import os
from pathlib import Path
import argparse
import glob
import numpy as np
import pandas as pd
from math import ceil, floor
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
import sys
sys.path.insert(0, "/home/dirac/HEP/OpenSamples") 
# ---- domain deps (unchanged) ----
from microboone_utils import *
from pynuml.io import File
from skimage.measure import block_reduce

def get_data_and_labels(input_file, f_downsample=6, start_evt=0, n_evts=1):
    """
    Read events [start_evt, start_evt + n_evts) from the file and return a batch.
    Robust to files where some groups (e.g., edep_table) are empty/missing.
    """
    # Try with full group set, then fall back without edep_table
    GROUP_TRIES = [
            ['event_table', 'wire_table', 'hit_table', 'edep_table'],
            ['event_table', 'wire_table', 'hit_table'],
    ]

    evt_list = None
    used_groups = None
    last_err = None

    for groups in GROUP_TRIES:
        try:
            f = File(input_file)
            for g in groups:
                try:
                    f.add_group(g)
                except Exception:
                    # harmless if already added or not required
                    pass
            # IMPORTANT: second arg is COUNT
            f.read_data(start_evt, n_evts)
            # build_evt also takes COUNT
            evt_list = f.build_evt(start_evt, n_evts)
            if len(evt_list) != n_evts:
                raise RuntimeError(f"Expected {n_evts} events, got {len(evt_list)}")
            used_groups = set(groups)
            break
        except Exception as e:
            last_err = e
            evt_list = None
            used_groups = None

    if evt_list is None:
        # Surface the original failure with context
        raise RuntimeError(
                f"Failed to build events from {input_file} "
                f"at [{start_evt}, +{n_evts}) even after group fallbacks"
        ) from last_err

    X_list, y_list = [], []
    extra_list = []

    for ev in evt_list:
        # -------- wires & ADCs --------
        wires = ev["wire_table"]
        planeadcs = [
                wires.query("local_plane==%i" % p)[['adc_%i' % i for i in range(0, ntimeticks())]].to_numpy()
                for p in range(0, nplanes())
        ]
        for p in range(nplanes()):
            planeadcs[p] = block_reduce(planeadcs[p], block_size=(1, f_downsample), func=np.sum)

        adccutoff = 10.0 * f_downsample / 6.0
        adcsaturation = 100.0 * f_downsample / 6.0
        for p in range(nplanes()):
            planeadcs[p][planeadcs[p] < adccutoff] = 0
            planeadcs[p][planeadcs[p] > adcsaturation] = adcsaturation

        # -------- hits & (optional) edeps --------

        hits = ev["hit_table"]

        edeps = None
        if "edep_table" in used_groups:  # we only *attempt* if we added the group
            try:
                edeps = ev["edep_table"]  # may raise if absent/empty
            except Exception:
                edeps = None
        # empty event
        if edeps is None or (hasattr(edeps, "empty") and edeps.empty):
            #print(f'empty event with Ev={ev['event_table']['nu_energy'].iloc[0]}' )
            continue  # do not include this event in the batch

        # not in active volume
        if not isPosInActiveVolume(ev['event_table']['nu_vtx_x'].iloc[0], ev['event_table']['nu_vtx_y'].iloc[0],
		                           ev['event_table']['nu_vtx_z'].iloc[0]):
            #print(f'No in active volume Ev={ev['event_table']['nu_energy'].iloc[0]}' )
            continue
	
        # Proceed as usual
        edeps = edeps.sort_values(by=["energy_fraction"], ascending=False, kind="mergesort") \
                                         .drop_duplicates(["hit_id"])
        hits = hits.merge(edeps, on=["hit_id"], how="left")

        # Stable dtype (avoid FutureWarning)
        hits["g4_id"] = pd.to_numeric(hits["g4_id"], errors="coerce").fillna(-1).astype("Int64")
        hits = hits.fillna(0).infer_objects(copy=False)

            
        # -------- truth masks --------
        planetruth = [np.zeros((nwires(p), ntimeticks())) for p in range(nplanes())]
        nrms = 2
        for p in range(nplanes()):
            nuhits = hits.query('local_plane==%i and g4_id>=0' % p)[['local_wire','local_time','rms']]
            for _, h in nuhits.iterrows():
                w = int(h['local_wire'])
                lo = max(0, floor(h['local_time'] - nrms*h['rms']))
                hi = min(ntimeticks(), ceil(h['local_time'] + nrms*h['rms']))
                if lo < hi:
                    planetruth[p][w][lo:hi] = 1

        for p in range(nplanes()):
            planetruth[p] = block_reduce(planetruth[p], block_size=(1, f_downsample), func=np.sum)
            planetruth[p] = np.multiply(planetruth[p], planeadcs[p])

        # -------- label --------
        is_cc = ev['event_table']['is_cc'].iloc[0]
        pdg = ev['event_table']['nu_pdg'].iloc[0]
        label = 0 if is_cc == 0 else (1 if abs(pdg) == 14 else (2 if abs(pdg) == 12 else -999))
        vtx_plane0 = ev['event_table']['nu_vtx_wire_pos_0'].iloc[0]
        vtx_plane1 = ev['event_table']['nu_vtx_wire_pos_1'].iloc[0]
        vtx_plane2 = ev['event_table']['nu_vtx_wire_pos_2'].iloc[0]
        Ev = ev['event_table']['nu_energy'].iloc[0]
        vtx = [ev['event_table']['nu_vtx_x'].iloc[0],
                ev['event_table']['nu_vtx_y'].iloc[0],
                ev['event_table']['nu_vtx_z'].iloc[0]
				]
        evt_info = {"wire_vtx": np.array([vtx_plane0, vtx_plane1, vtx_plane2], dtype=float), 
                        "nu_energy": Ev, "vtx_xyz": vtx}
            
		# -------- pad and append --------
        shapes = [arr.shape for arr in planetruth]  # (W_p, T_ds)
        T_ds = max(s[1] for s in shapes)
        max_W = max(s[0] for s in shapes)
        P = nplanes()
        X_evt = np.zeros((P, max_W, T_ds), dtype=np.float32)
        for p in range(P):
            Wp, Tp = planetruth[p].shape
            X_evt[p, :Wp, :Tp] = planetruth[p]

        X_list.append(X_evt)
        y_list.append(int(label))
        extra_list.append(evt_info)
            
    if len(X_list) == 0:
        return None, None, None
            
    X_batch = np.stack(X_list, axis=0)
    y_batch = np.asarray(y_list, dtype=np.int32)
    extra_info = np.array(extra_list, dtype=object)   
    return X_batch, y_batch, extra_info

def crop(plane_images, wires, f_downsample=6, pad_value=0.0):
    """
    Crop each plane image to a 500x500 pixel map
    Wire center = nu_vtx_wire_pos_p for each plane p
    Time center = weighted-average time index computed from pixels within the wire slab [x0:x1)
    Returns:
        crops: list of 3 arrays, each (500, 500) for planes 0,1,2.
    """
    if plane_images is None or len(plane_images) != 3:
        raise ValueError("plane_images must be a list/tuple of length 3 (one per plane).")
    _wires = wires['wire_vtx']
    wire_centers = [int(np.rint(_wires[0])),
                    int(np.rint(_wires[1])),
                    int(np.rint(_wires[2]))]

    # Fallback time center (downsampled) if the slab has no signal
    f_ds = max(int(f_downsample), 1)
    t_center_fallback = None

    HALF = 250  # half-size -> 500 window
    crops = []

    for p in range(3):
        img = np.asarray(plane_images[p])
        if img.ndim != 2:
            raise ValueError(f"Plane {p} image must be 2D (Wires, Ticks_ds). Got {img.shape}.")

        W, T = img.shape
        out = np.full((500, 500), pad_value, dtype=img.dtype)

        # Wire bounds centered on truth vertex wire for this plane
        cx = wire_centers[p]
        x0, x1 = cx - HALF, cx + HALF   # half-open [x0, x1) -> 500 wires

        # Clamp source slab to image bounds
        sx0, sx1 = max(0, x0), min(W, x1)

        # ----- Compute time center from the wire slab -----
        if sx0 < sx1:
            slab = img[sx0:sx1, :]               # shape: (Wx ~<= 500, T)
            weights = np.abs(slab)               # use magnitude as weights (robust if negatives)
            col_w = weights.sum(axis=0)          # per-time weights, shape (T,)
            total_w = col_w.sum()
            if total_w > 0:
                t_center = int(np.rint((np.arange(T) * col_w).sum() / total_w))
            else:
                # empty slab → fallback
                t_center = t_center_fallback if t_center_fallback is not None else (T // 2)
        else:
            # no overlap in wires → fallback
            t_center = t_center_fallback if t_center_fallback is not None else (T // 2)

        # Time bounds centered on computed t_center
        cy = int(np.clip(t_center, 0, T - 1))
        y0, y1 = cy - HALF, cy + HALF   # half-open [y0, y1) -> 500 ticks
        sy0, sy1 = max(0, y0), min(T, y1)

        # ----- Copy intersection into the 500×500 canvas -----
        if sx0 < sx1 and sy0 < sy1:
            dx0, dy0 = sx0 - x0, sy0 - y0
            dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)

            sx0, sx1, sy0, sy1 = map(int, (sx0, sx1, sy0, sy1))
            dx0, dx1, dy0, dy1 = map(int, (dx0, dx1, dy0, dy1))
            out[dx0:dx1, dy0:dy1] = img[sx0:sx1, sy0:sy1]

        crops.append(out)

    return crops


def parse_args():
    p = argparse.ArgumentParser(description="EventDataGenerator CLI")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Directory containing input .h5 files")
    p.add_argument("--output", type=str, required=True,
                   help="Output file .npz format")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Events per batch (default: 1)")
    p.add_argument("--file-pattern", type=str, default="*.h5",
                   help='Glob for files (default: "*.h5")')
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False,
                   help="Shuffle global event index each epoch (default: False)")
    p.add_argument("--f-downsample", type=int, default=6,
                   help="Time downsample factor (default: 6)")
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True,
                   help="Verbose scanning/logs (default: True)")
    return p.parse_args()

def main():
    args = parse_args()

    # Expand and sanity-check the data directory
    data_dir = str(Path(args.data_dir).expanduser())

    f = File(data_dir)
    print(len(f))
    X_all, y_all, info_all = [], [], []
    for e in range(len(f)):
        idx = e
        p,y,info = get_data_and_labels(data_dir, start_evt=e)
        if p is None:
            continue
        B, P, W, T = p.shape
        #print(info)
        
        for i in range(B):
            planes = [p[i, 0, :, :], p[i, 1, :, :], p[i, 2, :, :]] if P >= 3 else \
                        [p[i, j, :, :] for j in range(P)]
            try:
                cropped = crop(planes, info[0], f_downsample=6, pad_value=0)
            except Exception as e:
                print(f"buuu")
                continue
            X_all.append(np.stack(cropped, axis=0).astype(np.float32))  # (3,500,500)
        y_all.append(y[0])
        info_all.append(info[0])
        #print(f' idx {idx} event={info[0]}')
        if len(X_all) == 0:
            raise RuntimeError("No events were cropped. Check that 'info' contains vertex values.")
        #print(z)
    X_out = np.stack(X_all, axis=0)               # (N,3,500,500)
    y_out = np.asarray(y_all, dtype=np.int32)     # (N,)
    z_out = np.array(info_all, dtype=object)
    
    #out_path=str(Path(args.output).expanduser())
    #out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(args.output, X=X_out, y=y_out, z=z_out)
    print(f"[saved] {args.output} :: X{X_out.shape}, y{y_out.shape}, z{z_out.shape}")  
if __name__ == "__main__":
    main()
   
