import os
import glob
import numpy as np
import pandas as pd
from math import ceil, floor
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from tensorflow.keras.utils import Sequence
import sys
sys.path.insert(0, "/home/dirac/HEP/OpenSamples") 
# ---- domain deps (unchanged) ----
from microboone_utils import *
from pynuml.io import File
from skimage.measure import block_reduce


def _coalesce_consecutive(indices):
    """
    Given a sorted list of integers, yield (start, n) for each consecutive run.
    Example: [0,1,2,5,6] -> (0,3), (5,2)
    """
    for _, group in groupby(enumerate(indices), key=lambda t: t[0] - t[1]):
        run = list(map(itemgetter(1), group))
        start = run[0]
        length = len(run)
        yield start, length


class EventDataGenerator(Sequence):
    """
    Keras Sequence that indexes **events across all files**.
    """

    def __init__(
        self,
        data_dir,
        batch_size=8,
        file_pattern="*.h5",
        shuffle=True,
        f_downsample=6,
        scan_chunk=512,
        verbose=False,
    ):
        """
        Args:
            data_dir (str): Directory with HDF5 files.
            batch_size (int): Events per batch.
            file_pattern (str): Glob for files (default '*.h5').
            shuffle (bool): Shuffle the global event index each epoch.
            f_downsample (int): Downsample factor for time dimension.
            scan_chunk (int): Chunk size used while probing event counts.
            verbose (bool): Print scan info.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.f_downsample = f_downsample
        self.scan_chunk = scan_chunk
        self.verbose = verbose

        self.files = sorted(glob.glob(os.path.join(self.data_dir, file_pattern)))
        if not self.files:
            raise ValueError(f"No files found in '{data_dir}' with pattern '{file_pattern}'")

        # Pre-scan to build global (file, evt_idx) index
        self.file_event_counts = {}
        self.samples = []  # list of (file_path, evt_idx)
        total = 0
        for fp in self.files:
            n = self._probe_nevents(fp)
            self.file_event_counts[fp] = n
            if self.verbose:
                print(f"[scan] {os.path.basename(fp)} -> {n} events")
            for e in range(n):
                self.samples.append((fp, e))
            total += n

        if total == 0:
            raise ValueError("No events found across the provided files.")

        # establish deterministic order; shuffle if requested
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Infer a reference shape by reading a single event
        self._sample_shape = None
        for gid in self.indexes:                     # try events in shuffled (or sorted) order
            fp, evt = self.samples[gid]
            X_ref, _= self._load_events_for_file(fp, [evt])
            if X_ref is not None and getattr(X_ref, "ndim", 0) > 0 and X_ref.shape[0] > 0:
                self._sample_shape = X_ref.shape[1:] # (P, max_W, T_ds)
                break
        if self._sample_shape is None:
            raise RuntimeError("No usable events found to infer sample shape (all were skipped).")

    # ---------------- Keras Sequence API ----------------

    def __len__(self):
        return ceil(len(self.samples) / self.batch_size)

    # -----
    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end = min((batch_idx + 1) * self.batch_size, len(self.samples))
        batch_indices = self.indexes[start:end]

        from collections import defaultdict
        per_file = defaultdict(list)
        for global_idx in batch_indices:
            fp, evt = self.samples[global_idx]
            per_file[fp].append(evt)

        X_parts, y_parts = [], []
        info_parts = []
        # Load the requested window
        for fp, evts in per_file.items():
            evts = sorted(evts)
            for s, n in _coalesce_consecutive(evts):
                X_block, y_block, ex_block = self.get_data_and_labels(
                    fp, self.f_downsample, start_evt=s, n_evts=n
                )
                if X_block is None or y_block is None or X_block.shape[0] == 0:
                    continue
                X_parts.append(X_block)
                y_parts.append(y_block)
                info_parts.append(ex_block)
                
        # Top up to target batch_size by scanning forward
        have = sum(x.shape[0] for x in X_parts) if X_parts else 0
        probe = end
        while have < self.batch_size and probe < len(self.samples):
            fp, evt = self.samples[self.indexes[probe]]
            Xb, yb, exb = self.get_data_and_labels(fp, self.f_downsample, start_evt=evt, n_evts=1)
            probe += 1
            if Xb is not None and yb is not None and Xb.shape[0] > 0:
                X_parts.append(Xb)
                y_parts.append(yb)
                have += Xb.shape[0]
                info_parts.append(exb)

        # If still empty (e.g., we’re at the very end with only skipped events), fail clearly
        if not X_parts:
            raise IndexError("No usable events for this batch (all skipped).")

        X_batch = np.concatenate(X_parts, axis=0)
        y_batch = np.concatenate(y_parts, axis=0)
        z_batch = np.concatenate(info_parts, axis=0)
        return X_batch, y_batch, z_batch

    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # ---------------- Event counting (pre-scan) ----------------

    def _probe_nevents(self, file_path):
        
        f = File(file_path)
        return len(f)
        

    # ---------------- Helpers ----------------
    def _load_events_for_file(self, file_path, evt_indices_sorted):
        X_list, y_list = [], []
        for s, n in _coalesce_consecutive(evt_indices_sorted):
            Xb, yb, exb = self.get_data_and_labels(
                                            file_path, self.f_downsample, start_evt=s, n_evts=n)
            # skip empties/sentinels
            if Xb is None or yb is None:
                continue
            if getattr(Xb, "ndim", 0) == 0 or Xb.shape[0] == 0:
                continue
            X_list.append(Xb)
            y_list.append(yb)

        if not X_list:
            return None, None  # <-- caller must handle
        return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)

    # ---------------- Domain logic (range-aware) ----------------

    @staticmethod
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
            #print(f'event is empty? {edeps}')
            if edeps is None or (hasattr(edeps, "empty") and edeps.empty):
                    continue  # do not include this event in the batch

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
            vtx_time = ev['event_table']['nu_vtx_wire_time'].iloc[0]
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
            extra_list.append([vtx_plane0,vtx_plane1,vtx_plane2,vtx_time])

        if len(X_list) == 0:
            return None, None, None
            
        X_batch = np.stack(X_list, axis=0)
        y_batch = np.asarray(y_list, dtype=np.int32)
        extra_info = np.asarray(extra_list)
        return X_batch, y_batch, extra_info
