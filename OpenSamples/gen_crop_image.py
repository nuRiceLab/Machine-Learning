import os
from pathlib import Path
import numpy as np
from generator import *
import argparse

def crop(plane_images, event_vals, f_downsample=6, pad_value=0.0):
    """
    Crop each plane image to a 500x500 pixel map
    Wire center = nu_vtx_wire_pos_p for each plane p
    Time center = weighted-average time index computed from pixels within the wire slab [x0:x1)
    Returns:
        crops: list of 3 arrays, each (500, 500) for planes 0,1,2.
    """
    if plane_images is None or len(plane_images) != 3:
        raise ValueError("plane_images must be a list/tuple of length 3 (one per plane).")

    event_vals = np.asarray(event_vals, dtype=float)
    if event_vals.shape[0] != 4:
        raise ValueError("event_vals must be [wire0, wire1, wire2, nu_vtx_wire_time].")

    wire_centers = [int(np.rint(event_vals[0])),
                    int(np.rint(event_vals[1])),
                    int(np.rint(event_vals[2]))]

    # Fallback time center (downsampled) if the slab has no signal
    f_ds = max(int(f_downsample), 1)
    t_center_fallback = int(np.rint(event_vals[3] / f_ds)) if np.isfinite(event_vals[3]) else None

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

def save_all_crops_one_file(gen, out_path="cropped_all.npz", pad_value=0.0):
    """
    Iterate through the entire generator, crop each event to (500,500) per plane,
    and save ALL crops & labels to one file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    f_ds = getattr(gen, "f_downsample", 1)

    # ------- NPZ path (keep in memory then write once) -------
    X_all, y_all = [], []
    for b in range(len(gen)):
            
        p, y, info = gen[b]      # p:(B,P,W,T), y:(B,), info:(B, ...)
        p = np.asarray(p); y = np.asarray(y)
        B, P, W, T = p.shape
        for i in range(B):
            planes = [p[i, 0, :, :], p[i, 1, :, :], p[i, 2, :, :]] if P >= 3 else \
                         [p[i, j, :, :] for j in range(P)]
            try:
                cropped = crop(planes, info[i], f_downsample=f_ds, pad_value=pad_value)
            except Exception as e:
                print(f"[batch {b} idx {i}] crop failed: {e}")
                continue
            X_all.append(np.stack(cropped, axis=0).astype(np.float32))  # (3,500,500)
            y_all.append(int(y[i]))

    if len(X_all) == 0:
        raise RuntimeError("No events were cropped. Check that 'info' contains vertex values.")

    X_out = np.stack(X_all, axis=0)               # (N,3,500,500)
    y_out = np.asarray(y_all, dtype=np.int32)     # (N,)

    np.savez_compressed(out_path, X=X_out, y=y_out)
    print(f"[saved] {out_path} :: X{X_out.shape}, y{y_out.shape}")

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

    gen = EventDataGenerator(
        data_dir=data_dir,
        batch_size=args.batch_size,
        file_pattern=args.file_pattern,
        shuffle=args.shuffle,
        f_downsample=args.f_downsample,
        verbose=args.verbose,
    )

    # Example: peek one batch size only (no training here)
    print(f"Generator, cropping pixel maps. Batches: {len(gen)}; batch_size: {args.batch_size}")
    save_all_crops_one_file(gen, out_path=str(Path(args.output).expanduser()))
    
if __name__ == "__main__":
    main()
   