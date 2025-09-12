import os
import numpy as np
import h5py
from math import ceil
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    """
    Data generator for cropped MicroBooNE maps saved in one file.

    Supported inputs
    ----------------
    NPZ: keys 'X' -> (N, 3, 500, 500), 'y' -> (N,), optional 'info'
    Returns batches as:
      - (X, y) by default
    X shape per batch:
      - channels_last=True  -> (B, 500, 500, 3)
      - channels_last=False -> (B, 3, 500, 500)
    """

    def __init__(
        self,
        file_path,
        batch_size=32,
        shuffle=True,
        channels_last=True,
        dtype="float32",
        in_memory_npz=False,   # if True, loads NPZ arrays fully into RAM
    ):
        self.file_path = file_path
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.channels_last = bool(channels_last)
        self.dtype = np.dtype(dtype)
        self.in_memory_npz = bool(in_memory_npz)

        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npz":
            self.backend = "npz"
            self._npz = np.load(file_path, allow_pickle=True)
            self.X = self._npz["X"]  # ndarray via NPZ accessor
            self.y = self._npz["y"]
            self.info = self._npz["info"] if "info" in self._npz.files else None

            if self.in_memory_npz:
                # materialize to RAM (optional)
                self.X = np.asarray(self.X)
                self.y = np.asarray(self.y)
                if self.info is not None:
                    self.info = np.asarray(self.info, dtype=object)

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self.N = int(self.X.shape[0])
        if self.y.shape[0] != self.N:
            raise ValueError("Mismatched X and y lengths.")

        self.indexes = np.arange(self.N, dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # --- Keras Sequence API ---
    def __len__(self):
        return ceil(self.N / self.batch_size)

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        end = min((batch_index + 1) * self.batch_size, self.N)
        idxs = self.indexes[start:end]

        Xb = self._get_X(idxs)
        yb = self._get_y(idxs)

        # dtype + format
        Xb = Xb.astype(self.dtype, copy=False)
        if self.channels_last:
            # (B, 3, 500, 500) -> (B, 500, 500, 3)
            Xb = np.transpose(Xb, (0, 2, 3, 1))

        return Xb, yb

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # --- internal helpers ---
    def _get_X(self, idxs):
        return self.X[idxs]  # npz path

    def _get_y(self, idxs):
        y = self.y[idxs]
        return np.asarray(y, dtype=np.int32)

    def __del__(self):
        if hasattr(self, "_h5"):
            try:
                self._h5.close()
            except Exception:
                pass
