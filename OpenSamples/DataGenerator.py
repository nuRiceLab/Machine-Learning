import os, glob, bisect
import numpy as np
from math import ceil
from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(
        self,
        data_path,
        batch_size=32,
        shuffle=True,
        channels_last=True,
        dtype="float32",
        in_memory_npz=False,
        file_pattern="*.npz",
        recursive=False,
        num_classes=3,          # ### NEW
        one_hot=True,           # ### NEW
        y_dtype="float32",      # ### NEW: dtype for one-hot labels
    ):
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.channels_last = bool(channels_last)
        self.dtype = np.dtype(dtype)
        self.in_memory_npz = bool(in_memory_npz)
        self.num_classes = int(num_classes)     # ### NEW
        self.one_hot = bool(one_hot)            # ### NEW
        self.y_dtype = np.dtype(y_dtype)        # ### NEW

        # Resolve files
        if os.path.isdir(data_path):
            pattern = "**/" + file_pattern if recursive else file_pattern
            self.file_paths = sorted(glob.glob(os.path.join(data_path, pattern), recursive=recursive))
            if not self.file_paths:
                raise ValueError(f"No files matched pattern '{file_pattern}' in directory '{data_path}'.")
        else:
            ext = os.path.splitext(data_path)[1].lower()
            if ext != ".npz":
                raise ValueError(f"Unsupported file extension: {ext}. Expected a .npz or a directory.")
            self.file_paths = [data_path]

        # Per-file counts and cumulative index map
        self._file_counts = []
        if self.in_memory_npz:
            self._files_data = []
            for fp in self.file_paths:
                npz = np.load(fp, allow_pickle=True)
                X = np.asarray(npz["X"])  # (Ni, 3, 500, 500)
                y = np.asarray(npz["y"])
                self._files_data.append({"X": X, "y": y})
                self._file_counts.append(int(X.shape[0]))
        else:
            self._files_data = None
            for fp in self.file_paths:
                with np.load(fp, allow_pickle=True, mmap_mode="r") as npz:
                    n = int(npz["X"].shape[0])
                    self._file_counts.append(n)

        self._cum = np.cumsum([0] + self._file_counts)
        self.N = int(self._cum[-1])
        if self.N == 0:
            raise ValueError("No samples found across provided files.")

        self.indexes = np.arange(self.N, dtype=np.int64)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return ceil(self.N / self.batch_size)

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        end = min((batch_index + 1) * self.batch_size, self.N)
        gids = self.indexes[start:end]

        # group by file
        per_file_local = {}
        for g in gids:
            fidx = bisect.bisect_right(self._cum, g) - 1
            local = int(g - self._cum[fidx])
            per_file_local.setdefault(fidx, []).append(local)

        X_parts, y_parts = [], []

        if self.in_memory_npz:
            for fidx, locals_ in per_file_local.items():
                data = self._files_data[fidx]
                X_parts.append(data["X"][locals_])
                y_parts.append(data["y"][locals_])
        else:
            for fidx, locals_ in per_file_local.items():
                fp = self.file_paths[fidx]
                with np.load(fp, allow_pickle=True, mmap_mode="r") as npz:
                    X_parts.append(npz["X"][locals_])
                    y_parts.append(npz["y"][locals_])

        Xb = np.concatenate(X_parts, axis=0).astype(self.dtype, copy=False)
        yb = np.asarray(np.concatenate(y_parts, axis=0), dtype=np.int64)

        # channels_last
        if self.channels_last:
            Xb = np.transpose(Xb, (0, 2, 3, 1))  # (B, 500, 500, 3)

        # ### NEW: one-hot encode if requested
        if self.one_hot:
            # safety: clip/validate labels before indexing
            if (yb.min() < 0) or (yb.max() >= self.num_classes):
                raise ValueError(f"Label out of range. Found [{yb.min()}, {yb.max()}], "
                                 f"num_classes={self.num_classes}.")
            yb = np.eye(self.num_classes, dtype=self.y_dtype)[yb]  # (B, C)
        else:
            yb = yb.astype(np.int32, copy=False)

        return Xb, yb

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
