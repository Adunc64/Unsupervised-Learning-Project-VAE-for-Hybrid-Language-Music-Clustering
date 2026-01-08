# src/dataset.py
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class NumpyFeatureDataset(Dataset):
    def __init__(self, X: np.ndarray):
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("X must be 2D (N, D).")
        self.X = X.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.X[idx])

##medium##

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, X: np.ndarray):
        if X.ndim != 4:
            raise ValueError("SpectrogramDataset expects X shape (N, C, H, W).")
        self.X = X.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])

