# src/fusion.py
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def fuse_concat(Z_audio: np.ndarray, Z_lyrics: np.ndarray) -> np.ndarray:
    """
    Standardize each modality then concat.
    """
    if len(Z_audio) != len(Z_lyrics):
        raise ValueError(f"Row mismatch: audio={len(Z_audio)} vs lyrics={len(Z_lyrics)}")

    sa = StandardScaler()
    sl = StandardScaler()
    Za = sa.fit_transform(Z_audio)
    Zl = sl.fit_transform(Z_lyrics)
    return np.concatenate([Za, Zl], axis=1).astype(np.float32)
