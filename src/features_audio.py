# src/features_audio.py
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import librosa


def _fix_length_2d(x: np.ndarray, max_frames: int) -> np.ndarray:
    """Pad/truncate spectrogram time dimension to max_frames."""
    if x.shape[1] == max_frames:
        return x
    if x.shape[1] > max_frames:
        return x[:, :max_frames]
    pad = max_frames - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


def build_melspec_features(
    lyrics_csv_path: str,
    audio_root: str = "data/audio",
    cache_dir: str = "data/cache",
    sr: int = 22050,
    duration_sec: float = 30.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    max_frames: int = 1024,
) -> tuple[np.ndarray, np.ndarray, str]:
    """
    Builds mel-spectrogram features for each row in lyrics_csv_path.

    IMPORTANT:
    - Skips rows where audio file is missing
    - Skips rows where audio file exists but cannot be decoded (corrupt/unsupported)
    - Saves a filtered "aligned" CSV containing only rows that were kept

    Returns:
      X: (N, 1, n_mels, max_frames) float32 in [0,1]
      y_true: (N,) integer label for genre (for ARI)
      aligned_csv: path to aligned CSV used for audio+lyrics fusion

    Saves:
      data/cache/audio_melspec.npy
      data/cache/audio_genre_labels.npy
      data/cache/audio_genre_names.npy
      data/cache/lyrics_audio_aligned.csv
    """
    os.makedirs(cache_dir, exist_ok=True)

    out_x = os.path.join(cache_dir, "audio_melspec.npy")
    out_y = os.path.join(cache_dir, "audio_genre_labels.npy")
    out_names = os.path.join(cache_dir, "audio_genre_names.npy")
    aligned_csv = os.path.join(cache_dir, "lyrics_audio_aligned.csv")

    df = pd.read_csv(lyrics_csv_path)

    required = {"genre", "track"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"lyrics.csv must contain columns: {sorted(required)}. Missing: {sorted(missing)}")

    # Normalize to match folder names + filenames
    df = df.copy()
    df["genre"] = df["genre"].astype(str).str.strip().str.lower()
    df["track"] = df["track"].astype(str).str.strip()

    # Build label map from genres present in CSV
    genres_all = df["genre"].tolist()
    uniq = sorted(set(genres_all))
    genre_to_id = {g: i for i, g in enumerate(uniq)}

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    kept_indices: list[int] = []

    missing_files = 0
    bad_files = 0

    for i, row in df.iterrows():
        genre = row["genre"]
        track = row["track"]
        wav_path = os.path.join(audio_root, genre, track)

        # 1) Skip missing audio
        if not os.path.exists(wav_path):
            missing_files += 1
            continue

        # 2) Skip unreadable/corrupt audio
        try:
            y, _ = librosa.load(wav_path, sr=sr, mono=True, duration=duration_sec)
        except Exception:
            bad_files += 1
            continue

        # Mel spectrogram -> log
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
        )
        S = librosa.power_to_db(S, ref=np.max).astype(np.float32)  # (n_mels, T)

        # Fix time length
        S = _fix_length_2d(S, max_frames)

        # Normalize to [0,1] so Sigmoid decoder works well
        S_min, S_max = float(S.min()), float(S.max())
        if S_max > S_min:
            S = (S - S_min) / (S_max - S_min)
        else:
            S = np.zeros_like(S, dtype=np.float32)

        X_list.append(S[None, :, :])  # (1, n_mels, T)
        y_list.append(genre_to_id[genre])
        kept_indices.append(i)

        if len(kept_indices) % 100 == 0:
            print(f"[features_audio] kept {len(kept_indices)}/{len(df)}")

    if len(X_list) == 0:
        raise RuntimeError(
            "No usable audio files were loaded. Check: data/audio/<genre>/<track> paths and file formats."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y_true = np.array(y_list, dtype=np.int64)

    # Save caches
    np.save(out_x, X)
    np.save(out_y, y_true)
    np.save(out_names, np.array(uniq, dtype=object))

    # Save aligned CSV (only rows that matched usable audio)
    df_aligned = df.loc[kept_indices].copy()
    df_aligned.to_csv(aligned_csv, index=False, encoding="utf-8")

    print(f"[features_audio] Saved: {out_x} | shape={X.shape}")
    print(f"[features_audio] Skipped missing files: {missing_files}")
    print(f"[features_audio] Skipped unreadable files: {bad_files}")
    print(f"[features_audio] Aligned CSV: {aligned_csv} | rows={len(df_aligned)}")

    return X, y_true, aligned_csv


def load_melspec_features(cache_dir: str = "data/cache") -> tuple[np.ndarray, np.ndarray]:
    x_path = os.path.join(cache_dir, "audio_melspec.npy")
    y_path = os.path.join(cache_dir, "audio_genre_labels.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Missing cached audio features. Run build_melspec_features() first.")
    return np.load(x_path), np.load(y_path)
