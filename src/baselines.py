# src/baselines.py
from __future__ import annotations

import os
import numpy as np
from sklearn.decomposition import PCA

from src.clustering import kmeans_cluster


def pca_features(X: np.ndarray, n_components: int = 16, seed: int = 42) -> np.ndarray:
    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)
    return Z


def run_pca_kmeans(
    X: np.ndarray,
    out_dir: str,
    n_components: int = 16,
    n_clusters: int = 10,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      - Z_pca (N, n_components)
      - labels (N,)
    """
    os.makedirs(out_dir, exist_ok=True)
    Z = pca_features(X, n_components=n_components, seed=seed)
    labels = kmeans_cluster(Z, n_clusters=n_clusters, seed=seed)

    np.save(os.path.join(out_dir, "latent_pca.npy"), Z.astype(np.float32))
    np.save(os.path.join(out_dir, "labels_pca.npy"), labels.astype(np.int32))
    return Z, labels
