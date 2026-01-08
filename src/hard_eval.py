# src/hard_eval.py
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Purity = (1/N) * sum_k max_j |C_k ∩ T_j|
    Treat -1 (DBSCAN noise) as its own cluster if present.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    N = len(y_true)
    if N == 0:
        return float("nan")

    purity_sum = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx) == 0:
            continue
        true_labels = y_true[idx]
        # count majority label
        _, counts = np.unique(true_labels, return_counts=True)
        purity_sum += counts.max()

    return purity_sum / N


def evaluate_hard(Z: np.ndarray, labels: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Returns Silhouette, Davies-Bouldin, ARI, NMI, Purity.
    """
    Z = np.asarray(Z)
    labels = np.asarray(labels)
    y_true = np.asarray(y_true)

    # Need at least 2 clusters for silhouette/db
    if len(np.unique(labels)) < 2:
        return {"silhouette": "", "davies_bouldin": "", "ari": "", "nmi": "", "purity": ""}

    sil = silhouette_score(Z, labels)
    db = davies_bouldin_score(Z, labels)
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    pur = cluster_purity(y_true, labels)

    return {
        "silhouette": float(sil),
        "davies_bouldin": float(db),
        "ari": float(ari),
        "nmi": float(nmi),
        "purity": float(pur),
    }
