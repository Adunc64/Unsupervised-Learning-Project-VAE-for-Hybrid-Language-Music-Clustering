# src/evaluation.py
from __future__ import annotations

import os
import csv
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score


def evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Returns Silhouette and Calinski-Harabasz.
    """
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def append_metrics_csv(out_csv: str, row: dict):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    file_exists = os.path.exists(out_csv)
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[evaluation] Wrote metrics row to: {out_csv}")


##medium##
from sklearn.metrics import davies_bouldin_score, adjusted_rand_score


def evaluate_clustering_medium(X, labels, y_true=None) -> dict:
    """
    Returns:
      silhouette (higher better),
      davies_bouldin (lower better),
      ari (higher better) if y_true provided.
    """
    out = {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
    }
    if y_true is not None:
        out["ari"] = float(adjusted_rand_score(y_true, labels))
    else:
        out["ari"] = ""
    return out