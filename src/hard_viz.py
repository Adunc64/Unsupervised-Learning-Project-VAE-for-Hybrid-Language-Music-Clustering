# src/hard_viz.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BENGALI_RANGE = ("\u0980", "\u09FF")

def infer_language(text: str) -> str:
    if not isinstance(text, str):
        return "unk"
    for ch in text:
        if BENGALI_RANGE[0] <= ch <= BENGALI_RANGE[1]:
            return "bn"
    return "en"

def save_cluster_distribution(
    aligned_csv_path: str,
    labels: np.ndarray,
    out_png: str,
    group_col: str = "genre",
    title: str = "Cluster distribution",
):
    """
    Creates a stacked bar plot: cluster -> counts per group_col (genre or language).
    """
    df = pd.read_csv(aligned_csv_path)
    df["cluster"] = labels

    if group_col == "language":
        df["language"] = df["lyrics"].apply(infer_language)
        group_col = "language"

    # pivot: clusters x group counts
    pivot = df.pivot_table(index="cluster", columns=group_col, aggfunc="size", fill_value=0)
    pivot = pivot.sort_index()

    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))

    for col in pivot.columns:
        vals = pivot[col].values
        plt.bar(x, vals, bottom=bottom, label=str(col))
        bottom += vals

    plt.xticks(x, pivot.index.astype(str), rotation=0)
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()
