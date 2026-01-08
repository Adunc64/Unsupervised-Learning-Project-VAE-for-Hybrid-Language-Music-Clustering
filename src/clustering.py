# src/clustering.py
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


def kmeans_cluster(X: np.ndarray, n_clusters: int = 10, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    labels = km.fit_predict(X)
    return labels


def save_tsne_plot(X, labels, out_path, title, seed=42):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    X2 = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.title(title)
    cb = plt.colorbar(sc)
    cb.set_label("KMeans cluster id")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


##medium##
import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN


def agglomerative_cluster(X: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    return model.fit_predict(X)


def dbscan_cluster(X: np.ndarray, eps: float = 0.8, min_samples: int = 10) -> np.ndarray:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    return model.fit_predict(X)  # note: may output -1 for noise