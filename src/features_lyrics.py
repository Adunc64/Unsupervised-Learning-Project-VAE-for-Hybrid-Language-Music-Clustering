# # src/features_lyrics.py
# from __future__ import annotations

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer


# def build_tfidf_features(
#     meta_csv_path: str,
#     cache_dir: str = "data/cache",
#     max_features: int = 5000,
#     ngram_range: tuple[int, int] = (3, 5),
#     min_df: int = 2,
# ) -> np.ndarray:
#     """
#     Builds TF-IDF features from lyrics and caches:
#       - data/cache/lyrics_tfidf.npy
#       - data/cache/tfidf_vectorizer.joblib
#     Returns dense float32 array shape (N, D).
#     """
#     os.makedirs(cache_dir, exist_ok=True)
#     out_x = os.path.join(cache_dir, "lyrics_tfidf.npy")
#     out_vec = os.path.join(cache_dir, "tfidf_vectorizer.joblib")

#     df = pd.read_csv(meta_csv_path)
#     if "lyrics" not in df.columns:
#         raise ValueError("meta.csv must contain a 'lyrics' column.")

#     lyrics = df["lyrics"].fillna("").astype(str).tolist()

#     vectorizer = TfidfVectorizer(
#         analyzer="char_wb",          # important for Bangla + English
#         ngram_range=ngram_range,     # (3,5) works well
#         max_features=max_features,
#         min_df=min_df,
#         lowercase=False,             # don't force lowercasing (Bangla safe)
#     )

#     X_sparse = vectorizer.fit_transform(lyrics)
#     X = X_sparse.toarray().astype(np.float32)

#     np.save(out_x, X)
#     joblib.dump(vectorizer, out_vec)

#     print(f"[features_lyrics] Saved TF-IDF features: {out_x} | shape={X.shape}")
#     print(f"[features_lyrics] Saved vectorizer: {out_vec}")
#     return X


# def load_tfidf_features(cache_dir: str = "data/cache") -> np.ndarray:
#     path = os.path.join(cache_dir, "lyrics_tfidf.npy")
#     if not os.path.exists(path):
#         raise FileNotFoundError(
#             f"Missing cached TF-IDF at {path}. Run build_tfidf_features() first."
#         )
#     X = np.load(path)
#     return X


# src/features_lyrics.py
from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def build_tfidf_features(
    csv_path: str,
    text_col: str = "lyrics",
    cache_dir: str = "data/cache",
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
) -> np.ndarray:
    os.makedirs(cache_dir, exist_ok=True)
    out_x = os.path.join(cache_dir, "lyrics_tfidf.npy")
    out_vec = os.path.join(cache_dir, "tfidf_vectorizer.joblib")

    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"CSV must contain '{text_col}' column.")

    texts = df[text_col].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        lowercase=False,
    )
    X_sparse = vectorizer.fit_transform(texts)
    X = X_sparse.toarray().astype(np.float32)

    np.save(out_x, X)
    joblib.dump(vectorizer, out_vec)

    print(f"[features_lyrics] Saved TF-IDF: {out_x} | shape={X.shape}")
    return X


def load_tfidf_features(cache_dir: str = "data/cache") -> np.ndarray:
    path = os.path.join(cache_dir, "lyrics_tfidf.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cached TF-IDF at {path}")
    return np.load(path)


def build_lyrics_embeddings_svd(
    csv_path: str,
    text_col: str = "lyrics",
    cache_dir: str = "data/cache",
    tfidf_max_features: int = 5000,
    svd_dim: int = 128,
) -> np.ndarray:
    """
    Lyrics embedding = TF-IDF (char ngrams) -> TruncatedSVD.
    Saves: data/cache/lyrics_emb.npy
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, "lyrics_emb.npy")

    df = pd.read_csv(csv_path)
    texts = df[text_col].fillna("").astype(str).tolist()

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=tfidf_max_features, lowercase=False)
    X = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=svd_dim, random_state=42)
    Z = svd.fit_transform(X).astype(np.float32)

    np.save(out, Z)
    print(f"[features_lyrics] Saved lyrics embeddings: {out} | shape={Z.shape}")
    return Z


def load_lyrics_embeddings(cache_dir: str = "data/cache") -> np.ndarray:
    path = os.path.join(cache_dir, "lyrics_emb.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing cached lyrics embeddings at {path}")
    return np.load(path)
