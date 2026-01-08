# run.py
from __future__ import annotations

import os
import json
import argparse
import numpy as np

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src.hard_models import BetaVAE_MLP, CVAE_MLP, AutoEncoder_MLP
from src.hard_eval import evaluate_hard
from src.hard_viz import save_cluster_distribution


from src.features_lyrics import (
    build_tfidf_features, load_tfidf_features,
    build_lyrics_embeddings_svd, load_lyrics_embeddings
)
from src.features_audio import build_melspec_features, load_melspec_features
from src.training import train_vae, train_conv_vae
from src.baselines import run_pca_kmeans
from src.fusion import fuse_concat
from src.clustering import (
    kmeans_cluster, agglomerative_cluster, dbscan_cluster, save_tsne_plot
)
from src.evaluation import (
    evaluate_clustering, evaluate_clustering_medium, append_metrics_csv
)


def run_easy(args):
    out_dir = "results/easy"
    os.makedirs(out_dir, exist_ok=True)

    cache_x = "data/cache/lyrics_tfidf.npy"
    if not os.path.exists(cache_x):
        X = build_tfidf_features(args.meta, text_col="lyrics", max_features=args.max_features)
    else:
        X = load_tfidf_features()
        print(f"[easy] Loaded cached TF-IDF: {cache_x} | shape={X.shape}")

    _, Z_vae = train_vae(
        X,
        out_dir=out_dir,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    labels_vae = kmeans_cluster(Z_vae, n_clusters=args.k, seed=args.seed)
    np.save(os.path.join(out_dir, "labels_vae.npy"), labels_vae.astype(np.int32))

    Z_pca, labels_pca = run_pca_kmeans(
        X,
        out_dir=out_dir,
        n_components=args.latent_dim,
        n_clusters=args.k,
        seed=args.seed,
    )

    metrics_csv = os.path.join(out_dir, "metrics.csv")
    m_vae = evaluate_clustering(Z_vae, labels_vae)
    append_metrics_csv(metrics_csv, {"method": "VAE+KMeans", "k": args.k, **m_vae})

    m_pca = evaluate_clustering(Z_pca, labels_pca)
    append_metrics_csv(metrics_csv, {"method": "PCA+KMeans", "k": args.k, **m_pca})

    save_tsne_plot(
        Z_vae, labels_vae,
        os.path.join(out_dir, "tsne_vae.png"),
        "t-SNE: VAE latent (lyrics)",
        seed=args.seed
    )
    save_tsne_plot(
        Z_pca, labels_pca,
        os.path.join(out_dir, "tsne_pca.png"),
        "t-SNE: PCA features (lyrics)",
        seed=args.seed
    )

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nDONE ✅ Easy finished (results/easy/)")


def run_medium(args):
    """
    Medium:
    - mel-spectrogram + ConvVAE (audio)
    - lyrics embedding (TFIDF+SVD)
    - fuse (concat) and cluster with KMeans/Agglo/DBSCAN
    - metrics: silhouette, davies-bouldin, ARI (genre labels available)
    """
    out_dir = "results/medium"
    os.makedirs(out_dir, exist_ok=True)

    # --- Robust cache check for audio ---
    x_path = "data/cache/audio_melspec.npy"
    y_path = "data/cache/audio_genre_labels.npy"
    names_path = "data/cache/audio_genre_names.npy"
    aligned_path = "data/cache/lyrics_audio_aligned.csv"

    if not (os.path.exists(x_path) and os.path.exists(y_path) and os.path.exists(names_path) and os.path.exists(aligned_path)):
        # build_melspec_features must be the UPDATED version that:
        # - skips missing/unreadable wavs
        # - writes aligned CSV
        X_audio, y_true, aligned_csv = build_melspec_features(args.lyrics_csv)
    else:
        X_audio, y_true = load_melspec_features()
        aligned_csv = aligned_path
        print(f"[medium] Loaded cached audio: {x_path} | shape={X_audio.shape}")

    # --- Lyrics embeddings must be built on aligned_csv so rows match audio ---
    emb_path = "data/cache/lyrics_emb.npy"
    if not os.path.exists(emb_path):
        Z_lyrics = build_lyrics_embeddings_svd(aligned_csv, text_col="lyrics", svd_dim=args.lyrics_dim)
    else:
        Z_lyrics = load_lyrics_embeddings()
        # Safety: if cache was built from a different row set, rebuild
        if Z_lyrics.shape[0] != X_audio.shape[0]:
            print(
                f"[medium] lyrics_emb.npy rows ({Z_lyrics.shape[0]}) != audio rows ({X_audio.shape[0]}). Rebuilding lyrics embeddings..."
            )
            Z_lyrics = build_lyrics_embeddings_svd(aligned_csv, text_col="lyrics", svd_dim=args.lyrics_dim)
        else:
            print(f"[medium] Loaded cached lyrics emb: {emb_path} | shape={Z_lyrics.shape}")

    # 3) ConvVAE on audio -> latent
    _, Z_audio = train_conv_vae(
        X_audio,
        out_dir=out_dir,
        latent_dim=args.latent_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    # 4) Fusion
    Z_fused = fuse_concat(Z_audio, Z_lyrics)
    np.save(os.path.join(out_dir, "latent_fused.npy"), Z_fused.astype(np.float32))

    # 5) Clustering experiments
    metrics_csv = os.path.join(out_dir, "metrics.csv")

    def eval_and_save(name, Z, labels):
        # DBSCAN can output -1. If all are -1 or single cluster, silhouette fails.
        if len(set(labels)) < 2:
            row = {"method": name, "k": args.k, "silhouette": "", "davies_bouldin": "", "ari": ""}
        else:
            m = evaluate_clustering_medium(Z, labels, y_true=y_true)
            row = {"method": name, "k": args.k, **m}
        append_metrics_csv(metrics_csv, row)

    # ---- AUDIO latent only ----
    labels_k = kmeans_cluster(Z_audio, n_clusters=args.k, seed=args.seed)
    np.save(os.path.join(out_dir, "labels_audio_kmeans.npy"), labels_k.astype(np.int32))
    eval_and_save("AudioConvVAE+KMeans", Z_audio, labels_k)
    save_tsne_plot(
        Z_audio, labels_k,
        os.path.join(out_dir, "tsne_audio_kmeans.png"),
        "t-SNE: Audio ConvVAE latent (KMeans)",
        seed=args.seed
    )

    labels_ag = agglomerative_cluster(Z_audio, n_clusters=args.k)
    np.save(os.path.join(out_dir, "labels_audio_agglo.npy"), labels_ag.astype(np.int32))
    eval_and_save("AudioConvVAE+Agglo", Z_audio, labels_ag)
    save_tsne_plot(
        Z_audio, labels_ag,
        os.path.join(out_dir, "tsne_audio_agglo.png"),
        "t-SNE: Audio ConvVAE latent (Agglo)",
        seed=args.seed
    )

    labels_db = dbscan_cluster(Z_audio, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    np.save(os.path.join(out_dir, "labels_audio_dbscan.npy"), labels_db.astype(np.int32))
    eval_and_save("AudioConvVAE+DBSCAN", Z_audio, labels_db)
    save_tsne_plot(
        Z_audio, labels_db,
        os.path.join(out_dir, "tsne_audio_dbscan.png"),
        "t-SNE: Audio ConvVAE latent (DBSCAN)",
        seed=args.seed
    )

    # ---- FUSED (audio + lyrics) ----
    labels_fk = kmeans_cluster(Z_fused, n_clusters=args.k, seed=args.seed)
    np.save(os.path.join(out_dir, "labels_fused_kmeans.npy"), labels_fk.astype(np.int32))
    eval_and_save("Fused(Audio+Lyrics)+KMeans", Z_fused, labels_fk)
    save_tsne_plot(
        Z_fused, labels_fk,
        os.path.join(out_dir, "tsne_fused_kmeans.png"),
        "t-SNE: Fused (KMeans)",
        seed=args.seed
    )

    labels_fa = agglomerative_cluster(Z_fused, n_clusters=args.k)
    np.save(os.path.join(out_dir, "labels_fused_agglo.npy"), labels_fa.astype(np.int32))
    eval_and_save("Fused(Audio+Lyrics)+Agglo", Z_fused, labels_fa)
    save_tsne_plot(
        Z_fused, labels_fa,
        os.path.join(out_dir, "tsne_fused_agglo.png"),
        "t-SNE: Fused (Agglo)",
        seed=args.seed
    )

    labels_fd = dbscan_cluster(Z_fused, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    np.save(os.path.join(out_dir, "labels_fused_dbscan.npy"), labels_fd.astype(np.int32))
    eval_and_save("Fused(Audio+Lyrics)+DBSCAN", Z_fused, labels_fd)
    save_tsne_plot(
        Z_fused, labels_fd,
        os.path.join(out_dir, "tsne_fused_dbscan.png"),
        "t-SNE: Fused (DBSCAN)",
        seed=args.seed
    )

    # 6) Save config
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nDONE ✅ Medium finished (results/medium/)")

def run_hard(args):
    """
    Hard (from handout):
    - Beta-VAE or CVAE
    - Multi-modal clustering (audio + lyrics + genre info)
    - Metrics: Silhouette, NMI, ARI, Purity
    - Visualizations: latent plots + cluster distributions over languages/genres + recon examples
    - Baselines: PCA+KMeans, Autoencoder+KMeans, direct spectral clustering (MFCC)
    """
    out_dir = "results/hard"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load aligned data produced by Medium feature builder ----
    aligned_csv = "data/cache/lyrics_audio_aligned.csv"
    if not os.path.exists(aligned_csv):
        raise FileNotFoundError(
            "Missing data/cache/lyrics_audio_aligned.csv. Run medium once to build aligned cache."
        )

    df = pd.read_csv(aligned_csv)
    if "genre" not in df.columns:
        raise ValueError("aligned CSV must contain a 'genre' column.")

    # y_true as integers
    genres = sorted(df["genre"].unique().tolist())
    genre_to_id = {g: i for i, g in enumerate(genres)}
    y_true = df["genre"].map(genre_to_id).values.astype(int)

    # ---- Load cached features (must match 998 rows) ----
    X_audio = np.load("data/cache/audio_melspec.npy")  # (N,1,128,1024)
    Z_lyrics = np.load("data/cache/lyrics_emb.npy")    # (N,lyrics_dim)

    # sanity
    N = len(df)
    if X_audio.shape[0] != N or Z_lyrics.shape[0] != N:
        raise ValueError(f"Shape mismatch: aligned rows={N}, audio={X_audio.shape[0]}, lyrics={Z_lyrics.shape[0]}")

    # ---- 1) Audio latent (ConvVAE) from your existing training.py ----
    # Reuse medium's convVAE trainer (already in your pipeline)
    _, Z_audio = train_conv_vae(
        X_audio,
        out_dir=out_dir,
        latent_dim=args.latent_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    np.save(os.path.join(out_dir, "latent_audio_vae.npy"), Z_audio.astype(np.float32))

    # ---- 2) Fuse audio+lyrics ----
    Z_fused = fuse_concat(Z_audio, Z_lyrics)  # (N, latent_dim + lyrics_dim)
    np.save(os.path.join(out_dir, "latent_fused_raw.npy"), Z_fused.astype(np.float32))

    # optional: add genre one-hot into fused input (this is the “genre information” in features)
    if args.use_genre_feature:
        G = np.eye(len(genres), dtype=np.float32)[y_true]  # (N, num_genres)
        X_fused_in = np.concatenate([Z_fused, G], axis=1)
    else:
        X_fused_in = Z_fused

    # standardize for MLP models
    scaler = StandardScaler()
    X_fused_in = scaler.fit_transform(X_fused_in).astype(np.float32)
    np.save(os.path.join(out_dir, "fused_input.npy"), X_fused_in)

    # helper: clustering + save metrics
    metrics_csv = os.path.join(out_dir, "metrics.csv")

    def write_row(method: str, Z: np.ndarray, labels: np.ndarray):
        m = evaluate_hard(Z, labels, y_true=y_true)
        append_metrics_csv(metrics_csv, {"method": method, "k": args.k, **m})

    # ---- 3) Baseline A: PCA + KMeans on fused_input ----
    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.latent_dim, random_state=args.seed)
    Z_pca = pca.fit_transform(X_fused_in)
    labels_pca = kmeans_cluster(Z_pca, n_clusters=args.k, seed=args.seed)
    np.save(os.path.join(out_dir, "labels_pca_kmeans.npy"), labels_pca.astype(np.int32))
    write_row("PCA(Fused)+KMeans", Z_pca, labels_pca)
    save_tsne_plot(Z_pca, labels_pca, os.path.join(out_dir, "tsne_pca.png"), "t-SNE: PCA(Fused) + KMeans", seed=args.seed)

    # ---- 4) Baseline B: Autoencoder + KMeans ----
    import torch.nn.functional as F
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AutoEncoder_MLP(in_dim=X_fused_in.shape[1], hidden_dim=args.hidden_dim, bottleneck_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=args.lr)

    X_t = torch.tensor(X_fused_in, dtype=torch.float32).to(device)
    ae.train()
    for ep in range(1, args.ae_epochs + 1):
        opt.zero_grad()
        xhat, z = ae(X_t)
        loss = F.mse_loss(xhat, X_t)
        loss.backward()
        opt.step()
        if ep == 1 or ep % 5 == 0 or ep == args.ae_epochs:
            print(f"[hard][AE] epoch {ep:02d}/{args.ae_epochs} | loss={loss.item():.6f}")

    ae.eval()
    with torch.no_grad():
        _, Z_ae = ae(X_t)
    Z_ae = Z_ae.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "latent_ae.npy"), Z_ae.astype(np.float32))

    labels_ae = kmeans_cluster(Z_ae, n_clusters=args.k, seed=args.seed)
    np.save(os.path.join(out_dir, "labels_ae_kmeans.npy"), labels_ae.astype(np.int32))
    write_row("Autoencoder(Fused)+KMeans", Z_ae, labels_ae)
    save_tsne_plot(Z_ae, labels_ae, os.path.join(out_dir, "tsne_ae.png"), "t-SNE: Autoencoder(Fused) + KMeans", seed=args.seed)

    # ---- 5) Baseline C: Direct spectral-feature clustering ----
    # Try MFCC if available+valid; otherwise fall back to MelSpectrogram (which you already have).
    mfcc_path = "data/cache/audio_mfcc.npy"

    X_spec = None
    spec_name = None

    if os.path.exists(mfcc_path) and os.path.getsize(mfcc_path) > 0:
        try:
            X_mfcc = np.load(mfcc_path)
            X_spec = X_mfcc.reshape(X_mfcc.shape[0], -1).astype(np.float32)
            spec_name = "MFCC"
        except Exception as e:
            print(f"[hard] Could not load {mfcc_path} ({e}); falling back to MelSpec.")
            X_spec = None

    if X_spec is None:
        # Use mel-spectrogram features directly as the "spectral feature" baseline
        X_spec = X_audio.reshape(X_audio.shape[0], -1).astype(np.float32)
        spec_name = "MelSpec"

    X_spec = StandardScaler().fit_transform(X_spec).astype(np.float32)

    pca_m = PCA(n_components=min(64, X_spec.shape[1]), random_state=args.seed)
    Z_spec = pca_m.fit_transform(X_spec)

    labels_spec = kmeans_cluster(Z_spec, n_clusters=args.k, seed=args.seed)
    np.save(os.path.join(out_dir, f"labels_{spec_name.lower()}_kmeans.npy"), labels_spec.astype(np.int32))

    write_row(f"DirectSpectral({spec_name})+KMeans", Z_spec, labels_spec)
    save_tsne_plot(
        Z_spec, labels_spec,
        os.path.join(out_dir, f"tsne_{spec_name.lower()}.png"),
        f"t-SNE: {spec_name} PCA + KMeans",
        seed=args.seed
    )

    # ---- 6) Beta-VAE on fused_input ----
    beta = float(args.beta)
    model = BetaVAE_MLP(in_dim=X_fused_in.shape[1], hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for ep in range(1, args.epochs + 1):
        opt.zero_grad()
        xhat, mu, logvar, z = model(X_t)
        recon = F.mse_loss(xhat, X_t)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + beta * kl
        loss.backward()
        opt.step()
        if ep == 1 or ep % 5 == 0 or ep == args.epochs:
            print(f"[hard][BetaVAE] epoch {ep:02d}/{args.epochs} | loss={loss.item():.6f} | recon={recon.item():.6f} | kl={kl.item():.6f}")

    model.eval()
    with torch.no_grad():
        _, _, _, Z_bvae = model(X_t)
    Z_bvae = Z_bvae.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "latent_beta_vae.npy"), Z_bvae.astype(np.float32))
    torch.save(model.state_dict(), os.path.join(out_dir, "beta_vae_mlp.pt"))

    labels_bvae = kmeans_cluster(Z_bvae, n_clusters=args.k, seed=args.seed)
    np.save(os.path.join(out_dir, "labels_beta_vae_kmeans.npy"), labels_bvae.astype(np.int32))
    write_row("BetaVAE(Fused)+KMeans", Z_bvae, labels_bvae)
    save_tsne_plot(Z_bvae, labels_bvae, os.path.join(out_dir, "tsne_beta_vae.png"), "t-SNE: BetaVAE(Fused) + KMeans", seed=args.seed)

    # ---- 7) Optional CVAE conditioned on genre ----
    if args.use_cvae:
        cond = np.eye(len(genres), dtype=np.float32)[y_true]  # (N, num_genres)
        C_t = torch.tensor(cond, dtype=torch.float32).to(device)

        cvae = CVAE_MLP(in_dim=X_fused_in.shape[1], cond_dim=cond.shape[1], hidden_dim=args.hidden_dim, latent_dim=args.latent_dim).to(device)
        opt = torch.optim.Adam(cvae.parameters(), lr=args.lr)

        cvae.train()
        for ep in range(1, args.epochs + 1):
            opt.zero_grad()
            xhat, mu, logvar, z = cvae(X_t, C_t)
            recon = F.mse_loss(xhat, X_t)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl
            loss.backward()
            opt.step()
            if ep == 1 or ep % 5 == 0 or ep == args.epochs:
                print(f"[hard][CVAE] epoch {ep:02d}/{args.epochs} | loss={loss.item():.6f} | recon={recon.item():.6f} | kl={kl.item():.6f}")

        cvae.eval()
        with torch.no_grad():
            _, _, _, Z_cvae = cvae(X_t, C_t)
        Z_cvae = Z_cvae.detach().cpu().numpy()
        np.save(os.path.join(out_dir, "latent_cvae.npy"), Z_cvae.astype(np.float32))
        torch.save(cvae.state_dict(), os.path.join(out_dir, "cvae_mlp.pt"))

        labels_cvae = kmeans_cluster(Z_cvae, n_clusters=args.k, seed=args.seed)
        np.save(os.path.join(out_dir, "labels_cvae_kmeans.npy"), labels_cvae.astype(np.int32))
        write_row("CVAE(Fused|Genre)+KMeans", Z_cvae, labels_cvae)
        save_tsne_plot(Z_cvae, labels_cvae, os.path.join(out_dir, "tsne_cvae.png"), "t-SNE: CVAE(Fused|Genre) + KMeans", seed=args.seed)

        # cluster distributions for CVAE result
        save_cluster_distribution(aligned_csv, labels_cvae, os.path.join(out_dir, "cluster_vs_genre_cvae.png"),
                                  group_col="genre", title="Cluster distribution over genres (CVAE)")
        # save_cluster_distribution(aligned_csv, labels_cvae, os.path.join(out_dir, "cluster_vs_language_cvae.png"),
        #                           group_col="language", title="Cluster distribution over languages (CVAE)")

    # ---- 8) Cluster distributions (for BetaVAE result) ----
    save_cluster_distribution(aligned_csv, labels_bvae, os.path.join(out_dir, "cluster_vs_genre_betaVAE.png"),
                              group_col="genre", title="Cluster distribution over genres (BetaVAE)")
    # save_cluster_distribution(aligned_csv, labels_bvae, os.path.join(out_dir, "cluster_vs_language_betaVAE.png"),
    #                           group_col="language", title="Cluster distribution over languages (BetaVAE)")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nDONE ✅ Hard finished (results/hard/)")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="easy", choices=["easy", "medium", "hard"])

    # Easy (meta.csv)
    p.add_argument("--meta", type=str, default="data/meta.csv")
    p.add_argument("--max-features", type=int, default=5000)

    # Medium (lyrics.csv that matches audio tracks)
    p.add_argument("--lyrics-csv", type=str, default="data/lyrics/lyrics.csv")
    p.add_argument("--lyrics-dim", type=int, default=128)

    # shared
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)

    # DBSCAN
    p.add_argument("--dbscan-eps", type=float, default=0.8)
    p.add_argument("--dbscan-min-samples", type=int, default=10)

    # Hard options (MUST be defined before parse_args)
    p.add_argument("--beta", type=float, default=4.0)          # Beta-VAE strength
    p.add_argument("--use-cvae", action="store_true")          # if set: run CVAE version too
    p.add_argument("--use-genre-feature", action="store_true") # if set: concatenate one-hot genre to fused input
    p.add_argument("--ae-epochs", type=int, default=20)        # autoencoder baseline epochs

    args = p.parse_args()

    if args.task == "easy":
        run_easy(args)
    elif args.task == "medium":
        run_medium(args)
    else:
        run_hard(args)


if __name__ == "__main__":
    main()
