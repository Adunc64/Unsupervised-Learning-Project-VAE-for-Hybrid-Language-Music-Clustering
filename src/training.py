# src/training.py
from __future__ import annotations

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import NumpyFeatureDataset
from src.models import VAE


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_vae(
    X: np.ndarray,
    out_dir: str = "results/easy",
    hidden_dim: int = 512,
    latent_dim: int = 16,
    lr: float = 1e-3,
    batch_size: int = 64,
    epochs: int = 30,
    beta: float = 1.0,
    seed: int = 42,
) -> tuple[VAE, np.ndarray]:
    """
    Returns:
      - trained VAE
      - latent features (mu) shape (N, latent_dim)
    """
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = NumpyFeatureDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = VAE(input_dim=X.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0

        for batch in tqdm(dl, desc=f"[train] epoch {epoch}/{epochs}", leave=False):
            batch = batch.to(device)

            recon, mu, logvar = model(batch)
            loss, recon_loss, kl = VAE.loss_fn(batch, recon, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch.size(0)
            total_recon += recon_loss.item() * batch.size(0)
            total_kl += kl.item() * batch.size(0)

        n = len(ds)
        print(
            f"Epoch {epoch:02d} | loss={total_loss/n:.6f} | recon={total_recon/n:.6f} | kl={total_kl/n:.6f}"
        )

    # Save model
    model_path = os.path.join(out_dir, "vae_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[training] Saved VAE model: {model_path}")

    # Encode all -> latent
    Z = encode_mu(model, X, batch_size=256)
    z_path = os.path.join(out_dir, "latent_vae.npy")
    np.save(z_path, Z.astype(np.float32))
    print(f"[training] Saved latent vectors: {z_path} | shape={Z.shape}")

    return model, Z


@torch.no_grad()
def encode_mu(model: VAE, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()

    ds = NumpyFeatureDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    mus = []
    for batch in dl:
        batch = batch.to(device)
        h = model.enc(batch)
        mu = model.mu(h)
        mus.append(mu.cpu().numpy())

    return np.concatenate(mus, axis=0)

##medium##
# Add to src/training.py

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import SpectrogramDataset
from src.models import ConvVAE


def train_conv_vae(
    X: np.ndarray,
    out_dir: str = "results/medium",
    latent_dim: int = 16,
    lr: float = 1e-3,
    batch_size: int = 32,
    epochs: int = 20,
    beta: float = 1.0,
    seed: int = 42,
) -> tuple[ConvVAE, np.ndarray]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SpectrogramDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ConvVAE(in_channels=X.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total, total_r, total_kl = 0.0, 0.0, 0.0
        for batch in tqdm(dl, desc=f"[conv-vae] {epoch}/{epochs}", leave=False):
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, r, kl = ConvVAE.loss_fn(batch, recon, mu, logvar, beta=beta)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item() * batch.size(0)
            total_r += r.item() * batch.size(0)
            total_kl += kl.item() * batch.size(0)

        n = len(ds)
        print(f"Epoch {epoch:02d} | loss={total/n:.6f} | recon={total_r/n:.6f} | kl={total_kl/n:.6f}")

    # Save
    model_path = os.path.join(out_dir, "conv_vae_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[training] Saved ConvVAE: {model_path}")

    Z = encode_mu_conv(model, X, batch_size=128)
    z_path = os.path.join(out_dir, "latent_audio_vae.npy")
    np.save(z_path, Z.astype(np.float32))
    print(f"[training] Saved audio latent: {z_path} | shape={Z.shape}")

    return model, Z


@torch.no_grad()
def encode_mu_conv(model: ConvVAE, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()

    ds = SpectrogramDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)

    mus = []
    for batch in dl:
        batch = batch.to(device)
        recon, mu, logvar = model(batch)
        mus.append(mu.cpu().numpy())

    return np.concatenate(mus, axis=0)
