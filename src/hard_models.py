# src/hard_models.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class BetaVAE_MLP(nn.Module):
    """
    Simple MLP Beta-VAE for vector inputs (e.g., fused audio+lyrics(+genre)).
    """
    def __init__(self, in_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        xhat = self.dec(z)
        return xhat, mu, logvar, z


class CVAE_MLP(nn.Module):
    """
    Conditional VAE (CVAE) for vector inputs.
    Condition c is one-hot (genre, language, etc.)
    We concatenate condition to encoder input and decoder input.
    """
    def __init__(self, in_dim: int, cond_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.cond_dim = cond_dim

        self.enc = nn.Sequential(
            nn.Linear(in_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        # x: [B,in_dim], c: [B,cond_dim]
        h = self.enc(torch.cat([x, c], dim=1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        xhat = self.dec(torch.cat([z, c], dim=1))
        return xhat, mu, logvar, z


class AutoEncoder_MLP(nn.Module):
    """
    Deterministic autoencoder baseline for Autoencoder+KMeans.
    """
    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z
