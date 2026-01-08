# src/models.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, latent_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # TF-IDF is ~[0,1], sigmoid keeps recon bounded
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        recon = self.dec(z)
        return recon, mu, logvar

    @staticmethod
    def loss_fn(x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        # KL divergence
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, recon_loss, kl


###medium##
# Add below your existing VAE class in src/models.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    """
    Conv VAE for (N, 1, H, W) mel-spectrograms normalized to [0,1].
    """
    def __init__(self, in_channels: int = 1, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: downsample
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),           # /4
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),          # /8
            nn.ReLU(),
        )

        # We'll initialize these after we see input shape once
        self._enc_out_dim = None
        self.fc_mu = None
        self.fc_logvar = None
        self.fc_z = None

        # Decoder: upsample
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def _build_heads(self, x: torch.Tensor):
        with torch.no_grad():
            h = self.enc(x)
            self._enc_shape = h.shape[1:]  # (C, H, W)
            self._enc_out_dim = int(h.numel() / h.shape[0])

        self.fc_mu = nn.Linear(self._enc_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self._enc_out_dim, self.latent_dim)
        self.fc_z = nn.Linear(self.latent_dim, self._enc_out_dim)

        # Move to same device
        device = x.device
        self.fc_mu.to(device)
        self.fc_logvar.to(device)
        self.fc_z.to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.fc_mu is None:
            self._build_heads(x)

        h = self.enc(x)
        h_flat = h.view(h.size(0), -1)

        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)

        h2 = self.fc_z(z).view(x.size(0), *self._enc_shape)
        recon = self.dec(h2)
        return recon, mu, logvar

    @staticmethod
    def loss_fn(x, recon, mu, logvar, beta: float = 1.0):
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kl, recon_loss, kl
