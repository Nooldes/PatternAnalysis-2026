"""
modules.py - Model components for the HipMRI generative project.

Currently contains the convolutional VAE baseline (Kingma & Welling, 2014).
The VQ-VAE and PixelCNN prior are added alongside it so both models share
the same encoder/decoder style and can be compared fairly.

Inputs are single-channel 256x128 slices scaled to [0, 1].
Pure PyTorch: no NumPy is used in this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def down_block(in_ch, out_ch):
    """Conv layer that halves height and width (stride 2)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


def up_block(in_ch, out_ch):
    """Transposed conv layer that doubles height and width."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VAE(nn.Module):
    """Convolutional VAE baseline with a flat Gaussian latent.

    Encoder: 4 stride-2 convs, 256x128 -> 16x8 feature map, then linear
    layers predicting the latent mean and log-variance.
    Decoder: mirror of the encoder, sigmoid output in [0, 1].
    """

    def __init__(self, latent_dim=256, base_ch=32, in_shape=(256, 128)):
        super().__init__()
        self.latent_dim = latent_dim
        chs = [base_ch, base_ch * 2, base_ch * 4, base_ch * 4]
        self.feat_shape = (chs[-1], in_shape[0] // 16, in_shape[1] // 16)
        feat_size = chs[-1] * self.feat_shape[1] * self.feat_shape[2]

        self.encoder = nn.Sequential(
            down_block(1, chs[0]),
            down_block(chs[0], chs[1]),
            down_block(chs[1], chs[2]),
            down_block(chs[2], chs[3]),
        )
        self.fc_mu = nn.Linear(feat_size, latent_dim)
        self.fc_logvar = nn.Linear(feat_size, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, feat_size)
        self.decoder = nn.Sequential(
            up_block(chs[3], chs[2]),
            up_block(chs[2], chs[1]),
            up_block(chs[1], chs[0]),
            nn.ConvTranspose2d(chs[0], 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        # Clamp log-variance so exp(logvar) can't overflow and blow up the KL term.
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10.0, 10.0)

    def reparameterise(self, mu, logvar):
        """Sample z = mu + sigma * eps so gradients flow through mu and sigma.

        At evaluation the mean is used instead, giving deterministic
        reconstructions (as in the course CNN-VAE demo).
        """
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, *self.feat_shape)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n, device):
        """Generate n new slices by decoding z ~ N(0, I)."""
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """VAE loss = reconstruction (summed MSE per image) + beta * KL divergence.

    Both terms are summed over pixels/latent dims and averaged over the batch.
    Returns (total, reconstruction, kl) so each can be logged.
    """
    batch = x.size(0)
    recon_loss = F.mse_loss(recon, x, reduction="sum") / batch
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch
    return recon_loss + beta * kl, recon_loss, kl