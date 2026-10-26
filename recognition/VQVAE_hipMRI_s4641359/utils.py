"""
utils.py - Shared helpers: SSIM metric, parameter counting, plotting.

SSIM follows Wang et al. (2004), "Image quality assessment: from error
visibility to structural similarity", using an 11x11 Gaussian window
(sigma 1.5) on images scaled to [0, 1].
"""

import os

import matplotlib
matplotlib.use("Agg")  # save figures without needing a display (works on Rangpur)
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def _gaussian_window(size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]  # (1, 1, size, size)


def ssim(x, y, data_range=1.0):
    """Mean SSIM per image for batches of shape (N, 1, H, W). Returns (N,)."""
    window = _gaussian_window(device=x.device)
    pad = window.shape[-1] // 2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2

    mu_x = F.conv2d(x, window, padding=pad)
    mu_y = F.conv2d(y, window, padding=pad)
    var_x = F.conv2d(x * x, window, padding=pad) - mu_x ** 2
    var_y = F.conv2d(y * y, window, padding=pad) - mu_y ** 2
    cov = F.conv2d(x * y, window, padding=pad) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim_map.flatten(1).mean(dim=1)


def count_params(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_history(history, out_path):
    """Plot train/val loss and val SSIM per epoch to a PNG."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="validate")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend(); ax1.set_title("Loss")
    ax2.plot(epochs, history["val_ssim"], color="tab:green")
    ax2.axhline(0.6, linestyle="--", color="grey", label="target 0.6")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("SSIM"); ax2.legend(); ax2.set_title("Validation SSIM")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def save_image_grid(images, out_path, ncols=8, title=None):
    """Save a grid of (N, 1, H, W) images in [0, 1] to a PNG."""
    images = images.detach().cpu()
    n = images.size(0)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 2.2))
    for i, ax in enumerate(axes.flat if n > 1 else [axes]):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i, 0], cmap="gray", vmin=0, vmax=1)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)