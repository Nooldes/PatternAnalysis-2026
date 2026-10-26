"""
train.py - Train, validate and test models on HipMRI 2D slices.

Currently supports the VAE baseline. Each epoch it logs train/val loss
and validation SSIM, saves the best weights (by val SSIM), and at the end
reports test SSIM, parameter count, peak GPU memory and time per epoch.

Smoke test (quick check that everything runs):
    python train.py --model vae --data_root "path/to/keras_slices_data" --epochs 2

Outputs go to --out_dir (default: outputs/<model>/), which is git-ignored.
"""

import argparse
import json
import os
import time

import torch

from dataset import get_dataloaders
from modules import VAE, vae_loss
from utils import count_params, plot_history, save_image_grid, ssim


def run_epoch(model, loader, device, optimiser=None, beta=1.0):
    """One pass over a loader. Trains if an optimiser is given, else evaluates.

    Returns (mean loss, mean SSIM between inputs and reconstructions).
    """
    training = optimiser is not None
    model.train(training)
    total_loss, total_ssim, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(training):
        for x in loader:
            x = x.to(device, non_blocking=True)
            recon, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon, x, mu, logvar, beta)
            if training:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * x.size(0)
            total_ssim += ssim(recon.detach(), x).sum().item()
            n += x.size(0)
    return total_loss / n, total_ssim / n


def main():
    parser = argparse.ArgumentParser(description="Train a model on HipMRI 2D slices.")
    parser.add_argument("--model", default="vae", choices=["vae"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0, help="KL weight for the VAE")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.out_dir or os.path.join("outputs", args.model)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}")

    loaders = get_dataloaders(args.data_root, batch_size=args.batch_size)
    print({split: len(loader.dataset) for split, loader in loaders.items()})

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"Trainable parameters: {count_params(model):,}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train_loss": [], "val_loss": [], "val_ssim": []}
    best_ssim, epoch_times = -1.0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _ = run_epoch(model, loaders["train"], device, optimiser, args.beta)
        val_loss, val_ssim = run_epoch(model, loaders["validate"], device, beta=args.beta)
        epoch_times.append(time.time() - start)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_ssim"].append(val_ssim)
        print(f"Epoch {epoch:3d} | train loss {train_loss:9.2f} | "
              f"val loss {val_loss:9.2f} | val SSIM {val_ssim:.4f} | "
              f"{epoch_times[-1]:.1f}s")

        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    plot_history(history, os.path.join(out_dir, "training_curves.png"))

    # Test with the best weights.
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    test_loss, test_ssim = run_epoch(model, loaders["test"], device, beta=args.beta)

    # Example reconstructions and samples for a quick visual check.
    model.eval()
    with torch.no_grad():
        x = next(iter(loaders["test"]))[:8].to(device)
        recon, _, _ = model(x)
        save_image_grid(torch.cat([x, recon]), os.path.join(out_dir, "reconstructions.png"),
                        title="Top: test input | Bottom: reconstruction")
        save_image_grid(model.sample(16, device), os.path.join(out_dir, "samples.png"),
                        title="Random samples")

    peak_vram = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else 0.0
    results = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "params": count_params(model),
        "best_val_ssim": best_ssim,
        "test_loss": test_loss,
        "test_ssim": test_ssim,
        "peak_vram_gb": round(peak_vram, 3),
        "mean_epoch_seconds": round(sum(epoch_times) / len(epoch_times), 1),
    }
    print(json.dumps(results, indent=2))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()