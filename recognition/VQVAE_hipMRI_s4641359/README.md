# VQ-VAE + PixelCNN for Synthetic HipMRI Prostate Slices

**Author:** Nooldes (s4641359) · COMP3710 Pattern Analysis Project 2026 · Difficulty: **Hard** (VQ-VAE, Section 2.4)

## 1. Overview

### Problem

### How it works


### Key results


## 2. Feasibility Review


## 3. Data and Pre-processing

### Dataset


### Pre-processing


### Splits and justification

| Split | Slices | Patients |
|---|---|---|
| Train | 11,460 | 31 |
| Validate | 660 | 4 |
| Test | 540 | 3 |


## 4. Method

### Baseline: convolutional VAE


### VQ-VAE


### PixelCNN prior


### Final configuration and hyperparameters

| Model | Grid | Epochs | Batch | LR | Other |
|---|---|---|---|---|---|
| VAE baseline | - | 20 | 32 | 1e-3 | latent 256, KL weight 1 |
| VQ-VAE (main) | 64x32 | 15 | 32 | 2e-4 | K 512, D 64, beta 0.25 |
| VQ-VAE (ablation) | 32x16 | 12 | 32 | 2e-4 | n_down 3 |
| PixelCNN prior (final) | 64x32 | 10 (best 8) | 32 | 1e-3 | 64 ch, 60 blocks, position embedding, T = 1.0 |



## 5. Results

### Training


### Reconstruction: baseline vs VQ-VAE (identical test split)

| Model | Test SSIM | Params | Codebook perplexity |
|---|---|---|---|
| VAE baseline | 0.506 | 13.45 M | - |
| VQ-VAE 64x32 | **0.917** | 0.86 M | ~386 / 512 |
| VQ-VAE 32x16 | 0.778 | 0.99 M | ~419 / 512 |


### Per-slice and per-patient analysis

| Test patient | Slices | Mean SSIM (64x32) | Mean SSIM (32x16) |
|---|---|---|---|
| 040 | 420 | 0.920 | 0.781 |
| 041 | 60 | 0.915 | 0.773 |
| 042 | 60 | 0.904 | 0.759 |



### Generation: baseline vs VQ-VAE

| Slices | FID (Inception) ↓ | FD (VQ-VAE features) ↓ | Diversity: mean pairwise SSIM ↓ | NN-SSIM to train (mean / max) |
|---|---|---|---|---|
| VAE baseline samples | 420.7 | 0.17 | 0.519 | 0.436 / 0.542 |
| VQ-VAE + prior samples | **234.0** | 0.04 | **0.272** | 0.344 / 0.478 |
| Real train slices (reference) | 77.4 | 0.02 | 0.225 | n/a |
| VQ-VAE reconstructions of train (reference) | 108.8 | 0.01 | 0.250 | 0.908 / 0.949 |


### Resource profiling

| Model / task | Params | Peak VRAM | Time | Hardware |
|---|---|---|---|---|
| VAE baseline | 13.45 M | 0.47 GB | 48 s / epoch | RTX 2050 (laptop, 4 GB) |
| VQ-VAE 64x32 | 0.86 M | 0.67 GB | 113 s / epoch | RTX 2050 |
| VQ-VAE 32x16 | 0.99 M | 0.39 GB | 60 s / epoch | RTX 2050 |
| PixelCNN 32x16, 24 blocks + pos | 2.15 M | 0.53 GB | ~38 s / epoch; 64 samples in 18 s | RTX 2050 |
| **PixelCNN 64x32, 60 blocks + pos (final)** | 4.96 M | 4.26 GB | 92 s / epoch; 100 samples in 76 s | Colab A100 |
| Audit NN search (300 x 11,460 SSIM pairs) | - | - | 16 s | Colab A100 |


## 6. Engineering Dilemma Investigation

### Improving the prior: a sequence of diagnosed fixes

| Step | Prior | Rows visible | Test bits/code | What the samples showed |
|---|---|---|---|---|
| 1 | 64x32, 10 blocks | 13 / 64 | 5.11 | Horizontal streaks, no global layout |
| 2 | 32x16, 10 blocks | 13 / 32 | 5.84* | More coherent, blurrier |
| 3 | 32x16, 24 blocks | 27 / 32 | 5.76* | Bladder more often central, fewer bands |
| 4 | 64x32, 60 blocks | 63 / 64 | 4.90 | Better likelihood but WORSE samples (bands) |
| 5 | 32x16, 24 blocks + position | 27 / 32 | 5.71* | Bands gone, consistent body outline |
| 6 | **64x32, 60 blocks + position** | 63 / 64 | **4.88** | Sharp, plausible layout (bladder, rectum, round structures) |

### Sampling temperature

### Memorisation audit

| Group (100 slices) | Mean NN-SSIM | Max NN-SSIM | Mean code overlap | Max code overlap | Distinct NN patients |
|---|---|---|---|---|---|
| Generated | 0.344 | 0.447 | 6.4% | 11.6% | 18 |
| Test (VQ-VAE reconstructed) | 0.383 | 0.443 | 8.4% | 11.2% | 21 |
| Test (raw) | 0.366 | 0.432 | 8.4% | 11.2% | 21 |

### Latent interpolation

### Failure autopsy (3-5 cases)

### Recommendation to the project manager

## 7. Limitations and Future Work

## 8. Usage

### Dependencies

| Package | Version |
|---|---|
| Python | |
| torch | |
| nibabel | |
| numpy | |
| scipy | |
| matplotlib | |
| tqdm | |
| torchmetrics[image] (evaluate.py only) | |

### Reproducing the results

### Project structure

| File | Purpose |
|---|---|
| `dataset.py` | Nifti loading, scaling, splits, data audit |
| `modules.py` | VAE, VQ-VAE, PixelCNN prior (with optional position embedding) |
| `train.py` | Training / validation / testing for all three models |
| `predict.py` | Inference, reconstruction figures, per-slice / per-patient SSIM, sampling |
| `audit.py` | Memorisation audit: SSIM nearest neighbours, code overlap, patients |
| `evaluate.py` | Baseline vs VQ-VAE generation: FID, diversity, NN-SSIM, interpolation |
| `utils.py` | SSIM, checkpoints, plotting |
| `figures/` | Images used in this README |

## 9. Artificial Intelligence Usage Disclosure
Required (spec 6.1). Three parts:
**Tools:** Claude (Anthropic), used conversationally throughout.
**What AI did:** generated most of the code (list per file; each file also has
a header), explained concepts.
**What I did:** chose the problem/model/baseline; ran all training (laptop +
Colab); diagnosed issues and proposed changes (temperature hypothesis,
position/receptive-field reasoning, L1+SSIM and decoder-conv ideas); designed
the audit steps, interpreted all results, wrote this README.
**How I verified:** insert at the end

## 10. References
insert here at the end.