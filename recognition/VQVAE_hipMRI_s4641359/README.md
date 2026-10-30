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

Each step below changed one thing, chosen by diagnosing the previous result. The table summarises steps 1–6; step 7 tests sampling temperature.

| Step | Prior | Rows visible | Test bits/code | What the samples showed |
|---|---|---|---|---|
| 1 | 64x32, 10 blocks | 13 / 64 | 5.11 | Horizontal streaks, no global layout |
| 2 | 32x16, 10 blocks | 13 / 32 | 5.84* | More coherent, blurrier |
| 3 | 32x16, 24 blocks | 27 / 32 | 5.76* | Bladder more often central, fewer bands |
| 4 | 64x32, 60 blocks | 63 / 64 | 4.90 | Better likelihood but worse samples (bands) |
| 5 | 32x16, 24 blocks + position | 27 / 32 | 5.71* | Bands gone, consistent body outline |
| 6 | **64x32, 60 blocks + position** | 63 / 64 | **4.88** | Sharp, plausible layout (bladder, ring, round structures) |

*\* 32×16 bits/code are not directly comparable with 64×32 (each code covers 4× more pixels).*

**Step 1: Limited reach (64×32, 10 blocks).** The first iteration of the prior showed realistic local tissue texture, but there were horizontal streaks and no body outline. The problem was that the prior's reach was too small for the MRI slices. The first layer is a 7×7 mask-A convolution (3 rows of reach) and each block adds a 3×3 mask-B convolution (1 more row), so the prior sees 3 + N rows. With 10 blocks that is only 13 of 64 code rows. As a result, samples had realistic local texture but no coherent body structure: lower rows had no connection with the upper rows, patterns continued sideways as horizontal streaks, and larger structures appeared partially formed, misplaced or lacking symmetry.

![Step 1: 64x32, 10 blocks](figures/samples_64x32_10blocks.png)

**Step 2: Coarser grid (32×16, 10 blocks).** As a first test, I switched to the 32×16 grid, where the same 13 rows cover 13 of 32. Samples became more coherent (a consistent top edge and a central bright organ), confirming that reach was the issue. However, they were blurrier, because of the 32×16 grid: the 32×16 VQ-VAE reconstructs at SSIM 0.778 vs 0.917 for 64×32.

![Step 2: 32x16, 10 blocks](figures/samples_32x16_10blocks.png)

**Step 3: Deeper, thinner prior (32×16, 24 blocks).** The next fix was to increase the depth (number of blocks) while reducing the channel width. Since parameters grow with the square of the width but only linearly with depth, I halved the width (128 → 64 channels) and increased depth (10 → 24 blocks). The receptive field doubled from 13 to 27 rows with half the parameters (4.07M → 2.14M); the prior also trained faster per epoch and overfit less (the best validation epoch moved from 4 to 6). The samples were more coherent: a more consistent top edge, fewer bands, and a body outline starting to form. The bright organ (most likely the bladder) also appeared more often than in step 2 (in about 25 vs 13 of 64 samples).

![Step 3: 32x16, 24 blocks](figures/samples_32x16_24blocks.png)

**Step 4: Full-height prior (64×32, 60 blocks).** Since a larger receptive field improved coherence on the 32×16 grid, I applied the same fix to the sharper 64×32 grid, increasing the prior to 60 blocks so it could see 63 of 64 rows. This was trained on a Google Colab A100 instead of my laptop, since it needed more than 4 GB of VRAM. It gave the best test likelihood so far (4.90 bits per code), meaning it predicted unseen code grids best. However, it produced the worst-looking samples: sharp textures, but many horizontal black bands, an unrecognisable body outline, and a bladder in only 3 of 64 samples. Seeing the whole image was not enough, and a better likelihood does not guarantee better samples, so receptive field was necessary but not sufficient.

![Step 4: 64x32, 60 blocks](figures/samples_64x32_60blocks.png)

**Step 5: Position embedding (32×16, 24 blocks).** The black bands persisted even at full receptive field, so the reach was not the cause. The bands also ran sideways across whole rows, which matches how the prior generates codes row by row. In the MRI slices, the background spans whole rows at the top and bottom edges, so background to the left predicts background at the current position. Without position information, a background code anywhere in the image continues across the row, and then into the following rows, since the prior now sees background above and to the left, forming horizontal bands. This led to the conclusion that the prior had no idea where in the image it was. This is a consequence of convolution's design: convolutions share the same weights at every position, which gives translation equivariance, a useful bias when a pattern can appear anywhere in an image. HipMRI slices are different: they are spatially aligned, so position predicts content well (background around the edges, organs near the centre). Here, translation equivariance removes information the prior needs. Transformers face the same issue, since attention is unaware of input order, and it is usually solved by adding positional encodings to their inputs. I applied the same idea: a learned row vector and column vector are added to each code's embedding before the first masked convolution. For the code at row x, column y, the input becomes embedding(code) + row(x) + col(y), adding around 8k parameters. Because it depends only on coordinates and never on code values, it does not let the prior see future codes. To isolate its effect, I first added it to the 32×16, 24-block prior with every other setting unchanged. The bands disappeared, every sample gained a consistent body outline with a central bladder, and validation loss improved slightly (4.256 → 4.231).

![Step 5: 32x16, 24 blocks + position](figures/samples_32x16_24blocks_pos.png)

**Step 6: Final model (64×32, 60 blocks + position).** For the final model, I combined depth and position embedding on the 64×32 grid, since the two fixes cover each other's weaknesses. Depth lets the prior coordinate structures across the whole slice, but without coordinates it got lost and produced horizontal black bands. Position embedding tells it which row and column it is generating, so it knows what belongs there, but it had only been tested on the blurrier 32×16 grid. The final samples showed sharper detail and a more coherent body layout: a bright organ with a dark ring around it, a dark round structure below it, sometimes round structures on the sides, and a consistent body outline without black bands. It also achieved the best likelihood overall (4.88 bits per code, val CE 3.653). Several problems remained: half-formed or ambiguous organs, smeared regions, no clear bones, and areas that don't make anatomical sense. These could be caused by the PixelCNN's blind spot: it cannot see certain regions no matter how many layers are added, so structures there cannot be coordinated. Both fixes addressed problems of configuration, but neither changes the PixelCNN's architecture, so its limitations remain: the blind spot at each position, the local bias of convolutions, and errors that compound over long sequences of decisions. The remaining problems are consistent with these limits, which motivate the future work in Section 7.

![Step 6: 64x32, 60 blocks + position (final)](figures/samples_64x32_60blocks_pos.png)

**Step 7: Sampling temperature.** Before concluding that the problems came from the architecture, I tested whether they came from sampling rather than from what the prior learned. At T = 1.0, the prior occasionally draws rare, low-probability codes, which I hypothesised might produce the smeared patches. Lowering the sampling temperature suppresses these codes without retraining, so it isolates sampling as a cause. At each position, the prior gives a score for each code; before picking one, the scores are divided by T and turned into probabilities. T = 1.0 samples exactly as the model learned, while T < 1 sharpens the probabilities, so the most likely codes become even more likely and rare codes are almost never picked. I compared T = 1.0 with T = 0.8 and 0.7. At T = 0.8, samples were darker, with less tissue texture and more samples looking alike; at T = 0.7, many slices were nearly empty and dark, with large bright blobs. The most common codes in real slices are background and dark tissue, so lower T pushed the prior towards those, making samples darker and less detailed. The bright blobs arise when the prior starts a bright region: at low T, the most likely next code is more of that region, so it keeps choosing it and the area grows too large. My hypothesis that sampling caused the problems did not hold. The rarer codes actually carry the tissue detail, so removing them made things worse. The remaining problems come from what the prior learned, not from sampling randomness, so T = 1.0 was kept.

![T = 1.0](figures/temp_1.0.png)
![T = 0.8](figures/temp_0.8.png)
![T = 0.7](figures/temp_0.7.png)
 
### Memorisation audit
![NN-SSIM histogram](figures/audit_hist.png)
![Closest generated slices](figures/audit_top3.png)
![Nearest-neighbour patients](figures/audit_patients.png)
 
| Group (100 slices) | Mean NN-SSIM | Max NN-SSIM | Mean code overlap | Max code overlap | Distinct NN patients |
|---|---|---|---|---|---|
| Generated | 0.344 | 0.447 | 6.4% | 11.6% | 18 |
| Test (VQ-VAE reconstructed) | 0.383 | 0.443 | 8.4% | 11.2% | 21 |
| Test (raw) | 0.366 | 0.432 | 8.4% | 11.2% | 21 |
 

 
### Latent interpolation
![Latent interpolation](figures/interpolation.png)

 
### Failure autopsy (3-5 cases)
![Worst reconstructions, 32x16](figures/worst_cases_32x16.png)

 
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