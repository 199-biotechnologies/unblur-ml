# UnblurML

**A proof of concept demonstrating that Gaussian-blurred text can be recovered using deep learning — and why visual redaction methods are no longer trustworthy.**

By [Boris Djordjevic](https://github.com/199-biotechnologies)

---

## Why This Matters

We are entering an era where the traditional methods of obscuring sensitive information — blurring, pixelation, mosaic filters, and even simple redaction bars — are becoming dangerously unreliable.

For decades, people have relied on Gaussian blur to hide text in screenshots: seed phrases, passwords, personal details, classified documents. The assumption was always that blur is a one-way operation — once the information is smeared, it's gone. **That assumption is wrong.**

This project demonstrates that a commodity deep learning model, trained in under two hours on a single laptop, can recover blurred text with surprising accuracy. The model doesn't reconstruct the image — it classifies the blurred shape directly against a known vocabulary, bypassing the need for deblurring entirely.

### The Broader Implications

This is not an isolated result. The same principle applies to:

- **Mosaic/pixelation filters** — research has already shown these can be reversed with neural networks, as pixel-block patterns retain spatial frequency information that models can exploit.
- **Redaction by overlaying black bars** — if the bars don't fully cover the text, even a single pixel of exposed character can leak information. Combined with known font metrics and character-length analysis, partial redactions can be reconstructed.
- **Blurred faces and licence plates** — GAN-based methods (like PULSE and GFPGAN) can hallucinate plausible reconstructions from heavily degraded inputs.
- **Government redactions** — the Epstein files and similar high-profile document releases have drawn attention to the fragility of redaction techniques. Researchers and journalists have been attempting to reconstruct what lies beneath blurred, pixelated, or partially redacted sections of declassified documents. The techniques demonstrated here apply directly to those efforts.

**The core insight is simple**: if you know the vocabulary (BIP-39 wordlist, common names, addresses, document templates), classification is far easier than reconstruction. You don't need to unblur the image — you just need to determine which word from a known list best matches the blur pattern.

And the barrier to building these tools is collapsing. This entire model — data generation, training, inference — was built and trained in a single afternoon using open-source libraries on consumer hardware. No cloud compute. No special datasets. No PhD required.

**If your security depends on visual obscuration, it is time to reconsider.**

---

## What This Project Does

UnblurML recovers Gaussian-blurred BIP-39 seed phrase words using a classification approach. Given a blurred image of a single word, the model classifies it as one of 2,048 entries in the BIP-39 English wordlist.

### Key Results

| Blur Sigma | Top-1 Accuracy | Top-5 Accuracy | Top-20 Accuracy |
|---|---|---|---|
| 3 (mild) | 100% | 100% | 100% |
| 5 (medium) | 99.5% | 100% | 100% |
| 7 (heavy) | 96.0% | 100% | 100% |
| 8 (very heavy) | 93.0% | 99.0% | 100% |
| 10 (extreme) | 79.0% | 94.0% | 98.0% |
| 12 (near-obliteration) | 51.0% | 75.5% | 87.0% |

Edge case robustness (with base blur sigma 3–8):

| Degradation | Top-5 Accuracy |
|---|---|
| Heavy crop (25% removed) | 96.0% |
| JPEG compression (q=20–40) | 97.3% |
| Grey/faded text (40% contrast) | 100% |
| Inverted colours (light on dark) | 100% |
| Downscale 50% + upscale | 100% |
| Combined worst case | 94.7% |

Training time: **70 minutes** on Apple M4 Max (64GB).

### What Makes This Approach Work

The key innovation is treating deblurring as **classification rather than reconstruction**. Since BIP-39 has exactly 2,048 words, we don't need to recover the original pixels — we simply need to identify which word produced the blur pattern. This transforms an ill-posed inverse problem into a tractable 2,048-class classification task.

---

## Technical Approach

### On-the-Fly Synthetic Data Generation

Rather than pre-generating a fixed dataset, training data is synthesised in real time during training. Each sample is created by:

1. **Rendering** a random BIP-39 word in a random font (Menlo, Courier, SF Mono, etc.), at a random size, with random positioning jitter
2. **Applying Gaussian blur** at a random sigma level (controlled by curriculum)
3. **Augmenting** with real-world degradations: JPEG compression, downscaling, affine transforms, coarse dropout, brightness/contrast variation, and Gaussian noise

This means the model never sees the same image twice. Each epoch presents entirely fresh samples, preventing overfitting and enabling unlimited effective dataset size.

#### Rendering Diversity

Training samples include diverse rendering conditions to match real-world screenshots:

- **Standard** dark text on light background (30%)
- **Low contrast** — text nearly invisible against background, matching heavily blurred screenshots (25%)
- **Grey/faded text** — medium contrast reduction (15%)
- **Coloured backgrounds** — tinted UI themes (10%)
- **Inverted** — light text on dark background (10%)
- **Dark theme with coloured text** (10%)

### Curriculum Learning

The model starts with easy examples and progressively increases difficulty. Critically, the **lower bound** of blur sigma rises over time, forcing the model to spend its training budget on hard cases rather than wasting epochs on trivially readable text:

| Phase | Blur Sigma | Purpose |
|---|---|---|
| Warmup | 0.5–3.0 | Frozen backbone, train classifier head only |
| Easy | 1.0–5.0 | Learn word shapes with mild blur |
| Medium | 2.0–8.0 | Generalise to moderate blur |
| Hard | 3.0–11.0 | No more easy samples — focus on difficult cases |
| Harder | 4.0–14.0 | Push into extreme territory |
| Very Hard | 5.0–16.0 | Near information-theoretic limit |
| Maximum | 6.0–18.0 | Extreme blur recovery |

### Length-Constrained Inference

When approximate character lengths are known (from UI layout, font metrics, or manual estimation), the model's predictions can be filtered to only include words matching the expected length. This dramatically narrows the candidate pool — from 2,048 words down to 88–555 depending on length — and significantly boosts effective accuracy.

### Multi-Variation Ensemble

At inference time, multiple variations of the input are generated and their predictions averaged:

- **Contrast stretching** at multiple percentile levels (the single biggest improvement for real-world images)
- **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) at different clip limits
- **Brightness jitter** (±10–30 pixel values)
- **Scale jitter** (0.92x–1.08x)
- **Small crop jitter** (±5px random boundary shifts)

This stabilises predictions at high blur levels where small changes in crop boundaries can shift the top-1 prediction.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model (~70 min on Apple Silicon)
python -m src.train --model resnet18 --epochs 40 --time-limit 90

# Run inference on a blurred word image
python -m src.enhanced_inference path/to/blurred_word.png --model models/resnet18_best.pt

# Run benchmark
python -m src.benchmark --model models/resnet18_best.pt --output reports

# Generate PDF report
python -m src.generate_report
```

## Architecture

- **Backbone**: ResNet-18 (ImageNet pretrained, fine-tuned)
- **Input**: 128×384 px RGB images
- **Output**: 2,048-class softmax over BIP-39 wordlist
- **Training**: On-the-fly synthetic data, no disk dataset required
- **Optimiser**: AdamW with CosineAnnealingLR, differential learning rates (backbone 0.1×, head 1×)
- **Loss**: CrossEntropy with label smoothing (0.1)
- **Validation**: Fixed seed evaluation set covering full sigma range (0.5–12.0)

## Project Structure

```
src/
  train.py              # Training with curriculum learning
  enhanced_inference.py  # Multi-variation ensemble inference
  inference.py          # Basic single-pass inference
  benchmark.py          # Per-sigma and edge case evaluation
  generate_data.py      # Word rendering + Gaussian blur
  dataset.py            # On-the-fly data generation + augmentation
  models.py             # Model definitions
  generate_report.py    # PDF report builder
data/
  bip39_english.txt     # 2,048 word vocabulary
models/                 # Saved checkpoints (.pt, gitignored)
reports/                # Benchmark reports (PDF)
```

## Requirements

- Python 3.12+
- PyTorch 2.10+ (MPS backend for Apple Silicon, CUDA for NVIDIA)
- timm, albumentations, Pillow, OpenCV
- Optional: realesrgan (for super-resolution preprocessing)

## Hardware

Developed and tested on **Apple M4 Max** (64GB unified memory) using the PyTorch MPS backend. Works on CUDA GPUs. CPU training is possible but slow.

---

## Responsible Disclosure

This project is published as a **proof of concept** to raise awareness about the inadequacy of visual redaction methods. The goal is to encourage better security practices — not to enable attacks.

**If you are protecting sensitive information:**
- Do not rely on blur, pixelation, or mosaic filters
- Use solid-colour redaction bars that fully cover the text with margin
- Remove the underlying text data from the document, not just the visual layer
- Assume that any partially visible information can be recovered

## Licence

MIT

## Author

**Boris Djordjevic** — [199 Biotechnologies](https://github.com/199-biotechnologies)
