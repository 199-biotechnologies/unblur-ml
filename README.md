<div align="center">

# Unblur ML

**Recover Gaussian-blurred BIP-39 seed phrases using a CNN classifier. 100% accuracy at mild blur. 93% at heavy blur. Trained in 70 minutes on a laptop.**

<br />

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/unblur-ml?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/unblur-ml/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

<br />

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

Think blurring your seed phrase keeps it safe? It does not. This tool proves that a commodity deep learning model, trained on a single laptop in under two hours, can read blurred text with near-perfect accuracy. No cloud compute. No special datasets. Just a ResNet-18 and the BIP-39 wordlist. If your security depends on Gaussian blur, you have a problem.

[Why This Exists](#why-this-exists) | [How It Works](#how-it-works) | [Results](#results) | [Install](#install) | [Usage](#usage) | [Contributing](#contributing) | [License](#license)

</div>

## Why This Exists

People blur sensitive text in screenshots every day. Seed phrases, passwords, personal details, classified documents. The assumption is that blur is a one-way operation. Once the information is smeared, it's gone.

That assumption is wrong.

This project exists to demonstrate the problem. It is a security research tool, not an attack tool. The goal is to make you stop trusting visual redaction.

The same principle applies beyond seed phrases:

- **Pixelation and mosaic filters** retain spatial frequency information that neural networks can exploit
- **Black redaction bars** that don't fully cover text leak information through exposed pixels and character-length analysis
- **Blurred faces and license plates** can be reconstructed by GANs like PULSE and GFPGAN
- **Government redactions** on declassified documents are subject to the same vulnerabilities

The core insight: if you know the vocabulary, classification is far easier than reconstruction. You do not need to unblur the image. You just need to figure out which word from a known list produced that blur pattern.

The barrier to building these tools is collapsing. This entire model was built and trained in a single afternoon using open-source libraries on consumer hardware. No PhD required.

**If your security depends on visual obscuration, it is time to reconsider.**

## How It Works

Unblur ML treats deblurring as **classification, not reconstruction**. BIP-39 has exactly 2,048 words. Instead of recovering the original pixels, the model identifies which word produced the blur pattern. This transforms an ill-posed inverse problem into a tractable 2,048-class classification task.

### Architecture

- **Backbone**: ResNet-18 (ImageNet pretrained, fine-tuned)
- **Input**: 128x384 px RGB images
- **Output**: 2,048-class softmax over the BIP-39 English wordlist
- **Optimizer**: AdamW with CosineAnnealingLR, differential learning rates (backbone 0.1x, head 1x)
- **Loss**: CrossEntropy with label smoothing (0.1)

### On-the-Fly Data Generation

No static dataset. Every training sample is synthesized in real time:

1. Render a random BIP-39 word in a random font (Menlo, Courier, SF Mono), random size, random position
2. Apply Gaussian blur at a random sigma level
3. Augment with real-world degradations: JPEG compression, downscaling, affine transforms, noise, contrast variation

The model never sees the same image twice. Each epoch is entirely fresh data. This prevents overfitting and enables unlimited effective dataset size.

### Curriculum Learning

Training starts easy and gets progressively harder. The lower bound of blur sigma rises over time, forcing the model to spend its training budget on hard cases:

| Phase | Blur Sigma | Purpose |
|---|---|---|
| Warmup | 0.5 - 3.0 | Frozen backbone, train classifier head only |
| Easy | 1.0 - 5.0 | Learn word shapes with mild blur |
| Medium | 2.0 - 8.0 | Generalize to moderate blur |
| Hard | 3.0 - 11.0 | Focus on difficult cases |
| Harder | 4.0 - 14.0 | Push into extreme territory |
| Very Hard | 5.0 - 16.0 | Near information-theoretic limit |
| Maximum | 6.0 - 18.0 | Extreme blur recovery |

### Multi-Variation Ensemble Inference

At inference time, multiple variations of the input are generated and their predictions averaged:

- Contrast stretching at multiple percentile levels (single biggest accuracy boost for real-world images)
- CLAHE at different clip limits
- Brightness jitter, scale jitter, crop jitter

This stabilizes predictions at high blur levels where small crop boundary changes can shift the top-1 prediction.

### Length-Constrained Inference

When you know the approximate character length of the blurred word (from UI layout or font metrics), predictions can be filtered to only include words matching that length. This narrows the candidate pool from 2,048 words down to 88-555, dramatically boosting effective accuracy.

## Results

### Accuracy by Blur Level

| Blur Sigma | Top-1 Accuracy | Top-5 Accuracy | Top-20 Accuracy |
|---|---|---|---|
| 3 (mild) | **100%** | 100% | 100% |
| 5 (medium) | **99.5%** | 100% | 100% |
| 7 (heavy) | **96.0%** | 100% | 100% |
| 8 (very heavy) | **93.0%** | 99.0% | 100% |
| 10 (extreme) | **79.0%** | 94.0% | 98.0% |
| 12 (near-obliteration) | **51.0%** | 75.5% | 87.0% |

### Edge Case Robustness

With base blur sigma 3-8:

| Degradation | Top-5 Accuracy |
|---|---|
| Heavy crop (25% removed) | 96.0% |
| JPEG compression (q=20-40) | 97.3% |
| Grey/faded text (40% contrast) | 100% |
| Inverted colors (light on dark) | 100% |
| Downscale 50% + upscale | 100% |
| Combined worst case | 94.7% |

Training time: **70 minutes** on Apple M4 Max (64GB).

## Install

```bash
git clone https://github.com/199-biotechnologies/unblur-ml.git
cd unblur-ml
pip install -r requirements.txt
```

### Requirements

- Python 3.12+
- PyTorch 2.10+ (MPS backend for Apple Silicon, CUDA for NVIDIA)
- timm, albumentations, Pillow, OpenCV

## Usage

### Train a Model

```bash
python -m src.train --model resnet18 --epochs 40 --time-limit 90
```

Training takes about 70 minutes on Apple Silicon. Works on CUDA GPUs. CPU training is possible but slow.

### Run Inference on a Blurred Image

```bash
python -m src.enhanced_inference path/to/blurred_word.png --model models/resnet18_best.pt
```

### Benchmark

```bash
python -m src.benchmark --model models/resnet18_best.pt --output reports
```

### Generate Report

```bash
python -m src.generate_report
```

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

## Responsible Disclosure

This project is a proof of concept for security awareness. The goal is to encourage better security practices, not to enable attacks.

If you are protecting sensitive information:

- Do not rely on blur, pixelation, or mosaic filters
- Use solid-color redaction bars that fully cover the text with margin
- Remove the underlying text data from the document, not just the visual layer
- Assume that any partially visible information can be recovered

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)

---
<div align="center">

Built by [Boris Djordjevic](https://github.com/longevityboris) at [199 Biotechnologies](https://github.com/199-biotechnologies) | [Paperfoot AI](https://paperfoot.ai)

<br />

**If this is useful to you:**

[![Star this repo](https://img.shields.io/github/stars/199-biotechnologies/unblur-ml?style=for-the-badge&logo=github&label=%E2%AD%90%20Star%20this%20repo&color=yellow)](https://github.com/199-biotechnologies/unblur-ml/stargazers)
&nbsp;&nbsp;
[![Follow @longevityboris](https://img.shields.io/badge/Follow_%40longevityboris-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/longevityboris)

</div>
