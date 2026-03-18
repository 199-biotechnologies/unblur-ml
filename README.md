# unblur-ml

Recover Gaussian-blurred BIP-39 seed phrase words using deep learning. Given a blurred screenshot of a seed phrase, the model classifies each word as one of the 2,048 entries in the BIP-39 English wordlist.

## How It Works

This is a **classification** problem, not reconstruction. Since the vocabulary is fixed at 2,048 words, we don't need to rebuild pixel-perfect text — we just need to identify which word it is. A ResNet-18 backbone is fine-tuned with curriculum learning, progressively increasing blur difficulty during training.

### Key Results (v1)

| Blur Sigma | Top-1 Accuracy | Top-5 Accuracy |
|---|---|---|
| 1 (light) | 99.5% | 100% |
| 3 (mild) | 91.0% | 98.0% |
| 5 (medium) | 62.5% | 83.0% |

Training time: **14 minutes** on Apple M4 Max.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model (20 min on Apple Silicon)
python -m src.train --model resnet18 --epochs 20 --time-limit 20

# Run inference on a blurred image
python -m src.inference path/to/blurred_image.png --model models/resnet18_best.pt

# Run benchmark report
python -m src.benchmark --model models/resnet18_best.pt --output reports

# Generate PDF report
python -m src.generate_report
```

## Architecture

- **Backbone**: ResNet-18 (pretrained ImageNet, fine-tuned)
- **Input**: 128x384 px RGB images
- **Output**: 2,048-class softmax over BIP-39 wordlist
- **Training**: On-the-fly synthetic data generation — render word, blur, augment
- **Curriculum**: Sigma 0.5→3 → 5 → 7 → 9 → 12 over training epochs
- **Augmentation**: JPEG compression, affine transforms, coarse dropout, brightness

## Training Pipeline

```
BIP-39 word → Render (random font/size/color) → Gaussian blur (σ) → Augment → Model → Predict word
```

Data is generated on-the-fly: no disk dataset needed. Each epoch sees fresh random samples.

### Curriculum Learning

The model starts with easy samples and gradually increases difficulty:

| Phase | Blur Sigma | Purpose |
|---|---|---|
| Warmup | 0.5–3.0 | Frozen backbone, train head only |
| Easy | 0.5–3.0 | Learn word shapes |
| Medium | 0.5–5.0 | Generalize to moderate blur |
| Hard | 0.5–9.0 | Push boundaries |
| Maximum | 0.5–12.0 | Extreme blur recovery |

## Inference Modes

### Single word
```bash
python -m src.inference word_crop.png --mode word --model models/resnet18_best.pt
```

### Full line (auto-segments words)
```bash
python -m src.inference seed_phrase_line.png --mode line --model models/resnet18_best.pt
```

## Project Structure

```
src/
  train.py          # Training script with curriculum learning
  inference.py      # Single word + full line inference
  benchmark.py      # Per-sigma and edge case evaluation
  generate_data.py  # Word rendering + Gaussian blur
  dataset.py        # PyTorch dataset with on-the-fly generation
  models.py         # Model definitions (ResNet, EfficientNet, DINOv2)
  generate_report.py # PDF report builder
data/
  bip39_english.txt # 2,048 word vocabulary
models/             # Saved checkpoints
reports/            # Benchmark charts and PDF reports
```

## Requirements

- Python 3.12+
- PyTorch 2.10+ (MPS backend for Apple Silicon)
- timm, albumentations, Pillow, OpenCV

## Hardware

Optimized for **Apple Silicon** (M4 Max tested). Also works on CUDA GPUs. CPU training is possible but slow.
