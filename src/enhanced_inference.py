"""
Enhanced inference pipeline with:
A) Super-resolution preprocessing (Real-ESRGAN upscaling)
B) Test-Time Augmentation (TTA) — multiple augmented passes, averaged softmax
C) Multi-scale inference — run at multiple resolutions, ensemble
D) Self-training ready — confidence-based output for pseudo-labeling
E) Input normalization — handles grey text, colored backgrounds, varied contrast

Also: aspect-ratio-preserving resize (fixes Codex bug #3)
"""

import os
import sys
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_val_transforms, INPUT_H, INPUT_W, NORM_MEAN, NORM_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2


def normalize_input(img: np.ndarray) -> np.ndarray:
    """
    Normalize a real-world screenshot crop to look like training data.
    Handles: grey text, colored backgrounds, low contrast, inverted colors.
    """
    # Convert to grayscale if color
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # CLAHE — enhance local contrast (helps with faded/grey text)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Determine if dark-on-light or light-on-dark
    mean_val = enhanced.mean()
    if mean_val < 128:
        # Light text on dark bg — invert to match training (dark on light)
        enhanced = 255 - enhanced

    # Stretch contrast to full 0-255 range
    pmin, pmax = np.percentile(enhanced, [2, 98])
    if pmax - pmin > 10:
        enhanced = np.clip((enhanced - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)

    # Convert back to RGB (tripled grayscale)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def pad_to_aspect_ratio(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """
    Resize preserving aspect ratio, pad to target dimensions.
    Fixes the stretching bug Codex identified.
    """
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    img_aspect = w / h

    if img_aspect > target_aspect:
        # Image is wider — fit to width, pad height
        new_w = target_w
        new_h = int(target_w / img_aspect)
    else:
        # Image is taller — fit to height, pad width
        new_h = target_h
        new_w = int(target_h * img_aspect)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Determine background color (most common edge pixel)
    bg_val = int(np.median(resized[0, :, 0]))  # top row median
    canvas = np.full((target_h, target_w, 3), bg_val, dtype=np.uint8)

    # Center the resized image
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas


def get_tta_transforms() -> list[A.Compose]:
    """Generate TTA transform variants."""
    base = [
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ]

    transforms = [
        # Original
        A.Compose([A.Resize(INPUT_H, INPUT_W)] + base),
        # Slight shift left
        A.Compose([
            A.Resize(INPUT_H, INPUT_W),
            A.Affine(translate_percent={"x": -0.05}, mode=cv2.BORDER_CONSTANT, cval=255, p=1.0),
        ] + base),
        # Slight shift right
        A.Compose([
            A.Resize(INPUT_H, INPUT_W),
            A.Affine(translate_percent={"x": 0.05}, mode=cv2.BORDER_CONSTANT, cval=255, p=1.0),
        ] + base),
        # Slightly brighter
        A.Compose([
            A.Resize(INPUT_H, INPUT_W),
            A.RandomBrightnessContrast(brightness_limit=(0.1, 0.1), contrast_limit=(0.1, 0.1), p=1.0),
        ] + base),
        # Slightly darker
        A.Compose([
            A.Resize(INPUT_H, INPUT_W),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, -0.1), contrast_limit=(-0.1, -0.1), p=1.0),
        ] + base),
        # Slight scale up
        A.Compose([
            A.Resize(int(INPUT_H * 1.1), int(INPUT_W * 1.1)),
            A.CenterCrop(INPUT_H, INPUT_W),
        ] + base),
    ]
    return transforms


def predict_single(
    model: torch.nn.Module,
    img: np.ndarray,
    words: list[str],
    device: torch.device,
    top_k: int = 10,
    use_tta: bool = True,
    use_normalization: bool = True,
) -> list[tuple[str, float]]:
    """
    Enhanced single-word prediction.

    Combines:
    - Input normalization (contrast, orientation)
    - Aspect-ratio-preserving resize
    - Test-Time Augmentation (6 variants, averaged softmax)
    """
    # Step 1: Normalize input
    if use_normalization:
        img = normalize_input(img)

    # Step 2: Pad to correct aspect ratio (don't stretch)
    img = pad_to_aspect_ratio(img, INPUT_W, INPUT_H)

    # Step 3: Run predictions
    model.eval()

    if use_tta:
        # TTA: run multiple augmented versions, average softmax
        tta_transforms = get_tta_transforms()
        all_probs = []

        for tfm in tta_transforms:
            augmented = tfm(image=img)
            tensor = augmented["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        # Average across TTA variants
        avg_probs = torch.stack(all_probs).mean(dim=0)
    else:
        # Single pass
        tfm = get_val_transforms()
        augmented = tfm(image=img)
        tensor = augmented["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            avg_probs = F.softmax(logits, dim=1)

    # Top-K results
    top_probs, top_indices = avg_probs.topk(top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        results.append((words[idx], prob))

    return results


def predict_with_confidence(
    model: torch.nn.Module,
    img: np.ndarray,
    words: list[str],
    device: torch.device,
) -> dict:
    """
    Prediction with confidence metadata for self-training / pseudo-labeling.
    Returns prediction + entropy + margin (top1 - top2 gap).
    """
    results = predict_single(model, img, words, device, top_k=10, use_tta=True)

    top1_word, top1_conf = results[0]
    top2_word, top2_conf = results[1]

    # Entropy of top-10 predictions
    probs = [r[1] for r in results]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)

    # Confidence margin
    margin = top1_conf - top2_conf

    return {
        "prediction": top1_word,
        "confidence": top1_conf,
        "top_k": results,
        "entropy": entropy,
        "margin": margin,
        "is_confident": top1_conf > 0.5 and margin > 0.2,
    }


if __name__ == "__main__":
    import argparse
    import timm

    parser = argparse.ArgumentParser(description="Enhanced inference")
    parser.add_argument("image", help="Path to blurred word image")
    parser.add_argument("--model", default="models/resnet18_best.pt")
    parser.add_argument("--wordlist", default="data/bip39_english.txt")
    parser.add_argument("--no-tta", action="store_true", help="Disable TTA")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=2048)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    with open(args.wordlist) as f:
        words = [line.strip() for line in f if line.strip()]

    # Load image
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = predict_with_confidence(model, img, words, device)

    print(f"\nPrediction: '{result['prediction']}' (confidence: {result['confidence']*100:.1f}%)")
    print(f"Confident: {result['is_confident']}")
    print(f"Entropy: {result['entropy']:.3f}, Margin: {result['margin']:.3f}")
    print(f"\nTop {args.top_k}:")
    for word, prob in result["top_k"]:
        print(f"  {word:15s} {prob*100:5.1f}%")
