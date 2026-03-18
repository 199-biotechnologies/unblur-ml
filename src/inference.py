"""
Inference pipeline for deblurring seed phrases.

Handles:
- Single word crop -> classification
- Full line -> segmentation + classification
- Outputs top-K predictions with confidence scores
"""

import os
import sys
import json
import argparse

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model
from src.dataset import get_val_transforms, INPUT_H, INPUT_W


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    import timm
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    model = timm.create_model(model_name, pretrained=False, num_classes=2048)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name


def load_wordlist(path: str = "data/bip39_english.txt") -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def segment_line(img: np.ndarray, min_gap: int = 8) -> list[np.ndarray]:
    """
    Segment a line of blurred words by detecting gaps.
    Works by finding columns that are mostly background color.
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Determine if dark-on-light or light-on-dark
    mean_val = gray.mean()
    if mean_val > 128:
        # Dark text on light bg — gaps are bright columns
        col_means = gray.mean(axis=0)
        threshold = np.percentile(col_means, 75)
        is_gap = col_means > threshold
    else:
        # Light text on dark bg
        col_means = gray.mean(axis=0)
        threshold = np.percentile(col_means, 25)
        is_gap = col_means < threshold

    # Find contiguous gap/word regions
    segments = []
    in_word = False
    word_start = 0

    for i, gap in enumerate(is_gap):
        if not gap and not in_word:
            word_start = i
            in_word = True
        elif gap and in_word:
            if i - word_start > min_gap:
                segments.append((word_start, i))
            in_word = False

    if in_word:
        segments.append((word_start, len(is_gap)))

    # Merge segments that are too close (less than min_gap apart)
    merged = []
    for start, end in segments:
        if merged and start - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    # Extract word images with padding
    word_imgs = []
    pad = 5
    for start, end in merged:
        s = max(0, start - pad)
        e = min(img.shape[1], end + pad)
        word_img = img[:, s:e]
        if word_img.shape[1] > 5:  # skip tiny fragments
            word_imgs.append(word_img)

    return word_imgs


def predict_word(
    model: torch.nn.Module,
    img: np.ndarray,
    words: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Predict word from a single blurred image crop."""
    transform = get_val_transforms()

    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Resize to input dimensions
    img_resized = cv2.resize(img, (INPUT_W, INPUT_H))
    augmented = transform(image=img_resized)
    tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = probs.topk(top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        results.append((words[idx], prob))

    return results


def predict_line(
    model: torch.nn.Module,
    img: np.ndarray,
    words: list[str],
    device: torch.device,
    top_k: int = 5,
) -> list[list[tuple[str, float]]]:
    """Predict all words in a line image."""
    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # Segment into individual words
    word_imgs = segment_line(img)

    if not word_imgs:
        # If segmentation fails, treat whole image as one word
        return [predict_word(model, img, words, device, top_k)]

    results = []
    for word_img in word_imgs:
        preds = predict_word(model, word_img, words, device, top_k)
        results.append(preds)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict blurred words")
    parser.add_argument("image", help="Path to blurred image")
    parser.add_argument("--model", default="models/convnext_best.pt")
    parser.add_argument("--wordlist", default="data/bip39_english.txt")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["word", "line"], default="line")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, model_name = load_model(args.model, device)
    words = load_wordlist(args.wordlist)

    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if args.mode == "line":
        results = predict_line(model, img, words, device, args.top_k)
        for i, word_preds in enumerate(results):
            print(f"\nWord {i+1}:")
            for word, prob in word_preds:
                print(f"  {word:15s} {prob*100:5.1f}%")
    else:
        results = predict_word(model, img, words, device, args.top_k)
        print("Predictions:")
        for word, prob in results:
            print(f"  {word:15s} {prob*100:5.1f}%")
