"""
Synthetic data generator for blurred BIP-39 word images.

Renders each word in multiple fonts/sizes, applies Gaussian blur at various
sigma levels, and adds augmentations (crop, noise, brightness, partial overlap).
"""

import os
import random
import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# Monospace / system fonts available on macOS
FONT_PATHS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Courier.ttc",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/System/Library/Fonts/Supplemental/Andale Mono.ttf",
    "/System/Library/Fonts/Supplemental/PTMono.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/HelveticaNeue.ttc",
]

FONT_SIZES = [24, 28, 32, 36, 40, 48]
SIGMA_RANGE = (1.0, 25.0)
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 384  # wide enough for longest BIP-39 words


def get_available_fonts() -> list[str]:
    """Return font paths that actually exist on this system."""
    return [p for p in FONT_PATHS if os.path.exists(p)]


def load_wordlist(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def render_word(
    word: str,
    font_path: str,
    font_size: int,
    img_w: int = IMAGE_WIDTH,
    img_h: int = IMAGE_HEIGHT,
    text_color: tuple = (0, 0, 0),
    bg_color: tuple = (255, 255, 255),
    x_jitter: int = 0,
    y_jitter: int = 0,
) -> np.ndarray:
    """Render a word as a PIL image, return as numpy array (H, W, 3)."""
    img = Image.new("RGB", (img_w, img_h), bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), word, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # Center the text with optional jitter
    x = (img_w - tw) // 2 + x_jitter
    y = (img_h - th) // 2 + y_jitter
    draw.text((x, y), word, fill=text_color, font=font)

    return np.array(img)


def apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian blur with given sigma."""
    # Kernel size must be odd and large enough for the sigma
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(ksize, 3)
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def random_crop(img: np.ndarray, crop_fraction: float = 0.15) -> np.ndarray:
    """Randomly crop edges to simulate partial visibility."""
    h, w = img.shape[:2]
    top = random.randint(0, int(h * crop_fraction))
    bottom = random.randint(0, int(h * crop_fraction))
    left = random.randint(0, int(w * crop_fraction))
    right = random.randint(0, int(w * crop_fraction))

    cropped = img[top : h - bottom, left : w - right]
    # Resize back to original dimensions
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def add_noise(img: np.ndarray, noise_std: float = 10.0) -> np.ndarray:
    """Add Gaussian noise."""
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def adjust_brightness(img: np.ndarray, factor_range: tuple = (0.7, 1.3)) -> np.ndarray:
    """Random brightness adjustment."""
    factor = random.uniform(*factor_range)
    adjusted = img.astype(np.float32) * factor
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def vary_text_color() -> tuple[tuple, tuple]:
    """Return random text/bg color pair (dark on light or light on dark)."""
    if random.random() < 0.8:
        # Dark text on light background (most common)
        gray_bg = random.randint(230, 255)
        gray_fg = random.randint(0, 60)
    else:
        # Light text on dark background
        gray_bg = random.randint(0, 40)
        gray_fg = random.randint(200, 255)
    return (gray_fg, gray_fg, gray_fg), (gray_bg, gray_bg, gray_bg)


def generate_sample(
    word: str,
    label_idx: int,
    font_path: str,
    font_size: int,
    sigma: float,
    augment: bool = True,
) -> tuple[np.ndarray, int, dict]:
    """Generate a single blurred word sample with metadata."""
    text_color, bg_color = vary_text_color()
    x_jitter = random.randint(-15, 15) if augment else 0
    y_jitter = random.randint(-5, 5) if augment else 0

    # Render clean
    clean = render_word(
        word, font_path, font_size,
        text_color=text_color, bg_color=bg_color,
        x_jitter=x_jitter, y_jitter=y_jitter,
    )

    # Apply Gaussian blur
    blurred = apply_gaussian_blur(clean, sigma)

    if augment:
        # Random crop (simulates partial visibility)
        if random.random() < 0.3:
            blurred = random_crop(blurred, crop_fraction=0.12)

        # Add noise
        if random.random() < 0.4:
            blurred = add_noise(blurred, noise_std=random.uniform(3, 15))

        # Brightness variation
        if random.random() < 0.4:
            blurred = adjust_brightness(blurred)

    meta = {
        "word": word,
        "label": label_idx,
        "sigma": sigma,
        "font": os.path.basename(font_path),
        "font_size": font_size,
    }
    return blurred, label_idx, meta


def generate_dataset(
    wordlist_path: str,
    output_dir: str,
    samples_per_word: int = 50,
    split_ratios: tuple = (0.8, 0.1, 0.1),
    seed: int = 42,
):
    """Generate full training dataset."""
    random.seed(seed)
    np.random.seed(seed)

    words = load_wordlist(wordlist_path)
    fonts = get_available_fonts()
    if not fonts:
        raise RuntimeError("No fonts found! Check FONT_PATHS.")

    print(f"Words: {len(words)}, Fonts: {len(fonts)}, Samples/word: {samples_per_word}")
    print(f"Total samples: {len(words) * samples_per_word}")

    # Create split directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)

    # Pre-compute split assignments
    n_train = int(samples_per_word * split_ratios[0])
    n_val = int(samples_per_word * split_ratios[1])

    all_metadata = []

    for word_idx, word in enumerate(tqdm(words, desc="Generating")):
        for sample_idx in range(samples_per_word):
            font_path = random.choice(fonts)
            font_size = random.choice(FONT_SIZES)
            sigma = random.uniform(*SIGMA_RANGE)

            img, label, meta = generate_sample(
                word, word_idx, font_path, font_size, sigma, augment=True
            )

            # Determine split
            if sample_idx < n_train:
                split = "train"
            elif sample_idx < n_train + n_val:
                split = "val"
            else:
                split = "test"

            # Save image
            fname = f"{word_idx:04d}_{word}_{sample_idx:03d}.png"
            fpath = os.path.join(output_dir, split, fname)
            cv2.imwrite(fpath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            meta["split"] = split
            meta["path"] = fpath
            all_metadata.append(meta)

    # Save metadata
    import json
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    # Save word-to-index mapping
    w2i_path = os.path.join(output_dir, "word2idx.json")
    with open(w2i_path, "w") as f:
        json.dump({w: i for i, w in enumerate(words)}, f, indent=2)

    print(f"\nDataset saved to {output_dir}")
    print(f"  Train: {n_train * len(words)} samples")
    print(f"  Val: {n_val * len(words)} samples")
    print(f"  Test: {(samples_per_word - n_train - n_val) * len(words)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate blurred word dataset")
    parser.add_argument("--wordlist", default="data/bip39_english.txt")
    parser.add_argument("--output", default="data/generated")
    parser.add_argument("--samples-per-word", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        args.wordlist, args.output,
        samples_per_word=args.samples_per_word,
        seed=args.seed,
    )
