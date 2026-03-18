"""
PyTorch Dataset for blurred word images.
Loads from generated directory structure with on-the-fly augmentation.
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Standard input size for models
INPUT_H = 128
INPUT_W = 384

# Grayscale normalization (not ImageNet — our images are grayscale text)
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]


def get_train_transforms():
    """Training augmentation — simulates real-world screenshot degradation."""
    return A.Compose([
        A.Resize(INPUT_H, INPUT_W),
        # --- Compression artifacts (JPEG + video codec simulation) ---
        # JPEG compression: screenshots, saved images
        A.ImageCompression(quality_range=(20, 95), p=0.5),
        # Downscale+upscale: simulates video compression / low-res screenshots
        # (H.264/VP9 quantization looks like this at the pixel level)
        A.Downscale(scale_range=(0.4, 0.8), p=0.3),
        # --- Geometric transforms ---
        # Simulate partial crop / shifted view / slight rotation
        A.Affine(
            translate_percent={"x": (-0.12, 0.12), "y": (-0.08, 0.08)},
            scale=(0.85, 1.15),
            rotate=(-3, 3),
            mode=cv2.BORDER_CONSTANT, cval=255, p=0.4,
        ),
        # Simulate cropping from edges (partial word visibility)
        A.RandomCrop(height=int(INPUT_H * 0.85), width=int(INPUT_W * 0.85), p=0.2),
        A.Resize(INPUT_H, INPUT_W),  # resize back after crop
        # --- Occlusion / overlap ---
        A.CoarseDropout(
            num_holes_range=(1, 3), hole_height_range=(4, 16), hole_width_range=(8, 40),
            fill="random", p=0.25,
        ),
        # --- Color / exposure ---
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        # Slight noise (sensor noise, compression residuals)
        A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
        # --- Normalize ---
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Minimal transforms for validation."""
    return A.Compose([
        A.Resize(INPUT_H, INPUT_W),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])


class BlurredWordDataset(Dataset):
    """Dataset of blurred BIP-39 word images."""

    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform

        self.samples = []
        for fname in sorted(os.listdir(self.data_dir)):
            if not fname.endswith(".png"):
                continue
            parts = fname.rsplit("_", 1)
            label_and_word = parts[0]
            label_idx = int(label_and_word.split("_")[0])
            self.samples.append((fname, label_idx))

        print(f"  [{split}] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img_path = os.path.join(self.data_dir, fname)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


class OnTheFlyBlurDataset(Dataset):
    """
    Generate blurred samples on-the-fly during training.
    No disk I/O — renders, blurs, and augments in real-time.
    """

    def __init__(
        self,
        wordlist_path: str,
        fonts: list[str],
        samples_per_epoch: int = 102400,
        sigma_range: tuple = (0.5, 4.0),
        transform=None,
        font_sizes: list[int] = None,
    ):
        with open(wordlist_path) as f:
            self.words = [line.strip() for line in f if line.strip()]
        self.fonts = fonts
        self.samples_per_epoch = samples_per_epoch
        self.sigma_range = sigma_range
        self.transform = transform
        self.font_sizes = font_sizes or [24, 28, 32, 36, 40, 48]
        self.n_classes = len(self.words)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        from src.generate_data import render_word, apply_gaussian_blur

        label = random.randint(0, self.n_classes - 1)
        word = self.words[label]

        font_path = random.choice(self.fonts)
        font_size = random.choice(self.font_sizes)
        sigma = random.uniform(*self.sigma_range)

        # Grayscale text — dark on light (most common) or light on dark
        if random.random() < 0.85:
            gray_bg = random.randint(235, 255)
            gray_fg = random.randint(0, 50)
        else:
            gray_bg = random.randint(0, 30)
            gray_fg = random.randint(210, 255)
        text_color = (gray_fg, gray_fg, gray_fg)
        bg_color = (gray_bg, gray_bg, gray_bg)

        x_jitter = random.randint(-10, 10)
        y_jitter = random.randint(-4, 4)

        img = render_word(
            word, font_path, font_size,
            text_color=text_color, bg_color=bg_color,
            x_jitter=x_jitter, y_jitter=y_jitter,
        )
        img = apply_gaussian_blur(img, sigma)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]

        return img, label
