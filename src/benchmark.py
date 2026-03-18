"""
Benchmark and report generator.

Evaluates model across:
- Different blur sigma levels
- Partial crops (edge cases)
- JPEG compression artifacts
- Different fonts
- Generates visual report with example predictions
"""

import os
import sys
import json
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import get_model
from src.generate_data import (
    render_word, apply_gaussian_blur, get_available_fonts,
    random_crop, add_noise, load_wordlist,
)
from src.dataset import get_val_transforms, INPUT_H, INPUT_W
from src.inference import predict_word, load_model


def evaluate_by_sigma(model, words, fonts, device, n_per_sigma=200):
    """Evaluate accuracy at different blur levels."""
    sigma_bins = [1, 3, 5, 8, 12, 16, 20, 25]
    results = {}

    for sigma in sigma_bins:
        correct = 0
        correct_top5 = 0
        total = 0

        for _ in range(n_per_sigma):
            label = random.randint(0, len(words) - 1)
            word = words[label]
            font = random.choice(fonts)
            size = random.choice([16, 20, 24, 28, 32])

            img = render_word(word, font, size)
            img = apply_gaussian_blur(img, sigma)

            preds = predict_word(model, img, words, device, top_k=5)
            if preds[0][0] == word:
                correct += 1
            if any(p[0] == word for p in preds):
                correct_top5 += 1
            total += 1

        results[sigma] = {
            "top1": 100.0 * correct / total,
            "top5": 100.0 * correct_top5 / total,
        }
        print(f"  Sigma {sigma:2d}: Top1={results[sigma]['top1']:.1f}% Top5={results[sigma]['top5']:.1f}%")

    return results


def evaluate_edge_cases(model, words, fonts, device, n_each=150):
    """Evaluate specific edge cases."""
    cases = {}

    # 1. Heavy crop (30% edges removed)
    correct = 0
    total = 0
    for _ in range(n_each):
        label = random.randint(0, len(words) - 1)
        word = words[label]
        img = render_word(word, random.choice(fonts), random.choice([20, 24, 28]))
        img = apply_gaussian_blur(img, random.uniform(3, 15))
        img = random_crop(img, crop_fraction=0.25)
        preds = predict_word(model, img, words, device, top_k=5)
        if any(p[0] == word for p in preds):
            correct += 1
        total += 1
    cases["heavy_crop"] = 100.0 * correct / total
    print(f"  Heavy crop (25%): Top5={cases['heavy_crop']:.1f}%")

    # 2. JPEG compression (quality 20-40)
    correct = 0
    total = 0
    for _ in range(n_each):
        label = random.randint(0, len(words) - 1)
        word = words[label]
        img = render_word(word, random.choice(fonts), random.choice([20, 24, 28]))
        img = apply_gaussian_blur(img, random.uniform(3, 15))
        # JPEG compress
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, random.randint(20, 40)])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        preds = predict_word(model, img, words, device, top_k=5)
        if any(p[0] == word for p in preds):
            correct += 1
        total += 1
    cases["jpeg_heavy"] = 100.0 * correct / total
    print(f"  Heavy JPEG (q20-40): Top5={cases['jpeg_heavy']:.1f}%")

    # 3. Very noisy
    correct = 0
    total = 0
    for _ in range(n_each):
        label = random.randint(0, len(words) - 1)
        word = words[label]
        img = render_word(word, random.choice(fonts), random.choice([20, 24, 28]))
        img = apply_gaussian_blur(img, random.uniform(3, 15))
        img = add_noise(img, noise_std=25)
        preds = predict_word(model, img, words, device, top_k=5)
        if any(p[0] == word for p in preds):
            correct += 1
        total += 1
    cases["heavy_noise"] = 100.0 * correct / total
    print(f"  Heavy noise (std=25): Top5={cases['heavy_noise']:.1f}%")

    # 4. Crop + JPEG + Noise combined
    correct = 0
    total = 0
    for _ in range(n_each):
        label = random.randint(0, len(words) - 1)
        word = words[label]
        img = render_word(word, random.choice(fonts), random.choice([20, 24, 28]))
        img = apply_gaussian_blur(img, random.uniform(5, 18))
        img = random_crop(img, crop_fraction=0.15)
        img = add_noise(img, noise_std=15)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 50)])
        img = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        preds = predict_word(model, img, words, device, top_k=5)
        if any(p[0] == word for p in preds):
            correct += 1
        total += 1
    cases["combined_worst"] = 100.0 * correct / total
    print(f"  Combined worst case: Top5={cases['combined_worst']:.1f}%")

    # 5. Tiny font (hard to read even unblurred)
    correct = 0
    total = 0
    for _ in range(n_each):
        label = random.randint(0, len(words) - 1)
        word = words[label]
        img = render_word(word, random.choice(fonts), 14)
        img = apply_gaussian_blur(img, random.uniform(3, 10))
        preds = predict_word(model, img, words, device, top_k=5)
        if any(p[0] == word for p in preds):
            correct += 1
        total += 1
    cases["tiny_font"] = 100.0 * correct / total
    print(f"  Tiny font (14px): Top5={cases['tiny_font']:.1f}%")

    return cases


def generate_visual_examples(model, words, fonts, device, output_dir):
    """Generate visual grid showing predictions at various blur levels."""
    os.makedirs(output_dir, exist_ok=True)

    # Pick 8 diverse words
    sample_words = ["abandon", "crypto", "puzzle", "wrong", "ability", "zone", "mushroom", "abstract"]
    sample_indices = [words.index(w) for w in sample_words if w in words]
    sample_words = [words[i] for i in sample_indices]

    sigmas = [2, 5, 10, 15, 20, 25]

    fig, axes = plt.subplots(len(sample_words), len(sigmas), figsize=(24, 16))
    fig.suptitle("Predictions at Different Blur Levels (sigma)", fontsize=16, y=0.98)

    font = random.choice(fonts)

    for row, (word, label) in enumerate(zip(sample_words, sample_indices)):
        for col, sigma in enumerate(sigmas):
            img = render_word(word, font, 24)
            img_blurred = apply_gaussian_blur(img, sigma)

            preds = predict_word(model, img_blurred, words, device, top_k=3)
            top_word, top_conf = preds[0]

            ax = axes[row, col]
            ax.imshow(img_blurred)
            ax.set_xticks([])
            ax.set_yticks([])

            color = "green" if top_word == word else "red"
            ax.set_title(
                f"σ={sigma}\n'{top_word}' ({top_conf*100:.0f}%)",
                fontsize=9, color=color,
            )
            if col == 0:
                ax.set_ylabel(f"'{word}'", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "blur_levels_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Edge case examples
    fig, axes = plt.subplots(4, 6, figsize=(24, 12))
    fig.suptitle("Edge Case Examples", fontsize=16, y=0.98)

    edge_labels = ["Heavy Crop", "JPEG q=25", "Heavy Noise", "Combined"]
    for row_idx, (edge_name, edge_fn) in enumerate([
        ("Heavy Crop", lambda img: random_crop(img, 0.25)),
        ("JPEG q=25", lambda img: cv2.cvtColor(
            cv2.imdecode(
                cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 25])[1],
                cv2.IMREAD_COLOR
            ), cv2.COLOR_BGR2RGB)),
        ("Noise std=25", lambda img: add_noise(img, 25)),
        ("All combined", lambda img: add_noise(
            random_crop(
                cv2.cvtColor(
                    cv2.imdecode(
                        cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                      [cv2.IMWRITE_JPEG_QUALITY, 35])[1],
                        cv2.IMREAD_COLOR
                    ), cv2.COLOR_BGR2RGB),
                0.15),
            15)),
    ]):
        test_words = random.sample(sample_words, min(6, len(sample_words)))
        for col_idx, word in enumerate(test_words):
            label = words.index(word)
            img = render_word(word, random.choice(fonts), 24)
            img = apply_gaussian_blur(img, random.uniform(5, 15))
            img = edge_fn(img)

            preds = predict_word(model, img, words, device, top_k=3)
            top_word, top_conf = preds[0]

            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            color = "green" if top_word == word else "red"
            ax.set_title(f"'{top_word}' ({top_conf*100:.0f}%)", fontsize=9, color=color)
            if col_idx == 0:
                ax.set_ylabel(edge_name, fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "edge_cases_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def generate_report(model_path, wordlist, output_dir):
    """Full benchmark report."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model, model_name = load_model(model_path, device)
    words = load_wordlist(wordlist)
    fonts = get_available_fonts()

    random.seed(42)
    np.random.seed(42)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"BENCHMARK REPORT — {model_name}")
    print(f"{'='*60}")

    # 1. Accuracy by sigma
    print("\n[1] Accuracy by Blur Sigma:")
    sigma_results = evaluate_by_sigma(model, words, fonts, device)

    # 2. Edge cases
    print("\n[2] Edge Case Accuracy (Top-5):")
    edge_results = evaluate_edge_cases(model, words, fonts, device)

    # 3. Visual examples
    print("\n[3] Generating visual examples...")
    generate_visual_examples(model, words, fonts, device, output_dir)

    # 4. Plot sigma vs accuracy
    sigmas = sorted(sigma_results.keys())
    top1s = [sigma_results[s]["top1"] for s in sigmas]
    top5s = [sigma_results[s]["top5"] for s in sigmas]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sigmas, top1s, "o-", label="Top-1 Accuracy", linewidth=2, markersize=8)
    ax.plot(sigmas, top5s, "s-", label="Top-5 Accuracy", linewidth=2, markersize=8)
    ax.set_xlabel("Gaussian Blur Sigma", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy vs Blur Level — {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    path = os.path.join(output_dir, "sigma_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # 5. Edge case bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(edge_results.keys())
    vals = list(edge_results.values())
    colors = ["#2ecc71" if v > 70 else "#e74c3c" if v < 40 else "#f39c12" for v in vals]
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlabel("Top-5 Accuracy (%)", fontsize=12)
    ax.set_title("Edge Case Performance", fontsize=14)
    ax.set_xlim(0, 105)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=10)

    path = os.path.join(output_dir, "edge_cases_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # 6. Save full report JSON
    report = {
        "model": model_name,
        "sigma_results": {str(k): v for k, v in sigma_results.items()},
        "edge_cases": edge_results,
    }
    report_path = os.path.join(output_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {report_path}")

    # Print training history if available
    hist_path = os.path.join(os.path.dirname(model_path), f"{model_name}_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        print(f"\nTraining History ({len(history)} epochs):")
        for h in history:
            print(f"  Epoch {h['epoch']:2d}: "
                  f"Train Acc={h['train_acc']:.1f}% "
                  f"Val Top1={h['val_top1']:.1f}% "
                  f"Val Top5={h['val_top5']:.1f}%")

    print(f"\n{'='*60}")
    print("Report complete. Check {output_dir}/ for images and data.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/convnext_best.pt")
    parser.add_argument("--wordlist", default="data/bip39_english.txt")
    parser.add_argument("--output", default="reports")
    args = parser.parse_args()

    generate_report(args.model, args.wordlist, args.output)
