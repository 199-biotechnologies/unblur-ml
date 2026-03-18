"""
Enhanced inference pipeline v2 — integrates all improvements:

A) Super-resolution preprocessing (Real-ESRGAN 4x upscaling)
B) Adaptive TTA — only apply when confidence is low
C) Multi-scale inference — run at multiple resolutions, ensemble
D) Self-training ready — confidence-based output for pseudo-labeling
E) Adaptive normalization — detect input quality, normalize only when needed

Also: aspect-ratio-preserving resize, BIP-39 checksum validation ready.
"""

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_val_transforms, INPUT_H, INPUT_W, NORM_MEAN, NORM_STD
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# E) Adaptive Normalization
# ============================================================

def detect_input_quality(img: np.ndarray) -> dict:
    """Analyze input image to decide what normalization is needed."""
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img

    mean_val = gray.mean()
    std_val = gray.std()
    pmin, pmax = np.percentile(gray, [5, 95])
    contrast_range = pmax - pmin

    # Check if image has color (not grayscale)
    is_colored = False
    if len(img.shape) == 3 and img.shape[2] == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation_mean = hsv[:, :, 1].mean()
        is_colored = saturation_mean > 20

    return {
        "mean": mean_val,
        "std": std_val,
        "contrast_range": contrast_range,
        "is_inverted": mean_val < 128,       # Light on dark
        "is_low_contrast": contrast_range < 100,
        "is_colored": is_colored,
        "needs_normalization": (contrast_range < 100) or (mean_val < 128) or is_colored,
    }


def adaptive_normalize(img: np.ndarray) -> np.ndarray:
    """
    Only normalize when the input actually needs it.
    Clean blur → pass through untouched.
    Grey/faded/colored/inverted → fix it.
    """
    quality = detect_input_quality(img)

    if not quality["needs_normalization"]:
        # Input looks like clean training data — don't touch it
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    # Convert to grayscale
    if len(img.shape) == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()

    # Invert if dark background
    if quality["is_inverted"]:
        gray = 255 - gray

    # Only apply CLAHE if low contrast
    if quality["is_low_contrast"]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

    # Stretch contrast
    pmin, pmax = np.percentile(gray, [2, 98])
    if pmax - pmin > 10:
        gray = np.clip((gray - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# ============================================================
# A) Super-Resolution with Real-ESRGAN
# ============================================================

_esrgan_model = None

def get_esrgan():
    """Lazy-load Real-ESRGAN model (downloads weights on first use)."""
    global _esrgan_model
    if _esrgan_model is None:
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            _esrgan_model = RealESRGANer(
                scale=4, model_path=None, dni_weight=None,
                model=rrdb, tile=0, tile_pad=10, pre_pad=0, half=False,
            )
            # Download weights if needed
            import urllib.request
            weights_dir = os.path.expanduser("~/.cache/realesrgan")
            os.makedirs(weights_dir, exist_ok=True)
            weights_path = os.path.join(weights_dir, "RealESRGAN_x4plus.pth")
            if not os.path.exists(weights_path):
                print("Downloading Real-ESRGAN weights...")
                urllib.request.urlretrieve(
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
                    weights_path,
                )
            _esrgan_model = RealESRGANer(
                scale=4, model_path=weights_path, dni_weight=None,
                model=rrdb, tile=0, tile_pad=10, pre_pad=0, half=False,
            )
        except Exception as e:
            print(f"Real-ESRGAN not available: {e}")
            return None
    return _esrgan_model


def super_resolve(img: np.ndarray) -> np.ndarray:
    """4x upscale using Real-ESRGAN. Falls back to bicubic if unavailable."""
    esrgan = get_esrgan()
    if esrgan is not None:
        try:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            output, _ = esrgan.enhance(bgr, outscale=4)
            return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"ESRGAN failed, falling back to bicubic: {e}")

    # Fallback: bicubic 4x upscale
    h, w = img.shape[:2]
    return cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)


# ============================================================
# Aspect-ratio-preserving resize
# ============================================================

def pad_to_aspect_ratio(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize preserving aspect ratio, pad to target dimensions."""
    h, w = img.shape[:2]
    target_aspect = target_w / target_h
    img_aspect = w / h

    if img_aspect > target_aspect:
        new_w = target_w
        new_h = max(1, int(target_w / img_aspect))
    else:
        new_h = target_h
        new_w = max(1, int(target_h * img_aspect))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Background color from edge pixels
    bg_val = int(np.median(resized[0, :, 0]))
    canvas = np.full((target_h, target_w, 3), bg_val, dtype=np.uint8)

    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ============================================================
# B) Adaptive TTA
# ============================================================

def get_tta_transforms() -> list[A.Compose]:
    """TTA variants — shifts, brightness, scale."""
    base = [A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()]

    return [
        # Original
        A.Compose([A.Resize(INPUT_H, INPUT_W)] + base),
        # Slight shifts
        A.Compose([A.Resize(INPUT_H, INPUT_W),
                    A.Affine(translate_percent={"x": -0.04}, p=1.0)] + base),
        A.Compose([A.Resize(INPUT_H, INPUT_W),
                    A.Affine(translate_percent={"x": 0.04}, p=1.0)] + base),
        # Brightness variants
        A.Compose([A.Resize(INPUT_H, INPUT_W),
                    A.RandomBrightnessContrast(brightness_limit=(0.08, 0.08), contrast_limit=(0.08, 0.08), p=1.0)] + base),
        A.Compose([A.Resize(INPUT_H, INPUT_W),
                    A.RandomBrightnessContrast(brightness_limit=(-0.08, -0.08), contrast_limit=(-0.08, -0.08), p=1.0)] + base),
        # Scale up (zoom into center)
        A.Compose([A.Resize(int(INPUT_H * 1.08), int(INPUT_W * 1.08)),
                    A.CenterCrop(INPUT_H, INPUT_W)] + base),
    ]


def run_model_batch(model, imgs_list: list[np.ndarray], device) -> torch.Tensor:
    """Run model on a list of numpy images, return softmax probs."""
    tfm = A.Compose([
        A.Resize(INPUT_H, INPUT_W),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ToTensorV2(),
    ])
    tensors = []
    for img in imgs_list:
        tensors.append(tfm(image=img)["image"])
    batch = torch.stack(tensors).to(device)

    with torch.no_grad():
        logits = model(batch)
        return F.softmax(logits, dim=1)


# ============================================================
# C) Multi-Scale Inference
# ============================================================

SCALES = [
    (64, 192),    # v1 resolution
    (128, 384),   # v2 resolution (native)
    (192, 576),   # 1.5x
]


def predict_multiscale(
    model: torch.nn.Module,
    img: np.ndarray,
    device: torch.device,
    scales: list[tuple] = None,
) -> torch.Tensor:
    """Run model at multiple resolutions, average softmax."""
    scales = scales or SCALES
    all_probs = []

    base_norm = A.Normalize(mean=NORM_MEAN, std=NORM_STD)

    for h, w in scales:
        resized = pad_to_aspect_ratio(img, w, h)
        # Resize to model input
        final = cv2.resize(resized, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
        tfm = A.Compose([A.Normalize(mean=NORM_MEAN, std=NORM_STD), ToTensorV2()])
        tensor = tfm(image=final)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

    return torch.stack(all_probs).mean(dim=0)


# ============================================================
# Main prediction function
# ============================================================

def predict_word(
    model: torch.nn.Module,
    img: np.ndarray,
    words: list[str],
    device: torch.device,
    top_k: int = 10,
    use_sr: bool = False,        # A) Super-resolution
    use_tta: bool = True,        # B) Test-time augmentation
    use_multiscale: bool = False, # C) Multi-scale
    use_normalization: bool = True, # E) Adaptive normalization
    adaptive: bool = True,       # Auto-decide based on confidence
) -> list[tuple[str, float]]:
    """
    Full inference pipeline combining A-E.

    Default behavior (adaptive=True):
    1. Normalize only if input needs it (E)
    2. First pass without TTA
    3. If confidence < 0.7, retry with TTA (B)
    4. If still < 0.5, try multi-scale (C)
    5. If still < 0.3 and use_sr=True, try super-resolution (A)
    """
    # Ensure RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # E) Adaptive normalization
    if use_normalization:
        img_norm = adaptive_normalize(img)
    else:
        img_norm = img

    # Aspect-ratio-preserving resize
    img_padded = pad_to_aspect_ratio(img_norm, INPUT_W, INPUT_H)

    model.eval()

    if adaptive:
        # Step 1: Quick single pass
        tfm = get_val_transforms()
        tensor = tfm(image=img_padded)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)

        top1_conf = probs.max().item()

        # Step 2: If low confidence, add TTA
        if top1_conf < 0.7 and use_tta:
            tta_probs = [probs]
            for tfm_tta in get_tta_transforms()[1:]:  # skip original (already done)
                aug = tfm_tta(image=img_padded)
                t = aug["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    tta_probs.append(F.softmax(model(t), dim=1))
            probs = torch.stack(tta_probs).mean(dim=0)
            top1_conf = probs.max().item()

        # Step 3: If still low, try multi-scale
        if top1_conf < 0.5 and use_multiscale:
            ms_probs = predict_multiscale(model, img_norm, device)
            probs = (probs + ms_probs) / 2
            top1_conf = probs.max().item()

        # Step 4: If still low, try super-resolution
        if top1_conf < 0.3 and use_sr:
            sr_img = super_resolve(img)
            if use_normalization:
                sr_img = adaptive_normalize(sr_img)
            sr_padded = pad_to_aspect_ratio(sr_img, INPUT_W, INPUT_H)
            tfm = get_val_transforms()
            tensor = tfm(image=sr_padded)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                sr_probs = F.softmax(model(tensor), dim=1)
            probs = (probs + sr_probs) / 2

    else:
        # Non-adaptive: run everything requested
        all_probs = []

        # Base prediction
        tfm = get_val_transforms()
        tensor = tfm(image=img_padded)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            all_probs.append(F.softmax(model(tensor), dim=1))

        # TTA
        if use_tta:
            for tfm_tta in get_tta_transforms()[1:]:
                aug = tfm_tta(image=img_padded)
                t = aug["image"].unsqueeze(0).to(device)
                with torch.no_grad():
                    all_probs.append(F.softmax(model(t), dim=1))

        # Multi-scale
        if use_multiscale:
            all_probs.append(predict_multiscale(model, img_norm, device))

        # Super-resolution
        if use_sr:
            sr_img = super_resolve(img)
            if use_normalization:
                sr_img = adaptive_normalize(sr_img)
            sr_padded = pad_to_aspect_ratio(sr_img, INPUT_W, INPUT_H)
            tfm_sr = get_val_transforms()
            t_sr = tfm_sr(image=sr_padded)["image"].unsqueeze(0).to(device)
            with torch.no_grad():
                all_probs.append(F.softmax(model(t_sr), dim=1))

        probs = torch.stack(all_probs).mean(dim=0)

    # Extract top-K
    top_probs, top_indices = probs.topk(top_k, dim=1)
    results = []
    for i in range(top_k):
        idx = top_indices[0, i].item()
        prob = top_probs[0, i].item()
        results.append((words[idx], prob))

    return results


# ============================================================
# D) Confidence scoring for pseudo-labeling
# ============================================================

def predict_with_confidence(
    model: torch.nn.Module,
    img: np.ndarray,
    words: list[str],
    device: torch.device,
    **kwargs,
) -> dict:
    """Prediction with confidence metadata for self-training."""
    results = predict_word(model, img, words, device, top_k=10, **kwargs)

    top1_word, top1_conf = results[0]
    top2_word, top2_conf = results[1]

    probs = [r[1] for r in results]
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    margin = top1_conf - top2_conf

    return {
        "prediction": top1_word,
        "confidence": top1_conf,
        "top_k": results,
        "entropy": entropy,
        "margin": margin,
        "is_confident": top1_conf > 0.5 and margin > 0.2,
        "is_pseudo_label_candidate": top1_conf > 0.8 and margin > 0.4,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    import timm

    parser = argparse.ArgumentParser(description="Enhanced inference v2")
    parser.add_argument("image", help="Path to blurred word image")
    parser.add_argument("--model", default="models/resnet18_best.pt")
    parser.add_argument("--wordlist", default="data/bip39_english.txt")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sr", action="store_true", help="Enable super-resolution")
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--multiscale", action="store_true")
    parser.add_argument("--no-adaptive", action="store_true", help="Run all enhancements regardless of confidence")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=2048)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    with open(args.wordlist) as f:
        words = [line.strip() for line in f if line.strip()]

    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = predict_with_confidence(
        model, img, words, device,
        use_sr=args.sr,
        use_tta=not args.no_tta,
        use_normalization=not args.no_normalize,
        use_multiscale=args.multiscale,
        adaptive=not args.no_adaptive,
    )

    print(f"\nPrediction: '{result['prediction']}' ({result['confidence']*100:.1f}%)")
    print(f"Confident: {result['is_confident']} | Pseudo-label candidate: {result['is_pseudo_label_candidate']}")
    print(f"Entropy: {result['entropy']:.3f} | Margin: {result['margin']:.3f}")
    print(f"\nTop {args.top_k}:")
    for word, prob in result["top_k"]:
        print(f"  {word:15s} {prob*100:5.1f}%")
