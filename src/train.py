"""
Training script for blurred word classifier.

Optimized for Apple Silicon M4 Max with MPS backend.
Uses curriculum learning: starts with easy blur, increases over time.
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import (
    OnTheFlyBlurDataset,
    get_train_transforms, get_val_transforms,
)
from src.generate_data import get_available_fonts


NUM_CLASSES = 2048


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_model(model_name: str, pretrained: bool = True):
    """Create model via timm."""
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=NUM_CLASSES)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_name}")
    print(f"  Total params: {total:,}")
    print(f"  Trainable params: {trainable:,}")
    return model


def freeze_backbone(model):
    """Freeze everything except the classifier head."""
    for name, param in model.named_parameters():
        if "fc" not in name and "classifier" not in name and "head" not in name:
            param.requires_grad = False


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # OneCycleLR steps per batch
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.3f}",
            acc=f"{100.*correct/total:.1f}%",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
        )

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()

        _, top5_pred = outputs.topk(5, 1, True, True)
        correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "top1_acc": 100.0 * correct / total,
        "top5_acc": 100.0 * correct_top5 / total,
    }


def get_sigma_for_epoch(epoch: int, max_epochs: int) -> tuple:
    """Curriculum learning: gradually increase blur difficulty."""
    progress = epoch / max(max_epochs - 1, 1)

    if progress < 0.15:
        return (0.5, 3.0)   # Phase 1: learn word shapes
    elif progress < 0.35:
        return (0.5, 5.0)   # Phase 2: mild blur
    elif progress < 0.55:
        return (0.5, 7.0)   # Phase 3: medium blur
    elif progress < 0.75:
        return (0.5, 9.0)   # Phase 4: harder blur
    else:
        return (0.5, 12.0)  # Phase 5: push limits


def train(
    model_name: str = "resnet18",
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1e-3,
    samples_per_epoch: int = 51200,
    val_samples: int = 10240,
    output_dir: str = "models",
    wordlist: str = "data/bip39_english.txt",
    time_limit_minutes: float = 20.0,
):
    device = get_device()
    print(f"Device: {device}")

    model = create_model(model_name)
    model = model.to(device)

    fonts = get_available_fonts()
    print(f"\nUsing {len(fonts)} fonts")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    os.makedirs(output_dir, exist_ok=True)

    best_val_acc = 0
    history = []
    start_time = time.time()
    time_limit_sec = time_limit_minutes * 60

    print(f"\nTraining for up to {epochs} epochs (time limit: {time_limit_minutes}min)")
    print(f"  Batch size: {batch_size}, LR: {lr}")
    print(f"  Curriculum learning: sigma increases over epochs")
    print("=" * 60)

    for epoch in range(epochs):
        elapsed = time.time() - start_time
        if elapsed > time_limit_sec:
            print(f"\nTime limit reached ({time_limit_minutes}min). Stopping.")
            break

        # Curriculum: increase sigma over time
        sigma_range = get_sigma_for_epoch(epoch, epochs)
        print(f"\nEpoch {epoch+1}/{epochs} — sigma_range={sigma_range}")

        # Epoch 0: freeze backbone, train head only
        if epoch == 0:
            print("  [Warmup] Freezing backbone, training head only")
            freeze_backbone(model)
        elif epoch == 1:
            print("  [Unfreeze] Training all parameters")
            unfreeze_all(model)

        # Create fresh datasets with current sigma range
        train_ds = OnTheFlyBlurDataset(
            wordlist, fonts,
            samples_per_epoch=samples_per_epoch,
            sigma_range=sigma_range,
            transform=get_train_transforms(),
        )
        val_ds = OnTheFlyBlurDataset(
            wordlist, fonts,
            samples_per_epoch=val_samples,
            sigma_range=sigma_range,
            transform=get_val_transforms(),
        )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=4, persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=2, persistent_workers=True,
        )

        # Rebuild optimizer each time we change frozen params
        if epoch <= 1:
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                steps_per_epoch=len(train_loader),
                epochs=1,  # one cycle per epoch since we rebuild
                pct_start=0.2,
            )
        elif epoch == 2:
            # After unfreeze, new optimizer with differential LR
            backbone_params = []
            head_params = []
            for name, param in model.named_parameters():
                if "fc" in name or "classifier" in name or "head" in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
            optimizer = optim.AdamW([
                {"params": backbone_params, "lr": lr * 0.1},
                {"params": head_params, "lr": lr},
            ], weight_decay=0.01)
            remaining = epochs - epoch
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[lr * 0.1, lr],
                steps_per_epoch=len(train_loader),
                epochs=remaining,
                pct_start=0.1,
            )

        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scheduler
        )
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - start_time

        record = {
            "epoch": epoch + 1,
            "sigma_range": list(sigma_range),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1_acc"],
            "val_top5": val_metrics["top5_acc"],
            "epoch_time": epoch_time,
            "total_time": total_elapsed,
        }
        history.append(record)

        print(
            f"  Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Top1: {val_metrics['top1_acc']:.1f}% Top5: {val_metrics['top5_acc']:.1f}% | "
            f"Time: {epoch_time:.0f}s ({total_elapsed/60:.1f}min)"
        )

        if val_metrics["top1_acc"] > best_val_acc:
            best_val_acc = val_metrics["top1_acc"]
            save_path = os.path.join(output_dir, f"{model_name}_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "epoch": epoch + 1,
                "val_top1": val_metrics["top1_acc"],
                "val_top5": val_metrics["top5_acc"],
                "sigma_range": list(sigma_range),
            }, save_path)
            print(f"  -> Saved best model (top1: {best_val_acc:.1f}%)")

    # Save history
    hist_path = os.path.join(output_dir, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes")
    print(f"Best val top1: {best_val_acc:.1f}%")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples-per-epoch", type=int, default=51200)
    parser.add_argument("--val-samples", type=int, default=10240)
    parser.add_argument("--output", default="models")
    parser.add_argument("--wordlist", default="data/bip39_english.txt")
    parser.add_argument("--time-limit", type=float, default=20.0)
    args = parser.parse_args()

    train(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        output_dir=args.output,
        wordlist=args.wordlist,
        time_limit_minutes=args.time_limit,
    )
