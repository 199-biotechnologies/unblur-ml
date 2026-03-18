"""
Model definitions for blurred word classification.

Two approaches:
1. ConvNeXt V2 Tiny — fine-tuned end-to-end from ImageNet pretrained
2. DINOv2 ViT-B/14 — frozen backbone + trainable MLP head
"""

import torch
import torch.nn as nn
import timm


NUM_CLASSES = 2048


def create_convnext(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ConvNeXt V2 Tiny with custom classifier head."""
    model = timm.create_model(
        "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def create_efficientnet(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """EfficientNetV2-S as a lighter alternative."""
    model = timm.create_model(
        "tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


class DINOv2Classifier(nn.Module):
    """DINOv2 frozen backbone with trainable MLP classifier."""

    def __init__(self, num_classes: int = NUM_CLASSES, backbone: str = "dinov2_vitb14"):
        super().__init__()
        # Load DINOv2 from torch hub
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone)
        self.backbone.eval()

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DINOv2 ViT-B/14 outputs 768-dim embeddings
        embed_dim = self.backbone.embed_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # DINOv2 expects 224x224, we'll resize in transforms or here
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)


def get_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """Factory function."""
    if model_name == "convnext":
        return create_convnext(num_classes, pretrained)
    elif model_name == "efficientnet":
        return create_efficientnet(num_classes, pretrained)
    elif model_name == "dinov2":
        return DINOv2Classifier(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
