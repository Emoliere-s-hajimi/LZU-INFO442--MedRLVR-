"""Baseline models for glioma recurrence vs radiation necrosis classification.

Available baselines (all 3D, all return ``{seg, cls}`` dict):

    VGG family:
        - build_vgg11
        - build_vgg16

    U-Net family:
        - build_unet
        - build_attention_unet
        - build_nested_unet  (UNet++)

    ResNet family:
        - build_resnet10
        - build_resnet50
        - build_mresnet      (Multi-scale ResNet)

    Other:
        - build_densenet121
        - build_vmunet       (Vision Mamba UNet)
        - build_swin_unetr   (Swin Transformer UNETR)

Usage::

    from src.models.baselines import build_unet
    model = build_unet(config={'model': {'base_c': 16}})
    out = model(image)  # {'seg': ..., 'cls': ...}
"""
from ._common import BaselineWrapper, ClassificationHead3D, SegmentationHead3D
from .vgg import build_vgg11, build_vgg16, VGG11_3D, VGG16_3D
from .unet_family import (
    build_unet, build_attention_unet, build_nested_unet,
    UNet3D, AttentionUNet3D, NestedUNet3D,
)
from .resnet import (
    build_resnet10, build_resnet50, build_mresnet,
    ResNet10_3D, ResNet50_3D, MResNet3D,
)
from .densenet import build_densenet121, DenseNet3D
from .vmunet import build_vmunet, VMUNet3D
from .swin_unetr import build_swin_unetr, SwinUNETR3D


BASELINE_REGISTRY = {
    "vgg11": build_vgg11,
    "vgg16": build_vgg16,
    "unet": build_unet,
    "attention_unet": build_attention_unet,
    "nested_unet": build_nested_unet,
    "unetpp": build_nested_unet,  # alias
    "resnet10": build_resnet10,
    "resnet50": build_resnet50,
    "mresnet": build_mresnet,
    "densenet121": build_densenet121,
    "vmunet": build_vmunet,
    "swin_unetr": build_swin_unetr,
}


def build_baseline(name: str, config=None):
    """Factory: build a baseline model by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline: '{name}'. "
            f"Available: {sorted(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[name](config)


__all__ = [
    "BaselineWrapper", "ClassificationHead3D", "SegmentationHead3D",
    "build_vgg11", "build_vgg16",
    "build_unet", "build_attention_unet", "build_nested_unet",
    "build_resnet10", "build_resnet50", "build_mresnet",
    "build_densenet121",
    "build_vmunet", "build_swin_unetr",
    "BASELINE_REGISTRY", "build_baseline",
    "VGG11_3D", "VGG16_3D",
    "UNet3D", "AttentionUNet3D", "NestedUNet3D",
    "ResNet10_3D", "ResNet50_3D", "MResNet3D",
    "DenseNet3D", "VMUNet3D", "SwinUNETR3D",
]
