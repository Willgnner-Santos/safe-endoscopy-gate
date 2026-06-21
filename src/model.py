# -*- coding: utf-8 -*-
"""Construction of timm models for binary classification (1 logit)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import config as C


def build_model(model_key: str):
    """Returns a timm model with a binary head (num_classes=1)."""
    import timm
    name = C.MODELS[model_key]
    model = timm.create_model(name, pretrained=True, num_classes=1)
    return model


def param_groups(model, lr_backbone=None, lr_head=None):
    """Separates backbone (lower lr) from classification head (higher lr)."""
    lr_b = lr_backbone or C.LR_BACKBONE
    lr_h = lr_head or C.LR_HEAD
    head_keys = ("classifier", "fc", "head")
    head, backbone = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if any(k in n.lower() for k in head_keys) else backbone).append(p)
    groups = []
    if backbone: groups.append({"params": backbone, "lr": lr_b})
    if head:     groups.append({"params": head, "lr": lr_h})
    return groups
