"""Loader and staged-unfreezing helpers for MaizeLeaf-DMNet fine-tuning.

This is not a new architecture: it reuses the exact MaizeLeaf-DMNet v2
architecture and strictly initializes it from the selected epoch-21 checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from Models.model_factory import build_model


def _load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older PyTorch
        return torch.load(path, map_location="cpu")


def build_finetuned_dmnet(checkpoint_path, num_classes=7):
    """Build DMNet v2 without downloading ImageNet weights, then strict-load checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Starting checkpoint not found: {checkpoint_path}")

    checkpoint = _load_torch_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a full training checkpoint containing 'model_state_dict'.")
    if checkpoint.get("model_name") not in (None, "maizeleaf_dmnet", "maizeleaf_dmnet_finetuned"):
        raise ValueError(f"Checkpoint model is not MaizeLeaf-DMNet: {checkpoint.get('model_name')}")

    model = build_model(
        model_name="maizeleaf_dmnet",
        num_classes=num_classes,
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    # Keep the original factory key so existing Grad-CAM/model-factory helpers work.
    model.finetune_model_name = "maizeleaf_dmnet_finetuned"
    model.finetune_source_checkpoint = str(checkpoint_path)
    model.finetune_source_epoch = int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0)))
    model.finetune_source_macro_f1 = float(checkpoint.get("best_metric", float("nan")))
    return model, checkpoint


def set_trainable_stage(model: nn.Module, stage: str) -> Dict[str, int]:
    """Set classifier-only, progressive-tail, or full-network trainability."""
    if stage not in {"classifier_only", "progressive", "full"}:
        raise ValueError(f"Unknown fine-tuning stage: {stage}")

    for parameter in model.parameters():
        parameter.requires_grad = False

    if stage == "classifier_only":
        modules = [model.classifier]
    elif stage == "progressive":
        modules = [
            model.mobile_encoder,
            model.dense_encoder,
            model.fusion_gate,
            model.classifier,
            *list(model.mobilenet_features[13:]),
            model.densenet_features.denseblock4,
            model.densenet_features.norm5,
        ]
    else:
        for parameter in model.parameters():
            parameter.requires_grad = True
        modules = []

    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError(f"Stage {stage} produced no trainable parameters.")
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def _unique_trainable(modules):
    seen = set()
    parameters: List[nn.Parameter] = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                parameters.append(parameter)
                seen.add(id(parameter))
    return parameters


def get_stage_parameter_groups(model: nn.Module, stage: str, rates: Dict[str, float], weight_decay: float):
    """Return named parameter groups with differential learning rates."""
    if stage == "classifier_only":
        specs = [("classifier", [model.classifier], rates["classifier"])]
    elif stage == "progressive":
        specs = [
            ("mobilenet_tail", [*list(model.mobilenet_features[13:])], rates["backbone"]),
            ("densenet_tail", [model.densenet_features.denseblock4, model.densenet_features.norm5], rates["backbone"]),
            ("fusion_and_encoders", [model.mobile_encoder, model.dense_encoder, model.fusion_gate], rates["fusion"]),
            ("classifier", [model.classifier], rates["classifier"]),
        ]
    elif stage == "full":
        specs = [
            ("backbones", [model.mobilenet_features, model.densenet_features], rates["backbone"]),
            ("fusion_and_encoders", [model.mobile_encoder, model.dense_encoder, model.fusion_gate], rates["fusion"]),
            ("classifier", [model.classifier], rates["classifier"]),
        ]
    else:
        raise ValueError(f"Unknown fine-tuning stage: {stage}")

    groups = []
    for name, modules, lr in specs:
        params = _unique_trainable(modules)
        if params:
            groups.append({
                "name": name,
                "params": params,
                "lr": float(lr),
                "weight_decay": float(weight_decay),
            })

    covered = {id(p) for group in groups for p in group["params"]}
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if covered != expected:
        raise RuntimeError(
            f"Parameter-group coverage error: missing={len(expected-covered)}, "
            f"extra={len(covered-expected)}"
        )
    return groups
