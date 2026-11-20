"""Maize Leaf Classification - Optimizer Factory.

Supports Adam, AdamW, SGD and RMSprop. In addition to the original
single-group interface, build_optimizer accepts named parameter groups
with independent learning rates for staged fine-tuning.
"""

from __future__ import annotations

import torch

SUPPORTED_OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop")


def get_parameter_summary(model):
    total = trainable = frozen = 0
    trainable_names, frozen_names = [], []
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
            trainable_names.append(name)
        else:
            frozen += count
            frozen_names.append(name)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "trainable_parameter_names": trainable_names,
        "frozen_parameter_names": frozen_names,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
    }


def _prepare_parameter_groups(model, config, parameter_groups=None):
    all_trainable = {
        id(parameter): parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if not all_trainable:
        raise ValueError("The model has no trainable parameters.")

    if parameter_groups is None:
        return [{
            "params": list(all_trainable.values()),
            "lr": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
            "name": "all_trainable",
        }]

    prepared = []
    seen = set()
    for index, source_group in enumerate(parameter_groups):
        group = dict(source_group)
        source_params = group.pop("params", None)
        if source_params is None:
            raise ValueError(f"Parameter group {index} has no 'params'.")
        params = [p for p in source_params if p.requires_grad]
        if not params:
            continue
        for parameter in params:
            key = id(parameter)
            if key in seen:
                raise ValueError("A trainable parameter appears in more than one optimizer group.")
            if key not in all_trainable:
                raise ValueError("Optimizer group contains a parameter not owned by the model.")
            seen.add(key)
        group["params"] = params
        group.setdefault("lr", float(config.learning_rate))
        group.setdefault("weight_decay", float(config.weight_decay))
        group.setdefault("name", f"group_{index}")
        if float(group["lr"]) <= 0:
            raise ValueError(f"Learning rate must be positive in {group['name']}.")
        if float(group["weight_decay"]) < 0:
            raise ValueError(f"Weight decay cannot be negative in {group['name']}.")
        prepared.append(group)

    missing = set(all_trainable) - seen
    if missing:
        raise ValueError(
            f"Parameter groups omit {len(missing)} trainable parameter tensors. "
            "Every trainable parameter must be optimized exactly once."
        )
    if not prepared:
        raise ValueError("No trainable parameters remain in the supplied groups.")
    return prepared


def build_optimizer(model, config, parameter_groups=None):
    """Build an optimizer; optional groups enable differential learning rates."""
    optimizer_name = str(config.optimizer).lower()
    if optimizer_name not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Available: {SUPPORTED_OPTIMIZERS}")
    if config.learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")
    if config.weight_decay < 0:
        raise ValueError("Weight decay cannot be negative.")

    groups = _prepare_parameter_groups(model, config, parameter_groups)
    common = {
        "lr": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
    }
    eps = float(getattr(config, "optimizer_eps", 1e-8))
    betas = (
        float(getattr(config, "adam_beta1", 0.9)),
        float(getattr(config, "adam_beta2", 0.999)),
    )

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(groups, betas=betas, eps=eps, **common)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(groups, betas=betas, eps=eps, **common)
    elif optimizer_name == "sgd":
        momentum = float(getattr(config, "momentum", 0.9))
        nesterov = bool(getattr(config, "nesterov", False))
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov requires positive momentum.")
        optimizer = torch.optim.SGD(groups, momentum=momentum, nesterov=nesterov, **common)
    else:
        optimizer = torch.optim.RMSprop(
            groups,
            alpha=0.99,
            eps=eps,
            momentum=float(getattr(config, "momentum", 0.0)),
            **common,
        )
    return optimizer


def get_optimizer_summary(optimizer):
    groups = []
    for index, group in enumerate(optimizer.param_groups):
        groups.append({
            "group": index,
            "name": group.get("name", f"group_{index}"),
            "parameters": sum(parameter.numel() for parameter in group["params"]),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group.get("weight_decay", 0.0)),
            "momentum": group.get("momentum"),
            "betas": group.get("betas"),
        })
    return {
        "optimizer": optimizer.__class__.__name__,
        "parameter_groups": len(optimizer.param_groups),
        "total_optimized_parameters": sum(group["parameters"] for group in groups),
        "groups": groups,
    }


def print_optimizer_report(model, optimizer):
    parameter_info = get_parameter_summary(model)
    optimizer_info = get_optimizer_summary(optimizer)
    print("\n" + "=" * 70)
    print("OPTIMIZER AND PARAMETER REPORT")
    print("=" * 70)
    print(f"Optimizer: {optimizer_info['optimizer']}")
    print(f"Total parameters: {parameter_info['total_parameters']:,}")
    print(f"Trainable parameters: {parameter_info['trainable_parameters']:,}")
    print(f"Frozen parameters: {parameter_info['frozen_parameters']:,}")
    print(f"Trainable percentage: {parameter_info['trainable_percentage']:.2f}%")
    print(f"Optimized parameters: {optimizer_info['total_optimized_parameters']:,}")
    print("\nPARAMETER GROUPS")
    for group in optimizer_info["groups"]:
        print(
            f"{group['name']}: LR={group['learning_rate']:.2e}, "
            f"WD={group['weight_decay']:.2e}, Parameters={group['parameters']:,}"
        )
    print("=" * 70)
