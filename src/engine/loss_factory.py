
"""
Maize Leaf Classification
Centralized Loss Function Factory

Supported:
1. Cross Entropy Loss
2. Weighted Cross Entropy Loss
3. Focal Loss
4. Weighted Focal Loss
5. Label Smoothing
6. Automatic inverse-frequency class weights

Class weights are calculated ONLY from training data.

Validation and test labels must never be used to
calculate training loss weights.
"""

import csv

from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.class_names import (
    PROJECT_ROOT,
    CLASS_NAMES,
    NUM_CLASSES,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_MANIFEST = (
    PROJECT_ROOT /
    "data_splits" /
    "train.csv"
)

SUPPORTED_LOSSES = (
    "cross_entropy",
    "focal",
)

SUPPORTED_WEIGHTING = (
    "none",
    "inverse_frequency",
)


# ============================================================
# CLASS COUNT LOADER
# ============================================================

def load_class_counts_from_manifest(
    manifest_path=TRAIN_MANIFEST
):
    """
    Read class distribution exclusively from train.csv.

    Returns:
        Dictionary:
            {
                class_name: image_count
            }
    """

    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():

        raise FileNotFoundError(
            f"Training manifest not found: {manifest_path}"
        )

    counts = Counter()

    with open(
        manifest_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames or "class_name" not in reader.fieldnames:

            raise ValueError(
                "Training manifest must contain class_name."
            )

        for row in reader:

            class_name = row["class_name"]

            if class_name not in CLASS_NAMES:

                raise ValueError(
                    f"Unknown class in training manifest: {class_name}"
                )

            if "split" in row and row["split"] != "train":

                raise ValueError(
                    "Non-training record found in training manifest."
                )

            counts[class_name] += 1

    missing = set(CLASS_NAMES) - set(counts)

    if missing:

        raise ValueError(
            f"Training manifest has missing classes: {sorted(missing)}"
        )

    return {
        class_name: counts[class_name]
        for class_name in CLASS_NAMES
    }


# ============================================================
# NORMALIZE CLASS COUNTS
# ============================================================

def normalize_class_counts(
    class_counts,
    num_classes=NUM_CLASSES
):
    """
    Accept:
        1. Dictionary keyed by class names.
        2. Dictionary keyed by integer class indices.
        3. List or tuple in class-index order.
        4. Counter object.

    Returns:
        List of counts in class-index order.
    """

    if class_counts is None:

        raise ValueError(
            "Class counts are required for inverse-frequency weighting."
        )

    if isinstance(class_counts, dict):

        if all(
            class_name in class_counts
            for class_name in CLASS_NAMES
        ):

            counts = [
                class_counts[class_name]
                for class_name in CLASS_NAMES
            ]

        elif all(
            index in class_counts
            for index in range(num_classes)
        ):

            counts = [
                class_counts[index]
                for index in range(num_classes)
            ]

        else:

            raise ValueError(
                "Class count dictionary must contain all "
                "seven class names or all class indices."
            )

    elif isinstance(class_counts, (list, tuple)):

        counts = list(class_counts)

    else:

        raise TypeError(
            "Class counts must be a dictionary, list or tuple."
        )

    if len(counts) != num_classes:

        raise ValueError(
            f"Expected {num_classes} class counts, "
            f"received {len(counts)}."
        )

    validated = []

    for index, count in enumerate(counts):

        if isinstance(count, bool):

            raise ValueError(
                f"Invalid count for class {index}."
            )

        if not isinstance(count, (int, float)):

            raise TypeError(
                f"Class count must be numeric: class {index}."
            )

        if not float(count).is_integer() or count <= 0:

            raise ValueError(
                f"Class {index} must have a positive integer count."
            )

        validated.append(int(count))

    return validated


# ============================================================
# CLASS WEIGHT CALCULATION
# ============================================================

def calculate_class_weights(
    class_counts,
    method="inverse_frequency",
    device=None,
    num_classes=NUM_CLASSES
):
    """
    Inverse-frequency formula:

        weight_i = N / (C * n_i)

    Where:

        N = Total training images
        C = Number of classes
        n_i = Training images in class i

    The frequency-weighted average of class weights is 1.
    The ordinary arithmetic mean across classes need not be 1.
    """

    if method not in SUPPORTED_WEIGHTING:

        raise ValueError(
            f"Unsupported class weighting method: {method}"
        )

    if method == "none":

        return None

    counts = normalize_class_counts(
        class_counts,
        num_classes=num_classes
    )

    total_samples = sum(counts)

    weights = [

        total_samples / (
            num_classes * count
        )

        for count in counts

    ]

    weight_tensor = torch.tensor(
        weights,
        dtype=torch.float32
    )

    if not torch.isfinite(weight_tensor).all():

        raise ValueError(
            "Calculated class weights contain invalid values."
        )

    if device is not None:

        weight_tensor = weight_tensor.to(device)

    return weight_tensor


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss.

    Formula:

        FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)

    Features:
        - Optional class weighting.
        - Configurable gamma.
        - Label smoothing.
        - Mean, sum and none reductions.

    Input:
        logits: (batch_size, num_classes)
        targets: (batch_size,)
    """

    def __init__(
        self,
        gamma=2.0,
        alpha=None,
        label_smoothing=0.0,
        reduction="mean"
    ):

        super().__init__()

        if gamma < 0:

            raise ValueError(
                "Focal gamma cannot be negative."
            )

        if not 0 <= label_smoothing < 1:

            raise ValueError(
                "Label smoothing must be in [0, 1)."
            )

        if reduction not in ("mean", "sum", "none"):

            raise ValueError(
                "Reduction must be mean, sum or none."
            )

        if alpha is not None:

            if not isinstance(alpha, torch.Tensor):

                alpha = torch.as_tensor(
                    alpha,
                    dtype=torch.float32
                )

            alpha = alpha.detach().clone().float()

            if alpha.ndim != 1:

                raise ValueError(
                    "Alpha must be a one-dimensional tensor."
                )

            if not torch.isfinite(alpha).all():

                raise ValueError(
                    "Alpha contains invalid values."
                )

            if (alpha <= 0).any():

                raise ValueError(
                    "All alpha weights must be positive."
                )

        self.register_buffer(
            "alpha",
            alpha
        )

        self.gamma = float(gamma)

        self.label_smoothing = float(
            label_smoothing
        )

        self.reduction = reduction

    def forward(self, logits, targets):

        if logits.ndim != 2:

            raise ValueError(
                "Logits must have shape (batch_size, num_classes)."
            )

        if targets.ndim != 1:

            raise ValueError(
                "Targets must have shape (batch_size,)."
            )

        if logits.shape[0] != targets.shape[0]:

            raise ValueError(
                "Logits and targets batch sizes differ."
            )

        num_classes = logits.shape[1]

        if self.alpha is not None:

            if self.alpha.numel() != num_classes:

                raise ValueError(
                    "Alpha length does not match number of classes."
                )

        # Calculate log probabilities in FP32 for stability.

        log_probs = F.log_softmax(
            logits.float(),
            dim=1
        )

        probabilities = log_probs.exp()

        # Select probability of the actual target class.

        target_log_probs = log_probs.gather(
            dim=1,
            index=targets.long().unsqueeze(1)
        ).squeeze(1)

        target_probs = target_log_probs.exp()

        # Standard cross entropy with optional label smoothing.

        ce_loss = F.cross_entropy(
            logits.float(),
            targets.long(),
            reduction="none",
            label_smoothing=self.label_smoothing
        )

        # Focal modulation.

        focal_factor = (
            1.0 - target_probs
        ).clamp(min=0.0, max=1.0).pow(
            self.gamma
        )

        loss = focal_factor * ce_loss

        # Optional class weighting.

        if self.alpha is not None:

            alpha = self.alpha.to(
                device=logits.device,
                dtype=loss.dtype
            )

            alpha_t = alpha[
                targets.long()
            ]

            loss = alpha_t * loss

        if self.reduction == "mean":

            return loss.mean()

        if self.reduction == "sum":

            return loss.sum()

        return loss


# ============================================================
# LOSS FACTORY
# ============================================================

def build_loss(
    config,
    class_counts=None,
    device=None
):
    """
    Create the configured loss function.

    Args:
        config:
            TrainingConfig instance.

        class_counts:
            Training-set class distribution.

        device:
            Selected CPU/CUDA device.

    Returns:
        loss_function, class_weights
    """

    loss_name = config.loss.lower()

    weighting_method = config.class_weighting.lower()

    if loss_name not in SUPPORTED_LOSSES:

        raise ValueError(
            f"Unsupported loss: {loss_name}"
        )

    if weighting_method not in SUPPORTED_WEIGHTING:

        raise ValueError(
            f"Unsupported weighting: {weighting_method}"
        )

    class_weights = calculate_class_weights(
        class_counts=class_counts,
        method=weighting_method,
        device=device
    ) if weighting_method != "none" else None

    # --------------------------------------------------------
    # CROSS ENTROPY
    # --------------------------------------------------------

    if loss_name == "cross_entropy":

        loss_function = nn.CrossEntropyLoss(

            weight=class_weights,

            label_smoothing=config.label_smoothing,

        )

    # --------------------------------------------------------
    # FOCAL LOSS
    # --------------------------------------------------------

    elif loss_name == "focal":

        loss_function = FocalLoss(

            gamma=config.focal_gamma,

            alpha=class_weights,

            label_smoothing=config.label_smoothing,

            reduction="mean",

        )

    else:

        raise RuntimeError(
            "Unexpected loss factory state."
        )

    if device is not None:

        loss_function = loss_function.to(device)

    return loss_function, class_weights


# ============================================================
# LOSS SUMMARY
# ============================================================

def get_loss_summary(
    config,
    class_counts=None,
    class_weights=None
):
    """
    Generate loss configuration information
    for experiment reporting.
    """

    return {

        "loss_name": config.loss,

        "class_weighting": config.class_weighting,

        "label_smoothing": config.label_smoothing,

        "focal_gamma": (
            config.focal_gamma
            if config.loss == "focal"
            else None
        ),

        "class_counts": (
            {
                name: int(count)
                for name, count in zip(
                    CLASS_NAMES,
                    normalize_class_counts(class_counts)
                )
            }
            if class_counts is not None
            else None
        ),

        "class_weights": (
            {
                name: float(weight)
                for name, weight in zip(
                    CLASS_NAMES,
                    class_weights.detach().cpu().tolist()
                )
            }
            if class_weights is not None
            else None
        ),

        "weight_source": (
            "training_manifest_only"
            if class_weights is not None
            else None
        ),

    }


# ============================================================
# PRINT LOSS REPORT
# ============================================================

def print_loss_report(
    loss_function,
    class_counts=None,
    class_weights=None
):

    print("\n" + "=" * 70)

    print("LOSS FUNCTION REPORT")

    print("=" * 70)

    print(
        f"Loss: {loss_function.__class__.__name__}"
    )

    if class_counts is not None:

        counts = normalize_class_counts(
            class_counts
        )

        print("\nTRAINING CLASS DISTRIBUTION")

        for name, count in zip(CLASS_NAMES, counts):

            print(f"{name:<40}: {count}")

    if class_weights is not None:

        print("\nCLASS WEIGHTS")

        for name, weight in zip(
            CLASS_NAMES,
            class_weights.detach().cpu().tolist()
        ):

            print(f"{name:<40}: {weight:.6f}")

    print("=" * 70)


# ============================================================
# EXECUTION TEST
# ============================================================

def main():

    from configs.training_config import TrainingConfig

    print("=" * 70)

    print("MAIZE LEAF - LOSS FACTORY TEST")

    print("=" * 70)

    # --------------------------------------------------------
    # Load actual training distribution
    # --------------------------------------------------------

    class_counts = load_class_counts_from_manifest()

    print("\nTRAINING CLASS DISTRIBUTION")

    for name, count in class_counts.items():

        print(f"{name:<40}: {count}")

    assert sum(class_counts.values()) == 2623

    print("\n[PASS] Training-only class count loading")

    # --------------------------------------------------------
    # Test class weight calculation
    # --------------------------------------------------------

    
    weights = calculate_class_weights(
        class_counts,
        method="inverse_frequency"
    )

    assert weights.shape == (NUM_CLASSES,)

    assert torch.isfinite(weights).all()

    assert torch.all(weights > 0)

    # Verify frequency-weighted average, not ordinary mean.
    counts_tensor = torch.tensor(
        normalize_class_counts(class_counts),
        dtype=torch.float32
    )

    frequency_weighted_mean = (
        (weights * counts_tensor).sum()
        / counts_tensor.sum()
    )

    assert abs(
        frequency_weighted_mean.item() - 1.0
    ) < 1e-5

    print(
        f"[PASS] Inverse-frequency class weights "
        f"(weighted mean={frequency_weighted_mean.item():.6f})"
    )


    # --------------------------------------------------------
    # Test unweighted CE
    # --------------------------------------------------------

    config = TrainingConfig(
        model="custom_cnn",
        loss="cross_entropy",
        class_weighting="none"
    )

    ce_loss, ce_weights = build_loss(
        config,
        class_counts=class_counts
    )

    logits = torch.randn(16, NUM_CLASSES, requires_grad=True)

    targets = torch.randint(0, NUM_CLASSES, (16,))

    loss = ce_loss(logits, targets)

    loss.backward()

    assert torch.isfinite(loss)

    assert logits.grad is not None

    assert ce_weights is None

    print("[PASS] Cross Entropy Loss forward/backward")

    # --------------------------------------------------------
    # Test weighted CE
    # --------------------------------------------------------

    config = TrainingConfig(
        model="custom_cnn",
        loss="cross_entropy",
        class_weighting="inverse_frequency",
        label_smoothing=0.1
    )

    weighted_ce, ce_weights = build_loss(
        config,
        class_counts=class_counts
    )

    logits = torch.randn(
        16,
        NUM_CLASSES,
        requires_grad=True
    )

    loss = weighted_ce(logits, targets)

    loss.backward()

    assert torch.isfinite(loss)

    assert ce_weights is not None

    print("[PASS] Weighted Cross Entropy with label smoothing")

    # --------------------------------------------------------
    # Test unweighted Focal Loss
    # --------------------------------------------------------

    config = TrainingConfig(
        model="custom_cnn",
        loss="focal",
        class_weighting="none",
        focal_gamma=2.0
    )

    focal_loss, focal_weights = build_loss(
        config,
        class_counts=class_counts
    )

    logits = torch.randn(
        16,
        NUM_CLASSES,
        requires_grad=True
    )

    loss = focal_loss(logits, targets)

    loss.backward()

    assert torch.isfinite(loss)

    assert logits.grad is not None

    assert focal_weights is None

    print("[PASS] Focal Loss forward/backward")

    # --------------------------------------------------------
    # Test weighted Focal Loss
    # --------------------------------------------------------

    config = TrainingConfig(
        model="custom_cnn",
        loss="focal",
        class_weighting="inverse_frequency",
        focal_gamma=2.0,
        label_smoothing=0.05
    )

    weighted_focal, focal_weights = build_loss(
        config,
        class_counts=class_counts
    )

    logits = torch.randn(
        16,
        NUM_CLASSES,
        requires_grad=True
    )

    loss = weighted_focal(logits, targets)

    loss.backward()

    assert torch.isfinite(loss)

    assert focal_weights is not None

    print("[PASS] Weighted Focal Loss")

    # --------------------------------------------------------
    # Test focal reductions
    # --------------------------------------------------------

    test_logits = torch.randn(8, NUM_CLASSES)

    test_targets = torch.randint(
        0,
        NUM_CLASSES,
        (8,)
    )

    for reduction in ("mean", "sum", "none"):

        test_loss = FocalLoss(
            gamma=2.0,
            reduction=reduction
        )

        output = test_loss(
            test_logits,
            test_targets
        )

        if reduction == "none":

            assert output.shape == (8,)

        else:

            assert output.ndim == 0

        assert torch.isfinite(output).all()

    print("[PASS] Focal Loss reduction modes")

    # --------------------------------------------------------
    # Test class weight validation
    # --------------------------------------------------------

    try:

        calculate_class_weights(
            [100, 200],
            method="inverse_frequency"
        )

        raise AssertionError(
            "Invalid class counts were accepted."
        )

    except ValueError:

        print("[PASS] Invalid class count protection")

    # --------------------------------------------------------
    # Test summary
    # --------------------------------------------------------

    summary = get_loss_summary(
        config,
        class_counts,
        focal_weights
    )

    assert summary["weight_source"] == (
        "training_manifest_only"
    )

    print("[PASS] Loss experiment summary")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\nLOSS FACTORY TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
