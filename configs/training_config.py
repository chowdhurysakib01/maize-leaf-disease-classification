
"""
Maize Leaf Classification
Central Training Configuration

Project:
    D:/MaizeLeaf

Supports:
    9 classification models
    Custom optimizers
    Learning rate schedulers
    Early stopping
    Mixed precision
    Gradient accumulation
    Class weighting
    Weighted sampling
    Reproducible experiments
    Automatic evaluation / XAI / LLM pipeline

Python: 3.11+
"""

import argparse
import json
import warnings

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results"

SUPPORTED_MODELS = [
    "custom_cnn",
    "resnet50",
    "efficientnet_b0",
    "mobilenet_v3",
    "densenet121",
    "convnext_tiny",
    "small_vit",
    "riceleaf_dmnet",
    "mobnetfine",
]

SUPPORTED_OPTIMIZERS = [
    "adam",
    "adamw",
    "sgd",
    "rmsprop",
]

SUPPORTED_SCHEDULERS = [
    "none",
    "cosine",
    "step",
    "plateau",
    "onecycle",
]

SUPPORTED_LOSSES = [
    "cross_entropy",
    "focal",
]

SUPPORTED_MONITORS = [
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
]

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

@dataclass
class TrainingConfig:

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model: str = "resnet50"

    pretrained: bool = True

    freeze_backbone: bool = False

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    epochs: int = 100

    batch_size: int = 16

    image_size: int = 224

    learning_rate: float = 0.0001

    optimizer: str = "adamw"

    weight_decay: float = 0.0001

    momentum: float = 0.9

    nesterov: bool = False

    adam_beta1: float = 0.9

    adam_beta2: float = 0.999

    optimizer_eps: float = 1e-8

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler: str = "cosine"

    scheduler_step_size: int = 10

    scheduler_gamma: float = 0.1

    scheduler_patience: int = 5

    scheduler_factor: float = 0.5

    min_lr: float = 1e-7

    warmup_epochs: int = 0

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_stopping: bool = True

    patience: int = 15

    min_delta: float = 0.0001

    monitor: str = "val_macro_f1"

    monitor_mode: str = "auto"

    # --------------------------------------------------------
    # LOSS AND CLASS IMBALANCE
    # --------------------------------------------------------

    loss: str = "cross_entropy"

    label_smoothing: float = 0.0

    focal_gamma: float = 2.0

    class_weighting: str = "none"

    sampler: str = "none"

    # --------------------------------------------------------
    # IMAGE AUGMENTATION
    # --------------------------------------------------------

    augmentation: bool = True

    horizontal_flip: float = 0.5

    vertical_flip: float = 0.0

    rotation_degrees: float = 10.0

    brightness: float = 0.15

    contrast: float = 0.15

    saturation: float = 0.10

    hue: float = 0.02

    normalization: str = "imagenet"

    # --------------------------------------------------------
    # GPU / MIXED PRECISION
    # --------------------------------------------------------

    device: str = "auto"

    amp: bool = True

    amp_dtype: str = "float16"

    gradient_accumulation_steps: int = 1

    max_grad_norm: float = 1.0

    # --------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------

    num_workers: int = 0

    pin_memory: Optional[bool] = None

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    seed: int = 42

    deterministic: bool = True

    # --------------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------------

    run_name: Optional[str] = None

    results_dir: Path = DEFAULT_RESULTS_DIR

    resume: Optional[Path] = None

    # --------------------------------------------------------
    # AUTOMATIC PIPELINE
    # --------------------------------------------------------

    run_evaluation: bool = True

    run_benchmark: bool = True

    run_error_analysis: bool = True

    run_xai: bool = True

    run_llm: bool = True

    save_last_checkpoint: bool = True

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    def to_dict(self):

        data = asdict(self)

        data["results_dir"] = str(
            self.results_dir
        )

        data["resume"] = (
            str(self.resume)
            if self.resume is not None
            else None
        )

        return data


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_training_config(config):

    errors = []

    if config.model not in SUPPORTED_MODELS:

        errors.append(
            f"Unsupported model: {config.model}"
        )

    if config.epochs <= 0:

        errors.append("Epochs must be positive.")

    if config.batch_size <= 0:

        errors.append("Batch size must be positive.")

    if config.image_size <= 0:

        errors.append("Image size must be positive.")

    if config.learning_rate <= 0:

        errors.append("Learning rate must be positive.")

    if config.optimizer not in SUPPORTED_OPTIMIZERS:

        errors.append(
            f"Unsupported optimizer: {config.optimizer}"
        )

    if config.weight_decay < 0:

        errors.append(
            "Weight decay cannot be negative."
        )

    if not 0 <= config.momentum < 1:

        errors.append(
            "Momentum must be in [0, 1)."
        )

    if config.nesterov and config.optimizer != "sgd":

        errors.append(
            "Nesterov is supported only with SGD."
        )

    if not 0 < config.adam_beta1 < 1:

        errors.append("Adam beta1 must be in (0, 1).")

    if not 0 < config.adam_beta2 < 1:

        errors.append("Adam beta2 must be in (0, 1).")

    if config.optimizer_eps <= 0:

        errors.append("Optimizer epsilon must be positive.")

    if config.scheduler not in SUPPORTED_SCHEDULERS:

        errors.append(
            f"Unsupported scheduler: {config.scheduler}"
        )

    if config.scheduler_step_size <= 0:

        errors.append(
            "Scheduler step size must be positive."
        )

    if not 0 < config.scheduler_gamma <= 1:

        errors.append(
            "Scheduler gamma must be in (0, 1]."
        )

    if config.scheduler_patience < 0:

        errors.append(
            "Scheduler patience cannot be negative."
        )

    if not 0 < config.scheduler_factor < 1:

        errors.append(
            "Scheduler factor must be in (0, 1)."
        )

    if config.min_lr < 0:

        errors.append("Minimum learning rate cannot be negative.")

    if config.min_lr > config.learning_rate:

        errors.append(
            "Minimum learning rate cannot exceed initial LR."
        )

    if config.warmup_epochs < 0:

        errors.append(
            "Warmup epochs cannot be negative."
        )

    if config.warmup_epochs >= config.epochs:

        errors.append(
            "Warmup epochs must be less than total epochs."
        )

    if config.patience <= 0:

        errors.append(
            "Early stopping patience must be positive."
        )

    if config.min_delta < 0:

        errors.append(
            "Minimum improvement delta cannot be negative."
        )

    if config.monitor not in SUPPORTED_MONITORS:

        errors.append(
            f"Unsupported monitoring metric: {config.monitor}"
        )

    if config.monitor_mode not in ("auto", "min", "max"):

        errors.append(
            "Monitor mode must be auto, min or max."
        )

    if config.loss not in SUPPORTED_LOSSES:

        errors.append(
            f"Unsupported loss function: {config.loss}"
        )

    if not 0 <= config.label_smoothing < 1:

        errors.append(
            "Label smoothing must be in [0, 1)."
        )

    if config.focal_gamma < 0:

        errors.append(
            "Focal gamma cannot be negative."
        )

    if config.class_weighting not in (
        "none",
        "inverse_frequency"
    ):

        errors.append(
            "Unsupported class weighting method."
        )

    if config.sampler not in ("none", "weighted"):

        errors.append(
            "Sampler must be none or weighted."
        )

    if config.class_weighting == "inverse_frequency" and config.sampler == "weighted":

        warnings.warn(
            "Both inverse-frequency loss weighting and "
            "weighted sampling are enabled. This can "
            "over-correct class imbalance. Use this "
            "combination only in a deliberate experiment.",
            RuntimeWarning
        )

    for name, value in [
        ("horizontal_flip", config.horizontal_flip),
        ("vertical_flip", config.vertical_flip),
    ]:

        if not 0 <= value <= 1:

            errors.append(
                f"{name} must be between 0 and 1."
            )

    if config.rotation_degrees < 0:

        errors.append(
            "Rotation degrees cannot be negative."
        )

    for name, value in [
        ("brightness", config.brightness),
        ("contrast", config.contrast),
        ("saturation", config.saturation),
    ]:

        if not 0 <= value <= 1:

            errors.append(
                f"{name} must be between 0 and 1."
            )

    if not 0 <= config.hue <= 0.5:

        errors.append(
            "Hue must be between 0 and 0.5."
        )

    if config.normalization not in ("imagenet", "none"):

        errors.append(
            "Normalization must be imagenet or none."
        )

    if config.amp_dtype not in ("float16", "bfloat16"):

        errors.append(
            "AMP dtype must be float16 or bfloat16."
        )

    if config.gradient_accumulation_steps <= 0:

        errors.append(
            "Gradient accumulation steps must be positive."
        )

    if config.max_grad_norm < 0:

        errors.append(
            "Maximum gradient norm cannot be negative."
        )

    if config.num_workers < 0:

        errors.append(
            "DataLoader workers cannot be negative."
        )

    if config.seed < 0:

        errors.append(
            "Seed must be non-negative."
        )

    if config.device not in (
        "auto",
        "cpu",
        "cuda",
        "cuda:0"
    ):

        errors.append(
            "Supported devices: auto, cpu, cuda, cuda:0."
        )

    if config.resume is not None:

        if not config.resume.is_file():

            errors.append(
                f"Checkpoint does not exist: {config.resume}"
            )

    if errors:

        raise ValueError(
            "\nTRAINING CONFIGURATION ERRORS:\n- "
            + "\n- ".join(errors)
        )

    if config.scheduler == "onecycle":

        if config.warmup_epochs > 0:

            warnings.warn(
                "OneCycleLR already has its own internal "
                "learning-rate schedule. External warmup "
                "will not be applied.",
                RuntimeWarning
            )

    return True


# ============================================================
# COMMAND LINE ARGUMENTS
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(

        description=(
            "Maize Leaf Classification "
            "Training Configuration"
        ),

        formatter_class=argparse.ArgumentDefaultsHelpFormatter

    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model_group = parser.add_argument_group("Model")

    model_group.add_argument(
        "--model",
        required=True,
        choices=SUPPORTED_MODELS
    )

    model_group.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    model_group.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    training_group = parser.add_argument_group("Training")

    training_group.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    training_group.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    training_group.add_argument(
        "--image-size",
        type=int,
        default=224
    )

    training_group.add_argument(
        "--lr",
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=0.0001
    )

    training_group.add_argument(
        "--optimizer",
        choices=SUPPORTED_OPTIMIZERS,
        default="adamw"
    )

    training_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001
    )

    training_group.add_argument(
        "--momentum",
        type=float,
        default=0.9
    )

    training_group.add_argument(
        "--nesterov",
        action=argparse.BooleanOptionalAction,
        default=False
    )

    training_group.add_argument(
        "--adam-beta1",
        type=float,
        default=0.9
    )

    training_group.add_argument(
        "--adam-beta2",
        type=float,
        default=0.999
    )

    training_group.add_argument(
        "--optimizer-eps",
        type=float,
        default=1e-8
    )

    training_group.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1
    )

    training_group.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0
    )

    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler_group = parser.add_argument_group("Scheduler")

    scheduler_group.add_argument(
        "--scheduler",
        choices=SUPPORTED_SCHEDULERS,
        default="cosine"
    )

    scheduler_group.add_argument(
        "--scheduler-step-size",
        type=int,
        default=10
    )

    scheduler_group.add_argument(
        "--scheduler-gamma",
        type=float,
        default=0.1
    )

    scheduler_group.add_argument(
        "--scheduler-patience",
        type=int,
        default=5
    )

    scheduler_group.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5
    )

    scheduler_group.add_argument(
        "--min-lr",
        type=float,
        default=1e-7
    )

    scheduler_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=0
    )

    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    early_group = parser.add_argument_group("Early Stopping")

    early_group.add_argument(
        "--early-stopping",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    early_group.add_argument(
        "--patience",
        type=int,
        default=15
    )

    early_group.add_argument(
        "--min-delta",
        type=float,
        default=0.0001
    )

    early_group.add_argument(
        "--monitor",
        choices=SUPPORTED_MONITORS,
        default="val_macro_f1"
    )

    early_group.add_argument(
        "--monitor-mode",
        choices=["auto", "min", "max"],
        default="auto"
    )

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss_group = parser.add_argument_group("Loss and Imbalance")

    loss_group.add_argument(
        "--loss",
        choices=SUPPORTED_LOSSES,
        default="cross_entropy"
    )

    loss_group.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0
    )

    loss_group.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0
    )

    loss_group.add_argument(
        "--class-weighting",
        choices=["none", "inverse_frequency"],
        default="none"
    )

    loss_group.add_argument(
        "--sampler",
        choices=["none", "weighted"],
        default="none"
    )

    # --------------------------------------------------------
    # AUGMENTATION
    # --------------------------------------------------------

    augmentation_group = parser.add_argument_group(
        "Image Augmentation"
    )

    augmentation_group.add_argument(
        "--augmentation",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    augmentation_group.add_argument(
        "--horizontal-flip",
        type=float,
        default=0.5
    )

    augmentation_group.add_argument(
        "--vertical-flip",
        type=float,
        default=0.0
    )

    augmentation_group.add_argument(
        "--rotation-degrees",
        type=float,
        default=10.0
    )

    augmentation_group.add_argument(
        "--brightness",
        type=float,
        default=0.15
    )

    augmentation_group.add_argument(
        "--contrast",
        type=float,
        default=0.15
    )

    augmentation_group.add_argument(
        "--saturation",
        type=float,
        default=0.10
    )

    augmentation_group.add_argument(
        "--hue",
        type=float,
        default=0.02
    )

    augmentation_group.add_argument(
        "--normalization",
        choices=["imagenet", "none"],
        default="imagenet"
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    gpu_group = parser.add_argument_group("GPU and AMP")

    gpu_group.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        default="auto"
    )

    gpu_group.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    gpu_group.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16"
    )

    # --------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------

    data_group = parser.add_argument_group("Data Loading")

    data_group.add_argument(
        "--num-workers",
        type=int,
        default=0
    )

    data_group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None
    )

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    seed_group = parser.add_argument_group("Reproducibility")

    seed_group.add_argument(
        "--seed",
        type=int,
        default=42
    )

    seed_group.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    # --------------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------------

    experiment_group = parser.add_argument_group("Experiment")

    experiment_group.add_argument(
        "--run-name",
        type=str,
        default=None
    )

    experiment_group.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR
    )

    experiment_group.add_argument(
        "--resume",
        type=Path,
        default=None
    )

    # --------------------------------------------------------
    # AUTOMATIC PIPELINE
    # --------------------------------------------------------

    pipeline_group = parser.add_argument_group(
        "Automatic Pipeline"
    )

    pipeline_group.add_argument(
        "--run-evaluation",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    pipeline_group.add_argument(
        "--run-benchmark",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    pipeline_group.add_argument(
        "--run-error-analysis",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    pipeline_group.add_argument(
        "--run-xai",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    pipeline_group.add_argument(
        "--run-llm",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    pipeline_group.add_argument(
        "--save-last-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True
    )

    # --------------------------------------------------------
    # CONFIGURATION TEST
    # --------------------------------------------------------

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration without training."
    )

    return parser


# ============================================================
# PARSE CONFIGURATION
# ============================================================

def parse_training_config(argv=None):

    parser = build_parser()

    args = parser.parse_args(argv)

    args_dict = vars(args)

    args_dict.pop("validate_only", None)

    config = TrainingConfig(**args_dict)

    validate_training_config(config)

    return config


# ============================================================
# PRINT CONFIGURATION
# ============================================================

def print_config(config):

    print("\n" + "=" * 70)

    print("MAIZE LEAF TRAINING CONFIGURATION")

    print("=" * 70)

    data = config.to_dict()

    for key, value in data.items():

        print(f"{key:<35}: {value}")

    print("=" * 70)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

def save_config(config, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config.to_dict(),
            file,
            indent=4,
            ensure_ascii=False
        )

    return output_path


# ============================================================
# EXECUTION TEST
# ============================================================

def main():

    parser = build_parser()

    args = parser.parse_args()

    validate_only = args.validate_only

    args_dict = vars(args)

    args_dict.pop("validate_only", None)

    config = TrainingConfig(**args_dict)

    validate_training_config(config)

    print_config(config)

    if validate_only:

        print("\nCONFIGURATION VALIDATION: PASSED")

        print("No training was started.")

    else:

        print(
            "\nConfiguration is valid. "
            "Training engine has not been implemented yet."
        )

    print("=" * 70)


if __name__ == "__main__":

    main()
