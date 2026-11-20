
"""
Maize Leaf Classification - Complete Training and Research Pipeline
Project: D:/MaizeLeaf

Pipeline:
1. Fixed image-level train/validation/test splits
2. Model training and validation
3. Training history and timing
4. Best and last checkpoints
5. Training curves
6. Validation metrics and per-image predictions
7. Efficiency benchmarking
8. Error analysis
9. Grad-CAM
10. Grad-CAM++
11. LIME
12. English and Bangla LLM research reports

SHAP is intentionally excluded.
The held-out test set is never accessed.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import warnings

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Post-training pipeline defaults.
# These can be changed without modifying the training engine.

XAI_SAMPLES_PER_CLASS = 5

XAI_SAMPLE_STRATEGY = "balanced"

XAI_TARGET_CLASS = "predicted"

BENCHMARK_BATCH_SIZES = ["1", "8", "16"]

BENCHMARK_WARMUP = 5

BENCHMARK_ITERATIONS = 20

ERROR_TOP_K = 10

ERROR_CONFIDENCE_THRESHOLD = 0.80

ERROR_CONTACT_SHEET_LIMIT = 48

LLM_MODEL = os.environ.get(
    "MAIZE_LLM_MODEL",
    "qwen2.5:3b"
)

LLM_TIMEOUT = int(
    os.environ.get(
        "MAIZE_LLM_TIMEOUT",
        "1800"
    )
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from configs.training_config import (
    parse_training_config,
    print_config,
    save_config,
)

from Models.model_factory import (
    build_model_from_config,
    get_model_metadata,
    get_parameter_information,
    set_model_training_mode,
)

from src.data.class_names import (
    CLASS_NAMES,
    CLASS_TO_IDX,
    NUM_CLASSES,
)

from src.data.dataset_loader import (
    MaizeLeafDataset,
    load_all_split_records,
)

from src.data.preprocessing import (
    AugmentationConfig,
    build_transforms,
    get_preprocessing_summary,
)

from src.engine.optimizer_factory import (
    build_optimizer,
    get_optimizer_summary,
)

from src.engine.loss_factory import (
    build_loss,
    get_loss_summary,
)

from src.engine.scheduler_factory import (
    build_scheduler,
    get_scheduler_summary,
)

from src.engine.early_stopping import EarlyStopping

from src.engine.checkpoint import CheckpointManager

from src.engine.history_logger import TrainingHistoryLogger

from src.engine.metrics import ClassificationMetrics

from src.evaluation.prediction_exporter import export_predictions

from src.utils.device import (
    select_device,
    get_device_info,
    get_amp_settings,
    create_grad_scaler,
    autocast_context,
)

from src.utils.seed import (
    seed_everything,
    seed_worker,
    create_generator,
)


# ============================================================
# GENERAL UTILITIES
# ============================================================

def _json_default(value):

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.device):
        return str(value)

    if hasattr(value, "item"):

        try:
            return value.item()

        except Exception:
            pass

    return str(value)


def _write_json(data, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path
    )

    return path


def _safe_name(value):

    value = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value)
    )

    return value.strip("._-") or "run"


def _sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            digest.update(chunk)

    return digest.hexdigest()


def _get_split_metadata():

    metadata_path = (
        PROJECT_ROOT
        / "data_splits"
        / "split_metadata.json"
    )

    if not metadata_path.is_file():

        raise FileNotFoundError(
            f"Fixed split metadata not found: {metadata_path}\n"
            "Run the central split generator and verification first."
        )

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if int(metadata.get("num_classes", -1)) != NUM_CLASSES:

        raise RuntimeError(
            "Split metadata class count does not match "
            "the seven-class configuration."
        )

    if metadata.get("class_names") != list(CLASS_NAMES):

        raise RuntimeError(
            "Split metadata class order does not match class_names.py."
        )

    fingerprint = metadata.get(
        "split_assignment_fingerprint_sha256"
    )

    if not fingerprint:

        raise RuntimeError(
            "Split assignment fingerprint is missing."
        )

    return metadata, fingerprint


def _environment_metadata(device, amp_settings):

    info = get_device_info(device)

    return {

        "timestamp_local": (
            datetime.now().astimezone().isoformat()
        ),

        "python_version": sys.version,

        "platform": platform.platform(),

        "processor": platform.processor(),

        "pytorch_version": torch.__version__,

        "numpy_version": np.__version__,

        "cuda_available": torch.cuda.is_available(),

        "cuda_version": torch.version.cuda,

        "device": str(device),

        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),

        "device_information": info,

        "amp": {

            "enabled": bool(
                amp_settings["enabled"]
            ),

            "dtype": amp_settings["dtype"],

            "grad_scaler": bool(
                amp_settings["grad_scaler"]
            ),

            "reason": amp_settings["reason"],

        },

    }


def _parameter_bytes(model):

    return int(
        sum(
            p.numel() * p.element_size()
            for p in model.state_dict().values()
        )
    )


def _progress(iterable, **kwargs):

    if tqdm is not None:
        return tqdm(iterable, **kwargs)

    return iterable


def _make_run_directory(config):

    if config.resume:

        checkpoint_path = (
            Path(config.resume)
            .expanduser()
            .resolve()
        )

        if not checkpoint_path.is_file():

            raise FileNotFoundError(
                f"Resume checkpoint not found: {checkpoint_path}"
            )

        if checkpoint_path.parent.name != "checkpoints":

            raise ValueError(
                "Resume path must point to a checkpoint "
                "inside a run's checkpoints folder."
            )

        return (
            checkpoint_path.parent.parent,
            checkpoint_path
        )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    requested_name = (
        config.run_name
        or f"{config.model}_seed{config.seed}_{timestamp}"
    )

    run_name = _safe_name(requested_name)

    run_dir = (
        Path(config.results_dir)
        / config.model
        / run_name
    )

    if run_dir.exists() and any(run_dir.iterdir()):

        raise FileExistsError(
            f"Run directory already exists: {run_dir}\n"
            "Choose a different --run-name or resume."
        )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return run_dir, None


# ============================================================
# DATA LOADING
# ============================================================

def _build_dataloaders(config, device):

    records = load_all_split_records()

    train_records = records["train"]

    validation_records = records["validation"]

    aug = AugmentationConfig(

        horizontal_flip=(
            config.horizontal_flip
            if config.augmentation else 0.0
        ),

        vertical_flip=(
            config.vertical_flip
            if config.augmentation else 0.0
        ),

        rotation_degrees=(
            config.rotation_degrees
            if config.augmentation else 0.0
        ),

        brightness=(
            config.brightness
            if config.augmentation else 0.0
        ),

        contrast=(
            config.contrast
            if config.augmentation else 0.0
        ),

        saturation=(
            config.saturation
            if config.augmentation else 0.0
        ),

        hue=(
            config.hue
            if config.augmentation else 0.0
        ),

    )

    train_transform, validation_transform = build_transforms(

        image_size=config.image_size,

        augmentation=config.augmentation,

        normalization=config.normalization,

        augmentation_config=aug,

    )

    train_dataset = MaizeLeafDataset(
        train_records,
        transform=train_transform,
    )

    validation_dataset = MaizeLeafDataset(
        validation_records,
        transform=validation_transform,
    )

    counts = Counter(
        record["class_name"]
        for record in train_records
    )

    class_counts = {

        name: int(counts.get(name, 0))

        for name in CLASS_NAMES

    }

    if any(value <= 0 for value in class_counts.values()):

        raise RuntimeError(
            f"Missing training class: {class_counts}"
        )

    generator_train = create_generator(
        config.seed
    )

    generator_validation = create_generator(
        config.seed + 1
    )

    sampler = None

    shuffle = True

    if config.sampler == "weighted":

        sample_weights = [

            1.0 / class_counts[
                record["class_name"]
            ]

            for record in train_records

        ]

        sampler = WeightedRandomSampler(

            weights=torch.as_tensor(
                sample_weights,
                dtype=torch.double,
            ),

            num_samples=len(sample_weights),

            replacement=True,

            generator=create_generator(
                config.seed + 2
            ),

        )

        shuffle = False

    pin_memory = config.pin_memory

    if pin_memory is None:
        pin_memory = device.type == "cuda"

    common_loader_args = {

        "batch_size": config.batch_size,

        "num_workers": config.num_workers,

        "pin_memory": bool(pin_memory),

        "worker_init_fn": seed_worker,

        "persistent_workers": (
            config.num_workers > 0
        ),

    }

    train_loader = DataLoader(

        train_dataset,

        shuffle=shuffle,

        sampler=sampler,

        generator=generator_train,

        drop_last=False,

        **common_loader_args,

    )

    validation_loader = DataLoader(

        validation_dataset,

        shuffle=False,

        generator=generator_validation,

        drop_last=False,

        **common_loader_args,

    )

    split_counts = {

        "train": len(train_dataset),

        "validation": len(validation_dataset),

        "test": len(records["test"]),

    }

    data_metadata = {

        "project_root": str(PROJECT_ROOT),

        "dataset_directory": str(
            PROJECT_ROOT / "Dataset"
        ),

        "class_names": list(CLASS_NAMES),

        "class_to_idx": dict(CLASS_TO_IDX),

        "num_classes": NUM_CLASSES,

        "split_counts": split_counts,

        "training_class_counts": class_counts,

        "split_assignment_fingerprint_sha256": (
            _get_split_metadata()[1]
        ),

        "split_unit": "image",

        "group_ids_available": any(

            bool(r.get("group_id"))

            for r in train_records + validation_records

        ),

        "test_partition_policy": (
            "Held out; not used during training "
            "or validation-based model selection."
        ),

    }

    return (

        train_loader,

        validation_loader,

        class_counts,

        data_metadata,

        train_transform,

        validation_transform,

    )


# ============================================================
# METRICS AND TRAINING HELPERS
# ============================================================

def _make_metrics_tracker():

    return ClassificationMetrics(

        num_classes=NUM_CLASSES,

        class_names=list(CLASS_NAMES),

    )


def _get_monitor_value(
    config,
    validation_loss,
    validation_metrics,
):

    if config.monitor == "val_loss":

        return float(validation_loss)

    if config.monitor == "val_accuracy":

        return float(
            validation_metrics["accuracy"]
        )

    if config.monitor == "val_macro_f1":

        return float(
            validation_metrics["f1_macro"]
        )

    raise ValueError(
        f"Unsupported monitor: {config.monitor}"
    )


def _current_lr(optimizer):

    return float(
        optimizer.param_groups[0]["lr"]
    )


def _gpu_batch_memory(device):

    if device.type != "cuda":
        return None, None

    return (

        round(
            torch.cuda.memory_allocated(device)
            / (1024 ** 2),
            3
        ),

        round(
            torch.cuda.memory_reserved(device)
            / (1024 ** 2),
            3
        ),

    )


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    amp_settings,
    epoch,
    total_epochs,
    split,
    logger,
    optimizer=None,
    scaler=None,
    scheduler_controller=None,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    global_step=0,
):

    is_training = split == "train"

    set_model_training_mode(
        model,
        is_training
    )

    tracker = _make_metrics_tracker()

    running_loss = 0.0

    sample_count = 0

    correct_count = 0

    phase_start = time.perf_counter()

    batch_total = len(loader)

    if is_training:

        optimizer.zero_grad(
            set_to_none=True
        )

    iterator = _progress(

        enumerate(loader, start=1),

        total=batch_total,

        desc=(
            f"{split.upper():<10} "
            f"{epoch:03d}/{total_epochs:03d}"
        ),

        leave=False,

        dynamic_ncols=True,

    )

    for batch_index, batch in iterator:

        batch_start = time.perf_counter()

        images, labels = batch

        images = images.to(

            device,

            non_blocking=(
                device.type == "cuda"
            ),

        )

        labels = labels.to(

            device,

            non_blocking=(
                device.type == "cuda"
            ),

            dtype=torch.long,

        )

        batch_size = int(
            labels.shape[0]
        )

        if is_training:

            group_start = (

                (batch_index - 1)

                // gradient_accumulation_steps

            ) * gradient_accumulation_steps

            group_size = min(

                gradient_accumulation_steps,

                batch_total - group_start,

            )

        with torch.set_grad_enabled(is_training):

            with autocast_context(
                device,
                amp_settings
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels
                )

            if not torch.isfinite(loss):

                raise FloatingPointError(

                    f"Non-finite {split} loss at "

                    f"epoch {epoch}, batch {batch_index}"

                )

            if is_training:

                scaled_loss = (
                    loss / float(group_size)
                )

                scaler.scale(
                    scaled_loss
                ).backward()

                should_update = (

                    batch_index
                    % gradient_accumulation_steps == 0

                    or batch_index == batch_total

                )

                if should_update:

                    if (
                        max_grad_norm
                        and max_grad_norm > 0
                    ):

                        scaler.unscale_(
                            optimizer
                        )

                        torch.nn.utils.clip_grad_norm_(

                            model.parameters(),

                            max_grad_norm,

                        )

                    previous_scale = (
                        scaler.get_scale()
                    )

                    scaler.step(
                        optimizer
                    )

                    scaler.update()

                    current_scale = (
                        scaler.get_scale()
                    )

                    optimizer_updated = (
                        current_scale >= previous_scale
                    )

                    if optimizer_updated:

                        global_step += 1

                        if scheduler_controller is not None:

                            scheduler_controller.step_batch()

                    optimizer.zero_grad(
                        set_to_none=True
                    )

        loss_value = float(
            loss.detach().item()
        )

        running_loss += (
            loss_value * batch_size
        )

        sample_count += batch_size

        predictions = (
            logits.detach().argmax(dim=1)
        )

        correct_count += int(

            (predictions == labels)
            .sum()
            .item()

        )

        tracker.update(

            logits.detach(),

            labels.detach(),

        )

        batch_duration = (
            time.perf_counter() - batch_start
        )

        learning_rate = (

            _current_lr(optimizer)

            if optimizer is not None

            else 0.0

        )

        memory_allocated, memory_reserved = (
            _gpu_batch_memory(device)
        )

        logger.log_batch(

            epoch=epoch,

            batch=batch_index,

            global_step=global_step,

            loss=loss_value,

            accuracy=float(

                (predictions == labels)
                .float()
                .mean()
                .item()

            ),

            learning_rate=learning_rate,

            batch_duration_seconds=batch_duration,

            samples=batch_size,

            split=split,

            gpu_memory_allocated_mb=memory_allocated,

            gpu_memory_reserved_mb=memory_reserved,

        )

        if (
            tqdm is not None
            and hasattr(iterator, "set_postfix")
        ):

            iterator.set_postfix(

                loss=(
                    f"{running_loss / max(sample_count, 1):.4f}"
                ),

                acc=(
                    f"{100.0 * correct_count / max(sample_count, 1):.2f}%"
                ),

                lr=f"{learning_rate:.2e}",

            )

    if device.type == "cuda":

        torch.cuda.synchronize(device)

    duration = (
        time.perf_counter() - phase_start
    )

    mean_loss = (
        running_loss / max(sample_count, 1)
    )

    metrics_result = tracker.compute()

    metrics_result["loss"] = float(mean_loss)

    metrics_result["accuracy"] = float(

        correct_count / max(sample_count, 1)

    )

    return {

        "loss": float(mean_loss),

        "metrics_result": metrics_result,

        "metrics": metrics_result["metrics"],

        "samples": sample_count,

        "duration_seconds": duration,

        "samples_per_second": (

            sample_count / duration
            if duration > 0
            else None

        ),

        "global_step": global_step,

    }


# ============================================================
# TRAINING CURVES
# ============================================================

def _plot_training_curves(
    history_records,
    output_dir
):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    output_dir = (
        Path(output_dir)
        / "training_curves"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if not history_records:
        return []

    epochs = [

        int(item["epoch"])

        for item in history_records

    ]

    train_loss = [

        item.get("train_loss")

        for item in history_records

    ]

    val_loss = [

        item.get("validation_loss")

        for item in history_records

    ]

    train_acc = [

        item.get("train_metrics", {})
        .get("accuracy")

        for item in history_records

    ]

    val_acc = [

        item.get("validation_metrics", {})
        .get("accuracy")

        for item in history_records

    ]

    train_f1 = [

        item.get("train_metrics", {})
        .get("f1_macro")

        for item in history_records

    ]

    val_f1 = [

        item.get("validation_metrics", {})
        .get("f1_macro")

        for item in history_records

    ]

    learning_rates = [

        item.get("learning_rate")

        for item in history_records

    ]

    saved = []

    def save_plot(
        y1,
        y2,
        label1,
        label2,
        ylabel,
        title,
        filename,
    ):

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.plot(
            epochs,
            y1,
            marker="o",
            label=label1
        )

        ax.plot(
            epochs,
            y2,
            marker="o",
            label=label2
        )

        ax.set_xlabel("Epoch")

        ax.set_ylabel(ylabel)

        ax.set_title(title)

        ax.grid(True, alpha=0.3)

        ax.legend()

        fig.tight_layout()

        path = output_dir / filename

        fig.savefig(

            path,

            dpi=200,

            bbox_inches="tight",

        )

        plt.close(fig)

        saved.append(str(path))

    save_plot(

        train_loss,

        val_loss,

        "Training loss",

        "Validation loss",

        "Loss",

        "Training and validation loss",

        "loss_curve.png",

    )

    save_plot(

        train_acc,

        val_acc,

        "Training accuracy",

        "Validation accuracy",

        "Accuracy",

        "Training and validation accuracy",

        "accuracy_curve.png",

    )

    save_plot(

        train_f1,

        val_f1,

        "Training macro-F1",

        "Validation macro-F1",

        "Macro-F1",

        "Training and validation macro-F1",

        "macro_f1_curve.png",

    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.plot(

        epochs,

        learning_rates,

        marker="o",

        label="Learning rate",

    )

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Learning rate")

    ax.set_title("Learning-rate schedule")

    ax.grid(True, alpha=0.3)

    ax.legend()

    fig.tight_layout()

    path = (
        output_dir / "learning_rate_curve.png"
    )

    fig.savefig(

        path,

        dpi=200,

        bbox_inches="tight",

    )

    plt.close(fig)

    saved.append(str(path))

    return saved


# ============================================================
# ARCHITECTURE REPORT
# ============================================================

def _write_architecture_report(
    model,
    model_metadata,
    parameter_info,
    optimizer_summary,
    run_dir,
):

    architecture_path = (
        Path(run_dir)
        / "model_architecture.txt"
    )

    with open(
        architecture_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("=" * 88 + "\n")

        file.write(
            "MAIZE LEAF CLASSIFICATION - MODEL ARCHITECTURE REPORT\n"
        )

        file.write("=" * 88 + "\n\n")

        file.write(
            f"Model: {model_metadata.get('display_name')}\n"
        )

        file.write(
            f"Model key: {model_metadata.get('model_name')}\n"
        )

        file.write(
            f"Number of classes: {NUM_CLASSES}\n"
        )

        file.write(
            f"Pretrained requested: "
            f"{model_metadata.get('pretrained_requested')}\n"
        )

        file.write(
            f"Pretrained actually used: "
            f"{model_metadata.get('pretrained_used')}\n"
        )

        file.write(
            f"Backbone frozen: "
            f"{model_metadata.get('freeze_backbone')}\n\n"
        )

        file.write("PARAMETER SUMMARY\n")

        file.write("-" * 88 + "\n")

        file.write(
            f"Total parameters: "
            f"{parameter_info['total_parameters']:,}\n"
        )

        file.write(
            f"Trainable parameters: "
            f"{parameter_info['trainable_parameters']:,}\n"
        )

        file.write(
            f"Frozen parameters: "
            f"{parameter_info['frozen_parameters']:,}\n"
        )

        file.write(
            f"Trainable percentage: "
            f"{parameter_info['trainable_percentage']:.4f}%\n"
        )

        file.write(
            f"Non-trainable percentage: "
            f"{parameter_info['non_trainable_percentage']:.4f}%\n"
        )

        file.write(
            f"Optimizer parameter count: "
            f"{optimizer_summary['total_optimized_parameters']:,}\n"
        )

        file.write(
            f"Parameter storage estimate: "
            f"{_parameter_bytes(model) / (1024 ** 2):.3f} MB\n\n"
        )

        file.write("FULL PYTORCH ARCHITECTURE\n")

        file.write("-" * 88 + "\n")

        file.write(str(model))

        file.write("\n\nARCHITECTURE CONFIGURATION\n")

        file.write(
            json.dumps(
                model_metadata.get("architecture", {}),
                indent=2,
                ensure_ascii=False,
                default=_json_default,
            )
        )

        file.write("\n")

    _write_json(

        {

            "model_metadata": model_metadata,

            "parameter_information": parameter_info,

            "optimizer_information": optimizer_summary,

            "parameter_and_buffer_storage_estimate_mb": (
                _parameter_bytes(model) / (1024 ** 2)
            ),

        },

        Path(run_dir) / "model_architecture.json",

    )

    return architecture_path


# ============================================================
# RESUME HISTORY
# ============================================================

def _capture_history_for_resume(
    resume_run_dir
):

    history_dir = (
        Path(resume_run_dir)
        / "training_history"
    )

    file_names = [

        "training_history.json",

        "training_history.csv",

        "batch_history.csv",

        "batch_history.jsonl",

        "training_log.txt",

        "run_summary.json",

    ]

    saved_bytes = {}

    for name in file_names:

        path = history_dir / name

        if path.is_file():

            saved_bytes[name] = path.read_bytes()

    old_epochs = []

    old_start_timestamp = None

    old_total_seconds = 0.0

    old_batch_count = 0

    if "training_history.json" in saved_bytes:

        old_json = json.loads(

            saved_bytes["training_history.json"]
            .decode("utf-8")

        )

        old_epochs = old_json.get(
            "epochs",
            []
        )

        old_start_timestamp = old_json.get(
            "start_timestamp"
        )

    if "run_summary.json" in saved_bytes:

        old_summary = json.loads(

            saved_bytes["run_summary.json"]
            .decode("utf-8")

        )

        old_total_seconds = float(

            old_summary.get(
                "total_training_time_seconds",
                0.0
            )

        )

    if "batch_history.jsonl" in saved_bytes:

        old_batch_count = sum(

            1

            for line in (

                saved_bytes["batch_history.jsonl"]
                .decode("utf-8")
                .splitlines()

            )

            if line.strip()

        )

    return {

        "saved_bytes": saved_bytes,

        "old_epochs": old_epochs,

        "old_start_timestamp": old_start_timestamp,

        "old_total_seconds": old_total_seconds,

        "old_batch_count": old_batch_count,

    }


def _restore_logger_for_resume(
    logger,
    snapshot
):

    history_dir = logger.history_dir

    for name, content in snapshot["saved_bytes"].items():

        (history_dir / name).write_bytes(content)

    logger.epoch_records = snapshot["old_epochs"]

    logger.epochs_completed = len(
        snapshot["old_epochs"]
    )

    logger.batch_records_count = (
        snapshot["old_batch_count"]
    )

    if snapshot["old_start_timestamp"]:

        logger.start_timestamp = (
            snapshot["old_start_timestamp"]
        )

    logger.start_monotonic = (

        time.perf_counter()

        - snapshot["old_total_seconds"]

    )

    logger.status = "running"

    return {

        "previous_epochs": len(
            snapshot["old_epochs"]
        ),

        "previous_training_seconds": (
            snapshot["old_total_seconds"]
        ),

        "previous_batch_records": (
            snapshot["old_batch_count"]
        ),

    }


# ============================================================
# AUTOMATED PIPELINE UTILITIES
# ============================================================

def _pipeline_device(device):

    if device.type == "cuda":
        return "cuda:0"

    return "cpu"


def _record_skipped_stage(
    stage_name,
    reason
):

    print(
        f"\n[SKIPPED] {stage_name}: {reason}"
    )

    return {

        "stage": stage_name,

        "status": "skipped",

        "reason": reason,

        "duration_seconds": 0.0,

        "return_code": None,

        "log_file": None,

    }


def _run_pipeline_stage(
    stage_name,
    command,
    run_dir,
):

    log_directory = (
        Path(run_dir)
        / "pipeline_logs"
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    safe_stage_name = _safe_name(
        stage_name
    )

    log_path = (
        log_directory
        / f"{safe_stage_name}.log"
    )

    started_at = (
        datetime.now().astimezone().isoformat()
    )

    start_time = time.perf_counter()

    process = None

    return_code = None

    error_message = None

    print("\n" + "=" * 76)

    print(
        f"STARTING PIPELINE STAGE: {stage_name.upper()}"
    )

    print("=" * 76)

    print(
        "Command: "
        + subprocess.list2cmdline(
            [str(item) for item in command]
        )
    )

    print(
        f"Stage log: {log_path}"
    )

    try:

        with open(
            log_path,
            "w",
            encoding="utf-8",
            errors="replace"
        ) as log_file:

            log_file.write(
                f"Stage: {stage_name}\n"
            )

            log_file.write(
                f"Started: {started_at}\n"
            )

            log_file.write(
                "Command: "
                + subprocess.list2cmdline(
                    [str(item) for item in command]
                )
                + "\n\n"
            )

            log_file.flush()

            # Keep child-process output unbuffered so progress updates
            # are forwarded as soon as they are produced.
            child_environment = os.environ.copy()
            child_environment["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(

                [str(item) for item in command],

                cwd=str(PROJECT_ROOT),

                stdout=subprocess.PIPE,

                stderr=subprocess.STDOUT,

                text=False,

                bufsize=0,

                env=child_environment,

            )

            if process.stdout is not None:

                # Read raw bytes in chunks instead of iterating over text
                # lines. Text mode performs universal-newline conversion,
                # turning tqdm's carriage-return updates (\r) into new
                # lines and breaking the live progress bar.
                decoder = codecs.getincrementaldecoder("utf-8")(
                    errors="replace"
                )

                while True:

                    chunk = os.read(
                        process.stdout.fileno(),
                        4096
                    )

                    if not chunk:
                        break

                    output = decoder.decode(chunk)

                    if output:

                        log_file.write(output)

                        log_file.flush()

                        sys.stdout.write(output)

                        sys.stdout.flush()

                # Flush any remaining bytes held by the decoder.
                final_output = decoder.decode(
                    b"",
                    final=True
                )

                if final_output:

                    log_file.write(final_output)

                    log_file.flush()

                    sys.stdout.write(final_output)

                    sys.stdout.flush()

            return_code = process.wait()

    except KeyboardInterrupt:

        if (
            process is not None
            and process.poll() is None
        ):

            process.terminate()

            try:

                process.wait(timeout=10)

            except subprocess.TimeoutExpired:

                process.kill()

                process.wait()

        raise

    except Exception as error:

        error_message = repr(error)

        try:

            with open(
                log_path,
                "a",
                encoding="utf-8"
            ) as log_file:

                log_file.write(
                    "\nPIPELINE ERROR:\n"
                )

                log_file.write(
                    error_message + "\n"
                )

        except Exception:
            pass

    duration = (
        time.perf_counter() - start_time
    )

    finished_at = (
        datetime.now().astimezone().isoformat()
    )

    success = (

        error_message is None

        and return_code == 0

    )

    status = (
        "completed" if success else "failed"
    )

    print("\n" + "-" * 76)

    if success:

        print(
            f"[PASS] {stage_name} completed successfully."
        )

    else:

        print(
            f"[FAILED] {stage_name}"
        )

        print(
            f"Return code: {return_code}"
        )

        if error_message:

            print(
                f"Error: {error_message}"
            )

    print(
        f"Duration: {duration / 60:.2f} minutes"
    )

    print(
        f"Log: {log_path}"
    )

    print("-" * 76)

    return {

        "stage": stage_name,

        "status": status,

        "started_at": started_at,

        "finished_at": finished_at,

        "duration_seconds": duration,

        "return_code": return_code,

        "error": error_message,

        "command": [
            str(item)
            for item in command
        ],

        "log_file": str(log_path),

    }


# ============================================================
# COMPLETE POST-TRAINING PIPELINE
# ============================================================

def _run_post_training_pipeline(
    config,
    run_dir,
    device,
    amp_settings,
):

    run_dir = Path(run_dir).resolve()

    device_argument = _pipeline_device(
        device
    )

    stages = {}

    pipeline_start = time.perf_counter()

    pipeline_started_at = (
        datetime.now().astimezone().isoformat()
    )

    validation_dir = (
        run_dir
        / "evaluation"
        / "validation"
    )

    predictions_csv = (
        validation_dir
        / "predictions.csv"
    )

    print("\n")

    print("=" * 76)

    print(
        "MAIZE LEAF - AUTOMATED POST-TRAINING PIPELINE"
    )

    print("=" * 76)

    print(
        f"Run directory : {run_dir}"
    )

    print(
        f"Device        : {device_argument}"
    )

    print(
        f"XAI samples   : {XAI_SAMPLES_PER_CLASS} per class"
    )

    print(
        f"LLM model     : {LLM_MODEL}"
    )

    print(
        "SHAP          : EXCLUDED"
    )

    print(
        "Test set      : HELD OUT"
    )

    # --------------------------------------------------------
    # 1. EFFICIENCY BENCHMARKING
    # --------------------------------------------------------

    if config.run_benchmark:

        benchmark_command = [

            sys.executable,

            "-m",

            "src.benchmarking.efficiency_benchmark",

            "--run-dir",

            str(run_dir),

            "--device",

            device_argument,

            "--batch-sizes",

            *BENCHMARK_BATCH_SIZES,

            "--warmup",

            str(BENCHMARK_WARMUP),

            "--iterations",

            str(BENCHMARK_ITERATIONS),

        ]

        if amp_settings["enabled"]:

            benchmark_command.extend([

                "--amp",

                "--amp-dtype",

                str(amp_settings["dtype"]),

            ])

        stages["efficiency_benchmark"] = (
            _run_pipeline_stage(

                "efficiency_benchmark",

                benchmark_command,

                run_dir,

            )
        )

    else:

        stages["efficiency_benchmark"] = (
            _record_skipped_stage(

                "efficiency_benchmark",

                "Disabled in training configuration.",

            )
        )

    # --------------------------------------------------------
    # 2. ERROR ANALYSIS
    # --------------------------------------------------------

    if config.run_error_analysis:

        if not config.run_evaluation:

            stages["error_analysis"] = (
                _record_skipped_stage(

                    "error_analysis",

                    "Validation evaluation is disabled.",

                )
            )

        elif not predictions_csv.is_file():

            stages["error_analysis"] = (
                _record_skipped_stage(

                    "error_analysis",

                    "Validation predictions.csv is unavailable.",

                )
            )

        else:

            error_command = [

                sys.executable,

                "-m",

                "src.evaluation.error_analysis",

                "--evaluation-dir",

                str(validation_dir),

                "--top-k",

                str(ERROR_TOP_K),

                "--confidence-threshold",

                str(ERROR_CONFIDENCE_THRESHOLD),

                "--contact-sheet-limit",

                str(ERROR_CONTACT_SHEET_LIMIT),

            ]

            stages["error_analysis"] = (
                _run_pipeline_stage(

                    "error_analysis",

                    error_command,

                    run_dir,

                )
            )

    else:

        stages["error_analysis"] = (
            _record_skipped_stage(

                "error_analysis",

                "Disabled in training configuration.",

            )
        )

    # --------------------------------------------------------
    # 3. EXPLAINABLE AI
    # --------------------------------------------------------

    if config.run_xai:

        if not config.run_evaluation:

            for stage_name in [

                "gradcam",

                "gradcam_plus_plus",

                "lime",

            ]:

                stages[stage_name] = (
                    _record_skipped_stage(

                        stage_name,

                        "Validation evaluation is disabled.",

                    )
                )

        elif not predictions_csv.is_file():

            for stage_name in [

                "gradcam",

                "gradcam_plus_plus",

                "lime",

            ]:

                stages[stage_name] = (
                    _record_skipped_stage(

                        stage_name,

                        "Validation prediction file is unavailable.",

                    )
                )

        else:

            common_xai_arguments = [

                "--run-dir",

                str(run_dir),

                "--device",

                device_argument,

                "--samples-per-class",

                str(XAI_SAMPLES_PER_CLASS),

                "--sample-strategy",

                XAI_SAMPLE_STRATEGY,

                "--target-class",

                XAI_TARGET_CLASS,

                "--seed",

                str(config.seed),

            ]

            # GRAD-CAM

            gradcam_command = [

                sys.executable,

                "-m",

                "src.explainability.gradcam",

                *common_xai_arguments,

            ]

            stages["gradcam"] = (
                _run_pipeline_stage(

                    "gradcam",

                    gradcam_command,

                    run_dir,

                )
            )

            # GRAD-CAM++

            gradcam_pp_command = [

                sys.executable,

                "-m",

                "src.explainability.gradcam_plus_plus",

                *common_xai_arguments,

            ]

            stages["gradcam_plus_plus"] = (
                _run_pipeline_stage(

                    "gradcam_plus_plus",

                    gradcam_pp_command,

                    run_dir,

                )
            )

            # LIME

            lime_command = [

                sys.executable,

                "-m",

                "src.explainability.lime_explainer",

                *common_xai_arguments,

            ]

            stages["lime"] = (
                _run_pipeline_stage(

                    "lime",

                    lime_command,

                    run_dir,

                )
            )

    else:

        for stage_name in [

            "gradcam",

            "gradcam_plus_plus",

            "lime",

        ]:

            stages[stage_name] = (
                _record_skipped_stage(

                    stage_name,

                    "XAI disabled in training configuration.",

                )
            )

    # --------------------------------------------------------
    # 4. LOCAL LLM RESEARCH REPORTS
    # --------------------------------------------------------

    if config.run_llm:

        llm_script = (
            PROJECT_ROOT
            / "LLM"
            / "report_generator.py"
        )

        if not llm_script.is_file():

            stages["llm_reports"] = (
                _record_skipped_stage(

                    "llm_reports",

                    f"LLM report generator not found: {llm_script}",

                )
            )

        else:

            llm_command = [

                sys.executable,

                str(llm_script),

                "--run-dir",

                str(run_dir),

                "--model",

                LLM_MODEL,

                "--timeout",

                str(LLM_TIMEOUT),

            ]

            stages["llm_reports"] = (
                _run_pipeline_stage(

                    "llm_reports",

                    llm_command,

                    run_dir,

                )
            )

    else:

        stages["llm_reports"] = (
            _record_skipped_stage(

                "llm_reports",

                "LLM reporting disabled in training configuration.",

            )
        )

    # --------------------------------------------------------
    # 5. PIPELINE SUMMARY
    # --------------------------------------------------------

    pipeline_duration = (
        time.perf_counter() - pipeline_start
    )

    pipeline_finished_at = (
        datetime.now().astimezone().isoformat()
    )

    failed_stages = [

        name

        for name, result in stages.items()

        if result["status"] == "failed"

    ]

    completed_stages = [

        name

        for name, result in stages.items()

        if result["status"] == "completed"

    ]

    skipped_stages = [

        name

        for name, result in stages.items()

        if result["status"] == "skipped"

    ]

    pipeline_status = (

        "completed_with_warnings"

        if failed_stages

        else "completed"

    )

    pipeline_summary = {

        "project": "Maize Leaf Classification",

        "pipeline_status": pipeline_status,

        "started_at": pipeline_started_at,

        "finished_at": pipeline_finished_at,

        "duration_seconds": pipeline_duration,

        "duration_minutes": (
            pipeline_duration / 60.0
        ),

        "device": device_argument,

        "llm_model": LLM_MODEL,

        "shap_enabled": False,

        "test_partition_accessed": False,

        "xai_samples_per_class": (
            XAI_SAMPLES_PER_CLASS
        ),

        "completed_stages": completed_stages,

        "failed_stages": failed_stages,

        "skipped_stages": skipped_stages,

        "stage_results": stages,

    }

    pipeline_summary_path = (
        run_dir / "pipeline_summary.json"
    )

    _write_json(

        pipeline_summary,

        pipeline_summary_path,

    )

    # --------------------------------------------------------
    # PRINT FINAL PIPELINE STATUS
    # --------------------------------------------------------

    print("\n")

    print("=" * 76)

    print(
        "POST-TRAINING PIPELINE SUMMARY"
    )

    print("=" * 76)

    for name, result in stages.items():

        if result["status"] == "completed":

            symbol = "[PASS]"

        elif result["status"] == "failed":

            symbol = "[FAIL]"

        else:

            symbol = "[SKIP]"

        print(
            f"{symbol} {name}"
        )

    print("-" * 76)

    print(
        f"Completed stages : {len(completed_stages)}"
    )

    print(
        f"Failed stages    : {len(failed_stages)}"
    )

    print(
        f"Skipped stages   : {len(skipped_stages)}"
    )

    print(
        f"Pipeline time    : {pipeline_duration / 60:.2f} minutes"
    )

    print(
        f"Pipeline status  : {pipeline_status}"
    )

    print(
        f"Summary file     : {pipeline_summary_path}"
    )

    print("=" * 76)

    return pipeline_summary


# ============================================================
# MAIN TRAINING
# ============================================================

def train(config):

    # The complete configuration is saved to training_config.json.
    # Keep the normal terminal output concise.
    seed_information = seed_everything(

        seed=config.seed,

        deterministic=config.deterministic,

        warn_only=True,

    )

    device = select_device(
        config.device
    )

    if isinstance(device, str):

        device = torch.device(device)

    print("\n" + "=" * 76)

    print(
        "MAIZE LEAF CLASSIFICATION - TRAINING ENGINE"
    )

    print("=" * 76)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Model        : {config.model}"
    )

    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(device)}"
        )

        print(
            f"GPU memory   : "
            f"{torch.cuda.get_device_properties(device).total_memory / (1024 ** 3):.2f} GB"
        )

    amp_settings = get_amp_settings(

        device=device,

        enabled=config.amp,

        dtype=config.amp_dtype,

    )

    scaler = create_grad_scaler(
        amp_settings
    )

    run_dir, resume_checkpoint = (
        _make_run_directory(config)
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        f"Run directory: {run_dir}"
    )

    split_metadata, split_fingerprint = (
        _get_split_metadata()
    )

    # --------------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------------

    print("\nBUILDING MODEL")

    model = build_model_from_config(
        config
    )

    model = model.to(device)

    model_metadata = get_model_metadata(
        model
    )

    parameter_info = get_parameter_information(
        model
    )

    # --------------------------------------------------------
    # BUILD DATALOADERS
    # --------------------------------------------------------

    print(
        "\nBUILDING FIXED TRAIN/VALIDATION DATALOADERS"
    )

    (

        train_loader,

        validation_loader,

        class_counts,

        dataset_metadata,

        train_transform,

        validation_transform,

    ) = _build_dataloaders(
        config,
        device
    )

    accumulation_steps = int(
        config.gradient_accumulation_steps
    )

    optimizer_steps_per_epoch = math.ceil(

        len(train_loader)
        / accumulation_steps

    )

    # --------------------------------------------------------
    # LOSS, OPTIMIZER AND SCHEDULER
    # --------------------------------------------------------

    criterion, class_weights = build_loss(

        config=config,

        class_counts=class_counts,

        device=device,

    )

    optimizer = build_optimizer(
        model,
        config
    )

    scheduler_controller = build_scheduler(

        optimizer=optimizer,

        config=config,

        steps_per_epoch=optimizer_steps_per_epoch,

    )

    optimizer_summary = get_optimizer_summary(
        optimizer
    )

    loss_summary = get_loss_summary(

        config=config,

        class_counts=class_counts,

        class_weights=class_weights,

    )

    scheduler_summary = get_scheduler_summary(

        scheduler_controller,

        config,

    )

    if (

        optimizer_summary["total_optimized_parameters"]

        != parameter_info["trainable_parameters"]

    ):

        raise RuntimeError(

            "Optimizer parameter count does not match "
            "trainable model parameter count."

        )

    # --------------------------------------------------------
    # SAVE ARCHITECTURE
    # --------------------------------------------------------

    _write_architecture_report(

        model,

        model_metadata,

        parameter_info,

        optimizer_summary,

        run_dir,

    )

    save_config(

        config,

        run_dir / "training_config.json",

    )

    environment_metadata = _environment_metadata(

        device,

        amp_settings,

    )

    dataset_metadata["split_metadata"] = {

        "split_assignment_fingerprint_sha256": (
            split_fingerprint
        ),

        "split_unit": split_metadata.get(
            "split_unit"
        ),

        "grouping_method": split_metadata.get(
            "grouping_method"
        ),

        "random_seed": split_metadata.get(
            "random_seed"
        ),

        "split_ratios": split_metadata.get(
            "split_ratios"
        ),

        "effective_dataset_fingerprint_sha256": (
            split_metadata.get(
                "effective_dataset_fingerprint_sha256"
            )
        ),

    }

    _write_json(

        split_metadata,

        run_dir / "split_metadata_snapshot.json",

    )

    preprocessing_summary = get_preprocessing_summary(

        image_size=config.image_size,

        augmentation=config.augmentation,

        normalization=config.normalization,

        augmentation_config=AugmentationConfig(

            horizontal_flip=(
                config.horizontal_flip
                if config.augmentation else 0.0
            ),

            vertical_flip=(
                config.vertical_flip
                if config.augmentation else 0.0
            ),

            rotation_degrees=(
                config.rotation_degrees
                if config.augmentation else 0.0
            ),

            brightness=(
                config.brightness
                if config.augmentation else 0.0
            ),

            contrast=(
                config.contrast
                if config.augmentation else 0.0
            ),

            saturation=(
                config.saturation
                if config.augmentation else 0.0
            ),

            hue=(
                config.hue
                if config.augmentation else 0.0
            ),

        ),

    )

    experiment_metadata = {

        "config": config.to_dict(),

        "model": model_metadata,

        "parameters": parameter_info,

        "optimizer": optimizer_summary,

        "loss": loss_summary,

        "scheduler": scheduler_summary,

        "preprocessing": preprocessing_summary,

        "dataset": dataset_metadata,

        "environment": environment_metadata,

        "seed_information": seed_information,

    }

    _write_json(

        experiment_metadata,

        run_dir / "experiment_metadata.json",

    )

    (run_dir / "train_transform.txt").write_text(

        str(train_transform),

        encoding="utf-8",

    )

    (run_dir / "validation_transform.txt").write_text(

        str(validation_transform),

        encoding="utf-8",

    )

    # --------------------------------------------------------
    # HISTORY LOGGER
    # --------------------------------------------------------

    resume_history_snapshot = (

        _capture_history_for_resume(run_dir)

        if resume_checkpoint is not None

        else None

    )

    logger = TrainingHistoryLogger(

        output_dir=run_dir,

        run_name=run_dir.name,

        save_batch_history=True,

    )

    checkpoint_manager = CheckpointManager(

        output_dir=run_dir,

        save_last=config.save_last_checkpoint,

    )

    early_stopping = EarlyStopping(

        patience=config.patience,

        min_delta=config.min_delta,

        monitor=config.monitor,

        mode=config.monitor_mode,

        enabled=config.early_stopping,

        verbose=True,

    )

    compatibility = {

        "model_name": config.model,

        "num_classes": NUM_CLASSES,

        "class_names": list(CLASS_NAMES),

        "image_size": int(config.image_size),

        "normalization": config.normalization,

        "split_assignment_fingerprint_sha256": (
            split_fingerprint
        ),

    }

    start_epoch = 1

    global_step = 0

    resume_information = None

    if resume_checkpoint is not None:

        print(
            f"\nRESUMING FROM: {resume_checkpoint}"
        )

        logger_resume = _restore_logger_for_resume(

            logger,

            resume_history_snapshot,

        )

        resume_information = checkpoint_manager.load(

            model=model,

            checkpoint_path=resume_checkpoint,

            optimizer=optimizer,

            scheduler=scheduler_controller,

            scaler=scaler,

            early_stopping=early_stopping,

            restore_rng=True,

            expected_compatibility=compatibility,

            strict_compatibility=True,

        )

        start_epoch = int(
            resume_information["next_epoch"]
        )

        global_step = int(
            resume_information["global_step"]
        )

        logger.best_epoch = (
            early_stopping.best_epoch
        )

        logger.best_metric = (

            {

                "name": config.monitor,

                "value": early_stopping.best_value,

            }

            if early_stopping.best_value is not None

            else None

        )

        print(
            f"Previous history restored: {logger_resume}"
        )

        print(
            f"Next epoch: {start_epoch}"
        )

    logger.save_metadata(

        config=config.to_dict(),

        model_metadata=model_metadata,

        dataset_metadata=dataset_metadata,

        environment_metadata=environment_metadata,

    )

    # --------------------------------------------------------
    # TRAINING CONFIGURATION
    # --------------------------------------------------------

    print("\nTRAINING CONFIGURATION")

    print("-" * 76)

    print(
        f"Train images              : "
        f"{len(train_loader.dataset):,}"
    )

    print(
        f"Validation images         : "
        f"{len(validation_loader.dataset):,}"
    )

    print(
        "Test images               : HELD OUT - NOT USED"
    )

    print(
        f"Classes                   : {NUM_CLASSES}"
    )

    print(
        f"Batch size                : {config.batch_size}"
    )

    print(
        f"Training batches/epoch    : {len(train_loader)}"
    )

    print(
        f"Optimizer updates/epoch   : {optimizer_steps_per_epoch}"
    )

    print(
        f"Gradient accumulation     : {accumulation_steps}"
    )

    print(
        f"AMP                       : "
        f"{amp_settings['enabled']} ({amp_settings['dtype']})"
    )

    print(
        f"Initial learning rate     : "
        f"{config.learning_rate:.3e}"
    )

    print(
        f"Early stopping            : {config.early_stopping}"
    )

    print(
        f"Monitor                   : {config.monitor}"
    )

    print(
        f"Split fingerprint         : {split_fingerprint}"
    )

    print("-" * 76)

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            device
        )

    run_status = "completed"

    try:

        # ====================================================
        # TRAINING LOOP
        # ====================================================

        for epoch in range(

            start_epoch,

            config.epochs + 1,

        ):

            if device.type == "cuda":

                torch.cuda.reset_peak_memory_stats(
                    device
                )

            epoch_start_wall = (
                datetime.now().astimezone().isoformat()
            )

            epoch_start = time.perf_counter()

            learning_rate_start = _current_lr(
                optimizer
            )

            train_result = _run_epoch(

                model=model,

                loader=train_loader,

                criterion=criterion,

                device=device,

                amp_settings=amp_settings,

                epoch=epoch,

                total_epochs=config.epochs,

                split="train",

                logger=logger,

                optimizer=optimizer,

                scaler=scaler,

                scheduler_controller=scheduler_controller,

                gradient_accumulation_steps=accumulation_steps,

                max_grad_norm=config.max_grad_norm,

                global_step=global_step,

            )

            global_step = (
                train_result["global_step"]
            )

            validation_result = _run_epoch(

                model=model,

                loader=validation_loader,

                criterion=criterion,

                device=device,

                amp_settings=amp_settings,

                epoch=epoch,

                total_epochs=config.epochs,

                split="validation",

                logger=logger,

                optimizer=None,

                scaler=None,

                scheduler_controller=None,

                gradient_accumulation_steps=1,

                max_grad_norm=0.0,

                global_step=global_step,

            )

            monitored_value = _get_monitor_value(

                config,

                validation_result["loss"],

                validation_result["metrics"],

            )

            should_stop = early_stopping.step(

                monitored_value,

                epoch=epoch,

            )

            is_best = bool(
                early_stopping.improved
            )

            if is_best:

                logger.update_best(

                    epoch=epoch,

                    metric_name=config.monitor,

                    metric_value=monitored_value,

                )

            scheduler_controller.step_epoch(

                metric=monitored_value

            )

            epoch_duration = (

                time.perf_counter()

                - epoch_start

            )

            epoch_end_wall = (
                datetime.now().astimezone().isoformat()
            )

            next_learning_rate = _current_lr(
                optimizer
            )

            gpu_memory = logger.get_gpu_memory()

            logger.log_epoch(

                epoch=epoch,

                train_metrics=train_result["metrics_result"],

                validation_metrics=validation_result["metrics_result"],

                learning_rate=learning_rate_start,

                train_loss=train_result["loss"],

                validation_loss=validation_result["loss"],

                epoch_duration_seconds=epoch_duration,

                train_duration_seconds=train_result["duration_seconds"],

                validation_duration_seconds=validation_result["duration_seconds"],

                train_samples=train_result["samples"],

                validation_samples=validation_result["samples"],

                samples_per_second=train_result["samples_per_second"],

                gpu_memory=gpu_memory,

                epoch_start_timestamp=epoch_start_wall,

                epoch_end_timestamp=epoch_end_wall,

                extra={

                    "monitor": config.monitor,

                    "monitor_value": monitored_value,

                    "is_best": is_best,

                    "early_stopping_bad_epochs": (
                        early_stopping.bad_epochs
                    ),

                    "learning_rate_next_epoch": (
                        next_learning_rate
                    ),

                    "global_step": global_step,

                    "train_macro_f1": (
                        train_result["metrics"]["f1_macro"]
                    ),

                    "validation_macro_f1": (
                        validation_result["metrics"]["f1_macro"]
                    ),

                    "train_balanced_accuracy": (
                        train_result["metrics"]["balanced_accuracy"]
                    ),

                    "validation_balanced_accuracy": (
                        validation_result["metrics"]["balanced_accuracy"]
                    ),

                },

            )

            checkpoint_manager.save(

                model=model,

                epoch=epoch,

                global_step=global_step,

                optimizer=optimizer,

                scheduler=scheduler_controller,

                scaler=scaler,

                early_stopping=early_stopping,

                model_name=config.model,

                metric_name=config.monitor,

                metric_value=monitored_value,

                is_best=is_best,

                config=config.to_dict(),

                compatibility=compatibility,

                dataset_metadata=dataset_metadata,

                environment_metadata=environment_metadata,

                extra={

                    "run_directory": str(run_dir),

                    "completed_epochs": logger.epochs_completed,

                },

            )

            status_text = (
                "NEW BEST" if is_best else ""
            )

            print(

                f"\nEPOCH {epoch:03d}/{config.epochs:03d} | "

                f"Train loss {train_result['loss']:.5f} | "

                f"Train acc {train_result['metrics']['accuracy']:.4f} | "

                f"Val loss {validation_result['loss']:.5f} | "

                f"Val acc {validation_result['metrics']['accuracy']:.4f} | "

                f"Val macro-F1 {validation_result['metrics']['f1_macro']:.4f} | "

                f"LR {learning_rate_start:.3e}->{next_learning_rate:.3e} | "

                f"Time {epoch_duration / 60:.2f} min | "

                f"{status_text}"

            )

            if should_stop:

                run_status = "early_stopped"

                print(
                    "\nEARLY STOPPING ACTIVATED."
                )

                break

        # ====================================================
        # RESTORE BEST CHECKPOINT
        # ====================================================

        if checkpoint_manager.best_checkpoint_path.is_file():

            best_info = checkpoint_manager.load_model_weights(

                model,

                checkpoint_path=(
                    checkpoint_manager.best_checkpoint_path
                ),

                strict=True,

            )

            print(

                f"\nBEST CHECKPOINT RESTORED | "

                f"Epoch {best_info['best_epoch']} | "

                f"{config.monitor}={best_info['best_metric']}"

            )

        else:

            warnings.warn(

                "No best checkpoint was created. "
                "Final in-memory model will be used."

            )

        # ====================================================
        # FINAL VALIDATION
        # ====================================================

        prediction_export_metadata = None

        if config.run_evaluation:

            print(
                "\nGENERATING VALIDATION METRICS"
            )

            model.eval()

            validation_tracker = _make_metrics_tracker()

            validation_loss_sum = 0.0

            validation_samples = 0

            with torch.no_grad():

                iterator = _progress(

                    validation_loader,

                    desc="FINAL VALIDATION",

                    leave=True,

                    dynamic_ncols=True,

                )

                for images, labels in iterator:

                    images = images.to(

                        device,

                        non_blocking=(
                            device.type == "cuda"
                        ),

                    )

                    labels = labels.to(

                        device,

                        non_blocking=(
                            device.type == "cuda"
                        ),

                        dtype=torch.long,

                    )

                    with autocast_context(
                        device,
                        amp_settings
                    ):

                        logits = model(images)

                        loss = criterion(
                            logits,
                            labels
                        )

                    validation_loss_sum += (

                        float(loss.item())

                        * int(labels.size(0))

                    )

                    validation_samples += int(
                        labels.size(0)
                    )

                    validation_tracker.update(

                        logits,

                        labels,

                    )

            final_validation = (
                validation_tracker.compute()
            )

            final_validation["loss"] = (

                validation_loss_sum

                / max(validation_samples, 1)

            )

            validation_output = (

                run_dir

                / "evaluation"

                / "validation"

            )

            validation_artifacts = (
                validation_tracker.save(

                    validation_output,

                    split_name="validation",

                )
            )

            _write_json(

                final_validation,

                validation_output
                / "final_validation_metrics.json",

            )

            print(
                f"Validation metrics saved: {validation_output}"
            )

            # -----------------------------------------------
            # PER-IMAGE VALIDATION PREDICTION EXPORT
            # -----------------------------------------------

            print(
                "\nEXPORTING PER-IMAGE VALIDATION PREDICTIONS"
            )

            prediction_export_metadata = export_predictions(

                model=model,

                dataset=validation_loader.dataset,

                device=device,

                amp_settings=amp_settings,

                output_csv=(
                    validation_output / "predictions.csv"
                ),

                batch_size=config.batch_size,

                num_workers=0,

                pin_memory=config.pin_memory,

                model_name=config.model,

                run_name=run_dir.name,

                split_name="validation",

                split_fingerprint=split_fingerprint,

                dataset_metadata=dataset_metadata,

                training_metadata={

                    "best_epoch": early_stopping.best_epoch,

                    "best_monitor": config.monitor,

                    "best_monitor_value": (
                        early_stopping.best_value
                    ),

                    "training_config_path": str(

                        run_dir / "training_config.json"

                    ),

                },

            )

        else:

            final_validation = None

            validation_artifacts = {}

            prediction_export_metadata = None

        # ====================================================
        # TRAINING CURVES
        # ====================================================

        curve_paths = _plot_training_curves(

            logger.epoch_records,

            run_dir,

        )

        # ====================================================
        # CHECKPOINT INFORMATION
        # ====================================================

        best_checkpoint = (
            checkpoint_manager.best_checkpoint_path
        )

        checkpoint_size_mb = (

            best_checkpoint.stat().st_size
            / (1024 ** 2)

            if best_checkpoint.is_file()

            else None

        )

        # ====================================================
        # TRAINING TIME
        # ====================================================

        total_epoch_training_seconds = sum(

            float(
                record.get(
                    "epoch_duration_seconds",
                    0.0
                )
            )

            for record in logger.epoch_records

        )

        epochs_completed_this_run = sum(

            1

            for record in logger.epoch_records

            if int(record.get("epoch", 0)) >= start_epoch

        )

        # ====================================================
        # PRELIMINARY RUN SUMMARY
        # Saved before benchmarking so the benchmark can
        # retrieve training information.
        # ====================================================

        run_summary = {

            "model": config.model,

            "run_name": run_dir.name,

            "status": run_status,

            "best_epoch": early_stopping.best_epoch,

            "best_monitor": config.monitor,

            "best_monitor_value": early_stopping.best_value,

            "epochs_completed_this_run": (
                epochs_completed_this_run
            ),

            "total_training_time_seconds": (
                total_epoch_training_seconds
            ),

            "total_training_time_minutes": (
                total_epoch_training_seconds / 60.0
            ),

            "global_optimizer_steps": global_step,

            "best_checkpoint": (

                str(best_checkpoint)

                if best_checkpoint.is_file()

                else None

            ),

            "best_checkpoint_size_mb": (
                checkpoint_size_mb
            ),

            "total_parameters": (
                parameter_info["total_parameters"]
            ),

            "trainable_parameters": (
                parameter_info["trainable_parameters"]
            ),

            "frozen_parameters": (
                parameter_info["frozen_parameters"]
            ),

            "optimizer_parameters": (
                optimizer_summary[
                    "total_optimized_parameters"
                ]
            ),

            "split_assignment_fingerprint_sha256": (
                split_fingerprint
            ),

            "validation_metrics": (

                final_validation["metrics"]

                if final_validation

                else None

            ),

            "validation_loss": (

                final_validation["loss"]

                if final_validation

                else None

            ),

            "training_curve_files": curve_paths,

            "validation_artifacts": (
                validation_artifacts
            ),

            "prediction_export": (
                prediction_export_metadata
            ),

            "test_partition_accessed": False,

            "post_training_pipeline": None,

        }

        _write_json(

            run_summary,

            run_dir / "run_summary.json",

        )

        # ====================================================
        # AUTOMATED POST-TRAINING PIPELINE
        # ====================================================

        pipeline_summary = (
            _run_post_training_pipeline(

                config=config,

                run_dir=run_dir,

                device=device,

                amp_settings=amp_settings,

            )
        )

        if pipeline_summary["failed_stages"]:

            run_status = "completed_with_warnings"

        run_summary["status"] = run_status

        run_summary["post_training_pipeline"] = (
            pipeline_summary
        )

        run_summary["pipeline_status"] = (
            pipeline_summary["pipeline_status"]
        )

        run_summary["pipeline_summary_path"] = str(

            run_dir / "pipeline_summary.json"

        )

        run_summary["post_training_pipeline_time_seconds"] = (
            pipeline_summary["duration_seconds"]
        )

        run_summary["post_training_pipeline_time_minutes"] = (
            pipeline_summary["duration_minutes"]
        )

        # Total execution time includes training, evaluation,
        # prediction export and post-training pipeline stages.

        run_summary["total_run_time_seconds"] = (
            logger.elapsed_seconds()
        )

        run_summary["total_run_time_minutes"] = (

            run_summary["total_run_time_seconds"] / 60.0

        )

        _write_json(

            run_summary,

            run_dir / "run_summary.json",

        )

        # ====================================================
        # FINALIZE TRAINING HISTORY
        # ====================================================

        logger.finalize(

            status=run_status,

            best_epoch=early_stopping.best_epoch,

            best_metric=(

                {

                    "name": config.monitor,

                    "value": early_stopping.best_value,

                }

                if early_stopping.best_value is not None

                else None

            ),

            additional_information=run_summary,

        )

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        print("\n" + "=" * 76)

        print(
            "MAIZE LEAF RESEARCH PIPELINE COMPLETED"
        )

        print("=" * 76)

        print(
            f"Model                 : {config.model}"
        )

        print(
            f"Run status            : {run_status}"
        )

        print(
            f"Best epoch            : {early_stopping.best_epoch}"
        )

        print(

            f"Best {config.monitor:<16}: "
            f"{early_stopping.best_value}"

        )

        if checkpoint_size_mb is not None:

            print(

                f"Checkpoint size       : "
                f"{checkpoint_size_mb:.2f} MB"

            )

        else:

            print(
                "Checkpoint size       : N/A"
            )

        print(
            f"Run directory         : {run_dir}"
        )

        print(
            f"Training history      : "
            f"{run_dir / 'training_history'}"
        )

        print(
            f"Validation evaluation : "
            f"{run_dir / 'evaluation' / 'validation'}"
        )

        print(
            f"Training curves       : "
            f"{run_dir / 'training_curves'}"
        )

        print(
            f"Benchmarking          : "
            f"{run_dir / 'benchmarking'}"
        )

        print(
            f"Error analysis        : "
            f"{run_dir / 'evaluation' / 'validation' / 'error_analysis'}"
        )

        print(
            f"XAI results           : "
            f"{run_dir / 'Xai'}"
        )

        print(
            f"LLM reports           : "
            f"{run_dir / 'LLM'}"
        )

        print(
            f"Pipeline summary      : "
            f"{run_dir / 'pipeline_summary.json'}"
        )

        print(
            f"Total training time   : "
            f"{total_epoch_training_seconds / 60:.2f} minutes"
        )

        print(
            f"Total execution time  : "
            f"{run_summary['total_run_time_minutes']:.2f} minutes"
        )

        print(
            f"Failed pipeline stages: "
            f"{len(pipeline_summary['failed_stages'])}"
        )

        print(
            "Test partition        : NOT ACCESSED"
        )

        print(
            "SHAP                  : EXCLUDED"
        )

        print("=" * 76)

        return run_summary

    # ========================================================
    # INTERRUPTED TRAINING
    # ========================================================

    except KeyboardInterrupt:

        run_status = "interrupted"

        logger.finalize(

            status="interrupted",

            best_epoch=early_stopping.best_epoch,

            best_metric=(

                {

                    "name": config.monitor,

                    "value": early_stopping.best_value,

                }

                if early_stopping.best_value is not None

                else None

            ),

            additional_information={

                "message": (
                    "Training or post-training pipeline "
                    "interrupted by user."
                ),

                "last_completed_epoch": (
                    logger.epochs_completed
                ),

            },

        )

        print(

            "\nExecution interrupted. "
            "Completed history and checkpoints were preserved."

        )

        raise

    # ========================================================
    # FAILED EXECUTION
    # ========================================================

    except Exception as error:

        run_status = "failed"

        final_error = repr(error)

        try:

            logger.finalize(

                status="failed",

                best_epoch=early_stopping.best_epoch,

                best_metric=(

                    {

                        "name": config.monitor,

                        "value": early_stopping.best_value,

                    }

                    if early_stopping.best_value is not None

                    else None

                ),

                additional_information={

                    "error": final_error

                },

            )

        except Exception:
            pass

        _write_json(

            {

                "status": "failed",

                "error": final_error,

                "timestamp_local": (
                    datetime.now().astimezone().isoformat()
                ),

                "last_completed_epoch": (
                    logger.epochs_completed
                ),

            },

            run_dir / "failure_report.json",

        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

def main(argv=None):

    raw_arguments = (

        list(sys.argv[1:])

        if argv is None

        else list(argv)

    )

    validate_only = (
        "--validate-only" in raw_arguments
    )

    config = parse_training_config(
        raw_arguments
    )

    if validate_only:

        print_config(config)

        print(
            "\nTRAINING CONFIGURATION VALIDATION: PASSED"
        )

        print(
            "No training was started."
        )

        return

    if config.model == "mobnetfine":
        from src.engine.mobnetfine_trainer import train_progressive
        train_progressive(config)
    else:
        train(config)


if __name__ == "__main__":

    main()
