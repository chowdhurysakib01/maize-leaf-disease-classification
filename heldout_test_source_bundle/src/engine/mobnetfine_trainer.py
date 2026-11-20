"""
Progressive training engine for MobNetFine (MobileNetV3-Large + MSAF).

Training stages:
  1. classifier_only: train all newly-added MSAF/head layers.
  2. progressive: unfreeze MobileNetV3-Large features[13:] plus the MSAF head.
  3. full: unfreeze the complete MobileNetV3-Large feature extractor plus the head.

The epoch budget supplied through --epochs is split 15% / 25% / 60%.
For example, 20 epochs -> 3 + 5 + 12; 50 epochs -> 8 + 12 + 30.

The balanced objective is inverse-frequency weighted cross-entropy, with
weights calculated only from the training partition. The held-out test images
are not used for training, model selection, or post-training evaluation.
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from Models.MobNetFine import (
    get_stage_parameter_groups,
    set_trainable_stage,
)
from Models.model_factory import (
    build_model_from_config,
    get_model_metadata,
    get_parameter_information,
)
from configs.training_config import save_config
from src.data.class_names import CLASS_NAMES, NUM_CLASSES
from src.data.preprocessing import (
    AugmentationConfig,
    get_preprocessing_summary,
)
from src.engine.checkpoint import CheckpointManager
from src.engine.early_stopping import EarlyStopping
from src.engine.history_logger import TrainingHistoryLogger
from src.engine.loss_factory import build_loss, get_loss_summary
from src.engine.optimizer_factory import build_optimizer, get_optimizer_summary
from src.engine.scheduler_factory import build_scheduler, get_scheduler_summary
from src.evaluation.prediction_exporter import export_predictions
from src.utils.device import (
    autocast_context,
    create_grad_scaler,
    get_amp_settings,
    select_device,
)
from src.utils.seed import seed_everything

from src.engine import trainer as base_trainer


def _stage_epochs(total_epochs: int):
    """Return a 15% / 25% / 60% progressive epoch allocation."""
    total_epochs = int(total_epochs)
    if total_epochs < 3:
        raise ValueError("MobNetFine requires at least 3 total epochs.")

    stage1 = max(1, int(round(total_epochs * 0.15)))
    stage2 = max(1, int(round(total_epochs * 0.25)))
    stage3 = total_epochs - stage1 - stage2

    if stage3 < 1:
        stage3 = 1
        if stage2 > 1:
            stage2 -= 1
        else:
            stage1 -= 1

    return {
        "classifier_only": stage1,
        "progressive": stage2,
        "full": stage3,
    }


def _stage_rates(base_lr: float):
    """Differential learning rates for new head and pretrained backbone."""
    base_lr = float(base_lr)
    return {
        "classifier_only": {"head": base_lr},
        "progressive": {
            "backbone": base_lr * 0.05,
            "head": base_lr * 0.30,
        },
        "full": {
            "backbone": base_lr * 0.01,
            "head": base_lr * 0.10,
        },
    }


def _stage_config(config, epochs: int, rates):
    max_lr = max(float(value) for value in rates.values())
    min_lr = max(1e-9, min(float(value) for value in rates.values()) * 0.01)

    return SimpleNamespace(
        optimizer=config.optimizer,
        learning_rate=max_lr,
        weight_decay=float(config.weight_decay),
        optimizer_eps=float(config.optimizer_eps),
        adam_beta1=float(config.adam_beta1),
        adam_beta2=float(config.adam_beta2),
        momentum=float(config.momentum),
        nesterov=bool(config.nesterov),
        scheduler="cosine",
        epochs=int(epochs),
        min_lr=min_lr,
        warmup_epochs=0,
        monitor=config.monitor,
        monitor_mode=config.monitor_mode,
        scheduler_step_size=int(config.scheduler_step_size),
        scheduler_gamma=float(config.scheduler_gamma),
        scheduler_factor=float(config.scheduler_factor),
        scheduler_patience=int(config.scheduler_patience),
    )


def _make_training_strategy(config):
    epochs = _stage_epochs(config.epochs)
    rates = _stage_rates(config.learning_rate)
    return {
        "strategy": "progressive",
        "starting_weights": "ImageNet MobileNet_V3_Large_Weights.DEFAULT",
        "architecture": "MSAF",
        "objective": "inverse_frequency_weighted_cross_entropy",
        "class_weight_source": "train split only",
        "stage_epochs": epochs,
        "stage_learning_rates": rates,
        "backbone_unfreeze_plan": {
            "classifier_only": "All MobileNetV3 feature layers frozen; MSAF/head trainable",
            "progressive": "MobileNetV3 features[13:] and MSAF/head trainable",
            "full": "All MobileNetV3 feature layers and MSAF/head trainable",
        },
        "batchnorm_policy": "MobileNetV3 backbone BatchNorm running statistics remain frozen",
        "early_stopping": {
            "enabled": bool(config.early_stopping),
            "patience": int(config.patience),
            "monitor": config.monitor,
            "monitor_mode": config.monitor_mode,
            "active_stage": "full",
        },
    }


def _run_final_validation(
    model,
    validation_loader,
    criterion,
    device,
    amp_settings,
    run_dir,
    config,
    split_fingerprint,
    dataset_metadata,
    best_epoch,
    best_metric,
):
    model.eval()
    tracker = base_trainer._make_metrics_tracker()
    loss_sum = 0.0
    sample_count = 0

    with torch.no_grad():
        iterator = base_trainer._progress(
            validation_loader,
            desc="FINAL VALIDATION",
            leave=True,
            dynamic_ncols=True,
        )
        for images, labels in iterator:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            labels = labels.to(
                device,
                non_blocking=device.type == "cuda",
                dtype=torch.long,
            )

            with autocast_context(device, amp_settings):
                logits = model(images)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss during final validation.")

            loss_sum += float(loss.item()) * int(labels.size(0))
            sample_count += int(labels.size(0))
            tracker.update(logits, labels)

    final_metrics = tracker.compute()
    final_metrics["loss"] = loss_sum / max(sample_count, 1)

    validation_dir = run_dir / "evaluation" / "validation"
    validation_artifacts = tracker.save(
        validation_dir,
        split_name="validation",
    )
    base_trainer._write_json(
        final_metrics,
        validation_dir / "final_validation_metrics.json",
    )

    prediction_metadata = export_predictions(
        model=model,
        dataset=validation_loader.dataset,
        device=device,
        amp_settings=amp_settings,
        output_csv=validation_dir / "predictions.csv",
        batch_size=config.batch_size,
        # Keep prediction export single-process on Windows. This avoids
        # re-importing the full CUDA/PyTorch stack in spawned workers.
        num_workers=0,
        pin_memory=config.pin_memory,
        model_name=config.model,
        run_name=run_dir.name,
        split_name="validation",
        split_fingerprint=split_fingerprint,
        dataset_metadata=dataset_metadata,
        training_metadata={
            "best_epoch": best_epoch,
            "best_monitor": config.monitor,
            "best_monitor_value": best_metric,
            "training_config_path": str(run_dir / "training_config.json"),
            "training_strategy": "progressive MSAF fine-tuning",
        },
    )

    return final_metrics, validation_artifacts, prediction_metadata


def train_progressive(config):
    """Train MobNetFine and run the existing complete post-training pipeline."""
    if config.model != "mobnetfine":
        raise ValueError(
            "train_progressive only supports --model mobnetfine."
        )
    if not config.pretrained:
        raise ValueError(
            "MobNetFine is designed to start from ImageNet weights. "
            "Run with --pretrained."
        )
    if config.scheduler != "cosine":
        raise ValueError(
            "MobNetFine progressive fine-tuning uses the cosine scheduler. "
            "Run with --scheduler cosine."
        )
    if config.resume is not None:
        raise ValueError(
            "Progressive MobNetFine resume is not enabled in this version. "
            "Start a fresh run with a new --run-name."
        )

    # Enforce the approved balanced objective. Do not combine weighted loss
    # with weighted sampling, because that would apply imbalance correction twice.
    config.loss = "cross_entropy"
    config.class_weighting = "inverse_frequency"
    config.sampler = "none"

    seed_information = seed_everything(
        seed=config.seed,
        deterministic=config.deterministic,
        warn_only=True,
    )
    device = select_device(config.device)
    if not isinstance(device, torch.device):
        device = torch.device(device)

    amp_settings = get_amp_settings(
        device=device,
        enabled=config.amp,
        dtype=config.amp_dtype,
    )
    scaler = create_grad_scaler(amp_settings)

    run_dir, resume_checkpoint = base_trainer._make_run_directory(config)
    if resume_checkpoint is not None:
        raise RuntimeError("Unexpected resume checkpoint for MobNetFine.")

    print("\n" + "=" * 76)
    print("MOBNETFINE - PROGRESSIVE MSAF TRAINING ENGINE")
    print("=" * 76)
    print(f"Project root : {base_trainer.PROJECT_ROOT}")
    print(f"Model        : {config.model}")
    print(f"Device       : {device}")
    if device.type == "cuda":
        print(f"GPU          : {torch.cuda.get_device_name(device)}")
        print(
            "GPU memory   : "
            f"{torch.cuda.get_device_properties(device).total_memory / (1024 ** 3):.2f} GB"
        )
    print(f"Run directory: {run_dir}")
    print("Starting weights: ImageNet MobileNetV3-Large")
    print("Objective       : Inverse-frequency weighted cross-entropy")
    print("Test partition  : HELD OUT")

    logger = None
    run_status = "completed"
    stage_stopper = None

    try:
        split_metadata, split_fingerprint = base_trainer._get_split_metadata()

        print("\nBUILDING MOBNETFINE MSAF MODEL")
        model = build_model_from_config(config)

        # Move the model to the selected device before any CUDA batch is passed
        # through it. Keep master parameters in FP32: AMP autocast will select
        # FP16 for eligible operations while preserving FP32 model weights.
        # This mirrors the device placement performed by the standard trainer.
        model = model.to(device=device, dtype=torch.float32)

        parameter_devices = {parameter.device for parameter in model.parameters()}
        if parameter_devices != {device}:
            raise RuntimeError(
                "MobNetFine model parameters were not placed on the selected "
                f"device. Expected {device}; found {sorted(map(str, parameter_devices))}."
            )

        non_fp32_parameters = [
            name
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        ]
        if non_fp32_parameters:
            raise RuntimeError(
                "MobNetFine expects FP32 master parameters when AMP is enabled. "
                f"Unexpected parameter dtypes in: {non_fp32_parameters[:5]}"
            )

        print(f"[PASS] Model moved to {device}; FP32 master weights verified")

        # Record full architecture size before stage-wise freezing.
        full_parameter_stage = set_trainable_stage(model, "full")
        model_metadata = get_model_metadata(model)
        parameter_info = get_parameter_information(model)

        print("\nBUILDING FIXED TRAIN/VALIDATION DATALOADERS")
        (
            train_loader,
            validation_loader,
            class_counts,
            dataset_metadata,
            train_transform,
            validation_transform,
        ) = base_trainer._build_dataloaders(config, device)

        if dataset_metadata.get("split_assignment_fingerprint_sha256") != split_fingerprint:
            raise RuntimeError("Dataloader split fingerprint differs from split metadata.")

        # Confirm the separate train/validation CSV files match the class
        # counts recorded in the central split metadata before computing weights.
        class_split_counts = split_metadata.get("class_split_counts", {})
        if isinstance(class_split_counts, dict) and class_split_counts:
            expected_train_counts = {
                class_name: int(
                    class_split_counts.get(class_name, {}).get("train", -1)
                )
                for class_name in CLASS_NAMES
            }
            expected_validation_counts = {
                class_name: int(
                    class_split_counts.get(class_name, {}).get("validation", -1)
                )
                for class_name in CLASS_NAMES
            }
            if expected_train_counts != class_counts:
                raise RuntimeError(
                    "train.csv class counts do not match split_metadata.json. "
                    f"CSV={class_counts}; metadata={expected_train_counts}. "
                    "Refusing to compute class weights from an inconsistent split."
                )

            actual_validation_counts = {
                class_name: int(
                    sum(
                        1
                        for record in validation_loader.dataset.records
                        if record["class_name"] == class_name
                    )
                )
                for class_name in CLASS_NAMES
            }
            if expected_validation_counts != actual_validation_counts:
                raise RuntimeError(
                    "validation.csv class counts do not match split_metadata.json. "
                    f"CSV={actual_validation_counts}; metadata={expected_validation_counts}."
                )

        accumulation_steps = max(
            1,
            int(config.gradient_accumulation_steps),
        )
        optimizer_steps_per_epoch = math.ceil(
            len(train_loader) / accumulation_steps
        )

        criterion, class_weights = build_loss(
            config=config,
            class_counts=class_counts,
            device=device,
        )
        loss_summary = get_loss_summary(
            config=config,
            class_counts=class_counts,
            class_weights=class_weights,
        )

        strategy = _make_training_strategy(config)
        total_stage_epochs = sum(strategy["stage_epochs"].values())
        if total_stage_epochs != int(config.epochs):
            raise RuntimeError(
                f"Progressive stage allocation ({total_stage_epochs}) "
                f"does not equal configured epochs ({config.epochs})."
            )

        # Stage 1 optimizer/scheduler summary is included in the architecture report.
        stage1_stats = set_trainable_stage(model, "classifier_only")
        stage1_config = _stage_config(
            config,
            strategy["stage_epochs"]["classifier_only"],
            strategy["stage_learning_rates"]["classifier_only"],
        )
        stage1_groups = get_stage_parameter_groups(
            model,
            "classifier_only",
            strategy["stage_learning_rates"]["classifier_only"],
            config.weight_decay,
        )
        stage1_optimizer = build_optimizer(
            model,
            stage1_config,
            parameter_groups=stage1_groups,
        )
        stage1_scheduler = build_scheduler(
            optimizer=stage1_optimizer,
            config=stage1_config,
            steps_per_epoch=optimizer_steps_per_epoch,
        )
        optimizer_summary = get_optimizer_summary(stage1_optimizer)
        scheduler_summary = get_scheduler_summary(
            stage1_scheduler,
            stage1_config,
        )

        base_trainer._write_architecture_report(
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

        environment_metadata = base_trainer._environment_metadata(
            device,
            amp_settings,
        )

        dataset_metadata["split_metadata"] = {
            "split_assignment_fingerprint_sha256": split_fingerprint,
            "split_unit": split_metadata.get("split_unit"),
            "grouping_method": split_metadata.get("grouping_method"),
            "random_seed": split_metadata.get("random_seed"),
            "split_ratios": split_metadata.get("split_ratios"),
            "effective_dataset_fingerprint_sha256": split_metadata.get(
                "effective_dataset_fingerprint_sha256"
            ),
        }

        preprocessing_summary = get_preprocessing_summary(
            image_size=config.image_size,
            augmentation=config.augmentation,
            normalization=config.normalization,
            augmentation_config=AugmentationConfig(
                horizontal_flip=config.horizontal_flip if config.augmentation else 0.0,
                vertical_flip=config.vertical_flip if config.augmentation else 0.0,
                rotation_degrees=config.rotation_degrees if config.augmentation else 0.0,
                brightness=config.brightness if config.augmentation else 0.0,
                contrast=config.contrast if config.augmentation else 0.0,
                saturation=config.saturation if config.augmentation else 0.0,
                hue=config.hue if config.augmentation else 0.0,
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
            "fine_tuning_strategy": strategy,
        }
        base_trainer._write_json(
            experiment_metadata,
            run_dir / "experiment_metadata.json",
        )
        base_trainer._write_json(
            strategy,
            run_dir / "mobnetfine_training_strategy.json",
        )

        (run_dir / "train_transform.txt").write_text(
            str(train_transform),
            encoding="utf-8",
        )
        (run_dir / "validation_transform.txt").write_text(
            str(validation_transform),
            encoding="utf-8",
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

        compatibility = {
            "model_name": config.model,
            "num_classes": NUM_CLASSES,
            "class_names": list(CLASS_NAMES),
            "image_size": int(config.image_size),
            "normalization": config.normalization,
            "split_assignment_fingerprint_sha256": split_fingerprint,
        }

        logger.save_metadata(
            config=config.to_dict(),
            model_metadata=model_metadata,
            dataset_metadata=dataset_metadata,
            environment_metadata=environment_metadata,
        )

        # One best-metric tracker spans all three stages. Early stopping is
        # activated only after the final full-backbone stage begins.
        stage_stopper = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            monitor=config.monitor,
            mode=config.monitor_mode,
            enabled=False,
            verbose=True,
        )

        global_step = 0
        global_epoch = 0
        stage_summaries = []
        stage_optimizer_summaries = []
        best_metric = None
        best_epoch = None
        best_stage = None
        total_epoch_seconds = 0.0

        print("\nTRAINING CONFIGURATION")
        print("-" * 76)
        print(f"Train images              : {len(train_loader.dataset):,}")
        print(f"Validation images         : {len(validation_loader.dataset):,}")
        print("Test images               : HELD OUT - NOT USED")
        print(f"Classes                   : {NUM_CLASSES}")
        print(f"Batch size                : {config.batch_size}")
        print(f"Workers                   : {config.num_workers}")
        print(f"Image size                : {config.image_size}")
        print(f"Total epochs              : {config.epochs}")
        print(
            "Progressive stages        : "
            + " + ".join(
                str(value)
                for value in strategy["stage_epochs"].values()
            )
        )
        print(f"Training batches/epoch    : {len(train_loader)}")
        print(f"Optimizer updates/epoch   : {optimizer_steps_per_epoch}")
        print(f"AMP                       : {amp_settings['enabled']} ({amp_settings['dtype']})")
        print(f"Early stopping            : {config.early_stopping} (full stage only)")
        print(f"Monitor                   : {config.monitor}")
        print(f"Split fingerprint         : {split_fingerprint}")
        print("-" * 76)

        for stage_name in ("classifier_only", "progressive", "full"):
            stage_epochs = int(strategy["stage_epochs"][stage_name])
            rates = strategy["stage_learning_rates"][stage_name]

            # Start each new stage from the best weights found so far.
            if checkpoint_manager.best_checkpoint_path.is_file():
                checkpoint_manager.load_model_weights(
                    model,
                    checkpoint_path=checkpoint_manager.best_checkpoint_path,
                    strict=True,
                )

            stage_stats = set_trainable_stage(model, stage_name)
            stage_config = _stage_config(config, stage_epochs, rates)
            parameter_groups = get_stage_parameter_groups(
                model,
                stage_name,
                rates,
                config.weight_decay,
            )
            optimizer = build_optimizer(
                model,
                stage_config,
                parameter_groups=parameter_groups,
            )
            scheduler = build_scheduler(
                optimizer=optimizer,
                config=stage_config,
                steps_per_epoch=optimizer_steps_per_epoch,
            )
            stage_optimizer_summary = get_optimizer_summary(optimizer)
            stage_scheduler_summary = get_scheduler_summary(
                scheduler,
                stage_config,
            )
            stage_optimizer_summaries.append(
                {
                    "stage": stage_name,
                    "optimizer": stage_optimizer_summary,
                    "scheduler": stage_scheduler_summary,
                }
            )

            if stage_name == "full":
                stage_stopper.enabled = bool(config.early_stopping)
                stage_stopper.bad_epochs = 0
                stage_stopper.should_stop = False
                stage_stopper.stop_epoch = None

            print("\n" + "=" * 76)
            print(
                f"STAGE: {stage_name.upper()} "
                f"({stage_epochs} epochs)"
            )
            print(
                f"Trainable parameters: "
                f"{stage_stats['trainable_parameters']:,}/"
                f"{stage_stats['total_parameters']:,}"
            )
            print(
                "Learning-rate groups: "
                + ", ".join(
                    f"{group['name']}={group['learning_rate']:.2e}"
                    for group in stage_optimizer_summary["groups"]
                )
            )
            print("=" * 76)

            stage_completed = 0
            stage_start_best = stage_stopper.best_value

            for stage_epoch in range(1, stage_epochs + 1):
                global_epoch += 1
                epoch_start_wall = datetime.now().astimezone().isoformat()
                epoch_start = time.perf_counter()
                learning_rates_start = scheduler.get_last_lr()

                train_result = base_trainer._run_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    device=device,
                    amp_settings=amp_settings,
                    epoch=global_epoch,
                    total_epochs=config.epochs,
                    split="train",
                    logger=logger,
                    optimizer=optimizer,
                    scaler=scaler,
                    scheduler_controller=scheduler,
                    gradient_accumulation_steps=accumulation_steps,
                    max_grad_norm=config.max_grad_norm,
                    global_step=global_step,
                )
                global_step = train_result["global_step"]

                validation_result = base_trainer._run_epoch(
                    model=model,
                    loader=validation_loader,
                    criterion=criterion,
                    device=device,
                    amp_settings=amp_settings,
                    epoch=global_epoch,
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

                monitored_value = base_trainer._get_monitor_value(
                    config,
                    validation_result["loss"],
                    validation_result["metrics"],
                )

                should_stop = stage_stopper.step(
                    monitored_value,
                    epoch=global_epoch,
                )
                is_best = bool(stage_stopper.improved)
                if is_best:
                    best_metric = float(stage_stopper.best_value)
                    best_epoch = int(stage_stopper.best_epoch)
                    best_stage = stage_name
                    logger.update_best(
                        epoch=best_epoch,
                        metric_name=config.monitor,
                        metric_value=best_metric,
                    )

                scheduler.step_epoch(
                    metric=(
                        monitored_value
                        if scheduler.requires_metric
                        else None
                    )
                )

                epoch_duration = time.perf_counter() - epoch_start
                total_epoch_seconds += epoch_duration
                epoch_end_wall = datetime.now().astimezone().isoformat()
                learning_rates_next = scheduler.get_last_lr()

                logger.log_epoch(
                    epoch=global_epoch,
                    train_metrics=train_result["metrics_result"],
                    validation_metrics=validation_result["metrics_result"],
                    learning_rate=float(learning_rates_start[0]),
                    train_loss=train_result["loss"],
                    validation_loss=validation_result["loss"],
                    epoch_duration_seconds=epoch_duration,
                    train_duration_seconds=train_result["duration_seconds"],
                    validation_duration_seconds=validation_result["duration_seconds"],
                    train_samples=train_result["samples"],
                    validation_samples=validation_result["samples"],
                    samples_per_second=train_result["samples_per_second"],
                    gpu_memory=logger.get_gpu_memory(),
                    epoch_start_timestamp=epoch_start_wall,
                    epoch_end_timestamp=epoch_end_wall,
                    extra={
                        "stage": stage_name,
                        "stage_epoch": stage_epoch,
                        "monitor": config.monitor,
                        "monitor_value": monitored_value,
                        "is_best": is_best,
                        "learning_rates_start": learning_rates_start,
                        "learning_rates_next": learning_rates_next,
                        "early_stopping_bad_epochs": stage_stopper.bad_epochs,
                        "global_step": global_step,
                    },
                )

                checkpoint_manager.save(
                    model=model,
                    epoch=global_epoch,
                    global_step=global_step,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    early_stopping=stage_stopper,
                    model_name=config.model,
                    metric_name=config.monitor,
                    metric_value=monitored_value,
                    is_best=is_best,
                    config=config.to_dict(),
                    compatibility=compatibility,
                    dataset_metadata=dataset_metadata,
                    environment_metadata=environment_metadata,
                    extra={
                        "training_stage": stage_name,
                        "stage_epoch": stage_epoch,
                        "stage_epochs": strategy["stage_epochs"],
                        "training_strategy": "progressive_msaF",
                    },
                )

                stage_completed += 1
                print(
                    f"EPOCH {global_epoch:03d}/{config.epochs:03d} | "
                    f"{stage_name:<15} | "
                    f"Train loss {train_result['loss']:.5f} | "
                    f"Train acc {train_result['metrics']['accuracy']:.4f} | "
                    f"Val loss {validation_result['loss']:.5f} | "
                    f"Val acc {validation_result['metrics']['accuracy']:.4f} | "
                    f"Val macro-F1 {validation_result['metrics']['f1_macro']:.4f} | "
                    f"LR {','.join(f'{lr:.2e}' for lr in learning_rates_next)} | "
                    f"Time {epoch_duration / 60:.2f} min"
                    + (" | NEW BEST" if is_best else "")
                )

                if stage_name == "full" and should_stop:
                    run_status = "early_stopped"
                    print(
                        f"\n[EARLY STOPPING] Full stage stopped at "
                        f"global epoch {global_epoch}."
                    )
                    break

            stage_summaries.append(
                {
                    "stage": stage_name,
                    "requested_epochs": stage_epochs,
                    "completed_epochs": stage_completed,
                    "trainable_parameters": stage_stats["trainable_parameters"],
                    "best_metric_before_stage": stage_start_best,
                    "best_metric_after_stage": best_metric,
                    "best_epoch_after_stage": best_epoch,
                    "early_stopped": bool(
                        stage_name == "full" and stage_stopper.should_stop
                    ),
                }
            )

            if stage_name == "full" and stage_stopper.should_stop:
                break

        if not checkpoint_manager.best_checkpoint_path.is_file():
            raise RuntimeError("Training ended without a best_model.pth checkpoint.")

        # Always restore the validation-selected best weights before evaluation.
        checkpoint_manager.load_model_weights(
            model,
            checkpoint_path=checkpoint_manager.best_checkpoint_path,
            strict=True,
        )
        model.eval()

        final_validation = None
        validation_artifacts = {}
        prediction_export_metadata = None

        if config.run_evaluation:
            print("\nGENERATING VALIDATION METRICS")
            (
                final_validation,
                validation_artifacts,
                prediction_export_metadata,
            ) = _run_final_validation(
                model=model,
                validation_loader=validation_loader,
                criterion=criterion,
                device=device,
                amp_settings=amp_settings,
                run_dir=run_dir,
                config=config,
                split_fingerprint=split_fingerprint,
                dataset_metadata=dataset_metadata,
                best_epoch=best_epoch,
                best_metric=best_metric,
            )

        curve_paths = base_trainer._plot_training_curves(
            logger.epoch_records,
            run_dir,
        )

        best_checkpoint = checkpoint_manager.best_checkpoint_path
        checkpoint_size_mb = (
            best_checkpoint.stat().st_size / (1024 ** 2)
            if best_checkpoint.is_file()
            else None
        )

        stage3_optimizer_parameters = (
            stage_optimizer_summaries[-1]["optimizer"]["total_optimized_parameters"]
            if stage_optimizer_summaries
            else 0
        )

        training_status = run_status

        run_summary = {
            "model": config.model,
            "display_name": model_metadata.get("display_name"),
            "run_name": run_dir.name,
            "status": run_status,
            "training_strategy": strategy,
            "stage_summaries": stage_summaries,
            "stage_optimizer_summaries": stage_optimizer_summaries,
            "best_epoch": best_epoch,
            "best_stage": best_stage,
            "best_monitor": config.monitor,
            "best_monitor_value": best_metric,
            "epochs_completed_this_run": logger.epochs_completed,
            "total_training_time_seconds": total_epoch_seconds,
            "total_training_time_minutes": total_epoch_seconds / 60.0,
            "global_optimizer_steps": global_step,
            "best_checkpoint": str(best_checkpoint),
            "best_checkpoint_size_mb": checkpoint_size_mb,
            "total_parameters": parameter_info["total_parameters"],
            "trainable_parameters": parameter_info["total_parameters"],
            "frozen_parameters": 0,
            "optimizer_parameters": stage3_optimizer_parameters,
            "split_assignment_fingerprint_sha256": split_fingerprint,
            "training_class_counts": class_counts,
            "class_weights": loss_summary.get("class_weights"),
            "validation_metrics": (
                final_validation.get("metrics")
                if final_validation is not None
                else None
            ),
            "validation_loss": (
                final_validation.get("loss")
                if final_validation is not None
                else None
            ),
            "training_curve_files": curve_paths,
            "validation_artifacts": validation_artifacts,
            "prediction_export": prediction_export_metadata,
            "test_partition_accessed": False,
            "post_training_pipeline": None,
        }

        base_trainer._write_json(
            run_summary,
            run_dir / "run_summary.json",
        )

        pipeline_summary = base_trainer._run_post_training_pipeline(
            config=config,
            run_dir=run_dir,
            device=device,
            amp_settings=amp_settings,
        )
        if pipeline_summary["failed_stages"]:
            run_status = "completed_with_warnings"

        run_summary["status"] = run_status
        run_summary["post_training_pipeline"] = pipeline_summary
        run_summary["pipeline_status"] = pipeline_summary["pipeline_status"]
        run_summary["pipeline_summary_path"] = str(
            run_dir / "pipeline_summary.json"
        )
        run_summary["post_training_pipeline_time_seconds"] = (
            pipeline_summary["duration_seconds"]
        )
        run_summary["post_training_pipeline_time_minutes"] = (
            pipeline_summary["duration_minutes"]
        )
        run_summary["total_run_time_seconds"] = logger.elapsed_seconds()
        run_summary["total_run_time_minutes"] = (
            run_summary["total_run_time_seconds"] / 60.0
        )

        base_trainer._write_json(
            run_summary,
            run_dir / "run_summary.json",
        )

        logger.finalize(
            status=training_status,
            best_epoch=best_epoch,
            best_metric=(
                {"name": config.monitor, "value": best_metric}
                if best_metric is not None
                else None
            ),
            additional_information=run_summary,
        )

        print("\n" + "=" * 76)
        print("MOBNETFINE COMPLETE RESEARCH PIPELINE FINISHED")
        print("=" * 76)
        print(f"Run status            : {run_status}")
        print(f"Best epoch            : {best_epoch}")
        print(f"Best stage            : {best_stage}")
        print(f"Best {config.monitor:<16}: {best_metric}")
        print(f"Checkpoint size       : {checkpoint_size_mb:.2f} MB")
        print(f"Run directory         : {run_dir}")
        print(f"Training history      : {run_dir / 'training_history'}")
        print(f"Validation evaluation : {run_dir / 'evaluation' / 'validation'}")
        print(f"Training curves       : {run_dir / 'training_curves'}")
        print(f"Benchmarking          : {run_dir / 'benchmarking'}")
        print(f"Error analysis        : {run_dir / 'evaluation' / 'validation' / 'error_analysis'}")
        print(f"XAI results           : {run_dir / 'Xai'}")
        print(f"LLM reports           : {run_dir / 'LLM'}")
        print(f"Pipeline summary      : {run_dir / 'pipeline_summary.json'}")
        print(f"Total training time   : {total_epoch_seconds / 60.0:.2f} minutes")
        print(f"Total execution time  : {run_summary['total_run_time_minutes']:.2f} minutes")
        print(f"Failed pipeline stages: {len(pipeline_summary['failed_stages'])}")
        print("Test partition        : NOT ACCESSED")
        print("SHAP                  : EXCLUDED")
        print("=" * 76)

        return run_summary

    except KeyboardInterrupt:
        if logger is not None:
            logger.finalize(
                status="interrupted",
                best_epoch=stage_stopper.best_epoch if stage_stopper else None,
                best_metric=(
                    {
                        "name": config.monitor,
                        "value": stage_stopper.best_value,
                    }
                    if stage_stopper and stage_stopper.best_value is not None
                    else None
                ),
                additional_information={
                    "message": "MobNetFine progressive training interrupted by user.",
                    "last_completed_epoch": logger.epochs_completed,
                },
            )
        raise

    except Exception as error:
        if logger is not None:
            try:
                logger.finalize(
                    status="failed",
                    best_epoch=stage_stopper.best_epoch if stage_stopper else None,
                    best_metric=(
                        {
                            "name": config.monitor,
                            "value": stage_stopper.best_value,
                        }
                        if stage_stopper and stage_stopper.best_value is not None
                        else None
                    ),
                    additional_information={"error": repr(error)},
                )
            except Exception:
                pass

        base_trainer._write_json(
            {
                "status": "failed",
                "error": repr(error),
                "timestamp_local": datetime.now().astimezone().isoformat(),
                "last_completed_epoch": (
                    logger.epochs_completed if logger is not None else 0
                ),
            },
            run_dir / "failure_report.json",
        )
        raise
