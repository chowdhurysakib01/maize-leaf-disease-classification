"""Three-stage fine-tuning for RiceLeaf-DMNet v2.

Initializes strictly from the epoch-21 best checkpoint, checks the fixed split
fingerprint, trains only on train.csv, selects using validation.csv macro-F1,
and never opens the held-out test manifest or test images.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.riceleaf_dmnet_finetuned import (
    build_finetuned_dmnet,
    get_stage_parameter_groups,
    set_trainable_stage,
)
from src.data.class_names import CLASS_NAMES, NUM_CLASSES
from src.data.dataset_loader import (
    MaizeLeafDataset,
    load_split_csv,
    seed_worker,
    validate_split_records,
)
from src.data.preprocessing import AugmentationConfig, build_transforms
from src.engine.loss_factory import build_loss
from src.engine.optimizer_factory import (
    build_optimizer,
    get_optimizer_summary,
    get_parameter_summary,
)
from src.engine.scheduler_factory import build_scheduler, get_scheduler_summary
from src.utils.device import (
    autocast_context,
    create_grad_scaler,
    get_amp_settings,
    select_device,
)
from src.utils.seed import create_generator, seed_everything

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "Results"
    / "riceleaf_dmnet"
    / "final_riceleaf_dmnet_v2_seed42"
    / "checkpoints"
    / "best_model.pth"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune RiceLeaf-DMNet v2 from its best checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-name", default="finetune_from_epoch21_seed42")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "Results" / "riceleaf_dmnet_finetuned")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"], default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.0001)
    parser.add_argument("--epochs-stage1", type=int, default=3)
    parser.add_argument("--epochs-stage2", type=int, default=6)
    parser.add_argument("--epochs-stage3", type=int, default=12)
    parser.add_argument("--stage1-lr", type=float, default=5e-6)
    parser.add_argument("--stage2-backbone-lr", type=float, default=8e-7)
    parser.add_argument("--stage2-fusion-lr", type=float, default=4e-6)
    parser.add_argument("--stage2-classifier-lr", type=float, default=4e-6)
    parser.add_argument("--stage3-backbone-lr", type=float, default=5e-7)
    parser.add_argument("--stage3-fusion-lr", type=float, default=2e-6)
    parser.add_argument("--stage3-classifier-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--validate-only", action="store_true", help="Load checkpoint, verify splits and evaluate baseline only.")
    return parser.parse_args(argv)


def _load_split_metadata():
    path = PROJECT_ROOT / "data_splits" / "split_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Fixed split metadata not found: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if int(metadata.get("num_classes", -1)) != NUM_CLASSES:
        raise RuntimeError("Split metadata class count does not match the 7-class setup.")
    if metadata.get("class_names") != list(CLASS_NAMES):
        raise RuntimeError("Split metadata class order does not match class_names.py.")
    fingerprint = metadata.get("split_assignment_fingerprint_sha256")
    if not fingerprint:
        raise RuntimeError("Split assignment fingerprint is missing.")
    return metadata, fingerprint


def _build_loaders(args, source_config, device, expected_fingerprint):
    # Deliberately load train and validation only. The test manifest is never opened.
    records = {
        "train": load_split_csv("train"),
        "validation": load_split_csv("validation"),
    }
    validate_split_records(records)

    metadata, fingerprint = _load_split_metadata()
    if fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Current split fingerprint does not match the source checkpoint. "
            "Refusing to fine-tune on a different split."
        )

    counts = Counter(record["class_name"] for record in records["train"])
    class_counts = {name: int(counts.get(name, 0)) for name in CLASS_NAMES}
    if any(value <= 0 for value in class_counts.values()):
        raise RuntimeError(f"A training class is missing: {class_counts}")

    augmentation = AugmentationConfig(
        horizontal_flip=float(source_config.get("horizontal_flip", 0.5)),
        vertical_flip=float(source_config.get("vertical_flip", 0.0)),
        rotation_degrees=float(source_config.get("rotation_degrees", 10.0)),
        brightness=float(source_config.get("brightness", 0.15)),
        contrast=float(source_config.get("contrast", 0.15)),
        saturation=float(source_config.get("saturation", 0.10)),
        hue=float(source_config.get("hue", 0.02)),
    )
    image_size = int(source_config.get("image_size", 224))
    normalization = str(source_config.get("normalization", "imagenet"))
    train_transform, validation_transform = build_transforms(
        image_size=image_size,
        augmentation=bool(source_config.get("augmentation", True)),
        normalization=normalization,
        augmentation_config=augmentation,
    )

    train_dataset = MaizeLeafDataset(records["train"], transform=train_transform)
    validation_dataset = MaizeLeafDataset(records["validation"], transform=validation_transform)
    workers = max(0, int(args.num_workers))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=create_generator(args.seed),
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=create_generator(args.seed + 1),
        persistent_workers=workers > 0,
    )
    return train_loader, validation_loader, class_counts, fingerprint, metadata


def _metric_summary(targets, predictions):
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(
            targets,
            predictions,
            labels=list(range(NUM_CLASSES)),
            average="macro",
            zero_division=0,
        )),
    }


def _run_epoch(model, loader, criterion, device, amp_settings, scaler,
               optimizer=None, accumulation_steps=1, max_grad_norm=1.0):
    training = optimizer is not None
    model.train(training)
    # Keep pretrained backbone BatchNorm running statistics fixed. Gradients can
    # still update trainable backbone weights during progressive/full stages.
    model.mobilenet_features.eval()
    model.densenet_features.eval()

    total_loss = 0.0
    sample_count = 0
    all_targets, all_predictions = [], []
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_batches = len(loader)
    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda", dtype=torch.long)
        batch_size = int(labels.shape[0])
        with torch.set_grad_enabled(training):
            with autocast_context(device, amp_settings):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss encountered.")
            if training:
                group_start = ((batch_index - 1) // accumulation_steps) * accumulation_steps
                group_size = min(accumulation_steps, total_batches - group_start)
                scaler.scale(loss / float(group_size)).backward()
                should_step = (batch_index % accumulation_steps == 0) or (batch_index == total_batches)
                if should_step:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    trainable = [p for p in model.parameters() if p.requires_grad]
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size
        predictions = logits.detach().argmax(dim=1)
        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    metrics = _metric_summary(all_targets, all_predictions)
    metrics["loss"] = total_loss / max(1, sample_count)
    return metrics


def _stage_config(base, epochs, max_lr, min_lr):
    return SimpleNamespace(
        optimizer="adamw",
        learning_rate=float(max_lr),
        weight_decay=float(base),
        optimizer_eps=1e-8,
        adam_beta1=0.9,
        adam_beta2=0.999,
        momentum=0.9,
        nesterov=False,
        scheduler="cosine",
        epochs=int(epochs),
        min_lr=float(min_lr),
        warmup_epochs=0,
        monitor="val_macro_f1",
        monitor_mode="max",
        scheduler_step_size=10,
        scheduler_gamma=0.1,
        scheduler_factor=0.5,
        scheduler_patience=3,
        min_delta=0.0,
    )


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(temporary, path)


def train(args):
    seed_everything(args.seed, deterministic=True, warn_only=True)
    device = torch.device(select_device(args.device))
    print("\n" + "=" * 76)
    print("RICELEAF-DMNET v2 — THREE-STAGE FINE-TUNING")
    print("=" * 76)
    print(f"Project : {PROJECT_ROOT}")
    print(f"Device  : {device}")
    if device.type == "cuda":
        print(f"GPU     : {torch.cuda.get_device_name(device)}")
        print(f"VRAM    : {torch.cuda.get_device_properties(device).total_memory / (1024 ** 3):.2f} GB")

    model, source_checkpoint = build_finetuned_dmnet(args.checkpoint, num_classes=NUM_CLASSES)
    compatibility = source_checkpoint.get("compatibility", {})
    source_config = source_checkpoint.get("config", {})
    expected_fingerprint = compatibility.get("split_assignment_fingerprint_sha256")
    if not expected_fingerprint:
        raise RuntimeError("Source checkpoint does not contain the split fingerprint.")
    if int(compatibility.get("num_classes", -1)) != NUM_CLASSES:
        raise RuntimeError("Source checkpoint class count does not match this project.")
    if compatibility.get("class_names") != list(CLASS_NAMES):
        raise RuntimeError("Source checkpoint class order does not match this project.")
    if int(compatibility.get("image_size", 224)) != int(source_config.get("image_size", 224)):
        raise RuntimeError("Checkpoint image-size metadata is inconsistent.")

    train_loader, validation_loader, class_counts, split_fingerprint, split_metadata = _build_loaders(
        args, source_config, device, expected_fingerprint
    )
    saved_counts = source_checkpoint.get("dataset_metadata", {}).get("training_class_counts", {})
    if saved_counts and saved_counts != class_counts:
        raise RuntimeError(
            "Training class counts differ from the source checkpoint. "
            f"Checkpoint={saved_counts}; current={class_counts}"
        )

    model = model.to(device)
    amp_settings = get_amp_settings(device, enabled=args.amp, dtype=args.amp_dtype)
    scaler = create_grad_scaler(amp_settings)

    # User-selected balanced objective: apply inverse-frequency class weights
    # computed exclusively from the training split (never validation/test labels).
    loss_config = SimpleNamespace(
        loss=source_config.get("loss", "cross_entropy"),
        class_weighting="inverse_frequency",
        label_smoothing=float(source_config.get("label_smoothing", 0.0)),
        focal_gamma=float(source_config.get("focal_gamma", 2.0)),
    )
    criterion, class_weights = build_loss(loss_config, class_counts=class_counts, device=device)

    source_best = float(source_checkpoint.get("best_metric", float("nan")))
    baseline = _run_epoch(
        model, validation_loader, criterion, device, amp_settings, scaler,
        optimizer=None,
    )
    print(
        f"Loaded checkpoint epoch {source_checkpoint.get('best_epoch', source_checkpoint.get('epoch'))}; "
        f"recorded macro-F1={source_best:.4%}; reproduced validation macro-F1={baseline['macro_f1']:.4%}"
    )
    if math.isfinite(source_best) and abs(baseline["macro_f1"] - source_best) > 0.02:
        print("WARNING: reproduced baseline differs by more than 2 percentage points; verify preprocessing and validation files.")

    if args.validate_only:
        print("VALIDATION-ONLY: checkpoint, strict state load, split fingerprint and baseline evaluation passed.")
        return

    run_dir = (args.results_dir / args.run_name).resolve()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    history_fields = ["epoch", "stage", "stage_epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1", "learning_rates", "elapsed_seconds"]
    with history_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=history_fields).writeheader()

    best_metric = baseline["macro_f1"]
    best_epoch = 0
    best_stage = "source_checkpoint"
    best_state = _cpu_state_dict(model)
    global_epoch = 0
    stale_epochs = 0
    stage_results = []
    source_best_epoch = int(source_checkpoint.get("best_epoch", source_checkpoint.get("epoch", 0)))

    def make_payload(state, epoch, stage, metric):
        return {
            "checkpoint_version": 1,
            "model_name": "riceleaf_dmnet_finetuned",
            "base_model_name": "riceleaf_dmnet",
            "epoch": int(epoch),
            "completed_epoch": int(epoch),
            "best_epoch": int(best_epoch if epoch != 0 else 0),
            "best_metric_name": "val_macro_f1",
            "best_metric": float(metric),
            "model_state_dict": state,
            "source_checkpoint": str(args.checkpoint.resolve()),
            "source_best_epoch": source_best_epoch,
            "source_best_macro_f1": source_best,
            "fine_tuning_stage": stage,
            "compatibility": compatibility,
            "split_assignment_fingerprint_sha256": split_fingerprint,
            "config": {
                "strategy": "classifier_only -> progressive_unfreezing -> full_network",
                "batch_size": args.batch_size,
                "accumulation_steps": args.accumulation_steps,
                "seed": args.seed,
                "amp": amp_settings,
                "stage_epochs": [args.epochs_stage1, args.epochs_stage2, args.epochs_stage3],
            },
        }

    _atomic_torch_save(make_payload(best_state, 0, best_stage, best_metric), checkpoint_dir / "best_model.pth")
    _write_json({
        "model_name": "riceleaf_dmnet_finetuned",
        "base_model_name": "riceleaf_dmnet",
        "best_epoch": 0,
        "best_stage": best_stage,
        "best_metric_name": "val_macro_f1",
        "best_metric_value": best_metric,
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_best_epoch": source_best_epoch,
        "source_best_macro_f1": source_best,
        "checkpoint": str(checkpoint_dir / "best_model.pth"),
    }, checkpoint_dir / "best_model_info.json")

    stages = [
        {
            "name": "classifier_only",
            "epochs": args.epochs_stage1,
            "rates": {"classifier": args.stage1_lr, "fusion": args.stage1_lr, "backbone": args.stage1_lr},
            "max_lr": args.stage1_lr,
        },
        {
            "name": "progressive",
            "epochs": args.epochs_stage2,
            "rates": {"backbone": args.stage2_backbone_lr, "fusion": args.stage2_fusion_lr, "classifier": args.stage2_classifier_lr},
            "max_lr": max(args.stage2_backbone_lr, args.stage2_fusion_lr, args.stage2_classifier_lr),
        },
        {
            "name": "full",
            "epochs": args.epochs_stage3,
            "rates": {"backbone": args.stage3_backbone_lr, "fusion": args.stage3_fusion_lr, "classifier": args.stage3_classifier_lr},
            "max_lr": max(args.stage3_backbone_lr, args.stage3_fusion_lr, args.stage3_classifier_lr),
        },
    ]

    for stage in stages:
        if stage["epochs"] <= 0:
            continue
        stage_stats = set_trainable_stage(model, stage["name"])
        groups = get_stage_parameter_groups(model, stage["name"], stage["rates"], args.weight_decay)
        stage_config = _stage_config(
            args.weight_decay,
            stage["epochs"],
            stage["max_lr"],
            min(1e-8, stage["max_lr"] / 100.0),
        )
        optimizer = build_optimizer(model, stage_config, parameter_groups=groups)
        scheduler = build_scheduler(
            optimizer=optimizer,
            config=stage_config,
            steps_per_epoch=math.ceil(len(train_loader) / max(1, args.accumulation_steps)),
        )
        print("\n" + "-" * 72)
        print(f"STAGE: {stage['name']} | epochs={stage['epochs']}")
        print(f"Trainable parameters: {stage_stats['trainable_parameters']:,}/{stage_stats['total_parameters']:,}")
        print("Optimizer groups:", get_optimizer_summary(optimizer)["groups"])

        stage_best = best_metric
        stage_stale = 0
        completed_stage_epochs = 0
        for stage_epoch in range(1, stage["epochs"] + 1):
            global_epoch += 1
            start_time = time.perf_counter()
            train_metrics = _run_epoch(
                model, train_loader, criterion, device, amp_settings, scaler,
                optimizer=optimizer,
                accumulation_steps=max(1, args.accumulation_steps),
                max_grad_norm=args.max_grad_norm,
            )
            val_metrics = _run_epoch(
                model, validation_loader, criterion, device, amp_settings, scaler,
                optimizer=None,
            )
            scheduler.step_epoch(metric=val_metrics["macro_f1"] if scheduler.requires_metric else None)
            elapsed = time.perf_counter() - start_time
            learning_rates = scheduler.get_last_lr()
            row = {
                "epoch": global_epoch,
                "stage": stage["name"],
                "stage_epoch": stage_epoch,
                "train_loss": train_metrics["loss"],
                "validation_loss": val_metrics["loss"],
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
                "learning_rates": json.dumps(learning_rates),
                "elapsed_seconds": round(elapsed, 3),
            }
            with history_path.open("a", newline="", encoding="utf-8") as file:
                csv.DictWriter(file, fieldnames=history_fields).writerow(row)

            improved = val_metrics["macro_f1"] > best_metric + args.min_delta
            if improved:
                best_metric = val_metrics["macro_f1"]
                best_epoch = global_epoch
                best_stage = stage["name"]
                best_state = _cpu_state_dict(model)
                stale_epochs = 0
                stage_stale = 0
                _atomic_torch_save(
                    make_payload(best_state, best_epoch, best_stage, best_metric),
                    checkpoint_dir / "best_model.pth",
                )
                _write_json({
                    "model_name": "riceleaf_dmnet_finetuned",
                    "base_model_name": "riceleaf_dmnet",
                    "best_epoch": best_epoch,
                    "best_stage": best_stage,
                    "best_metric_name": "val_macro_f1",
                    "best_metric_value": best_metric,
                    "source_checkpoint": str(args.checkpoint.resolve()),
                    "source_best_epoch": source_best_epoch,
                    "source_best_macro_f1": source_best,
                    "checkpoint": str(checkpoint_dir / "best_model.pth"),
                }, checkpoint_dir / "best_model_info.json")
            else:
                stale_epochs += 1
                stage_stale += 1

            completed_stage_epochs += 1
            print(
                f"[{stage['name']:<16}] {stage_epoch:02d}/{stage['epochs']:02d} "
                f"train_loss={train_metrics['loss']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4%} "
                f"val_macro_f1={val_metrics['macro_f1']:.4%} "
                f"best={best_metric:.4%} "
                f"lr={','.join(f'{x:.1e}' for x in learning_rates)}"
            )
            if stage_stale >= args.patience:
                print(f"Early stop within {stage['name']} after {stage_epoch} epochs without validation improvement.")
                break

        stage_results.append({
            "stage": stage["name"],
            "requested_epochs": stage["epochs"],
            "completed_epochs": completed_stage_epochs,
            "trainable_parameters": stage_stats["trainable_parameters"],
            "best_macro_f1_after_stage": best_metric,
        })
        # Save one stage-end recovery point instead of rewriting a 150+ MB file every epoch.
        _atomic_torch_save(
            make_payload(_cpu_state_dict(model), global_epoch, stage["name"], val_metrics["macro_f1"]),
            checkpoint_dir / "last_checkpoint.pth",
        )
        # Start the next stage from the best validation weights, not a worse tail epoch.
        model.load_state_dict(best_state, strict=True)

    model.load_state_dict(best_state, strict=True)
    summary = {
        "model_name": "riceleaf_dmnet_finetuned",
        "base_model": "RiceLeaf-DMNet v2",
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_best_epoch": source_best_epoch,
        "source_recorded_val_macro_f1": source_best,
        "fine_tuning_objective": "balanced_class_weighted_cross_entropy",
        "loss_name": loss_config.loss,
        "class_weighting": loss_config.class_weighting,
        "class_weights_from_training_split": (
            {
                name: float(weight)
                for name, weight in zip(
                    CLASS_NAMES,
                    class_weights.detach().cpu().tolist(),
                )
            }
            if class_weights is not None
            else None
        ),
        "best_checkpoint_selection_metric": "validation_macro_f1",
        "reproduced_baseline_val_loss": baseline["loss"],
        "reproduced_baseline_val_accuracy": baseline["accuracy"],
        "reproduced_baseline_val_macro_f1": baseline["macro_f1"],
        "best_finetuned_val_macro_f1": best_metric,
        "absolute_macro_f1_change": best_metric - baseline["macro_f1"],
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "split_assignment_fingerprint_sha256": split_fingerprint,
        "train_images": len(train_loader.dataset),
        "validation_images": len(validation_loader.dataset),
        "test_set_accessed": False,
        "stages": stage_results,
        "history_csv": str(history_path),
        "best_checkpoint": str(checkpoint_dir / "best_model.pth"),
    }
    _write_json(summary, run_dir / "training_summary.json")
    print("\n" + "=" * 76)
    print("FINE-TUNING COMPLETE")
    print(f"Baseline validation Macro-F1 : {baseline['macro_f1']:.4%}")
    print(f"Best fine-tuned Macro-F1      : {best_metric:.4%}")
    print(f"Change                        : {(best_metric - baseline['macro_f1'])*100:+.3f} percentage points")
    print(f"Best checkpoint               : {checkpoint_dir / 'best_model.pth'}")
    print("Held-out test set             : NOT ACCESSED")
    print("=" * 76)


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size <= 0 or args.accumulation_steps <= 0:
        raise ValueError("Batch size and accumulation steps must be positive.")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative.")
    if args.patience <= 0:
        raise ValueError("Patience must be positive.")
    if any(value < 0 for value in [args.epochs_stage1, args.epochs_stage2, args.epochs_stage3]):
        raise ValueError("Stage epochs cannot be negative.")
    train(args)


if __name__ == "__main__":
    main()
