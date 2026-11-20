"""Complete validation evaluation for the fine-tuned RiceLeaf-DMNet checkpoint.

Creates an isolated compatibility run under:
    Results/riceleaf_dmnet/finetuned_eval_seed42

Uses ONLY the fixed validation split. The held-out test manifest is never opened.
The original fine-tuning checkpoint and training results are never modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.model_factory import build_model
from src.data.class_names import CLASS_NAMES, NUM_CLASSES
from src.data.dataset_loader import MaizeLeafDataset, load_split_csv
from src.data.preprocessing import build_transforms
from src.evaluation.prediction_exporter import export_predictions
from src.engine.metrics import ClassificationMetrics
from src.utils.device import get_amp_settings

EXPECTED_SPLIT_FINGERPRINT = (
    "8a64e1a8a53eecd2dd2eeedbfb63c7e783a193304a2529da1be4e8c7bff441de"
)
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "Results"
    / "riceleaf_dmnet_finetuned"
    / "RiceLeaf-DMNet-Finetuned_seed42"
    / "checkpoints"
    / "best_model.pth"
)
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "Results"
    / "riceleaf_dmnet"
    / "finetuned_eval_seed42"
)


def _load_torch(path: Path) -> dict[str, Any]:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise TypeError(f"Invalid checkpoint type: {type(data).__name__}")
    return data


def _find_fingerprint(value: Any) -> str | None:
    if isinstance(value, dict):
        direct = value.get("split_assignment_fingerprint_sha256")
        if isinstance(direct, str):
            return direct
        for child in value.values():
            found = _find_fingerprint(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_fingerprint(child)
            if found:
                return found
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=4, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _verify_checkpoint_and_split(checkpoint: dict[str, Any]) -> str:
    compatibility = checkpoint.get("compatibility", {})
    checkpoint_fingerprint = (
        checkpoint.get("split_assignment_fingerprint_sha256")
        or compatibility.get("split_assignment_fingerprint_sha256")
    )
    if checkpoint_fingerprint != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(
            "Fine-tuned checkpoint split fingerprint does not match the "
            "project's verified fixed split. Evaluation stopped."
        )

    central_metadata = _read_json(PROJECT_ROOT / "data_splits" / "split_metadata.json")
    current_fingerprint = _find_fingerprint(central_metadata)
    if current_fingerprint != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError(
            "Current data_splits/split_metadata.json does not match the "
            "fine-tuning checkpoint fingerprint. Evaluation stopped."
        )
    return current_fingerprint


def _prepare_eval_run(
    checkpoint_path: Path,
    run_dir: Path,
    overwrite: bool,
) -> tuple[dict[str, Any], str]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Fine-tuned best checkpoint not found: {checkpoint_path}\n"
            "Expected Results/riceleaf_dmnet_finetuned/"
            "RiceLeaf-DMNet-Finetuned_seed42/checkpoints/best_model.pth"
        )

    checkpoint = _load_torch(checkpoint_path)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint is missing model_state_dict.")
    if checkpoint.get("model_name") not in (
        "riceleaf_dmnet_finetuned",
        "riceleaf_dmnet",
    ):
        raise RuntimeError(
            f"Unexpected checkpoint model_name: {checkpoint.get('model_name')!r}"
        )
    fingerprint = _verify_checkpoint_and_split(checkpoint)

    # Grad-CAM and the benchmarker expect the registry key to match the
    # Results/<model_key>/<run_name> layout. Make a compatibility copy only;
    # never edit the user's original fine-tuning checkpoint.
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Evaluation directory already exists: {run_dir}\n"
                "Use --overwrite to replace this evaluation directory only."
            )
        shutil.rmtree(run_dir)

    if run_dir.parent.name != "riceleaf_dmnet":
        raise ValueError(
            "The evaluation run directory must be inside Results/riceleaf_dmnet/ "
            "so the existing model factory and XAI tools can reconstruct DMNet."
        )

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation" / "validation").mkdir(parents=True, exist_ok=True)

    eval_checkpoint = dict(checkpoint)
    eval_checkpoint["model_name"] = "riceleaf_dmnet"
    eval_checkpoint["base_model_name"] = "riceleaf_dmnet"
    eval_checkpoint["evaluation_checkpoint_alias"] = True
    torch.save(eval_checkpoint, run_dir / "checkpoints" / "best_model.pth")

    # Preserve the source training configuration where available, but set
    # pretrained=False for reconstruction: all learned weights are in the checkpoint.
    source_config_path = (
        PROJECT_ROOT
        / "Results"
        / "riceleaf_dmnet"
        / "final_riceleaf_dmnet_v2_seed42"
        / "training_config.json"
    )
    if source_config_path.is_file():
        training_config = _read_json(source_config_path)
    else:
        training_config = {}

    training_config.update({
        "model": "riceleaf_dmnet",
        "pretrained": False,
        "freeze_backbone": False,
        "num_classes": NUM_CLASSES,
        "image_size": int(checkpoint.get("compatibility", {}).get("image_size", 224)),
        "normalization": "imagenet",
        "seed": 42,
        "run_name": run_dir.name,
        "run_evaluation": False,
        "run_benchmark": False,
        "run_error_analysis": False,
        "run_xai": False,
        "run_llm": False,
    })
    _write_json(run_dir / "training_config.json", training_config)

    run_summary = {
        "model": "riceleaf_dmnet",
        "run_name": run_dir.name,
        "status": "evaluation_only",
        "best_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))),
        "best_monitor": checkpoint.get("best_metric_name", "val_macro_f1"),
        "best_monitor_value": float(checkpoint.get("best_metric", 0.0)),
        "split_assignment_fingerprint_sha256": fingerprint,
        "test_partition_accessed": False,
        "source_model_name": "riceleaf_dmnet_finetuned",
        "source_checkpoint": str(checkpoint_path.resolve()),
        "backbone_initialization": "ImageNet pretrained, then fine-tuned; reconstruction uses checkpoint weights only.",
    }
    _write_json(run_dir / "run_summary.json", run_summary)

    # Copy the original DMNet v2 split snapshot when present. Otherwise use
    # the current central split metadata (already fingerprint-verified above).
    source_snapshot = (
        PROJECT_ROOT
        / "Results"
        / "riceleaf_dmnet"
        / "final_riceleaf_dmnet_v2_seed42"
        / "split_metadata_snapshot.json"
    )
    if not source_snapshot.is_file():
        source_snapshot = PROJECT_ROOT / "data_splits" / "split_metadata.json"
    shutil.copy2(source_snapshot, run_dir / "split_metadata_snapshot.json")

    # Preserve the fine-tuning curve/history beside the evaluation outputs.
    source_history = (
        checkpoint_path.parent.parent / "history.csv"
    )
    if source_history.is_file():
        shutil.copy2(source_history, run_dir / "fine_tuning_history.csv")

    _write_json(run_dir / "evaluation_metadata.json", {
        "evaluation_type": "post-training validation evaluation",
        "model": "RiceLeaf-DMNet Fine-Tuned",
        "checkpoint_used": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))),
        "checkpoint_stage": checkpoint.get("fine_tuning_stage"),
        "checkpoint_macro_f1": checkpoint.get("best_metric"),
        "base_model": checkpoint.get("base_model_name", "riceleaf_dmnet"),
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "image_size": training_config["image_size"],
        "normalization": "ImageNet",
        "split_fingerprint": fingerprint,
        "split_used": "validation only",
        "test_partition_accessed": False,
        "backbone_initialization": "ImageNet pretrained, then fine-tuned",
        "note": "The model is reconstructed without downloading pretrained weights because the checkpoint already contains the complete fine-tuned state_dict.",
    })

    return checkpoint, fingerprint


def _export_validation_predictions(
    checkpoint: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Path:
    print("\n[1/7] EXPORTING VALIDATION PREDICTIONS")
    model = build_model(
        model_name="riceleaf_dmnet",
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
    )
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint loading reported incompatible keys.")
    model.to(device).eval()

    records = load_split_csv("validation")
    if len(records) != 564:
        raise RuntimeError(
            f"Expected 564 validation records, found {len(records)}. "
            "Check that the fixed split is unchanged."
        )
    _, eval_transform = build_transforms(
        image_size=224,
        augmentation=False,
        normalization="imagenet",
        interpolation="bilinear",
    )
    dataset = MaizeLeafDataset(records=records, transform=eval_transform)
    output_csv = run_dir / "evaluation" / "validation" / "predictions.csv"
    amp_settings = get_amp_settings(
        device=device,
        enabled=False,
        dtype="float16",
    )
    export_predictions(
        model=model,
        dataset=dataset,
        device=device,
        amp_settings=amp_settings,
        output_csv=output_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        model_name="riceleaf_dmnet_finetuned",
        run_name=run_dir.name,
        split_name="validation",
        split_fingerprint=EXPECTED_SPLIT_FINGERPRINT,
        dataset_metadata={
            "num_images": len(dataset),
            "split": "validation",
            "test_partition_accessed": False,
        },
        training_metadata={
            "source_model": "RiceLeaf-DMNet v2",
            "fine_tuned": True,
            "checkpoint_epoch": int(checkpoint.get("best_epoch", 0)),
            "checkpoint_stage": checkpoint.get("fine_tuning_stage"),
            "best_validation_macro_f1_during_finetuning": checkpoint.get("best_metric"),
        },
    )
    del model, dataset, records
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_csv


def _read_predictions(prediction_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    targets: list[int] = []
    probabilities: list[list[float]] = []
    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            targets.append(int(row["true_index"]))
            probabilities.append([
                float(row[f"prob_class_{i}"])
                for i in range(NUM_CLASSES)
            ])
    y = np.asarray(targets, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(y) != 564 or p.shape != (564, NUM_CLASSES):
        raise RuntimeError(f"Unexpected prediction array shapes: y={y.shape}, p={p.shape}")
    if not np.isfinite(p).all():
        raise FloatingPointError("Non-finite probabilities found in predictions.csv")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Prediction probabilities do not sum to one.")
    return y, p


def _save_calibration_report(y: np.ndarray, p: np.ndarray, output_dir: Path) -> dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = p.argmax(axis=1)
    confidence = p.max(axis=1)
    correct = (pred == y).astype(np.float64)
    one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[y]

    # Multiclass Brier score: mean_i sum_k (p_ik - y_ik)^2.
    brier = float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

    # Top-label ECE with 10 equal-width confidence bins, matching the
    # project's error-analysis convention.
    edges = np.linspace(0.0, 1.0, 11)
    bins = []
    ece = 0.0
    for i in range(10):
        lower, upper = float(edges[i]), float(edges[i + 1])
        if i == 9:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count:
            mean_conf = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += (count / len(y)) * abs(accuracy - mean_conf)
        else:
            mean_conf = None
            accuracy = None
        bins.append({
            "bin": i + 1,
            "confidence_lower": lower,
            "confidence_upper": upper,
            "sample_count": count,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
        })

    class_brier = {}
    for class_idx, class_name in enumerate(CLASS_NAMES):
        target_binary = (y == class_idx).astype(np.float64)
        class_brier[class_name] = float(np.mean((p[:, class_idx] - target_binary) ** 2))

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibration_bins.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(bins[0].keys()))
        writer.writeheader()
        writer.writerows(bins)

    fig, ax = plt.subplots(figsize=(7, 6))
    valid_bins = [b for b in bins if b["sample_count"] > 0]
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    if valid_bins:
        ax.plot(
            [b["mean_confidence"] for b in valid_bins],
            [b["accuracy"] for b in valid_bins],
            marker="o",
            label="Fine-tuned DMNet",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_title("Validation Reliability Diagram")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_reliability.png", dpi=300)
    plt.close(fig)

    report = {
        "split": "validation",
        "num_samples": int(len(y)),
        "accuracy": float(np.mean(pred == y)),
        "mean_top_label_confidence": float(confidence.mean()),
        "top_label_ece_10_bins": float(ece),
        "multiclass_brier_score": brier,
        "multiclass_brier_definition": "Mean over samples of the sum across seven classes of (predicted_probability - one_hot_target)^2.",
        "classwise_brier_scores": class_brier,
        "calibration_bins": bins,
        "test_partition_accessed": False,
    }
    (output_dir / "calibration_report.json").write_text(
        json.dumps(report, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def _run_command(label: str, command: list[str]) -> None:
    print("\n" + "=" * 76)
    print(label)
    print("=" * 76)
    print("Command:", subprocess.list2cmdline(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _correct_benchmark_provenance(run_dir: Path) -> None:
    """Record the true initialization history in benchmark metadata.

    The benchmark reconstructs the network with pretrained=False to avoid
    downloading weights; the checkpoint itself contains ImageNet-initialized
    and subsequently fine-tuned weights. Correct the factory-only flags so
    the exported report describes the actual training provenance.
    """
    path = run_dir / "benchmarking" / "efficiency_benchmark.json"
    if not path.is_file():
        return
    data = _read_json(path)
    metadata = data.setdefault("model_metadata", {})
    metadata["pretrained_requested"] = True
    metadata["pretrained_used"] = True
    metadata["checkpoint_initialization"] = (
        "ImageNet pretrained MobileNetV3-Large and DenseNet121, "
        "then fine-tuned; evaluation reconstruction loaded checkpoint weights "
        "without downloading pretrained weights."
    )
    architecture = metadata.get("architecture")
    if isinstance(architecture, dict):
        architecture["pretrained_backbones"] = True
        architecture["fine_tuned_from_pretrained_checkpoint"] = True
    data["evaluation_model_name"] = "riceleaf_dmnet_finetuned"
    data["checkpoint_provenance"] = metadata["checkpoint_initialization"]
    _write_json(path, data)

    report_path = run_dir / "benchmarking" / "efficiency_benchmark_report.txt"
    if report_path.is_file():
        with report_path.open("a", encoding="utf-8") as f:
            f.write("\nCHECKPOINT PROVENANCE\n")
            f.write("ImageNet pretrained MobileNetV3-Large + DenseNet121, then fine-tuned.\n")
            f.write("The benchmark reconstructed the architecture from the saved checkpoint without downloading weights.\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete validation evaluation for fine-tuned RiceLeaf-DMNet."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--lime-num-samples", type=int, default=1000)
    parser.add_argument("--lime-num-features", type=int, default=10)
    parser.add_argument("--lime-num-segments", type=int, default=80)
    parser.add_argument("--benchmark-iterations", type=int, default=20)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-xai", action="store_true", help="Skip Grad-CAM, Grad-CAM++ and LIME.")
    parser.add_argument("--skip-benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()

    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid batch size or worker count.")
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be positive.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this environment.")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()

    print("=" * 76)
    print("RICELEAF-DMNET FINE-TUNED - COMPLETE VALIDATION EVALUATION")
    print("=" * 76)
    print("Project root :", PROJECT_ROOT)
    print("Checkpoint  :", checkpoint_path)
    print("Output      :", run_dir)
    print("Device      :", device)
    print("Split       : validation only (564 images)")
    print("Test set    : HELD OUT - NOT ACCESSED")

    checkpoint, fingerprint = _prepare_eval_run(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        overwrite=args.overwrite,
    )
    print("[PASS] Checkpoint fingerprint matches the fixed split")
    print("[PASS] Evaluation compatibility run prepared")

    # Strictly verify architecture reconstruction before running any pipeline.
    model_check = build_model(
        model_name="riceleaf_dmnet",
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
    )
    model_check.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del model_check
    print("[PASS] Fine-tuned checkpoint strictly matches RiceLeaf-DMNet v2 architecture")

    prediction_csv = _export_validation_predictions(
        checkpoint=checkpoint,
        run_dir=run_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Classification report, confusion matrices, ROC-AUC, PR-AUC and curves.
    print("\n[2/7] GENERATING CLASSIFICATION / ROC / PR METRICS")
    y_true, probabilities = _read_predictions(prediction_csv)
    metric_engine = ClassificationMetrics(
        num_classes=NUM_CLASSES,
        class_names=list(CLASS_NAMES),
    )
    metric_engine.update_from_probabilities(probabilities, y_true)
    evaluation_dir = run_dir / "evaluation" / "validation"
    metric_paths = metric_engine.save(evaluation_dir, split_name="validation")
    print("[PASS] Classification report, confusion matrices, ROC and PR artifacts saved")

    # Calibration report adds multiclass Brier score to the project's ECE analysis.
    print("\n[3/7] GENERATING CALIBRATION REPORT")
    calibration = _save_calibration_report(y_true, probabilities, evaluation_dir)
    print(f"[PASS] Top-label ECE (10 bins): {calibration['top_label_ece_10_bins']:.6f}")
    print(f"[PASS] Multiclass Brier score : {calibration['multiclass_brier_score']:.6f}")

    # Existing error-analysis engine also writes confidence bins, ECE and its
    # own reliability plot plus error/contact-sheet reports.
    _run_command(
        "[4/7] ERROR ANALYSIS",
        [
            sys.executable, "-m", "src.evaluation.error_analysis",
            "--evaluation-dir", str(evaluation_dir),
            "--top-k", "10",
            "--confidence-threshold", "0.8",
            "--contact-sheet-limit", "48",
        ],
    )

    if not args.skip_xai:
        xai_common = [
            "--run-dir", str(run_dir),
            "--device", str(device),
            "--samples-per-class", str(args.samples_per_class),
            "--sample-strategy", "balanced",
            "--target-class", "predicted",
            "--seed", "42",
        ]
        _run_command(
            "[5/7] GRAD-CAM",
            [sys.executable, "-m", "src.explainability.gradcam", *xai_common],
        )
        _run_command(
            "[6/7] GRAD-CAM++",
            [sys.executable, "-m", "src.explainability.gradcam_plus_plus", *xai_common],
        )
        _run_command(
            "[7/7] LIME",
            [
                sys.executable, "-m", "src.explainability.lime_explainer",
                *xai_common,
                "--num-samples", str(args.lime_num_samples),
                "--num-features", str(args.lime_num_features),
                "--num-segments", str(args.lime_num_segments),
                "--prediction-batch-size", "8",
            ],
        )

    if not args.skip_benchmark:
        benchmark_command = [
            sys.executable, "-m", "src.benchmarking.efficiency_benchmark",
            "--run-dir", str(run_dir),
            "--device", str(device),
            "--batch-sizes", "1", "8", "16",
            "--warmup", str(args.benchmark_warmup),
            "--iterations", str(args.benchmark_iterations),
        ]
        if device.type == "cuda":
            benchmark_command.extend(["--amp", "--amp-dtype", "float16"])
        _run_command("EFFICIENCY BENCHMARK", benchmark_command)
        _correct_benchmark_provenance(run_dir)

    final_summary = {
        "model": "riceleaf_dmnet_finetuned",
        "display_name": "RiceLeaf-DMNet Fine-Tuned",
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("best_epoch", checkpoint.get("epoch", 0))),
        "checkpoint_stage": checkpoint.get("fine_tuning_stage"),
        "checkpoint_validation_macro_f1_during_finetuning": checkpoint.get("best_metric"),
        "split": "validation",
        "split_fingerprint": fingerprint,
        "test_partition_accessed": False,
        "classification_metrics": str(evaluation_dir / "metrics.json"),
        "classification_report": str(evaluation_dir / "classification_report.csv"),
        "confusion_matrix": str(evaluation_dir / "confusion_matrix.png"),
        "roc_curve": str(evaluation_dir / "roc_curves.png"),
        "pr_curve": str(evaluation_dir / "precision_recall_curves.png"),
        "calibration_report": str(evaluation_dir / "calibration_report.json"),
        "calibration_reliability": str(evaluation_dir / "calibration_reliability.png"),
        "error_analysis_dir": str(evaluation_dir / "error_analysis"),
        "gradcam_dir": str(run_dir / "Xai" / "gradcam"),
        "gradcam_plus_plus_dir": str(run_dir / "Xai" / "gradcam_plus_plus"),
        "lime_dir": str(run_dir / "Xai" / "lime"),
        "benchmark_dir": str(run_dir / "benchmarking"),
        "validation_accuracy": float(np.mean(probabilities.argmax(axis=1) == y_true)),
        "validation_macro_f1": float(metric_engine.compute()["metrics"]["f1_macro"]),
        "validation_macro_roc_auc": metric_engine.compute()["metrics"]["roc_auc_ovr_macro"],
        "validation_macro_average_precision": metric_engine.compute()["metrics"]["pr_auc_macro_average_precision"],
        "top_label_ece_10_bins": calibration["top_label_ece_10_bins"],
        "multiclass_brier_score": calibration["multiclass_brier_score"],
        "elapsed_seconds": time.perf_counter() - started,
        "metric_artifacts": metric_paths,
    }
    _write_json(run_dir / "complete_evaluation_summary.json", final_summary)

    print("\n" + "=" * 76)
    print("FINE-TUNED DMNET VALIDATION EVALUATION COMPLETED")
    print("=" * 76)
    print(f"Accuracy        : {final_summary['validation_accuracy']:.6f}")
    print(f"Macro-F1        : {final_summary['validation_macro_f1']:.6f}")
    print(f"Macro ROC-AUC   : {final_summary['validation_macro_roc_auc']:.6f}")
    print(f"Macro PR-AUC/AP : {final_summary['validation_macro_average_precision']:.6f}")
    print(f"Top-label ECE   : {final_summary['top_label_ece_10_bins']:.6f}")
    print(f"Multiclass Brier: {final_summary['multiclass_brier_score']:.6f}")
    print("Test partition : NOT ACCESSED")
    print("Output         :", run_dir)
    print("Summary        :", run_dir / "complete_evaluation_summary.json")
    print("=" * 76)


if __name__ == "__main__":
    main()
