"""
Held-out test evaluation for every unique trained checkpoint in D:/MaizeLeaf.

Uses ONLY data_splits/test.csv for predictions and XAI.
Never trains, fine-tunes, selects checkpoints, or changes source checkpoints.
Saves all artifacts under Results/heldout_test_evaluation/.
Deduplicates copied fine-tuned checkpoint aliases by SHA-256.
Checkpoint files are NOT copied into the output folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.model_factory import build_model_from_config, get_parameter_information
from src.data.class_names import CLASS_NAMES, NUM_CLASSES
from src.data.dataset_loader import MaizeLeafDataset, load_split_csv
from src.data.preprocessing import build_transforms
from src.engine.metrics import ClassificationMetrics
from src.evaluation.prediction_exporter import export_predictions
from src.utils.device import get_amp_settings

EXPECTED_SPLIT_FINGERPRINT = (
    "8a64e1a8a53eecd2dd2eeedbfb63c7e783a193304a2529da1be4e8c7bff441de"
)
EXPECTED_TEST_COUNT = 562
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Results" / "heldout_test_evaluation"
DEVICE = torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return default


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )


def find_fingerprint(value: Any):
    if isinstance(value, dict):
        direct = value.get("split_assignment_fingerprint_sha256")
        if isinstance(direct, str):
            return direct
        for child in value.values():
            found = find_fingerprint(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_fingerprint(child)
            if found:
                return found
    return None


def load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Invalid checkpoint object at {path}")
    state = checkpoint.get("model_state_dict")
    if state is None:
        state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise KeyError(f"No model_state_dict/state_dict found in {path}")
    return checkpoint, state


def candidate_run_root(checkpoint_path: Path) -> Path:
    root = checkpoint_path.parent.parent
    if root.name.lower() == "fine_tuning":
        root = root.parent
    return root


def candidate_score(path: Path, run_root: Path) -> int:
    score = 0
    if "fine_tuning" in [part.lower() for part in path.parts]:
        score += 10
    if (run_root / "history.csv").is_file():
        score += 6
    if (run_root / "training_config.json").is_file():
        score += 3
    if run_root.name.lower().endswith("_complete"):
        score -= 8
    if run_root.name.lower().startswith("final_"):
        score += 2
    return score


def infer_seed(run_name: str, config: dict, checkpoint: dict):
    match = re.search(r"seed[_-]?(\d+)", run_name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    for value in (checkpoint.get("seed"), config.get("seed")):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def normalize_model_key(value):
    if not value:
        return None
    key = str(value).strip().lower()
    return {
        "maizeleaf_dmnet_finetuned": "maizeleaf_dmnet",
        "maizeleaf-dmnet": "maizeleaf_dmnet",
        "mobilenetv3": "mobilenet_v3",
        "mobilenet_v3_large": "mobilenet_v3",
        "customcnn": "custom_cnn",
    }.get(key, key)


def discover_unique_checkpoints(results_root: Path, output_root: Path):
    candidates = []
    for checkpoint_path in results_root.rglob("best_model.pth"):
        try:
            checkpoint_path.resolve().relative_to(output_root.resolve())
            continue
        except ValueError:
            pass
        run_root = candidate_run_root(checkpoint_path)
        candidates.append({
            "checkpoint_path": checkpoint_path.resolve(),
            "run_root": run_root.resolve(),
            "sha256": sha256_file(checkpoint_path),
            "score": candidate_score(checkpoint_path, run_root),
        })

    if not candidates:
        raise FileNotFoundError(f"No best_model.pth checkpoints found under {results_root}")

    grouped = defaultdict(list)
    for item in candidates:
        grouped[item["sha256"]].append(item)

    unique = []
    for digest, group in grouped.items():
        group.sort(key=lambda x: x["score"], reverse=True)
        canonical = group[0]
        checkpoint, state_for_metadata = load_checkpoint(canonical["checkpoint_path"])

        config = read_json(canonical["run_root"] / "training_config.json", {})
        if not config:
            for alternative in group:
                alternative_config = read_json(alternative["run_root"] / "training_config.json", {})
                if alternative_config:
                    config = alternative_config
                    break

        model_key = normalize_model_key(config.get("model") or checkpoint.get("model_name"))
        if not model_key and "finetuned_eval" in canonical["run_root"].name.lower():
            model_key = "maizeleaf_dmnet"

        run_name = canonical["run_root"].name
        seed = infer_seed(run_name, config, checkpoint)

        if model_key == "maizeleaf_dmnet" and "finetuned_eval" in run_name.lower():
            display_name = "MaizeLeaf-DMNet Fine-Tuned"
        elif model_key == "maizeleaf_dmnet" and "_v2_" in run_name.lower():
            display_name = "MaizeLeaf-DMNet v2"
        elif model_key == "maizeleaf_dmnet" and run_name.lower() == "final_maizeleaf_dmnet_seed42":
            display_name = "MaizeLeaf-DMNet v1 (legacy)"
        else:
            display_name = model_key or "unknown"

        checkpoint_model_name = checkpoint.get("model_name")
        del state_for_metadata
        del checkpoint

        unique.append({
            "model_key": model_key,
            "display_name": display_name,
            "run_name": run_name,
            "seed": seed,
            "checkpoint_path": canonical["checkpoint_path"],
            "run_root": canonical["run_root"],
            "checkpoint_sha256": digest,
            "checkpoint_size_bytes": canonical["checkpoint_path"].stat().st_size,
            "checkpoint_model_name": checkpoint_model_name,
            "config": config,
            "aliases": [str(x["checkpoint_path"]) for x in group[1:]],
        })

    unique.sort(key=lambda x: (
        str(x["model_key"] or ""),
        0 if x["seed"] == 42 else 1 if x["seed"] == 123 else 2,
        x["run_name"],
    ))
    return unique


def build_model_for_run(item):
    model_key = item["model_key"]
    supported = {
        "custom_cnn", "resnet50", "efficientnet_b0", "mobilenet_v3",
        "densenet121", "convnext_tiny", "small_vit", "maizeleaf_dmnet",
        "mobnetfine",
    }
    if model_key not in supported:
        raise ValueError(f"Unsupported model key: {model_key}")

    # The uploaded source bundle contains DMNet v2 only. The old v1 run used
    # a different concatenation architecture, so do not mis-reconstruct it.
    if model_key == "maizeleaf_dmnet" and item["run_name"].lower() == "final_maizeleaf_dmnet_seed42":
        raise RuntimeError(
            "Legacy MaizeLeaf-DMNet v1 checkpoint: current Models/MaizeLeaf-DMNet.py "
            "contains v2 only. Original v1 model source is required for strict reconstruction."
        )

    config = dict(item["config"] or {})
    config.update({
        "model": model_key,
        "pretrained": False,
        "freeze_backbone": False,
        "num_classes": NUM_CLASSES,
    })
    config.setdefault("image_size", 224)
    config.setdefault("normalization", "imagenet")
    config.setdefault("vit_embed_dim", 192)
    config.setdefault("vit_depth", 6)
    config.setdefault("vit_heads", 3)

    kwargs = {}
    if model_key == "small_vit":
        kwargs = {
            "image_size": int(config["image_size"]),
            "embed_dim": int(config["vit_embed_dim"]),
            "depth": int(config["vit_depth"]),
            "num_heads": int(config["vit_heads"]),
        }

    model = build_model_from_config(SimpleNamespace(**config), **kwargs)
    checkpoint, state = load_checkpoint(item["checkpoint_path"])
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}

    checkpoint_for_xai = dict(checkpoint)
    checkpoint_for_xai["model_name"] = model_key

    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict state-dict mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model.to(DEVICE), checkpoint_for_xai, config


def read_prediction_arrays(prediction_csv: Path):
    y, probabilities = [], []
    with prediction_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            y.append(int(row["true_index"]))
            probabilities.append([
                float(row[f"prob_class_{index}"]) for index in range(NUM_CLASSES)
            ])
    y = np.asarray(y, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(y) != EXPECTED_TEST_COUNT or probabilities.shape != (EXPECTED_TEST_COUNT, NUM_CLASSES):
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_COUNT} test predictions; "
            f"got y={y.shape}, probabilities={probabilities.shape}"
        )
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise RuntimeError("Invalid or non-finite prediction probabilities.")
    return y, probabilities


def save_calibration(y_true, probabilities, output_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == y_true).astype(float)
    one_hot = np.eye(NUM_CLASSES, dtype=np.float64)[y_true]
    brier = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

    edges = np.linspace(0.0, 1.0, 11)
    bins, ece = [], 0.0
    for index in range(10):
        low, high = float(edges[index]), float(edges[index + 1])
        mask = (confidence >= low) & (
            (confidence <= high) if index == 9 else (confidence < high)
        )
        count = int(mask.sum())
        if count:
            mean_conf = float(confidence[mask].mean())
            accuracy = float(correct[mask].mean())
            ece += (count / len(y_true)) * abs(accuracy - mean_conf)
        else:
            mean_conf, accuracy = None, None
        bins.append({
            "bin": index + 1,
            "confidence_lower": low,
            "confidence_upper": high,
            "sample_count": count,
            "mean_confidence": mean_conf,
            "accuracy": accuracy,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "calibration_bins.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bins[0]))
        writer.writeheader()
        writer.writerows(bins)

    valid = [row for row in bins if row["sample_count"]]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    if valid:
        ax.plot(
            [row["mean_confidence"] for row in valid],
            [row["accuracy"] for row in valid],
            marker="o",
            label="Model",
        )
    ax.set(
        xlim=(0, 1), ylim=(0, 1),
        xlabel="Mean predicted confidence",
        ylabel="Observed accuracy",
        title="Held-out Test Reliability Diagram",
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "calibration_reliability.png", dpi=300)
    plt.close(fig)

    report = {
        "split": "test",
        "num_samples": int(len(y_true)),
        "accuracy": float(np.mean(predicted == y_true)),
        "mean_top_label_confidence": float(confidence.mean()),
        "top_label_ece_10_bins": float(ece),
        "multiclass_brier_score": brier,
        "classwise_brier_scores": {
            CLASS_NAMES[index]: float(np.mean(
                (probabilities[:, index] - (y_true == index).astype(float)) ** 2
            ))
            for index in range(NUM_CLASSES)
        },
        "calibration_bins": bins,
        "test_partition_accessed": True,
    }
    write_json(output_dir / "calibration_report.json", report)
    return report


def run_error_analysis(eval_dir: Path):
    from argparse import Namespace
    from src.evaluation.error_analysis import run_error_analysis as run
    run(Namespace(
        evaluation_dir=eval_dir,
        top_k=10,
        confidence_threshold=0.80,
        contact_sheet_limit=48,
        no_contact_sheet=False,
    ))


def run_inference_benchmark(model, device, image_size, checkpoint_size_bytes, output_dir):
    device = torch.device(device)
    model.eval()
    batch_results = []

    for batch_size in (1, 8, 16):
        try:
            sample = torch.randn(batch_size, 3, image_size, image_size, device=device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            def forward():
                if device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        return model(sample)
                return model(sample)

            with torch.inference_mode():
                for _ in range(5):
                    _ = forward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                iterations = 20
                for _ in range(iterations):
                    _ = forward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start

            peak_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                if device.type == "cuda" else None
            )
            batch_results.append({
                "batch_size": batch_size,
                "iterations": iterations,
                "total_elapsed_seconds": elapsed,
                "batch_latency_ms": elapsed / iterations * 1000,
                "per_image_latency_ms": elapsed / iterations / batch_size * 1000,
                "throughput_images_per_second": iterations * batch_size / elapsed,
                "peak_gpu_allocated_mb": peak_mb,
                "status": "completed",
            })
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            batch_results.append({
                "batch_size": batch_size,
                "status": "out_of_memory",
                "error": str(exc),
            })
        except Exception as exc:
            batch_results.append({
                "batch_size": batch_size,
                "status": "failed",
                "error": repr(exc),
            })

    try:
        parameters = get_parameter_information(model)["total_parameters"]
    except Exception:
        parameters = sum(parameter.numel() for parameter in model.parameters())

    result = {
        "benchmark_type": "synthetic_input_inference_benchmark",
        "note": "Synthetic tensors are used; test images and labels are not used for timing.",
        "device": str(device),
        "image_size": image_size,
        "checkpoint_size_mb": round(checkpoint_size_bytes / (1024 ** 2), 3),
        "parameters": int(parameters),
        "parameters_millions": round(parameters / 1e6, 4),
        "batch_results": batch_results,
    }
    write_json(output_dir / "efficiency_benchmark.json", result)

    lines = [
        "HELD-OUT EVALUATION - INFERENCE BENCHMARK",
        f"Device: {device}",
        f"Input: 3x{image_size}x{image_size}",
        "Synthetic inputs only; no test labels used.",
        f"Checkpoint size: {result['checkpoint_size_mb']} MB",
        f"Parameters: {parameters:,}",
        "",
        "Batch | per-image latency ms | throughput img/s | peak GPU MB | status",
    ]
    for row in batch_results:
        lines.append(
            f"{row['batch_size']} | {row.get('per_image_latency_ms', '')} | "
            f"{row.get('throughput_images_per_second', '')} | "
            f"{row.get('peak_gpu_allocated_mb', '')} | {row['status']}"
        )
    (output_dir / "efficiency_benchmark_report.txt").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result


def replace_split_terms(value):
    if isinstance(value, dict):
        return {
            str(key).replace("validation", "test").replace("Validation", "Test"):
            replace_split_terms(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [replace_split_terms(item) for item in value]
    if isinstance(value, str):
        return (
            value.replace("VALIDATION", "TEST")
            .replace("Validation", "Test")
            .replace("validation", "test")
        )
    return value


def relabel_xai_reports(xai_root: Path):
    for path in xai_root.rglob("*.json"):
        data = read_json(path)
        if data is not None:
            write_json(path, replace_split_terms(data))
    for path in xai_root.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
            text = (
                text.replace("VALIDATION", "TEST")
                .replace("Validation", "Test")
                .replace("validation", "test")
            )
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass


def run_test_xai(model, checkpoint, checkpoint_path, run_dir, eval_dir,
                 test_records, device, samples_per_class, lime_samples,
                 lime_features, lime_segments):
    from argparse import Namespace
    from src.explainability import gradcam as common
    from src.explainability import gradcam_plus_plus as gradcam_pp
    from src.explainability import lime_explainer as lime

    xai_root = run_dir / "Xai" / "test"
    xai_root.mkdir(parents=True, exist_ok=True)

    originals = {
        "load_validation_records": common.load_validation_records,
        "read_prediction_lookup": common.read_prediction_lookup,
        "load_best_model": common.load_best_model,
    }

    # Redirect the existing validation-oriented XAI engines to the held-out
    # test records and test predictions. No validation records are loaded.
    common.load_validation_records = lambda: test_records

    def test_prediction_lookup(_run_dir):
        lookup = {}
        with (eval_dir / "predictions.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                key = str(Path(row["image_path"]).expanduser().resolve()).casefold()
                lookup[key] = str(row["correct"]).strip().lower() in {"true", "1"}
        return lookup

    common.read_prediction_lookup = test_prediction_lookup
    common.load_best_model = lambda _run_dir, _config, _model_name, _device: (
        model, checkpoint, checkpoint_path
    )

    try:
        common_args = {
            "run_dir": run_dir,
            "device": str(device),
            "samples_per_class": samples_per_class,
            "sample_strategy": "balanced",
            "target_class": "predicted",
            "seed": 42,
            "no_contact_sheet": False,
        }

        common.run_gradcam(Namespace(
            **common_args, output_dir=xai_root / "gradcam"
        ))
        gradcam_pp.run_gradcam_plus_plus(Namespace(
            **common_args,
            output_dir=xai_root / "gradcam_plus_plus",
            eps=1e-8,
        ))
        lime.run_lime(Namespace(
            **common_args,
            output_dir=xai_root / "lime",
            num_samples=lime_samples,
            num_features=lime_features,
            num_segments=lime_segments,
            compactness=10.0,
            sigma=0.8,
            prediction_batch_size=8,
            hide_color=None,
        ))

        relabel_xai_reports(xai_root)
        return {
            "status": "completed",
            "samples_per_class": samples_per_class,
            "images_per_method": samples_per_class * NUM_CLASSES,
        }
    finally:
        common.load_validation_records = originals["load_validation_records"]
        common.read_prediction_lookup = originals["read_prediction_lookup"]
        common.load_best_model = originals["load_best_model"]


def prepare_output_run(item, output_root, fingerprint, test_manifest_hash):
    run_dir = output_root / item["model_key"] / item["run_name"]
    if run_dir.exists():
        previous = read_json(run_dir / "complete_test_evaluation_summary.json")
        if previous and previous.get("status") == "completed":
            return run_dir, True
        # Only clear an incomplete folder inside the dedicated test-evaluation root.
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation" / "test").mkdir(parents=True, exist_ok=True)

    config = dict(item["config"] or {})
    config.update({
        "model": item["model_key"],
        "pretrained": False,
        "freeze_backbone": False,
        "num_classes": NUM_CLASSES,
        "seed": item["seed"],
        "run_name": item["run_name"],
        "normalization": config.get("normalization", "imagenet"),
    })
    config.setdefault("image_size", 224)
    write_json(run_dir / "training_config.json", config)

    snapshot = read_json(PROJECT_ROOT / "data_splits" / "split_metadata.json")
    write_json(run_dir / "split_metadata_snapshot.json", snapshot)
    write_json(
        run_dir / "source_run_summary.json",
        read_json(item["run_root"] / "run_summary.json", {}) or {},
    )
    write_json(run_dir / "run_summary.json", {
        "model": item["model_key"],
        "run_name": item["run_name"],
        "seed": item["seed"],
        "status": "heldout_test_evaluation",
        "source_checkpoint": str(item["checkpoint_path"]),
        "source_run_directory": str(item["run_root"]),
        "checkpoint_sha256": item["checkpoint_sha256"],
        "split_assignment_fingerprint_sha256": fingerprint,
        "test_manifest_sha256": test_manifest_hash,
        "test_partition_accessed": True,
        "training_or_finetuning_performed": False,
    })
    write_json(run_dir / "evaluation_metadata.json", {
        "evaluation_type": "post-training held-out test evaluation",
        "model": item["display_name"],
        "model_key": item["model_key"],
        "run_name": item["run_name"],
        "seed": item["seed"],
        "checkpoint_used": str(item["checkpoint_path"]),
        "checkpoint_sha256": item["checkpoint_sha256"],
        "checkpoint_size_bytes": item["checkpoint_size_bytes"],
        "num_classes": NUM_CLASSES,
        "class_names": list(CLASS_NAMES),
        "image_size": int(config["image_size"]),
        "normalization": config["normalization"],
        "split_fingerprint": fingerprint,
        "split_used": "test",
        "test_images": EXPECTED_TEST_COUNT,
        "test_manifest_sha256": test_manifest_hash,
        "test_partition_accessed": True,
        "checkpoint_copied_to_output": False,
    })
    return run_dir, False


def evaluate_one(item, args, device, test_records, fingerprint, test_manifest_hash):
    run_dir, already_done = prepare_output_run(
        item, args.output_root, fingerprint, test_manifest_hash
    )
    if already_done:
        print(f"[SKIP] Already completed: {item['model_key']} / {item['run_name']}")
        return read_json(run_dir / "complete_test_evaluation_summary.json")

    print("\n" + "=" * 84)
    print(f"HELD-OUT TEST: {item['display_name']} | {item['run_name']} | seed={item['seed']}")
    print(f"Checkpoint: {item['checkpoint_path']}")
    print("=" * 84)

    started = time.perf_counter()
    model, checkpoint, config = build_model_for_run(item)
    image_size = int(config.get("image_size", 224))
    normalization = config.get("normalization", "imagenet")

    _, eval_transform = build_transforms(
        image_size=image_size,
        augmentation=False,
        normalization=normalization,
        interpolation="bilinear",
    )
    dataset = MaizeLeafDataset(records=test_records, transform=eval_transform)
    eval_dir = run_dir / "evaluation" / "test"

    amp_settings = get_amp_settings(
        device=device, enabled=False, dtype="float16"
    )
    export_predictions(
        model=model,
        dataset=dataset,
        device=device,
        amp_settings=amp_settings,
        output_csv=eval_dir / "predictions.csv",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        model_name=item["model_key"],
        run_name=item["run_name"],
        split_name="test",
        split_fingerprint=fingerprint,
        dataset_metadata={
            "num_images": len(dataset),
            "split": "test",
            "test_partition_accessed": True,
            "test_manifest_sha256": test_manifest_hash,
        },
        training_metadata={
            "source_run": str(item["run_root"]),
            "seed": item["seed"],
            "checkpoint_sha256": item["checkpoint_sha256"],
            "fine_tuned": "finetuned" in item["run_name"].lower(),
        },
    )

    y_true, probabilities = read_prediction_arrays(eval_dir / "predictions.csv")
    metric_engine = ClassificationMetrics(
        num_classes=NUM_CLASSES, class_names=list(CLASS_NAMES)
    )
    metric_engine.update_from_probabilities(probabilities, y_true)
    metric_paths = metric_engine.save(eval_dir, split_name="test")
    calibration = save_calibration(y_true, probabilities, eval_dir)
    run_error_analysis(eval_dir)

    benchmark = run_inference_benchmark(
        model=model,
        device=device,
        image_size=image_size,
        checkpoint_size_bytes=item["checkpoint_size_bytes"],
        output_dir=run_dir / "benchmarking",
    )

    xai_result = {"status": "not_run"}
    if not args.skip_xai:
        try:
            xai_result = run_test_xai(
                model=model,
                checkpoint=checkpoint,
                checkpoint_path=item["checkpoint_path"],
                run_dir=run_dir,
                eval_dir=eval_dir,
                test_records=test_records,
                device=device,
                samples_per_class=args.samples_per_class,
                lime_samples=args.lime_num_samples,
                lime_features=args.lime_num_features,
                lime_segments=args.lime_num_segments,
            )
        except Exception as exc:
            xai_result = {"status": "failed", "error": repr(exc)}
            write_json(run_dir / "Xai" / "test" / "xai_failure.json", xai_result)
            print(f"[WARNING] XAI failed for this run: {exc}")

    computed = metric_engine.compute()
    metrics = computed["metrics"]
    summary = {
        "status": "completed",
        "model": item["model_key"],
        "display_name": item["display_name"],
        "run_name": item["run_name"],
        "seed": item["seed"],
        "source_run_directory": str(item["run_root"]),
        "checkpoint_path": str(item["checkpoint_path"]),
        "checkpoint_sha256": item["checkpoint_sha256"],
        "checkpoint_size_mb": round(item["checkpoint_size_bytes"] / (1024 ** 2), 3),
        "test_samples": int(len(y_true)),
        "test_accuracy": metrics["accuracy"],
        "test_precision_macro": metrics["precision_macro"],
        "test_recall_macro": metrics["recall_macro"],
        "test_macro_f1": metrics["f1_macro"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
        "test_pr_auc_macro_average_precision": metrics["pr_auc_macro_average_precision"],
        "test_ece_10_bins": calibration["top_label_ece_10_bins"],
        "test_multiclass_brier_score": calibration["multiclass_brier_score"],
        "parameters": benchmark["parameters"],
        "parameters_millions": benchmark["parameters_millions"],
        "benchmark_batch_results": benchmark["batch_results"],
        "xai": xai_result,
        "metric_artifacts": metric_paths,
        "test_manifest_sha256": test_manifest_hash,
        "split_assignment_fingerprint_sha256": fingerprint,
        "test_partition_accessed": True,
        "training_or_finetuning_performed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "note": (
            "Included as an exploratory run; prior validation-overfitting concerns "
            "should be considered when interpreting MobNetFine."
            if item["model_key"] == "mobnetfine" else None
        ),
    }
    write_json(run_dir / "complete_test_evaluation_summary.json", summary)

    del model, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"[PASS] Test accuracy={metrics['accuracy']:.4f}, "
        f"macro-F1={metrics['f1_macro']:.4f}, "
        f"ROC-AUC={metrics['roc_auc_ovr_macro']:.4f}, "
        f"PR-AUC={metrics['pr_auc_macro_average_precision']:.4f}"
    )
    return summary


def save_master_outputs(output_root: Path, summaries: list, manifest: list):
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for summary in summaries:
        if not summary:
            continue
        row = {
            "model": summary.get("model"),
            "display_name": summary.get("display_name"),
            "run_name": summary.get("run_name"),
            "seed": summary.get("seed"),
            "test_samples": summary.get("test_samples"),
            "accuracy": summary.get("test_accuracy"),
            "precision_macro": summary.get("test_precision_macro"),
            "recall_macro": summary.get("test_recall_macro"),
            "macro_f1": summary.get("test_macro_f1"),
            "balanced_accuracy": summary.get("test_balanced_accuracy"),
            "roc_auc_ovr_macro": summary.get("test_roc_auc_ovr_macro"),
            "pr_auc_macro_average_precision": summary.get("test_pr_auc_macro_average_precision"),
            "ece_10_bins": summary.get("test_ece_10_bins"),
            "multiclass_brier_score": summary.get("test_multiclass_brier_score"),
            "parameters": summary.get("parameters"),
            "parameters_millions": summary.get("parameters_millions"),
            "checkpoint_size_mb": summary.get("checkpoint_size_mb"),
            "xai_status": (summary.get("xai") or {}).get("status"),
            "checkpoint_path": summary.get("checkpoint_path"),
            "test_partition_accessed": summary.get("test_partition_accessed"),
        }
        for batch in summary.get("benchmark_batch_results") or []:
            suffix = f"batch{batch.get('batch_size')}"
            row[f"{suffix}_latency_ms_per_image"] = batch.get("per_image_latency_ms")
            row[f"{suffix}_throughput_images_per_second"] = batch.get("throughput_images_per_second")
            row[f"{suffix}_gpu_peak_mb"] = batch.get("peak_gpu_allocated_mb")
        rows.append(row)

    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output_root / "master_test_comparison.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_json(output_root / "master_test_comparison.json", rows)
        try:
            import pandas as pd
            pd.DataFrame(rows).to_excel(
                output_root / "master_test_comparison.xlsx", index=False
            )
        except Exception as exc:
            (output_root / "xlsx_export_unavailable.txt").write_text(
                f"Excel export unavailable: {exc}\nCSV remains the master comparison.",
                encoding="utf-8",
            )

    if manifest:
        fields = sorted({key for row in manifest for key in row})
        with (output_root / "evaluation_manifest.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest)


def main():
    global DEVICE
    parser = argparse.ArgumentParser(
        description="Evaluate every unique MaizeLeaf best checkpoint on the held-out test split."
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "cuda:0"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--lime-num-samples", type=int, default=1000)
    parser.add_argument("--lime-num-features", type=int, default=10)
    parser.add_argument("--lime-num-segments", type=int, default=80)
    parser.add_argument("--skip-xai", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("Invalid batch size or worker count.")

    if args.device == "auto":
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        DEVICE = torch.device("cuda:0" if args.device == "cuda" else args.device)
    if DEVICE.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")

    output_root = args.output_root.expanduser().resolve()
    results_root = PROJECT_ROOT / "Results"
    output_root.mkdir(parents=True, exist_ok=True)

    split_meta = read_json(PROJECT_ROOT / "data_splits" / "split_metadata.json")
    fingerprint = find_fingerprint(split_meta)
    if fingerprint != EXPECTED_SPLIT_FINGERPRINT:
        raise RuntimeError("Central split fingerprint mismatch. Held-out evaluation stopped.")

    test_manifest_path = PROJECT_ROOT / "data_splits" / "test.csv"
    if not test_manifest_path.is_file():
        raise FileNotFoundError(test_manifest_path)
    test_manifest_hash = sha256_file(test_manifest_path)

    # This is the only split manifest loaded by this evaluator.
    test_records = load_split_csv("test")
    if len(test_records) != EXPECTED_TEST_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_COUNT} test images, found {len(test_records)}."
        )
    if any(str(record.get("split", "")).lower() != "test" for record in test_records):
        raise RuntimeError("Non-test records found in data_splits/test.csv.")
    if len({record["relative_path"] for record in test_records}) != len(test_records):
        raise RuntimeError("Duplicate image paths found in the test manifest.")
    if {record["label"] for record in test_records} != set(range(NUM_CLASSES)):
        raise RuntimeError("Test manifest does not contain all seven classes.")

    items = discover_unique_checkpoints(results_root, output_root)

    print("=" * 84)
    print("MAIZE LEAF - COMPLETE HELD-OUT TEST EVALUATION")
    print("=" * 84)
    print("Output root       :", output_root)
    print("Device            :", DEVICE)
    print("Unique checkpoints:", len(items))
    print("Test images       :", len(test_records))
    print("Split fingerprint :", fingerprint)
    print("Test manifest SHA :", test_manifest_hash)
    print("Train/validation manifests: NOT OPENED")
    print("Training/fine-tuning: NOT PERFORMED")
    print("Checkpoint files copied to output: NO")

    summaries, manifest = [], []
    for item in items:
        base = {
            "model": item["model_key"],
            "run_name": item["run_name"],
            "seed": item["seed"],
            "checkpoint_path": str(item["checkpoint_path"]),
            "checkpoint_sha256": item["checkpoint_sha256"],
            "duplicate_alias_count": len(item["aliases"]),
            "duplicate_aliases": " ; ".join(item["aliases"]),
        }
        try:
            if item["model_key"] is None:
                raise RuntimeError("Could not identify model key from config/checkpoint.")
            summary = evaluate_one(
                item, args, DEVICE, test_records, fingerprint, test_manifest_hash
            )
            if summary:
                summaries.append(summary)
                manifest.append({**base, "status": "completed"})
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            print(f"[FAILED] {item['run_name']}: {error_text}")
            print(traceback.format_exc())
            manifest.append({**base, "status": "failed", "error": error_text})
            if item["model_key"]:
                failure_dir = output_root / item["model_key"] / item["run_name"]
                failure_dir.mkdir(parents=True, exist_ok=True)
                write_json(failure_dir / "evaluation_failure.json", {
                    "status": "failed",
                    "model": item["model_key"],
                    "run_name": item["run_name"],
                    "checkpoint_path": str(item["checkpoint_path"]),
                    "error": error_text,
                    "traceback": traceback.format_exc(),
                })
        save_master_outputs(output_root, summaries, manifest)

    save_master_outputs(output_root, summaries, manifest)
    print("\n" + "=" * 84)
    print("HELD-OUT TEST EVALUATION FINISHED")
    print("Completed runs :", sum(1 for row in manifest if row["status"] == "completed"))
    print("Failed runs    :", sum(1 for row in manifest if row["status"] == "failed"))
    print("Master CSV     :", output_root / "master_test_comparison.csv")
    print("Manifest       :", output_root / "evaluation_manifest.csv")
    print("=" * 84)


if __name__ == "__main__":
    main()
