
"""
MAIZE LEAF CLASSIFICATION
GRAD-CAM++ EXPLAINABILITY ENGINE

File:
src/explainability/gradcam_plus_plus.py

Uses the verified shared Grad-CAM utilities.
Only the fixed validation partition is accessed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw

from src.explainability import gradcam as common


CLASS_NAMES = common.CLASS_NAMES
NUM_CLASSES = common.NUM_CLASSES
PROJECT_ROOT = common.PROJECT_ROOT


def print_section(title):
    """
    Display only the two requested progress-stage headings.

    All routine setup/report output is hidden by the public wrapper.
    The tqdm progress bar remains visible because tqdm writes to stderr.
    """
    visible_sections = {
        "GENERATING GRAD-CAM++ EXPLANATIONS",
        "GENERATING GRAD-CAM++ SUCCESSFUL",
    }

    if title not in visible_sections:
        return

    stream = sys.__stdout__ or sys.stdout
    stream.write("\n" + "=" * 76 + "\n")
    stream.write(title + "\n")
    stream.write("=" * 76 + "\n")
    stream.flush()


def compute_gradcam_plus_plus(
    model,
    target_layer,
    reshape_transform,
    input_tensor,
    target_class,
    eps=1e-8,
):
    """
    Grad-CAM++ computation.

    alpha = grad^2 / (2*grad^2 + sum(A*grad^3))
    weights = sum(alpha * ReLU(grad))
    CAM = ReLU(sum(weights * activation))
    """

    captured = {}

    def forward_hook(module, inputs, output):
        activation = common.extract_feature_tensor(output)
        captured["activation"] = activation

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():

            logits = common.extract_logits(model(input_tensor))

            if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
                raise RuntimeError(
                    f"Expected [B,{NUM_CLASSES}] logits, "
                    f"received {tuple(logits.shape)}"
                )

            if input_tensor.shape[0] != 1:
                raise ValueError(
                    "Grad-CAM++ requires batch size 1."
                )

            if not 0 <= int(target_class) < NUM_CLASSES:
                raise ValueError("Invalid target class.")

            activation = captured.get("activation")

            if activation is None:
                raise RuntimeError(
                    "Target-layer activation was not captured."
                )

            if not activation.requires_grad:
                raise RuntimeError(
                    "Target activation does not require gradients."
                )

            target_score = logits[0, int(target_class)]

            gradients = torch.autograd.grad(
                outputs=target_score,
                inputs=activation,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

        activation = common.reshape_feature_tensor(
            activation.detach(),
            reshape_transform,
        )

        gradients = common.reshape_feature_tensor(
            gradients.detach(),
            reshape_transform,
        )

        if activation.ndim != 4 or gradients.ndim != 4:
            raise RuntimeError(
                "Grad-CAM++ requires 4D spatial feature maps."
            )

        if activation.shape != gradients.shape:
            raise RuntimeError(
                "Activation and gradient shapes do not match."
            )

        gradients_squared = gradients.pow(2)
        gradients_cubed = gradients_squared * gradients

        sum_activation_gradient_cubed = (
            activation * gradients_cubed
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        denominator = (
            2.0 * gradients_squared
            + sum_activation_gradient_cubed
        )

        denominator = torch.where(
            denominator.abs() < eps,
            torch.where(
                denominator < 0,
                torch.full_like(denominator, -eps),
                torch.full_like(denominator, eps),
            ),
            denominator,
        )

        alpha = gradients_squared / denominator

        weights = (
            alpha * F.relu(gradients)
        ).sum(
            dim=(2, 3),
            keepdim=True,
        )

        cam = F.relu(
            (weights * activation).sum(
                dim=1,
                keepdim=True,
            )
        )

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().float().cpu().numpy()

        cam = common.normalize_cam(cam)

        if not np.isfinite(cam).all():
            raise FloatingPointError(
                "Non-finite Grad-CAM++ values."
            )

        return (
            cam,
            float(target_score.detach().float().cpu().item()),
        )

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def save_visualization(
    original_image,
    cam,
    sample_slug,
    directories,
):

    height, width = cam.shape

    resampling = getattr(
        Image,
        "Resampling",
        Image,
    ).LANCZOS

    display_image = original_image.resize(
        (width, height),
        resampling,
    )

    original_array = np.asarray(
        display_image,
        dtype=np.uint8,
    )

    colormap = plt.get_cmap("jet")

    heatmap_array = (
        colormap(cam)[..., :3] * 255
    ).astype(np.uint8)

    overlay_array = np.clip(
        original_array.astype(np.float32) * 0.55
        + heatmap_array.astype(np.float32) * 0.45,
        0,
        255,
    ).astype(np.uint8)

    paths = {
        "original_file": directories["originals"]
            / f"{sample_slug}_original.png",

        "heatmap_file": directories["heatmaps"]
            / f"{sample_slug}_heatmap.png",

        "overlay_file": directories["overlays"]
            / f"{sample_slug}_overlay.png",

        "comparison_file": directories["comparisons"]
            / f"{sample_slug}_comparison.png",

        "cam_array_file": directories["arrays"]
            / f"{sample_slug}_cam.npy",
    }

    Image.fromarray(original_array).save(
        paths["original_file"]
    )

    Image.fromarray(heatmap_array).save(
        paths["heatmap_file"]
    )

    Image.fromarray(overlay_array).save(
        paths["overlay_file"]
    )

    np.save(
        paths["cam_array_file"],
        cam.astype(np.float32),
    )

    comparison = Image.new(
        "RGB",
        (width * 3, height + 38),
        "white",
    )

    comparison.paste(
        Image.fromarray(original_array),
        (0, 0),
    )

    comparison.paste(
        Image.fromarray(heatmap_array),
        (width, 0),
    )

    comparison.paste(
        Image.fromarray(overlay_array),
        (width * 2, 0),
    )

    draw = ImageDraw.Draw(comparison)

    for index, label in enumerate(
        ("Original", "Grad-CAM++", "Overlay")
    ):

        draw.text(
            (
                index * width + max(5, width // 2 - 35),
                height + 10,
            ),
            label,
            fill="black",
        )

    comparison.save(
        paths["comparison_file"]
    )

    comparison.close()
    display_image.close()

    return {
        key: str(value)
        for key, value in paths.items()
    }


def write_report(
    summary,
    class_summary,
    results,
    output_path,
):

    lines = [
        "=" * 78,
        "MAIZE LEAF CLASSIFICATION - GRAD-CAM++ REPORT",
        "=" * 78,
        "",
        f"Model: {summary['model_name']}",
        f"Run: {summary['run_name']}",
        f"Checkpoint: {summary['checkpoint']}",
        f"Checkpoint epoch: {summary['checkpoint_epoch']}",
        f"Partition: {summary['partition']}",
        f"Target layer: {summary['target_layer']}",
        f"Target class policy: {summary['target_class_policy']}",
        f"Samples explained: {summary['sample_count']}",
        "",
        "OVERALL SUMMARY",
        "-" * 78,
        f"Correct: {summary['correct_predictions']}",
        f"Incorrect: {summary['incorrect_predictions']}",
        f"Subset accuracy: {summary['sample_accuracy']:.6f}",
        f"Mean confidence: {summary['mean_confidence']:.6f}",
        f"Mean CAM intensity: {summary['mean_cam_intensity']:.6f}",
        f"Mean CAM time: {summary['mean_runtime_seconds']:.6f}",
        f"Total runtime: {summary['total_runtime_seconds']:.4f}",
        "",
        "CLASS-WISE SUMMARY",
        "-" * 78,
    ]

    for item in class_summary:

        lines.extend([
            "",
            item["class"],
            f"  Selected: {item['selected_images']}",
            f"  Correct: {item['correct_predictions']}",
            f"  Incorrect: {item['incorrect_predictions']}",
            f"  Subset accuracy: {item['sample_accuracy']}",
            f"  Mean confidence: {item['mean_confidence']}",
            f"  Mean CAM intensity: {item['mean_cam_intensity']}",
        ])

    lines.extend([
        "",
        "SAMPLE DETAILS",
        "-" * 78,
    ])

    for item in results:

        lines.extend([
            "",
            f"Sample: {item['sample_id']}",
            f"Actual: {item['actual_class']}",
            f"Predicted: {item['predicted_class']}",
            f"Confidence: {item['confidence']:.6f}",
            f"Correct: {item['correct']}",
            f"Target class: {item['target_class']}",
            f"CAM mean: {item['cam_mean']:.6f}",
        ])

    lines.extend([
        "",
        "DATA INTEGRITY",
        "-" * 78,
        f"Split fingerprint: {summary['split_fingerprint']}",
        f"Seed: {summary['seed']}",
        "Validation manifest only: YES",
        "Train manifest loaded: NO",
        "Test manifest loaded: NO",
        "Test set used: NO",
        f"Group IDs available: {summary['validation_group_ids_available']}",
        "",
        "LIMITATIONS",
        "-" * 78,
        "Grad-CAM++ is a post-hoc explanation method.",
        "Highlighted regions do not establish biological causality.",
        "Heatmap intensity is not disease severity.",
        "The balanced subset is not a replacement for full validation metrics.",
        "Visual interpretation requires domain expertise.",
        "",
        "=" * 78,
    ])

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _run_gradcam_plus_plus_internal(args):

    run_dir = Path(
        args.run_dir
    ).expanduser().resolve()

    if not run_dir.is_dir():
        raise NotADirectoryError(
            f"Run directory not found: {run_dir}"
        )

    if args.samples_per_class < 1:
        raise ValueError(
            "Samples per class must be at least 1."
        )

    if args.seed < 0:
        raise ValueError(
            "Seed must be non-negative."
        )

    if not np.isfinite(args.eps) or args.eps <= 0:
        raise ValueError(
            "Epsilon must be positive and finite."
        )

    if args.device == "auto":

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    else:

        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    if args.device == "cuda":
        device = torch.device("cuda:0")

    print_section(
        "MAIZE LEAF - GRAD-CAM++ EXPLAINABILITY ENGINE"
    )

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Run directory: {run_dir}")
    print(f"Device       : {device}")

    if device.type == "cuda":

        print(
            f"GPU          : {torch.cuda.get_device_name(device)}"
        )

        print(
            f"GPU memory   : "
            f"{torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print_section("VERIFYING FIXED DATASET SPLIT")

    split_fingerprint = common.verify_split(run_dir)

    print_section("LOADING TRAINING CONFIGURATION")

    config, config_data, model_name = (
        common.load_training_configuration(run_dir)
    )

    print(f"Model      : {model_name}")
    print(f"Image size : {config.image_size}")
    print(f"Classes    : {NUM_CLASSES}")

    for index, name in enumerate(CLASS_NAMES):
        print(f"{index}: {name}")

    print_section("LOADING BEST MODEL")

    model, checkpoint, checkpoint_path = (
        common.load_best_model(
            run_dir,
            config,
            model_name,
            device,
        )
    )

    print_section("CONFIGURING GRAD-CAM++ TARGET LAYER")

    target_layer, reshape_transform, target_layer_name = (
        common.get_target_configuration(
            model,
            model_name,
        )
    )

    print_section("BUILDING VALIDATION DATA")

    validation_records = common.load_validation_records()

    group_ids_available = any(
        str(record.get("group_id", "")).strip()
        for record in validation_records
    )

    if not group_ids_available:
        print("[INFO] Validation group IDs are empty.")

    validation_transform, image_size = (
        common.build_validation_transform(config)
    )

    print(f"[PASS] Image size: {image_size}")
    print("[PASS] Validation preprocessing configured")
    print("[PASS] Test partition not accessed")

    print_section("SELECTING VALIDATION SAMPLES")

    prediction_lookup = common.read_prediction_lookup(
        run_dir
    )

    selected_samples = common.select_samples(
        records=validation_records,
        predictions_lookup=prediction_lookup,
        samples_per_class=args.samples_per_class,
        strategy=args.sample_strategy,
        seed=args.seed,
    )

    print(
        f"\nTotal selected samples: {len(selected_samples)}"
    )

    if args.output_dir:

        output_dir = Path(args.output_dir).expanduser()

        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

        output_dir = output_dir.resolve()

    else:

        output_dir = (
            run_dir
            / "Xai"
            / "gradcam_plus_plus"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    directories = {
        name: output_dir / name
        for name in (
            "originals",
            "heatmaps",
            "overlays",
            "comparisons",
            "arrays",
        )
    }

    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    print_section("GENERATING GRAD-CAM++ EXPLANATIONS")

    results = []
    total_started = time.perf_counter()

    try:

        from tqdm.auto import tqdm

        iterator = tqdm(
            enumerate(selected_samples, start=1),
            total=len(selected_samples),
            desc="GRAD-CAM++",
            dynamic_ncols=True,
        )

    except ImportError:

        iterator = enumerate(
            selected_samples,
            start=1,
        )

    for index, record in iterator:

        sample_started = time.perf_counter()

        image_path = common.resolve_image_path(
            common.get_record_image_path(record)
        )

        actual_class = int(record["label"])
        actual_class_name = CLASS_NAMES[actual_class]

        with Image.open(image_path) as source:

            original_image = source.convert("RGB")
            display_original = original_image.copy()

            input_tensor = validation_transform(
                original_image
            ).unsqueeze(0).to(device)

        model.eval()

        with torch.no_grad():

            logits = common.extract_logits(
                model(input_tensor)
            )

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )[0]

        if probabilities.numel() != NUM_CLASSES:
            raise RuntimeError(
                "Prediction probability count is not seven."
            )

        if not torch.isfinite(probabilities).all():
            raise FloatingPointError(
                f"Non-finite prediction: {image_path}"
            )

        predicted_class = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[predicted_class].item()
        )

        correct = actual_class == predicted_class

        target_class = (
            predicted_class
            if args.target_class == "predicted"
            else actual_class
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        cam_started = time.perf_counter()

        cam, target_logit = compute_gradcam_plus_plus(
            model=model,
            target_layer=target_layer,
            reshape_transform=reshape_transform,
            input_tensor=input_tensor,
            target_class=target_class,
            eps=args.eps,
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        cam_runtime = time.perf_counter() - cam_started

        sample_id = f"sample_{index:04d}"

        sample_slug = (
            f"{sample_id}_"
            f"true-{common.safe_slug(actual_class_name)}_"
            f"pred-{common.safe_slug(CLASS_NAMES[predicted_class])}"
        )

        visual_files = save_visualization(
            original_image=display_original,
            cam=cam,
            sample_slug=sample_slug,
            directories=directories,
        )

        display_original.close()

        result = {

            "sample_id": sample_id,

            "image_path": str(image_path),

            "relative_path": str(record.get("relative_path", "")),

            "group_id": str(record.get("group_id", "")),

            "split": str(record.get("split", "validation")),

            "actual_class_index": actual_class,

            "actual_class": actual_class_name,

            "predicted_class_index": predicted_class,

            "predicted_class": CLASS_NAMES[predicted_class],

            "target_class_index": target_class,

            "target_class": CLASS_NAMES[target_class],

            "target_class_policy": args.target_class,

            "confidence": confidence,

            "target_logit": target_logit,

            "correct": bool(correct),

            "cam_mean": float(cam.mean()),

            "cam_std": float(cam.std()),

            "cam_max": float(cam.max()),

            "cam_active_fraction": float(np.mean(cam > 0.5)),

            "runtime_seconds": float(cam_runtime),

            "sample_runtime_seconds": float(
                time.perf_counter() - sample_started
            ),

            "target_layer": target_layer_name,

            **visual_files,
        }

        results.append(result)

        del input_tensor

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"\n[{index}/{len(selected_samples)}] "
            f"{actual_class_name} | "
            f"Predicted: {CLASS_NAMES[predicted_class]} | "
            f"Confidence: {confidence:.4f} | "
            f"{'CORRECT' if correct else 'INCORRECT'}",
            flush=True,
        )

    total_runtime = time.perf_counter() - total_started

    print_section("GENERATING GRAD-CAM++ REPORTS")

    correct_count = sum(
        bool(item["correct"])
        for item in results
    )

    incorrect_count = len(results) - correct_count

    class_summary = common.calculate_class_summary(
        results
    )

    mean_confidence = float(np.mean([
        item["confidence"] for item in results
    ]))

    mean_cam_intensity = float(np.mean([
        item["cam_mean"] for item in results
    ]))

    mean_runtime = float(np.mean([
        item["runtime_seconds"] for item in results
    ]))

    common.save_csv(
        results,
        output_dir / "gradcam_plus_plus_results.csv",
    )

    common.save_csv(
        class_summary,
        output_dir / "class_summary.csv",
    )

    selected_rows = []

    for index, record in enumerate(
        selected_samples,
        start=1,
    ):

        selected_rows.append({

            "sample_id": f"sample_{index:04d}",

            "image_path": str(
                common.resolve_image_path(
                    common.get_record_image_path(record)
                )
            ),

            "relative_path": str(
                record.get("relative_path", "")
            ),

            "group_id": str(record.get("group_id", "")),

            "split": str(record.get("split", "validation")),

            "actual_class": CLASS_NAMES[int(record["label"])],

            "actual_class_index": int(record["label"]),
        })

    common.save_csv(
        selected_rows,
        output_dir / "selected_samples.csv",
    )

    contact_sheet_path = None

    if not args.no_contact_sheet:

        contact_sheet_path = common.create_contact_sheet(
            results,
            output_dir / "gradcam_plus_plus_contact_sheet.jpg",
            columns=4,
        )

    summary = {

        "status": "COMPLETED",

        "method": "Grad-CAM++",

        "model_name": model_name,

        "run_name": run_dir.name,

        "checkpoint": str(checkpoint_path),

        "checkpoint_epoch": checkpoint.get("epoch"),

        "checkpoint_sha256": common.sha256_file(checkpoint_path),

        "partition": "validation",

        "validation_images_available": len(validation_records),

        "validation_manifest_only": True,

        "train_manifest_loaded": False,

        "held_out_test_manifest_loaded": False,

        "held_out_test_accessed": False,

        "validation_group_ids_available": group_ids_available,

        "group_level_leakage_verified": False,

        "sample_count": len(results),

        "samples_per_class_requested": args.samples_per_class,

        "sample_strategy": args.sample_strategy,

        "target_class_policy": args.target_class,

        "target_layer": target_layer_name,

        "class_names": list(CLASS_NAMES),

        "correct_predictions": correct_count,

        "incorrect_predictions": incorrect_count,

        "sample_accuracy": correct_count / len(results),

        "mean_confidence": mean_confidence,

        "mean_cam_intensity": mean_cam_intensity,

        "mean_runtime_seconds": mean_runtime,

        "total_runtime_seconds": total_runtime,

        "device": str(device),

        "gpu_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),

        "image_size": image_size,

        "seed": args.seed,

        "eps": args.eps,

        "split_fingerprint": split_fingerprint,

        "class_summary": class_summary,

        "contact_sheet": (
            str(contact_sheet_path)
            if contact_sheet_path else None
        ),

        "output_directory": str(output_dir),

        "held_out_test_used": False,

        "limitations": [
            "Grad-CAM++ is a post-hoc visual explanation method.",
            "Highlighted regions do not establish biological causality.",
            "Heatmap intensity is not a direct measure of disease severity.",
            "The balanced explanation subset is not a substitute for full validation metrics.",
            "Visual interpretation requires domain expertise.",
        ],
    }

    common.save_json(
        summary,
        output_dir / "gradcam_plus_plus_summary.json",
    )

    write_report(
        summary,
        class_summary,
        results,
        output_dir / "gradcam_plus_plus_report.txt",
    )

    print_section("GENERATING GRAD-CAM++ SUCCESSFUL")

    print(f"Model                 : {model_name}")
    print(f"Target layer          : {target_layer_name}")
    print(f"Validation images     : {len(validation_records)}")
    print(f"Samples explained     : {len(results)}")
    print(f"Correct predictions   : {correct_count}")
    print(f"Incorrect predictions : {incorrect_count}")
    print(f"Balanced-subset acc.  : {correct_count / len(results):.4f}")
    print(f"Mean confidence       : {mean_confidence:.4f}")
    print(f"Mean CAM intensity    : {mean_cam_intensity:.4f}")
    print(f"Mean CAM runtime      : {mean_runtime:.4f} seconds")
    print(f"Total runtime         : {total_runtime:.2f} seconds")
    print("Train manifest read   : NO")
    print("Test manifest read    : NO")
    print("Test set used         : NO")
    print(f"Output directory      : {output_dir}")
    print("=" * 76)

    return summary



def run_gradcam_plus_plus(args):
    """
    Run Grad-CAM++ with a minimal terminal interface.

    Visible terminal output:
      1. GENERATING GRAD-CAM++ EXPLANATIONS
      2. One tqdm progress bar for all selected images
      3. GENERATING GRAD-CAM++ SUCCESSFUL

    Per-image confidence lines, setup diagnostics, report-generation
    messages, and the detailed final statistics are suppressed.
    Files and reports are still generated normally. Exceptions are
    allowed to propagate so their traceback remains visible.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_gradcam_plus_plus_internal(args)


def build_parser():

    parser = argparse.ArgumentParser(
        description="Maize Leaf Classification - Grad-CAM++ Engine"
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        default="auto",
    )

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--sample-strategy",
        choices=[
            "balanced",
            "correct",
            "incorrect",
            "random",
        ],
        default="balanced",
    )

    parser.add_argument(
        "--target-class",
        choices=["predicted", "true"],
        default="predicted",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    if not np.isfinite(args.eps) or args.eps <= 0:
        parser.error("--eps must be positive and finite.")

    run_gradcam_plus_plus(args)


if __name__ == "__main__":
    main()
