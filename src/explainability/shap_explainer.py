
"""
MAIZE LEAF CLASSIFICATION
SHAP EXPLAINABILITY ENGINE

File:
src/explainability/shap_explainer.py

Method:
SHAP GradientExplainer (Expected Gradients)

Supported:
1. Custom CNN
2. ResNet50
3. EfficientNet-B0
4. MobileNetV3-Large
5. DenseNet121
6. ConvNeXt-Tiny
7. Small Vision Transformer

Features:
- Best checkpoint reconstruction
- Seven-class explanation
- Validation-only background dataset
- Reproducible class-balanced sampling
- Positive and negative attribution maps
- Absolute attribution maps
- RGB channel attribution
- Spatial region attribution
- SHAP completeness analysis
- Original images and overlays
- Raw SHAP array exports
- Contact sheet
- CSV, JSON and TXT reports

The held-out test partition is never accessed.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageOps

from src.explainability import gradcam as common


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = common.CLASS_NAMES
NUM_CLASSES = common.NUM_CLASSES
PROJECT_ROOT = common.PROJECT_ROOT


# ============================================================
# PRINT UTILITIES
# ============================================================

def print_section(title):

    print("\n" + "=" * 76)
    print(title)
    print("=" * 76, flush=True)


# ============================================================
# VALIDATION
# ============================================================

def validate_args(args):

    if args.samples_per_class < 1:
        raise ValueError("samples-per-class must be positive.")

    if args.nsamples < 10:
        raise ValueError("nsamples must be at least 10.")

    if args.background_per_class < 1:
        raise ValueError("background-per-class must be positive.")

    if args.explainer_batch_size < 1:
        raise ValueError("Explainer batch size must be positive.")

    if args.local_smoothing < 0:
        raise ValueError("Local smoothing cannot be negative.")

    if args.grid_size < 1:
        raise ValueError("Grid size must be positive.")


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path, transform, image_size):

    image_path = common.resolve_image_path(image_path)

    with Image.open(image_path) as source:

        image = ImageOps.exif_transpose(source).convert("RGB")

        image = image.resize(
            (image_size, image_size),
            Image.Resampling.BILINEAR,
        )

        original = np.asarray(
            image,
            dtype=np.uint8,
        ).copy()

        tensor = transform(image)

    if tensor.shape != (3, image_size, image_size):

        raise RuntimeError(
            f"Unexpected input tensor shape: {tensor.shape}"
        )

    if not torch.isfinite(tensor).all():

        raise FloatingPointError(
            "Non-finite input tensor."
        )

    return original, tensor


# ============================================================
# BACKGROUND SELECTION
# ============================================================

def select_background_records(
    validation_records,
    selected_samples,
    background_per_class,
    seed,
):

    excluded_paths = {

        common.normalize_path(
            common.resolve_image_path(record["image_path"])
        )

        for record in selected_samples
    }

    background_records = []

    print("\nSelecting SHAP background images...")

    for class_index, class_name in enumerate(CLASS_NAMES):

        candidates = [

            record

            for record in validation_records

            if int(record["label"]) == class_index

            and common.normalize_path(
                common.resolve_image_path(record["image_path"])
            ) not in excluded_paths

        ]

        rng = random.Random(
            seed + 50000 + class_index * 101
        )

        rng.shuffle(candidates)

        chosen = candidates[:background_per_class]

        if len(chosen) != background_per_class:

            raise RuntimeError(
                f"Insufficient independent validation "
                f"background images for {class_name}."
            )

        background_records.extend(
            dict(record) for record in chosen
        )

        print(
            f"[BACKGROUND] {class_name}: "
            f"{len(chosen)}/{background_per_class}"
        )

    return background_records


# ============================================================
# TARGET CLASS MODEL
# ============================================================

class TargetLogitModel(nn.Module):

    """
    Exposes one target-class logit as a [B, 1] output.

    SHAP explains the target-class logit rather
    than the softmax probability.
    """

    def __init__(self, model, target_class):

        super().__init__()

        self.model = model
        self.target_class = int(target_class)

    def forward(self, x):

        logits = common.extract_logits(self.model(x))

        if logits.shape[1] != NUM_CLASSES:

            raise RuntimeError(
                "Expected seven-class model output."
            )

        return logits[
            :,
            self.target_class:self.target_class + 1
        ]


# ============================================================
# SHAP VALUE NORMALIZATION
# ============================================================

def extract_shap_array(raw_values):

    if isinstance(raw_values, tuple):
        raw_values = raw_values[0]

    if isinstance(raw_values, list):

        if len(raw_values) != 1:

            raise RuntimeError(
                "Expected one SHAP output. "
                f"Received {len(raw_values)} outputs."
            )

        raw_values = raw_values[0]

    if torch.is_tensor(raw_values):

        raw_values = (
            raw_values.detach()
            .cpu()
            .numpy()
        )

    values = np.asarray(
        raw_values,
        dtype=np.float32,
    )

    values = np.squeeze(values)

    if values.ndim != 3:

        raise RuntimeError(
            "Unexpected SHAP array shape after "
            f"normalization: {values.shape}"
        )

    if values.shape[0] == 3:
        pass

    elif values.shape[-1] == 3:

        values = np.transpose(
            values,
            (2, 0, 1),
        )

    else:

        raise RuntimeError(
            "SHAP array must contain three RGB channels. "
            f"Received: {values.shape}"
        )

    if not np.isfinite(values).all():

        raise FloatingPointError(
            "SHAP values contain NaN or infinity."
        )

    return values.astype(np.float32)


# ============================================================
# SHAP ATTRIBUTION MAPS
# ============================================================

def calculate_attribution_maps(channel_values):

    signed_map = np.sum(
        channel_values,
        axis=0,
    )

    absolute_map = np.sum(
        np.abs(channel_values),
        axis=0,
    )

    maximum = float(np.max(np.abs(signed_map)))

    if maximum > 1e-12:

        normalized_signed = signed_map / maximum

    else:

        normalized_signed = np.zeros_like(signed_map)

    absolute_maximum = float(absolute_map.max())

    if absolute_maximum > 1e-12:

        normalized_absolute = absolute_map / absolute_maximum

    else:

        normalized_absolute = np.zeros_like(absolute_map)

    return (
        signed_map.astype(np.float32),
        absolute_map.astype(np.float32),
        normalized_signed.astype(np.float32),
        normalized_absolute.astype(np.float32),
    )


# ============================================================
# IMAGE UTILITIES
# ============================================================

def save_image(array, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        np.asarray(array, dtype=np.uint8)
    ).save(path)

    return str(path)


def blend_overlay(original, heatmap, intensity):

    original = original.astype(np.float32)
    heatmap = heatmap.astype(np.float32)

    intensity = np.asarray(
        intensity,
        dtype=np.float32,
    )

    intensity = np.clip(
        intensity,
        0,
        0.65,
    )[..., None]

    result = (
        original * (1.0 - intensity)
        + heatmap * intensity
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# SHAP VISUALIZATION
# ============================================================

def create_visualizations(
    original,
    channel_values,
    sample_slug,
    directories,
):

    (
        signed_map,
        absolute_map,
        normalized_signed,
        normalized_absolute,
    ) = calculate_attribution_maps(channel_values)

    height, width = signed_map.shape

    # --------------------------------------------------------
    # SIGNED HEATMAP
    # --------------------------------------------------------

    coolwarm = plt.get_cmap("coolwarm")

    signed_rgb = (
        coolwarm(
            (normalized_signed + 1.0) / 2.0
        )[..., :3] * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # ABSOLUTE IMPORTANCE MAP
    # --------------------------------------------------------

    magma = plt.get_cmap("magma")

    absolute_rgb = (
        magma(normalized_absolute)[..., :3] * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # COMBINED OVERLAY
    # --------------------------------------------------------

    overlay = blend_overlay(
        original,
        signed_rgb,
        0.55 * normalized_absolute,
    )

    # --------------------------------------------------------
    # POSITIVE AND NEGATIVE MAPS
    # --------------------------------------------------------

    positive_mask = signed_map > 0
    negative_mask = signed_map < 0

    red = np.zeros_like(original, dtype=np.uint8)
    red[..., 0] = 255

    blue = np.zeros_like(original, dtype=np.uint8)
    blue[..., 2] = 255

    positive_intensity = (
        0.55 * np.maximum(normalized_signed, 0)
    )

    negative_intensity = (
        0.55 * np.maximum(-normalized_signed, 0)
    )

    positive_overlay = blend_overlay(
        original,
        red,
        positive_intensity,
    )

    negative_overlay = blend_overlay(
        original,
        blue,
        negative_intensity,
    )

    # --------------------------------------------------------
    # MASKS
    # --------------------------------------------------------

    positive_mask_image = (
        positive_mask.astype(np.uint8) * 255
    )

    negative_mask_image = (
        negative_mask.astype(np.uint8) * 255
    )

    # --------------------------------------------------------
    # FILE PATHS
    #
    # IMPORTANT:
    # These keys MUST match the directories dictionary
    # created inside run_shap().
    # --------------------------------------------------------

    paths = {

        "original_file":
            directories["originals"]
            / f"{sample_slug}_original.png",

        "signed_heatmap_file":
            directories["signed_heatmaps"]
            / f"{sample_slug}_signed.png",

        "absolute_heatmap_file":
            directories["absolute_heatmaps"]
            / f"{sample_slug}_absolute.png",

        "overlay_file":
            directories["overlays"]
            / f"{sample_slug}_overlay.png",

        "positive_overlay_file":
            directories["positive"]
            / f"{sample_slug}_positive.png",

        "negative_overlay_file":
            directories["negative"]
            / f"{sample_slug}_negative.png",

        "positive_mask_file":
            directories["masks"]
            / f"{sample_slug}_positive_mask.png",

        "negative_mask_file":
            directories["masks"]
            / f"{sample_slug}_negative_mask.png",

        "attribution_array_file":
            directories["arrays"]
            / f"{sample_slug}_attributions.npz",

        "comparison_file":
            directories["comparisons"]
            / f"{sample_slug}_comparison.png",
    }

    # --------------------------------------------------------
    # SAVE IMAGES
    # --------------------------------------------------------

    save_image(original, paths["original_file"])

    save_image(
        signed_rgb,
        paths["signed_heatmap_file"],
    )

    save_image(
        absolute_rgb,
        paths["absolute_heatmap_file"],
    )

    save_image(
        overlay,
        paths["overlay_file"],
    )

    save_image(
        positive_overlay,
        paths["positive_overlay_file"],
    )

    save_image(
        negative_overlay,
        paths["negative_overlay_file"],
    )

    save_image(
        positive_mask_image,
        paths["positive_mask_file"],
    )

    save_image(
        negative_mask_image,
        paths["negative_mask_file"],
    )

    # --------------------------------------------------------
    # RAW ATTRIBUTION DATA
    # --------------------------------------------------------

    np.savez_compressed(

        paths["attribution_array_file"],

        channel_shap_values=channel_values,

        signed_pixel_attribution=signed_map,

        absolute_pixel_attribution=absolute_map,

        normalized_signed_map=normalized_signed,

        normalized_absolute_map=normalized_absolute,

        positive_mask=positive_mask,

        negative_mask=negative_mask,
    )

    # --------------------------------------------------------
    # COMPARISON PANEL
    # --------------------------------------------------------

    panels = [
        original,
        signed_rgb,
        absolute_rgb,
        overlay,
        positive_overlay,
        negative_overlay,
    ]

    labels = [
        "Original",
        "Signed SHAP",
        "Absolute SHAP",
        "Combined",
        "Positive",
        "Negative",
    ]

    panel_height = height + 35

    comparison = Image.new(
        "RGB",
        (
            width * len(panels),
            panel_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(comparison)

    for index, (panel, label) in enumerate(
        zip(panels, labels)
    ):

        panel_image = Image.fromarray(panel)

        comparison.paste(
            panel_image,
            (index * width, 0),
        )

        draw.text(
            (
                index * width + 8,
                height + 10,
            ),
            label,
            fill="black",
        )

        panel_image.close()

    comparison.save(paths["comparison_file"])
    comparison.close()

    return (
        {
            key: str(value)
            for key, value in paths.items()
        },
        signed_map,
        absolute_map,
        positive_mask,
        negative_mask,
    )


# ============================================================
# SPATIAL REGION ANALYSIS
# ============================================================

def calculate_region_importance(
    signed_map,
    absolute_map,
    sample_id,
    grid_size,
):

    height, width = signed_map.shape

    y_edges = np.linspace(
        0,
        height,
        grid_size + 1,
        dtype=int,
    )

    x_edges = np.linspace(
        0,
        width,
        grid_size + 1,
        dtype=int,
    )

    rows = []

    for row in range(grid_size):

        for column in range(grid_size):

            y0 = y_edges[row]
            y1 = y_edges[row + 1]

            x0 = x_edges[column]
            x1 = x_edges[column + 1]

            signed_region = signed_map[
                y0:y1,
                x0:x1,
            ]

            absolute_region = absolute_map[
                y0:y1,
                x0:x1,
            ]

            if signed_region.size == 0:
                continue

            rows.append({

                "sample_id": sample_id,

                "grid_row": row,

                "grid_column": column,

                "x_start": x0,

                "x_end": x1,

                "y_start": y0,

                "y_end": y1,

                "mean_signed_attribution": float(
                    signed_region.mean()
                ),

                "mean_absolute_attribution": float(
                    absolute_region.mean()
                ),

                "sum_absolute_attribution": float(
                    absolute_region.sum()
                ),

                "positive_pixel_fraction": float(
                    np.mean(signed_region > 0)
                ),

                "negative_pixel_fraction": float(
                    np.mean(signed_region < 0)
                ),

            })

    return rows


# ============================================================
# CHANNEL ANALYSIS
# ============================================================

def calculate_channel_importance(
    channel_values,
    sample_id,
):

    rows = []

    channel_names = [
        "Red",
        "Green",
        "Blue",
    ]

    for index, channel_name in enumerate(channel_names):

        values = channel_values[index]

        rows.append({

            "sample_id": sample_id,

            "channel": channel_name,

            "mean_signed_attribution": float(
                values.mean()
            ),

            "mean_absolute_attribution": float(
                np.abs(values).mean()
            ),

            "sum_absolute_attribution": float(
                np.abs(values).sum()
            ),

        })

    return rows


# ============================================================
# CLASS SUMMARY
# ============================================================

def calculate_class_summary(results):

    summaries = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        rows = [

            item

            for item in results

            if item["actual_class_index"] == class_index

        ]

        correct = [
            item for item in rows
            if item["correct"]
        ]

        summaries.append({

            "class_index": class_index,

            "class": class_name,

            "selected_images": len(rows),

            "correct_predictions": len(correct),

            "incorrect_predictions": (
                len(rows) - len(correct)
            ),

            "sample_accuracy": (
                len(correct) / len(rows)
                if rows else None
            ),

            "mean_confidence": (
                float(np.mean([
                    item["confidence"]
                    for item in rows
                ]))
                if rows else None
            ),

            "mean_absolute_attribution": (
                float(np.mean([
                    item["mean_absolute_attribution"]
                    for item in rows
                ]))
                if rows else None
            ),

            "mean_completeness_error": (
                float(np.mean([
                    item["absolute_completeness_error"]
                    for item in rows
                ]))
                if rows else None
            ),

            "mean_runtime_seconds": (
                float(np.mean([
                    item["runtime_seconds"]
                    for item in rows
                ]))
                if rows else None
            ),

        })

    return summaries


# ============================================================
# TEXT REPORT
# ============================================================

def write_report(
    summary,
    class_summary,
    output_path,
):

    lines = [

        "=" * 78,

        "MAIZE LEAF CLASSIFICATION - SHAP REPORT",

        "=" * 78,

        "",

        f"Model: {summary['model_name']}",

        f"Run: {summary['run_name']}",

        f"Method: {summary['method']}",

        f"Partition: {summary['partition']}",

        f"Checkpoint: {summary['checkpoint']}",

        "",

        "SHAP CONFIGURATION",

        "-" * 78,

        f"Algorithm: {summary['algorithm']}",

        f"Samples per class: {summary['samples_per_class']}",

        f"Background images: {summary['background_count']}",

        f"Expected gradient samples: {summary['nsamples']}",

        f"Explainer batch size: {summary['explainer_batch_size']}",

        f"Local smoothing: {summary['local_smoothing']}",

        f"Target policy: {summary['target_class_policy']}",

        "",

        "OVERALL RESULTS",

        "-" * 78,

        f"Samples explained: {summary['sample_count']}",

        f"Correct: {summary['correct_predictions']}",

        f"Incorrect: {summary['incorrect_predictions']}",

        f"Subset accuracy: {summary['sample_accuracy']:.6f}",

        f"Mean confidence: {summary['mean_confidence']:.6f}",

        f"Mean absolute attribution: "
        f"{summary['mean_absolute_attribution']:.8f}",

        f"Mean absolute completeness error: "
        f"{summary['mean_completeness_error']:.8f}",

        f"Mean runtime: "
        f"{summary['mean_runtime_seconds']:.4f} seconds",

        f"Total runtime: "
        f"{summary['total_runtime_seconds']:.2f} seconds",

        "",

        "CLASS-WISE RESULTS",

        "-" * 78,

    ]

    for item in class_summary:

        lines.extend([

            "",

            item["class"],

            f"  Selected: {item['selected_images']}",

            f"  Correct: {item['correct_predictions']}",

            f"  Incorrect: {item['incorrect_predictions']}",

            f"  Accuracy: {item['sample_accuracy']}",

            f"  Mean absolute attribution: "
            f"{item['mean_absolute_attribution']}",

            f"  Mean completeness error: "
            f"{item['mean_completeness_error']}",

        ])

    lines.extend([

        "",

        "DATA INTEGRITY",

        "-" * 78,

        f"Split fingerprint: {summary['split_fingerprint']}",

        f"Seed: {summary['seed']}",

        "Validation manifest only: YES",

        "Training manifest accessed: NO",

        "Test manifest accessed: NO",

        "Test partition used: NO",

        "",

        "INTERPRETATION",

        "-" * 78,

        "Positive attribution increases the selected class logit",

        "relative to the background reference distribution.",

        "Negative attribution decreases the selected class logit.",

        "Absolute attribution represents contribution magnitude.",

        "Completeness error measures the difference between",

        "the output difference and the sum of estimated SHAP values.",

        "SHAP attribution does not establish biological causality.",

        "SHAP importance does not measure disease severity.",

        "Results depend on the selected background distribution.",

        "",

        "=" * 78,

    ])

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# MAIN SHAP ENGINE
# ============================================================

def run_shap(args):

    validate_args(args)

    try:

        import shap

    except ImportError as error:

        raise ImportError(
            "SHAP is not installed. Run: "
            "python -m pip install shap"
        ) from error

    run_dir = Path(
        args.run_dir
    ).expanduser().resolve()

    if not run_dir.is_dir():

        raise NotADirectoryError(
            f"Run directory not found: {run_dir}"
        )

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(args.device)

    if device.type == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        if args.device == "cuda":

            device = torch.device("cuda:0")

    # --------------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------------

    print_section(
        "MAIZE LEAF - SHAP EXPLAINABILITY ENGINE"
    )

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Run directory: {run_dir}")
    print(f"Device       : {device}")
    print(f"SHAP version : {shap.__version__}")

    if device.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(device)}"
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

    # --------------------------------------------------------
    # SPLIT VERIFICATION
    # --------------------------------------------------------

    print_section(
        "VERIFYING FIXED DATASET SPLIT"
    )

    split_fingerprint = common.verify_split(run_dir)

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    print_section(
        "LOADING TRAINING CONFIGURATION"
    )

    config, config_data, model_name = (
        common.load_training_configuration(run_dir)
    )

    image_size = int(config.image_size)

    print(f"Model      : {model_name}")
    print(f"Image size : {image_size}")
    print(f"Classes    : {NUM_CLASSES}")

    for index, name in enumerate(CLASS_NAMES):

        print(f"{index}: {name}")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print_section(
        "LOADING BEST MODEL"
    )

    model, checkpoint, checkpoint_path = (
        common.load_best_model(
            run_dir,
            config,
            model_name,
            device,
        )
    )

    model.eval()

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    print_section(
        "BUILDING VALIDATION DATA"
    )

    validation_records = (
        common.load_validation_records()
    )

    validation_transform, image_size = (
        common.build_validation_transform(config)
    )

    print(
        f"[PASS] Validation records: {len(validation_records)}"
    )

    print("[PASS] Fixed seven-class mapping")
    print("[PASS] Evaluation preprocessing")
    print("[PASS] Test partition not accessed")

    # --------------------------------------------------------
    # SAMPLE SELECTION
    # --------------------------------------------------------

    print_section(
        "SELECTING EXPLANATION SAMPLES"
    )

    prediction_lookup = (
        common.read_prediction_lookup(run_dir)
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

    if not selected_samples:

        raise RuntimeError(
            "No explanation samples were selected."
        )

    # --------------------------------------------------------
    # BACKGROUND DATA
    # --------------------------------------------------------

    print_section(
        "BUILDING VALIDATION BACKGROUND"
    )

    background_records = select_background_records(

        validation_records=validation_records,

        selected_samples=selected_samples,

        background_per_class=args.background_per_class,

        seed=args.seed,

    )

    background_tensors = []
    background_rows = []

    for index, record in enumerate(
        background_records,
        start=1,
    ):

        path = common.resolve_image_path(
            record["image_path"]
        )

        _, tensor = load_image(
            path,
            validation_transform,
            image_size,
        )

        background_tensors.append(tensor)

        background_rows.append({

            "background_id": f"background_{index:03d}",

            "image_path": str(path),

            "class_index": int(record["label"]),

            "class_name": CLASS_NAMES[
                int(record["label"])
            ],

            "split": "validation",

        })

    background_tensor = torch.stack(
        background_tensors,
        dim=0,
    ).to(device)

    print(
        f"[PASS] Background images: {len(background_tensor)}"
    )

    print(
        f"[PASS] Background tensor: "
        f"{tuple(background_tensor.shape)}"
    )

    print(
        "[PASS] Background references use validation partition only"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    if args.output_dir:

        output_dir = Path(
            args.output_dir
        ).expanduser()

        if not output_dir.is_absolute():

            output_dir = PROJECT_ROOT / output_dir

        output_dir = output_dir.resolve()

    else:

        output_dir = (
            run_dir
            / "Xai"
            / "shap_gradient"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # DIRECTORY STRUCTURE
    # --------------------------------------------------------

    directories = {

        name: output_dir / name

        for name in (

            "originals",

            "signed_heatmaps",

            "absolute_heatmaps",

            "overlays",

            "positive",

            "negative",

            "masks",

            "arrays",

            "comparisons",

        )

    }

    for directory in directories.values():

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    print("[PASS] SHAP output directories initialized")

    # --------------------------------------------------------
    # SHAP EXPLAINER CACHE
    # --------------------------------------------------------

    print_section(
        "INITIALIZING SHAP GRADIENT EXPLAINER"
    )

    print("Algorithm: Expected Gradients")

    print(
        f"Background images: {len(background_records)}"
    )

    print(
        f"Samples per explanation: {args.nsamples}"
    )

    print(
        f"Explainer batch size: {args.explainer_batch_size}"
    )

    print(
        "[INFO] One target-class explainer is cached per class."
    )

    explainer_cache = {}
    expected_value_cache = {}

    def get_explainer(target_class):

        target_class = int(target_class)

        if target_class not in explainer_cache:

            target_model = TargetLogitModel(

                model=model,

                target_class=target_class,

            ).to(device)

            target_model.eval()

            explainer = shap.GradientExplainer(

                target_model,

                background_tensor,

                batch_size=args.explainer_batch_size,

                local_smoothing=args.local_smoothing,

            )

            with torch.no_grad():

                baseline_outputs = target_model(
                    background_tensor
                )

                expected_value = float(
                    baseline_outputs.mean().item()
                )

            if not np.isfinite(expected_value):

                raise FloatingPointError(
                    "Invalid SHAP background expected value."
                )

            explainer_cache[target_class] = (
                target_model,
                explainer,
            )

            expected_value_cache[target_class] = (
                expected_value
            )

            print(
                f"[PASS] Explainer initialized: "
                f"{CLASS_NAMES[target_class]}"
            )

        return (

            explainer_cache[target_class][0],

            explainer_cache[target_class][1],

            expected_value_cache[target_class],

        )

    # --------------------------------------------------------
    # EXPLANATION GENERATION
    # --------------------------------------------------------

    print_section(
        "GENERATING SHAP EXPLANATIONS"
    )

    from tqdm.auto import tqdm

    results = []
    region_rows = []
    channel_rows = []

    started = time.perf_counter()

    for index, record in tqdm(

        enumerate(
            selected_samples,
            start=1,
        ),

        total=len(selected_samples),

        desc="SHAP",

        dynamic_ncols=True,

    ):

        sample_started = time.perf_counter()

        image_path = common.resolve_image_path(
            record["image_path"]
        )

        actual_class = int(record["label"])

        actual_class_name = CLASS_NAMES[actual_class]

        original, input_tensor = load_image(

            image_path,

            validation_transform,

            image_size,

        )

        input_tensor = input_tensor.unsqueeze(
            0
        ).to(device)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            logits = common.extract_logits(
                model(input_tensor)
            )

            probabilities = F.softmax(
                logits.float(),
                dim=1,
            )[0]

        if probabilities.numel() != NUM_CLASSES:

            raise RuntimeError(
                "Model output does not contain seven classes."
            )

        if not torch.isfinite(probabilities).all():

            raise FloatingPointError(
                "Invalid model probabilities."
            )

        predicted_class = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[predicted_class].item()
        )

        correct = (
            actual_class == predicted_class
        )

        target_class = (

            predicted_class

            if args.target_class == "predicted"

            else actual_class

        )

        target_logit = float(
            logits[0, target_class].item()
        )

        target_probability = float(
            probabilities[target_class].item()
        )

        # ----------------------------------------------------
        # TARGET EXPLAINER
        # ----------------------------------------------------

        target_model, explainer, expected_value = (
            get_explainer(target_class)
        )

        # ----------------------------------------------------
        # COMPUTE SHAP
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize(device)

        explanation_started = time.perf_counter()

        raw_values = explainer.shap_values(

            input_tensor,

            nsamples=args.nsamples,

            rseed=args.seed + index,

        )

        if device.type == "cuda":

            torch.cuda.synchronize(device)

        explanation_runtime = (
            time.perf_counter() - explanation_started
        )

        channel_values = extract_shap_array(raw_values)

        if channel_values.shape != (
            3,
            image_size,
            image_size,
        ):

            raise RuntimeError(
                "Unexpected SHAP spatial dimensions: "
                f"{channel_values.shape}"
            )

        # ----------------------------------------------------
        # ATTRIBUTION MAPS
        # ----------------------------------------------------

        (
            signed_map,
            absolute_map,
            normalized_signed,
            normalized_absolute,
        ) = calculate_attribution_maps(channel_values)

        # ----------------------------------------------------
        # COMPLETENESS ANALYSIS
        # ----------------------------------------------------

        shap_sum = float(channel_values.sum())

        output_difference = (
            target_logit - expected_value
        )

        completeness_error = (
            output_difference - shap_sum
        )

        absolute_completeness_error = abs(
            completeness_error
        )

        relative_completeness_error = (

            absolute_completeness_error

            / max(
                abs(output_difference),
                1e-8,
            )

        )

        mean_absolute_attribution = float(
            np.abs(channel_values).mean()
        )

        sum_absolute_attribution = float(
            np.abs(channel_values).sum()
        )

        mean_signed_pixel_attribution = float(
            signed_map.mean()
        )

        mean_absolute_pixel_attribution = float(
            absolute_map.mean()
        )

        positive_fraction = float(
            np.mean(signed_map > 0)
        )

        negative_fraction = float(
            np.mean(signed_map < 0)
        )

        # ----------------------------------------------------
        # SAMPLE IDENTIFICATION
        # ----------------------------------------------------

        sample_id = f"sample_{index:04d}"

        sample_slug = (

            f"{sample_id}_"

            f"true-{common.safe_slug(actual_class_name)}_"

            f"pred-{common.safe_slug(CLASS_NAMES[predicted_class])}"

        )

        # ----------------------------------------------------
        # VISUALIZATION
        # ----------------------------------------------------

        (
            visual_files,
            signed_map,
            absolute_map,
            positive_mask,
            negative_mask,
        ) = create_visualizations(

            original=original,

            channel_values=channel_values,

            sample_slug=sample_slug,

            directories=directories,

        )

        # ----------------------------------------------------
        # SPATIAL REGION ANALYSIS
        # ----------------------------------------------------

        sample_region_rows = calculate_region_importance(

            signed_map=signed_map,

            absolute_map=absolute_map,

            sample_id=sample_id,

            grid_size=args.grid_size,

        )

        for row in sample_region_rows:

            row["actual_class"] = actual_class_name

            row["predicted_class"] = CLASS_NAMES[
                predicted_class
            ]

            row["target_class"] = CLASS_NAMES[
                target_class
            ]

        region_rows.extend(sample_region_rows)

        # ----------------------------------------------------
        # RGB CHANNEL ANALYSIS
        # ----------------------------------------------------

        sample_channel_rows = calculate_channel_importance(

            channel_values,

            sample_id,

        )

        for row in sample_channel_rows:

            row["actual_class"] = actual_class_name

            row["target_class"] = CLASS_NAMES[
                target_class
            ]

        channel_rows.extend(sample_channel_rows)

        # ----------------------------------------------------
        # RESULT RECORD
        # ----------------------------------------------------

        result = {

            "sample_id": sample_id,

            "image_path": str(image_path),

            "relative_path": str(
                record.get("relative_path", "")
            ),

            "group_id": str(
                record.get("group_id", "")
            ),

            "split": "validation",

            "actual_class_index": actual_class,

            "actual_class": actual_class_name,

            "predicted_class_index": predicted_class,

            "predicted_class": CLASS_NAMES[
                predicted_class
            ],

            "target_class_index": target_class,

            "target_class": CLASS_NAMES[
                target_class
            ],

            "target_class_policy": args.target_class,

            "confidence": confidence,

            "target_probability": target_probability,

            "target_logit": target_logit,

            "background_expected_value": expected_value,

            "output_difference": output_difference,

            "shap_sum": shap_sum,

            "completeness_error": completeness_error,

            "absolute_completeness_error": (
                absolute_completeness_error
            ),

            "relative_completeness_error": (
                relative_completeness_error
            ),

            "mean_absolute_attribution": (
                mean_absolute_attribution
            ),

            "sum_absolute_attribution": (
                sum_absolute_attribution
            ),

            "mean_signed_pixel_attribution": (
                mean_signed_pixel_attribution
            ),

            "mean_absolute_pixel_attribution": (
                mean_absolute_pixel_attribution
            ),

            "positive_pixel_fraction": positive_fraction,

            "negative_pixel_fraction": negative_fraction,

            "positive_pixel_count": int(
                positive_mask.sum()
            ),

            "negative_pixel_count": int(
                negative_mask.sum()
            ),

            "runtime_seconds": float(
                explanation_runtime
            ),

            "sample_runtime_seconds": float(
                time.perf_counter() - sample_started
            ),

            "correct": bool(correct),

            **visual_files,

        }

        results.append(result)

        print(

            f"\n[{index}/{len(selected_samples)}] "

            f"{actual_class_name} | "

            f"Predicted: {CLASS_NAMES[predicted_class]} | "

            f"Confidence: {confidence:.4f} | "

            f"{'CORRECT' if correct else 'INCORRECT'}",

            flush=True,

        )

        print(

            f"    Target logit: {target_logit:.5f} | "

            f"SHAP sum: {shap_sum:.5f} | "

            f"Completeness error: {completeness_error:.5f} | "

            f"Time: {explanation_runtime:.2f}s",

            flush=True,

        )

        del input_tensor

        if device.type == "cuda":

            torch.cuda.empty_cache()

    total_runtime = (
        time.perf_counter() - started
    )

    # --------------------------------------------------------
    # REPORT GENERATION
    # --------------------------------------------------------

    print_section(
        "GENERATING SHAP REPORTS"
    )

    if not results:

        raise RuntimeError(
            "No SHAP explanations were generated."
        )

    correct_count = sum(
        item["correct"]
        for item in results
    )

    incorrect_count = (
        len(results) - correct_count
    )

    mean_confidence = float(
        np.mean([
            item["confidence"]
            for item in results
        ])
    )

    mean_absolute_attribution = float(
        np.mean([
            item["mean_absolute_attribution"]
            for item in results
        ])
    )

    mean_completeness_error = float(
        np.mean([
            item["absolute_completeness_error"]
            for item in results
        ])
    )

    mean_runtime = float(
        np.mean([
            item["runtime_seconds"]
            for item in results
        ])
    )

    class_summary = calculate_class_summary(results)

    # --------------------------------------------------------
    # CSV FILES
    # --------------------------------------------------------

    common.save_csv(
        results,
        output_dir / "shap_results.csv",
    )

    print("[PASS] shap_results.csv")

    common.save_csv(
        region_rows,
        output_dir / "shap_regions.csv",
    )

    print("[PASS] shap_regions.csv")

    common.save_csv(
        channel_rows,
        output_dir / "shap_channel_summary.csv",
    )

    print("[PASS] shap_channel_summary.csv")

    common.save_csv(
        class_summary,
        output_dir / "class_summary.csv",
    )

    print("[PASS] class_summary.csv")

    # --------------------------------------------------------
    # SELECTED SAMPLE RECORDS
    # --------------------------------------------------------

    selected_rows = []

    for index, record in enumerate(
        selected_samples,
        start=1,
    ):

        selected_rows.append({

            "sample_id": f"sample_{index:04d}",

            "image_path": str(
                common.resolve_image_path(
                    record["image_path"]
                )
            ),

            "actual_class": CLASS_NAMES[
                int(record["label"])
            ],

            "actual_class_index": int(
                record["label"]
            ),

            "split": "validation",

        })

    common.save_csv(
        selected_rows,
        output_dir / "selected_samples.csv",
    )

    print("[PASS] selected_samples.csv")

    common.save_csv(
        background_rows,
        output_dir / "background_samples.csv",
    )

    print("[PASS] background_samples.csv")

    # --------------------------------------------------------
    # CONTACT SHEET
    # --------------------------------------------------------

    contact_sheet_path = None

    if not args.no_contact_sheet:

        contact_sheet_path = common.create_contact_sheet(

            results,

            output_dir / "shap_contact_sheet.jpg",

            columns=4,

        )

        if contact_sheet_path:

            print("[PASS] shap_contact_sheet.jpg")

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    summary = {

        "status": "COMPLETED",

        "method": "SHAP",

        "algorithm": "GradientExplainer - Expected Gradients",

        "model_name": model_name,

        "run_name": run_dir.name,

        "checkpoint": str(checkpoint_path),

        "checkpoint_epoch": checkpoint.get("epoch"),

        "checkpoint_sha256": common.sha256_file(
            checkpoint_path
        ),

        "partition": "validation",

        "validation_images_available": len(
            validation_records
        ),

        "sample_count": len(results),

        "samples_per_class": args.samples_per_class,

        "sample_strategy": args.sample_strategy,

        "target_class_policy": args.target_class,

        "background_count": len(background_records),

        "background_per_class": (
            args.background_per_class
        ),

        "background_policy": (
            "Class-balanced independent validation images; "
            "selected explanation images excluded."
        ),

        "nsamples": args.nsamples,

        "explainer_batch_size": (
            args.explainer_batch_size
        ),

        "local_smoothing": args.local_smoothing,

        "grid_size": args.grid_size,

        "image_size": image_size,

        "class_names": list(CLASS_NAMES),

        "correct_predictions": correct_count,

        "incorrect_predictions": incorrect_count,

        "sample_accuracy": (
            correct_count / len(results)
        ),

        "mean_confidence": mean_confidence,

        "mean_absolute_attribution": (
            mean_absolute_attribution
        ),

        "mean_completeness_error": (
            mean_completeness_error
        ),

        "mean_runtime_seconds": mean_runtime,

        "total_runtime_seconds": total_runtime,

        "device": str(device),

        "gpu_name": (

            torch.cuda.get_device_name(device)

            if device.type == "cuda"

            else None

        ),

        "seed": args.seed,

        "split_fingerprint": split_fingerprint,

        "shap_version": shap.__version__,

        "torch_version": torch.__version__,

        "background_samples": background_rows,

        "class_summary": class_summary,

        "contact_sheet": contact_sheet_path,

        "output_directory": str(output_dir),

        "validation_manifest_only": True,

        "train_manifest_loaded": False,

        "test_manifest_loaded": False,

        "held_out_test_used": False,

        "limitations": [

            "GradientExplainer uses expected gradients.",

            "SHAP values are approximate.",

            "Results depend on the background distribution.",

            "The model output explained is the target-class logit.",

            "Positive attribution increases the selected class logit.",

            "Negative attribution decreases the selected class logit.",

            "Completeness error is an approximation diagnostic.",

            "Attributions do not establish biological causality.",

            "Attribution magnitude does not measure disease severity.",

        ],

    }

    common.save_json(
        summary,
        output_dir / "shap_summary.json",
    )

    print("[PASS] shap_summary.json")

    write_report(

        summary,

        class_summary,

        output_dir / "shap_report.txt",

    )

    print("[PASS] shap_report.txt")

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print_section(
        "SHAP COMPLETED SUCCESSFULLY"
    )

    print(f"Model                 : {model_name}")

    print(
        f"Validation images     : {len(validation_records)}"
    )

    print(
        f"Background images     : {len(background_records)}"
    )

    print(
        f"Samples explained     : {len(results)}"
    )

    print(
        f"Correct predictions   : {correct_count}"
    )

    print(
        f"Incorrect predictions : {incorrect_count}"
    )

    print(
        f"Subset accuracy       : "
        f"{correct_count / len(results):.4f}"
    )

    print(
        f"Mean confidence       : {mean_confidence:.4f}"
    )

    print(
        f"Mean absolute SHAP    : "
        f"{mean_absolute_attribution:.8f}"
    )

    print(
        f"Mean completeness err : "
        f"{mean_completeness_error:.8f}"
    )

    print(
        f"Mean SHAP runtime     : "
        f"{mean_runtime:.4f} seconds"
    )

    print(
        f"Total runtime         : "
        f"{total_runtime:.2f} seconds"
    )

    print("Train manifest read   : NO")
    print("Test manifest read    : NO")
    print("Test set used         : NO")

    print(
        f"Output directory      : {output_dir}"
    )

    print("=" * 76)

    return summary


# ============================================================
# COMMAND LINE
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(

        description=(
            "Maize Leaf Classification - SHAP "
            "GradientExplainer Engine"
        )

    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
            "cuda:0",
        ],
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
        choices=[
            "predicted",
            "true",
        ],
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
        "--nsamples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--background-per-class",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--explainer-batch-size",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--local-smoothing",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=14,
    )

    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
    )

    return parser


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    parser = build_parser()

    args = parser.parse_args()

    try:

        run_shap(args)

    except KeyboardInterrupt:

        print(
            "\n[SHAP] Interrupted by user."
        )

        raise

    except Exception as error:

        print(
            f"\n[SHAP ERROR] "
            f"{type(error).__name__}: {error}"
        )

        raise


if __name__ == "__main__":

    main()
