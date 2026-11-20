
"""
MAIZE LEAF CLASSIFICATION
LIME EXPLAINABILITY ENGINE

File:
src/explainability/lime_explainer.py

Supports all seven project architectures.

Uses:
- Fixed validation manifest only
- Existing best checkpoint
- Existing preprocessing
- Existing prediction records
- Reproducible class-balanced sampling

Exports:
- Original images
- Superpixel segmentation
- Positive feature overlays
- Negative feature overlays
- Combined explanation overlays
- Signed feature-weight maps
- Raw segmentation and weight arrays
- Per-feature CSV
- Per-image CSV
- Class summary
- JSON metadata
- TXT report
- Contact sheet

The held-out test partition is never accessed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import io
import random
import sys
import time
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic, mark_boundaries

try:
    from tqdm.auto import tqdm as progress_tqdm
except ImportError:
    progress_tqdm = None

from src.explainability import gradcam as common


CLASS_NAMES = common.CLASS_NAMES
NUM_CLASSES = common.NUM_CLASSES
PROJECT_ROOT = common.PROJECT_ROOT


# ============================================================
# GENERAL UTILITIES
# ============================================================

def print_section(title):

    visible_sections = {
        "GENERATING LIME EXPLANATIONS",
        "GENERATING LIME SUCCESSFUL",
    }

    if title not in visible_sections:
        return

    # Use the real terminal stream so these headings remain visible
    # while routine setup/report messages are redirected.
    stream = sys.__stdout__

    stream.write("\n" + "=" * 76 + "\n")
    stream.write(title + "\n")
    stream.write("=" * 76 + "\n")
    stream.flush()


def get_version(package):

    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def explain_instance_without_internal_progress(explainer, **kwargs):
    """
    Run LIME without its per-perturbation tqdm bar.

    Newer LIME releases expose progress_bar=False directly.
    The installed project version does not, so for that version
    the tqdm symbol in lime_image.py is temporarily replaced by
    an identity wrapper. The original symbol is always restored.
    """

    parameters = inspect.signature(
        explainer.explain_instance
    ).parameters

    if "progress_bar" in parameters:
        kwargs["progress_bar"] = False
        return explainer.explain_instance(**kwargs)

    bound_method = explainer.explain_instance
    function = getattr(bound_method, "__func__", bound_method)
    function_globals = getattr(function, "__globals__", {})

    original_tqdm = function_globals.get("tqdm")

    # If this LIME build does not use a module-level tqdm,
    # simply call the explainer normally.
    if original_tqdm is None:
        return explainer.explain_instance(**kwargs)

    def identity_progress(iterable, *args, **inner_kwargs):
        return iterable

    function_globals["tqdm"] = identity_progress

    try:
        return explainer.explain_instance(**kwargs)
    finally:
        function_globals["tqdm"] = original_tqdm


def validate_arguments(args):

    if args.samples_per_class < 1:
        raise ValueError(
            "samples-per-class must be at least 1."
        )

    if args.num_samples < 100:
        raise ValueError(
            "LIME num-samples must be at least 100."
        )

    if args.num_features < 1:
        raise ValueError(
            "num-features must be positive."
        )

    if args.num_segments < 2:
        raise ValueError(
            "num-segments must be at least 2."
        )

    if args.compactness <= 0:
        raise ValueError(
            "SLIC compactness must be positive."
        )

    if args.sigma < 0:
        raise ValueError(
            "SLIC sigma cannot be negative."
        )

    if args.prediction_batch_size < 1:
        raise ValueError(
            "Prediction batch size must be positive."
        )

    if args.hide_color is not None:
        if not 0 <= args.hide_color <= 255:
            raise ValueError(
                "Hide color must be between 0 and 255."
            )

    if args.seed < 0:
        raise ValueError(
            "Seed must be non-negative."
        )


def to_uint8_rgb(image):

    image = np.asarray(image)

    if image.ndim == 2:

        image = np.stack(
            [image, image, image],
            axis=-1,
        )

    if image.ndim != 3:
        raise ValueError(
            f"Invalid image dimensions: {image.shape}"
        )

    if image.shape[-1] == 1:

        image = np.repeat(
            image,
            3,
            axis=-1,
        )

    elif image.shape[-1] == 4:

        image = image[..., :3]

    if image.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB image: {image.shape}"
        )

    if not np.isfinite(image).all():
        raise ValueError(
            "Image contains non-finite values."
        )

    if np.issubdtype(image.dtype, np.floating):

        image = image.astype(np.float32)

        if image.min() >= 0 and image.max() <= 1:

            image = image * 255.0

    return np.ascontiguousarray(
        np.clip(image, 0, 255).astype(np.uint8)
    )


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


# ============================================================
# MODEL-AGNOSTIC PREDICTION FUNCTION
# ============================================================

def build_classifier_fn(
    model,
    transform,
    device,
    prediction_batch_size,
):
    """
    LIME supplies perturbed RGB images.

    This function applies the exact evaluation
    preprocessing used by the project and returns
    seven-class probability distributions.
    """

    def classifier_fn(images):

        if isinstance(images, np.ndarray):

            if images.ndim == 3:

                image_list = [images]

            elif images.ndim == 4:

                image_list = [
                    images[index]
                    for index in range(len(images))
                ]

            else:

                raise ValueError(
                    f"Invalid LIME batch shape: {images.shape}"
                )

        else:

            image_list = list(images)

        if not image_list:

            return np.empty(
                (0, NUM_CLASSES),
                dtype=np.float32,
            )

        all_probabilities = []

        model.eval()

        for start in range(
            0,
            len(image_list),
            prediction_batch_size,
        ):

            batch_images = image_list[
                start:
                start + prediction_batch_size
            ]

            tensors = []

            for image_array in batch_images:

                image_array = to_uint8_rgb(
                    image_array
                )

                with Image.fromarray(
                    image_array
                ) as pil_image:

                    tensor = transform(
                        pil_image
                    )

                tensors.append(tensor)

            batch_tensor = torch.stack(
                tensors,
                dim=0,
            ).to(device)

            with torch.inference_mode():

                logits = common.extract_logits(
                    model(batch_tensor)
                )

                if logits.shape[1] != NUM_CLASSES:

                    raise RuntimeError(
                        "Model output does not contain "
                        "exactly seven classes."
                    )

                probabilities = F.softmax(
                    logits.float(),
                    dim=1,
                )

            probabilities = probabilities.cpu().numpy()

            if not np.isfinite(probabilities).all():

                raise FloatingPointError(
                    "Non-finite prediction probabilities."
                )

            if not np.allclose(
                probabilities.sum(axis=1),
                1.0,
                atol=1e-4,
            ):

                raise RuntimeError(
                    "Prediction probabilities do not sum to one."
                )

            all_probabilities.append(
                probabilities
            )

            del batch_tensor

        return np.concatenate(
            all_probabilities,
            axis=0,
        )

    return classifier_fn


# ============================================================
# SLIC SEGMENTATION
# ============================================================

def build_segmentation_function(args):

    def segmentation_function(image):

        image = to_uint8_rgb(image)

        segments = slic(
            image,
            n_segments=args.num_segments,
            compactness=args.compactness,
            sigma=args.sigma,
            start_label=0,
            channel_axis=-1,
            convert2lab=True,
            enforce_connectivity=True,
        )

        return segments.astype(np.int32)

    return segmentation_function


# ============================================================
# LIME VISUALIZATION
# ============================================================

def create_lime_visualizations(
    image,
    segments,
    local_weights,
    sample_slug,
    directories,
):

    image = to_uint8_rgb(image)

    height, width = image.shape[:2]

    positive_features = [
        (segment_id, weight)
        for segment_id, weight in local_weights
        if weight > 0
    ]

    negative_features = [
        (segment_id, weight)
        for segment_id, weight in local_weights
        if weight < 0
    ]

    positive_ids = [
        segment_id
        for segment_id, _ in positive_features
    ]

    negative_ids = [
        segment_id
        for segment_id, _ in negative_features
    ]

    positive_mask = np.isin(
        segments,
        positive_ids,
    )

    negative_mask = np.isin(
        segments,
        negative_ids,
    )

    # --------------------------------------------------------
    # Positive overlay
    # Green: supports the explained target class
    # --------------------------------------------------------

    green = np.array(
        [0, 255, 0],
        dtype=np.float32,
    )

    positive_overlay = image.astype(
        np.float32
    ).copy()

    positive_overlay[positive_mask] = (
        positive_overlay[positive_mask] * 0.50
        + green * 0.50
    )

    positive_overlay = np.clip(
        positive_overlay,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Negative overlay
    # Red: weighs against the explained target class
    # --------------------------------------------------------

    red = np.array(
        [255, 0, 0],
        dtype=np.float32,
    )

    negative_overlay = image.astype(
        np.float32
    ).copy()

    negative_overlay[negative_mask] = (
        negative_overlay[negative_mask] * 0.50
        + red * 0.50
    )

    negative_overlay = np.clip(
        negative_overlay,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Combined overlay
    # --------------------------------------------------------

    combined_overlay = image.astype(
        np.float32
    ).copy()

    combined_overlay[positive_mask] = (
        combined_overlay[positive_mask] * 0.50
        + green * 0.50
    )

    combined_overlay[negative_mask] = (
        combined_overlay[negative_mask] * 0.50
        + red * 0.50
    )

    combined_overlay = np.clip(
        combined_overlay,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Superpixel boundaries
    # --------------------------------------------------------

    boundary_image = mark_boundaries(
        image.astype(np.float32) / 255.0,
        segments,
        color=(1.0, 1.0, 0.0),
        mode="thick",
    )

    boundary_image = np.clip(
        boundary_image * 255,
        0,
        255,
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Signed feature weight map
    # --------------------------------------------------------

    weight_map = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    for segment_id, weight in local_weights:

        weight_map[
            segments == segment_id
        ] = weight

    maximum_weight = float(
        np.abs(weight_map).max()
    )

    if maximum_weight > 0:

        normalized_weights = (
            weight_map / maximum_weight
        )

    else:

        normalized_weights = weight_map

    colormap = plt.get_cmap(
        "coolwarm"
    )

    weight_rgb = (
        colormap(
            (normalized_weights + 1.0) / 2.0
        )[..., :3] * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    paths = {

        "original_file":
            directories["originals"]
            / f"{sample_slug}_original.png",

        "segmentation_file":
            directories["segmentations"]
            / f"{sample_slug}_segments.png",

        "positive_overlay_file":
            directories["positive"]
            / f"{sample_slug}_positive.png",

        "negative_overlay_file":
            directories["negative"]
            / f"{sample_slug}_negative.png",

        "overlay_file":
            directories["overlays"]
            / f"{sample_slug}_overlay.png",

        "weight_map_file":
            directories["weight_maps"]
            / f"{sample_slug}_weights.png",

        "positive_mask_file":
            directories["masks"]
            / f"{sample_slug}_positive_mask.png",

        "negative_mask_file":
            directories["masks"]
            / f"{sample_slug}_negative_mask.png",

        "segments_array_file":
            directories["arrays"]
            / f"{sample_slug}_segments.npy",

        "weights_array_file":
            directories["arrays"]
            / f"{sample_slug}_weights.npy",

        "comparison_file":
            directories["comparisons"]
            / f"{sample_slug}_comparison.png",
    }

    save_image(
        image,
        paths["original_file"],
    )

    save_image(
        boundary_image,
        paths["segmentation_file"],
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
        combined_overlay,
        paths["overlay_file"],
    )

    save_image(
        weight_rgb,
        paths["weight_map_file"],
    )

    save_image(
        positive_mask.astype(np.uint8) * 255,
        paths["positive_mask_file"],
    )

    save_image(
        negative_mask.astype(np.uint8) * 255,
        paths["negative_mask_file"],
    )

    np.save(
        paths["segments_array_file"],
        segments.astype(np.int32),
    )

    np.save(
        paths["weights_array_file"],
        weight_map.astype(np.float32),
    )

    # --------------------------------------------------------
    # Comparison panel
    # --------------------------------------------------------

    panels = [
        image,
        boundary_image,
        positive_overlay,
        negative_overlay,
        weight_rgb,
    ]

    labels = [
        "Original",
        "Superpixels",
        "Positive",
        "Negative",
        "Weight map",
    ]

    panel_width = width * len(panels)

    panel_height = height + 35

    comparison = Image.new(
        "RGB",
        (panel_width, panel_height),
        "white",
    )

    from PIL import ImageDraw

    draw = ImageDraw.Draw(
        comparison
    )

    for index, (panel, label) in enumerate(
        zip(panels, labels)
    ):

        panel_image = Image.fromarray(
            panel
        )

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

    comparison.save(
        paths["comparison_file"]
    )

    comparison.close()

    return {
        key: str(value)
        for key, value in paths.items()
    }, positive_ids, negative_ids, weight_map


# ============================================================
# CLASS SUMMARY
# ============================================================

def calculate_class_summary(results):

    summaries = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        rows = [
            item
            for item in results
            if item["actual_class_index"] == class_index
        ]

        correct = [
            item
            for item in rows
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

            "mean_local_model_score": (
                float(np.mean([
                    item["local_model_score"]
                    for item in rows
                ]))
                if rows else None
            ),

            "mean_absolute_feature_weight": (
                float(np.mean([
                    item["mean_absolute_feature_weight"]
                    for item in rows
                ]))
                if rows else None
            ),

            "mean_explanation_runtime": (
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

        "MAIZE LEAF CLASSIFICATION - LIME EXPLANATION REPORT",

        "=" * 78,

        "",

        f"Model: {summary['model_name']}",

        f"Run: {summary['run_name']}",

        f"Checkpoint: {summary['checkpoint']}",

        f"Checkpoint epoch: {summary['checkpoint_epoch']}",

        f"Method: {summary['method']}",

        f"Partition: {summary['partition']}",

        f"Samples explained: {summary['sample_count']}",

        "",

        "LIME CONFIGURATION",

        "-" * 78,

        f"Perturbation samples: {summary['num_samples']}",

        f"Maximum features: {summary['num_features']}",

        f"Requested superpixels: {summary['num_segments']}",

        f"SLIC compactness: {summary['compactness']}",

        f"SLIC sigma: {summary['sigma']}",

        f"Prediction batch size: {summary['prediction_batch_size']}",

        f"Target class policy: {summary['target_class_policy']}",

        "",

        "OVERALL RESULTS",

        "-" * 78,

        f"Correct: {summary['correct_predictions']}",

        f"Incorrect: {summary['incorrect_predictions']}",

        f"Subset accuracy: {summary['sample_accuracy']:.6f}",

        f"Mean confidence: {summary['mean_confidence']:.6f}",

        f"Mean local model R2: {summary['mean_local_model_score']:.6f}",

        f"Mean absolute feature weight: {summary['mean_absolute_feature_weight']:.6f}",

        f"Mean explanation time: {summary['mean_runtime_seconds']:.4f} seconds",

        f"Total runtime: {summary['total_runtime_seconds']:.2f} seconds",

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

            f"  Subset accuracy: {item['sample_accuracy']}",

            f"  Mean local model R2: {item['mean_local_model_score']}",

            f"  Mean absolute feature weight: {item['mean_absolute_feature_weight']}",

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

        "Held-out test set used: NO",

        f"Group IDs available: {summary['validation_group_ids_available']}",

        "",

        "INTERPRETATION",

        "-" * 78,

        "Positive superpixels support the selected target class",

        "under the local surrogate model.",

        "Negative superpixels weigh against the selected target class.",

        "Local model R2 measures the surrogate's local fit.",

        "LIME weights are not biological causal effects.",

        "Superpixel importance does not establish disease severity.",

        "Interpretations depend on segmentation and perturbation settings.",

        "",

        "=" * 78,

    ])

    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ============================================================
# MAIN LIME ENGINE
# ============================================================

def _run_lime_internal(args):

    validate_arguments(args)

    run_dir = Path(
        args.run_dir
    ).expanduser().resolve()

    if not run_dir.is_dir():

        raise NotADirectoryError(
            f"Run directory not found: {run_dir}"
        )

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

    print_section(
        "MAIZE LEAF - LIME EXPLAINABILITY ENGINE"
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

    # --------------------------------------------------------
    # Fixed split verification
    # --------------------------------------------------------

    print_section(
        "VERIFYING FIXED DATASET SPLIT"
    )

    split_fingerprint = common.verify_split(
        run_dir
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    print_section(
        "LOADING TRAINING CONFIGURATION"
    )

    config, config_data, model_name = (
        common.load_training_configuration(
            run_dir
        )
    )

    print(f"Model      : {model_name}")

    print(f"Image size : {config.image_size}")

    print(f"Classes    : {NUM_CLASSES}")

    for index, name in enumerate(CLASS_NAMES):

        print(f"{index}: {name}")

    # --------------------------------------------------------
    # Checkpoint
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

    # --------------------------------------------------------
    # Validation records
    # --------------------------------------------------------

    print_section(
        "BUILDING VALIDATION DATA"
    )

    validation_records = (
        common.load_validation_records()
    )

    group_ids_available = any(
        str(record.get("group_id", "")).strip()
        for record in validation_records
    )

    if not group_ids_available:

        print(
            "[INFO] Validation group IDs are empty."
        )

    validation_transform, image_size = (
        common.build_validation_transform(
            config
        )
    )

    print(
        f"[PASS] Evaluation image size: {image_size}"
    )

    print(
        "[PASS] Validation preprocessing configured"
    )

    print(
        "[PASS] Test partition not accessed"
    )

    # --------------------------------------------------------
    # Prediction lookup and selection
    # --------------------------------------------------------

    print_section(
        "SELECTING VALIDATION SAMPLES"
    )

    prediction_lookup = (
        common.read_prediction_lookup(
            run_dir
        )
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

    # --------------------------------------------------------
    # LIME configuration
    # --------------------------------------------------------

    print_section(
        "INITIALIZING LIME EXPLAINER"
    )

    print(
        f"Perturbation samples : {args.num_samples}"
    )

    print(
        f"Maximum features    : {args.num_features}"
    )

    print(
        f"SLIC superpixels    : {args.num_segments}"
    )

    print(
        f"Prediction batch    : {args.prediction_batch_size}"
    )

    print(
        "[PASS] Model-agnostic classifier configured"
    )

    explainer = LimeImageExplainer(
        kernel_width=0.25,

        verbose=False,

        feature_selection="auto",

        random_state=args.seed,
    )

    classifier_fn = build_classifier_fn(
        model=model,

        transform=validation_transform,

        device=device,

        prediction_batch_size=args.prediction_batch_size,
    )

    segmentation_function = (
        build_segmentation_function(args)
    )

    # --------------------------------------------------------
    # Output directories
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
            / "lime"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    directories = {

        "originals": output_dir / "originals",

        "segmentations": output_dir / "segmentations",

        "positive": output_dir / "positive",

        "negative": output_dir / "negative",

        "overlays": output_dir / "overlays",

        "weight_maps": output_dir / "weight_maps",

        "masks": output_dir / "masks",

        "arrays": output_dir / "arrays",

        "comparisons": output_dir / "comparisons",
    }

    for directory in directories.values():

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # Explanation generation
    # --------------------------------------------------------

    print_section(
        "GENERATING LIME EXPLANATIONS"
    )

    results = []

    feature_rows = []

    selected_rows = []

    total_started = time.perf_counter()

    progress_bar = None

    if progress_tqdm is not None:
        progress_bar = progress_tqdm(
            total=len(selected_samples),
            desc="LIME",
            dynamic_ncols=True,
            unit="image",
        )

    for index, record in enumerate(
        selected_samples,
        start=1,
    ):

        sample_started = time.perf_counter()

        image_path = common.resolve_image_path(
            common.get_record_image_path(record)
        )

        actual_class = int(
            record["label"]
        )

        actual_class_name = CLASS_NAMES[
            actual_class
        ]

        # Load and resize original image.
        with Image.open(image_path) as source:

            original_image = source.convert("RGB")

            original_image = original_image.resize(
                (image_size, image_size),
                Image.Resampling.BILINEAR,
            )

            image_array = np.asarray(
                original_image,
                dtype=np.uint8,
            ).copy()

        # Original prediction.
        baseline_probabilities = classifier_fn(
            image_array[np.newaxis, ...]
        )[0]

        predicted_class = int(
            np.argmax(baseline_probabilities)
        )

        confidence = float(
            baseline_probabilities[predicted_class]
        )

        correct = (
            actual_class == predicted_class
        )

        # Choose the class whose decision is explained.
        target_class = (
            predicted_class
            if args.target_class == "predicted"
            else actual_class
        )

        target_probability = float(
            baseline_probabilities[target_class]
        )

        sample_id = f"sample_{index:04d}"

        sample_slug = (
            f"{sample_id}_"
            f"true-{common.safe_slug(actual_class_name)}_"
            f"pred-{common.safe_slug(CLASS_NAMES[predicted_class])}"
        )

        # ----------------------------------------------------
        # LIME perturbation and explanation
        # ----------------------------------------------------

        explanation_started = time.perf_counter()

        explanation = explain_instance_without_internal_progress(

            explainer,

            image=image_array,

            classifier_fn=classifier_fn,

            labels=(target_class,),

            top_labels=None,

            hide_color=args.hide_color,

            num_features=args.num_features,

            num_samples=args.num_samples,

            batch_size=args.prediction_batch_size,

            segmentation_fn=segmentation_function,

            random_seed=args.seed + index,
        )

        explanation_runtime = (
            time.perf_counter()
            - explanation_started
        )

        segments = np.asarray(
            explanation.segments,
            dtype=np.int32,
        )

        if segments.shape != (
            image_size,
            image_size,
        ):

            raise RuntimeError(
                f"Unexpected segmentation shape: {segments.shape}"
            )

        local_exp = explanation.local_exp.get(
            target_class,
            [],
        )

        if not local_exp:

            raise RuntimeError(
                f"No LIME explanation generated for "
                f"class {target_class}."
            )

        local_weights = sorted(
            [
                (int(segment_id), float(weight))
                for segment_id, weight in local_exp
            ],
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        # ----------------------------------------------------
        # Surrogate statistics
        # ----------------------------------------------------

        local_model_score = float(
            explanation.score
        )

        local_prediction = float(
            np.asarray(
                explanation.local_pred
            ).reshape(-1)[0]
        )

        intercept = float(
            explanation.intercept[target_class]
        )

        mean_absolute_weight = float(
            np.mean([
                abs(weight)
                for _, weight in local_weights
            ])
        )

        positive_count = sum(
            weight > 0
            for _, weight in local_weights
        )

        negative_count = sum(
            weight < 0
            for _, weight in local_weights
        )

        # ----------------------------------------------------
        # Visual artifacts
        # ----------------------------------------------------

        visual_files, positive_ids, negative_ids, weight_map = (
            create_lime_visualizations(
                image=image_array,

                segments=segments,

                local_weights=local_weights,

                sample_slug=sample_slug,

                directories=directories,
            )
        )

        unique_segments = np.unique(
            segments
        )

        segment_areas = {
            int(segment_id): int(
                np.count_nonzero(
                    segments == segment_id
                )
            )
            for segment_id in unique_segments
        }

        for rank, (segment_id, weight) in enumerate(
            local_weights,
            start=1,
        ):

            area = segment_areas.get(
                segment_id,
                0,
            )

            feature_rows.append({

                "sample_id": sample_id,

                "image_path": str(image_path),

                "actual_class": actual_class_name,

                "predicted_class": CLASS_NAMES[predicted_class],

                "target_class": CLASS_NAMES[target_class],

                "feature_rank": rank,

                "segment_id": segment_id,

                "weight": weight,

                "direction": (
                    "positive"
                    if weight > 0
                    else "negative"
                    if weight < 0
                    else "neutral"
                ),

                "segment_area_pixels": area,

                "segment_area_fraction": (
                    area / (image_size * image_size)
                ),

                "local_model_score": local_model_score,
            })

        # ----------------------------------------------------
        # Sample record
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

            "split": str(
                record.get("split", "validation")
            ),

            "actual_class_index": actual_class,

            "actual_class": actual_class_name,

            "predicted_class_index": predicted_class,

            "predicted_class": CLASS_NAMES[predicted_class],

            "target_class_index": target_class,

            "target_class": CLASS_NAMES[target_class],

            "target_class_policy": args.target_class,

            "confidence": confidence,

            "target_probability": target_probability,

            "correct": bool(correct),

            "local_model_score": local_model_score,

            "local_prediction": local_prediction,

            "intercept": intercept,

            "mean_absolute_feature_weight": mean_absolute_weight,

            "positive_feature_count": positive_count,

            "negative_feature_count": negative_count,

            "total_superpixels": int(
                len(unique_segments)
            ),

            "explained_features": len(
                local_weights
            ),

            "positive_segments": len(
                positive_ids
            ),

            "negative_segments": len(
                negative_ids
            ),

            "runtime_seconds": float(
                explanation_runtime
            ),

            "sample_runtime_seconds": float(
                time.perf_counter() - sample_started
            ),

            "top_features_json": str([
                {
                    "segment_id": segment_id,
                    "weight": weight,
                }
                for segment_id, weight in local_weights
            ]),

            **visual_files,
        }

        results.append(result)

        selected_rows.append({

            "sample_id": sample_id,

            "image_path": str(image_path),

            "relative_path": str(
                record.get("relative_path", "")
            ),

            "group_id": str(
                record.get("group_id", "")
            ),

            "split": "validation",

            "actual_class": actual_class_name,

            "actual_class_index": actual_class,

        })

        del explanation

        if device.type == "cuda":

            torch.cuda.empty_cache()

        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.close()

    total_runtime = (
        time.perf_counter()
        - total_started
    )

    # --------------------------------------------------------
    # Generate reports
    # --------------------------------------------------------

    print_section(
        "GENERATING LIME REPORTS"
    )

    if not results:

        raise RuntimeError(
            "No LIME explanations were generated."
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

    mean_local_score = float(
        np.mean([
            item["local_model_score"]
            for item in results
        ])
    )

    mean_absolute_weight = float(
        np.mean([
            item["mean_absolute_feature_weight"]
            for item in results
        ])
    )

    mean_runtime = float(
        np.mean([
            item["runtime_seconds"]
            for item in results
        ])
    )

    class_summary = calculate_class_summary(
        results
    )

    # --------------------------------------------------------
    # CSV exports
    # --------------------------------------------------------

    common.save_csv(
        results,
        output_dir / "lime_results.csv",
    )

    common.save_csv(
        feature_rows,
        output_dir / "lime_features.csv",
    )

    common.save_csv(
        class_summary,
        output_dir / "class_summary.csv",
    )

    common.save_csv(
        selected_rows,
        output_dir / "selected_samples.csv",
    )

    print(
        "[PASS] lime_results.csv"
    )

    print(
        "[PASS] lime_features.csv"
    )

    print(
        "[PASS] class_summary.csv"
    )

    print(
        "[PASS] selected_samples.csv"
    )

    # --------------------------------------------------------
    # Contact sheet
    # --------------------------------------------------------

    contact_sheet_path = None

    if not args.no_contact_sheet:

        contact_sheet_path = (
            common.create_contact_sheet(
                results,
                output_dir / "lime_contact_sheet.jpg",
                columns=4,
            )
        )

        if contact_sheet_path:

            print(
                "[PASS] lime_contact_sheet.jpg"
            )

    # --------------------------------------------------------
    # Summary metadata
    # --------------------------------------------------------

    summary = {

        "status": "COMPLETED",

        "method": "LIME",

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

        "validation_manifest_only": True,

        "train_manifest_loaded": False,

        "held_out_test_manifest_loaded": False,

        "held_out_test_used": False,

        "validation_group_ids_available": (
            group_ids_available
        ),

        "sample_count": len(results),

        "samples_per_class_requested": (
            args.samples_per_class
        ),

        "sample_strategy": args.sample_strategy,

        "target_class_policy": args.target_class,

        "num_samples": args.num_samples,

        "num_features": args.num_features,

        "num_segments": args.num_segments,

        "compactness": args.compactness,

        "sigma": args.sigma,

        "prediction_batch_size": (
            args.prediction_batch_size
        ),

        "hide_color": args.hide_color,

        "image_size": image_size,

        "class_names": list(CLASS_NAMES),

        "correct_predictions": correct_count,

        "incorrect_predictions": incorrect_count,

        "sample_accuracy": (
            correct_count / len(results)
        ),

        "mean_confidence": mean_confidence,

        "mean_local_model_score": mean_local_score,

        "mean_absolute_feature_weight": (
            mean_absolute_weight
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

        "lime_version": get_version("lime"),

        "scikit_image_version": get_version("scikit-image"),

        "torch_version": torch.__version__,

        "class_summary": class_summary,

        "contact_sheet": contact_sheet_path,

        "output_directory": str(output_dir),

        "limitations": [

            "LIME is a local model-agnostic surrogate explanation.",

            "Positive features support the selected target class locally.",

            "Negative features weigh against the selected target class locally.",

            "Local model R2 is not overall classifier accuracy.",

            "Explanation quality depends on segmentation and perturbation settings.",

            "Superpixel weights do not establish biological causality.",

            "The balanced explanation subset is not a substitute for full validation metrics.",

        ],
    }

    common.save_json(
        summary,
        output_dir / "lime_summary.json",
    )

    write_report(
        summary,
        class_summary,
        output_dir / "lime_report.txt",
    )

    print(
        "[PASS] lime_summary.json"
    )

    print(
        "[PASS] lime_report.txt"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_section(
        "GENERATING LIME SUCCESSFUL"
    )

    print(f"Model                 : {model_name}")

    print(f"Validation images     : {len(validation_records)}")

    print(f"Samples explained     : {len(results)}")

    print(f"Correct predictions   : {correct_count}")

    print(f"Incorrect predictions : {incorrect_count}")

    print(
        f"Subset accuracy      : "
        f"{correct_count / len(results):.4f}"
    )

    print(
        f"Mean confidence      : {mean_confidence:.4f}"
    )

    print(
        f"Mean local model R2  : {mean_local_score:.4f}"
    )

    print(
        f"Mean absolute weight : {mean_absolute_weight:.6f}"
    )

    print(
        f"Mean LIME runtime    : {mean_runtime:.4f} seconds"
    )

    print(
        f"Total runtime        : {total_runtime:.2f} seconds"
    )

    print(
        "Train manifest read  : NO"
    )

    print(
        "Test manifest read   : NO"
    )

    print(
        "Test set used        : NO"
    )

    print(
        f"Output directory     : {output_dir}"
    )

    print("=" * 76)

    return summary




def run_lime(args):
    """
    Run LIME with a minimal terminal interface.

    Routine setup and report messages are hidden. The two requested
    section headings are printed to the real terminal, while tqdm's
    single image-level progress bar remains visible on stderr.
    Errors are allowed to propagate to main() and remain visible.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        return _run_lime_internal(args)

# ============================================================
# COMMAND LINE
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description="Maize Leaf Classification - LIME Engine"
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
        "--num-samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--num-features",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num-segments",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--compactness",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--prediction-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--hide-color",
        type=int,
        default=None,
        help="Replacement pixel value. Default: mean image color.",
    )

    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    try:

        run_lime(args)

    except KeyboardInterrupt:

        print(
            "\n[LIME] Interrupted by user."
        )

        raise

    except Exception as error:

        print(
            f"\n[LIME ERROR] {type(error).__name__}: {error}"
        )

        raise


if __name__ == "__main__":

    main()
