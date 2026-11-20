"""
MAIZE LEAF CLASSIFICATION
GRAD-CAM EXPLAINABILITY ENGINE

File:
src/explainability/gradcam.py

Supported models:
1. Custom CNN
2. ResNet50
3. EfficientNet-B0
4. MobileNetV3-Large
5. DenseNet121
6. ConvNeXt-Tiny
7. Small Vision Transformer

Features:
- Best checkpoint reconstruction
- Architecture-specific target layers
- Grad-CAM computation
- Vision Transformer token reshaping
- Correct and incorrect sample analysis
- Class-balanced reproducible sampling
- Original image export
- Heatmap export
- Overlay export
- Raw CAM array export
- Contact sheet
- CSV, JSON and TXT reports
- Fixed validation split verification

The held-out test partition is never accessed.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import hashlib
import inspect
import json
import math
import random
import re
import sys
import time

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_SPLIT_FINGERPRINT = (
    "8a64e1a8a53eecd2dd2eeedbfb63c7e783a193304a2529da1be4e8c7bff441de"
)


from src.data.class_names import (
    CLASS_NAMES,
    NUM_CLASSES,
)

from src.data.dataset_loader import (
    load_split_csv,
)

from src.data.preprocessing import (
    build_transforms,
)

import Models.model_factory as model_factory


# ============================================================
# GENERAL UTILITIES
# ============================================================

def print_section(title):

    visible_sections = {
        "GENERATING GRAD-CAM EXPLANATIONS",
        "GENERATING GRAD-CAM SUCCESSFUL",
    }

    if title not in visible_sections:
        return

    # Print directly to the real terminal while routine stdout is hidden.
    stream = sys.__stdout__ or sys.stdout

    stream.write("\n" + "=" * 76 + "\n")
    stream.write(title + "\n")
    stream.write("=" * 76 + "\n")
    stream.flush()


def save_json(data, path):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)

    return path


def save_csv(rows, path, fields=None):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if fields is None:

        fields = (
            list(rows[0].keys())
            if rows
            else []
        )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(rows)

    return path


def sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


def safe_slug(value):

    return re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(value),
    ).strip("_").lower()


def json_load(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def normalize_path(path):

    return str(
        Path(path).expanduser().resolve()
    ).casefold()


def get_record_image_path(record):
    """Return the image path from a dataset manifest record.

    The current dataset loader uses the key ``path``. The alternate
    names are retained only for compatibility with older manifests.
    """

    if not isinstance(record, dict):

        raise TypeError(
            f"Dataset record must be a dictionary, received {type(record)}"
        )

    for key in ("path", "image_path", "filepath", "relative_path"):

        value = record.get(key)

        if value is not None and str(value).strip():

            return value

    raise KeyError(
        "Dataset record has no usable image path. "
        "Expected one of: path, image_path, filepath, relative_path."
    )


def resolve_image_path(value):

    path = Path(str(value)).expanduser()

    if not path.is_absolute():

        path = PROJECT_ROOT / path

    path = path.resolve()

    if not path.is_file():

        raise FileNotFoundError(
            f"Dataset image not found: {path}"
        )

    return path


def extract_logits(output):

    if torch.is_tensor(output):

        logits = output

    elif isinstance(output, dict):

        if "logits" not in output:

            raise ValueError(
                "Model output dictionary has no logits key."
            )

        logits = output["logits"]

    elif isinstance(output, (tuple, list)):

        if not output:

            raise ValueError(
                "Model returned an empty output."
            )

        logits = output[0]

    else:

        raise TypeError(
            f"Unsupported model output: {type(output)}"
        )

    if not torch.is_tensor(logits):

        raise TypeError(
            "Model logits must be a tensor."
        )

    if logits.ndim != 2:

        raise ValueError(
            f"Expected logits [B,C], received {tuple(logits.shape)}"
        )

    return logits


# ============================================================
# CONFIGURATION LOADING
# ============================================================

def load_training_configuration(run_dir):

    config_path = (
        run_dir / "training_config.json"
    )

    if not config_path.is_file():

        raise FileNotFoundError(
            f"Training configuration not found: {config_path}"
        )

    config_data = json_load(config_path)

    if (
        isinstance(config_data, dict)
        and isinstance(config_data.get("config"), dict)
    ):

        config_data = config_data["config"]

    elif (
        isinstance(config_data, dict)
        and isinstance(config_data.get("training_config"), dict)
    ):

        config_data = config_data["training_config"]

    if not isinstance(config_data, dict):

        raise ValueError(
            "Invalid training configuration format."
        )

    config_data = dict(config_data)

    model_name = config_data.get(
        "model",
        run_dir.parent.name,
    )

    if model_name != run_dir.parent.name:

        raise RuntimeError(
            "Model name in training_config.json does not "
            "match the Results directory."
        )

    # The checkpoint already contains the trained weights.
    # Avoid downloading ImageNet weights during reconstruction.

    config_data["model"] = model_name

    config_data["pretrained"] = False

    config_data["freeze_backbone"] = False

    config_data["num_classes"] = NUM_CLASSES

    config_data.setdefault(
        "image_size",
        224,
    )

    config_data.setdefault(
        "dropout",
        0.35,
    )

    config_data.setdefault(
        "vit_embed_dim",
        192,
    )

    config_data.setdefault(
        "vit_depth",
        6,
    )

    config_data.setdefault(
        "vit_heads",
        3,
    )

    return (
        SimpleNamespace(**config_data),
        config_data,
        model_name,
    )


# ============================================================
# SPLIT FINGERPRINT VERIFICATION
# ============================================================

def find_fingerprint(data):

    if isinstance(data, dict):

        for key, value in data.items():

            if (
                key == "split_assignment_fingerprint_sha256"
                and isinstance(value, str)
            ):

                return value

        for value in data.values():

            result = find_fingerprint(value)

            if result:
                return result

    elif isinstance(data, list):

        for value in data:

            result = find_fingerprint(value)

            if result:
                return result

    return None


def verify_split(run_dir):

    snapshot_path = (
        run_dir / "split_metadata_snapshot.json"
    )

    root_metadata_path = (
        PROJECT_ROOT
        / "data_splits"
        / "split_metadata.json"
    )

    if not snapshot_path.is_file():

        raise FileNotFoundError(
            "Run split metadata snapshot is missing."
        )

    if not root_metadata_path.is_file():

        raise FileNotFoundError(
            "Central split metadata is missing."
        )

    snapshot = json_load(snapshot_path)

    root_metadata = json_load(
        root_metadata_path
    )

    run_fingerprint = find_fingerprint(
        snapshot
    )

    current_fingerprint = find_fingerprint(
        root_metadata
    )

    if not run_fingerprint:

        raise RuntimeError(
            "Run split fingerprint is missing."
        )

    if not current_fingerprint:

        raise RuntimeError(
            "Current split fingerprint is missing."
        )

    if run_fingerprint != EXPECTED_SPLIT_FINGERPRINT:

        raise RuntimeError(
            "Run fingerprint does not match the "
            "verified seven-class split."
        )

    if current_fingerprint != EXPECTED_SPLIT_FINGERPRINT:

        raise RuntimeError(
            "Current dataset split has changed."
        )

    if run_fingerprint != current_fingerprint:

        raise RuntimeError(
            "Run and current split fingerprints differ."
        )

    print(
        "[PASS] Run split fingerprint"
    )

    print(
        "[PASS] Current central split fingerprint"
    )

    print(
        "[PASS] Fixed seven-class dataset"
    )

    return run_fingerprint


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_best_model(
    run_dir,
    config,
    model_name,
    device,
):

    checkpoint_path = (
        run_dir
        / "checkpoints"
        / "best_model.pth"
    )

    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if not isinstance(checkpoint, dict):

        raise RuntimeError(
            "Invalid checkpoint format."
        )

    checkpoint_model_name = checkpoint.get("model_name")

    if checkpoint_model_name and checkpoint_model_name != model_name:

        raise RuntimeError(
            f"Checkpoint model mismatch: expected {model_name}, "
            f"found {checkpoint_model_name}."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if state_dict is None:

        raise KeyError(
            "Checkpoint does not contain model_state_dict."
        )

    if not isinstance(state_dict, dict):

        raise TypeError(
            "model_state_dict must be a dictionary."
        )

    print(
        "\nBUILDING MODEL ARCHITECTURE"
    )

    model_kwargs = {}

    if model_name == "small_vit":

        model_kwargs = {
            "image_size": int(getattr(config, "image_size", 224)),
            "embed_dim": int(getattr(config, "vit_embed_dim", 192)),
            "depth": int(getattr(config, "vit_depth", 6)),
            "num_heads": int(getattr(config, "vit_heads", 3)),
        }

    model = model_factory.build_model_from_config(
        config,
        **model_kwargs,
    )

    if not isinstance(model, nn.Module):

        raise TypeError(
            "Model factory did not return a PyTorch model."
        )

    # Enable gradients for XAI even if the original training
    # configuration froze some backbone parameters.

    for parameter in model.parameters():

        parameter.requires_grad_(True)

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if incompatible.missing_keys or incompatible.unexpected_keys:

        raise RuntimeError(
            "Checkpoint architecture compatibility failed."
        )

    model = model.to(device)

    model.eval()

    if device.type == "cuda":

        torch.cuda.empty_cache()

    print(
        f"[PASS] Model reconstructed: {model_name}"
    )

    print(
        "[PASS] Best checkpoint loaded"
    )

    print(
        f"[PASS] Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'Unknown')}"
    )

    print(
        f"[PASS] Checkpoint size: "
        f"{checkpoint_path.stat().st_size / (1024 ** 2):.2f} MB"
    )

    return (
        model,
        checkpoint,
        checkpoint_path,
    )


# ============================================================
# TARGET LAYER CONFIGURATION
# ============================================================

def call_factory_helper(function, model, model_name):

    if not callable(function):
        return None

    try:

        signature = inspect.signature(
            function
        )

        parameters = list(
            signature.parameters.values()
        )

        if len(parameters) >= 2:

            return function(
                model,
                model_name,
            )

        return function(model)

    except (TypeError, ValueError):

        try:

            return function(model)

        except TypeError:

            return function(
                model,
                model_name,
            )


def fallback_target_layer(
    model,
    model_name,
):

    # The project factory wraps torchvision models in a ``backbone``
    # attribute. Supporting both wrapped and unwrapped models makes the
    # fallback safe if the factory helper is unavailable.
    backbone = getattr(model, "backbone", model)

    if model_name == "custom_cnn":

        return model.features[-1].block[3]

    if model_name == "resnet50":

        return backbone.layer4[-1]

    if model_name in (
        "efficientnet_b0",
        "mobilenet_v3",
        "mobnetfine",
        "convnext_tiny",
    ):

        return backbone.features[-1]

    if model_name == "densenet121":

        return (
            backbone.features
            .denseblock4
            .denselayer16
            .conv2
        )

    if model_name == "small_vit":

        if hasattr(model, "blocks"):

            return model.blocks[-1].norm2

        if hasattr(model, "patch_embed"):

            return model.patch_embed

    raise ValueError(
        f"No Grad-CAM target layer available for {model_name}."
    )

def get_target_configuration(
    model,
    model_name,
):

    target_layer = None

    reshape_transform = None

    target_function = getattr(
        model_factory,
        "get_gradcam_target_layer",
        None,
    )

    reshape_function = getattr(
        model_factory,
        "get_gradcam_reshape_transform",
        None,
    )

    if callable(target_function):

        try:

            result = call_factory_helper(
                target_function,
                model,
                model_name,
            )

            if isinstance(result, tuple):

                target_layer = result[0]

                if len(result) > 1:

                    reshape_transform = result[1]

            elif isinstance(result, dict):

                target_layer = result.get(
                    "target_layer",
                    result.get("layer"),
                )

                reshape_transform = result.get(
                    "reshape_transform"
                )

            else:

                target_layer = result

        except Exception as error:

            print(
                f"[WARNING] Factory target lookup failed: {error}"
            )

    if not isinstance(target_layer, nn.Module):

        target_layer = fallback_target_layer(
            model,
            model_name,
        )

    if reshape_transform is None and callable(reshape_function):

        try:

            reshape_transform = call_factory_helper(
                reshape_function,
                model,
                model_name,
            )

        except Exception as error:

            print(
                f"[WARNING] Factory reshape lookup failed: {error}"
            )

    layer_name = next(
        (
            name
            for name, module in model.named_modules()
            if module is target_layer
        ),
        target_layer.__class__.__name__,
    )

    print(
        f"[PASS] Grad-CAM target layer: {layer_name}"
    )

    if reshape_transform is not None:

        print(
            "[PASS] Architecture reshape transform available"
        )

    return (
        target_layer,
        reshape_transform,
        layer_name,
    )


# ============================================================
# FEATURE TENSOR HANDLING
# ============================================================

def extract_feature_tensor(output):

    if torch.is_tensor(output):

        return output

    if isinstance(output, (tuple, list)):

        for value in output:

            if torch.is_tensor(value):

                return value

    if isinstance(output, dict):

        for value in output.values():

            if torch.is_tensor(value):

                return value

    raise RuntimeError(
        "Grad-CAM target did not return a tensor."
    )


def reshape_feature_tensor(
    tensor,
    reshape_transform=None,
):

    if tensor.ndim == 4:

        return tensor

    if callable(reshape_transform):

        transformed = reshape_transform(
            tensor
        )

        if not torch.is_tensor(transformed):

            raise RuntimeError(
                "ViT reshape transform did not return a tensor."
            )

        if transformed.ndim == 4:

            return transformed

    # Generic Vision Transformer token handling.
    #
    # Input: B x N x C
    # Output: B x C x H x W
    #
    # A class token is removed when N-1 is a perfect square.

    if tensor.ndim == 3:

        batch_size, token_count, channels = tensor.shape

        spatial_tokens = token_count

        if math.isqrt(token_count - 1) ** 2 == token_count - 1:

            tensor = tensor[:, 1:, :]

            spatial_tokens = token_count - 1

        side = math.isqrt(
            spatial_tokens
        )

        if side * side != spatial_tokens:

            raise RuntimeError(
                "Cannot reshape transformer tokens "
                "into a square spatial feature map."
            )

        tensor = tensor.reshape(
            batch_size,
            side,
            side,
            channels,
        )

        tensor = tensor.permute(
            0,
            3,
            1,
            2,
        ).contiguous()

        return tensor

    raise RuntimeError(
        f"Unsupported Grad-CAM feature shape: {tuple(tensor.shape)}"
    )


# ============================================================
# GRAD-CAM COMPUTATION
# ============================================================

def normalize_cam(cam):

    cam = np.asarray(
        cam,
        dtype=np.float32,
    )

    cam = np.nan_to_num(
        cam,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    cam = np.maximum(
        cam,
        0,
    )

    minimum = float(
        cam.min()
    )

    maximum = float(
        cam.max()
    )

    if maximum - minimum < 1e-12:

        return np.zeros_like(
            cam,
            dtype=np.float32,
        )

    return (
        cam - minimum
    ) / (
        maximum - minimum
    )


def compute_gradcam(
    model,
    target_layer,
    reshape_transform,
    input_tensor,
    target_class,
):

    captured = {}

    def forward_hook(
        module,
        inputs,
        output,
    ):

        activation = extract_feature_tensor(
            output
        )

        captured["activation"] = activation

        if not activation.requires_grad:

            raise RuntimeError(
                "Target activation does not require gradients."
            )

        def save_gradient(gradient):

            captured["gradient"] = gradient

        activation.register_hook(
            save_gradient
        )

    handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        model.zero_grad(
            set_to_none=True
        )

        with torch.enable_grad():

            logits = extract_logits(
                model(input_tensor)
            )

            if logits.shape[1] != NUM_CLASSES:

                raise RuntimeError(
                    "Model output does not match seven classes."
                )

            if not 0 <= target_class < NUM_CLASSES:

                raise ValueError(
                    "Invalid Grad-CAM target class."
                )

            target_score = logits[
                0,
                target_class,
            ]

            target_score.backward()

        activation = captured.get(
            "activation"
        )

        gradient = captured.get(
            "gradient"
        )

        if activation is None or gradient is None:

            raise RuntimeError(
                "Grad-CAM activations or gradients were not captured."
            )

        activation = reshape_feature_tensor(
            activation,
            reshape_transform,
        )

        gradient = reshape_feature_tensor(
            gradient,
            reshape_transform,
        )

        if activation.shape != gradient.shape:

            raise RuntimeError(
                "Activation and gradient dimensions do not match."
            )

        # Global average pooling of gradients.
        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted combination of activation maps.
        cam = torch.sum(
            weights * activation,
            dim=1,
            keepdim=True,
        )

        cam = torch.relu(
            cam
        )

        # Resize to model input resolution.
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()

        cam = normalize_cam(
            cam
        )

        return (
            cam,
            float(target_score.detach().cpu().item()),
        )

    finally:

        handle.remove()

        model.zero_grad(
            set_to_none=True
        )


# ============================================================
# VALIDATION DATA AND PREPROCESSING
# ============================================================

def build_validation_transform(config):

    image_size = int(
        getattr(config, "image_size", 224)
    )

    if image_size <= 0:

        raise ValueError("Configured image size must be positive.")

    # Match src.data.dataset_loader.get_transforms(): its evaluation
    # pipeline uses bilinear resizing and ImageNet normalization.
    # No random augmentation is applied to validation images.
    _, validation_transform = build_transforms(
        image_size=image_size,
        augmentation=False,
        normalization="imagenet",
        interpolation="bilinear",
    )

    return (
        validation_transform,
        image_size,
    )

def load_validation_records():

    # IMPORTANT: load_split_csv("validation") reads only validation.csv.
    # Do not call load_all_split_records() here because that also opens
    # the held-out test manifest.
    validation_records = load_split_csv("validation")

    if not isinstance(validation_records, list):

        raise TypeError(
            "load_split_csv('validation') must return a list of records."
        )

    if not validation_records:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    seen_paths = set()

    for record in validation_records:

        image_value = get_record_image_path(record)

        path = normalize_path(
            resolve_image_path(image_value)
        )

        if path in seen_paths:

            raise RuntimeError(
                f"Duplicate validation image path: {path}"
            )

        seen_paths.add(path)

        label = int(
            record["label"]
        )

        if not 0 <= label < NUM_CLASSES:

            raise RuntimeError(
                f"Invalid validation label: {label}"
            )

        if (
            record.get("class_name")
            and record["class_name"] != CLASS_NAMES[label]
        ):

            raise RuntimeError(
                "Validation class name/index mismatch."
            )

        declared_split = str(
            record.get("split", "validation")
        ).strip().lower()

        if declared_split != "validation":

            raise RuntimeError(
                f"Non-validation record encountered: split={declared_split!r}"
            )

    print(
        f"[PASS] Validation records loaded: "
        f"{len(validation_records)}"
    )

    print(
        "[PASS] Seven-class validation mapping"
    )

    print(
        "[PASS] Validation image-path uniqueness"
    )

    print(
        "[PASS] Validation manifest only; train/test manifests not loaded"
    )

    return validation_records

# ============================================================
# PREDICTION LOOKUP FOR BALANCED SAMPLE SELECTION
# ============================================================

def read_prediction_lookup(run_dir):

    prediction_path = (
        run_dir
        / "evaluation"
        / "validation"
        / "predictions.csv"
    )

    if not prediction_path.is_file():

        print(
            "[WARNING] predictions.csv not found. "
            "Using random class-balanced sampling."
        )

        return {}

    lookup = {}

    with open(
        prediction_path,
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required = {
            "image_path",
            "correct",
        }

        if not reader.fieldnames or not required.issubset(
            set(reader.fieldnames)
        ):

            raise ValueError(
                "Validation predictions CSV has an invalid schema."
            )

        for row in reader:

            path = normalize_path(
                row["image_path"]
            )

            correct_value = str(
                row["correct"]
            ).strip().lower()

            if correct_value not in (
                "true",
                "false",
                "1",
                "0",
            ):

                raise ValueError(
                    "Invalid correct field in predictions.csv."
                )

            lookup[path] = (
                correct_value in ("true", "1")
            )

    print(
        f"[PASS] Existing prediction records loaded: "
        f"{len(lookup)}"
    )

    return lookup


# ============================================================
# REPRODUCIBLE CLASS-BALANCED SAMPLE SELECTION
# ============================================================

def select_samples(
    records,
    predictions_lookup,
    samples_per_class,
    strategy,
    seed,
):

    selected = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_records = [
            record
            for record in records
            if int(record["label"]) == class_index
        ]

        if not class_records:

            raise RuntimeError(
                f"No validation images for {class_name}."
            )

        rng = random.Random(
            seed + class_index * 1009
        )

        rng.shuffle(
            class_records
        )

        if not predictions_lookup or strategy == "random":

            chosen = class_records[
                :samples_per_class
            ]

        else:

            correct_records = []

            incorrect_records = []

            unknown_records = []

            for record in class_records:

                key = normalize_path(
                    resolve_image_path(
                        get_record_image_path(record)
                    )
                )

                if key not in predictions_lookup:

                    unknown_records.append(
                        record
                    )

                elif predictions_lookup[key]:

                    correct_records.append(
                        record
                    )

                else:

                    incorrect_records.append(
                        record
                    )

            rng.shuffle(correct_records)
            rng.shuffle(incorrect_records)
            rng.shuffle(unknown_records)

            if strategy == "correct":

                pool = correct_records + unknown_records

                chosen = pool[
                    :samples_per_class
                ]

            elif strategy == "incorrect":

                pool = incorrect_records + unknown_records

                chosen = pool[
                    :samples_per_class
                ]

            else:

                # BALANCED:
                # Aim for an approximately equal number
                # of correct and incorrect predictions.

                correct_target = (
                    samples_per_class + 1
                ) // 2

                incorrect_target = (
                    samples_per_class
                ) // 2

                chosen = (
                    correct_records[:correct_target]
                    +
                    incorrect_records[:incorrect_target]
                )

                remaining = [
                    record
                    for record in class_records
                    if record not in chosen
                ]

                rng.shuffle(
                    remaining
                )

                chosen.extend(
                    remaining[
                        :samples_per_class - len(chosen)
                    ]
                )

                chosen = chosen[
                    :samples_per_class
                ]

        for record in chosen:

            selected.append(
                dict(record)
            )

        print(
            f"[SAMPLE] {class_name}: "
            f"{len(chosen)}/{samples_per_class}"
        )

    selected.sort(
        key=lambda item: (
            int(item["label"]),
            str(get_record_image_path(item)).lower(),
        )
    )

    if not selected:

        raise RuntimeError(
            "No Grad-CAM samples were selected."
        )

    return selected


# ============================================================
# IMAGE VISUALIZATION
# ============================================================

def resize_display_image(
    image,
    width,
    height,
):

    resampling = getattr(
        Image,
        "Resampling",
        Image,
    ).LANCZOS

    return image.resize(
        (width, height),
        resampling,
    )


def save_visualization(
    original_image,
    cam,
    sample_id,
    directories,
):

    height, width = cam.shape

    display_image = resize_display_image(
        original_image,
        width,
        height,
    )

    original_array = np.asarray(
        display_image,
        dtype=np.uint8,
    )

    colormap = plt.get_cmap(
        "jet"
    )

    heatmap_array = (
        colormap(cam)[..., :3] * 255
    ).astype(
        np.uint8
    )

    overlay_array = np.clip(
        original_array.astype(np.float32) * 0.55
        +
        heatmap_array.astype(np.float32) * 0.45,
        0,
        255,
    ).astype(
        np.uint8
    )

    original_path = (
        directories["originals"]
        / f"{sample_id}_original.png"
    )

    heatmap_path = (
        directories["heatmaps"]
        / f"{sample_id}_heatmap.png"
    )

    overlay_path = (
        directories["overlays"]
        / f"{sample_id}_overlay.png"
    )

    comparison_path = (
        directories["comparisons"]
        / f"{sample_id}_comparison.png"
    )

    array_path = (
        directories["arrays"]
        / f"{sample_id}_cam.npy"
    )

    Image.fromarray(
        original_array
    ).save(
        original_path
    )

    Image.fromarray(
        heatmap_array
    ).save(
        heatmap_path
    )

    Image.fromarray(
        overlay_array
    ).save(
        overlay_path
    )

    np.save(
        array_path,
        cam.astype(np.float32),
    )

    # Three-panel comparison.
    panel_height = height + 35

    comparison = Image.new(
        "RGB",
        (
            width * 3,
            panel_height,
        ),
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

    draw = ImageDraw.Draw(
        comparison
    )

    draw.text(
        (width // 2 - 25, height + 8),
        "Original",
        fill="black",
    )

    draw.text(
        (width + width // 2 - 25, height + 8),
        "Grad-CAM",
        fill="black",
    )

    draw.text(
        (width * 2 + width // 2 - 25, height + 8),
        "Overlay",
        fill="black",
    )

    comparison.save(
        comparison_path
    )

    comparison.close()

    display_image.close()

    return {

        "original_file": str(original_path),

        "heatmap_file": str(heatmap_path),

        "overlay_file": str(overlay_path),

        "comparison_file": str(comparison_path),

        "cam_array_file": str(array_path),
    }


# ============================================================
# CONTACT SHEET
# ============================================================

def create_contact_sheet(
    result_records,
    output_path,
    columns=4,
):

    if not result_records:

        return None

    tile_width = 320
    tile_height = 265

    rows = math.ceil(
        len(result_records) / columns
    )

    sheet = Image.new(
        "RGB",
        (
            columns * tile_width,
            rows * tile_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(
        sheet
    )

    loaded = 0

    for index, record in enumerate(
        result_records
    ):

        overlay_path = Path(
            record["overlay_file"]
        )

        if not overlay_path.is_file():
            continue

        try:

            with Image.open(
                overlay_path
            ) as source:

                image = source.convert("RGB")

                image = resize_display_image(
                    image,
                    300,
                    205,
                )

            column = index % columns
            row = index // columns

            x = column * tile_width
            y = row * tile_height

            sheet.paste(
                image,
                (
                    x + 10,
                    y + 5,
                ),
            )

            label = (
                f"Actual: {record['actual_class']}\n"
                f"Predicted: {record['predicted_class']}\n"
                f"Confidence: {record['confidence']:.3f}\n"
                f"Correct: {record['correct']}"
            )

            draw.text(
                (
                    x + 10,
                    y + 215,
                ),
                label,
                fill="black",
            )

            image.close()

            loaded += 1

        except Exception as error:

            print(
                f"[WARNING] Contact sheet image skipped: {error}"
            )

    if loaded == 0:

        sheet.close()

        return None

    sheet.save(
        output_path,
        quality=92,
    )

    sheet.close()

    return str(output_path)


# ============================================================
# CLASS SUMMARY
# ============================================================

def calculate_class_summary(
    result_records,
):

    summary = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        rows = [
            item
            for item in result_records
            if item["actual_class_index"] == class_index
        ]

        correct = [
            item
            for item in rows
            if item["correct"]
        ]

        incorrect = [
            item
            for item in rows
            if not item["correct"]
        ]

        summary.append({

            "class_index": class_index,

            "class": class_name,

            "selected_images": len(rows),

            "correct_predictions": len(correct),

            "incorrect_predictions": len(incorrect),

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

            "mean_cam_intensity": (
                float(np.mean([
                    item["cam_mean"]
                    for item in rows
                ]))
                if rows else None
            ),

            "mean_cam_std": (
                float(np.mean([
                    item["cam_std"]
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

    return summary


# ============================================================
# TEXT REPORT
# ============================================================

def create_report(
    summary,
    class_summary,
    result_records,
    output_path,
):

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "=" * 78 + "\n"
        )

        file.write(
            "MAIZE LEAF CLASSIFICATION - GRAD-CAM REPORT\n"
        )

        file.write(
            "=" * 78 + "\n\n"
        )

        file.write(
            f"Model: {summary['model_name']}\n"
        )

        file.write(
            f"Run: {summary['run_name']}\n"
        )

        file.write(
            f"Checkpoint: {summary['checkpoint']}\n"
        )

        file.write(
            f"Checkpoint epoch: {summary['checkpoint_epoch']}\n"
        )

        file.write(
            f"Partition: {summary['partition']}\n"
        )

        file.write(
            f"Target layer: {summary['target_layer']}\n"
        )

        file.write(
            f"Target class policy: {summary['target_class_policy']}\n"
        )

        file.write(
            f"Samples explained: {summary['sample_count']}\n\n"
        )

        file.write(
            "OVERALL EXPLANATION SUMMARY\n"
        )

        file.write(
            "-" * 78 + "\n"
        )

        file.write(
            f"Correct predictions: {summary['correct_predictions']}\n"
        )

        file.write(
            f"Incorrect predictions: {summary['incorrect_predictions']}\n"
        )

        file.write(
            f"Sample accuracy: {summary['sample_accuracy']:.6f}\n"
        )

        file.write(
            f"Mean confidence: {summary['mean_confidence']:.6f}\n"
        )

        file.write(
            f"Mean CAM intensity: {summary['mean_cam_intensity']:.6f}\n"
        )

        file.write(
            f"Mean execution time: {summary['mean_runtime_seconds']:.4f} seconds\n\n"
        )

        file.write(
            "CLASS-WISE SUMMARY\n"
        )

        file.write(
            "-" * 78 + "\n"
        )

        for item in class_summary:

            file.write(
                f"\n{item['class']}\n"
            )

            file.write(
                f"  Samples: {item['selected_images']}\n"
            )

            file.write(
                f"  Correct: {item['correct_predictions']}\n"
            )

            file.write(
                f"  Incorrect: {item['incorrect_predictions']}\n"
            )

            file.write(
                f"  Sample accuracy: {item['sample_accuracy']}\n"
            )

            file.write(
                f"  Mean confidence: {item['mean_confidence']}\n"
            )

            file.write(
                f"  Mean CAM intensity: {item['mean_cam_intensity']}\n"
            )

        file.write(
            "\nEXPLAINED SAMPLES\n"
        )

        file.write(
            "-" * 78 + "\n"
        )

        for item in result_records:

            file.write(
                f"\nSample: {item['sample_id']}\n"
            )

            file.write(
                f"Actual: {item['actual_class']}\n"
            )

            file.write(
                f"Predicted: {item['predicted_class']}\n"
            )

            file.write(
                f"Confidence: {item['confidence']:.6f}\n"
            )

            file.write(
                f"Correct: {item['correct']}\n"
            )

            file.write(
                f"Target class: {item['target_class']}\n"
            )

            file.write(
                f"CAM mean: {item['cam_mean']:.6f}\n"
            )

        file.write(
            "\nREPRODUCIBILITY AND DATA INTEGRITY\n"
        )

        file.write(
            "-" * 78 + "\n"
        )

        file.write(
            f"Split fingerprint: {summary['split_fingerprint']}\n"
        )

        file.write(
            f"Random seed: {summary['seed']}\n"
        )

        file.write(
            "Validation partition only: YES\n"
        )

        file.write(
            "Train manifest loaded: NO\n"
        )

        file.write(
            "Held-out test manifest loaded: NO\n"
        )

        file.write(
            "Held-out test partition used: NO\n"
        )

        file.write(
            f"Validation group IDs available: "
            f"{summary['validation_group_ids_available']}\n"
        )

        file.write(
            "Group-level leakage independently verified: NO\n"
        )

        file.write(
            "\nLIMITATIONS\n"
        )

        file.write(
            "-" * 78 + "\n"
        )

        file.write(
            "Grad-CAM is a post-hoc visual explanation.\n"
        )

        file.write(
            "Highlighted regions do not establish biological causality.\n"
        )

        file.write(
            "Visual interpretation requires expert inspection of the original leaf.\n"
        )

        file.write(
            "Heatmap intensity is not a direct measure of disease severity.\n"
        )

        file.write(
            "\n" + "=" * 78 + "\n"
        )


# ============================================================
# MAIN GRAD-CAM ENGINE
# ============================================================

def _run_gradcam_internal(args):

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

        device = torch.device(
            args.device
        )

    if device.type == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA requested but unavailable."
        )

    if device.type == "cuda":

        device = torch.device(
            "cuda:0"
            if args.device == "cuda"
            else args.device
        )

    print_section(
        "MAIZE LEAF - GRAD-CAM EXPLAINABILITY ENGINE"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Run directory: {run_dir}"
    )

    print(
        f"Device       : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU          : {torch.cuda.get_device_name(device)}"
        )

        print(
            f"GPU memory   : "
            f"{torch.cuda.get_device_properties(device).total_memory / (1024 ** 3):.2f} GB"
        )

    # --------------------------------------------------------
    # REPRODUCIBILITY
    # --------------------------------------------------------

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            args.seed
        )

    # --------------------------------------------------------
    # SPLIT VERIFICATION
    # --------------------------------------------------------

    print_section(
        "VERIFYING FIXED DATASET SPLIT"
    )

    split_fingerprint = verify_split(
        run_dir
    )

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    print_section(
        "LOADING TRAINING CONFIGURATION"
    )

    config, config_data, model_name = (
        load_training_configuration(
            run_dir
        )
    )

    print(
        f"Model      : {model_name}"
    )

    print(
        f"Image size : {config.image_size}"
    )

    print(
        f"Classes    : {NUM_CLASSES}"
    )

    if list(CLASS_NAMES) == []:

        raise RuntimeError(
            "Class configuration is empty."
        )

    for index, name in enumerate(CLASS_NAMES):

        print(
            f"{index}: {name}"
        )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print_section(
        "LOADING BEST MODEL"
    )

    model, checkpoint, checkpoint_path = (
        load_best_model(
            run_dir,
            config,
            model_name,
            device,
        )
    )

    # --------------------------------------------------------
    # TARGET LAYER
    # --------------------------------------------------------

    print_section(
        "CONFIGURING GRAD-CAM TARGET LAYER"
    )

    (
        target_layer,
        reshape_transform,
        target_layer_name,
    ) = get_target_configuration(
        model,
        model_name,
    )

    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    print_section(
        "BUILDING VALIDATION DATA"
    )

    validation_records = load_validation_records()

    validation_group_ids_available = any(
        str(record.get("group_id", "")).strip()
        for record in validation_records
    )

    if not validation_group_ids_available:

        print(
            "[INFO] Validation records contain no group IDs. "
            "Group-level leakage cannot be independently verified."
        )

    validation_transform, image_size = (
        build_validation_transform(
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
    # SAMPLE SELECTION
    # --------------------------------------------------------

    print_section(
        "SELECTING VALIDATION SAMPLES"
    )

    prediction_lookup = read_prediction_lookup(
        run_dir
    )

    selected_samples = select_samples(

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
            / "gradcam"
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

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # --------------------------------------------------------
    # GENERATE GRAD-CAM
    # --------------------------------------------------------

    print_section(
        "GENERATING GRAD-CAM EXPLANATIONS"
    )

    results = []

    started = time.perf_counter()

    try:

        from tqdm.auto import tqdm

        iterator = tqdm(
            enumerate(selected_samples, start=1),
            total=len(selected_samples),
            desc="GRAD-CAM",
            dynamic_ncols=True,
        )

    except ImportError:

        iterator = enumerate(
            selected_samples,
            start=1,
        )

    for index, record in iterator:

        sample_started = time.perf_counter()

        image_path = resolve_image_path(
            get_record_image_path(record)
        )

        actual_class = int(
            record["label"]
        )

        actual_class_name = CLASS_NAMES[
            actual_class
        ]

        # Match MaizeLeafDataset.__getitem__ exactly: RGB conversion without
        # an additional EXIF orientation transform.
        with Image.open(image_path) as source:

            original_image = source.convert("RGB")

            # Separate image copy for saved visualization.
            display_original = original_image.copy()

            # Validation preprocessing exactly as training.
            input_tensor = validation_transform(
                original_image
            ).unsqueeze(0).to(device)

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            logits = extract_logits(
                model(input_tensor)
            )

            probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )[0]

            if not torch.isfinite(probabilities).all():

                raise FloatingPointError(
                    f"Non-finite class probabilities for image: {image_path}"
                )

            predicted_class = int(
                probabilities.argmax().item()
            )

            confidence = float(
                probabilities[predicted_class].item()
            )

        if probabilities.numel() != NUM_CLASSES:

            raise RuntimeError(
                "Prediction probability count is not seven."
            )

        correct = (
            actual_class == predicted_class
        )

        if args.target_class == "predicted":

            target_class = predicted_class

        else:

            target_class = actual_class

        target_class_name = CLASS_NAMES[
            target_class
        ]

        # ----------------------------------------------------
        # GRAD-CAM COMPUTATION
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize(
                device
            )

        cam_started = time.perf_counter()

        cam, target_score = compute_gradcam(

            model=model,

            target_layer=target_layer,

            reshape_transform=reshape_transform,

            input_tensor=input_tensor,

            target_class=target_class,
        )

        if device.type == "cuda":

            torch.cuda.synchronize(
                device
            )

        cam_runtime = (
            time.perf_counter() - cam_started
        )

        # ----------------------------------------------------
        # SAMPLE IDENTIFICATION
        # ----------------------------------------------------

        sample_id = (
            f"sample_{index:04d}"
        )

        sample_slug = (
            f"{sample_id}_"
            f"true-{safe_slug(actual_class_name)}_"
            f"pred-{safe_slug(CLASS_NAMES[predicted_class])}"
        )

        # ----------------------------------------------------
        # SAVE VISUALIZATIONS
        # ----------------------------------------------------

        visual_files = save_visualization(

            original_image=display_original,

            cam=cam,

            sample_id=sample_slug,

            directories=directories,
        )

        display_original.close()

        # ----------------------------------------------------
        # RESULT RECORD
        # ----------------------------------------------------

        record_result = {

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

            "target_class": target_class_name,

            "target_class_policy": args.target_class,

            "confidence": confidence,

            "target_logit": target_score,

            "correct": correct,

            "cam_mean": float(cam.mean()),

            "cam_std": float(cam.std()),

            "cam_max": float(cam.max()),

            "cam_active_fraction": float(
                np.mean(cam > 0.5)
            ),

            "runtime_seconds": float(
                cam_runtime
            ),

            "sample_runtime_seconds": float(
                time.perf_counter() - sample_started
            ),

            "target_layer": target_layer_name,

            **visual_files,
        }

        results.append(
            record_result
        )

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

    total_runtime = (
        time.perf_counter() - started
    )

    # --------------------------------------------------------
    # CLASS-WISE SUMMARY
    # --------------------------------------------------------

    print_section(
        "GENERATING GRAD-CAM REPORTS"
    )

    correct_count = sum(
        item["correct"]
        for item in results
    )

    incorrect_count = (
        len(results) - correct_count
    )

    class_summary = calculate_class_summary(
        results
    )

    mean_confidence = float(
        np.mean([
            item["confidence"]
            for item in results
        ])
    )

    mean_cam_intensity = float(
        np.mean([
            item["cam_mean"]
            for item in results
        ])
    )

    mean_runtime = float(
        np.mean([
            item["runtime_seconds"]
            for item in results
        ])
    )

    # --------------------------------------------------------
    # SAVE CSV FILES
    # --------------------------------------------------------

    save_csv(
        results,
        output_dir / "gradcam_results.csv",
    )

    print(
        "[PASS] gradcam_results.csv"
    )

    save_csv(
        class_summary,
        output_dir / "class_summary.csv",
    )

    print(
        "[PASS] class_summary.csv"
    )

    selected_rows = []

    for index, record in enumerate(
        selected_samples,
        start=1,
    ):

        selected_rows.append({

            "sample_id": f"sample_{index:04d}",

            "image_path": str(
                resolve_image_path(
                    get_record_image_path(record)
                )
            ),

            "relative_path": str(record.get("relative_path", "")),

            "group_id": str(record.get("group_id", "")),

            "split": str(record.get("split", "validation")),

            "actual_class": CLASS_NAMES[
                int(record["label"])
            ],

            "actual_class_index": int(
                record["label"]
            ),
        })

    save_csv(
        selected_rows,
        output_dir / "selected_samples.csv",
    )

    print(
        "[PASS] selected_samples.csv"
    )

    # --------------------------------------------------------
    # CONTACT SHEET
    # --------------------------------------------------------

    contact_sheet_path = None

    if not args.no_contact_sheet:

        contact_sheet_path = create_contact_sheet(

            results,

            output_dir / "gradcam_contact_sheet.jpg",

            columns=4,
        )

        if contact_sheet_path:

            print(
                "[PASS] gradcam_contact_sheet.jpg"
            )

    # --------------------------------------------------------
    # SUMMARY JSON
    # --------------------------------------------------------

    summary = {

        "status": "COMPLETED",

        "method": "Grad-CAM",

        "model_name": model_name,

        "run_name": run_dir.name,

        "checkpoint": str(checkpoint_path),

        "checkpoint_epoch": checkpoint.get(
            "epoch"
        ),

        "checkpoint_sha256": sha256_file(
            checkpoint_path
        ),

        "partition": "validation",

        "validation_images_available": len(
            validation_records
        ),

        "validation_manifest_only": True,

        "train_manifest_loaded": False,

        "held_out_test_manifest_loaded": False,

        "held_out_test_accessed": False,

        "validation_group_ids_available": validation_group_ids_available,

        "group_level_leakage_verified": False,

        "sample_count": len(results),

        "samples_per_class_requested": (
            args.samples_per_class
        ),

        "sample_strategy": args.sample_strategy,

        "target_class_policy": args.target_class,

        "target_layer": target_layer_name,

        "class_names": list(CLASS_NAMES),

        "correct_predictions": correct_count,

        "incorrect_predictions": incorrect_count,

        "sample_accuracy": (
            correct_count / len(results)
        ),

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

        "split_fingerprint": split_fingerprint,

        "class_summary": class_summary,

        "contact_sheet": contact_sheet_path,

        "output_directory": str(
            output_dir
        ),

        "held_out_test_used": False,

        "interpretation": (
            "Grad-CAM highlights spatial features that contribute "
            "to the selected class score. It does not establish "
            "biological causality."
        ),

        "limitations": [

            "Grad-CAM is a post-hoc explanation.",

            "Heatmap intensity does not measure disease severity.",

            "Highlighted regions do not prove biological causality.",

            "Visual interpretation requires expert inspection.",

            "The explained sample subset is not a substitute "
            "for full-dataset classification evaluation.",
        ],
    }

    save_json(
        summary,
        output_dir / "gradcam_summary.json",
    )

    print(
        "[PASS] gradcam_summary.json"
    )

    # --------------------------------------------------------
    # TEXT REPORT
    # --------------------------------------------------------

    create_report(

        summary=summary,

        class_summary=class_summary,

        result_records=results,

        output_path=(
            output_dir / "gradcam_report.txt"
        ),
    )

    print(
        "[PASS] gradcam_report.txt"
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print_section(
        "GENERATING GRAD-CAM SUCCESSFUL"
    )

    print(
        f"Model                : {model_name}"
    )

    print(
        f"Target layer         : {target_layer_name}"
    )

    print(
        f"Validation images    : {len(validation_records)}"
    )

    print(
        f"Samples explained    : {len(results)}"
    )

    print(
        f"Correct predictions  : {correct_count}"
    )

    print(
        f"Incorrect predictions: {incorrect_count}"
    )

    print(
        f"Sample accuracy      : "
        f"{correct_count / len(results):.4f}"
    )

    print(
        f"Mean confidence      : {mean_confidence:.4f}"
    )

    print(
        f"Mean CAM intensity   : {mean_cam_intensity:.4f}"
    )

    print(
        f"Mean CAM runtime     : {mean_runtime:.4f} seconds"
    )

    print(
        f"Total runtime        : {total_runtime:.2f} seconds"
    )

    print(
        f"Train manifest read  : NO"
    )

    print(
        f"Test manifest read   : NO"
    )

    print(
        f"Test set used        : NO"
    )

    print(
        f"Output directory     : {output_dir}"
    )

    print("=" * 76)

    return summary




def run_gradcam(args):
    """
    Run Grad-CAM with a minimal terminal interface.

    Routine setup, per-image messages, and report messages are hidden.
    The single image-level tqdm progress bar remains visible because
    tqdm writes to stderr. The two requested headings are written
    directly to the real terminal. Exceptions remain visible.
    """

    with contextlib.redirect_stdout(io.StringIO()):
        return _run_gradcam_internal(args)

# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Maize Leaf Classification - Grad-CAM Engine"
        )
    )

    parser.add_argument(

        "--run-dir",

        type=Path,

        required=True,

        help="Completed model run directory.",
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

        "--no-contact-sheet",

        action="store_true",
    )

    args = parser.parse_args()

    run_gradcam(
        args
    )


if __name__ == "__main__":

    main()