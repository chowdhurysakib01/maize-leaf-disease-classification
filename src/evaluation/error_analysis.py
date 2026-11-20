"""
MAIZE LEAF CLASSIFICATION
ERROR ANALYSIS ENGINE

File:
src/evaluation/error_analysis.py

Features:
1. Overall prediction analysis
2. Class-wise precision, recall and F1
3. False positives and false negatives
4. Confusion-pair analysis
5. High-confidence wrong predictions
6. Confidence distribution
7. Expected calibration error (ECE)
8. Misclassified image export
9. Misclassified image contact sheet
10. Confusion matrix visualizations
11. CSV, JSON, PNG and TXT reports

Input:
evaluation/<split>/predictions.csv
evaluation/<split>/predictions_metadata.json

The original dataset and fixed split are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import re
import sys

from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageOps


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


NUM_CLASSES = 7

EXPECTED_SPLIT_FINGERPRINT = (
    "8a64e1a8a53eecd2dd2eeedbfb63c7e783a193304a2529da1be4e8c7bff441de"
)


DEFAULT_CLASS_NAMES = [
    "Fall Armyworm (Insect) Damage",
    "Healthy Leaf",
    "Leaf Spot",
    "Magnesium Deficiency",
    "Nitrogen Deficiency",
    "Northern Leaf Blight",
    "Phosphorus Deficiency",
]


# ============================================================
# GENERAL UTILITIES
# ============================================================

def write_json(data, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

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


def write_csv(rows, path, fieldnames=None):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:

        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
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


def normalize_name(value):

    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).strip().lower(),
    )


def safe_float(value, default=None):

    if value is None or str(value).strip() == "":
        return default

    try:

        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (ValueError, TypeError):

        return default


def safe_int(value, default=None):

    number = safe_float(value)

    if number is None:
        return default

    if not number.is_integer():
        return default

    return int(number)


def first_nonempty(row, keys):

    for key in keys:

        value = row.get(key)

        if value is not None and str(value).strip() != "":
            return value

    return None


def parse_boolean(value):

    if isinstance(value, bool):
        return value

    if value is None:
        return None

    value = str(value).strip().lower()

    if value in ("true", "1", "yes", "correct"):
        return True

    if value in ("false", "0", "no", "incorrect"):
        return False

    return None


# ============================================================
# CLASS CONFIGURATION
# ============================================================

def load_project_class_names():

    try:

        module = importlib.import_module(
            "src.data.class_names"
        )

        candidates = [
            "CLASS_NAMES",
            "CLASS_NAMES_LIST",
            "CLASSES",
            "CLASS_LABELS",
            "CLASS_TO_IDX",
        ]

        for attribute in candidates:

            value = getattr(
                module,
                attribute,
                None,
            )

            if isinstance(value, (list, tuple)):

                names = list(value)

                if len(names) == NUM_CLASSES:
                    return [str(x) for x in names]

            if isinstance(value, dict):

                try:

                    names = [
                        str(value[i])
                        for i in range(NUM_CLASSES)
                    ]

                    return names

                except (KeyError, TypeError):
                    pass

    except (ImportError, AttributeError):
        pass

    return DEFAULT_CLASS_NAMES.copy()


def normalize_class_names(value):

    if isinstance(value, dict):

        try:

            return [
                str(value[str(i)] if str(i) in value else value[i])
                for i in range(NUM_CLASSES)
            ]

        except (KeyError, TypeError):
            return None

    if isinstance(value, (list, tuple)):

        names = [str(item) for item in value]

        if len(names) == NUM_CLASSES:
            return names

    return None


def find_metadata_class_names(data):

    if isinstance(data, dict):

        for key in (
            "class_names",
            "class_labels",
            "classes",
        ):

            if key in data:

                names = normalize_class_names(
                    data[key]
                )

                if names is not None:
                    return names

        for value in data.values():

            result = find_metadata_class_names(
                value
            )

            if result is not None:
                return result

    elif isinstance(data, list):

        for value in data:

            result = find_metadata_class_names(
                value
            )

            if result is not None:
                return result

    return None


def create_class_mapping(class_names):

    if len(class_names) != NUM_CLASSES:

        raise ValueError(
            f"Expected exactly {NUM_CLASSES} classes, "
            f"received {len(class_names)}."
        )

    mapping = {}

    for index, name in enumerate(class_names):

        mapping[normalize_name(name)] = index

    return mapping


def parse_class_index(
    index_value,
    class_value,
    class_mapping,
):

    if index_value is not None:

        index = safe_int(index_value)

        if index is None:

            raise ValueError(
                f"Invalid class index: {index_value}"
            )

        return index

    if class_value is None:

        raise ValueError(
            "Neither class index nor class name is available."
        )

    numeric_index = safe_int(class_value)

    if numeric_index is not None:
        return numeric_index

    normalized = normalize_name(class_value)

    if normalized not in class_mapping:

        raise ValueError(
            f"Unknown class name: {class_value}"
        )

    return class_mapping[normalized]


# ============================================================
# METADATA AND FINGERPRINT VERIFICATION
# ============================================================

def load_metadata(path):

    if not path.is_file():
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def find_split_fingerprints(data):

    found = []

    if isinstance(data, dict):

        for key, value in data.items():

            key_lower = key.lower()

            if (
                "fingerprint" in key_lower
                and "split" in key_lower
                and isinstance(value, str)
            ):

                found.append(value)

            found.extend(
                find_split_fingerprints(value)
            )

    elif isinstance(data, list):

        for value in data:

            found.extend(
                find_split_fingerprints(value)
            )

    return found


def verify_split_fingerprint(
    metadata,
    evaluation_dir,
):

    sources = []

    if metadata:
        sources.append(metadata)

    run_dir = evaluation_dir.parent.parent

    for filename in (
        "run_summary.json",
        "experiment_metadata.json",
        "split_metadata_snapshot.json",
    ):

        path = run_dir / filename

        if path.is_file():

            try:

                sources.append(
                    load_metadata(path)
                )

            except (OSError, ValueError):
                pass

    fingerprints = []

    for source in sources:

        fingerprints.extend(
            find_split_fingerprints(source)
        )

    if not fingerprints:

        print(
            "[WARNING] Split fingerprint not found "
            "in prediction metadata."
        )

        return False

    mismatches = [
        value
        for value in fingerprints
        if value != EXPECTED_SPLIT_FINGERPRINT
    ]

    if mismatches:

        raise RuntimeError(
            "Split fingerprint mismatch. "
            "Error analysis aborted to protect "
            "research reproducibility."
        )

    print(
        "[PASS] Fixed split fingerprint verified"
    )

    return True


# ============================================================
# PREDICTION CSV LOADING
# ============================================================

def get_probability_column(
    row,
    class_index,
    class_name,
):

    candidates = [
        f"prob_class_{class_index}",
        f"probability_{class_index}",
        f"probability_{class_name}",
        f"prob_{class_name}",
    ]

    for key in candidates:

        if key in row:

            value = safe_float(
                row[key]
            )

            if value is not None:
                return value

    return None


def load_predictions(
    csv_path,
    class_names,
):

    if not csv_path.is_file():

        raise FileNotFoundError(
            f"Prediction CSV not found: {csv_path}"
        )

    class_mapping = create_class_mapping(
        class_names
    )

    predictions = []

    probability_columns = []

    supplied_correct_mismatches = 0

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames:

            raise ValueError(
                "Prediction CSV has no header."
            )

        headers = set(reader.fieldnames)

        probability_columns = [
            key
            for key in (
                [
                    f"prob_class_{i}"
                    for i in range(NUM_CLASSES)
                ]
                +
                [
                    f"probability_{i}"
                    for i in range(NUM_CLASSES)
                ]
            )
            if key in headers
        ]

        if len(probability_columns) not in (
            0,
            NUM_CLASSES,
        ):

            raise ValueError(
                "Incomplete probability columns. "
                "Expected all seven class probability columns."
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):

            true_index_value = first_nonempty(
                row,
                [
                    "true_index",
                    "actual_class_index",
                    "true_class_index",
                    "label_index",
                    "target",
                ],
            )

            predicted_index_value = first_nonempty(
                row,
                [
                    "predicted_index",
                    "predicted_class_index",
                    "prediction_index",
                    "pred_index",
                ],
            )

            true_class_value = first_nonempty(
                row,
                [
                    "true_class",
                    "actual_class",
                    "true_label",
                    "actual_label",
                ],
            )

            predicted_class_value = first_nonempty(
                row,
                [
                    "predicted_class",
                    "predicted_label",
                    "prediction",
                ],
            )

            true_index = parse_class_index(
                true_index_value,
                true_class_value,
                class_mapping,
            )

            predicted_index = parse_class_index(
                predicted_index_value,
                predicted_class_value,
                class_mapping,
            )

            if not 0 <= true_index < NUM_CLASSES:

                raise ValueError(
                    f"Invalid true class index at CSV row "
                    f"{row_number}: {true_index}"
                )

            if not 0 <= predicted_index < NUM_CLASSES:

                raise ValueError(
                    f"Invalid predicted class index at CSV row "
                    f"{row_number}: {predicted_index}"
                )

            probabilities = []

            for class_index in range(NUM_CLASSES):

                probability = get_probability_column(
                    row,
                    class_index,
                    class_names[class_index],
                )

                probabilities.append(
                    probability
                )

            probability_available = all(
                value is not None
                for value in probabilities
            )

            if probability_available:

                if any(
                    value < 0 or value > 1
                    for value in probabilities
                ):

                    raise ValueError(
                        f"Probability outside [0,1] "
                        f"at CSV row {row_number}."
                    )

                probability_sum = sum(
                    probabilities
                )

                if not np.isclose(
                    probability_sum,
                    1.0,
                    atol=0.02,
                ):

                    raise ValueError(
                        f"Class probabilities do not sum "
                        f"to approximately 1 at CSV row {row_number}: "
                        f"{probability_sum}"
                    )

                calculated_confidence = max(
                    probabilities
                )

            else:

                probability_sum = None

                calculated_confidence = None

            confidence = safe_float(
                first_nonempty(
                    row,
                    [
                        "confidence",
                        "max_probability",
                        "prediction_confidence",
                    ],
                ),
                default=calculated_confidence,
            )

            if confidence is not None:

                if not 0 <= confidence <= 1:

                    raise ValueError(
                        f"Invalid confidence at CSV row {row_number}."
                    )

            correct = (
                true_index == predicted_index
            )

            supplied_correct = parse_boolean(
                row.get("correct")
            )

            if (
                supplied_correct is not None
                and supplied_correct != correct
            ):

                supplied_correct_mismatches += 1

            image_path = first_nonempty(
                row,
                [
                    "image_path",
                    "file_path",
                    "filepath",
                    "path",
                ],
            )

            predictions.append({

                "sample_index": len(predictions),

                "image_path": (
                    str(image_path)
                    if image_path is not None
                    else ""
                ),

                "true_index": true_index,

                "true_class": class_names[true_index],

                "predicted_index": predicted_index,

                "predicted_class": class_names[
                    predicted_index
                ],

                "confidence": confidence,

                "correct": correct,

                "probabilities": probabilities,

                "probability_available": (
                    probability_available
                ),

                "probability_sum": probability_sum,
            })

    if not predictions:

        raise ValueError(
            "Prediction CSV contains no prediction records."
        )

    return (
        predictions,
        len(probability_columns) == NUM_CLASSES,
        supplied_correct_mismatches,
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def calculate_confusion_matrix(predictions):

    matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=np.int64,
    )

    for prediction in predictions:

        matrix[
            prediction["true_index"],
            prediction["predicted_index"],
        ] += 1

    return matrix


def calculate_class_metrics(
    matrix,
    class_names,
):

    total = int(matrix.sum())

    report = []

    for index, class_name in enumerate(class_names):

        true_positive = int(
            matrix[index, index]
        )

        support = int(
            matrix[index, :].sum()
        )

        predicted_count = int(
            matrix[:, index].sum()
        )

        false_positive = (
            predicted_count - true_positive
        )

        false_negative = (
            support - true_positive
        )

        true_negative = (
            total
            - true_positive
            - false_positive
            - false_negative
        )

        precision = (
            true_positive / predicted_count
            if predicted_count > 0
            else 0.0
        )

        recall = (
            true_positive / support
            if support > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        error_rate = (
            false_negative / support
            if support > 0
            else 0.0
        )

        row_without_diagonal = (
            matrix[index].copy()
        )

        row_without_diagonal[index] = 0

        most_confused_index = int(
            np.argmax(row_without_diagonal)
        )

        most_confused_count = int(
            row_without_diagonal[
                most_confused_index
            ]
        )

        if most_confused_count > 0:

            most_confused_with = class_names[
                most_confused_index
            ]

        else:

            most_confused_with = "None"

        report.append({

            "class_index": index,

            "class": class_name,

            "support": support,

            "correct": true_positive,

            "incorrect": false_negative,

            "accuracy_recall": recall,

            "precision": precision,

            "recall": recall,

            "f1_score": f1,

            "false_positive": false_positive,

            "false_negative": false_negative,

            "true_negative": true_negative,

            "error_rate": error_rate,

            "most_confused_with": most_confused_with,

            "confusion_count": most_confused_count,
        })

    return report


# ============================================================
# CONFUSION PAIR ANALYSIS
# ============================================================

def calculate_confusion_pairs(
    matrix,
    class_names,
):

    pairs = []

    for true_index in range(NUM_CLASSES):

        for predicted_index in range(NUM_CLASSES):

            if true_index == predicted_index:
                continue

            count = int(
                matrix[
                    true_index,
                    predicted_index,
                ]
            )

            if count == 0:
                continue

            support = int(
                matrix[true_index].sum()
            )

            error_percentage = (
                count / support * 100
                if support > 0
                else 0.0
            )

            pairs.append({

                "actual_class": class_names[
                    true_index
                ],

                "predicted_class": class_names[
                    predicted_index
                ],

                "actual_class_index": true_index,

                "predicted_class_index": predicted_index,

                "confusion_count": count,

                "percentage_of_actual_class": (
                    error_percentage
                ),
            })

    pairs.sort(
        key=lambda item: item["confusion_count"],
        reverse=True,
    )

    return pairs


# ============================================================
# CONFIDENCE AND CALIBRATION ANALYSIS
# ============================================================

def calculate_confidence_analysis(
    predictions,
):

    available = [
        item
        for item in predictions
        if item["confidence"] is not None
    ]

    if not available:

        return {
            "available": False,
            "reason": "No confidence values available.",
            "confidence_bins": [],
            "ece": None,
        }

    bins = np.linspace(
        0.0,
        1.0,
        11,
    )

    bin_results = []

    ece = 0.0

    total = len(available)

    for index in range(10):

        lower = float(bins[index])
        upper = float(bins[index + 1])

        if index == 9:

            selected = [
                item
                for item in available
                if lower <= item["confidence"] <= upper
            ]

        else:

            selected = [
                item
                for item in available
                if lower <= item["confidence"] < upper
            ]

        count = len(selected)

        if count:

            mean_confidence = float(
                np.mean([
                    item["confidence"]
                    for item in selected
                ])
            )

            accuracy = float(
                np.mean([
                    item["correct"]
                    for item in selected
                ])
            )

            ece += (
                count / total
            ) * abs(
                accuracy - mean_confidence
            )

        else:

            mean_confidence = None
            accuracy = None

        bin_results.append({

            "bin": index + 1,

            "confidence_lower": lower,

            "confidence_upper": upper,

            "sample_count": count,

            "mean_confidence": mean_confidence,

            "accuracy": accuracy,
        })

    correct_confidence = [
        item["confidence"]
        for item in available
        if item["correct"]
    ]

    incorrect_confidence = [
        item["confidence"]
        for item in available
        if not item["correct"]
    ]

    return {

        "available": True,

        "sample_count": total,

        "mean_confidence": float(
            np.mean([
                item["confidence"]
                for item in available
            ])
        ),

        "mean_correct_confidence": (
            float(np.mean(correct_confidence))
            if correct_confidence
            else None
        ),

        "mean_incorrect_confidence": (
            float(np.mean(incorrect_confidence))
            if incorrect_confidence
            else None
        ),

        "ece": float(ece),

        "confidence_bins": bin_results,
    }


# ============================================================
# MISCLASSIFIED PREDICTIONS
# ============================================================

def prepare_error_records(
    predictions,
    class_names,
    confidence_threshold,
):

    errors = [
        item
        for item in predictions
        if not item["correct"]
    ]

    errors.sort(
        key=lambda item: (
            item["confidence"]
            if item["confidence"] is not None
            else -1
        ),
        reverse=True,
    )

    error_records = []

    for item in errors:

        record = {

            "sample_index": item["sample_index"],

            "image_path": item["image_path"],

            "actual_class": item["true_class"],

            "predicted_class": item["predicted_class"],

            "actual_class_index": item["true_index"],

            "predicted_class_index": item["predicted_index"],

            "confidence": item["confidence"],

            "error_type": (
                "False Negative for actual class; "
                "False Positive for predicted class"
            ),
        }

        for index, class_name in enumerate(class_names):

            record[
                f"prob_class_{index}"
            ] = item["probabilities"][index]

        error_records.append(record)

    high_confidence_errors = [
        item
        for item in error_records
        if (
            item["confidence"] is not None
            and item["confidence"] >= confidence_threshold
        )
    ]

    return (
        error_records,
        high_confidence_errors,
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_confusion_matrix(
    matrix,
    class_names,
    output_path,
    normalized=False,
):

    if normalized:

        row_sums = matrix.sum(
            axis=1,
            keepdims=True,
        )

        display_matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(
                matrix,
                dtype=float,
            ),
            where=row_sums != 0,
        ) * 100

    else:

        display_matrix = matrix

    fig, ax = plt.subplots(
        figsize=(13, 11)
    )

    image = ax.imshow(
        display_matrix,
        cmap="Blues",
        aspect="auto",
    )

    fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    ax.set_xticks(
        range(NUM_CLASSES)
    )

    ax.set_yticks(
        range(NUM_CLASSES)
    )

    ax.set_xticklabels(
        class_names,
        rotation=45,
        ha="right",
        fontsize=9,
    )

    ax.set_yticklabels(
        class_names,
        fontsize=9,
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "Actual class"
    )

    ax.set_title(
        "Normalized Confusion Matrix (%)"
        if normalized
        else "Confusion Matrix"
    )

    threshold = (
        display_matrix.max() / 2
        if display_matrix.size
        else 0
    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            value = display_matrix[i, j]

            text = (
                f"{value:.1f}%"
                if normalized
                else f"{int(value)}"
            )

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_class_errors(
    class_report,
    output_path,
):

    names = [
        item["class"]
        for item in class_report
    ]

    error_rates = [
        item["error_rate"] * 100
        for item in class_report
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars = ax.bar(
        names,
        error_rates,
    )

    ax.set_title(
        "Class-wise Misclassification Rate"
    )

    ax.set_ylabel(
        "Error rate (%)"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylim(
        0,
        max(100, max(error_rates, default=0) * 1.15),
    )

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    for bar, value in zip(
        bars,
        error_rates,
    ):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_confusion_pairs(
    pairs,
    output_path,
    top_k,
):

    selected = pairs[:top_k]

    if not selected:
        return False

    labels = [
        (
            f"{item['actual_class']}\n"
            f"→ {item['predicted_class']}"
        )
        for item in selected
    ]

    values = [
        item["confusion_count"]
        for item in selected
    ]

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    bars = ax.barh(
        labels[::-1],
        values[::-1],
    )

    ax.set_title(
        f"Top {len(selected)} Confused Class Pairs"
    )

    ax.set_xlabel(
        "Number of incorrect predictions"
    )

    for bar, value in zip(
        bars,
        values[::-1],
    ):

        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value}",
            va="center",
        )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


def plot_confidence_distribution(
    predictions,
    output_path,
):

    correct = [
        item["confidence"]
        for item in predictions
        if item["correct"]
        and item["confidence"] is not None
    ]

    incorrect = [
        item["confidence"]
        for item in predictions
        if not item["correct"]
        and item["confidence"] is not None
    ]

    if not correct and not incorrect:
        return False

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    bins = np.linspace(
        0,
        1,
        21,
    )

    if correct:

        ax.hist(
            correct,
            bins=bins,
            alpha=0.65,
            label="Correct predictions",
        )

    if incorrect:

        ax.hist(
            incorrect,
            bins=bins,
            alpha=0.65,
            label="Incorrect predictions",
        )

    ax.set_title(
        "Confidence Distribution"
    )

    ax.set_xlabel(
        "Prediction confidence"
    )

    ax.set_ylabel(
        "Number of images"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


def plot_reliability_diagram(
    confidence_analysis,
    output_path,
):

    if not confidence_analysis["available"]:
        return False

    bins = confidence_analysis[
        "confidence_bins"
    ]

    valid = [
        item
        for item in bins
        if item["sample_count"] > 0
    ]

    if not valid:
        return False

    confidence = [
        item["mean_confidence"]
        for item in valid
    ]

    accuracy = [
        item["accuracy"]
        for item in valid
    ]

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.plot(
        confidence,
        accuracy,
        marker="o",
        label="Model calibration",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel(
        "Mean confidence"
    )

    ax.set_ylabel(
        "Observed accuracy"
    )

    ax.set_title(
        "Confidence Reliability Diagram"
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    return True


# ============================================================
# MISCLASSIFIED IMAGE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    error_records,
    output_path,
    limit=48,
    columns=4,
):

    if limit <= 0:
        return None

    if not error_records:
        return None

    tile_width = 340
    tile_height = 260

    selected = error_records[:limit]

    rows = math.ceil(
        len(selected) / columns
    )

    sheet = Image.new(
        "RGB",
        (
            columns * tile_width,
            rows * tile_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(sheet)

    loaded = 0

    resampling = getattr(
        Image,
        "Resampling",
        Image,
    ).LANCZOS

    for position, record in enumerate(
        selected
    ):

        image_path = record["image_path"]

        if not image_path:
            continue

        path = Path(image_path)

        if not path.is_absolute():

            candidate = PROJECT_ROOT / path

            if candidate.is_file():
                path = candidate

        if not path.is_file():
            continue

        try:

            with Image.open(path) as source:

                image = source.convert("RGB")

                image = ImageOps.contain(
                    image,
                    (
                        tile_width - 20,
                        175,
                    ),
                    method=resampling,
                )

            column = position % columns
            row = position // columns

            x = column * tile_width
            y = row * tile_height

            image_x = (
                x
                + (tile_width - image.width) // 2
            )

            sheet.paste(
                image,
                (
                    image_x,
                    y + 5,
                ),
            )

            confidence = record["confidence"]

            confidence_text = (
                f"{confidence:.3f}"
                if confidence is not None
                else "N/A"
            )

            text = (
                f"Actual: {record['actual_class']}\n"
                f"Predicted: {record['predicted_class']}\n"
                f"Confidence: {confidence_text}"
            )

            draw.text(
                (
                    x + 8,
                    y + 185,
                ),
                text,
                fill="black",
            )

            loaded += 1

        except Exception:
            continue

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
# SUMMARY REPORT
# ============================================================

def create_text_report(
    summary,
    class_report,
    confusion_pairs,
    error_records,
    high_confidence_errors,
    output_path,
):

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("=" * 88 + "\n")

        file.write(
            "MAIZE LEAF CLASSIFICATION - ERROR ANALYSIS REPORT\n"
        )

        file.write("=" * 88 + "\n\n")

        file.write(
            f"Model: {summary['model_name']}\n"
        )

        file.write(
            f"Run: {summary['run_name']}\n"
        )

        file.write(
            f"Split: {summary['split_name']}\n"
        )

        file.write(
            f"Timestamp: {summary['timestamp']}\n\n"
        )

        file.write("OVERALL PERFORMANCE\n")
        file.write("-" * 88 + "\n")

        file.write(
            f"Total predictions: {summary['total_predictions']}\n"
        )

        file.write(
            f"Correct: {summary['correct_predictions']}\n"
        )

        file.write(
            f"Incorrect: {summary['incorrect_predictions']}\n"
        )

        file.write(
            f"Accuracy: {summary['accuracy']:.6f}\n"
        )

        file.write(
            f"Error rate: {summary['error_rate']:.6f}\n"
        )

        file.write(
            f"High-confidence errors: "
            f"{summary['high_confidence_error_count']}\n\n"
        )

        file.write("CLASS-WISE ERROR ANALYSIS\n")
        file.write("-" * 88 + "\n")

        for item in class_report:

            file.write(
                f"\n{item['class']}\n"
            )

            file.write(
                f"  Support: {item['support']}\n"
            )

            file.write(
                f"  Correct: {item['correct']}\n"
            )

            file.write(
                f"  Incorrect: {item['incorrect']}\n"
            )

            file.write(
                f"  Precision: {item['precision']:.4f}\n"
            )

            file.write(
                f"  Recall: {item['recall']:.4f}\n"
            )

            file.write(
                f"  F1: {item['f1_score']:.4f}\n"
            )

            file.write(
                f"  False positives: {item['false_positive']}\n"
            )

            file.write(
                f"  False negatives: {item['false_negative']}\n"
            )

            file.write(
                f"  Error rate: {item['error_rate']:.4f}\n"
            )

            file.write(
                f"  Most confused with: "
                f"{item['most_confused_with']}\n"
            )

        file.write(
            "\nTOP CONFUSION PAIRS\n"
        )

        file.write("-" * 88 + "\n")

        for index, item in enumerate(
            confusion_pairs[:10],
            start=1,
        ):

            file.write(
                f"{index}. {item['actual_class']} -> "
                f"{item['predicted_class']}: "
                f"{item['confusion_count']} images "
                f"({item['percentage_of_actual_class']:.2f}%)\n"
            )

        file.write(
            "\nHIGH-CONFIDENCE INCORRECT PREDICTIONS\n"
        )

        file.write("-" * 88 + "\n")

        for item in high_confidence_errors[:20]:

            file.write(
                f"\nActual: {item['actual_class']}\n"
            )

            file.write(
                f"Predicted: {item['predicted_class']}\n"
            )

            file.write(
                f"Confidence: {item['confidence']}\n"
            )

            file.write(
                f"Image: {item['image_path']}\n"
            )

        confidence = summary["confidence_analysis"]

        file.write(
            "\nCONFIDENCE ANALYSIS\n"
        )

        file.write("-" * 88 + "\n")

        if confidence["available"]:

            file.write(
                f"Mean confidence: "
                f"{confidence['mean_confidence']:.6f}\n"
            )

            file.write(
                f"Mean correct confidence: "
                f"{confidence['mean_correct_confidence']}\n"
            )

            file.write(
                f"Mean incorrect confidence: "
                f"{confidence['mean_incorrect_confidence']}\n"
            )

            file.write(
                f"Expected Calibration Error: "
                f"{confidence['ece']:.6f}\n"
            )

        else:

            file.write(
                "Confidence information unavailable.\n"
            )

        file.write(
            "\nDATA INTEGRITY\n"
        )

        file.write("-" * 88 + "\n")

        file.write(
            f"Prediction CSV SHA256: "
            f"{summary['prediction_csv_sha256']}\n"
        )

        file.write(
            f"Split fingerprint verified: "
            f"{summary['split_fingerprint_verified']}\n"
        )

        file.write(
            f"Correct-field mismatches: "
            f"{summary['correct_field_mismatches']}\n"
        )

        file.write(
            "\n" + "=" * 88 + "\n"
        )

        file.write(
            "END OF ERROR ANALYSIS REPORT\n"
        )

        file.write(
            "=" * 88 + "\n"
        )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def run_error_analysis(args):

    evaluation_dir = Path(
        args.evaluation_dir
    ).expanduser().resolve()

    if not evaluation_dir.is_dir():

        raise NotADirectoryError(
            evaluation_dir
        )

    if args.top_k < 1:

        raise ValueError(
            "top-k must be at least 1."
        )

    if not 0 <= args.confidence_threshold <= 1:

        raise ValueError(
            "Confidence threshold must be between 0 and 1."
        )

    prediction_path = (
        evaluation_dir / "predictions.csv"
    )

    metadata_path = (
        evaluation_dir / "predictions_metadata.json"
    )

    metadata = load_metadata(
        metadata_path
    )

    print("\n" + "=" * 76)

    print(
        "MAIZE LEAF - ERROR ANALYSIS ENGINE"
    )

    print("=" * 76)

    print(
        f"Evaluation directory: {evaluation_dir}"
    )

    # Fingerprint verification
    fingerprint_verified = verify_split_fingerprint(
        metadata,
        evaluation_dir,
    )

    # Class mapping
    class_names = (
        find_metadata_class_names(metadata)
        or load_project_class_names()
    )

    if len(class_names) != NUM_CLASSES:

        raise RuntimeError(
            "Class configuration mismatch. "
            "Exactly seven classes are required."
        )

    print(
        f"\nNumber of classes: {NUM_CLASSES}"
    )

    for index, name in enumerate(class_names):

        print(
            f"{index}: {name}"
        )

    # Load predictions
    print(
        "\nLOADING PREDICTIONS"
    )

    (
        predictions,
        probabilities_available,
        correct_field_mismatches,
    ) = load_predictions(
        prediction_path,
        class_names,
    )

    total = len(predictions)

    correct_count = sum(
        item["correct"]
        for item in predictions
    )

    incorrect_count = (
        total - correct_count
    )

    accuracy = (
        correct_count / total
    )

    error_rate = (
        incorrect_count / total
    )

    print(
        f"[PASS] Predictions loaded: {total}"
    )

    print(
        f"[PASS] Correct: {correct_count}"
    )

    print(
        f"[PASS] Incorrect: {incorrect_count}"
    )

    print(
        f"[PASS] Accuracy: {accuracy:.6f}"
    )

    if probabilities_available:

        print(
            "[PASS] Seven-class probability verification"
        )

    else:

        print(
            "[WARNING] Probability columns unavailable."
        )

    if correct_field_mismatches:

        print(
            "[WARNING] Supplied correct-field mismatches: "
            f"{correct_field_mismatches}"
        )

    # Confusion matrix
    matrix = calculate_confusion_matrix(
        predictions
    )

    class_report = calculate_class_metrics(
        matrix,
        class_names,
    )

    confusion_pairs = calculate_confusion_pairs(
        matrix,
        class_names,
    )

    # Errors
    (
        error_records,
        high_confidence_errors,
    ) = prepare_error_records(
        predictions,
        class_names,
        args.confidence_threshold,
    )

    # Confidence
    confidence_analysis = calculate_confidence_analysis(
        predictions
    )

    # Output directory
    output_dir = (
        evaluation_dir / "error_analysis"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nGENERATING ERROR ANALYSIS ARTIFACTS"
    )

    # 1. Confusion matrix CSV
    confusion_csv_rows = []

    for i, name in enumerate(class_names):

        record = {
            "actual_class": name
        }

        for j, predicted_name in enumerate(class_names):

            record[predicted_name] = int(
                matrix[i, j]
            )

        confusion_csv_rows.append(record)

    write_csv(
        confusion_csv_rows,
        output_dir / "confusion_matrix.csv",
    )

    print(
        "[PASS] confusion_matrix.csv"
    )

    # 2. Normalized confusion matrix CSV
    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    normalized_matrix = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_sums != 0,
    )

    normalized_rows = []

    for i, name in enumerate(class_names):

        record = {
            "actual_class": name
        }

        for j, predicted_name in enumerate(class_names):

            record[predicted_name] = float(
                normalized_matrix[i, j]
            )

        normalized_rows.append(record)

    write_csv(
        normalized_rows,
        output_dir / "normalized_confusion_matrix.csv",
    )

    print(
        "[PASS] normalized_confusion_matrix.csv"
    )

    # 3. Class-wise error report
    write_csv(
        class_report,
        output_dir / "per_class_error_analysis.csv",
    )

    print(
        "[PASS] per_class_error_analysis.csv"
    )

    # 4. Confusion pairs
    write_csv(
        confusion_pairs,
        output_dir / "confusion_pairs.csv",
    )

    print(
        "[PASS] confusion_pairs.csv"
    )

    # 5. All incorrect predictions
    error_fields = [

        "sample_index",
        "image_path",
        "actual_class",
        "predicted_class",
        "actual_class_index",
        "predicted_class_index",
        "confidence",
        "error_type",
    ] + [
        f"prob_class_{i}"
        for i in range(NUM_CLASSES)
    ]

    write_csv(
        error_records,
        output_dir / "misclassified_predictions.csv",
        fieldnames=error_fields,
    )

    print(
        "[PASS] misclassified_predictions.csv"
    )

    # 6. High confidence errors
    write_csv(
        high_confidence_errors,
        output_dir / "high_confidence_errors.csv",
        fieldnames=error_fields,
    )

    print(
        "[PASS] high_confidence_errors.csv"
    )

    # 7. Confidence bins
    write_csv(
        confidence_analysis["confidence_bins"],
        output_dir / "confidence_bins.csv",
    )

    print(
        "[PASS] confidence_bins.csv"
    )

    # 8. Plot artifacts
    plot_confusion_matrix(
        matrix,
        class_names,
        output_dir / "confusion_matrix.png",
        normalized=False,
    )

    print(
        "[PASS] confusion_matrix.png"
    )

    plot_confusion_matrix(
        matrix,
        class_names,
        output_dir / "normalized_confusion_matrix.png",
        normalized=True,
    )

    print(
        "[PASS] normalized_confusion_matrix.png"
    )

    plot_class_errors(
        class_report,
        output_dir / "class_wise_error_rates.png",
    )

    print(
        "[PASS] class_wise_error_rates.png"
    )

    if plot_confusion_pairs(
        confusion_pairs,
        output_dir / "top_confusion_pairs.png",
        args.top_k,
    ):

        print(
            "[PASS] top_confusion_pairs.png"
        )

    if plot_confidence_distribution(
        predictions,
        output_dir / "confidence_distribution.png",
    ):

        print(
            "[PASS] confidence_distribution.png"
        )

    if plot_reliability_diagram(
        confidence_analysis,
        output_dir / "confidence_reliability.png",
    ):

        print(
            "[PASS] confidence_reliability.png"
        )

    # 9. Contact sheet
    contact_sheet_path = None

    if not args.no_contact_sheet:

        contact_sheet_path = create_contact_sheet(
            error_records,
            output_dir / "misclassified_contact_sheet.jpg",
            limit=args.contact_sheet_limit,
        )

        if contact_sheet_path:

            print(
                "[PASS] misclassified_contact_sheet.jpg"
            )

        else:

            print(
                "[WARNING] Contact sheet not generated. "
                "No accessible misclassified images found."
            )

    # 10. Overall summary
    supports = {
        item["class"]: item["support"]
        for item in class_report
    }

    absent_classes = [
        name
        for name, support in supports.items()
        if support == 0
    ]

    macro_precision = float(
        np.mean([
            item["precision"]
            for item in class_report
        ])
    )

    macro_recall = float(
        np.mean([
            item["recall"]
            for item in class_report
        ])
    )

    macro_f1 = float(
        np.mean([
            item["f1_score"]
            for item in class_report
        ])
    )

    run_dir = evaluation_dir.parent.parent

    summary = {

        "timestamp": datetime.now().astimezone().isoformat(),

        "model_name": run_dir.parent.name,

        "run_name": run_dir.name,

        "split_name": evaluation_dir.name,

        "num_classes": NUM_CLASSES,

        "class_names": class_names,

        "total_predictions": total,

        "correct_predictions": correct_count,

        "incorrect_predictions": incorrect_count,

        "accuracy": accuracy,

        "error_rate": error_rate,

        "macro_precision": macro_precision,

        "macro_recall": macro_recall,

        "macro_f1": macro_f1,

        "class_support": supports,

        "absent_classes": absent_classes,

        "class_wise_metrics": class_report,

        "confusion_matrix": matrix.tolist(),

        "normalized_confusion_matrix": (
            normalized_matrix.tolist()
        ),

        "top_confusion_pairs": confusion_pairs[
            :args.top_k
        ],

        "high_confidence_threshold": (
            args.confidence_threshold
        ),

        "high_confidence_error_count": len(
            high_confidence_errors
        ),

        "confidence_analysis": confidence_analysis,

        "probability_columns_available": (
            probabilities_available
        ),

        "correct_field_mismatches": (
            correct_field_mismatches
        ),

        "split_fingerprint_verified": (
            fingerprint_verified
        ),

        "expected_split_fingerprint": (
            EXPECTED_SPLIT_FINGERPRINT
        ),

        "prediction_csv_sha256": sha256_file(
            prediction_path
        ),

        "prediction_csv": str(
            prediction_path
        ),

        "output_directory": str(
            output_dir
        ),

        "contact_sheet": contact_sheet_path,

        "analysis_method": (
            "Prediction-based classification error analysis"
        ),

        "limitations": [
            "Confusion patterns do not establish the biological cause of an error.",
            "High confidence is not equivalent to model correctness.",
            "Visual error interpretation requires reviewing the original images.",
            "No causal conclusions about maize stress symptoms are inferred automatically.",
        ],
    }

    write_json(
        summary,
        output_dir / "error_analysis_summary.json",
    )

    print(
        "[PASS] error_analysis_summary.json"
    )

    # 11. Full text report
    create_text_report(
        summary,
        class_report,
        confusion_pairs,
        error_records,
        high_confidence_errors,
        output_dir / "error_analysis_report.txt",
    )

    print(
        "[PASS] error_analysis_report.txt"
    )

    # Final output
    print("\n" + "=" * 76)

    print(
        "ERROR ANALYSIS COMPLETED"
    )

    print("=" * 76)

    print(
        f"Total predictions       : {total}"
    )

    print(
        f"Correct                 : {correct_count}"
    )

    print(
        f"Incorrect               : {incorrect_count}"
    )

    print(
        f"Accuracy                : {accuracy:.6f}"
    )

    print(
        f"Macro-F1                : {macro_f1:.6f}"
    )

    print(
        f"High-confidence errors  : "
        f"{len(high_confidence_errors)}"
    )

    print(
        f"Output directory        : {output_dir}"
    )

    print("=" * 76)

    return summary


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Maize Leaf Classification Error Analysis"
        )
    )

    parser.add_argument(
        "--evaluation-dir",
        required=True,
        help=(
            "Directory containing predictions.csv "
            "and predictions_metadata.json."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top confusion pairs to plot.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.80,
        help=(
            "Threshold for identifying high-confidence "
            "incorrect predictions."
        ),
    )

    parser.add_argument(
        "--contact-sheet-limit",
        type=int,
        default=48,
        help="Maximum number of misclassified images in contact sheet.",
    )

    parser.add_argument(
        "--no-contact-sheet",
        action="store_true",
        help="Disable contact sheet generation.",
    )

    args = parser.parse_args()

    run_error_analysis(args)


if __name__ == "__main__":
    main()