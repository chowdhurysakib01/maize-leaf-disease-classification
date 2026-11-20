
"""
Maize Leaf Classification
Centralized Classification Metrics Engine

Metrics:
    - Accuracy
    - Macro Precision
    - Macro Recall
    - Macro F1-score
    - Balanced Accuracy
    - Multiclass ROC-AUC (One-vs-Rest, Macro)
    - Multiclass PR-AUC (Macro Average Precision)
    - Per-class Precision, Recall and F1
    - Per-class ROC-AUC and Average Precision

Artifacts:
    - metrics.json
    - classification_report.csv
    - confusion_matrix.csv
    - normalized_confusion_matrix.csv
    - confusion_matrix.png
    - roc_curves.png
    - precision_recall_curves.png
    - roc_auc_per_class.csv
    - pr_auc_per_class.csv

Supports:
    - Batch-wise metric accumulation.
    - CPU and CUDA tensors.
    - Seven-class classification.
    - Training, validation and test evaluation.
    - Independent metric calculation.
    - Reproducible evaluation.

IMPORTANT:
    This module does not generate or modify dataset splits.
"""

import csv
import json
import tempfile

from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
)

from src.data import class_names as class_config


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

NUM_CLASSES = int(class_config.NUM_CLASSES)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
# CLASS NAME RESOLUTION
# ============================================================

def get_default_class_names(num_classes=NUM_CLASSES):
    """
    Resolve class names from centralized class configuration.

    Supports common class_names.py structures.
    """

    candidate_attributes = [
        "CLASS_NAMES",
        "CLASS_LABELS",
        "CLASSES",
        "IDX_TO_CLASS",
        "INDEX_TO_CLASS",
        "CLASS_TO_IDX",
        "CLASS_MAPPING",
    ]

    for attribute in candidate_attributes:

        value = getattr(
            class_config,
            attribute,
            None
        )

        if isinstance(value, (list, tuple)):

            names = list(value)

            if len(names) == num_classes:

                return [str(name) for name in names]

        if isinstance(value, dict):

            # Index -> class name.

            if all(
                isinstance(key, int)
                for key in value.keys()
            ):

                names = [
                    value[index]
                    for index in range(num_classes)
                    if index in value
                ]

                if len(names) == num_classes:

                    return [str(name) for name in names]

            # Class name -> index.

            elif all(
                isinstance(index, int)
                for index in value.values()
            ):

                names = [
                    name
                    for name, index in sorted(
                        value.items(),
                        key=lambda item: item[1]
                    )
                ]

                if len(names) == num_classes:

                    return [str(name) for name in names]

    # Known seven-class dataset fallback.

    if num_classes == len(DEFAULT_CLASS_NAMES):

        return DEFAULT_CLASS_NAMES.copy()

    return [
        f"Class_{index}"
        for index in range(num_classes)
    ]


# ============================================================
# NUMPY CONVERSION
# ============================================================

def to_numpy(value):
    """
    Convert PyTorch tensors or array-like values to NumPy.
    """

    if torch.is_tensor(value):

        return value.detach().cpu().numpy()

    return np.asarray(value)


# ============================================================
# CLASSIFICATION METRICS ENGINE
# ============================================================

class ClassificationMetrics:
    """
    Accumulates predictions across batches and calculates
    multiclass classification metrics.

    Example:

        tracker = ClassificationMetrics(num_classes=7)

        for images, labels in loader:

            logits = model(images)

            tracker.update(logits, labels)

        results = tracker.compute()

        tracker.save(output_dir)
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        class_names=None,
        zero_division=0
    ):

        if num_classes <= 1:

            raise ValueError(
                "num_classes must be greater than 1."
            )

        self.num_classes = int(num_classes)

        self.class_names = (

            list(class_names)

            if class_names is not None

            else get_default_class_names(num_classes)

        )

        if len(self.class_names) != self.num_classes:

            raise ValueError(
                "Number of class names must match num_classes."
            )

        if len(set(self.class_names)) != len(self.class_names):

            raise ValueError(
                "Class names must be unique."
            )

        self.zero_division = zero_division

        self.reset()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self._targets = []

        self._predictions = []

        self._probabilities = []

    # ========================================================
    # UPDATE USING MODEL LOGITS
    # ========================================================

    def update(self, logits, targets):
        """
        Update metrics using model logits.

        logits:
            [B, num_classes]

        targets:
            [B]

        Softmax is applied automatically.
        """

        if not torch.is_tensor(logits):

            logits = torch.as_tensor(logits)

        if not torch.is_tensor(targets):

            targets = torch.as_tensor(targets)

        if logits.ndim != 2:

            raise ValueError(
                "Logits must have shape [B, num_classes]."
            )

        if logits.shape[1] != self.num_classes:

            raise ValueError(
                f"Expected {self.num_classes} logits, "
                f"received {logits.shape[1]}."
            )

        if targets.ndim != 1:

            targets = targets.reshape(-1)

        if logits.shape[0] != targets.shape[0]:

            raise ValueError(
                "Logits and targets have different batch sizes."
            )

        if not torch.isfinite(logits).all():

            raise ValueError(
                "Logits contain NaN or infinite values."
            )

        probabilities = torch.softmax(
            logits.detach().float(),
            dim=1
        )

        self.update_from_probabilities(
            probabilities,
            targets
        )

    # ========================================================
    # UPDATE USING PROBABILITIES
    # ========================================================

    def update_from_probabilities(
        self,
        probabilities,
        targets
    ):
        """
        Update metrics using predicted class probabilities.

        probabilities:
            [B, num_classes]

        targets:
            [B]
        """

        probabilities = to_numpy(
            probabilities
        ).astype(np.float64, copy=False)

        targets = to_numpy(
            targets
        ).astype(np.int64, copy=False).reshape(-1)

        if probabilities.ndim != 2:

            raise ValueError(
                "Probabilities must have shape [B, num_classes]."
            )

        if probabilities.shape[1] != self.num_classes:

            raise ValueError(
                "Probability dimension does not match num_classes."
            )

        if probabilities.shape[0] != len(targets):

            raise ValueError(
                "Probability and target batch sizes differ."
            )

        if len(targets) == 0:

            raise ValueError(
                "Empty batches cannot update metrics."
            )

        if not np.isfinite(probabilities).all():

            raise ValueError(
                "Probabilities contain NaN or infinite values."
            )

        if np.any(probabilities < 0):

            raise ValueError(
                "Probabilities cannot be negative."
            )

        if np.any(probabilities > 1 + 1e-6):

            raise ValueError(
                "Probabilities cannot exceed 1."
            )

        if np.any(targets < 0) or np.any(
            targets >= self.num_classes
        ):

            raise ValueError(
                "Target labels are outside the configured class range."
            )

        row_sums = probabilities.sum(axis=1)

        if np.any(row_sums <= 0):

            raise ValueError(
                "Probability rows must have positive sums."
            )

        # Normalize small floating-point deviations.

        probabilities = (
            probabilities / row_sums[:, None]
        )

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        self._targets.append(targets.copy())

        self._predictions.append(
            predictions.copy()
        )

        self._probabilities.append(
            probabilities.copy()
        )

    # ========================================================
    # COLLECT ACCUMULATED DATA
    # ========================================================

    def _collect(self):

        if not self._targets:

            raise RuntimeError(
                "No predictions available. "
                "Call update() before compute()."
            )

        targets = np.concatenate(
            self._targets
        )

        predictions = np.concatenate(
            self._predictions
        )

        probabilities = np.concatenate(
            self._probabilities,
            axis=0
        )

        return targets, predictions, probabilities

    # ========================================================
    # COMPUTE CLASSIFICATION METRICS
    # ========================================================

    def compute(self):
        """
        Calculate all classification metrics.

        ROC-AUC:
            One-vs-Rest, macro average.

        PR-AUC:
            Macro average of per-class Average Precision.

        Missing classes are retained in the report.
        """

        targets, predictions, probabilities = self._collect()

        labels = np.arange(
            self.num_classes
        )

        # ----------------------------------------------------
        # Basic metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            targets,
            predictions
        )

        precision, recall, f1, support = (
            precision_recall_fscore_support(

                targets,

                predictions,

                labels=labels,

                average=None,

                zero_division=self.zero_division

            )
        )

        macro_precision = precision_score = float(
            np.mean(precision)
        )

        macro_recall = float(
            np.mean(recall)
        )

        macro_f1 = float(
            np.mean(f1)
        )

        balanced_accuracy = balanced_accuracy_score(
            targets,
            predictions
        )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            targets,
            predictions,
            labels=labels
        )

        row_totals = cm.sum(axis=1, keepdims=True)

        normalized_cm = np.divide(

            cm.astype(np.float64),

            row_totals,

            out=np.zeros_like(
                cm,
                dtype=np.float64
            ),

            where=row_totals != 0

        )

        # ----------------------------------------------------
        # ROC-AUC and PR-AUC
        # ----------------------------------------------------

        roc_auc_values = []

        pr_auc_values = []

        per_class_roc_auc = []

        per_class_pr_auc = []

        for index in labels:

            binary_targets = (
                targets == index
            ).astype(np.int32)

            positive_count = int(
                binary_targets.sum()
            )

            negative_count = int(
                len(binary_targets) - positive_count
            )

            # ROC-AUC requires positive and negative examples.

            if positive_count > 0 and negative_count > 0:

                class_roc_auc = float(
                    roc_auc_score(
                        binary_targets,
                        probabilities[:, index]
                    )
                )

                roc_auc_values.append(
                    class_roc_auc
                )

            else:

                class_roc_auc = None

            # Average Precision requires at least one positive.

            if positive_count > 0:

                class_pr_auc = float(
                    average_precision_score(

                        binary_targets,

                        probabilities[:, index]

                    )
                )

                pr_auc_values.append(
                    class_pr_auc
                )

            else:

                class_pr_auc = None

            per_class_roc_auc.append(
                class_roc_auc
            )

            per_class_pr_auc.append(
                class_pr_auc
            )

        roc_auc_macro = (

            float(np.mean(roc_auc_values))

            if roc_auc_values

            else None

        )

        pr_auc_macro = (

            float(np.mean(pr_auc_values))

            if pr_auc_values

            else None

        )

        # ----------------------------------------------------
        # Per-class report
        # ----------------------------------------------------

        per_class = []

        for index in labels:

            per_class.append({

                "class_index": int(index),

                "class_name": self.class_names[index],

                "support": int(support[index]),

                "precision": float(precision[index]),

                "recall": float(recall[index]),

                "f1_score": float(f1[index]),

                "roc_auc": per_class_roc_auc[index],

                "average_precision": per_class_pr_auc[index],

            })

        # ----------------------------------------------------
        # Macro and weighted report
        # ----------------------------------------------------

        report = classification_report(

            targets,

            predictions,

            labels=labels,

            target_names=self.class_names,

            output_dict=True,

            zero_division=self.zero_division

        )

        return {

            "num_classes": self.num_classes,

            "num_samples": int(len(targets)),

            "class_names": self.class_names,

            "metrics": {

                "accuracy": float(accuracy),

                "precision_macro": float(macro_precision),

                "recall_macro": float(macro_recall),

                "f1_macro": float(macro_f1),

                "balanced_accuracy": float(
                    balanced_accuracy
                ),

                "roc_auc_ovr_macro": roc_auc_macro,

                "pr_auc_macro_average_precision": pr_auc_macro,

            },

            "per_class": per_class,

            "confusion_matrix": cm.tolist(),

            "normalized_confusion_matrix": (
                normalized_cm.tolist()
            ),

            "classification_report": report,

            "auc_definition": {

                "roc_auc": (
                    "Macro average of one-vs-rest "
                    "per-class ROC-AUC."
                ),

                "pr_auc": (
                    "Macro average of per-class "
                    "Average Precision scores."
                ),

            },

        }

    # ========================================================
    # SAVE CONFUSION MATRIX CSV
    # ========================================================

    def _save_confusion_matrix(
        self,
        cm,
        output_dir,
        normalized=False
    ):

        filename = (

            "normalized_confusion_matrix.csv"

            if normalized

            else "confusion_matrix.csv"

        )

        filepath = output_dir / filename

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                ["Actual / Predicted"] + self.class_names
            )

            for index, row in enumerate(cm):

                writer.writerow(
                    [self.class_names[index]]
                    + list(row)
                )

        return filepath

    # ========================================================
    # SAVE CLASSIFICATION REPORT CSV
    # ========================================================

    def _save_classification_report(
        self,
        result,
        output_dir
    ):

        filepath = (
            output_dir / "classification_report.csv"
        )

        columns = [

            "class_index",

            "class_name",

            "support",

            "precision",

            "recall",

            "f1_score",

            "roc_auc",

            "average_precision",

        ]

        with open(
            filepath,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )

            writer.writeheader()

            for row in result["per_class"]:

                writer.writerow(row)

            # Macro average.

            macro = result["classification_report"][
                "macro avg"
            ]

            writer.writerow({

                "class_index": "",

                "class_name": "Macro Average",

                "support": int(
                    sum(
                        row["support"]
                        for row in result["per_class"]
                    )
                ),

                "precision": macro["precision"],

                "recall": macro["recall"],

                "f1_score": macro["f1-score"],

                "roc_auc": "",

                "average_precision": "",

            })

            # Weighted average.

            weighted = result["classification_report"][
                "weighted avg"
            ]

            writer.writerow({

                "class_index": "",

                "class_name": "Weighted Average",

                "support": int(
                    sum(
                        row["support"]
                        for row in result["per_class"]
                    )
                ),

                "precision": weighted["precision"],

                "recall": weighted["recall"],

                "f1_score": weighted["f1-score"],

                "roc_auc": "",

                "average_precision": "",

            })

        return filepath

    # ========================================================
    # SAVE PER-CLASS AUC CSV
    # ========================================================

    def _save_auc_reports(
        self,
        result,
        output_dir
    ):

        roc_path = (
            output_dir / "roc_auc_per_class.csv"
        )

        pr_path = (
            output_dir / "pr_auc_per_class.csv"
        )

        columns = [
            "class_index",
            "class_name",
            "auc",
            "support"
        ]

        with open(
            roc_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )

            writer.writeheader()

            for row in result["per_class"]:

                writer.writerow({

                    "class_index": row["class_index"],

                    "class_name": row["class_name"],

                    "auc": row["roc_auc"],

                    "support": row["support"],

                })

        with open(
            pr_path,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )

            writer.writeheader()

            for row in result["per_class"]:

                writer.writerow({

                    "class_index": row["class_index"],

                    "class_name": row["class_name"],

                    "auc": row["average_precision"],

                    "support": row["support"],

                })

        return roc_path, pr_path

    # ========================================================
    # SAVE CONFUSION MATRIX IMAGE
    # ========================================================

    def _plot_confusion_matrix(
        self,
        cm,
        output_dir,
        normalized=False
    ):

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(
            figsize=(11, 9)
        )

        image = ax.imshow(
            cm,
            interpolation="nearest"
        )

        fig.colorbar(
            image,
            ax=ax
        )

        ax.set(

            xticks=np.arange(self.num_classes),

            yticks=np.arange(self.num_classes),

            xticklabels=self.class_names,

            yticklabels=self.class_names,

            xlabel="Predicted Class",

            ylabel="Actual Class",

            title=(

                "Normalized Confusion Matrix"

                if normalized

                else "Confusion Matrix"

            ),

        )

        plt.setp(

            ax.get_xticklabels(),

            rotation=45,

            ha="right",

            rotation_mode="anchor"

        )

        for row in range(self.num_classes):

            for column in range(self.num_classes):

                value = cm[row, column]

                text = (

                    f"{value:.3f}"

                    if normalized

                    else f"{int(value)}"

                )

                ax.text(

                    column,

                    row,

                    text,

                    ha="center",

                    va="center"

                )

        fig.tight_layout()

        filename = (

            "normalized_confusion_matrix.png"

            if normalized

            else "confusion_matrix.png"

        )

        filepath = output_dir / filename

        fig.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        return filepath

    # ========================================================
    # SAVE ROC AND PRECISION-RECALL CURVES
    # ========================================================

    def _plot_auc_curves(
        self,
        targets,
        probabilities,
        output_dir
    ):

        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        roc_filepath = (
            output_dir / "roc_curves.png"
        )

        pr_filepath = (
            output_dir / "precision_recall_curves.png"
        )

        # ----------------------------------------------------
        # ROC curves
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        roc_curve_rows = []

        for index in range(self.num_classes):

            binary_targets = (
                targets == index
            ).astype(np.int32)

            if len(np.unique(binary_targets)) < 2:

                continue

            fpr, tpr, thresholds = roc_curve(

                binary_targets,

                probabilities[:, index]

            )

            auc_value = roc_auc_score(

                binary_targets,

                probabilities[:, index]

            )

            ax.plot(

                fpr,

                tpr,

                label=(
                    f"{self.class_names[index]} "
                    f"(AUC={auc_value:.4f})"
                )

            )

            for fpr_value, tpr_value, threshold in zip(
                fpr,
                tpr,
                thresholds
            ):

                roc_curve_rows.append({

                    "class_index": index,

                    "class_name": self.class_names[index],

                    "fpr": float(fpr_value),

                    "tpr": float(tpr_value),

                    "threshold": float(threshold),

                })

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Random classifier"
        )

        ax.set_xlabel(
            "False Positive Rate"
        )

        ax.set_ylabel(
            "True Positive Rate"
        )

        ax.set_title(
            "Multiclass ROC Curves (One-vs-Rest)"
        )

        ax.legend(
            loc="lower right",
            fontsize=8
        )

        ax.grid(
            alpha=0.3
        )

        fig.tight_layout()

        fig.savefig(
            roc_filepath,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        # Save ROC coordinates.

        with open(
            output_dir / "roc_curve_data.csv",
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            columns = [
                "class_index",
                "class_name",
                "fpr",
                "tpr",
                "threshold"
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )

            writer.writeheader()

            writer.writerows(
                roc_curve_rows
            )

        # ----------------------------------------------------
        # Precision-recall curves
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 8)
        )

        pr_curve_rows = []

        for index in range(self.num_classes):

            binary_targets = (
                targets == index
            ).astype(np.int32)

            if binary_targets.sum() == 0:

                continue

            precision_values, recall_values, thresholds = (
                precision_recall_curve(

                    binary_targets,

                    probabilities[:, index]

                )
            )

            ap_value = average_precision_score(

                binary_targets,

                probabilities[:, index]

            )

            ax.plot(

                recall_values,

                precision_values,

                label=(
                    f"{self.class_names[index]} "
                    f"(AP={ap_value:.4f})"
                )

            )

            for position, (
                precision_value,
                recall_value
            ) in enumerate(zip(
                precision_values,
                recall_values
            )):

                threshold = (

                    float(thresholds[position])

                    if position < len(thresholds)

                    else ""

                )

                pr_curve_rows.append({

                    "class_index": index,

                    "class_name": self.class_names[index],

                    "recall": float(recall_value),

                    "precision": float(precision_value),

                    "threshold": threshold,

                })

        ax.set_xlabel(
            "Recall"
        )

        ax.set_ylabel(
            "Precision"
        )

        ax.set_title(
            "Multiclass Precision-Recall Curves"
        )

        ax.legend(
            loc="best",
            fontsize=8
        )

        ax.grid(
            alpha=0.3
        )

        fig.tight_layout()

        fig.savefig(
            pr_filepath,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        # Save PR coordinates.

        with open(
            output_dir / "pr_curve_data.csv",
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            columns = [
                "class_index",
                "class_name",
                "recall",
                "precision",
                "threshold"
            ]

            writer = csv.DictWriter(
                file,
                fieldnames=columns
            )

            writer.writeheader()

            writer.writerows(
                pr_curve_rows
            )

        return roc_filepath, pr_filepath

    # ========================================================
    # SAVE ALL METRICS
    # ========================================================

    def save(
        self,
        output_dir,
        split_name="test"
    ):
        """
        Save all metrics and graphical artifacts.

        output_dir:
            Model-specific evaluation directory.

        split_name:
            train, validation or test.
        """

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        targets, predictions, probabilities = self._collect()

        result = self.compute()

        # ----------------------------------------------------
        # Main metrics JSON
        # ----------------------------------------------------

        result["split_name"] = split_name

        result["prediction_summary"] = {

            "total_predictions": int(len(targets)),

            "correct_predictions": int(
                np.sum(targets == predictions)
            ),

            "incorrect_predictions": int(
                np.sum(targets != predictions)
            ),

        }

        metrics_path = (
            output_dir / "metrics.json"
        )

        with open(
            metrics_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                ensure_ascii=False,
                allow_nan=False
            )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm_path = self._save_confusion_matrix(

            result["confusion_matrix"],

            output_dir,

            normalized=False

        )

        normalized_cm_path = self._save_confusion_matrix(

            result["normalized_confusion_matrix"],

            output_dir,

            normalized=True

        )

        cm_image = self._plot_confusion_matrix(

            np.asarray(
                result["confusion_matrix"]
            ),

            output_dir,

            normalized=False

        )

        normalized_cm_image = self._plot_confusion_matrix(

            np.asarray(
                result["normalized_confusion_matrix"]
            ),

            output_dir,

            normalized=True

        )

        # ----------------------------------------------------
        # Classification report
        # ----------------------------------------------------

        report_path = self._save_classification_report(

            result,

            output_dir

        )

        # ----------------------------------------------------
        # AUC reports
        # ----------------------------------------------------

        roc_auc_path, pr_auc_path = self._save_auc_reports(

            result,

            output_dir

        )

        # ----------------------------------------------------
        # ROC and PR curves
        # ----------------------------------------------------

        roc_image, pr_image = self._plot_auc_curves(

            targets,

            probabilities,

            output_dir

        )

        # ----------------------------------------------------
        # Return generated artifact paths
        # ----------------------------------------------------

        return {

            "metrics_json": str(metrics_path),

            "classification_report": str(report_path),

            "confusion_matrix_csv": str(cm_path),

            "normalized_confusion_matrix_csv": str(
                normalized_cm_path
            ),

            "confusion_matrix_png": str(cm_image),

            "normalized_confusion_matrix_png": str(
                normalized_cm_image
            ),

            "roc_auc_csv": str(roc_auc_path),

            "pr_auc_csv": str(pr_auc_path),

            "roc_curve_png": str(roc_image),

            "pr_curve_png": str(pr_image),

            "roc_curve_data_csv": str(
                output_dir / "roc_curve_data.csv"
            ),

            "pr_curve_data_csv": str(
                output_dir / "pr_curve_data.csv"
            ),

        }


# ============================================================
# STANDALONE METRIC FUNCTION
# ============================================================

def compute_classification_metrics(
    y_true,
    y_prob,
    class_names=None,
    num_classes=NUM_CLASSES
):
    """
    Calculate metrics directly from ground-truth labels
    and predicted probabilities.
    """

    tracker = ClassificationMetrics(

        num_classes=num_classes,

        class_names=class_names

    )

    tracker.update_from_probabilities(
        y_prob,
        y_true
    )

    return tracker.compute()


# ============================================================
# METRICS ENGINE TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF - CLASSIFICATION METRICS TEST")

    print("=" * 70)

    print(f"\nProject: {PROJECT_ROOT}")

    print(f"Classes: {NUM_CLASSES}")

    print("\nCLASS CONFIGURATION")

    class_names = get_default_class_names()

    for index, name in enumerate(class_names):

        print(
            f"{index}: {name}"
        )

    assert len(class_names) == NUM_CLASSES

    print("\n[PASS] Seven-class configuration")

    # --------------------------------------------------------
    # Initialize metric tracker
    # --------------------------------------------------------

    tracker = ClassificationMetrics(
        num_classes=NUM_CLASSES
    )

    print("[PASS] Metrics initialization")

    # --------------------------------------------------------
    # Generate synthetic predictions
    # --------------------------------------------------------

    torch.manual_seed(42)

    sample_count = 140

    targets = torch.arange(
        NUM_CLASSES
    ).repeat_interleave(20)

    # Generate synthetic logits.

    logits = torch.randn(
        sample_count,
        NUM_CLASSES
    )

    # Give each target class a stronger corresponding logit.

    logits[
        torch.arange(sample_count),
        targets
    ] += 2.0

    # --------------------------------------------------------
    # Test batch-wise accumulation
    # --------------------------------------------------------

    batch_size = 16

    for start in range(
        0,
        sample_count,
        batch_size
    ):

        end = min(
            start + batch_size,
            sample_count
        )

        tracker.update(

            logits[start:end],

            targets[start:end]

        )

    print("[PASS] Batch-wise metric accumulation")

    # --------------------------------------------------------
    # Compute metrics
    # --------------------------------------------------------

    result = tracker.compute()

    metrics = result["metrics"]

    assert result["num_samples"] == sample_count

    assert result["num_classes"] == NUM_CLASSES

    required_metrics = [

        "accuracy",

        "precision_macro",

        "recall_macro",

        "f1_macro",

        "balanced_accuracy",

        "roc_auc_ovr_macro",

        "pr_auc_macro_average_precision",

    ]

    for metric_name in required_metrics:

        assert metric_name in metrics

        value = metrics[metric_name]

        assert value is not None

        assert 0 <= value <= 1

        print(
            f"[PASS] {metric_name}: {value:.4f}"
        )

    # --------------------------------------------------------
    # Confusion matrix validation
    # --------------------------------------------------------

    cm = np.asarray(
        result["confusion_matrix"]
    )

    assert cm.shape == (
        NUM_CLASSES,
        NUM_CLASSES
    )

    assert cm.sum() == sample_count

    print("[PASS] Confusion matrix")

    # --------------------------------------------------------
    # Per-class validation
    # --------------------------------------------------------

    assert len(
        result["per_class"]
    ) == NUM_CLASSES

    for row in result["per_class"]:

        assert 0 <= row["precision"] <= 1

        assert 0 <= row["recall"] <= 1

        assert 0 <= row["f1_score"] <= 1

    print("[PASS] Per-class classification metrics")

    # --------------------------------------------------------
    # Save artifact smoke test
    # --------------------------------------------------------

    test_root = (
        PROJECT_ROOT / "Pipeline_Test"
    )

    test_root.mkdir(
        parents=True,
        exist_ok=True
    )

    with tempfile.TemporaryDirectory(

        prefix="metrics_smoke_test_",

        dir=test_root

    ) as temporary_directory:

        output_paths = tracker.save(

            output_dir=temporary_directory,

            split_name="test"

        )

        for name, filepath in output_paths.items():

            assert Path(filepath).is_file(), (
                f"Missing artifact: {name}"
            )

            print(
                f"[PASS] Artifact: {name}"
            )

    # --------------------------------------------------------
    # Reset validation
    # --------------------------------------------------------

    tracker.reset()

    assert len(tracker._targets) == 0

    assert len(tracker._predictions) == 0

    assert len(tracker._probabilities) == 0

    print("[PASS] Metrics reset")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\nCLASSIFICATION METRICS TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
