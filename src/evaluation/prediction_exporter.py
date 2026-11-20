
"""
MAIZE LEAF CLASSIFICATION
Per-Image Prediction Exporter

Exports:
    image_path
    true_class
    predicted_class
    true_index
    predicted_index
    confidence
    correct
    prob_class_0 ... prob_class_6

Also saves:
    Per-class accuracy
    Confusion matrix
    Prediction timing
    GPU/device information
    Split fingerprint
    CSV SHA-256 integrity
    Training metadata
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import tempfile
import time

from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.class_names import (
    CLASS_NAMES,
    NUM_CLASSES,
)

from src.data.dataset_loader import (
    MaizeLeafDataset,
)

from src.utils.device import (
    autocast_context,
)


# ============================================================
# JSON UTILITIES
# ============================================================

def _json_default(value):

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return str(value)


def atomic_json_save(data, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        )

        file.flush()
        os.fsync(file.fileno())

    os.replace(
        temporary_path,
        output_path,
    )

    return output_path


def _sha256_file(path):

    digest = hashlib.sha256()

    with open(path, "rb") as file:

        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================
# MODEL OUTPUT HANDLING
# ============================================================

def extract_logits(output):

    if torch.is_tensor(output):

        logits = output

    elif isinstance(output, dict):

        if "logits" not in output:
            raise ValueError(
                "Model output dictionary must contain logits."
            )

        logits = output["logits"]

    elif isinstance(output, (tuple, list)):

        if not output:
            raise ValueError(
                "Model returned an empty tuple/list."
            )

        logits = output[0]

    else:

        raise TypeError(
            f"Unsupported model output: {type(output).__name__}"
        )

    if not torch.is_tensor(logits):
        raise TypeError(
            "Extracted logits are not a PyTorch tensor."
        )

    if logits.ndim != 2:

        raise ValueError(
            f"Expected [batch, classes], received {logits.shape}"
        )

    return logits


def _amp_metadata(amp_settings):

    amp_settings = amp_settings or {}

    return {
        "enabled": bool(
            amp_settings.get("enabled", False)
        ),
        "dtype": amp_settings.get("dtype"),
        "grad_scaler": bool(
            amp_settings.get("grad_scaler", False)
        ),
        "reason": amp_settings.get("reason"),
    }


# ============================================================
# PATH-AWARE DATASET
# ============================================================

def create_path_dataloader(
    dataset,
    batch_size=16,
    num_workers=0,
    pin_memory=True,
):

    if not hasattr(dataset, "records"):

        raise TypeError(
            "Dataset must expose its original records."
        )

    if not hasattr(dataset, "transform"):

        raise TypeError(
            "Dataset must expose its transform."
        )

    if batch_size <= 0 or num_workers < 0:

        raise ValueError(
            "Invalid DataLoader configuration."
        )

    path_dataset = MaizeLeafDataset(
        records=dataset.records,
        transform=dataset.transform,
        return_path=True,
    )

    loader = DataLoader(
        path_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(pin_memory),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    return loader


# ============================================================
# MAIN PREDICTION EXPORTER
# ============================================================

def export_predictions(
    model,
    dataset,
    device,
    amp_settings,
    output_csv,
    batch_size=16,
    num_workers=0,
    pin_memory=None,
    model_name=None,
    run_name=None,
    split_name="validation",
    split_fingerprint=None,
    dataset_metadata=None,
    training_metadata=None,
):

    """
    Export predictions for every image.

    Training pipeline:
        Best checkpoint -> Validation prediction export.

    The held-out test partition must not be used during
    training or validation-based model selection.
    """

    if not isinstance(model, torch.nn.Module):

        raise TypeError(
            "Model must be a PyTorch module."
        )

    if not isinstance(dataset, MaizeLeafDataset):

        raise TypeError(
            "Dataset must be MaizeLeafDataset."
        )

    if batch_size <= 0 or num_workers < 0:

        raise ValueError(
            "Invalid DataLoader configuration."
        )

    if split_name not in (
        "train",
        "validation",
        "test",
        "external",
    ):

        raise ValueError(
            f"Invalid split name: {split_name}"
        )

    if len(CLASS_NAMES) != NUM_CLASSES:

        raise RuntimeError(
            "CLASS_NAMES and NUM_CLASSES are inconsistent."
        )

    device = torch.device(device)

    if pin_memory is None:
        pin_memory = device.type == "cuda"

    output_csv = Path(output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = output_csv.with_name(
        "predictions_metadata.json"
    )

    # --------------------------------------------------------
    # DATALOADER
    # --------------------------------------------------------

    loader = create_path_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # --------------------------------------------------------
    # CSV COLUMNS
    # --------------------------------------------------------

    fieldnames = [
        "image_path",
        "true_class",
        "predicted_class",
        "true_index",
        "predicted_index",
        "confidence",
        "correct",
    ] + [
        f"prob_class_{index}"
        for index in range(NUM_CLASSES)
    ]

    # --------------------------------------------------------
    # COUNTERS
    # --------------------------------------------------------

    total_samples = 0
    correct_samples = 0

    seen_paths = set()

    class_total = [
        0 for _ in range(NUM_CLASSES)
    ]

    class_correct = [
        0 for _ in range(NUM_CLASSES)
    ]

    confusion_matrix = [
        [0 for _ in range(NUM_CLASSES)]
        for _ in range(NUM_CLASSES)
    ]

    confidence_sum = 0.0

    was_training = model.training

    temporary_path = None

    started_at = datetime.now().astimezone()

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # PREDICTION LOOP
    # --------------------------------------------------------

    try:

        model.eval()

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            suffix=".tmp",
            prefix="predictions_",
            dir=str(output_csv.parent),
            delete=False,
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            writer = csv.DictWriter(
                temporary_file,
                fieldnames=fieldnames,
            )

            writer.writeheader()

            try:

                from tqdm.auto import tqdm

                iterator = tqdm(
                    loader,
                    desc=f"EXPORT {split_name.upper()}",
                    dynamic_ncols=True,
                    leave=True,
                )

            except ImportError:

                iterator = loader

            with torch.inference_mode():

                for images, labels, image_paths in iterator:

                    images = images.to(
                        device,
                        non_blocking=(device.type == "cuda"),
                    )

                    labels = labels.to(
                        device,
                        dtype=torch.long,
                        non_blocking=(device.type == "cuda"),
                    )

                    # ----------------------------------------
                    # FORWARD PASS
                    # ----------------------------------------

                    with autocast_context(
                        device,
                        amp_settings,
                    ):

                        logits = extract_logits(
                            model(images)
                        )

                    if logits.shape[1] != NUM_CLASSES:

                        raise ValueError(
                            f"Model returned {logits.shape[1]} outputs; "
                            f"expected {NUM_CLASSES} classes."
                        )

                    if not torch.isfinite(logits).all():

                        raise FloatingPointError(
                            "Model produced non-finite logits."
                        )

                    # ----------------------------------------
                    # PROBABILITIES
                    # ----------------------------------------

                    probabilities = torch.softmax(
                        logits.float(),
                        dim=1,
                    )

                    if not torch.isfinite(probabilities).all():

                        raise FloatingPointError(
                            "Non-finite prediction probabilities."
                        )

                    probability_sums = probabilities.sum(
                        dim=1
                    )

                    if not torch.allclose(
                        probability_sums,
                        torch.ones_like(probability_sums),
                        atol=1e-5,
                        rtol=1e-5,
                    ):

                        raise RuntimeError(
                            "Prediction probabilities do not sum to one."
                        )

                    predictions = probabilities.argmax(
                        dim=1
                    )

                    confidences = probabilities.max(
                        dim=1
                    ).values

                    labels_cpu = (
                        labels.detach().cpu().tolist()
                    )

                    predictions_cpu = (
                        predictions.detach().cpu().tolist()
                    )

                    confidence_cpu = (
                        confidences.detach().cpu().tolist()
                    )

                    probabilities_cpu = (
                        probabilities.detach().cpu().tolist()
                    )

                    # ----------------------------------------
                    # PER-IMAGE EXPORT
                    # ----------------------------------------

                    for (
                        image_path,
                        true_index,
                        predicted_index,
                        confidence,
                        probs,
                    ) in zip(
                        image_paths,
                        labels_cpu,
                        predictions_cpu,
                        confidence_cpu,
                        probabilities_cpu,
                    ):

                        true_index = int(true_index)
                        predicted_index = int(predicted_index)

                        if not 0 <= true_index < NUM_CLASSES:

                            raise ValueError(
                                f"Invalid true class index: {true_index}"
                            )

                        if not 0 <= predicted_index < NUM_CLASSES:

                            raise ValueError(
                                f"Invalid predicted class index: {predicted_index}"
                            )

                        if len(probs) != NUM_CLASSES:

                            raise RuntimeError(
                                "Unexpected probability vector size."
                            )

                        resolved_path = str(
                            Path(image_path).resolve()
                        )

                        if resolved_path in seen_paths:

                            raise RuntimeError(
                                f"Duplicate image path: {resolved_path}"
                            )

                        seen_paths.add(resolved_path)

                        is_correct = (
                            true_index == predicted_index
                        )

                        total_samples += 1

                        correct_samples += int(
                            is_correct
                        )

                        class_total[true_index] += 1

                        class_correct[true_index] += int(
                            is_correct
                        )

                        confusion_matrix[
                            true_index
                        ][predicted_index] += 1

                        confidence_sum += float(
                            confidence
                        )

                        row = {
                            "image_path": resolved_path,
                            "true_class": CLASS_NAMES[true_index],
                            "predicted_class": CLASS_NAMES[predicted_index],
                            "true_index": true_index,
                            "predicted_index": predicted_index,
                            "confidence": (
                                f"{float(confidence):.10f}"
                            ),
                            "correct": str(
                                is_correct
                            ).lower(),
                        }

                        for class_index, probability in enumerate(probs):

                            row[
                                f"prob_class_{class_index}"
                            ] = f"{float(probability):.10f}"

                        writer.writerow(row)

            temporary_file.flush()

            os.fsync(
                temporary_file.fileno()
            )

        # ----------------------------------------------------
        # INTEGRITY VALIDATION
        # ----------------------------------------------------

        if total_samples != len(dataset):

            raise RuntimeError(
                f"Expected {len(dataset)} predictions, "
                f"received {total_samples}."
            )

        if total_samples == 0:

            raise RuntimeError(
                "No predictions generated."
            )

        os.replace(
            temporary_path,
            output_csv,
        )

        temporary_path = None

    finally:

        if temporary_path is not None:

            temporary_path.unlink(
                missing_ok=True
            )

        model.train(was_training)

    if device.type == "cuda":

        torch.cuda.synchronize(device)

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    completed_at = datetime.now().astimezone()

    accuracy = (
        correct_samples / total_samples
    )

    # --------------------------------------------------------
    # PER-CLASS SUMMARY
    # --------------------------------------------------------

    per_class_summary = {}

    for class_index, class_name in enumerate(CLASS_NAMES):

        count = class_total[class_index]

        per_class_summary[class_name] = {
            "class_index": class_index,
            "total_images": count,
            "correct_predictions": class_correct[class_index],
            "incorrect_predictions": (
                count - class_correct[class_index]
            ),
            "accuracy": (
                class_correct[class_index] / count
                if count > 0
                else None
            ),
        }

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    metadata = {

        "model_name": model_name,

        "run_name": run_name,

        "split_name": split_name,

        "created_at": started_at.isoformat(),

        "completed_at": completed_at.isoformat(),

        "elapsed_seconds": elapsed_seconds,

        "images_per_second": (
            total_samples / elapsed_seconds
            if elapsed_seconds > 0
            else None
        ),

        "total_images": total_samples,

        "correct_predictions": correct_samples,

        "incorrect_predictions": (
            total_samples - correct_samples
        ),

        "accuracy": accuracy,

        "mean_confidence": (
            confidence_sum / total_samples
        ),

        "num_classes": NUM_CLASSES,

        "class_names": list(CLASS_NAMES),

        "per_class_summary": per_class_summary,

        "confusion_matrix": confusion_matrix,

        "device": str(device),

        "amp": _amp_metadata(amp_settings),

        "batch_size": int(batch_size),

        "num_batches": len(loader),

        "num_workers": int(num_workers),

        "pin_memory": bool(pin_memory),

        "split_assignment_fingerprint_sha256": split_fingerprint,

        "dataset_metadata": dataset_metadata or {},

        "training_metadata": training_metadata or {},

        "output_csv": str(
            output_csv.resolve()
        ),

        "metadata_path": str(
            metadata_path.resolve()
        ),

        "csv_sha256": _sha256_file(output_csv),

        "probability_columns": [
            f"prob_class_{index}"
            for index in range(NUM_CLASSES)
        ],
    }

    atomic_json_save(
        metadata,
        metadata_path,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\n" + "=" * 76)

    print("PREDICTION EXPORT COMPLETED")

    print("=" * 76)

    print(f"Model             : {model_name}")

    print(f"Run               : {run_name}")

    print(f"Split             : {split_name}")

    print(f"Total predictions : {total_samples}")

    print(f"Correct           : {correct_samples}")

    print(
        f"Incorrect         : {total_samples - correct_samples}"
    )

    print(f"Accuracy          : {accuracy:.6f}")

    print(
        f"Mean confidence   : {metadata['mean_confidence']:.6f}"
    )

    print(
        f"Time              : {elapsed_seconds:.2f} seconds"
    )

    print(
        f"Images/second     : {metadata['images_per_second']:.3f}"
    )

    print(f"CSV               : {output_csv}")

    print(f"Metadata          : {metadata_path}")

    print("=" * 76)

    return metadata


# ============================================================
# SELF TEST
# ============================================================

def main():

    import csv
    import tempfile

    from PIL import Image
    from torchvision import transforms

    print("=" * 76)

    print("MAIZE LEAF - PREDICTION EXPORTER TEST")

    print("=" * 76)

    with tempfile.TemporaryDirectory() as temporary_directory:

        temporary_directory = Path(
            temporary_directory
        )

        records = []

        for index in range(4):

            image_path = (
                temporary_directory / f"image_{index}.png"
            )

            image = Image.new(
                "RGB",
                (32, 32),
                color=(index * 40, 100, 150),
            )

            image.save(image_path)

            records.append({
                "path": image_path,
                "label": index,
                "class_name": CLASS_NAMES[index],
                "relative_path": image_path.name,
            })

        dataset = MaizeLeafDataset(
            records=records,
            transform=transforms.ToTensor(),
            return_path=False,
        )

        class TestModel(torch.nn.Module):

            def __init__(self):

                super().__init__()

                self.pool = torch.nn.AdaptiveAvgPool2d(1)

                self.fc = torch.nn.Linear(
                    3,
                    NUM_CLASSES,
                )

            def forward(self, images):

                images = self.pool(images).flatten(1)

                return self.fc(images)

        model = TestModel()

        output_csv = (
            temporary_directory / "predictions.csv"
        )

        amp_settings = {
            "enabled": False,
            "dtype": "float16",
            "torch_dtype": None,
            "grad_scaler": False,
            "reason": "CPU self-test",
        }

        metadata = export_predictions(
            model=model,
            dataset=dataset,
            device="cpu",
            amp_settings=amp_settings,
            output_csv=output_csv,
            batch_size=2,
            num_workers=0,
            model_name="test_model",
            run_name="prediction_exporter_self_test",
            split_name="validation",
            split_fingerprint="self-test-fingerprint",
            dataset_metadata={
                "test_dataset": True,
            },
            training_metadata={
                "test_run": True,
            },
        )

        assert output_csv.is_file()

        assert metadata["total_images"] == 4

        assert metadata["split_name"] == "validation"

        assert (
            metadata["split_assignment_fingerprint_sha256"]
            == "self-test-fingerprint"
        )

        with open(
            output_csv,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:

            rows = list(
                csv.DictReader(file)
            )

        assert len(rows) == 4

        required_columns = {
            "image_path",
            "true_class",
            "predicted_class",
            "true_index",
            "predicted_index",
            "confidence",
            "correct",
        }

        assert required_columns.issubset(
            rows[0].keys()
        )

        for class_index in range(NUM_CLASSES):

            assert (
                f"prob_class_{class_index}" in rows[0]
            )

        for row in rows:

            probability_sum = sum(
                float(row[f"prob_class_{index}"])
                for index in range(NUM_CLASSES)
            )

            assert abs(probability_sum - 1.0) < 1e-5

        metadata_file = (
            temporary_directory / "predictions_metadata.json"
        )

        assert metadata_file.is_file()

        loaded_metadata = json.loads(
            metadata_file.read_text(encoding="utf-8")
        )

        assert loaded_metadata["total_images"] == 4

        assert (
            len(loaded_metadata["confusion_matrix"])
            == NUM_CLASSES
        )

        assert (
            loaded_metadata["csv_sha256"]
            == _sha256_file(output_csv)
        )

    print("[PASS] Dataset path loading")
    print("[PASS] Model prediction")
    print("[PASS] Seven-class probability export")
    print("[PASS] Confidence calculation")
    print("[PASS] Correct/incorrect classification")
    print("[PASS] CSV generation")
    print("[PASS] Metadata generation")
    print("[PASS] Probability sum verification")
    print("[PASS] Per-class summary and confusion matrix")
    print("[PASS] Split fingerprint metadata")
    print("[PASS] CSV SHA-256 integrity")

    print("\nPREDICTION EXPORTER TEST: PASSED")

    print("=" * 76)


if __name__ == "__main__":
    main()
