
"""
Maize Leaf Classification
Complete Training History Logger

Records:
    1. Batch-wise training history.
    2. Epoch-wise training history.
    3. Training and validation metrics.
    4. Learning rate history.
    5. Batch and epoch processing times.
    6. Cumulative training time.
    7. GPU memory information.
    8. Training start/end timestamps.
    9. Model, dataset and environment metadata.
    10. Final experiment summary.

Artifacts:
    training_history.csv
    training_history.json
    batch_history.csv
    batch_history.jsonl
    run_metadata.json
    run_summary.json
    training_log.txt

All history is saved incrementally.
"""

import csv
import json
import os
import time

from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch


# ============================================================
# SERIALIZATION
# ============================================================

def to_jsonable(value):

    if isinstance(value, dict):

        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):

        return [
            to_jsonable(item)
            for item in value
        ]

    if torch.is_tensor(value):

        if value.numel() == 1:

            return value.detach().cpu().item()

        return value.detach().cpu().tolist()

    if isinstance(value, np.ndarray):

        return value.tolist()

    if isinstance(value, np.generic):

        return value.item()

    if isinstance(value, Path):

        return str(value)

    if isinstance(value, datetime):

        return value.isoformat()

    if isinstance(value, (str, int, float, bool)) or value is None:

        return value

    return str(value)


# ============================================================
# TIMESTAMP
# ============================================================

def current_timestamps():

    local_now = datetime.now().astimezone()

    utc_now = datetime.now(timezone.utc)

    return {

        "local": local_now.isoformat(),

        "utc": utc_now.isoformat(),

    }


# ============================================================
# ATOMIC JSON WRITER
# ============================================================

def atomic_write_json(data, filepath):

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = filepath.with_name(
        filepath.name + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(

            to_jsonable(data),

            file,

            indent=4,

            ensure_ascii=False,

            allow_nan=False

        )

    os.replace(
        temporary_path,
        filepath
    )


# ============================================================
# ATOMIC CSV WRITER
# ============================================================

def atomic_write_csv(records, filepath):

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not records:

        return

    fieldnames = []

    for record in records:

        for key in record:

            if key not in fieldnames:

                fieldnames.append(key)

    temporary_path = filepath.with_name(
        filepath.name + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=fieldnames,

            extrasaction="ignore"

        )

        writer.writeheader()

        writer.writerows(records)

    os.replace(
        temporary_path,
        filepath
    )


# ============================================================
# COMPLETE TRAINING HISTORY LOGGER
# ============================================================

class TrainingHistoryLogger:

    def __init__(
        self,
        output_dir,
        run_name=None,
        save_batch_history=True
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.run_name = run_name or self.output_dir.name

        self.save_batch_history = save_batch_history

        # ----------------------------------------------------
        # History directory
        # ----------------------------------------------------

        self.history_dir = (
            self.output_dir / "training_history"
        )

        self.history_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ----------------------------------------------------
        # File paths
        # ----------------------------------------------------

        self.epoch_csv_path = (
            self.history_dir / "training_history.csv"
        )

        self.epoch_json_path = (
            self.history_dir / "training_history.json"
        )

        self.batch_csv_path = (
            self.history_dir / "batch_history.csv"
        )

        self.batch_jsonl_path = (
            self.history_dir / "batch_history.jsonl"
        )

        self.metadata_path = (
            self.history_dir / "run_metadata.json"
        )

        self.summary_path = (
            self.history_dir / "run_summary.json"
        )

        self.log_path = (
            self.history_dir / "training_log.txt"
        )

        # ----------------------------------------------------
        # Start time
        # ----------------------------------------------------

        self.start_timestamp = current_timestamps()

        self.start_monotonic = time.perf_counter()

        self.end_timestamp = None

        # ----------------------------------------------------
        # History storage
        # ----------------------------------------------------

        self.epoch_records = []

        self.batch_records_count = 0

        self.epochs_completed = 0

        self.best_epoch = None

        self.best_metric = None

        self.status = "running"

        # ----------------------------------------------------
        # Initialize files
        # ----------------------------------------------------

        self._initialize_files()

        self._write_log(

            "TRAINING RUN INITIALIZED"

        )

        self._write_log(
            f"Run name: {self.run_name}"
        )

        self._write_log(
            f"Start local time: {self.start_timestamp['local']}"
        )

        self._write_log(
            f"Start UTC time: {self.start_timestamp['utc']}"
        )

    # ========================================================
    # INITIALIZE FILES
    # ========================================================

    def _initialize_files(self):

        atomic_write_json(

            {

                "run_name": self.run_name,

                "status": "running",

                "start_timestamp": self.start_timestamp,

                "epochs": [],

            },

            self.epoch_json_path

        )

        if self.save_batch_history:

            with open(
                self.batch_csv_path,
                "w",
                newline="",
                encoding="utf-8-sig"
            ) as file:

                writer = csv.writer(file)

                writer.writerow([

                    "timestamp_local",

                    "timestamp_utc",

                    "epoch",

                    "batch",

                    "global_step",

                    "split",

                    "loss",

                    "accuracy",

                    "learning_rate",

                    "batch_duration_seconds",

                    "elapsed_seconds",

                    "samples",

                    "samples_per_second",

                    "gpu_memory_allocated_mb",

                    "gpu_memory_reserved_mb",

                ])

            self.batch_jsonl_path.write_text(
                "",
                encoding="utf-8"
            )

    # ========================================================
    # WRITE TEXT LOG
    # ========================================================

    def _write_log(self, message):

        timestamp = current_timestamps()["local"]

        with open(
            self.log_path,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                f"[{timestamp}] {message}\n"
            )

    # ========================================================
    # ELAPSED TIME
    # ========================================================

    def elapsed_seconds(self):

        return (
            time.perf_counter()
            - self.start_monotonic
        )

    # ========================================================
    # SAVE RUN METADATA
    # ========================================================

    def save_metadata(
        self,
        config=None,
        model_metadata=None,
        dataset_metadata=None,
        environment_metadata=None
    ):

        metadata = {

            "run_name": self.run_name,

            "start_timestamp": self.start_timestamp,

            "config": config or {},

            "model": model_metadata or {},

            "dataset": dataset_metadata or {},

            "environment": environment_metadata or {},

        }

        atomic_write_json(
            metadata,
            self.metadata_path
        )

        self._write_log(
            "RUN METADATA SAVED"
        )

        return self.metadata_path

    # ========================================================
    # GPU MEMORY
    # ========================================================

    @staticmethod
    def get_gpu_memory():

        if not torch.cuda.is_available():

            return {

                "allocated_mb": None,

                "reserved_mb": None,

                "peak_allocated_mb": None,

                "peak_reserved_mb": None,

            }

        return {

            "allocated_mb": round(

                torch.cuda.memory_allocated() / (1024 ** 2),

                3

            ),

            "reserved_mb": round(

                torch.cuda.memory_reserved() / (1024 ** 2),

                3

            ),

            "peak_allocated_mb": round(

                torch.cuda.max_memory_allocated() / (1024 ** 2),

                3

            ),

            "peak_reserved_mb": round(

                torch.cuda.max_memory_reserved() / (1024 ** 2),

                3

            ),

        }

    # ========================================================
    # BATCH-WISE HISTORY
    # ========================================================

    def log_batch(
        self,
        epoch,
        batch,
        global_step,
        loss,
        accuracy,
        learning_rate,
        batch_duration_seconds,
        samples,
        split="train",
        gpu_memory_allocated_mb=None,
        gpu_memory_reserved_mb=None
    ):

        if not self.save_batch_history:

            return

        if split not in (
            "train",
            "validation",
            "test"
        ):

            raise ValueError(
                f"Invalid split name: {split}"
            )

        timestamps = current_timestamps()

        duration = float(
            batch_duration_seconds
        )

        sample_count = int(samples)

        record = {

            "timestamp_local": timestamps["local"],

            "timestamp_utc": timestamps["utc"],

            "epoch": int(epoch),

            "batch": int(batch),

            "global_step": int(global_step),

            "split": split,

            "loss": float(loss),

            "accuracy": float(accuracy),

            "learning_rate": float(learning_rate),

            "batch_duration_seconds": duration,

            "elapsed_seconds": float(
                self.elapsed_seconds()
            ),

            "samples": sample_count,

            "samples_per_second": (

                sample_count / duration

                if duration > 0

                else None

            ),

            "gpu_memory_allocated_mb": (
                gpu_memory_allocated_mb
            ),

            "gpu_memory_reserved_mb": (
                gpu_memory_reserved_mb
            ),

        }

        # ----------------------------------------------------
        # Incremental CSV write
        # ----------------------------------------------------

        with open(
            self.batch_csv_path,
            "a",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=list(record.keys())

            )

            writer.writerow(record)

        # ----------------------------------------------------
        # Incremental JSONL write
        # ----------------------------------------------------

        with open(
            self.batch_jsonl_path,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(

                json.dumps(
                    record,
                    ensure_ascii=False,
                    allow_nan=False
                )

                + "\n"

            )

        self.batch_records_count += 1

    # ========================================================
    # EXTRACT METRIC VALUES
    # ========================================================

    @staticmethod
    def _extract_metrics(metrics):

        if metrics is None:

            return {}

        if "metrics" in metrics:

            return metrics["metrics"]

        return metrics

    # ========================================================
    # EPOCH-WISE HISTORY
    # ========================================================

    def log_epoch(
        self,
        epoch,
        train_metrics,
        validation_metrics,
        learning_rate,
        train_loss=None,
        validation_loss=None,
        epoch_duration_seconds=None,
        train_duration_seconds=None,
        validation_duration_seconds=None,
        train_samples=None,
        validation_samples=None,
        samples_per_second=None,
        gpu_memory=None,
        epoch_start_timestamp=None,
        epoch_end_timestamp=None,
        extra=None
    ):

        timestamps = current_timestamps()

        train_values = self._extract_metrics(
            train_metrics
        )

        validation_values = self._extract_metrics(
            validation_metrics
        )

        memory = gpu_memory or self.get_gpu_memory()

        epoch_duration = (

            float(epoch_duration_seconds)

            if epoch_duration_seconds is not None

            else None

        )

        # ----------------------------------------------------
        # Build epoch record
        # ----------------------------------------------------

        record = {

            "epoch": int(epoch),

            "timestamp_local": timestamps["local"],

            "timestamp_utc": timestamps["utc"],

            "epoch_start_timestamp": (
                epoch_start_timestamp
            ),

            "epoch_end_timestamp": (
                epoch_end_timestamp
            ),

            "elapsed_seconds": float(
                self.elapsed_seconds()
            ),

            "epoch_duration_seconds": epoch_duration,

            "epoch_duration_minutes": (

                epoch_duration / 60

                if epoch_duration is not None

                else None

            ),

            "train_duration_seconds": (
                train_duration_seconds
            ),

            "validation_duration_seconds": (
                validation_duration_seconds
            ),

            "learning_rate": float(
                learning_rate
            ),

            "train_loss": train_loss,

            "validation_loss": validation_loss,

            "train_samples": train_samples,

            "validation_samples": validation_samples,

            "samples_per_second": samples_per_second,

            "gpu_memory_allocated_mb": (
                memory.get("allocated_mb")
            ),

            "gpu_memory_reserved_mb": (
                memory.get("reserved_mb")
            ),

            "gpu_peak_allocated_mb": (
                memory.get("peak_allocated_mb")
            ),

            "gpu_peak_reserved_mb": (
                memory.get("peak_reserved_mb")
            ),

            "train_metrics": train_values,

            "validation_metrics": validation_values,

            "extra": extra or {},

        }

        record = to_jsonable(record)

        # ----------------------------------------------------
        # Store history
        # ----------------------------------------------------

        self.epoch_records.append(record)

        self.epochs_completed += 1

        # ----------------------------------------------------
        # Save JSON after every epoch
        # ----------------------------------------------------

        atomic_write_json(

            {

                "run_name": self.run_name,

                "status": self.status,

                "start_timestamp": self.start_timestamp,

                "last_saved_timestamp": timestamps,

                "epochs_completed": self.epochs_completed,

                "epochs": self.epoch_records,

            },

            self.epoch_json_path

        )

        # ----------------------------------------------------
        # Flatten records for CSV
        # ----------------------------------------------------

        flattened_records = []

        for item in self.epoch_records:

            flattened = {

                key: value

                for key, value in item.items()

                if key not in (
                    "train_metrics",
                    "validation_metrics",
                    "extra"
                )

            }

            for key, value in item.get(
                "train_metrics",
                {}
            ).items():

                flattened[
                    f"train_{key}"
                ] = value

            for key, value in item.get(
                "validation_metrics",
                {}
            ).items():

                flattened[
                    f"validation_{key}"
                ] = value

            for key, value in item.get(
                "extra",
                {}
            ).items():

                flattened[
                    f"extra_{key}"
                ] = value

            flattened_records.append(flattened)

        atomic_write_csv(

            flattened_records,

            self.epoch_csv_path

        )

        # ----------------------------------------------------
        # Text log
        # ----------------------------------------------------

        self._write_log(

            f"Epoch {epoch} completed | "

            f"Train Loss: {train_loss} | "

            f"Validation Loss: {validation_loss} | "

            f"LR: {learning_rate} | "

            f"Duration: {epoch_duration}"

        )

        return record

    # ========================================================
    # BEST MODEL INFORMATION
    # ========================================================

    def update_best(
        self,
        epoch,
        metric_name,
        metric_value
    ):

        self.best_epoch = int(epoch)

        self.best_metric = {

            "name": str(metric_name),

            "value": float(metric_value),

        }

        self._write_log(

            f"New best result: "

            f"{metric_name}={metric_value} "

            f"at epoch {epoch}"

        )

    # ========================================================
    # FINALIZE RUN
    # ========================================================

    def finalize(
        self,
        status="completed",
        best_epoch=None,
        best_metric=None,
        additional_information=None
    ):

        allowed_statuses = {

            "completed",

            "early_stopped",

            "interrupted",

            "failed",

        }

        if status not in allowed_statuses:

            raise ValueError(
                f"Invalid run status: {status}"
            )

        self.status = status

        self.end_timestamp = current_timestamps()

        total_seconds = float(
            self.elapsed_seconds()
        )

        if best_epoch is not None:

            self.best_epoch = best_epoch

        if best_metric is not None:

            self.best_metric = best_metric

        summary = {

            "run_name": self.run_name,

            "status": self.status,

            "start_timestamp": self.start_timestamp,

            "end_timestamp": self.end_timestamp,

            "total_training_time_seconds": total_seconds,

            "total_training_time_minutes": (
                total_seconds / 60
            ),

            "total_training_time_hours": (
                total_seconds / 3600
            ),

            "epochs_completed": self.epochs_completed,

            "batch_records_saved": self.batch_records_count,

            "best_epoch": self.best_epoch,

            "best_metric": self.best_metric,

            "additional_information": (
                additional_information or {}
            ),

        }

        atomic_write_json(
            summary,
            self.summary_path
        )

        # Update master history status.

        atomic_write_json(

            {

                "run_name": self.run_name,

                "status": self.status,

                "start_timestamp": self.start_timestamp,

                "end_timestamp": self.end_timestamp,

                "epochs_completed": self.epochs_completed,

                "epochs": self.epoch_records,

            },

            self.epoch_json_path

        )

        self._write_log(

            f"TRAINING RUN FINALIZED: {self.status}"

        )

        self._write_log(

            f"Total training time: {total_seconds:.2f} seconds"

        )

        self._write_log(

            f"Epochs completed: {self.epochs_completed}"

        )

        return summary


# ============================================================
# HISTORY LOGGER TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF - TRAINING HISTORY LOGGER TEST")

    print("=" * 70)

    project_root = Path(__file__).resolve().parents[2]

    test_root = project_root / "Pipeline_Test"

    test_root.mkdir(
        parents=True,
        exist_ok=True
    )

    import tempfile

    with tempfile.TemporaryDirectory(

        prefix="history_logger_test_",

        dir=test_root

    ) as temporary_directory:

        # ----------------------------------------------------
        # Initialize logger
        # ----------------------------------------------------

        logger = TrainingHistoryLogger(

            output_dir=temporary_directory,

            run_name="history_smoke_test",

            save_batch_history=True

        )

        print("[PASS] History logger initialization")

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        logger.save_metadata(

            config={

                "epochs": 2,

                "batch_size": 16,

                "learning_rate": 0.0001,

            },

            model_metadata={

                "model_name": "resnet50",

                "num_classes": 7,

            },

            dataset_metadata={

                "train": 2623,

                "validation": 564,

                "test": 562,

                "split_seed": 42,

            },

            environment_metadata={

                "device": "cuda",

                "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",

            }

        )

        assert logger.metadata_path.is_file()

        print("[PASS] Run metadata saving")

        # ----------------------------------------------------
        # Batch history
        # ----------------------------------------------------

        for epoch in range(1, 3):

            for batch in range(1, 4):

                logger.log_batch(

                    epoch=epoch,

                    batch=batch,

                    global_step=(
                        (epoch - 1) * 3 + batch
                    ),

                    loss=1.0 / epoch,

                    accuracy=0.80,

                    learning_rate=0.0001,

                    batch_duration_seconds=0.25,

                    samples=16,

                    split="train",

                    gpu_memory_allocated_mb=1500,

                    gpu_memory_reserved_mb=2000,

                )

        assert logger.batch_records_count == 6

        print("[PASS] Batch-wise history")

        # ----------------------------------------------------
        # Epoch history
        # ----------------------------------------------------

        for epoch in range(1, 3):

            logger.log_epoch(

                epoch=epoch,

                train_metrics={

                    "metrics": {

                        "accuracy": 0.80 + epoch * 0.01,

                        "precision_macro": 0.79,

                        "recall_macro": 0.80,

                        "f1_macro": 0.795,

                        "balanced_accuracy": 0.80,

                        "roc_auc_ovr_macro": 0.94,

                        "pr_auc_macro_average_precision": 0.85,

                    }

                },

                validation_metrics={

                    "metrics": {

                        "accuracy": 0.78 + epoch * 0.01,

                        "precision_macro": 0.77,

                        "recall_macro": 0.78,

                        "f1_macro": 0.775,

                        "balanced_accuracy": 0.78,

                        "roc_auc_ovr_macro": 0.92,

                        "pr_auc_macro_average_precision": 0.83,

                    }

                },

                learning_rate=0.0001,

                train_loss=0.50 / epoch,

                validation_loss=0.60 / epoch,

                epoch_duration_seconds=10.5,

                train_duration_seconds=8.5,

                validation_duration_seconds=2.0,

                train_samples=2623,

                validation_samples=564,

                samples_per_second=300,

                gpu_memory={

                    "allocated_mb": 1500,

                    "reserved_mb": 2000,

                    "peak_allocated_mb": 1700,

                    "peak_reserved_mb": 2100,

                },

                extra={

                    "optimizer": "adamw",

                    "scheduler": "cosine",

                }

            )

        assert logger.epochs_completed == 2

        print("[PASS] Epoch-wise history")

        print("[PASS] Training/validation metrics")

        print("[PASS] Epoch duration and timestamps")

        print("[PASS] GPU memory history")

        # ----------------------------------------------------
        # Best epoch
        # ----------------------------------------------------

        logger.update_best(

            epoch=2,

            metric_name="val_macro_f1",

            metric_value=0.775

        )

        print("[PASS] Best epoch tracking")

        # ----------------------------------------------------
        # Finalize
        # ----------------------------------------------------

        summary = logger.finalize(

            status="completed"

        )

        assert summary["epochs_completed"] == 2

        assert summary["batch_records_saved"] == 6

        print("[PASS] Run finalization")

        # ----------------------------------------------------
        # Artifact validation
        # ----------------------------------------------------

        required_files = [

            logger.epoch_csv_path,

            logger.epoch_json_path,

            logger.batch_csv_path,

            logger.batch_jsonl_path,

            logger.metadata_path,

            logger.summary_path,

            logger.log_path,

        ]

        for filepath in required_files:

            assert filepath.is_file(), (
                f"Missing file: {filepath.name}"
            )

            assert filepath.stat().st_size > 0

            print(
                f"[PASS] {filepath.name}"
            )

        # ----------------------------------------------------
        # History integrity
        # ----------------------------------------------------

        with open(
            logger.epoch_json_path,
            "r",
            encoding="utf-8"
        ) as file:

            saved_history = json.load(file)

        assert len(
            saved_history["epochs"]
        ) == 2

        assert (
            saved_history["status"]
            == "completed"
        )

        print("[PASS] Saved history integrity")

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\nTRAINING HISTORY LOGGER TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
