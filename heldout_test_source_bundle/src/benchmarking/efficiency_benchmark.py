"""
MAIZE LEAF CLASSIFICATION
EFFICIENCY BENCHMARKING ENGINE

Project: D:/MaizeLeaf

Measures:
- Total, trainable and frozen parameters
- Estimated MACs and FLOPs per image
- Model state size and checkpoint size
- Inference latency: mean, median, P95
- Throughput: images per second
- GPU peak allocated and reserved memory
- Original training duration

Uses the best trained checkpoint.
Does not access train, validation or test image files.
Uses synthetic input for model profiling and timing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Models.model_factory import (
    build_model_from_config,
    get_model_metadata,
    get_parameter_information,
)

from src.engine.checkpoint import CheckpointManager

from src.data.class_names import NUM_CLASSES

from src.utils.device import (
    get_amp_settings,
    autocast_context,
)


EXPECTED_SPLIT_FINGERPRINT = (
    "8a64e1a8a53eecd2dd2eeedbfb63c7e783a193304a2529da1be4e8c7bff441de"
)


# ============================================================
# UTILITIES
# ============================================================

def write_json(data, path):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

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

    os.replace(temporary, path)

    return path


def extract_logits(output):

    if torch.is_tensor(output):
        logits = output

    elif isinstance(output, dict):
        if "logits" not in output:
            raise ValueError(
                "Model output dictionary has no logits."
            )
        logits = output["logits"]

    elif isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Empty model output.")
        logits = output[0]

    else:
        raise TypeError(
            f"Unsupported output type: {type(output)}"
        )

    if not torch.is_tensor(logits):
        raise TypeError("Model logits must be a tensor.")

    if logits.ndim != 2:
        raise ValueError(
            f"Expected [batch, classes], got {logits.shape}"
        )

    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} output classes, "
            f"got {logits.shape[1]}"
        )

    return logits


def synchronize(device):

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def human_time(seconds):

    if seconds is None:
        return "N/A"

    seconds = int(seconds)

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return (
        f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    )


def get_training_duration(run_dir):

    possible_files = [
        run_dir / "run_summary.json",
        run_dir / "training_history" / "run_summary.json",
    ]

    for path in possible_files:

        if not path.is_file():
            continue

        try:

            data = json.loads(
                path.read_text(encoding="utf-8")
            )

            value = data.get(
                "total_training_time_seconds"
            )

            if value is None:

                additional = data.get(
                    "additional_information",
                    {},
                )

                value = additional.get(
                    "total_training_time_seconds"
                )

            if value is not None:
                return float(value)

        except (ValueError, OSError):
            continue

    return None


def get_split_fingerprint(run_dir):

    summary_path = run_dir / "run_summary.json"

    if summary_path.is_file():

        data = json.loads(
            summary_path.read_text(encoding="utf-8")
        )

        fingerprint = data.get(
            "split_assignment_fingerprint_sha256"
        )

        if fingerprint:
            return fingerprint

    metadata_path = (
        run_dir / "experiment_metadata.json"
    )

    if metadata_path.is_file():

        data = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        dataset = data.get("dataset", {})

        split = dataset.get("split_metadata", {})

        fingerprint = split.get(
            "split_assignment_fingerprint_sha256"
        )

        if fingerprint:
            return fingerprint

    raise RuntimeError(
        "Could not retrieve the run's split fingerprint. "
        "Benchmarking stopped to protect experiment traceability."
    )


# ============================================================
# MODEL LOADING
# ============================================================

def load_trained_model(run_dir, device):

    config_path = run_dir / "training_config.json"

    checkpoint_path = (
        run_dir / "checkpoints" / "best_model.pth"
    )

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config_data = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    if "config" in config_data:
        config_data = config_data["config"]

    model_name = config_data.get(
        "model",
        run_dir.parent.name,
    )

    # The trained checkpoint contains the complete weights.
    # Do not download pretrained weights again.
    config_data["model"] = model_name
    config_data["pretrained"] = False
    config_data.setdefault("freeze_backbone", False)

    config_data.setdefault(
        "num_classes",
        NUM_CLASSES,
    )

    config = SimpleNamespace(**config_data)

    print("\nBUILDING MODEL")

    model = build_model_from_config(config)

    model = model.to(device)

    print("Loading best checkpoint...")

    manager = CheckpointManager(
        output_dir=run_dir,
        save_last=True,
    )

    checkpoint_information = manager.load_model_weights(
        model,
        checkpoint_path=checkpoint_path,
        strict=True,
    )

    model.eval()

    print("[PASS] Model architecture reconstructed")
    print("[PASS] Best checkpoint loaded")

    return (
        model,
        config,
        checkpoint_path,
        checkpoint_information,
    )


# ============================================================
# PARAMETER AND MODEL SIZE
# ============================================================

def calculate_model_information(
    model,
    checkpoint_path,
):

    parameter_info = get_parameter_information(model)

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    frozen_parameters = (
        total_parameters - trainable_parameters
    )

    parameter_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    buffer_bytes = sum(
        buffer.numel() * buffer.element_size()
        for buffer in model.buffers()
    )

    state_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
        if torch.is_tensor(tensor)
    )

    checkpoint_bytes = checkpoint_path.stat().st_size

    information = {

        "total_parameters": int(total_parameters),

        "trainable_parameters": int(trainable_parameters),

        "frozen_parameters": int(frozen_parameters),

        "trainable_percentage": (
            100.0 * trainable_parameters
            / max(total_parameters, 1)
        ),

        "frozen_percentage": (
            100.0 * frozen_parameters
            / max(total_parameters, 1)
        ),

        "parameter_storage_bytes": int(parameter_bytes),

        "parameter_storage_mb": (
            parameter_bytes / (1024 ** 2)
        ),

        "buffer_storage_bytes": int(buffer_bytes),

        "state_dict_storage_bytes": int(state_bytes),

        "state_dict_storage_mb": (
            state_bytes / (1024 ** 2)
        ),

        "best_checkpoint_bytes": int(checkpoint_bytes),

        "best_checkpoint_mb": (
            checkpoint_bytes / (1024 ** 2)
        ),

        "factory_parameter_information": parameter_info,
    }

    return information


# ============================================================
# MAC AND FLOP COUNTING
# ============================================================

def calculate_macs(model, image_size, device):

    """
    Estimate MACs using forward hooks.

    Counted:
        Conv1d
        Conv2d
        ConvTranspose1d
        ConvTranspose2d
        Linear
        MultiheadAttention

    One MAC is conventionally reported as two FLOPs.

    Elementwise operations, activation functions, normalization,
    pooling and custom functional operations are not included.
    """

    counters = {
        "macs": 0,
        "conv1d_macs": 0,
        "conv2d_macs": 0,
        "conv_transpose_macs": 0,
        "linear_macs": 0,
        "attention_macs": 0,
    }

    handles = []

    attention_output_projections = {
        id(module.out_proj)
        for module in model.modules()
        if isinstance(module, nn.MultiheadAttention)
    }

    def conv_hook(module, inputs, output):

        if not torch.is_tensor(output):
            return

        spatial_kernel = math.prod(
            module.kernel_size
        )

        macs = (
            output.numel()
            * (module.in_channels // module.groups)
            * spatial_kernel
        )

        counters["macs"] += int(macs)

        if isinstance(module, nn.Conv1d):
            counters["conv1d_macs"] += int(macs)
        else:
            counters["conv2d_macs"] += int(macs)

    def transpose_hook(module, inputs, output):

        if not torch.is_tensor(output):
            return

        spatial_kernel = math.prod(
            module.kernel_size
        )

        macs = (
            output.numel()
            * (module.in_channels // module.groups)
            * spatial_kernel
        )

        counters["macs"] += int(macs)

        counters["conv_transpose_macs"] += int(macs)

    def linear_hook(module, inputs, output):

        if not torch.is_tensor(output):
            return

        macs = (
            output.numel()
            * module.in_features
        )

        counters["macs"] += int(macs)

        counters["linear_macs"] += int(macs)

    def attention_hook(module, inputs, output):

        query = inputs[0]
        key = inputs[1] if len(inputs) > 1 else query
        value = inputs[2] if len(inputs) > 2 else key

        if query is None or key is None or value is None:
            return

        if query.ndim == 2:

            batch = 1
            q_length = query.shape[0]
            k_length = key.shape[0]

        elif module.batch_first:

            batch = query.shape[0]
            q_length = query.shape[1]
            k_length = key.shape[1]

        else:

            batch = query.shape[1]
            q_length = query.shape[0]
            k_length = key.shape[0]

        embed_dim = module.embed_dim
        num_heads = module.num_heads

        key_dim = module.kdim or embed_dim
        value_dim = module.vdim or embed_dim

        head_dim = embed_dim // num_heads

        q_projection = (
            batch * q_length * embed_dim * embed_dim
        )

        k_projection = (
            batch * k_length * key_dim * embed_dim
        )

        v_projection = (
            batch * k_length * value_dim * embed_dim
        )

        output_projection = (
            batch * q_length * embed_dim * embed_dim
        )

        attention_products = (
            2
            * batch
            * num_heads
            * q_length
            * k_length
            * head_dim
        )

        macs = (
            q_projection
            + k_projection
            + v_projection
            + output_projection
            + attention_products
        )

        counters["macs"] += int(macs)

        counters["attention_macs"] += int(macs)

    # Register hooks.
    for module in model.modules():

        if isinstance(module, nn.MultiheadAttention):

            handles.append(
                module.register_forward_hook(
                    attention_hook
                )
            )

        elif isinstance(module, nn.Conv1d):

            handles.append(
                module.register_forward_hook(
                    conv_hook
                )
            )

        elif isinstance(module, nn.Conv2d):

            handles.append(
                module.register_forward_hook(
                    conv_hook
                )
            )

        elif isinstance(
            module,
            (nn.ConvTranspose1d, nn.ConvTranspose2d),
        ):

            handles.append(
                module.register_forward_hook(
                    transpose_hook
                )
            )

        elif isinstance(module, nn.Linear):

            if id(module) in attention_output_projections:
                continue

            handles.append(
                module.register_forward_hook(
                    linear_hook
                )
            )

    was_training = model.training

    model.eval()

    dummy = torch.randn(
        1,
        3,
        image_size,
        image_size,
        device=device,
    )

    try:

        with torch.inference_mode():

            output = model(dummy)

            extract_logits(output)

        synchronize(device)

    finally:

        for handle in handles:
            handle.remove()

        model.train(was_training)

        del dummy

    macs = int(counters["macs"])
    flops = int(macs * 2)

    return {

        "input_shape": [
            1,
            3,
            image_size,
            image_size,
        ],

        "macs_per_image": macs,

        "gmacs_per_image": macs / 1e9,

        "flops_per_image": flops,

        "gflops_per_image": flops / 1e9,

        "operation_breakdown_macs": counters,

        "counting_method": (
            "Forward-hook estimate: convolution, linear and "
            "MultiheadAttention operations."
        ),

        "excluded_operations": [
            "Elementwise operations",
            "ReLU and other activations",
            "Batch normalization",
            "Pooling",
            "Softmax",
            "Custom functional operations not represented by counted modules",
        ],

        "flops_convention": (
            "1 MAC = 2 FLOPs (one multiplication and one addition)."
        ),
    }


# ============================================================
# INFERENCE BENCHMARK
# ============================================================

def benchmark_batch_size(
    model,
    batch_size,
    image_size,
    device,
    amp_settings,
    warmup,
    iterations,
):

    if batch_size < 1:
        raise ValueError("Batch size must be positive.")

    input_tensor = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    latencies_ms = []
    events = []

    baseline_allocated = None
    baseline_reserved = None

    try:

        model.eval()

        # Warmup
        with torch.inference_mode():

            for _ in range(warmup):

                with autocast_context(
                    device,
                    amp_settings,
                ):

                    output = model(input_tensor)

                    extract_logits(output)

        synchronize(device)

        if device.type == "cuda":

            baseline_allocated = (
                torch.cuda.memory_allocated(device)
            )

            baseline_reserved = (
                torch.cuda.memory_reserved(device)
            )

            torch.cuda.reset_peak_memory_stats(device)

        # Actual measurement
        synchronize(device)

        wall_start = time.perf_counter()

        with torch.inference_mode():

            for _ in range(iterations):

                if device.type == "cuda":

                    start_event = torch.cuda.Event(
                        enable_timing=True
                    )

                    end_event = torch.cuda.Event(
                        enable_timing=True
                    )

                    start_event.record()

                    with autocast_context(
                        device,
                        amp_settings,
                    ):

                        output = model(input_tensor)

                        extract_logits(output)

                    end_event.record()

                    events.append(
                        (start_event, end_event)
                    )

                else:

                    iteration_start = time.perf_counter()

                    with autocast_context(
                        device,
                        amp_settings,
                    ):

                        output = model(input_tensor)

                        extract_logits(output)

                    iteration_end = time.perf_counter()

                    latencies_ms.append(
                        (
                            iteration_end
                            - iteration_start
                        ) * 1000.0
                    )

        synchronize(device)

        wall_seconds = (
            time.perf_counter() - wall_start
        )

        if device.type == "cuda":

            latencies_ms = [
                start.elapsed_time(end)
                for start, end in events
            ]

            peak_allocated = (
                torch.cuda.max_memory_allocated(device)
            )

            peak_reserved = (
                torch.cuda.max_memory_reserved(device)
            )

            additional_peak_allocated = max(
                0,
                peak_allocated - baseline_allocated,
            )

            additional_peak_reserved = max(
                0,
                peak_reserved - baseline_reserved,
            )

        else:

            peak_allocated = None
            peak_reserved = None

            additional_peak_allocated = None
            additional_peak_reserved = None

        total_images = batch_size * iterations

        mean_latency = statistics.mean(
            latencies_ms
        )

        median_latency = statistics.median(
            latencies_ms
        )

        p95_latency = float(
            np.percentile(
                latencies_ms,
                95,
            )
        )

        throughput = (
            total_images / wall_seconds
            if wall_seconds > 0
            else None
        )

        result = {

            "batch_size": batch_size,

            "status": "success",

            "warmup_iterations": warmup,

            "measurement_iterations": iterations,

            "total_images": total_images,

            "mean_latency_ms": mean_latency,

            "median_latency_ms": median_latency,

            "p95_latency_ms": p95_latency,

            "minimum_latency_ms": min(latencies_ms),

            "maximum_latency_ms": max(latencies_ms),

            "per_image_mean_latency_ms": (
                mean_latency / batch_size
            ),

            "throughput_images_per_second": throughput,

            "measurement_wall_seconds": wall_seconds,

            "gpu_peak_allocated_mb": (
                peak_allocated / (1024 ** 2)
                if peak_allocated is not None
                else None
            ),

            "gpu_peak_reserved_mb": (
                peak_reserved / (1024 ** 2)
                if peak_reserved is not None
                else None
            ),

            "gpu_additional_peak_allocated_mb": (
                additional_peak_allocated / (1024 ** 2)
                if additional_peak_allocated is not None
                else None
            ),

            "gpu_additional_peak_reserved_mb": (
                additional_peak_reserved / (1024 ** 2)
                if additional_peak_reserved is not None
                else None
            ),
        }

        return result

    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:

        if (
            device.type == "cuda"
            and (
                isinstance(error, torch.cuda.OutOfMemoryError)
                or "out of memory" in str(error).lower()
            )
        ):

            torch.cuda.empty_cache()

            return {
                "batch_size": batch_size,
                "status": "out_of_memory",
                "error": str(error),
                "throughput_images_per_second": None,
                "mean_latency_ms": None,
                "median_latency_ms": None,
                "p95_latency_ms": None,
            }

        raise

    finally:

        del input_tensor

        if device.type == "cuda":
            torch.cuda.empty_cache()


# ============================================================
# REPORT GENERATION
# ============================================================

def save_benchmark_results(
    run_dir,
    model,
    model_metadata,
    model_information,
    operation_information,
    results,
    training_duration,
    device,
    amp_settings,
    image_size,
    checkpoint_path,
    split_fingerprint,
):

    output_dir = run_dir / "benchmarking"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gpu_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None
    )

    gpu_total_memory = (
        torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else None
    )

    summary = {

        "model_name": run_dir.parent.name,

        "run_name": run_dir.name,

        "timestamp": datetime.now().astimezone().isoformat(),

        "device": str(device),

        "gpu_name": gpu_name,

        "gpu_total_memory_gb": (
            gpu_total_memory / (1024 ** 3)
            if gpu_total_memory is not None
            else None
        ),

        "pytorch_version": torch.__version__,

        "cuda_version": torch.version.cuda,

        "image_size": image_size,

        "num_classes": NUM_CLASSES,

        "split_assignment_fingerprint_sha256": split_fingerprint,

        "checkpoint": str(checkpoint_path),

        "training_duration_seconds": training_duration,

        "training_duration_formatted": human_time(
            training_duration
        ),

        "amp": {
            "enabled": amp_settings["enabled"],
            "dtype": amp_settings["dtype"],
            "reason": amp_settings["reason"],
        },

        "model_metadata": model_metadata,

        "model_information": model_information,

        "operation_information": operation_information,

        "inference_benchmarks": results,

        "benchmark_methodology": {
            "input_source": "Synthetic random tensors",
            "dataset_accessed": False,
            "train_images_used": False,
            "validation_images_used": False,
            "test_images_used": False,
            "model_mode": "eval",
            "gradient_calculation": False,
            "warmup_excluded_from_latency": True,
            "throughput_includes_python_loop_overhead": True,
        },
    }

    json_path = output_dir / "efficiency_benchmark.json"

    write_json(summary, json_path)

    # CSV
    csv_path = output_dir / "efficiency_benchmark.csv"

    fields = [
        "batch_size",
        "status",
        "mean_latency_ms",
        "median_latency_ms",
        "p95_latency_ms",
        "minimum_latency_ms",
        "maximum_latency_ms",
        "per_image_mean_latency_ms",
        "throughput_images_per_second",
        "measurement_wall_seconds",
        "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb",
        "gpu_additional_peak_allocated_mb",
        "gpu_additional_peak_reserved_mb",
    ]

    with open(
        csv_path,
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

        for row in results:
            writer.writerow(row)

    # Text report
    report_path = (
        output_dir / "efficiency_benchmark_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write("=" * 88 + "\n")

        file.write(
            "MAIZE LEAF CLASSIFICATION - EFFICIENCY BENCHMARK REPORT\n"
        )

        file.write("=" * 88 + "\n\n")

        file.write(
            f"Model: {run_dir.parent.name}\n"
        )

        file.write(
            f"Run: {run_dir.name}\n"
        )

        file.write(
            f"Timestamp: {summary['timestamp']}\n"
        )

        file.write(
            f"Device: {device}\n"
        )

        file.write(
            f"GPU: {gpu_name or 'CPU'}\n"
        )

        file.write(
            f"Image size: {image_size}x{image_size}\n"
        )

        file.write(
            f"Classes: {NUM_CLASSES}\n\n"
        )

        file.write("MODEL PARAMETERS\n")
        file.write("-" * 88 + "\n")

        file.write(
            f"Total parameters: "
            f"{model_information['total_parameters']:,}\n"
        )

        file.write(
            f"Trainable parameters: "
            f"{model_information['trainable_parameters']:,}\n"
        )

        file.write(
            f"Frozen parameters: "
            f"{model_information['frozen_parameters']:,}\n"
        )

        file.write(
            f"Parameter storage: "
            f"{model_information['parameter_storage_mb']:.3f} MB\n"
        )

        file.write(
            f"State dict storage: "
            f"{model_information['state_dict_storage_mb']:.3f} MB\n"
        )

        file.write(
            f"Best checkpoint size: "
            f"{model_information['best_checkpoint_mb']:.3f} MB\n\n"
        )

        file.write("COMPUTATIONAL COMPLEXITY\n")
        file.write("-" * 88 + "\n")

        file.write(
            f"MACs/image: "
            f"{operation_information['macs_per_image']:,}\n"
        )

        file.write(
            f"GMACs/image: "
            f"{operation_information['gmacs_per_image']:.6f}\n"
        )

        file.write(
            f"FLOPs/image: "
            f"{operation_information['flops_per_image']:,}\n"
        )

        file.write(
            f"GFLOPs/image: "
            f"{operation_information['gflops_per_image']:.6f}\n\n"
        )

        file.write(
            "NOTE: MAC/FLOP counts are estimates from supported "
            "module hooks. See JSON for excluded operations.\n\n"
        )

        file.write("TRAINING INFORMATION\n")
        file.write("-" * 88 + "\n")

        file.write(
            f"Total training duration: "
            f"{human_time(training_duration)}\n"
        )

        file.write(
            f"Training duration seconds: "
            f"{training_duration}\n\n"
        )

        file.write("INFERENCE PERFORMANCE\n")
        file.write("-" * 88 + "\n")

        for row in results:

            file.write(
                f"\nBatch size: {row['batch_size']}\n"
            )

            file.write(
                f"Status: {row['status']}\n"
            )

            if row["status"] != "success":
                file.write(
                    f"Error: {row.get('error', 'N/A')}\n"
                )
                continue

            file.write(
                f"Mean latency: "
                f"{row['mean_latency_ms']:.4f} ms\n"
            )

            file.write(
                f"Median latency: "
                f"{row['median_latency_ms']:.4f} ms\n"
            )

            file.write(
                f"P95 latency: "
                f"{row['p95_latency_ms']:.4f} ms\n"
            )

            file.write(
                f"Per-image latency: "
                f"{row['per_image_mean_latency_ms']:.4f} ms\n"
            )

            file.write(
                f"Throughput: "
                f"{row['throughput_images_per_second']:.4f} images/sec\n"
            )

            file.write(
                f"GPU peak allocated: "
                f"{row.get('gpu_peak_allocated_mb')} MB\n"
            )

            file.write(
                f"GPU peak reserved: "
                f"{row.get('gpu_peak_reserved_mb')} MB\n"
            )

        file.write("\n" + "=" * 88 + "\n")

        file.write(
            "END OF EFFICIENCY BENCHMARK REPORT\n"
        )

        file.write("=" * 88 + "\n")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "report": str(report_path),
    }


# ============================================================
# MAIN
# ============================================================

def run_benchmark(args):

    run_dir = Path(args.run_dir).expanduser().resolve()

    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)

    if args.warmup < 0 or args.iterations < 1:
        raise ValueError(
            "Warmup must be >= 0 and iterations must be >= 1."
        )

    if any(size < 1 for size in args.batch_sizes):
        raise ValueError("All batch sizes must be positive.")

    if args.device == "auto":

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    else:

        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA requested but CUDA is unavailable."
        )

    print("\n" + "=" * 76)
    print("MAIZE LEAF - EFFICIENCY BENCHMARKING ENGINE")
    print("=" * 76)

    print(f"Run directory : {run_dir}")
    print(f"Device        : {device}")

    if device.type == "cuda":
        print(
            f"GPU           : {torch.cuda.get_device_name(device)}"
        )

    split_fingerprint = get_split_fingerprint(run_dir)

    if split_fingerprint != EXPECTED_SPLIT_FINGERPRINT:

        raise RuntimeError(
            "Run split fingerprint does not match the fixed "
            "7-class research split. Benchmark aborted."
        )

    print("[PASS] Fixed split fingerprint")

    (
        model,
        config,
        checkpoint_path,
        checkpoint_information,
    ) = load_trained_model(
        run_dir,
        device,
    )

    image_size = int(
        config.image_size
    )

    amp_settings = get_amp_settings(
        device=device,
        enabled=args.amp,
        dtype=args.amp_dtype,
    )

    print("\nMODEL INFORMATION")

    model_metadata = get_model_metadata(model)

    model_information = calculate_model_information(
        model,
        checkpoint_path,
    )

    print(
        f"Total parameters     : "
        f"{model_information['total_parameters']:,}"
    )

    print(
        f"Trainable parameters : "
        f"{model_information['trainable_parameters']:,}"
    )

    print(
        f"Frozen parameters    : "
        f"{model_information['frozen_parameters']:,}"
    )

    print(
        f"Checkpoint size      : "
        f"{model_information['best_checkpoint_mb']:.2f} MB"
    )

    print("\nCOMPUTATIONAL COMPLEXITY")

    operation_information = calculate_macs(
        model=model,
        image_size=image_size,
        device=device,
    )

    print(
        f"MACs/image : "
        f"{operation_information['macs_per_image']:,}"
    )

    print(
        f"FLOPs/image: "
        f"{operation_information['flops_per_image']:,}"
    )

    print(
        "\nINFERENCE BENCHMARK"
    )

    print(
        f"AMP        : {amp_settings['enabled']}"
    )

    print(
        f"Warmup     : {args.warmup}"
    )

    print(
        f"Iterations : {args.iterations}"
    )

    results = []

    for batch_size in args.batch_sizes:

        print("\n" + "-" * 70)

        print(
            f"Benchmarking batch size: {batch_size}"
        )

        result = benchmark_batch_size(
            model=model,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
            amp_settings=amp_settings,
            warmup=args.warmup,
            iterations=args.iterations,
        )

        results.append(result)

        if result["status"] == "success":

            print(
                f"Mean latency : "
                f"{result['mean_latency_ms']:.4f} ms"
            )

            print(
                f"Median       : "
                f"{result['median_latency_ms']:.4f} ms"
            )

            print(
                f"P95          : "
                f"{result['p95_latency_ms']:.4f} ms"
            )

            print(
                f"Throughput   : "
                f"{result['throughput_images_per_second']:.2f} images/sec"
            )

            print(
                f"Peak GPU RAM : "
                f"{result.get('gpu_peak_allocated_mb')} MB"
            )

            print("[PASS] Batch benchmark")

        else:

            print(
                f"[WARNING] Batch size {batch_size}: "
                f"{result['status']}"
            )

    training_duration = get_training_duration(
        run_dir
    )

    output_files = save_benchmark_results(
        run_dir=run_dir,
        model=model,
        model_metadata=model_metadata,
        model_information=model_information,
        operation_information=operation_information,
        results=results,
        training_duration=training_duration,
        device=device,
        amp_settings=amp_settings,
        image_size=image_size,
        checkpoint_path=checkpoint_path,
        split_fingerprint=split_fingerprint,
    )

    print("\n" + "=" * 76)
    print("EFFICIENCY BENCHMARK COMPLETED")
    print("=" * 76)

    print(f"Training time : {human_time(training_duration)}")

    print(f"JSON          : {output_files['json']}")

    print(f"CSV           : {output_files['csv']}")

    print(f"Report        : {output_files['report']}")

    print("=" * 76)

    return output_files


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Maize Leaf Classification Efficiency Benchmark"
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed model run directory.",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "cuda:0"],
        default="auto",
    )

    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 16],
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--amp",
        action="store_true",
    )

    parser.add_argument(
        "--amp-dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )

    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()