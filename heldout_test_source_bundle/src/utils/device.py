
"""
Maize Leaf Classification
GPU / CPU Device Management

Features:
1. Automatic CUDA detection.
2. Manual GPU/CPU selection.
3. GPU information and memory monitoring.
4. FP16 / BF16 Automatic Mixed Precision.
5. Gradient scaler configuration.
6. CUDA synchronization.
7. Peak GPU memory tracking.

Tested design:
    Windows
    NVIDIA CUDA
    PyTorch 2.x
"""

from contextlib import nullcontext
from dataclasses import dataclass

import torch


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_DEVICE = "auto"

DEFAULT_AMP = True

DEFAULT_AMP_DTYPE = "float16"


# ============================================================
# DEVICE SELECTION
# ============================================================

def select_device(device=DEFAULT_DEVICE):
    """
    Select CPU or NVIDIA CUDA device.

    Supported:
        auto
        cpu
        cuda
        cuda:0
        cuda:1
    """

    device = str(device).lower().strip()

    if device == "auto":

        if torch.cuda.is_available():

            return torch.device("cuda:0")

        return torch.device("cpu")

    if device == "cpu":

        return torch.device("cpu")

    if device == "cuda":

        device = "cuda:0"

    selected_device = torch.device(device)

    if selected_device.type == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        index = (
            selected_device.index
            if selected_device.index is not None
            else 0
        )

        if index >= torch.cuda.device_count():

            raise ValueError(
                f"CUDA device {index} does not exist. "
                f"Available GPUs: {torch.cuda.device_count()}"
            )

        selected_device = torch.device(
            f"cuda:{index}"
        )

    elif selected_device.type != "cpu":

        raise ValueError(
            f"Unsupported device: {device}"
        )

    return selected_device


# ============================================================
# GPU INFORMATION
# ============================================================

def get_device_info(device=None):

    if device is None:

        device = select_device()

    device = torch.device(device)

    info = {

        "device": str(device),

        "device_type": device.type,

        "cuda_available": torch.cuda.is_available(),

        "pytorch_version": torch.__version__,

        "cuda_version": torch.version.cuda,

        "gpu_name": None,

        "gpu_index": None,

        "gpu_memory_total_gb": None,

        "gpu_memory_free_gb": None,

        "gpu_memory_allocated_gb": None,

        "gpu_memory_reserved_gb": None,

        "compute_capability": None,

        "bf16_supported": False,

    }

    if device.type == "cuda":

        index = device.index or 0

        properties = torch.cuda.get_device_properties(
            index
        )

        total_memory = properties.total_memory

        free_memory, _ = torch.cuda.mem_get_info(
            device
        )

        info.update({

            "gpu_name": properties.name,

            "gpu_index": index,

            "gpu_memory_total_gb": round(
                total_memory / (1024 ** 3), 3
            ),

            "gpu_memory_free_gb": round(
                free_memory / (1024 ** 3), 3
            ),

            "gpu_memory_allocated_gb": round(
                torch.cuda.memory_allocated(device)
                / (1024 ** 3), 3
            ),

            "gpu_memory_reserved_gb": round(
                torch.cuda.memory_reserved(device)
                / (1024 ** 3), 3
            ),

            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),

            "bf16_supported": (
                torch.cuda.is_bf16_supported()
            ),

        })

    return info


# ============================================================
# AMP CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class AMPConfig:

    enabled: bool = DEFAULT_AMP

    dtype: str = DEFAULT_AMP_DTYPE


def get_amp_settings(
    device,
    enabled=DEFAULT_AMP,
    dtype=DEFAULT_AMP_DTYPE
):
    """
    Configure Automatic Mixed Precision.

    Supported:
        float16 / fp16
        bfloat16 / bf16

    FP16 uses GradScaler.
    BF16 normally does not require GradScaler.
    """

    device = torch.device(device)

    dtype = str(dtype).lower().strip()

    aliases = {

        "fp16": "float16",

        "half": "float16",

        "float16": "float16",

        "bf16": "bfloat16",

        "bfloat16": "bfloat16",

    }

    if dtype not in aliases:

        raise ValueError(
            "AMP dtype must be float16 or bfloat16."
        )

    dtype = aliases[dtype]

    if not enabled:

        return {

            "enabled": False,

            "dtype": dtype,

            "torch_dtype": None,

            "grad_scaler": False,

            "reason": "AMP disabled by user.",

        }

    if device.type != "cuda":

        return {

            "enabled": False,

            "dtype": dtype,

            "torch_dtype": None,

            "grad_scaler": False,

            "reason": "CUDA unavailable or CPU selected.",

        }

    if dtype == "float16":

        return {

            "enabled": True,

            "dtype": dtype,

            "torch_dtype": torch.float16,

            "grad_scaler": True,

            "reason": "CUDA FP16 mixed precision enabled.",

        }

    if dtype == "bfloat16":

        if not torch.cuda.is_bf16_supported():

            raise RuntimeError(
                "BF16 is not supported by the selected GPU."
            )

        return {

            "enabled": True,

            "dtype": dtype,

            "torch_dtype": torch.bfloat16,

            "grad_scaler": False,

            "reason": "CUDA BF16 mixed precision enabled.",

        }


# ============================================================
# GRADIENT SCALER
# ============================================================

def create_grad_scaler(amp_settings):
    """
    Create GradScaler for FP16 CUDA training.

    BF16 and full precision do not require scaling.
    """

    enabled = bool(
        amp_settings["enabled"]
        and amp_settings["grad_scaler"]
    )

    if hasattr(torch, "amp") and hasattr(
        torch.amp, "GradScaler"
    ):

        try:

            return torch.amp.GradScaler(
                "cuda",
                enabled=enabled
            )

        except TypeError:

            return torch.amp.GradScaler(
                enabled=enabled
            )

    return torch.cuda.amp.GradScaler(
        enabled=enabled
    )


# ============================================================
# AUTOCAST CONTEXT
# ============================================================

def autocast_context(
    device,
    amp_settings
):
    """
    Return an autocast context for model forward passes.
    """

    device = torch.device(device)

    if not amp_settings["enabled"]:

        return nullcontext()

    if device.type != "cuda":

        return nullcontext()

    return torch.autocast(
        device_type="cuda",
        dtype=amp_settings["torch_dtype"],
        enabled=True
    )


# ============================================================
# CUDA SYNCHRONIZATION
# ============================================================

def synchronize_device(device=None):

    if device is None:

        device = select_device()

    device = torch.device(device)

    if device.type == "cuda":

        torch.cuda.synchronize(device)


# ============================================================
# GPU MEMORY MANAGEMENT
# ============================================================

def reset_peak_memory_stats(device=None):

    if device is None:

        device = select_device()

    device = torch.device(device)

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats(
            device
        )


def get_gpu_memory_stats(device=None):

    if device is None:

        device = select_device()

    device = torch.device(device)

    if device.type != "cuda":

        return {

            "allocated_gb": 0.0,

            "reserved_gb": 0.0,

            "peak_allocated_gb": 0.0,

            "peak_reserved_gb": 0.0,

        }

    return {

        "allocated_gb": round(
            torch.cuda.memory_allocated(device)
            / (1024 ** 3), 4
        ),

        "reserved_gb": round(
            torch.cuda.memory_reserved(device)
            / (1024 ** 3), 4
        ),

        "peak_allocated_gb": round(
            torch.cuda.max_memory_allocated(device)
            / (1024 ** 3), 4
        ),

        "peak_reserved_gb": round(
            torch.cuda.max_memory_reserved(device)
            / (1024 ** 3), 4
        ),

    }


# ============================================================
# DEVICE DIAGNOSTICS
# ============================================================

def print_device_report(device=None):

    if device is None:

        device = select_device()

    info = get_device_info(device)

    print("\n" + "=" * 70)

    print("MAIZE LEAF - COMPUTE DEVICE REPORT")

    print("=" * 70)

    for key, value in info.items():

        print(f"{key:<32}: {value}")

    print("=" * 70)


# ============================================================
# EXECUTION TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF - GPU MANAGEMENT TEST")

    print("=" * 70)

    # --------------------------------------------------------
    # TEST 1: Device detection
    # --------------------------------------------------------

    device = select_device("auto")

    print(f"\nSelected device: {device}")

    print("[PASS] Automatic device selection")

    # --------------------------------------------------------
    # TEST 2: GPU information
    # --------------------------------------------------------

    print_device_report(device)

    # --------------------------------------------------------
    # TEST 3: AMP FP16 configuration
    # --------------------------------------------------------

    amp_settings = get_amp_settings(
        device,
        enabled=True,
        dtype="float16"
    )

    print("\nFP16 AMP CONFIGURATION")

    print(f"Enabled: {amp_settings['enabled']}")

    print(f"GradScaler: {amp_settings['grad_scaler']}")

    print(f"Reason: {amp_settings['reason']}")

    print("[PASS] FP16 AMP configuration")

    # --------------------------------------------------------
    # TEST 4: GradScaler
    # --------------------------------------------------------

    scaler = create_grad_scaler(
        amp_settings
    )

    if amp_settings["grad_scaler"]:

        assert scaler.is_enabled()

    else:

        assert not scaler.is_enabled()

    print("[PASS] GradScaler initialization")

    # --------------------------------------------------------
    # TEST 5: Autocast forward pass
    # --------------------------------------------------------

    model_input = torch.randn(
        32,
        128,
        device=device
    )

    model_weight = torch.randn(
        128,
        64,
        device=device
    )

    with autocast_context(
        device,
        amp_settings
    ):

        output = model_input @ model_weight

    assert output.shape == (32, 64)

    assert torch.isfinite(output).all()

    print("[PASS] AMP forward pass")

    # --------------------------------------------------------
    # TEST 6: Backward pass
    # --------------------------------------------------------

    model_input = torch.randn(
        32,
        128,
        device=device,
        requires_grad=True
    )

    with autocast_context(
        device,
        amp_settings
    ):

        output = model_input @ model_weight

        loss = output.float().square().mean()

    if scaler.is_enabled():

        scaler.scale(loss).backward()

    else:

        loss.backward()

    assert model_input.grad is not None

    assert torch.isfinite(model_input.grad).all()

    print("[PASS] AMP backward pass")

    # --------------------------------------------------------
    # TEST 7: BF16 if supported
    # --------------------------------------------------------

    if device.type == "cuda" and torch.cuda.is_bf16_supported():

        bf16_settings = get_amp_settings(
            device,
            enabled=True,
            dtype="bfloat16"
        )

        with autocast_context(
            device,
            bf16_settings
        ):

            bf16_output = model_input.detach() @ model_weight

        assert torch.isfinite(bf16_output).all()

        print("[PASS] BF16 forward pass")

    else:

        print("[SKIP] BF16 unavailable")

    # --------------------------------------------------------
    # TEST 8: GPU memory
    # --------------------------------------------------------

    synchronize_device(device)

    memory = get_gpu_memory_stats(device)

    print("\nGPU MEMORY")

    for key, value in memory.items():

        print(f"{key:<28}: {value} GB")

    print("[PASS] GPU memory monitoring")

    # --------------------------------------------------------
    # TEST 9: CPU selection
    # --------------------------------------------------------

    cpu_device = select_device("cpu")

    assert cpu_device.type == "cpu"

    print("[PASS] Manual CPU selection")

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\nGPU MANAGEMENT TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
