
"""
Maize Leaf Classification
Reproducibility and Random Seed Management

Controls:
1. Python random
2. NumPy random
3. PyTorch CPU
4. PyTorch CUDA
5. DataLoader workers
6. Deterministic backend settings
7. RNG state saving/restoration for checkpoint resuming

Import this module early in the training entry point,
before CUDA initialization.
"""

import os

# Set before importing torch where possible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import warnings

import numpy as np
import torch


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_SEED = 42


# ============================================================
# GLOBAL SEED
# ============================================================

def seed_everything(
    seed=DEFAULT_SEED,
    deterministic=True,
    warn_only=True
):
    """
    Set random seeds across Python, NumPy and PyTorch.

    Args:
        seed:
            Non-negative random seed.

        deterministic:
            Enable deterministic PyTorch algorithms and
            cuDNN settings where supported.

        warn_only:
            Warn instead of raising an exception when an
            operation has no deterministic implementation.

    Returns:
        Dictionary containing reproducibility settings.
    """

    if not isinstance(seed, int) or seed < 0:

        raise ValueError(
            "Seed must be a non-negative integer."
        )

    # Python hash seed is fixed at interpreter startup.
    # This environment value is also inherited by child processes.
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python random.
    random.seed(seed)

    # NumPy random.
    np.random.seed(seed)

    # PyTorch CPU.
    torch.manual_seed(seed)

    # PyTorch CUDA.
    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # cuDNN reproducibility.
    torch.backends.cudnn.deterministic = deterministic

    torch.backends.cudnn.benchmark = not deterministic

    # PyTorch deterministic algorithms.
    if hasattr(torch, "use_deterministic_algorithms"):

        try:

            torch.use_deterministic_algorithms(
                deterministic,
                warn_only=warn_only
            )

        except Exception as error:

            warnings.warn(
                f"Could not configure deterministic algorithms: {error}",
                RuntimeWarning
            )

    return {

        "seed": seed,

        "deterministic": deterministic,

        "warn_only": warn_only,

        "python_seeded": True,

        "numpy_seeded": True,

        "torch_cpu_seeded": True,

        "torch_cuda_seeded": torch.cuda.is_available(),

        "cudnn_deterministic": (
            torch.backends.cudnn.deterministic
        ),

        "cudnn_benchmark": (
            torch.backends.cudnn.benchmark
        ),

        "cuda_available": torch.cuda.is_available(),

    }


# ============================================================
# DATA LOADER WORKER SEED
# ============================================================

def seed_worker(worker_id):
    """
    Initialize Python and NumPy random states inside
    each PyTorch DataLoader worker.

    PyTorch automatically assigns a worker-specific
    initial seed using the DataLoader generator.
    """

    worker_seed = torch.initial_seed() % (2**32)

    random.seed(worker_seed)

    np.random.seed(worker_seed)


# ============================================================
# PYTORCH GENERATOR
# ============================================================

def create_generator(seed=DEFAULT_SEED):
    """
    Create a dedicated CPU generator for DataLoader
    shuffling and worker seed generation.
    """

    if not isinstance(seed, int) or seed < 0:

        raise ValueError(
            "Seed must be a non-negative integer."
        )

    generator = torch.Generator(device="cpu")

    generator.manual_seed(seed)

    return generator


# ============================================================
# RNG STATE CAPTURE
# ============================================================

def capture_rng_state():
    """
    Capture current random states.

    Save this dictionary inside training checkpoints
    to support reproducible training resumption.
    """

    state = {

        "python": random.getstate(),

        "numpy": np.random.get_state(),

        "torch_cpu": torch.get_rng_state(),

        "torch_cuda": None,

    }

    if torch.cuda.is_available():

        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


# ============================================================
# RNG STATE RESTORE
# ============================================================

def restore_rng_state(state):
    """
    Restore previously captured random states.

    Args:
        state:
            Dictionary returned by capture_rng_state().
    """

    required_keys = {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda"
    }

    missing = required_keys - set(state.keys())

    if missing:

        raise ValueError(
            f"Invalid RNG state. Missing keys: {missing}"
        )

    random.setstate(state["python"])

    np.random.set_state(state["numpy"])

    torch.set_rng_state(state["torch_cpu"])

    cuda_state = state["torch_cuda"]

    if cuda_state is not None:

        if torch.cuda.is_available():

            if len(cuda_state) != torch.cuda.device_count():

                warnings.warn(
                    "Saved CUDA RNG device count differs from "
                    "the current environment.",
                    RuntimeWarning
                )

            device_count = min(
                len(cuda_state),
                torch.cuda.device_count()
            )

            for index in range(device_count):

                torch.cuda.set_rng_state(
                    cuda_state[index],
                    device=index
                )

        else:

            warnings.warn(
                "Checkpoint contains CUDA RNG states, "
                "but CUDA is unavailable.",
                RuntimeWarning
            )


# ============================================================
# REPRODUCIBILITY INFORMATION
# ============================================================

def get_reproducibility_info(seed=DEFAULT_SEED):
    """
    Return environment and seed configuration
    for experiment metadata.
    """

    return {

        "seed": seed,

        "python_hash_seed": os.environ.get(
            "PYTHONHASHSEED"
        ),

        "cublas_workspace_config": os.environ.get(
            "CUBLAS_WORKSPACE_CONFIG"
        ),

        "numpy_version": np.__version__,

        "pytorch_version": torch.__version__,

        "cuda_available": torch.cuda.is_available(),

        "cuda_version": torch.version.cuda,

        "gpu_count": torch.cuda.device_count(),

        "gpu_names": [

            torch.cuda.get_device_name(index)

            for index in range(torch.cuda.device_count())

        ],

        "cudnn_version": torch.backends.cudnn.version(),

        "cudnn_deterministic": (
            torch.backends.cudnn.deterministic
        ),

        "cudnn_benchmark": (
            torch.backends.cudnn.benchmark
        ),

    }


# ============================================================
# EXECUTION TEST
# ============================================================

def main():

    print("=" * 70)

    print("MAIZE LEAF - REPRODUCIBILITY TEST")

    print("=" * 70)

    seed = 42

    print(f"\nTesting seed: {seed}")

    # --------------------------------------------------------
    # TEST 1: Global seeding
    # --------------------------------------------------------

    settings = seed_everything(
        seed=seed,
        deterministic=True,
        warn_only=True
    )

    print("\n[PASS] Global random seed configuration")

    # --------------------------------------------------------
    # TEST 2: Python reproducibility
    # --------------------------------------------------------

    random.seed(seed)

    python_first = [
        random.random()
        for _ in range(5)
    ]

    random.seed(seed)

    python_second = [
        random.random()
        for _ in range(5)
    ]

    assert python_first == python_second

    print("[PASS] Python random reproducibility")

    # --------------------------------------------------------
    # TEST 3: NumPy reproducibility
    # --------------------------------------------------------

    np.random.seed(seed)

    numpy_first = np.random.rand(5)

    np.random.seed(seed)

    numpy_second = np.random.rand(5)

    assert np.array_equal(
        numpy_first,
        numpy_second
    )

    print("[PASS] NumPy random reproducibility")

    # --------------------------------------------------------
    # TEST 4: PyTorch CPU reproducibility
    # --------------------------------------------------------

    torch.manual_seed(seed)

    torch_first = torch.rand(5)

    torch.manual_seed(seed)

    torch_second = torch.rand(5)

    assert torch.equal(
        torch_first,
        torch_second
    )

    print("[PASS] PyTorch CPU reproducibility")

    # --------------------------------------------------------
    # TEST 5: CUDA reproducibility
    # --------------------------------------------------------

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        cuda_first = torch.rand(
            5,
            device="cuda"
        )

        torch.cuda.manual_seed_all(seed)

        cuda_second = torch.rand(
            5,
            device="cuda"
        )

        assert torch.equal(
            cuda_first,
            cuda_second
        )

        print("[PASS] NVIDIA CUDA random reproducibility")

    else:

        print("[SKIP] CUDA unavailable")

    # --------------------------------------------------------
    # TEST 6: DataLoader generator
    # --------------------------------------------------------

    generator_one = create_generator(seed)

    generator_two = create_generator(seed)

    values_one = torch.rand(
        10,
        generator=generator_one
    )

    values_two = torch.rand(
        10,
        generator=generator_two
    )

    assert torch.equal(
        values_one,
        values_two
    )

    print("[PASS] DataLoader generator reproducibility")

    # --------------------------------------------------------
    # TEST 7: RNG state restoration
    # --------------------------------------------------------

    seed_everything(seed)

    state = capture_rng_state()

    expected_python = random.random()

    expected_numpy = np.random.rand()

    expected_torch = torch.rand(1)

    restore_rng_state(state)

    actual_python = random.random()

    actual_numpy = np.random.rand()

    actual_torch = torch.rand(1)

    assert expected_python == actual_python

    assert expected_numpy == actual_numpy

    assert torch.equal(
        expected_torch,
        actual_torch
    )

    print("[PASS] RNG state capture and restoration")

    # --------------------------------------------------------
    # TEST 8: Environment information
    # --------------------------------------------------------

    info = get_reproducibility_info(seed)

    print("\nENVIRONMENT")

    print(f"PyTorch: {info['pytorch_version']}")

    print(f"CUDA: {info['cuda_version']}")

    print(f"GPU: {info['gpu_names']}")

    print(
        f"cuDNN Deterministic: "
        f"{settings['cudnn_deterministic']}"
    )

    print(
        f"cuDNN Benchmark: "
        f"{settings['cudnn_benchmark']}"
    )

    print("\nREPRODUCIBILITY TEST: PASSED")

    print("=" * 70)


if __name__ == "__main__":

    main()
