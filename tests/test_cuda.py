"""
tests/test_cuda.py
───────────────────
Smoke tests for CUDA availability.

These tests are intentionally minimal — they verify the environment is wired
correctly before any model code is written.
"""

import importlib
import sys

import pytest


# ── torch import ──────────────────────────────────────────────────────────────


def test_torch_importable():
    """PyTorch must be importable."""
    torch = importlib.import_module("torch")
    assert torch is not None


def test_torch_version_major():
    """Require torch >= 2.x everywhere; >= 2.5 on Linux (Ubuntu target)."""
    import platform
    import torch

    major, minor, *_ = torch.__version__.split(".")
    assert int(major) >= 2, f"Expected torch >= 2.x, got {torch.__version__}"
    if platform.system() == "Linux" and int(major) == 2:
        # On the Ubuntu target, enforce the cu121 wheel (2.5.1)
        assert int(minor) >= 5, (
            f"Expected torch >= 2.5 on Linux/CUDA target, got {torch.__version__}. "
            "Run `make setup` to install the cu121 wheel."
        )


@pytest.mark.skipif(
    __import__("platform").system() != "Linux",
    reason="cu121 wheel check only enforced on Linux target (macOS uses CPU-only wheel for dev)",
)
def test_cuda_compiled():
    """PyTorch must have been compiled with CUDA (not CPU-only wheel) on Linux."""
    import torch

    assert torch.version.cuda is not None, (
        "torch.version.cuda is None — you have a CPU-only PyTorch wheel. "
        "Reinstall via: pip install torch==2.5.1+cu121 "
        "--index-url https://download.pytorch.org/whl/cu121"
    )


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="No CUDA GPU available — hardware/driver issue, not a code bug",
)
def test_cuda_available():
    """torch.cuda.is_available() must be True on target hardware."""
    import torch

    assert torch.cuda.is_available()


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="No CUDA GPU available",
)
def test_cuda_tensor_ops():
    """Basic GPU tensor allocation and arithmetic must succeed."""
    import torch

    a = torch.zeros(4, 4, device="cuda")
    b = torch.ones(4, 4, device="cuda")
    c = a + b
    assert c.sum().item() == 16.0


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="No CUDA GPU available",
)
def test_gpu_memory_reported():
    """At least one GPU must report > 0 bytes of total memory."""
    import torch

    props = torch.cuda.get_device_properties(0)
    assert props.total_memory > 0, "GPU reports 0 bytes total memory — something is wrong"
