"""Utility helpers for configuring PyTorch GPU runtime (CUDA, MPS, CPU)"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import torch


def configure_mps_environment(force_disable_cpu_fallback: bool = False) -> None:
    """Tune environment variables for the PyTorch MPS backend."""
    if force_disable_cpu_fallback:
        os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)

    # Keep watermark low to avoid unexpected thrashing; torch obeys first-set value
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


def get_preferred_device(preferred: str = "mps") -> torch.device:
    """Return the preferred torch device when available, else CPU."""
    preferred_lower = preferred.lower()
    
    # Prioridad: CUDA > MPS > CPU
    if preferred_lower == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred_lower == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():  # Fallback a CUDA si está disponible
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Fallback a MPS si está disponible
        return torch.device("mps")
    
    return torch.device("cpu")


@contextmanager
def autocast_to_cpu_on_mps(tensor: torch.Tensor) -> Generator[torch.Tensor, None, None]:
    """Yield tensor on CPU when tensor lives on MPS; keep original device alive."""
    original_device = tensor.device
    if original_device.type == "mps":
        tensor_cpu = tensor.detach().to("cpu")
        try:
            yield tensor_cpu
        finally:
            tensor_cpu = tensor_cpu.to("cpu")  # ensure reference drop
    else:
        yield tensor


def empty_mps_cache() -> None:
    """Free cached MPS memory when backend is active."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

