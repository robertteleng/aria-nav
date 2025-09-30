"""Tests for MPS utility helpers."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from core.vision import mps_utils as utils


class DummyTorch:
    def __init__(self):
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
        self.mps = SimpleNamespace(empty_cache=self._empty)
        self.device_calls = []

    def device(self, name):
        self.device_calls.append(name)
        return SimpleNamespace(type=name)

    def _empty(self):
        self.mps_cleared = True


@pytest.fixture()
def torch_stub(monkeypatch: pytest.MonkeyPatch):
    stub = DummyTorch()
    monkeypatch.setattr(utils, "torch", stub)
    return stub


def test_configure_mps_environment_sets_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    utils.configure_mps_environment()
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.0"


def test_get_preferred_device_uses_cpu_when_unavailable(torch_stub):
    device = utils.get_preferred_device("mps")
    assert device.type == "cpu"


def test_empty_mps_cache_invokes_backend(torch_stub, monkeypatch: pytest.MonkeyPatch):
    torch_stub.backends.mps.is_available = lambda: True
    utils.empty_mps_cache()
    assert getattr(torch_stub, "mps_cleared", False)


def test_autocast_to_cpu_on_mps(torch_stub):
    tensor_mps = SimpleNamespace(device=SimpleNamespace(type="mps"), detach=lambda: tensor_mps)

    with utils.autocast_to_cpu_on_mps(tensor_mps) as tensor:
        assert tensor is tensor_mps

    tensor_cpu = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    with utils.autocast_to_cpu_on_mps(tensor_cpu) as tensor:
        assert tensor is tensor_cpu
