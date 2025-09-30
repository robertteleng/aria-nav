"""Constructor and configuration tests for YoloProcessor."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.vision import yolo_processor as yp_module
from core.vision.yolo_processor import YoloProcessor, YoloRuntimeConfig


class StubYOLO:
    def __init__(self, model):
        self.model = model
        self.device_called = None

    def to(self, device):
        self.device_called = device

    def fuse(self):
        self.fused = True


@pytest.fixture()
def stubbed_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yp_module, "configure_mps_environment", lambda *_: None)
    monkeypatch.setattr(yp_module.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(yp_module, "YOLO", StubYOLO)
    monkeypatch.setattr(yp_module, "get_preferred_device", lambda _: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(yp_module, "empty_mps_cache", lambda: None)


def test_constructor_raises_with_conflicting_arguments(stubbed_env):
    config = YoloRuntimeConfig.from_defaults()
    with pytest.raises(ValueError):
        YoloProcessor(runtime_config=config, profile="rgb")


def test_constructor_uses_overrides(stubbed_env, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(yp_module.Config, "PROFILE_PIPELINE", False, raising=False)
    processor = YoloProcessor(profile="slam", confidence=0.5, image_size=320)

    assert processor.runtime_config.profile_name == "slam"
    assert processor.conf_threshold == 0.5
    assert processor.img_size == 320
    assert processor.device_str == "cpu"


def test_runtime_config_passthrough(stubbed_env):
    config = YoloRuntimeConfig.from_defaults().with_overrides(device="cpu")
    processor = YoloProcessor(runtime_config=config)

    assert processor.runtime_config == config
    assert processor.device_str == "cpu"
