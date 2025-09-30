"""Tests for YoloRuntimeConfig option handling."""

from __future__ import annotations

import pytest

from core.vision.yolo_processor import YoloRuntimeConfig


def test_defaults_reference_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("utils.config.Config.YOLO_MODEL", "model.pt")
    monkeypatch.setattr("utils.config.Config.YOLO_DEVICE", "cpu")
    monkeypatch.setattr("utils.config.Config.YOLO_CONFIDENCE", 0.42)
    monkeypatch.setattr("utils.config.Config", "YOLO_FRAME_SKIP", 2)

    config = YoloRuntimeConfig.from_defaults()

    assert config.model == "model.pt"
    assert config.frame_skip == 2
    assert config.confidence == 0.42


def test_for_profile_returns_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("utils.config.Config.YOLO_MODEL", "base.pt")
    config = YoloRuntimeConfig.for_profile("slam")

    assert config.profile_name == "slam"
    assert config.image_size == 256
    assert config.frame_skip == 3

    with pytest.raises(ValueError):
        YoloRuntimeConfig.for_profile("unknown")


def test_with_overrides_accepts_aliases():
    base = YoloRuntimeConfig.from_defaults()
    updated = base.with_overrides(imgsz=512, conf=0.7, max_det=5, iou=0.3, frame_skip=5, device="cpu")

    assert updated.image_size == 512
    assert updated.confidence == 0.7
    assert updated.max_detections == 5
    assert updated.iou_threshold == 0.3
    assert updated.frame_skip == 5
    assert updated.device == "cpu"

    with pytest.raises(ValueError):
        base.with_overrides(unsupported=1)
