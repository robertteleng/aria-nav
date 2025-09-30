"""Unit tests for ImageEnhancer behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from core.vision import image_enhancer as image_enhancer_module
from core.vision.image_enhancer import ImageEnhancer


class DummyClahe:
    def __init__(self) -> None:
        self.calls = 0

    def apply(self, channel):
        self.calls += 1
        return channel


@pytest.fixture()
def base_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(image_enhancer_module.Config, "LOW_LIGHT_ENHANCEMENT", True, raising=False)
    monkeypatch.setattr(image_enhancer_module.Config, "AUTO_ENHANCEMENT", True, raising=False)
    monkeypatch.setattr(image_enhancer_module.Config, "LOW_LIGHT_THRESHOLD", 50.0, raising=False)
    monkeypatch.setattr(image_enhancer_module.Config, "GAMMA_CORRECTION", 0.0, raising=False)
    return monkeypatch


def test_enhancement_disabled_returns_original(monkeypatch: pytest.MonkeyPatch):
    dummy = DummyClahe()
    monkeypatch.setattr(image_enhancer_module.cv2, "createCLAHE", lambda **_: dummy)
    monkeypatch.setattr(image_enhancer_module.Config, "LOW_LIGHT_ENHANCEMENT", False, raising=False)

    enhancer = ImageEnhancer()
    frame = np.full((4, 4, 3), 120, dtype=np.uint8)

    result = enhancer.enhance_frame(frame)

    assert np.array_equal(result, frame)
    assert dummy.calls == 0


def test_auto_detection_skips_well_lit_frame(base_config, monkeypatch: pytest.MonkeyPatch):
    dummy = DummyClahe()
    monkeypatch.setattr(image_enhancer_module.cv2, "createCLAHE", lambda **_: dummy)
    enhancer = ImageEnhancer()

    # Frame clearly above threshold (avg brightness 200 > 50)
    bright_frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    result = enhancer.enhance_frame(bright_frame)

    assert np.array_equal(result, bright_frame)
    assert dummy.calls == 0  # CLAHE not invoked when skipping


def test_enhance_frame_invokes_clahe_when_low_light(base_config, monkeypatch: pytest.MonkeyPatch):
    # Force enhancement without auto gating
    monkeypatch.setattr(image_enhancer_module.Config, "AUTO_ENHANCEMENT", False, raising=False)
    dummy = DummyClahe()
    monkeypatch.setattr(image_enhancer_module.cv2, "createCLAHE", lambda **_: dummy)

    # Ensure LUT is identity to simplify assertions
    monkeypatch.setattr(image_enhancer_module.cv2, "LUT", lambda img, table: img)

    enhancer = ImageEnhancer()
    dark_frame = np.zeros((4, 4, 3), dtype=np.uint8)

    result = enhancer.enhance_frame(dark_frame)

    assert result.shape == dark_frame.shape
    assert dummy.calls > 0


def test_detect_low_light_uses_threshold(base_config, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(image_enhancer_module.cv2, "createCLAHE", lambda **_: DummyClahe())
    enhancer = ImageEnhancer()

    low_frame = np.zeros((2, 2, 3), dtype=np.uint8)
    bright_frame = np.full((2, 2, 3), 255, dtype=np.uint8)

    assert enhancer._detect_low_light(low_frame) is True
    assert enhancer._detect_low_light(bright_frame) is False
