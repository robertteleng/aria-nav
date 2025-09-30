"""Unit tests for the SDK-focused Observer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core import observer as observer_module
from core.observer import Observer


class CameraIdStub:
    Rgb = "rgb"
    Slam1 = "slam1"
    Slam2 = "slam2"


@pytest.fixture()
def observer_env(monkeypatch: pytest.MonkeyPatch):
    def fake_rotate(image, flag):
        return image + 1

    def fake_cvt_color(image, code):
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        return image + 2

    monkeypatch.setattr(observer_module.cv2, "rotate", fake_rotate)
    monkeypatch.setattr(observer_module.cv2, "cvtColor", fake_cvt_color)
    monkeypatch.setattr(observer_module.aria, "CameraId", CameraIdStub)
    monkeypatch.setattr(observer_module.Config, "RGB_CAMERA_COLOR_SPACE", "RGB", raising=False)

    return Observer()


def test_on_image_received_updates_rgb_frame(observer_env):
    obs = observer_env
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    record = SimpleNamespace(camera_id=CameraIdStub.Rgb)

    obs.on_image_received(frame, record)

    assert obs.current_frames["rgb"].tolist() == (frame + 3).tolist()
    assert obs.frame_counts["rgb"] == 1


def test_on_image_received_converts_grayscale(observer_env):
    obs = observer_env
    grayscale = np.zeros((2, 2), dtype=np.uint8)
    record = SimpleNamespace(camera_id=CameraIdStub.Slam1)

    obs.on_image_received(grayscale, record)

    stored = obs.current_frames["slam1"]
    assert stored.shape == (2, 2, 3)
    assert obs.frame_counts["slam1"] == 1


def test_on_imu_received_appends_samples(observer_env):
    obs = observer_env
    sample = SimpleNamespace(accel_msec2=(1.0, 2.0, 2.0), capture_timestamp_ns=123)

    obs.on_imu_received([sample], imu_idx=0)

    assert obs.imu_data["imu0"][0]["timestamp"] == 123
    assert obs.motion_magnitude_history
