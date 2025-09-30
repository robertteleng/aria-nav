"""Tests for the NavigationPipeline module."""

from __future__ import annotations

import numpy as np
import pytest

from core.navigation.navigation_pipeline import NavigationPipeline, PipelineResult
from utils import config as config_module


class FakeEnhancer:
    def __init__(self):
        self.calls = 0

    def enhance_frame(self, frame):
        self.calls += 1
        return frame + 1


class FakeDepthEstimator:
    def __init__(self):
        self.calls = 0
        self.model = object()

    def estimate_depth(self, frame):
        self.calls += 1
        return np.ones_like(frame) * 42


class FakeYolo:
    def __init__(self):
        self.calls = []

    def process_frame(self, frame, depth_map):
        self.calls.append((frame, depth_map))
        return [{"name": "person"}]


@pytest.fixture()
def numpy_frame() -> np.ndarray:
    return np.zeros((2, 2, 3), dtype=np.uint8)


def test_process_runs_all_stages(monkeypatch: pytest.MonkeyPatch, numpy_frame: np.ndarray) -> None:
    monkeypatch.setattr(config_module.Config, "DEPTH_ENABLED", True, raising=False)
    monkeypatch.setattr(config_module.Config, "DEPTH_FRAME_SKIP", 1, raising=False)

    enhancer = FakeEnhancer()
    depth_estimator = FakeDepthEstimator()
    yolo = FakeYolo()

    pipeline = NavigationPipeline(
        yolo_processor=yolo,
        image_enhancer=enhancer,
        depth_estimator=depth_estimator,
    )

    result = pipeline.process(numpy_frame, profile=True)

    assert isinstance(result, PipelineResult)
    assert enhancer.calls == 1
    assert depth_estimator.calls == 1
    assert len(yolo.calls) == 1
    assert np.array_equal(result.frame, numpy_frame + 1)
    assert np.array_equal(result.depth_map, np.ones_like(numpy_frame) * 42)
    assert set(result.timings.keys()) == {"enhance", "depth", "yolo"}


def test_process_respects_depth_frame_skip(monkeypatch: pytest.MonkeyPatch, numpy_frame: np.ndarray) -> None:
    monkeypatch.setattr(config_module.Config, "DEPTH_ENABLED", True, raising=False)
    monkeypatch.setattr(config_module.Config, "DEPTH_FRAME_SKIP", 2, raising=False)

    depth_estimator = FakeDepthEstimator()
    yolo = FakeYolo()
    pipeline = NavigationPipeline(yolo_processor=yolo, depth_estimator=depth_estimator)

    first = pipeline.process(numpy_frame)
    second = pipeline.process(numpy_frame)

    # Primer frame no genera depth map; segundo sí y se reutiliza para salida
    assert depth_estimator.calls == 1
    assert first.depth_map is None
    assert second.depth_map is not None
    assert pipeline.get_latest_depth_map() is second.depth_map


def test_process_handles_missing_depth(monkeypatch: pytest.MonkeyPatch, numpy_frame: np.ndarray) -> None:
    monkeypatch.setattr(config_module.Config, "DEPTH_ENABLED", False, raising=False)

    yolo = FakeYolo()
    pipeline = NavigationPipeline(yolo_processor=yolo, depth_estimator=None)

    result = pipeline.process(numpy_frame)

    assert result.depth_map is None
    assert pipeline.get_latest_depth_map() is None
    assert len(yolo.calls) == 1
