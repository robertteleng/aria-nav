"""Tests for the Builder convenience helpers."""

from __future__ import annotations

import types

import pytest

from core.navigation import builder as builder_module
from utils.config import Config


class StubCoordinator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.peripheral_calls = []

    def attach_peripheral_system(self, workers, audio_router):
        self.peripheral_calls.append((workers, audio_router))


class StubSlamDetectionWorker:
    def __init__(self, source, target_fps):
        self.source = source
        self.target_fps = target_fps
        self.started = False

    def start(self):
        self.started = True


class StubAudioRouter:
    def __init__(self, audio_system, telemetry=None):
        self.audio_system = audio_system
        self.telemetry = telemetry
        self.started = False

    def start(self):
        self.started = True

    def enqueue_from_rgb(self, *args, **kwargs):
        pass


@pytest.fixture()
def stubbed_builder(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(builder_module, "YoloProcessor", lambda: "yolo", raising=False)
    monkeypatch.setattr(builder_module, "AudioSystem", lambda: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(builder_module, "FrameRenderer", lambda: "renderer", raising=False)
    monkeypatch.setattr(builder_module, "ImageEnhancer", lambda: "enhancer", raising=False)
    monkeypatch.setattr(builder_module, "NavigationPipeline", lambda **kwargs: "pipeline", raising=False)
    monkeypatch.setattr(builder_module, "NavigationDecisionEngine", lambda: "decision", raising=False)
    monkeypatch.setattr(builder_module, "NavigationAudioRouter", StubAudioRouter, raising=False)
    monkeypatch.setattr(builder_module, "Coordinator", StubCoordinator, raising=False)
    monkeypatch.setattr(builder_module, "SlamDetectionWorker", StubSlamDetectionWorker, raising=False)

    # CameraSource stub with simple values
    CameraSourceStub = types.SimpleNamespace(SLAM1=types.SimpleNamespace(value="slam1"), SLAM2=types.SimpleNamespace(value="slam2"))
    monkeypatch.setattr(builder_module, "CameraSource", CameraSourceStub, raising=False)

    return builder_module.Builder()


def test_build_full_system_returns_coordinator(monkeypatch: pytest.MonkeyPatch, stubbed_builder):
    monkeypatch.setattr(Config, "PERIPHERAL_VISION_ENABLED", False, raising=False)

    coordinator = stubbed_builder.build_full_system(enable_dashboard=False)

    assert isinstance(coordinator, StubCoordinator)
    assert coordinator.kwargs["navigation_pipeline"] == "pipeline"
    assert coordinator.kwargs["decision_engine"] == "decision"
    audio_router = coordinator.kwargs["audio_router"]
    assert isinstance(audio_router, StubAudioRouter)
    assert audio_router.audio_system is coordinator.kwargs["audio_system"]


def test_build_full_system_attaches_slam(monkeypatch: pytest.MonkeyPatch, stubbed_builder):
    monkeypatch.setattr(Config, "PERIPHERAL_VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(Config, "SLAM_TARGET_FPS", 5, raising=False)

    coordinator = stubbed_builder.build_full_system(enable_dashboard=False)

    assert coordinator.peripheral_calls
    workers, audio_router = coordinator.peripheral_calls[0]
    assert all(isinstance(worker, StubSlamDetectionWorker) for worker in workers.values())
    assert audio_router is coordinator.kwargs["audio_router"]


def test_build_full_system_passes_telemetry(monkeypatch: pytest.MonkeyPatch, stubbed_builder):
    monkeypatch.setattr(Config, "PERIPHERAL_VISION_ENABLED", False, raising=False)
    telemetry = object()

    coordinator = stubbed_builder.build_full_system(
        enable_dashboard=False,
        telemetry=telemetry,
    )

    audio_router = coordinator.kwargs["audio_router"]
    assert audio_router.telemetry is telemetry
