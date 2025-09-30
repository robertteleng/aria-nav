"""Tests for NavigationAudioRouter prioritization and throttling."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from core.audio.navigation_audio_router import (
    EventPriority,
    NavigationAudioRouter,
    NavigationEvent,
)
from core.vision.slam_detection_worker import CameraSource


class DummyAudioSystem:
    def __init__(self) -> None:
        self.spoken = []

    def speak_async(self, message: str) -> bool:
        self.spoken.append(message)
        return True

    def set_repeat_cooldown(self, *_args, **_kwargs):
        pass

    def set_announcement_cooldown(self, *_args, **_kwargs):
        pass


@pytest.fixture()
def audio_system() -> DummyAudioSystem:
    return DummyAudioSystem()


@pytest.fixture()
def router(audio_system: DummyAudioSystem) -> NavigationAudioRouter:
    nav_router = NavigationAudioRouter(audio_system)
    return nav_router


def create_nav_event(priority: EventPriority, source: str = "rgb", **kwargs) -> NavigationEvent:
    payload = {
        "timestamp": time.time(),
        "source": source,
        "priority": priority,
        "message": "test message",
        "metadata": {
            "cooldown": kwargs.get("cooldown", 0.0),
        },
    }
    payload.update(kwargs)
    return NavigationEvent(**payload)


def test_should_announce_obeys_global_cooldown(router: NavigationAudioRouter):
    event = create_nav_event(EventPriority.HIGH)
    router._last_global_announcement = time.time()

    allowed, reason = router._should_announce(event)

    assert not allowed
    assert reason == "global_cooldown"


def test_should_announce_blocks_slam_spacing(router: NavigationAudioRouter):
    event = create_nav_event(EventPriority.MEDIUM, source=CameraSource.SLAM1.value)
    router._last_source_announcement[CameraSource.SLAM1.value] = time.time()

    allowed, reason = router._should_announce(event)

    assert not allowed
    assert reason == "source_cooldown"


def test_enqueue_from_slam_wraps_event(router: NavigationAudioRouter):
    slam_event = SimpleNamespace(
        source=CameraSource.SLAM1,
        timestamp=123.0,
        object_name="person",
        zone="left",
        distance="close",
        confidence=0.8,
        bbox=(0, 0, 10, 10),
        frame_index=7,
    )

    router.enqueue_from_slam(slam_event, "person close", EventPriority.HIGH)

    priority_value, _counter, nav_event = router.event_queue.get_nowait()

    assert priority_value == EventPriority.HIGH.value
    assert nav_event.source == CameraSource.SLAM1.value
    assert nav_event.raw_event is slam_event
    assert nav_event.message == "person close"


def test_run_processes_queue_and_speaks(router: NavigationAudioRouter, audio_system: DummyAudioSystem):
    router.start()
    event = create_nav_event(EventPriority.CRITICAL)
    router.enqueue(event)

    # Give the worker loop a brief moment to process.
    time.sleep(0.1)
    router.stop()

    assert audio_system.spoken
    assert audio_system.spoken[0] == event.message


def test_enqueue_drops_when_queue_full(router: NavigationAudioRouter):
    for _ in range(router.event_queue.maxsize):
        router.enqueue(create_nav_event(EventPriority.LOW))

    dropped_event = create_nav_event(EventPriority.LOW)
    router.enqueue(dropped_event)

    assert router.events_dropped == 1


def test_run_skips_when_tts_rejects():
    class RejectingAudio:
        def __init__(self) -> None:
            self.calls = []

        def speak_async(self, message: str) -> bool:
            self.calls.append(message)
            return False

        def set_repeat_cooldown(self, *_args, **_kwargs):
            pass

        def set_announcement_cooldown(self, *_args, **_kwargs):
            pass

    audio = RejectingAudio()
    router = NavigationAudioRouter(audio)
    router.start()
    router.enqueue(create_nav_event(EventPriority.HIGH))
    time.sleep(0.1)
    router.stop()

    assert audio.calls  # speech intent attempted
    assert router.events_skipped >= 1


def test_start_stop_lifecycle_creates_session(tmp_path, audio_system: DummyAudioSystem):
    router = NavigationAudioRouter(audio_system)
    router.log_path = tmp_path / "audio.jsonl"

    router.start()
    assert router._running is True
    assert router.metrics["events_enqueued"] == 0

    router.stop()
    assert router._running is False
    assert router.log_path.exists()


def test_stop_without_start_is_safe(audio_system: DummyAudioSystem):
    router = NavigationAudioRouter(audio_system)
    router.stop()  # Should not raise
    assert router._running is False
