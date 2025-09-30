"""Tests for RgbAudioRouter behaviour."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.audio.navigation_audio_router import EventPriority
from core.navigation.navigation_decision_engine import DecisionCandidate
from core.navigation.rgb_audio_router import RgbAudioRouter


class DummyAudioSystem:
    def __init__(self) -> None:
        self.repeat_cooldown = None
        self.announcement_cooldown = None
        self.queued_messages = []

    def set_repeat_cooldown(self, value: float) -> None:
        self.repeat_cooldown = value

    def set_announcement_cooldown(self, value: float) -> None:
        self.announcement_cooldown = value

    def queue_message(self, message: str) -> None:
        self.queued_messages.append(message)


@pytest.fixture()
def audio_system() -> DummyAudioSystem:
    return DummyAudioSystem()


@pytest.fixture()
def decision_candidate() -> DecisionCandidate:
    nav_object = {
        "class": "person",
        "zone": "left",
        "distance": "cerca",
        "priority": 10,
    }
    metadata = {
        "class": "person",
        "zone": "left",
        "distance": "cerca",
        "priority": 10,
        "cooldown": 2.0,
    }
    return DecisionCandidate(nav_object=nav_object, metadata=metadata, priority=EventPriority.CRITICAL)


def test_route_uses_navigation_router(audio_system: DummyAudioSystem, decision_candidate: DecisionCandidate) -> None:
    calls = SimpleNamespace(enqueue=False, cooldown=None)

    class FakeRouter:
        _running = True

        def set_source_cooldown(self, source: str, value: float) -> None:
            calls.cooldown = (source, value)

        def enqueue_from_rgb(self, message, priority, metadata):
            calls.enqueue = True
            assert "Warning" in message
            assert priority == EventPriority.CRITICAL
            assert metadata["cooldown"] == 2.0

    router = FakeRouter()
    rgb_router = RgbAudioRouter(audio_system, router)

    rgb_router.route(decision_candidate)

    assert calls.enqueue is True
    assert calls.cooldown == ("rgb", 2.0)
    assert audio_system.queued_messages == []


def test_route_fallbacks_to_audio_system(audio_system: DummyAudioSystem, decision_candidate: DecisionCandidate) -> None:
    class FailingRouter:
        _running = True

        def set_source_cooldown(self, *_args, **_kwargs):
            raise RuntimeError("boom")

        def enqueue_from_rgb(self, *_args, **_kwargs):
            raise AssertionError("should not be called")

    rgb_router = RgbAudioRouter(audio_system, FailingRouter())

    rgb_router.route(decision_candidate)

    assert audio_system.repeat_cooldown == pytest.approx(2.0)
    assert audio_system.announcement_cooldown == pytest.approx(1.0)
    assert audio_system.queued_messages[0].startswith("Warning, person")


def test_route_handles_missing_cooldown_metadata(audio_system: DummyAudioSystem) -> None:
    nav_object = {"class": "bus", "zone": "right", "distance": "medio", "priority": 12}
    candidate = DecisionCandidate(nav_object=nav_object, metadata={}, priority=EventPriority.HIGH)

    class FakeRouter:
        _running = True

        def __init__(self) -> None:
            self.cooldown = None
            self.messages = []

        def set_source_cooldown(self, source: str, value: float) -> None:
            self.cooldown = (source, value)

        def enqueue_from_rgb(self, message, priority, metadata):
            self.messages.append(message)

    router = FakeRouter()
    rgb_router = RgbAudioRouter(audio_system, router)

    rgb_router.route(candidate)

    assert router.cooldown == ("rgb", 0.0)
    assert "at medium distance" in router.messages[0].lower()


def test_route_without_router_uses_audio_system(audio_system: DummyAudioSystem) -> None:
    nav_object = {"class": "traffic light", "zone": "center", "distance": "medio", "priority": 9}
    metadata = {"cooldown": 0.5}
    candidate = DecisionCandidate(nav_object=nav_object, metadata=metadata, priority=EventPriority.MEDIUM)

    rgb_router = RgbAudioRouter(audio_system, None)

    rgb_router.route(candidate)

    assert audio_system.repeat_cooldown == pytest.approx(0.5)
    assert any("traffic light" in msg.lower() for msg in audio_system.queued_messages)
