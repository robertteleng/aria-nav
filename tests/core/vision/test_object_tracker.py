"""Unit tests for ObjectTracker consistency logic."""

from __future__ import annotations

import pytest

from core.vision import object_tracker as tracker_module
from core.vision.object_tracker import ObjectTracker


@pytest.fixture()
def tracker_env(monkeypatch: pytest.MonkeyPatch):
    clock = {"value": 0.0}

    def fake_time():
        return clock["value"]

    monkeypatch.setattr(tracker_module.time, "time", fake_time)
    tracker = ObjectTracker(history_size=5)
    return tracker, clock


def advance(clock, seconds: float) -> None:
    clock["value"] += seconds


def make_detection(name: str, zone: str) -> dict:
    return {"name": name, "zone": zone}


def test_tracker_requires_history(tracker_env) -> None:
    tracker, clock = tracker_env
    detect = [make_detection("person", "center")]

    for _ in range(2):
        advance(clock, 0.1)
        result = tracker.update(detect)
        assert result == []  # insufficient history yet


def test_tracker_returns_consistent_objects(tracker_env) -> None:
    tracker, clock = tracker_env

    sequence = [
        [make_detection("person", "left")],
        [make_detection("person", "left"), make_detection("dog", "left")],
        [make_detection("person", "left")],
        [make_detection("dog", "left")],
    ]

    outputs = []
    for detections in sequence:
        advance(clock, 0.2)
        outputs.append(tracker.update(detections))

    assert make_detection("person", "left") in outputs[-1]
    assert make_detection("dog", "left") not in outputs[-1]
