"""Tests for NavigationDecisionEngine decision logic."""

from __future__ import annotations

import time
from typing import Dict

import pytest

from core.navigation.navigation_decision_engine import (
    DecisionCandidate,
    NavigationDecisionEngine,
)


@pytest.fixture()
def engine() -> NavigationDecisionEngine:
    return NavigationDecisionEngine()


def make_detection(name: str, bbox: Dict[str, float], confidence: float = 0.9) -> Dict[str, object]:
    return {
        "name": name,
        "bbox": bbox,
        "confidence": confidence,
    }


def test_analyze_filters_unknown_objects(engine: NavigationDecisionEngine) -> None:
    detections = [
        make_detection("person", {"x": 0, "y": 0, "w": 100, "h": 210}),
        make_detection("dog", {"x": 0, "y": 0, "w": 80, "h": 150}),
    ]

    result = engine.analyze(detections)

    assert len(result) == 1
    assert result[0]["class"] == "person"


def test_analyze_assigns_priority_and_zone(engine: NavigationDecisionEngine) -> None:
    detections = [make_detection("car", {"x": 250, "y": 0, "w": 150, "h": 160})]

    result = engine.analyze(detections)

    assert result[0]["zone"] == "center"
    assert result[0]["distance"] == "cerca"
    assert result[0]["priority"] > result[0]["original_priority"]


def test_analyze_orders_by_priority(engine: NavigationDecisionEngine) -> None:
    detections = [
        make_detection("traffic light", {"x": 320, "y": 0, "w": 50, "h": 60}),
        make_detection("car", {"x": 200, "y": 0, "w": 160, "h": 170}),
    ]

    ordered = engine.analyze(detections)

    assert ordered[0]["class"] == "car"
    assert ordered[1]["class"] == "traffic light"


def test_evaluate_returns_candidate_with_metadata(engine: NavigationDecisionEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    detections = [make_detection("person", {"x": 50, "y": 0, "w": 120, "h": 220})]
    items = engine.analyze(detections)

    # Congelar tiempo para controlar cooldown.
    now = time.time()
    monkeypatch.setattr(engine, "last_announcement_time", now - 10)

    candidate = engine.evaluate(items, motion_state="walking")

    assert isinstance(candidate, DecisionCandidate)
    assert candidate.metadata["class"] == "person"
    assert candidate.metadata["cooldown"] == pytest.approx(1.5)
    assert candidate.priority.name in {"CRITICAL", "HIGH"}


def test_evaluate_respects_cooldown(engine: NavigationDecisionEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    detections = [make_detection("person", {"x": 50, "y": 0, "w": 120, "h": 220})]
    items = engine.analyze(detections)

    now = time.time()
    monkeypatch.setattr(engine, "last_announcement_time", now - 0.5)

    candidate = engine.evaluate(items, motion_state="stationary")

    assert candidate is None


def test_evaluate_rejects_low_priority_events(engine: NavigationDecisionEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    detections = [make_detection("chair", {"x": 10, "y": 0, "w": 60, "h": 70})]
    items = engine.analyze(detections)

    assert len(items) == 1

    now = time.time()
    monkeypatch.setattr(engine, "last_announcement_time", now - 10)

    candidate = engine.evaluate(items, motion_state="stationary")

    assert candidate is None


def test_evaluate_updates_last_timestamp(engine: NavigationDecisionEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    detections = [make_detection("stop sign", {"x": 300, "y": 0, "w": 120, "h": 140})]
    items = engine.analyze(detections)

    start = time.time() - 20
    monkeypatch.setattr(engine, "last_announcement_time", start)

    candidate = engine.evaluate(items, motion_state="stationary")

    assert candidate is not None
    assert engine.last_announcement_time > start
