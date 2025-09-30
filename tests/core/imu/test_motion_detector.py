"""Tests for SimpleMotionDetector variance thresholds."""

from __future__ import annotations

from core.imu.motion_detector import SimpleMotionDetector


def test_motion_detector_requires_history():
    detector = SimpleMotionDetector()
    for _ in range(5):
        state = detector.update(0.1, timestamp_ns=0)
    assert state == "stationary"


def test_motion_detector_stationary_and_walking():
    detector = SimpleMotionDetector()
    for value in [0.05] * 12:
        detector.update(value, 0)
    assert detector.last_motion_state == "stationary"

    walking_sequence = [0.0, 2.0] * 10
    for value in walking_sequence:
        state = detector.update(value, 0)
    assert state == "walking"


def test_motion_detector_hysteresis():
    detector = SimpleMotionDetector()
    for value in [0.0, 2.0] * 10:
        detector.update(value, 0)
    assert detector.last_motion_state == "walking"

    mid_variance = [0.7, 0.8] * 10
    for value in mid_variance:
        state = detector.update(value, 0)
    assert state == "walking"
