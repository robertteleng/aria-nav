"""
Simple motion detection from IMU accelerometer data.

This module provides motion state detection based on accelerometer magnitude
variance analysis. It distinguishes between stationary and walking states using
hysteresis thresholds to prevent rapid state oscillation.

Features:
- Rolling window variance analysis (~0.5s history)
- Hysteresis thresholds for stable state transitions
- Type-safe motion state literals
- Minimal computational overhead

Motion States:
- 'stationary': Low variance (< 0.5 m/s²) - user is standing still
- 'walking': High variance (> 1.0 m/s²) - user is moving

Usage:
    detector = SimpleMotionDetector()
    state = detector.update(magnitude=9.85, timestamp_ns=timestamp)
    if state == "walking":
        # Apply walking-specific logic
"""

from collections import deque
from typing import Literal

MotionState = Literal["stationary", "walking"]


class SimpleMotionDetector:
    """Detect motion state from IMU accelerometer magnitude variance."""

    def __init__(self) -> None:
        self.magnitude_history = deque(maxlen=50)  # ~0.5 seconds of history
        self.last_motion_state: MotionState = "stationary"

        # Thresholds for motion detection
        self.stationary_threshold = 0.5  # m/s² variance for "stationary"
        self.walking_threshold = 1.0     # m/s² variance for "walking"

    def update(self, magnitude: float, timestamp_ns: int) -> MotionState:
        """Update motion state based on acceleration variance."""

        self.last_magnitude = magnitude
        self.magnitude_history.append(magnitude)

        if len(self.magnitude_history) < 10:  # Need minimum history
            return self.last_motion_state

        # Calculate variance in window
        magnitudes = list(self.magnitude_history)
        variance = self._calculate_variance(magnitudes)

        # Determine state based on variance
        if variance < self.stationary_threshold:
            new_state: MotionState = "stationary"
        elif variance > self.walking_threshold:
            new_state: MotionState = "walking"
        else:
            new_state = self.last_motion_state  # Maintain previous state (hysteresis)

        self.last_motion_state = new_state
        return new_state

    def _calculate_variance(self, values):
        """Calculate standard deviation of values list."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5  # Standard deviation
