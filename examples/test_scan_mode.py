#!/usr/bin/env python3
"""🆕 Test script for scan mode feature (NOA-inspired)."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio.audio_system import AudioSystem


def test_scan_mode():
    """Test the new scan_scene() feature with mock detections."""
    print("[TEST] Initializing AudioSystem...")
    audio = AudioSystem()

    # Wait for TTS to initialize
    time.sleep(0.5)

    # Test 1: Empty scene
    print("\n[TEST 1] Empty scene")
    audio.scan_scene([])
    time.sleep(2)

    # Test 2: Single object ahead
    print("\n[TEST 2] Single person ahead")
    mock_detections = [
        {"class": "person", "zone": "center", "priority": 10.0},
    ]
    audio.scan_scene(mock_detections)
    time.sleep(3)

    # Test 3: Multiple objects in different zones
    print("\n[TEST 3] Complex scene")
    mock_detections = [
        {"class": "person", "zone": "center", "priority": 10.0},
        {"class": "chair", "zone": "center", "priority": 5.0},
        {"class": "table", "zone": "left", "priority": 4.0},
        {"class": "bottle", "zone": "right", "priority": 2.0},
        {"class": "laptop", "zone": "right", "priority": 2.0},
    ]
    audio.scan_scene(mock_detections)
    time.sleep(4)

    # Test 4: Many objects (should limit to top 5)
    print("\n[TEST 4] Overcrowded scene (limit to 5)")
    mock_detections = [
        {"class": "person", "zone": "center", "priority": 10.0},
        {"class": "car", "zone": "left", "priority": 8.0},
        {"class": "chair", "zone": "center", "priority": 5.0},
        {"class": "table", "zone": "left", "priority": 4.0},
        {"class": "bottle", "zone": "right", "priority": 2.0},
        {"class": "laptop", "zone": "right", "priority": 2.0},
        {"class": "door", "zone": "center", "priority": 3.0},
        {"class": "couch", "zone": "left", "priority": 3.0},
    ]
    audio.scan_scene(mock_detections)
    time.sleep(5)

    print("\n[TEST] Complete! Check audio output.")
    audio.close()


if __name__ == "__main__":
    test_scan_mode()
