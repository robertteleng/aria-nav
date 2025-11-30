#!/usr/bin/env python3
"""🧪 Integration test for all 3 NOA-inspired improvements."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio.audio_system import AudioSystem
from core.navigation.navigation_decision_engine import NavigationDecisionEngine
from core.navigation.rgb_audio_router import RgbAudioRouter


def test_dynamic_volume():
    """Test 1: Dynamic volume by distance."""
    print("\n" + "="*60)
    print("TEST 1: DYNAMIC VOLUME BY DISTANCE")
    print("="*60)

    audio = AudioSystem()
    time.sleep(0.5)

    test_cases = [
        ("very_close", "left", True, "Person muy cerca izquierda (LOUD)"),
        ("far", "left", True, "Person lejos izquierda (quiet)"),
        ("close", "center", False, "Chair cerca centro (medium-high)"),
        ("medium", "right", False, "Table medio-lejos derecha (medium-low)"),
    ]

    for distance, zone, is_critical, description in test_cases:
        print(f"\n[BEEP] {description}")
        audio.play_spatial_beep(zone, is_critical, distance)
        time.sleep(1.5)

    print("\n✅ Volume should vary: very_close=loud, far=quiet")
    audio.close()


def test_scan_mode():
    """Test 2: Scan mode for scene summary."""
    print("\n" + "="*60)
    print("TEST 2: SCAN MODE (Scene Summary)")
    print("="*60)

    audio = AudioSystem()
    time.sleep(0.5)

    # Simulate coordinator's current_detections
    mock_detections = [
        {"class": "person", "zone": "center", "priority": 10.0},
        {"class": "chair", "zone": "center", "priority": 5.0},
        {"class": "table", "zone": "left", "priority": 4.0},
        {"class": "car", "zone": "right", "priority": 8.0},
        {"class": "bottle", "zone": "right", "priority": 2.0},
    ]

    print("\n[SCAN] Triggering scene scan...")
    audio.scan_scene(mock_detections)
    print("Expected: 'Scanning. Ahead: person, chair. Left: table. Right: car, bottle.'")
    time.sleep(5)

    print("\n✅ TTS should announce grouped objects by zone")
    audio.close()


def test_object_tracking():
    """Test 3: Object tracking with per-instance cooldowns."""
    print("\n" + "="*60)
    print("TEST 3: OBJECT TRACKING (Per-Instance Cooldowns)")
    print("="*60)

    decision_engine = NavigationDecisionEngine()

    # Simulate 2 people at different positions
    print("\n[FRAME 1] Two people detected (different positions)")
    detections_frame1 = [
        {
            "class": "person",
            "bbox": (100, 200, 50, 100),  # Left person
            "zone": "left",
            "distance": "very_close",
            "priority": 10.0,
        },
        {
            "class": "person",
            "bbox": (400, 200, 50, 100),  # Right person
            "zone": "right",
            "distance": "very_close",
            "priority": 10.0,
        },
    ]

    candidate1 = decision_engine.evaluate(detections_frame1)
    if candidate1:
        print(f"  ✅ Announced: {candidate1.nav_object['class']} (track_id={candidate1.nav_object.get('track_id')})")
    else:
        print("  ❌ Nothing announced")

    # Same frame, second person should also trigger (different instance)
    time.sleep(0.1)
    # Remove first person, evaluate second
    detections_frame1b = [detections_frame1[1]]
    candidate1b = decision_engine.evaluate(detections_frame1b)
    if candidate1b:
        print(f"  ✅ Announced: {candidate1b.nav_object['class']} (track_id={candidate1b.nav_object.get('track_id')})")
    else:
        print("  ❌ Nothing announced (expected if cooldown active)")

    # Frame 2: Left person moves slightly (same ID)
    print("\n[FRAME 2] Left person moves slightly (0.5s later)")
    time.sleep(0.5)
    detections_frame2 = [
        {
            "class": "person",
            "bbox": (105, 205, 50, 100),  # Slightly moved (high IoU)
            "zone": "left",
            "distance": "very_close",
            "priority": 10.0,
        },
    ]

    candidate2 = decision_engine.evaluate(detections_frame2)
    if candidate2:
        print(f"  ❌ UNEXPECTED: Announced despite cooldown (track_id={candidate2.nav_object.get('track_id')})")
    else:
        print("  ✅ Blocked by tracker cooldown (same instance)")

    # Frame 3: After cooldown expires
    print("\n[FRAME 3] After cooldown expires (2s later)")
    time.sleep(2.0)
    candidate3 = decision_engine.evaluate(detections_frame2)
    if candidate3:
        print(f"  ✅ Announced again: {candidate3.nav_object['class']} (track_id={candidate3.nav_object.get('track_id')})")
    else:
        print("  ❌ Should announce after cooldown")

    # Stats
    stats = decision_engine.object_tracker.get_stats()
    print(f"\n[TRACKER STATS] {stats}")
    print("✅ Tracker should show 2 IDs assigned (person_0, person_1)")


def main():
    """Run all tests."""
    print("\n" + "🧪 NOA-INSPIRED IMPROVEMENTS TEST SUITE")
    print("=" * 60)

    try:
        test_dynamic_volume()
        time.sleep(2)

        test_scan_mode()
        time.sleep(2)

        test_object_tracking()

        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETE")
        print("="*60)
        print("\nSUMMARY:")
        print("  1. Dynamic Volume: Listen for volume differences by distance")
        print("  2. Scan Mode: Should announce 'Scanning. Ahead/Left/Right: ...'")
        print("  3. Object Tracking: Check logs for track_id assignments")
        print("\nCheck console output and audio for verification.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
