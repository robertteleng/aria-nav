#!/usr/bin/env python3
"""Test script to verify beeps and TTS work independently (bug fix verification)."""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.audio.audio_system import AudioSystem


def test_beep_tts_independence():
    """Verify that beeps no longer block TTS announcements."""
    print("[TEST] Initializing AudioSystem...")
    audio = AudioSystem()

    # Wait for TTS to initialize
    time.sleep(0.5)

    # Test 1: TTS announcement followed by beeps (should both play)
    print("\n[TEST 1] TTS + Beep immediately after")
    audio.speak_async("Person ahead", force=True)
    time.sleep(0.1)  # Small delay to let TTS start
    audio.play_spatial_beep("center", is_critical=False, distance="close")
    time.sleep(3)

    # Test 2: Beeps followed by TTS (should both play)
    print("\n[TEST 2] Beep + TTS immediately after")
    audio.play_spatial_beep("left", is_critical=False, distance="medium")
    time.sleep(0.1)
    audio.speak_async("Car on the left", force=True)
    time.sleep(3)

    # Test 3: Simultaneous beeps and TTS (the critical test)
    print("\n[TEST 3] Simultaneous beeps and TTS")
    # Start TTS
    audio.speak_async("Obstacle directly ahead", force=True)
    # Immediately play beeps (should NOT block TTS)
    audio.play_spatial_beep("center", is_critical=True, distance="very_close")
    time.sleep(3)

    # Test 4: Multiple beeps while TTS is speaking
    print("\n[TEST 4] Multiple beeps during TTS")
    audio.speak_async("Multiple objects detected around you", force=True)
    time.sleep(0.2)
    audio.play_spatial_beep("left", is_critical=False, distance="close")
    time.sleep(0.3)
    audio.play_spatial_beep("right", is_critical=False, distance="medium")
    time.sleep(0.3)
    audio.play_spatial_beep("center", is_critical=False, distance="far")
    time.sleep(3)

    # Test 5: Rapid fire (stress test)
    print("\n[TEST 5] Rapid fire TTS + beeps")
    for i in range(3):
        audio.speak_async(f"Alert {i+1}", force=True)
        audio.play_spatial_beep("center", is_critical=False, distance="close")
        time.sleep(0.5)
    time.sleep(3)

    print("\n[TEST] Complete!")
    print("\n[RESULTS] Check audio output:")
    print("  ✓ All TTS announcements should have been heard")
    print("  ✓ All beeps should have played")
    print("  ✓ No blocking should have occurred")

    # Print stats
    stats = audio.get_beep_stats()
    print(f"\n[STATS] Beep statistics:")
    print(f"  - Normal beeps: {stats['normal_beeps']}")
    print(f"  - Critical beeps: {stats['critical_beeps']}")

    audio.close()


if __name__ == "__main__":
    test_beep_tts_independence()
