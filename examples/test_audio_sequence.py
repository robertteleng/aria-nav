
import sys
import os
import time

# Add src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from core.audio.audio_system import AudioSystem
from core.audio.navigation_audio_router import NavigationAudioRouter, NavigationEvent, EventPriority, SLAM1_SOURCE, SLAM2_SOURCE, RGB_SOURCE

def run_audio_demonstration():
    """
    This script demonstrates the functionality of the AudioSystem and NavigationAudioRouter.
    It simulates a sequence of navigation events to produce an "audiotira" or audio strip.
    """
    print("--- Starting Audio System Demonstration ---")

    # 1. Initialize the core components
    try:
        audio_system = AudioSystem()
        router = NavigationAudioRouter(audio_system)
    except Exception as e:
        print(f"Error initializing audio components: {e}")
        print("Please ensure all dependencies are installed (pyttsx3, sounddevice, numpy).")
        return

    # 2. Start the router's processing thread
    router.start()
    print("[INFO] NavigationAudioRouter started.")
    time.sleep(1)

    # 3. Demonstrate spatial beeps
    print("\n--- Testing Spatial Beeps ---")
    print("Playing normal beep on the left...")
    audio_system.play_spatial_beep("left", is_critical=False)
    time.sleep(1)

    print("Playing critical beep on the right...")
    audio_system.play_spatial_beep("right", is_critical=True)
    time.sleep(1)

    print("Playing normal beep in the center...")
    audio_system.play_spatial_beep("center", is_critical=False)
    time.sleep(2)

    # 4. Simulate a sequence of navigation events
    print("\n--- Simulating Navigation Event Sequence ---")

    # Event 1: A simple, low-priority message
    print("\n[SEQ 1] Enqueuing a low-priority message from RGB camera.")
    router.enqueue_from_rgb(
        message="Path is clear ahead.",
        priority=EventPriority.LOW
    )
    time.sleep(2) # Wait for it to be spoken

    # Event 2: A medium-priority warning
    print("\n[SEQ 2] Enqueuing a medium-priority obstacle warning from SLAM1.")
    router.enqueue(
        NavigationEvent(
            timestamp=time.time(),
            source=SLAM1_SOURCE,
            priority=EventPriority.MEDIUM,
            message="Obstacle detected to your left."
        )
    )
    time.sleep(2) # Wait for it to be spoken

    # Event 3: A high-priority event that should be spoken immediately
    print("\n[SEQ 3] Enqueuing a high-priority direction change from RGB.")
    router.enqueue_from_rgb(
        message="Turn right at the next corner.",
        priority=EventPriority.HIGH
    )
    time.sleep(2)

    # Event 4: A critical, urgent warning that should interrupt if necessary
    print("\n[SEQ 4] Enqueuing a CRITICAL warning from SLAM2.")
    router.enqueue(
        NavigationEvent(
            timestamp=time.time(),
            source=SLAM2_SOURCE,
            priority=EventPriority.CRITICAL,
            message="Stop! Obstacle directly in front."
        )
    )
    time.sleep(2)

    # Event 5: A repeated message that should be skipped by the cooldown logic
    print("\n[SEQ 5] Enqueuing the same critical warning again. Should be skipped.")
    router.enqueue(
        NavigationEvent(
            timestamp=time.time(),
            source=SLAM2_SOURCE,
            priority=EventPriority.CRITICAL,
            message="Stop! Obstacle directly in front."
        )
    )
    time.sleep(2) # Give it time to be (correctly) skipped

    # Event 6: Another message from the same source, might be affected by source-specific cooldown
    print("\n[SEQ 6] Enqueuing another message from SLAM2. May be skipped by source cooldown.")
    router.enqueue(
        NavigationEvent(
            timestamp=time.time(),
            source=SLAM2_SOURCE,
            priority=EventPriority.MEDIUM,
            message="Checking surroundings again."
        )
    )
    time.sleep(3) # Wait longer for source cooldown

    print("\n--- Demonstration Complete ---")
    print("Waiting a few seconds for any remaining queued messages...")
    time.sleep(5)

    # 5. Clean up
    print("[INFO] Stopping NavigationAudioRouter.")
    router.stop()
    audio_system.close()
    print("--- End of Demonstration ---")

if __name__ == "__main__":
    run_audio_demonstration()
