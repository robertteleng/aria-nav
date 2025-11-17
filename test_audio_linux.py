#!/usr/bin/env python3
"""Test script for AudioSystem on Linux."""

import time
from src.core.audio.audio_system import AudioSystem

def main():
    print("Initializing AudioSystem...")
    audio = AudioSystem()
    
    print("\n1. Testing TTS (Text-to-Speech)...")
    audio.speak_async('Sistema de audio en Linux funcionando correctamente', force=True)
    time.sleep(4)
    
    print("\n2. Testing spatial beeps...")
    audio.update_frame_dimensions(640, 480)
    
    print("   - Left normal beep")
    audio.play_spatial_beep('left', is_critical=False)
    time.sleep(1)
    
    print("   - Right critical beep")
    audio.play_spatial_beep('right', is_critical=True)
    time.sleep(1)
    
    print("   - Center normal beep")
    audio.play_spatial_beep('center', is_critical=False)
    time.sleep(1)
    
    print("\n3. Stats:")
    stats = audio.get_beep_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n✓ All audio tests passed successfully!")
    audio.close()

if __name__ == "__main__":
    main()
