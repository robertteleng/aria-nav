#!/usr/bin/env python3
"""Test TTS at different speeds"""

import pyttsx3
import time

print("Probando diferentes velocidades de TTS...\n")

try:
    engine = pyttsx3.init()
    
    test_message = "Chair detected on the right"
    
    speeds = [
        (200, "Muy rápido (200)"),
        (150, "Rápido (150)"),
        (130, "Normal (130)"),
        (110, "Lento (110)"),
        (90, "Muy lento (90)")
    ]
    
    for rate, description in speeds:
        print(f"\n{description}...")
        engine.setProperty('rate', rate)
        engine.say(f"Test speed {rate}. {test_message}")
        engine.runAndWait()
        time.sleep(0.5)
    
    print("\n\n✓ Test completado")
    print("¿Cuál velocidad se escuchó mejor? Recomiendo 130 para claridad.")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
