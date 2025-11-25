#!/usr/bin/env python3
"""Simple TTS test for Linux"""

import platform
import subprocess
import shutil

print(f"Sistema: {platform.system()}")
print(f"Probando TTS engines disponibles...\n")

# Test 1: espeak-ng
print("=== Test 1: espeak-ng directo ===")
if shutil.which('espeak-ng'):
    print("✓ espeak-ng encontrado")
    try:
        result = subprocess.run(
            ['espeak-ng', 'Hello from espeak'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"✗ Error: {e}")
else:
    print("✗ espeak-ng no encontrado")

print("\n=== Test 2: pyttsx3 ===")
try:
    import pyttsx3
    print("✓ pyttsx3 importado")
    
    engine = pyttsx3.init()
    print(f"Engine: {engine}")
    
    # Get properties
    rate = engine.getProperty('rate')
    volume = engine.getProperty('volume')
    voices = engine.getProperty('voices')
    
    print(f"Rate: {rate}")
    print(f"Volume: {volume}")
    print(f"Voices available: {len(voices) if voices else 0}")
    
    if voices:
        for i, voice in enumerate(voices[:3]):
            print(f"  Voice {i}: {voice.name}")
    
    # Try to speak
    print("\nIntentando hablar con pyttsx3...")
    engine.say("Hello from pyttsx3")
    engine.runAndWait()
    print("✓ pyttsx3.say() completado")
    
except ImportError:
    print("✗ pyttsx3 no instalado")
    print("Instalar con: pip install pyttsx3")
except Exception as e:
    print(f"✗ Error con pyttsx3: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test 3: espeak con subprocess.Popen ===")
if shutil.which('espeak-ng'):
    try:
        print("Ejecutando: espeak-ng 'Test with Popen'")
        process = subprocess.Popen(
            ['espeak-ng', 'Test with Popen'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(timeout=5)
        print(f"Return code: {process.returncode}")
        if stdout:
            print(f"Stdout: {stdout.decode()}")
        if stderr:
            print(f"Stderr: {stderr.decode()}")
    except Exception as e:
        print(f"✗ Error: {e}")

print("\n=== Test 4: spd-say (speech-dispatcher) ===")
if shutil.which('spd-say'):
    print("✓ spd-say encontrado")
    try:
        result = subprocess.run(
            ['spd-say', 'Hello from speech dispatcher'],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"Return code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
    except Exception as e:
        print(f"✗ Error: {e}")
else:
    print("✗ spd-say no encontrado")

print("\n=== Resumen ===")
print("Si no escuchas audio, verifica:")
print("1. Salida de audio en WSL: pulseaudio o pipewire")
print("2. DISPLAY variable para X11")
print("3. Audio device: aplay -l")
