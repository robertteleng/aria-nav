# 🔊 Audio Configuration Guide

> **Complete audio system setup for macOS and Linux**  
> Last updated: November 20, 2025  
> Status: ✅ Active

## 📑 Table of Contents

1. [Overview](#overview)
2. [macOS Configuration](#macos-configuration)
3. [Linux Configuration](#linux-configuration)
4. [Audio Router Settings](#audio-router-settings)
5. [Beep System](#beep-system)
6. [Testing Audio](#testing-audio)
7. [Troubleshooting](#troubleshooting)

---

## Overview

The Aria Navigation System uses two types of audio feedback:

1. **Voice Commands (TTS)** - Spoken navigation instructions
2. **Beeps** - Distance-based warning sounds

### Audio Components

```
┌─────────────────────┐
│ Navigation Engine   │
│ (Detections)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Navigation Audio    │
│ Router              │
│ (Priority Queue)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Audio System        │
│ (TTS + Beeps)       │
└─────────────────────┘
```

---

## macOS Configuration

### Built-in TTS (say command)

macOS uses the native `say` command for text-to-speech.

#### 1. Verify Installation

```bash
# Check if 'say' is available
which say
# Output: /usr/bin/say

# Test basic TTS
say "Hello world"
```

#### 2. Configure Voice

```bash
# List available voices
say -v ?

# Test different voices
say -v Alex "Testing Alex voice"
say -v Samantha "Testing Samantha voice"
say -v Daniel "Testing Daniel voice"
```

#### 3. Adjust Speech Rate

```bash
# Faster speech (default is ~175 words/min)
say -r 200 "This is faster"

# Slower speech
say -r 150 "This is slower"
```

#### 4. Project Configuration

Edit `src/core/audio/audio_system.py`:

```python
class AudioSystem:
    def __init__(self):
        # macOS configuration
        self.voice = "Alex"          # Change voice
        self.rate = 175              # Words per minute
        self.use_say = True
        
    def speak(self, text: str):
        """Use macOS 'say' command"""
        cmd = ['say', '-v', self.voice, '-r', str(self.rate), text]
        subprocess.run(cmd, check=False)
```

#### 5. Permissions

Grant Terminal microphone/accessibility access:

1. **System Preferences** → **Security & Privacy**
2. **Privacy** tab → **Microphone**
3. Enable Terminal/iTerm
4. **Accessibility** → Enable Terminal/iTerm

---

## Linux Configuration

Linux offers multiple TTS options. We recommend **espeak** for speed.

### Option 1: espeak (Recommended)

Fast, lightweight, but robotic voice.

#### Installation

```bash
# Ubuntu/Debian
sudo apt install espeak espeak-data libespeak-dev -y

# Verify
espeak --version
```

#### Basic Usage

```bash
# Test TTS
espeak "Hello world"

# Adjust speed (-s flag, default 175)
espeak -s 200 "Faster speech"
espeak -s 150 "Slower speech"

# Adjust pitch (-p flag, 0-99, default 50)
espeak -p 60 "Higher pitch"

# Adjust volume (-a flag, 0-200, default 100)
espeak -a 150 "Louder"
```

#### Project Configuration

Edit `src/core/audio/audio_system.py`:

```python
import subprocess

class AudioSystem:
    def __init__(self):
        # Linux espeak configuration
        self.tts_engine = "espeak"
        self.rate = 175       # Words per minute
        self.pitch = 50       # 0-99
        self.amplitude = 100  # 0-200
        
    def speak(self, text: str):
        """Use espeak for TTS"""
        cmd = [
            'espeak',
            '-s', str(self.rate),
            '-p', str(self.pitch),
            '-a', str(self.amplitude),
            text
        ]
        subprocess.run(cmd, check=False)
```

### Option 2: pyttsx3 (More Natural)

More natural voice, but slower performance.

#### Installation

```bash
# Install pyttsx3 and espeak backend
pip install pyttsx3
sudo apt install espeak espeak-data libespeak-dev
```

#### Project Configuration

```python
import pyttsx3

class AudioSystem:
    def __init__(self):
        # Initialize pyttsx3 engine
        self.engine = pyttsx3.init()
        
        # Configure voice properties
        self.engine.setProperty('rate', 175)    # Speed
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Select voice (optional)
        voices = self.engine.getProperty('voices')
        # self.engine.setProperty('voice', voices[0].id)  # Male voice
        # self.engine.setProperty('voice', voices[1].id)  # Female voice
        
    def speak(self, text: str):
        """Use pyttsx3 for TTS"""
        self.engine.say(text)
        self.engine.runAndWait()
```

### Option 3: Festival (Alternative)

Another TTS option with better quality than espeak.

```bash
# Install
sudo apt install festival festvox-us-slt-hts

# Test
echo "Hello world" | festival --tts

# In Python
subprocess.run(['festival', '--tts'], input=text.encode())
```

---

## Audio Router Settings

Configure audio behavior in `src/utils/config.py`:

### Basic Settings

```python
# Audio System
AUDIO_ENABLED = True
BEEP_ENABLED = True
AUDIO_COOLDOWN_SECONDS = 2.0  # Minimum time between voice commands

# Priority System
AUDIO_PRIORITY_CRITICAL = 3   # Immediate obstacles
AUDIO_PRIORITY_HIGH = 2       # Important detections
AUDIO_PRIORITY_NORMAL = 1     # Regular updates

# Queue Management
MAX_AUDIO_QUEUE_SIZE = 10     # Maximum queued commands
AUDIO_QUEUE_TIMEOUT = 5.0     # Drop old commands after N seconds
```

### Per-Source Cooldowns

```python
# Different cooldown for each camera
RGB_AUDIO_COOLDOWN = 2.0      # Center camera
SLAM1_AUDIO_COOLDOWN = 3.0    # Left camera
SLAM2_AUDIO_COOLDOWN = 3.0    # Right camera
```

### Voice vs Beeps

```python
# When to use voice vs beeps
USE_BEEPS_FOR_CLOSE_OBJECTS = True
BEEP_DISTANCE_THRESHOLD = 1.0  # meters

# When object closer than threshold, use beeps instead of voice
# When farther, use voice commands
```

---

## Beep System

### Configuration

```python
# In config.py
BEEP_ENABLED = True

# Critical beeps (< 0.5m)
CRITICAL_BEEP_FREQUENCY = 880  # Hz (high pitch)
CRITICAL_BEEP_DURATION = 0.15  # seconds
CRITICAL_BEEP_INTERVAL = 0.3   # seconds between beeps

# Normal beeps (0.5m - 1.0m)
NORMAL_BEEP_FREQUENCY = 440    # Hz (lower pitch)
NORMAL_BEEP_DURATION = 0.2     # seconds
NORMAL_BEEP_INTERVAL = 0.6     # seconds between beeps
```

### Beep Behavior

Distance-based beep patterns:

| Distance | Beep Type | Frequency | Interval | Pattern |
|----------|-----------|-----------|----------|---------|
| < 0.5m | Critical | 880 Hz | 0.3s | Rapid |
| 0.5-1.0m | Normal | 440 Hz | 0.6s | Moderate |
| > 1.0m | None | - | - | Voice only |

### Custom Beep Implementation

Edit `src/core/audio/audio_system.py`:

```python
import numpy as np
import sounddevice as sd

class AudioSystem:
    def generate_beep(self, frequency: int, duration: float):
        """Generate beep tone"""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration))
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Add fade in/out to avoid clicks
        fade_samples = int(0.01 * sample_rate)
        wave[:fade_samples] *= np.linspace(0, 1, fade_samples)
        wave[-fade_samples:] *= np.linspace(1, 0, fade_samples)
        
        return wave
    
    def play_beep(self, frequency: int, duration: float):
        """Play beep through speakers"""
        wave = self.generate_beep(frequency, duration)
        sd.play(wave, samplerate=44100)
        sd.wait()
```

---

## Testing Audio

### 1. Test TTS Directly

```bash
# macOS
say "Testing text to speech"

# Linux (espeak)
espeak "Testing text to speech"

# Linux (pyttsx3)
python -c "import pyttsx3; e = pyttsx3.init(); e.say('Testing'); e.runAndWait()"
```

### 2. Test in Debug Mode

```bash
# Run system in debug mode
python src/main.py debug

# Press 't' to trigger test audio command
# Should hear: "Testing audio system"
```

### 3. Test Audio Router

```python
# In Python console
from src.core.audio.navigation_audio_router import NavigationAudioRouter
from src.core.audio.audio_system import AudioSystem

router = NavigationAudioRouter(AudioSystem())
router.enqueue_audio("rgb", "Test message", priority=2)

# Check metrics
print(router.get_metrics())
```

### 4. Test Beeps

```python
from src.core.audio.audio_system import AudioSystem

audio = AudioSystem()

# Test critical beep
audio.play_beep(frequency=880, duration=0.15)

# Test normal beep
audio.play_beep(frequency=440, duration=0.2)
```

---

## Troubleshooting

### macOS Issues

#### "say" command not found

```bash
# Reinstall Xcode tools
xcode-select --install

# Add to PATH
export PATH="/usr/bin:$PATH"
```

#### No sound output

1. Check **System Preferences** → **Sound** → **Output**
2. Verify volume is not muted
3. Test with: `afplay /System/Library/Sounds/Glass.aiff`

#### Permission denied

Grant accessibility permissions:
- **System Preferences** → **Security & Privacy** → **Privacy** → **Accessibility**
- Add Terminal/your IDE

### Linux Issues

#### espeak not installed

```bash
sudo apt update
sudo apt install espeak espeak-data libespeak-dev -y
```

#### No audio device found

```bash
# Check audio devices
aplay -l

# Install PulseAudio if needed
sudo apt install pulseaudio

# Restart audio service
pulseaudio --kill
pulseaudio --start
```

#### sounddevice errors (for beeps)

```bash
# Install portaudio
sudo apt install libportaudio2 portaudio19-dev

# Install Python package
pip install sounddevice
```

### General Issues

#### Audio commands not playing

Check configuration:
```python
# In config.py
AUDIO_ENABLED = True  # Must be True
AUDIO_COOLDOWN_SECONDS = 2.0  # Not too high
```

#### Voice too fast/slow

Adjust rate:
```python
# macOS
say -r 150 "slower"  # 150 words/min

# Linux espeak
espeak -s 150 "slower"  # 150 words/min
```

#### Audio cutting out

Reduce queue size or increase cooldown:
```python
AUDIO_COOLDOWN_SECONDS = 3.0  # Longer cooldown
MAX_AUDIO_QUEUE_SIZE = 5      # Smaller queue
```

---

## Advanced Configuration

### Spatial Audio (Future)

For stereo positioning (left/right/center):

```python
# Future feature
SPATIAL_AUDIO_ENABLED = True
AUDIO_PAN_STRENGTH = 0.7  # 0.0 = mono, 1.0 = full stereo
```

### Dynamic Volume

Adjust volume based on motion state:

```python
# Louder when moving
if motion_state == "walking":
    volume_multiplier = 1.2
else:
    volume_multiplier = 1.0
```

### Custom Voice Messages

Override default messages:

```python
# In navigation_decision_engine.py
AUDIO_MESSAGES = {
    "obstacle_center": "Obstacle ahead",
    "obstacle_left": "Obstacle on left",
    "obstacle_right": "Obstacle on right",
    "clear_path": "Path clear",
}
```

---

## See Also

- [Setup Guide](../setup/SETUP.md) - Installation instructions
- [Quick Reference](QUICK_REFERENCE.md) - Common commands
- [Architecture](../architecture/audio_spatial_summary.md) - Audio system design
- [Linux Audio Migration](../migration/LINUX_AUDIO.md) - macOS → Linux guide

---

**Audio configured?** Test with `python src/main.py debug` and press `t` 🔊
