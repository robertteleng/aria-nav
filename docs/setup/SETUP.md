# 🚀 Setup Guide - Aria Navigation System

> **Complete installation and configuration guide**  
> Last updated: November 20, 2025  
> Status: ✅ Active

## 📑 Table of Contents

1. [System Requirements](#system-requirements)
2. [macOS Setup](#macos-setup)
3. [Linux Setup (Ubuntu)](#linux-setup-ubuntu)
4. [Aria SDK Installation](#aria-sdk-installation)
5. [Project Installation](#project-installation)
6. [Configuration](#configuration)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Minimum Requirements

#### Hardware
- **Processor**: 4+ cores (Intel/AMD x86_64 or Apple Silicon)
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: 
  - macOS: Apple Silicon (M1/M2/M3) or AMD GPU with Metal support
  - Linux: NVIDIA GPU with 4GB+ VRAM (RTX 2060 or better recommended)
- **Storage**: 5GB free space for dependencies and models
- **USB**: USB-C or USB-A port for Aria glasses connection

#### Meta Aria Glasses
- Meta Aria smart glasses with streaming enabled
- Profile 28 or compatible profile configured
- USB cable or WiFi network for streaming

### Software Requirements

#### macOS
- **OS**: macOS 13 (Ventura) or later
- **Xcode Tools**: `xcode-select --install`
- **Homebrew**: (optional but recommended)
- **Python**: 3.10 or 3.11

#### Linux (Ubuntu)
- **OS**: Ubuntu 22.04 LTS or later
- **Python**: 3.10 or 3.11
- **NVIDIA Drivers**: If using CUDA (535.x or later)
- **CUDA Toolkit**: 11.8 or 12.x (for GPU acceleration)

---

## macOS Setup

### 1. Install Xcode Command Line Tools

```bash
# Install Xcode tools
xcode-select --install

# Verify installation
xcode-select -p
# Should output: /Library/Developer/CommandLineTools
```

### 2. Install Python 3.10+

**Option A: Using Homebrew (Recommended)**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Verify installation
python3.11 --version
```

**Option B: Official Python Installer**
- Download from [python.org](https://www.python.org/downloads/)
- Install and add to PATH

### 3. Verify TTS (Text-to-Speech)

```bash
# macOS uses built-in 'say' command
which say
# Should output: /usr/bin/say

# Test TTS
say "Hello, testing audio system"
```

### 4. Install Git

```bash
# Check if installed
git --version

# Install if needed
brew install git
```

---

## Linux Setup (Ubuntu)

### 1. Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. Install Python 3.10+

```bash
# Install Python 3.10
sudo apt install python3.10 python3.10-venv python3.10-dev python3-pip -y

# Verify installation
python3.10 --version
```

### 3. Install Build Tools

```bash
# Essential build tools
sudo apt install build-essential cmake git pkg-config -y

# Additional libraries
sudo apt install libopencv-dev libavcodec-dev libavformat-dev libswscale-dev -y
```

### 4. Install Audio System (espeak)

```bash
# Install espeak for TTS
sudo apt install espeak espeak-data libespeak-dev -y

# Test TTS
espeak "Hello, testing audio system"
```

### 5. (Optional) NVIDIA GPU Setup

**Only if you have NVIDIA GPU and want CUDA acceleration:**

```bash
# Install NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot

# Verify driver installation
nvidia-smi
```

For full CUDA setup, see [NUC Migration Guide](../migration/NUC_MIGRATION.md).

---

## Aria SDK Installation

### 1. Request Access

1. Visit [Project Aria Developer Portal](https://projectaria.com/)
2. Request developer access
3. Accept terms and conditions
4. Download SDK for your platform

### 2. Install Aria SDK

**macOS:**
```bash
# Download from portal (example)
# cd ~/Downloads
# tar -xzf aria-sdk-mac.tar.gz

# Follow Meta's official installation guide
# Usually involves copying libraries and setting PYTHONPATH
```

**Linux:**
```bash
# Similar process for Linux
# Follow Meta's official documentation
```

### 3. Install Python Bindings

```bash
# Install projectaria-tools
pip install projectaria-tools

# Verify installation
python -c "import projectaria_tools; print(projectaria_tools.__version__)"
```

### 4. Configure Aria Profile

Connect your Aria glasses and enable Profile 28:

```bash
# Use Aria CLI tool (from Meta SDK)
aria-cli configure-profile --profile profile28

# Or use companion app
```

See [Meta Aria Profiles Guide](meta_aria_profiles.md) for details.

---

## Project Installation

### 1. Clone Repository

```bash
# Clone project
git clone https://github.com/robertteleng/aria-nav.git
cd aria-nav

# Checkout appropriate branch
git checkout main  # or feature/fase4-tensorrt
```

### 2. Create Virtual Environment

**macOS / Linux:**
```bash
# Create venv
python3.10 -m venv .venv

# Activate
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip wheel setuptools
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip wheel setuptools
```

### 3. Install PyTorch

**macOS (Apple Silicon):**
```bash
# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Verify MPS is available
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```

**Linux (CUDA):**
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

**CPU Only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install Project Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install ultralytics opencv-python numpy scipy \
    transformers timm pillow pytest flask \
    projectaria-tools
```

### 5. Download Model Weights

```bash
# YOLO model
mkdir -p checkpoints
cd checkpoints

# Download YOLOv12n (example using ultralytics)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Or manually place your trained model
# cp /path/to/yolo12n.pt checkpoints/

# Depth model (downloads automatically on first run)
# Or pre-download:
# python -c "from transformers import AutoModel; AutoModel.from_pretrained('LiheYoung/depth-anything-small-hf')"
```

---

## Configuration

### 1. Basic Configuration

Edit `src/utils/config.py`:

```python
# Device configuration
YOLO_DEVICE = "mps"  # macOS: "mps", Linux: "cuda", CPU: "cpu"
DEPTH_DEVICE = "mps"  # Same as above

# Enable/disable features
DEPTH_ENABLED = True
PERIPHERAL_VISION_ENABLED = True
BEEP_ENABLED = True

# Performance tuning (adjust based on your hardware)
YOLO_FRAME_SKIP = 3     # Lower = more detections, higher CPU
DEPTH_FRAME_SKIP = 12   # Lower = more depth maps, higher CPU

# Audio settings
AUDIO_COOLDOWN_SECONDS = 2.0
```

### 2. Aria Connection Settings

```python
# In config.py or create .env file
STREAMING_INTERFACE = "usb"  # or "wifi"
STREAMING_PROFILE_NAME = "profile28"

# For WiFi streaming
ARIA_DEVICE_IP = "192.168.1.100"  # Your Aria's IP
```

### 3. Dashboard Selection

Choose your preferred visualization:

- `opencv` - Simple OpenCV windows (lowest overhead)
- `rerun` - Interactive 3D visualization
- `web` - Web dashboard at http://localhost:5000

Set in config or choose at runtime.

---

## Verification

### 1. Run Tests

```bash
# Activate environment
source .venv/bin/activate

# Run test suite
pytest tests/ -v

# Run specific tests
pytest tests/test_navigation_pipeline.py -v
```

### 2. Test Without Hardware (Mock Mode)

```bash
# Test with synthetic frames
python src/main.py debug

# Should see:
# - Mock frames being processed
# - YOLO detections
# - Audio commands
# - Dashboard running
```

### 3. Test With Aria Hardware

```bash
# Connect Aria glasses via USB or WiFi
# Run main system
python src/main.py

# Choose dashboard when prompted
# Press 't' to test audio
# Press 'q' to quit
```

### 4. Verify GPU Acceleration

**macOS:**
```bash
python -c "
import torch
print('MPS available:', torch.backends.mps.is_available())
print('MPS built:', torch.backends.mps.is_built())
"
```

**Linux:**
```bash
python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0))
"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

---

## Troubleshooting

### Common Issues

#### 1. Aria Not Detected

**USB Connection:**
```bash
# macOS
system_profiler SPUSBDataType | grep -i aria

# Linux
lsusb | grep -i meta
```

**Solution:**
- Try different USB port
- Check cable connection
- Restart Aria glasses
- Update Aria firmware

#### 2. MPS/CUDA Not Available

**macOS MPS:**
```bash
# Check macOS version
sw_vers

# Must be macOS 13+ on Apple Silicon
# If not available, use CPU mode:
# Set YOLO_DEVICE = "cpu" in config.py
```

**Linux CUDA:**
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Audio Not Working

**macOS:**
```bash
# Test say command
say "test"

# Check System Preferences → Sound
# Grant Terminal microphone/accessibility permissions
```

**Linux:**
```bash
# Test espeak
espeak "test"

# If not working:
sudo apt install espeak espeak-data

# Check audio devices
aplay -l
```

#### 4. Low FPS / Performance Issues

**Quick Fixes:**
```python
# In config.py, increase frame skipping:
YOLO_FRAME_SKIP = 5      # Higher = less frequent processing
DEPTH_FRAME_SKIP = 15    # Higher = less frequent depth estimation

# Reduce model size:
YOLO_MODEL = "yolo11n"   # Nano model (fastest)
DEPTH_ANYTHING_VARIANT = "vits"  # Small variant

# Disable features:
PERIPHERAL_VISION_ENABLED = False
DEPTH_ENABLED = False  # Test without depth first
```

#### 5. Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python version
python --version  # Should be 3.10 or 3.11

# Verify virtual environment is activated
which python  # Should point to .venv/bin/python
```

#### 6. Model Download Fails

```bash
# Manually download models
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
"

# Or download directly and place in checkpoints/
# https://github.com/ultralytics/assets/releases
```

---

## Next Steps

After successful installation:

1. 📖 **Read**: [Quick Reference Guide](../guides/QUICK_REFERENCE.md)
2. 🧪 **Test**: [Mock Observer Guide](../guides/MOCK_OBSERVER_GUIDE.md)
3. 🏗️ **Understand**: [Architecture Overview](../architecture/architecture_document.md)
4. 🚀 **Migrate**: [NUC Migration Guide](../migration/NUC_MIGRATION.md) (for production)

---

## Getting Help

- 📚 **Documentation**: [docs/INDEX.md](../INDEX.md)
- 🐛 **Issues**: Check [TROUBLESHOOTING](../TROUBLESHOOTING.md) and quick steps in [Contributing](../development/CONTRIBUTING.md)
- 💬 **Questions**: Open GitHub issue with `[Question]` tag

---

## Additional Resources

- [Meta Aria Developer Portal](https://projectaria.com/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)

---

**Installation complete?** Test with `python src/main.py debug` 🎉
