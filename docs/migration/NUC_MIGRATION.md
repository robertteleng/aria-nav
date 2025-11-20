# 🚀 Intel NUC 11 + RTX 2060 Migration Guide

> **Consolidated migration guide for moving from macOS (MPS) to Intel NUC with NVIDIA CUDA**  
> Last updated: November 20, 2025  
> Status: ✅ Active

## Table of Contents
1. [Hardware & Software Targets](#hardware--software-targets)
2. [Pre-Migration Checklist](#pre-migration-checklist)
3. [Phase 1: Environment Setup](#phase-1-environment-setup)
4. [Phase 2: Code Migration](#phase-2-code-migration)
5. [Phase 3: Optimization](#phase-3-optimization)
6. [Phase 4: Validation](#phase-4-validation)

---

## Hardware & Software Targets

### Target Hardware
- **Device**: Intel NUC 11 Enthusiast
- **GPU**: NVIDIA RTX 2060 (6GB VRAM)
- **OS**: Ubuntu 22.04 LTS
- **Connectivity**: USB-C/USB-A for Meta Aria glasses

### Current Baseline (macOS)
- **Platform**: macOS (Apple Silicon M1/M2)
- **Acceleration**: Metal Performance Shaders (MPS)
- **Performance**: ~18-20 FPS with frame skipping
- **Limitations**: MPS backend instability, limited VRAM

### Expected Improvements
- **Performance**: 3-5x speedup (target: 60+ FPS)
- **Stability**: Better CUDA maturity vs MPS
- **TensorRT**: Native optimization for depth + YOLO
- **Memory**: 6GB dedicated VRAM vs shared MPS

---

## Pre-Migration Checklist

### 1. Capture Current Baseline
```bash
# On macOS
pip freeze > reports/requirements_mac_$(date +%Y%m%d).txt
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"

# Capture performance metrics
python src/main.py --profile
cp -r logs/profiling/ logs/baseline_mac/
```

### 2. Inventory Assets
- [ ] Model weights: `checkpoints/*.pt`, `checkpoints/*.pth`
- [ ] TensorRT engines: `checkpoints/*.engine` (will need regeneration)
- [ ] Config files: `utils/config.py`, environment variables
- [ ] Aria SDK credentials and calibration data

### 3. Verify Compatibility
- [ ] `projectaria-tools` supports Ubuntu 22.04 x86_64
- [ ] `aria-sdk` has Linux build available
- [ ] All Python dependencies support Linux

---

## Phase 1: Environment Setup

### 1.1. Install Ubuntu 22.04 LTS

```bash
# Download Ubuntu 22.04 LTS
# Create bootable USB with Rufus (Windows) or dd (Linux/Mac)
# Install on NUC with full disk or dual boot
```

### 1.2. Install NVIDIA Drivers

```bash
# Auto-install recommended drivers
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot

# Verify installation
nvidia-smi  # Should show RTX 2060
```

Expected output:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx       Driver Version: 535.xx       CUDA Version: 12.2    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce RTX 2060  Off  | 00000000:01:00.0 On  |                  N/A |
```

### 1.3. Install CUDA Toolkit & cuDNN

```bash
# Install CUDA 11.8 (recommended for PyTorch compatibility)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

Install cuDNN:
```bash
# Download cuDNN from NVIDIA (requires account)
# https://developer.nvidia.com/cudnn

# Extract and copy files
tar -xvf cudnn-linux-x86_64-8.x.x.x_cudaX.Y-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

### 1.4. Setup Python Environment

```bash
# Install Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Clone repository
git clone https://github.com/\<your-user\>/aria-nav.git
cd aria-nav

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True NVIDIA GeForce RTX 2060

# Install other dependencies
pip install ultralytics opencv-python numpy projectaria-tools transformers pytest
```

---

## Phase 2: Code Migration

### 2.1. Device Configuration Changes

**File: `src/utils/config.py`**

```python
# OLD (macOS)
YOLO_DEVICE = "mps"
DEPTH_DEVICE = "mps"
YOLO_FORCE_MPS = True

# NEW (Linux CUDA)
YOLO_DEVICE = "cuda"
DEPTH_DEVICE = "cuda"
YOLO_FORCE_MPS = False
```

### 2.2. Audio System Migration

macOS uses `say` command - Linux needs alternatives.

**Option 1: pyttsx3 (Recommended)**
```bash
pip install pyttsx3
sudo apt-get install espeak espeak-data libespeak-dev
```

**File: `src/core/audio/audio_system.py`**
```python
import pyttsx3

class AudioSystem:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)  # Speed
        self.engine.setProperty('volume', 1.0)
    
    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()
```

**Option 2: espeak (Faster, less natural)**
```python
import subprocess

def speak(text: str):
    subprocess.run(['espeak', '-s', '175', text])
```

See [LINUX_AUDIO.md](LINUX_AUDIO.md) for complete guide.

### 2.3. Remove MPS-Specific Code

Search and remove MPS references:
```bash
grep -r "mps" src/
grep -r "MPS" src/
```

**File: `src/core/vision/yolo_processor.py`**
```python
# Remove MPS workarounds
if self.device == 'mps':
    # MPS-specific hacks
    ...
```

---

## Phase 3: Optimization

### 3.1. TensorRT Conversion

**Depth Anything v2:**
```bash
python export_tensorrt_slam.py
# Generates: checkpoints/depth_anything_v2_vits.engine
```

**YOLO:**
```bash
yolo export model=yolo12n.pt format=engine device=0 imgsz=640
# Generates: yolo12n.engine
```

**Update config:**
```python
# src/utils/config.py
DEPTH_USE_TENSORRT = True
DEPTH_TENSORRT_PATH = "checkpoints/depth_anything_v2_vits.engine"

YOLO_USE_TENSORRT = True
YOLO_TENSORRT_PATH = "checkpoints/yolo12n.engine"
```

### 3.2. Remove Frame Skipping

With better GPU, reduce or eliminate frame skipping:
```python
# config.py
YOLO_FRAME_SKIP = 0      # Was 3 on macOS
DEPTH_FRAME_SKIP = 2     # Was 12 on macOS
```

### 3.3. Enable Async Processing

```python
# config.py
PERIPHERAL_VISION_ENABLED = True  # SLAM cameras
ASYNC_TELEMETRY = True            # Background logging
```

---

## Phase 4: Validation

### 4.1. Functional Tests

```bash
# Run test suite
pytest tests/ -v

# Test with mock hardware
python src/main.py debug

# Test with real Aria
python src/main.py
```

### 4.2. Performance Benchmarks

```bash
# Profile pipeline
python benchmarks/benchmark_1_performance.py

# Compare with baseline
python tools/compare_benchmarks.py \
    logs/baseline_mac/performance.json \
    logs/profiling/performance.json
```

### 4.3. Expected Metrics

| Metric | macOS (MPS) | NUC (CUDA) | Target |
|--------|-------------|------------|--------|
| **Overall FPS** | 18-20 | 45-60 | 60+ |
| **YOLO Latency** | 45-60ms | 10-15ms | <20ms |
| **Depth Latency** | 120-180ms | 25-40ms | <50ms |
| **Frame Skip** | 3x YOLO, 12x Depth | 0x YOLO, 2x Depth | Minimal |
| **GPU Util** | ~60% | ~80% | Optimal |

---

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch sizes or model size
YOLO_MODEL = "yolo12n"  # Instead of yolo12m
DEPTH_MODEL = "vits"    # Instead of vitb
```

### Audio Not Working
```bash
# Test espeak
espeak "Hello world"

# Check audio devices
aplay -l

# Install missing codecs
sudo apt-get install pulseaudio
```

### Aria SDK Connection Issues
```bash
# Check USB permissions
sudo usermod -aG plugdev $USER
sudo udevadm control --reload-rules
```

### Low FPS Despite Good GPU
- Check CUDA version matches PyTorch
- Verify TensorRT engines are loading
- Monitor GPU usage: `watch -n 1 nvidia-smi`
- Check thermal throttling

---

## Next Steps After Migration

1. **Fine-tune parameters** based on real-world performance
2. **Optimize TensorRT engines** with FP16/INT8 quantization
3. **Enable all features** that were disabled on macOS
4. **Collect new benchmarks** for documentation
5. **Update README** with Linux-specific instructions

---

## References

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [PyTorch CUDA Setup](https://pytorch.org/get-started/locally/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Linux Audio Setup](LINUX_AUDIO.md)
- [Audio Router Migration](AUDIO_ROUTER_MIGRATION.md)

---

**Migration issues?** Open an issue with logs from `logs/` directory.
