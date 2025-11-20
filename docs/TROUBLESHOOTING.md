# 🔧 Troubleshooting Guide

> **Complete guide to debugging common issues in Aria Navigation System**  
> Compiled from 10 iterations of development and bug fixes  
> Last updated: November 20, 2025

---

## 📋 Quick Navigation

| Problem Category | Link |
|-----------------|------|
| [Installation & Setup](#installation--setup) | Environment, dependencies, hardware |
| [TensorRT & ONNX](#tensorrt--onnx-issues) | Model export, inference engines |
| [Depth Estimation](#depth-estimation-issues) | Depth not running, accuracy problems |
| [YOLO Detection](#yolo-detection-issues) | Detection failures, low accuracy |
| [Audio System](#audio-system-issues) | TTS not working, audio routing |
| [Multiprocessing](#multiprocessing-issues) | Worker crashes, queue overflows |
| [Performance](#performance-issues) | Low FPS, high latency |
| [Aria Hardware](#aria-hardware-issues) | USB connection, streaming failures |
| [GPU & CUDA](#gpu--cuda-issues) | Memory errors, driver issues |
| [Logging & Telemetry](#logging--telemetry-issues) | Log corruption, telemetry spikes |

---

## 🚨 Critical Bug Fixes (Historical)

### ✅ FIXED: Depth Not Executing (Nov 17, 2025)

**Symptom:**
```
✓ Depth-Anything-V2 TensorRT engine loaded successfully
⚠️ Estimator created but model is None
⚠️ Depth estimator initialized without model (disabled)
⚠️ Depth estimator model failed to load - depth estimation disabled
```

**Root Cause:**
Pipeline validation checks only looked for PyTorch model, ignoring ONNX Runtime session.

**Solution:**
```python
# BEFORE (broken)
if self.depth_estimator.model is not None:
    # depth code

# AFTER (fixed)
if (self.depth_estimator.model is not None or 
    self.depth_estimator.ort_session is not None):
    # depth code
```

**Files Changed:**
- `src/core/navigation/navigation_pipeline.py` (lines 95, 141, 179, 512)

**Verification:**
```bash
python run.py --telemetry
# Check logs for "Depth estimation: 27ms"
```

---

### ✅ FIXED: Depth Resize Bug (Nov 17, 2025)

**Symptom:**
- ONNX depth runs but takes 315ms (slower than PyTorch!)
- Logs show processing 1408x1408 images

**Root Cause:**
Conditional resize logic failed to resize 1408x1408 Aria frames to 384x384 ONNX input size.

**Solution:**
```python
# BEFORE (buggy)
if self.input_size and max(rgb_input.shape[:2]) > self.input_size:
    scale = self.input_size / max(rgb_input.shape[:2])
    new_size = (int(rgb_input.shape[1] * scale), int(rgb_input.shape[0] * scale))
    rgb_resized = cv2.resize(rgb_input, new_size, interpolation=cv2.INTER_AREA)
else:
    rgb_resized = rgb_input  # ❌ Processed 1408x1408!

# AFTER (fixed)
if self.input_size:
    rgb_resized = cv2.resize(
        rgb_input, 
        (self.input_size, self.input_size),  # ✅ Always 384x384
        interpolation=cv2.INTER_AREA
    )
else:
    rgb_resized = rgb_input
```

**Files Changed:**
- `src/core/vision/depth_estimator.py` (line 346)

**Impact:**
- **Before:** 315ms @ 1408x1408
- **After:** 27ms @ 384x384 (11.7x speedup)

---

### ✅ FIXED: ONNX Running on CPU (Nov 17, 2025)

**Symptom:**
- ONNX loaded successfully
- Depth estimation takes 192ms (should be ~30ms)
- No CUDA errors visible

**Root Cause:**
TensorRT Execution Provider failed silently, fell back to CPU.

**Solution:**
```python
# BEFORE (fallback to CPU)
providers=[
    'TensorrtExecutionProvider',  # Fails silently
    'CUDAExecutionProvider',      # Never reached
    'CPUExecutionProvider'        # Used instead!
]

# AFTER (force CUDA only)
providers=[
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    })
]
```

**Files Changed:**
- `src/core/vision/depth_estimator.py` (line 131)

**Verification:**
```python
# Check ONNX RT provider
import onnxruntime as ort
session = ort.InferenceSession("depth.onnx")
print(session.get_providers())
# Should show: ['CUDAExecutionProvider']
```

---

### ✅ FIXED: Motion State Always Stationary (Nov 20, 2025)

**Symptom:**
- IMU data streaming correctly
- Motion state always returns 'stationary'
- No state changes detected

**Root Cause:**
Hysteresis logic broken - line 216 always returned default 'stationary' instead of maintaining last state.

**Solution:**
```python
# BEFORE (broken)
def _estimate_motion_state(self) -> str:
    # ... compute std_dev
    if std_dev < 0.3:
        return 'stationary'
    elif std_dev > 0.6:
        return 'walking'
    else:
        return 'stationary'  # ❌ Always returns stationary!

# AFTER (fixed)
def _estimate_motion_state(self) -> str:
    # ... compute std_dev
    if std_dev < 0.3:
        return 'stationary'
    elif std_dev > 0.6:
        return 'walking'
    else:
        return self._last_motion_state or 'stationary'  # ✅ Hysteresis

def _update_last_motion_state(self, state: str):
    self._last_motion_state = state
```

**Files Changed:**
- `src/core/observer.py` (line 216, added _update_last_motion_state method)

**Verification:**
```bash
python run.py --telemetry
# Walk around, check logs for "Motion: walking"
```

---

## Installation & Setup

### Issue: `projectaria-tools` Not Found

**Symptom:**
```bash
ModuleNotFoundError: No module named 'projectaria_tools'
```

**Solutions:**

**1. macOS Installation:**
```bash
# Install from source (recommended)
brew install cmake
git clone https://github.com/facebookresearch/projectaria_tools.git
cd projectaria_tools
./build_tools.sh
pip install .

# Or use pre-built wheel
pip install projectaria-tools
```

**2. Linux/WSL2 Installation:**
```bash
# WSL2 DOES NOT SUPPORT Aria USB streaming!
# Use native Linux instead

# Ubuntu/Debian
sudo apt install cmake build-essential
git clone https://github.com/facebookresearch/projectaria_tools.git
cd projectaria_tools
./build_tools.sh
pip install .
```

**3. Verify Installation:**
```python
python -c "import projectaria_tools; print(projectaria_tools.__version__)"
```

---

### Issue: CUDA Not Available

**Symptom:**
```python
torch.cuda.is_available()  # Returns False
```

**Solutions:**

**1. Check NVIDIA Driver:**
```bash
nvidia-smi
# Should show GPU and CUDA version
```

**2. Install CUDA Toolkit:**
```bash
# Linux
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Verify
nvcc --version
```

**3. Install PyTorch with CUDA:**
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**4. Verify:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

### Issue: Permission Denied on USB Device

**Symptom:**
```
Error: Cannot open USB device: Permission denied
```

**Solution (Linux only):**
```bash
# Add user to plugdev group
sudo usermod -a -G plugdev $USER

# Create udev rule for Aria glasses
sudo tee /etc/udev/rules.d/99-aria.rules << EOF
SUBSYSTEM=="usb", ATTR{idVendor}=="2833", MODE="0666"
EOF

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger

# Logout and login
```

---

## TensorRT & ONNX Issues

### Issue: TensorRT Export Fails

**Symptom:**
```
RuntimeError: Unsupported operator: xxx
TensorRT export failed
```

**Solution:**

**1. Use ONNX Intermediate Format:**
```bash
# Export to ONNX first
yolo export model=yolo12n.pt format=onnx opset=17

# Convert ONNX to TensorRT manually
trtexec --onnx=yolo12n.onnx \
        --saveEngine=yolo12n.engine \
        --fp16 \
        --workspace=4096 \
        --verbose
```

**2. Check ONNX Compatibility:**
```python
import onnx
model = onnx.load("yolo12n.onnx")
onnx.checker.check_model(model)
```

**3. If Still Fails:**
- Use ONNX Runtime instead of TensorRT
- Or simplify model architecture
- Or use PyTorch backend (slower)

---

### Issue: ONNX Model Wrong Input Size

**Symptom:**
```
RuntimeError: Input size mismatch: got [1, 3, 640, 640], expected [1, 3, 384, 384]
```

**Solution:**

**Check ONNX Input Shape:**
```python
import onnx
model = onnx.load("depth.onnx")
input_shape = model.graph.input[0].type.tensor_type.shape.dim
print([d.dim_value for d in input_shape])  # [1, 3, 384, 384]
```

**Fix Preprocessing:**
```python
# Always resize to ONNX input size
rgb_resized = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_AREA)
```

**Re-export with Dynamic Axes:**
```python
torch.onnx.export(
    model, dummy_input, "model.onnx",
    dynamic_axes={
        'image': {0: 'batch', 2: 'height', 3: 'width'},
        'output': {0: 'batch'}
    }
)
```

---

### Issue: TensorRT Engine Crashes

**Symptom:**
```
Segmentation fault (core dumped)
CUDA error: invalid device function
```

**Solution:**

**1. Check CUDA Compute Capability:**
```python
import torch
capability = torch.cuda.get_device_capability()
print(f"Compute capability: {capability}")

# TensorRT requires >= 5.3
```

**2. Re-build Engine for Target GPU:**
```bash
# On target machine (not source machine!)
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```

**3. Check TensorRT Version:**
```python
import tensorrt as trt
print(trt.__version__)  # Should match CUDA version
```

---

## Depth Estimation Issues

### Issue: Depth Returns None

**Symptom:**
```python
depth_map = depth_estimator.estimate_depth(frame)
print(depth_map)  # None
```

**Solutions:**

**1. Check Model Loading:**
```python
# Add to DepthEstimator
def _is_model_loaded(self) -> bool:
    if self.use_onnx:
        return self.session is not None
    else:
        return self.model is not None

# Use in pipeline
if not depth_estimator._is_model_loaded():
    print("ERROR: Model not loaded!")
```

**2. Verify Input Shape:**
```python
print(f"Frame shape: {frame.shape}")  # Should be (H, W, 3)
print(f"Input size: {depth_estimator.input_size}")  # Should match model
```

**3. Check Preprocessing:**
```python
# In estimate_depth() add debug prints
print(f"Input shape: {rgb_input.shape}")
print(f"Resized shape: {rgb_resized.shape}")
print(f"Tensor shape: {input_tensor.shape}")
```

**4. Test with Dummy Input:**
```python
import numpy as np
dummy = np.random.rand(480, 640, 3).astype(np.uint8)
depth = depth_estimator.estimate_depth(dummy)
print(f"Depth output: {depth.shape if depth is not None else None}")
```

---

### Issue: Depth Very Slow (>200ms)

**Symptom:**
- Depth estimation takes 200-300ms
- Expected ~30ms with ONNX/CUDA

**Solutions:**

**1. Check Execution Provider:**
```python
print(depth_estimator.session.get_providers())
# Should show: ['CUDAExecutionProvider']
# NOT: ['CPUExecutionProvider']
```

**2. Check Input Size:**
```python
# Large inputs = slow inference
print(f"Processing size: {rgb_resized.shape}")
# Should be (384, 384, 3) for depth
```

**3. Verify GPU Usage:**
```bash
# Run in another terminal
watch -n 1 nvidia-smi

# Look for python process using GPU
# GPU Memory should be 3-5GB
```

**4. Disable Unnecessary Post-processing:**
```python
# In depth_estimator.py
def estimate_depth(self, image):
    # ... inference
    # Skip heavy post-processing if not needed
    # depth = cv2.GaussianBlur(depth, (5, 5), 0)  # Optional
    return depth
```

---

### Issue: Depth Map Quality Poor

**Symptom:**
- Depth map looks noisy/pixelated
- Poor edge definition

**Solutions:**

**1. Increase Input Resolution:**
```python
# In config or depth_estimator
input_size = 518  # Instead of 384
# Note: Slower inference (~50ms vs 27ms)
```

**2. Enable Post-processing:**
```python
depth = cv2.GaussianBlur(depth, (5, 5), 0)
depth = cv2.bilateralFilter(depth, 9, 75, 75)
```

**3. Use Better Model:**
```python
# Load larger variant
model_name = "depth_anything_v2_vitb.onnx"  # Base instead of Small
# Note: Slower but more accurate
```

---

## YOLO Detection Issues

### Issue: No Detections

**Symptom:**
```python
results = yolo_processor.detect(frame)
print(len(results))  # 0
```

**Solutions:**

**1. Lower Confidence Threshold:**
```python
# In yolo_processor or config
confidence_threshold = 0.3  # Instead of 0.5
```

**2. Check Image Quality:**
```python
# Display frame to verify it's not corrupted
cv2.imshow("Frame", frame)
cv2.waitKey(0)
```

**3. Test with Known Image:**
```python
# Download COCO test image
import requests
from PIL import Image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img = Image.open(requests.get(url, stream=True).raw)
results = yolo_processor.detect(np.array(img))
print(f"Detections: {len(results)}")  # Should detect cats
```

**4. Verify Model Loaded:**
```python
print(yolo_processor.model)  # Should not be None
print(yolo_processor.model.names)  # Should show class names
```

---

### Issue: Low Detection Accuracy

**Symptom:**
- Many false positives/negatives
- Wrong classes detected

**Solutions:**

**1. Increase Model Size:**
```python
# Use larger YOLO variant
model = "yolo11s.pt"  # Small instead of Nano
model = "yolo11m.pt"  # Medium (slower, more accurate)
```

**2. Fine-tune Thresholds:**
```python
confidence_threshold = 0.5  # Higher = fewer false positives
iou_threshold = 0.5  # Higher = fewer duplicate boxes
```

**3. Increase Input Resolution:**
```python
yolo_input_size = 640  # Standard
yolo_input_size = 1280  # High accuracy (slower)
```

**4. Filter Classes:**
```python
# Only detect persons and obstacles
allowed_classes = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign']
results = [r for r in results if r['class'] in allowed_classes]
```

---

## Audio System Issues

### Issue: TTS Not Working (macOS)

**Symptom:**
```
Audio message sent but nothing plays
No error messages
```

**Solutions:**

**1. Test TTS Directly:**
```bash
say "Test message"
# Should hear audio
```

**2. Check Audio Output:**
```bash
system_profiler SPAudioDataType
# Verify output device available
```

**3. Check Cooldown:**
```python
# Audio may be rate-limited
print(audio_system.cooldown_active('person'))
# If True, messages are suppressed
```

**4. Enable Debug Logging:**
```python
# In audio_system.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

### Issue: TTS Not Working (Linux)

**Symptom:**
```
ModuleNotFoundError: No module named 'pyttsx3'
```

**Solutions:**

**1. Install TTS Engine:**
```bash
# Install espeak
sudo apt install espeak

# Install pyttsx3
pip install pyttsx3
```

**2. Configure Audio System:**
```python
# In audio_system.py (Linux version)
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)
engine.say("Test message")
engine.runAndWait()
```

**3. Alternative: Use System Command:**
```python
import subprocess
subprocess.run(['espeak', 'Test message'])
```

---

### Issue: Audio Telemetry Spikes

**Symptom:**
```
FPS drops from 18 to 14 every 50 frames
Spikes visible in logs: 250-350ms
```

**Solution (Already Fixed in v2.0):**

Use async telemetry logger:
```python
# In coordinator or pipeline
from src.utils.telemetry_logger import AsyncTelemetryLogger

logger = AsyncTelemetryLogger(session_dir)
logger.log_audio_event(event)  # Non-blocking
```

**Verification:**
```bash
# Check FPS stays consistent
tail -f logs/session_*/performance.jsonl | jq '.fps'
```

---

## Multiprocessing Issues

### Issue: Worker Process Crashes

**Symptom:**
```
Process SlamWorker-1 terminated unexpectedly
No error messages in logs
```

**Solutions:**

**1. Check GPU Memory:**
```bash
nvidia-smi
# Each worker uses 1-2GB
# Ensure enough VRAM available
```

**2. Limit Per-Process Memory:**
```python
# In worker process
import torch
torch.cuda.set_per_process_memory_fraction(0.25, device=0)
```

**3. Add Error Handling:**
```python
def _worker_loop(self):
    try:
        # ... worker code
    except Exception as e:
        print(f"Worker error: {e}")
        import traceback
        traceback.print_exc()
```

**4. Check Queue Sizes:**
```python
# Reduce queue sizes to prevent memory bloat
input_queue = Queue(maxsize=2)  # Was 10
output_queue = Queue(maxsize=5)  # Was 20
```

---

### Issue: Queue Overflow/Deadlock

**Symptom:**
```
Warning: Frame dropped (queue full)
Process hangs, no progress
```

**Solutions:**

**1. Use Timeouts:**
```python
# Non-blocking put/get
try:
    queue.put(item, timeout=0.1)
except:
    print("Queue full, dropping frame")

try:
    item = queue.get(timeout=0.01)
except:
    return None  # No items available
```

**2. Reduce Queue Size:**
```python
# Drop old frames faster
queue = Queue(maxsize=1)  # Only keep latest
```

**3. Monitor Queue Depth:**
```python
print(f"Queue size: {queue.qsize()} / {queue.maxsize}")
```

**4. Implement Backpressure:**
```python
if queue.qsize() > queue.maxsize * 0.8:
    print("Queue nearly full, skipping frame")
    continue
```

---

### Issue: Serialization Slow (pickle)

**Symptom:**
```
Frame transfer takes 5-10ms
CPU usage high during transfers
```

**Solutions:**

**1. Use Shared Memory (Future):**
```python
# Not implemented yet, planned for v2.1
from multiprocessing import shared_memory
# Share np.ndarray without pickle
```

**2. Reduce Frame Size:**
```python
# Downsample before sending
frame_small = cv2.resize(frame, (640, 480))
queue.put(frame_small)
```

**3. Use torch.multiprocessing:**
```python
import torch.multiprocessing as mp
mp.set_start_method('spawn')
# Better tensor sharing
```

---

## Performance Issues

### Issue: Low FPS (<10 FPS)

**Symptom:**
- Sustained FPS below 10
- Expected 18-20 FPS

**Debugging Steps:**

**1. Profile Pipeline:**
```python
import time

# Add timing to each step
t0 = time.time()
detections = yolo_processor.detect(frame)
t1 = time.time()
print(f"YOLO: {(t1-t0)*1000:.2f}ms")

depth = depth_estimator.estimate_depth(frame)
t2 = time.time()
print(f"Depth: {(t2-t1)*1000:.2f}ms")
```

**2. Check GPU Utilization:**
```bash
nvidia-smi
# Should see 85-95% GPU usage
# If low, check CPU bottlenecks
```

**3. Check Model Backends:**
```python
# Verify using fast backends
print(yolo_processor.model_path)  # Should end with .engine
print(depth_estimator.use_onnx)   # Should be True
print(depth_estimator.session.get_providers())  # Should be CUDA
```

**4. Disable Unnecessary Features:**
```python
# Temporarily disable components
DEPTH_ENABLED = False
SLAM_ENABLED = False
# Re-test FPS
```

---

### Issue: High Latency (>100ms)

**Symptom:**
- End-to-end latency > 100ms
- Expected ~50ms

**Solutions:**

**1. Reduce Frame Skipping:**
```python
# In config
YOLO_SKIP_FRAMES = 1  # Instead of 3
DEPTH_SKIP_FRAMES = 3  # Instead of 12
```

**2. Optimize Display:**
```python
# Display at lower resolution
frame_display = cv2.resize(frame, (960, 540))
cv2.imshow("Display", frame_display)
```

**3. Use Async Display:**
```python
# Display in separate thread
from threading import Thread
Thread(target=lambda: cv2.imshow("Display", frame), daemon=True).start()
```

---

## Aria Hardware Issues

### Issue: Cannot Connect to Aria

**Symptom:**
```
Error: No Aria device found
USB device not detected
```

**Solutions:**

**1. Check USB Connection:**
```bash
lsusb | grep 2833
# Should show Meta device
```

**2. Check Driver:**
```bash
# macOS
system_profiler SPUSBDataType | grep -A 10 "Meta"

# Linux
dmesg | grep usb
```

**3. Try Different USB Port:**
- Use USB 3.0 port (not 2.0)
- Avoid USB hubs
- Try different cable

**4. Restart Aria Glasses:**
- Power off completely
- Wait 10 seconds
- Power on
- Wait for blue LED

---

### Issue: Streaming Starts Then Stops

**Symptom:**
```
Streaming starts
After 5-10 seconds: Connection lost
Error: Device disconnected
```

**Solutions:**

**1. Check USB Power:**
```bash
# Increase USB power limit (Linux)
echo 1 | sudo tee /sys/bus/usb/devices/*/power/autosuspend
```

**2. Use Powered USB Hub:**
- Aria draws significant power
- Some laptops can't provide enough

**3. Check Streaming Profile:**
```python
# Use less demanding profile
profile = aria.StreamingProfile.Profile18  # Instead of Profile28
```

**4. Reduce Frame Rate:**
```python
# In observer configuration
rgb_fps = 30  # Instead of 60
```

---

## GPU & CUDA Issues

### Issue: CUDA Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**

**1. Check Memory Usage:**
```bash
nvidia-smi
# Check total memory usage
```

**2. Reduce Model Sizes:**
```python
# Use smaller YOLO variant
model = "yolo11n.pt"  # Nano (smallest)

# Reduce batch size
batch_size = 1
```

**3. Clear GPU Cache:**
```python
import torch
torch.cuda.empty_cache()
```

**4. Limit Per-Process Memory:**
```python
torch.cuda.set_per_process_memory_fraction(0.3, device=0)
```

**5. Use CPU for Some Operations:**
```python
# Run YOLO on GPU, depth on CPU
depth_estimator = DepthEstimator(device='cpu')
```

---

### Issue: CUDA Driver Version Mismatch

**Symptom:**
```
RuntimeError: CUDA driver version is insufficient for CUDA runtime version
```

**Solution:**

**1. Check Versions:**
```bash
nvidia-smi  # Driver version
nvcc --version  # CUDA toolkit version
```

**2. Update NVIDIA Driver:**
```bash
# Ubuntu
sudo ubuntu-drivers autoinstall

# Or download from NVIDIA website
```

**3. Install Matching CUDA Toolkit:**
```bash
# Match driver version (see nvidia-smi)
# Driver 525+ → CUDA 12.x
# Driver 470-524 → CUDA 11.x
```

---

## Logging & Telemetry Issues

### Issue: Log Files Not Created

**Symptom:**
```
No logs in logs/session_*/
Telemetry logger initialized but no output
```

**Solutions:**

**1. Check Permissions:**
```bash
ls -la logs/
chmod 755 logs/
```

**2. Check Session Directory:**
```python
print(f"Session dir: {telemetry_logger.session_dir}")
ls -la logs/session_*/
```

**3. Force Flush:**
```python
# In telemetry logger
logger.log_event(event)
logger.flush()  # Force write
```

---

### Issue: JSONL Files Corrupted

**Symptom:**
```
JSONDecodeError: Extra data
Invalid JSONL format
```

**Solution:**

**1. Check Async Logger:**
```python
# Use AsyncTelemetryLogger (v2.0+)
from src.utils.telemetry_logger import AsyncTelemetryLogger

logger = AsyncTelemetryLogger(session_dir)
# Handles buffering automatically
```

**2. Validate JSONL:**
```bash
# Check each line is valid JSON
cat logs/session_*/performance.jsonl | while read line; do echo "$line" | jq . > /dev/null || echo "Invalid: $line"; done
```

**3. Recovery:**
```bash
# Filter valid lines only
cat corrupted.jsonl | jq -c . 2>/dev/null > valid.jsonl
```

---

## General Debugging Strategies

### Enable Verbose Logging

```python
# In run.py or main script
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Use Mock Observer

```bash
# Test without hardware
python run.py --mock
```

### Profile Performance

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Run pipeline

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

### Check System Resources

```bash
# CPU/Memory usage
htop

# GPU usage
watch -n 1 nvidia-smi

# Disk I/O
iotop

# Network (if using WiFi streaming)
iftop
```

---

## 📚 Additional Resources

### Internal Documentation
- [Setup Guide](setup/SETUP.md) - Complete installation instructions
- [Architecture Document](architecture/architecture_document.md) - System design
- [Problem Solving Guide](development/problem_solving_guide.md) - Debugging strategies
- [CUDA Optimization](migration/CUDA_OPTIMIZATION.md) - Performance optimization guide

### External Resources
- [Aria SDK Documentation](https://facebookresearch.github.io/projectaria_tools/)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [ONNX Runtime Troubleshooting](https://onnxruntime.ai/docs/reference/troubleshooting.html)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)

---

## 🆘 Getting Help

### Before Opening an Issue

1. ✅ Check this troubleshooting guide
2. ✅ Search existing issues on GitHub
3. ✅ Enable debug logging and collect logs
4. ✅ Test with mock observer (if applicable)
5. ✅ Verify hardware/driver setup

### Issue Template

```markdown
**System Information:**
- OS: [macOS 14.0 / Ubuntu 22.04 / etc.]
- Python version: [3.10]
- CUDA version: [12.1]
- GPU: [RTX 3070]
- Aria SDK version: [1.x.x]

**Issue Description:**
[Clear description of the problem]

**Steps to Reproduce:**
1. [Step 1]
2. [Step 2]
3. [Error occurs]

**Expected Behavior:**
[What should happen]

**Actual Behavior:**
[What actually happens]

**Logs:**
```
[Paste relevant log excerpts]
```

**Additional Context:**
[Any other relevant information]
```

---

**Troubleshooting Status:** ✅ Comprehensive  
**Coverage:** 10+ iterations of bug fixes  
**Last Updated:** November 20, 2025

---

*For development workflow, see [development_workflow.md](development/development_workflow.md)*  
*For quick reference, see [QUICK_REFERENCE.md](guides/QUICK_REFERENCE.md)*
