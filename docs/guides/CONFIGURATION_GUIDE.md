# ⚙️ Configuration Guide: Tuning the Aria Navigation System

> **Complete guide to system configuration and parameter tuning**
> Last updated: November 25, 2025

## Overview

The Aria Navigation System uses a centralized configuration class (`Config`) in `src/utils/config.py`. This guide explains all configuration options, their impact, and how to tune them for different scenarios.

---

## Table of Contents

1. [Quick Configuration](#quick-configuration)
2. [Device Detection](#device-detection)
3. [CUDA Optimizations](#cuda-optimizations)
4. [YOLO Configuration](#yolo-configuration)
5. [Depth Estimation](#depth-estimation)
6. [Audio System](#audio-system)
7. [Spatial Processing](#spatial-processing)
8. [Performance Tuning](#performance-tuning)
9. [Advanced Options](#advanced-options)

---

## 1. Quick Configuration

### Common Scenarios

**Scenario 1: Maximum Performance (Default)**
```python
# src/utils/config.py
Config.YOLO_FRAME_SKIP = 3          # Process every 3rd frame
Config.DEPTH_FRAME_SKIP = 12        # Depth every 12th frame
Config.PHASE6_HYBRID_STREAMS = True # GPU parallelization
```
**Result:** 18-22 FPS, balanced accuracy/speed

**Scenario 2: Maximum Accuracy (Slower)**
```python
Config.YOLO_FRAME_SKIP = 1          # Process every frame
Config.DEPTH_FRAME_SKIP = 6         # Depth every 6th frame
Config.YOLO_RGB_IMAGE_SIZE = 640    # Keep high resolution
```
**Result:** 10-12 FPS, higher detection accuracy

**Scenario 3: Maximum Speed (Lower Accuracy)**
```python
Config.YOLO_FRAME_SKIP = 5          # Skip more frames
Config.DEPTH_FRAME_SKIP = 15        # Less frequent depth
Config.YOLO_RGB_IMAGE_SIZE = 416    # Lower resolution
Config.PERIPHERAL_VISION_ENABLED = False  # Disable SLAM
```
**Result:** 25-30 FPS, reduced accuracy

---

## 2. Device Detection

### Automatic Device Selection

```python
def detect_device():
    """Auto-detect best device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"
```

**Configuration:**
```python
Config.YOLO_DEVICE = DEVICE  # Auto-detected
```

**Manual Override:**
```python
# Force specific device
Config.YOLO_DEVICE = "cuda:1"  # Use second GPU
Config.DEPTH_ANYTHING_DEVICE = "cuda:0"  # Use first GPU
```

### Device Optimization

**CUDA Optimizations** (Enabled automatically on NVIDIA GPUs):
```python
torch.backends.cudnn.benchmark = True  # Auto-tune kernels
torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')
```

**Impact:** +15-20% performance on NVIDIA GPUs

---

## 3. CUDA Optimizations

### Phase 1 Optimizations (Always Enabled)

```python
Config.CUDA_OPTIMIZATIONS = True
Config.PINNED_MEMORY = True           # Faster CPU→GPU transfers
Config.NON_BLOCKING_TRANSFER = True   # Async transfers
Config.CUDA_STREAMS = True            # Stream scheduling
```

**What they do:**
- **cuDNN Benchmark:** Auto-selects fastest convolution algorithms
- **TF32:** Uses TensorFloat-32 for matmul (2x faster, minimal accuracy loss)
- **Pinned Memory:** Eliminates copy overhead for GPU transfers
- **Streams:** Parallel GPU operations (depth + YOLO overlap)

### Phase 6 Hybrid Streams

```python
Config.PHASE6_HYBRID_STREAMS = True
```

**Enables:**
- Parallel depth estimation and YOLO inference
- CUDA stream synchronization
- +3% FPS improvement

**Disable if:**
- Single GPU with limited VRAM
- Debugging inference issues

---

## 4. YOLO Configuration

### Resolution Settings

```python
# RGB Camera (high resolution for primary view)
Config.YOLO_RGB_IMAGE_SIZE = 640
Config.YOLO_RGB_CONFIDENCE = 0.50
Config.YOLO_RGB_MAX_DETECTIONS = 20

# SLAM Cameras (lower resolution for peripheral vision)
Config.YOLO_SLAM_IMAGE_SIZE = 256
Config.YOLO_SLAM_CONFIDENCE = 0.60
Config.YOLO_SLAM_MAX_DETECTIONS = 8
```

**Trade-offs:**

| Size | FPS | Accuracy | Use Case |
|------|-----|----------|----------|
| 256 | Fast | Low | Peripheral vision, quick scans |
| 416 | Medium | Medium | Balanced performance |
| 640 | Slow | High | Primary RGB camera (default) |
| 1280 | Very Slow | Highest | Offline analysis only |

**Tuning Confidence:**
- **Lower (0.3-0.4):** More detections, more false positives
- **Default (0.5):** Balanced
- **Higher (0.6-0.7):** Fewer detections, fewer false positives

### Frame Skip Strategy

```python
Config.YOLO_FRAME_SKIP = 3  # Process every 3rd frame
```

**Why skip?**
- YOLO inference (40ms) slower than frame rate (60 FPS = 16ms)
- Objects don't move significantly in 3 frames (~50ms)
- Frees GPU for depth estimation

**Guidelines:**
- **YOLO_FRAME_SKIP = 1:** No skip (process every frame)
- **YOLO_FRAME_SKIP = 3:** Balanced (default)
- **YOLO_FRAME_SKIP = 5:** Aggressive skip (faster, less responsive)

### Model Selection

```python
Config.YOLO_MODEL = "checkpoints/yolo12n.pt"
Config.USE_TENSORRT = True  # Auto-detect .engine file
```

**YOLO Model Variants:**
- **yolo12n (nano):** Fastest, lowest accuracy (default)
- **yolo12s (small):** 2x slower, better accuracy
- **yolo12m (medium):** 4x slower, even better accuracy

**TensorRT Export:**
```bash
# Convert PyTorch → TensorRT
yolo export model=yolo12n.pt format=engine half=True device=0
```

---

## 5. Depth Estimation

### Backend Selection

```python
Config.DEPTH_BACKEND = "depth_anything_v2"  # or "midas"
Config.DEPTH_ANYTHING_MODEL = "Small"  # Small, Base, Large
```

**Comparison:**

| Backend | Model | Latency | Accuracy | Use Case |
|---------|-------|---------|----------|----------|
| MiDaS | Small | 50ms | Medium | Legacy support |
| Depth-Anything v2 | Small | 27ms | High | Default (best) |
| Depth-Anything v2 | Base | 60ms | Higher | High-accuracy mode |
| Depth-Anything v2 | Large | 120ms | Highest | Offline only |

### Depth Processing

```python
Config.DEPTH_ENABLED = True
Config.DEPTH_FRAME_SKIP = 12  # Every 12th frame
Config.DEPTH_INPUT_SIZE = 384  # Resolution
```

**Frame Skip Rationale:**
- Depth estimation slower than YOLO
- Depth changes slowly (objects don't teleport)
- 12 frames @ 60 FPS = 200ms update rate (acceptable)

**Distance Thresholds:**
```python
Config.DEPTH_CLOSE_THRESHOLD = 0.7   # > 0.7 = "close" (<2m)
Config.DEPTH_MEDIUM_THRESHOLD = 0.4  # > 0.4 = "medium" (2-5m)
# < 0.4 = "far" (>5m)
```

**Tuning:**
- **Indoor:** Lower thresholds (0.6, 0.3) - shorter distances
- **Outdoor:** Higher thresholds (0.8, 0.5) - longer distances

### Distance Estimation Method

```python
Config.DISTANCE_METHOD = "depth_only"  # or "area_only", "hybrid"
```

**Options:**
1. **depth_only:** Use depth map exclusively (most accurate)
2. **area_only:** Estimate from bbox size (fallback if depth unavailable)
3. **hybrid:** Combine both methods (experimental)

---

## 6. Audio System

### TTS Configuration

```python
Config.TTS_RATE = 190  # Words per minute
Config.AUDIO_COOLDOWN = 3.0  # Seconds between commands
Config.AUDIO_QUEUE_SIZE = 12  # Command buffer size
```

**Tuning TTS Rate:**
- **Slower (150 WPM):** Clearer for new users
- **Default (190 WPM):** Balanced
- **Faster (220 WPM):** More responsive, harder to understand

### Cooldown Management

```python
# Global cooldown (any command → any command)
Config.AUDIO_GLOBAL_COOLDOWN = 0.3  # seconds

# Per-command cooldown
Config.CRITICAL_COOLDOWN_WALKING = 1.0  # Critical alerts
Config.CRITICAL_COOLDOWN_STATIONARY = 2.0  # When user stationary
Config.NORMAL_COOLDOWN = 2.5  # Normal obstacles
```

**Why cooldowns?**
- Without: 18 FPS × detections = audio spam
- With: Max 0.5-1.0 commands/sec = digestible

### Priority System

**Critical Priority (High urgency):**
```python
Config.CRITICAL_ALLOWED_CLASSES = {
    "person", "car", "truck", "bus", "bicycle", "motorcycle"
}
Config.CRITICAL_DISTANCE_WALKING = {"very_close", "close"}
Config.CRITICAL_DISTANCE_STATIONARY = {"very_close"}
```

**Normal Priority (Obstacles):**
```python
Config.NORMAL_ALLOWED_CLASSES = {
    "chair", "table", "bottle", "door", "laptop", "couch", "bed"
}
Config.NORMAL_DISTANCE = {"close", "medium"}
Config.NORMAL_PERSISTENCE_FRAMES = 2  # Must persist 2 frames
```

**Tuning Priorities:**
- Add classes: Extend `CRITICAL_ALLOWED_CLASSES`
- Change thresholds: Adjust distance requirements
- Filter flickering: Increase `NORMAL_PERSISTENCE_FRAMES`

### Spatial Audio Beeps

```python
Config.AUDIO_SPATIAL_BEEPS_ENABLED = True

# Critical alerts (high-pitched, long)
Config.BEEP_CRITICAL_FREQUENCY = 1000  # Hz
Config.BEEP_CRITICAL_DURATION = 0.3  # seconds

# Normal alerts (low-pitched, short)
Config.BEEP_NORMAL_FREQUENCY = 500  # Hz
Config.BEEP_NORMAL_DURATION = 0.1  # seconds
Config.BEEP_NORMAL_COUNT = 2  # Two beeps
```

**Why beeps?**
- Faster than TTS (instant feedback)
- Directional cues (left/right panning)
- Attention-grabbing for critical alerts

---

## 7. Spatial Processing

### Zone System

```python
Config.ZONE_SYSTEM = "five_zones"  # or "four_quadrants"
```

**Five Zones:**
```
┌─────────────────────────────────┐
│ TOP   │   TOP      │  TOP       │
│ LEFT  │   CENTER   │  RIGHT     │
├───────┼────────────┼────────────┤
│       │            │            │
│ LEFT  │   CENTER   │  RIGHT     │
│       │   (FOCUS)  │            │
├───────┼────────────┼────────────┤
│ BOT   │   BOT      │  BOT       │
│ LEFT  │   CENTER   │  RIGHT     │
└───────┴────────────┴────────────┘
```

**Configuration:**
```python
Config.ZONE_LEFT_BOUNDARY = 0.33
Config.ZONE_RIGHT_BOUNDARY = 0.67
Config.CENTER_ZONE_WIDTH_RATIO = 0.4  # 40% of frame width
Config.CENTER_ZONE_PRIORITY_BOOST = 1.5  # 1.5x priority for center
```

**Tuning:**
- **Narrow center (0.3):** Focus on straight ahead only
- **Wide center (0.5):** More permissive, announce more objects
- **Priority boost (1.5-2.0):** Emphasize center zone

### Distance Buckets

```python
# Area-based estimation (fallback)
Config.DISTANCE_VERY_CLOSE = 0.10  # Bbox > 10% of frame
Config.DISTANCE_CLOSE = 0.04       # Bbox > 4% of frame
Config.DISTANCE_MEDIUM = 0.015     # Bbox > 1.5% of frame
```

**Guidelines:**
- **Indoor:** Lower thresholds (smaller rooms)
- **Outdoor:** Higher thresholds (larger spaces)

---

## 8. Performance Tuning

### Multiprocessing (Phase 2)

```python
Config.PHASE2_MULTIPROC_ENABLED = True  # Best: 16.6 FPS
Config.PHASE2_QUEUE_MAXSIZE = 1  # Minimize latency
Config.PHASE2_BACKPRESSURE_TIMEOUT = 0.1  # Fast timeout
```

**Trade-offs:**
- **Enabled:** +60% FPS (10.9 → 16.6), adds IPC overhead
- **Disabled:** Sequential processing, easier debugging

**Queue Sizing:**
- **Maxsize = 1:** Minimal latency (default)
- **Maxsize = 3:** More buffering, higher throughput

### Shared Memory (Phase 3)

```python
Config.USE_SHARED_MEMORY = False  # DISABLED (race conditions)
```

**Status:** Experimental, causes 19s spikes and 36% FPS drop
**Don't enable unless debugging**

### Double Buffering (Phase 7)

```python
Config.PHASE7_DOUBLE_BUFFERING = False  # Keep disabled
```

**Status:** Doesn't solve IPC overhead, adds complexity
**Don't enable**

### Input Resizing

```python
Config.INPUT_RESIZE_ENABLED = True
Config.INPUT_RESIZE_WIDTH = 896
Config.INPUT_RESIZE_HEIGHT = 896
```

**Impact:**
- Reduces IPC overhead (smaller frames to transfer)
- Slight accuracy loss (less resolution)
- +10% FPS improvement

**Tuning:**
- **1024×1024:** Better accuracy, slower
- **896×896:** Balanced (default)
- **768×768:** Faster, lower accuracy

### Peripheral Vision

```python
Config.PERIPHERAL_VISION_ENABLED = True
Config.SLAM_FRAME_SKIP = 4  # Process every 4th SLAM frame
```

**Impact:**
- **Enabled:** +15% awareness, -10% FPS
- **Disabled:** +10% FPS, no lateral detection

**When to disable:**
- Debugging RGB pipeline
- Need maximum FPS
- Indoor-only usage (less lateral threats)

---

## 9. Advanced Options

### Image Enhancement

```python
Config.LOW_LIGHT_ENHANCEMENT = True
Config.GAMMA_CORRECTION = 1.1  # 1.0-1.8
Config.AUTO_ENHANCEMENT = True
Config.LOW_LIGHT_THRESHOLD = 120.0  # 0-255
```

**CLAHE Settings:**
```python
Config.CLAHE_CLIP_LIMIT = 3.0  # Contrast strength
Config.CLAHE_TILE_SIZE = (8, 8)  # Grid size
```

**Impact:** +5-10% detection accuracy in low light

### Aria Streaming

```python
Config.STREAMING_PROFILE = "profile28"  # 30 FPS
Config.STREAMING_INTERFACE = "wifi"  # or "usb"
Config.STREAMING_WIFI_DEVICE_IP = "192.168.0.204"
```

**Profiles:**
- **profile15:** WiFi optimized (lower bandwidth)
- **profile28:** USB optimized (higher quality)

### Profiling & Debugging

```python
Config.PROFILE_PIPELINE = True
Config.PROFILE_WINDOW_FRAMES = 30
Config.DEBUG_FRAME_INTERVAL = 100
Config.ENHANCEMENT_DEBUG = True
```

**Outputs:**
- Frame-level timing in telemetry
- Periodic performance summaries
- Debug logs for enhancement decisions

---

## 10. Configuration Examples

### Example 1: Indoor Navigation (High Accuracy)

```python
# config.py overrides
Config.YOLO_RGB_IMAGE_SIZE = 640
Config.YOLO_FRAME_SKIP = 2
Config.DEPTH_FRAME_SKIP = 8
Config.DEPTH_CLOSE_THRESHOLD = 0.6  # Lower for indoor
Config.PERIPHERAL_VISION_ENABLED = True
Config.AUDIO_COOLDOWN = 2.0  # More frequent updates
Config.NORMAL_ALLOWED_CLASSES.add("stairs")  # Add indoor hazards
```

**Result:** 12-15 FPS, high accuracy, frequent audio updates

### Example 2: Outdoor Navigation (Balanced)

```python
Config.YOLO_RGB_IMAGE_SIZE = 640
Config.YOLO_FRAME_SKIP = 3
Config.DEPTH_FRAME_SKIP = 12
Config.DEPTH_CLOSE_THRESHOLD = 0.7
Config.CRITICAL_DISTANCE_WALKING = {"very_close", "close", "medium"}
Config.AUDIO_COOLDOWN = 3.0
```

**Result:** 18-22 FPS (default), balanced performance

### Example 3: Demo Mode (Maximum FPS)

```python
Config.YOLO_RGB_IMAGE_SIZE = 416
Config.YOLO_FRAME_SKIP = 5
Config.DEPTH_ENABLED = False  # Disable depth
Config.PERIPHERAL_VISION_ENABLED = False
Config.LOW_LIGHT_ENHANCEMENT = False
Config.AUDIO_COOLDOWN = 1.0
```

**Result:** 25-30 FPS, reduced accuracy, impressive demo

### Example 4: Research/Analysis (Offline)

```python
Config.YOLO_RGB_IMAGE_SIZE = 1280
Config.YOLO_FRAME_SKIP = 1  # Process all
Config.DEPTH_FRAME_SKIP = 1  # Process all
Config.DEPTH_ANYTHING_MODEL = "Large"
Config.YOLO_RGB_CONFIDENCE = 0.3  # Lower threshold
Config.PROFILE_PIPELINE = True
```

**Result:** 2-4 FPS, maximum accuracy, detailed profiling

---

## 11. Configuration File Template

Create `config_override.py` for custom settings:

```python
# config_override.py
from utils.config import Config

# My custom configuration
Config.YOLO_FRAME_SKIP = 2
Config.DEPTH_FRAME_SKIP = 10
Config.AUDIO_COOLDOWN = 2.5
Config.PERIPHERAL_VISION_ENABLED = True

# Indoor navigation
Config.DEPTH_CLOSE_THRESHOLD = 0.6
Config.DEPTH_MEDIUM_THRESHOLD = 0.35

# Audio preferences
Config.TTS_RATE = 200
Config.AUDIO_SPATIAL_BEEPS_ENABLED = True

# Debug mode
Config.PROFILE_PIPELINE = True
Config.ENHANCEMENT_DEBUG = True
```

**Usage:**
```python
# main.py
from utils.config import Config
import config_override  # Apply overrides
```

---

## 12. Configuration Best Practices

### 1. Start with Defaults

Don't change settings unless you have a specific reason. The defaults are tuned for balanced performance.

### 2. Change One Thing at a Time

When tuning, adjust one parameter, test, then adjust another. This helps isolate cause and effect.

### 3. Profile First

Use `Config.PROFILE_PIPELINE = True` to identify bottlenecks before optimizing.

### 4. Document Changes

Comment why you changed a setting:
```python
Config.YOLO_FRAME_SKIP = 5  # Increased for demo (need 30 FPS)
```

### 5. Test Edge Cases

- Low light conditions
- Crowded scenes (many objects)
- Fast motion (tracking stress test)
- SLAM camera occlusion

### 6. Monitor Telemetry

Check `logs/session_*/telemetry/summary.json` for:
- Average FPS
- Latency distribution
- Detection counts
- Audio events frequency

---

## 13. Troubleshooting

### Problem: Low FPS (<10 FPS)

**Check:**
1. GPU utilization: `nvidia-smi`
2. Frame skip settings: `YOLO_FRAME_SKIP`, `DEPTH_FRAME_SKIP`
3. TensorRT enabled: `USE_TENSORRT = True`
4. Multiprocessing: `PHASE2_MULTIPROC_ENABLED = True`

**Solution:**
```python
Config.YOLO_FRAME_SKIP = 5
Config.DEPTH_FRAME_SKIP = 15
Config.PERIPHERAL_VISION_ENABLED = False
```

### Problem: Audio Spam

**Check:**
1. Cooldown settings: `AUDIO_COOLDOWN`, `AUDIO_GLOBAL_COOLDOWN`
2. Persistence: `NORMAL_PERSISTENCE_FRAMES`

**Solution:**
```python
Config.AUDIO_COOLDOWN = 3.0  # Increase
Config.NORMAL_PERSISTENCE_FRAMES = 3  # More frames
Config.CRITICAL_COOLDOWN_WALKING = 1.5  # Increase
```

### Problem: Missing Detections

**Check:**
1. Confidence threshold: `YOLO_RGB_CONFIDENCE`
2. Frame skip: `YOLO_FRAME_SKIP`
3. Image size: `YOLO_RGB_IMAGE_SIZE`

**Solution:**
```python
Config.YOLO_RGB_CONFIDENCE = 0.4  # Lower
Config.YOLO_FRAME_SKIP = 2  # Less skipping
Config.YOLO_RGB_IMAGE_SIZE = 640  # Higher resolution
```

### Problem: CUDA OOM

**Check:**
1. Image sizes: `YOLO_RGB_IMAGE_SIZE`, `DEPTH_INPUT_SIZE`
2. Batch size (should be 1)
3. Multiple models loaded

**Solution:**
```python
Config.YOLO_RGB_IMAGE_SIZE = 416  # Reduce
Config.DEPTH_INPUT_SIZE = 256  # Reduce
Config.PERIPHERAL_VISION_ENABLED = False  # Disable
```

---

## 14. Performance Tuning Checklist

- [ ] Profile pipeline: Enable `PROFILE_PIPELINE`
- [ ] Check GPU usage: `nvidia-smi dmon`
- [ ] Verify TensorRT: Models should be `.engine` files
- [ ] Tune frame skip: Balance FPS vs responsiveness
- [ ] Test audio cooldown: Ensure no spam
- [ ] Validate detections: Check confidence thresholds
- [ ] Monitor telemetry: Review `summary.json`
- [ ] Test edge cases: Low light, crowds, fast motion

---

## Conclusion

The configuration system provides fine-grained control over all aspects of the Aria Navigation System. Start with defaults, profile to find bottlenecks, then tune incrementally. Remember:

- **Performance vs Accuracy:** Always a trade-off
- **User Experience:** Audio quality > FPS
- **Safety:** Better to miss 1 frame than give wrong info

For detailed metrics and optimization strategies, see [DEEP_DIVE.md](../architecture/DEEP_DIVE.md) and [DATA_FLOW.md](../architecture/DATA_FLOW.md).

---

**Last Updated:** November 25, 2025
**Author:** Roberto Rojas Sahuquillo
