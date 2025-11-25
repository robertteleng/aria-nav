# 🌊 Data Flow Analysis: Frame-by-Frame Journey

> **Detailed trace of how data flows through the Aria Navigation System**
> Last updated: November 25, 2025

## Overview

This document traces a single RGB frame from capture to audio output, showing timing, transformations, and decision points.

---

## Timeline: Frame N = 180 (Example at 60 FPS, 3 seconds in)

```
Time (ms)  | Component              | Operation                      | Data Shape
-----------|------------------------|--------------------------------|------------------
0          | Aria SDK               | Capture RGB frame              | (480, 640, 3) uint8
2          | Observer               | Undistortion (fisheye)         | (480, 640, 3) uint8
4          | Coordinator            | Route to RGB pipeline          | (480, 640, 3) uint8
6          | ImageEnhancer          | Brightness/contrast adjust     | (480, 640, 3) uint8
8          | NavigationPipeline     | Check frame skip (180 % 3 == 0)| YES → run YOLO
10         | YOLOProcessor          | Resize to 640x640              | (640, 640, 3) uint8
12         | YOLOProcessor          | HWC → BCHW, normalize          | (1, 3, 640, 640) float32
14         | TensorRT Engine        | GPU inference (FP16)           | 40ms latency
54         | YOLOProcessor          | NMS post-processing            | List[Detection]
56         | NavigationPipeline     | Check depth skip (180 % 12 == 0)| YES → run Depth
58         | DepthEstimator         | Resize to 518x518              | (518, 518, 3) uint8
60         | DepthEstimator         | Normalize to [0, 1]            | (518, 518, 3) float32
62         | DepthEstimator         | HWC → BCHW                     | (1, 3, 518, 518)
64         | ONNX Runtime (CUDA)    | GPU inference                  | 27ms latency
91         | DepthEstimator         | Resize depth map back          | (480, 640) float32
93         | NavigationPipeline     | Fuse depth with detections     | List[Detection+Depth]
95         | ObjectTracker          | Update tracks                  | List[TrackedObject]
97         | NavigationPipeline     | Classify zones (L/C/R)         | List[TrackedObject+Zone]
99         | DecisionEngine         | Calculate priorities           | List[Priority]
101        | DecisionEngine         | Generate top command           | "Close person center"
103        | AudioRouter            | Check cooldown (2s)            | PASS
105        | AudioRouter            | Enqueue TTS command            | Queue.put()
107        | AudioSystem            | TTS synthesis                  | WAV buffer
150        | AudioSystem            | Play audio to speaker          | User hears alert
```

**Total Latency:** 107ms (capture → decision)
**Audio Latency:** +43ms (TTS synthesis)
**End-to-End:** 150ms

---

## Detailed Flow Diagrams

### 1. Frame Acquisition (Observer Layer)

```
┌────────────────────────────────────────────────┐
│            ARIA GLASSES                        │
│  • RGB Camera: 640x480 @ 60fps                │
│  • Fisheye lens (180° FOV)                    │
└────────────────┬───────────────────────────────┘
                 │ USB-C / WiFi stream
                 ▼
┌────────────────────────────────────────────────┐
│          AriaObserver.get_rgb_frame()         │
├────────────────────────────────────────────────┤
│  1. Poll Aria SDK                             │
│     └─ aria_sdk.get_image_data()              │
│  2. Check if new frame available              │
│     └─ if timestamp > last_timestamp          │
│  3. Undistort fisheye (optional)              │
│     └─ cv2.undistort() using calibration      │
│  4. Rotate 90° CCW (device orientation)       │
│     └─ np.rot90(frame, -1)                    │
│  5. Return contiguous array                   │
│     └─ np.ascontiguousarray(frame)            │
└────────────────┬───────────────────────────────┘
                 │ RGB frame ready
                 ▼
         Coordinator.run()
```

**Key Transformations:**
1. **Fisheye → Rectified:** Remove lens distortion using Aria calibration
2. **Rotation:** Aria SDK returns portrait, we need landscape
3. **Contiguous Memory:** Required for YOLO (avoids copy)

**Timing:**
- SDK fetch: <1ms (already buffered)
- Undistortion: ~2ms (if enabled)
- Rotation: <1ms (view operation)

---

### 2. Image Enhancement

```
┌────────────────────────────────────────────────┐
│      ImageEnhancer.enhance(frame)             │
├────────────────────────────────────────────────┤
│  Input: (480, 640, 3) uint8 RGB               │
│                                                │
│  1. Convert to HSV color space                │
│     └─ cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) │
│                                                │
│  2. Brightness adjustment                     │
│     └─ H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2] │
│     └─ V = V * brightness_factor (1.2)        │
│                                                │
│  3. Contrast normalization                    │
│     └─ V = (V - V.mean()) * contrast + V.mean() │
│                                                │
│  4. Convert back to RGB                       │
│     └─ cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)   │
│                                                │
│  Output: (480, 640, 3) uint8 RGB (enhanced)   │
└────────────────────────────────────────────────┘
```

**Why enhance?**
- Meta Aria cameras optimized for recording, not real-time CV
- Low-light conditions common in indoor navigation
- Improves YOLO detection confidence by 5-10%

**Configuration:** `Config.ENHANCEMENT_ENABLED = True`

---

### 3. YOLO Detection (TensorRT)

```
┌────────────────────────────────────────────────┐
│        YOLOProcessor.detect(frame)             │
├────────────────────────────────────────────────┤
│  Input: (480, 640, 3) uint8 RGB               │
│                                                │
│  [Preprocessing - 2ms]                        │
│  1. Resize to square                          │
│     └─ cv2.resize(frame, (640, 640))          │
│  2. Normalize to [0, 1]                       │
│     └─ frame_float = frame.astype(float32) / 255.0 │
│  3. Transpose HWC → CHW                       │
│     └─ chw = np.transpose(hwc, (2, 0, 1))     │
│  4. Add batch dimension                       │
│     └─ bchw = chw[None, ...]  # (1, 3, 640, 640) │
│                                                │
│  [TensorRT Inference - 40ms]                  │
│  5. Copy to GPU                               │
│     └─ input_buffer = cuda.memcpy_htod(bchw)  │
│  6. Execute engine                            │
│     └─ context.execute_v2(bindings)           │
│  7. Copy outputs to CPU                       │
│     └─ boxes, scores, classes = fetch_outputs() │
│                                                │
│  [Post-processing - 3ms]                      │
│  8. Non-Maximum Suppression (NMS)             │
│     └─ Filter overlapping boxes (IoU > 0.45)  │
│  9. Confidence filtering                      │
│     └─ Keep only scores > 0.5                 │
│  10. Create DetectedObject instances          │
│     └─ [DetectedObject(...) for each box]     │
│                                                │
│  Output: List[DetectedObject]                 │
│          └─ class_id, confidence, bbox, ...   │
└────────────────────────────────────────────────┘
```

**TensorRT Optimizations:**
- **FP16 Precision:** 2x faster than FP32, minimal accuracy loss
- **Layer Fusion:** Convolution + BatchNorm + ReLU → single kernel
- **Dynamic Shapes:** Disabled (fixed 640x640 for optimal performance)

**Detection Format:**
```python
@dataclass
class DetectedObject:
    class_id: int          # COCO class (0-79)
    class_name: str        # "person", "car", etc.
    confidence: float      # 0.0 - 1.0
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    timestamp: float       # Frame capture time
    zone: str              # "left", "center", "right" (added later)
    distance: float        # meters (added by depth fusion)
    distance_bucket: str   # "close", "medium", "far"
    track_id: Optional[int] # Tracking ID (added by tracker)
```

---

### 4. Depth Estimation (ONNX)

```
┌────────────────────────────────────────────────┐
│     DepthEstimator.estimate(frame)             │
├────────────────────────────────────────────────┤
│  Input: (480, 640, 3) uint8 RGB               │
│                                                │
│  [Preprocessing - 3ms]                        │
│  1. Resize to model input size                │
│     └─ resized = cv2.resize(frame, (518, 518)) │
│  2. Normalize to [0, 1]                       │
│     └─ normalized = resized.astype(float32) / 255.0 │
│  3. Transpose HWC → CHW                       │
│     └─ chw = np.transpose(hwc, (2, 0, 1))     │
│  4. Add batch dimension                       │
│     └─ bchw = chw[None, ...]  # (1, 3, 518, 518) │
│                                                │
│  [ONNX Runtime Inference - 27ms]              │
│  5. Run session (CUDA Execution Provider)     │
│     └─ output = session.run(["depth"], {"image": bchw}) │
│  6. Squeeze batch dimension                   │
│     └─ depth_map = output[0][0]  # (518, 518)  │
│                                                │
│  [Post-processing - 2ms]                      │
│  7. Resize to original frame size             │
│     └─ depth_resized = cv2.resize(depth_map,  │
│                                    (640, 480)) │
│  8. Normalize to [0, 255] for visualization   │
│     └─ depth_vis = (depth_map / depth_map.max()) * 255 │
│                                                │
│  Output: (480, 640) float32 inverse depth     │
│          └─ Closer objects = higher values    │
└────────────────────────────────────────────────┘
```

**Model Architecture:** Depth-Anything v2 Small
- **Backbone:** DINOv2 (Vision Transformer)
- **Decoder:** Dense Prediction Transformer (DPT)
- **Training:** MiDaS-style relative depth

**Inverse Depth:**
- Model outputs inverse depth: `d_inv = 1 / d_real`
- Why? More stable gradients during training
- Convert to real depth: `d_real = 1 / d_inv`

---

### 5. Depth-Detection Fusion

```
┌────────────────────────────────────────────────┐
│  NavigationPipeline._fuse_depth(dets, depth_map) │
├────────────────────────────────────────────────┤
│  For each detection:                           │
│                                                │
│  1. Extract bounding box coordinates          │
│     └─ x1, y1, x2, y2 = det.bbox              │
│                                                │
│  2. Crop depth map to bbox region             │
│     └─ bbox_depth = depth_map[y1:y2, x1:x2]   │
│                                                │
│  3. Calculate statistics                      │
│     └─ mean_inv_depth = np.mean(bbox_depth)   │
│     └─ std_inv_depth = np.std(bbox_depth)     │
│                                                │
│  4. Convert to real depth (meters)            │
│     └─ distance = 1.0 / (mean_inv_depth + 1e-6) │
│                                                │
│  5. Classify distance bucket                  │
│     └─ if distance < 2.0: bucket = "close"    │
│        elif distance < 5.0: bucket = "medium" │
│        else: bucket = "far"                   │
│                                                │
│  6. Update detection object                   │
│     └─ det.distance = distance                │
│        det.distance_bucket = bucket           │
└────────────────────────────────────────────────┘
```

**Challenges:**
1. **Occlusion:** Depth map includes background behind transparent objects
   - **Solution:** Use median instead of mean for robust estimate
2. **Scale ambiguity:** Relative depth (not absolute)
   - **Solution:** Calibrate buckets empirically (2m, 5m thresholds)

---

### 6. Object Tracking

```
┌────────────────────────────────────────────────┐
│       ObjectTracker.update(detections)         │
├────────────────────────────────────────────────┤
│  State: self.tracks = {track_id: Track(...)}  │
│                                                │
│  1. Match current detections to existing tracks│
│     └─ For each detection:                    │
│         • Calculate IoU with all tracks       │
│         • If IoU > 0.3: assign to track       │
│         • Else: create new track              │
│                                                │
│  2. Update matched tracks                     │
│     └─ track.update(detection)                │
│        • Kalman filter prediction (optional)  │
│        • Update bbox, confidence              │
│        • Increment hit counter                │
│                                                │
│  3. Handle unmatched tracks                   │
│     └─ track.miss_count += 1                  │
│        if track.miss_count > 5:               │
│            delete track (object left frame)   │
│                                                │
│  4. Handle new detections                     │
│     └─ Create Track(id=next_id, ...)          │
│        self.tracks[next_id] = track           │
│        next_id += 1                            │
│                                                │
│  5. Return tracked detections                 │
│     └─ [det with track_id assigned]           │
└────────────────────────────────────────────────┘
```

**Tracking Algorithm:** IoU-based (simple but effective)
- **IoU = Intersection over Union:** Overlap area / Total area
- **Threshold:** 0.3 (tuned empirically)

**Why track?**
- Reduce flickering (same object gets same ID across frames)
- Enable temporal reasoning ("person was here 5 frames ago")
- Future: Predict motion trajectories

---

### 7. Spatial Classification

```
┌────────────────────────────────────────────────┐
│  NavigationPipeline._classify_zone(bbox)       │
├────────────────────────────────────────────────┤
│  Input: bbox = (x1, y1, x2, y2)               │
│         frame_width = 640                     │
│                                                │
│  1. Calculate bbox center                     │
│     └─ center_x = (x1 + x2) / 2               │
│                                                │
│  2. Define zone boundaries                    │
│     └─ left_boundary = frame_width * 0.33     │
│        right_boundary = frame_width * 0.66    │
│                                                │
│  3. Classify based on center_x                │
│     └─ if center_x < left_boundary:           │
│            zone = "left"                      │
│        elif center_x > right_boundary:        │
│            zone = "right"                     │
│        else:                                  │
│            zone = "center"                    │
│                                                │
│  Output: "left" | "center" | "right"          │
└────────────────────────────────────────────────┘
```

**Zone Diagram:**

```
┌─────────────────────────────────┐
│ LEFT  │   CENTER   │  RIGHT     │
│ 0-33% │   33-66%   │  66-100%   │
├───────┼────────────┼────────────┤
│       │            │            │
│   🚶  │     🚗     │            │
│       │            │     🚲     │
│       │            │            │
└───────┴────────────┴────────────┘
```

**Why 3 zones?**
- Simple audio commands: "person left" vs "person right"
- User can turn head to center the object
- Balance: more zones = cognitive overload

---

### 8. Decision Engine (Priority Calculation)

```
┌────────────────────────────────────────────────┐
│  DecisionEngine.process_detections(rgb, slam)  │
├────────────────────────────────────────────────┤
│  1. Combine RGB + SLAM detections             │
│     └─ all_dets = rgb_dets + slam_dets        │
│                                                │
│  2. Calculate priority for each               │
│     └─ For det in all_dets:                   │
│         priority = 0                          │
│         # Distance factor                     │
│         if det.distance_bucket == "close":    │
│             priority += 100                   │
│         elif det.distance_bucket == "medium": │
│             priority += 50                    │
│         else:                                 │
│             priority += 10                    │
│                                                │
│         # Zone factor                         │
│         if det.zone == "center":              │
│             priority += 30                    │
│         else:                                 │
│             priority += 10                    │
│                                                │
│         # Class factor                        │
│         if det.class_name in DANGEROUS:       │
│             priority += 40                    │
│         elif det.class_name in MOVING:        │
│             priority += 20                    │
│                                                │
│         # Motion state                        │
│         if self.motion_state == "stationary": │
│             priority *= 0.5                   │
│                                                │
│         det.priority = priority               │
│                                                │
│  3. Sort by priority (descending)             │
│     └─ sorted_dets = sorted(all_dets,         │
│                             key=lambda d: d.priority, │
│                             reverse=True)     │
│                                                │
│  4. Generate command for top detection        │
│     └─ top_det = sorted_dets[0]               │
│        command = self._generate_command(top_det) │
│                                                │
│  5. Send to audio router                      │
│     └─ self.audio_router.route_command(command) │
└────────────────────────────────────────────────┘
```

**Priority Formula:**

```
Priority = Distance Factor + Zone Factor + Class Factor

Where:
  Distance Factor = {100 (close), 50 (medium), 10 (far)}
  Zone Factor = {30 (center), 10 (left/right)}
  Class Factor = {40 (dangerous), 20 (moving), 0 (static)}

If user stationary: Priority *= 0.5
```

**Example Scenarios:**

| Detection | Distance | Zone | Class | Priority | Reason |
|-----------|----------|------|-------|----------|--------|
| Car | close | center | car | 170 | High threat |
| Person | medium | left | person | 80 | Medium threat |
| Chair | far | right | chair | 20 | Low priority |

---

### 9. Audio Routing (Cooldown Management)

```
┌────────────────────────────────────────────────┐
│    AudioRouter.route_command(command)          │
├────────────────────────────────────────────────┤
│  State: self.last_command_time = {}           │
│         self.audio_queue = Queue()            │
│                                                │
│  1. Check if command should be sent           │
│     └─ now = time.time()                      │
│        last_time = self.last_command_time.get(command, 0) │
│        elapsed = now - last_time              │
│                                                │
│        if elapsed < AUDIO_COOLDOWN_SECONDS:   │
│            return  # Skip (too soon)          │
│                                                │
│  2. Update last command time                  │
│     └─ self.last_command_time[command] = now  │
│                                                │
│  3. Enqueue command                           │
│     └─ self.audio_queue.put(command)          │
│                                                │
│  4. Trigger audio system (separate thread)    │
│     └─ Audio thread pulls from queue          │
│        Synthesizes TTS                        │
│        Plays to speaker                       │
└────────────────────────────────────────────────┘
```

**Cooldown Rationale:**
- **Without cooldown:** 18 commands/sec → audio spam
- **With 2s cooldown:** Max 0.5 commands/sec → digestible
- **Per-command cooldown:** "car left" and "person right" can interleave

**Configuration:**
```python
Config.AUDIO_COOLDOWN_SECONDS = 2.0  # Tunable
```

---

### 10. Audio System (TTS Synthesis)

```
┌────────────────────────────────────────────────┐
│     AudioSystem._audio_worker_thread()         │
├────────────────────────────────────────────────┤
│  Runs in background daemon thread             │
│                                                │
│  while not self.stop_event.is_set():          │
│      1. Block on queue (wait for command)     │
│         └─ command = self.queue.get(timeout=1) │
│                                                │
│      2. Synthesize TTS                        │
│         └─ self.tts_engine.say(command)       │
│            self.tts_engine.runAndWait()       │
│            # Blocking call (~43ms for 3 words) │
│                                                │
│      3. Play to speaker                       │
│         └─ Audio output via system API        │
│            (ALSA on Linux, CoreAudio on macOS) │
│                                                │
│      4. Mark task done                        │
│         └─ self.queue.task_done()             │
└────────────────────────────────────────────────┘
```

**TTS Engine:** pyttsx3 (wrapper for espeak/sapi5)
- **Speed:** 150 WPM (tunable)
- **Voice:** System default (can select)
- **Latency:** ~43ms for short commands

**Alternative TTS:**
- **Google Cloud TTS:** Higher quality, requires internet
- **Coqui TTS:** Local neural TTS, higher latency
- **Current:** pyttsx3 for low-latency local TTS

---

## Performance Analysis

### Latency Breakdown (Frame 180)

| Stage | Duration (ms) | % of Total | Parallelizable? |
|-------|---------------|------------|-----------------|
| Capture + Undistort | 2 | 1.9% | No (hardware) |
| Enhancement | 2 | 1.9% | No (sequential) |
| YOLO Inference | 40 | 37.4% | Partially (CUDA streams) |
| Depth Inference | 27 | 25.2% | Partially (CUDA streams) |
| Depth Fusion | 2 | 1.9% | No (CPU) |
| Tracking | 2 | 1.9% | No (CPU) |
| Decision Engine | 3 | 2.8% | No (CPU) |
| Audio Routing | 2 | 1.9% | No (CPU) |
| **Total Pipeline** | **107** | **100%** | |
| TTS Synthesis | 43 | (Additional) | Yes (separate thread) |
| **End-to-End** | **150** | | |

### Frame Skip Impact

```
Without frame skip (every frame):
  YOLO: 60 fps * 40ms = 2400ms/sec = OVERLOAD
  Depth: 60 fps * 27ms = 1620ms/sec = OVERLOAD
  Total: 4020ms/sec → 0.25 FPS ❌

With frame skip (YOLO every 3rd, Depth every 12th):
  YOLO: 20 fps * 40ms = 800ms/sec
  Depth: 5 fps * 27ms = 135ms/sec
  Other: 60 fps * 20ms = 1200ms/sec
  Total: 2135ms/sec → 0.47 FPS (still tight)

With frame skip + pipeline optimization:
  Actual measured: 18-22 FPS ✓
  (Parallel GPU ops + efficient CPU code)
```

### CUDA Streams Benefit (Phase 6)

```
Sequential (Baseline):
┌─────────────┐
│   YOLO 40ms │───┐
└─────────────┘   │
┌─────────────┐   │ 67ms total
│   Depth 27ms│───┘
└─────────────┘

Parallel (Phase 6):
┌─────────────┐
│   YOLO 40ms │
└─────────────┘
┌─────────────┐
│   Depth 27ms│ (overlapped)
└─────────────┘
Total: max(40, 27) + sync = 42ms

Savings: 67 - 42 = 25ms (37% reduction)
```

**Actual gain:** Only ~3ms improvement
**Why?** TensorRT (YOLO) and ONNX (Depth) use different GPU contexts, limiting overlap

---

## Memory Flow

### GPU Memory (VRAM)

```
┌────────────────────────────────────────┐
│        NVIDIA RTX 2060 (6GB)           │
├────────────────────────────────────────┤
│                                        │
│  ┌──────────────────┐                 │
│  │  YOLO TensorRT   │  800 MB         │
│  │  • Weights       │                 │
│  │  • Activations   │                 │
│  └──────────────────┘                 │
│                                        │
│  ┌──────────────────┐                 │
│  │  Depth ONNX      │  500 MB         │
│  │  • Weights       │                 │
│  │  • Activations   │                 │
│  └──────────────────┘                 │
│                                        │
│  ┌──────────────────┐                 │
│  │  CUDA Context    │  200 MB         │
│  │  • Kernels       │                 │
│  │  • Buffers       │                 │
│  └──────────────────┘                 │
│                                        │
│  ┌──────────────────┐                 │
│  │  Free Memory     │  4.5 GB (75%)   │
│  └──────────────────┘                 │
│                                        │
└────────────────────────────────────────┘
```

### CPU Memory (RAM)

```
┌────────────────────────────────────────┐
│           System RAM (32GB)            │
├────────────────────────────────────────┤
│                                        │
│  ┌──────────────────┐                 │
│  │  Frame Buffers   │  ~200 MB        │
│  │  • RGB queue     │  (10 frames)    │
│  │  • SLAM queue    │  (10 frames)    │
│  └──────────────────┘                 │
│                                        │
│  ┌──────────────────┐                 │
│  │  Detection Lists │  ~50 MB         │
│  │  • Current frame │                 │
│  │  • History (100) │                 │
│  └──────────────────┘                 │
│                                        │
│  ┌──────────────────┐                 │
│  │  Telemetry Logs  │  ~100 MB        │
│  │  • JSONL buffers │                 │
│  │  • Queue         │                 │
│  └──────────────────┘                 │
│                                        │
│  ┌──────────────────┐                 │
│  │  Free Memory     │  ~30 GB         │
│  └──────────────────┘                 │
│                                        │
└────────────────────────────────────────┘
```

---

## Edge Cases & Error Handling

### 1. No Detections in Frame

```python
# Frame N: Empty street
detections = yolo.detect(frame)  # Returns []

# Pipeline continues
tracked = tracker.update([])  # No tracks to update
decision_engine.process([])   # No commands generated

# Result: Silence (no audio spam)
```

### 2. CUDA Out of Memory

```python
try:
    depth_map = depth_estimator.estimate(frame)
except RuntimeError as e:
    if "out of memory" in str(e):
        logger.warning("CUDA OOM, skipping depth for this frame")
        depth_map = None  # Graceful degradation
        # Continue without depth
```

### 3. Audio Queue Overflow

```python
# AudioRouter: Queue maxsize = 10
try:
    self.audio_queue.put(command, timeout=0.1)
except queue.Full:
    logger.warning(f"Audio queue full, dropping: {command}")
    # Drop oldest command, not current
```

### 4. Observer Frame Drop

```python
# Observer: Frame N+1 arrives before N is processed
rgb_frame = observer.get_rgb_frame()
if rgb_frame is None:
    logger.debug("Frame drop detected")
    continue  # Skip this iteration, wait for next

# Result: Graceful handling, no crash
```

---

## Debugging Tools

### 1. Timing Instrumentation

```python
# Example: Measure YOLO latency
from utils.profiler import Profiler

with Profiler("yolo_inference"):
    detections = yolo.detect(frame)

# Logs: [PROFILER] yolo_inference: 42.3ms
```

### 2. Frame-Level Telemetry

```python
# Telemetry output (performance.jsonl):
{
  "timestamp": 1700000000.123,
  "frame_number": 180,
  "fps": 19.2,
  "yolo_latency_ms": 40.1,
  "depth_latency_ms": 27.3,
  "total_latency_ms": 78.5
}
```

### 3. Detection Logging

```python
# Telemetry output (detections.jsonl):
{
  "timestamp": 1700000000.123,
  "frame_number": 180,
  "detections": [
    {
      "class_name": "person",
      "confidence": 0.89,
      "bbox": [120, 200, 250, 450],
      "distance": 3.2,
      "distance_bucket": "medium",
      "zone": "center",
      "priority": 140
    }
  ]
}
```

---

## Conclusion

This document traced a single frame through the entire Aria Navigation System, from capture to audio output. Key takeaways:

1. **Pipeline Latency:** 107ms (capture → decision)
2. **Bottlenecks:** YOLO (40ms) + Depth (27ms) dominate
3. **Frame Skip:** Essential for real-time performance
4. **Optimization:** TensorRT + ONNX provide 3-4x speedup
5. **Future:** Further gains possible with model quantization (INT8)

**Total Processing Time per Frame:** ~150ms (including TTS)
**Effective FPS:** 18-22 FPS (limited by inference, not I/O)
**User Experience:** Responsive, real-time navigation assistance

---

**Last Updated:** November 25, 2025
**Author:** Roberto Rojas Sahuquillo
