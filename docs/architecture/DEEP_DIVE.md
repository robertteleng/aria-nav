# 🧠 Deep Dive: Aria Navigation System Architecture

> **Comprehensive technical analysis of the navigation system for visually impaired users**
> Last updated: November 25, 2025

## 📚 Table of Contents

1. [System Overview](#system-overview)
2. [Core Architecture](#core-architecture)
3. [Data Flow](#data-flow)
4. [Component Deep Dive](#component-deep-dive)
5. [Performance Optimization](#performance-optimization)
6. [Design Decisions](#design-decisions)

---

## 1. System Overview

### 1.1 Purpose

The Aria Navigation System is a real-time assistive navigation solution for visually impaired users. It combines:
- **Computer Vision** (YOLO object detection + Depth estimation)
- **Spatial Audio** (TTS with directional cues)
- **IMU Sensors** (Motion state detection)
- **Real-time Processing** (18-22 FPS on RTX 2060)

### 1.2 Key Metrics

| Metric | Value | Context |
|--------|-------|---------|
| **FPS** | 18-22 fps | Real-time navigation |
| **YOLO Latency** | ~40ms | TensorRT FP16 optimized |
| **Depth Latency** | ~27ms | ONNX Runtime CUDA |
| **End-to-End** | ~48ms | From capture to decision |
| **GPU Memory** | 1.5GB / 6GB | 75% headroom available |
| **Code Size** | 13,397 LOC | Python |

### 1.3 Hardware Stack

```
┌────────────────────────────────────────────┐
│         META ARIA GLASSES                  │
├────────────────────────────────────────────┤
│ • RGB Camera: 640x480 @ 60fps             │
│ • SLAM Cameras (L/R): 640x480 @ 30fps     │
│ • IMU: 1000Hz sampling                     │
│ • Speaker: Spatial audio output            │
└────────────────────────────────────────────┘
           ↓ USB-C / WiFi
┌────────────────────────────────────────────┐
│      COMPUTE UNIT (Intel NUC + RTX 2060)   │
├────────────────────────────────────────────┤
│ • CPU: Intel Core i7 (8 cores)            │
│ • GPU: NVIDIA RTX 2060 (6GB VRAM)         │
│ • RAM: 32GB DDR4                           │
│ • OS: Ubuntu 22.04 LTS                     │
└────────────────────────────────────────────┘
```

---

## 2. Core Architecture

### 2.1 Architectural Pattern: Separated Concerns

The system follows a **clean separation of concerns** architecture:

```
main.py
  ├── Observer (Hardware I/O)
  ├── Coordinator (Navigation Pipeline)
  └── PresentationManager (UI/Dashboards)
```

**Why this pattern?**
- **Testability**: Each component can be tested independently
- **Mock Support**: MockObserver enables development without hardware
- **Maintainability**: Changes in one layer don't affect others
- **Scalability**: Easy to add new cameras, sensors, or output modes

### 2.2 Component Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│                    MAIN.PY (Entry)                      │
└──────────────┬────────────────────────────┬─────────────┘
               ▼                            ▼
    ┌──────────────────┐        ┌──────────────────────┐
    │    OBSERVER      │        │   COORDINATOR        │
    │  (Hardware I/O)  │        │ (Navigation Logic)   │
    └──────────────────┘        └──────────────────────┘
               │                            │
        ┌──────┴──────┐            ┌───────┴────────┐
        ▼             ▼            ▼                ▼
   ┌────────┐   ┌─────────┐  ┌─────────┐    ┌──────────┐
   │  RGB   │   │  SLAM   │  │Pipeline │    │  Audio   │
   │ Stream │   │ Stream  │  │         │    │  Router  │
   └────────┘   └─────────┘  └─────────┘    └──────────┘
```

### 2.3 Directory Structure Explained

```
src/
├── main.py                    # Entry point with mode selection
├── core/                      # Core system components
│   ├── observer.py           # Real Aria hardware interface
│   ├── mock_observer.py      # Mock for development
│   ├── audio/                # Audio system
│   │   ├── audio_system.py          # TTS engine wrapper
│   │   └── navigation_audio_router.py  # Audio routing logic
│   ├── hardware/             # Device management
│   │   └── device_manager.py       # Device init/cleanup
│   ├── imu/                  # Motion detection
│   │   └── motion_detector.py      # Walking/stationary states
│   ├── navigation/           # Navigation pipeline
│   │   ├── coordinator.py          # Main pipeline orchestrator
│   │   ├── navigation_pipeline.py  # RGB processing pipeline
│   │   ├── navigation_decision_engine.py  # Decision logic
│   │   ├── builder.py              # Component factory
│   │   ├── rgb_audio_router.py     # RGB audio routing
│   │   └── slam_audio_router.py    # SLAM audio routing
│   ├── processing/           # Multiprocessing workers
│   │   ├── central_worker.py       # RGB worker
│   │   ├── slam_worker.py          # SLAM worker
│   │   └── shared_memory_manager.py # Zero-copy (experimental)
│   ├── telemetry/            # Metrics & logging
│   │   └── loggers/
│   │       ├── telemetry_logger.py     # Main telemetry
│   │       └── depth_logger.py         # Depth-specific logs
│   └── vision/               # Computer vision
│       ├── yolo_processor.py       # YOLO detection
│       ├── depth_estimator.py      # Depth-Anything v2
│       ├── image_enhancer.py       # Preprocessing
│       ├── object_tracker.py       # Temporal tracking
│       ├── slam_detection_worker.py # SLAM detection
│       ├── detected_object.py      # Detection data class
│       └── gpu_utils.py            # CUDA utilities
├── presentation/             # User interface
│   ├── presentation_manager.py   # UI orchestrator
│   ├── dashboards/
│   │   ├── opencv_dashboard.py   # OpenCV visualization
│   │   ├── rerun_dashboard.py    # Rerun 3D viz
│   │   └── web_dashboard.py      # Web-based UI
│   └── renderers/
│       └── frame_renderer.py     # Frame drawing utils
├── utils/                    # Utilities
│   ├── config.py                 # Configuration management
│   ├── ctrl_handler.py           # Ctrl+C handling
│   ├── memory_profiler.py        # Memory monitoring
│   ├── profiler.py               # Performance profiling
│   ├── resource_monitor.py       # Resource tracking
│   └── system_monitor.py         # System health
└── external/                 # External dependencies
    └── depth_anything_v2/        # Depth estimation model
```

---

## 3. Data Flow

### 3.1 Main Loop Flow

```
┌──────────────┐
│  Aria SDK    │ Streaming @ 60fps (RGB) / 30fps (SLAM)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Observer   │ Frame acquisition + undistortion
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Coordinator │ Route to pipelines
└──────┬───────┘
       │
   ┌───┴────┐
   ▼        ▼
┌─────┐  ┌──────┐
│ RGB │  │ SLAM │ Parallel processing
│ Pipe│  │ Pipe │
└─────┘  └──────┘
   │        │
   └────┬───┘
        ▼
┌───────────────┐
│ Decision      │ Prioritization
│ Engine        │
└────────┬──────┘
         │
         ▼
┌───────────────┐
│ Audio Router  │ TTS queue
└────────┬──────┘
         │
         ▼
┌───────────────┐
│ Audio System  │ Speaker output
└───────────────┘
```

### 3.2 RGB Pipeline (Detailed)

```python
# Frame N arrives from RGB camera (60 FPS)
Frame(rgb, timestamp)
    ↓
[1] Image Enhancement (~2ms)
    • Brightness adjustment
    • Contrast normalization
    • Gamma correction
    ↓
[2] YOLO Detection (~40ms with TensorRT)
    • Input: 640x640 RGB
    • Model: YOLOv12n TensorRT FP16
    • Output: [class_id, confidence, bbox, ...]
    ↓
[3] Depth Estimation (~27ms with ONNX CUDA, every 12th frame)
    • Model: Depth-Anything v2 Small
    • Input: 518x518 RGB
    • Output: Depth map (inverse depth)
    ↓
[4] Depth-Detection Fusion (~1ms)
    • Map bbox to depth values
    • Calculate mean depth in bbox
    • Classify: close (<2m) / medium (2-5m) / far (>5m)
    ↓
[5] Object Tracking (~2ms)
    • Temporal consistency
    • ID assignment
    • Filter flickering
    ↓
[6] Spatial Classification (~1ms)
    • Zone: left / center / right
    • Based on bbox center x-coordinate
    ↓
[7] Decision Engine (~3ms)
    • Priority calculation
    • Threat assessment
    • Command generation
    ↓
[8] Audio Routing (~2ms)
    • Cooldown check (2s default)
    • Queue management
    • TTS trigger
```

**Total Latency:** ~78ms (with depth) or ~51ms (without depth)
**Frame Skip:** YOLO every 3rd frame, Depth every 12th frame
**Effective FPS:** 18-22 FPS

### 3.3 SLAM Pipeline (Peripheral Vision)

```python
# Frame N arrives from SLAM cameras (30 FPS)
Frame(slam_left, slam_right, timestamp)
    ↓
[1] Rectification (~5ms)
    • Fisheye undistortion using Aria SDK calibration
    ↓
[2] YOLO Detection (both cameras) (~40ms each, parallel)
    • Reuse RGB YOLO model
    • Same TensorRT optimization
    ↓
[3] Lateral Threat Detection (~2ms)
    • Focus on "close" + "person"/"car"/"bicycle"
    • No depth estimation (inferred from bbox size)
    ↓
[4] Audio Command (~2ms)
    • "Caution left" / "Caution right"
    • Lower priority than RGB
```

**Purpose:** Peripheral awareness for side obstacles

---

## 4. Component Deep Dive

### 4.1 Observer Pattern

**File:** `src/core/observer.py`

**Responsibility:** Abstract hardware interface

```python
class Observer(ABC):
    @abstractmethod
    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Return latest RGB frame or None"""
        pass

    @abstractmethod
    def get_slam_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return (left_frame, right_frame) or (None, None)"""
        pass

    @abstractmethod
    def get_imu_data(self) -> Dict[str, Any]:
        """Return IMU readings"""
        pass
```

**Implementations:**
1. **AriaObserver** (`observer.py`): Real hardware via Project Aria SDK
2. **MockObserver** (`mock_observer.py`): Synthetic data for testing

**Why this pattern?**
- Enables **unit testing** without hardware
- Supports **CI/CD pipelines** (no physical device needed)
- Facilitates **multi-platform development** (Mac, Linux, Windows)

### 4.2 Coordinator: Pipeline Orchestrator

**File:** `src/core/navigation/coordinator.py`

**Key Responsibilities:**
1. Initialize vision models (YOLO, Depth)
2. Create RGB and SLAM pipelines
3. Manage multiprocessing workers (Phase 6 disabled by default)
4. Route frames from Observer to pipelines
5. Collect detections and pass to Decision Engine
6. Manage graceful shutdown

**Main Loop:**

```python
def run(self):
    """Main coordination loop"""
    while not self.stop_event.is_set():
        # 1. Get frames from Observer
        rgb_frame = self.observer.get_rgb_frame()
        slam_left, slam_right = self.observer.get_slam_frames()

        # 2. Process RGB
        if rgb_frame is not None:
            detections_rgb = self.rgb_pipeline.process_frame(
                rgb_frame,
                frame_number=self.frame_count
            )
            self.frame_count += 1

        # 3. Process SLAM (if enabled)
        if Config.PERIPHERAL_VISION_ENABLED:
            detections_slam = self.slam_pipeline.process_frames(...)

        # 4. Decision Engine
        self.decision_engine.process_detections(
            rgb_detections=detections_rgb,
            slam_detections=detections_slam
        )

        # 5. Update dashboards
        self.presentation_manager.update(...)
```

### 4.3 Navigation Pipeline: RGB Processing

**File:** `src/core/navigation/navigation_pipeline.py`

**Architecture:** Modular stage-based pipeline

```python
class NavigationPipeline:
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.yolo = YOLOProcessor(model="yolo12n", device="cuda:0")
        self.depth = DepthEstimator(model="small", device="cuda:0")
        self.tracker = ObjectTracker()

    def process_frame(self, frame, frame_number):
        # Stage 1: Enhancement
        enhanced = self.enhancer.enhance(frame)

        # Stage 2: YOLO (every 3rd frame)
        if frame_number % Config.YOLO_FRAME_SKIP == 0:
            detections = self.yolo.detect(enhanced)
        else:
            detections = []  # Reuse previous

        # Stage 3: Depth (every 12th frame)
        if frame_number % Config.DEPTH_FRAME_SKIP == 0:
            depth_map = self.depth.estimate(enhanced)
            self._fuse_depth(detections, depth_map)

        # Stage 4: Tracking
        tracked = self.tracker.update(detections)

        # Stage 5: Spatial classification
        for det in tracked:
            det.zone = self._classify_zone(det.bbox)

        return tracked
```

**Frame Skip Strategy:**
- **YOLO:** Every 3rd frame (effective 20 FPS from 60 FPS input)
- **Depth:** Every 12th frame (effective 5 FPS)
- **Why?** Balance between responsiveness and throughput

### 4.4 YOLO Processor: TensorRT Optimization

**File:** `src/core/vision/yolo_processor.py`

**Key Optimizations:**

1. **TensorRT Export:**
```python
# Export PyTorch → TensorRT FP16
model.export(
    format="engine",
    half=True,  # FP16 precision
    workspace=4,  # GB
    device=0
)
```

**Result:** 40ms latency (vs 120ms PyTorch)

2. **Batch Size = 1:**
   - Optimized for real-time single-frame inference
   - Minimizes latency

3. **Input Size = 640x640:**
   - Balance between accuracy and speed
   - YOLO12n (nano) variant

4. **Confidence Threshold:**
   - Default: 0.5 (adjustable)
   - Trade-off: lower = more false positives, higher = miss objects

**Classes Detected:** 80 COCO classes (person, car, bicycle, etc.)

### 4.5 Depth Estimator: Depth-Anything v2

**File:** `src/core/vision/depth_estimator.py`

**Model:** Depth-Anything v2 Small (ONNX Runtime CUDA)

**Pipeline:**

```python
def estimate(self, rgb_image: np.ndarray) -> np.ndarray:
    # 1. Resize to 518x518
    resized = cv2.resize(rgb_image, (518, 518))

    # 2. Normalize [0, 255] → [0, 1]
    normalized = resized.astype(np.float32) / 255.0

    # 3. HWC → BCHW
    input_tensor = np.transpose(normalized, (2, 0, 1))[None, ...]

    # 4. ONNX inference
    depth_map = self.session.run(
        output_names=["depth"],
        input_feed={"image": input_tensor}
    )[0]

    # 5. Resize back to original
    depth_map_resized = cv2.resize(depth_map, (width, height))

    return depth_map_resized
```

**Optimization:** ONNX Runtime with CUDA Execution Provider
- **Latency:** ~27ms (vs 80ms PyTorch)
- **Model Size:** 50MB
- **Output:** Inverse depth (closer = larger values)

**Depth-Detection Fusion:**

```python
def _fuse_depth(self, detections, depth_map):
    """Map depth values to bounding boxes"""
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        # Crop depth map to bbox
        bbox_depth = depth_map[y1:y2, x1:x2]
        # Calculate mean inverse depth
        mean_depth = np.mean(bbox_depth)
        # Convert to meters (inverse depth)
        det.distance = 1.0 / (mean_depth + 1e-6)
        # Classify
        if det.distance < 2.0:
            det.distance_bucket = "close"
        elif det.distance < 5.0:
            det.distance_bucket = "medium"
        else:
            det.distance_bucket = "far"
```

### 4.6 Decision Engine: Prioritization Logic

**File:** `src/core/navigation/navigation_decision_engine.py`

**Core Algorithm:**

```python
def _calculate_priority(self, detection: DetectedObject) -> float:
    """
    Priority = f(distance, zone, class, motion_state)
    Higher priority = more urgent
    """
    priority = 0.0

    # 1. Distance factor (closer = higher)
    if detection.distance_bucket == "close":
        priority += 100
    elif detection.distance_bucket == "medium":
        priority += 50
    else:
        priority += 10

    # 2. Zone factor (center > left/right)
    if detection.zone == "center":
        priority += 30
    else:
        priority += 10

    # 3. Class factor (dangerous objects prioritized)
    if detection.class_name in ["car", "truck", "bus"]:
        priority += 40
    elif detection.class_name in ["person", "bicycle", "motorcycle"]:
        priority += 20

    # 4. Motion state (stationary = lower priority)
    if self.motion_state == "stationary":
        priority *= 0.5  # Reduce urgency if user not moving

    return priority
```

**Audio Command Generation:**

```python
def _generate_command(self, detection: DetectedObject) -> str:
    """Generate natural language audio command"""
    # Template: "{distance} {class} {direction}"
    distance = "Close" if detection.distance_bucket == "close" else ""
    class_name = detection.class_name
    direction = detection.zone  # "left", "center", "right"

    return f"{distance} {class_name} {direction}".strip()
```

**Example:** "Close person center" → TTS → User hears alert

### 4.7 Audio Router: Command Routing

**File:** `src/core/audio/navigation_audio_router.py`

**Key Features:**

1. **Cooldown Management:**
```python
def _should_send_command(self, command: str) -> bool:
    """Enforce cooldown to avoid audio spam"""
    now = time.time()
    last_time = self.last_command_time.get(command, 0)
    if now - last_time < Config.AUDIO_COOLDOWN_SECONDS:
        return False  # Too soon
    self.last_command_time[command] = now
    return True
```

2. **Priority Queue:**
   - High-priority commands can interrupt low-priority
   - FIFO within same priority level

3. **Distance Beeps:**
   - Sonar-like beeps for close objects
   - Frequency inversely proportional to distance

### 4.8 Telemetry System: Async Logging

**File:** `src/core/telemetry/loggers/telemetry_logger.py`

**Architecture:** Non-blocking async I/O

```python
class AsyncTelemetryLogger:
    def __init__(self):
        self._write_queue = queue.Queue(maxsize=2000)
        self._flush_thread = threading.Thread(
            target=self._flush_worker,
            daemon=True
        )
        self._flush_thread.start()

    def log_performance(self, fps, latency):
        """Queue write (non-blocking)"""
        self._write_queue.put(("performance", {
            "timestamp": time.time(),
            "fps": fps,
            "latency": latency
        }))

    def _flush_worker(self):
        """Background thread flushes to disk"""
        buffer = []
        while not self._shutdown_flag.is_set():
            try:
                item = self._write_queue.get(timeout=self._flush_interval)
                buffer.append(item)
                if len(buffer) >= self._buffer_size:
                    self._flush_buffer(buffer)
                    buffer.clear()
            except queue.Empty:
                if buffer:
                    self._flush_buffer(buffer)
                    buffer.clear()
```

**Benefits:**
- **No I/O blocking:** Main thread never waits for disk writes
- **Batch writes:** Reduces syscalls
- **Graceful shutdown:** atexit handler ensures final flush

**Output Structure:**
```
logs/session_YYYY-MM-DD_HH-MM-SS/
├── telemetry/
│   ├── performance.jsonl
│   ├── detections.jsonl
│   ├── audio_events.jsonl
│   └── summary.json
├── decision_engine.log
└── audio_system.log
```

---

## 5. Performance Optimization

### 5.1 Phase Evolution

| Phase | Focus | FPS | Key Change |
|-------|-------|-----|------------|
| Baseline | PyTorch models | 3.5 | N/A |
| Phase 1-2 | TensorRT + ONNX | 10.2 | Model optimization |
| Phase 3 | Shared Memory | 6.6 | Race conditions (reverted) |
| Phase 4 | Frame skip + tuning | 18.4 | Smart frame skip |
| Phase 5 | Non-blocking queues | 18.8 | Queue timeouts |
| Phase 6 | CUDA Streams (hybrid) | 19.0 | Depth || YOLO parallel |

**Current:** Phase 6 (19.0 FPS sustained)

### 5.2 CUDA Streams (Phase 6)

**Concept:** Overlap depth estimation and YOLO inference on GPU

```python
# Create CUDA streams
stream_depth = torch.cuda.Stream()
stream_yolo = torch.cuda.Stream()

# Parallel execution
with torch.cuda.stream(stream_depth):
    depth_map = depth_estimator.estimate(frame)

with torch.cuda.stream(stream_yolo):
    detections = yolo.detect(frame)

# Synchronize
torch.cuda.synchronize()
```

**Result:** +0.6 FPS improvement (3% gain)

**Limitation:** YOLO (TensorRT) and Depth (ONNX) use different frameworks, limiting parallelization

### 5.3 Memory Management

**GPU Memory Usage:**
- YOLO TensorRT: ~800MB
- Depth ONNX: ~500MB
- CUDA context: ~200MB
- **Total:** ~1.5GB / 6GB (25% utilization)

**Headroom:** 4.5GB available for future features (tracking, VLM, etc.)

### 5.4 Bottleneck Analysis

**Profiling Tools:**
- `utils/profiler.py` - Function-level timing
- `utils/memory_profiler.py` - Memory allocation tracking
- `utils/resource_monitor.py` - System-wide monitoring

**Current Bottlenecks:**
1. **YOLO inference:** 40ms (51% of latency)
2. **Depth estimation:** 27ms (35%)
3. **Image preprocessing:** 5ms (6%)
4. **Other (tracking, decision):** 6ms (8%)

**Next Optimization Targets:**
- YOLO to YOLOv11 TensorRT (potential 10ms reduction)
- Depth model quantization (INT8 vs FP16)

---

## 6. Design Decisions

### 6.1 Why Separated Architecture?

**Decision:** Split into Observer, Coordinator, PresentationManager

**Rationale:**
- **Testability:** Each component has clear boundaries
- **Mock Support:** Develop without hardware
- **Reusability:** Swap Observer implementations (real/mock/replay)
- **Maintainability:** Changes isolated to specific layers

### 6.2 Why Frame Skip?

**Decision:** YOLO every 3rd frame, Depth every 12th frame

**Rationale:**
- **Temporal consistency:** Objects don't move much in 50ms
- **Throughput:** Process more total frames per second
- **Trade-off:** Slight lag (acceptable for navigation)

**Alternative considered:** Always process every frame → 8 FPS (too slow)

### 6.3 Why TensorRT + ONNX?

**Decision:** TensorRT for YOLO, ONNX for Depth

**Rationale:**
- **TensorRT:** Best for YOLO (3x speedup)
- **ONNX:** Best for Depth-Anything (portable, CUDA EP)
- **Mixed stack:** Each model uses optimal framework

**Alternative considered:** All PyTorch → 3.5 FPS (unacceptable)

### 6.4 Why Async Telemetry?

**Decision:** Background thread for logging

**Rationale:**
- **Non-blocking:** Main loop never waits for I/O
- **Batch writes:** Reduces syscalls from 18/sec to 0.5/sec
- **Performance:** Eliminated 250ms spikes

**Alternative considered:** Synchronous logging → periodic lag spikes

### 6.5 Why SQLite for MLflow?

**Decision:** SQLite backend instead of FileStore

**Rationale:**
- **Deprecation:** FileStore deprecated in MLflow 2.0
- **Performance:** Faster queries for experiment comparison
- **Portability:** Single .db file

**Alternative considered:** Remote tracking server → unnecessary complexity

### 6.6 Why Project-Local MLflow?

**Decision:** mlruns/ in project root (not ~/mlruns)

**Rationale:**
- **Portability:** Easy to move project
- **Isolation:** No pollution of home directory
- **Version control:** Can .gitignore mlruns/ centrally

---

## 7. Future Roadmap

### 7.1 Short-Term (Q1 2025)

- [ ] Multi-language support (Spanish, English)
- [ ] Object tracking (ByteTrack)
- [ ] Fisheye undistortion optimization

### 7.2 Mid-Term (Q2 2025)

- [ ] VLM integration (Moondream2 for scene descriptions)
- [ ] Mobile companion app
- [ ] Cloud telemetry dashboard

### 7.3 Long-Term (Q3-Q4 2025)

- [ ] Multi-user deployment
- [ ] Real-world pilot study
- [ ] Publication + Open Source release

---

## 8. Development Workflow

### 8.1 Adding a New Feature

1. **Create feature branch:** `git checkout -b feature/my-feature`
2. **Write tests:** `tests/test_my_feature.py`
3. **Implement:** `src/core/my_module/my_feature.py`
4. **Document:** Update relevant docs in `docs/`
5. **Profile:** Use `utils/profiler.py` to measure impact
6. **Update CHANGELOG:** Add entry to `CHANGELOG.md`
7. **Create PR:** Merge to `main` via pull request

### 8.2 Debugging Guide

**Common Issues:**

1. **Low FPS:**
   - Check GPU utilization: `nvidia-smi`
   - Profile: `python src/main.py` with profiler enabled
   - Look for I/O blocking, memory leaks

2. **Audio lag:**
   - Check cooldown settings in `config.py`
   - Verify queue size: `audio_router._queue.qsize()`
   - TTS engine performance (pyttsx3 vs espeak)

3. **CUDA OOM:**
   - Reduce batch size (already 1, but check multi-frame buffers)
   - Lower depth estimation resolution (518 → 384)
   - Disable peripheral vision temporarily

**Debugging Tools:**
- `pytest tests/ -v` - Run test suite
- `python -m cProfile src/main.py` - CPU profiling
- `nvidia-smi dmon` - GPU monitoring
- MLflow UI - Compare experiments

---

## 9. References

### 9.1 Key Papers

- **YOLO:** "You Only Look Once: Unified, Real-Time Object Detection" (Redmon et al., 2016)
- **Depth-Anything:** "Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data" (Yang et al., 2024)
- **Assistive Navigation:** "Computer Vision for Blind Navigation: A Review" (Zeng & Weber, 2022)

### 9.2 External Resources

- [Project Aria SDK](https://facebookresearch.github.io/projectaria_tools/)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)

---

**Last Updated:** November 25, 2025
**Author:** Roberto Rojas Sahuquillo
**Status:** Living document (updated as system evolves)
