# Changelog

All notable changes to the Aria Navigation System project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned
- Hardware migration to Intel NUC + RTX 2060
- Multi-language audio support (Spanish, English)
- Mobile companion app
- Cloud telemetry dashboard
- Object tracking to reduce flicker
- FP16/INT8 quantization experiments

---

## [2.0.0] - 2025-11-20 - 🚀 Production Release

### Added
- **[FEAT]** Async telemetry logger with zero FPS impact
- **[FEAT]** Batch writing for all log streams (2s intervals)
- **[FEAT]** Session-based log organization (`logs/session_*/`)
- **[FEAT]** Multiple telemetry streams:
  - `performance.jsonl` - FPS, latency metrics
  - `detections.jsonl` - YOLO detection events
  - `audio_events.jsonl` - Audio routing decisions
  - `audio_telemetry.jsonl` - Audio system statistics
- **[DOCS]** Complete documentation reorganization (45+ files)
- **[DOCS]** PROJECT_TIMELINE.md with 10-iteration history
- **[DOCS]** Comprehensive SETUP.md (400+ lines)
- **[DOCS]** Audio configuration guide (audio_config.md, 350+ lines)
- **[DOCS]** Testing guide with pytest examples (300+ lines)

### Changed
- **[REFACTOR]** Moved from sync to async telemetry logging
- **[PERF]** Eliminated I/O spikes (250ms → 0ms)
- **[DOCS]** Reorganized docs/ into 9 categories

### Fixed
- **[FIX]** Motion state detection hysteresis (observer.py line 216)
- **[FIX]** _last_motion_state initialization
- **[FIX]** State persistence between frames
- **[FIX]** All broken documentation links in INDEX.md

### Performance
- **18.4 FPS** sustained (no degradation from logging)
- **~48ms** end-to-end latency
- Zero sync I/O blocking

---

## [1.9.0] - 2025-11-18 - 🔧 Optimization Complete

### Added
- **[FEAT]** TensorRT YOLO engine (`yolo12n.engine`)
- **[FEAT]** ONNX Runtime depth estimator with CUDA EP
- **[FEAT]** Multiprocessing for SLAM cameras
- **[FEAT]** Inter-process communication queues
- **[FEAT]** Frame buffer management
- **[TOOL]** `export_tensorrt_slam.py` - TensorRT export script
- **[DOCS]** FASE_4_FINAL_RESULTS.md with benchmarks

### Changed
- **[PERF]** YOLO resolution: 256x256 → 640x640
- **[PERF]** Depth resolution: 256x256 → 384x384
- **[REFACTOR]** Depth estimator to use ONNX Runtime
- **[REFACTOR]** SLAM detection moved to separate processes

### Fixed
- **[CRITICAL]** Depth not executing on TensorRT pipeline
  - Root cause: Invalid model validation checks
  - Solution: Added `_is_model_loaded()` method
  - File: `src/core/vision/depth_estimator.py`
- **[FIX]** ONNX model loading error handling
- **[FIX]** TensorRT engine validation

### Performance
- **18.4 FPS** (+426% from 3.5 FPS baseline)
- **YOLO:** 100ms → 40ms (2.5x speedup)
- **Depth:** 315ms → 27ms (11.7x speedup)
- **Latency:** 283ms → 48ms (83% reduction)

### Technical Details
```
Phase 1: GPU Optimizations
- cuDNN benchmark mode
- TensorFloat-32 (TF32)
- Pinned memory
Result: Minimal gains (GPU not saturated)

Phase 2: TensorRT + ONNX
- TensorRT for YOLO (FP16)
- ONNX Runtime for Depth (CUDA EP)
Result: 3.5 → 12.0 FPS (+243%)

Phase 3: Multiprocessing
- Separate processes for cameras
- Async SLAM workers
Result: 12.0 → 18.4 FPS (+53%)
```

---

## [1.8.0] - 2025-11-10 - 📊 Dashboard Suite

### Added
- **[FEAT]** OpenCV dashboard with improved HUD
- **[FEAT]** Rerun 3D visualization dashboard
- **[FEAT]** Web dashboard (Flask + WebSockets)
  - Live video streams (MJPEG)
  - Real-time metrics
  - Detection logs
  - Audio event history
  - SLAM peripheral views
- **[FEAT]** Dashboard selection at startup (`--dashboard` flag)
- **[UI]** Audio/performance HUD overlay
- **[UI]** SLAM minimap views
- **[UI]** Depth visualization mini-map

### Changed
- **[REFACTOR]** Dashboard abstraction layer
- **[IMPROVE]** OpenCV rendering performance
- **[IMPROVE]** WebSocket frame encoding (JPEG compression)

### Technical Details
- Flask server @ localhost:5000
- WebSocket updates @ 10 Hz
- MJPEG streaming for video
- Bootstrap 5 + Chart.js UI
- Rerun logging for 3D viz

---

## [1.7.0] - 2025-11-05 - 🎵 Audio Routing Refactor

### Added
- **[FEAT]** Centralized `NavigationAudioRouter`
- **[FEAT]** Per-source priority queues (RGB, SLAM1, SLAM2)
- **[FEAT]** Dynamic cooldown management
- **[FEAT]** Audio event telemetry (JSONL)
- **[FEAT]** Session summaries
- **[FEAT]** Queue overflow handling
- **[METRICS]** Audio routing metrics and monitoring

### Changed
- **[REFACTOR]** Unified audio routing architecture
  - RgbAudioRouter → NavigationAudioRouter
  - SlamAudioRouter → NavigationAudioRouter
  - IMU events → NavigationAudioRouter
- **[IMPROVE]** Source-specific cooldowns
- **[IMPROVE]** Event deduplication logic

### Removed
- **[DEPRECATED]** Separate audio routers per source

### Architecture
```
BEFORE:
RgbAudioRouter → AudioSystem
SlamAudioRouter → AudioSystem

AFTER:
RgbAudioRouter ────┐
SlamAudioRouter ───┤→ NavigationAudioRouter → AudioSystem
IMU Events ────────┘
```

---

## [1.6.0] - 2025-10-30 - 👁️ SLAM Peripheral Vision

### Added
- **[FEAT]** SLAM1 (left) camera streaming
- **[FEAT]** SLAM2 (right) camera streaming
- **[FEAT]** Asynchronous SLAM detection workers
- **[FEAT]** Lateral obstacle detection
- **[FEAT]** SLAM audio routing
- **[FEAT]** Independent cooldowns per camera
- **[FEAT]** Event deduplication
- **[CLASS]** `SlamDetectionWorker` - Background detection
- **[CLASS]** `SlamAudioRouter` - SLAM-specific routing
- **[EVENT]** `SlamDetectionEvent` - Lateral detection events

### Changed
- **[IMPROVE]** Detection priority for lateral obstacles
- **[IMPROVE]** YOLO profile for SLAM (lower resolution)

### Performance
- SLAM processing in background threads (no FPS impact)
- ~18-20 FPS sustained with 3 cameras

---

## [1.5.0] - 2025-10-25 - 🏃 Motion State Detection

### Added
- **[FEAT]** IMU data streaming (2 sensors: left/right wrists)
- **[FEAT]** Motion state detection (stationary/walking)
- **[FEAT]** Acceleration magnitude analysis
- **[FEAT]** Hysteresis for stability
- **[METHOD]** `_estimate_motion_state()` - State inference
- **[METHOD]** `_update_last_motion_state()` - State persistence

### Changed
- **[IMPROVE]** Observer IMU callbacks
- **[IMPROVE]** Motion-aware processing

### Technical Details
```python
Motion Thresholds:
- Stationary: std < 0.3 m/s²
- Walking: std > 0.6 m/s²
- Hysteresis: Maintains last valid state
```

### Performance
- Zero FPS impact (~0ms overhead)
- IMU @ 1000 Hz (downsampled to 10 Hz)

---

## [1.4.0] - 2025-10-20 - 🌙 Low-Light Enhancement

### Added
- **[FEAT]** CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **[FEAT]** Gamma correction
- **[FEAT]** Auto-enhancement mode
- **[FEAT]** Image quality metrics
- **[CLASS]** `ImageEnhancer` module
- **[METHOD]** CLAHE on LAB color space
- **[METHOD]** Adaptive gamma correction

### Changed
- **[IMPROVE]** Detection accuracy in low light (+15%)

### Performance
- Enhancement overhead: ~3-5ms per frame
- FPS: 18-20 sustained

### Technical Details
```python
CLAHE Parameters:
- Clip limit: 2.0
- Tile grid: 8x8
- Color space: LAB
```

---

## [1.3.0] - 2025-10-15 - 🌊 Depth Estimation

### Added
- **[FEAT]** Depth Anything v2 integration (ViT-Small)
- **[FEAT]** Distance-based warnings
- **[FEAT]** Depth visualization (mini-map)
- **[FEAT]** Combined YOLO + Depth pipeline
- **[CLASS]** `DepthEstimator` - Depth processing
- **[STRATEGY]** Frame skipping (YOLO: 3x, Depth: 12x)
- **[FEAT]** Depth-YOLO coordinate mapping

### Changed
- **[REFACTOR]** Pipeline to support dual models
- **[IMPROVE]** Memory management for simultaneous inference

### Performance
- Depth inference: ~120-180ms (PyTorch MPS)
- Overall FPS: 18-20 (with frame skip)
- Resolution: 256x256 → 384x384

### Technical Details
```
Model: Depth Anything v2 (ViT-Small)
Backend: PyTorch MPS (macOS)
Output: Relative depth map (normalized 0-1)
Post-processing: Gaussian blur + normalization
```

---

## [1.2.0] - 2025-10-08 - 🎵 Spatial Audio System

### Added
- **[FEAT]** Zone-based detection (left/center/right)
- **[FEAT]** macOS TTS integration (`say` command)
- **[FEAT]** Priority-based audio queue
- **[FEAT]** Cooldown mechanism (avoid spam)
- **[FEAT]** Distance-based priority
- **[CLASS]** `NavigationDecisionEngine` - Decision logic
- **[CLASS]** `AudioSystem` - TTS interface
- **[CLASS]** `RgbAudioRouter` - Audio routing

### Changed
- **[IMPROVE]** Zone assignment algorithm
- **[IMPROVE]** Audio message templates

### Architecture
```
Detections → NavigationDecisionEngine
               ↓
          Zone Assignment
               ↓
          Priority Queue
               ↓
          Audio System (TTS)
```

### Performance
- Audio overhead: ~5-10ms per message
- FPS: 20-25 with audio feedback

### Technical Details
```python
Zones:
- Left: x < 0.33
- Center: 0.33 <= x <= 0.66
- Right: x > 0.66

Cooldowns:
- Person: 3s
- Obstacle: 5s
- Lateral: 2s
```

---

## [1.1.0] - 2025-10-03 - 🚀 MVP: RGB + YOLO Detection

### Added
- **[FEAT]** Aria USB streaming @ 60 FPS
- **[FEAT]** YOLOv11n integration (CPU inference)
- **[FEAT]** Frame rotation and preprocessing
- **[FEAT]** OpenCV real-time display
- **[FEAT]** Detection confidence thresholds
- **[CLASS]** `Observer` - Aria SDK interface
- **[CLASS]** `YoloProcessor` - Detection engine
- **[CLASS]** `OpenCVDashboard` - Visualization
- **[FEAT]** Mock observer for testing

### Technical Details
```
Profile: Profile28 (60 FPS RGB streaming)
Model: YOLOv11n (fastest variant)
Backend: CPU (avoid MPS bugs)
Resolution: 640x640
Preprocessing: np.ascontiguousarray() + rotation
```

### Performance
- 20-30 FPS (lightweight, detection only)
- Detection latency: ~40ms

### Decisions Made
```
✓ Profile28 (60fps streaming)
✓ USB over WiFi (more stable)
✓ CPU inference (avoid MPS bugs)
✓ YOLOv11n (fastest variant)
✓ np.ascontiguousarray() for YOLO
✓ Rotation: np.rot90(image, -1)
✗ Undistortion (degraded detection)
```

---

## [1.0.0] - 2025-09-25 - 🎯 Initial Setup

### Added
- **[INIT]** Project structure
- **[INIT]** Python environment (venv)
- **[INIT]** Requirements.txt with dependencies:
  - `projectaria-tools`
  - `opencv-python`
  - `ultralytics` (YOLOv11)
  - `torch` (PyTorch)
  - `numpy`
- **[DOCS]** Initial README.md
- **[DOCS]** Git repository initialization

### Technical Stack
- Python 3.10+
- macOS development environment
- VS Code editor
- Git version control

---

## Legend

### Commit Types
- **[FEAT]** - New feature
- **[FIX]** - Bug fix
- **[PERF]** - Performance improvement
- **[REFACTOR]** - Code refactoring
- **[DOCS]** - Documentation
- **[TEST]** - Tests
- **[CHORE]** - Maintenance
- **[CRITICAL]** - Critical bug fix
- **[UI]** - User interface
- **[METRICS]** - Monitoring/telemetry
- **[TOOL]** - Development tool
- **[IMPROVE]** - Enhancement
- **[CLASS]** - New class
- **[METHOD]** - New method
- **[EVENT]** - New event type
- **[STRATEGY]** - Algorithm/strategy
- **[INIT]** - Initialization
- **[DEPRECATED]** - Deprecated feature

---

## Statistics

### Development Time
- **Total Duration:** ~10 weeks
- **Iterations:** 10 (8 base + 2 optimization)
- **Commits:** 100+ commits
- **Files Changed:** 80+ Python files

### Performance Journey
```
v1.1.0: 25-30 FPS (YOLO only)
v1.2.0: 20-25 FPS (+ Audio)
v1.3.0: 18-20 FPS (+ Depth)
v1.4.0: 18-20 FPS (+ Enhancement)
v1.5.0: 18-20 FPS (+ Motion)
v1.6.0: 18-20 FPS (+ SLAM)
v1.7.0: 18-20 FPS (+ Routing)
v1.8.0: 18-20 FPS (+ Dashboards)
v1.9.0: 18.4 FPS (+ TensorRT/ONNX) ← 426% improvement
v2.0.0: 18.4 FPS (+ Async telemetry)
```

### Features Added
- ✅ Real-time object detection (YOLOv11n)
- ✅ Depth estimation (Depth Anything v2)
- ✅ Spatial audio feedback (TTS)
- ✅ Motion state detection (IMU)
- ✅ Peripheral vision (SLAM cameras)
- ✅ Multiple dashboards (OpenCV/Rerun/Web)
- ✅ Production telemetry (async logging)
- ✅ Complete documentation (45+ files)

---

**Project Status:** ✅ Production Ready  
**Current Version:** 2.0.0  
**Next Target:** 3.0.0 (Hardware Migration → Intel NUC + RTX 2060)

---

*For detailed technical information, see [PROJECT_TIMELINE.md](docs/PROJECT_TIMELINE.md)*
