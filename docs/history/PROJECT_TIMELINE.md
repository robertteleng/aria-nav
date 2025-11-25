# 📅 Project Timeline & Development History

> **Complete development journey of Aria Navigation System**  
> Period: October 2024 - November 2025  
> Last updated: November 20, 2025

---

## 🎯 Project Overview

**Duration:** 10 iterations (8 base + 2 advanced optimizations)  
**Total Time:** ~10 weeks  
**Final Performance:** 18.4 FPS (5.3x improvement from initial 3.5 FPS)  
**Status:** ✅ Production Ready

---

## 📊 Development Timeline

### Iteration 1: RGB + YOLO Detection (Week 1)
**Goal:** Basic pipeline with real-time object detection  
**Duration:** ~1 week

#### Achievements
- ✅ Aria USB streaming @ 60 FPS
- ✅ YOLOv11n integration (CPU inference)
- ✅ Frame rotation and preprocessing
- ✅ OpenCV real-time display

#### Technical Decisions
```
✓ Profile28 (60fps streaming)
✓ USB over WiFi (more stable)
✓ CPU inference (avoid MPS bugs)
✓ YOLOv11n (fastest variant)
✓ np.ascontiguousarray() for YOLO
✓ Rotation: np.rot90(image, -1)
✗ Undistortion (degraded detection)
```

#### Key Files Created
- `src/core/observer.py` - Aria SDK interface
- `src/core/vision/yolo_processor.py` - Detection engine
- `src/presentation/opencv_dashboard.py` - Visualization

**Performance:** 20-30 FPS (lightweight, detection only)

---

### Iteration 2: Audio + Zones (Week 2)
**Goal:** Spatial audio feedback system  
**Duration:** ~1 week

#### Achievements
- ✅ Zone-based detection (left/center/right)
- ✅ macOS TTS integration (`say` command)
- ✅ Priority-based audio queue
- ✅ Cooldown mechanism (avoid spam)

#### Architecture
```
Detections → NavigationDecisionEngine
               ↓
          Zone Assignment
               ↓
          Priority Queue
               ↓
          Audio System (TTS)
```

#### Key Components
- `src/core/navigation/navigation_decision_engine.py`
- `src/core/audio/audio_system.py`
- `src/core/navigation/rgb_audio_router.py`

**Performance:** 20-25 FPS (with audio feedback)

---

### Iteration 3: Depth Estimation (Week 3)
**Goal:** Distance estimation for navigation  
**Duration:** ~1.5 weeks

#### Achievements
- ✅ Depth Anything v2 integration
- ✅ Distance-based warnings
- ✅ Depth visualization (mini-map)
- ✅ Combined YOLO + Depth pipeline

#### Technical Stack
```
Depth Model: Depth Anything v2 (ViT-Small)
Backend: PyTorch MPS (macOS)
Resolution: 256x256 → 384x384
Inference: ~120-180ms per frame
```

#### Challenges Solved
- Memory management for dual models
- Frame skipping strategy (YOLO: 3x, Depth: 12x)
- Depth-YOLO coordinate mapping

**Performance:** 18-20 FPS (with aggressive frame skip)

---

### Iteration 4: Low-Light Enhancement (Week 4)
**Goal:** Improve detection in poor lighting  
**Duration:** ~3 days

#### Achievements
- ✅ CLAHE (Contrast Limited Adaptive Histogram Equalization)
- ✅ Gamma correction
- ✅ Auto-enhancement mode
- ✅ Image quality metrics

#### Implementation
```python
# ImageEnhancer module
- CLAHE on LAB color space
- Adaptive gamma correction
- Minimal latency (<5ms)
```

#### Key Files
- `src/core/vision/image_enhancer.py`

**Performance:** 18-20 FPS (enhancement adds ~3ms)

---

### Iteration 5: Motion + IMU (Week 5)
**Goal:** Motion state awareness  
**Duration:** ~4 days

#### Achievements
- ✅ IMU data streaming (2 sensors)
- ✅ Motion state detection (stationary/walking)
- ✅ Acceleration magnitude analysis
- ✅ Hysteresis for stability

#### Motion States
```
Stationary: std < 0.3 m/s²
Walking:    std > 0.6 m/s²
Hysteresis: Maintains last valid state
```

#### Key Files
- `src/core/observer.py` (IMU callbacks)
- Motion detection in Observer class

**Performance:** No impact (~0ms overhead)

---

### Iteration 6: SLAM Cameras (Week 6)
**Goal:** Peripheral vision for lateral obstacles  
**Duration:** ~1 week

#### Achievements
- ✅ SLAM1 (left) + SLAM2 (right) streaming
- ✅ Asynchronous detection workers
- ✅ Lateral event prioritization
- ✅ Integrated SLAM audio routing

#### Architecture
```
Observer → SLAM frames → SlamDetectionWorker (async)
                              ↓
                      SlamDetectionEvent
                              ↓
                      SlamAudioRouter
                              ↓
                   NavigationAudioRouter (unified)
```

#### Technical Details
- Background thread processing
- YOLO profile for SLAM (lower res)
- Independent cooldowns per camera
- Event deduplication

#### Key Files
- `src/core/vision/slam_detection_worker.py`
- `src/core/navigation/slam_audio_router.py`

**Performance:** 18-20 FPS RGB (SLAM async, no impact)

---

### Iteration 7: Audio Routing Refactor (Week 7)
**Goal:** Unified audio management system  
**Duration:** ~1 week

#### Achievements
- ✅ Centralized `NavigationAudioRouter`
- ✅ Per-source priority queues (RGB/SLAM1/SLAM2)
- ✅ Dynamic cooldowns
- ✅ Telemetry logging (JSONL)
- ✅ Metrics and monitoring

#### Architecture Evolution
```
BEFORE:
RgbAudioRouter → AudioSystem
SlamAudioRouter → AudioSystem

AFTER:
RgbAudioRouter ────┐
SlamAudioRouter ───┤→ NavigationAudioRouter → AudioSystem
IMU Events ────────┘
```

#### Features
- Source-specific cooldowns
- Queue overflow management
- Audio event telemetry
- Session summaries

#### Key Files
- `src/core/audio/navigation_audio_router.py`
- `logs/audio_telemetry.jsonl`

**Performance:** 18-20 FPS (minimal audio overhead)

---

### Iteration 8: Dashboards (Week 8)
**Goal:** Multiple visualization options  
**Duration:** ~5 days

#### Achievements
- ✅ OpenCV dashboard (improved)
- ✅ Rerun 3D visualization
- ✅ Web dashboard (Flask + WebSockets)
- ✅ Dashboard selection at startup

#### Dashboards Created

**1. OpenCV** (default)
- Live RGB + detection overlays
- Depth mini-map
- SLAM peripheral views
- Audio/performance HUD

**2. Rerun**
- 3D point clouds
- Trajectory visualization
- Multi-camera views
- Timeline playback

**3. Web Dashboard**
- Browser-based @ localhost:5000
- Real-time metrics
- Video streams (MJPEG)
- Beep statistics
- Detection log
- SLAM events

#### Key Files
- `src/presentation/dashboards/opencv_dashboard.py`
- `src/presentation/dashboards/rerun_dashboard.py`
- `src/presentation/dashboards/web_dashboard.py`
- `src/presentation/dashboards/dashboard_html_template.py`

**Performance:** 18-20 FPS (all dashboards)

---

## 🚀 Advanced Optimization Phase

### Iteration 9: CUDA Optimization (Week 9-10)
**Goal:** 3-5x performance improvement  
**Duration:** ~2 weeks

#### Phase 1: Quick Wins
- ✅ Resolution increase (256→640 YOLO, 256→384 Depth)
- ✅ cuDNN benchmark mode
- ✅ TensorFloat-32 (TF32)
- ✅ Pinned memory
- ✅ Non-blocking transfers

**Result:** Minimal gains (GPU not bottleneck)

#### Phase 2: TensorRT + ONNX
- ✅ YOLO TensorRT export (`yolo12n.engine`)
- ✅ Depth ONNX export (`depth_anything_v2_vits.onnx`)
- ✅ ONNX Runtime with CUDA Execution Provider
- ⚠️ Critical bug: Depth not executing (fixed)

**Key Fix:** Model validation checks in pipeline
```python
# BEFORE (broken):
if self.depth_estimator.model is None:
    return None

# AFTER (fixed):
if not self.depth_estimator._is_model_loaded():
    return None
```

**Result:** 3.5 FPS → 12.0 FPS (+243%)

#### Phase 3: Multiprocessing
- ✅ Separate processes for cameras
- ✅ Inter-process communication (queues)
- ✅ SLAM offloaded to workers
- ✅ Frame buffer management

**Result:** 12.0 FPS → 18.4 FPS (+53%)

#### Phase 4: CUDA Streams (Attempted)
- ⚠️ Complex implementation
- ⚠️ Marginal gains vs complexity
- ❌ Deferred for future

#### Final Performance
```
Baseline:  3.5 FPS  (pre-optimization)
Phase 2:  12.0 FPS  (+243%)
Phase 3:  18.4 FPS  (+426% total)

YOLO:     100ms → 40ms   (2.5x)
Depth:    315ms → 27ms   (11.7x)
Latency:  283ms → 48ms   (83% reduction)
```

#### Key Files
- `export_tensorrt_slam.py` - TensorRT export script
- `checkpoints/*.engine` - TensorRT models
- `checkpoints/*.onnx` - ONNX models
- Updated `depth_estimator.py` with ONNX Runtime
- `src/core/processing/multiproc_types.py`

---

### Iteration 10: Telemetry + Async Logging (Week 10)
**Goal:** Production-ready telemetry without performance impact  
**Duration:** ~2 days

#### Achievements
- ✅ Async telemetry logger (background thread)
- ✅ Batch writes (reduce I/O)
- ✅ Multiple log streams
  - `performance.jsonl`
  - `detections.jsonl`
  - `audio_events.jsonl`
  - `audio_telemetry.jsonl`
- ✅ Session-based organization
- ✅ Zero FPS impact

#### Before/After
```
BEFORE: Sync writes blocking main thread
- Spikes: 250-350ms every 50 frames
- FPS drops: 18 → 14-15 FPS

AFTER: Async background writes
- Steady: 18.4 FPS
- No spikes
- Batch writes every 2s
```

#### Key Files
- `src/utils/telemetry_logger.py` (AsyncTelemetryLogger)
- `logs/session_*/` structure

---

## 📊 Overall Progress

### Performance Evolution

| Iteration | FPS | Key Feature | Latency |
|-----------|-----|-------------|---------|
| 1 | 25-30 | YOLO only | ~40ms |
| 2 | 20-25 | + Audio | ~50ms |
| 3 | 18-20 | + Depth | ~120ms |
| 4 | 18-20 | + Enhancement | ~125ms |
| 5 | 18-20 | + Motion | ~125ms |
| 6 | 18-20 | + SLAM | ~125ms |
| 7 | 18-20 | + Audio Routing | ~125ms |
| 8 | 18-20 | + Dashboards | ~125ms |
| 9 | **18.4** | **+ TensorRT/ONNX** | **~48ms** |
| 10 | **18.4** | + Async Logging | **~48ms** |

### Feature Additions

```
 Week 1    Week 2    Week 3    Week 4    Week 5    Week 6    Week 7    Week 8    Week 9-10
   │         │         │         │         │         │         │         │          │
   ▼         ▼         ▼         ▼         ▼         ▼         ▼         ▼          ▼
 YOLO → Audio → Depth → Enhance → Motion → SLAM → Routing → Dash → OPTIMIZATION
  ✓       ✓       ✓       ✓        ✓       ✓        ✓        ✓        ✓
```

---

## 🏗️ Architecture Evolution

### Phase 1: MVP (Iterations 1-2)
```
Aria → Observer → YOLO → Display
                   ↓
                 Audio
```

### Phase 2: Full Pipeline (Iterations 3-5)
```
Aria → Observer → Enhancer → Depth → YOLO → DecisionEngine → Audio
                                                  ↓
                                              Display
```

### Phase 3: Complete System (Iterations 6-8)
```
                    ┌─ RGB → Pipeline → RgbAudioRouter ──┐
Aria → Observer ────┤                                     ├→ NavigationAudioRouter → Audio
                    └─ SLAM → Workers → SlamAudioRouter ─┘
                              ↓
                         Dashboards
```

### Phase 4: Optimized (Iterations 9-10)
```
                    ┌─ RGB → [TensorRT Pipeline] → Router ──┐
Aria → Observer ────┤                                        ├→ Unified Router → Audio
                    └─ SLAM → [Multiproc Workers] → Router ─┘
                              ↓
                         Dashboards
                              ↓
                    [Async Telemetry Logger]
```

---

## 💡 Key Lessons Learned

### What Worked Well
1. **Incremental development** - One feature at a time
2. **Mock observer** - Rapid iteration without hardware
3. **Profile early** - Identified bottlenecks quickly
4. **TensorRT/ONNX** - Massive gains (11x depth, 2.5x YOLO)
5. **Async logging** - Zero-overhead telemetry

### What Didn't Work
1. **MPS backend** - Too unstable, switched to CUDA/CPU
2. **CUDA Streams** - Complexity vs marginal gains
3. **Over-engineering** - Keep it simple initially
4. **WiFi streaming** - USB more reliable

### Technical Debt Paid
- Audio routing refactor (iteration 7)
- Pipeline validation checks (iteration 9)
- Proper multiprocessing (iteration 9)
- Documentation reorganization (iteration 10)

---

## 📚 Documentation Created

### Core Documentation (45+ files)
- Setup guides (macOS/Linux)
- Architecture documents
- API references
- Testing guides
- Migration guides
- Troubleshooting
- Development workflow

### Historical Records
- Daily development notes
- Phase implementation details
- CUDA optimization notes
- Performance benchmarks

---

## 🎯 Final Statistics

### Code Metrics
- **Python Files:** 80+
- **Lines of Code:** ~15,000
- **Test Files:** 20+
- **Documentation:** 45+ MD files (2,500+ lines)

### Performance Metrics
- **FPS Improvement:** 426% (3.5 → 18.4 FPS)
- **Latency Reduction:** 83% (283ms → 48ms)
- **YOLO Speedup:** 2.5x (TensorRT)
- **Depth Speedup:** 11.7x (ONNX + CUDA EP)

### Features Delivered
- ✅ Real-time object detection
- ✅ Depth estimation
- ✅ Spatial audio feedback
- ✅ Motion state detection
- ✅ Peripheral vision (SLAM)
- ✅ Multiple dashboards
- ✅ Production telemetry
- ✅ Complete documentation

---

## 🚀 Future Work

### Planned Enhancements
- [ ] NUC + RTX 2060 migration (60+ FPS target)
- [ ] Multi-language audio support
- [ ] Mobile companion app
- [ ] Cloud telemetry dashboard
- [ ] Object tracking (reduce flicker)
- [ ] Semantic segmentation
- [ ] Path planning visualization

### Technical Improvements
- [ ] CUDA Streams (when proven beneficial)
- [ ] FP16/INT8 quantization
- [ ] Model distillation
- [ ] Dynamic batching
- [ ] Better memory pooling

---

**Project Status:** ✅ Production Ready  
**Next Milestone:** Hardware Migration → Intel NUC + RTX 2060  
**Target Performance:** 60+ FPS sustained

---

*For detailed phase documentation, see `docs/archive/cuda/` and `docs/history/`*
