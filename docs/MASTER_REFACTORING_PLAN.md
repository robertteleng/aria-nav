# Master Refactoring Plan - Aria Navigation System

**Objetivo:** Refactoring completo y sistemático del codebase (~14,000 LOC)
**Estrategia:** Layer-by-layer, component-by-component cleanup
**Prioridad:** Limpieza, simplicidad, documentación, mantenibilidad

---

## 📋 Checklist General por Archivo

Para CADA archivo Python, aplicar en orden:

### 1. **Imports Cleanup**
- [ ] Remove unused imports
- [ ] Sort imports: stdlib → third-party → local
- [ ] Remove duplicate imports
- [ ] Use explicit imports (avoid `import *`)

### 2. **Remove Dead Code**
- [ ] Delete commented code blocks
- [ ] Remove unreachable code (after `return`/`raise`)
- [ ] Delete unused functions/methods
- [ ] Remove unused variables
- [ ] Delete deprecated code paths

### 3. **Remove Emojis**
- [ ] Remove ALL emojis from code
- [ ] Keep emojis ONLY in user-facing strings (TTS messages)
- [ ] Replace emoji comments with clear English

### 4. **Simplify Logic**
- [ ] Extract nested conditionals
- [ ] Remove unnecessary `else` after `return`
- [ ] Simplify boolean expressions
- [ ] Replace complex ternaries with `if`/`else`
- [ ] Flatten deep nesting (max 3 levels)

### 5. **Remove Duplication**
- [ ] Extract repeated code to functions
- [ ] Consolidate similar logic
- [ ] Use constants for magic numbers
- [ ] Share common code between modules

### 6. **Improve Names**
- [ ] Use descriptive variable names (no `x`, `tmp`, `data`)
- [ ] Follow PEP8 naming: `snake_case` for functions/vars
- [ ] Use full words (no abbreviations unless standard)
- [ ] Rename misleading names

### 7. **Documentation**
- [ ] Add module docstring (purpose, usage)
- [ ] Document all public functions/classes
- [ ] Add inline comments for complex logic ONLY
- [ ] Remove obvious comments (`# increment i`)
- [ ] Use English for ALL comments

### 8. **Type Hints**
- [ ] Add type hints to function signatures
- [ ] Use `Optional[T]` for nullable types
- [ ] Import types from `typing`
- [ ] Add return type hints

### 9. **Error Handling**
- [ ] Remove bare `except:` (specify exceptions)
- [ ] Don't catch exceptions you can't handle
- [ ] Log errors before re-raising
- [ ] Remove unnecessary try/except

### 10. **Testing Readiness**
- [ ] Make functions pure when possible
- [ ] Reduce global state
- [ ] Inject dependencies (no hardcoded)
- [ ] Separate logic from I/O

---

## 🗂️ Refactoring Order (Layer-by-Layer)

### **LAYER 1: Core Infrastructure** (Foundation)
Critical components that everything depends on.

#### 1.1 Configuration & Utils
- [ ] `src/utils/config.py` - Consolidate all configs, remove duplication
- [ ] `src/utils/ctrl_handler.py` - Simplify signal handling
- [ ] `src/utils/profiler.py` - Clean profiling code
- [ ] `src/utils/system_monitor.py` - Remove unused monitors
- [ ] `src/utils/resource_monitor.py` - Merge with system_monitor if duplicate
- [ ] `src/utils/memory_profiler.py` - Keep only if used

**Goals:**
- Single source of truth for Config
- Remove monitoring code if unused
- Clear, documented utilities

---

### **LAYER 2: Hardware & SDK Interface** (Device Connection)
Aria SDK interaction, lowest hardware layer.

#### 2.1 Device Management
- [ ] `src/core/hardware/device_manager.py` - Clean SDK connection logic
- [ ] `src/core/observer.py` - Simplify frame capture from SDK
- [ ] `src/core/mock_observer.py` - Ensure parity with Observer

**Goals:**
- Clear separation: DeviceManager (connection) vs Observer (frames)
- Remove duplicated SDK calls
- Document calibration flow

---

### **LAYER 3: Vision & Detection** (YOLO + Depth)
Computer vision processing pipeline.

#### 3.1 YOLO Detection
- [ ] `src/core/vision/yolo_processor.py` - Clean YOLO inference logic
- [ ] `src/core/vision/detected_object.py` - Simplify object representation
- [ ] `src/core/vision/gpu_utils.py` - Remove if unused, or consolidate

#### 3.2 Depth Estimation
- [ ] `src/core/vision/depth_estimator.py` - Clean depth inference
- [ ] `src/core/vision/image_enhancer.py` - Remove if not used

#### 3.3 Multi-Camera Workers
- [ ] `src/core/vision/slam_detection_worker.py` - Simplify worker logic
- [ ] Remove `src/core/vision/object_tracker.py` - **DEPRECATED** (replaced by global_object_tracker)

#### 3.4 Cross-Camera Tracking (Recent Work)
- [ ] `src/core/vision/global_object_tracker.py` - Already clean from Phase 2
- [ ] `src/core/vision/camera_geometry.py` - Already clean from Phase 3

**Goals:**
- Single, clear YOLO pipeline
- Remove deprecated ObjectTracker
- Document multi-camera flow

---

### **LAYER 4: Navigation Logic** (Decision Engine)
High-level navigation decisions and audio routing.

#### 4.1 Decision Engine
- [ ] `src/core/navigation/navigation_decision_engine.py` - Simplify priority logic
- [ ] `src/core/navigation/navigation_pipeline.py` - Clean pipeline orchestration

#### 4.2 Audio Routing
- [ ] `src/core/navigation/slam_audio_router.py` - Remove duplication with rgb_router
- [ ] `src/core/navigation/rgb_audio_router.py` - Consolidate common logic
- [ ] Extract shared router base class if needed

#### 4.3 Coordinator & Builder
- [ ] `src/core/navigation/coordinator.py` - Simplify orchestration
- [ ] `src/core/navigation/builder.py` - Clean dependency injection

**Goals:**
- Clear decision flow (analyze → evaluate → route → announce)
- Remove duplicated routing logic
- Document priority system

---

### **LAYER 5: Audio System** (TTS Output)
Text-to-speech and audio management.

#### 5.1 Audio Core
- [ ] `src/core/audio/audio_system.py` - Clean TTS interface
- [ ] `src/core/audio/navigation_audio_router.py` - Simplify queue management

**Goals:**
- Simple audio interface: speak(message, priority)
- Remove queue complexity if unnecessary
- Document priority/interrupt logic

---

### **LAYER 6: IMU & Motion** (Motion Detection)
Accelerometer/gyroscope processing.

#### 6.1 Motion Detection
- [ ] `src/core/imu/motion_detector.py` - Simplify walking/stationary detection

**Goals:**
- Clear motion state: "walking" | "stationary"
- Document thresholds

---

### **LAYER 7: Multiprocessing** (Performance)
Parallel processing for SLAM cameras.

#### 7.1 Multiproc Infrastructure
- [ ] `src/core/processing/multiproc_types.py` - Clean shared types
- [ ] `src/core/processing/shared_memory_manager.py` - Simplify if complex
- [ ] `src/core/processing/slam_worker.py` - Clean worker logic
- [ ] `src/core/processing/central_worker.py` - Remove if unused

**Goals:**
- Clear worker responsibilities
- Remove dead multiproc code
- Document IPC mechanisms

---

### **LAYER 8: Telemetry & Logging** (Observability)
Metrics, logs, debugging.

#### 8.1 Loggers
- [ ] `src/core/telemetry/loggers/telemetry_logger.py` - Simplify async logging
- [ ] `src/core/telemetry/loggers/navigation_logger.py` - Remove duplication
- [ ] `src/core/telemetry/loggers/depth_logger.py` - Merge if similar

**Goals:**
- Single logger interface
- Remove redundant loggers
- Clear log levels (DEBUG, INFO, WARNING)

---

### **LAYER 9: External Dependencies** (Third-Party)
External libraries (Depth Anything V2).

#### 9.1 Depth Anything V2
- [ ] Review `src/external/depth_anything_v2/**` - Keep as-is or minimal changes
- Document if modifications were made

**Goals:**
- Isolate external code
- Document any custom modifications

---

### **LAYER 10: Main Entry Point** (Application)
Top-level orchestration.

#### 10.1 Main
- [ ] `src/main.py` - Simplify initialization flow
- [ ] Remove redundant setup
- [ ] Clear error handling

**Goals:**
- Linear, readable main()
- Clear startup sequence
- Graceful shutdown

---

## 📊 Metrics to Track

For each layer, measure:

```python
# Before refactoring
- Lines of code (LOC)
- Number of functions
- Cyclomatic complexity
- Import count
- Duplicate code %

# After refactoring
- LOC reduction
- Functions removed
- Complexity reduction
- Clearer responsibilities
```

---

## 🛠️ Refactoring Workflow

For EACH file:

### Step 1: Analysis (Read-Only)
```bash
# Count lines
wc -l file.py

# Find TODOs/FIXMEs
grep -n "TODO\|FIXME" file.py

# Find emojis
grep -P "[\x{1F600}-\x{1F64F}]" file.py

# Check imports
grep "^import\|^from" file.py

# Find dead code (functions never called)
# Use IDE "Find Usages" or grep
```

### Step 2: Plan Changes
- List specific changes needed
- Identify dependencies (what breaks if changed)
- Check if file has tests

### Step 3: Refactor
- Make ONE type of change at a time
- Test after each change (if tests exist)
- Commit frequently with clear messages

### Step 4: Document
- Add/update docstrings
- Update inline comments
- Note breaking changes

### Step 5: Verify
- Run tests (if exist)
- Check imports still work
- No syntax errors

---

## 🚦 Priority Levels

### P0 - Critical (Do First)
Files that block everything else:
- `config.py`
- `coordinator.py`
- `navigation_decision_engine.py`
- `yolo_processor.py`

### P1 - High (Do Second)
Core functionality:
- `navigation_pipeline.py`
- `global_object_tracker.py`
- Audio routers
- `audio_system.py`

### P2 - Medium (Do Third)
Supporting systems:
- Workers
- Loggers
- Utils

### P3 - Low (Do Last)
Nice-to-have:
- External code
- Profilers
- Monitors (if unused)

---

## 📝 Commit Strategy

**Atomic commits** for each file:

```bash
# Example commit sequence for one file:
git commit -m "refactor(config): remove unused imports"
git commit -m "refactor(config): extract audio labels to constants"
git commit -m "refactor(config): add type hints to Config class"
git commit -m "docs(config): add comprehensive module docstring"
```

**Summary commits** for each layer:

```bash
git commit -m "refactor(layer-1): complete utils and config cleanup

- config.py: consolidated constants, removed duplication
- ctrl_handler.py: simplified signal handling
- Removed unused profilers and monitors

Stats:
- -245 lines of code
- -12 unused functions
- Complexity reduced 40%"
```

---

## 🎯 Success Criteria

After complete refactoring:

- [ ] **0 emojis** in code (except TTS messages)
- [ ] **100% English** comments and docs
- [ ] **No commented code** blocks
- [ ] **No unused imports**
- [ ] **No duplicate logic** (DRY)
- [ ] **All public APIs documented**
- [ ] **Type hints** on 80%+ functions
- [ ] **Clear separation** of concerns
- [ ] **<200 lines** per file (target, not strict)
- [ ] **<10 complexity** per function (McCabe)

---

## 📅 Estimated Timeline

Assuming 2-4 hours per layer:

- **Layer 1 (Utils/Config):** 3 hours
- **Layer 2 (Hardware):** 2 hours
- **Layer 3 (Vision):** 6 hours (largest)
- **Layer 4 (Navigation):** 5 hours
- **Layer 5 (Audio):** 2 hours
- **Layer 6 (IMU):** 1 hour
- **Layer 7 (Multiproc):** 3 hours
- **Layer 8 (Telemetry):** 3 hours
- **Layer 9 (External):** 1 hour (minimal)
- **Layer 10 (Main):** 2 hours

**Total:** ~28-30 hours of focused work

Split into daily sessions:
- **Week 1:** Layers 1-3 (foundation + vision)
- **Week 2:** Layers 4-6 (navigation + audio + IMU)
- **Week 3:** Layers 7-10 (multiproc + logging + main)

---

## 🔧 Tools to Use

### Static Analysis
```bash
# Find unused code
vulture src/

# Check complexity
radon cc src/ -a

# Find duplicates
pylint --disable=all --enable=duplicate-code src/

# Type checking
mypy src/
```

### Code Formatting
```bash
# Auto-format (after manual refactor)
black src/
isort src/
```

### Documentation
```bash
# Generate API docs
pdoc src/ -o docs/api/
```

---

## 📖 Documentation Plan

Create comprehensive docs alongside refactoring:

### Module-Level Docs
For each major component, create:
- `docs/architecture/MODULE_NAME.md`
- Flow diagrams (PlantUML)
- API reference
- Usage examples

### Architecture Docs
- `docs/architecture/SYSTEM_OVERVIEW.md`
- `docs/architecture/DATA_FLOW.md`
- `docs/architecture/CROSS_CAMERA_TRACKING.md`
- `docs/architecture/AUDIO_PIPELINE.md`

---

## 🚀 Getting Started

### Next Immediate Action

**Start with Layer 1, File 1:**

```bash
# Create feature branch
git checkout -b refactor/layer-1-utils-config

# Analyze first file
wc -l src/utils/config.py
grep -n "TODO\|FIXME" src/utils/config.py

# Begin refactoring (see checklist above)
```

---

**Ready to start?** Reply with:
- "Start Layer 1" - Begin with Utils/Config
- "Analyze first" - Get detailed analysis of current state
- "Custom order" - Specify which files to tackle first
