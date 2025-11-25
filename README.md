# Aria Navigation System

> **Assistive navigation system for visually impaired users using Meta Aria glasses**  
> Combines computer vision, spatial analysis, and prioritized audio feedback in real-time.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/status-active-success.svg)]()

---

## ⚡ Quick Start

```bash
# Clone and install
git clone https://github.com/<your-user>/aria-nav.git
cd aria-nav
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run with Aria hardware
python src/main.py

# Test without hardware
python src/main.py debug
```

**Controls:** `q` = quit | `t` = test audio | `Ctrl+C` = emergency stop

---

## 🎯 What It Does

**Real-time navigation assistance using:**
- 🎥 **RGB Camera** - Object detection (YOLO) + depth estimation (Depth-Anything v2) + fisheye rectification
- 👀 **Peripheral Vision** - SLAM cameras for lateral obstacle detection (rectified)
- 🧭 **IMU Sensors** - Motion state tracking (stationary/walking)
- 🔊 **Spatial Audio** - Prioritized voice commands + beep alerts
- 📊 **Live Dashboards** - OpenCV, Rerun, or Web visualization
- 🎯 **Image Rectification** - SDK-based undistortion for all fisheye cameras

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Meta Aria   │───▶│  Observer    │───▶│ Pipeline        │
│ (RGB+SLAM+  │    │ (SDK Bridge) │    │ (Vision + AI)   │
│  IMU)       │    └──────────────┘    └─────────────────┘
└─────────────┘                                │
                                               ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Audio       │◀───│ Navigation   │◀───│ Decision        │
│ System      │    │ Audio Router │    │ Engine          │
└─────────────┘    └──────────────┘    └─────────────────┘
```

**Key Components:**
- **Observer** - Hardware interface (cameras + IMU)
- **Pipeline** - Enhancement → Depth → Detection
- **Decision Engine** - Spatial reasoning and prioritization
- **Audio Router** - Cooldown management and queue coordination
- **Presentation** - Multi-dashboard rendering

---

## 📋 Requirements

### Hardware
- Meta Aria glasses with `profile28` enabled
- Mac (Apple Silicon recommended) or Linux with NVIDIA GPU
- USB-C connection or WiFi streaming

### Software
- Python 3.10+
- PyTorch with MPS (macOS) or CUDA (Linux)
- Aria SDK (from Meta)
- See [Setup Guide](docs/setup/SETUP.md) for detailed instructions

---

## 🚀 Installation

### macOS (Current)
```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy projectaria-tools transformers pytest

# Verify TTS
which say  # Should return /usr/bin/say
```

### Linux (Migration Target)
See [NUC Migration Guide](docs/migration/NUC_MIGRATION.md) for CUDA setup.

---

## 📖 Documentation

| Resource | Description |
|----------|-------------|
| [📚 Documentation Index](docs/INDEX.md) | Central hub for all documentation |
| [🚀 Quick Reference](docs/guides/QUICK_REFERENCE.md) | Common commands and workflows |
| [🏗️ Architecture](docs/architecture/architecture_document.md) | System design and components |
| [🔧 Setup Guide](docs/setup/SETUP.md) | Detailed installation instructions |
| [🧪 Testing Guide](docs/testing/README.md) | Test strategy and execution |
| [🐛 Troubleshooting](docs/TROUBLESHOOTING.md) | Catálogo de síntomas→acciones |
| [🤝 Contributing](docs/development/CONTRIBUTING.md) | Workflow, ramas, commits, pruebas |

---

## 🎛️ Configuration

Main settings in `src/utils/config.py`:

```python
# Vision
YOLO_DEVICE = "mps"          # GPU device (mps/cuda/cpu)
DEPTH_ENABLED = True          # Enable depth estimation
PERIPHERAL_VISION_ENABLED = True  # SLAM cameras

# Audio
AUDIO_COOLDOWN_SECONDS = 2.0  # Minimum time between commands
BEEP_ENABLED = True           # Distance beeps

# Performance
YOLO_FRAME_SKIP = 3          # Process every Nth frame
DEPTH_FRAME_SKIP = 12        # Depth estimation frequency
```

---

## 🧪 Testing

```bash
# Run full test suite
pytest tests/ -v

# Specific tests
pytest tests/test_navigation_pipeline.py
pytest tests/test_audio_router.py

# With coverage
pytest --cov=src --cov-report=html

# Mock hardware test
python examples/test_mock_basic.py
```

---

## 📊 Performance

### Current (Linux CUDA + TensorRT) ✅
- **FPS:** 18-22 fps (RTX 2060)
- **YOLO Latency:** ~40ms (TensorRT FP16)
- **Depth Latency:** ~27ms (ONNX Runtime CUDA)
- **End-to-end:** ~48ms
- **GPU Memory:** ~1.5GB / 6GB

### Optimization Journey
```
v1.0: 3.5 FPS (baseline)
v1.9: 18.4 FPS (+426% with TensorRT/ONNX)
v2.0: 19.0 FPS (+3% with Phase 6 hybrid streams)
```

See [CHANGELOG.md](CHANGELOG.md) for detailed performance history.

---

## 📁 Project Structure

```
aria-nav/
├── src/
│   ├── core/              # Core system (hardware, vision, audio, navigation)
│   ├── presentation/      # UI and visualization
│   └── utils/             # Configuration and utilities
├── tests/                 # Test suite
├── benchmarks/            # Performance benchmarks
├── docs/                  # Documentation
│   ├── INDEX.md          # 📚 Start here
│   ├── guides/           # User guides
│   ├── architecture/     # System design
│   ├── development/      # Dev workflows
│   ├── migration/        # Platform migration
│   └── testing/          # Test documentation
├── logs/                  # Runtime logs and telemetry
└── checkpoints/           # Model weights
```

---

## 🗺️ Roadmap

- [x] RGB pipeline with YOLO + Depth
- [x] Peripheral vision (SLAM cameras)
- [x] Audio routing with priorities
- [x] Web dashboard
- [x] Motion state detection
- [x] TensorRT optimization (YOLO FP16)
- [x] NUC + RTX 2060 migration
- [x] MLflow experiment tracking
- [ ] Multi-language support
- [ ] Mobile companion app
- [ ] Fisheye undistortion optimization

---

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines on:
- Code style and conventions
- Testing requirements
- Pull request process
- Development workflow

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Roberto Rojas Sahuquillo**  
Master's Thesis 2025  
Universidad [Your University]

---

## 🙏 Acknowledgments

- Meta Aria team for Project Aria SDK
- Open source community (Ultralytics, Depth-Anything, PyTorch)
- Accessibility research lab

---

## 📞 Support

- 📚 **Documentation:** [docs/INDEX.md](docs/INDEX.md)
- 🐛 **Issues:** Check [Problem Solving Guide](docs/development/problem_solving_guide.md)
- 💬 **Discussions:** Open an issue with `[Question]` tag

---

**Status:** 🔬 Innovation & Research | 🚀 Active Development
