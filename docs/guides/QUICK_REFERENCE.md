# 🚀 Quick Reference Guide

> **Common commands and workflows for daily development**  
> Last updated: November 20, 2025

## 🏃 Quick Start Commands

### Running the System

```bash
# Main mode (with Aria hardware)
python src/main.py

# Debug mode (mock hardware)
python src/main.py debug

# With specific dashboard
python src/main.py           # Interactive prompt
# Options: opencv, rerun, web

# Profile mode (performance metrics)
python src/main.py --profile
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Specific test file
pytest tests/test_navigation_pipeline.py -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Mock observer test
python examples/test_mock_basic.py
```

### Benchmarks

```bash
# Performance benchmark
python benchmarks/benchmark_1_performance.py

# Precision testing
python benchmarks/benchmark_2_precision.py

# Distance estimation
python benchmarks/benchmark_3_distances.py
```

---

## 🎮 Runtime Controls

| Key | Action |
|-----|--------|
| `q` | Quit system gracefully |
| `t` | Test audio system (RGB queue) |
| `Ctrl+C` | Emergency stop |

---

## 📊 Viewing Logs

### Session Logs
```bash
# Latest session
ls -lt logs/session_* | head -1

# View performance log
cat logs/session_*/performance.jsonl | jq

# Analyze session
python logs/analyze_session.py logs/session_1234567890/
```

### Telemetry
```bash
# Audio events
cat logs/audio_telemetry.jsonl | jq

# Detection events
cat logs/detections.jsonl | jq '.class_name' -r | sort | uniq -c
```

---

## 🔧 Configuration

### Main Config File
```bash
# Edit configuration
nano src/utils/config.py

# Key settings:
YOLO_DEVICE = "mps"  # or "cuda" or "cpu"
DEPTH_ENABLED = True
PERIPHERAL_VISION_ENABLED = True
```

### Environment Variables
```bash
# Create .env file
cat > .env << EOF
ARIA_DEVICE_IP=192.168.1.100
LOG_LEVEL=DEBUG
EOF
```

---

## 🐛 Common Issues

### Aria Not Connecting
```bash
# Check USB connection
system_profiler SPUSBDataType | grep Aria  # macOS
lsusb | grep Aria                           # Linux

# Check WiFi streaming
ping <aria-ip>
aria-doctor  # If available from SDK
```

### Audio Not Working

**macOS:**
```bash
# Test TTS
say "Testing audio"

# Check permissions
# System Preferences → Security & Privacy → Accessibility
```

**Linux:**
```bash
# Test espeak
espeak "Testing audio"

# Install if missing
sudo apt-get install espeak
```

### Low FPS

```bash
# Check GPU usage
# macOS: Activity Monitor → GPU tab
# Linux: watch -n 1 nvidia-smi

# Reduce load
# Edit config.py:
YOLO_FRAME_SKIP = 3
DEPTH_FRAME_SKIP = 12
```

---

## 🔄 Git Workflow

### Daily Development
```bash
# Start new feature
git checkout -b feature/my-feature

# Regular commits
git add .
git commit -m "feat: add new feature"

# Update from main
git fetch origin
git rebase origin/main
```

### Commit Message Format
```
<type>: <description>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation only
- test: Adding tests
- refactor: Code restructuring
- perf: Performance improvement
- chore: Maintenance
```

See [Contributing](../development/CONTRIBUTING.md) for branch/commit conventions.

---

## 📦 Dependencies

### Update Dependencies
```bash
# Activate environment
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows

# Update packages
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# Freeze current versions
pip freeze > requirements.txt
```

### Add New Package
```bash
pip install <package-name>
pip freeze | grep <package-name> >> requirements.txt
```

---

## 🏗️ Project Structure

```
aria-nav/
├── src/
│   ├── core/           # Core system components
│   ├── presentation/   # UI and visualization
│   └── utils/          # Utilities and config
├── tests/              # Test suite
├── benchmarks/         # Performance benchmarks
├── docs/               # Documentation
├── logs/               # Runtime logs
└── checkpoints/        # Model weights
```

---

## 📚 More Resources

- **Full Documentation**: See [docs/INDEX.md](../INDEX.md)
- **Setup Guide**: [docs/setup/SETUP.md](../setup/SETUP.md)
- **Architecture**: [docs/architecture/architecture_document.md](../architecture/architecture_document.md)
- **Migration**: [docs/migration/NUC_MIGRATION.md](../migration/NUC_MIGRATION.md)

---

## 🆘 Getting Help

1. Check [Contributing](../development/CONTRIBUTING.md) for quick debug pointers; then [TROUBLESHOOTING](../TROUBLESHOOTING.md)
2. Review logs in `logs/` directory
3. Run diagnostics: `python tools/diagnostics.py` (if available)
4. Open issue with reproduction steps

---

**Quick tip:** Bookmark this page for easy reference during development!
