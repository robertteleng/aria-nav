# 🧪 Testing Documentation

> **Test suite overview and execution guide**  
> Last updated: November 20, 2025  
> Status: ✅ Active

## 📑 Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Writing Tests](#writing-tests)
6. [CI/CD Integration](#cicd-integration)

---

## Overview

The Aria Navigation System uses **pytest** for comprehensive testing across all components.

### Test Coverage

- ✅ **Core Components**: Pipeline, audio router, vision modules
- ✅ **Integration Tests**: End-to-end system behavior
- ✅ **Mock Hardware**: Testing without Aria glasses
- ✅ **Performance**: Benchmarks and profiling
- ✅ **Configuration**: Validation of settings

### Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_navigation_pipeline.py -v
```

---

## Test Structure

```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures
│
├── core/                          # Core system tests
│   ├── test_observer.py          # Hardware observer
│   ├── test_navigation_pipeline.py # Vision pipeline
│   ├── test_audio_router.py      # Audio routing
│   ├── test_slam_worker.py       # SLAM detection
│   └── test_motion_detector.py   # IMU processing
│
├── integration/                   # Integration tests
│   ├── test_full_system.py       # End-to-end
│   └── test_coordinator.py       # Component coordination
│
├── benchmarks/                    # Performance tests
│   ├── benchmark_1_performance.py
│   ├── benchmark_2_precision.py
│   └── benchmark_3_distances.py
│
└── fixtures/                      # Test data
    ├── mock_frames.py
    └── test_images/
```

---

## Running Tests

### Basic Execution

```bash
# Activate environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with summary
pytest tests/ -v --tb=short

# Stop on first failure
pytest tests/ -x
```

### Test Selection

```bash
# Run specific file
pytest tests/core/test_navigation_pipeline.py

# Run specific test
pytest tests/core/test_audio_router.py::test_priority_queue

# Run by marker
pytest tests/ -m "integration"
pytest tests/ -m "slow"

# Run by keyword
pytest tests/ -k "audio"
pytest tests/ -k "pipeline and not slow"
```

### Coverage Reports

```bash
# HTML coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Terminal coverage
pytest tests/ --cov=src --cov-report=term

# XML for CI/CD
pytest tests/ --cov=src --cov-report=xml
```

### Verbose Output

```bash
# Show print statements
pytest tests/ -v -s

# Show locals on failure
pytest tests/ -v -l

# Full traceback
pytest tests/ -v --tb=long
```

---

## Test Categories

### 1. Unit Tests

Test individual components in isolation.

**Location:** `tests/core/`

**Examples:**
```bash
# Test YOLO processor
pytest tests/core/test_yolo_processor.py -v

# Test depth estimator
pytest tests/core/test_depth_estimator.py -v

# Test audio system
pytest tests/core/test_audio_system.py -v
```

### 2. Integration Tests

Test component interactions and data flow.

**Location:** `tests/integration/`

**Examples:**
```bash
# Test full pipeline
pytest tests/integration/test_full_system.py -v

# Test coordinator
pytest tests/integration/test_coordinator.py -v
```

### 3. Mock Hardware Tests

Test system without physical Aria glasses.

**Location:** `examples/`

**Examples:**
```bash
# Basic mock test
python examples/test_mock_basic.py

# Mock observer test
python examples/test_mock_observer.py

# Audio sequence test
python examples/test_audio_sequence.py
```

See: [Mock Observer Guide](../guides/MOCK_OBSERVER_GUIDE.md)

### 4. Performance Benchmarks

Measure system performance and throughput.

**Location:** `benchmarks/`

**Examples:**
```bash
# Overall performance
python benchmarks/benchmark_1_performance.py

# Detection precision
python benchmarks/benchmark_2_precision.py

# Distance estimation accuracy
python benchmarks/benchmark_3_distances.py
```

### 5. Stress Tests

Test system under load and edge cases.

**Examples:**
```bash
# Async stress test
python test_async_stress.py

# Telemetry stress test
python test_async_telemetry.py

# System async test
./test_system_async.sh
```

---

## Test Documentation

### Navigation Audio Testing

**File:** `navigation_audio_testing.md`

Comprehensive testing guide for the audio routing system:
- Priority queue behavior
- Cooldown mechanisms
- Source-specific routing
- Metrics validation

**Read full guide:** [navigation_audio_testing.md](navigation_audio_testing.md)

---

## Writing Tests

### Test Template

```python
# tests/core/test_my_component.py
import pytest
from src.core.my_module import MyComponent

class TestMyComponent:
    """Test suite for MyComponent"""
    
    @pytest.fixture
    def component(self):
        """Create component instance for testing"""
        return MyComponent(param1="value1")
    
    def test_basic_functionality(self, component):
        """Test basic component behavior"""
        result = component.process(input_data)
        assert result is not None
        assert result.status == "success"
    
    def test_edge_case(self, component):
        """Test edge case handling"""
        with pytest.raises(ValueError):
            component.process(invalid_input)
    
    @pytest.mark.slow
    def test_performance(self, component):
        """Test performance requirements"""
        import time
        start = time.time()
        component.process(large_input)
        duration = time.time() - start
        assert duration < 0.1  # Must complete in < 100ms
```

### Using Fixtures

```python
# tests/conftest.py - Shared fixtures
import pytest
import numpy as np

@pytest.fixture
def mock_frame():
    """Generate synthetic RGB frame"""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_detection():
    """Generate mock YOLO detection"""
    return {
        'class_name': 'person',
        'confidence': 0.85,
        'bbox': [100, 100, 200, 300],
        'distance': 2.5
    }
```

### Markers

```python
# Mark tests for selective execution
@pytest.mark.integration
def test_full_pipeline():
    pass

@pytest.mark.slow
def test_heavy_computation():
    pass

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_gpu_acceleration():
    pass
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      
      - name: Run tests
        run: |
          pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```bash
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest
        name: pytest
        entry: pytest tests/core -v
        language: system
        pass_filenames: false
        always_run: true
```

---

## Test Best Practices

### 1. Test Naming

```python
# Good: Descriptive test names
def test_audio_router_respects_cooldown():
    pass

def test_yolo_filters_low_confidence_detections():
    pass

# Bad: Vague names
def test_audio():
    pass

def test_yolo_1():
    pass
```

### 2. Arrange-Act-Assert Pattern

```python
def test_detection_priority():
    # Arrange: Set up test data
    router = NavigationAudioRouter()
    critical_msg = create_critical_message()
    normal_msg = create_normal_message()
    
    # Act: Execute the operation
    router.enqueue(critical_msg)
    router.enqueue(normal_msg)
    
    # Assert: Verify the result
    next_msg = router.dequeue()
    assert next_msg == critical_msg  # Critical first
```

### 3. Independent Tests

```python
# Good: Each test is independent
def test_pipeline_with_depth():
    pipeline = create_fresh_pipeline()
    result = pipeline.process(frame)
    assert result.has_depth

def test_pipeline_without_depth():
    pipeline = create_fresh_pipeline()
    pipeline.disable_depth()
    result = pipeline.process(frame)
    assert not result.has_depth

# Bad: Tests depend on execution order
pipeline = None  # Global state

def test_create_pipeline():
    global pipeline
    pipeline = Pipeline()

def test_use_pipeline():  # Fails if test_create_pipeline didn't run
    result = pipeline.process(frame)
```

### 4. Mock External Dependencies

```python
from unittest.mock import Mock, patch

def test_audio_system_without_hardware():
    """Test audio without actual TTS"""
    with patch('subprocess.run') as mock_run:
        audio = AudioSystem()
        audio.speak("test")
        
        # Verify TTS command was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert 'say' in args or 'espeak' in args
```

---

## Continuous Testing

### Watch Mode

```bash
# Install pytest-watch
pip install pytest-watch

# Run tests on file changes
ptw tests/ -- -v
```

### Quick Feedback Loop

```bash
# Fast tests only (exclude benchmarks)
pytest tests/core -v -m "not slow"

# Test single component during development
pytest tests/core/test_audio_router.py -v -s
```

---

## Test Metrics

Track test health over time:

- **Coverage:** Target 80%+ for core modules
- **Duration:** Keep test suite under 2 minutes
- **Flakiness:** Fix or mark flaky tests
- **Maintenance:** Update tests with code changes

### Coverage Goals

| Module | Current | Target |
|--------|---------|--------|
| Core Vision | 85% | 90% |
| Audio System | 78% | 85% |
| Navigation | 82% | 90% |
| Presentation | 65% | 75% |

---

## Troubleshooting Tests

### Common Issues

#### Import Errors

```bash
# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in editable mode
pip install -e .
```

#### Fixture Not Found

```bash
# Check conftest.py is present
ls tests/conftest.py

# Verify fixture name matches
pytest --fixtures  # List all fixtures
```

#### Tests Hang

```bash
# Set timeout
pytest tests/ --timeout=30

# Identify slow tests
pytest tests/ --durations=10
```

---

## Next Steps

- 📖 **Learn**: [Mock Observer Guide](../guides/MOCK_OBSERVER_GUIDE.md)
- 🎯 **Practice**: Run example tests in `examples/`
- 📝 **Contribute**: Add tests for new features
- 🔍 **Review**: Check [navigation_audio_testing.md](navigation_audio_testing.md)

---

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Coverage.py](https://coverage.readthedocs.io/)

---

**Ready to test?** Run `pytest tests/ -v` 🧪
