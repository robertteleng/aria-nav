"""Test básico de integración - valida componentes principales"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_logging_setup():
    """Verifica que el logging se inicializa correctamente"""
    from core.telemetry.telemetry_logger import TelemetryLogger
    from core.telemetry.loggers.navigation_logger import get_navigation_logger
    from core.telemetry.loggers.depth_logger import get_depth_logger
    
    # Create telemetry logger
    telemetry = TelemetryLogger()
    print(f"✓ TelemetryLogger created: {telemetry.session_dir}")
    
    # Create navigation logger
    nav_logger = get_navigation_logger(session_dir=telemetry.session_dir)
    print(f"✓ NavigationLogger created: {nav_logger.log_dir}")
    
    # Create depth logger
    depth_logger = get_depth_logger(session_dir=str(telemetry.session_dir))
    print(f"✓ DepthLogger created: {depth_logger.log_dir}")
    
    # Log test entries
    telemetry.log_frame_performance(frame_number=1, fps=30.0, latency_ms=33.3)
    nav_logger.decision.info("Test decision log")
    depth_logger.log("Test depth log")
    
    print(f"\n✅ All loggers working - check logs in: {telemetry.session_dir}")
    return True


def test_tensorrt_models():
    """Verifica que los modelos TensorRT se pueden cargar"""
    from pathlib import Path
    
    checkpoints = Path("checkpoints")
    yolo_engine = checkpoints / "yolo12n.engine"
    depth_engine = checkpoints / "depth_anything_v2_vits.engine"
    
    if not yolo_engine.exists():
        print(f"⚠️  YOLO engine not found: {yolo_engine}")
        return False
    
    if not depth_engine.exists():
        print(f"⚠️  Depth engine not found: {depth_engine}")
        return False
    
    print(f"✓ YOLO engine found: {yolo_engine}")
    print(f"✓ Depth engine found: {depth_engine}")
    
    # Try loading (without running inference)
    try:
        import tensorrt as trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(yolo_engine, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
            print(f"✓ YOLO engine loaded successfully")
            
        with open(depth_engine, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
            print(f"✓ Depth engine loaded successfully")
            
        print("\n✅ TensorRT models are valid")
        return True
        
    except Exception as e:
        print(f"❌ TensorRT error: {e}")
        return False


def test_config():
    """Verifica que la configuración es válida"""
    from utils.config import (
        YOLO_CONFIDENCE_THRESHOLD,
        NORMAL_CENTER_TOLERANCE,
        CRITICAL_CENTER_TOLERANCE,
        AUDIO_GLOBAL_COOLDOWN
    )
    
    print(f"✓ YOLO confidence: {YOLO_CONFIDENCE_THRESHOLD}")
    print(f"✓ Normal tolerance: {NORMAL_CENTER_TOLERANCE}")
    print(f"✓ Critical tolerance: {CRITICAL_CENTER_TOLERANCE}")
    print(f"✓ Audio cooldown: {AUDIO_GLOBAL_COOLDOWN}s")
    
    print("\n✅ Configuration loaded")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("ARIA-NAV BASIC INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_config),
        ("Logging Setup", test_logging_setup),
        ("TensorRT Models", test_tensorrt_models),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"TEST: {name}")
        print("=" * 60)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print(f"\n\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(r for _, r in results)
    if all_passed:
        print(f"\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  Some tests failed")
        sys.exit(1)
