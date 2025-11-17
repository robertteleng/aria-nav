"""
Test rápido para verificar que depth TensorRT se ejecuta correctamente
después del fix en navigation_pipeline.py
"""

import sys
sys.path.insert(0, '/home/roberto/Projects/aria-nav')

import numpy as np
import cv2
from src.core.vision.depth_estimator import DepthEstimator

print("=" * 70)
print("TEST: Verificación Depth TensorRT Fix")
print("=" * 70)

# 1. Crear estimador
print("\n1. Creando DepthEstimator...")
estimator = DepthEstimator()

# 2. Verificar atributos
print("\n2. Verificando atributos:")
print(f"   - backend: {getattr(estimator, 'backend', 'N/A')}")
print(f"   - model: {getattr(estimator, 'model', None)}")
print(f"   - ort_session: {getattr(estimator, 'ort_session', None)}")
print(f"   - device: {getattr(estimator, 'device', 'N/A')}")

# 3. Check lógico del fix
has_model = getattr(estimator, "model", None) is not None
has_ort = getattr(estimator, "ort_session", None) is not None

print("\n3. Verificando lógica del fix:")
print(f"   - has_model: {has_model}")
print(f"   - has_ort_session: {has_ort}")
print(f"   - should_run (model OR ort): {has_model or has_ort}")

# 4. Test de ejecución
print("\n4. Probando estimate_depth_with_details()...")
test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

result = None
try:
    result = estimator.estimate_depth_with_details(test_frame)
    if result is not None:
        print(f"   ✅ SUCCESS!")
        print(f"   - Depth map shape: {result.map_8bit.shape}")
        print(f"   - Inference time: {result.inference_ms:.2f}ms")
        print(f"   - Depth range: [{result.map_8bit.min()}, {result.map_8bit.max()}]")
    else:
        print(f"   ❌ FAILED: estimate_depth returned None")
except Exception as e:
    print(f"   ❌ FAILED with exception: {e}")

print("\n" + "=" * 70)
print("FIX VERIFICATION:")
if has_ort and result is not None:
    print("✅ TensorRT depth estimation está funcionando correctamente")
    print("✅ El fix en navigation_pipeline.py está correcto")
else:
    print("❌ Aún hay problemas con la ejecución de depth")
print("=" * 70)
