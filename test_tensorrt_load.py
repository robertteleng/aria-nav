#!/usr/bin/env python3
"""Test script to verify TensorRT engine loading"""

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.utils.config import Config
from src.core.vision.yolo_processor import YOLOProcessor

print("\n" + "="*60)
print("Testing TensorRT YOLO Loading")
print("="*60)

config = Config()
print(f"\nConfig:")
print(f"  USE_TENSORRT: {config.USE_TENSORRT}")
print(f"  YOLO_MODEL: {config.YOLO_MODEL}")
print(f"  DEPTH_ENABLED: {config.DEPTH_ENABLED}")
print(f"  PHASE2_MULTIPROC_ENABLED: {config.PHASE2_MULTIPROC_ENABLED}")

print(f"\nInitializing YOLO Processor...")
processor = YOLOProcessor(config)

print(f"\n✓ Processor initialized successfully")
print(f"  Model: {processor.model}")
print(f"  Device: {processor.device_str}")

# Run a quick inference test
import numpy as np
dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

print(f"\nRunning test inference...")
import time
start = time.perf_counter()
results = processor.process(dummy_frame, 1, "rgb")
elapsed = (time.perf_counter() - start) * 1000
print(f"  Inference time: {elapsed:.2f}ms")
print(f"  Detections: {len(results)}")

print("\n" + "="*60)
print("✓ Test completed successfully")
print("="*60 + "\n")
