#!/usr/bin/env python3
"""Quick test: YOLO with TensorRT engine"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import time
from core.vision.yolo_processor import YoloProcessor

print("🔧 Creating YoloProcessor with TensorRT...")
processor = YoloProcessor(profile="rgb")

print("\n📊 Running inference test...")
# Create dummy frame
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# Warmup
print("  Warmup...")
for _ in range(3):
    processor.process_frame(frame)

# Benchmark
print("  Benchmarking 30 frames...")
times = []
for i in range(30):
    start = time.perf_counter()
    detections = processor.process_frame(frame)
    elapsed = time.perf_counter() - start
    times.append(elapsed)
    if i % 10 == 0:
        print(f"    Frame {i}: {elapsed*1000:.1f}ms")

avg_ms = np.mean(times) * 1000
fps = 1000 / avg_ms
print(f"\n✅ Results:")
print(f"   Avg inference: {avg_ms:.1f}ms")
print(f"   Throughput: {fps:.1f} FPS")
print(f"   Min: {min(times)*1000:.1f}ms | Max: {max(times)*1000:.1f}ms")
