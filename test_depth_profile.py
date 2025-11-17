#!/usr/bin/env python3
"""Quick test to profile depth TensorRT inference."""

import sys
sys.path.insert(0, '/home/roberto/Projects/aria-nav/src')

from core.vision.depth_estimator import DepthEstimator
import numpy as np
import time

print('='*70)
print('TEST: Depth TensorRT Performance Profiling')
print('='*70)

estimator = DepthEstimator()

print(f'\nConfig:')
print(f'  input_size: {estimator.input_size}')
print(f'  backend: {estimator.backend}')
print(f'  ort_session: {estimator.ort_session is not None}')
print(f'  model: {estimator.model is not None}')

# Test con frame tamaño Aria (1408x1408)
test_frame = np.random.randint(0, 255, (1408, 1408, 3), dtype=np.uint8)
print(f'\nTest frame shape: {test_frame.shape}')

# Warmup
print(f'\nWarmup (3 iterations)...')
for i in range(3):
    _ = estimator.estimate_depth_with_details(test_frame)

# Benchmark con profiling
print(f'\nBenchmark (10 iterations)...')
print('-'*70)
times = []
for i in range(10):
    start = time.perf_counter()
    result = estimator.estimate_depth_with_details(test_frame)
    elapsed = (time.perf_counter() - start) * 1000
    times.append(elapsed)
    print(f'Iteration {i+1}: TOTAL={elapsed:.1f}ms | shape={result.map_8bit.shape}')

avg = sum(times) / len(times)
print('='*70)
print(f'RESULTS:')
print(f'  Average: {avg:.1f}ms')
print(f'  Min: {min(times):.1f}ms')
print(f'  Max: {max(times):.1f}ms')
print(f'  Output shape: {result.map_8bit.shape}')
print('='*70)

if avg < 50:
    print('✅ EXCELLENT - TensorRT optimization working!')
elif avg < 100:
    print('✅ GOOD - Decent performance')
else:
    print('⚠️  SLOW - Expected ~30-35ms, investigation needed')
