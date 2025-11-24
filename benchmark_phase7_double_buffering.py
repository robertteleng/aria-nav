#!/usr/bin/env python3
"""
📊 Phase 7 Double Buffering Benchmark
Measures performance improvements from Double Buffering (2x workers).

Comparison:
- Baseline: Single Buffering (1x CentralWorker, 1x SlamWorker)
- Phase 7: Double Buffering (2x CentralWorker, 2x SlamWorker)

Expected improvement:
- FPS: 19.0 → ~25.0 FPS (+31%)
- VRAM: 1.5GB → ~2.4-3.0GB
- Latency: Reduced due to parallel processing

Author: Roberto Rojas Sahuquillo
Date: January 2025
"""

import sys
import os
import argparse
import time
import signal
# import psutil

# Configure multiprocessing spawn BEFORE any torch imports
if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from utils.config import Config
from core.navigation.navigation_pipeline import NavigationPipeline
from core.vision.yolo_processor import YoloProcessor

@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    fps: float
    vram_gb: float

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0

def benchmark_pipeline(pipeline, test_frames, iterations=100, warmup=10):
    print(f"  Warming up ({warmup} iterations)...", end="", flush=True)
    for i in range(warmup):
        pipeline.process(test_frames[i % len(test_frames)])
    print(" Done")
    
    print(f"  Benchmarking ({iterations} iterations)...", end="", flush=True)
    start_time = time.time()
    for i in range(iterations):
        pipeline.process(test_frames[i % len(test_frames)])
        # Simulate camera frame rate (don't push faster than 30fps input)
        # time.sleep(0.033) 
    
    total_time = time.time() - start_time
    fps = iterations / total_time
    mean_ms = (total_time / iterations) * 1000
    
    print(f" Done ({fps:.1f} FPS)")
    return fps, mean_ms

def run_baseline_benchmark():
    print("\n" + "=" * 70)
    print("🔍 BENCHMARK: Baseline (Single Buffering)")
    print("=" * 70)
    
    # Disable double buffering
    Config.PHASE7_DOUBLE_BUFFERING = False
    
    try:
        # Initialize pipeline
        yolo = YoloProcessor.from_profile("rgb")
        pipeline = NavigationPipeline(yolo_processor=yolo, camera_id='rgb')
        
        # Create test frames
        test_frames = [np.random.randint(0, 255, (1408, 1408, 3), dtype=np.uint8) for _ in range(10)]
        
        # Benchmark
        fps, mean_ms = benchmark_pipeline(pipeline, test_frames, iterations=100)
        vram = get_vram_usage()
        
        print(f"\n  Results (Baseline):")
        print(f"    FPS:        {fps:.1f}")
        print(f"    Latency:    {mean_ms:.1f} ms")
        print(f"    VRAM:       {vram:.2f} GB")
        
        pipeline.shutdown()
        return BenchmarkResult("Baseline", 100, mean_ms, 0, fps, vram)
        
    except Exception as e:
        print(f"Error in baseline benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_double_buffering_benchmark():
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK: Phase 7 (Double Buffering)")
    print("=" * 70)
    
    # Enable double buffering
    Config.PHASE7_DOUBLE_BUFFERING = True
    
    try:
        # Initialize pipeline
        yolo = YoloProcessor.from_profile("rgb")
        pipeline = NavigationPipeline(yolo_processor=yolo, camera_id='rgb')
        
        # Verify worker count
        worker_count = len(pipeline.workers)
        print(f"  Active workers: {worker_count} (Expected: 4)")
        if worker_count != 4:
            print("  ⚠️ WARNING: Worker count mismatch!")
        
        # Create test frames
        test_frames = [np.random.randint(0, 255, (1408, 1408, 3), dtype=np.uint8) for _ in range(10)]
        
        # Benchmark
        fps, mean_ms = benchmark_pipeline(pipeline, test_frames, iterations=200) # More iterations for stability
        vram = get_vram_usage()
        
        print(f"\n  Results (Double Buffering):")
        print(f"    FPS:        {fps:.1f}")
        print(f"    Latency:    {mean_ms:.1f} ms")
        print(f"    VRAM:       {vram:.2f} GB")
        
        pipeline.shutdown()
        return BenchmarkResult("Double Buffering", 200, mean_ms, 0, fps, vram)
        
    except Exception as e:
        print(f"Error in double buffering benchmark: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_crash_test():
    print("\n" + "=" * 70)
    print("💥 TEST: Worker Crash Recovery")
    print("=" * 70)
    
    Config.PHASE7_DOUBLE_BUFFERING = True
    
    try:
        yolo = YoloProcessor.from_profile("rgb")
        pipeline = NavigationPipeline(yolo_processor=yolo, camera_id='rgb')
        
        test_frames = [np.random.randint(0, 255, (1408, 1408, 3), dtype=np.uint8) for _ in range(10)]
        
        print("  1. Running normal operation...")
        benchmark_pipeline(pipeline, test_frames, iterations=20, warmup=0)
        
        print("  2. Killing one CentralWorker...")
        # Find a central worker process
        target_worker = pipeline.worker_instances["central"][0]["proc"]
        print(f"     Terminating {target_worker.name} (PID: {target_worker.pid})")
        os.kill(target_worker.pid, signal.SIGKILL)
        time.sleep(1) # Allow system to detect
        
        print("  3. Running post-crash operation...")
        fps, _ = benchmark_pipeline(pipeline, test_frames, iterations=50, warmup=0)
        
        print(f"  4. Post-crash FPS: {fps:.1f}")
        
        # Verify health check caught it
        pipeline._monitor_worker_health()
        if not pipeline.worker_health[target_worker.name]:
            print("  ✅ Health check correctly identified crashed worker")
        else:
            print("  ❌ Health check failed to identify crash")
            
        pipeline.shutdown()
        
    except Exception as e:
        print(f"Error in crash test: {e}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Phase 7 Benchmark")
    parser.add_argument("--mode", type=str, choices=["baseline", "double_buffering", "crash_test", "all"], default="all")
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot benchmark")
        return
        
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    baseline_res = None
    phase7_res = None
    
    if args.mode in ["baseline", "all"]:
        baseline_res = run_baseline_benchmark()
        # Cleanup between runs
        torch.cuda.empty_cache()
        time.sleep(2)
        
    if args.mode in ["double_buffering", "all"]:
        phase7_res = run_double_buffering_benchmark()
        torch.cuda.empty_cache()
        time.sleep(2)
        
    if args.mode in ["crash_test", "all"]:
        run_crash_test()
        
    # Comparison
    if baseline_res and phase7_res:
        print("\n" + "=" * 70)
        print("📊 FINAL COMPARISON")
        print("=" * 70)
        
        fps_improvement = (phase7_res.fps - baseline_res.fps) / baseline_res.fps * 100
        vram_increase = phase7_res.vram_gb - baseline_res.vram_gb
        
        print(f"FPS Improvement: {fps_improvement:+.1f}% ({baseline_res.fps:.1f} → {phase7_res.fps:.1f} FPS)")
        print(f"VRAM Increase:   {vram_increase:+.2f} GB ({baseline_res.vram_gb:.2f} → {phase7_res.vram_gb:.2f} GB)")
        
        if phase7_res.fps >= 24.0:
            print("✅ Target FPS met (~25 FPS)")
        else:
            print("⚠️ Target FPS missed")
            
        if phase7_res.vram_gb < 6.0:
            print("✅ VRAM within limits (<6GB)")
        else:
            print("❌ VRAM exceeded limits!")

if __name__ == "__main__":
    main()
