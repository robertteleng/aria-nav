#!/usr/bin/env python3
"""
📊 Phase 6 Hybrid Mode Benchmark
Measures performance improvements from CUDA Streams in multiprocessing mode.

Comparison:
- Baseline: Multiprocessing without CUDA Streams (sequential YOLO + Depth)
- Phase 6: Multiprocessing with CUDA Streams (parallel YOLO + Depth)

Expected improvement:
- RGB latency: 67ms → 40ms (~40% faster)
- System FPS: 18.4 → ~25 FPS (+36%)
- VRAM: 1.5GB → 2.0GB (+0.5GB)

Author: Roberto Rojas Sahuquillo
Date: January 2025
"""

import sys
import os

# Configure multiprocessing spawn BEFORE any torch imports
if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from utils.config import Config
from core.navigation.navigation_pipeline import NavigationPipeline
from core.vision.yolo_processor import YoloProcessor


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    vram_gb: float


def benchmark_pipeline(
    pipeline: NavigationPipeline,
    test_frames: List[np.ndarray],
    warmup: int = 10,
    iterations: int = 100
) -> List[float]:
    """
    Benchmark pipeline processing with multiple iterations.
    
    Args:
        pipeline: NavigationPipeline to benchmark
        test_frames: List of test frames to process
        warmup: Number of warmup iterations (discarded)
        iterations: Number of benchmark iterations
    
    Returns:
        List of processing times in milliseconds
    """
    timings = []
    
    # Warmup
    print(f"  Warming up ({warmup} iterations)...", end="", flush=True)
    for i in range(warmup):
        frame = test_frames[i % len(test_frames)]
        pipeline.process(frame)
    print(" Done")
    
    # Benchmark
    print(f"  Benchmarking ({iterations} iterations)...", end="", flush=True)
    for i in range(iterations):
        frame = test_frames[i % len(test_frames)]
        
        start = time.perf_counter()
        pipeline.process(frame)
        torch.cuda.synchronize()  # Wait for GPU to finish
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        
        timings.append(elapsed_ms)
    
    print(" Done")
    return timings


def analyze_timings(timings: List[float], name: str) -> BenchmarkResult:
    """Compute statistics from timing measurements"""
    timings_sorted = sorted(timings)
    n = len(timings)
    
    mean_ms = np.mean(timings)
    std_ms = np.std(timings)
    min_ms = min(timings)
    max_ms = max(timings)
    p50_ms = timings_sorted[int(n * 0.50)]
    p95_ms = timings_sorted[int(n * 0.95)]
    p99_ms = timings_sorted[int(n * 0.99)]
    
    # VRAM usage
    vram_gb = 0.0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / (1024**3)
    
    return BenchmarkResult(
        name=name,
        iterations=len(timings),
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        vram_gb=vram_gb
    )


def print_results(result: BenchmarkResult):
    """Pretty print benchmark results"""
    print(f"\n  {result.name}:")
    print(f"    Iterations: {result.iterations}")
    print(f"    Mean:       {result.mean_ms:.2f} ms ± {result.std_ms:.2f} ms")
    print(f"    Median:     {result.p50_ms:.2f} ms")
    print(f"    P95:        {result.p95_ms:.2f} ms")
    print(f"    P99:        {result.p99_ms:.2f} ms")
    print(f"    Min:        {result.min_ms:.2f} ms")
    print(f"    Max:        {result.max_ms:.2f} ms")
    print(f"    VRAM:       {result.vram_gb:.2f} GB")
    print(f"    FPS:        {1000.0 / result.mean_ms:.1f}")


def compare_results(baseline: BenchmarkResult, phase6: BenchmarkResult):
    """Compare baseline vs Phase 6 results"""
    print("\n" + "=" * 70)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 70)
    
    latency_improvement = (baseline.mean_ms - phase6.mean_ms) / baseline.mean_ms * 100
    fps_baseline = 1000.0 / baseline.mean_ms
    fps_phase6 = 1000.0 / phase6.mean_ms
    fps_improvement = (fps_phase6 - fps_baseline) / fps_baseline * 100
    vram_increase = phase6.vram_gb - baseline.vram_gb
    
    print(f"\n  Latency Reduction:")
    print(f"    Baseline:  {baseline.mean_ms:.2f} ms")
    print(f"    Phase 6:   {phase6.mean_ms:.2f} ms")
    print(f"    Improvement: {latency_improvement:+.1f}% ({baseline.mean_ms - phase6.mean_ms:+.2f} ms)")
    
    print(f"\n  FPS Increase:")
    print(f"    Baseline:  {fps_baseline:.1f} FPS")
    print(f"    Phase 6:   {fps_phase6:.1f} FPS")
    print(f"    Improvement: {fps_improvement:+.1f}% ({fps_phase6 - fps_baseline:+.1f} FPS)")
    
    print(f"\n  VRAM Usage:")
    print(f"    Baseline:  {baseline.vram_gb:.2f} GB")
    print(f"    Phase 6:   {phase6.vram_gb:.2f} GB")
    print(f"    Increase:  {vram_increase:+.2f} GB")
    
    # Target validation
    print(f"\n  Target Validation:")
    target_latency = 40.0  # ms
    target_fps = 25.0
    target_vram = 2.5  # GB
    
    if phase6.mean_ms <= target_latency:
        print(f"    ✅ Latency target met: {phase6.mean_ms:.2f} ms ≤ {target_latency} ms")
    else:
        print(f"    ❌ Latency target missed: {phase6.mean_ms:.2f} ms > {target_latency} ms")
    
    if fps_phase6 >= target_fps:
        print(f"    ✅ FPS target met: {fps_phase6:.1f} FPS ≥ {target_fps} FPS")
    else:
        print(f"    ⚠️  FPS target missed: {fps_phase6:.1f} FPS < {target_fps} FPS")
    
    if phase6.vram_gb <= target_vram:
        print(f"    ✅ VRAM target met: {phase6.vram_gb:.2f} GB ≤ {target_vram} GB")
    else:
        print(f"    ⚠️  VRAM exceeded: {phase6.vram_gb:.2f} GB > {target_vram} GB")
    
    # Overall assessment
    print(f"\n  Overall Assessment:")
    if latency_improvement > 20 and phase6.vram_gb <= target_vram:
        print(f"    ✅ Phase 6 provides significant improvement with acceptable memory cost")
    elif latency_improvement > 10:
        print(f"    ⚠️  Phase 6 provides moderate improvement")
    else:
        print(f"    ❌ Phase 6 improvement below expectations")


def main():
    """Run Phase 6 benchmark"""
    print("\n" + "=" * 70)
    print("📊 PHASE 6 HYBRID MODE - PERFORMANCE BENCHMARK")
    print("=" * 70)
    print("\nObjective: Measure performance improvements from CUDA Streams")
    print("Expected: 67ms → 40ms latency, 18.4 → 25 FPS\n")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot benchmark")
        return 1
    
    # Configuration
    warmup_iterations = 10
    benchmark_iterations = 100
    num_test_frames = 10
    
    print(f"Configuration:")
    print(f"  Warmup iterations: {warmup_iterations}")
    print(f"  Benchmark iterations: {benchmark_iterations}")
    print(f"  Test frames: {num_test_frames}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    
    # Create test frames (1408x1408 for RGB profile)
    print(f"\n📸 Generating test frames...")
    test_frames = [
        np.random.randint(0, 255, (1408, 1408, 3), dtype=np.uint8)
        for _ in range(num_test_frames)
    ]
    print(f"  ✅ Generated {num_test_frames} frames of shape {test_frames[0].shape}")
    
    # ========================================
    # Benchmark 1: Baseline (Streams disabled)
    # ========================================
    print("\n" + "=" * 70)
    print("🔍 BENCHMARK 1: Baseline (Sequential Processing)")
    print("=" * 70)
    print("  Mode: Multiprocessing WITHOUT CUDA Streams")
    print("  Expected: ~67ms (YOLO 40ms + Depth 27ms sequential)")
    
    # Temporarily disable Phase 6
    original_phase6 = getattr(Config, 'PHASE6_HYBRID_STREAMS', False)
    Config.PHASE6_HYBRID_STREAMS = False
    
    yolo_baseline = YoloProcessor.from_profile("rgb")
    pipeline_baseline = NavigationPipeline(
        yolo_processor=yolo_baseline,
        camera_id='rgb'
    )
    
    print(f"\n  Pipeline configuration:")
    print(f"    CUDA Streams: {pipeline_baseline.use_cuda_streams}")
    print(f"    Multiprocessing: {pipeline_baseline.multiproc_enabled}")
    
    timings_baseline = benchmark_pipeline(
        pipeline_baseline,
        test_frames,
        warmup=warmup_iterations,
        iterations=benchmark_iterations
    )
    result_baseline = analyze_timings(timings_baseline, "Baseline")
    print_results(result_baseline)
    
    # Cleanup
    del pipeline_baseline
    del yolo_baseline
    torch.cuda.empty_cache()
    
    # ========================================
    # Benchmark 2: Phase 6 (Streams enabled)
    # ========================================
    print("\n" + "=" * 70)
    print("🚀 BENCHMARK 2: Phase 6 (Parallel Processing)")
    print("=" * 70)
    print("  Mode: Multiprocessing WITH CUDA Streams")
    print("  Expected: ~40ms (YOLO + Depth parallel)")
    
    # Re-enable Phase 6
    Config.PHASE6_HYBRID_STREAMS = True
    
    yolo_phase6 = YoloProcessor.from_profile("rgb")
    pipeline_phase6 = NavigationPipeline(
        yolo_processor=yolo_phase6,
        camera_id='rgb'
    )
    
    print(f"\n  Pipeline configuration:")
    print(f"    CUDA Streams: {pipeline_phase6.use_cuda_streams}")
    print(f"    Multiprocessing: {pipeline_phase6.multiproc_enabled}")
    
    timings_phase6 = benchmark_pipeline(
        pipeline_phase6,
        test_frames,
        warmup=warmup_iterations,
        iterations=benchmark_iterations
    )
    result_phase6 = analyze_timings(timings_phase6, "Phase 6")
    print_results(result_phase6)
    
    # Restore original Phase 6 setting
    Config.PHASE6_HYBRID_STREAMS = original_phase6
    
    # ========================================
    # Comparison
    # ========================================
    compare_results(result_baseline, result_phase6)
    
    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("\n💡 Next steps:")
    print("  1. If results are good, test full system: python run.py")
    print("  2. Monitor VRAM during full run: nvidia-smi")
    print("  3. Check for memory leaks in long sessions (10+ minutes)")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
