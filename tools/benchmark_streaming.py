#!/usr/bin/env python3
"""
Benchmark para identificar si el streaming de Aria es el bottleneck

Ejecutar:
    python tools/benchmark_streaming.py --mock    # Con MockObserver
    python tools/benchmark_streaming.py           # Con gafas reales

Interpretación:
    Capture latency <5ms:  ✅ Streaming NO es bottleneck
    Capture latency 5-10ms: ⚠️ Podría mejorar
    Capture latency >10ms:  ❌ Streaming ES bottleneck (considerar FASE 3)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from core.mock_observer import MockObserver
from core.observer import Observer
from core.hardware.device_manager import DeviceManager


def benchmark_capture_latency(observer, duration_sec=10, samples=100):
    """
    Mide la latencia de captura de frames
    
    Returns:
        dict con estadísticas de latencia
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Capture Latency")
    print(f"{'='*60}\n")
    
    latencies = []
    frame_sizes = []
    
    print(f"[1/2] Calentamiento (3s)...")
    time.sleep(3)
    
    print(f"[2/2] Capturando {samples} frames...")
    
    for i in range(samples):
        # Medir latencia de get_latest_frame()
        start = time.perf_counter()
        frame = observer.get_latest_frame('rgb')
        latency = (time.perf_counter() - start) * 1000  # ms
        
        if frame is not None:
            latencies.append(latency)
            frame_sizes.append(frame.nbytes / 1024)  # KB
            
        # Progress
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{samples} frames")
        
        # Throttle para no saturar
        time.sleep(0.01)
    
    # Estadísticas
    latencies = np.array(latencies)
    frame_sizes = np.array(frame_sizes)
    
    stats = {
        'mean_latency_ms': np.mean(latencies),
        'median_latency_ms': np.median(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'max_latency_ms': np.max(latencies),
        'std_latency_ms': np.std(latencies),
        'mean_frame_size_kb': np.mean(frame_sizes),
        'total_samples': len(latencies),
    }
    
    return stats, latencies


def benchmark_frame_availability(observer, duration_sec=10):
    """
    Mide qué tan rápido llegan frames nuevos
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Frame Delivery Rate")
    print(f"{'='*60}\n")
    
    print(f"Monitoreando durante {duration_sec}s...")
    
    frame_timestamps = []
    last_frame = None
    start_time = time.time()
    
    while time.time() - start_time < duration_sec:
        frame = observer.get_latest_frame('rgb')
        
        # Detectar frame nuevo (comparar identidad)
        if frame is not None and (last_frame is None or frame is not last_frame):
            frame_timestamps.append(time.time())
            last_frame = frame
        
        time.sleep(0.001)  # Poll cada 1ms
    
    # Calcular intervalos entre frames
    if len(frame_timestamps) > 1:
        intervals = np.diff(frame_timestamps) * 1000  # ms
        fps = len(frame_timestamps) / duration_sec
        
        stats = {
            'fps': fps,
            'mean_interval_ms': np.mean(intervals),
            'std_interval_ms': np.std(intervals),
            'min_interval_ms': np.min(intervals),
            'max_interval_ms': np.max(intervals),
            'total_frames': len(frame_timestamps),
        }
    else:
        stats = {'error': 'No frames received'}
    
    return stats


def benchmark_throughput(observer, duration_sec=5):
    """
    Mide throughput máximo de captura
    """
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Maximum Throughput")
    print(f"{'='*60}\n")
    
    print(f"Captura continua durante {duration_sec}s...")
    
    frame_count = 0
    bytes_transferred = 0
    start_time = time.time()
    
    while time.time() - start_time < duration_sec:
        frame = observer.get_latest_frame('rgb')
        if frame is not None:
            frame_count += 1
            bytes_transferred += frame.nbytes
    
    elapsed = time.time() - start_time
    
    stats = {
        'throughput_fps': frame_count / elapsed,
        'throughput_mbps': (bytes_transferred / elapsed) / (1024 * 1024),
        'total_frames': frame_count,
        'total_mb': bytes_transferred / (1024 * 1024),
    }
    
    return stats


def print_results(capture_stats, delivery_stats, throughput_stats):
    """
    Imprime resultados y diagnóstico
    """
    print(f"\n{'='*60}")
    print(f"📊 RESULTADOS")
    print(f"{'='*60}\n")
    
    # Capture Latency
    print("1️⃣  CAPTURE LATENCY (get_latest_frame)")
    print(f"   Mean:   {capture_stats['mean_latency_ms']:.2f}ms")
    print(f"   Median: {capture_stats['median_latency_ms']:.2f}ms")
    print(f"   P95:    {capture_stats['p95_latency_ms']:.2f}ms")
    print(f"   P99:    {capture_stats['p99_latency_ms']:.2f}ms")
    print(f"   Max:    {capture_stats['max_latency_ms']:.2f}ms")
    print(f"   Std:    {capture_stats['std_latency_ms']:.2f}ms")
    
    # Frame Delivery
    print(f"\n2️⃣  FRAME DELIVERY RATE")
    if 'fps' in delivery_stats:
        print(f"   FPS:              {delivery_stats['fps']:.1f}")
        print(f"   Mean interval:    {delivery_stats['mean_interval_ms']:.2f}ms")
        print(f"   Interval std:     {delivery_stats['std_interval_ms']:.2f}ms")
        print(f"   Min/Max interval: {delivery_stats['min_interval_ms']:.2f} / {delivery_stats['max_interval_ms']:.2f}ms")
    else:
        print(f"   ERROR: {delivery_stats.get('error', 'Unknown')}")
    
    # Throughput
    print(f"\n3️⃣  MAXIMUM THROUGHPUT")
    print(f"   FPS:        {throughput_stats['throughput_fps']:.1f}")
    print(f"   Bandwidth:  {throughput_stats['throughput_mbps']:.2f} MB/s")
    print(f"   Total:      {throughput_stats['total_frames']} frames, {throughput_stats['total_mb']:.1f} MB")
    
    # Diagnóstico
    print(f"\n{'='*60}")
    print(f"🔍 DIAGNÓSTICO")
    print(f"{'='*60}\n")
    
    mean_latency = capture_stats['mean_latency_ms']
    p95_latency = capture_stats['p95_latency_ms']
    
    if mean_latency < 2.0 and p95_latency < 5.0:
        diagnosis = "✅ EXCELENTE"
        detail = "Streaming NO es bottleneck. Latencia mínima."
        recommendation = "SKIP FASE 3. Ir directo a FASE 4 (TensorRT)."
        color = "🟢"
    elif mean_latency < 5.0 and p95_latency < 10.0:
        diagnosis = "✅ BUENO"
        detail = "Streaming NO es bottleneck principal."
        recommendation = "FASE 3 opcional. FASE 4 dará más ganancia."
        color = "🟢"
    elif mean_latency < 10.0 and p95_latency < 15.0:
        diagnosis = "⚠️ ACEPTABLE"
        detail = "Streaming tiene overhead moderado."
        recommendation = "Evaluar FASE 3 DESPUÉS de FASE 4. Medir nuevo baseline."
        color = "🟡"
    else:
        diagnosis = "❌ BOTTLENECK DETECTADO"
        detail = "Streaming tiene latencia alta."
        recommendation = "CONSIDERAR FASE 3 (GStreamer) para reducir overhead de captura."
        color = "🔴"
    
    print(f"{color} {diagnosis}")
    print(f"   {detail}")
    print(f"\n   📋 Recomendación:")
    print(f"   {recommendation}")
    
    # Frame size
    frame_size = capture_stats['mean_frame_size_kb']
    print(f"\n   📦 Frame size: {frame_size:.1f} KB")
    if frame_size > 5000:  # >5MB
        print(f"      ⚠️ Frames grandes. Considerar reducir resolución o comprimir.")
    
    # Jitter
    if 'std_interval_ms' in delivery_stats:
        jitter = delivery_stats['std_interval_ms']
        if jitter > 10:
            print(f"\n   ⚠️ Jitter alto ({jitter:.1f}ms). Frames no llegan de manera consistente.")
            print(f"      Posible causa: Observer en threading (GIL) o DDS congestion.")
    
    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark streaming latency")
    parser.add_argument('--mock', action='store_true', help='Use MockObserver')
    parser.add_argument('--duration', type=int, default=10, help='Duration in seconds')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples')
    args = parser.parse_args()
    
    observer = None
    device_manager = None
    
    try:
        print(f"\n{'='*60}")
        print(f"🔍 STREAMING BOTTLENECK ANALYZER")
        print(f"{'='*60}\n")
        
        # Setup observer
        if args.mock:
            print("📋 Modo: MockObserver")
            observer = MockObserver(mode='synthetic', fps=30, resolution=(1408, 1408))
            observer.start()
            print("✅ MockObserver iniciado\n")
        else:
            print("📋 Modo: Aria SDK (hardware real)")
            device_manager = DeviceManager()
            rgb_calib = device_manager.start_streaming()
            observer = Observer(rgb_calib=rgb_calib)
            print("✅ Aria streaming iniciado\n")
        
        # Run benchmarks
        capture_stats, latencies = benchmark_capture_latency(
            observer, 
            duration_sec=args.duration, 
            samples=args.samples
        )
        
        delivery_stats = benchmark_frame_availability(
            observer, 
            duration_sec=args.duration
        )
        
        throughput_stats = benchmark_throughput(
            observer, 
            duration_sec=5
        )
        
        # Print results
        print_results(capture_stats, delivery_stats, throughput_stats)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Benchmark interrumpido por usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if observer is not None:
            if hasattr(observer, 'stop'):
                observer.stop()
        if device_manager is not None:
            device_manager.cleanup()
        
        print("✅ Cleanup completo")


if __name__ == '__main__':
    main()
