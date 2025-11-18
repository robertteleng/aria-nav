#!/usr/bin/env python3
"""
Test de stress para AsyncTelemetryLogger.
Simula carga alta de escritura para validar que no hay bloqueos.
"""

import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.telemetry.telemetry_logger import AsyncTelemetryLogger

def main():
    print("\n🔥 AsyncTelemetryLogger Stress Test")
    print("=" * 60)
    
    telemetry = AsyncTelemetryLogger(
        flush_interval=2.0,
        buffer_size=100,
        queue_maxsize=2000
    )
    
    print(f"Session: {telemetry.session_id}")
    print(f"Target: 1000 frames @ 20 FPS simulation\n")
    
    # Simular 1000 frames a ~20 FPS
    num_frames = 1000
    target_fps = 20
    frame_time = 1.0 / target_fps
    
    start = time.time()
    total_log_time = 0
    
    for frame_num in range(num_frames):
        frame_start = time.time()
        
        # Timing de logging (no debería bloquear)
        log_start = time.time()
        
        # Performance metric (siempre)
        telemetry.log_frame_performance(
            frame_number=frame_num,
            fps=random.uniform(18, 22),
            latency_ms=random.uniform(40, 55)
        )
        
        # Detections (probabilidad alta)
        if random.random() < 0.7:  # 70% de frames con detección
            for _ in range(random.randint(1, 3)):  # 1-3 objetos
                telemetry.log_detection(
                    frame_number=frame_num,
                    source=random.choice(["rgb", "slam1", "slam2"]),
                    object_class=random.choice(["person", "chair", "door", "table"]),
                    confidence=random.uniform(0.5, 0.95),
                    distance_normalized=random.uniform(0.1, 0.9)
                )
        
        # Audio events (menos frecuentes)
        if random.random() < 0.1:  # 10% de frames
            telemetry.log_audio_event(
                action="enqueued",
                source=random.choice(["rgb", "slam1", "slam2"]),
                priority=random.randint(1, 4),
                message=f"Object detected at frame {frame_num}"
            )
        
        log_time = time.time() - log_start
        total_log_time += log_time
        
        # Warning si logging toma mucho tiempo (>5ms = problema)
        if log_time > 0.005:
            print(f"⚠️  Frame {frame_num}: log took {log_time*1000:.1f}ms")
        
        # Simular tiempo de procesamiento restante
        elapsed = time.time() - frame_start
        sleep_time = max(0, frame_time - elapsed)
        time.sleep(sleep_time)
        
        # Progress cada 100 frames
        if (frame_num + 1) % 100 == 0:
            print(f"  ✓ Frame {frame_num + 1}/{num_frames}")
    
    total_time = time.time() - start
    actual_fps = num_frames / total_time
    avg_log_time = (total_log_time / num_frames) * 1000
    
    print(f"\n{'=' * 60}")
    print(f"✅ Stress test complete!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Actual FPS: {actual_fps:.1f}")
    print(f"  Avg log time: {avg_log_time:.3f}ms/frame")
    print(f"  Log overhead: {(total_log_time/total_time)*100:.1f}%")
    
    # Esperar flush final
    print(f"\n⏳ Waiting for final flush (2s)...")
    time.sleep(2.5)
    
    # Verificar resultados
    print(f"\n📊 Results:")
    perf_lines = len(telemetry.performance_log.read_text().strip().split('\n'))
    det_lines = len(telemetry.detections_log.read_text().strip().split('\n'))
    audio_lines = len(telemetry.audio_log.read_text().strip().split('\n'))
    
    print(f"  Performance: {perf_lines} lines (expected: {num_frames})")
    print(f"  Detections:  {det_lines} lines")
    print(f"  Audio:       {audio_lines} lines")
    
    if perf_lines == num_frames:
        print(f"\n🎉 SUCCESS: All {num_frames} frames logged without loss!")
    else:
        print(f"\n⚠️  WARNING: Lost {num_frames - perf_lines} frames")
    
    print(f"\nLogs: {telemetry.output_dir}")

if __name__ == "__main__":
    main()
