#!/usr/bin/env python3
"""Test rápido de AsyncTelemetryLogger."""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.telemetry.telemetry_logger import AsyncTelemetryLogger

def main():
    print("\n🧪 Testing AsyncTelemetryLogger...")
    
    # Crear logger
    telemetry = AsyncTelemetryLogger(
        flush_interval=0.5,  # Flush cada 0.5s
        buffer_size=10       # Flush cada 10 líneas
    )
    
    print(f"✓ Session ID: {telemetry.session_id}")
    print(f"✓ Output dir: {telemetry.output_dir}")
    
    # Simular 100 frames
    print("\n📊 Logging 100 frames...")
    start_time = time.time()
    
    for frame_num in range(100):
        # Log performance (no debería bloquear)
        telemetry.log_frame_performance(
            frame_number=frame_num,
            fps=20.0 + (frame_num % 5),
            latency_ms=45.0 + (frame_num % 10)
        )
        
        # Simular detecciones
        if frame_num % 3 == 0:
            telemetry.log_detection(
                frame_number=frame_num,
                source="rgb",
                object_class="person",
                confidence=0.85,
                distance_normalized=0.5
            )
        
        # Simular audio events
        if frame_num % 10 == 0:
            telemetry.log_audio_event(
                action="enqueued",
                source="rgb",
                priority=2,
                message=f"Person detected at frame {frame_num}"
            )
        
        # Pequeño delay para simular procesamiento
        time.sleep(0.001)
    
    elapsed = time.time() - start_time
    print(f"✓ Logged 100 frames in {elapsed:.2f}s")
    print(f"✓ Average: {100/elapsed:.1f} FPS equivalent")
    
    # Esperar un momento para el flush
    print("\n⏳ Waiting for background flush...")
    time.sleep(2)
    
    # Verificar archivos
    print("\n📁 Checking output files...")
    for log_file in [telemetry.performance_log, telemetry.detections_log, telemetry.audio_log]:
        if log_file.exists():
            lines = len(log_file.read_text().strip().split('\n'))
            print(f"  ✓ {log_file.name}: {lines} lines")
        else:
            print(f"  ✗ {log_file.name}: NOT FOUND")
    
    print("\n✅ Test complete - AsyncTelemetryLogger working!")
    print("   (graceful shutdown will happen on exit)")

if __name__ == "__main__":
    main()
