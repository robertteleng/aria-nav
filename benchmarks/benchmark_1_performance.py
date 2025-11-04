#!/usr/bin/env python3
"""Prueba 1: Performance - FPS >25 y latencia <200ms"""

import json
import sys
from pathlib import Path


def test_performance(session_dir: Path, skip_warmup=True):
    perf_file = session_dir / "performance.jsonl"
    
    # Cargar datos
    frames = []
    with open(perf_file) as f:
        for line in f:
            frames.append(json.loads(line.strip()))
    
    # NUEVO: Excluir primeros 100 frames (warm-up)
    if skip_warmup and len(frames) > 100:
        print(f"⚠️  Excluyendo primeros 100 frames (warm-up)")
        frames = frames[100:]
    
    fps_vals = [f['fps'] for f in frames]
    lat_vals = [f['latency_ms'] for f in frames]
    
    # Métricas
    avg_fps = sum(fps_vals) / len(fps_vals)
    min_fps = min(fps_vals)
    max_fps = max(fps_vals)
    frames_below_25 = sum(1 for f in fps_vals if f < 25)
    frames_above_200ms = sum(1 for l in lat_vals if l > 200)
    
    # Criterios de éxito
    fps_ok = avg_fps >= 25
    latency_ok = (frames_above_200ms / len(lat_vals)) < 0.05
    
    print(f"\n{'='*50}")
    print("PRUEBA 1: PERFORMANCE")
    print(f"{'='*50}")
    print(f"📊 Frames analizados: {len(frames)}")
    print(f"\n📈 FPS:")
    print(f"   Promedio: {avg_fps:.2f} {'✅' if fps_ok else '❌'} (objetivo: ≥25)")
    print(f"   Mínimo:   {min_fps:.2f}")
    print(f"   Máximo:   {max_fps:.2f}")
    print(f"   <25fps:   {frames_below_25} ({100*frames_below_25/len(fps_vals):.1f}%)")
    
    print(f"\n⏱️  LATENCIA:")
    print(f"   Promedio: {sum(lat_vals)/len(lat_vals):.1f}ms")
    print(f"   Máxima:   {max(lat_vals):.1f}ms")
    print(f"   >200ms:   {frames_above_200ms} ({100*frames_above_200ms/len(lat_vals):.1f}%) {'✅' if latency_ok else '❌'} (<5%)")
    
    # Análisis de tendencia
    print(f"\n📉 TENDENCIA FPS:")
    tercio = len(fps_vals) // 3
    fps_inicio = sum(fps_vals[:tercio]) / tercio
    fps_medio = sum(fps_vals[tercio:2*tercio]) / tercio
    fps_final = sum(fps_vals[2*tercio:]) / (len(fps_vals) - 2*tercio)
    
    print(f"   Primer tercio:  {fps_inicio:.2f} FPS")
    print(f"   Segundo tercio: {fps_medio:.2f} FPS")
    print(f"   Tercer tercio:  {fps_final:.2f} FPS")
    
    print(f"\n{'='*50}")
    print(f"{'APROBADO ✅' if (fps_ok and latency_ok) else 'FALLIDO ❌'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_1_performance.py <session_dir>")
        sys.exit(1)
    
    test_performance(Path(sys.argv[1]))