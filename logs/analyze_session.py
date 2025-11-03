#!/usr/bin/env python3
"""Análisis básico de sesión de telemetría."""

import json
import sys
from pathlib import Path
from typing import Dict, List


def load_jsonl(path: Path) -> List[Dict]:
    """Cargar archivo JSONL."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def analyze_session(session_dir: Path) -> None:
    """Análisis básico de una sesión."""
    
    # 1. Performance
    perf = load_jsonl(session_dir / "performance.jsonl")
    fps_vals = [p['fps'] for p in perf]
    lat_vals = [p['latency_ms'] for p in perf]
    
    print(f"\n📊 PERFORMANCE ({len(perf)} frames)")
    print(f"   FPS:      {sum(fps_vals)/len(fps_vals):.1f} avg  |  {min(fps_vals):.1f} min  |  {max(fps_vals):.1f} max")
    print(f"   Latencia: {sum(lat_vals)/len(lat_vals):.1f}ms avg  |  {max(lat_vals):.1f}ms max")
    print(f"   ⚠️  Frames <25fps: {sum(1 for f in fps_vals if f < 25)}")
    
    # 2. Detecciones
    dets = load_jsonl(session_dir / "detections.jsonl")
    by_class = {}
    by_source = {}
    for d in dets:
        by_class[d['object_class']] = by_class.get(d['object_class'], 0) + 1
        by_source[d['source']] = by_source.get(d['source'], 0) + 1
    
    print(f"\n🔍 DETECCIONES ({len(dets)} total)")
    print(f"   Por clase: {by_class}")
    print(f"   Por fuente: {by_source}")
    
    # 3. Summary
    with open(session_dir / "summary.json") as f:
        summary = json.load(f)
    
    print(f"\n⏱️  SESIÓN")
    print(f"   Duración: {summary['duration_seconds']:.1f}s")
    print(f"   ID: {summary['session_id']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python analyze_session.py logs/session_XXXXX/")
        sys.exit(1)
    
    session_dir = Path(sys.argv[1])
    analyze_session(session_dir)