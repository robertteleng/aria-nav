#!/usr/bin/env python3
"""Prueba 3: Distancias - MAE entre estimadas vs reales"""

import json
import sys
from pathlib import Path


def test_distances(session_dir: Path, distances_gt_path: Path):
    dets_file = session_dir / "detections.jsonl"
    
    # Cargar detecciones
    detections = []
    with open(dets_file) as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get('distance'):  # Solo RGB tiene distancias
                detections.append(d)
    
    # Cargar ground truth (formato: frame,object_id,distance_meters)
    with open(distances_gt_path) as f:
        gt = {int(r[0]): float(r[2]) for r in [l.strip().split(',') for l in f if l.strip()]}
    
    # Convertir distancias categóricas a numéricas
    dist_map = {'very_close': 0.5, 'close': 1.5, 'medium': 3.0, 'far': 5.0}
    
    errors = []
    for d in detections:
        frame = d['frame_number']
        if frame in gt:
            pred = dist_map.get(d['distance'], 0)
            real = gt[frame]
            errors.append(abs(pred - real))
    
    mae = sum(errors) / len(errors) if errors else 0
    
    print(f"\n{'='*50}")
    print("PRUEBA 3: DISTANCIAS")
    print(f"{'='*50}")
    print(f"MAE: {mae:.2f}m {'✅' if mae < 0.5 else '❌'} (objetivo: <0.5m)")
    print(f"Muestras: {len(errors)}")


if __name__ == "__main__":
    test_distances(Path(sys.argv[1]), Path(sys.argv[2]))