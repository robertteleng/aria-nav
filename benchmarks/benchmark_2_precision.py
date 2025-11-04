#!/usr/bin/env python3
"""Prueba 2: Precisión - Calcular P/R/F1 vs ground truth"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def test_precision(session_dir: Path, ground_truth_path: Path):
    dets_file = session_dir / "detections.jsonl"
    
    # Cargar detecciones
    detections = []
    with open(dets_file) as f:
        for line in f:
            detections.append(json.loads(line.strip()))
    
    # Cargar ground truth (formato: frame,class,x,y,w,h)
    with open(ground_truth_path) as f:
        gt = [line.strip().split(',') for line in f if line.strip()]
    
    # Agrupar por clase
    pred_by_class = defaultdict(int)
    for d in detections:
        pred_by_class[d['object_class']] += 1
    
    gt_by_class = defaultdict(int)
    for row in gt:
        gt_by_class[row[1]] += 1
    
    # Calcular métricas (simplificado - necesitas matching real)
    print(f"\n{'='*50}")
    print("PRUEBA 2: PRECISIÓN")
    print(f"{'='*50}")
    print(f"Detectadas: {pred_by_class}")
    print(f"Ground truth: {dict(gt_by_class)}")
    print("\n⚠️  Necesitas implementar IoU matching para P/R/F1 real")


if __name__ == "__main__":
    test_precision(Path(sys.argv[1]), Path(sys.argv[2]))