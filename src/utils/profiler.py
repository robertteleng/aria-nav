#!/usr/bin/env python3
"""
GPU + CPU profiling con métricas detalladas
Usa torch.profiler para CUDA + custom metrics
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np

@dataclass
class ProfileMetrics:
    """Métricas de profiling por componente"""
    component: str
    call_count: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_time_ms: float
    p95_time_ms: float
    p99_time_ms: float

class PerformanceProfiler:
    """
    Profiler ligero para medir cuellos de botella sin overhead
    
    Uso:
        profiler = PerformanceProfiler()
        
        with profiler.measure("yolo_inference"):
            result = yolo.process(frame)
        
        profiler.print_report()
    """
    
    def __init__(self, output_dir: str = "logs/profiling"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.measurements: Dict[str, List[float]] = {}
        self.start_time = time.time()
        self.enabled = True
    
    def measure(self, component: str):
        """Context manager para medir tiempo de componente"""
        return ProfileContext(self, component)
    
    def record(self, component: str, duration_ms: float):
        """Registrar medición manual"""
        if not self.enabled:
            return
        
        if component not in self.measurements:
            self.measurements[component] = []
        self.measurements[component].append(duration_ms)
    
    def get_metrics(self, component: str) -> ProfileMetrics:
        """Calcular métricas estadísticas para un componente"""
        times = self.measurements.get(component, [])
        if not times:
            return ProfileMetrics(
                component=component,
                call_count=0,
                total_time_ms=0,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
            )
        
        times_sorted = sorted(times)
        return ProfileMetrics(
            component=component,
            call_count=len(times),
            total_time_ms=sum(times),
            avg_time_ms=np.mean(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            p50_time_ms=np.percentile(times, 50),
            p95_time_ms=np.percentile(times, 95),
            p99_time_ms=np.percentile(times, 99),
        )
    
    def print_report(self):
        """Imprimir reporte de performance"""
        print("\n" + "="*80)
        print("🔍 PERFORMANCE PROFILING REPORT")
        print("="*80)
        
        # Ordenar por tiempo total descendente
        components = sorted(
            self.measurements.keys(),
            key=lambda c: sum(self.measurements[c]),
            reverse=True
        )
        
        print(f"\n{'Component':<30} {'Calls':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'P95(ms)':>10} {'P99(ms)':>10}")
        print("-" * 80)
        
        for component in components:
            metrics = self.get_metrics(component)
            print(
                f"{component:<30} "
                f"{metrics.call_count:>8} "
                f"{metrics.total_time_ms:>12.2f} "
                f"{metrics.avg_time_ms:>10.2f} "
                f"{metrics.p95_time_ms:>10.2f} "
                f"{metrics.p99_time_ms:>10.2f}"
            )
        
        print("="*80)
        
        # FPS estimado
        total_runtime = time.time() - self.start_time
        total_calls = sum(len(times) for times in self.measurements.values())
        if total_runtime > 0:
            fps = total_calls / total_runtime / len(self.measurements)
            print(f"Estimated FPS: {fps:.2f}")
            print(f"Total runtime: {total_runtime:.2f}s")
    
    def save_report(self, filename: str = None):
        """Guardar reporte en JSON"""
        if filename is None:
            filename = f"profile_{int(time.time())}.json"
        
        output_path = self.output_dir / filename
        
        report = {
            "timestamp": time.time(),
            "runtime_seconds": time.time() - self.start_time,
            "components": {
                component: asdict(self.get_metrics(component))
                for component in self.measurements.keys()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Report saved: {output_path}")
        return output_path

class ProfileContext:
    """Context manager para mediciones"""
    
    def __init__(self, profiler: PerformanceProfiler, component: str):
        self.profiler = profiler
        self.component = component
        self.start = None
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        duration_ms = (time.perf_counter() - self.start) * 1000
        self.profiler.record(self.component, duration_ms)

# Global profiler instance
_global_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler

def enable_profiling():
    """Enable global profiling"""
    get_profiler().enabled = True

def disable_profiling():
    """Disable global profiling"""
    get_profiler().enabled = False
