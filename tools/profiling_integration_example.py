"""
Ejemplo de integración del profiler en NavigationPipeline

ANTES:
    result = self.yolo.process(frame)
    depth_map = self.depth_estimator.estimate(frame)

DESPUÉS:
    from utils.profiler import get_profiler
    profiler = get_profiler()
    
    with profiler.measure("yolo_inference"):
        result = self.yolo.process(frame)
    
    with profiler.measure("depth_estimation"):
        depth_map = self.depth_estimator.estimate(frame)
    
    # Al final del programa
    profiler.print_report()
    profiler.save_report()
"""

# Ejemplo completo para _process_multiproc:

def _process_multiproc(self, frames_dict, profile):
    from utils.profiler import get_profiler
    profiler = get_profiler() if profile else None
    
    frame_id = self.frames_processed
    self.frames_processed += 1
    
    # Enhancement
    if profiler:
        with profiler.measure("image_enhancement"):
            central_frame = self.image_enhancer.enhance_frame(frames_dict["central"])
    else:
        central_frame = self.image_enhancer.enhance_frame(frames_dict["central"])
    
    # Enqueue
    if profiler:
        with profiler.measure("enqueue_frames"):
            self._enqueue_frames(frame_id, frames_dict, central_frame, timestamp)
    else:
        self._enqueue_frames(frame_id, frames_dict, central_frame, timestamp)
    
    # Collect
    if profiler:
        with profiler.measure("collect_results"):
            results = self._collect_results_with_overlap(frame_id, central_frame, profile)
    else:
        results = self._collect_results_with_overlap(frame_id, central_frame, profile)
    
    # Merge
    if profiler:
        with profiler.measure("merge_results"):
            return self._merge_results(results, frames_dict)
    else:
        return self._merge_results(results, frames_dict)

# Componentes a medir en Phase 2:
COMPONENTS_TO_PROFILE = [
    "image_enhancement",          # Cuánto tarda enhancement
    "enqueue_frames",             # Overhead de queue.put
    "collect_results",            # Tiempo esperando workers (blocking)
    "merge_results",              # Overhead de merge
    "queue_put_central",          # Granular: solo central
    "queue_put_slam",             # Granular: solo SLAM
    "queue_get_central",          # Granular: blocking en central
    "queue_get_slam",             # Granular: non-blocking SLAM
    "worker_central_total",       # Tiempo total en CentralWorker
    "worker_central_depth",       # Solo depth en worker
    "worker_central_yolo",        # Solo YOLO en worker
    "worker_slam_total",          # Tiempo total en SlamWorker
    "pickle_serialize",           # Overhead de serialización
    "pickle_deserialize",         # Overhead de deserialización
]

# Para encontrar cuellos de botella:
# 1. Ejecutar con profiling: python src/main.py debug --profile
# 2. Ver reporte: cat logs/profiling/latest_stats.txt
# 3. Identificar componentes con:
#    - avg_time_ms alto
#    - p95/p99 muy por encima de avg (variabilidad)
#    - total_time_ms alto (se llama mucho)
