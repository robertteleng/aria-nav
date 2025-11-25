"""Telemetría centralizada - Sistema de métricas para navegación asistida."""

import json
import time
import threading
import queue
import atexit
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PerformanceMetric:
    """Métrica de rendimiento por frame."""
    timestamp: float
    frame_number: int
    fps: float
    latency_ms: float


@dataclass
class DetectionMetric:
    """Métrica de detección de objetos."""
    timestamp: float
    frame_number: int
    source: str  # "rgb", "slam1", "slam2"
    object_class: str
    confidence: float
    distance_bucket: Optional[str] = None  # Categoría (very_close, medium...)
    distance_normalized: Optional[float] = None  # Profundidad normalizada (0-1)
    distance_raw: Optional[float] = None  # Profundidad cruda (escala del modelo)
    distance_meters: Optional[float] = None  # Reservado para futuras calibraciones


@dataclass
class AudioMetric:
    """Métrica de evento de audio."""
    timestamp: float
    action: str  # "enqueued", "spoken", "skipped", "dropped"
    source: str  # "rgb", "slam1", "slam2"
    priority: int  # 1=CRITICAL, 2=HIGH, 3=MEDIUM, 4=LOW
    message: str
    reason: Optional[str] = None  # Razón si skipped/dropped


class TelemetryLogger:
    """
    Logger centralizado de métricas del sistema (thread-safe).
    - Performance: FPS, latencia
    - Detecciones: objetos, confianza, distancia
    - Audio: eventos de navegación, prioridades
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Inicializar nueva sesión de telemetría.
        
        Args:
            output_dir: Directorio base para logs (default: logs/)
        """
        # 🔒 Locks para thread-safety
        self._write_lock = threading.Lock()
        self._buffer_lock = threading.Lock()
        
        # Setup carpeta base
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[3]
            output_dir = project_root / "logs"
        
        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear sesión única con timestamp legible
        self.session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_start = time.time()
        self.session_dir = base_dir / f"session_{self.session_timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear subcarpeta telemetry dentro de la sesión
        self.output_dir = self.session_dir / "telemetry"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de log en subcarpeta telemetry
        self.performance_log = self.output_dir / "performance.jsonl"
        self.detections_log = self.output_dir / "detections.jsonl"
        self.audio_log = self.output_dir / "audio_events.jsonl"
        self.system_log = self.output_dir / "system.jsonl"
        self.resource_log = self.output_dir / "resources.jsonl"
        
        # Buffers en memoria (protegidos por _buffer_lock)
        self.performance_buffer: List[PerformanceMetric] = []
        self.detection_buffer: List[DetectionMetric] = []
        self.audio_buffer: List[AudioMetric] = []
        
        # Log de inicio
        self._log_system_event("session_start", {
            "session": self.session_timestamp,
            "timestamp": self.session_start
        })
        
        print(f"[TELEMETRY] Nueva sesión: {self.session_timestamp}")
        print(f"[TELEMETRY] Carpeta: {self.output_dir}")
    
    def get_session_dir(self) -> Path:
        """Return the session directory path for use by other loggers."""
        return self.session_dir
    
    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------
    
    def log_frame_performance(
        self,
        frame_number: int,
        fps: float,
        latency_ms: float,
        timing_breakdown: dict = None
    ) -> None:
        """
        Registrar métricas de rendimiento de un frame.
        
        Args:
            frame_number: Número de frame
            fps: Frames por segundo actual
            latency_ms: Latencia de procesamiento en milisegundos
            timing_breakdown: Diccionario con timing detallado de componentes
        
        Thread-safe: Puede ser llamado desde cualquier thread.
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            frame_number=frame_number,
            fps=fps,
            latency_ms=latency_ms
        )
        
        with self._buffer_lock:
            self.performance_buffer.append(metric)
        
        # Write with timing breakdown if available
        data = asdict(metric)
        if timing_breakdown:
            data['timing'] = timing_breakdown
        self._write_jsonl(self.performance_log, data)
    
    # ------------------------------------------------------------------
    # Detection Metrics
    # ------------------------------------------------------------------
    
    def log_detection(
        self,
        frame_number: int,
        source: str,
        object_class: str,
        confidence: float,
        distance_bucket: Optional[str] = None,
        distance_normalized: Optional[float] = None,
        distance_raw: Optional[float] = None,
        distance_meters: Optional[float] = None,
    ) -> None:
        """
        Registrar una detección individual.
        
        Args:
            frame_number: Número de frame
            source: Fuente ("rgb", "slam1", "slam2")
            object_class: Clase del objeto ("person", "chair", etc.)
            confidence: Confianza de la detección (0.0-1.0)
            distance_bucket: Categoría simbólica de distancia
            distance_normalized: Valor normalizado 0-1 basado en profundidad/área
            distance_raw: Valor crudo promedio del mapa de profundidad
            distance_meters: Distancia estimada en metros (si se calibra)
        
        Thread-safe: Puede ser llamado desde cualquier thread.
        """
        metric = DetectionMetric(
            timestamp=time.time(),
            frame_number=frame_number,
            source=source,
            object_class=object_class,
            confidence=confidence,
            distance_bucket=distance_bucket,
            distance_normalized=distance_normalized,
            distance_raw=distance_raw,
            distance_meters=distance_meters,
        )
        
        with self._buffer_lock:
            self.detection_buffer.append(metric)
        
        self._write_jsonl(self.detections_log, asdict(metric))
    
    def log_detections_batch(
        self,
        frame_number: int,
        source: str,
        detections: List[Dict[str, Any]]
    ) -> None:
        """
        Registrar múltiples detecciones de un frame.
        
        Args:
            frame_number: Número de frame
            source: Fuente ("rgb", "slam1", "slam2")
            detections: Lista de detecciones [{'name': 'person', 'confidence': 0.95, ...}]
        
        Thread-safe: Puede ser llamado desde cualquier thread.
        """
        for det in detections:
            self.log_detection(
                frame_number=frame_number,
                source=source,
                object_class=det.get('name', det.get('class', 'unknown')),
                confidence=det.get('confidence', 0.0),
                distance_bucket=det.get('distance'),
                distance_normalized=det.get('distance_normalized'),
                distance_raw=det.get('distance_raw'),
                distance_meters=det.get('distance_meters'),
            )
    
    # ------------------------------------------------------------------
    # Audio Metrics
    # ------------------------------------------------------------------
    
    def log_audio_event(
        self,
        action: str,
        source: str,
        priority: int,
        message: str,
        reason: Optional[str] = None
    ) -> None:
        """
        Registrar evento de audio de navegación.
        
        Args:
            action: "enqueued", "spoken", "skipped", "dropped"
            source: "rgb", "slam1", "slam2"
            priority: 1=CRITICAL, 2=HIGH, 3=MEDIUM, 4=LOW
            message: Mensaje de audio reproducido/intentado
            reason: Razón si fue skipped/dropped (ej: "cooldown", "queue_full")
        
        Thread-safe: Puede ser llamado desde cualquier thread.
        """
        metric = AudioMetric(
            timestamp=time.time(),
            action=action,
            source=source,
            priority=priority,
            message=message,
            reason=reason
        )
        
        with self._buffer_lock:
            self.audio_buffer.append(metric)
        
        self._write_jsonl(self.audio_log, asdict(metric))
    
    # ------------------------------------------------------------------
    # System Events
    # ------------------------------------------------------------------
    
    def _log_system_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Registrar eventos del sistema."""
        payload = {
            "timestamp": time.time(),
            "session": self.session_timestamp,
            "event_type": event_type,
            **data
        }
        self._write_jsonl(self.system_log, payload)
    
    def log_error(self, error_type: str, message: str, **kwargs: Any) -> None:
        """Registrar un error del sistema."""
        self._log_system_event("error", {
            "error_type": error_type,
            "message": message,
            **kwargs
        })
    
    def log_resources(self, resource_data: Dict[str, Any]) -> None:
        """
        Registrar métricas de recursos del sistema (CPU, GPU, RAM).
        
        Args:
            resource_data: Dict con claves:
                - cpu_pct: float (%)
                - ram_used_mb: float
                - ram_total_mb: float
                - gpu_present: bool
                - gpu_mem_used_mb: float (opcional)
                - gpu_mem_total_mb: float (opcional)
                - gpu_util_pct: float (opcional)
        """
        payload = {
            "timestamp": time.time(),
            "session": self.session_timestamp,
            **resource_data
        }
        self._write_jsonl(self.resource_log, payload)
    
    # ------------------------------------------------------------------
    # Session Management
    # ------------------------------------------------------------------
    
    def finalize_session(self) -> Dict[str, Any]:
        """
        Finalizar sesión y generar resumen.
        
        Returns:
            Dict con estadísticas de la sesión
        
        Thread-safe: Puede ser llamado desde cualquier thread.
        """
        session_duration = time.time() - self.session_start
        
        # 🔒 Copiar buffers de forma thread-safe
        with self._buffer_lock:
            perf_copy = list(self.performance_buffer)
            det_copy = list(self.detection_buffer)
            audio_copy = list(self.audio_buffer)
        
        # Calcular estadísticas (ahora con copias locales)
        fps_values = [m.fps for m in perf_copy]
        latencies = [m.latency_ms for m in perf_copy]
        
        # Contar detecciones por clase
        detections_by_class: Dict[str, int] = {}
        detections_by_source: Dict[str, int] = {}
        detections_by_bucket: Dict[str, int] = {}
        distance_norm_values: List[float] = []
        distance_raw_values: List[float] = []
        for det in det_copy:
            detections_by_class[det.object_class] = detections_by_class.get(det.object_class, 0) + 1
            detections_by_source[det.source] = detections_by_source.get(det.source, 0) + 1
            if det.distance_bucket:
                detections_by_bucket[det.distance_bucket] = detections_by_bucket.get(det.distance_bucket, 0) + 1
            if det.distance_normalized is not None:
                distance_norm_values.append(det.distance_normalized)
            if det.distance_raw is not None:
                distance_raw_values.append(det.distance_raw)
        
        # Contar eventos de audio
        audio_by_action: Dict[str, int] = {}
        audio_by_source: Dict[str, int] = {}
        for audio in audio_copy:
            audio_by_action[audio.action] = audio_by_action.get(audio.action, 0) + 1
            audio_by_source[audio.source] = audio_by_source.get(audio.source, 0) + 1
        
        summary = {
            "session": self.session_timestamp,
            "duration_seconds": session_duration,
            "total_frames": len(perf_copy),
            "avg_fps": sum(fps_values) / len(fps_values) if fps_values else 0,
            "min_fps": min(fps_values) if fps_values else 0,
            "max_fps": max(fps_values) if fps_values else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "frames_below_25fps": sum(1 for fps in fps_values if fps < 25),
            "total_detections": len(det_copy),
            "detections_by_class": detections_by_class,
            "detections_by_source": detections_by_source,
            "detections_by_distance_bucket": detections_by_bucket,
            "avg_distance_normalized": (
                sum(distance_norm_values) / len(distance_norm_values)
                if distance_norm_values else None
            ),
            "avg_distance_raw": (
                sum(distance_raw_values) / len(distance_raw_values)
                if distance_raw_values else None
            ),
            "total_audio_events": len(audio_copy),
            "audio_by_action": audio_by_action,
            "audio_by_source": audio_by_source,
        }
        
        # Log evento de cierre
        self._log_system_event("session_end", summary)
        
        # Guardar resumen en carpeta telemetry
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[TELEMETRY] Sesión finalizada: {self.session_timestamp}")
        print(f"[TELEMETRY] Resumen guardado: {summary_path.name}")
        
        return summary
    
    # ------------------------------------------------------------------
    # Analysis Helpers
    # ------------------------------------------------------------------
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Obtener estadísticas de rendimiento en tiempo real.
        
        Thread-safe: Puede ser llamado desde cualquier thread.
        """
        with self._buffer_lock:
            if not self.performance_buffer:
                return {}
            
            fps_values = [m.fps for m in self.performance_buffer]
            latencies = [m.latency_ms for m in self.performance_buffer]
        
        return {
            "avg_fps": sum(fps_values) / len(fps_values),
            "min_fps": min(fps_values),
            "max_fps": max(fps_values),
            "avg_latency_ms": sum(latencies) / len(latencies),
            "max_latency_ms": max(latencies),
            "frames_below_25fps": sum(1 for fps in fps_values if fps < 25),
            "frames_above_200ms": sum(1 for lat in latencies if lat > 200),
        }
    
    def get_detection_summary(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de detecciones.
        
        Thread-safe: Puede ser llamado desde cualquier thread.
        """
        with self._buffer_lock:
            by_class: Dict[str, int] = {}
            by_source: Dict[str, int] = {}
            by_bucket: Dict[str, int] = {}
            normalized_values: List[float] = []
            raw_values: List[float] = []
            
            for det in self.detection_buffer:
                by_class[det.object_class] = by_class.get(det.object_class, 0) + 1
                by_source[det.source] = by_source.get(det.source, 0) + 1
                if det.distance_bucket:
                    by_bucket[det.distance_bucket] = by_bucket.get(det.distance_bucket, 0) + 1
                if det.distance_normalized is not None:
                    normalized_values.append(det.distance_normalized)
                if det.distance_raw is not None:
                    raw_values.append(det.distance_raw)
        
        return {
            "total_detections": len(self.detection_buffer),
            "by_class": by_class,
            "by_source": by_source,
            "by_distance_bucket": by_bucket,
            "avg_distance_normalized": (
                sum(normalized_values) / len(normalized_values)
                if normalized_values else None
            ),
            "avg_distance_raw": (
                sum(raw_values) / len(raw_values)
                if raw_values else None
            ),
        }
    
    def get_audio_summary(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de audio.
        
        Thread-safe: Puede ser llamado desde cualquier thread.
        """
        with self._buffer_lock:
            by_action: Dict[str, int] = {}
            by_source: Dict[str, int] = {}
            by_priority: Dict[int, int] = {}
            
            for audio in self.audio_buffer:
                by_action[audio.action] = by_action.get(audio.action, 0) + 1
                by_source[audio.source] = by_source.get(audio.source, 0) + 1
                by_priority[audio.priority] = by_priority.get(audio.priority, 0) + 1
            
            spoken = by_action.get("spoken", 0)
            total = len(self.audio_buffer)
        
        return {
            "total_events": total,
            "by_action": by_action,
            "by_source": by_source,
            "by_priority": by_priority,
            "spoken_ratio": spoken / total if total > 0 else 0,
        }
    
    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    
    def _write_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Escribir línea JSON de forma thread-safe.
        
        Thread-safe: Usa lock para escritura atómica.
        """
        try:
            line = json.dumps(data, ensure_ascii=True)
        except TypeError:
            line = json.dumps({"error": "serialization_failed", "repr": repr(data)})
        
        with self._write_lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


# ======================================================================
# ASYNC TELEMETRY LOGGER - Non-blocking I/O
# ======================================================================

class AsyncTelemetryLogger(TelemetryLogger):
    """
    Telemetry logger con I/O asíncrona para eliminar bottlenecks.

    Features:
    - Queue para escrituras no bloqueantes
    - Background thread con batch writes
    - Flush interval configurable (default: 1.0s)
    - Buffer size configurable (default: 100 líneas)
    - Graceful shutdown con atexit
    - Export automático a MLflow al finalizar sesión

    Beneficios:
    - Elimina spikes de 250-300ms cada ~80 frames
    - Batch writes reducen syscalls
    - Main thread nunca bloqueado por I/O

    Riesgo:
    - Últimos frames pueden perderse si crash abrupto
      (mitigado con atexit flush)
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        flush_interval: float = 2.0,
        buffer_size: int = 100,
        queue_maxsize: int = 2000,
        mlflow_enabled: bool = True,
        mlflow_experiment: str = "aria-navigation"
    ):
        """
        Inicializar async telemetry logger.

        Args:
            output_dir: Directorio base para logs
            flush_interval: Segundos entre flushes automáticos (default: 2.0s)
            buffer_size: Líneas acumuladas antes de flush forzado (default: 100)
            queue_maxsize: Tamaño máximo de la cola (default: 2000, previene OOM)
            mlflow_enabled: Exportar a MLflow al finalizar (default: True)
            mlflow_experiment: Nombre del experimento MLflow
        """
        # Queue para escrituras asíncronas (ANTES de super().__init__)
        self._write_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._flush_interval = flush_interval
        self._buffer_size = buffer_size
        self._shutdown_flag = threading.Event()

        # MLflow config
        self._mlflow_enabled = mlflow_enabled
        self._mlflow_experiment = mlflow_experiment
        self._mlflow_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlflow")

        # Ahora sí, inicializar base class (que llama a _write_jsonl)
        super().__init__(output_dir)

        # Background thread para flush
        self._flush_thread = threading.Thread(
            target=self._flush_worker,
            daemon=True,  # Daemon para no bloquear exit
            name="AsyncTelemetryFlusher"
        )
        self._flush_thread.start()

        # Registrar shutdown handler para flush final
        atexit.register(self._shutdown)

        self._log_system_event("async_telemetry_enabled", {
            "flush_interval": flush_interval,
            "buffer_size": buffer_size,
            "queue_maxsize": queue_maxsize,
            "mlflow_enabled": mlflow_enabled,
            "mlflow_experiment": mlflow_experiment
        })

        print(f"[TELEMETRY] Modo asíncrono activado (flush={flush_interval}s, buffer={buffer_size})")
        if mlflow_enabled:
            print(f"[TELEMETRY] MLflow habilitado: {mlflow_experiment}")
    
    def _write_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Queue write en lugar de bloquear.
        
        Override del método síncrono para usar queue.
        """
        try:
            self._write_queue.put_nowait((path, data))
        except queue.Full:
            # Log error pero no bloquear main thread
            print(f"[TELEMETRY WARNING] Queue full, dropping write to {path.name}")
    
    def _flush_worker(self) -> None:
        """
        Background thread para batch writes.
        
        Lógica:
        1. Recoger writes de la queue con timeout
        2. Acumular en buffers por archivo
        3. Flush cuando buffer lleno O timeout
        """
        buffers: Dict[Path, List[str]] = {}
        last_flush = time.time()
        
        while not self._shutdown_flag.is_set():
            try:
                # Esperar write con timeout
                path, data = self._write_queue.get(timeout=0.1)
                
                # Serializar JSON
                try:
                    line = json.dumps(data, ensure_ascii=True)
                except TypeError:
                    line = json.dumps({"error": "serialization_failed"})
                
                # Acumular en buffer
                if path not in buffers:
                    buffers[path] = []
                buffers[path].append(line)
                
                # Flush si buffer lleno
                if len(buffers[path]) >= self._buffer_size:
                    self._flush_buffer(path, buffers[path])
                    buffers[path] = []
                    last_flush = time.time()
                
            except queue.Empty:
                # Timeout: flush todos los buffers si ha pasado el intervalo
                now = time.time()
                if now - last_flush >= self._flush_interval:
                    for file_path, lines in list(buffers.items()):
                        if lines:
                            self._flush_buffer(file_path, lines)
                            buffers[file_path] = []
                    last_flush = now
        
        # Flush final al shutdown
        for file_path, lines in buffers.items():
            if lines:
                self._flush_buffer(file_path, lines)
    
    def _flush_buffer(self, path: Path, lines: List[str]) -> None:
        """
        Batch write a disco.
        
        Escribe múltiples líneas de una vez para reducir syscalls.
        """
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to flush {path.name}: {e}")
    
    def _shutdown(self) -> None:
        """
        Graceful shutdown: flush pendiente y detener thread.

        Llamado automáticamente por atexit.
        """
        print("[TELEMETRY] Flushing pending writes...")
        self._shutdown_flag.set()

        # Esperar a que el thread termine (con timeout)
        self._flush_thread.join(timeout=5.0)

        if self._flush_thread.is_alive():
            print("[TELEMETRY WARNING] Flush thread did not terminate cleanly")
        else:
            print("[TELEMETRY] Shutdown complete")

        # Shutdown MLflow executor
        self._mlflow_executor.shutdown(wait=False)

    def finalize_session(
        self,
        model_name: str = "yolo12n",
        resolution: int = 640,
        depth_enabled: bool = True,
        tensorrt_enabled: bool = True,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Finalizar sesión, generar resumen y exportar a MLflow.

        Args:
            model_name: Nombre del modelo YOLO
            resolution: Resolución de entrada
            depth_enabled: Si depth estimation estaba activo
            tensorrt_enabled: Si TensorRT estaba activo
            extra_params: Parámetros adicionales para MLflow

        Returns:
            Dict con estadísticas de la sesión
        """
        # Llamar al método padre para generar summary.json
        summary = super().finalize_session()

        # Export a MLflow (async, no bloquea)
        if self._mlflow_enabled:
            self._mlflow_executor.submit(
                self._export_to_mlflow,
                summary,
                model_name,
                resolution,
                depth_enabled,
                tensorrt_enabled,
                extra_params or {}
            )

        return summary

    def _export_to_mlflow(
        self,
        summary: Dict[str, Any],
        model_name: str,
        resolution: int,
        depth_enabled: bool,
        tensorrt_enabled: bool,
        extra_params: Dict[str, Any]
    ) -> None:
        """Export de sesión a MLflow (ejecutado en thread separado)."""
        try:
            import mlflow

            # SQLite backend dentro del proyecto
            project_root = Path(__file__).resolve().parents[4]  # src/core/telemetry/loggers/ -> root
            mlruns_dir = project_root / "mlruns"
            mlruns_dir.mkdir(exist_ok=True)
            tracking_uri = f"sqlite:///{mlruns_dir / 'mlflow.db'}"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self._mlflow_experiment)

            with mlflow.start_run(run_name=self.session_timestamp):
                # Parámetros de configuración
                mlflow.log_params({
                    "model_name": model_name,
                    "resolution": resolution,
                    "depth_enabled": depth_enabled,
                    "tensorrt_enabled": tensorrt_enabled,
                    **extra_params
                })

                # Métricas de performance
                mlflow.log_metrics({
                    "duration_seconds": summary.get("duration_seconds", 0),
                    "total_frames": summary.get("total_frames", 0),
                    "avg_fps": summary.get("avg_fps", 0),
                    "min_fps": summary.get("min_fps", 0),
                    "max_fps": summary.get("max_fps", 0),
                    "avg_latency_ms": summary.get("avg_latency_ms", 0),
                    "max_latency_ms": summary.get("max_latency_ms", 0),
                    "frames_below_25fps": summary.get("frames_below_25fps", 0),
                    "total_detections": summary.get("total_detections", 0),
                    "total_audio_events": summary.get("total_audio_events", 0),
                })

                # Detecciones por clase
                for cls, count in summary.get("detections_by_class", {}).items():
                    safe_cls = cls.replace(" ", "_").replace("-", "_")
                    mlflow.log_metric(f"det_{safe_cls}", count)

                # Detecciones por distancia
                for bucket, count in summary.get("detections_by_distance_bucket", {}).items():
                    mlflow.log_metric(f"dist_{bucket}", count)

                # Tags
                mlflow.set_tag("session_dir", str(self.session_dir))

                # Artifact: guardar summary.json
                summary_path = self.output_dir / "summary.json"
                if summary_path.exists():
                    mlflow.log_artifact(str(summary_path))

            print(f"[MLflow] Sesión exportada: {self.session_timestamp}")
            print(f"[MLflow] Ver resultados: mlflow ui --backend-store-uri \"{tracking_uri}\"")

        except ImportError:
            print("[MLflow] mlflow no instalado - skip export (pip install mlflow)")
        except Exception as e:
            print(f"[MLflow] Error exportando: {e}")

