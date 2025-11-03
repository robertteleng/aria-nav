"""Telemetría centralizada - Sistema de métricas para TFM."""

import json
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


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
    
    Captura automáticamente:
    - Performance: FPS, latencia
    - Detecciones: objetos, confianza, distancia
    - Audio: eventos de navegación, prioridades
    
    Cada sesión genera una carpeta única en logs/session_XXXXX/
    
    Thread-safety: Puede ser llamado desde múltiples threads de forma segura.
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
        
        # Crear sesión única
        self.session_id = int(time.time() * 1000)  # Milisegundos
        self.session_start = time.time()
        self.output_dir = base_dir / f"session_{self.session_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivos de log
        self.performance_log = self.output_dir / "performance.jsonl"
        self.detections_log = self.output_dir / "detections.jsonl"
        self.audio_log = self.output_dir / "audio_events.jsonl"
        self.system_log = self.output_dir / "system.jsonl"
        
        # Buffers en memoria (protegidos por _buffer_lock)
        self.performance_buffer: List[PerformanceMetric] = []
        self.detection_buffer: List[DetectionMetric] = []
        self.audio_buffer: List[AudioMetric] = []
        
        # Log de inicio
        self._log_system_event("session_start", {
            "session_id": self.session_id,
            "timestamp": self.session_start
        })
        
        print(f"[TELEMETRY] Nueva sesión: {self.session_id}")
        print(f"[TELEMETRY] Carpeta: {self.output_dir}")
    
    # ------------------------------------------------------------------
    # Performance Metrics
    # ------------------------------------------------------------------
    
    def log_frame_performance(
        self,
        frame_number: int,
        fps: float,
        latency_ms: float
    ) -> None:
        """
        Registrar métricas de rendimiento de un frame.
        
        Args:
            frame_number: Número de frame
            fps: Frames por segundo actual
            latency_ms: Latencia de procesamiento en milisegundos
        
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
        
        self._write_jsonl(self.performance_log, asdict(metric))
    
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
            "session_id": self.session_id,
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
            "session_id": self.session_id,
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
        
        # Guardar resumen en archivo separado
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[TELEMETRY] Sesión finalizada: {self.session_id}")
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
