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
    distance: Optional[float] = None  # En metros (None para SLAM)


class TelemetryLogger:
    """
    Logger centralizado de métricas del sistema.
    
    Captura automáticamente:
    - Performance: FPS, latencia
    - Detecciones: objetos, confianza, distancia
    
    Cada sesión genera una carpeta única en logs/session_XXXXX/
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Inicializar nueva sesión de telemetría.
        
        Args:
            output_dir: Directorio base para logs (default: logs/)
        """
        self._write_lock = threading.Lock()
        
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
        self.system_log = self.output_dir / "system.jsonl"
        
        # Buffers en memoria
        self.performance_buffer: List[PerformanceMetric] = []
        self.detection_buffer: List[DetectionMetric] = []
        
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
        """
        metric = PerformanceMetric(
            timestamp=time.time(),
            frame_number=frame_number,
            fps=fps,
            latency_ms=latency_ms
        )
        
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
        distance: Optional[float] = None
    ) -> None:
        """
        Registrar una detección individual.
        
        Args:
            frame_number: Número de frame
            source: Fuente ("rgb", "slam1", "slam2")
            object_class: Clase del objeto ("person", "chair", etc.)
            confidence: Confianza de la detección (0.0-1.0)
            distance: Distancia en metros (None si no disponible)
        """
        metric = DetectionMetric(
            timestamp=time.time(),
            frame_number=frame_number,
            source=source,
            object_class=object_class,
            confidence=confidence,
            distance=distance
        )
        
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
            detections: Lista de detecciones [{'class': 'person', 'confidence': 0.95, ...}]
        """
        for det in detections:
            self.log_detection(
                frame_number=frame_number,
                source=source,
                object_class=det.get('class', 'unknown'),
                confidence=det.get('confidence', 0.0),
                distance=det.get('distance')
            )
    
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
        """
        session_duration = time.time() - self.session_start
        
        # Calcular estadísticas
        fps_values = [m.fps for m in self.performance_buffer]
        latencies = [m.latency_ms for m in self.performance_buffer]
        
        # Contar detecciones por clase
        detections_by_class: Dict[str, int] = {}
        for det in self.detection_buffer:
            detections_by_class[det.object_class] = detections_by_class.get(det.object_class, 0) + 1
        
        summary = {
            "session_id": self.session_id,
            "duration_seconds": session_duration,
            "total_frames": len(self.performance_buffer),
            "total_detections": len(self.detection_buffer),
            "avg_fps": sum(fps_values) / len(fps_values) if fps_values else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "detections_by_class": detections_by_class,
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
        """Obtener estadísticas de rendimiento en tiempo real."""
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
        """Obtener estadísticas de detecciones."""
        by_class: Dict[str, int] = {}
        by_source: Dict[str, int] = {}
        
        for det in self.detection_buffer:
            by_class[det.object_class] = by_class.get(det.object_class, 0) + 1
            by_source[det.source] = by_source.get(det.source, 0) + 1
        
        return {
            "total_detections": len(self.detection_buffer),
            "by_class": by_class,
            "by_source": by_source,
        }
    
    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------
    
    def _write_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """Escribir línea JSON de forma thread-safe."""
        try:
            line = json.dumps(data, ensure_ascii=True)
        except TypeError:
            line = json.dumps({"error": "serialization_failed", "repr": repr(data)})
        
        with self._write_lock:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')