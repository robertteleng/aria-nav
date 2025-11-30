"""
Centralized telemetry system for assisted navigation metrics.

This module provides thread-safe telemetry logging with both synchronous and
asynchronous I/O options. It tracks performance, detections, and audio events
across multi-camera navigation sessions.

Features:
- Thread-safe metric collection with locks
- JSONL format for efficient streaming analytics
- Session-based organization with unique timestamps
- Real-time summary statistics
- Optional async I/O for zero-overhead logging
- MLflow integration for experiment tracking

Metric Types:
- PerformanceMetric: FPS, latency per frame
- DetectionMetric: Object detections with confidence and distance
- AudioMetric: Navigation audio events with priorities

Loggers:
- TelemetryLogger: Synchronous I/O (simple, reliable)
- AsyncTelemetryLogger: Non-blocking I/O with batch writes (high performance)

Usage:
    # Synchronous logger
    from core.telemetry.loggers.telemetry_logger import TelemetryLogger

    logger = TelemetryLogger()
    logger.log_frame_performance(frame_number=42, fps=28.5, latency_ms=35.2)
    logger.log_detection(frame_number=42, source="rgb", object_class="person", confidence=0.95)
    summary = logger.finalize_session()

    # Async logger (for production)
    from core.telemetry.loggers.telemetry_logger import AsyncTelemetryLogger

    logger = AsyncTelemetryLogger(flush_interval=2.0, buffer_size=100)
    logger.log_frame_performance(frame_number=42, fps=28.5, latency_ms=35.2)
    summary = logger.finalize_session(model_name="yolo12n", resolution=640)
"""

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
    """Performance metric per frame."""
    timestamp: float
    frame_number: int
    fps: float
    latency_ms: float


@dataclass
class DetectionMetric:
    """Object detection metric."""
    timestamp: float
    frame_number: int
    source: str  # "rgb", "slam1", "slam2"
    object_class: str
    confidence: float
    distance_bucket: Optional[str] = None  # Category (very_close, medium...)
    distance_normalized: Optional[float] = None  # Normalized depth (0-1)
    distance_raw: Optional[float] = None  # Raw depth (model scale)
    distance_meters: Optional[float] = None  # Reserved for future calibrations


@dataclass
class AudioMetric:
    """Audio event metric."""
    timestamp: float
    action: str  # "enqueued", "spoken", "skipped", "dropped"
    source: str  # "rgb", "slam1", "slam2"
    priority: int  # 1=CRITICAL, 2=HIGH, 3=MEDIUM, 4=LOW
    message: str
    reason: Optional[str] = None  # Reason if skipped/dropped


class TelemetryLogger:
    """
    Centralized thread-safe metrics logger.
    - Performance: FPS, latency
    - Detections: objects, confidence, distance
    - Audio: navigation events, priorities
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize new telemetry session.

        Args:
            output_dir: Base directory for logs (default: logs/)
        """
        # Locks for thread-safety
        self._write_lock = threading.Lock()
        self._buffer_lock = threading.Lock()

        # Setup base folder
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[4]
            output_dir = project_root / "logs"

        base_dir = Path(output_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Create unique session with readable timestamp
        self.session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_start = time.time()
        self.session_dir = base_dir / f"session_{self.session_timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Log files directly in session directory (no telemetry subfolder)
        self.output_dir = self.session_dir

        # Log files directly in session_YYYY-MM-DD_HH-MM-SS/
        self.performance_log = self.output_dir / "performance.jsonl"
        self.detections_log = self.output_dir / "detections.jsonl"
        self.audio_log = self.output_dir / "audio_events.jsonl"
        self.system_log = self.output_dir / "system.jsonl"
        self.resource_log = self.output_dir / "resources.jsonl"

        # In-memory buffers (protected by _buffer_lock)
        self.performance_buffer: List[PerformanceMetric] = []
        self.detection_buffer: List[DetectionMetric] = []
        self.audio_buffer: List[AudioMetric] = []

        # Log start event
        self._log_system_event("session_start", {
            "session": self.session_timestamp,
            "timestamp": self.session_start
        })

        print(f"[TELEMETRY] New session: {self.session_timestamp}")
        print(f"[TELEMETRY] Folder: {self.output_dir}")
    
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
        Record performance metrics for a frame.

        Args:
            frame_number: Frame number
            fps: Current frames per second
            latency_ms: Processing latency in milliseconds
            timing_breakdown: Dict with detailed component timing

        Thread-safe: Can be called from any thread.
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
        Record an individual detection.

        Args:
            frame_number: Frame number
            source: Source ("rgb", "slam1", "slam2")
            object_class: Object class ("person", "chair", etc.)
            confidence: Detection confidence (0.0-1.0)
            distance_bucket: Symbolic distance category
            distance_normalized: Normalized value 0-1 based on depth/area
            distance_raw: Raw average value from depth map
            distance_meters: Estimated distance in meters (if calibrated)

        Thread-safe: Can be called from any thread.
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
        Record multiple detections from a frame.

        Args:
            frame_number: Frame number
            source: Source ("rgb", "slam1", "slam2")
            detections: List of detections [{'name': 'person', 'confidence': 0.95, ...}]

        Thread-safe: Can be called from any thread.
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
        Record navigation audio event.

        Args:
            action: "enqueued", "spoken", "skipped", "dropped"
            source: "rgb", "slam1", "slam2"
            priority: 1=CRITICAL, 2=HIGH, 3=MEDIUM, 4=LOW
            message: Audio message played/attempted
            reason: Reason if skipped/dropped (e.g., "cooldown", "queue_full")

        Thread-safe: Can be called from any thread.
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
        """Record system events."""
        payload = {
            "timestamp": time.time(),
            "session": self.session_timestamp,
            "event_type": event_type,
            **data
        }
        self._write_jsonl(self.system_log, payload)

    def log_error(self, error_type: str, message: str, **kwargs: Any) -> None:
        """Record a system error."""
        self._log_system_event("error", {
            "error_type": error_type,
            "message": message,
            **kwargs
        })

    def log_resources(self, resource_data: Dict[str, Any]) -> None:
        """
        Record system resource metrics (CPU, GPU, RAM).

        Args:
            resource_data: Dict with keys:
                - cpu_pct: float (%)
                - ram_used_mb: float
                - ram_total_mb: float
                - gpu_present: bool
                - gpu_mem_used_mb: float (optional)
                - gpu_mem_total_mb: float (optional)
                - gpu_util_pct: float (optional)
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
        Finalize session and generate summary.

        Returns:
            Dict with session statistics

        Thread-safe: Can be called from any thread.
        """
        session_duration = time.time() - self.session_start

        # Copy buffers thread-safely
        with self._buffer_lock:
            perf_copy = list(self.performance_buffer)
            det_copy = list(self.detection_buffer)
            audio_copy = list(self.audio_buffer)

        # Calculate statistics (now with local copies)
        fps_values = [m.fps for m in perf_copy]
        latencies = [m.latency_ms for m in perf_copy]

        # Count detections by class
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

        # Count audio events
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
        
        # Log close event
        self._log_system_event("session_end", summary)

        # Save summary in telemetry folder
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[TELEMETRY] Session finalized: {self.session_timestamp}")
        print(f"[TELEMETRY] Summary saved: {summary_path.name}")
        
        return summary
    
    # ------------------------------------------------------------------
    # Analysis Helpers
    # ------------------------------------------------------------------
    
    def get_performance_summary(self) -> Dict[str, float]:
        """
        Get real-time performance statistics.

        Thread-safe: Can be called from any thread.
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
        Get detection statistics.

        Thread-safe: Can be called from any thread.
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
        Get audio statistics.

        Thread-safe: Can be called from any thread.
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
        Write JSON line thread-safely.

        Thread-safe: Uses lock for atomic writes.
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
    Telemetry logger with async I/O to eliminate bottlenecks.

    Features:
    - Queue for non-blocking writes
    - Background thread with batch writes
    - Configurable flush interval (default: 1.0s)
    - Configurable buffer size (default: 100 lines)
    - Graceful shutdown with atexit
    - Automatic MLflow export on session finalization

    Benefits:
    - Eliminates 250-300ms spikes every ~80 frames
    - Batch writes reduce syscalls
    - Main thread never blocked by I/O

    Risk:
    - Last frames may be lost if abrupt crash
      (mitigated with atexit flush)
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
        Initialize async telemetry logger.

        Args:
            output_dir: Base directory for logs
            flush_interval: Seconds between automatic flushes (default: 2.0s)
            buffer_size: Lines accumulated before forced flush (default: 100)
            queue_maxsize: Maximum queue size (default: 2000, prevents OOM)
            mlflow_enabled: Export to MLflow on finalization (default: True)
            mlflow_experiment: MLflow experiment name
        """
        # Queue for async writes (BEFORE super().__init__)
        self._write_queue: queue.Queue = queue.Queue(maxsize=queue_maxsize)
        self._flush_interval = flush_interval
        self._buffer_size = buffer_size
        self._shutdown_flag = threading.Event()

        # MLflow config
        self._mlflow_enabled = mlflow_enabled
        self._mlflow_experiment = mlflow_experiment
        self._mlflow_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlflow")

        # Now initialize base class (which calls _write_jsonl)
        super().__init__(output_dir)

        # Background thread for flush
        self._flush_thread = threading.Thread(
            target=self._flush_worker,
            daemon=True,  # Daemon to not block exit
            name="AsyncTelemetryFlusher"
        )
        self._flush_thread.start()

        # Register shutdown handler for final flush
        atexit.register(self._shutdown)

        self._log_system_event("async_telemetry_enabled", {
            "flush_interval": flush_interval,
            "buffer_size": buffer_size,
            "queue_maxsize": queue_maxsize,
            "mlflow_enabled": mlflow_enabled,
            "mlflow_experiment": mlflow_experiment
        })

        print(f"[TELEMETRY] Async mode enabled (flush={flush_interval}s, buffer={buffer_size})")
        if mlflow_enabled:
            print(f"[TELEMETRY] MLflow enabled: {mlflow_experiment}")
    
    def _write_jsonl(self, path: Path, data: Dict[str, Any]) -> None:
        """
        Queue write instead of blocking.

        Override of synchronous method to use queue.
        """
        try:
            self._write_queue.put_nowait((path, data))
        except queue.Full:
            # Log error but don't block main thread
            print(f"[TELEMETRY WARNING] Queue full, dropping write to {path.name}")

    def _flush_worker(self) -> None:
        """
        Background thread for batch writes.

        Logic:
        1. Collect writes from queue with timeout
        2. Accumulate in buffers per file
        3. Flush when buffer full OR timeout
        """
        buffers: Dict[Path, List[str]] = {}
        last_flush = time.time()
        
        while not self._shutdown_flag.is_set():
            try:
                # Wait for write with timeout
                path, data = self._write_queue.get(timeout=0.1)

                # Serialize JSON
                try:
                    line = json.dumps(data, ensure_ascii=True)
                except TypeError:
                    line = json.dumps({"error": "serialization_failed"})

                # Accumulate in buffer
                if path not in buffers:
                    buffers[path] = []
                buffers[path].append(line)

                # Flush if buffer full
                if len(buffers[path]) >= self._buffer_size:
                    self._flush_buffer(path, buffers[path])
                    buffers[path] = []
                    last_flush = time.time()

            except queue.Empty:
                # Timeout: flush all buffers if interval passed
                now = time.time()
                if now - last_flush >= self._flush_interval:
                    for file_path, lines in list(buffers.items()):
                        if lines:
                            self._flush_buffer(file_path, lines)
                            buffers[file_path] = []
                    last_flush = now

        # Final flush on shutdown
        for file_path, lines in buffers.items():
            if lines:
                self._flush_buffer(file_path, lines)

    def _flush_buffer(self, path: Path, lines: List[str]) -> None:
        """
        Batch write to disk.

        Writes multiple lines at once to reduce syscalls.
        """
        try:
            with open(path, 'a', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n')
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to flush {path.name}: {e}")
    
    def _shutdown(self) -> None:
        """
        Graceful shutdown: flush pending writes and stop thread.

        Called automatically by atexit.
        """
        print("[TELEMETRY] Flushing pending writes...")
        self._shutdown_flag.set()

        # Wait for thread to terminate (with timeout)
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
        Finalize session, generate summary, and export to MLflow.

        Args:
            model_name: YOLO model name
            resolution: Input resolution
            depth_enabled: Whether depth estimation was active
            tensorrt_enabled: Whether TensorRT was active
            extra_params: Additional parameters for MLflow

        Returns:
            Dict with session statistics
        """
        # Call parent method to generate summary.json
        summary = super().finalize_session()

        # Export to MLflow (async, non-blocking)
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
        """Export session to MLflow (executed in separate thread)."""
        try:
            import mlflow

            # SQLite backend local - prefer project's `mlruns/` folder
            try:
                project_root = Path(__file__).resolve().parents[4]
                mlruns_dir = project_root / "mlruns"
                mlruns_dir.mkdir(parents=True, exist_ok=True)
                db_path = mlruns_dir / "mlflow.db"
                tracking_uri = f"sqlite:///{db_path}"
            except Exception as e:
                # Fallback: use user's home directory if project path creation fails
                fallback_dir = Path.home() / "mlruns"
                try:
                    fallback_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                db_path = fallback_dir / "mlflow.db"
                tracking_uri = f"sqlite:///{db_path}"

            print(f"[TELEMETRY] MLflow tracking URI: {tracking_uri}")
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

                # Detections by class
                for cls, count in summary.get("detections_by_class", {}).items():
                    safe_cls = cls.replace(" ", "_").replace("-", "_")
                    mlflow.log_metric(f"det_{safe_cls}", count)

                # Detections by distance
                for bucket, count in summary.get("detections_by_distance_bucket", {}).items():
                    mlflow.log_metric(f"dist_{bucket}", count)

                # Tags
                mlflow.set_tag("session_dir", str(self.session_dir))

                # Artifact: save summary.json
                summary_path = self.output_dir / "summary.json"
                if summary_path.exists():
                    mlflow.log_artifact(str(summary_path))

            print(f"[MLflow] Session exported: {self.session_timestamp}")
            print(f"[MLflow] View results: mlflow ui --backend-store-uri \"{tracking_uri}\"")

        except ImportError:
            print("[MLflow] mlflow not installed - skip export (pip install mlflow)")
        except Exception as e:
            print(f"[MLflow] Export error: {e}")

