"""Modular pipeline for RGB frame processing."""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch  # FASE 1 / Tarea 4: Para CUDA streams

from core.processing.shared_memory_manager import SharedMemoryRingBuffer

from utils.config import Config
from utils.depth_logger import get_depth_logger
from utils.profiler import get_profiler
from utils.system_monitor import get_monitor

log = logging.getLogger("NavigationPipeline")

try:
    from core.vision.depth_estimator import DepthEstimator
except Exception:
    DepthEstimator = None


@dataclass
class PipelineResult:
    """Container for pipeline outputs."""

    frame: np.ndarray
    detections: Any
    depth_map: Optional[np.ndarray]
    depth_raw: Optional[np.ndarray]
    timings: Dict[str, float]


class NavigationPipeline:
    """Encapsulates enhancement, depth estimation and YOLO detection."""

    def __init__(
        self,
        yolo_processor,
        *,
        image_enhancer=None,
        depth_estimator=None,
        camera_id: str = 'rgb',  # PHASE 6: Identify camera (rgb/slam1/slam2)
    ) -> None:
        # PHASE 6: Store camera ID
        self.camera_id = camera_id
        
        # FASE 2: Check multiprocessing mode FIRST
        self.multiproc_enabled = getattr(Config, "PHASE2_MULTIPROC_ENABLED", False)
        
        # CRITICAL: Main process must NOT initialize CUDA when multiprocessing
        # Only workers should touch GPU to avoid spawn conflicts
        if self.multiproc_enabled:
            log.info("[Pipeline] Multiprocessing mode - skipping GPU initialization in main process")
            self.yolo_processor = None
            self.depth_estimator = None
        else:
            self.yolo_processor = yolo_processor
            self.depth_estimator = depth_estimator or self._build_depth_estimator()
        
        self.image_enhancer = image_enhancer
        self.depth_frame_skip = max(1, getattr(Config, "DEPTH_FRAME_SKIP", 1))
        self.latest_depth_map: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.frames_processed = 0
        
        # Profiling & monitoring
        self.profiler = get_profiler()
        self.system_monitor = get_monitor()
        self.last_monitor_check = time.time()
        self.monitor_interval = 5.0  # Print stats every 5s
        
        # FASE 2: Multiprocessing setup
        if self.multiproc_enabled:
            self._init_multiproc()
        else:
            log.info("[Pipeline] Running in sequential mode")
        
        # PHASE 6: Hybrid mode - CUDA Streams configuration
        # Allows streams in main process even with multiprocessing enabled
        enable_streams = getattr(Config, 'CUDA_STREAMS', False) and torch.cuda.is_available()
        phase6_hybrid = getattr(Config, 'PHASE6_HYBRID_STREAMS', False)
        
        if self.multiproc_enabled and phase6_hybrid:
            # PHASE 6: Hybrid mode active
            if self.camera_id == 'rgb':
                # Main process (RGB camera): Enable streams for Depth + YOLO parallelization
                self.use_cuda_streams = enable_streams
                if self.use_cuda_streams:
                    self.yolo_stream = torch.cuda.Stream()
                    self.depth_stream = torch.cuda.Stream()
                    print("[INFO] ✅ PHASE 6: Hybrid mode - CUDA streams in main process")
                    print("[INFO]    → Depth + YOLO parallel on RGB camera")
                else:
                    self.yolo_stream = None
                    self.depth_stream = None
                    print("[INFO] 🔄 Main process: Sequential (streams disabled in config)")
            else:
                # Workers (SLAM cameras): No streams (YOLO-only, nothing to parallelize)
                self.use_cuda_streams = False
                self.yolo_stream = None
                self.depth_stream = None
                print(f"[INFO] 🔄 Worker {self.camera_id}: Sequential (YOLO-only, no streams needed)")
        elif not self.multiproc_enabled:
            # Original behavior: Single-process mode
            self.use_cuda_streams = enable_streams
            if self.use_cuda_streams:
                self.yolo_stream = torch.cuda.Stream()
                self.depth_stream = torch.cuda.Stream()
                print("[INFO] ✓ CUDA streams habilitados (YOLO + Depth en paralelo)")
            else:
                self.yolo_stream = None
                self.depth_stream = None
                if torch.cuda.is_available():
                    print("[INFO] ⚠️ CUDA streams deshabilitados (ejecución secuencial)")
        else:
            # Legacy behavior: Multiprocessing without Phase 6
            self.use_cuda_streams = False
            self.yolo_stream = None
            self.depth_stream = None
            print("[INFO] 🔄 Multiprocessing mode - GPU work handled by workers")
        
        # Log depth estimator status (only for non-multiproc or main process)
        if not self.multiproc_enabled or (self.multiproc_enabled and self.camera_id == 'rgb'):
            if self.depth_estimator is None:
                print("[WARN] ⚠️ Depth estimator is None - depth estimation disabled")
            elif getattr(self.depth_estimator, "model", None) is None and getattr(self.depth_estimator, "ort_session", None) is None:
                print("[WARN] ⚠️ Depth estimator model failed to load - depth estimation disabled")
            else:
                print(f"[INFO] ✅ Depth estimator initialized: {getattr(self.depth_estimator, 'backend', 'unknown')}")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray, *, profile: bool = False, frames_dict: Optional[Dict[str, np.ndarray]] = None) -> PipelineResult:
        """Run the RGB processing pipeline."""
        
        if self.multiproc_enabled and frames_dict is not None:
            return self._process_multiproc(frames_dict, profile)

        self.frames_processed += 1

        timings: Dict[str, float] = {}

        processed_frame = frame
        if self.image_enhancer is not None:
            enhance_start = time.perf_counter() if profile else None
            try:
                processed_frame = self.image_enhancer.enhance_frame(frame)
            except Exception as err:
                print(f"[WARN] Image enhancement skipped: {err}")
                processed_frame = frame
            if profile and enhance_start is not None:
                timings["enhance"] = time.perf_counter() - enhance_start

        # FASE 1 / Tarea 4: Ejecutar Depth y YOLO en paralelo con CUDA streams
        depth_map = None
        depth_raw = None
        
        if self.use_cuda_streams:
            # ========== EJECUCIÓN PARALELA ==========
            depth_start = time.perf_counter() if profile else None
            yolo_start = time.perf_counter() if profile else None
            
            # Stream 1: Depth estimation (operación más lenta)
            with torch.cuda.stream(self.depth_stream):
                if self.depth_estimator is not None and (getattr(self.depth_estimator, "model", None) is not None or getattr(self.depth_estimator, "ort_session", None) is not None):
                    try:
                        if self.frames_processed % self.depth_frame_skip == 0:
                            depth_prediction = None
                            if hasattr(self.depth_estimator, "estimate_depth_with_details"):
                                depth_prediction = self.depth_estimator.estimate_depth_with_details(processed_frame)
                                if depth_prediction is not None:
                                    self.latest_depth_map = depth_prediction.map_8bit
                                    self.latest_depth_raw = getattr(depth_prediction, "raw", None)
                            else:
                                depth_candidate = self.depth_estimator.estimate_depth(processed_frame)
                                if depth_candidate is not None:
                                    self.latest_depth_map = depth_candidate
                                    self.latest_depth_raw = None
                    except Exception as err:
                        print(f"[WARN] Depth estimation skipped: {err}")
            
            # Stream 2: YOLO detection (operación más rápida)
            with torch.cuda.stream(self.yolo_stream):
                # Usar depth map anterior (el actual se está computando en paralelo)
                detections = self.yolo_processor.process_frame(
                    processed_frame,
                    self.latest_depth_map,
                    self.latest_depth_raw,
                )
            
            # Sincronizar ambos streams antes de continuar
            torch.cuda.synchronize()
            
            if profile and depth_start is not None:
                timings["depth"] = time.perf_counter() - depth_start
            if profile and yolo_start is not None:
                timings["yolo"] = time.perf_counter() - yolo_start
            
            depth_map = self.latest_depth_map
            depth_raw = self.latest_depth_raw
        else:
            # ========== EJECUCIÓN SECUENCIAL (fallback) ==========
            if self.depth_estimator is not None and (getattr(self.depth_estimator, "model", None) is not None or getattr(self.depth_estimator, "ort_session", None) is not None):
                depth_start = time.perf_counter() if profile else None
                try:
                    if self.frames_processed % self.depth_frame_skip == 0:
                        depth_prediction = None
                        if hasattr(self.depth_estimator, "estimate_depth_with_details"):
                            depth_prediction = self.depth_estimator.estimate_depth_with_details(processed_frame)
                            if depth_prediction is not None:
                                self.latest_depth_map = depth_prediction.map_8bit
                                self.latest_depth_raw = getattr(depth_prediction, "raw", None)
                                if self.frames_processed % 100 == 0:
                                    print(f"[DEPTH] Frame {self.frames_processed}: Depth computed, shape={depth_prediction.map_8bit.shape}, inference={depth_prediction.inference_ms:.1f}ms")
                        else:
                            depth_candidate = self.depth_estimator.estimate_depth(processed_frame)
                            if depth_candidate is not None:
                                self.latest_depth_map = depth_candidate
                                self.latest_depth_raw = None
                                if self.frames_processed % 100 == 0:
                                    print(f"[DEPTH] Frame {self.frames_processed}: Depth computed, shape={depth_candidate.shape}")
                    depth_map = self.latest_depth_map
                    depth_raw = self.latest_depth_raw
                except Exception as err:
                    print(f"[WARN] Depth estimation skipped: {err}")
                if profile and depth_start is not None:
                    timings["depth"] = time.perf_counter() - depth_start

            yolo_start = time.perf_counter() if profile else None
            detections = self.yolo_processor.process_frame(
                processed_frame,
                depth_map,
                depth_raw,
            )
            if profile and yolo_start is not None:
                timings["yolo"] = time.perf_counter() - yolo_start

        # CRITICAL: Tag detections with camera_source for proper rendering separation
        # This prevents RGB/SLAM detection mixing in web dashboard display
        if detections:
            for det in detections:
                det['camera_source'] = self.camera_id  # 'rgb' for main camera

        return PipelineResult(
            frame=processed_frame,
            detections=detections,
            depth_map=depth_map,
            depth_raw=depth_raw,
            timings=timings,
        )

    def get_latest_depth_map(self) -> Optional[np.ndarray]:
        return self.latest_depth_map

    def get_latest_depth_raw(self) -> Optional[np.ndarray]:
        return self.latest_depth_raw

    # ------------------------------------------------------------------
    # FASE 2: Multiprocessing helpers
    # ------------------------------------------------------------------

    def _init_multiproc(self) -> None:
        import torch.multiprocessing as mp
        # Note: start_method must be set in run.py BEFORE imports
        
        from core.processing.central_worker import central_gpu_worker
        from core.processing.slam_worker import slam_gpu_worker
        
        # Phase 7: Double Buffering Configuration
        self.double_buffering = getattr(Config, "PHASE7_DOUBLE_BUFFERING", False)
        self.worker_health_interval = getattr(Config, "PHASE7_WORKER_HEALTH_CHECK_INTERVAL", 5.0)
        self.graceful_degradation = getattr(Config, "PHASE7_GRACEFUL_DEGRADATION", True)
        self.last_health_check = time.time()
        
        # Worker instance tracking
        self.worker_instances = {
            "central": [],
            "slam": []
        }
        self.worker_health = {}  # {name: is_alive}
        self.submit_counter = 0
        
        self.result_queue = mp.Queue(maxsize=getattr(Config, "PHASE2_RESULT_QUEUE_MAXSIZE", 10))
        self.stop_event = mp.Event()
        
        # Events for readiness
        self.central_ready = mp.Event()
        self.slam_ready = mp.Event()
        
        self.workers = []
        
        if self.double_buffering:
            log.info("[Pipeline] 🚀 Initializing DOUBLE BUFFERING (2x workers per type)")
            
            # Create 2 sets of queues
            self.central_queues = [mp.Queue(maxsize=2), mp.Queue(maxsize=2)]
            self.slam_queues = [mp.Queue(maxsize=2), mp.Queue(maxsize=2)]
            
            # Create 2 RGB Workers
            for i in range(2):
                worker_name = f"RGBWorker_{'A' if i==0 else 'B'}"
                p = mp.Process(
                    target=central_gpu_worker,
                    args=(self.central_queues[i], self.result_queue, self.stop_event, self.central_ready),
                    name=worker_name,
                )
                self.workers.append(p)
                self.worker_instances["rgb"].append({"proc": p, "queue": self.central_queues[i], "idx": i})
                self.worker_health[worker_name] = True

            # Create 2 SLAM Workers
            for i in range(2):
                worker_name = f"SlamWorker_{'A' if i==0 else 'B'}"
                p = mp.Process(
                    target=slam_gpu_worker,
                    args=(self.slam_queues[i], self.result_queue, self.stop_event, self.slam_ready),
                    name=worker_name,
                )
                self.workers.append(p)
                self.worker_instances["slam"].append({"proc": p, "queue": self.slam_queues[i], "idx": i})
                self.worker_health[worker_name] = True
                
        else:
            log.info("[Pipeline] Initializing SINGLE BUFFERING (1 worker per type)")
            # Legacy single queue mode
            self.central_queue = mp.Queue(maxsize=2)
            self.slam_queue = mp.Queue(maxsize=2)
            self.central_queues = [self.central_queue]
            self.slam_queues = [self.slam_queue]
            
            # RGB Worker
            p_central = mp.Process(
                target=central_gpu_worker,
                args=(self.central_queue, self.result_queue, self.stop_event, self.central_ready),
                name="RGBWorker",
            )
            self.workers.append(p_central)
            self.worker_instances["rgb"].append({"proc": p_central, "queue": self.central_queue, "idx": 0})
            self.worker_health["RGBWorker"] = True
            
            # SLAM Worker
            p_slam = mp.Process(
                target=slam_gpu_worker,
                args=(self.slam_queue, self.result_queue, self.stop_event, self.slam_ready),
                name="SLAMWorker",
            )
            self.workers.append(p_slam)
            self.worker_instances["slam"].append({"proc": p_slam, "queue": self.slam_queue, "idx": 0})
            self.worker_health["SLAMWorker"] = True
        
        # Start all workers
        for worker in self.workers:
            worker.start()
        
        # Wait for workers to signal ready via Events
        log.info("[Pipeline] Waiting for workers to load models and signal ready...")
        start_wait = time.time()
        max_wait = 60.0 if self.double_buffering else 30.0  # More time for double workers
        
        # Wait for RGB worker(s) - shared event, so first one triggers it
        # Ideally we'd have separate events, but for now we assume if one is ready, others are close
        if not self.central_ready.wait(timeout=max_wait):
            log.error("[Pipeline] RGB worker failed to signal ready")
            raise RuntimeError("RGB worker initialization timeout")
        log.info(f"[Pipeline] RGB worker(s) ready signal received")
        
        # Wait for SLAM worker(s)
        remaining_time = max_wait - (time.time() - start_wait)
        if not self.slam_ready.wait(timeout=max(remaining_time, 1.0)):
            log.error("[Pipeline] SLAM worker failed to signal ready")
            raise RuntimeError("SLAM worker initialization timeout")
        
        elapsed = time.time() - start_wait
        log.info(f"[Pipeline] All workers ready in {elapsed:.1f}s")
        
        # Final check if workers are alive
        for worker in self.workers:
            if not worker.is_alive():
                exitcode = worker.exitcode
                log.error(f"[Pipeline] Worker {worker.name} failed! exitcode={exitcode}")
                raise RuntimeError(f"Worker {worker.name} crashed (exitcode={exitcode})")
        
        log.info(f"[Pipeline] Workers initialized: {[w.name for w in self.workers]}")
        
        self.stats = {
            "frames_processed": 0,
            "dropped_central": 0,
            "dropped_slam": 0,
            "timeout_errors": 0,
            "stale_results": 0,  # SOLUTION #3: Track stale result usage
            "last_stats_time": time.time(),
        }
        
        # Overlap/pipelining support
        self.pending_results = {}  # {camera: ResultMessage}
        self.last_frame_id = -1
        
        # SOLUTION #3: Cache for non-blocking with age tracking
        self.cached_result = None
        self.result_timestamp = 0
        self.first_result_received = False

        # PHASE 3: Shared Memory Initialization (optional)
        self.use_shared_memory = getattr(Config, "USE_SHARED_MEMORY", False)
        self.shm_central = None
        self.shm_slam1 = None
        self.shm_slam2 = None
        
        if self.use_shared_memory:
            # Aria RGB camera delivers 1408x1408 frames
            # SLAM cameras also deliver similar resolution
            self.shm_shape = (1408, 1408, 3)
            self.shm_dtype = np.uint8
            
            try:
                self.shm_central = SharedMemoryRingBuffer(
                    name_prefix="aria_rgb",
                    count=getattr(Config, "PHASE2_QUEUE_MAXSIZE", 2) + 2, # +2 for safety buffer
                    shape=self.shm_shape,
                    dtype=self.shm_dtype,
                    create=True
                )
                
                self.shm_slam1 = SharedMemoryRingBuffer(
                    name_prefix="aria_slam1",
                    count=getattr(Config, "PHASE2_SLAM_QUEUE_MAXSIZE", 4) + 2,
                    shape=self.shm_shape,
                    dtype=self.shm_dtype,
                    create=True
                )
                
                self.shm_slam2 = SharedMemoryRingBuffer(
                    name_prefix="aria_slam2",
                    count=getattr(Config, "PHASE2_SLAM_QUEUE_MAXSIZE", 4) + 2,
                    shape=self.shm_shape,
                    dtype=self.shm_dtype,
                    create=True
                )
                log.info("[Pipeline] Shared Memory buffers initialized")
            except Exception as e:
                log.error(f"[Pipeline] Failed to init shared memory: {e}")
                raise
        else:
            log.info("[Pipeline] Shared Memory disabled - using direct frame passing")

    def _process_multiproc(self, frames_dict: Dict[str, np.ndarray], profile: bool) -> PipelineResult:
        frame_id = self.frames_processed
        self.frames_processed += 1
        
        # Enhancement solo en central
        central_frame = frames_dict.get("central")
        if central_frame is None:
            log.warning("No central frame in frames_dict, falling back")
            return self._process_sequential(frames_dict.get("central", np.zeros((480, 640, 3), dtype=np.uint8)), profile)
        
        # OPTIMIZATION: Resize input frames to reduce IPC overhead
        if getattr(Config, "INPUT_RESIZE_ENABLED", False):
            import cv2
            target_w = getattr(Config, "INPUT_RESIZE_WIDTH", 1024)
            target_h = getattr(Config, "INPUT_RESIZE_HEIGHT", 1024)
            if central_frame.shape[1] != target_w or central_frame.shape[0] != target_h:
                central_frame = cv2.resize(central_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                # Also resize SLAM frames
                for key in ["slam1", "slam2"]:
                    if key in frames_dict and frames_dict[key] is not None:
                        frames_dict[key] = cv2.resize(frames_dict[key], (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        if self.image_enhancer:
            try:
                central_frame = self.image_enhancer.enhance_frame(central_frame)
            except Exception as err:
                log.warning(f"Image enhancement failed: {err}")
        
        # Phase 1: Enqueue frames to workers (non-blocking)
        timestamp = time.time()
        self._enqueue_frames(frame_id, frames_dict, central_frame, timestamp)
        
        # Phase 2: Collect results with overlap (puede ser de frame anterior)
        results = self._collect_results_with_overlap(frame_id, central_frame, profile)
        
        # Update stats
        self.stats["frames_processed"] += 1
        if time.time() - self.stats["last_stats_time"] > getattr(Config, "PHASE2_STATS_INTERVAL", 5.0):
            self._print_stats()
        
        # Merge results
        return self._merge_results(results, frames_dict)
    
    def _monitor_worker_health(self) -> None:
        """Check if workers are alive and handle failures"""
        if not self.graceful_degradation:
            return
            
        for category, instances in self.worker_instances.items():
            for instance in instances:
                worker = instance["proc"]
                if not worker.is_alive() and self.worker_health.get(worker.name, False):
                    log.error(f"[Pipeline] ⚠️ Worker {worker.name} CRASHED! Exitcode: {worker.exitcode}")
                    self.worker_health[worker.name] = False
                    
                    # Check if we have any workers left in this category
                    alive_count = sum(1 for inst in instances if self.worker_health.get(inst["proc"].name, False))
                    if alive_count == 0:
                        log.critical(f"[Pipeline] ❌ ALL {category} workers dead! System failure imminent.")
                    else:
                        log.warning(f"[Pipeline] ⚠️ Degrading performance: {alive_count} {category} workers remaining")

    def _enqueue_frames(self, frame_id: int, frames_dict: Dict[str, np.ndarray], central_frame: np.ndarray, timestamp: float) -> None:
        """Phase 1: Enviar frames a workers (no espera resultados)"""
        
        # Monitor health periodically
        if time.time() - self.last_health_check > self.worker_health_interval:
            self._monitor_worker_health()
            self.last_health_check = time.time()
        
        # Determine target worker index (Round Robin)
        # If double buffering is off, this is always 0
        # If on, it toggles 0, 1, 0, 1...
        worker_idx = self.submit_counter % 2 if self.double_buffering else 0
        self.submit_counter += 1
        
            # Distribute to workers
        try:
            # RGB
            # Check if target worker is alive
            target_instance = self.worker_instances["rgb"][worker_idx]
            if not self.worker_health.get(target_instance["proc"].name, True):
                # Try the other worker if this one is dead
                other_idx = (worker_idx + 1) % 2
                other_instance = self.worker_instances["central"][other_idx]
                if self.worker_health.get(other_instance["proc"].name, True):
                    target_instance = other_instance
                    # log.debug(f"Redirecting frame {frame_id} to {target_instance['proc'].name}")
                else:
                    # Both dead
                    return

            target_queue = target_instance["queue"]
            
            if self.use_shared_memory:
                # Shared Memory mode (Zero-Copy)
                def _prepare_frame(frame, target_shape):
                    if frame.shape != target_shape:
                        import cv2
                        return cv2.resize(frame, (target_shape[1], target_shape[0]))
                    return frame
                
                central_ready = _prepare_frame(central_frame, self.shm_shape)
                buf_idx = self.shm_central.put(central_ready)
                
                with self.profiler.measure("queue_put_central"):
                    # SOLUTION #3: put_nowait - drop frame if worker busy
                    try:
                        target_queue.put_nowait(
                            {
                                "frame_id": frame_id,
                                "camera": "rgb",
                                "shm_name": "aria_rgb",
                                "buffer_index": buf_idx,
                                "shape": self.shm_shape,
                                "dtype": self.shm_dtype,
                                "timestamp": timestamp,
                            }
                        )
                    except queue.Full:
                        self.stats["dropped_central"] += 1
                        if self.frames_processed % 30 == 0:
                            log.warning(f"Dropped RGB frame {frame_id} (queue full)")
            else:
                # Direct mode (legacy - frame in queue)
                with self.profiler.measure("queue_put_central"):
                    # SOLUTION #3: put_nowait - drop frame if worker busy
                    try:
                        target_queue.put_nowait(
                            {
                                "frame_id": frame_id,
                                "camera": "rgb",
                                "frame": central_frame,  # Direct pass
                                "timestamp": timestamp,
                            }
                        )
                    except queue.Full:
                        self.stats["dropped_central"] += 1
                        if self.frames_processed % 30 == 0:
                            log.warning(f"Dropped RGB frame {frame_id} (queue full)")
        except Exception as e:
            log.error(f"Error enqueuing RGB frame: {e}")
        
        # SLAM
        # Similar logic for SLAM workers
        target_instance_slam = self.worker_instances["slam"][worker_idx]
        if not self.worker_health.get(target_instance_slam["proc"].name, True):
             # Try the other worker if this one is dead
            other_idx = (worker_idx + 1) % 2
            other_instance = self.worker_instances["slam"][other_idx]
            if self.worker_health.get(other_instance["proc"].name, True):
                target_instance_slam = other_instance
            else:
                return
        
        target_queue_slam = target_instance_slam["queue"]

        if self.use_shared_memory:
            def _prepare_frame(frame, target_shape):
                if frame.shape != target_shape:
                    import cv2
                    return cv2.resize(frame, (target_shape[1], target_shape[0]))
                return frame
            
            for camera, shm_buffer in [("slam1", self.shm_slam1), ("slam2", self.shm_slam2)]:
                slam_frame = frames_dict.get(camera)
                if slam_frame is not None:
                    try:
                        slam_ready = _prepare_frame(slam_frame, self.shm_shape)
                        buf_idx = shm_buffer.put(slam_ready)
                        
                        # SOLUTION #3: put_nowait for SLAM too
                        try:
                            target_queue_slam.put_nowait(
                                {
                                    "frame_id": frame_id,
                                    "camera": camera,
                                    "shm_name": f"aria_{camera}",
                                    "buffer_index": buf_idx,
                                    "shape": self.shm_shape,
                                    "dtype": self.shm_dtype,
                                    "timestamp": timestamp,
                                }
                            )
                        except queue.Full:
                            self.stats["dropped_slam"] += 1
                    except Exception as e:
                        log.error(f"Error enqueuing {camera} frame: {e}")
        else:
            # Direct mode
            for camera in ["slam1", "slam2"]:
                slam_frame = frames_dict.get(camera)
                if slam_frame is not None:
                    try:
                        # SOLUTION #3: put_nowait for SLAM direct mode
                        target_queue_slam.put_nowait(
                            {
                                "frame_id": frame_id,
                                "camera": camera,
                                "frame": slam_frame,  # Direct pass
                                "timestamp": timestamp,
                            }
                        )
                    except queue.Full:
                        self.stats["dropped_slam"] += 1
                    except Exception as e:
                        log.error(f"Error enqueuing {camera} frame: {e}")
        
    
    def _collect_results_with_overlap(self, frame_id: int, central_frame: np.ndarray, profile: bool) -> Dict[str, Any]:
        """SOLUTION #3: Non-blocking collection with age-limited cache"""
        results = {}
        
        # System monitoring cada N segundos
        if time.time() - self.last_monitor_check > self.monitor_interval:
            self._print_system_metrics()
            self.last_monitor_check = time.time()
        
        # Track if we got a NEW result in this iteration
        got_new_result = False
        
        # SOLUTION #3: Drain all available results (non-blocking)
        while True:
            try:
                result = self.result_queue.get_nowait()
                # Update cache with latest result
                self.cached_result = result
                self.result_timestamp = time.time()
                self.first_result_received = True
                got_new_result = True
                # Organize by camera
                self.pending_results[result.camera] = result
            except queue.Empty:
                break
        
        # SPECIAL CASE: First frame - must wait for initial model load
        if not self.first_result_received:
            log.info(f"[Frame {frame_id}] Waiting for first result (model loading)...")
            try:
                result = self.result_queue.get(timeout=15.0)
                self.cached_result = result
                self.result_timestamp = time.time()
                self.first_result_received = True
                got_new_result = True
                self.pending_results[result.camera] = result
                log.info(f"[Frame {frame_id}] First result received after model load")
            except queue.Empty:
                log.error(f"RGB worker timeout on frame {frame_id} after 15s")
                self.stats["timeout_errors"] += 1
                raise RuntimeError("RGB worker not responding")
        
        # SOLUTION #3: If no new result, use cached result from previous frame
        if not got_new_result and self.cached_result is not None:
            # CRITICAL: Only use cached result if it's from the central camera
            # Prevents SLAM detections from appearing in RGB frame
            if self.cached_result.camera == "rgb":
                # Check age limit (100ms safety threshold)
                age_ms = (time.time() - self.result_timestamp) * 1000
                if age_ms > 100:
                    log.warning(f"[Frame {frame_id}] Stale result: {age_ms:.1f}ms old (>100ms limit)")
                    self.stats["stale_results"] += 1
                
                # Use cached result (from previous frame)
                self.pending_results["rgb"] = self.cached_result
                if frame_id % 30 == 0:
                    log.info(f"[Frame {frame_id}] Using cached result ({age_ms:.1f}ms old)")
            else:
                # Cached result is from SLAM, don't use it for central
                if frame_id % 30 == 0:
                    log.warning(f"[Frame {frame_id}] No central result, cached is {self.cached_result.camera}")
        
        # Return available results
        results = dict(self.pending_results)
        self.pending_results.clear()
        
        return results

    def _merge_results(self, results: Dict[str, Any], frames_dict: Dict[str, np.ndarray]) -> PipelineResult:
        rgb_result = results.get("rgb")
        
        # Solo incluir detecciones de la cámara RGB en el PipelineResult
        # Las SLAM se manejan por separado en main.py
        all_detections = []
        if rgb_result:
            for det in rgb_result.detections:
                det["camera"] = "rgb"
                det["camera_source"] = "rgb"  # CRITICAL: Tag for frame renderer filtering
                all_detections.append(det)
        
        # Update latest depth (keep reference, but clear old one first)
        if rgb_result and rgb_result.depth_map is not None:
            # Clear old references to allow GC
            self.latest_depth_map = None
            self.latest_depth_raw = None
            # Assign new
            self.latest_depth_map = rgb_result.depth_map
            self.latest_depth_raw = rgb_result.depth_raw
        
        timings = {}
        if rgb_result:
            timings["multiproc"] = True
            timings["rgb_ms"] = rgb_result.latency_ms
            timings["slam1_ms"] = results.get("slam1", type("obj", (), {"latency_ms": 0})).latency_ms
            timings["slam2_ms"] = results.get("slam2", type("obj", (), {"latency_ms": 0})).latency_ms
        
        # Clear result objects to help GC (they contain large numpy arrays)
        for result in results.values():
            result.depth_map = None
            result.depth_raw = None
        
        return PipelineResult(
            frame=frames_dict.get("rgb", np.zeros((480, 640, 3), dtype=np.uint8)),
            detections=all_detections,
            depth_map=self.latest_depth_map,
            depth_raw=self.latest_depth_raw,
            timings=timings,
        )

    def _print_stats(self) -> None:
        elapsed = time.time() - self.stats["last_stats_time"]
        fps = self.stats["frames_processed"] / elapsed if elapsed > 0 else 0
        
        log.info(
            f"[STATS] {elapsed:.1f}s window: FPS={fps:.1f} | "
            f"Dropped RGB={self.stats['dropped_central']} | "
            f"Dropped SLAM={self.stats['dropped_slam']} | "
            f"Stale results={self.stats['stale_results']} | "
            f"Timeout errors={self.stats['timeout_errors']}"
        )
        
        self.stats["frames_processed"] = 0
        self.stats["dropped_central"] = 0
        self.stats["dropped_slam"] = 0
        self.stats["stale_results"] = 0
        self.stats["last_stats_time"] = time.time()
    
    def _print_system_metrics(self) -> None:
        """Print system resource metrics (CPU/RAM/GPU)"""
        metrics = self.system_monitor.get_metrics()
        
        log.info("[System Metrics]")
        log.info(f"  CPU: {metrics.cpu_percent:.1f}%")
        log.info(f"  RAM: {metrics.ram_used_gb:.2f}/{metrics.ram_total_gb:.2f}GB ({metrics.ram_percent:.1f}%)")
        
        if metrics.gpu_utilization is not None:
            log.info(f"  GPU Usage: {metrics.gpu_utilization:.1f}%")
            log.info(f"  GPU VRAM: {metrics.gpu_memory_used_gb:.2f}/{metrics.gpu_memory_total_gb:.2f}GB ({metrics.gpu_memory_percent:.1f}%)")
            if metrics.gpu_temperature_c:
                log.info(f"  GPU Temp: {metrics.gpu_temperature_c}°C")
        
        # Alertas
        if metrics.ram_percent > 85:
            log.warning("⚠️ RAM usage > 85%!")
        if metrics.gpu_memory_percent and metrics.gpu_memory_percent > 90:
            log.warning("⚠️ GPU VRAM > 90%!")

    def _process_sequential(self, frame: np.ndarray, profile: bool) -> PipelineResult:
        """Fallback to original sequential processing."""
        return self.process(frame, profile=profile)

    def shutdown(self) -> None:
        if not hasattr(self, "stop_event"):
            return
        
        log.info("[Pipeline] Initiating shutdown...")
        self.stop_event.set()
        
        for worker in self.workers:
            worker.join(timeout=2)
            if worker.is_alive():
                log.warning(f"Terminating {worker.name}")
                worker.terminate()
                worker.join(timeout=1)
        
        # Clean up all queues
        queues_to_clean = [self.result_queue]
        if hasattr(self, 'central_queues'):
            queues_to_clean.extend(self.central_queues)
        if hasattr(self, 'slam_queues'):
            queues_to_clean.extend(self.slam_queues)
            
        for q in queues_to_clean:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break
        
        log.info("[Pipeline] Shutdown complete")
        
        # Cleanup Shared Memory
        try:
            if hasattr(self, 'shm_central'): self.shm_central.cleanup()
            if hasattr(self, 'shm_slam1'): self.shm_slam1.cleanup()
            if hasattr(self, 'shm_slam2'): self.shm_slam2.cleanup()
        except Exception as e:
            log.error(f"[Pipeline] Error cleaning up shared memory: {e}")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_depth_estimator(self):
        logger = get_depth_logger()
        logger.section("NavigationPipeline: Building Depth Estimator")
        
        depth_enabled = getattr(Config, "DEPTH_ENABLED", False)
        logger.log(f"DEPTH_ENABLED={depth_enabled}")
        logger.log(f"DepthEstimator class={'Available' if DepthEstimator else 'None'}")
        print(f"[DEBUG] _build_depth_estimator: DEPTH_ENABLED={depth_enabled}, DepthEstimator={'Available' if DepthEstimator else 'None'}")
        
        if not depth_enabled:
            logger.log("Depth estimation disabled in config - returning None")
            print("[INFO] Depth estimation disabled in config")
            return None
        if DepthEstimator is None:
            logger.log("DepthEstimator class not available (import failed) - returning None")
            print("[WARN] DepthEstimator class not available (import failed)")
            return None
            
        try:
            logger.log("Creating DepthEstimator instance...")
            print("[INFO] Creating DepthEstimator instance...")
            estimator = DepthEstimator()
            
            if getattr(estimator, "model", None) is None and getattr(estimator, "ort_session", None) is None:
                logger.log("⚠️ Estimator created but model is None")
                print("[WARN] ⚠️ Depth estimator initialized without model (disabled)")
            else:
                logger.log(f"✅ Estimator created successfully with backend: {getattr(estimator, 'backend', 'unknown')}")
            
            return estimator
        except Exception as err:
            logger.log(f"❌ Exception during estimator creation: {err}")
            print(f"[ERROR] ❌ Depth estimator init failed: {err}")
            import traceback
            tb = traceback.format_exc()
            logger.log(f"Traceback:\n{tb}")
            traceback.print_exc()
            return None


__all__ = ["NavigationPipeline", "PipelineResult"]
