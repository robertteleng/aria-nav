"""Modular pipeline for RGB frame processing."""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch  # FASE 1 / Tarea 4: Para CUDA streams

from utils.config import Config
from utils.depth_logger import get_depth_logger

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
    ) -> None:
        self.yolo_processor = yolo_processor
        self.image_enhancer = image_enhancer
        self.depth_estimator = depth_estimator or self._build_depth_estimator()

        self.depth_frame_skip = max(1, getattr(Config, "DEPTH_FRAME_SKIP", 1))
        self.latest_depth_map: Optional[np.ndarray] = None
        self.latest_depth_raw: Optional[np.ndarray] = None
        self.frames_processed = 0
        
        # FASE 2: Multiprocessing setup
        self.multiproc_enabled = getattr(Config, "PHASE2_MULTIPROC_ENABLED", False)
        if self.multiproc_enabled:
            self._init_multiproc()
        else:
            log.info("[Pipeline] Running in sequential mode")
        
        # FASE 1 / Tarea 4: CUDA Streams para paralelización
        self.use_cuda_streams = getattr(Config, 'CUDA_STREAMS', False) and torch.cuda.is_available()
        if self.use_cuda_streams:
            self.yolo_stream = torch.cuda.Stream()
            self.depth_stream = torch.cuda.Stream()
            print("[INFO] ✓ CUDA streams habilitados (YOLO + Depth en paralelo)")
        else:
            self.yolo_stream = None
            self.depth_stream = None
            if torch.cuda.is_available():
                print("[INFO] ⚠️ CUDA streams deshabilitados (ejecución secuencial)")
        
        # Log depth estimator status
        if self.depth_estimator is None:
            print("[WARN] ⚠️ Depth estimator is None - depth estimation disabled")
        elif getattr(self.depth_estimator, "model", None) is None:
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
                if self.depth_estimator is not None and getattr(self.depth_estimator, "model", None) is not None:
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
            if self.depth_estimator is not None and getattr(self.depth_estimator, "model", None) is not None:
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
        mp.set_start_method('spawn', force=True)
        
        from core.processing.central_worker import central_gpu_worker
        from core.processing.slam_worker import slam_gpu_worker
        
        self.central_queue = mp.Queue(maxsize=getattr(Config, "PHASE2_QUEUE_MAXSIZE", 2))
        self.slam_queue = mp.Queue(maxsize=getattr(Config, "PHASE2_SLAM_QUEUE_MAXSIZE", 4))
        self.result_queue = mp.Queue(maxsize=getattr(Config, "PHASE2_RESULT_QUEUE_MAXSIZE", 6))
        self.stop_event = mp.Event()
        
        self.workers = [
            mp.Process(
                target=central_gpu_worker,
                args=(self.central_queue, self.result_queue, self.stop_event),
                name="CentralWorker",
            ),
            mp.Process(
                target=slam_gpu_worker,
                args=(self.slam_queue, self.result_queue, self.stop_event),
                name="SLAMWorker",
            ),
        ]
        
        for worker in self.workers:
            worker.start()
        
        log.info("[Pipeline] Multiprocessing workers started")
        
        self.stats = {
            "frames_processed": 0,
            "dropped_central": 0,
            "dropped_slam": 0,
            "timeout_errors": 0,
            "last_stats_time": time.time(),
        }

    def _process_multiproc(self, frames_dict: Dict[str, np.ndarray], profile: bool) -> PipelineResult:
        frame_id = self.frames_processed
        self.frames_processed += 1
        timestamp = time.time()
        
        # Enhancement solo en central
        central_frame = frames_dict.get("central")
        if central_frame is None:
            log.warning("No central frame in frames_dict, falling back")
            return self._process_sequential(frames_dict.get("central", np.zeros((480, 640, 3), dtype=np.uint8)), profile)
        
        if self.image_enhancer:
            try:
                central_frame = self.image_enhancer.enhance_frame(central_frame)
            except Exception as err:
                log.warning(f"Image enhancement failed: {err}")
        
        # Distribute to workers
        try:
            self.central_queue.put(
                {
                    "frame_id": frame_id,
                    "camera": "central",
                    "frame": central_frame,
                    "timestamp": timestamp,
                },
                timeout=0.1,
            )
        except queue.Full:
            self.stats["dropped_central"] += 1
            log.warning(f"Dropped central frame {frame_id}")
        
        for camera in ["slam1", "slam2"]:
            slam_frame = frames_dict.get(camera)
            if slam_frame is not None:
                try:
                    self.slam_queue.put(
                        {
                            "frame_id": frame_id,
                            "camera": camera,
                            "frame": slam_frame,
                            "timestamp": timestamp,
                        },
                        timeout=0.05,
                    )
                except queue.Full:
                    self.stats["dropped_slam"] += 1
        
        # Collect results (espera 3 cámaras)
        results = {}
        timeout_count = 0
        
        for _ in range(3):
            try:
                result = self.result_queue.get(timeout=1.0)
                results[result.camera] = result
                timeout_count = 0
            except queue.Empty:
                timeout_count += 1
                self.stats["timeout_errors"] += 1
                
                if timeout_count >= 3:
                    log.error("3 consecutive timeouts, falling back to sequential")
                    Config.PHASE2_MULTIPROC_ENABLED = False
                    self.multiproc_enabled = False
                    return self._process_sequential(central_frame, profile)
        
        # Update stats
        self.stats["frames_processed"] += 1
        if time.time() - self.stats["last_stats_time"] > getattr(Config, "PHASE2_STATS_INTERVAL", 5.0):
            self._print_stats()
        
        # Merge results
        return self._merge_results(results, frames_dict)

    def _merge_results(self, results: Dict[str, Any], frames_dict: Dict[str, np.ndarray]) -> PipelineResult:
        central_result = results.get("central")
        
        all_detections = []
        for camera, result in results.items():
            for det in result.detections:
                det["camera"] = camera
                all_detections.append(det)
        
        # Update latest depth
        if central_result and central_result.depth_map is not None:
            self.latest_depth_map = central_result.depth_map
            self.latest_depth_raw = central_result.depth_raw
        
        timings = {}
        if central_result:
            timings["multiproc"] = True
            timings["central_ms"] = central_result.latency_ms
            timings["slam1_ms"] = results.get("slam1", type("obj", (), {"latency_ms": 0})).latency_ms
            timings["slam2_ms"] = results.get("slam2", type("obj", (), {"latency_ms": 0})).latency_ms
        
        return PipelineResult(
            frame=frames_dict.get("central", np.zeros((480, 640, 3), dtype=np.uint8)),
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
            f"Dropped central={self.stats['dropped_central']} | "
            f"Dropped SLAM={self.stats['dropped_slam']} | "
            f"Timeout errors={self.stats['timeout_errors']}"
        )
        
        self.stats["frames_processed"] = 0
        self.stats["dropped_central"] = 0
        self.stats["dropped_slam"] = 0
        self.stats["last_stats_time"] = time.time()

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
        
        for q in [self.central_queue, self.slam_queue, self.result_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except:
                    break
        
        log.info("[Pipeline] Shutdown complete")

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
            
            if getattr(estimator, "model", None) is None:
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
