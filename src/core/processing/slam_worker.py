"""
SLAM camera GPU worker for peripheral vision processing (Phase 2 multiprocessing).

This module provides a dedicated GPU worker process for SLAM peripheral cameras
(SLAM1, SLAM2) running YOLO detection with specialized 256x256 models optimized
for fast peripheral vision.

Architecture:
- Dedicated GPU process (CUDA device 0)
- YOLO-only processing (no depth estimation for SLAM cameras)
- Zero-copy shared memory ring buffer support
- Lightweight SLAM profile (256x256 YOLO model)
- Headless operation (offscreen Qt platform)

Models:
- YOLO SLAM profile (256x256 resolution for peripheral cameras)
- TensorRT/PyTorch backend

Performance:
- Lower latency than RGB processing (no depth estimation)
- Typical latency: 5-10ms per frame on NVIDIA GPU
- Zero-copy frame transfers via shared memory

Usage:
    # Entry point for multiprocessing.Process
    from multiprocessing import Process, Queue, Event

    slam_queue = Queue()
    result_queue = Queue()
    stop_event = Event()
    ready_event = Event()

    p = Process(target=slam_gpu_worker, args=(slam_queue, result_queue, stop_event, ready_event))
    p.start()
    ready_event.wait()  # Wait for model to load
"""

import logging
import os
import queue
import time

import torch
from torch import cuda

from core.processing.multiproc_types import ResultMessage
from core.vision.yolo_processor import YoloProcessor
from core.processing.shared_memory_manager import SharedMemoryRingBuffer
from utils.config import Config

log = logging.getLogger("SlamWorker")


def _set_cuda_device() -> None:
    if not cuda.is_available():
        raise RuntimeError("SLAM worker requires CUDA")
    cuda.set_device(0)
    log.info("[SlamWorker] CUDA device set to 0")


class SlamWorker:
    """SLAM camera worker for peripheral vision processing."""

    def __init__(self, slam_queue, result_queue, stop_event, ready_event=None):
        self.slam_queue = slam_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.ready_event = ready_event
        self.yolo_processor: YoloProcessor | None = None

    def _load_model(self) -> None:
        self.yolo_processor = YoloProcessor.from_profile("slam")
        log.info("[SlamWorker] Model loaded")

    def _process_frame(self, msg: dict) -> ResultMessage:
        assert self.yolo_processor is not None

        # Shared Memory Support
        if "buffer_index" in msg:
            if not hasattr(self, "shm_reader") or self.shm_reader is None:
                try:
                    # Lazy init reader
                    self.shm_reader = SharedMemoryRingBuffer(
                        name_prefix=msg["shm_name"],
                        count=getattr(Config, "PHASE2_SLAM_QUEUE_MAXSIZE", 4) + 2,
                        shape=msg["shape"],
                        dtype=msg["dtype"],
                        create=False
                    )
                    log.info(f"[SlamWorker] Attached to shared memory: {msg['shm_name']}")
                except Exception as e:
                    log.error(f"[SlamWorker] Failed to attach to SHM: {e}")
                    raise

            frame = self.shm_reader.get(msg["buffer_index"])
        else:
            frame = msg["frame"]

        start_time = time.perf_counter()
        detections = self.yolo_processor.process_frame(frame)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Mark detections as SLAM for correct filtering in navigation pipeline
        for det in detections:
            det["camera_source"] = "slam"

        return ResultMessage(
            frame_id=msg.get("frame_id", -1),
            camera=msg.get("camera", "slam"),
            detections=detections,
            latency_ms=latency_ms,
            profiling={
                "yolo_ms": latency_ms,
                "gpu_mem_mb": torch.cuda.memory_allocated() / 1e6,
            },
        )

    def run_loop(self) -> None:
        """Main worker loop for SLAM processing."""
        # Configure environment for headless operation
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

        log.info("[SlamWorker] Starting run loop")
        try:
            _set_cuda_device()
            self._load_model()

            # Signal ready to parent
            if self.ready_event:
                self.ready_event.set()
                log.info("[SlamWorker] Ready event signaled")

            while not self.stop_event.is_set():
                try:
                    msg = self.slam_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                result = self._process_frame(msg)
                self.result_queue.put(result)
        except Exception as err:  # noqa: BLE001
            log.critical("[SlamWorker] Fatal error", exc_info=err)
            self.stop_event.set()
        finally:
            torch.cuda.empty_cache()
            if hasattr(self, "shm_reader") and self.shm_reader:
                try:
                    self.shm_reader.cleanup()
                except:
                    pass
            log.info("[SlamWorker] Shutdown complete")


def slam_gpu_worker(slam_queue, result_queue, stop_event, ready_event=None) -> None:
    """Entry point for SLAM GPU worker process."""
    SlamWorker(slam_queue, result_queue, stop_event, ready_event).run_loop()
