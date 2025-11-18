import logging
import os
import queue
import time

import torch
from torch import cuda

from core.processing.multiproc_types import ResultMessage
from core.vision.yolo_processor import YoloProcessor

log = logging.getLogger("SlamWorker")


def _set_cuda_device() -> None:
    if not cuda.is_available():
        raise RuntimeError("SLAM worker requires CUDA")
    cuda.set_device(0)
    log.info("[SlamWorker] CUDA device set to 0")


class SlamWorker:
    def __init__(self, slam_queue, result_queue, stop_event):
        self.slam_queue = slam_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.yolo_processor: YoloProcessor | None = None

    def _load_model(self) -> None:
        self.yolo_processor = YoloProcessor.from_profile("slam")
        log.info("[SlamWorker] Model loaded")

    def _process_frame(self, msg: dict) -> ResultMessage:
        assert self.yolo_processor is not None
        start_time = time.perf_counter()
        detections = self.yolo_processor.process_frame(msg["frame"])
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 🔧 FIX: Marcar detecciones como SLAM para filtrado correcto
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
        # Configure environment for headless operation
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        
        log.info("[SlamWorker] Starting run loop")
        try:
            _set_cuda_device()
            self._load_model()

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
            log.info("[SlamWorker] Shutdown complete")


def slam_gpu_worker(slam_queue, result_queue, stop_event) -> None:
    SlamWorker(slam_queue, result_queue, stop_event).run_loop()
