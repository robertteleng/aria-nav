import logging
import os
import queue
import time
from typing import Any

import torch
from torch import cuda

from core.processing.multiproc_types import ResultMessage
from core.vision.depth_estimator import DepthEstimator
from core.vision.yolo_processor import YoloProcessor

log = logging.getLogger("CentralWorker")


def _set_cuda_device() -> None:
    if not cuda.is_available():
        raise RuntimeError("Central worker requires CUDA")
    cuda.set_device(0)
    log.info("[CentralWorker] CUDA device set to 0")


class CentralWorker:
    def __init__(self, central_queue, result_queue, stop_event):
        self.central_queue = central_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.depth_estimator: DepthEstimator | None = None
        self.yolo_processor: YoloProcessor | None = None
        self.depth_stream: Any = None
        self.yolo_stream: Any = None

    def _load_models(self) -> None:
        # Skip depth in multiprocessing for now (HuggingFace model loading issues)
        import os
        skip_depth = os.environ.get("ARIA_SKIP_DEPTH", "0") == "1"
        
        if not skip_depth:
            os.environ["ARIA_SKIP_DEPTH"] = "1"  # Force skip in worker
            log.info("[CentralWorker] Depth disabled in multiprocessing mode")
        
        self.depth_estimator = None  # Skip for now
        self.yolo_processor = YoloProcessor()
        self.depth_stream = torch.cuda.Stream()
        self.yolo_stream = torch.cuda.Stream()
        log.info("[CentralWorker] Models loaded and streams created")

    def _process_frame(self, msg: dict) -> ResultMessage:
        assert self.yolo_processor is not None
        assert self.depth_stream is not None and self.yolo_stream is not None

        frame = msg["frame"]
        frame_id = msg.get("frame_id", -1)
        start_time = time.perf_counter()

        depth_map = None
        depth_raw = None
        depth_ms = 0.0
        
        # Skip depth processing (disabled in multiprocessing)
        # with torch.cuda.stream(self.depth_stream):
        #     ...

        with torch.cuda.stream(self.yolo_stream):
            yolo_start = time.perf_counter()
            detections = self.yolo_processor.process_frame(frame, None, None)
            yolo_ms = (time.perf_counter() - yolo_start) * 1000

        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start_time) * 1000

        return ResultMessage(
            frame_id=frame_id,
            camera="central",
            detections=detections,
            depth_map=depth_map,
            depth_raw=depth_raw,
            latency_ms=latency_ms,
            profiling={
                "depth_ms": depth_ms,
                "yolo_ms": yolo_ms,
                "gpu_mem_mb": torch.cuda.memory_allocated() / 1e6,
            },
        )

    def run_loop(self) -> None:
        # Configure environment for headless operation
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        
        log.info("[CentralWorker] Starting run loop")
        try:
            _set_cuda_device()
            self._load_models()

            while not self.stop_event.is_set():
                try:
                    msg = self.central_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                result = self._process_frame(msg)
                self.result_queue.put(result)
        except Exception as err:  # noqa: BLE001
            log.critical("[CentralWorker] Fatal error", exc_info=err)
            self.stop_event.set()
        finally:
            torch.cuda.empty_cache()
            log.info("[CentralWorker] Shutdown complete")


def central_gpu_worker(central_queue, result_queue, stop_event) -> None:
    CentralWorker(central_queue, result_queue, stop_event).run_loop()
