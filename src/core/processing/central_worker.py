import logging
import os
import queue
import time
from typing import Any

import numpy as np
import torch
from torch import cuda

from core.processing.multiproc_types import ResultMessage
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
        self.depth_model: Any = None
        self.depth_processor: Any = None
        self.yolo_processor: YoloProcessor | None = None
        self.depth_stream: Any = None
        self.yolo_stream: Any = None

    def _load_models(self) -> None:
        """Load Depth Anything V2 (native implementation) + YOLO"""
        import sys
        print("[WORKER] _load_models() called", flush=True)
        log.info("[CentralWorker] Python path in worker:")
        for p in sys.path[:3]:
            log.info(f"  - {p}")
        
        print("[WORKER] Step 2.1: Starting model loading...", flush=True)
        log.info("[CentralWorker] Step 2.1: Starting model loading...")
        
        try:
            print("[WORKER] Step 2.2: Importing DepthAnythingV2...", flush=True)
            log.info("[CentralWorker] Step 2.2: Importing DepthAnythingV2...")
            from external.depth_anything_v2.dpt import DepthAnythingV2
            print("[WORKER] ✅ Import successful", flush=True)
            log.info("[CentralWorker] ✅ Import successful")
            
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            }
            
            log.info("[CentralWorker] Step 2.3: Creating model instance...")
            self.depth_model = DepthAnythingV2(**model_configs['vits'])
            log.info("[CentralWorker] ✅ Instance created")
            
            checkpoint_path = 'checkpoints/depth_anything_v2_vits.pth'
            log.info(f"[CentralWorker] Step 2.4: Loading checkpoint {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            log.info("[CentralWorker] ✅ Checkpoint loaded to CPU")
            
            log.info("[CentralWorker] Step 2.5: Loading state dict...")
            self.depth_model.load_state_dict(state_dict)
            log.info("[CentralWorker] ✅ State dict loaded")
            
            log.info("[CentralWorker] Step 2.6: Moving to CUDA...")
            self.depth_model.to('cuda')
            log.info("[CentralWorker] ✅ Moved to CUDA")
            
            self.depth_model.eval()
            log.info("[CentralWorker] ✅ Depth model ready")
            
            # YOLO
            log.info("[CentralWorker] Step 2.7: Loading YOLO...")
            self.yolo_processor = YoloProcessor()
            log.info("[CentralWorker] ✅ YOLO loaded")
            
            # CUDA streams
            self.depth_stream = torch.cuda.Stream()
            self.yolo_stream = torch.cuda.Stream()
            
            log.info("[CentralWorker] ✅✅✅ ALL MODELS LOADED SUCCESSFULLY ✅✅✅")
        except Exception as e:
            log.critical(f"[CentralWorker] ❌ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _process_frame(self, msg: dict) -> ResultMessage:
        assert self.yolo_processor is not None
        assert self.depth_model is not None
        assert self.depth_stream is not None and self.yolo_stream is not None

        frame = msg["frame"]
        frame_id = msg.get("frame_id", -1)
        start_time = time.perf_counter()

        depth_map = None
        depth_raw = None
        depth_ms = 0.0

        # Parallel execution: Depth + YOLO with CUDA streams
        with torch.cuda.stream(self.depth_stream):
            depth_start = time.perf_counter()
            print(f"[WORKER] Frame {frame_id}: Starting depth inference...", flush=True)
            
            # Convert numpy to GPU tensor (H, W, 3) uint8
            frame_tensor = torch.from_numpy(frame).to('cuda', non_blocking=True)
            
            # Depth Anything V2 GPU-optimized inference (no CPU transfers)
            with torch.no_grad():
                depth_tensor = self.depth_model.infer_image_gpu(frame_tensor, 384)  # 384 for speed
            
            infer_time = (time.perf_counter() - depth_start) * 1000
            print(f"[WORKER] Frame {frame_id}: Depth inference took {infer_time:.1f}ms", flush=True)
            
            # Transfer to CPU only once for final result
            depth_raw = depth_tensor.cpu().numpy()
            
            # Normalize to 0-255 for depth_map
            depth_min = depth_raw.min()
            depth_max = depth_raw.max()
            if depth_max > depth_min:
                depth_map = ((depth_raw - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            else:
                depth_map = np.zeros_like(depth_raw, dtype=np.uint8)
            
            depth_ms = (time.perf_counter() - depth_start) * 1000
            print(f"[WORKER] Frame {frame_id}: Total depth processing {depth_ms:.1f}ms", flush=True)

        with torch.cuda.stream(self.yolo_stream):
            yolo_start = time.perf_counter()
            detections = self.yolo_processor.process_frame(frame, depth_map, depth_raw)
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
        
        print("[WORKER] run_loop started", flush=True)
        log.info("[CentralWorker] Starting run loop")
        try:
            print("[WORKER] Setting CUDA device...", flush=True)
            log.info("[CentralWorker] Step 1: Setting CUDA device...")
            _set_cuda_device()
            print("[WORKER] CUDA device set, loading models...", flush=True)
            log.info("[CentralWorker] Step 2: CUDA device set, loading models...")
            self._load_models()
            print("[WORKER] Models loaded, entering main loop...", flush=True)
            log.info("[CentralWorker] Step 3: Models loaded, entering main loop...")

            while not self.stop_event.is_set():
                try:
                    msg = self.central_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                log.info(f"[CentralWorker] Processing frame {msg.get('frame_id', -1)}...")
                result = self._process_frame(msg)
                self.result_queue.put(result)
                log.info(f"[CentralWorker] Frame {msg.get('frame_id', -1)} done, latency={result.latency_ms:.2f}ms")
        except Exception as err:  # noqa: BLE001
            log.critical("[CentralWorker] Fatal error", exc_info=err)
            import traceback
            traceback.print_exc()
            self.stop_event.set()
        finally:
            torch.cuda.empty_cache()
            log.info("[CentralWorker] Shutdown complete")


def central_gpu_worker(central_queue, result_queue, stop_event) -> None:
    print("[WORKER ENTRY] central_gpu_worker started", flush=True)
    CentralWorker(central_queue, result_queue, stop_event).run_loop()
    print("[WORKER EXIT] central_gpu_worker exiting", flush=True)
