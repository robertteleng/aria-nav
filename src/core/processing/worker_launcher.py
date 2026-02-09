"""
Worker launcher that restores CUDA visibility before importing worker modules.

This is needed because:
1. Main process hides CUDA (CUDA_VISIBLE_DEVICES="") for FastDDS compatibility
2. Workers need CUDA for GPU inference
3. With spawn, we need to restore CUDA BEFORE importing torch
"""
import os


def central_gpu_worker_wrapper(central_queue, result_queue, stop_event, ready_event=None):
    """Wrapper that restores CUDA before importing the actual worker."""
    # Restore CUDA FIRST, before any torch imports
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("NUMBA_DISABLE_CUDA", None)

    # NOW import and run the actual worker
    from core.processing.central_worker import central_gpu_worker
    central_gpu_worker(central_queue, result_queue, stop_event, ready_event)


def slam_gpu_worker_wrapper(slam_queue, result_queue, stop_event, ready_event=None):
    """Wrapper that restores CUDA before importing the actual worker."""
    # Restore CUDA FIRST, before any torch imports
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("NUMBA_DISABLE_CUDA", None)

    # NOW import and run the actual worker
    from core.processing.slam_worker import slam_gpu_worker
    slam_gpu_worker(slam_queue, result_queue, stop_event, ready_event)
