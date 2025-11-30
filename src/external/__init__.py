"""
External third-party dependencies for the Aria Navigation System.

This package contains external libraries and models used for TensorRT optimization
workflows in the navigation system.

Modules:
- depth_anything_v2: Depth Anything V2 model architecture for TensorRT export
  - Source: https://github.com/DepthAnything/Depth-Anything-V2
  - License: Apache 2.0
  - Purpose: Required for TensorRT/ONNX export workflow
  - Status: ACTIVE in production when TensorRT .engine files are available

TensorRT Workflow:
   This local copy is essential for the TensorRT optimization pipeline because
   the HuggingFace transformers wrapper doesn't expose the underlying model
   architecture needed for torch.onnx.export().

Production Runtime Flow:
   1. Config sets DEPTH_BACKEND = "depth_anything_v2" and USE_TENSORRT = True
   2. If TensorRT .engine file exists in checkpoints/:
      - Uses ONNX Runtime with TensorRT backend (FAST - production mode)
   3. If .engine file NOT found (e.g., on dev machines without checkpoints/):
      - Falls back to HuggingFace transformers with PyTorch (SLOW - dev mode)
      - See core/vision/depth_estimator.py lines 147-191 for fallback logic

Note on checkpoints/ folder:
   The checkpoints/ folder containing .engine files is in .gitignore and does
   not sync between machines. This is why development machines may not have
   TensorRT models and will use PyTorch fallback.

TensorRT Export Workflow:
   To re-export models to ONNX/TensorRT format, use:
   - tools/export_depth_tensorrt.py (requires this local depth_anything_v2 copy)
   - See docs/cuda optimization/FASE_4_TENSORRT_NOTES.md for detailed workflow
"""
