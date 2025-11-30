"""
External third-party dependencies for the Aria Navigation System.

This package contains external libraries and models that are not part of the
core navigation system but were essential for model optimization workflows.

Modules:
- depth_anything_v2: Depth Anything V2 model architecture (LEGACY - not currently used)
  - Source: https://github.com/DepthAnything/Depth-Anything-V2
  - License: Apache 2.0
  - Purpose: Originally used for TensorRT/ONNX export workflow
  - Status: LEGACY - Current implementation uses HuggingFace transformers instead

⚠️ IMPORTANT: The depth_anything_v2 code is currently NOT used at runtime.
   The active depth estimation uses `transformers.AutoModelForDepthEstimation`
   from HuggingFace (see core/vision/depth_estimator.py lines 181-191).

Historical Context:
   This local copy was needed for TensorRT optimization workflow because
   the HuggingFace wrapper doesn't expose the underlying model architecture
   needed for torch.onnx.export(). See docs/cuda optimization/FASE_4_TENSORRT_NOTES.md
   for the original export workflow that required direct access to DepthAnythingV2 class.

Current Runtime Flow:
   1. Config sets DEPTH_BACKEND = "depth_anything_v2"
   2. DepthEstimator loads via HuggingFace transformers (NOT from src/external/)
   3. Optional: If TensorRT ONNX exists, uses ONNX Runtime instead

If you need to re-export to ONNX/TensorRT:
   See tools/export_depth_tensorrt.py for the export script that uses
   this local copy.
"""
