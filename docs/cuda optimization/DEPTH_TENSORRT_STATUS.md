# Depth-Anything-V2 TensorRT Integration Status

**Date:** 2025-11-17  
**Branch:** `feature/fase4-tensorrt`

---

## ✅ Completed

### 1. TensorRT Engine Export
- **Script:** `tools/export_depth_tensorrt.py`
- **Engine:** `checkpoints/depth_anything_v2_vits.engine` (50.9 MB, FP16)
- **Export time:** ~110 seconds
- **Status:** ✅ Successfully generated

### 2. Export Pipeline
```
HuggingFace model → ONNX → TensorRT engine
depth-anything/Depth-Anything-V2-Small-hf → .onnx → .engine
```

**ONNX Export:**
- Input: `pixel_values` (1, 3, 384, 384)
- Output: `predicted_depth`
- Dynamic axes configured
- Warnings about TracerWarning (expected for dynamic ops)

**TensorRT Conversion:**
- FP16 precision enabled
- Optimization profile: Fixed shape (1, 3, 384, 384)
- Workspace: 4GB
- Build time: 109.6 seconds
- Final size: 50.9 MB (vs 95MB .pth original)

---

## ⏳ Pending - Runtime Integration

### Challenge: Transformers vs TensorRT

**Current Implementation** (`depth_estimator.py`):
```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

self.processor = AutoImageProcessor.from_pretrained(name)
self.model = AutoModelForDepthEstimation.from_pretrained(name)
self.model.to(self.device)
self.model.eval()
```

**TensorRT Integration Requires:**
1. **Low-level inference:** Replace Transformers API with TensorRT runtime
2. **Memory management:** Manual CUDA memory allocation (pycuda or torch.cuda)
3. **Preprocessing:** Replicate `AutoImageProcessor` logic manually
4. **Postprocessing:** Match Transformers output format

### Integration Options

#### Option A: PyCUDA (Pure TensorRT)
**Pros:**
- Maximum performance
- Direct TensorRT control

**Cons:**
- Requires `pycuda` (compilation issues on this system)
- Complex memory management
- Need to replicate preprocessing logic

**Code sketch:**
```python
import tensorrt as trt
import pycuda.driver as cuda

# Load engine
with open('depth_anything_v2_vits.engine', 'rb') as f:
    engine = trt.Runtime(trt.Logger()).deserialize_cuda_engine(f.read())

# Allocate buffers, run inference...
```

#### Option B: PyTorch + TensorRT (Hybrid)
**Pros:**
- Keep PyTorch preprocessing
- Use `torch.cuda` for memory (already available)
- Easier integration

**Cons:**
- Slightly slower than pure TensorRT
- Still requires TensorRT runtime code

**Code sketch:**
```python
import torch
import tensorrt as trt

# Load engine
runtime = trt.Runtime(trt.Logger())
engine = runtime.deserialize_cuda_engine(engine_data)
context = engine.create_execution_context()

# Use torch tensors as buffers
input_tensor = torch.from_numpy(input_data).cuda()
output_tensor = torch.empty(output_shape).cuda()

# Bind and execute
context.set_tensor_address('pixel_values', input_tensor.data_ptr())
context.set_tensor_address('predicted_depth', output_tensor.data_ptr())
context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
```

#### Option C: ONNX Runtime (Simpler Alternative)
**Pros:**
- Simpler API than raw TensorRT
- Still gets TensorRT acceleration via TensorRT Execution Provider
- No pycuda needed

**Cons:**
- Not pure TensorRT (adds abstraction layer)
- Slightly less performant

**Code sketch:**
```python
import onnxruntime as ort

session = ort.InferenceSession(
    'depth_anything_v2_vits.onnx',
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider']
)
outputs = session.run(None, {'pixel_values': input_data})
```

---

## 📊 Expected Performance

**Current (PyTorch):**
- Inference time: ~60ms per frame
- FPS impact: 9.27 FPS with depth (vs 22.79 without)

**Expected (TensorRT):**
- Inference time: ~15-20ms per frame (3-4x speedup)
- FPS target: ~18-20 FPS with depth enabled
- Combined with TensorRT YOLO: ~20 FPS full pipeline

---

## 🚀 Recommended Next Steps

### Immediate (Quickest path to working system):

**1. Use ONNX Runtime** (1-2 hours)
```bash
pip install onnxruntime-gpu
```

Update `depth_estimator.py`:
- Keep `.onnx` file (don't delete in export script)
- Load with ONNX Runtime + TensorRT EP
- Test performance vs PyTorch

**2. If performance insufficient, implement Option B** (4-6 hours)
- Pure TensorRT with torch.cuda memory
- More complex but maximum performance

### Alternative (Skip TensorRT for depth):

**Optimize PyTorch Depth** instead:
- Increase `DEPTH_FRAME_SKIP` (process every 2-3 frames)
- Reduce `DEPTH_INPUT_SIZE` (384 → 256)
- Trade-off: accuracy vs speed

Expected FPS with optimizations: ~15-16 FPS (vs 9.27 current)

---

## 📁 Files Created

```
tools/export_depth_tensorrt.py          ✅ Export script
tools/test_depth_engine.py              ⏳ Test script (needs pycuda)
checkpoints/depth_anything_v2_vits.engine  ✅ TensorRT engine (50.9 MB)
checkpoints/depth_anything_v2_vits.onnx    🗑️ Deleted (can regenerate)
```

---

## 🎯 Decision Point

**Current state:**
- YOLO TensorRT: ✅ Working (22.79 FPS)
- Depth TensorRT: ⏳ Engine ready, integration pending

**Options:**
1. **Complete TensorRT depth integration** (4-8 hours total work)
   - Expected: 18-20 FPS full pipeline
   
2. **Optimize PyTorch depth** (1-2 hours)
   - Expected: 15-16 FPS full pipeline
   - Simpler, less risk
   
3. **Skip depth optimization for now**
   - Current: 9.27 FPS with depth
   - Focus on other features

**Recommendation:** Option 2 (optimize PyTorch) is the pragmatic choice given compilation issues and time investment required for full TensorRT integration.

---

**Status:** 🟡 Depth TensorRT engine ready but not integrated  
**Next:** Decision on integration approach
