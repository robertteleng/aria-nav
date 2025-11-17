#!/usr/bin/env python3
"""
Test Depth-Anything-V2 TensorRT engine inference.
Validates that the engine loads and runs correctly.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import tensorrt as trt
import torch
import time

def load_engine(engine_path: str):
    """Load TensorRT engine from file."""
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    print(f"🔧 Loading TensorRT engine: {engine_path}")
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    
    if engine is None:
        raise RuntimeError("Failed to load engine")
    
    print(f"✅ Engine loaded successfully")
    print(f"   Inputs: {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT]}")
    print(f"   Outputs: {[engine.get_tensor_name(i) for i in range(engine.num_io_tensors) if engine.get_tensor_mode(engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT]}")
    
    return engine

def run_inference(engine, input_data: np.ndarray):
    """Run inference with TensorRT engine."""
    context = engine.create_execution_context()
    
    # Get tensor names
    input_name = None
    output_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            input_name = name
        elif engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            output_name = name
    
    if not input_name or not output_name:
        raise RuntimeError("Could not find input/output tensors")
    
    # Allocate device memory
    import pycuda.driver as cuda
    import pycuda.autoinit
    
    # Input
    d_input = cuda.mem_alloc(input_data.nbytes)
    cuda.memcpy_htod(d_input, input_data)
    
    # Output (need to know shape - get from engine)
    output_shape = engine.get_tensor_shape(output_name)
    output_size = np.prod(output_shape) * np.dtype(np.float32).itemsize
    d_output = cuda.mem_alloc(output_size)
    
    # Set tensor addresses
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    # Run inference
    start = time.perf_counter()
    context.execute_async_v3(cuda.Stream().handle)
    cuda.Context.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    # Copy output back
    output_data = np.empty(output_shape, dtype=np.float32)
    cuda.memcpy_dtoh(output_data, d_output)
    
    return output_data, elapsed

def main():
    engine_path = "checkpoints/depth_anything_v2_vits.engine"
    
    if not Path(engine_path).exists():
        print(f"❌ Engine not found: {engine_path}")
        print("   Run: python tools/export_depth_tensorrt.py")
        return
    
    try:
        # Load engine
        engine = load_engine(engine_path)
        
        # Create dummy input (1, 3, 384, 384)
        print("\n📊 Running test inference...")
        dummy_input = np.random.randn(1, 3, 384, 384).astype(np.float32)
        
        # Warmup
        print("  Warmup...")
        for _ in range(3):
            _ = run_inference(engine, dummy_input)
        
        # Benchmark
        print("  Benchmarking 30 iterations...")
        times = []
        for i in range(30):
            _, elapsed = run_inference(engine, dummy_input)
            times.append(elapsed)
            if i % 10 == 0:
                print(f"    Iteration {i}: {elapsed:.1f}ms")
        
        avg_ms = np.mean(times)
        fps = 1000 / avg_ms
        
        print(f"\n✅ Results:")
        print(f"   Avg inference: {avg_ms:.1f}ms")
        print(f"   Throughput: {fps:.1f} FPS")
        print(f"   Min: {min(times):.1f}ms | Max: {max(times):.1f}ms")
        print(f"\n🎯 Expected improvement: PyTorch ~60ms → TensorRT ~{avg_ms:.1f}ms")
        print(f"   Speedup: {60/avg_ms:.1f}x")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
