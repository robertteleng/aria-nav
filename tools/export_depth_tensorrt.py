#!/usr/bin/env python3
"""
Export Depth-Anything-V2 model to TensorRT engine format.

Usage:
    python tools/export_depth_tensorrt.py
    
This script converts depth_anything_v2_vits.pth → depth_anything_v2_vits.engine
with FP16 precision for RTX 2060 inference.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def export_depth_to_onnx(
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    output_path: str = "checkpoints/depth_anything_v2_vits.onnx",
    input_size: int = 384,
) -> Path:
    """
    Export Depth-Anything-V2 to ONNX format (intermediate step).
    
    Args:
        model_name: HuggingFace model identifier
        output_path: Path for output ONNX file
        input_size: Input image size
        
    Returns:
        Path to generated ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"🔧 Loading Depth-Anything-V2 model: {model_name}")
    
    # Load model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.eval()
    model.cuda()
    
    print(f"📊 Model loaded successfully")
    print(f"   • Input size: {input_size}x{input_size}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size, device='cuda')
    
    print(f"🚀 Exporting to ONNX...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['pixel_values'],
        output_names=['predicted_depth'],
        dynamic_axes={
            'pixel_values': {0: 'batch'},
            'predicted_depth': {0: 'batch'}
        }
    )
    
    print(f"✅ ONNX export complete: {output_path}")
    print(f"   Size: {output_path.stat().st_size / (1024**2):.1f} MB")
    
    return output_path


def onnx_to_tensorrt(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 4,
) -> Path:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Path to ONNX file
        engine_path: Path for output engine file
        fp16: Use FP16 precision
        workspace_gb: Workspace size in GB
        
    Returns:
        Path to generated engine file
    """
    try:
        import tensorrt as trt
    except ImportError:
        raise ImportError(
            "TensorRT not found. Install with: "
            "pip install tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs"
        )
    
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    
    print(f"🔧 Converting ONNX → TensorRT engine...")
    print(f"   • Input: {onnx_path}")
    print(f"   • Output: {engine_path}")
    print(f"   • Precision: {'FP16' if fp16 else 'FP32'}")
    print(f"   • Workspace: {workspace_gb}GB")
    
    # Create builder and network
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"📖 Parsing ONNX model...")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                print(f"❌ Parser error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX file")
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    
    if fp16 and builder.platform_has_fast_fp16:
        print("✅ Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        if fp16:
            print("⚠️  FP16 requested but not supported, using FP32")
    
    # Create optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    
    # Set min/opt/max shapes (batch, channels, height, width)
    # For fixed batch size of 1
    min_shape = (1, 3, 384, 384)
    opt_shape = (1, 3, 384, 384)
    max_shape = (1, 3, 384, 384)
    
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    print(f"✅ Optimization profile configured for input: {input_name}")
    
    # Build engine
    print(f"🏗️  Building TensorRT engine (this may take several minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")
    
    # Save engine
    engine_path.parent.mkdir(exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✅ TensorRT engine saved: {engine_path}")
    print(f"   Size: {engine_path.stat().st_size / (1024**2):.1f} MB")
    
    return engine_path


def export_depth_to_tensorrt(
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    output_name: str = "depth_anything_v2_vits.engine",
    input_size: int = 384,
    fp16: bool = True,
    keep_onnx: bool = False,
) -> Path:
    """
    Complete pipeline: HuggingFace model → ONNX → TensorRT engine.
    
    Args:
        model_name: HuggingFace model identifier
        output_name: Name for output .engine file
        input_size: Input image size
        fp16: Use FP16 precision
        keep_onnx: Keep intermediate ONNX file
        
    Returns:
        Path to generated engine file
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - TensorRT export requires NVIDIA GPU")
    
    # Step 1: Export to ONNX
    onnx_path = Path("checkpoints") / output_name.replace('.engine', '.onnx')
    
    if not onnx_path.exists():
        print("\n" + "="*60)
        print("STEP 1/2: Exporting to ONNX")
        print("="*60)
        export_depth_to_onnx(model_name, str(onnx_path), input_size)
    else:
        print(f"✓ ONNX file already exists: {onnx_path}")
    
    # Step 2: Convert ONNX to TensorRT
    print("\n" + "="*60)
    print("STEP 2/2: Converting to TensorRT")
    print("="*60)
    
    engine_path = Path("checkpoints") / output_name
    onnx_to_tensorrt(str(onnx_path), str(engine_path), fp16=fp16)
    
    # Cleanup ONNX if requested
    if not keep_onnx and onnx_path.exists():
        print(f"🧹 Cleaning up intermediate ONNX file...")
        onnx_path.unlink()
    
    return engine_path


if __name__ == "__main__":
    try:
        print("="*60)
        print("Depth-Anything-V2 TensorRT Export")
        print("="*60)
        
        engine_path = export_depth_to_tensorrt(
            model_name="depth-anything/Depth-Anything-V2-Small-hf",
            output_name="depth_anything_v2_vits.engine",
            input_size=384,
            fp16=True,
            keep_onnx=True,  # KEEP ONNX for ONNX Runtime!
        )
        
        print("\n" + "="*60)
        print("🎯 Export Complete!")
        print("="*60)
        print(f"✅ TensorRT engine ready: {engine_path}")
        print(f"\nNext steps:")
        print(f"  1. Update depth_estimator.py to load .engine files")
        print(f"  2. Test performance with: python run.py")
        print(f"  3. Expected improvement: 60ms → 15-20ms per frame")
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
