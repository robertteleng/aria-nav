#!/usr/bin/env python3
"""
Export YOLO model to TensorRT engine format.

Usage:
    python tools/export_yolo_tensorrt.py
    
This script converts checkpoints/yolo12n.pt → checkpoints/yolo12n.engine
with FP16 precision for RTX 2060 inference.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from ultralytics import YOLO

def export_yolo_to_tensorrt(
    model_path: str = "checkpoints/yolo12n.pt",
    output_name: str = "yolo12n.engine",
    imgsz: int = 640,
    half: bool = True,
    device: str = "0",
) -> Path:
    """
    Export YOLO PyTorch model to TensorRT engine.
    
    Args:
        model_path: Path to .pt model file
        output_name: Name for output .engine file (placed in checkpoints/)
        imgsz: Input image size for inference
        half: Use FP16 precision (requires GPU with compute capability >= 7.0)
        device: CUDA device index
        
    Returns:
        Path to generated .engine file
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path_obj}")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available - TensorRT export requires NVIDIA GPU")
    
    print(f"🔧 Loading YOLO model from {model_path_obj}")
    model = YOLO(str(model_path))
    
    # Determine output path
    output_dir = Path("checkpoints")
    output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 Exporting to TensorRT engine...")
    print(f"   • Image size: {imgsz}x{imgsz}")
    print(f"   • Precision: {'FP16' if half else 'FP32'}")
    print(f"   • Device: cuda:{device}")
    
    # Export to TensorRT
    # ultralytics will create the .engine file automatically
    export_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        device=device,
        dynamic=False,  # Fixed input size for maximum performance
        simplify=True,
        workspace=4,  # 4GB workspace for optimization
    )
    
    # Move to checkpoints/ if needed
    export_path = Path(export_path)
    final_path = output_dir / output_name
    
    if export_path != final_path:
        if final_path.exists():
            final_path.unlink()
        export_path.rename(final_path)
        print(f"✅ Moved to {final_path}")
    
    print(f"✅ TensorRT export complete: {final_path}")
    print(f"   Size: {final_path.stat().st_size / (1024**2):.1f} MB")
    
    return final_path


if __name__ == "__main__":
    try:
        engine_path = export_yolo_to_tensorrt(
            model_path="checkpoints/yolo12n.pt",
            output_name="yolo12n.engine",
            imgsz=640,
            half=True,
        )
        print(f"\n🎯 Ready to use: {engine_path}")
        print("   Update vision_worker.py to load this .engine file")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        sys.exit(1)
