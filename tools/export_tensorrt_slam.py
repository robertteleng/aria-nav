#!/usr/bin/env python3
"""
Export YOLO model to TensorRT for SLAM profile (256x256).

SLAM usa resolución 256x256 (diferente de RGB 640x640), por lo que necesita
su propio engine TensorRT.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ultralytics.models.yolo import YOLO
import torch

def export_slam_tensorrt():
    """Export YOLO model to TensorRT with SLAM resolution (256x256)."""
    
    print("\n🔧 Exporting YOLO TensorRT Engine for SLAM")
    print("=" * 60)
    
    # Paths
    model_path = Path("checkpoints/yolo12n.pt")
    output_name = "yolo12n_slam256"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA not available! TensorRT requires CUDA.")
        return False
    
    print(f"✓ Model: {model_path}")
    print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
    print(f"✓ Target resolution: 256x256 (SLAM profile)")
    print(f"✓ Output: checkpoints/{output_name}.engine")
    
    # Load model
    print("\n📦 Loading YOLO model...")
    model = YOLO(str(model_path))
    
    # Export to TensorRT
    print("\n🔥 Exporting to TensorRT (this takes 2-3 minutes)...")
    print("   - Device: GPU (CUDA)")
    print("   - Image size: 256x256")
    print("   - Half precision: FP16")
    print("   - Dynamic batching: Disabled")
    print("   - Optimization: Max performance")
    
    try:
        # Export with SLAM-specific settings
        export_path = model.export(
            format='engine',          # TensorRT
            imgsz=256,               # SLAM resolution
            half=True,               # FP16 precision
            device=0,                # GPU 0
            workspace=4,             # 4GB workspace
            verbose=True,
            simplify=True,
            dynamic=False,           # Fixed batch size = 1
            batch=1,
        )
        
        print(f"\n✅ Export successful!")
        print(f"   Engine: {export_path}")
        
        # Rename to include resolution in filename
        src = Path(export_path)
        dst = src.parent / f"{output_name}.engine"
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        
        print(f"\n✓ Renamed to: {dst}")
        print(f"✓ Size: {dst.stat().st_size / 1024 / 1024:.1f} MB")
        
        # Test inference
        print("\n🧪 Testing inference...")
        import numpy as np
        test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        model_trt = YOLO(str(dst))
        results = model_trt.predict(test_img, imgsz=256, verbose=False)
        
        print(f"✓ Inference test passed!")
        if results and results[0].boxes is not None:
            print(f"✓ Detections: {len(results[0].boxes)}")
        else:
            print(f"✓ No detections in test image (expected)")
        
        print(f"\n{'=' * 60}")
        print("🎉 SLAM TensorRT engine ready!")
        print(f"\nNext steps:")
        print(f"1. Update yolo_processor.py to use {output_name}.engine for SLAM")
        print(f"2. Test with: python run.py test")
        print(f"3. Benchmark performance improvement")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = export_slam_tensorrt()
    sys.exit(0 if success else 1)
