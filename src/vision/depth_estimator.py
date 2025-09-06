#!/usr/bin/env python3
"""MiDaS depth estimation module"""

import torch
import cv2
import numpy as np
from typing import Tuple, Optional
from utils.config import Config

class DepthEstimator:
    """Handle MiDaS depth estimation for spatial awareness"""
    
    def __init__(self):
        if not Config.DEPTH_ENABLED:
            self.model = None
            return
            
        print(f"[INFO] Loading MiDaS {Config.MIDAS_MODEL}...")
        try:
            self.device = torch.device(Config.MIDAS_DEVICE)
            self.model = torch.hub.load("intel-isl/MiDaS", Config.MIDAS_MODEL)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if Config.MIDAS_MODEL in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
                
            print(f"[INFO] ✓ MiDaS {Config.MIDAS_MODEL} loaded")
        except Exception as e:
            print(f"[ERROR] MiDaS loading failed: {e}")
            self.model = None
    
    def estimate_depth(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth map from RGB frame"""
        if self.model is None:
            return None
            
        try:
            # Convert BGR to RGB if needed
            if len(rgb_frame.shape) == 3:
                rgb_input = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_input = rgb_frame
            
            # Apply MiDaS transforms
            input_tensor = self.transform(rgb_input).to(self.device)
            
            # Inference
            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=rgb_input.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # Convert to numpy and normalize
            depth_map = prediction.cpu().numpy()
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
            return depth_normalized
            
        except Exception as e:
            print(f"[ERROR] Depth estimation failed: {e}")
            return None