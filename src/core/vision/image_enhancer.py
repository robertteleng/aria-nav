"""
Image enhancement module for low-light conditions
"""

import cv2
import numpy as np
from utils.config import Config

class ImageEnhancer:
    """Handle image enhancement for low-light conditions"""
    
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        print("[INFO] ✓ ImageEnhancer initialized")

        self.auto_enhancement = Config.AUTO_ENHANCEMENT
        self.low_light_threshold = Config.LOW_LIGHT_THRESHOLD
        self.frame_count = 0
    
        print("[INFO] ✓ ImageEnhancer initialized")
        print(f"[INFO]   - Auto detection: {self.auto_enhancement}")
        print(f"[INFO]   - Threshold: {self.low_light_threshold}")

    def _detect_low_light(self, frame):
        """Detecta automáticamente si hay poca luz"""
        if not self.auto_enhancement:
            return True  # Siempre aplicar si auto está off
        
        # Calcular brillo promedio
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Debug cada 30 frames
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            is_low = avg_brightness < self.low_light_threshold
            print(f"[AUTO-LIGHT] Brightness: {avg_brightness:.1f}, "
                f"Low-light: {is_low}, Threshold: {self.low_light_threshold}")
        
        return avg_brightness < self.low_light_threshold
    
    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance frame for better visibility in low-light"""
        
        # 1. Verificar si el sistema está habilitado
        if not Config.LOW_LIGHT_ENHANCEMENT:
            return frame
        
        # 2. Si auto detection está habilitado, verificar luz
        if Config.AUTO_ENHANCEMENT and not self._detect_low_light(frame):
            return frame  # No enhancement needed - buena iluminación
        
        # 3. Aplicar enhancement
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l_enhanced = self.clahe.apply(l)
            
            # Merge channels back
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Optional: Slight gamma correction
            if Config.GAMMA_CORRECTION > 0:
                gamma = Config.GAMMA_CORRECTION
                inv_gamma = 1.0 / gamma
                table = np.array([(i / 255.0) ** inv_gamma * 255 
                                for i in np.arange(256)]).astype("uint8")
                enhanced_frame = cv2.LUT(enhanced_frame, table)
            
            return enhanced_frame
            
        except Exception as e:
            print(f"[WARN] Image enhancement failed: {e}")
            return frame
        
    def set_always_on(self):
        """Enhancement siempre activo"""
        self.auto_enhancement = False
        print("[INFO] Enhancement: Always ON")
    
    def set_auto(self):
        """Enhancement solo en low light"""
        self.auto_enhancement = True
        print("[INFO] Enhancement: Auto detection")