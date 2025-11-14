#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MockObserver: Simula el comportamiento del Observer de Aria sin hardware real

Permite desarrollo y testing sin las gafas Aria mediante:
1. Generación de frames sintéticos
2. Replay de videos grabados
3. Imágenes estáticas con variaciones simuladas

Compatible 100% con la API del Observer real para drop-in replacement.

Author: Roberto Rojas Sahuquillo
Date: Noviembre 2025
Version: 1.0 - Mock para desarrollo sin hardware
"""

import numpy as np
import cv2
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from pathlib import Path
import logging

log = logging.getLogger("MockObserver")


class MockObserver:
    """
    Mock del Observer de Aria para desarrollo sin hardware.
    
    Modos de operación:
    - 'synthetic': Genera frames sintéticos con objetos simulados
    - 'video': Reproduce un video en loop
    - 'static': Imagen estática con pequeñas variaciones
    
    Uso:
        # Modo sintético (default)
        observer = MockObserver(mode='synthetic', fps=60)
        
        # Modo video
        observer = MockObserver(mode='video', video_path='data/session.mp4')
        
        # Modo estático
        observer = MockObserver(mode='static', image_path='data/frame.jpg')
    """
    
    def __init__(
        self,
        mode: str = 'synthetic',
        fps: int = 60,
        resolution: Tuple[int, int] = (1408, 1408),
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
        rgb_calib: Optional[Any] = None,
        buffer_size: int = 30,
    ):
        """
        Inicializa el MockObserver.
        
        Args:
            mode: 'synthetic', 'video', o 'static'
            fps: Frames por segundo a simular
            resolution: (width, height) de los frames
            video_path: Ruta al video para modo 'video'
            image_path: Ruta a imagen para modo 'static'
            rgb_calib: Calibración RGB (opcional, para compatibilidad)
            buffer_size: Tamaño del buffer de frames
        """
        self.mode = mode
        self.fps = fps
        self.resolution = resolution
        self.video_path = video_path
        self.image_path = image_path
        self.rgb_calib = rgb_calib
        self.buffer_size = buffer_size
        
        # Buffer circular para frames (como Observer real)
        self.frame_buffer = deque(maxlen=buffer_size)
        self.frame_lock = threading.Lock()
        
        # Estado
        self.running = False
        self.frame_count = 0
        self.start_time = None
        self._generator_thread = None
        
        # Inicializar según modo
        self._init_mode()
        
        print(f"[MockObserver] Initialized in '{mode}' mode @ {fps} FPS, resolution {resolution}")
    
    def _init_mode(self):
        """Inicializa recursos según el modo seleccionado."""
        if self.mode == 'video':
            if not self.video_path or not Path(self.video_path).exists():
                raise ValueError(f"Video file not found: {self.video_path}")
            self.video_capture = cv2.VideoCapture(self.video_path)
            if not self.video_capture.isOpened():
                raise ValueError(f"Cannot open video: {self.video_path}")
            print(f"[MockObserver] Loaded video: {self.video_path}")
            
        elif self.mode == 'static':
            if not self.image_path or not Path(self.image_path).exists():
                raise ValueError(f"Image file not found: {self.image_path}")
            self.static_image = cv2.imread(self.image_path)
            if self.static_image is None:
                raise ValueError(f"Cannot read image: {self.image_path}")
            # Resize to target resolution
            self.static_image = cv2.resize(self.static_image, self.resolution)
            print(f"[MockObserver] Loaded image: {self.image_path}")
            
        elif self.mode == 'synthetic':
            # Nada que inicializar para sintético
            print(f"[MockObserver] Synthetic frame generation ready")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def start(self):
        """Inicia la generación de frames (compatible con Observer real)."""
        if self.running:
            print("[MockObserver] Already running")
            return
        
        self.running = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Iniciar thread de generación de frames
        self._generator_thread = threading.Thread(target=self._generate_frames, daemon=True)
        self._generator_thread.start()
        
        print(f"[MockObserver] Started frame generation")
    
    def stop(self):
        """Detiene la generación de frames."""
        self.running = False
        if self._generator_thread:
            self._generator_thread.join(timeout=2.0)
        
        if self.mode == 'video' and hasattr(self, 'video_capture'):
            self.video_capture.release()
        
        print(f"[MockObserver] Stopped (generated {self.frame_count} frames)")
    
    def _generate_frames(self):
        """Thread loop que genera frames según el modo."""
        frame_interval = 1.0 / self.fps
        
        while self.running:
            loop_start = time.time()
            
            # Generar frame según modo
            if self.mode == 'synthetic':
                frame = self._generate_synthetic_frame()
            elif self.mode == 'video':
                frame = self._get_video_frame()
            elif self.mode == 'static':
                frame = self._get_static_frame()
            else:
                frame = None
            
            if frame is not None:
                # Agregar al buffer (thread-safe)
                with self.frame_lock:
                    self.frame_buffer.append({
                        'frame': frame,
                        'timestamp': time.time(),
                        'frame_id': self.frame_count
                    })
                
                self.frame_count += 1
            
            # Sleep para mantener FPS target
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Genera un frame sintético con objetos simulados.
        Simula una escena indoor con personas, sillas, mesas, etc.
        """
        # Base: fondo gris con ruido
        frame = np.random.randint(100, 150, (*self.resolution[::-1], 3), dtype=np.uint8)
        
        # Agregar algunos objetos simulados
        num_objects = np.random.randint(2, 6)
        
        for _ in range(num_objects):
            # Objeto aleatorio (simulando personas, sillas, etc)
            obj_type = np.random.choice(['person', 'chair', 'table', 'bottle'])
            
            # Posición y tamaño aleatorios
            x = np.random.randint(100, self.resolution[0] - 200)
            y = np.random.randint(100, self.resolution[1] - 200)
            w = np.random.randint(80, 300)
            h = np.random.randint(100, 400)
            
            # Color según tipo
            if obj_type == 'person':
                color = (180, 150, 120)  # Tono piel
            elif obj_type == 'chair':
                color = (139, 69, 19)    # Marrón
            elif obj_type == 'table':
                color = (160, 82, 45)    # Marrón claro
            else:
                color = (0, 150, 200)    # Azul (botella)
            
            # Dibujar rectángulo con gradiente simple
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            
            # Borde más oscuro
            darker = tuple(int(c * 0.7) for c in color)
            cv2.rectangle(frame, (x, y), (x + w, y + h), darker, 3)
        
        # Agregar timestamp como texto (útil para debugging)
        timestamp = f"Frame: {self.frame_count} | {time.time():.2f}s"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2)
        
        # Agregar indicador de modo
        cv2.putText(frame, "MOCK MODE: SYNTHETIC", (10, self.resolution[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def _get_video_frame(self) -> Optional[np.ndarray]:
        """Lee el siguiente frame del video (loop infinito)."""
        ret, frame = self.video_capture.read()
        
        if not ret:
            # Reiniciar video al final
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.video_capture.read()
            print("[MockObserver] Video loop restarted")
        
        if ret and frame is not None:
            # Resize al tamaño esperado
            frame = cv2.resize(frame, self.resolution)
            
            # Agregar indicador
            cv2.putText(frame, f"MOCK MODE: VIDEO | Frame {self.frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return frame
        
        return None
    
    def _get_static_frame(self) -> np.ndarray:
        """
        Retorna la imagen estática con pequeñas variaciones.
        Útil para testing de estabilidad y consistency.
        """
        if self.static_image is None:
            # Fallback a frame negro si no hay imagen
            frame = np.zeros((*self.resolution[::-1], 3), dtype=np.uint8)
        else:
            frame = self.static_image.copy()
        
        # Agregar pequeño ruido aleatorio (simula vibración natural)
        noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Pequeño desplazamiento aleatorio (simula movimiento mínimo)
        shift_x = int(np.random.randint(-2, 3))
        shift_y = int(np.random.randint(-2, 3))
        M = np.array([[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]], dtype=np.float32)
        frame = cv2.warpAffine(frame, M, self.resolution, flags=cv2.INTER_LINEAR)
        
        # Indicador
        cv2.putText(frame, f"MOCK MODE: STATIC | Frame {self.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def get_latest_frame(self, camera: str = 'rgb') -> Optional[np.ndarray]:
        """
        Obtiene el frame más reciente del buffer.
        Compatible con API del Observer real.
        
        Args:
            camera: 'rgb', 'slam1', o 'slam2' (solo 'rgb' implementado en mock)
        """
        # Mock solo soporta RGB, ignora slam1/slam2
        if camera not in ['rgb', 'slam1', 'slam2']:
            return None
        
        with self.frame_lock:
            if not self.frame_buffer:
                return None
            return self.frame_buffer[-1]['frame'].copy()
    
    def get_frame_data(self) -> Optional[Dict[str, Any]]:
        """
        Obtiene datos completos del frame más reciente.
        Compatible con API del Observer real.
        """
        with self.frame_lock:
            if not self.frame_buffer:
                return None
            return self.frame_buffer[-1].copy()
    
    def get_buffer_size(self) -> int:
        """Retorna el número de frames en el buffer."""
        with self.frame_lock:
            return len(self.frame_buffer)
    
    def get_motion_state(self) -> Dict[str, Any]:
        """
        Retorna estado de movimiento simulado.
        Compatible con API del Observer real.
        """
        # Mock simula estado estacionario
        return {
            'state': 'stationary',
            'magnitude': 9.8,  # Gravedad estándar
            'timestamp': time.time(),
            'history_length': 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de operación."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        actual_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'mode': self.mode,
            'frames_generated': self.frame_count,
            'elapsed_time': elapsed,
            'target_fps': self.fps,
            'actual_fps': actual_fps,
            'buffer_size': self.get_buffer_size(),
            'running': self.running
        }
    
    def __enter__(self):
        """Context manager support."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.stop()
        return False
