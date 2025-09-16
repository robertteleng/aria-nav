#!/usr/bin/env python3
"""
Mac sender completo con objetos detectables por YOLO
"""

import cv2
import imagezmq
import time
import numpy as np

# IP del Jetson (cambiar si es diferente)
sender = imagezmq.ImageSender("tcp://192.168.0.25:5555", REQ_REP=True)

frame_id = 0
success = 0

try:
    for frame_id in range(100):
        # Frame con fondo negro y objetos detectables
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Dibujar objetos que YOLO puede detectar
        # Rectángulo verde (puede ser detectado como objeto)
        cv2.rectangle(frame, (100, 100), (200, 400), (0, 255, 0), -1)
        
        # Círculo azul (otro objeto)
        cv2.circle(frame, (400, 200), 50, (255, 0, 0), -1)
        
        # Texto del frame
        cv2.putText(frame, f"Frame {frame_id}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        try:
            _, jpg = cv2.imencode('.jpg', frame)
            reply = sender.send_jpg("mac", jpg)
            if reply == b'OK':
                success += 1
                if frame_id % 10 == 0:
                    print(f"✅ Frame {frame_id} sent")
        except Exception as e:
            print(f"❌ Frame {frame_id}: {e}")
        
        time.sleep(1/30)  # 30 FPS
    
    print(f"📊 Success: {success}/100")
    
except KeyboardInterrupt:
    print("Stopped")
finally:
    sender.close()