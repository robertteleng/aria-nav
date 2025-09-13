#!/usr/bin/env python3
import cv2
import numpy as np
import imagezmq
import time

print("[MAC] Conectando a Jetson 192.168.8.204:5555...")
sender = imagezmq.ImageSender(connect_to='tcp://192.168.8.204:5555')

for i in range(10):
    # Frame de prueba
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f"Mac Frame #{i}", (50, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    try:
        reply = sender.send_image("mac-test", frame)
        print(f"[MAC] Frame {i} enviado - Reply: {reply}")
    except Exception as e:
        print(f"[MAC] Error: {e}")
        break
    
    time.sleep(1)

print("[MAC] Test completado")
