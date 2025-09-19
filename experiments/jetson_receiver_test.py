import imagezmq
import time
import cv2
import numpy as np
from ultralytics import YOLO

# Cargar YOLO
print("Loading YOLO...")
model = YOLO('yolov8n.pt')
print("✅ YOLO loaded")

print("🔗 Starting receiver on port 5555...")
image_hub = imagezmq.ImageHub(open_port="tcp://*:5555", REQ_REP=True)
print("✅ Ready for frames from Mac")

received = 0
start = time.time()

try:
    while True:
        name, jpg = image_hub.recv_jpg()
        received += 1
        
        # Decode frame
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        # YOLO detection
        results = model(frame, verbose=False)
        detections = len(results[0].boxes) if results[0].boxes is not None else 0
        
        image_hub.send_reply(b'OK')
        
        if received % 10 == 0:
            fps = received / (time.time() - start)
            print(f"📡 Frame {received}, FPS: {fps:.1f}, Detections: {detections}")
            
except KeyboardInterrupt:
    print(f"\n📊 FINAL: {received} frames")
finally:
    image_hub.close()