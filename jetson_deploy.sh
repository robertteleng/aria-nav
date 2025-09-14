#!/bin/bash
# Jetson Deploy - Complete initial setup

JETSON_USER="jetson"
JETSON_IP="192.168.8.204"
CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r36.2.0-py3"

echo "🚀 JETSON DEPLOY - Complete initial setup"

# Pre-checks
echo "🔍 Checking SSH connection..."
if ! ssh -o ConnectTimeout=5 $JETSON_USER@$JETSON_IP "echo 'OK'" > /dev/null 2>&1; then
    echo "❌ SSH connection failed"
    exit 1
fi

echo "🐳 Checking Docker + NVIDIA runtime..."
if ! ssh $JETSON_USER@$JETSON_IP "docker run --rm --runtime nvidia hello-world" > /dev/null 2>&1; then
    echo "❌ NVIDIA runtime failed"  
    exit 1
fi

# Setup directory
echo "📂 Setting up Jetson directory..."
ssh $JETSON_USER@$JETSON_IP "mkdir -p ~/aria-navigation"

# Copy complete architecture
echo "📤 Copying complete architecture..."
scp -r src/ $JETSON_USER@$JETSON_IP:~/aria-navigation/

echo "📄 Copying root files..."
scp main.py $JETSON_USER@$JETSON_IP:~/aria-navigation/
if [ -f "environment.yml" ]; then
    scp environment.yml $JETSON_USER@$JETSON_IP:~/aria-navigation/
fi

# Create Jetson server using your architecture
cat > temp_jetson_server.py << 'EOF'
#!/usr/bin/env python3
import sys
import imagezmq
import cv2
from pathlib import Path

sys.path.insert(0, '/workspace/src')

from core.navigation.builder import build_navigation_system
from utils.ctrl_handler import CtrlCHandler

class JetsonProcessor:
    def __init__(self):
        self.coordinator = build_navigation_system(enable_dashboard=False)
        self.image_hub = imagezmq.ImageHub(open_port="tcp://0.0.0.0:5555")
        print("Jetson processor ready - using your complete architecture")
    
    def run(self):
        ctrl_handler = CtrlCHandler()
        while not ctrl_handler.should_stop:
            try:
                rpi_name, frame = self.image_hub.recv_image()
                annotated_frame = self.coordinator.process_frame(frame)
                self.image_hub.send_reply(b'OK')
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    processor = JetsonProcessor()
    processor.run()
EOF

scp temp_jetson_server.py $JETSON_USER@$JETSON_IP:~/aria-navigation/jetson_server.py
rm temp_jetson_server.py

# Create Mac sender
cat > mac_sender.py << 'EOF'
#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import cv2
import imagezmq
from core.hardware.device_manager import DeviceManager
from utils.ctrl_handler import CtrlCHandler

class MacSender:
    def __init__(self):
        self.sender = imagezmq.ImageSender(connect_to="tcp://192.168.8.204:5555")
        self.device_manager = DeviceManager()
    
    def on_image_received(self, image, record):
        if record.camera_id.name == "RGB":
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            try:
                reply = self.sender.send_image("mac", rotated)
            except Exception as e:
                print(f"Send error: {e}")
    
    def start(self):
        self.device_manager.connect()
        self.device_manager.start_streaming()
        self.device_manager.register_observer(self)
        self.device_manager.subscribe()
        
        ctrl_handler = CtrlCHandler()
        while not ctrl_handler.should_stop:
            pass
        
        self.device_manager.cleanup()

if __name__ == "__main__":
    sender = MacSender()
    sender.start()
EOF

# Pull container image
echo "🐳 Pulling container image..."
ssh $JETSON_USER@$JETSON_IP "docker pull $CONTAINER_IMAGE" > /dev/null 2>&1

# Install dependencies in container
echo "📦 Installing dependencies..."
ssh $JETSON_USER@$JETSON_IP "cd ~/aria-navigation && docker run --rm -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE bash -c 'pip install ultralytics imagezmq pyttsx3'"

# Test imports
echo "🧪 Testing architecture imports..."
if ssh $JETSON_USER@$JETSON_IP "cd ~/aria-navigation && docker run --rm -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE python3 -c 'import sys; sys.path.insert(0, \"/workspace/src\"); from core.navigation.builder import build_navigation_system; print(\"✅ Architecture OK\")'"; then
    echo "✅ Setup completed successfully!"
else
    echo "❌ Architecture test failed"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Jetson: ssh $JETSON_USER@$JETSON_IP"
echo "   cd ~/aria-navigation"
echo "   docker run --rm --runtime nvidia --network host -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE python3 jetson_server.py"
echo ""
echo "2. Mac: python3 mac_sender.py"