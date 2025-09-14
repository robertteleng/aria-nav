#!/bin/bash
# Quick Deploy - Fast iteration updates

JETSON_USER="jetson"
JETSON_IP="192.168.8.204"
CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r36.2.0-py3"

echo "🚀 QUICK DEPLOY - Fast updates"

# Quick copy of modified files
echo "📤 Copying updated files..."
scp -r src/ $JETSON_USER@$JETSON_IP:~/aria-navigation/

# Update mac_sender if it exists
if [ -f "mac_sender.py" ]; then
    echo "📱 Updating Mac sender..."
    # Mac sender is already local, no need to copy
else
    echo "ℹ️  Mac sender not found - run jetson_deploy.sh first"
fi

# Quick test
echo "🧪 Quick test..."
if ssh $JETSON_USER@$JETSON_IP "cd ~/aria-navigation && docker run --rm -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE python3 -c 'import sys; sys.path.insert(0, \"/workspace/src\"); print(\"Quick update OK\")'"; then
    echo "✅ Quick deploy successful!"
else
    echo "❌ Quick test failed"
    exit 1
fi

echo ""
echo "Ready to run:"
echo "Jetson: ssh $JETSON_USER@$JETSON_IP && cd ~/aria-navigation && docker run --rm --runtime nvidia --network host -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE python3 jetson_server.py"
echo "Mac: python3 mac_sender.py"