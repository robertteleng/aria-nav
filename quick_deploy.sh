#!/bin/bash
# =================================================================
# QUICK DEPLOY - Test rápido en Jetson Container
# =================================================================

echo "🚀 QUICK DEPLOY - Jetson Container Test"
echo "📱 Ejecutando desde Mac hacia Jetson"

# Configuration
JETSON_USER="jetson"
JETSON_IP="192.168.8.204"
PROJECT_DIR="$HOME/aria-navigation"
CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r36.2.0-py3"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}📁 Project: $PROJECT_DIR${NC}"
echo -e "${YELLOW}🎯 Target: $JETSON_USER@$JETSON_IP${NC}"
echo ""

# Check if we're in the right directory and file exists
if [ ! -f "src/communication/jetson_components_migration.py" ]; then
    echo -e "${RED}❌ jetson_components_migration.py not found in src/communication/${NC}"
    echo "💡 Make sure you're in the project root directory (aria-navigation-tfm)"
    echo "💡 And that src/communication/jetson_components_migration.py exists"
    exit 1
fi

# Copy files to Jetson
echo "📤 Copying files to Jetson..."
if scp jetson_server.py $JETSON_USER@$JETSON_IP:~/aria-navigation/; then
    echo -e "${GREEN}✅ Files copied successfully${NC}"
else
    echo -e "${RED}❌ Failed to copy files${NC}"
    exit 1
fi

# Test on Jetson
echo "🧪 Running test on Jetson container..."
echo "Command: ssh $JETSON_USER@$JETSON_IP 'cd ~/aria-navigation && docker run --rm --runtime nvidia -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE python3 jetson_server.py test'"

if ssh $JETSON_USER@$JETSON_IP "cd ~/aria-navigation && docker run --rm --runtime nvidia -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE python3 jetson_server.py test"; then
    echo ""
    echo -e "${GREEN}✅ QUICK DEPLOY SUCCESSFUL!${NC}"
    echo "🎯 Jetson container is ready for processing"
    echo ""
    echo "Next steps:"
    echo "1. Start Jetson server: ssh $JETSON_USER@$JETSON_IP 'cd ~/aria-navigation && docker run --rm --runtime nvidia -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE python3 jetson_server.py run'"
    echo "2. Run Mac sender to send frames"
else
    echo ""
    echo -e "${RED}❌ QUICK DEPLOY FAILED${NC}"
    echo "🔍 Check the error messages above"
fi