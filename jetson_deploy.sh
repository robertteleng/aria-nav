#!/bin/bash
# =================================================================
# JETSON DEPLOYMENT SCRIPT - TFM Navigation System
# Migración híbrida: Componentes Core al Jetson Container
# =================================================================

echo "🚀 JETSON DEPLOYMENT - TFM Navigation System"
echo "🎯 Migración componentes Core desde Mac"
echo "=========================================================="

# =================================================================
# CONFIGURATION
# =================================================================
PROJECT_DIR="$HOME/aria-navigation"
JETSON_USER="jetson"
JETSON_IP="192.168.8.204"
CONTAINER_IMAGE="nvcr.io/nvidia/l4t-ml:r36.2.0-py3"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}📁 Project Directory: $PROJECT_DIR${NC}"
echo -e "${BLUE}🎯 Target Jetson: $JETSON_USER@$JETSON_IP${NC}"
echo -e "${BLUE}🐳 Container: $CONTAINER_IMAGE${NC}"
echo ""

# =================================================================
# FUNCTIONS
# =================================================================

check_connection() {
    echo "🔍 Verificando conexión SSH..."
    if ssh -o ConnectTimeout=5 $JETSON_USER@$JETSON_IP "echo 'SSH OK'" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ SSH connection: OK${NC}"
        return 0
    else
        echo -e "${RED}❌ SSH connection: FAILED${NC}"
        echo "💡 Verificar:"
        echo "   - Jetson encendido y conectado a red"
        echo "   - IP correcta: $JETSON_IP"
        echo "   - SSH keys configuradas"
        return 1
    fi
}

check_docker_access() {
    echo "🐳 Verificando acceso a Docker..."
    if ssh $JETSON_USER@$JETSON_IP "docker --version" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Docker access: OK${NC}"
        return 0
    else
        echo -e "${RED}❌ Docker access: FAILED${NC}"
        echo "💡 En Jetson ejecutar: sudo usermod -aG docker $JETSON_USER"
        return 1
    fi
}

check_nvidia_runtime() {
    echo "🎮 Verificando NVIDIA container runtime..."
    if ssh $JETSON_USER@$JETSON_IP "docker run --rm --runtime nvidia hello-world" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ NVIDIA runtime: OK${NC}"
        return 0
    else
        echo -e "${RED}❌ NVIDIA runtime: FAILED${NC}"
        echo "💡 Verificar nvidia-container-runtime instalado"
        return 1
    fi
}

check_local_files() {
    echo "📁 Verificando archivos locales..."
    
    if [ ! -f "src/communication/jetson_server.py" ]; then
        echo -e "${RED}❌ jetson_server.py not found in src/communication/${NC}"
        return 1
    fi
    
    if [ ! -f "src/communication/protocols.py" ]; then
        echo -e "${RED}❌ protocols.py not found in src/communication/${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Required files found${NC}"
    return 0
}

setup_jetson_directory() {
    echo "📂 Configurando directorio en Jetson..."
    
    if ssh $JETSON_USER@$JETSON_IP "mkdir -p ~/aria-navigation"; then
        echo -e "${GREEN}✅ Jetson directory created${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to create Jetson directory${NC}"
        return 1
    fi
}

copy_files_to_jetson() {
    echo "📤 Copiando archivos al Jetson..."
    
    # Copy main files
    if scp src/communication/jetson_server.py src/communication/protocols.py $JETSON_USER@$JETSON_IP:~/aria-navigation/; then
        echo -e "${GREEN}✅ Core files copied${NC}"
    else
        echo -e "${RED}❌ Failed to copy core files${NC}"
        return 1
    fi
    
    # Copy helper script
    if scp run_jetson_server.sh $JETSON_USER@$JETSON_IP:~/aria-navigation/; then
        echo -e "${GREEN}✅ Helper script copied${NC}"
        ssh $JETSON_USER@$JETSON_IP "chmod +x ~/aria-navigation/run_jetson_server.sh"
    else
        echo -e "${YELLOW}⚠️ Helper script copy failed (not critical)${NC}"
    fi
    
    return 0
}

pull_container_image() {
    echo "🐳 Descargando imagen del container..."
    
    if ssh $JETSON_USER@$JETSON_IP "docker pull $CONTAINER_IMAGE" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Container image ready${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️ Container pull failed - will download on first run${NC}"
        return 0  # Not critical, will download on first run
    fi
}

test_deployment() {
    echo "🧪 Testing deployment..."
    
    if ssh $JETSON_USER@$JETSON_IP "cd ~/aria-navigation && docker run --rm --runtime nvidia -v \$(pwd):/workspace -w /workspace $CONTAINER_IMAGE python3 jetson_server.py test"; then
        echo -e "${GREEN}✅ Deployment test PASSED${NC}"
        return 0
    else
        echo -e "${RED}❌ Deployment test FAILED${NC}"
        return 1
    fi
}

print_usage_instructions() {
    echo ""
    echo -e "${YELLOW}🎯 DEPLOYMENT COMPLETED SUCCESSFULLY!${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Start Jetson server:"
    echo "   ssh $JETSON_USER@$JETSON_IP"
    echo "   cd ~/aria-navigation"
    echo "   bash run_jetson_server.sh run"
    echo ""
    echo "2. Start Mac client (in another terminal):"
    echo "   cd ~/aria-navigation-tfm"
    echo "   python3 src/communication/mac_client.py"
    echo ""
    echo "3. Monitor system:"
    echo "   - Jetson will show processing stats"
    echo "   - Mac will display processed dashboard"
    echo "   - Audio commands executed on Jetson"
    echo ""
}

# =================================================================
# MAIN DEPLOYMENT FLOW
# =================================================================

main() {
    echo "Starting deployment process..."
    echo ""
    
    # Pre-flight checks
    if ! check_local_files; then
        echo -e "${RED}❌ Local files missing - aborting${NC}"
        exit 1
    fi
    
    if ! check_connection; then
        echo -e "${RED}❌ Cannot connect to Jetson - aborting${NC}"
        exit 1
    fi
    
    if ! check_docker_access; then
        echo -e "${RED}❌ Docker not accessible - aborting${NC}"
        exit 1
    fi
    
    if ! check_nvidia_runtime; then
        echo -e "${RED}❌ NVIDIA runtime not available - aborting${NC}"
        exit 1
    fi
    
    # Setup and deployment
    if ! setup_jetson_directory; then
        echo -e "${RED}❌ Directory setup failed - aborting${NC}"
        exit 1
    fi
    
    if ! copy_files_to_jetson; then
        echo -e "${RED}❌ File copy failed - aborting${NC}"
        exit 1
    fi
    
    # Optional optimizations
    pull_container_image
    
    # Final test
    if ! test_deployment; then
        echo -e "${RED}❌ Deployment test failed - check logs${NC}"
        exit 1
    fi
    
    # Success
    print_usage_instructions
}

# =================================================================
# SCRIPT EXECUTION
# =================================================================

# Check if we're in the right directory
if [ ! -f "src/communication/jetson_server.py" ]; then
    echo -e "${RED}❌ Run this script from aria-navigation-tfm project root${NC}"
    echo "💡 Current directory: $(pwd)"
    echo "💡 Expected: ~/aria-navigation-tfm"
    exit 1
fi

# Run main deployment
main

echo ""
echo -e "${GREEN}🎉 JETSON DEPLOYMENT COMPLETE!${NC}"