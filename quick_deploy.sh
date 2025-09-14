#!/bin/bash
# =================================================================
# JETSON QUICK DEPLOYMENT SCRIPT - TFM Navigation System  
# Sync rápido: Solo archivos modificados + restart automático
# =================================================================

# Configuration
JETSON_IP="${JETSON_IP:-192.168.8.204}"
JETSON_USER="${JETSON_USER:-aria}"
JETSON_PATH="~/jetson-aria"
LOCAL_SRC="./src"

echo "⚡ QUICK DEPLOY - Solo archivos modificados"
echo "============================================"
echo "📱 Mac → 🤖 Jetson (Fast Sync)"

# Verificar estructura básica
if [[ ! -d "src" ]]; then
    echo "❌ No encontrado src/ - usar ./jetson_deploy.sh primero"
    exit 1
fi

# Test conexión rápido
echo "🔍 Testing conexión..."
if ! ssh -o ConnectTimeout=3 ${JETSON_USER}@${JETSON_IP} "echo OK" > /dev/null 2>&1; then
    echo "❌ Jetson no accesible"
    exit 1
fi
echo "✅ Conexión OK"

# Sync solo archivos modificados (rsync incremental)
echo "📦 Sync incremental..."
rsync -avz --delete \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    --itemize-changes \
    ${LOCAL_SRC}/ ${JETSON_USER}@${JETSON_IP}:${JETSON_PATH}/src/ | grep -E '^[<>cf]' || echo "No changes"

# Restart jetson_server.py si estaba ejecutándose  
echo "🔄 Restart automático..."
ssh ${JETSON_USER}@${JETSON_IP} "
    cd ${JETSON_PATH}
    
    # Kill proceso anterior si existe
    pkill -f 'python3 jetson_server.py' || echo 'No proceso anterior'
    
    # Wait a moment
    sleep 1
    
    # Test que el código funciona
    python3 jetson_server.py test && echo '✅ Quick deploy OK' || echo '❌ Deploy failed'
"

echo "⚡ QUICK DEPLOY COMPLETADO"
echo "💡 Para restart manual: ssh ${JETSON_USER}@${JETSON_IP} 'cd ${JETSON_PATH} && python3 jetson_server.py'"