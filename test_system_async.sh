#!/bin/bash
# Test rápido del sistema con AsyncTelemetryLogger

cd /home/roberto/Projects/aria-nav

echo "🧪 Testing aria-nav with AsyncTelemetryLogger..."
echo "📊 Running 50 frames in debug mode..."
echo ""

# Ejecutar test y capturar output
/home/roberto/Projects/aria-nav/.venv/bin/python run.py test 2>&1 | tee /tmp/aria_test.log | grep -E "(TELEMETRY|FPS|Modo asíncrono|Nueva sesión|frame [0-9]+:)" | tail -30

echo ""
echo "✅ Test complete. Latest session:"
LATEST_SESSION=$(ls -td logs/session_* 2>/dev/null | head -1)
if [ -n "$LATEST_SESSION" ]; then
    echo "   $LATEST_SESSION"
    echo ""
    echo "📁 Files generated:"
    ls -lh "$LATEST_SESSION"/*.jsonl 2>/dev/null | awk '{print "   "$9": "$5}'
    echo ""
    echo "📊 Line counts:"
    wc -l "$LATEST_SESSION"/*.jsonl 2>/dev/null | tail -1 | awk '{print "   Total: "$1" lines"}'
fi
