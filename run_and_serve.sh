#!/bin/bash
# Complete pipeline: Generate reports → Prepare web data → Start server → Open browser

set -e  # Exit on error

echo "=========================================="
echo "Workflow Benchmark Pipeline"
echo "=========================================="
echo ""

# Step 1: Run benchmarks
echo "[1/4] Running benchmarks..."
if ! poetry run python main.py; then
    echo "❌ Benchmark failed!"
    exit 1
fi
echo "✅ Benchmarks complete"
echo ""

# Step 2: Prepare web data
echo "[2/4] Preparing web data..."
if ! bash scripts/prepare_web_data.sh; then
    echo "❌ Web data preparation failed!"
    exit 1
fi
echo "✅ Web data ready"
echo ""

# Step 3: Start web server in background
echo "[3/4] Starting web server..."

# Detect Python command (python3 or python)
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo "❌ Python not found! Please install Python 3."
        exit 1
    fi
fi

cd web
$PYTHON_CMD serve.py --no-open &
SERVER_PID=$!
cd ..
echo "✅ Server starting (PID: $SERVER_PID)"
echo ""

# Wait for server to start and get the actual port
echo "Waiting for server to start..."
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start (PID $SERVER_PID not found)!"
    echo "   Check if Python is installed: python3 --version"
    exit 1
fi

# Give it a moment more to actually bind to port
sleep 1

# Try to detect the port (check common ports)
DETECTED_PORT=8000
for port in 8000 8001 8002 8003; do
    # Try multiple methods to check if port is open
    if (python3 -c "import socket; s=socket.socket(); s.settimeout(0.5); result=s.connect_ex(('localhost', $port)); s.close(); exit(0 if result == 0 else 1)" 2>/dev/null) || \
       (command -v curl >/dev/null && curl -s "http://localhost:$port" >/dev/null 2>&1) || \
       (command -v nc >/dev/null && nc -z localhost $port 2>/dev/null); then
        DETECTED_PORT=$port
        break
    fi
done

URL="http://localhost:${DETECTED_PORT}"
echo "✅ Server running at $URL"
echo ""

# Step 4: Open browser
echo "[4/4] Opening browser..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open "$URL" 2>/dev/null || echo "Please open $URL in your browser"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start "$URL"
else
    echo "Please open $URL in your browser"
fi
echo "✅ Browser opened"
echo ""

echo "=========================================="
echo "🎉 Pipeline Complete!"
echo "=========================================="
echo ""
echo "📊 Dashboard: $URL"
echo "📁 Results: results/$(date +%Y-%m-%d)/"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping server...'; kill $SERVER_PID 2>/dev/null; echo '✅ Server stopped'; exit 0" INT TERM

# Keep script running
wait $SERVER_PID

