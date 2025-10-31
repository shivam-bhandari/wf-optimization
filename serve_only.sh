#!/bin/bash
# Quick serve: Just start server and open browser (no benchmark run)

echo "🚀 Starting web server..."

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
sleep 3

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start!"
    exit 1
fi

# Try to detect the port
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

# Open browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$URL"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    xdg-open "$URL" 2>/dev/null || echo "Please open $URL in your browser"
fi

echo "📊 Dashboard: $URL"
echo "Press Ctrl+C to stop the server"
echo ""

# Wait for Ctrl+C
trap "echo ''; echo 'Stopping server...'; kill $SERVER_PID 2>/dev/null; echo '✅ Stopped'; exit 0" INT TERM
wait $SERVER_PID

