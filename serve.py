#!/usr/bin/env python3
"""
Local dev server for the web dashboard.
- Serves files from ./web at http://localhost:8000
- Opens default browser
- Handles JS MIME type for .js files
- Stops gracefully on Ctrl+C
"""
import http.server
import socketserver
import webbrowser
import sys
import os
import argparse

PORT = 8000
ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = ROOT
os.chdir(WEB_DIR)

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Set JS correct MIME
        if self.path.endswith('.js'):
            self.send_header('Content-Type', 'application/javascript')
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        return super().end_headers()
    
    def log_message(self, format, *args):
        # Suppress default logging, we'll handle it ourselves
        pass

def find_free_port(start_port=8000):
    """Find a free port starting from start_port."""
    import socket
    for port in range(start_port, start_port + 10):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError("Could not find a free port")

def main():
    parser = argparse.ArgumentParser(description='Serve web dashboard locally')
    parser.add_argument('--port', type=int, default=PORT, help='Port to serve on (default: 8000)')
    parser.add_argument('--no-open', action='store_true', help='Do not open browser automatically')
    args = parser.parse_args()
    
    port = args.port
    # Try to use specified port, or find a free one
    if port == PORT:
        try:
            port = find_free_port(PORT)
        except RuntimeError:
            print(f"[ERROR] Port {PORT} is in use and no free port found. Try a different port.")
            sys.exit(1)
    
    try:
        url = f"http://localhost:{port}"
        print(f"\n{'='*50}")
        print(f"[INFO] Serving web dashboard at {url}")
        print(f"[INFO] Serving from: {WEB_DIR}")
        print(f"Press Ctrl+C to stop the server.")
        print(f"{'='*50}\n")
        
        if not args.no_open:
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"[WARN] Could not open browser automatically: {e}")
                print(f"[INFO] Please open {url} manually in your browser.\n")
        
        # Bind to 0.0.0.0 to allow connections from any interface
        with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
            httpd.serve_forever()
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"[ERROR] Port {port} is already in use.")
            print(f"[INFO] Try running with --port <different_port> or kill the process using port {port}")
            sys.exit(1)
        else:
            print(f"[ERROR] Failed to start server: {e}")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user (Ctrl+C). Exiting.")
        sys.exit(0)

if __name__ == "__main__":
    main()
