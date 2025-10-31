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
from functools import partial

PORT = 8000
ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = ROOT
os.chdir(WEB_DIR)

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Set JS correct MIME
        if self.path.endswith('.js'):
            self.send_header('Content-Type', 'application/javascript')
        return super().end_headers()

try:
    url = f"http://localhost:{PORT}"
    print(f"\n[INFO] Serving web dashboard at {url}")
    print("Press Ctrl+C to stop the server.\n")
    webbrowser.open(url)
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        httpd.serve_forever()
except KeyboardInterrupt:
    print("\nServer stopped by user (Ctrl+C). Exiting.")
    sys.exit(0)
