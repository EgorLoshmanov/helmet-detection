from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from urllib.parse import urlsplit


def make_server(monitor, host: str, port: int) -> ThreadingHTTPServer:
    index = (Path(__file__).parent / "static" / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(5)

        def reply(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def json(self, value, status=200) -> None:
            self.reply(json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            route = urlsplit(self.path).path
            if route == "/":
                self.reply(index, "text/html; charset=utf-8")
            elif route == "/api/status":
                self.json(monitor.snapshot()[0])
            elif route == "/api/settings":
                self.json(monitor.settings())
            elif route == "/api/events":
                self.json(monitor.journal.recent())
            elif route == "/api/metrics.csv":
                self.reply(monitor.metrics_csv().encode("utf-8"), "text/csv; charset=utf-8")
            elif route == "/frame.jpg":
                jpeg = monitor.snapshot()[1]
                if jpeg is None:
                    self.json({"error": "Frame unavailable"}, 503)
                else:
                    self.reply(jpeg, "image/jpeg")
            elif re.fullmatch(r"/snapshots/[1-9][0-9]*\.jpg", route):
                jpeg = monitor.journal.snapshot(int(route.rsplit("/", 1)[1][:-4]))
                if jpeg is None:
                    self.json({"error": "Snapshot not found"}, 404)
                else:
                    self.reply(jpeg, "image/jpeg")
            else:
                self.json({"error": "Not found"}, 404)

        def do_POST(self) -> None:
            if self.path != "/api/settings":
                self.json({"error": "Not found"}, 404)
                return
            # Browser writes must originate from this panel, with JSON content.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://{self.headers.get('Host')}":
                self.json({"error": "Cross-origin write rejected"}, 403)
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.json({"error": "Expected application/json"}, 415)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Body must be between 1 and 8192 bytes")
                values = json.loads(self.rfile.read(length))
                self.json(monitor.configure(values))
            except (ValueError, UnicodeError) as exc:
                self.json({"error": str(exc)}, 400)
            except OSError as exc:
                self.json({"error": f"Could not save settings: {exc}"}, 500)

    return ThreadingHTTPServer((host, port), Handler)
