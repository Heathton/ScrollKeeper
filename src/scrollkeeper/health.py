from __future__ import annotations

import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

log = logging.getLogger(__name__)


class Heartbeat:
    """Liveness signal: the event loop calls `beat()` regularly, the probe thread checks the age.

    A stalled loop (the failure that drops Discord voice) stops beating and the probe turns unhealthy.
    """

    def __init__(self, max_age_seconds: float = 60.0) -> None:
        self.max_age_seconds = max_age_seconds
        self._last = time.monotonic()

    def beat(self) -> None:
        self._last = time.monotonic()

    def is_alive(self) -> bool:
        return time.monotonic() - self._last <= self.max_age_seconds


def start_health_server(heartbeat: Heartbeat, port: int) -> ThreadingHTTPServer | None:
    """Serve `GET /healthz` from a daemon thread. A port of 0 or less disables the server."""
    if port <= 0:
        return None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path.rstrip("/") != "/healthz":
                self.send_response(404)
                self.end_headers()
                return
            alive = heartbeat.is_alive()
            body = b"ok\n" if alive else b"event loop stalled\n"
            self.send_response(200 if alive else 503)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, name="healthz", daemon=True).start()
    log.info("Health endpoint listening on port %s", server.server_address[1])
    return server
