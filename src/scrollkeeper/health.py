from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
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


def start_health_server(
    heartbeat: Heartbeat,
    port: int,
    metrics_provider: Callable[[], str] | None = None,
) -> ThreadingHTTPServer | None:
    """Serve `GET /healthz` (and `GET /metrics`, when a provider is given) from a
    daemon thread. A port of 0 or less disables the server.

    Both endpoints answer from this probe thread, so they stay up even when the
    bot's event loop has stalled: `/healthz` reports the stall (503), `/metrics`
    keeps serving the last recorded session state.
    """
    if port <= 0:
        return None

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.rstrip("/")
            if path == "/healthz":
                alive = heartbeat.is_alive()
                body = b"ok\n" if alive else b"event loop stalled\n"
                self._respond(200 if alive else 503, "text/plain", body)
            elif path == "/metrics" and metrics_provider is not None:
                try:
                    body = metrics_provider().encode("utf-8")
                except Exception:
                    log.exception("Failed to render /metrics")
                    self._respond(500, "text/plain", b"metrics unavailable\n")
                    return
                self._respond(200, "text/plain; version=0.0.4; charset=utf-8", body)
            else:
                self._respond(404, "text/plain", b"not found\n")

        def _respond(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    threading.Thread(target=server.serve_forever, name="health", daemon=True).start()
    log.info("Health endpoint listening on port %s", server.server_address[1])
    return server
