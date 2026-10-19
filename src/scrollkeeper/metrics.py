"""Prometheus metrics for session processing state, served by the health server.

A session stuck in `processing` (a wedged LLM or speech-to-text backend, or a
worker that dies without crashing the bot) leaves no other signal: `/healthz`
only watches the event loop, and per-call timeouts bound one call, not a whole
run. This module renders the Prometheus text exposition for the session state
in SQLite; the health server serves it at `GET /metrics` on the same port as
`/healthz` (disabled with `SCROLLKEEPER_HEALTH_PORT=0`). Stdlib only.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from .health import Heartbeat
from .storage import Storage

log = logging.getLogger(__name__)


def _seconds_since(iso: str | None) -> float | None:
    """Age in seconds of a stored `datetime.utcnow().isoformat()` value."""
    if not iso:
        return None
    try:
        return max(0.0, time.time() - datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())
    except ValueError:
        log.warning("Unparseable session timestamp %r", iso)
        return None


def render_metrics(storage: Storage, heartbeat: Heartbeat) -> str:
    """Prometheus text exposition of the session processing state.

    A scrape reads the `sessions` table once and renders whatever it gets; the
    health server turns a failure into a 500 (a scrape error, visible in
    Prometheus) rather than a dropped endpoint.
    """
    lines = [
        "# HELP scrollkeeper_event_loop_alive 1 when the bot's event loop has beaten recently",
        "# TYPE scrollkeeper_event_loop_alive gauge",
        f"scrollkeeper_event_loop_alive {1 if heartbeat.is_alive() else 0}",
        "# HELP scrollkeeper_sessions Number of sessions in each live state",
        "# TYPE scrollkeeper_sessions gauge",
        f'scrollkeeper_sessions{{status="recording"}} {storage.count_sessions("recording")}',
        f'scrollkeeper_sessions{{status="processing"}} {storage.count_sessions("processing")}',
        "# HELP scrollkeeper_session_processing_seconds Seconds a session has spent in its current processing run",
        "# TYPE scrollkeeper_session_processing_seconds gauge",
        "# HELP scrollkeeper_session_processing_attempts Processing attempts counted for each session in processing",
        "# TYPE scrollkeeper_session_processing_attempts gauge",
    ]
    for session in storage.sessions_in_processing():
        labels = f'session_id="{session["id"]}",guild_id="{session["guild_id"]}"'
        # The age keeps growing across bot restarts (a restart continues the
        # same run); rows from before the column existed fall back to the
        # recording start, which overstates the age rather than hiding a stuck
        # session.
        age = _seconds_since(session["processing_started_at"])
        if age is None:
            age = _seconds_since(session["started_at"])
        if age is not None:
            lines.append(f"scrollkeeper_session_processing_seconds{{{labels}}} {age:.3f}")
        lines.append(f"scrollkeeper_session_processing_attempts{{{labels}}} {int(session['processing_attempts'] or 0)}")
    return "\n".join(lines) + "\n"
