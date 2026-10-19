from __future__ import annotations

import socket
import tempfile
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from scrollkeeper.health import Heartbeat, start_health_server
from scrollkeeper.metrics import render_metrics
from scrollkeeper.storage import Storage


def _samples(text: str) -> list[tuple[str, dict[str, str], float]]:
    """The exposition as (metric, labels, value) triples, skipping HELP/TYPE lines."""
    samples: list[tuple[str, dict[str, str], float]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value_text = line.partition(" ")
        labels: dict[str, str] = {}
        if name.endswith("}"):
            name, _, label_blob = name.partition("{")
            for pair in label_blob.rstrip("}").split(","):
                key, _, value = pair.partition("=")
                labels[key] = value.strip('"')
        samples.append((name, labels, float(value_text)))
    return samples


def _value(text: str, metric: str, **labels: str) -> float | None:
    """The value of `metric` whose labels include all of `labels`, or None if absent."""
    for name, sample_labels, value in _samples(text):
        if name == metric and all(sample_labels.get(key) == value_ for key, value_ in labels.items()):
            return value
    return None


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage = Storage(Path(self._tmp.name))

    def test_processing_session_age_and_attempts(self) -> None:
        session_id = self.storage.create_session(1, 100, 200, "s")
        self.storage.queue_session(session_id)
        self.storage.begin_processing_attempt(session_id)
        text = render_metrics(self.storage, Heartbeat())
        age = _value(text, "scrollkeeper_session_processing_seconds", session_id=str(session_id), guild_id="1")
        self.assertIsNotNone(age)
        assert age is not None
        self.assertGreater(age, 0.0)
        self.assertEqual(_value(text, "scrollkeeper_session_processing_attempts", session_id=str(session_id), guild_id="1"), 1.0)

    def test_counts_report_live_states(self) -> None:
        self.storage.create_session(1, 100, 200, "recording")
        queued = self.storage.create_session(1, 101, 201, "queued")
        self.storage.queue_session(queued)
        text = render_metrics(self.storage, Heartbeat())
        self.assertEqual(_value(text, "scrollkeeper_sessions", status="recording"), 1.0)
        self.assertEqual(_value(text, "scrollkeeper_sessions", status="processing"), 1.0)

    def test_event_loop_alive_follows_the_heartbeat(self) -> None:
        heartbeat = Heartbeat(max_age_seconds=60)
        self.assertEqual(_value(render_metrics(self.storage, heartbeat), "scrollkeeper_event_loop_alive"), 1.0)
        heartbeat.max_age_seconds = -1
        self.assertEqual(_value(render_metrics(self.storage, heartbeat), "scrollkeeper_event_loop_alive"), 0.0)

    def test_completed_session_exports_no_processing_samples(self) -> None:
        session_id = self.storage.create_session(1, 100, 200, "s")
        self.storage.queue_session(session_id)
        self.storage.finalize_session(session_id, "transcript.txt", "summary.md")
        text = render_metrics(self.storage, Heartbeat())
        self.assertIsNone(_value(text, "scrollkeeper_session_processing_seconds", session_id=str(session_id)))
        self.assertIsNone(_value(text, "scrollkeeper_session_processing_attempts", session_id=str(session_id)))
        with self.storage.connection() as conn:
            row = conn.execute("SELECT processing_started_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
        self.assertIsNone(row["processing_started_at"])

    def test_restart_does_not_reset_processing_age(self) -> None:
        session_id = self.storage.create_session(1, 100, 200, "s")
        self.storage.queue_session(session_id)
        with self.storage.connection() as conn:
            before = conn.execute("SELECT processing_started_at FROM sessions WHERE id = ?", (session_id,)).fetchone()[
                "processing_started_at"
            ]
        # A bot crash mid-run counts an attempt on recovery but starts no new run.
        self.storage.begin_processing_attempt(session_id)
        self.storage.begin_processing_attempt(session_id)
        with self.storage.connection() as conn:
            after = conn.execute("SELECT processing_started_at FROM sessions WHERE id = ?", (session_id,)).fetchone()[
                "processing_started_at"
            ]
        self.assertEqual(before, after)

    def test_reprocessing_starts_the_clock_again(self) -> None:
        session_id = self.storage.create_session(1, 100, 200, "s")
        self.storage.queue_session(session_id)
        self.storage.finalize_session(session_id, "transcript.txt", "summary.md")
        with self.storage.connection() as conn:
            conn.execute("UPDATE sessions SET processing_started_at = NULL WHERE id = ?", (session_id,))
        self.storage.reset_session_llm_processing(session_id)
        with self.storage.connection() as conn:
            row = conn.execute("SELECT processing_started_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
        self.assertIsNotNone(row["processing_started_at"])

    def test_pre_upgrade_row_falls_back_to_recording_start(self) -> None:
        session_id = self.storage.create_session(1, 100, 200, "s")
        self.storage.queue_session(session_id)
        old = (datetime.utcnow() - timedelta(hours=3)).isoformat()
        with self.storage.connection() as conn:
            conn.execute(
                "UPDATE sessions SET processing_started_at = NULL, started_at = ? WHERE id = ?",
                (old, session_id),
            )
        age = _value(
            render_metrics(self.storage, Heartbeat()),
            "scrollkeeper_session_processing_seconds",
            session_id=str(session_id),
        )
        self.assertIsNotNone(age)
        assert age is not None
        self.assertGreater(age, 2.9 * 3600)
        self.assertLess(age, 3.2 * 3600)


class HealthMetricsEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage = Storage(Path(self._tmp.name))

    def _start(self, heartbeat: Heartbeat, metrics_provider: Any = None) -> tuple[Any, int]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = start_health_server(heartbeat, port, metrics_provider=metrics_provider)
        self.assertIsNotNone(server)
        return server, port

    def test_metrics_served_alongside_healthz(self) -> None:
        server, port = self._start(Heartbeat(), lambda: render_metrics(self.storage, Heartbeat()))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics") as resp:
                self.assertEqual(resp.status, 200)
                self.assertIn("text/plain; version=0.0.4", resp.headers.get("Content-Type"))
                self.assertIn(b"scrollkeeper_event_loop_alive 1", resp.read())
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz") as resp:
                self.assertEqual(resp.status, 200)
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/nope")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()

    def test_metrics_500_when_the_provider_raises(self) -> None:
        def boom() -> str:
            raise RuntimeError("database is locked")

        server, port = self._start(Heartbeat(), boom)
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics")
            self.assertEqual(ctx.exception.code, 500)
        finally:
            server.shutdown()
            server.server_close()

    def test_metrics_404_without_a_provider(self) -> None:
        server, port = self._start(Heartbeat())
        try:
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics")
            self.assertEqual(ctx.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
