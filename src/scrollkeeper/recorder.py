"""Writer thread for live recording: one Ogg Opus track per speaker, written as packets arrive.

voice_recv calls the sink from its own packet thread. That thread only enqueues; this thread
does all file and SQLite I/O, so neither the packet thread nor the asyncio event loop waits on
disk.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from .audio import SpeakerTrack
from .storage import Storage

log = logging.getLogger(__name__)

# End an Ogg page and hand it to the OS this often (bounds what a killed pod loses).
FLUSH_SECONDS = 1.0
# fsync this often (bounds what a node crash loses).
FSYNC_SECONDS = 10.0


@dataclass(slots=True, frozen=True)
class Speaker:
    user_id: int
    display_name: str
    character_name: str


@dataclass(slots=True)
class _OpenTrack:
    track_id: int
    track: SpeakerTrack


_STOP = object()


class TrackRecorder:
    def __init__(
        self,
        storage: Storage,
        session_id: int,
        record_dir: Path,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.storage = storage
        self.session_id = session_id
        self.record_dir = record_dir
        self._clock = clock
        self._queue: queue.Queue = queue.Queue()
        self._tracks: dict[int, _OpenTrack] = {}
        self._failed_users: set[int] = set()
        self._thread = threading.Thread(target=self._run, name=f"recorder-{session_id}", daemon=True)

    def start(self) -> None:
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def submit(self, speaker: Speaker, ssrc: int, rtp_timestamp: int, payload: bytes) -> None:
        """Queue one Opus packet. Thread-safe and non-blocking (called from the voice packet thread)."""
        self._queue.put((speaker, ssrc, rtp_timestamp, payload, self._clock()))

    def stop(self) -> None:
        """Write out everything queued, close the tracks, and wait for the thread. Blocking."""
        if self._thread.is_alive():
            self._queue.put(_STOP)
            self._thread.join()

    def _run(self) -> None:
        last_flush = last_fsync = self._clock()
        while True:
            try:
                item = self._queue.get(timeout=FLUSH_SECONDS / 2)
            except queue.Empty:
                item = None
            if item is _STOP:
                break
            if item is not None:
                try:
                    self._write(*item)
                except Exception:
                    # One speaker's broken track must not stop the others from recording.
                    speaker = item[0]
                    if speaker.user_id not in self._failed_users:
                        log.exception("Could not record packet for user %s", speaker.user_id)
                        self._failed_users.add(speaker.user_id)
            now = self._clock()
            if now - last_flush >= FLUSH_SECONDS:
                self._flush(fsync=now - last_fsync >= FSYNC_SECONDS)
                last_flush = now
                if now - last_fsync >= FSYNC_SECONDS:
                    last_fsync = now
        self._close_all()

    def _write(self, speaker: Speaker, ssrc: int, rtp_timestamp: int, payload: bytes, clock: float) -> None:
        open_track = self._tracks.get(speaker.user_id)
        if open_track is None:
            open_track = self._open(speaker, clock)
        open_track.track.add(ssrc, rtp_timestamp, payload, clock)

    def _open(self, speaker: Speaker, first_packet_clock: float) -> _OpenTrack:
        path = self.record_dir / f"{speaker.user_id}.ogg"
        # The track starts when its first packet arrived, not when this thread got to it
        # (they differ if the disk stalled), so tracks of different speakers stay aligned.
        started_at = datetime.utcnow() - timedelta(seconds=max(0.0, self._clock() - first_packet_clock))
        # Row first: if the bot dies right after, recovery still finds (and skips) the track.
        track_id = self.storage.create_track(
            self.session_id, speaker.user_id, speaker.display_name, speaker.character_name, path, started_at
        )
        serial = zlib.crc32(f"{self.session_id}:{speaker.user_id}".encode())
        open_track = _OpenTrack(track_id, SpeakerTrack(path.open("wb"), serial))
        self._tracks[speaker.user_id] = open_track
        log.info("Recording session %s: new track for %s at %s", self.session_id, speaker.character_name, path)
        return open_track

    def _flush(self, fsync: bool) -> None:
        for open_track in self._tracks.values():
            try:
                open_track.track.flush()
                if fsync:
                    os.fsync(open_track.track.handle.fileno())
            except OSError:
                log.exception("Could not flush track %s", open_track.track_id)

    def _close_all(self) -> None:
        ended_at = datetime.utcnow()
        for open_track in self._tracks.values():
            try:
                open_track.track.close()
                self.storage.close_track(open_track.track_id, ended_at)
                log.info(
                    "Closed track %s (%s packets, %.1f s)",
                    open_track.track_id,
                    open_track.track.packets,
                    open_track.track.writer.samples_written / 48000,
                )
            except Exception:
                log.exception("Could not close track %s", open_track.track_id)
        self._tracks.clear()
