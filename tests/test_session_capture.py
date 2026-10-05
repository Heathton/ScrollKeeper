from __future__ import annotations

import asyncio
import tempfile
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scrollkeeper.audio import FRAME_SAMPLES, SpeakerTrack
from scrollkeeper.models import SpeakerSegment, TimedText, TranscriptionResult
from scrollkeeper.storage import PROCESS_TRANSCRIBE, Storage
from scrollkeeper.wiki import CampaignWiki

from test_audio import VOICE
from test_llm_summary import fake_settings

DISCORD_IMPORT_ERROR: str | None = None
try:
    from scrollkeeper.session_manager import MAX_PROCESSING_ATTEMPTS, SessionAudioSink, SessionManager
    from test_session_wiki import SummaryLLM
except ModuleNotFoundError as exc:
    SessionManager = None
    SummaryLLM = object
    DISCORD_IMPORT_ERROR = str(exc)


class TranscribingLLM(SummaryLLM):
    """Fake STT: returns the scripted words for each track (keyed by user id), records uploads."""

    def __init__(self, words_by_user: dict[int, list[tuple[str, float, float]]]) -> None:
        super().__init__()
        self.words_by_user = words_by_user
        self.uploads: list[Path] = []
        self.fail = False

    async def transcribe_track(self, audio_path: Path) -> TranscriptionResult:
        self.uploads.append(audio_path)
        if self.fail:
            raise RuntimeError("speech-to-text service unavailable")
        user_id = int(audio_path.read_text())
        words = [TimedText(*word) for word in self.words_by_user.get(user_id, [])]
        return TranscriptionResult(text=" ".join(w.text for w in words), words=words)


def fake_decode(source: Path, destination: Path) -> None:
    """Stands in for ffmpeg: the 'decoded' file just names the speaker; rejects files marked broken."""
    if source.read_bytes().startswith(b"broken"):
        raise RuntimeError("ffmpeg could not decode")
    destination.write_text(source.stem)


def write_track(path: Path, seconds: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    track = SpeakerTrack(path.open("wb"), serial=1)
    for index in range(int(seconds * 50)):
        track.add(1, index * FRAME_SAMPLES, VOICE, index * 0.02)
    track.close()


@unittest.skipUnless(SessionManager is not None, f"discord dependency unavailable: {DISCORD_IMPORT_ERROR}")
class SessionCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.storage = Storage(root / "data")
        self.spool = root / "spool"
        self.llm = TranscribingLLM(
            {
                10: [("Who", 1.0, 1.2), ("goes", 1.3, 1.5), ("there?", 1.6, 2.0), ("Show", 9.0, 9.2), ("yourself.", 9.3, 9.8)],
                20: [("A", 3.0, 3.1), ("friend.", 3.2, 3.6)],
            }
        )
        self.wiki = CampaignWiki(self.storage, self.llm, fake_settings())
        self.manager = SessionManager(self.storage, self.llm, self.wiki, spool_dir=self.spool, audio_retention_days=7)
        self.notices: list[tuple[int, str]] = []
        self.completions: list[tuple] = []

        async def notice(channel_id: int, message: str) -> None:
            self.notices.append((channel_id, message))

        async def completion(guild_id, channel_id, artifacts, error) -> None:
            self.completions.append((guild_id, channel_id, artifacts, error))

        self.manager.set_notice_handler(notice)
        self.manager.set_completion_handler(completion)
        self.decode_patch = patch("scrollkeeper.session_manager.decode_for_transcription", side_effect=fake_decode)
        self.decode_patch.start()
        self.storage.register_character(1, 10, "Mira")
        self.storage.register_character(1, 20, "Varric")

    async def asyncTearDown(self) -> None:
        self.decode_patch.stop()
        self._tmp.cleanup()

    def recorded_session(self, status: str = "recording") -> tuple[int, datetime]:
        """A session with two spooled tracks, as the recorder leaves it; Varric joined 2 s late."""
        session_id = self.storage.create_session(1, 2, 3, "Test")
        started = datetime.fromisoformat(self.storage.get_session(session_id)["started_at"])
        for user_id, name, offset in ((10, "Mira", 0), (20, "Varric", 2)):
            path = self.spool / str(session_id) / f"{user_id}.ogg"
            write_track(path)
            self.storage.create_track(session_id, user_id, name.lower(), name, path, started + timedelta(seconds=offset))
        self.storage.set_session_status(session_id, status)
        return session_id, started

    async def drain_queue(self) -> None:
        while self.manager.processing_tasks:
            await asyncio.gather(*self.manager.processing_tasks.values())

    # --- live capture ------------------------------------------------------------------

    def make_sink(self) -> tuple[SessionAudioSink, list]:
        submitted: list = []
        recorder = types.SimpleNamespace(submit=lambda *args: submitted.append(args))
        session = self.manager._session_from_row(self.storage.get_session(self.storage.create_session(1, 2, 3, None)))
        session.recorder = recorder
        return SessionAudioSink(self.manager, session), submitted

    @staticmethod
    def voice(timestamp: int = 0, payload: bytes = VOICE, ssrc: int = 5):
        return types.SimpleNamespace(packet=types.SimpleNamespace(ssrc=ssrc, timestamp=timestamp), opus=payload)

    def test_sink_records_registered_users_only_and_looks_names_up_once(self) -> None:
        sink, submitted = self.make_sink()
        posted: list[str] = []
        self.manager.post_threadsafe = lambda _channel, message: posted.append(message)
        player = types.SimpleNamespace(id=10, display_name="mira#1", bot=False)
        guest = types.SimpleNamespace(id=30, display_name="guest", bot=False)
        music_bot = types.SimpleNamespace(id=40, display_name="DJ", bot=True)

        with patch.object(self.storage, "get_registered_character_name", wraps=self.storage.get_registered_character_name) as lookup:
            for timestamp in range(0, 5 * FRAME_SAMPLES, FRAME_SAMPLES):
                for user in (player, guest, music_bot):
                    sink.write(user, self.voice(timestamp))
            sink.write(player, self.voice(payload=b""))  # lost packet: nothing to write
            sink.write(None, self.voice())
        self.assertEqual(lookup.call_count, 2)  # player and guest, once each; bots are never looked up
        self.assertEqual([(args[0].character_name, args[2]) for args in submitted], [("Mira", t) for t in range(0, 5 * 960, 960)])
        self.assertEqual(len(posted), 1)
        self.assertIn("guest", posted[0])

        # The guest registers mid-session: recorded from their next packet on.
        self.storage.register_character(1, 30, "Brother Aldous")
        self.manager.active_sessions[1] = sink.session
        sink.session.sink = sink
        self.manager.forget_speaker(1, 30)
        sink.write(guest, self.voice())
        self.assertEqual(submitted[-1][0].character_name, "Brother Aldous")

    # --- processing ---------------------------------------------------------------------

    async def test_end_to_end_tracks_are_archived_transcribed_and_merged_by_time(self) -> None:
        session_id, _ = self.recorded_session(status="processing")
        self.storage.queue_session(session_id, PROCESS_TRANSCRIBE)
        self.manager._ensure_worker(1)
        await self.drain_queue()

        archive = self.storage.sessions_dir / str(session_id) / "audio"
        self.assertEqual(sorted(p.name for p in archive.iterdir()), ["10.ogg", "20.ogg"])
        self.assertFalse((self.spool / str(session_id)).exists())
        self.assertEqual([Path(t["path"]).parent for t in self.storage.get_session_tracks(session_id)], [archive, archive])
        self.assertTrue(all(not upload.exists() for upload in self.llm.uploads), "scratch files are deleted")

        transcript = (self.storage.sessions_dir / str(session_id) / "transcript.md").read_text()
        self.assertEqual(
            transcript,
            "# Transcript\n\nMira: Who goes there?\n\nVarric: A friend.\n\nMira: Show yourself.\n",
        )
        self.assertIn("[00:00:05] Varric: A friend.", self.llm.extract_calls[0][1])  # 2 s late + 3 s into the track
        self.assertEqual(self.storage.get_session(session_id)["status"], "completed")
        self.assertIsNone(self.completions[0][3])
        self.assertIn("Transcribing 2 speaker track(s)", self.notices[0][1])

    async def test_undecodable_track_is_skipped_but_stt_failure_fails_the_session(self) -> None:
        session_id, _ = self.recorded_session(status="processing")
        (self.spool / str(session_id) / "20.ogg").write_bytes(b"broken")
        self.storage.queue_session(session_id)
        self.manager._ensure_worker(1)
        await self.drain_queue()
        self.assertEqual(self.storage.get_session(session_id)["status"], "completed")
        self.assertTrue(any("Could not decode the audio of: Varric" in message for _, message in self.notices))

        self.llm.fail = True
        await self.manager.reprocess_session(1, session_id)
        await self.drain_queue()
        self.assertEqual(self.storage.get_session(session_id)["status"], "failed")
        self.assertIn("unavailable", self.completions[-1][3])

    async def test_no_tracks_explains_the_opt_in(self) -> None:
        session_id = self.storage.create_session(1, 2, 3, None)
        self.storage.queue_session(session_id)
        self.manager._ensure_worker(1)
        await self.drain_queue()
        self.assertIn("register-character", self.completions[0][3])

    # --- restart recovery ---------------------------------------------------------------

    async def test_startup_recovers_interrupted_recording_and_processing(self) -> None:
        recording_id, _ = self.recorded_session(status="recording")
        processing_id, _ = self.recorded_session(status="processing")
        self.storage.queue_session(processing_id)
        self.storage.begin_processing_attempt(processing_id)  # it was mid-way when the bot died

        await self.manager.start()
        await self.drain_queue()

        for session_id in (recording_id, processing_id):
            self.assertEqual(self.storage.get_session(session_id)["status"], "completed")
        self.assertIsNotNone(self.storage.get_session(recording_id)["interrupted_at"])
        messages = [message for _, message in self.notices]
        self.assertTrue(any(f"#{recording_id}** was still recording" in message for message in messages))
        self.assertTrue(any(f"Resuming processing of session **#{processing_id}" in message for message in messages))
        transcript = (self.storage.sessions_dir / str(recording_id) / "transcript.md").read_text()
        self.assertIn("interrupted by a bot restart", transcript)
        await self.manager.start()  # a second on_ready does nothing
        self.assertEqual(len(self.completions), 2)

    async def test_session_that_keeps_crashing_the_bot_is_given_up(self) -> None:
        session_id, _ = self.recorded_session(status="processing")
        self.storage.queue_session(session_id)
        for _ in range(MAX_PROCESSING_ATTEMPTS):
            self.storage.begin_processing_attempt(session_id)
        await self.manager.start()
        await self.drain_queue()
        self.assertEqual(self.storage.get_session(session_id)["status"], "failed")
        self.assertIn("giving up", self.completions[0][3])
        self.assertEqual(self.llm.uploads, [])

    # --- old recordings and retention ----------------------------------------------------

    async def test_legacy_clip_session_is_converted_and_reprocessed(self) -> None:
        session_id = self.storage.create_session(1, 2, 3, "Old")
        started = datetime.fromisoformat(self.storage.get_session(session_id)["started_at"])
        audio_dir = self.storage.sessions_dir / str(session_id) / "audio"
        audio_dir.mkdir(parents=True)
        import struct

        for user_id, name, offset in ((10, "Mira", 1), (20, "Varric", 3)):
            clip = audio_dir / f"{offset}_{user_id}_{name}.skopus"
            clip.write_bytes(b"".join(struct.pack("<HIH", i, i * 960, len(VOICE)) + VOICE for i in range(10)))
            at = started + timedelta(seconds=offset)
            self.storage.add_transcript_segment(session_id, SpeakerSegment(user_id, name.lower(), name, at, at, clip, "old text"))
        self.storage.set_session_status(session_id, "completed")

        await self.manager.reprocess_session(1, session_id)
        await self.drain_queue()

        tracks = self.storage.get_session_tracks(session_id)
        self.assertEqual([(t["user_id"], Path(t["path"]).name) for t in tracks], [(10, "10.ogg"), (20, "20.ogg")])
        self.assertEqual(tracks[1]["started_at"], (started + timedelta(seconds=3)).isoformat())
        transcript = (self.storage.sessions_dir / str(session_id) / "transcript.md").read_text()
        self.assertNotIn("old text", transcript)
        self.assertIn("Varric: A friend.", transcript)

    async def test_audio_is_deleted_after_the_retention_period(self) -> None:
        session_id, _ = self.recorded_session(status="processing")
        self.storage.queue_session(session_id)
        self.manager._ensure_worker(1)
        await self.drain_queue()
        audio_dir = self.storage.sessions_dir / str(session_id) / "audio"
        self.assertTrue(audio_dir.exists())

        self.assertEqual(self.manager.delete_expired_audio(now=datetime.utcnow() + timedelta(days=6)), [])
        self.assertEqual(self.manager.delete_expired_audio(now=datetime.utcnow() + timedelta(days=8)), [session_id])
        self.assertFalse(audio_dir.exists())
        self.assertTrue((self.storage.sessions_dir / str(session_id) / "transcript.md").exists())
        with self.assertRaisesRegex(RuntimeError, "deleted"):
            await self.manager.reprocess_session(1, session_id)


if __name__ == "__main__":
    unittest.main()
