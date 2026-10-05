from __future__ import annotations

import io
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from scrollkeeper.audio import (
    FRAME_SAMPLES,
    OPUS_SILENCE,
    PRE_SKIP_SAMPLES,
    SpeakerTrack,
    TrackTimeline,
    convert_legacy_clips,
    decode_for_transcription,
    ogg_crc,
    ogg_duration_seconds,
    opus_packet_samples,
)
from scrollkeeper.recorder import Speaker, TrackRecorder
from scrollkeeper.storage import MIGRATIONS, PROCESS_LLM_ONLY, PROCESS_TRANSCRIBE, Storage

# A made-up 20 ms CELT packet (TOC 0xFC: config 31, stereo, one frame) with a recognisable body.
VOICE = b"\xfc" + b"voice"
FRAMES_PRE_SKIP = PRE_SKIP_SAMPLES // FRAME_SAMPLES


def read_ogg(data: bytes) -> list[dict]:
    """Minimal Ogg reader for tests: pages with flags, granule, sequence, CRC check, and packets."""
    pages, offset = [], 0
    while offset < len(data):
        capture, version, flags, granule, serial, sequence, crc, count = struct.unpack_from("<4sBBqIIIB", data, offset)
        assert capture == b"OggS" and version == 0
        lacing = data[offset + 27 : offset + 27 + count]
        size = 27 + count + sum(lacing)
        page = bytearray(data[offset : offset + size])
        page[22:26] = b"\0\0\0\0"
        body = data[offset + 27 + count : offset + size]
        packets, current, position = [], b"", 0
        for value in lacing:
            current += body[position : position + value]
            position += value
            if value < 255:
                packets.append(current)
                current = b""
        pages.append(
            {"flags": flags, "granule": granule, "serial": serial, "sequence": sequence, "crc_ok": ogg_crc(bytes(page)) == crc, "packets": packets}
        )
        offset += size
    return pages


def audio_packets(pages: list[dict]) -> list[bytes]:
    """Packets after the two header pages and the pre-skip silence."""
    return [packet for page in pages[2:] for packet in page["packets"]][FRAMES_PRE_SKIP:]


class OpusBasicsTests(unittest.TestCase):
    def test_ogg_crc_check_value(self) -> None:
        # CRC-32 with polynomial 0x04C11DB7, zero init, no reflection or final xor (Ogg's).
        self.assertEqual(ogg_crc(b"123456789"), 0x89A1897F)

    def test_packet_durations_from_toc(self) -> None:
        self.assertEqual(opus_packet_samples(OPUS_SILENCE), 960)  # CELT 20 ms
        self.assertEqual(opus_packet_samples(bytes([0x18])), 2880)  # SILK 60 ms (config 3)
        self.assertEqual(opus_packet_samples(bytes([0xFB, 0x03])), 2880)  # code 3: three 20 ms frames
        self.assertEqual(opus_packet_samples(bytes([0x79])), 1920)  # hybrid 20 ms, code 1: two frames
        self.assertEqual(opus_packet_samples(b""), 0)
        self.assertEqual(opus_packet_samples(bytes([0xFB, 0x3F])), 0)  # 63 frames: longer than 120 ms


class OggOpusWriterTests(unittest.TestCase):
    def test_headers_pages_and_packets_round_trip(self) -> None:
        handle = io.BytesIO()
        track = SpeakerTrack(handle, serial=77)
        for index in range(300):  # more than one page's worth of packets
            track.add(1, 1000 + index * FRAME_SAMPLES, VOICE + bytes([index % 256]), index * 0.02)
            if index == 100:
                track.flush()
        track.writer.close()
        pages = read_ogg(handle.getvalue())

        self.assertTrue(all(page["crc_ok"] for page in pages))
        self.assertEqual([page["sequence"] for page in pages], list(range(len(pages))))
        self.assertTrue(all(page["serial"] == 77 for page in pages))
        self.assertEqual(pages[0]["flags"], 0x02)  # BOS
        self.assertEqual(pages[-1]["flags"], 0x04)  # EOS
        magic, version, channels, pre_skip, rate, gain, mapping = struct.unpack("<8sBBHIhB", pages[0]["packets"][0])
        self.assertEqual((magic, version, channels, pre_skip, rate, mapping), (b"OpusHead", 1, 2, PRE_SKIP_SAMPLES, 48000, 0))
        self.assertTrue(pages[1]["packets"][0].startswith(b"OpusTags"))
        packets = audio_packets(pages)
        self.assertEqual(packets, [VOICE + bytes([index % 256]) for index in range(300)])
        self.assertEqual(pages[-1]["granule"], PRE_SKIP_SAMPLES + 300 * FRAME_SAMPLES)
        granules = [page["granule"] for page in pages]
        self.assertEqual(granules, sorted(granules))

    def test_rtp_gap_becomes_silence_frames(self) -> None:
        handle = io.BytesIO()
        track = SpeakerTrack(handle, serial=1)
        track.add(5, 1000, VOICE, clock=0.0)
        track.add(5, 1000 + 3 * 48000, VOICE, clock=3.0)  # 3 s later by RTP and by arrival
        track.writer.close()
        packets = audio_packets(read_ogg(handle.getvalue()))
        self.assertEqual(packets[0], VOICE)
        self.assertEqual(packets[1:-1], [OPUS_SILENCE] * (3 * 50 - 1))
        self.assertEqual(packets[-1], VOICE)
        self.assertEqual(track.writer.samples_written, 3 * 48000 + FRAME_SAMPLES)

    def test_unsizable_packet_keeps_time_with_silence(self) -> None:
        handle = io.BytesIO()
        track = SpeakerTrack(handle, serial=1)
        track.add(5, 0, b"", clock=0.0)
        self.assertEqual(track.writer.samples_written, FRAME_SAMPLES)

    def test_duration_from_last_page_granule(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.ogg"
            track = SpeakerTrack(path.open("wb"), serial=1)
            for index in range(100):
                track.add(1, index * FRAME_SAMPLES, VOICE, index * 0.02)
            track.close()
            self.assertAlmostEqual(ogg_duration_seconds(path), 2.0)


class TrackTimelineTests(unittest.TestCase):
    def test_trusts_rtp_within_tolerance(self) -> None:
        timeline = TrackTimeline(tolerance_seconds=2.0)
        self.assertEqual(timeline.target(1, 5000, clock=10.0, written=0), 0)
        # Arrives 0.5 s late (jitter): RTP decides the position.
        self.assertEqual(timeline.target(1, 5000 + 48000, clock=11.5, written=960), 48000)

    def test_new_ssrc_reanchors_to_arrival_time(self) -> None:
        timeline = TrackTimeline()
        timeline.target(1, 5000, clock=0.0, written=0)
        self.assertEqual(timeline.target(2, 999, clock=4.0, written=960), 4 * 48000)
        self.assertEqual(timeline.target(2, 999 + 960, clock=4.02, written=4 * 48000 + 960), 4 * 48000 + 960)

    def test_rtp_that_disagrees_with_the_clock_is_overruled(self) -> None:
        timeline = TrackTimeline(tolerance_seconds=2.0)
        timeline.target(1, 0, clock=0.0, written=0)
        # RTP claims an hour passed but only 5 s did: no hour of silence.
        self.assertEqual(timeline.target(1, 3600 * 48000, clock=5.0, written=960), 5 * 48000)
        # A client that froze its RTP clock across a 30 s pause: placed by arrival time.
        self.assertEqual(timeline.target(1, 3600 * 48000 + 960, clock=35.0, written=5 * 48000 + 960), 35 * 48000)

    def test_rtp_wraparound(self) -> None:
        timeline = TrackTimeline()
        timeline.target(1, 2**32 - 960, clock=0.0, written=0)
        self.assertEqual(timeline.target(1, 960, clock=0.04, written=960), 1920)

    def test_never_places_before_what_is_written_after_reanchor(self) -> None:
        timeline = TrackTimeline()
        timeline.target(1, 0, clock=0.0, written=0)
        self.assertEqual(timeline.target(2, 0, clock=1.0, written=5 * 48000), 5 * 48000)


class LegacyConversionTests(unittest.TestCase):
    def write_clip(self, path: Path, first_ts: int, count: int) -> None:
        with path.open("wb") as handle:
            for index in range(count):
                handle.write(struct.pack("<HIH", index, first_ts + index * FRAME_SAMPLES, len(VOICE)) + VOICE)

    def test_clips_are_placed_at_their_start_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            start = datetime(2026, 1, 1, 20, 0, 0)
            self.write_clip(root / "a.skopus", 123, 50)  # 1 s
            self.write_clip(root / "b.skopus", 999_999, 25)  # 0.5 s, 5 s after the first clip
            (root / "broken.skopus").write_bytes(b"\x01\x02")
            clips = [
                (start + timedelta(seconds=5), root / "b.skopus"),
                (start, root / "a.skopus"),
                (start + timedelta(seconds=2), root / "broken.skopus"),
            ]
            with self.assertLogs("scrollkeeper.audio", "WARNING"):
                track_start = convert_legacy_clips(clips, root / "out" / "1.ogg", serial=3)
            self.assertEqual(track_start, start)
            packets = audio_packets(read_ogg((root / "out" / "1.ogg").read_bytes()))
            self.assertEqual(packets[:50], [VOICE] * 50)
            self.assertEqual(packets[50:250], [OPUS_SILENCE] * 200)  # 1 s .. 5 s
            self.assertEqual(packets[250:], [VOICE] * 25)
            self.assertAlmostEqual(ogg_duration_seconds(root / "out" / "1.ogg"), 5.5)

    def test_nothing_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "1.ogg"
            self.assertIsNone(convert_legacy_clips([(datetime(2026, 1, 1), Path(tmp) / "missing.skopus")], out, 1))
            self.assertFalse(out.exists())


class DecodeForTranscriptionTests(unittest.TestCase):
    def test_ffmpeg_command_downmixes_and_resamples_with_soxr(self) -> None:
        with patch("scrollkeeper.audio.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            decode_for_transcription(Path("/in/1.ogg"), Path("/scratch/1.flac"))
        command = run.call_args.args[0]
        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("aresample=resampler=soxr", command)
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-c:a") + 1], "flac")
        self.assertEqual(command[-1], "/scratch/1.flac")

    def test_ffmpeg_failure_raises_with_its_message(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", "Invalid data found when processing input")
        with patch("scrollkeeper.audio.subprocess.run", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "Invalid data"):
                decode_for_transcription(Path("/in/1.ogg"), Path("/scratch/1.flac"))

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
    def test_real_ffmpeg_reads_a_track_and_a_truncated_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.ogg"
            track = SpeakerTrack(path.open("wb"), serial=9)
            for index in range(500):  # 10 s, a page per second like the recorder
                track.add(1, index * FRAME_SAMPLES, OPUS_SILENCE, index * 0.02)
                if index % 50 == 49:
                    track.flush()
            track.handle.close()  # killed: no EOS page
            data = path.read_bytes()
            cut = Path(tmp) / "cut.ogg"
            cut.write_bytes(data[: data.rfind(b"OggS", 0, data.rfind(b"OggS")) + 60])  # mid-page
            for source, expected in ((path, 10.0), (cut, 8.0)):
                flac = source.with_suffix(".flac")
                decode_for_transcription(source, flac)
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels:format=duration", "-of", "default=nw=1", str(flac)],
                    capture_output=True, text=True, check=True,
                ).stdout
                self.assertIn("sample_rate=16000", probe)
                self.assertIn("channels=1", probe)
                duration = float(probe.split("duration=")[1].split()[0])
                self.assertAlmostEqual(duration, expected, delta=0.05)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class TrackRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name) / "data")
        self.session_id = self.storage.create_session(1, 2, 3, "Test")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_writes_one_track_per_speaker_and_records_it(self) -> None:
        clock = FakeClock()
        record_dir = Path(self._tmp.name) / "spool" / str(self.session_id)
        recorder = TrackRecorder(self.storage, self.session_id, record_dir, clock=clock)
        recorder.start()
        mira, varric = Speaker(10, "mira#1", "Mira"), Speaker(20, "varric#2", "Varric")
        for index in range(50):
            clock.now = index * 0.02
            recorder.submit(mira, 111, index * FRAME_SAMPLES, VOICE)
        clock.now = 2.0
        recorder.submit(varric, 222, 7, VOICE)
        recorder.stop()

        tracks = {row["user_id"]: row for row in self.storage.get_session_tracks(self.session_id)}
        self.assertEqual(set(tracks), {10, 20})
        self.assertEqual(tracks[10]["character_name"], "Mira")
        self.assertEqual(Path(tracks[10]["path"]), record_dir / "10.ogg")
        self.assertIsNotNone(tracks[10]["ended_at"])
        self.assertEqual(audio_packets(read_ogg((record_dir / "10.ogg").read_bytes())), [VOICE] * 50)
        self.assertAlmostEqual(ogg_duration_seconds(record_dir / "20.ogg"), 0.02)

    def test_track_start_is_when_the_first_packet_arrived(self) -> None:
        clock = FakeClock()
        recorder = TrackRecorder(self.storage, self.session_id, Path(self._tmp.name) / "rec", clock=clock)
        clock.now = 100.0
        recorder.submit(Speaker(10, "mira", "Mira"), 1, 0, VOICE)  # queued before the thread runs
        clock.now = 103.0  # the writer thread gets to it 3 s later
        before = datetime.utcnow()
        recorder.start()
        recorder.stop()
        started_at = datetime.fromisoformat(self.storage.get_session_tracks(self.session_id)[0]["started_at"])
        self.assertAlmostEqual((before - started_at).total_seconds(), 3.0, delta=0.5)


class StorageMigrationTests(unittest.TestCase):
    def test_upgrades_a_pre_track_database_and_keeps_its_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            conn = sqlite3.connect(data / "scrollkeeper.db")
            conn.executescript(
                """
                CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, guild_id INTEGER NOT NULL,
                    voice_channel_id INTEGER NOT NULL, text_channel_id INTEGER NOT NULL, title TEXT,
                    started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL, transcript_path TEXT, summary_path TEXT);
                CREATE TABLE transcript_segments (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL, display_name TEXT NOT NULL, character_name TEXT NOT NULL,
                    started_at TEXT NOT NULL, ended_at TEXT NOT NULL, audio_path TEXT NOT NULL, transcript_text TEXT);
                INSERT INTO sessions VALUES (1, 5, 6, 7, 'Old', '2026-01-01T20:00:00', NULL, 'completed', NULL, NULL);
                INSERT INTO transcript_segments VALUES (1, 1, 10, 'mira', 'Mira', '2026-01-01T20:00:01',
                    '2026-01-01T20:00:02', '/old/1.skopus', 'Hello.');
                """
            )
            conn.close()
            storage = Storage(data)
            with storage.connection() as conn:
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], len(MIGRATIONS))
            row = storage.get_session_segments(1)[0]
            self.assertEqual((row["transcript_text"], row["track_id"]), ("Hello.", None))
            self.assertEqual(storage.get_session(1)["processing_attempts"], 0)

    def test_processing_queue_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            first = storage.create_session(1, 2, 3, None)
            second = storage.create_session(1, 2, 3, None)
            other_guild = storage.create_session(9, 2, 3, None)
            self.assertIsNone(storage.next_queued_session(1))
            storage.queue_session(second, PROCESS_TRANSCRIBE)
            storage.reset_session_llm_processing(first)
            storage.queue_session(other_guild)
            queued = storage.next_queued_session(1)
            self.assertEqual((queued["id"], queued["processing_kind"]), (first, PROCESS_LLM_ONLY))
            self.assertEqual(storage.begin_processing_attempt(first), 1)
            self.assertEqual(storage.begin_processing_attempt(first), 2)
            storage.set_session_status(first, "failed")
            self.assertEqual(storage.next_queued_session(1)["id"], second)

            stopped = datetime(2026, 1, 1, 21, 0, 0)
            recording = storage.create_session(1, 2, 3, None)
            self.assertEqual([r["id"] for r in storage.get_sessions_with_status("recording")], [recording])
            storage.mark_session_interrupted(recording, stopped)
            row = storage.get_session(recording)
            self.assertEqual((row["status"], row["processing_kind"], row["interrupted_at"]), ("processing", PROCESS_TRANSCRIBE, stopped.isoformat()))

    def test_reset_for_reprocessing_drops_track_utterances_only(self) -> None:
        from scrollkeeper.models import SpeakerSegment

        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            session_id = storage.create_session(1, 2, 3, None)
            at = datetime(2026, 1, 1, 20, 0, 0)
            track_id = storage.create_track(session_id, 10, "mira", "Mira", Path("/a/10.ogg"), at)
            storage.add_transcript_segment(session_id, SpeakerSegment(10, "mira", "Mira", at, at, Path("/old.skopus"), "Old."))
            storage.add_transcript_segment(session_id, SpeakerSegment(10, "mira", "Mira", at, at, Path("/a/10.ogg"), "New.", track_id))
            storage.reset_session_processing(session_id)
            rows = storage.get_session_segments(session_id)
            self.assertEqual([(r["track_id"], r["transcript_text"]) for r in rows], [(None, None)])


if __name__ == "__main__":
    unittest.main()
