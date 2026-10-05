from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from scrollkeeper.models import SpeakerSegment, TimedText, TranscriptionResult
from scrollkeeper.storage import Storage
from scrollkeeper.transcript import merge_lines, render_compact, render_timed, split_utterances


def words(*items: tuple[str, float, float]) -> list[TimedText]:
    return [TimedText(text, start, end) for text, start, end in items]


class SplitUtterancesTests(unittest.TestCase):
    def test_splits_at_pauses_longer_than_the_threshold(self) -> None:
        result = TranscriptionResult(
            text="Yes. I attack the goblin. Nat 20!",
            words=words(
                ("Yes.", 1.0, 1.3),
                ("I", 3.0, 3.1), ("attack", 3.2, 3.5), ("the", 3.9, 4.0), ("goblin.", 4.1, 4.6),
                ("Nat", 5.5, 5.7), ("20!", 5.8, 6.1),
            ),
        )
        utterances = split_utterances(result, pause_seconds=0.8)
        self.assertEqual(
            [(u.text, u.start, u.end) for u in utterances],
            [("Yes.", 1.0, 1.3), ("I attack the goblin.", 3.0, 4.6), ("Nat 20!", 5.5, 6.1)],
        )

    def test_short_utterances_are_kept(self) -> None:
        result = TranscriptionResult(text="No.", words=words(("No.", 12.0, 12.2)))
        self.assertEqual([u.text for u in split_utterances(result)], ["No."])

    def test_falls_back_to_segments_then_text(self) -> None:
        with_segments = TranscriptionResult(text="A B", segments=words((" A ", 1.0, 2.0), ("", 3.0, 4.0), ("B", 5.0, 6.0)))
        self.assertEqual([(u.text, u.start) for u in split_utterances(with_segments)], [("A", 1.0), ("B", 5.0)])
        self.assertEqual([(u.text, u.start) for u in split_utterances(TranscriptionResult(text=" Hi "))], [("Hi", 0.0)])
        self.assertEqual(split_utterances(TranscriptionResult(text="")), [])


class MergeLinesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))
        self.session_id = self.storage.create_session(1, 2, 3, None)
        self.start = datetime(2026, 1, 1, 20, 0, 0)
        self.tracks = {
            name: self.storage.create_track(self.session_id, user_id, name.lower(), name, Path(f"/a/{user_id}.ogg"), self.start)
            for user_id, name in ((10, "Mira"), (20, "Varric"))
        }

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def add(self, name: str, start: float, end: float, text: str, track: bool = True) -> None:
        self.storage.add_transcript_segment(
            self.session_id,
            SpeakerSegment(
                10 if name == "Mira" else 20, name.lower(), name,
                self.start + timedelta(seconds=start), self.start + timedelta(seconds=end),
                Path("/a.ogg"), text, self.tracks[name] if track else None,
            ),
        )

    def test_orders_speakers_by_time_and_joins_consecutive_lines(self) -> None:
        # Inserted per track (as processing does), not in time order.
        self.add("Mira", 1.0, 2.0, "Who goes there?")
        self.add("Mira", 2.9, 3.5, "Show yourself.")
        self.add("Mira", 10.0, 11.0, "Fine.")
        self.add("Varric", 4.0, 5.0, "A friend.")
        self.add("Varric", 5.5, 6.0, "Mostly.")
        self.add("Varric", 70.0, 71.0, "Still here.")
        self.add("Varric", 120.0, 121.0, "Hello?")  # same speaker, but over 30 s later
        lines = merge_lines(self.storage.get_session_segments(self.session_id))
        self.assertEqual(
            [(line.speaker, (line.started_at - self.start).total_seconds(), line.text) for line in lines],
            [
                ("Mira", 1.0, "Who goes there? Show yourself."),
                ("Varric", 4.0, "A friend. Mostly."),
                ("Mira", 10.0, "Fine."),
                ("Varric", 70.0, "Still here."),
                ("Varric", 120.0, "Hello?"),
            ],
        )
        self.assertEqual(
            render_timed(lines[:2], self.start),
            "# Transcript\n\n[00:00:01] Mira: Who goes there? Show yourself.\n[00:00:04] Varric: A friend. Mostly.\n",
        )
        self.assertEqual(
            render_compact(lines[:1], note="Interrupted."),
            "# Transcript\n\n_Interrupted._\n\nMira: Who goes there? Show yourself.\n",
        )

    def test_legacy_clip_rows_are_ignored_once_tracks_are_transcribed(self) -> None:
        self.add("Mira", 1.0, 2.0, "Old clip text.", track=False)
        self.assertEqual([line.text for line in merge_lines(self.storage.get_session_segments(self.session_id))], ["Old clip text."])
        self.add("Mira", 1.0, 2.0, "New track text.")
        self.assertEqual([line.text for line in merge_lines(self.storage.get_session_segments(self.session_id))], ["New track text."])


if __name__ == "__main__":
    unittest.main()
