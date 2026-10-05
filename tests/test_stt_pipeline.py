from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docker" / "stt"))

from pipeline import (  # noqa: E402
    SAMPLE_RATE,
    RegionCutter,
    build_response,
    group_words,
    transcribe_track,
)

WINDOW = 4


def concat(parts: list[list[float]]) -> list[float]:
    return [sample for part in parts for sample in part]


@dataclass
class FakeVad:
    """Marks samples equal to 1.0 as speech; emits a segment after `min_silence` silent windows,
    like sherpa-onnx's VoiceActivityDetector (absolute start index, samples of the speech)."""

    min_silence: int = 2
    window_sizes: list[int] = field(default_factory=list)
    _queue: list[SimpleNamespace] = field(default_factory=list)
    _position: int = 0
    _speech_start: int | None = None
    _speech: list[float] = field(default_factory=list)
    _silent_windows: int = 0

    def accept_waveform(self, samples: list[float]) -> None:
        self.window_sizes.append(len(samples))
        if any(sample == 1.0 for sample in samples):
            if self._speech_start is None:
                self._speech_start = self._position
            self._speech.extend(samples)
            self._silent_windows = 0
        elif self._speech_start is not None:
            self._silent_windows += 1
            if self._silent_windows >= self.min_silence:
                self._emit()
        self._position += len(samples)

    def _emit(self) -> None:
        self._queue.append(SimpleNamespace(start=self._speech_start, samples=self._speech))
        self._speech_start, self._speech, self._silent_windows = None, [], 0

    def flush(self) -> None:
        if self._speech_start is not None:
            self._emit()

    def empty(self) -> bool:
        return not self._queue

    @property
    def front(self) -> SimpleNamespace:
        return self._queue[0]

    def pop(self) -> None:
        self._queue.pop(0)


def chunked(samples: list[float], size: int) -> list[list[float]]:
    return [samples[i : i + size] for i in range(0, len(samples), size)]


class GroupWordsTests(unittest.TestCase):
    def test_joins_subword_tokens_and_attaches_punctuation(self) -> None:
        tokens = [" Well", ",", " I", " don", "'", "t", " w", "ish"]
        stamps = [0.40, 0.64, 0.72, 0.88, 0.96, 1.04, 1.04, 1.12]
        durations = [0.24, 0.08, 0.16, 0.08, 0.08, 0.0, 0.08, 0.16]
        words = group_words(tokens, stamps, durations, offset=10.0)
        self.assertEqual([w.word for w in words], ["Well,", "I", "don't", "wish"])
        self.assertEqual((words[0].start, words[0].end), (10.4, 10.72))
        self.assertEqual((words[2].start, words[2].end), (10.88, 11.04))
        self.assertEqual((words[3].start, words[3].end), (11.04, 11.28))

    def test_sentencepiece_marker_and_missing_durations(self) -> None:
        words = group_words(["▁Str", "ahd", "▁left"], [0.0, 0.2, 0.4], [], offset=1.0, region_end=3.0)
        self.assertEqual([w.word for w in words], ["Strahd", "left"])
        self.assertEqual((words[0].start, words[0].end), (1.0, 1.4))  # ends at the next token's start
        self.assertEqual(words[1].end, 2.2)  # last token capped at MAX_TOKEN_SECONDS

    def test_empty(self) -> None:
        self.assertEqual(group_words([], [], [], offset=0.0), [])


class RegionCutterTests(unittest.TestCase):
    def test_feeds_fixed_windows_and_pads_regions_without_overlap(self) -> None:
        # 8 silent, 4 speech, 8 silent, 4 speech, 8 silent samples.
        audio = [0.0] * 8 + [1.0] * 4 + [0.0] * 8 + [1.0] * 4 + [0.0] * 8
        vad = FakeVad()
        cutter = RegionCutter(vad, concat, window_size=WINDOW, pad_samples=6, keep_samples=100)
        regions = [r for chunk in chunked(audio, 3) for r in cutter.feed(chunk)] + list(cutter.finish())
        self.assertTrue(all(size == WINDOW for size in vad.window_sizes))
        self.assertEqual([(r.start_sample, len(r.samples)) for r in regions], [(2, 16), (18, 12)])
        self.assertEqual(regions[0].samples, audio[2:18])
        self.assertEqual(cutter.samples_seen, len(audio))

    def test_flushes_trailing_speech_and_partial_window(self) -> None:
        audio = [0.0] * 4 + [1.0] * 6
        cutter = RegionCutter(FakeVad(), concat, window_size=WINDOW, pad_samples=0, keep_samples=100)
        regions = [r for r in cutter.feed(audio)] + list(cutter.finish())
        self.assertEqual([(r.start_sample, r.samples) for r in regions], [(4, [1.0] * 6)])

    def test_history_is_bounded(self) -> None:
        cutter = RegionCutter(FakeVad(), concat, window_size=WINDOW, pad_samples=0, keep_samples=8)
        for chunk in chunked([0.0] * 100, 4):
            list(cutter.feed(chunk))
        self.assertLessEqual(cutter._history_len, 12)
        self.assertEqual(cutter._history_start + cutter._history_len, 100)


class TranscribeTrackTests(unittest.TestCase):
    def test_offsets_words_to_track_time_and_skips_empty_regions(self) -> None:
        second = SAMPLE_RATE
        audio = [0.0] * (2 * second) + [1.0] * second + [0.0] * second + [1.0] * second + [0.0] * second
        decoded = iter(
            [
                SimpleNamespace(text=" Hello there", tokens=[" Hello", " there"], timestamps=[0.1, 0.5], durations=[0.3, 0.3]),
                SimpleNamespace(text="", tokens=[], timestamps=[], durations=[]),
            ]
        )
        cutter = RegionCutter(FakeVad(min_silence=1), concat, window_size=400, pad_samples=0)
        segments, duration = transcribe_track(chunked(audio, 3000), cutter, lambda samples: next(decoded))
        self.assertEqual(duration, 6.0)
        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual((segment.text, segment.start, segment.end), ("Hello there", 2.1, 2.8))
        self.assertEqual([(w.word, w.start, w.end) for w in segment.words], [("Hello", 2.1, 2.4), ("there", 2.5, 2.8)])


class BuildResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        cutter = RegionCutter(FakeVad(), concat, window_size=WINDOW, pad_samples=0)
        audio = [0.0] * 4 + [1.0] * 4 + [0.0] * 8 + [1.0] * 4 + [0.0] * 8
        results = iter(
            [
                SimpleNamespace(text="Roll", tokens=[" Roll"], timestamps=[0.0], durations=[0.1]),
                SimpleNamespace(text="initiative.", tokens=[" initiative", "."], timestamps=[0.0, 0.2], durations=[0.2, 0.0]),
            ]
        )
        self.segments, self.duration = transcribe_track([audio], cutter, lambda samples: next(results))

    def test_verbose_json_with_words(self) -> None:
        response = build_response(self.segments, self.duration, "verbose_json", ["word"])
        self.assertEqual(response["text"], "Roll initiative.")
        self.assertEqual([w["word"] for w in response["words"]], ["Roll", "initiative."])
        self.assertNotIn("segments", response)

    def test_verbose_json_defaults_to_segments(self) -> None:
        response = build_response(self.segments, self.duration, "verbose_json", [])
        self.assertEqual([s["text"] for s in response["segments"]], ["Roll", "initiative."])
        self.assertNotIn("words", response)

    def test_json_and_text(self) -> None:
        self.assertEqual(build_response(self.segments, self.duration, "json", ["word"]), {"text": "Roll initiative."})
        self.assertEqual(build_response(self.segments, self.duration, "text", []), "Roll initiative.")


if __name__ == "__main__":
    unittest.main()
