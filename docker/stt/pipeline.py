"""Long-track transcription pipeline: VAD regions -> per-region decode -> words with track times.

This module has no third-party imports so it can be unit tested without sherpa-onnx, numpy or
models. Sample buffers are opaque sequences (numpy arrays in the service, lists in tests); the
only operation that differs between them, concatenation, is injected.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

SAMPLE_RATE = 16000

Samples = Any  # numpy float32 array in the service; a list of floats in tests


class Vad(Protocol):
    """The subset of sherpa_onnx.VoiceActivityDetector the pipeline uses."""

    def accept_waveform(self, samples: Samples) -> None: ...
    def empty(self) -> bool: ...
    def pop(self) -> None: ...
    def flush(self) -> None: ...

    @property
    def front(self) -> Any: ...  # has .start (absolute sample index) and .samples


class Decoded(Protocol):
    """The subset of a sherpa-onnx offline recognition result the pipeline uses."""

    text: str
    tokens: Sequence[str]
    timestamps: Sequence[float]
    durations: Sequence[float]


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Region:
    start_sample: int
    samples: Samples


# Without durations, a word ends at the next token's start, capped so a pause is not swallowed.
MAX_TOKEN_SECONDS = 0.8


def _is_word_start(token: str) -> bool:
    return token.startswith((" ", "▁"))


def group_words(
    tokens: Sequence[str],
    timestamps: Sequence[float],
    durations: Sequence[float] | None,
    offset: float,
    region_end: float | None = None,
) -> list[Word]:
    """Join sub-word tokens into words with start/end times offset into track time.

    A token beginning with a space (or SentencePiece's "▁") starts a new word; others, including
    punctuation, extend the current word. Ends come from token durations (TDT models) when
    present, otherwise from the next token's start.
    """
    if not durations or len(durations) != len(tokens):
        durations = None
    words: list[Word] = []
    text = ""
    start = 0.0
    last_index = -1

    def token_end(index: int) -> float:
        if durations is not None:
            return timestamps[index] + max(durations[index], 0.0)
        cap = timestamps[index] + MAX_TOKEN_SECONDS
        if index + 1 < len(timestamps):
            return min(timestamps[index + 1], cap)
        return cap if region_end is None else min(region_end - offset, cap)

    def close() -> None:
        word = text.strip()
        if word:
            words.append(Word(word, round(offset + start, 3), round(offset + token_end(last_index), 3)))

    for index, token in enumerate(tokens):
        if not text or _is_word_start(token):
            if text:
                close()
            text = token.replace("▁", " ")
            start = timestamps[index]
        else:
            text += token
        last_index = index
    if text:
        close()
    return words


class RegionCutter:
    """Feeds audio to the VAD in fixed windows and yields padded speech regions.

    Silero starts regions slightly late, so each region is cut from a rolling history buffer with
    `pad_samples` added on both sides (never overlapping the previous region).
    """

    def __init__(
        self,
        vad: Vad,
        concat: Callable[[list[Samples]], Samples],
        window_size: int = 512,
        pad_samples: int = SAMPLE_RATE // 4,
        keep_samples: int = SAMPLE_RATE * 60,
    ) -> None:
        self.vad = vad
        self.concat = concat
        self.window_size = window_size
        self.pad_samples = pad_samples
        self.keep_samples = keep_samples
        self._pending: list[Samples] = []  # input not yet fed to the VAD (< one window)
        self._pending_len = 0
        self._history: list[Samples] = []  # recent input, starting at sample _history_start
        self._history_start = 0
        self._history_len = 0
        self._last_end = 0
        self.samples_seen = 0

    def feed(self, chunk: Samples) -> Iterator[Region]:
        if not len(chunk):
            return
        self.samples_seen += len(chunk)
        self._remember(chunk)
        self._pending.append(chunk)
        self._pending_len += len(chunk)
        if self._pending_len < self.window_size:
            return
        buffered = self.concat(self._pending)
        usable = len(buffered) - len(buffered) % self.window_size
        for index in range(0, usable, self.window_size):
            self.vad.accept_waveform(buffered[index : index + self.window_size])
            yield from self._drain()
        rest = buffered[usable:]
        self._pending = [rest] if len(rest) else []
        self._pending_len = len(rest)

    def finish(self) -> Iterator[Region]:
        if self._pending_len:
            self.vad.accept_waveform(self.concat(self._pending))
            self._pending, self._pending_len = [], 0
        self.vad.flush()
        yield from self._drain()

    def _remember(self, chunk: Samples) -> None:
        self._history.append(chunk)
        self._history_len += len(chunk)
        while self._history and self._history_len - len(self._history[0]) >= self.keep_samples:
            dropped = self._history.pop(0)
            self._history_len -= len(dropped)
            self._history_start += len(dropped)

    def _drain(self) -> Iterator[Region]:
        while not self.vad.empty():
            segment = self.vad.front
            start, length = int(segment.start), len(segment.samples)
            self.vad.pop()
            yield self._cut(start, start + length)

    def _cut(self, start: int, end: int) -> Region:
        history_end = self._history_start + self._history_len
        cut_start = max(start - self.pad_samples, self._last_end, self._history_start)
        cut_end = min(end + self.pad_samples, history_end)
        self._last_end = cut_end
        buffered = self.concat(self._history)
        return Region(cut_start, buffered[cut_start - self._history_start : cut_end - self._history_start])


@dataclass
class Progress:
    regions: int
    audio_seconds: float
    elapsed_seconds: float


def transcribe_track(
    chunks: Iterable[Samples],
    cutter: RegionCutter,
    decode: Callable[[Samples], Decoded],
    on_progress: Callable[[Progress], None] | None = None,
    progress_every_seconds: float = 30.0,
) -> tuple[list[Segment], float]:
    """Transcribe a whole track region by region. Returns (segments, track duration in seconds)."""
    segments: list[Segment] = []
    started = last_report = time.monotonic()

    def handle(region: Region) -> None:
        nonlocal last_report
        offset = region.start_sample / SAMPLE_RATE
        region_end = offset + len(region.samples) / SAMPLE_RATE
        result = decode(region.samples)
        text = str(result.text).strip()
        if text:
            words = group_words(
                list(result.tokens),
                list(result.timestamps),
                list(getattr(result, "durations", None) or []),
                offset,
                region_end,
            )
            seg_start = words[0].start if words else round(offset, 3)
            seg_end = words[-1].end if words else round(region_end, 3)
            segments.append(Segment(len(segments), seg_start, seg_end, text, words))
        now = time.monotonic()
        if on_progress is not None and now - last_report >= progress_every_seconds:
            last_report = now
            on_progress(Progress(len(segments), cutter.samples_seen / SAMPLE_RATE, now - started))

    for chunk in chunks:
        for region in cutter.feed(chunk):
            handle(region)
    for region in cutter.finish():
        handle(region)
    return segments, cutter.samples_seen / SAMPLE_RATE


def build_response(
    segments: list[Segment],
    duration: float,
    response_format: str,
    granularities: Iterable[str],
) -> dict[str, object] | str:
    """Shape the result like OpenAI's /v1/audio/transcriptions response."""
    text = " ".join(segment.text for segment in segments).strip()
    if response_format == "text":
        return text
    if response_format != "verbose_json":
        return {"text": text}
    wanted = set(granularities) or {"segment"}
    response: dict[str, object] = {
        "task": "transcribe",
        "language": "english",
        "duration": round(duration, 3),
        "text": text,
    }
    if "segment" in wanted:
        response["segments"] = [
            {"id": seg.id, "start": seg.start, "end": seg.end, "text": seg.text} for seg in segments
        ]
    if "word" in wanted:
        response["words"] = [
            {"word": word.word, "start": word.start, "end": word.end}
            for seg in segments
            for word in seg.words
        ]
    return response
