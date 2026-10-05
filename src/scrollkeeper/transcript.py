"""Transcript assembly: per-track words -> utterances -> one time-ordered transcript for all speakers."""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import TimedText, TranscriptionResult

# A pause longer than this between two words of one speaker starts a new utterance.
UTTERANCE_PAUSE_SECONDS = 0.8
# Consecutive utterances of one speaker are joined into one transcript line unless they are
# further apart than this, so long monologues still get a fresh timestamp now and then.
JOIN_MAX_GAP_SECONDS = 30.0
# When a transcript is split for the summarizer, each cut is placed at the longest pause within
# this fraction of the target part size on either side of the ideal cut.
CUT_WINDOW = 0.25


@dataclass(slots=True)
class TranscriptLine:
    speaker: str
    started_at: datetime
    text: str
    ended_at: datetime | None = None  # end of the line's last utterance; None means unknown


def split_utterances(result: TranscriptionResult, pause_seconds: float = UTTERANCE_PAUSE_SECONDS) -> list[TimedText]:
    """Cut one track's transcription into utterances at pauses between words.

    Falls back to the service's segments when it returned no word timestamps, and to the whole
    text at the start of the track when it returned neither.
    """
    if result.words:
        utterances: list[TimedText] = []
        current: list[TimedText] = []
        for word in result.words:
            if current and word.start - current[-1].end > pause_seconds:
                utterances.append(_join(current))
                current = []
            current.append(word)
        if current:
            utterances.append(_join(current))
        return utterances
    segments = [segment for segment in result.segments if segment.text.strip()]
    if segments:
        return [TimedText(segment.text.strip(), segment.start, segment.end) for segment in segments]
    text = result.text.strip()
    return [TimedText(text, 0.0, 0.0)] if text else []


def _join(words: list[TimedText]) -> TimedText:
    text = " ".join(word.text.strip() for word in words if word.text.strip())
    return TimedText(text, words[0].start, words[-1].end)


def merge_lines(rows: list[sqlite3.Row]) -> list[TranscriptLine]:
    """Order utterance rows by start time across speakers and join one speaker's consecutive lines.

    When a session has per-speaker tracks, rows from older clip-based recordings are ignored.
    """
    if any(row["track_id"] is not None for row in rows):
        rows = [row for row in rows if row["track_id"] is not None]
    items = []
    for row in rows:
        text = (row["transcript_text"] or "").strip()
        if text:
            speaker = row["character_name"] or row["display_name"]
            items.append((datetime.fromisoformat(row["started_at"]), datetime.fromisoformat(row["ended_at"]), speaker, text))
    items.sort(key=lambda item: item[0])

    lines: list[TranscriptLine] = []
    line_end = datetime.min  # end of the last line's latest utterance
    for started_at, ended_at, speaker, text in items:
        if lines and lines[-1].speaker == speaker and (started_at - line_end).total_seconds() <= JOIN_MAX_GAP_SECONDS:
            lines[-1].text = f"{lines[-1].text} {text}"
            line_end = max(line_end, ended_at)
        else:
            lines.append(TranscriptLine(speaker, started_at, text))
            line_end = ended_at
        lines[-1].ended_at = line_end
    return lines


def compact_line(line: TranscriptLine) -> str:
    return f"{line.speaker}: {line.text}"


def render_compact(lines: list[TranscriptLine], note: str = "") -> str:
    """`Speaker: text` lines without timestamps, the summarizer's view (fewer tokens)."""
    out = ["# Transcript", ""]
    if note:
        out.extend([f"_{note}_", ""])
    for line in lines:
        out.extend([compact_line(line), ""])
    return "\n".join(out).strip() + "\n"


def render_timed(lines: list[TranscriptLine], session_started_at: datetime, note: str = "") -> str:
    """`[HH:MM:SS] Speaker: text` lines, offsets from the session start, so facts can cite a time."""
    out = ["# Transcript", ""]
    if note:
        out.extend([f"_{note}_", ""])
    for line in lines:
        out.append(f"[{format_offset(line.started_at - session_started_at)}] {line.speaker}: {line.text}")
    return "\n".join(out).strip() + "\n"


def split_at_pauses(lines: list[TranscriptLine], max_chars: int) -> list[list[TranscriptLine]]:
    """Split a transcript into parts of at most `max_chars` (compact lines), cutting at pauses.

    The parts are balanced in size; each cut goes at the longest silence between lines within a
    window around the ideal cut, so a scene is less likely to be split in the middle. A single
    line longer than `max_chars` becomes a part of its own.
    """
    sizes = [len(compact_line(line)) + 2 for line in lines]  # rendered with a blank line after
    parts: list[list[TranscriptLine]] = []
    start = 0
    while start < len(lines):
        remaining = sum(sizes[start:])
        if remaining <= max_chars:
            parts.append(lines[start:])
            break
        target = remaining / math.ceil(remaining / max_chars)
        low, high = target * (1 - CUT_WINDOW), min(max_chars, target * (1 + CUT_WINDOW))
        best: tuple[float, float, int] | None = None  # (pause, -distance from target, cut index)
        size = 0
        for index in range(start, len(lines)):
            if index > start and low <= size <= high:
                candidate = (_pause_before(lines, index), -abs(size - target), index)
                best = max(best, candidate) if best is not None else candidate
            size += sizes[index]
            if size > high:
                break
        if best is not None:
            cut = best[2]
        else:  # No cut point in the window (very long lines): take what fits, at least one line.
            cut, size = start + 1, sizes[start]
            while cut < len(lines) and size + sizes[cut] <= max_chars:
                size += sizes[cut]
                cut += 1
        parts.append(lines[start:cut])
        start = cut
    return parts


def _pause_before(lines: list[TranscriptLine], index: int) -> float:
    previous = lines[index - 1]
    previous_end = previous.ended_at or previous.started_at
    return (lines[index].started_at - previous_end).total_seconds()


def format_offset(offset: timedelta) -> str:
    seconds = max(0, int(offset.total_seconds()))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
