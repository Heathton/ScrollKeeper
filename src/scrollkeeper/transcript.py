"""Transcript assembly: per-track words -> utterances -> one time-ordered transcript for all speakers."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from .models import TimedText, TranscriptionResult

# A pause longer than this between two words of one speaker starts a new utterance.
UTTERANCE_PAUSE_SECONDS = 0.8
# Consecutive utterances of one speaker are joined into one transcript line unless they are
# further apart than this, so long monologues still get a fresh timestamp now and then.
JOIN_MAX_GAP_SECONDS = 30.0


@dataclass(slots=True)
class TranscriptLine:
    speaker: str
    started_at: datetime
    text: str


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
    return lines


def render_compact(lines: list[TranscriptLine], note: str = "") -> str:
    """`Speaker: text` lines without timestamps, the summarizer's view (fewer tokens)."""
    out = ["# Transcript", ""]
    if note:
        out.extend([f"_{note}_", ""])
    for line in lines:
        out.extend([f"{line.speaker}: {line.text}", ""])
    return "\n".join(out).strip() + "\n"


def render_timed(lines: list[TranscriptLine], session_started_at: datetime, note: str = "") -> str:
    """`[HH:MM:SS] Speaker: text` lines, offsets from the session start, so facts can cite a time."""
    out = ["# Transcript", ""]
    if note:
        out.extend([f"_{note}_", ""])
    for line in lines:
        out.append(f"[{format_offset(line.started_at - session_started_at)}] {line.speaker}: {line.text}")
    return "\n".join(out).strip() + "\n"


def format_offset(offset: timedelta) -> str:
    seconds = max(0, int(offset.total_seconds()))
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"
