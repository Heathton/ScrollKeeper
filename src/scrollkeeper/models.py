from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass(slots=True)
class SpeakerSegment:
    discord_user_id: int
    discord_display_name: str
    character_name: str
    started_at: datetime
    ended_at: datetime
    audio_path: Path
    transcript_text: str = ""


@dataclass(slots=True)
class SessionArtifacts:
    session_id: int
    transcript_markdown: str
    session_notes_markdown: str
    cinematic_summary_markdown: str
    note_updates: list[dict]
    transcript_path: Path
    summary_path: Path
    exported_notes_path: Path | None = None


@dataclass(slots=True)
class CampaignNote:
    guild_id: int
    note_type: str
    title: str
    content: str
    source_session_id: int | None = None
    metadata: dict = field(default_factory=dict)
