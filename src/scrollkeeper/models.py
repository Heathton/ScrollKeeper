from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


ENTITY_TYPES = ("Character", "Faction", "Location", "Item", "Mystery", "PointOfInterest", "Quest")
FACT_KINDS = ("observed", "pinned", "imported")
QUEST_STATUSES = ("offered", "active", "completed", "failed", "abandoned")


@dataclass(slots=True)
class SpeakerSegment:
    discord_user_id: int
    discord_display_name: str
    character_name: str
    started_at: datetime
    ended_at: datetime
    audio_path: Path
    transcript_text: str = ""
    track_id: int | None = None


@dataclass(slots=True)
class TimedText:
    """A word or an utterance, with start/end in seconds from the start of its audio track."""

    text: str
    start: float
    end: float


@dataclass(slots=True)
class TranscriptionResult:
    """What the speech-to-text service returned for one track."""

    text: str
    words: list[TimedText] = field(default_factory=list)
    segments: list[TimedText] = field(default_factory=list)


@dataclass(slots=True)
class Campaign:
    id: int
    guild_id: int
    name: str
    is_active: bool = False


@dataclass(slots=True)
class Entity:
    id: int
    campaign_id: int
    type: str
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    short_description: str = ""
    merged_into: int | None = None
    status: str = ""  # Quests only: offered, active, completed, failed or abandoned.

    def names(self) -> list[str]:
        return [self.canonical_name, *self.aliases]


@dataclass(slots=True)
class Fact:
    id: int
    campaign_id: int
    entity_id: int
    kind: str
    text: str
    session_id: int | None = None
    transcript_ts: str | None = None
    source_ref: str | None = None
    created_at: str = ""
    superseded_by: int | None = None
    retracted_at: str | None = None

    @property
    def active(self) -> bool:
        return self.superseded_by is None and self.retracted_at is None

    def source_label(self) -> str:
        """Short human-readable provenance, e.g. `session 12 @ 01:23:45`, `pinned`, `journal:abc`."""
        if self.kind == "pinned":
            return "pinned"
        if self.session_id is not None:
            label = f"session {self.session_id}"
            if self.transcript_ts:
                label += f" @ {self.transcript_ts}"
            return label
        if self.source_ref:
            return self.source_ref
        return self.kind


@dataclass(slots=True)
class Page:
    entity_id: int
    markdown: str
    source_fact_ids: list[int]
    updated_at: str


@dataclass(slots=True)
class SearchDoc:
    """One searchable document: a wiki page (`ref_id` = entity id), a session summary or a
    transcript chunk (`ref_id` = session id; `part` numbers a session's chunks)."""

    kind: str
    ref_id: int
    title: str
    body: str
    part: int = 0
    session_id: int | None = None
    start_ts: str | None = None
    id: int = 0
    campaign_id: int = 0


@dataclass(slots=True)
class DuplicateCandidate:
    first: Entity
    second: Entity
    reason: str


@dataclass(slots=True)
class WikiChangeReport:
    new_entities: list[Entity] = field(default_factory=list)
    updated_entities: list[Entity] = field(default_factory=list)
    facts_added: int = 0
    facts_retracted: int = 0
    possible_duplicates: list[DuplicateCandidate] = field(default_factory=list)
    page_failures: list[str] = field(default_factory=list)
    pages_without_embedding: int = 0
    renamed: list[tuple[str, str]] = field(default_factory=list)
    error: str | None = None


@dataclass(slots=True)
class SessionArtifacts:
    session_id: int
    transcript_markdown: str
    session_notes_markdown: str
    cinematic_summary_markdown: str
    transcript_path: Path
    summary_path: Path
    wiki_report: WikiChangeReport | None = None


_ARTICLE_PREFIX = re.compile(r"^(the|a|an)\s+")
_APOSTROPHES = re.compile(r"['\u2019]")
_NON_WORD = re.compile(r"[^\w\s]")


def normalize_name(name: str) -> str:
    """Lower-case, drop punctuation and a leading article, collapse whitespace (for alias matching).

    Apostrophes are removed rather than split on, so "Varric's Ledger" becomes "varrics ledger" and
    does not contain the name "varric".
    """
    text = _NON_WORD.sub(" ", _APOSTROPHES.sub("", name.casefold()))
    text = " ".join(text.split())
    return _ARTICLE_PREFIX.sub("", text)
