from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import CampaignNote, SpeakerSegment


class Storage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.notes_dir = self.data_dir / "notes"
        self.sessions_dir = self.data_dir / "sessions"
        self.db_path = self.data_dir / "scrollkeeper.db"
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS character_registry (
                    guild_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    character_name TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (guild_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    voice_channel_id INTEGER NOT NULL,
                    text_channel_id INTEGER NOT NULL,
                    title TEXT,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    status TEXT NOT NULL,
                    transcript_path TEXT,
                    summary_path TEXT
                );

                CREATE TABLE IF NOT EXISTS transcript_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    user_id INTEGER NOT NULL,
                    display_name TEXT NOT NULL,
                    character_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT NOT NULL,
                    audio_path TEXT NOT NULL,
                    transcript_text TEXT
                );

                CREATE TABLE IF NOT EXISTS campaign_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    guild_id INTEGER NOT NULL,
                    note_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_session_id INTEGER,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS note_embeddings (
                    note_id INTEGER PRIMARY KEY REFERENCES campaign_notes(id) ON DELETE CASCADE,
                    embedding_json TEXT NOT NULL
                );
                """
            )

    def register_character(self, guild_id: int, user_id: int, character_name: str) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO character_registry (guild_id, user_id, character_name, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(guild_id, user_id) DO UPDATE SET
                    character_name = excluded.character_name,
                    updated_at = excluded.updated_at
                """,
                (guild_id, user_id, character_name, now),
            )

    def get_character_name(self, guild_id: int, user_id: int, fallback_name: str) -> str:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT character_name
                FROM character_registry
                WHERE guild_id = ? AND user_id = ?
                """,
                (guild_id, user_id),
            ).fetchone()
        return row["character_name"] if row else fallback_name

    def create_session(
        self,
        guild_id: int,
        voice_channel_id: int,
        text_channel_id: int,
        title: str | None,
    ) -> int:
        started_at = datetime.utcnow().isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (
                    guild_id, voice_channel_id, text_channel_id, title,
                    started_at, status
                )
                VALUES (?, ?, ?, ?, ?, 'recording')
                """,
                (guild_id, voice_channel_id, text_channel_id, title, started_at),
            )
            return int(cursor.lastrowid)

    def set_session_status(self, session_id: int, status: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE sessions SET status = ? WHERE id = ?",
                (status, session_id),
            )

    def finalize_session(
        self,
        session_id: int,
        transcript_path: str,
        summary_path: str,
    ) -> None:
        ended_at = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = 'completed',
                    ended_at = ?,
                    transcript_path = ?,
                    summary_path = ?
                WHERE id = ?
                """,
                (ended_at, transcript_path, summary_path, session_id),
            )

    def add_transcript_segment(self, session_id: int, segment: SpeakerSegment) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO transcript_segments (
                    session_id, user_id, display_name, character_name,
                    started_at, ended_at, audio_path, transcript_text
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    segment.discord_user_id,
                    segment.discord_display_name,
                    segment.character_name,
                    segment.started_at.isoformat(),
                    segment.ended_at.isoformat(),
                    str(segment.audio_path),
                    segment.transcript_text,
                ),
            )

    def update_segment_transcript(self, session_id: int, audio_path: str, transcript_text: str) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE transcript_segments
                SET transcript_text = ?
                WHERE session_id = ? AND audio_path = ?
                """,
                (transcript_text, session_id, audio_path),
            )

    def get_session_segments(self, session_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM transcript_segments
                WHERE session_id = ?
                ORDER BY started_at ASC
                """,
                (session_id,),
            ).fetchall()
        return list(rows)

    def upsert_campaign_note(self, note: CampaignNote, embedding: list[float]) -> None:
        now = datetime.utcnow().isoformat()
        metadata_json = json.dumps(note.metadata, ensure_ascii=True)
        embedding_json = json.dumps(embedding, ensure_ascii=True)
        with self.connection() as conn:
            existing = conn.execute(
                """
                SELECT id
                FROM campaign_notes
                WHERE guild_id = ? AND note_type = ? AND title = ?
                """,
                (note.guild_id, note.note_type, note.title),
            ).fetchone()
            if existing:
                note_id = int(existing["id"])
                conn.execute(
                    """
                    UPDATE campaign_notes
                    SET content = ?, source_session_id = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (note.content, note.source_session_id, metadata_json, now, note_id),
                )
            else:
                cursor = conn.execute(
                    """
                    INSERT INTO campaign_notes (
                        guild_id, note_type, title, content, source_session_id,
                        metadata_json, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        note.guild_id,
                        note.note_type,
                        note.title,
                        note.content,
                        note.source_session_id,
                        metadata_json,
                        now,
                        now,
                    ),
                )
                note_id = int(cursor.lastrowid)
            conn.execute(
                """
                INSERT INTO note_embeddings (note_id, embedding_json)
                VALUES (?, ?)
                ON CONFLICT(note_id) DO UPDATE SET embedding_json = excluded.embedding_json
                """,
                (note_id, embedding_json),
            )

    def get_recent_notes(self, guild_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM campaign_notes
                WHERE guild_id = ?
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 200
                """,
                (guild_id,),
            ).fetchall()
        return list(rows)

    def semantic_search_notes(
        self,
        guild_id: int,
        query_embedding: list[float],
        limit: int = 8,
    ) -> list[sqlite3.Row]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT cn.*, ne.embedding_json
                FROM campaign_notes cn
                JOIN note_embeddings ne ON ne.note_id = cn.id
                WHERE cn.guild_id = ?
                """,
                (guild_id,),
            ).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            score = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b, strict=False))
    denom_a = math.sqrt(sum(x * x for x in a))
    denom_b = math.sqrt(sum(y * y for y in b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)
