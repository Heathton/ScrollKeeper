from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import FACT_KINDS, Campaign, Entity, Fact, Page, SearchDoc, SpeakerSegment, normalize_name, session_date


# Forward-only schema migrations, applied in order; `PRAGMA user_version` records how many ran.
# The base tables (characters, sessions, transcript segments) are created in `_init_db` with
# IF NOT EXISTS and count as version 0. Migration 4 rebuilds `character_registry` per campaign,
# so the IF NOT EXISTS there is a no-op on any database past version 3.
MIGRATIONS: list[str] = [
    # 1: campaign wiki (entities, aliases, append-only facts, pages rebuilt from facts).
    """
    CREATE TABLE entities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        guild_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        canonical_name TEXT NOT NULL,
        name_norm TEXT NOT NULL,
        short_description TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL DEFAULT '',
        created_session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
        merged_into INTEGER REFERENCES entities(id),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX entities_guild_name ON entities(guild_id, name_norm);

    CREATE TABLE entity_aliases (
        entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
        alias TEXT NOT NULL,
        alias_norm TEXT NOT NULL,
        PRIMARY KEY (entity_id, alias_norm)
    );
    CREATE INDEX entity_aliases_norm ON entity_aliases(alias_norm);

    CREATE TABLE facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        guild_id INTEGER NOT NULL,
        entity_id INTEGER NOT NULL REFERENCES entities(id),
        session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
        transcript_ts TEXT,
        source_ref TEXT,
        text TEXT NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('observed', 'pinned', 'imported')),
        created_at TEXT NOT NULL,
        created_by_user_id INTEGER,
        superseded_by INTEGER REFERENCES facts(id),
        retracted_at TEXT,
        retraction_reason TEXT
    );
    CREATE INDEX facts_entity ON facts(entity_id);
    CREATE INDEX facts_session ON facts(guild_id, session_id);
    CREATE INDEX facts_source_ref ON facts(guild_id, source_ref);

    CREATE TABLE pages (
        entity_id INTEGER PRIMARY KEY REFERENCES entities(id) ON DELETE CASCADE,
        markdown TEXT NOT NULL,
        source_fact_ids_json TEXT NOT NULL,
        embedding_json TEXT,
        updated_at TEXT NOT NULL
    );
    """,
    # 2: per-speaker Ogg Opus tracks; utterance rows point at their track; the processing queue
    # lives on `sessions` (status 'processing' + kind) so it survives restarts.
    """
    CREATE TABLE audio_tracks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL,
        display_name TEXT NOT NULL,
        character_name TEXT NOT NULL,
        path TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT,
        UNIQUE (session_id, user_id)
    );

    ALTER TABLE transcript_segments ADD COLUMN track_id INTEGER REFERENCES audio_tracks(id) ON DELETE CASCADE;
    CREATE INDEX transcript_segments_session ON transcript_segments(session_id);

    ALTER TABLE sessions ADD COLUMN processing_kind TEXT;
    ALTER TABLE sessions ADD COLUMN processing_attempts INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE sessions ADD COLUMN interrupted_at TEXT;
    ALTER TABLE sessions ADD COLUMN audio_deleted_at TEXT;
    """,
    # 3: hybrid retrieval. Searchable documents (wiki pages, session summaries, transcript
    # chunks) are derived from the tables above by `SearchIndex.refresh`, with an FTS5 keyword
    # index and an embedding that records the model and dimension it was made with. Page
    # embeddings from the old HTTP endpoint are dropped: they came from another model.
    """
    CREATE TABLE search_docs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        guild_id INTEGER NOT NULL,
        kind TEXT NOT NULL CHECK (kind IN ('page', 'summary', 'transcript')),
        ref_id INTEGER NOT NULL,
        part INTEGER NOT NULL DEFAULT 0,
        session_id INTEGER,
        start_ts TEXT,
        title TEXT NOT NULL,
        body TEXT NOT NULL,
        version TEXT NOT NULL,
        embedding BLOB,
        embed_model TEXT,
        embed_dim INTEGER,
        UNIQUE (guild_id, kind, ref_id, part)
    );

    CREATE VIRTUAL TABLE search_fts USING fts5(
        title, body, content='search_docs', content_rowid='id',
        tokenize='porter unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER search_docs_ai AFTER INSERT ON search_docs BEGIN
        INSERT INTO search_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
    END;
    CREATE TRIGGER search_docs_ad AFTER DELETE ON search_docs BEGIN
        INSERT INTO search_fts(search_fts, rowid, title, body) VALUES ('delete', old.id, old.title, old.body);
    END;
    CREATE TRIGGER search_docs_au AFTER UPDATE OF title, body ON search_docs BEGIN
        INSERT INTO search_fts(search_fts, rowid, title, body) VALUES ('delete', old.id, old.title, old.body);
        INSERT INTO search_fts(rowid, title, body) VALUES (new.id, new.title, new.body);
    END;

    ALTER TABLE pages DROP COLUMN embedding_json;
    """,
    # 4: campaigns. A server has several campaigns and one active one. Sessions belong to a
    # campaign; characters, the wiki and the search index are kept per campaign (the wiki and
    # search tables' guild_id column becomes campaign_id). Existing data moves into a "Default"
    # campaign per server.
    """
    CREATE TABLE campaigns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        guild_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        name_norm TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        UNIQUE (guild_id, name_norm)
    );
    INSERT INTO campaigns (guild_id, name, name_norm, is_active, created_at)
    SELECT guild_id, 'Default', 'default', 1, strftime('%Y-%m-%dT%H:%M:%f', 'now')
    FROM (
        SELECT guild_id FROM sessions
        UNION SELECT guild_id FROM character_registry
        UNION SELECT guild_id FROM entities
        UNION SELECT guild_id FROM search_docs
    );

    ALTER TABLE sessions ADD COLUMN campaign_id INTEGER REFERENCES campaigns(id);
    UPDATE sessions SET campaign_id = (SELECT c.id FROM campaigns c WHERE c.guild_id = sessions.guild_id);
    CREATE INDEX sessions_campaign ON sessions(campaign_id);

    CREATE TABLE character_registry_v4 (
        campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL,
        character_name TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (campaign_id, user_id)
    );
    INSERT INTO character_registry_v4 (campaign_id, user_id, character_name, updated_at)
    SELECT c.id, r.user_id, r.character_name, r.updated_at
    FROM character_registry r JOIN campaigns c ON c.guild_id = r.guild_id;
    DROP TABLE character_registry;
    ALTER TABLE character_registry_v4 RENAME TO character_registry;

    DROP INDEX entities_guild_name;
    ALTER TABLE entities RENAME COLUMN guild_id TO campaign_id;
    UPDATE entities SET campaign_id = (SELECT c.id FROM campaigns c WHERE c.guild_id = entities.campaign_id);
    CREATE INDEX entities_campaign_name ON entities(campaign_id, name_norm);

    DROP INDEX facts_session;
    DROP INDEX facts_source_ref;
    ALTER TABLE facts RENAME COLUMN guild_id TO campaign_id;
    UPDATE facts SET campaign_id = (SELECT c.id FROM campaigns c WHERE c.guild_id = facts.campaign_id);
    CREATE INDEX facts_session ON facts(campaign_id, session_id);
    CREATE INDEX facts_source_ref ON facts(campaign_id, source_ref);

    ALTER TABLE search_docs RENAME COLUMN guild_id TO campaign_id;
    UPDATE search_docs SET campaign_id = (SELECT c.id FROM campaigns c WHERE c.guild_id = search_docs.campaign_id);
    """,
    # 5: VTT journal import (#15). Dated session recaps become sessions with a summary and no
    # audio (`journal_id` set). `journal_entries` maps each imported journal entry to what it
    # became, with a hash of its content, so a re-import updates instead of duplicating.
    # `journal_imports` is the import job queue, so an import resumes after a restart.
    """
    ALTER TABLE sessions ADD COLUMN journal_id TEXT;

    CREATE TABLE journal_entries (
        campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
        journal_id TEXT NOT NULL,
        name TEXT NOT NULL,
        folder TEXT NOT NULL,
        action TEXT NOT NULL,
        entity_id INTEGER REFERENCES entities(id),
        session_id INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
        content_hash TEXT NOT NULL,
        imported_at TEXT NOT NULL,
        PRIMARY KEY (campaign_id, journal_id)
    );

    CREATE TABLE journal_imports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        campaign_id INTEGER NOT NULL REFERENCES campaigns(id) ON DELETE CASCADE,
        guild_id INTEGER NOT NULL,
        text_channel_id INTEGER NOT NULL,
        path TEXT NOT NULL,
        status TEXT NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        finished_at TEXT
    );
    """,
    # 6: per-campaign session numbers (#24). Sessions are numbered 1, 2, 3... within their
    # campaign, by date; the id stays the storage key (folders, fact sources).
    """
    ALTER TABLE sessions ADD COLUMN number INTEGER;
    UPDATE sessions SET number = (
        SELECT COUNT(*) FROM sessions s
        WHERE s.campaign_id IS sessions.campaign_id
          AND (s.started_at < sessions.started_at OR (s.started_at = sessions.started_at AND s.id <= sessions.id))
    );
    CREATE UNIQUE INDEX sessions_campaign_number ON sessions(campaign_id, number);
    """,
]

# A session's number: the next one in its campaign.
_NEXT_SESSION_NUMBER = "(SELECT COALESCE(MAX(number), 0) + 1 FROM sessions WHERE campaign_id = ?)"
# Facts with the number and start of the session they come from (for citations, see `Fact`).
_FACTS_WITH_SESSION = """
    SELECT f.*, s.number AS session_number, s.started_at AS session_started_at, s.journal_id AS session_journal_id
    FROM facts f LEFT JOIN sessions s ON s.id = f.session_id
"""

DEFAULT_CAMPAIGN = "Default"

# Values of sessions.processing_kind: run speech-to-text then the LLM steps, or the LLM steps only.
PROCESS_TRANSCRIBE = "transcribe"
PROCESS_LLM_ONLY = "llm"


class Storage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.sessions_dir = self.data_dir / "sessions"
        self.db_path = self.data_dir / "scrollkeeper.db"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
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

                """
            )
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            for target_version, script in enumerate(MIGRATIONS[version:], start=version + 1):
                conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {target_version};\nCOMMIT;")

    # --- Campaigns: several per server, one active ------------------------------------------

    def active_campaign(self, guild_id: int) -> Campaign:
        """The server's active campaign, creating the "Default" campaign on first use."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM campaigns WHERE guild_id = ? AND is_active = 1", (guild_id,)
            ).fetchone()
            if row is None:
                row = self._activate_campaign(conn, guild_id, DEFAULT_CAMPAIGN)
        return _campaign_from_row(row)

    def switch_campaign(self, guild_id: int, name: str) -> tuple[Campaign, bool]:
        """Make `name` the server's active campaign, creating it if needed. Returns (campaign, created)."""
        name = " ".join(name.split())
        if not normalize_name(name):
            raise ValueError("A campaign needs a name.")
        with self.connection() as conn:
            existed = self._find_campaign(conn, guild_id, name) is not None
            row = self._activate_campaign(conn, guild_id, name)
        return _campaign_from_row(row), not existed

    def list_campaigns(self, guild_id: int) -> list[Campaign]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM campaigns WHERE guild_id = ? ORDER BY name COLLATE NOCASE", (guild_id,)
            ).fetchall()
        return [_campaign_from_row(row) for row in rows]

    def get_campaign(self, campaign_id: int) -> Campaign | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
        return _campaign_from_row(row) if row else None

    def _find_campaign(self, conn: sqlite3.Connection, guild_id: int, name: str) -> sqlite3.Row | None:
        return conn.execute(
            "SELECT * FROM campaigns WHERE guild_id = ? AND name_norm = ?", (guild_id, normalize_name(name))
        ).fetchone()

    def _activate_campaign(self, conn: sqlite3.Connection, guild_id: int, name: str) -> sqlite3.Row:
        conn.execute(
            """
            INSERT OR IGNORE INTO campaigns (guild_id, name, name_norm, is_active, created_at)
            VALUES (?, ?, ?, 0, ?)
            """,
            (guild_id, name, normalize_name(name), datetime.utcnow().isoformat()),
        )
        row = self._find_campaign(conn, guild_id, name)
        conn.execute(
            "UPDATE campaigns SET is_active = (id = ?) WHERE guild_id = ?", (int(row["id"]), guild_id)
        )
        return self._find_campaign(conn, guild_id, name)

    # --- Characters (per campaign) ----------------------------------------------------------

    def register_character(self, campaign_id: int, user_id: int, character_name: str) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO character_registry (campaign_id, user_id, character_name, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(campaign_id, user_id) DO UPDATE SET
                    character_name = excluded.character_name,
                    updated_at = excluded.updated_at
                """,
                (campaign_id, user_id, character_name, now),
            )

    def list_registered_characters(self, campaign_id: int) -> list[str]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT character_name FROM character_registry WHERE campaign_id = ? ORDER BY character_name",
                (campaign_id,),
            ).fetchall()
        return [row["character_name"] for row in rows]

    def get_character_name(self, campaign_id: int, user_id: int, fallback_name: str) -> str:
        return self.get_registered_character_name(campaign_id, user_id) or fallback_name

    def get_registered_character_name(self, campaign_id: int, user_id: int) -> str | None:
        """The name from `!register-character`, or None if the user hasn't registered (and isn't recorded)."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT character_name
                FROM character_registry
                WHERE campaign_id = ? AND user_id = ?
                """,
                (campaign_id, user_id),
            ).fetchone()
        return row["character_name"] if row else None

    def registered_user_ids(self, campaign_id: int) -> set[int]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT user_id FROM character_registry WHERE campaign_id = ?", (campaign_id,)
            ).fetchall()
        return {int(row["user_id"]) for row in rows}

    # --- Sessions ---------------------------------------------------------------------------

    def create_session(
        self,
        guild_id: int,
        voice_channel_id: int,
        text_channel_id: int,
        title: str | None,
        campaign_id: int | None = None,
    ) -> int:
        """Create a recording session in `campaign_id` (default: the server's active campaign)."""
        if campaign_id is None:
            campaign_id = self.active_campaign(guild_id).id
        started_at = datetime.utcnow().isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (
                    guild_id, campaign_id, number, voice_channel_id, text_channel_id, title,
                    started_at, status
                )
                VALUES (?, ?, {_NEXT_SESSION_NUMBER}, ?, ?, ?, ?, 'recording')
                """.format(_NEXT_SESSION_NUMBER=_NEXT_SESSION_NUMBER),
                (guild_id, campaign_id, campaign_id, voice_channel_id, text_channel_id, title, started_at),
            )
            return int(cursor.lastrowid)

    def set_session_status(self, session_id: int, status: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE sessions SET status = ? WHERE id = ?",
                (status, session_id),
            )

    def queue_session(self, session_id: int, kind: str = PROCESS_TRANSCRIBE) -> None:
        """Put a session on the processing queue (status 'processing'); the queue survives restarts."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = 'processing', processing_kind = ?, processing_attempts = 0
                WHERE id = ?
                """,
                (kind, session_id),
            )

    def reset_session_processing(self, session_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = 'processing',
                    processing_kind = ?,
                    processing_attempts = 0,
                    transcript_path = NULL,
                    summary_path = NULL
                WHERE id = ?
                """,
                (PROCESS_TRANSCRIBE, session_id),
            )
            # Track utterances are recreated by speech-to-text; legacy clip rows keep their row.
            conn.execute(
                "DELETE FROM transcript_segments WHERE session_id = ? AND track_id IS NOT NULL",
                (session_id,),
            )
            conn.execute(
                """
                UPDATE transcript_segments
                SET transcript_text = NULL
                WHERE session_id = ?
                """,
                (session_id,),
            )

    def reset_session_llm_processing(self, session_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = 'processing',
                    processing_kind = ?,
                    processing_attempts = 0,
                    summary_path = NULL
                WHERE id = ?
                """,
                (PROCESS_LLM_ONLY, session_id),
            )

    def next_queued_session(self, guild_id: int) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE guild_id = ? AND status = 'processing' ORDER BY id LIMIT 1",
                (guild_id,),
            ).fetchone()

    def begin_processing_attempt(self, session_id: int) -> int:
        """Count a processing attempt (so a session that crashes the bot isn't retried forever)."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE sessions SET processing_attempts = processing_attempts + 1 WHERE id = ?",
                (session_id,),
            )
            row = conn.execute("SELECT processing_attempts FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return int(row["processing_attempts"]) if row else 0

    def get_sessions_with_status(self, status: str) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return list(conn.execute("SELECT * FROM sessions WHERE status = ? ORDER BY id", (status,)).fetchall())

    def mark_session_interrupted(self, session_id: int, interrupted_at: datetime) -> None:
        """A recording cut short by a restart: queue it for processing and note when capture stopped."""
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = 'processing', processing_kind = ?, processing_attempts = 0, interrupted_at = ?
                WHERE id = ?
                """,
                (PROCESS_TRANSCRIBE, interrupted_at.isoformat(), session_id),
            )

    def sessions_for_audio_cleanup(self, ended_before: datetime) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return list(
                conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE status = 'completed' AND audio_deleted_at IS NULL
                      AND ended_at IS NOT NULL AND ended_at < ?
                    ORDER BY id
                    """,
                    (ended_before.isoformat(),),
                ).fetchall()
            )

    def mark_audio_deleted(self, session_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE sessions SET audio_deleted_at = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), session_id),
            )

    # --- Per-speaker audio tracks -----------------------------------------------------------

    def create_track(
        self,
        session_id: int,
        user_id: int,
        display_name: str,
        character_name: str,
        path: Path,
        started_at: datetime,
    ) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO audio_tracks (session_id, user_id, display_name, character_name, path, started_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, display_name, character_name, str(path), started_at.isoformat()),
            )
            return int(cursor.lastrowid)

    def close_track(self, track_id: int, ended_at: datetime) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE audio_tracks SET ended_at = ? WHERE id = ?", (ended_at.isoformat(), track_id))

    def update_track_path(self, track_id: int, path: Path) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE audio_tracks SET path = ? WHERE id = ?", (str(path), track_id))

    def get_session_tracks(self, session_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return list(
                conn.execute(
                    "SELECT * FROM audio_tracks WHERE session_id = ? ORDER BY started_at, id", (session_id,)
                ).fetchall()
            )

    def replace_track_segments(self, track: sqlite3.Row, character_name: str, segments: list[SpeakerSegment]) -> None:
        """Store a track's utterances, replacing any from an earlier run, in one transaction."""
        with self.connection() as conn:
            conn.execute("DELETE FROM transcript_segments WHERE track_id = ?", (track["id"],))
            conn.execute(
                "UPDATE audio_tracks SET character_name = ? WHERE id = ?", (character_name, track["id"])
            )
            for segment in segments:
                self._insert_segment(conn, int(track["session_id"]), segment)

    def get_session(self, session_id: int) -> sqlite3.Row | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM sessions
                WHERE id = ?
                """,
                (session_id,),
            ).fetchone()
        return row

    def get_session_by_number(self, campaign_id: int, number: int) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM sessions WHERE campaign_id = ? AND number = ?", (campaign_id, number)
            ).fetchone()

    def get_latest_session(self, campaign_id: int) -> sqlite3.Row | None:
        """The campaign's latest recorded session (imported journal recaps don't count)."""
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM sessions
                WHERE campaign_id = ? AND journal_id IS NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (campaign_id,),
            ).fetchone()
        return row

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
            self._insert_segment(conn, session_id, segment)

    def _insert_segment(self, conn: sqlite3.Connection, session_id: int, segment: SpeakerSegment) -> None:
        conn.execute(
            """
            INSERT INTO transcript_segments (
                session_id, user_id, display_name, character_name,
                started_at, ended_at, audio_path, transcript_text, track_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                segment.track_id,
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

    # --- Campaign wiki: entities, aliases, append-only facts, pages -------------------------

    def create_entity(
        self,
        campaign_id: int,
        entity_type: str,
        name: str,
        aliases: list[str] | None = None,
        short_description: str = "",
        created_session_id: int | None = None,
    ) -> int:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO entities (
                    campaign_id, type, canonical_name, name_norm, short_description,
                    created_session_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    entity_type,
                    name.strip(),
                    normalize_name(name),
                    short_description.strip(),
                    created_session_id,
                    now,
                    now,
                ),
            )
            entity_id = int(cursor.lastrowid)
            for alias in aliases or []:
                self._insert_alias(conn, entity_id, name, alias)
        return entity_id

    def get_entity(self, campaign_id: int, entity_id: int) -> Entity | None:
        """Return the live entity for `entity_id`, following merges to the surviving entity."""
        with self.connection() as conn:
            seen: set[int] = set()
            current = entity_id
            while current not in seen:
                seen.add(current)
                row = conn.execute(
                    "SELECT * FROM entities WHERE campaign_id = ? AND id = ?",
                    (campaign_id, current),
                ).fetchone()
                if row is None:
                    return None
                if row["merged_into"] is None:
                    return self._entity_from_row(conn, row)
                current = int(row["merged_into"])
        return None

    def list_entities(self, campaign_id: int) -> list[Entity]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM entities
                WHERE campaign_id = ? AND merged_into IS NULL
                ORDER BY type, canonical_name COLLATE NOCASE
                """,
                (campaign_id,),
            ).fetchall()
            return [self._entity_from_row(conn, row) for row in rows]

    def find_entities_by_name(self, campaign_id: int, name: str) -> list[Entity]:
        """Live entities whose canonical name or an alias normalizes to the same text as `name`."""
        norm = normalize_name(name)
        if not norm:
            return []
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT e.*
                FROM entities e
                LEFT JOIN entity_aliases a ON a.entity_id = e.id
                WHERE e.campaign_id = ? AND e.merged_into IS NULL
                  AND (e.name_norm = ? OR a.alias_norm = ?)
                ORDER BY e.id
                """,
                (campaign_id, norm, norm),
            ).fetchall()
            return [self._entity_from_row(conn, row) for row in rows]

    def add_alias(self, entity_id: int, alias: str) -> bool:
        with self.connection() as conn:
            row = conn.execute("SELECT canonical_name FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if row is None:
                return False
            added = self._insert_alias(conn, entity_id, row["canonical_name"], alias)
            if added:
                self._touch_entity(conn, entity_id)
            return added

    def rename_entity(self, entity_id: int, new_name: str) -> None:
        """Change the canonical name; the old name is kept as an alias."""
        new_name = new_name.strip()
        with self.connection() as conn:
            row = conn.execute("SELECT canonical_name FROM entities WHERE id = ?", (entity_id,)).fetchone()
            if row is None:
                raise KeyError(entity_id)
            old_name = row["canonical_name"]
            conn.execute(
                "UPDATE entities SET canonical_name = ?, name_norm = ? WHERE id = ?",
                (new_name, normalize_name(new_name), entity_id),
            )
            conn.execute(
                "DELETE FROM entity_aliases WHERE entity_id = ? AND alias_norm = ?",
                (entity_id, normalize_name(new_name)),
            )
            self._insert_alias(conn, entity_id, new_name, old_name)
            self._touch_entity(conn, entity_id)

    def set_entity_status(self, entity_id: int, status: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE entities SET status = ? WHERE id = ?", (status, entity_id))

    def active_facts_by_type(self, campaign_id: int, entity_type: str) -> dict[int, list[str]]:
        """Active fact texts of every live entity of one type, keyed by entity id."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT f.entity_id, f.text
                FROM facts f
                JOIN entities e ON e.id = f.entity_id
                WHERE e.campaign_id = ? AND e.type = ? AND e.merged_into IS NULL
                  AND f.superseded_by IS NULL AND f.retracted_at IS NULL
                ORDER BY f.id
                """,
                (campaign_id, entity_type),
            ).fetchall()
        result: dict[int, list[str]] = {}
        for row in rows:
            result.setdefault(int(row["entity_id"]), []).append(row["text"])
        return result

    def set_entity_short_description(self, entity_id: int, short_description: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE entities SET short_description = ? WHERE id = ?",
                (short_description.strip(), entity_id),
            )
            self._touch_entity(conn, entity_id)

    def merge_entities(self, source_id: int, target_id: int) -> None:
        """Fold `source_id` into `target_id`: facts and aliases move, the source keeps a pointer."""
        if source_id == target_id:
            raise ValueError("Cannot merge an entity into itself.")
        with self.connection() as conn:
            source = conn.execute("SELECT * FROM entities WHERE id = ?", (source_id,)).fetchone()
            target = conn.execute("SELECT * FROM entities WHERE id = ?", (target_id,)).fetchone()
            if source is None or target is None:
                raise KeyError(source_id if source is None else target_id)
            conn.execute("UPDATE facts SET entity_id = ? WHERE entity_id = ?", (target_id, source_id))
            aliases = conn.execute(
                "SELECT alias FROM entity_aliases WHERE entity_id = ?",
                (source_id,),
            ).fetchall()
            for alias in [source["canonical_name"], *[row["alias"] for row in aliases]]:
                self._insert_alias(conn, target_id, target["canonical_name"], alias)
            conn.execute("DELETE FROM pages WHERE entity_id = ?", (source_id,))
            conn.execute(
                "UPDATE entities SET merged_into = ? WHERE merged_into = ? OR id = ?",
                (target_id, source_id, source_id),
            )
            self._touch_entity(conn, target_id)

    def add_fact(
        self,
        campaign_id: int,
        entity_id: int,
        text: str,
        kind: str,
        session_id: int | None = None,
        transcript_ts: str | None = None,
        source_ref: str | None = None,
        created_by_user_id: int | None = None,
    ) -> int:
        if kind not in FACT_KINDS:
            raise ValueError(f"Unknown fact kind: {kind}")
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO facts (
                    campaign_id, entity_id, session_id, transcript_ts, source_ref, text, kind,
                    created_at, created_by_user_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    entity_id,
                    session_id,
                    transcript_ts or None,
                    source_ref,
                    text.strip(),
                    kind,
                    now,
                    created_by_user_id,
                ),
            )
            return int(cursor.lastrowid)

    def get_fact(self, campaign_id: int, fact_id: int) -> Fact | None:
        with self.connection() as conn:
            row = conn.execute(
                f"{_FACTS_WITH_SESSION} WHERE f.campaign_id = ? AND f.id = ?",
                (campaign_id, fact_id),
            ).fetchone()
        return _fact_from_row(row) if row else None

    def get_facts(self, campaign_id: int, fact_ids: list[int]) -> dict[int, Fact]:
        if not fact_ids:
            return {}
        placeholders = ",".join("?" for _ in fact_ids)
        with self.connection() as conn:
            rows = conn.execute(
                f"{_FACTS_WITH_SESSION} WHERE f.campaign_id = ? AND f.id IN ({placeholders})",
                (campaign_id, *fact_ids),
            ).fetchall()
        return {int(row["id"]): _fact_from_row(row) for row in rows}

    def get_entity_facts(self, entity_id: int, active_only: bool = True) -> list[Fact]:
        query = f"{_FACTS_WITH_SESSION} WHERE f.entity_id = ?"
        if active_only:
            query += " AND f.superseded_by IS NULL AND f.retracted_at IS NULL"
        query += " ORDER BY f.id"
        with self.connection() as conn:
            rows = conn.execute(query, (entity_id,)).fetchall()
        return [_fact_from_row(row) for row in rows]

    def count_active_facts(self, campaign_id: int) -> dict[int, int]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT entity_id, COUNT(*) AS n
                FROM facts
                WHERE campaign_id = ? AND superseded_by IS NULL AND retracted_at IS NULL
                GROUP BY entity_id
                """,
                (campaign_id,),
            ).fetchall()
        return {int(row["entity_id"]): int(row["n"]) for row in rows}

    def active_fact_ids_by_entity(self, campaign_id: int) -> dict[int, set[int]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT f.entity_id, f.id
                FROM facts f
                JOIN entities e ON e.id = f.entity_id
                WHERE f.campaign_id = ? AND e.merged_into IS NULL
                  AND f.superseded_by IS NULL AND f.retracted_at IS NULL
                """,
                (campaign_id,),
            ).fetchall()
        result: dict[int, set[int]] = {}
        for row in rows:
            result.setdefault(int(row["entity_id"]), set()).add(int(row["id"]))
        return result

    def page_source_ids_by_entity(self, campaign_id: int) -> dict[int, set[int]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT p.entity_id, p.source_fact_ids_json
                FROM pages p
                JOIN entities e ON e.id = p.entity_id
                WHERE e.campaign_id = ?
                """,
                (campaign_id,),
            ).fetchall()
        return {int(row["entity_id"]): set(json.loads(row["source_fact_ids_json"])) for row in rows}

    def supersede_fact(self, fact_id: int, replacement_fact_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE facts SET superseded_by = ? WHERE id = ?",
                (replacement_fact_id, fact_id),
            )

    def retract_fact(self, fact_id: int, reason: str) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE facts
                SET retracted_at = ?, retraction_reason = ?
                WHERE id = ? AND retracted_at IS NULL
                """,
                (now, reason, fact_id),
            )

    def retract_session_facts(self, campaign_id: int, session_id: int, reason: str) -> tuple[int, set[int]]:
        """Retract the active observed facts from one session (before re-extraction).

        Returns (retracted count, affected entity ids).
        """
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, entity_id
                FROM facts
                WHERE campaign_id = ? AND session_id = ? AND kind = 'observed'
                  AND superseded_by IS NULL AND retracted_at IS NULL
                """,
                (campaign_id, session_id),
            ).fetchall()
            conn.executemany(
                "UPDATE facts SET retracted_at = ?, retraction_reason = ? WHERE id = ?",
                [(now, reason, int(row["id"])) for row in rows],
            )
        return len(rows), {int(row["entity_id"]) for row in rows}

    def retract_source_facts(self, campaign_id: int, source_ref: str, reason: str) -> tuple[int, set[int]]:
        """Retract the active imported facts from one source (a journal entry, before re-import).

        Pinned facts are never touched. Returns (retracted count, affected entity ids).
        """
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, entity_id
                FROM facts
                WHERE campaign_id = ? AND source_ref = ? AND kind = 'imported'
                  AND superseded_by IS NULL AND retracted_at IS NULL
                """,
                (campaign_id, source_ref),
            ).fetchall()
            conn.executemany(
                "UPDATE facts SET retracted_at = ?, retraction_reason = ? WHERE id = ?",
                [(now, reason, int(row["id"])) for row in rows],
            )
        return len(rows), {int(row["entity_id"]) for row in rows}

    def set_entity_type(self, entity_id: int, entity_type: str) -> None:
        with self.connection() as conn:
            conn.execute("UPDATE entities SET type = ? WHERE id = ?", (entity_type, entity_id))
            self._touch_entity(conn, entity_id)

    # --- VTT journal import (#15) -----------------------------------------------------------

    def get_journal_entry(self, campaign_id: int, journal_id: str) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM journal_entries WHERE campaign_id = ? AND journal_id = ?",
                (campaign_id, journal_id),
            ).fetchone()

    def record_journal_entry(
        self,
        campaign_id: int,
        journal_id: str,
        name: str,
        folder: str,
        action: str,
        content_hash: str,
        entity_id: int | None = None,
        session_id: int | None = None,
    ) -> None:
        """Remember that a journal entry was imported (its content hash makes a re-import skip it)."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO journal_entries (
                    campaign_id, journal_id, name, folder, action, entity_id, session_id, content_hash, imported_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(campaign_id, journal_id) DO UPDATE SET
                    name = excluded.name,
                    folder = excluded.folder,
                    action = excluded.action,
                    entity_id = excluded.entity_id,
                    session_id = excluded.session_id,
                    content_hash = excluded.content_hash,
                    imported_at = excluded.imported_at
                """,
                (
                    campaign_id,
                    journal_id,
                    name,
                    folder,
                    action,
                    entity_id,
                    session_id,
                    content_hash,
                    datetime.utcnow().isoformat(),
                ),
            )

    def upsert_journal_session(
        self, campaign_id: int, journal_id: str, title: str, started_at: datetime, summary_markdown: str
    ) -> int:
        """Create or update the session that holds an imported recap, and write its summary file.

        The session is 'completed' with a summary and no audio, so search indexes the recap like
        any session summary. `ended_at` changes on every update, which re-indexes it.
        """
        with self.connection() as conn:
            row = conn.execute(
                "SELECT id FROM sessions WHERE campaign_id = ? AND journal_id = ?", (campaign_id, journal_id)
            ).fetchone()
            if row is None:
                guild = conn.execute("SELECT guild_id FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
                cursor = conn.execute(
                    """
                    INSERT INTO sessions (
                        guild_id, campaign_id, number, voice_channel_id, text_channel_id, title, started_at,
                        status, journal_id
                    )
                    VALUES (?, ?, {_NEXT_SESSION_NUMBER}, 0, 0, ?, ?, 'completed', ?)
                    """.format(_NEXT_SESSION_NUMBER=_NEXT_SESSION_NUMBER),
                    (int(guild["guild_id"]), campaign_id, campaign_id, title, started_at.isoformat(), journal_id),
                )
                session_id = int(cursor.lastrowid)
            else:
                session_id = int(row["id"])
            summary_path = self.sessions_dir / str(session_id) / "summary.md"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary_markdown, encoding="utf-8")
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, started_at = ?, ended_at = ?, status = 'completed', summary_path = ?
                WHERE id = ?
                """,
                (title, started_at.isoformat(), datetime.utcnow().isoformat(), str(summary_path), session_id),
            )
        return session_id

    def create_journal_import(self, campaign_id: int, guild_id: int, text_channel_id: int, path: Path) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO journal_imports (campaign_id, guild_id, text_channel_id, path, status, created_at)
                VALUES (?, ?, ?, ?, 'running', ?)
                """,
                (campaign_id, guild_id, text_channel_id, str(path), datetime.utcnow().isoformat()),
            )
            return int(cursor.lastrowid)

    def running_journal_imports(self) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return list(
                conn.execute("SELECT * FROM journal_imports WHERE status = 'running' ORDER BY id").fetchall()
            )

    def begin_journal_import_attempt(self, import_id: int) -> int:
        """Count an attempt (so an import that crashes the bot isn't retried forever)."""
        with self.connection() as conn:
            conn.execute("UPDATE journal_imports SET attempts = attempts + 1 WHERE id = ?", (import_id,))
            row = conn.execute("SELECT attempts FROM journal_imports WHERE id = ?", (import_id,)).fetchone()
        return int(row["attempts"]) if row else 0

    def finish_journal_import(self, import_id: int, status: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE journal_imports SET status = ?, finished_at = ? WHERE id = ?",
                (status, datetime.utcnow().isoformat(), import_id),
            )

    def get_page(self, entity_id: int) -> Page | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM pages WHERE entity_id = ?", (entity_id,)).fetchone()
        return _page_from_row(row) if row else None

    def save_page(self, entity_id: int, markdown: str, source_fact_ids: list[int]) -> None:
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO pages (entity_id, markdown, source_fact_ids_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    markdown = excluded.markdown,
                    source_fact_ids_json = excluded.source_fact_ids_json,
                    updated_at = excluded.updated_at
                """,
                (entity_id, markdown, json.dumps(sorted(set(source_fact_ids))), now),
            )

    def delete_page(self, entity_id: int) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM pages WHERE entity_id = ?", (entity_id,))

    def list_pages(self, campaign_id: int) -> list[tuple[Entity, Page]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT e.*, p.markdown, p.source_fact_ids_json, p.updated_at AS page_updated_at
                FROM pages p
                JOIN entities e ON e.id = p.entity_id
                WHERE e.campaign_id = ? AND e.merged_into IS NULL
                ORDER BY e.type, e.canonical_name COLLATE NOCASE
                """,
                (campaign_id,),
            ).fetchall()
            return [
                (
                    self._entity_from_row(conn, row),
                    Page(
                        entity_id=int(row["id"]),
                        markdown=row["markdown"],
                        source_fact_ids=json.loads(row["source_fact_ids_json"]),
                        updated_at=row["page_updated_at"],
                    ),
                )
                for row in rows
            ]

    # --- Search index (#8): documents derived from pages and sessions -----------------------

    def search_campaign_ids(self) -> list[int]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT campaign_id FROM entities
                UNION SELECT campaign_id FROM sessions WHERE campaign_id IS NOT NULL
                ORDER BY campaign_id
                """
            ).fetchall()
        return [int(row["campaign_id"]) for row in rows]

    def page_doc_sources(self, campaign_id: int) -> list[sqlite3.Row]:
        """Live pages with a version string that changes when the page text or the entity's names do."""
        with self.connection() as conn:
            return list(
                conn.execute(
                    """
                    SELECT e.id AS entity_id, p.updated_at || '|' || e.updated_at || '|' || e.canonical_name AS version
                    FROM pages p
                    JOIN entities e ON e.id = p.entity_id
                    WHERE e.campaign_id = ? AND e.merged_into IS NULL
                    ORDER BY e.id
                    """,
                    (campaign_id,),
                ).fetchall()
            )

    def summarized_sessions(self, campaign_id: int) -> list[sqlite3.Row]:
        """Sessions with a written summary; `ended_at` changes each time a session is (re)processed."""
        with self.connection() as conn:
            return list(
                conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE campaign_id = ? AND summary_path IS NOT NULL AND status = 'completed'
                    ORDER BY id
                    """,
                    (campaign_id,),
                ).fetchall()
            )

    def search_doc_versions(self, campaign_id: int) -> dict[tuple[str, int, int], tuple[int, str]]:
        """(kind, ref_id, part) -> (doc id, version) for every indexed document of a campaign."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id, kind, ref_id, part, version FROM search_docs WHERE campaign_id = ?",
                (campaign_id,),
            ).fetchall()
        return {(row["kind"], int(row["ref_id"]), int(row["part"])): (int(row["id"]), row["version"]) for row in rows}

    def replace_search_docs(
        self,
        campaign_id: int,
        kind: str,
        ref_id: int,
        docs: list[SearchDoc],
        version: str,
    ) -> None:
        """Store the documents made from one source (a page, a summary, one session's transcript),
        replacing the earlier ones. A document whose text is unchanged keeps its embedding."""
        with self.connection() as conn:
            existing = {
                int(row["part"]): row
                for row in conn.execute(
                    "SELECT * FROM search_docs WHERE campaign_id = ? AND kind = ? AND ref_id = ?",
                    (campaign_id, kind, ref_id),
                ).fetchall()
            }
            for doc in docs:
                old = existing.pop(doc.part, None)
                if old is None:
                    conn.execute(
                        """
                        INSERT INTO search_docs (campaign_id, kind, ref_id, part, session_id, start_ts, title, body, version)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (campaign_id, kind, ref_id, doc.part, doc.session_id, doc.start_ts, doc.title, doc.body, version),
                    )
                elif old["title"] == doc.title and old["body"] == doc.body:
                    conn.execute(
                        "UPDATE search_docs SET version = ?, session_id = ?, start_ts = ? WHERE id = ?",
                        (version, doc.session_id, doc.start_ts, old["id"]),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE search_docs
                        SET title = ?, body = ?, version = ?, session_id = ?, start_ts = ?,
                            embedding = NULL, embed_model = NULL, embed_dim = NULL
                        WHERE id = ?
                        """,
                        (doc.title, doc.body, version, doc.session_id, doc.start_ts, old["id"]),
                    )
            for old in existing.values():
                conn.execute("DELETE FROM search_docs WHERE id = ?", (old["id"],))

    def delete_search_docs(self, campaign_id: int, kind: str, ref_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                "DELETE FROM search_docs WHERE campaign_id = ? AND kind = ? AND ref_id = ?",
                (campaign_id, kind, ref_id),
            )

    def clear_search_index(self, campaign_id: int) -> None:
        """Drop every search document of a campaign (they are rebuilt from pages and sessions)."""
        with self.connection() as conn:
            conn.execute("DELETE FROM search_docs WHERE campaign_id = ?", (campaign_id,))

    def docs_needing_embedding(self, campaign_id: int, model: str, kinds: tuple[str, ...]) -> list[SearchDoc]:
        """Documents without an embedding from `model` (new, changed, or embedded by another model)."""
        placeholders = ",".join("?" for _ in kinds)
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM search_docs
                WHERE campaign_id = ? AND kind IN ({placeholders})
                  AND (embedding IS NULL OR embed_model IS NOT ?)
                ORDER BY id
                """,
                (campaign_id, *kinds, model),
            ).fetchall()
        return [_search_doc_from_row(row) for row in rows]

    def set_doc_embedding(self, doc_id: int, title: str, body: str, model: str, vector: bytes, dim: int) -> bool:
        """Store an embedding unless the document changed while it was being computed."""
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE search_docs SET embedding = ?, embed_model = ?, embed_dim = ?
                WHERE id = ? AND title = ? AND body = ?
                """,
                (vector, model, dim, doc_id, title, body),
            )
            return cursor.rowcount > 0

    def doc_embeddings(self, campaign_id: int, model: str, dim: int, kinds: tuple[str, ...]) -> list[tuple[int, bytes]]:
        placeholders = ",".join("?" for _ in kinds)
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, embedding FROM search_docs
                WHERE campaign_id = ? AND kind IN ({placeholders}) AND embed_model = ? AND embed_dim = ?
                  AND embedding IS NOT NULL
                """,
                (campaign_id, *kinds, model, dim),
            ).fetchall()
        return [(int(row["id"]), bytes(row["embedding"])) for row in rows]

    def keyword_search(self, campaign_id: int, match: str, kinds: tuple[str, ...], limit: int) -> list[int]:
        """Doc ids matching an FTS5 query, best BM25 first (titles weigh more than bodies)."""
        placeholders = ",".join("?" for _ in kinds)
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT d.id
                FROM search_fts
                JOIN search_docs d ON d.id = search_fts.rowid
                WHERE search_fts MATCH ? AND d.campaign_id = ? AND d.kind IN ({placeholders})
                ORDER BY bm25(search_fts, 4.0, 1.0)
                LIMIT ?
                """,
                (match, campaign_id, *kinds, limit),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def get_search_docs(self, doc_ids: list[int]) -> dict[int, SearchDoc]:
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        with self.connection() as conn:
            rows = conn.execute(f"SELECT * FROM search_docs WHERE id IN ({placeholders})", doc_ids).fetchall()
        return {int(row["id"]): _search_doc_from_row(row) for row in rows}

    def page_doc_ids(self, campaign_id: int, entity_ids: list[int]) -> dict[int, int]:
        """entity id -> search doc id of its page."""
        if not entity_ids:
            return {}
        placeholders = ",".join("?" for _ in entity_ids)
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, ref_id FROM search_docs
                WHERE campaign_id = ? AND kind = 'page' AND ref_id IN ({placeholders})
                """,
                (campaign_id, *entity_ids),
            ).fetchall()
        return {int(row["ref_id"]): int(row["id"]) for row in rows}

    def count_unembedded_docs(self, campaign_id: int, model: str, kinds: tuple[str, ...]) -> int:
        return len(self.docs_needing_embedding(campaign_id, model, kinds))

    def _insert_alias(self, conn: sqlite3.Connection, entity_id: int, canonical_name: str, alias: str) -> bool:
        alias = alias.strip()
        alias_norm = normalize_name(alias)
        if not alias_norm or alias_norm == normalize_name(canonical_name):
            return False
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO entity_aliases (entity_id, alias, alias_norm)
            VALUES (?, ?, ?)
            """,
            (entity_id, alias, alias_norm),
        )
        return cursor.rowcount > 0

    def _touch_entity(self, conn: sqlite3.Connection, entity_id: int) -> None:
        conn.execute(
            "UPDATE entities SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), entity_id),
        )

    def _entity_from_row(self, conn: sqlite3.Connection, row: sqlite3.Row) -> Entity:
        aliases = conn.execute(
            "SELECT alias FROM entity_aliases WHERE entity_id = ? ORDER BY alias COLLATE NOCASE",
            (row["id"],),
        ).fetchall()
        return Entity(
            id=int(row["id"]),
            campaign_id=int(row["campaign_id"]),
            type=row["type"],
            canonical_name=row["canonical_name"],
            aliases=[alias["alias"] for alias in aliases],
            short_description=row["short_description"] or "",
            merged_into=row["merged_into"],
            status=row["status"] or "",
        )


def _campaign_from_row(row: sqlite3.Row) -> Campaign:
    return Campaign(
        id=int(row["id"]),
        guild_id=int(row["guild_id"]),
        name=row["name"],
        is_active=bool(row["is_active"]),
    )


def _fact_from_row(row: sqlite3.Row) -> Fact:
    return Fact(
        id=int(row["id"]),
        campaign_id=int(row["campaign_id"]),
        entity_id=int(row["entity_id"]),
        kind=row["kind"],
        text=row["text"],
        session_id=row["session_id"],
        transcript_ts=row["transcript_ts"],
        source_ref=row["source_ref"],
        created_at=row["created_at"],
        superseded_by=row["superseded_by"],
        retracted_at=row["retracted_at"],
        session_number=row["session_number"] if "session_number" in row.keys() else None,
        session_date=(
            session_date(row["session_started_at"], row["session_journal_id"] is not None) or None
            if "session_started_at" in row.keys()
            else None
        ),
    )


def _search_doc_from_row(row: sqlite3.Row) -> SearchDoc:
    return SearchDoc(
        id=int(row["id"]),
        campaign_id=int(row["campaign_id"]),
        kind=row["kind"],
        ref_id=int(row["ref_id"]),
        part=int(row["part"]),
        title=row["title"],
        body=row["body"],
        session_id=row["session_id"],
        start_ts=row["start_ts"],
    )


def _page_from_row(row: sqlite3.Row) -> Page:
    return Page(
        entity_id=int(row["entity_id"]),
        markdown=row["markdown"],
        source_fact_ids=json.loads(row["source_fact_ids_json"]),
        updated_at=row["updated_at"],
    )
