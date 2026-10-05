from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .models import FACT_KINDS, Entity, Fact, Page, SpeakerSegment, normalize_name


# Forward-only schema migrations, applied in order; `PRAGMA user_version` records how many ran.
# The base tables (characters, sessions, transcript segments) are created in `_init_db` with
# IF NOT EXISTS and count as version 0.
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
]


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

    def list_registered_characters(self, guild_id: int) -> list[str]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT character_name FROM character_registry WHERE guild_id = ? ORDER BY character_name",
                (guild_id,),
            ).fetchall()
        return [row["character_name"] for row in rows]

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

    def reset_session_processing(self, session_id: int) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET status = 'processing',
                    transcript_path = NULL,
                    summary_path = NULL
                WHERE id = ?
                """,
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
                    summary_path = NULL
                WHERE id = ?
                """,
                (session_id,),
            )

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

    def get_latest_session(self, guild_id: int) -> sqlite3.Row | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM sessions
                WHERE guild_id = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (guild_id,),
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

    # --- Campaign wiki: entities, aliases, append-only facts, pages -------------------------

    def create_entity(
        self,
        guild_id: int,
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
                    guild_id, type, canonical_name, name_norm, short_description,
                    created_session_id, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
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

    def get_entity(self, guild_id: int, entity_id: int) -> Entity | None:
        """Return the live entity for `entity_id`, following merges to the surviving entity."""
        with self.connection() as conn:
            seen: set[int] = set()
            current = entity_id
            while current not in seen:
                seen.add(current)
                row = conn.execute(
                    "SELECT * FROM entities WHERE guild_id = ? AND id = ?",
                    (guild_id, current),
                ).fetchone()
                if row is None:
                    return None
                if row["merged_into"] is None:
                    return self._entity_from_row(conn, row)
                current = int(row["merged_into"])
        return None

    def list_entities(self, guild_id: int) -> list[Entity]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM entities
                WHERE guild_id = ? AND merged_into IS NULL
                ORDER BY type, canonical_name COLLATE NOCASE
                """,
                (guild_id,),
            ).fetchall()
            return [self._entity_from_row(conn, row) for row in rows]

    def find_entities_by_name(self, guild_id: int, name: str) -> list[Entity]:
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
                WHERE e.guild_id = ? AND e.merged_into IS NULL
                  AND (e.name_norm = ? OR a.alias_norm = ?)
                ORDER BY e.id
                """,
                (guild_id, norm, norm),
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

    def active_facts_by_type(self, guild_id: int, entity_type: str) -> dict[int, list[str]]:
        """Active fact texts of every live entity of one type, keyed by entity id."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT f.entity_id, f.text
                FROM facts f
                JOIN entities e ON e.id = f.entity_id
                WHERE e.guild_id = ? AND e.type = ? AND e.merged_into IS NULL
                  AND f.superseded_by IS NULL AND f.retracted_at IS NULL
                ORDER BY f.id
                """,
                (guild_id, entity_type),
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
        guild_id: int,
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
                    guild_id, entity_id, session_id, transcript_ts, source_ref, text, kind,
                    created_at, created_by_user_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
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

    def get_fact(self, guild_id: int, fact_id: int) -> Fact | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM facts WHERE guild_id = ? AND id = ?",
                (guild_id, fact_id),
            ).fetchone()
        return _fact_from_row(row) if row else None

    def get_facts(self, guild_id: int, fact_ids: list[int]) -> dict[int, Fact]:
        if not fact_ids:
            return {}
        placeholders = ",".join("?" for _ in fact_ids)
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM facts WHERE guild_id = ? AND id IN ({placeholders})",
                (guild_id, *fact_ids),
            ).fetchall()
        return {int(row["id"]): _fact_from_row(row) for row in rows}

    def get_entity_facts(self, entity_id: int, active_only: bool = True) -> list[Fact]:
        query = "SELECT * FROM facts WHERE entity_id = ?"
        if active_only:
            query += " AND superseded_by IS NULL AND retracted_at IS NULL"
        query += " ORDER BY id"
        with self.connection() as conn:
            rows = conn.execute(query, (entity_id,)).fetchall()
        return [_fact_from_row(row) for row in rows]

    def count_active_facts(self, guild_id: int) -> dict[int, int]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT entity_id, COUNT(*) AS n
                FROM facts
                WHERE guild_id = ? AND superseded_by IS NULL AND retracted_at IS NULL
                GROUP BY entity_id
                """,
                (guild_id,),
            ).fetchall()
        return {int(row["entity_id"]): int(row["n"]) for row in rows}

    def active_fact_ids_by_entity(self, guild_id: int) -> dict[int, set[int]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT f.entity_id, f.id
                FROM facts f
                JOIN entities e ON e.id = f.entity_id
                WHERE f.guild_id = ? AND e.merged_into IS NULL
                  AND f.superseded_by IS NULL AND f.retracted_at IS NULL
                """,
                (guild_id,),
            ).fetchall()
        result: dict[int, set[int]] = {}
        for row in rows:
            result.setdefault(int(row["entity_id"]), set()).add(int(row["id"]))
        return result

    def page_source_ids_by_entity(self, guild_id: int) -> dict[int, set[int]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT p.entity_id, p.source_fact_ids_json
                FROM pages p
                JOIN entities e ON e.id = p.entity_id
                WHERE e.guild_id = ?
                """,
                (guild_id,),
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

    def retract_session_facts(self, guild_id: int, session_id: int, reason: str) -> tuple[int, set[int]]:
        """Retract the active observed facts from one session (before re-extraction).

        Returns (retracted count, affected entity ids).
        """
        now = datetime.utcnow().isoformat()
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, entity_id
                FROM facts
                WHERE guild_id = ? AND session_id = ? AND kind = 'observed'
                  AND superseded_by IS NULL AND retracted_at IS NULL
                """,
                (guild_id, session_id),
            ).fetchall()
            conn.executemany(
                "UPDATE facts SET retracted_at = ?, retraction_reason = ? WHERE id = ?",
                [(now, reason, int(row["id"])) for row in rows],
            )
        return len(rows), {int(row["entity_id"]) for row in rows}

    def get_page(self, entity_id: int) -> Page | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM pages WHERE entity_id = ?", (entity_id,)).fetchone()
        return _page_from_row(row) if row else None

    def save_page(
        self,
        entity_id: int,
        markdown: str,
        source_fact_ids: list[int],
        embedding: list[float] | None,
    ) -> None:
        now = datetime.utcnow().isoformat()
        embedding_json = json.dumps(embedding, ensure_ascii=True) if embedding else None
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO pages (entity_id, markdown, source_fact_ids_json, embedding_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    markdown = excluded.markdown,
                    source_fact_ids_json = excluded.source_fact_ids_json,
                    embedding_json = excluded.embedding_json,
                    updated_at = excluded.updated_at
                """,
                (entity_id, markdown, json.dumps(sorted(set(source_fact_ids))), embedding_json, now),
            )

    def update_page_embedding(self, entity_id: int, embedding: list[float]) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE pages SET embedding_json = ? WHERE entity_id = ?",
                (json.dumps(embedding, ensure_ascii=True), entity_id),
            )

    def delete_page(self, entity_id: int) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM pages WHERE entity_id = ?", (entity_id,))

    def list_pages(self, guild_id: int) -> list[tuple[Entity, Page]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT e.*, p.markdown, p.source_fact_ids_json, p.updated_at AS page_updated_at
                FROM pages p
                JOIN entities e ON e.id = p.entity_id
                WHERE e.guild_id = ? AND e.merged_into IS NULL
                ORDER BY e.type, e.canonical_name COLLATE NOCASE
                """,
                (guild_id,),
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

    def semantic_search_pages(
        self,
        guild_id: int,
        query_embedding: list[float],
        limit: int = 8,
    ) -> list[tuple[Entity, Page]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT e.*, p.markdown, p.source_fact_ids_json, p.embedding_json,
                       p.updated_at AS page_updated_at
                FROM pages p
                JOIN entities e ON e.id = p.entity_id
                WHERE e.guild_id = ? AND e.merged_into IS NULL AND p.embedding_json IS NOT NULL
                """,
                (guild_id,),
            ).fetchall()
            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                score = cosine_similarity(query_embedding, json.loads(row["embedding_json"]))
                scored.append((score, row))
            scored.sort(key=lambda item: item[0], reverse=True)
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
                for _, row in scored[:limit]
            ]

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
            guild_id=int(row["guild_id"]),
            type=row["type"],
            canonical_name=row["canonical_name"],
            aliases=[alias["alias"] for alias in aliases],
            short_description=row["short_description"] or "",
            merged_into=row["merged_into"],
            status=row["status"] or "",
        )


def _fact_from_row(row: sqlite3.Row) -> Fact:
    return Fact(
        id=int(row["id"]),
        guild_id=int(row["guild_id"]),
        entity_id=int(row["entity_id"]),
        kind=row["kind"],
        text=row["text"],
        session_id=row["session_id"],
        transcript_ts=row["transcript_ts"],
        source_ref=row["source_ref"],
        created_at=row["created_at"],
        superseded_by=row["superseded_by"],
        retracted_at=row["retracted_at"],
    )


def _page_from_row(row: sqlite3.Row) -> Page:
    return Page(
        entity_id=int(row["entity_id"]),
        markdown=row["markdown"],
        source_fact_ids=json.loads(row["source_fact_ids_json"]),
        updated_at=row["updated_at"],
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    numerator = sum(x * y for x, y in zip(a, b, strict=False))
    denom_a = math.sqrt(sum(x * x for x in a))
    denom_b = math.sqrt(sum(y * y for y in b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return numerator / (denom_a * denom_b)
