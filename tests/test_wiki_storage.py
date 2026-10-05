from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from scrollkeeper.storage import MIGRATIONS, Storage


UNVERSIONED_SCHEMA = """
CREATE TABLE sessions (
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
INSERT INTO sessions (guild_id, voice_channel_id, text_channel_id, started_at, status)
VALUES (1, 2, 3, '2026-01-01T00:00:00', 'completed');
"""


class StorageTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = Path(self._tmp.name)
        self.storage = Storage(self.data_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class MigrationTests(unittest.TestCase):
    def test_fresh_database_is_at_latest_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            with storage.connection() as conn:
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], len(MIGRATIONS))

    def test_unversioned_database_is_migrated_without_losing_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(Path(tmp) / "scrollkeeper.db")
            conn.executescript(UNVERSIONED_SCHEMA)
            conn.close()

            storage = Storage(Path(tmp))
            Storage(Path(tmp))  # Re-opening must be a no-op.

            with storage.connection() as conn:
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], len(MIGRATIONS))
            self.assertIsNotNone(storage.get_session(1))
            entity_id = storage.create_entity(1, "Character", "Varric")
            storage.add_fact(1, entity_id, "Runs the docks.", "observed", session_id=1, transcript_ts="00:01:02")
            self.assertEqual(len(storage.get_entity_facts(entity_id)), 1)


class EntityStorageTests(StorageTestCase):
    def test_find_by_canonical_name_or_alias_is_normalized(self) -> None:
        entity_id = self.storage.create_entity(1, "Faction", "The Black Hand", aliases=["Hand of Shadows"])
        self.assertEqual([e.id for e in self.storage.find_entities_by_name(1, "black hand")], [entity_id])
        self.assertEqual([e.id for e in self.storage.find_entities_by_name(1, "HAND OF SHADOWS!")], [entity_id])
        self.assertEqual(self.storage.find_entities_by_name(2, "black hand"), [])
        ledger = self.storage.create_entity(1, "Item", "Varric’s Ledger")
        self.assertEqual([e.id for e in self.storage.find_entities_by_name(1, "varrics ledger")], [ledger])

    def test_alias_equal_to_name_is_ignored(self) -> None:
        entity_id = self.storage.create_entity(1, "Character", "Varric", aliases=["varric", "Lord Varric"])
        self.assertEqual(self.storage.get_entity(1, entity_id).aliases, ["Lord Varric"])
        self.assertFalse(self.storage.add_alias(entity_id, "Lord Varric"))
        self.assertTrue(self.storage.add_alias(entity_id, "The Old Lord"))

    def test_rename_keeps_old_name_as_alias(self) -> None:
        entity_id = self.storage.create_entity(1, "Character", "Varik", aliases=["Varric"])
        self.storage.rename_entity(entity_id, "Varric")
        entity = self.storage.get_entity(1, entity_id)
        self.assertEqual(entity.canonical_name, "Varric")
        self.assertEqual(entity.aliases, ["Varik"])

    def test_merge_moves_facts_and_names_and_redirects(self) -> None:
        source = self.storage.create_entity(1, "Character", "Lord Varric", aliases=["The Old Lord"])
        target = self.storage.create_entity(1, "Character", "Varric")
        fact_id = self.storage.add_fact(1, source, "Owns a tower.", "observed")
        self.storage.save_page(source, "page", [fact_id])

        self.storage.merge_entities(source, target)

        merged = self.storage.get_entity(1, source)
        self.assertEqual(merged.id, target)
        self.assertEqual(sorted(merged.aliases), ["Lord Varric", "The Old Lord"])
        self.assertEqual([f.id for f in self.storage.get_entity_facts(target)], [fact_id])
        self.assertIsNone(self.storage.get_page(source))
        self.assertEqual([e.id for e in self.storage.list_entities(1)], [target])
        self.assertEqual([e.id for e in self.storage.find_entities_by_name(1, "the old lord")], [target])


class FactStorageTests(StorageTestCase):
    def test_retract_session_facts_only_touches_observed_facts_from_that_session(self) -> None:
        a = self.storage.create_entity(1, "Character", "A")
        b = self.storage.create_entity(1, "Character", "B")
        session_one = self.storage.create_session(1, 2, 3, None)
        session_two = self.storage.create_session(1, 2, 3, None)
        retract_me = self.storage.add_fact(1, a, "a1", "observed", session_id=session_one)
        self.storage.add_fact(1, b, "b1", "observed", session_id=session_two)
        self.storage.add_fact(1, a, "pinned", "pinned")

        count, entity_ids = self.storage.retract_session_facts(1, session_one, "reprocessed")

        self.assertEqual((count, entity_ids), (1, {a}))
        self.assertFalse(self.storage.get_fact(1, retract_me).active)
        self.assertEqual([f.text for f in self.storage.get_entity_facts(a)], ["pinned"])
        self.assertEqual(len(self.storage.get_entity_facts(a, active_only=False)), 2)

    def test_supersede_keeps_history(self) -> None:
        entity_id = self.storage.create_entity(1, "Item", "Sword")
        old = self.storage.add_fact(1, entity_id, "It is cursed.", "observed")
        new = self.storage.add_fact(1, entity_id, "It is blessed.", "pinned")
        self.storage.supersede_fact(old, new)
        self.assertEqual(self.storage.get_fact(1, old).superseded_by, new)
        self.assertEqual([f.id for f in self.storage.get_entity_facts(entity_id)], [new])

    def test_unknown_fact_kind_is_rejected(self) -> None:
        entity_id = self.storage.create_entity(1, "Item", "Sword")
        with self.assertRaises(ValueError):
            self.storage.add_fact(1, entity_id, "text", "rumour")


if __name__ == "__main__":
    unittest.main()
