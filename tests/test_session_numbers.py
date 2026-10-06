from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from scrollkeeper import storage as storage_module
from scrollkeeper.models import Fact, session_date
from scrollkeeper.search import SearchIndex, summary_doc
from scrollkeeper.storage import MIGRATIONS, Storage
from scrollkeeper.wiki import CampaignWiki, format_fact

from test_llm_summary import fake_settings
from test_session_wiki import SummaryLLM

DISCORD_IMPORT_ERROR: str | None = None
try:
    from scrollkeeper.session_manager import SessionManager
except ModuleNotFoundError as exc:
    SessionManager = None
    DISCORD_IMPORT_ERROR = str(exc)


class SessionNumberStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_each_campaign_numbers_its_own_sessions(self) -> None:
        first = self.storage.active_campaign(1)
        second, _ = self.storage.switch_campaign(1, "Second")
        a = self.storage.create_session(1, 2, 3, None, campaign_id=first.id)
        b = self.storage.create_session(1, 2, 3, None, campaign_id=second.id)
        c = self.storage.create_session(1, 2, 3, None, campaign_id=first.id)
        recap = self.storage.upsert_journal_session(second.id, "j1", "Journal recap", datetime(2024, 3, 10), "# Notes\n")
        self.assertEqual([self.storage.get_session(i)["number"] for i in (a, b, c, recap)], [1, 1, 2, 2])
        self.assertEqual(int(self.storage.get_session_by_number(first.id, 2)["id"]), c)
        # Updating a recap keeps its number.
        self.storage.upsert_journal_session(second.id, "j1", "Journal recap", datetime(2024, 3, 10), "# Changed\n")
        self.assertEqual(self.storage.get_session(recap)["number"], 2)

    def test_facts_carry_their_session_number_and_date(self) -> None:
        campaign = self.storage.active_campaign(1)
        self.storage.create_session(1, 2, 3, None, campaign_id=campaign.id)
        recap = self.storage.upsert_journal_session(campaign.id, "j1", "Journal recap", datetime(2024, 3, 10), "x")
        entity = self.storage.create_entity(campaign.id, "Character", "Varric")
        fact_id = self.storage.add_fact(campaign.id, entity, "Varric runs the docks.", "imported", recap)
        observed = self.storage.add_fact(campaign.id, entity, "Varric fled.", "observed", recap, "00:10:00")
        fact = self.storage.get_fact(campaign.id, fact_id)
        self.assertEqual((fact.session_id, fact.session_number, fact.session_date), (recap, 2, "2024-03-10"))
        self.assertEqual(fact.source_label(), "session 2, 2024-03-10")
        self.assertEqual(format_fact(fact), f"[F{fact_id}] (session 2, 2024-03-10) Varric runs the docks.")
        labels = [f.source_label() for f in self.storage.get_entity_facts(entity)]
        self.assertEqual(labels, ["session 2, 2024-03-10", "session 2, 2024-03-10 @ 00:10:00"])
        self.assertEqual(self.storage.get_facts(campaign.id, [observed])[observed].session_number, 2)

    def test_a_fact_without_session_details_falls_back_to_the_id(self) -> None:
        self.assertEqual(Fact(1, 1, 1, "observed", "x", session_id=9, transcript_ts="00:01:00").source_label(), "session 9 @ 00:01:00")

    def test_migration_numbers_existing_sessions_by_date_within_each_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(storage_module, "MIGRATIONS", MIGRATIONS[:5]):
                old = Storage(Path(tmp))
                first = old.active_campaign(1).id
                second = old.switch_campaign(1, "Second")[0].id
                rows = [
                    (first, "2026-10-01T23:00:00"),  # recorded first, played later
                    (second, "2026-09-01T23:00:00"),
                    (first, "2024-03-10T00:00:00"),  # an imported recap of an older session
                    (first, "2026-10-08T23:00:00"),
                ]
                with old.connection() as conn:
                    ids = [
                        conn.execute(
                            "INSERT INTO sessions (guild_id, campaign_id, voice_channel_id, text_channel_id, started_at, status)"
                            " VALUES (1, ?, 2, 3, ?, 'completed')",
                            row,
                        ).lastrowid
                        for row in rows
                    ]
            storage = Storage(Path(tmp))
            self.assertEqual([storage.get_session(i)["number"] for i in ids], [2, 1, 1, 3])
            with storage.connection() as conn:
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], len(MIGRATIONS))
                with self.assertRaises(sqlite3.IntegrityError):
                    conn.execute("UPDATE sessions SET number = 1 WHERE id = ?", (ids[3],))
            self.assertEqual(storage.get_session(storage.create_session(1, 2, 3, None, campaign_id=first))["number"], 4)


class SessionDateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tz = os.environ.get("TZ")
        os.environ["TZ"] = "America/Toronto"
        time.tzset()

    def tearDown(self) -> None:
        if self._tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = self._tz
        time.tzset()

    def test_recorded_sessions_use_the_local_date_and_recaps_their_own(self) -> None:
        # 21:30 in Toronto is 01:30 UTC the next day.
        self.assertEqual(session_date("2026-10-06T01:30:00"), "2026-10-05")
        self.assertEqual(session_date("2024-03-10T00:00:00", imported=True), "2024-03-10")
        self.assertEqual(session_date("now"), "")

    def test_summary_titles_use_the_number_and_local_date(self) -> None:
        row = {"id": 40, "number": 3, "title": "The Tower", "started_at": "2026-10-06T01:30:00", "journal_id": None}
        self.assertEqual(summary_doc(row, "Notes").title, "session 3: The Tower (2026-10-05)")


class SearchNumberTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_summaries_are_retitled_and_cited_by_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            first = storage.active_campaign(1)
            second, _ = storage.switch_campaign(1, "Second")
            storage.create_session(1, 2, 3, None, campaign_id=first.id)  # takes id 1
            recap = storage.upsert_journal_session(second.id, "j1", "Journal recap", datetime(2024, 3, 10), "Docks recap.")
            index = SearchIndex(storage, SummaryLLM(), None)
            # An index built before numbers existed is re-titled: the version includes the number.
            with storage.connection() as conn:
                conn.execute("UPDATE sessions SET number = NULL WHERE id = ?", (recap,))
            index.sync_documents(second.id)
            with storage.connection() as conn:
                conn.execute("UPDATE sessions SET number = 1 WHERE id = ?", (recap,))
            index.sync_documents(second.id)
            (doc,) = storage.get_search_docs([doc_id for doc_id, _ in storage.search_doc_versions(second.id).values()]).values()
            self.assertEqual(doc.title, "session 1: Journal recap (2024-03-10)")
            _context, labels = index.build_context(second.id, [doc])
            self.assertEqual(labels, {"S1": "session 1, 2024-03-10 summary"})


@unittest.skipUnless(SessionManager is not None, f"discord dependency unavailable: {DISCORD_IMPORT_ERROR}")
class ReprocessByNumberTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))
        llm = SummaryLLM()
        self.manager = SessionManager(self.storage, llm, CampaignWiki(self.storage, llm, fake_settings()))
        self.manager._ensure_worker = lambda _guild_id: None

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_reprocess_takes_the_number_in_the_active_campaign(self) -> None:
        first = self.storage.active_campaign(1)
        self.storage.create_session(1, 2, 3, None, campaign_id=first.id)
        second, _ = self.storage.switch_campaign(1, "Second")
        target = self.storage.create_session(1, 2, 3, None, campaign_id=second.id)
        self.storage.upsert_journal_session(second.id, "j1", "Journal recap", datetime(2024, 3, 10), "x")

        self.assertEqual(await self.manager.reprocess_llm_only(1, 1), 1)
        self.assertEqual(self.storage.get_session(target)["status"], "processing")
        self.assertIn("Session #1", self.manager.session_status(1))
        with self.assertRaisesRegex(RuntimeError, "has no session #7"):
            await self.manager.reprocess_llm_only(1, 7)
        with self.assertRaisesRegex(RuntimeError, "Session #2 is a recap imported from the journal"):
            await self.manager.reprocess_llm_only(1, 2)


if __name__ == "__main__":
    unittest.main()
