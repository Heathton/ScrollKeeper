from __future__ import annotations

import sqlite3
import tempfile
import types
import unittest
from pathlib import Path

from scrollkeeper.storage import MIGRATIONS, Storage
from scrollkeeper.wiki import CampaignWiki

from test_bot_integration import FakeCtx, FakeSessionManager, build_bot, build_settings
from test_llm_summary import fake_settings
from test_session_wiki import SummaryLLM
from test_wiki import extraction

DISCORD_IMPORT_ERROR: str | None = None
try:
    from scrollkeeper.session_manager import SessionManager
except ModuleNotFoundError as exc:
    SessionManager = None
    DISCORD_IMPORT_ERROR = str(exc)


class CampaignStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_first_use_creates_an_active_default_campaign(self) -> None:
        campaign = self.storage.active_campaign(7)
        self.assertEqual((campaign.name, campaign.is_active), ("Default", True))
        self.assertEqual(self.storage.active_campaign(7).id, campaign.id)
        self.assertEqual([c.name for c in self.storage.list_campaigns(7)], ["Default"])

    def test_switch_creates_once_and_keeps_one_active_campaign_per_server(self) -> None:
        default = self.storage.active_campaign(7)
        rime, created = self.storage.switch_campaign(7, "  Rime   of the Frostmaiden ")
        self.assertTrue(created)
        self.assertEqual(rime.name, "Rime of the Frostmaiden")
        self.assertEqual(self.storage.active_campaign(7).id, rime.id)

        again, created = self.storage.switch_campaign(7, "rime of the frostmaiden")
        self.assertFalse(created)
        self.assertEqual(again.id, rime.id)
        self.assertEqual(again.name, "Rime of the Frostmaiden")

        back, _ = self.storage.switch_campaign(7, "default")
        self.assertEqual(back.id, default.id)
        active = [c.name for c in self.storage.list_campaigns(7) if c.is_active]
        self.assertEqual(active, ["Default"])
        # Another server's campaigns are separate.
        self.assertEqual(self.storage.active_campaign(8).name, "Default")
        self.assertNotEqual(self.storage.active_campaign(8).id, default.id)

    def test_switch_needs_a_name(self) -> None:
        with self.assertRaises(ValueError):
            self.storage.switch_campaign(7, "  ")

    def test_characters_and_sessions_belong_to_a_campaign(self) -> None:
        first = self.storage.active_campaign(7)
        self.storage.register_character(first.id, 10, "Mira")
        first_session = self.storage.create_session(7, 2, 3, None)
        second, _ = self.storage.switch_campaign(7, "Second")
        self.storage.register_character(second.id, 10, "Brenna")

        self.assertEqual(self.storage.get_registered_character_name(first.id, 10), "Mira")
        self.assertEqual(self.storage.get_registered_character_name(second.id, 10), "Brenna")
        self.assertEqual(self.storage.registered_user_ids(second.id), {10})
        self.assertEqual(self.storage.get_session(first_session)["campaign_id"], first.id)
        self.assertEqual(int(self.storage.get_latest_session(first.id)["id"]), first_session)
        self.assertIsNone(self.storage.get_latest_session(second.id))

    def test_wiki_entities_are_kept_per_campaign(self) -> None:
        first = self.storage.active_campaign(7)
        second, _ = self.storage.switch_campaign(7, "Second")
        self.storage.create_entity(first.id, "Character", "Varric")
        self.assertEqual(len(self.storage.find_entities_by_name(first.id, "Varric")), 1)
        self.assertEqual(self.storage.find_entities_by_name(second.id, "Varric"), [])
        self.assertEqual(self.storage.list_entities(second.id), [])


class CampaignMigrationTests(unittest.TestCase):
    def test_version_3_data_moves_into_a_default_campaign_per_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "scrollkeeper.db"
            conn = sqlite3.connect(db)
            conn.executescript(
                """
                CREATE TABLE character_registry (guild_id INTEGER NOT NULL, user_id INTEGER NOT NULL,
                    character_name TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (guild_id, user_id));
                CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, guild_id INTEGER NOT NULL,
                    voice_channel_id INTEGER NOT NULL, text_channel_id INTEGER NOT NULL, title TEXT,
                    started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL, transcript_path TEXT,
                    summary_path TEXT);
                CREATE TABLE transcript_segments (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL, display_name TEXT NOT NULL, character_name TEXT NOT NULL,
                    started_at TEXT NOT NULL, ended_at TEXT NOT NULL, audio_path TEXT NOT NULL, transcript_text TEXT);
                """
            )
            for script in MIGRATIONS[:3]:
                conn.executescript(script)
            conn.executescript(
                """
                PRAGMA user_version = 3;
                INSERT INTO character_registry VALUES (500, 10, 'Mira', 'now');
                INSERT INTO sessions (guild_id, voice_channel_id, text_channel_id, started_at, status)
                    VALUES (500, 2, 3, 'now', 'completed'), (900, 2, 3, 'now', 'completed');
                INSERT INTO entities (guild_id, type, canonical_name, name_norm, created_at, updated_at)
                    VALUES (500, 'Character', 'Varric', 'varric', 'now', 'now');
                INSERT INTO facts (guild_id, entity_id, session_id, text, kind, created_at)
                    VALUES (500, 1, 1, 'Runs the docks.', 'observed', 'now');
                INSERT INTO search_docs (guild_id, kind, ref_id, title, body, version)
                    VALUES (500, 'page', 1, 'Varric', 'Runs the docks.', 'v1');
                """
            )
            conn.commit()
            conn.close()

            storage = Storage(Path(tmp))
            campaign = storage.active_campaign(500)
            self.assertEqual(campaign.name, "Default")
            self.assertEqual(storage.get_registered_character_name(campaign.id, 10), "Mira")
            self.assertEqual(storage.get_session(1)["campaign_id"], campaign.id)
            self.assertEqual(storage.get_session(2)["campaign_id"], storage.active_campaign(900).id)
            self.assertEqual([e.canonical_name for e in storage.list_entities(campaign.id)], ["Varric"])
            self.assertEqual(storage.get_fact(campaign.id, 1).text, "Runs the docks.")
            self.assertEqual(len(storage.search_doc_versions(campaign.id)), 1)
            self.assertEqual(storage.search_campaign_ids(), sorted({campaign.id, storage.active_campaign(900).id}))
            with storage.connection() as conn:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertEqual(version, len(MIGRATIONS))


@unittest.skipUnless(SessionManager is not None, f"discord dependency unavailable: {DISCORD_IMPORT_ERROR}")
class CampaignSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))
        self.llm = SummaryLLM()
        self.wiki = CampaignWiki(self.storage, self.llm, fake_settings())
        self.manager = SessionManager(self.storage, self.llm, self.wiki)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _session_with_transcript(self) -> int:
        from datetime import datetime

        from scrollkeeper.models import SpeakerSegment

        session_id = self.storage.create_session(1, 2, 3, "Test")
        at = datetime.fromisoformat(self.storage.get_session(session_id)["started_at"])
        self.storage.add_transcript_segment(
            session_id, SpeakerSegment(1, "Mira", "Mira", at, at, Path("/nonexistent/1.wav"), "Varric owes us.")
        )
        return session_id

    async def test_switch_is_refused_while_recording(self) -> None:
        self.manager.active_sessions[1] = types.SimpleNamespace(closed=False)
        with self.assertRaisesRegex(RuntimeError, "recording"):
            await self.manager.switch_campaign(1, "Other")
        self.manager.active_sessions.clear()
        campaign, created = await self.manager.switch_campaign(1, "Other")
        self.assertTrue(created)
        self.assertEqual(campaign.name, "Other")

    async def test_a_session_updates_the_wiki_of_its_own_campaign_after_a_switch(self) -> None:
        first = self.storage.active_campaign(1)
        session_id = self._session_with_transcript()
        second, _ = await self.manager.switch_campaign(1, "Second")
        self.llm.extractions = [
            extraction(
                new_entities=[{"name": "Varric", "type": "Character", "aliases": [], "short_description": ""}],
                facts=[{"entity": "Varric", "text": "Varric owes the party.", "timestamp": "00:00:00"}],
            )
        ]
        session = self.manager._session_from_row(self.storage.get_session(session_id))
        await self.manager._process_closed_session(1, session, transcribe_audio=False)

        self.assertEqual([e.canonical_name for e in self.storage.list_entities(first.id)], ["Varric"])
        self.assertEqual(self.storage.list_entities(second.id), [])

    async def test_reprocess_defaults_to_the_latest_session_of_the_active_campaign(self) -> None:
        session_id = self._session_with_transcript()
        await self.manager.switch_campaign(1, "Second")
        with self.assertRaisesRegex(RuntimeError, "active campaign"):
            await self.manager.reprocess_llm_only(1)
        await self.manager.switch_campaign(1, "Default")
        self.manager._ensure_worker = lambda _guild_id: None
        self.assertEqual(await self.manager.reprocess_llm_only(1), session_id)

    async def test_transcript_is_saved_even_when_the_summary_fails(self) -> None:
        session_id = self._session_with_transcript()

        async def broken(*_args, **_kwargs):
            raise RuntimeError("LLM unreachable")

        self.llm.summarize_session = broken
        session = self.manager._session_from_row(self.storage.get_session(session_id))
        with self.assertRaisesRegex(RuntimeError, "LLM unreachable"):
            await self.manager._process_closed_session(1, session, transcribe_audio=False)
        self.assertIn("Varric owes us.", (session.base_dir / "transcript.md").read_text())
        self.assertFalse((session.base_dir / "summary.md").exists())


class CampaignSessionManager(FakeSessionManager):
    """The bot-test session manager, with campaigns backed by a real Storage."""

    async def list_campaigns(self, guild_id: int):
        return self.storage.list_campaigns(guild_id)

    async def switch_campaign(self, guild_id: int, name: str):
        if getattr(self, "recording", False):
            raise RuntimeError("Cannot switch campaigns while a session is recording.")
        return self.storage.switch_campaign(guild_id, name)


@unittest.skipUnless(build_bot is not None, "discord dependency unavailable")
class CampaignCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        from unittest.mock import patch

        from test_bot_integration import BrokenPageLLM, UnavailableEmbedder

        self._tmp = tempfile.TemporaryDirectory()
        settings = build_settings()
        settings.data_dir = Path(self._tmp.name)
        self.storage = Storage(settings.data_dir)
        self._patches = [
            patch("scrollkeeper.bot.LocalAIService", BrokenPageLLM),
            patch("scrollkeeper.bot.SessionManager", CampaignSessionManager),
            patch("scrollkeeper.bot.LocalEmbedder", UnavailableEmbedder),
        ]
        for item in self._patches:
            item.start()
        self.bot = build_bot(settings)
        self.guild = types.SimpleNamespace(id=1)

    async def asyncTearDown(self) -> None:
        await self.bot.close()
        for item in self._patches:
            item.stop()
        self._tmp.cleanup()

    async def _run(self, name: str, *args, **kwargs) -> FakeCtx:
        ctx = FakeCtx(guild=self.guild)
        ctx.author = types.SimpleNamespace(id=55)
        await self.bot.get_command(name).callback(ctx, *args, **kwargs)
        return ctx

    async def test_switch_list_and_current_campaign(self) -> None:
        ctx = await self._run("list-campaigns")
        self.assertIn("**Default** (active)", ctx.sent[-1])

        ctx = await self._run("switch-campaign", campaign_name="Curse of Strahd")
        self.assertIn("Created campaign **Curse of Strahd**", ctx.replies[0])
        ctx = await self._run("current-campaign")
        self.assertEqual(ctx.replies[0], "Active campaign: **Curse of Strahd**.")

        ctx = await self._run("switch-campaign", campaign_name="default")
        self.assertEqual(ctx.replies[0], "Active campaign is now **Default**.")
        ctx = await self._run("list-campaigns")
        self.assertIn("- Curse of Strahd\n- **Default** (active)", ctx.sent[-1])

    async def test_switch_refusal_is_reported(self) -> None:
        FakeSessionManager.last_instance.recording = True
        ctx = await self._run("switch-campaign", campaign_name="Other")
        self.assertIn("Cannot switch campaigns", ctx.replies[0])
        self.assertEqual(self.storage.active_campaign(1).name, "Default")

    async def test_register_character_and_wiki_commands_use_the_active_campaign(self) -> None:
        ctx = await self._run("register-character", character_name="Brenna")
        self.assertIn("in campaign **Default**", ctx.replies[0])
        default = self.storage.active_campaign(1)
        self.assertEqual(self.storage.get_registered_character_name(default.id, 55), "Brenna")

        await self._run("switch-campaign", campaign_name="Other")
        ctx = await self._run("entity", ref="Brenna")
        self.assertIn("No entity matches", ctx.replies[0])
        ctx = await self._run("entities")
        self.assertEqual(ctx.replies[0], "No campaign wiki entities yet.")

        await self._run("switch-campaign", campaign_name="Default")
        ctx = await self._run("entities")
        self.assertIn("Brenna", ctx.sent[-1])


if __name__ == "__main__":
    unittest.main()
