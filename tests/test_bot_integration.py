from __future__ import annotations

import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scrollkeeper.config import Settings
from scrollkeeper.models import SessionArtifacts

DISCORD_IMPORT_ERROR: str | None = None
try:
    from scrollkeeper.bot import build_bot
except ModuleNotFoundError as exc:
    build_bot = None
    DISCORD_IMPORT_ERROR = str(exc)


class FakeTextChannel:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, message: str) -> None:
        self.messages.append(message)


class FakeCtx:
    def __init__(self, guild: object | None) -> None:
        self.guild = guild
        self.replies: list[str] = []
        self.sent: list[str] = []

    async def reply(self, message: str) -> None:
        self.replies.append(message)

    async def send(self, message: str) -> None:
        self.sent.append(message)


class FakeStorage:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def register_character(self, *_args, **_kwargs) -> None:
        return


class FakeLocalAIService:
    def __init__(self, _settings: Settings) -> None:
        pass


class FakeCampaignWiki:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    async def ensure_player_characters(self, _guild_id: int) -> list[int]:
        return []


class FakeSessionManager:
    last_instance: "FakeSessionManager | None" = None

    def __init__(self, _storage: FakeStorage, _llm: FakeLocalAIService, _wiki: FakeCampaignWiki) -> None:
        self.completion_handler = None
        self.notice_handler = None
        self.end_session_calls: list[int] = []
        self.reprocess_calls: list[tuple[int, int | None]] = []
        self.reprocess_llm_calls: list[tuple[int, int | None]] = []
        self.answer_calls: list[tuple[int, str]] = []
        self.status_map: dict[int, str] = {}
        FakeSessionManager.last_instance = self

    def set_completion_handler(self, handler) -> None:
        self.completion_handler = handler

    def set_notice_handler(self, handler) -> None:
        self.notice_handler = handler

    async def end_session(self, guild) -> int:
        self.end_session_calls.append(guild.id)
        self.status_map[guild.id] = "Session #42 status: processing. Transcribing and generating notes."
        return 42

    async def answer_campaign_question(self, guild_id: int, question: str, on_wait=None) -> str:
        self.answer_calls.append((guild_id, question))
        return "Campaign answer from notes."

    async def reprocess_session(self, guild_id: int, session_id: int | None = None) -> int:
        self.reprocess_calls.append((guild_id, session_id))
        self.status_map[guild_id] = "Session #84 status: processing. Reprocessing saved audio and regenerating notes."
        return 84

    async def reprocess_llm_only(self, guild_id: int, session_id: int | None = None) -> int:
        self.reprocess_llm_calls.append((guild_id, session_id))
        self.status_map[guild_id] = (
            "Session #85 status: processing. Reprocessing summaries and notes from existing transcript text."
        )
        return 85

    def session_status(self, guild_id: int) -> str:
        return self.status_map.get(guild_id, "No active or recent session.")


def build_settings() -> Settings:
    return Settings(
        discord_bot_token="token",
        command_prefix="!",
        data_dir=Path("/tmp/scrollkeeper-tests"),
        bot_name="ScrollKeeper",
        stt_base_url="http://stt/v1",
        stt_model="whisper-1",
        stt_timeout_seconds=600,
        llm_base_url="http://llm/v1",
        llm_model="test-model",
        llm_api_key="",
        llm_timeout_seconds=900,
        embed_base_url="http://llm/v1",
        embed_model="test-embed",
        wait_notice_seconds=20,
        health_port=0,
    )


@unittest.skipUnless(build_bot is not None, f"discord dependency unavailable: {DISCORD_IMPORT_ERROR}")
class BotIntegrationHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_session_returns_immediately_and_posts_completion_later(self) -> None:
        with (
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
            patch("scrollkeeper.bot.CampaignWiki", FakeCampaignWiki),
            patch("scrollkeeper.bot.discord.TextChannel", FakeTextChannel),
        ):
            bot = build_bot(build_settings())
            try:
                guild = types.SimpleNamespace(id=999)
                ctx = FakeCtx(guild=guild)
                end_command = bot.get_command("end-session")
                self.assertIsNotNone(end_command)

                await end_command.callback(ctx)

                self.assertEqual(
                    ctx.replies[0],
                    "Ending the session. Processing will continue in the background.",
                )
                self.assertIn("Session **#42** is now in processing.", ctx.sent[0])
                manager = FakeSessionManager.last_instance
                self.assertIsNotNone(manager)
                self.assertIsNotNone(manager.completion_handler)

                channel = FakeTextChannel()
                bot.get_channel = lambda _cid: channel
                bot.get_guild = lambda _gid: None
                artifacts = SessionArtifacts(
                    session_id=42,
                    transcript_markdown="# Transcript\n",
                    session_notes_markdown="Session notes body",
                    cinematic_summary_markdown="Cinematic body",
                    transcript_path=Path("/tmp/transcript.md"),
                    summary_path=Path("/tmp/summary.md"),
                )

                await manager.completion_handler(guild.id, 1234, artifacts, None)

                self.assertEqual(len(channel.messages), 1)
                self.assertIn("## Session 42", channel.messages[0])
                self.assertIn("### Session Notes", channel.messages[0])
            finally:
                await bot.close()

    async def test_completion_handler_splits_long_summary_across_messages(self) -> None:
        with (
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
            patch("scrollkeeper.bot.CampaignWiki", FakeCampaignWiki),
            patch("scrollkeeper.bot.discord.TextChannel", FakeTextChannel),
        ):
            bot = build_bot(build_settings())
            try:
                manager = FakeSessionManager.last_instance
                self.assertIsNotNone(manager)
                self.assertIsNotNone(manager.completion_handler)

                channel = FakeTextChannel()
                bot.get_channel = lambda _cid: channel
                bot.get_guild = lambda _gid: None
                artifacts = SessionArtifacts(
                    session_id=77,
                    transcript_markdown="# Transcript\n",
                    session_notes_markdown=("Session note line.\n" * 300).strip(),
                    cinematic_summary_markdown=("Cinematic line.\n" * 300).strip(),
                    transcript_path=Path("/tmp/transcript.md"),
                    summary_path=Path("/tmp/summary.md"),
                )

                await manager.completion_handler(777, 4321, artifacts, None)

                self.assertGreater(len(channel.messages), 1)
                self.assertTrue(all(len(message) <= 1900 for message in channel.messages))
            finally:
                await bot.close()

    async def test_campaign_question_works_while_processing(self) -> None:
        with (
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
            patch("scrollkeeper.bot.CampaignWiki", FakeCampaignWiki),
            patch("scrollkeeper.bot.discord.TextChannel", FakeTextChannel),
        ):
            bot = build_bot(build_settings())
            try:
                guild = types.SimpleNamespace(id=101)
                ctx = FakeCtx(guild=guild)
                question_command = bot.get_command("campaign-question")
                self.assertIsNotNone(question_command)

                await question_command.callback(ctx, question="What happened in the crypt?")

                self.assertEqual(ctx.replies[0], "Searching campaign notes.")
                self.assertEqual(ctx.sent[0], "Campaign answer from notes.")
            finally:
                await bot.close()

    async def test_completion_handler_posts_failure_message(self) -> None:
        with (
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
            patch("scrollkeeper.bot.CampaignWiki", FakeCampaignWiki),
            patch("scrollkeeper.bot.discord.TextChannel", FakeTextChannel),
        ):
            bot = build_bot(build_settings())
            try:
                manager = FakeSessionManager.last_instance
                self.assertIsNotNone(manager)
                self.assertIsNotNone(manager.completion_handler)

                channel = FakeTextChannel()
                bot.get_channel = lambda _cid: channel
                bot.get_guild = lambda _gid: None

                await manager.completion_handler(200, 12, None, "whisper timeout")

                self.assertEqual(len(channel.messages), 1)
                self.assertIn("Session processing failed: whisper timeout", channel.messages[0])
            finally:
                await bot.close()

    async def test_reprocess_session_uses_latest_by_default(self) -> None:
        with (
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
            patch("scrollkeeper.bot.CampaignWiki", FakeCampaignWiki),
            patch("scrollkeeper.bot.discord.TextChannel", FakeTextChannel),
        ):
            bot = build_bot(build_settings())
            try:
                guild = types.SimpleNamespace(id=303)
                ctx = FakeCtx(guild=guild)
                reprocess_command = bot.get_command("reprocess-session")
                self.assertIsNotNone(reprocess_command)

                await reprocess_command.callback(ctx)

                self.assertEqual(ctx.replies[0], "Reprocessing saved session audio in the background.")
                self.assertIn("Session **#84** is now reprocessing from saved audio.", ctx.sent[0])
                manager = FakeSessionManager.last_instance
                self.assertIsNotNone(manager)
                self.assertEqual(manager.reprocess_calls, [(303, None)])
            finally:
                await bot.close()

    async def test_reprocess_llm_uses_latest_by_default(self) -> None:
        with (
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
            patch("scrollkeeper.bot.CampaignWiki", FakeCampaignWiki),
            patch("scrollkeeper.bot.discord.TextChannel", FakeTextChannel),
        ):
            bot = build_bot(build_settings())
            try:
                guild = types.SimpleNamespace(id=404)
                ctx = FakeCtx(guild=guild)
                reprocess_command = bot.get_command("reprocess-llm")
                self.assertIsNotNone(reprocess_command)

                await reprocess_command.callback(ctx)

                self.assertEqual(ctx.replies[0], "Reprocessing summaries and notes from existing transcript text.")
                self.assertIn("Session **#85** is now reprocessing LLM outputs only (speech-to-text skipped).", ctx.sent[0])
                manager = FakeSessionManager.last_instance
                self.assertIsNotNone(manager)
                self.assertEqual(manager.reprocess_llm_calls, [(404, None)])
            finally:
                await bot.close()


class BrokenPageLLM:
    """LLM stand-in whose page rewrites fail (e.g. the inference host is down)."""

    def __init__(self, _settings: Settings) -> None:
        pass

    async def rewrite_page(self, *_args, **_kwargs) -> dict:
        raise RuntimeError("inference host unreachable")

    async def embed_text(self, *_args, **_kwargs) -> list[float]:
        return [1.0]


@unittest.skipUnless(build_bot is not None, f"discord dependency unavailable: {DISCORD_IMPORT_ERROR}")
class WikiCommandTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        import tempfile

        from scrollkeeper.storage import Storage

        self._tmp = tempfile.TemporaryDirectory()
        settings = build_settings()
        settings.data_dir = Path(self._tmp.name)
        self.storage = Storage(settings.data_dir)
        self._patches = [
            patch("scrollkeeper.bot.LocalAIService", BrokenPageLLM),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
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

    async def test_ambiguous_name_asks_for_an_id(self) -> None:
        self.storage.create_entity(1, "Character", "Varric")
        self.storage.create_entity(1, "Faction", "Varric")
        ctx = FakeCtx(guild=self.guild)
        await self.bot.get_command("entity").callback(ctx, ref="varric")
        self.assertIn("matches several entities", ctx.replies[0])

    async def test_pin_fact_is_saved_even_when_the_page_cannot_be_rewritten(self) -> None:
        entity_id = self.storage.create_entity(1, "Character", "Varric")
        ctx = FakeCtx(guild=self.guild)
        ctx.author = types.SimpleNamespace(id=55)
        await self.bot.get_command("pin-fact").callback(ctx, "Varric", text="Varric is a dwarf.")

        self.assertIn("The change was saved", ctx.replies[0])
        facts = self.storage.get_entity_facts(entity_id)
        self.assertEqual([(f.kind, f.text) for f in facts], [("pinned", "Varric is a dwarf.")])

    async def test_merge_entity_by_quoted_name(self) -> None:
        source = self.storage.create_entity(1, "Character", "Lord Varric")
        target = self.storage.create_entity(1, "Character", "Varric")
        ctx = FakeCtx(guild=self.guild)
        await self.bot.get_command("merge-entity").callback(ctx, "Lord Varric", f"#{target}")
        self.assertEqual(self.storage.get_entity(1, source).id, target)
        self.assertIn("Merged.", ctx.sent[-1])


if __name__ == "__main__":
    unittest.main()
