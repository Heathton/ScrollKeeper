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


class FakeServices:
    def __init__(self, _settings: Settings) -> None:
        self.warmup_calls = 0
        self.shutdown_calls = 0

    async def ensure_ollama_running(self) -> None:
        self.warmup_calls += 1

    async def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeLocalAIService:
    def __init__(self, _services: FakeServices) -> None:
        pass


class FakeSessionManager:
    last_instance: "FakeSessionManager | None" = None

    def __init__(self, _storage: FakeStorage, _llm: FakeLocalAIService) -> None:
        self.completion_handler = None
        self.end_session_calls: list[int] = []
        self.answer_calls: list[tuple[int, str]] = []
        self.status_map: dict[int, str] = {}
        FakeSessionManager.last_instance = self

    def set_completion_handler(self, handler) -> None:
        self.completion_handler = handler

    async def end_session(self, guild) -> int:
        self.end_session_calls.append(guild.id)
        self.status_map[guild.id] = "Session #42 status: processing. Transcribing and generating notes."
        return 42

    async def answer_campaign_question(self, guild_id: int, question: str) -> str:
        self.answer_calls.append((guild_id, question))
        return "Campaign answer from notes."

    def session_status(self, guild_id: int) -> str:
        return self.status_map.get(guild_id, "No active or recent session.")


def build_settings() -> Settings:
    return Settings(
        discord_bot_token="token",
        command_prefix="!",
        data_dir=Path("/tmp/scrollkeeper-tests"),
        bot_name="ScrollKeeper",
        docker_network="scrollkeeper-net",
        whisper_image="scrollkeeper-whisper:latest",
        whisper_container="scrollkeeper-whisper",
        whisper_port=9000,
        whisper_model="small.en",
        ollama_image="ollama/ollama:latest",
        ollama_container="scrollkeeper-ollama",
        ollama_port=11434,
        ollama_model="qwen3.5:9b",
        ollama_embed_model="embeddinggemma",
        ollama_idle_timeout=0,
        gpu_policy="concurrent",
    )


@unittest.skipUnless(build_bot is not None, f"discord dependency unavailable: {DISCORD_IMPORT_ERROR}")
class BotIntegrationHarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_end_session_returns_immediately_and_posts_completion_later(self) -> None:
        with (
            patch("scrollkeeper.bot.DockerServiceManager", FakeServices),
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
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
                    note_updates=[{"note_type": "Events", "title": "A", "content": "B"}],
                    transcript_path=Path("/tmp/transcript.md"),
                    summary_path=Path("/tmp/summary.md"),
                )

                await manager.completion_handler(guild.id, 1234, artifacts, None)

                self.assertEqual(len(channel.messages), 1)
                self.assertIn("## Session 42", channel.messages[0])
                self.assertIn("### Session Notes", channel.messages[0])
            finally:
                await bot.close()

    async def test_campaign_question_works_while_processing(self) -> None:
        with (
            patch("scrollkeeper.bot.DockerServiceManager", FakeServices),
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
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
            patch("scrollkeeper.bot.DockerServiceManager", FakeServices),
            patch("scrollkeeper.bot.Storage", FakeStorage),
            patch("scrollkeeper.bot.LocalAIService", FakeLocalAIService),
            patch("scrollkeeper.bot.SessionManager", FakeSessionManager),
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


if __name__ == "__main__":
    unittest.main()
