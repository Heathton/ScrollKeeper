from __future__ import annotations

import asyncio

import discord
from discord.ext import commands
from discord.ext import voice_recv

from .config import Settings
from .llm import LocalAIService
from .models import CampaignNote
from .service_manager import DockerServiceManager
from .session_manager import SessionManager
from .storage import Storage
from .voice_compat import apply_voice_recv_compatibility_patch


class ScrollKeeperBot(commands.Bot):
    def __init__(self, services: DockerServiceManager, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.services = services
        self.ollama_warmup_started = False

    async def close(self) -> None:
        await self.services.shutdown()
        await super().close()


def build_bot(settings: Settings) -> commands.Bot:
    apply_voice_recv_compatibility_patch()

    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.members = True

    services = DockerServiceManager(settings)
    bot = ScrollKeeperBot(
        services,
        command_prefix=settings.command_prefix,
        intents=intents,
    )
    storage = Storage(settings.data_dir)
    llm = LocalAIService(services)
    sessions = SessionManager(storage, llm)
    discord_message_limit = 1900

    def _split_long_message(message: str, max_len: int = discord_message_limit) -> list[str]:
        if len(message) <= max_len:
            return [message]

        chunks: list[str] = []
        block = message.strip()
        while len(block) > max_len:
            split_at = block.rfind("\n\n", 0, max_len)
            if split_at < 0:
                split_at = block.rfind("\n", 0, max_len)
            if split_at < 0:
                split_at = max_len
            chunk = block[:split_at].strip()
            if chunk:
                chunks.append(chunk)
            block = block[split_at:].lstrip()
        if block:
            chunks.append(block)
        return chunks

    async def on_session_processed(
        guild_id: int,
        text_channel_id: int,
        artifacts,
        error_message: str | None,
    ) -> None:
        channel = bot.get_channel(text_channel_id)
        if not isinstance(channel, discord.TextChannel):
            guild = bot.get_guild(guild_id)
            if guild:
                fetched = guild.get_channel(text_channel_id)
                if isinstance(fetched, discord.TextChannel):
                    channel = fetched
        if not isinstance(channel, discord.TextChannel):
            return
        if error_message:
            await channel.send(f"Session processing failed: {error_message[:1800]}")
            return
        if artifacts is None:
            await channel.send("Session processing finished without generated artifacts.")
            return
        response = [
            f"## Session {artifacts.session_id}",
            "",
            "### Session Notes",
            artifacts.session_notes_markdown,
            "",
            "### Cinematic Summary",
            artifacts.cinematic_summary_markdown,
        ]
        if artifacts.note_updates:
            response.extend(["", f"Updated {len(artifacts.note_updates)} campaign notes."])
        for chunk in _split_long_message("\n".join(response)):
            await channel.send(chunk)

    sessions.set_completion_handler(on_session_processed)

    @bot.event
    async def on_ready() -> None:
        if bot.user:
            print(f"{bot.user} is ready.")
        if settings.gpu_policy == "concurrent" and not bot.ollama_warmup_started:
            bot.ollama_warmup_started = True
            asyncio.create_task(_warm_ollama())

    async def _warm_ollama() -> None:
        try:
            await services.ensure_ollama_running()
        except Exception as exc:
            print(f"Ollama warmup failed: {exc}")

    @bot.command(name="register-character")
    async def register_character(ctx: commands.Context, *, character_name: str) -> None:
        if ctx.guild is None or ctx.author is None:
            await ctx.reply("This command must be used in a server.")
            return
        storage.register_character(ctx.guild.id, ctx.author.id, character_name.strip())
        await ctx.reply(f"Registered character name: **{character_name.strip()}**")

    @bot.command(name="join")
    async def join(ctx: commands.Context) -> None:
        if ctx.guild is None or not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command must be used in a server.")
            return
        if ctx.author.voice is None or ctx.author.voice.channel is None:
            await ctx.reply("Join a voice channel first, then invite me with this command.")
            return
        channel = ctx.author.voice.channel
        existing = ctx.guild.voice_client
        if existing and existing.channel and existing.channel.id != channel.id:
            await existing.move_to(channel)
            await ctx.reply(f"Moved to voice channel **{channel.name}**.")
            return
        if existing and existing.channel and existing.channel.id == channel.id:
            await ctx.reply(f"I am already in **{channel.name}**.")
            return
        await channel.connect(cls=voice_recv.VoiceRecvClient)
        await ctx.reply(f"Joined **{channel.name}**. Use `!start-session` when you are ready.")

    @bot.command(name="start-session")
    async def start_session(ctx: commands.Context, *, title: str | None = None) -> None:
        if ctx.guild is None or not isinstance(ctx.author, discord.Member):
            await ctx.reply("This command must be used in a server.")
            return
        if ctx.author.voice is None or ctx.author.voice.channel is None:
            await ctx.reply("Join the voice channel you want recorded first.")
            return
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.reply("Use this from a server text channel.")
            return
        try:
            session = await sessions.start_session(
                guild=ctx.guild,
                voice_channel=ctx.author.voice.channel,
                text_channel=ctx.channel,
                title=title,
            )
        except RuntimeError as exc:
            await ctx.reply(str(exc))
            return
        await ctx.reply(
            f"Session **#{session.session_id}** is now recording in **{ctx.author.voice.channel.name}**."
        )

    @bot.command(name="end-session")
    async def end_session(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Ending the session. Processing will continue in the background.")
        try:
            session_id = await sessions.end_session(ctx.guild)
        except Exception as exc:
            await ctx.reply(str(exc))
            return
        await ctx.send(
            f"Session **#{session_id}** is now in processing. "
            "You can continue using `!campaign-question` while this runs."
        )

    @bot.command(name="campaign-question")
    async def campaign_question(ctx: commands.Context, *, question: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Searching campaign notes.")
        try:
            answer = await sessions.answer_campaign_question(ctx.guild.id, question.strip())
        except Exception as exc:
            await ctx.send(f"Could not answer right now: {str(exc)[:1800]}")
            return
        await ctx.send(answer[:1900])

    @bot.command(name="list-notes")
    async def list_notes(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        notes = storage.get_recent_notes(ctx.guild.id)
        if not notes:
            await ctx.reply("No campaign notes exist yet.")
            return
        lines = ["Recent campaign notes (use `!correct-note <id> <new content>` to edit):", ""]
        for note in notes[:25]:
            preview = " ".join(str(note["content"]).split())
            if len(preview) > 110:
                preview = preview[:107] + "..."
            lines.append(f"- #{note['id']} [{note['note_type']}] {note['title']}: {preview}")
        for chunk in _split_long_message("\n".join(lines)):
            await ctx.send(chunk)

    @bot.command(name="correct-note")
    async def correct_note(ctx: commands.Context, note_id: int, *, corrected_content: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        corrected = corrected_content.strip()
        if not corrected:
            await ctx.reply("Corrected content cannot be empty.")
            return
        note = storage.get_campaign_note_by_id(ctx.guild.id, note_id)
        if note is None:
            await ctx.reply(f"Could not find note #{note_id} in this server.")
            return
        metadata = {}
        try:
            # Preserve existing metadata and mark manual edits for auditability.
            import json

            metadata = json.loads(note["metadata_json"] or "{}")
            if not isinstance(metadata, dict):
                metadata = {}
        except Exception:
            metadata = {}
        metadata["manually_corrected"] = True
        metadata["corrected_by_user_id"] = getattr(ctx.author, "id", None)

        updated = storage.update_campaign_note_content(
            ctx.guild.id,
            note_id,
            corrected,
            metadata=metadata,
        )
        if not updated:
            await ctx.reply(f"Could not update note #{note_id}.")
            return
        embedding = await llm.embed_text(f"{note['note_type']}\n{note['title']}\n{corrected}")
        storage.upsert_campaign_note(
            CampaignNote(
                guild_id=ctx.guild.id,
                note_type=note["note_type"],
                title=note["title"],
                content=corrected,
                source_session_id=note["source_session_id"],
                metadata=metadata,
            ),
            embedding,
        )
        await ctx.reply(f"Updated note #{note_id}: **[{note['note_type']}] {note['title']}**")

    @bot.command(name="session-status")
    async def session_status(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply(sessions.session_status(ctx.guild.id))

    @bot.command(name="reprocess-session")
    async def reprocess_session(ctx: commands.Context, session_id: int | None = None) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Reprocessing saved session audio in the background.")
        try:
            resolved_session_id = await sessions.reprocess_session(ctx.guild.id, session_id=session_id)
        except Exception as exc:
            await ctx.send(str(exc))
            return
        await ctx.send(
            f"Session **#{resolved_session_id}** is now reprocessing from saved audio. "
            "Use `!session-status` to check progress."
        )

    @bot.command(name="reprocess-llm")
    async def reprocess_llm(ctx: commands.Context, session_id: int | None = None) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Reprocessing summaries and notes from existing transcript text.")
        try:
            resolved_session_id = await sessions.reprocess_llm_only(ctx.guild.id, session_id=session_id)
        except Exception as exc:
            await ctx.send(str(exc))
            return
        await ctx.send(
            f"Session **#{resolved_session_id}** is now reprocessing LLM outputs only (Whisper skipped). "
            "Use `!session-status` to check progress."
        )

    return bot
