from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands
from discord.ext import voice_recv

from .config import Settings
from .llm import LocalAIService
from .health import Heartbeat, start_health_server
from .models import ENTITY_TYPES, Entity
from .session_manager import SessionManager
from .storage import Storage
from .wiki import CampaignWiki, format_change_report
from .voice_compat import apply_voice_recv_compatibility_patch


log = logging.getLogger(__name__)


class ScrollKeeperBot(commands.Bot):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heartbeat = Heartbeat()
        self._heartbeat_task: asyncio.Task | None = None

    async def setup_hook(self) -> None:
        self._heartbeat_task = asyncio.create_task(self._beat_forever())

    async def _beat_forever(self) -> None:
        while True:
            self.heartbeat.beat()
            await asyncio.sleep(10)

    async def close(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        await super().close()


def build_bot(settings: Settings) -> commands.Bot:
    apply_voice_recv_compatibility_patch()

    intents = discord.Intents.default()
    intents.message_content = True
    intents.voice_states = True
    intents.members = True

    bot = ScrollKeeperBot(
        command_prefix=settings.command_prefix,
        intents=intents,
    )
    storage = Storage(settings.data_dir)
    llm = LocalAIService(settings)
    wiki = CampaignWiki(storage, llm, settings)
    sessions = SessionManager(
        storage,
        llm,
        wiki,
        spool_dir=settings.spool_dir,
        audio_retention_days=settings.audio_retention_days,
    )
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
        if artifacts.wiki_report is not None:
            response.extend(["", format_change_report(artifacts.wiki_report)])
        for chunk in _split_long_message("\n".join(response)):
            await channel.send(chunk)

    async def post_notice(text_channel_id: int, message: str) -> None:
        channel = bot.get_channel(text_channel_id)
        if isinstance(channel, discord.TextChannel):
            await channel.send(message)

    sessions.set_completion_handler(on_session_processed)
    sessions.set_notice_handler(post_notice)

    @bot.event
    async def on_ready() -> None:
        if bot.user:
            print(f"{bot.user} is ready.")
        # Picks up sessions a restart interrupted (runs once; on_ready also fires on reconnects).
        await sessions.start()

    @bot.command(name="register-character")
    async def register_character(ctx: commands.Context, *, character_name: str) -> None:
        if ctx.guild is None or ctx.author is None:
            await ctx.reply("This command must be used in a server.")
            return
        storage.register_character(ctx.guild.id, ctx.author.id, character_name.strip())
        sessions.forget_speaker(ctx.guild.id, ctx.author.id)  # start recording them if a session is live
        await wiki.ensure_player_characters(ctx.guild.id)
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
        voice_channel = ctx.author.voice.channel
        registered = await asyncio.to_thread(storage.registered_user_ids, ctx.guild.id)
        unregistered = [
            member.display_name
            for member in getattr(voice_channel, "members", [])
            if not member.bot and member.id not in registered
        ]
        notice = [
            f"Session **#{session.session_id}** is now recording in **{voice_channel.name}**.",
            "Recording notice: the voices of players who have run `!register-character` are recorded and "
            "transcribed for the session notes. Everyone else, and bots, are not recorded.",
        ]
        if unregistered:
            notice.append(f"Not being recorded (no character registered): {', '.join(unregistered)}.")
        await ctx.reply("\n".join(notice))

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
            answer = await sessions.answer_campaign_question(
                ctx.guild.id,
                question.strip(),
                on_wait=ctx.send,
            )
        except Exception as exc:
            await ctx.send(f"Could not answer right now: {str(exc)[:1800]}")
            return
        await ctx.send(answer[:1900])

    async def _send_long(ctx: commands.Context, message: str) -> None:
        for chunk in _split_long_message(message):
            await ctx.send(chunk)

    async def _one_entity(ctx: commands.Context, ref: str) -> Entity | None:
        """Resolve `#id`, name or alias to exactly one entity, replying when that isn't possible."""
        matches = await wiki.resolve_entity(ctx.guild.id, ref)
        if not matches:
            await ctx.reply(f"No entity matches **{ref}**. Use `!entities` to list them.")
            return None
        if len(matches) > 1:
            listed = ", ".join(f"#{e.id} {e.canonical_name} ({e.type})" for e in matches)
            await ctx.reply(f"**{ref}** matches several entities: {listed}. Use the `#id` instead.")
            return None
        return matches[0]

    async def _wiki_update(ctx: commands.Context, update) -> bool:
        """Await a wiki change; the stored change survives an LLM failure, so say so instead of failing."""
        try:
            await update
            return True
        except Exception as exc:
            log.exception("Wiki page update failed")
            await ctx.reply(
                f"The change was saved, but the page could not be rewritten right now ({str(exc)[:300]}). "
                "It is retried after the next processed session."
            )
            return False

    async def _active_fact(ctx: commands.Context, fact_id: int):
        fact = await asyncio.to_thread(storage.get_fact, ctx.guild.id, fact_id)
        if fact is None:
            await ctx.reply(f"Could not find fact F{fact_id} in this server.")
            return None
        if not fact.active:
            await ctx.reply(f"Fact F{fact_id} was already corrected or retracted.")
            return None
        return fact

    @bot.command(name="entities")
    async def list_entities(ctx: commands.Context, entity_type: str | None = None) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        entities = await asyncio.to_thread(storage.list_entities, ctx.guild.id)
        if entity_type:
            wanted = entity_type.strip().casefold()
            entities = [e for e in entities if e.type.casefold() == wanted]
        if not entities:
            if entity_type:
                await ctx.reply(f"No entities of type **{entity_type}**. Types: {', '.join(ENTITY_TYPES)}.")
            else:
                await ctx.reply("No campaign wiki entities yet.")
            return
        counts = await asyncio.to_thread(storage.count_active_facts, ctx.guild.id)
        lines = ["Campaign wiki entities (`!entity <name or #id>` to read one):"]
        current_type = None
        for entity in entities:
            if entity.type != current_type:
                current_type = entity.type
                lines.extend(["", f"**{current_type}**"])
            line = f"- #{entity.id} {entity.canonical_name} ({counts.get(entity.id, 0)} facts)"
            if entity.short_description:
                line += f": {entity.short_description[:90]}"
            lines.append(line)
        await _send_long(ctx, "\n".join(lines))

    @bot.command(name="entity")
    async def show_entity(ctx: commands.Context, *, ref: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        entity = await _one_entity(ctx, ref)
        if entity is None:
            return
        await _send_long(ctx, await wiki.render_entity(ctx.guild.id, entity))

    @bot.command(name="merge-entity")
    async def merge_entity(ctx: commands.Context, source_ref: str, target_ref: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        source = await _one_entity(ctx, source_ref)
        target = await _one_entity(ctx, target_ref) if source else None
        if source is None or target is None:
            return
        if source.id == target.id:
            await ctx.reply("Those are the same entity.")
            return
        await ctx.reply(f"Merging #{source.id} {source.canonical_name} into #{target.id} {target.canonical_name}.")
        if not await _wiki_update(ctx, wiki.merge(ctx.guild.id, source, target, on_wait=ctx.send)):
            return
        await ctx.send(f"Merged. **{source.canonical_name}** is now an alias of **{target.canonical_name}** (#{target.id}).")

    @bot.command(name="rename-entity")
    async def rename_entity(ctx: commands.Context, ref: str, *, new_name: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        entity = await _one_entity(ctx, ref)
        if entity is None:
            return
        if not await _wiki_update(ctx, wiki.rename(ctx.guild.id, entity, new_name.strip(), on_wait=ctx.send)):
            return
        await ctx.reply(f"Renamed #{entity.id} to **{new_name.strip()}** (old name kept as an alias).")

    @bot.command(name="add-alias")
    async def add_alias(ctx: commands.Context, ref: str, *, alias: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        entity = await _one_entity(ctx, ref)
        if entity is None:
            return
        if await wiki.has_name(ctx.guild.id, entity, alias):
            await ctx.reply(f"#{entity.id} {entity.canonical_name} already has that name.")
            return
        if await _wiki_update(ctx, wiki.add_alias(ctx.guild.id, entity, alias.strip(), on_wait=ctx.send)):
            await ctx.reply(f"Added alias **{alias.strip()}** to #{entity.id} {entity.canonical_name}.")

    @bot.command(name="pin-fact")
    async def pin_fact(ctx: commands.Context, ref: str, *, text: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        entity = await _one_entity(ctx, ref)
        if entity is None:
            return
        fact_id = await asyncio.to_thread(
            storage.add_fact, ctx.guild.id, entity.id, text.strip(), "pinned", None, None, None, getattr(ctx.author, "id", None)
        )
        if await _wiki_update(ctx, wiki.refresh_after_change(ctx.guild.id, entity.id, on_wait=ctx.send)):
            await ctx.reply(f"Pinned F{fact_id} on #{entity.id} {entity.canonical_name}; its page was rewritten.")

    @bot.command(name="correct-fact")
    async def correct_fact(ctx: commands.Context, fact_id: int, *, text: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        fact = await _active_fact(ctx, fact_id)
        if fact is None:
            return
        new_id = await asyncio.to_thread(
            storage.add_fact, ctx.guild.id, fact.entity_id, text.strip(), "pinned", None, None, None, getattr(ctx.author, "id", None)
        )
        await asyncio.to_thread(storage.supersede_fact, fact.id, new_id)
        if await _wiki_update(ctx, wiki.refresh_after_change(ctx.guild.id, fact.entity_id, on_wait=ctx.send)):
            await ctx.reply(f"F{fact_id} is superseded by pinned fact F{new_id}; the page was rewritten.")

    @bot.command(name="retract-fact")
    async def retract_fact(ctx: commands.Context, fact_id: int, *, reason: str = "retracted by a user") -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        fact = await _active_fact(ctx, fact_id)
        if fact is None:
            return
        await asyncio.to_thread(storage.retract_fact, fact.id, reason.strip())
        if await _wiki_update(ctx, wiki.refresh_after_change(ctx.guild.id, fact.entity_id, on_wait=ctx.send)):
            await ctx.reply(f"Retracted F{fact_id}; the page was rewritten.")

    @bot.command(name="rebuild-pages")
    async def rebuild_pages(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Rebuilding every wiki page from its facts. This makes one LLM call per entity.")
        rebuilt, failures = await wiki.rebuild_all_pages(ctx.guild.id, on_wait=ctx.send)
        message = f"Rebuilt {rebuilt} wiki page(s)."
        if failures:
            message += f" Could not rebuild: {', '.join(failures)} (retried after the next processed session)."
        await ctx.send(message)

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
            f"Session **#{resolved_session_id}** is now reprocessing LLM outputs only (speech-to-text skipped). "
            "Use `!session-status` to check progress."
        )

    return bot
