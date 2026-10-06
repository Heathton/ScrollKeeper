from __future__ import annotations

import asyncio
import logging

import discord
from discord.ext import commands
from discord.ext import voice_recv

from .config import Settings
from .llm import LocalAIService
from .health import Heartbeat, start_health_server
from .journal import MAX_UPLOAD_BYTES, JournalImporter, parse_rules
from .models import ENTITY_TYPES, Entity
from .embeddings import EMBEDDING_MODELS, LocalEmbedder
from .search import SearchIndex
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
    embedder = LocalEmbedder(
        EMBEDDING_MODELS[settings.embed_model],
        settings.embed_model_dir or settings.data_dir / "models",
        threads=settings.embed_threads,
    )
    search = SearchIndex(storage, llm, embedder)
    wiki = CampaignWiki(storage, llm, settings, search=search)
    journal = JournalImporter(storage, wiki, parse_rules(settings.journal_rules))
    sessions = SessionManager(
        storage,
        llm,
        wiki,
        spool_dir=settings.spool_dir,
        audio_retention_days=settings.audio_retention_days,
        search=search,
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
            f"## Session {artifacts.session_number or artifacts.session_id}"
            + (f" ({artifacts.session_date})" if artifacts.session_date else ""),
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

    async def post_long_notice(text_channel_id: int, message: str) -> None:
        for chunk in _split_long_message(message):
            await post_notice(text_channel_id, chunk)

    journal.set_notice_handler(post_long_notice)

    @bot.event
    async def on_ready() -> None:
        if bot.user:
            print(f"{bot.user} is ready.")
        # Picks up sessions a restart interrupted (runs once; on_ready also fires on reconnects).
        await sessions.start()
        # Resumes a journal import a restart interrupted.
        await journal.start()
        # Loads (first time: downloads) the embedding model and indexes anything new, in the background.
        search.start()

    async def _campaign_id(ctx: commands.Context) -> int:
        return (await sessions.active_campaign(ctx.guild.id)).id

    @bot.command(name="switch-campaign")
    async def switch_campaign(ctx: commands.Context, *, campaign_name: str) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        try:
            campaign, created = await sessions.switch_campaign(ctx.guild.id, campaign_name)
        except RuntimeError as exc:
            await ctx.reply(str(exc))
            return
        if created:
            await ctx.reply(
                f"Created campaign **{campaign.name}** and made it active. "
                "Players need to `!register-character` for this campaign before they are recorded."
            )
        else:
            await ctx.reply(f"Active campaign is now **{campaign.name}**.")

    @bot.command(name="current-campaign")
    async def current_campaign(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        campaign = await sessions.active_campaign(ctx.guild.id)
        await ctx.reply(f"Active campaign: **{campaign.name}**.")

    @bot.command(name="list-campaigns")
    async def list_campaigns(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await sessions.active_campaign(ctx.guild.id)  # creates "Default" on first use
        campaigns = await sessions.list_campaigns(ctx.guild.id)
        lines = ["Campaigns (`!switch-campaign <name>` to change):"]
        lines.extend(f"- **{c.name}** (active)" if c.is_active else f"- {c.name}" for c in campaigns)
        await _send_long(ctx, "\n".join(lines))

    @bot.command(name="register-character")
    async def register_character(ctx: commands.Context, *, character_name: str) -> None:
        if ctx.guild is None or ctx.author is None:
            await ctx.reply("This command must be used in a server.")
            return
        campaign = await sessions.active_campaign(ctx.guild.id)
        await asyncio.to_thread(storage.register_character, campaign.id, ctx.author.id, character_name.strip())
        sessions.forget_speaker(ctx.guild.id, ctx.author.id)  # start recording them if a session is live
        await wiki.ensure_player_characters(campaign.id)
        await ctx.reply(f"Registered character name **{character_name.strip()}** in campaign **{campaign.name}**.")

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
        registered = await asyncio.to_thread(storage.registered_user_ids, session.campaign_id)
        campaign = await asyncio.to_thread(storage.get_campaign, session.campaign_id)
        unregistered = [
            member.display_name
            for member in getattr(voice_channel, "members", [])
            if not member.bot and member.id not in registered
        ]
        notice = [
            f"Session **#{session.number or session.session_id}** is now recording in **{voice_channel.name}** "
            f"for campaign **{campaign.name if campaign else session.campaign_id}**.",
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
            session_number = await sessions.end_session(ctx.guild)
        except Exception as exc:
            await ctx.reply(str(exc))
            return
        await ctx.send(
            f"Session **#{session_number}** is now in processing. "
            "You can continue using `!campaign-question` while this runs."
        )

    async def _answer(ctx: commands.Context, question: str, deep: bool) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Searching campaign notes and session transcripts." if deep else "Searching campaign notes.")
        try:
            answer = await sessions.answer_campaign_question(
                await _campaign_id(ctx), question.strip(), on_wait=ctx.send, deep=deep
            )
        except Exception as exc:
            log.exception("Could not answer a campaign question")
            await ctx.send(f"Could not answer right now: {str(exc)[:1800]}")
            return
        await _send_long(ctx, answer)

    @bot.command(name="campaign-question")
    async def campaign_question(ctx: commands.Context, *, question: str) -> None:
        await _answer(ctx, question, deep=False)

    @bot.command(name="deep-question")
    async def deep_question(ctx: commands.Context, *, question: str) -> None:
        await _answer(ctx, question, deep=True)

    @bot.command(name="reindex")
    async def reindex(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        if not _can_manage(ctx):
            await ctx.reply("Only members with the Manage Server permission can rebuild the search index.")
            return
        await ctx.reply("Rebuilding the search index: every wiki page and session summary is re-embedded on CPU.")
        try:
            total, pending = await search.reindex(await _campaign_id(ctx))
        except Exception as exc:
            log.exception("Reindex failed")
            await ctx.send(f"Reindex failed: {str(exc)[:1800]}")
            return
        message = f"Indexed {total} document(s)."
        if pending:
            message += (
                f" {pending} could not be embedded (the embedding model is not available; see the bot log). "
                "Name and keyword search still find them."
            )
        await ctx.send(message)

    def _can_manage(ctx: commands.Context) -> bool:
        permissions = getattr(ctx.author, "guild_permissions", None)
        return bool(getattr(permissions, "manage_guild", False))

    async def _send_long(ctx: commands.Context, message: str) -> None:
        for chunk in _split_long_message(message):
            await ctx.send(chunk)

    async def _one_entity(ctx: commands.Context, ref: str) -> Entity | None:
        """Resolve `#id`, name or alias to exactly one entity, replying when that isn't possible."""
        matches = await wiki.resolve_entity(await _campaign_id(ctx), ref)
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
        fact = await asyncio.to_thread(storage.get_fact, await _campaign_id(ctx), fact_id)
        if fact is None:
            await ctx.reply(f"Could not find fact F{fact_id} in this campaign.")
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
        campaign_id = await _campaign_id(ctx)
        entities = await asyncio.to_thread(storage.list_entities, campaign_id)
        if entity_type:
            wanted = entity_type.strip().casefold()
            entities = [e for e in entities if e.type.casefold() == wanted]
        if not entities:
            if entity_type:
                await ctx.reply(f"No entities of type **{entity_type}**. Types: {', '.join(ENTITY_TYPES)}.")
            else:
                await ctx.reply("No campaign wiki entities yet.")
            return
        counts = await asyncio.to_thread(storage.count_active_facts, campaign_id)
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
        await _send_long(ctx, await wiki.render_entity(entity.campaign_id, entity))

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
        if not await _wiki_update(ctx, wiki.merge(target.campaign_id, source, target, on_wait=ctx.send)):
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
        if not await _wiki_update(ctx, wiki.rename(entity.campaign_id, entity, new_name.strip(), on_wait=ctx.send)):
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
        if await wiki.has_name(entity.campaign_id, entity, alias):
            await ctx.reply(f"#{entity.id} {entity.canonical_name} already has that name.")
            return
        if await _wiki_update(ctx, wiki.add_alias(entity.campaign_id, entity, alias.strip(), on_wait=ctx.send)):
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
            storage.add_fact, entity.campaign_id, entity.id, text.strip(), "pinned", None, None, None, getattr(ctx.author, "id", None)
        )
        if await _wiki_update(ctx, wiki.refresh_after_change(entity.campaign_id, entity.id, on_wait=ctx.send)):
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
            storage.add_fact, fact.campaign_id, fact.entity_id, text.strip(), "pinned", None, None, None, getattr(ctx.author, "id", None)
        )
        await asyncio.to_thread(storage.supersede_fact, fact.id, new_id)
        if await _wiki_update(ctx, wiki.refresh_after_change(fact.campaign_id, fact.entity_id, on_wait=ctx.send)):
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
        if await _wiki_update(ctx, wiki.refresh_after_change(fact.campaign_id, fact.entity_id, on_wait=ctx.send)):
            await ctx.reply(f"Retracted F{fact_id}; the page was rewritten.")

    @bot.command(name="rebuild-pages")
    async def rebuild_pages(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Rebuilding every wiki page from its facts. This makes one LLM call per entity.")
        rebuilt, failures = await wiki.rebuild_all_pages(await _campaign_id(ctx), on_wait=ctx.send)
        message = f"Rebuilt {rebuilt} wiki page(s)."
        if failures:
            message += f" Could not rebuild: {', '.join(failures)} (retried after the next processed session)."
        await ctx.send(message)

    @bot.command(name="import-journal")
    async def import_journal(ctx: commands.Context, *, options: str = "") -> None:
        # Open to every member for now: the game master who runs imports may not administer the server.
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        option = options.strip().lower()
        attachments = getattr(ctx.message, "attachments", [])
        if not attachments or option not in {"", "preview"}:
            await ctx.reply(
                "Attach a journal export (JSON) to `!import-journal` to import it into the active campaign, "
                "or to `!import-journal preview` to see what it would import first."
            )
            return
        attachment = attachments[0]
        if attachment.size > MAX_UPLOAD_BYTES:
            await ctx.reply(f"That file is too large ({attachment.size // (1024 * 1024)} MB).")
            return
        raw = await attachment.read()
        if option == "preview":
            try:
                preview = await asyncio.to_thread(journal.preview, raw)
            except ValueError as exc:
                await ctx.reply(str(exc))
                return
            await _send_long(ctx, preview)
            return
        campaign = await sessions.active_campaign(ctx.guild.id)
        try:
            plan = await journal.begin(ctx.guild.id, campaign.id, ctx.channel.id, raw)
        except (ValueError, RuntimeError) as exc:
            await ctx.reply(str(exc))
            return
        await ctx.reply(
            f"Importing {len(plan.entries)} journal entries into campaign **{campaign.name}** in the background. "
            "It makes an LLM call per long entry, recap and changed page, so a large journal takes hours. "
            "`!import-status` shows progress; the report is posted here when it finishes."
        )

    @bot.command(name="import-status")
    async def import_status(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply(journal.status(ctx.guild.id))

    @bot.command(name="session-status")
    async def session_status(ctx: commands.Context) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply(sessions.session_status(ctx.guild.id))

    @bot.command(name="reprocess-session")
    async def reprocess_session(ctx: commands.Context, number: int | None = None) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Reprocessing saved session audio in the background.")
        try:
            resolved = await sessions.reprocess_session(ctx.guild.id, number)
        except Exception as exc:
            await ctx.send(str(exc))
            return
        await ctx.send(
            f"Session **#{resolved}** is now reprocessing from saved audio. "
            "Use `!session-status` to check progress."
        )

    @bot.command(name="reprocess-llm")
    async def reprocess_llm(ctx: commands.Context, number: int | None = None) -> None:
        if ctx.guild is None:
            await ctx.reply("This command must be used in a server.")
            return
        await ctx.reply("Reprocessing summaries and notes from existing transcript text.")
        try:
            resolved = await sessions.reprocess_llm_only(ctx.guild.id, number)
        except Exception as exc:
            await ctx.send(str(exc))
            return
        await ctx.send(
            f"Session **#{resolved}** is now reprocessing LLM outputs only (speech-to-text skipped). "
            "Use `!session-status` to check progress."
        )

    return bot
