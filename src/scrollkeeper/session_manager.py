from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import sqlite3
import tempfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

import discord
from discord.ext import voice_recv

from .audio import LEGACY_OPUS_EXTENSION, convert_legacy_clips, decode_for_transcription, ogg_duration_seconds
from .llm import LocalAIService
from .models import Campaign, SessionArtifacts, SpeakerSegment, WikiChangeReport
from .recorder import Speaker, TrackRecorder
from .storage import PROCESS_LLM_ONLY, PROCESS_TRANSCRIBE, Storage
from .transcript import TranscriptLine, format_offset, merge_lines, render_compact, render_timed, split_utterances
from .search import SearchIndex
from .wiki import CampaignWiki


log = logging.getLogger(__name__)

# A session whose processing was started this many times without finishing (the bot died each
# time) is marked failed instead of being retried again on the next start.
MAX_PROCESSING_ATTEMPTS = 3
RETENTION_CHECK_SECONDS = 6 * 3600

CompletionHandler = Callable[[int, int, SessionArtifacts | None, str | None], Awaitable[None]]
NoticeHandler = Callable[[int, str], Awaitable[None]]

__all__ = ["ActiveSession", "SessionManager", "format_offset"]


@dataclass(slots=True)
class ActiveSession:
    session_id: int
    guild_id: int
    campaign_id: int
    voice_channel_id: int
    text_channel_id: int
    title: str | None
    base_dir: Path
    audio_dir: Path
    started_at: datetime
    record_dir: Path | None = None
    voice_client: voice_recv.VoiceRecvClient | None = None
    sink: "SessionAudioSink | None" = None
    recorder: TrackRecorder | None = None
    reconnect_task: asyncio.Task | None = None
    interrupted_at: datetime | None = None
    closed: bool = False


@dataclass(slots=True)
class SessionStatus:
    state: str
    session_id: int | None = None
    message: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)


class SessionAudioSink(voice_recv.AudioSink):
    """Receives Opus packets on voice_recv's packet thread and hands them to the session recorder.

    Only users who ran `!register-character` are recorded; bots never are. Who a user is gets
    decided once, on their first packet, and cached for the session.
    """

    def __init__(self, manager: "SessionManager", session: ActiveSession) -> None:
        super().__init__()
        self.manager = manager
        self.session = session
        self.speakers: dict[int, Speaker | None] = {}

    def wants_opus(self) -> bool:
        return True

    def write(self, user: discord.User | discord.Member | None, data: voice_recv.VoiceData) -> None:
        packet = getattr(data, "packet", None)
        payload = getattr(data, "opus", None)
        # Lost packets arrive as empty fake packets; their time becomes silence on the track.
        if user is None or packet is None or not payload:
            return
        speaker = self._speaker(user)
        recorder = self.session.recorder
        if speaker is None or recorder is None:
            return
        recorder.submit(speaker, packet.ssrc, packet.timestamp, bytes(payload))

    def _speaker(self, user: discord.User | discord.Member) -> Speaker | None:
        if user.id in self.speakers:
            return self.speakers[user.id]
        speaker: Speaker | None = None
        display_name = getattr(user, "display_name", None) or getattr(user, "name", str(user.id))
        if not getattr(user, "bot", False):
            character_name = self.manager.storage.get_registered_character_name(self.session.campaign_id, user.id)
            if character_name:
                speaker = Speaker(user.id, display_name, character_name)
            else:
                self.manager.post_threadsafe(
                    self.session.text_channel_id,
                    f"**{display_name}** is speaking but hasn't run `!register-character`, so they are not being recorded.",
                )
        self.speakers[user.id] = speaker
        return speaker

    def cleanup(self) -> None:
        # The recorder belongs to the session and is closed when the session ends, not when
        # voice_recv stops listening (which also happens around reconnects).
        pass


class SessionManager:
    def __init__(
        self,
        storage: Storage,
        llm: LocalAIService,
        wiki: CampaignWiki,
        spool_dir: Path | None = None,
        audio_retention_days: int = 0,
        search: SearchIndex | None = None,
    ) -> None:
        self.storage = storage
        self.llm = llm
        self.wiki = wiki
        self.search = search
        self.spool_dir = spool_dir
        self.audio_retention_days = audio_retention_days
        self.active_sessions: dict[int, ActiveSession] = {}
        self.processing_tasks: dict[int, asyncio.Task] = {}
        self.statuses: dict[int, SessionStatus] = {}
        self._completion_handler: CompletionHandler | None = None
        self._notice_handler: NoticeHandler | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._retention_task: asyncio.Task | None = None

    def set_completion_handler(self, handler: CompletionHandler) -> None:
        self._completion_handler = handler

    def set_notice_handler(self, handler: NoticeHandler) -> None:
        """Handler(text_channel_id, message) used to post "still waiting" notices during processing."""
        self._notice_handler = handler

    def _channel_notifier(self, text_channel_id: int) -> Callable[[str], Awaitable[None]]:
        async def notify(message: str) -> None:
            if self._notice_handler:
                try:
                    await self._notice_handler(text_channel_id, message)
                except Exception:
                    log.warning("Could not post notice to channel %s", text_channel_id, exc_info=True)

        return notify

    def post_threadsafe(self, text_channel_id: int, message: str) -> None:
        """Post a channel notice from a non-asyncio thread (the voice packet thread)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        notify = self._channel_notifier(text_channel_id)
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(notify(message)))

    async def start(self) -> None:
        """Recover sessions a restart interrupted and start background jobs. Call once the bot is ready."""
        if self._started:
            return
        self._started = True
        self._loop = asyncio.get_running_loop()
        notices, guild_ids = await asyncio.to_thread(self._recover_after_restart)
        for text_channel_id, message in notices:
            await self._channel_notifier(text_channel_id)(message)
        for guild_id in guild_ids:
            self._ensure_worker(guild_id)
        if self.audio_retention_days > 0:
            self._retention_task = asyncio.create_task(self._retention_loop())

    def _recover_after_restart(self) -> tuple[list[tuple[int, str]], set[int]]:
        notices: list[tuple[int, str]] = []
        guild_ids: set[int] = set()
        for row in self.storage.get_sessions_with_status("recording"):
            session_id = int(row["id"])
            stopped_at = self._last_audio_write(row)
            self.storage.mark_session_interrupted(session_id, stopped_at)
            guild_ids.add(int(row["guild_id"]))
            log.warning("Session %s was recording when the bot stopped; queued with audio up to %s", session_id, stopped_at)
            offset = format_offset(stopped_at - datetime.fromisoformat(row["started_at"]))
            notices.append(
                (
                    int(row["text_channel_id"]),
                    f"Session **#{session_id}** was still recording when I restarted. The audio up to {offset} "
                    "into the session was saved and is being processed; anything after that was not recorded.",
                )
            )
        for row in self.storage.get_sessions_with_status("processing"):
            guild_ids.add(int(row["guild_id"]))
            if row["interrupted_at"] is None or int(row["processing_attempts"]) > 0:
                notices.append(
                    (int(row["text_channel_id"]), f"Resuming processing of session **#{row['id']}** after a restart.")
                )
        return notices, guild_ids

    def _last_audio_write(self, row: sqlite3.Row) -> datetime:
        latest = datetime.fromisoformat(row["started_at"])
        for track in self.storage.get_session_tracks(int(row["id"])):
            path = Path(track["path"])
            with contextlib.suppress(OSError):
                latest = max(latest, datetime.utcfromtimestamp(path.stat().st_mtime))
        return latest

    def forget_speaker(self, guild_id: int, user_id: int) -> None:
        """Re-check a user on their next packet (e.g. they just registered a character mid-session)."""
        session = self.active_sessions.get(guild_id)
        if session is not None and session.sink is not None:
            session.sink.speakers.pop(user_id, None)

    # --- Campaigns -------------------------------------------------------------------------

    async def active_campaign(self, guild_id: int) -> Campaign:
        return await asyncio.to_thread(self.storage.active_campaign, guild_id)

    async def list_campaigns(self, guild_id: int) -> list[Campaign]:
        return await asyncio.to_thread(self.storage.list_campaigns, guild_id)

    async def switch_campaign(self, guild_id: int, name: str) -> tuple[Campaign, bool]:
        """Make `name` the active campaign (creating it if needed). Returns (campaign, created).

        Not allowed while a session records: its players registered characters in the current
        campaign. Queued and processing sessions are fine; each keeps the campaign it was recorded in.
        """
        existing = self.active_sessions.get(guild_id)
        if existing is not None and not existing.closed:
            raise RuntimeError("Cannot switch campaigns while a session is recording. Use `!end-session` first.")
        try:
            return await asyncio.to_thread(self.storage.switch_campaign, guild_id, name)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc

    async def start_session(
        self,
        guild: discord.Guild,
        voice_channel: discord.VoiceChannel,
        text_channel: discord.TextChannel,
        title: str | None,
    ) -> ActiveSession:
        existing = self.active_sessions.get(guild.id)
        if existing and not existing.closed:
            raise RuntimeError("A session is already active in this server.")
        self._loop = asyncio.get_running_loop()

        campaign = await asyncio.to_thread(self.storage.active_campaign, guild.id)
        session_id = await asyncio.to_thread(
            self.storage.create_session,
            guild_id=guild.id,
            voice_channel_id=voice_channel.id,
            text_channel_id=text_channel.id,
            title=title,
            campaign_id=campaign.id,
        )
        row = await asyncio.to_thread(self.storage.get_session, session_id)
        session = self._session_from_row(row)
        session.closed = False
        session.record_dir = self._record_dir(session_id, session.audio_dir)
        session.recorder = TrackRecorder(self.storage, session_id, session.record_dir)
        await asyncio.to_thread(session.recorder.start)

        try:
            voice_client = await self._connect_voice(voice_channel)
            sink = SessionAudioSink(self, session)
            voice_client.listen(sink)
        except Exception:
            await asyncio.to_thread(session.recorder.stop)
            await asyncio.to_thread(self.storage.set_session_status, session_id, "failed")
            raise
        session.voice_client = voice_client
        session.sink = sink
        session.reconnect_task = asyncio.create_task(self._monitor_voice_connection(guild, session))
        self.active_sessions[guild.id] = session
        self._set_status(guild.id, "recording", session_id, "Recording in progress.")
        return session

    async def end_session(self, guild: discord.Guild) -> int:
        session = self.active_sessions.get(guild.id)
        if session is None or session.closed:
            raise RuntimeError("No active session for this server.")

        log.info("Ending session %s for guild %s", session.session_id, guild.id)
        await self._stop_recording_session(session)
        self.active_sessions.pop(guild.id, None)
        await asyncio.to_thread(self.storage.queue_session, session.session_id, PROCESS_TRANSCRIBE)
        self._set_status(guild.id, "processing", session.session_id, "Transcribing and generating notes.")
        self._ensure_worker(guild.id)
        return session.session_id

    async def reprocess_session(self, guild_id: int, session_id: int | None = None) -> int:
        row = await self._session_row_for_reprocess(guild_id, session_id)
        resolved_session_id = int(row["id"])
        if row["audio_deleted_at"]:
            raise RuntimeError(f"Session #{resolved_session_id}'s audio was deleted (audio retention), so it can't be re-transcribed.")
        tracks = await asyncio.to_thread(self.storage.get_session_tracks, resolved_session_id)
        audio_dir = self.storage.sessions_dir / str(resolved_session_id) / "audio"
        if not tracks and not audio_dir.exists():
            raise RuntimeError(f"Session #{resolved_session_id} has no recorded audio to reprocess.")

        await asyncio.to_thread(self.storage.reset_session_processing, resolved_session_id)
        self._set_status(
            guild_id,
            "processing",
            resolved_session_id,
            "Reprocessing saved audio and regenerating notes.",
        )
        self._ensure_worker(guild_id)
        return resolved_session_id

    async def reprocess_llm_only(self, guild_id: int, session_id: int | None = None) -> int:
        row = await self._session_row_for_reprocess(guild_id, session_id)
        resolved_session_id = int(row["id"])
        await asyncio.to_thread(self.storage.reset_session_llm_processing, resolved_session_id)
        self._set_status(
            guild_id,
            "processing",
            resolved_session_id,
            "Reprocessing summaries and notes from existing transcript text.",
        )
        self._ensure_worker(guild_id)
        return resolved_session_id

    async def _session_row_for_reprocess(self, guild_id: int, session_id: int | None) -> sqlite3.Row:
        if guild_id in self.active_sessions:
            raise RuntimeError("Cannot reprocess while a live session is recording in this server.")
        if guild_id in self.processing_tasks:
            raise RuntimeError("A session is already processing in this server.")
        if session_id is not None:
            row = await asyncio.to_thread(self.storage.get_session, session_id)
        else:
            campaign = await asyncio.to_thread(self.storage.active_campaign, guild_id)
            row = await asyncio.to_thread(self.storage.get_latest_session, campaign.id)
        if row is None:
            raise RuntimeError("No saved session was found to reprocess in the active campaign.")
        if int(row["guild_id"]) != guild_id:
            raise RuntimeError("That session does not belong to this server.")
        if row["journal_id"]:
            raise RuntimeError(
                f"Session #{row['id']} is a recap imported from the journal; run `!import-journal` again to update it."
            )
        return row

    async def answer_campaign_question(
        self,
        campaign_id: int,
        question: str,
        on_wait: Callable[[str], Awaitable[None]] | None = None,
        deep: bool = False,
    ) -> str:
        if self.search is None:
            raise RuntimeError("Campaign search is not configured.")
        return await self.search.answer(campaign_id, question, deep=deep, on_wait=on_wait)

    def session_status(self, guild_id: int) -> str:
        status = self.statuses.get(guild_id)
        if status is None:
            return "No active or recent session."
        session_label = f"Session #{status.session_id}" if status.session_id else "Session"
        if status.message:
            return f"{session_label} status: {status.state}. {status.message}"
        return f"{session_label} status: {status.state}."

    # --- Processing queue (sessions with status 'processing', oldest first, one per guild) -----

    def _ensure_worker(self, guild_id: int) -> None:
        task = self.processing_tasks.get(guild_id)
        if task is not None and not task.done():
            return
        task = asyncio.create_task(self._process_queue(guild_id))
        self.processing_tasks[guild_id] = task

        def _done(finished: asyncio.Task) -> None:
            if self.processing_tasks.get(guild_id) is finished:
                self.processing_tasks.pop(guild_id, None)

        task.add_done_callback(_done)

    async def _process_queue(self, guild_id: int) -> None:
        while True:
            row = await asyncio.to_thread(self.storage.next_queued_session, guild_id)
            if row is None:
                return
            session = self._session_from_row(row)
            attempts = await asyncio.to_thread(self.storage.begin_processing_attempt, session.session_id)
            if attempts > MAX_PROCESSING_ATTEMPTS:
                message = (
                    f"Processing was interrupted {attempts - 1} times; giving up. "
                    "Check the bot logs, then use `!reprocess-session` to try again."
                )
                log.error("Session %s: %s", session.session_id, message)
                await asyncio.to_thread(self.storage.set_session_status, session.session_id, "failed")
                self._set_status(guild_id, "failed", session.session_id, message)
                await self._report_completion(guild_id, session, None, message)
                continue
            transcribe = row["processing_kind"] != PROCESS_LLM_ONLY
            self._set_status(guild_id, "processing", session.session_id, "Transcribing and generating notes.")
            await self._process_session_job(guild_id, session, transcribe_audio=transcribe)

    async def _process_session_job(self, guild_id: int, session: ActiveSession, transcribe_audio: bool = True) -> None:
        artifacts: SessionArtifacts | None = None
        error_message: str | None = None
        try:
            log.info("Starting post-session processing for session %s", session.session_id)
            artifacts = await self._process_closed_session(guild_id, session, transcribe_audio=transcribe_audio)
            self._set_status(
                guild_id,
                "completed",
                session.session_id,
                "Session processing is complete.",
            )
            log.info("Completed post-session processing for session %s", session.session_id)
        except Exception as exc:
            error_message = str(exc)
            log.exception("Session %s processing failed", session.session_id)
            await asyncio.to_thread(self.storage.set_session_status, session.session_id, "failed")
            self._set_status(
                guild_id,
                "failed",
                session.session_id,
                f"Session processing failed: {error_message}",
            )
        await self._report_completion(guild_id, session, artifacts, error_message)

    async def _report_completion(
        self, guild_id: int, session: ActiveSession, artifacts: SessionArtifacts | None, error_message: str | None
    ) -> None:
        # A failed Discord post must not stop the queue worker from taking the next session.
        if self._completion_handler:
            try:
                await self._completion_handler(guild_id, session.text_channel_id, artifacts, error_message)
            except Exception:
                log.exception("Could not report the result of session %s", session.session_id)

    async def _process_closed_session(
        self,
        guild_id: int,
        session: ActiveSession,
        transcribe_audio: bool = True,
    ) -> SessionArtifacts:
        notify = self._channel_notifier(session.text_channel_id)
        if transcribe_audio:
            await self._transcribe_session(guild_id, session, notify)
        else:
            segments = await asyncio.to_thread(self.storage.get_session_segments, session.session_id)
            transcribed_segments = sum(1 for segment in segments if (segment["transcript_text"] or "").strip())
            if transcribed_segments == 0:
                raise RuntimeError(
                    "No transcript text exists for this session yet. Run `!reprocess-session` first."
                )
            log.info(
                "Skipping speech-to-text for session %s and reusing %s existing transcript segment(s)",
                session.session_id,
                transcribed_segments,
            )

        lines = await asyncio.to_thread(self._transcript_lines, session)
        note = self._interruption_note(session)
        transcript_markdown = render_compact(lines, note)
        transcript_path = session.base_dir / "transcript.md"
        summary_path = session.base_dir / "summary.md"
        # Written before the summary, so a failed LLM step still leaves a readable transcript.
        await asyncio.to_thread(self._write_output, transcript_path, transcript_markdown)
        glossary = await asyncio.to_thread(self.wiki.spelling_glossary, session.campaign_id)
        log.info("Generating summary for session %s (%s names in the spelling glossary)", session.session_id, len(glossary))
        self._set_status(guild_id, "processing", session.session_id, "Writing the session summary.")
        summary_payload = await self.llm.summarize_session(lines, note, glossary, on_wait=notify)
        session_notes = summary_payload["session_notes_markdown"].strip()
        cinematic = summary_payload["cinematic_summary_markdown"].strip()

        summary_markdown = (
            "# Session Notes\n\n"
            f"{session_notes}\n\n"
            "# Cinematic Summary\n\n"
            f"{cinematic}\n"
        )
        await asyncio.to_thread(self._write_output, summary_path, summary_markdown)

        # The summary is saved already; a wiki failure is reported but doesn't fail the session.
        log.info("Updating the campaign wiki from session %s", session.session_id)
        try:
            timed_transcript = await asyncio.to_thread(self._build_timed_transcript, session)
            wiki_report = await self.wiki.process_session(
                session.campaign_id,
                session.session_id,
                timed_transcript,
                on_wait=notify,
                on_progress=lambda message: self._set_status(guild_id, "processing", session.session_id, message),
            )
        except Exception as exc:
            log.exception("Campaign wiki update failed for session %s", session.session_id)
            wiki_report = WikiChangeReport(error=str(exc))

        await asyncio.to_thread(
            self.storage.finalize_session,
            session.session_id,
            str(transcript_path),
            str(summary_path),
        )
        # Index the new summary and transcript; a failure only delays it to the next change.
        if self.search is not None:
            try:
                await self.search.refresh(session.campaign_id)
            except Exception:
                log.exception("Could not index session %s for search", session.session_id)
        return SessionArtifacts(
            session_id=session.session_id,
            transcript_markdown=transcript_markdown,
            session_notes_markdown=session_notes,
            cinematic_summary_markdown=cinematic,
            transcript_path=transcript_path,
            summary_path=summary_path,
            wiki_report=wiki_report,
        )

    @staticmethod
    def _write_output(path: Path, markdown: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")

    # --- Speech-to-text, one whole track per speaker -----------------------------------------

    async def _transcribe_session(
        self,
        guild_id: int,
        session: ActiveSession,
        notify: Callable[[str], Awaitable[None]],
    ) -> None:
        tracks = await asyncio.to_thread(self._tracks_for_processing, session)
        if not tracks:
            raise RuntimeError(
                "No audio was recorded for this session. Only players who have run `!register-character` are recorded."
            )
        durations = await asyncio.to_thread(self._track_durations, tracks)
        total = timedelta(seconds=int(sum(durations)))
        log.info("Session %s: transcribing %s track(s), %s of audio", session.session_id, len(tracks), total)
        await notify(
            f"Transcribing {len(tracks)} speaker track(s) ({format_offset(total)} of audio in total). "
            "This can take a while; `!session-status` shows progress."
        )
        transcribed = 0
        skipped: list[str] = []
        scratch_parent = self.spool_dir if self.spool_dir is not None else None
        if scratch_parent is not None:
            await asyncio.to_thread(scratch_parent.mkdir, parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="scrollkeeper-stt-", dir=scratch_parent) as scratch:
            for index, track in enumerate(tracks, start=1):
                name = track["character_name"]
                self._set_status(
                    guild_id,
                    "processing",
                    session.session_id,
                    f"Transcribing track {index}/{len(tracks)} ({name}, {format_offset(timedelta(seconds=int(durations[index - 1])))}).",
                )
                source = Path(track["path"])
                flac_path = Path(scratch) / f"track-{track['id']}.flac"
                try:
                    if not source.exists():
                        raise RuntimeError(f"{source} is missing")
                    await asyncio.to_thread(decode_for_transcription, source, flac_path)
                except Exception as exc:
                    # A track with no complete audio page (e.g. cut off within its first second)
                    # can't be decoded; skip it rather than lose the rest of the session.
                    log.warning("Session %s: skipping undecodable track %s: %s", session.session_id, source, exc)
                    skipped.append(name)
                    continue
                try:
                    result = await self.llm.transcribe_track(flac_path)
                finally:
                    with contextlib.suppress(OSError):
                        await asyncio.to_thread(flac_path.unlink)
                await asyncio.to_thread(self._store_track_transcript, session, track, result)
                transcribed += 1
        if transcribed == 0:
            raise RuntimeError("No speaker track could be decoded for transcription.")
        if skipped:
            await notify(f"Could not decode the audio of: {', '.join(skipped)}. The transcript is missing them.")

    def _store_track_transcript(self, session: ActiveSession, track: sqlite3.Row, result) -> None:
        track_start = datetime.fromisoformat(track["started_at"])
        user_id = int(track["user_id"])
        # Use the current registry name, so a character renamed since the recording is fixed on reprocess.
        name = self.storage.get_registered_character_name(session.campaign_id, user_id) or track["character_name"]
        segments = [
            SpeakerSegment(
                discord_user_id=user_id,
                discord_display_name=track["display_name"],
                character_name=name,
                started_at=track_start + timedelta(seconds=utterance.start),
                ended_at=track_start + timedelta(seconds=utterance.end),
                audio_path=Path(track["path"]),
                transcript_text=utterance.text,
                track_id=int(track["id"]),
            )
            for utterance in split_utterances(result)
        ]
        self.storage.replace_track_segments(track, name, segments)
        log.info("Session %s: %s utterance(s) from %s", session.session_id, len(segments), name)

    @staticmethod
    def _track_durations(tracks: list[sqlite3.Row]) -> list[float]:
        durations = []
        for track in tracks:
            try:
                durations.append(ogg_duration_seconds(Path(track["path"])))
            except OSError:
                durations.append(0.0)
        return durations

    def _tracks_for_processing(self, session: ActiveSession) -> list[sqlite3.Row]:
        """The session's tracks, moved from the recording spool into the archive (blocking)."""
        tracks = self.storage.get_session_tracks(session.session_id)
        if not tracks:
            self._convert_legacy_session(session)
            return self.storage.get_session_tracks(session.session_id)
        for track in tracks:
            source = Path(track["path"])
            destination = session.audio_dir / source.name
            if source == destination:
                continue
            if source.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                log.info("Moved %s to the archive at %s", source, destination)
            if destination.exists():
                self.storage.update_track_path(int(track["id"]), destination)
        if self.spool_dir is not None:
            with contextlib.suppress(OSError):
                (self.spool_dir / str(session.session_id)).rmdir()
        return self.storage.get_session_tracks(session.session_id)

    def _convert_legacy_session(self, session: ActiveSession) -> None:
        """Turn a pre-track session's `.skopus` clips into one Ogg Opus track per speaker (once)."""
        clips: dict[int, list[sqlite3.Row]] = {}
        for row in self.storage.get_session_segments(session.session_id):
            if row["track_id"] is None and str(row["audio_path"]).endswith(LEGACY_OPUS_EXTENSION):
                clips.setdefault(int(row["user_id"]), []).append(row)
        for user_id, rows in clips.items():
            out_path = session.audio_dir / f"{user_id}.ogg"
            started_at = convert_legacy_clips(
                [(datetime.fromisoformat(row["started_at"]), Path(row["audio_path"])) for row in rows],
                out_path,
                serial=zlib.crc32(f"{session.session_id}:{user_id}".encode()),
            )
            if started_at is None:
                continue
            last = rows[-1]
            track_id = self.storage.create_track(
                session.session_id, user_id, last["display_name"], last["character_name"], out_path, started_at
            )
            self.storage.close_track(track_id, max(datetime.fromisoformat(row["ended_at"]) for row in rows))
            log.info("Session %s: converted %s legacy clip(s) of user %s into %s", session.session_id, len(rows), user_id, out_path)

    # --- Transcript ------------------------------------------------------------------------

    def _interruption_note(self, session: ActiveSession) -> str:
        if session.interrupted_at is None:
            return ""
        offset = format_offset(session.interrupted_at - session.started_at)
        return f"The recording was interrupted by a bot restart at {offset}; later audio was not captured."

    def _transcript_lines(self, session: ActiveSession) -> list[TranscriptLine]:
        return merge_lines(self.storage.get_session_segments(session.session_id))

    def _build_timed_transcript(self, session: ActiveSession) -> str:
        """Transcript with `[HH:MM:SS]` offsets from the session start, so facts can cite a time."""
        lines = merge_lines(self.storage.get_session_segments(session.session_id))
        return render_timed(lines, session.started_at, self._interruption_note(session))

    # --- Recording -------------------------------------------------------------------------

    def _record_dir(self, session_id: int, audio_dir: Path) -> Path:
        return self.spool_dir / str(session_id) if self.spool_dir is not None else audio_dir

    def _session_from_row(self, row: sqlite3.Row) -> ActiveSession:
        session_id = int(row["id"])
        base_dir = self.storage.sessions_dir / str(session_id)
        interrupted_at = row["interrupted_at"]
        return ActiveSession(
            session_id=session_id,
            guild_id=int(row["guild_id"]),
            campaign_id=int(row["campaign_id"]),
            voice_channel_id=int(row["voice_channel_id"]),
            text_channel_id=int(row["text_channel_id"]),
            title=row["title"],
            base_dir=base_dir,
            audio_dir=base_dir / "audio",
            started_at=datetime.fromisoformat(row["started_at"]),
            interrupted_at=datetime.fromisoformat(interrupted_at) if interrupted_at else None,
            closed=True,
        )

    async def _stop_recording_session(self, session: ActiveSession) -> None:
        session.closed = True
        if session.reconnect_task:
            session.reconnect_task.cancel()
        if session.voice_client and session.voice_client.is_listening():
            session.voice_client.stop_listening()
        if session.voice_client and session.voice_client.is_connected():
            await session.voice_client.disconnect(force=True)
        if session.recorder:
            # Drains packets still queued, writes the final pages, records each track's end.
            await asyncio.to_thread(session.recorder.stop)

    def _set_status(self, guild_id: int, state: str, session_id: int | None, message: str) -> None:
        self.statuses[guild_id] = SessionStatus(
            state=state,
            session_id=session_id,
            message=message,
            updated_at=datetime.utcnow(),
        )

    async def _connect_voice(self, voice_channel: discord.VoiceChannel) -> voice_recv.VoiceRecvClient:
        existing_client = voice_channel.guild.voice_client
        if existing_client and not isinstance(existing_client, voice_recv.VoiceRecvClient):
            await existing_client.disconnect(force=True)
            existing_client = None
        if existing_client and existing_client.channel and existing_client.channel.id != voice_channel.id:
            await existing_client.move_to(voice_channel)
            client = existing_client
        elif existing_client:
            client = existing_client
        else:
            client = await voice_channel.connect(cls=voice_recv.VoiceRecvClient)
        if not isinstance(client, voice_recv.VoiceRecvClient):
            raise RuntimeError("Voice client does not support receiving audio.")
        return client

    async def _monitor_voice_connection(self, guild: discord.Guild, session: ActiveSession) -> None:
        while not session.closed:
            await asyncio.sleep(5)
            voice_client = session.voice_client
            if voice_client and voice_client.is_connected() and voice_client.is_listening():
                continue
            if voice_client and voice_client.is_connected():
                if session.sink is None:
                    session.sink = SessionAudioSink(self, session)
                try:
                    voice_client.listen(session.sink)
                    continue
                except Exception:
                    await asyncio.sleep(10)
                    continue
            channel = guild.get_channel(session.voice_channel_id)
            if not isinstance(channel, discord.VoiceChannel):
                continue
            try:
                session.voice_client = await self._connect_voice(channel)
                if session.sink is None:
                    session.sink = SessionAudioSink(self, session)
                if not session.voice_client.is_listening():
                    session.voice_client.listen(session.sink)
            except Exception:
                await asyncio.sleep(10)

    # --- Audio retention -------------------------------------------------------------------

    async def _retention_loop(self) -> None:
        while True:
            try:
                await asyncio.to_thread(self.delete_expired_audio)
            except Exception:
                log.exception("Audio retention check failed")
            await asyncio.sleep(RETENTION_CHECK_SECONDS)

    def delete_expired_audio(self, now: datetime | None = None) -> list[int]:
        """Delete the audio of sessions summarized more than `audio_retention_days` ago (blocking)."""
        if self.audio_retention_days <= 0:
            return []
        cutoff = (now or datetime.utcnow()) - timedelta(days=self.audio_retention_days)
        deleted = []
        for row in self.storage.sessions_for_audio_cleanup(cutoff):
            session_id = int(row["id"])
            shutil.rmtree(self.storage.sessions_dir / str(session_id) / "audio", ignore_errors=True)
            if self.spool_dir is not None:
                shutil.rmtree(self.spool_dir / str(session_id), ignore_errors=True)
            self.storage.mark_audio_deleted(session_id)
            deleted.append(session_id)
            log.info("Deleted the audio of session %s (older than %s days)", session_id, self.audio_retention_days)
        return deleted
