from __future__ import annotations

import asyncio
import contextlib
import logging
import struct
import wave
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Awaitable, Callable

import discord
from discord.ext import voice_recv

from .llm import LocalAIService
from .models import CampaignNote, SessionArtifacts, SpeakerSegment
from .storage import Storage


log = logging.getLogger(__name__)
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2
SEGMENT_IDLE_SECONDS = 1.5
MIN_SEGMENT_SECONDS = 0.75
OPUS_PACKET_EXTENSION = ".skopus"
OPUS_FRAME_DURATION_SECONDS = 0.02
OPUS_PACKET_HEADER = struct.Struct("<HIH")

CompletionHandler = Callable[[int, int, SessionArtifacts | None, str | None], Awaitable[None]]


@dataclass(slots=True)
class BufferedOpusPacket:
    sequence: int
    timestamp: int
    payload: bytes


@dataclass(slots=True)
class ActiveSpeakerBuffer:
    user_id: int
    display_name: str
    character_name: str
    started_at: datetime
    last_packet_at: datetime
    packets: list[BufferedOpusPacket] = field(default_factory=list)


@dataclass(slots=True)
class ActiveSession:
    session_id: int
    guild_id: int
    voice_channel_id: int
    text_channel_id: int
    title: str | None
    base_dir: Path
    audio_dir: Path
    started_at: datetime
    voice_client: voice_recv.VoiceRecvClient | None = None
    sink: "SessionAudioSink | None" = None
    monitor_task: asyncio.Task | None = None
    reconnect_task: asyncio.Task | None = None
    closed: bool = False


@dataclass(slots=True)
class SessionStatus:
    state: str
    session_id: int | None = None
    message: str = ""
    updated_at: datetime = field(default_factory=datetime.utcnow)


class SessionAudioSink(voice_recv.AudioSink):
    def __init__(self, manager: "SessionManager", session: ActiveSession) -> None:
        super().__init__()
        self.manager = manager
        self.session = session
        self.buffers: dict[int, ActiveSpeakerBuffer] = {}

    def wants_opus(self) -> bool:
        return True

    def write(self, user: discord.User | discord.Member | None, data: voice_recv.VoiceData) -> None:
        opus_payload = getattr(data, "opus", None)
        packet = getattr(data, "packet", None)
        if user is None or not opus_payload or packet is None:
            return
        now = datetime.utcnow()
        display_name = getattr(user, "display_name", None) or getattr(user, "name", str(user.id))
        character_name = self.manager.storage.get_character_name(
            self.session.guild_id,
            user.id,
            display_name,
        )
        buffer = self.buffers.get(user.id)
        if buffer is None:
            buffer = ActiveSpeakerBuffer(
                user_id=user.id,
                display_name=display_name,
                character_name=character_name,
                started_at=now,
                last_packet_at=now,
            )
            self.buffers[user.id] = buffer
        buffer.packets.append(
            BufferedOpusPacket(
                sequence=packet.sequence,
                timestamp=packet.timestamp,
                payload=bytes(opus_payload),
            )
        )
        buffer.last_packet_at = now

    def cleanup(self) -> None:
        for user_id in list(self.buffers):
            self.manager.flush_user_buffer(self.session, self.buffers, user_id, force=True)


class SessionManager:
    def __init__(self, storage: Storage, llm: LocalAIService) -> None:
        self.storage = storage
        self.llm = llm
        self.active_sessions: dict[int, ActiveSession] = {}
        self.processing_tasks: dict[int, asyncio.Task] = {}
        self.statuses: dict[int, SessionStatus] = {}
        self._completion_handler: CompletionHandler | None = None

    def set_completion_handler(self, handler: CompletionHandler) -> None:
        self._completion_handler = handler

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

        session_id = self.storage.create_session(
            guild_id=guild.id,
            voice_channel_id=voice_channel.id,
            text_channel_id=text_channel.id,
            title=title,
        )
        base_dir = self.storage.sessions_dir / str(session_id)
        audio_dir = base_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        session = ActiveSession(
            session_id=session_id,
            guild_id=guild.id,
            voice_channel_id=voice_channel.id,
            text_channel_id=text_channel.id,
            title=title,
            base_dir=base_dir,
            audio_dir=audio_dir,
            started_at=datetime.utcnow(),
        )

        voice_client = await self._connect_voice(voice_channel)
        sink = SessionAudioSink(self, session)
        voice_client.listen(sink)
        session.voice_client = voice_client
        session.sink = sink
        session.monitor_task = asyncio.create_task(self._monitor_buffers(session))
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
        self.storage.set_session_status(session.session_id, "processing")
        self._set_status(guild.id, "processing", session.session_id, "Transcribing and generating notes.")

        task = asyncio.create_task(self._process_session_job(guild.id, session))
        self.processing_tasks[guild.id] = task
        task.add_done_callback(lambda _: self.processing_tasks.pop(guild.id, None))
        return session.session_id

    async def reprocess_session(self, guild_id: int, session_id: int | None = None) -> int:
        if guild_id in self.active_sessions:
            raise RuntimeError("Cannot reprocess while a live session is recording in this server.")
        if guild_id in self.processing_tasks:
            raise RuntimeError("A session is already processing in this server.")

        row = self.storage.get_session(session_id) if session_id is not None else self.storage.get_latest_session(guild_id)
        if row is None:
            raise RuntimeError("No saved session was found to reprocess.")
        if int(row["guild_id"]) != guild_id:
            raise RuntimeError("That session does not belong to this server.")

        resolved_session_id = int(row["id"])
        base_dir = self.storage.sessions_dir / str(resolved_session_id)
        audio_dir = base_dir / "audio"
        if not audio_dir.exists():
            raise RuntimeError(f"Session #{resolved_session_id} has no recorded audio directory to reprocess.")

        session = ActiveSession(
            session_id=resolved_session_id,
            guild_id=int(row["guild_id"]),
            voice_channel_id=int(row["voice_channel_id"]),
            text_channel_id=int(row["text_channel_id"]),
            title=row["title"],
            base_dir=base_dir,
            audio_dir=audio_dir,
            started_at=datetime.fromisoformat(row["started_at"]),
            closed=True,
        )

        self.storage.reset_session_processing(resolved_session_id)
        self._set_status(
            guild_id,
            "processing",
            resolved_session_id,
            "Reprocessing saved audio and regenerating notes.",
        )
        task = asyncio.create_task(self._process_session_job(guild_id, session, transcribe_audio=True))
        self.processing_tasks[guild_id] = task
        task.add_done_callback(lambda _: self.processing_tasks.pop(guild_id, None))
        return resolved_session_id

    async def reprocess_llm_only(self, guild_id: int, session_id: int | None = None) -> int:
        if guild_id in self.active_sessions:
            raise RuntimeError("Cannot reprocess while a live session is recording in this server.")
        if guild_id in self.processing_tasks:
            raise RuntimeError("A session is already processing in this server.")

        row = self.storage.get_session(session_id) if session_id is not None else self.storage.get_latest_session(guild_id)
        if row is None:
            raise RuntimeError("No saved session was found to reprocess.")
        if int(row["guild_id"]) != guild_id:
            raise RuntimeError("That session does not belong to this server.")

        resolved_session_id = int(row["id"])
        base_dir = self.storage.sessions_dir / str(resolved_session_id)
        audio_dir = base_dir / "audio"

        session = ActiveSession(
            session_id=resolved_session_id,
            guild_id=int(row["guild_id"]),
            voice_channel_id=int(row["voice_channel_id"]),
            text_channel_id=int(row["text_channel_id"]),
            title=row["title"],
            base_dir=base_dir,
            audio_dir=audio_dir,
            started_at=datetime.fromisoformat(row["started_at"]),
            closed=True,
        )

        self.storage.reset_session_llm_processing(resolved_session_id)
        self._set_status(
            guild_id,
            "processing",
            resolved_session_id,
            "Reprocessing summaries and notes from existing transcript text.",
        )
        task = asyncio.create_task(self._process_session_job(guild_id, session, transcribe_audio=False))
        self.processing_tasks[guild_id] = task
        task.add_done_callback(lambda _: self.processing_tasks.pop(guild_id, None))
        return resolved_session_id

    async def answer_campaign_question(self, guild_id: int, question: str) -> str:
        query_embedding = await self.llm.embed_text(question)
        results = self.storage.semantic_search_notes(guild_id, query_embedding, limit=8)
        if not results:
            return "I do not have any campaign notes saved yet."
        context_chunks = []
        for row in results:
            context_chunks.append(f"[{row['note_type']}] {row['title']}\n{row['content']}")
        return await self.llm.answer_question(question, "\n\n".join(context_chunks))

    def session_status(self, guild_id: int) -> str:
        status = self.statuses.get(guild_id)
        if status is None:
            return "No active or recent session."
        session_label = f"Session #{status.session_id}" if status.session_id else "Session"
        if status.message:
            return f"{session_label} status: {status.state}. {status.message}"
        return f"{session_label} status: {status.state}."

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
            self.storage.set_session_status(session.session_id, "failed")
            self._set_status(
                guild_id,
                "failed",
                session.session_id,
                f"Session processing failed: {error_message}",
            )
        if self._completion_handler:
            await self._completion_handler(guild_id, session.text_channel_id, artifacts, error_message)

    async def _process_closed_session(
        self,
        guild_id: int,
        session: ActiveSession,
        transcribe_audio: bool = True,
    ) -> SessionArtifacts:
        segments = self.storage.get_session_segments(session.session_id)
        log.info("Session %s has %s recorded audio segments", session.session_id, len(segments))
        if not segments:
            raise RuntimeError(
                "No decodable audio was captured for this session. "
                "Voice packets were received, but no valid speaker audio could be saved."
            )
        if transcribe_audio:
            transcribed_segments = 0
            skipped_decode_segments = 0
            await self.llm.services.prepare_for_transcription()
            try:
                for segment in segments:
                    source_audio_path = Path(segment["audio_path"])
                    try:
                        wav_path = self._prepare_segment_for_transcription(source_audio_path)
                    except Exception as exc:
                        skipped_decode_segments += 1
                        log.warning(
                            "Skipping undecodable segment for session %s (%s): %s",
                            session.session_id,
                            source_audio_path,
                            exc,
                        )
                        continue
                    log.info(
                        "Transcribing session %s segment %s",
                        session.session_id,
                        wav_path,
                    )
                    transcript_text = await self.llm.transcribe_audio_segment(wav_path)
                    self.storage.update_segment_transcript(
                        session.session_id,
                        segment["audio_path"],
                        transcript_text,
                    )
                    transcribed_segments += 1
                    log.info(
                        "Finished transcribing session %s segment %s",
                        session.session_id,
                        wav_path,
                    )
                if transcribed_segments == 0:
                    raise RuntimeError(
                        "No audio segments could be transcribed. All saved segments failed to decode."
                    )
                if skipped_decode_segments:
                    log.warning(
                        "Session %s skipped %s undecodable segment(s) and transcribed %s segment(s).",
                        session.session_id,
                        skipped_decode_segments,
                        transcribed_segments,
                    )
            finally:
                await self.llm.services.stop_whisper()
                await self.llm.services.recover_after_transcription()
        else:
            transcribed_segments = sum(
                1
                for segment in segments
                if (segment["transcript_text"] or "").strip()
            )
            if transcribed_segments == 0:
                raise RuntimeError(
                    "No transcript text exists for this session yet. Run `!reprocess-session` first."
                )
            log.info(
                "Skipping Whisper for session %s and reusing %s existing transcript segment(s)",
                session.session_id,
                transcribed_segments,
            )

        transcript_markdown = self._build_transcript_markdown(session.session_id)
        existing_notes_context = self._format_existing_notes_context(guild_id)
        log.info("Generating summary and notes for session %s", session.session_id)
        summary_payload = await self.llm.summarize_session(transcript_markdown, existing_notes_context)
        session_notes = summary_payload["session_notes_markdown"].strip()
        cinematic = summary_payload["cinematic_summary_markdown"].strip()
        note_updates = summary_payload["note_updates"]

        transcript_path = session.base_dir / "transcript.md"
        summary_path = session.base_dir / "summary.md"
        transcript_path.write_text(transcript_markdown, encoding="utf-8")
        summary_markdown = (
            "# Session Notes\n\n"
            f"{session_notes}\n\n"
            "# Cinematic Summary\n\n"
            f"{cinematic}\n"
        )
        summary_path.write_text(summary_markdown, encoding="utf-8")

        for update in note_updates:
            content = str(update["content"]).strip()
            title = str(update["title"]).strip()
            note_type = str(update["note_type"]).strip()
            if not content or not title or not note_type:
                continue
            note = CampaignNote(
                guild_id=guild_id,
                note_type=note_type,
                title=title,
                content=content,
                source_session_id=session.session_id,
                metadata=update.get("metadata", {}),
            )
            embedding = await self.llm.embed_text(f"{note.note_type}\n{note.title}\n{note.content}")
            self.storage.upsert_campaign_note(note, embedding)

        exported_notes_path = self._export_notes_snapshot(guild_id, session.base_dir)
        self.storage.finalize_session(
            session.session_id,
            str(transcript_path),
            str(summary_path),
        )
        return SessionArtifacts(
            session_id=session.session_id,
            transcript_markdown=transcript_markdown,
            session_notes_markdown=session_notes,
            cinematic_summary_markdown=cinematic,
            note_updates=note_updates,
            transcript_path=transcript_path,
            summary_path=summary_path,
            exported_notes_path=exported_notes_path,
        )

    async def _stop_recording_session(self, session: ActiveSession) -> None:
        session.closed = True
        if session.monitor_task:
            session.monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await session.monitor_task
        if session.sink:
            session.sink.cleanup()
        if session.voice_client and session.voice_client.is_listening():
            session.voice_client.stop_listening()
        if session.voice_client and session.voice_client.is_connected():
            await session.voice_client.disconnect(force=True)
        if session.reconnect_task:
            session.reconnect_task.cancel()

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

    async def _monitor_buffers(self, session: ActiveSession) -> None:
        while not session.closed:
            await asyncio.sleep(0.5)
            if session.sink is None:
                continue
            for user_id in list(session.sink.buffers):
                self.flush_user_buffer(session, session.sink.buffers, user_id, force=False)

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

    def flush_user_buffer(
        self,
        session: ActiveSession,
        buffers: dict[int, ActiveSpeakerBuffer],
        user_id: int,
        force: bool,
    ) -> None:
        buffer = buffers.get(user_id)
        if buffer is None:
            return
        if not buffer.packets:
            buffers.pop(user_id, None)
            return
        now = datetime.utcnow()
        idle_for = now - buffer.last_packet_at
        duration_seconds = len(buffer.packets) * OPUS_FRAME_DURATION_SECONDS
        if not force:
            if idle_for < timedelta(seconds=SEGMENT_IDLE_SECONDS):
                return
            if duration_seconds < MIN_SEGMENT_SECONDS:
                buffers.pop(user_id, None)
                return

        ended_at = buffer.last_packet_at
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in buffer.character_name).strip("_") or "speaker"
        timestamp = int(buffer.started_at.timestamp() * 1000)
        audio_path = session.audio_dir / f"{timestamp}_{user_id}_{safe_name}{OPUS_PACKET_EXTENSION}"
        self._write_opus_packet_file(audio_path, buffer.packets)
        segment = SpeakerSegment(
            discord_user_id=buffer.user_id,
            discord_display_name=buffer.display_name,
            character_name=buffer.character_name,
            started_at=buffer.started_at,
            ended_at=ended_at,
            audio_path=audio_path,
        )
        self.storage.add_transcript_segment(session.session_id, segment)
        log.info(
            "Saved Opus segment for session %s user %s to %s (%.2fs, %s packets)",
            session.session_id,
            buffer.user_id,
            audio_path,
            duration_seconds,
            len(buffer.packets),
        )
        buffers.pop(user_id, None)

    def _prepare_segment_for_transcription(self, audio_path: Path) -> Path:
        if audio_path.suffix.lower() == ".wav":
            return audio_path
        if audio_path.suffix.lower() != OPUS_PACKET_EXTENSION:
            raise RuntimeError(f"Unsupported segment format for transcription: {audio_path}")
        wav_path = audio_path.with_suffix(".wav")
        if wav_path.exists():
            return wav_path
        self._decode_opus_packet_file_to_wav(audio_path, wav_path)
        return wav_path

    def _write_opus_packet_file(self, audio_path: Path, packets: list[BufferedOpusPacket]) -> None:
        with audio_path.open("wb") as handle:
            for packet in packets:
                handle.write(
                    OPUS_PACKET_HEADER.pack(
                        packet.sequence,
                        packet.timestamp,
                        len(packet.payload),
                    )
                )
                handle.write(packet.payload)

    def _read_opus_packet_file(self, audio_path: Path) -> list[BufferedOpusPacket]:
        packets: list[BufferedOpusPacket] = []
        with audio_path.open("rb") as handle:
            while True:
                header = handle.read(OPUS_PACKET_HEADER.size)
                if not header:
                    break
                if len(header) != OPUS_PACKET_HEADER.size:
                    raise RuntimeError(f"Corrupted Opus packet header in {audio_path}")
                sequence, timestamp, payload_size = OPUS_PACKET_HEADER.unpack(header)
                payload = handle.read(payload_size)
                if len(payload) != payload_size:
                    raise RuntimeError(f"Corrupted Opus packet payload in {audio_path}")
                packets.append(
                    BufferedOpusPacket(
                        sequence=sequence,
                        timestamp=timestamp,
                        payload=payload,
                    )
                )
        return packets

    def _decode_opus_packet_file_to_wav(self, audio_path: Path, wav_path: Path) -> None:
        packets = self._read_opus_packet_file(audio_path)
        if not packets:
            raise RuntimeError(f"No Opus packets were captured in {audio_path}")

        decoder = discord.opus.Decoder()
        pcm_frames = bytearray()
        expected_sequence: int | None = None
        max_gap_frames = int(SEGMENT_IDLE_SECONDS / OPUS_FRAME_DURATION_SECONDS)
        corrupted_packets = 0

        for packet in packets:
            if expected_sequence is not None:
                gap = (packet.sequence - expected_sequence) % 65536
                if 0 < gap <= max_gap_frames:
                    for _ in range(gap):
                        pcm_frames.extend(decoder.decode(None, fec=False))

            try:
                pcm_frames.extend(decoder.decode(packet.payload, fec=False))
            except discord.opus.OpusError:
                corrupted_packets += 1
                # Packet loss concealment keeps the timeline aligned when a frame is corrupt.
                with contextlib.suppress(discord.opus.OpusError):
                    pcm_frames.extend(decoder.decode(None, fec=False))
            expected_sequence = (packet.sequence + 1) % 65536

        if not pcm_frames:
            raise RuntimeError(f"No decodable PCM frames were produced from {audio_path}")

        with wave.open(str(wav_path), "wb") as wav_file:
            wav_file.setnchannels(CHANNELS)
            wav_file.setsampwidth(SAMPLE_WIDTH)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(bytes(pcm_frames))

        if corrupted_packets:
            log.warning(
                "Decoded Opus segment %s to WAV %s with %s corrupted packet(s) concealed",
                audio_path,
                wav_path,
                corrupted_packets,
            )
            return
        log.info("Decoded Opus segment %s to WAV %s", audio_path, wav_path)

    def _build_transcript_markdown(self, session_id: int) -> str:
        rows = self.storage.get_session_segments(session_id)
        lines = ["# Transcript", ""]
        for row in rows:
            transcript_text = (row["transcript_text"] or "").strip()
            if not transcript_text:
                continue
            speaker = row["character_name"] or row["display_name"]
            lines.append(f"{speaker}: {transcript_text}")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _format_existing_notes_context(self, guild_id: int) -> str:
        notes = self.storage.get_recent_notes(guild_id)
        if not notes:
            return "No prior campaign notes."
        rendered = []
        for note in notes:
            rendered.append(f"[{note['note_type']}] {note['title']}\n{note['content']}")
        return "\n\n".join(rendered)

    def _export_notes_snapshot(self, guild_id: int, base_dir: Path) -> Path:
        notes = self.storage.get_recent_notes(guild_id)
        export_path = base_dir / "campaign_notes.md"
        lines = ["# Campaign Notes Snapshot", ""]
        for note in notes:
            lines.append(f"## [{note['note_type']}] {note['title']}")
            lines.append("")
            lines.append(note["content"])
            lines.append("")
        export_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return export_path
