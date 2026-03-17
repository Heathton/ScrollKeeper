# ScrollKeeper

ScrollKeeper is a Discord bot for tabletop campaigns. It can join a voice channel, record speaker-separated audio, transcribe the session with in-character names, produce narrative/session summaries, update campaign notes, and answer lore questions from Discord.

## What this MVP includes

- Text commands to join a voice channel, start/end a session, register character names, and ask campaign questions
- Voice receive pipeline built for `discord-ext-voice-recv`
- Persistent SQLite storage for sessions, transcripts, character mappings, notes, and embeddings
- File archives for audio segments, transcripts, summaries, and generated notes
- Retrieval-backed campaign Q&A over saved notes
- Local Whisper transcription service started on demand in Docker
- Local Ollama inference service started on demand for summaries, embeddings, and campaign Q&A

## Commands

- `!register-character <character name>`: map your Discord user to an in-game character
- `!join`: bot joins your current voice channel
- `!start-session [title]`: begin recording/transcription for the active voice channel
- `!end-session`: stop recording, finalize transcript, generate summaries/notes, and post the result
- `!campaign-question <question>`: ask about campaign notes
- `!session-status`: show the current session state

## Docker Setup

1. Copy `.env.example` to `.env` and fill in your Discord bot token.
2. Create a Discord bot with the `MESSAGE CONTENT`, `SERVER MEMBERS INTENT`, and `VOICE STATES INTENT` enabled.
3. Invite the bot to your server with voice permissions.
4. Start the stack:

```bash
docker compose up --build
```

The bot container mounts the Docker socket and starts sibling containers as needed:

- A local Whisper service is built from `docker/whisper` and started only during transcript finalization.
- An Ollama container is kept running in `concurrent` mode so question answering remains available during session processing.
- Set `SCROLLKEEPER_GPU_POLICY=serialize` to unload Ollama models while Whisper transcribes if GPU pressure is too high.
- `SCROLLKEEPER_OLLAMA_IDLE_TIMEOUT=0` disables automatic Ollama shutdown (recommended for concurrency tests).

## Storage layout

- `data/scrollkeeper.db`: SQLite database
- `data/sessions/<session-id>/audio`: recorded WAV segments
- `data/sessions/<session-id>/transcript.md`: finalized transcript
- `data/sessions/<session-id>/summary.md`: session notes + cinematic summary
- `data/notes`: exported campaign note snapshots

## Important implementation notes

- Discord voice receive in Python relies on `discord-ext-voice-recv`.
- The bot records speaker-specific WAV segments and transcribes them after the session ends. This is simpler and more reliable than trying to stream partial text live.
- Notes are indexed with embeddings stored in SQLite. Transcript text is archived but intentionally excluded from retrieval, matching your requirement.
- If the voice connection drops mid-session, the bot will try to reconnect to the same channel and continue the session.
- The bot calls the local Whisper HTTP service for speech-to-text, then stops that container after transcription completes.
- The bot calls Ollama's local REST API for summaries, note updates, embeddings, and campaign Q&A.
- `!end-session` now queues background processing so users can still run `!campaign-question` while transcription and note generation continue.
- The first Ollama use may take a while because the requested models need to be pulled into the persistent `scrollkeeper_ollama` volume.

## Recommended Defaults For Your Hardware

- `SCROLLKEEPER_OLLAMA_MODEL=qwen3.5:9b`
- `SCROLLKEEPER_WHISPER_MODEL=small.en`
- `SCROLLKEEPER_GPU_POLICY=concurrent`
- `SCROLLKEEPER_OLLAMA_IDLE_TIMEOUT=0`

## Next improvements

- Incremental live transcript updates during the call
- Better diarization fallback when Discord user audio is unavailable
- Rich slash commands and admin-only maintenance commands
- Structured campaign schema tuning for your exact note taxonomy
