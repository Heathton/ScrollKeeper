# ScrollKeeper

ScrollKeeper is a Discord bot for tabletop campaigns. It can join a voice channel, record speaker-separated audio, transcribe the session with in-character names, produce narrative/session summaries, update campaign notes, and answer lore questions from Discord.

## What this MVP includes

- Text commands to join a voice channel, start/end a session, register character names, and ask campaign questions
- Voice receive pipeline built for `discord-ext-voice-recv`
- Persistent SQLite storage for sessions, transcripts, character mappings, notes, and embeddings
- File archives for audio segments, transcripts, summaries, and generated notes
- Retrieval-backed campaign Q&A over saved notes
- Speech-to-text through an OpenAI-compatible `/v1/audio/transcriptions` endpoint (a CPU Parakeet-TDT service with word timestamps is included)
- Summaries, embeddings, and campaign Q&A through any OpenAI-compatible LLM endpoint (`/v1/chat/completions`, `/v1/embeddings`)

## Commands

- `!register-character <character name>`: map your Discord user to an in-game character
- `!join`: bot joins your current voice channel
- `!start-session [title]`: begin recording/transcription for the active voice channel
- `!end-session`: stop recording, finalize transcript, generate summaries/notes, and post the result
- `!campaign-question <question>`: ask about campaign notes
- `!list-notes`: list recent campaign notes with IDs
- `!correct-note <note-id> <corrected content>`: manually fix an incorrect campaign note
- `!session-status`: show the current session state
- `!reprocess-session [session-id]`: rerun speech-to-text + summary/note generation from saved audio
- `!reprocess-llm [session-id]`: rerun summary/note generation only from existing transcript text (skips speech-to-text)

## Prerequisites

- Python 3.11+ (`pip install -e .`) for running tests and the bot directly; `ffmpeg` and `libopus` if you run it outside a container.
- Docker with the compose plugin for building images and the local stack.
- An OpenAI-compatible LLM endpoint (see Configuration).

## Configuration

All configuration comes from environment variables (a `.env` file is optional and only for local development). Copy `.env.example` for the full list. Required:

- `DISCORD_BOT_TOKEN`
- `SCROLLKEEPER_STT_BASE_URL`: OpenAI-compatible speech-to-text server, including `/v1` (the bot calls `POST /audio/transcriptions` with `response_format=verbose_json` and word timestamps).
- `SCROLLKEEPER_LLM_BASE_URL` and `SCROLLKEEPER_LLM_MODEL`: OpenAI-compatible chat server (`POST /chat/completions`). `SCROLLKEEPER_LLM_API_KEY` is optional.
- `SCROLLKEEPER_EMBED_MODEL`: embedding model served at `POST /embeddings` on `SCROLLKEEPER_EMBED_BASE_URL` (defaults to the LLM base URL).

Optional:

- `SCROLLKEEPER_LLM_TIMEOUT_SECONDS=900` / `SCROLLKEEPER_STT_TIMEOUT_SECONDS=600`: read timeouts. They are long on purpose because the LLM host may cold-start for several minutes.
- `SCROLLKEEPER_WAIT_NOTICE_SECONDS=20`: if a request takes longer than this, the bot posts a "waking the inference box" notice in the Discord channel.
- `SCROLLKEEPER_HEALTH_PORT=8080`: serves `GET /healthz` for Kubernetes liveness probes (`0` disables). It returns 503 if the bot's event loop has stalled for over a minute.
- `SCROLLKEEPER_SUMMARY_SINGLE_PASS_MAX_CHARS=90000` sets when the bot switches from single-pass summary generation to chunked summarization.
- `SCROLLKEEPER_SUMMARY_CHUNK_CHARS=45000` sets chunk size used when transcripts are too long for single-pass summarization.
- `SCROLLKEEPER_SUMMARY_PROMPT_APPEND=` appends your own instructions to the summary/note-generation system prompt.

Logs go to stdout. The bot never starts other containers and does not need the Docker socket.

## Local development with Docker Compose

1. Copy `.env.example` to `.env`; fill in the Discord bot token and your LLM endpoint and model.
2. Create a Discord bot with the `MESSAGE CONTENT`, `SERVER MEMBERS INTENT`, and `VOICE STATES INTENT` enabled, and invite it with voice permissions.
3. Start the stack:

```bash
docker compose up --build
```

Compose runs the bot and the CPU speech-to-text service (`docker/stt`, below) that implements `/v1/audio/transcriptions`. The LLM is not started by compose: point `SCROLLKEEPER_LLM_BASE_URL` at any OpenAI-compatible server.

## Speech-to-text service

`docker/stt` serves NVIDIA Parakeet-TDT 0.6B v2 (int8 ONNX, English) through sherpa-onnx on CPU. It was chosen by the benchmark in #3 (about 9x realtime at 3 threads on a 4-core Skylake, ~1.7 GB RAM). The model and the Silero VAD model are downloaded and checksum-verified at image build time, so the container needs no network or volume at runtime.

- `POST /v1/audio/transcriptions`: multipart `file` (any format ffmpeg reads), `response_format` = `json` | `text` | `verbose_json`, and `timestamp_granularities[]` = `segment` and/or `word`. `verbose_json` returns `text`, `duration`, `segments` (one per speech region) and `words`, each with `start`/`end` in seconds from the start of the file. `model` and `prompt` are accepted and ignored; hotwords are not supported (the benchmark found them unusable), so names are corrected downstream.
- Long input: a whole per-speaker track of several hours is fine. The audio is streamed through ffmpeg, cut into speech regions by Silero VAD, and decoded region by region; progress is logged every 30 seconds. Requests are handled one at a time (others queue), since parallel decodes would compete for the same cores.
- `GET /health`: `{"status": "ok", "model": ..., "threads": ..., "busy": ...}`. The model loads before the server accepts connections, so use it as a readiness probe.

Service settings (environment variables of the STT container, not the bot):

- `STT_THREADS=3`: ONNX Runtime threads. 3 leaves a core free on a 4-core node; going from 2 to 3 threads gave ~20% more speed in the benchmark.
- `STT_MAX_SPEECH_SECONDS=20`: longest speech region decoded in one call (longer speech is split by the VAD). Memory grows with region length.
- `STT_MODEL_DIR=/models/parakeet-tdt-0.6b-v2-int8` and `STT_VAD_MODEL=/models/silero_vad.onnx`: override to mount different model files.

## Container images

Pushing a `v*` tag runs `.github/workflows/images.yml`, which publishes `ghcr.io/<owner>/scrollkeeper` (the bot) and `ghcr.io/<owner>/scrollkeeper-stt` (speech-to-text) tagged with the version. Pin these tags in deployments; there is no `latest`.

## Storage layout

- `data/scrollkeeper.db`: SQLite database
- `data/sessions/<session-id>/audio`: recorded WAV segments
- `data/sessions/<session-id>/transcript.md`: finalized transcript
- `data/sessions/<session-id>/summary.md`: session notes + cinematic summary
- `data/notes`: exported campaign note snapshots

## Important implementation notes

- Discord voice receive in Python relies on `discord-ext-voice-recv`.
- The bot records speaker-specific WAV segments and transcribes them after the session ends. This is simpler and more reliable than trying to stream partial text live.
- Transcript markdown is saved in speaker-only format (`**Speaker:** line`) without timestamp prefixes to reduce context-token overhead in summarization.
- Session summaries are generated from the current session transcript only, so prior campaign notes are not used as summary source material.
- Notes are indexed with embeddings stored in SQLite. Transcript text is archived but intentionally excluded from retrieval, matching your requirement.
- If the voice connection drops mid-session, the bot will try to reconnect to the same channel and continue the session.
- `discord-ext-voice-recv` is pinned to a commit SHA in `pyproject.toml`; change it deliberately, in its own PR. If Python voice receive keeps breaking, the fallback is a small Node `@discordjs/voice` recorder feeding this pipeline.
- The bot calls the configured speech-to-text endpoint per speaker segment after the session ends.
- Summaries, note updates, embeddings, and campaign Q&A go to the configured OpenAI-compatible LLM endpoint.
- Long completion posts are split across multiple Discord messages automatically to avoid message-length truncation.
- `!end-session` now queues background processing so users can still run `!campaign-question` while transcription and note generation continue.

## Next improvements

- Incremental live transcript updates during the call
- Better diarization fallback when Discord user audio is unavailable
- Rich slash commands and admin-only maintenance commands
- Structured campaign schema tuning for your exact note taxonomy
