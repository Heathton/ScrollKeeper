# ScrollKeeper

ScrollKeeper is a Discord bot for tabletop campaigns. It can join a voice channel, record speaker-separated audio, transcribe the session with in-character names, produce narrative/session summaries, maintain a campaign wiki built from sourced facts, and answer lore questions from Discord.

## What this MVP includes

- Text commands to join a voice channel, start/end a session, register character names, and ask campaign questions
- Voice receive pipeline built for `discord-ext-voice-recv`
- Persistent SQLite storage for sessions, transcripts, character mappings, the campaign wiki, and embeddings
- File archives for audio segments, transcripts, summaries, and an Obsidian-style wiki export
- A campaign wiki: entities with aliases, an append-only log of facts with their source (session and transcript time), and pages rewritten from those facts
- Retrieval-backed campaign Q&A over wiki pages
- Speech-to-text through an OpenAI-compatible `/v1/audio/transcriptions` endpoint (a CPU faster-whisper service is included)
- Summaries, embeddings, and campaign Q&A through any OpenAI-compatible LLM endpoint (`/v1/chat/completions`, `/v1/embeddings`)

## Commands

- `!register-character <character name>`: map your Discord user to an in-game character
- `!join`: bot joins your current voice channel
- `!start-session [title]`: begin recording/transcription for the active voice channel
- `!end-session`: stop recording, finalize transcript, write the summary, update the campaign wiki, and post the result with a wiki change report
- `!campaign-question <question>`: ask about the campaign (answers come from wiki pages)
- `!session-status`: show the current session state
- `!reprocess-session [session-id]`: rerun speech-to-text + summary/note generation from saved audio
- `!reprocess-llm [session-id]`: rerun the summary and wiki update from existing transcript text (skips speech-to-text). The session's earlier extracted facts are retracted and replaced.

### Campaign wiki

Entities are referred to by `#id`, name or alias. Put names with spaces in quotes when another argument follows (`!merge-entity "Lord Varric" Varric`).

- `!entities [type]`: list entities (types: Character, Faction, Location, Item, Mystery, PointOfInterest, Quest) with fact counts
- `!entity <name>`: show an entity's page and the facts it cites (fact ids `F12`, with session and transcript time)
- `!merge-entity <from> <into>`: merge a duplicate; its facts move over and its names become aliases
- `!rename-entity <entity> <new name>`: change the canonical name (the old name stays as an alias)
- `!add-alias <entity> <alias>`: add another name, nickname or common mis-transcription
- `!pin-fact <entity> <text>`: add an authoritative fact; pinned facts override anything that conflicts
- `!correct-fact <fact-id> <text>`: replace a wrong fact with a pinned one (the old fact is kept as history)
- `!retract-fact <fact-id> [reason]`: withdraw a wrong fact
- `!rebuild-pages`: rewrite every page from its facts (after a page-layout change); one LLM call per entity

After each processed session the bot posts the new, updated and renamed entities and **possible duplicates** so they can be merged.

Pages follow a fixed layout per type (`PAGE_LAYOUTS` in `src/scrollkeeper/wiki.py`): an opening paragraph, then sections in a set order, each written only when facts support it.

| Type | Sections |
|---|---|
| Character | Appearance & Personality, Relationships, History, Status, Open Questions |
| Player character (registered) | Relationships, History |
| Faction | Members & Leadership, Allies & Enemies, Activities, Relationship with the Party, Open Questions |
| Location | Notable Places, Notable People, History, Current State, Open Questions |
| Item | Properties & Effects, Provenance, History, Open Questions |
| Mystery | Clues, Theories, Status |
| PointOfInterest | Features, Dangers, History, Open Questions |
| Quest | Status, Progress, People & Places, Open Questions |

Quests are entities too. Extraction records each quest's giver, objective, reward and status changes on the Quest only, naming the entities involved; the quest page holds all the details and its status (offered, active, completed, failed, abandoned). Any entity named in a quest's facts gets a **Quests** list (quest name and current status) added when its page is shown or exported. The list is built from current data, not written by the LLM, so it never goes stale.

How the bot decides whether someone or something is already known:

- Fact extraction sees an index of every entity (id, type, names). Entities named in the transcript chunk, including near spellings such as "Varic" for "Varric", also show their one-line description.
- Before a proposed entity is created, the bot looks for existing entities with the same name or alias, a similar spelling, a shared name word ("Lord Thane" / "Varric Thane"), or the name appearing in a description ("the harbour master"). If it finds any, a short LLM check decides `same`, `different` or `unsure`. `same` attaches the facts and learns the new name as an alias. `unsure` creates the entity and flags the pair in the change report.
- A name that matches several entities is reported instead of being guessed silently.
- When a described entity's real name is revealed ("the harbour master was Aldous Penn"), the entity is renamed and the old label becomes an alias.
- Names registered with `!register-character` become Character entities, so player characters always exist.

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
- `SCROLLKEEPER_LLM_EXTRA_BODY=`: a JSON object merged into every chat request for server-specific options. Reasoning models can spend minutes thinking before answering, and a proxy in front of the LLM may cut long requests. With a Qwen3-family model on vLLM whose chat template defaults to `xhigh` effort, one extraction call took about 4 minutes. On a 19k-character transcript chunk, the same call took about 50 seconds at `low` or `medium`. Medium extracted the most complete facts, so use `{"chat_template_kwargs": {"reasoning_effort": "medium"}}`; `{"chat_template_kwargs": {"enable_thinking": false}}` is about three times faster but makes more mistakes, such as stating claims as facts. Check which kwargs your model's chat template accepts.
- `SCROLLKEEPER_WAIT_NOTICE_SECONDS=20`: if a request takes longer than this, the bot posts a "waking the inference box" notice in the Discord channel.
- `SCROLLKEEPER_HEALTH_PORT=8080`: serves `GET /healthz` for Kubernetes liveness probes (`0` disables). It returns 503 if the bot's event loop has stalled for over a minute.
- `SCROLLKEEPER_SUMMARY_SINGLE_PASS_MAX_CHARS=90000` sets when the bot switches from single-pass summary generation to chunked summarization.
- `SCROLLKEEPER_SUMMARY_CHUNK_CHARS=45000` sets chunk size used when transcripts are too long for single-pass summarization.
- `SCROLLKEEPER_SUMMARY_PROMPT_APPEND=` appends your own instructions to the summary system prompt.
- `SCROLLKEEPER_EXTRACT_CHUNK_CHARS=24000` sets the transcript chunk size for campaign fact extraction (each chunk is sent with the entity index).
- `SCROLLKEEPER_WIKI_EXPORT=1` writes the Obsidian-style export to `data/wiki/` after wiki changes (`0` disables).

Fact extraction and page rewrites request schema-enforced JSON (`response_format: {"type": "json_schema"}`), which vLLM and other OpenAI-compatible servers support.

Logs go to stdout. The bot never starts other containers and does not need the Docker socket.

## Local development with Docker Compose

1. Copy `.env.example` to `.env`; fill in the Discord bot token and your LLM endpoint and model.
2. Create a Discord bot with the `MESSAGE CONTENT`, `SERVER MEMBERS INTENT`, and `VOICE STATES INTENT` enabled, and invite it with voice permissions.
3. Start the stack:

```bash
docker compose up --build
```

Compose runs the bot and a CPU faster-whisper service (`docker/whisper`) that implements `/v1/audio/transcriptions`. The LLM is not started by compose: point `SCROLLKEEPER_LLM_BASE_URL` at any OpenAI-compatible server.

## Container images

Pushing a `v*` tag runs `.github/workflows/images.yml`, which publishes `ghcr.io/<owner>/scrollkeeper` (the bot) and `ghcr.io/<owner>/scrollkeeper-stt` (speech-to-text) tagged with the version. Pin these tags in deployments; there is no `latest`.

## Storage layout

- `data/scrollkeeper.db`: SQLite database
- `data/sessions/<session-id>/audio`: recorded WAV segments
- `data/sessions/<session-id>/transcript.md`: finalized transcript
- `data/sessions/<session-id>/summary.md`: session notes + cinematic summary
- `data/wiki/<guild-id>/<Type>/<Name>.md`: Obsidian-style wiki export, one file per entity with `aliases` frontmatter, `[[links]]` and session citations. It is regenerated from the database, so edits there are overwritten; use the commands above.

## Important implementation notes

- Discord voice receive in Python relies on `discord-ext-voice-recv`.
- The bot records speaker-specific WAV segments and transcribes them after the session ends. This is simpler and more reliable than trying to stream partial text live.
- Transcript markdown is saved in speaker-only format (`**Speaker:** line`) without timestamp prefixes to reduce context-token overhead in summarization.
- Session summaries are generated from the current session transcript only, so prior campaign notes are not used as summary source material.
- Wiki pages are indexed with embeddings stored in SQLite. Transcript text is archived but intentionally excluded from retrieval.
- The wiki pipeline after each session: (1) summary from the transcript only; (2) fact extraction per timestamped transcript chunk, given the entity index, attaching facts to existing entities or proposing new ones; (3) rewrite of every page whose facts changed. Pinned facts are authoritative. Retracting or superseding a fact rebuilds the page from the remaining facts.
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
