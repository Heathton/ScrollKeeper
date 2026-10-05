# ScrollKeeper

ScrollKeeper is a Discord bot for tabletop campaigns. It can join a voice channel, record speaker-separated audio, transcribe the session with in-character names, produce narrative/session summaries, maintain a campaign wiki built from sourced facts, and answer lore questions from Discord.

## What this MVP includes

- Text commands to join a voice channel, start/end a session, register character names, and ask campaign questions
- Voice receive pipeline built for `discord-ext-voice-recv`
- Persistent SQLite storage for sessions, transcripts, character mappings, the campaign wiki, and the search index
- Per-speaker Ogg Opus audio tracks, transcripts, summaries, and an Obsidian-style wiki export
- A campaign wiki: entities with aliases, an append-only log of facts with their source (session and transcript time), and pages rewritten from those facts
- Campaign Q&A with cited answers, from hybrid search (names and aliases, keywords, and embeddings) over wiki pages and session summaries, and optionally transcripts
- Speech-to-text through an OpenAI-compatible `/v1/audio/transcriptions` endpoint (a CPU Parakeet-TDT service with word timestamps is included)
- Summaries, wiki updates and answers through any OpenAI-compatible LLM endpoint (`/v1/chat/completions`); embeddings run inside the bot on CPU

## Commands

- `!list-campaigns`: list this server's campaigns and mark the active one
- `!current-campaign`: show the active campaign
- `!switch-campaign <name>`: make another campaign active, creating it if no campaign has that name (case-insensitive). Not allowed while a session is recording.
- `!register-character <character name>`: map your Discord user to an in-game character in the active campaign. **Recording is opt-in:** only players who have registered a character in the active campaign are recorded (bots never are).
- `!join`: bot joins your current voice channel
- `!start-session [title]`: begin recording the active voice channel. The bot posts a recording notice and names anyone in the channel who is not being recorded.
- `!end-session`: stop recording, finalize transcript, write the summary, update the campaign wiki, and post the result with a wiki change report
- `!campaign-question <question>`: ask about the campaign. Answers come from wiki pages and session summaries and cite their sources (see [Campaign questions](#campaign-questions))
- `!deep-question <question>`: the same, but also searches session transcripts (slower to read, but finds what never made it into the notes)
- `!reindex`: rebuild the search index and re-embed everything (needs the Manage Server permission). Rarely needed: the index updates itself after every change and when the embedding model changes
- `!session-status`: show the current session state
- `!reprocess-session [session-id]`: rerun speech-to-text + summary/note generation from saved audio (also works for sessions recorded before per-speaker tracks; see [Recording](#recording))
- `!reprocess-llm [session-id]`: rerun the summary and wiki update from existing transcript text (skips speech-to-text). The session's earlier extracted facts are retracted and replaced.

### Campaigns

A server can run several campaigns, and one of them is active. The first command that needs a campaign creates one called **Default**. Character registrations, sessions, the campaign wiki, and questions all belong to a campaign:

- `!start-session` records into the active campaign. Each session keeps its campaign, so switching while earlier sessions are still processing is fine; they update the wiki of the campaign they were recorded in.
- Wiki commands (`!entities`, `!entity`, `!pin-fact`, ...) and questions (`!campaign-question`, `!deep-question`) work on the active campaign only. The same name in two campaigns is two different entities.
- `!reprocess-session` and `!reprocess-llm` without a session id pick the latest session of the active campaign.
- Players register a character separately in each campaign.

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
| Character | Appearance & Personality, Relationships, History, Status |
| Player character (registered) | Relationships, History |
| Faction | Members & Leadership, Allies & Enemies, Activities, Relationship with the Party |
| Location | Notable Places, Notable People, History, Current State |
| Item | Properties & Effects, Provenance, History |
| Mystery | Clues, Theories, Status |
| PointOfInterest | Features, Dangers, History |
| Quest | Status, Progress, People & Places, Open Questions |

Quests are entities too. Extraction records each quest's giver, objective, reward and status changes on the Quest only, naming the entities involved; the quest page holds all the details and its status (offered, active, completed, failed, abandoned). Any entity named in a quest's facts gets a **Quests** list (quest name and current status) added when its page is shown or exported. The list is built from current data, not written by the LLM, so it never goes stale.

How the bot decides whether someone or something is already known:

- Fact extraction sees an index of every entity (id, type, names). Entities named in the transcript chunk, including near spellings such as "Varic" for "Varric", also show their one-line description.
- Before a proposed entity is created, the bot looks for existing entities with the same name or alias, a similar spelling, a shared name word ("Lord Thane" / "Varric Thane"), or the name appearing in a description ("the harbour master"). If it finds any, a short LLM check decides `same`, `different` or `unsure`. `same` attaches the facts and learns the new name as an alias. `unsure` creates the entity and flags the pair in the change report.
- A name that matches several entities is reported instead of being guessed silently.
- When a described entity's real name is revealed ("the harbour master was Aldous Penn"), the entity is renamed and the old label becomes an alias.
- Names registered with `!register-character` become Character entities, so player characters always exist.

## Campaign questions

`!campaign-question` searches every wiki page and session summary three ways, and merges the three rankings with reciprocal rank fusion:

1. **Names and aliases** in the question, including near spellings ("Thalren" for "Thalrin Vey"). The matching entities' pages get extra weight.
2. **Keywords**: SQLite FTS5 full-text search (BM25, English stemming; titles weigh more than text). The names of mentioned entities are added as search terms, so "the Grey Warden" also finds summaries that say "Thalrin".
3. **Meaning**: cosine similarity of embeddings, for questions that share no words with the notes ("where are the lizards?" finds the page about lizardfolk).

The top documents go to the LLM as numbered sources. Answers cite them: wiki facts as `(session 12 @ 01:43:10)`, summaries as `(session 12 summary)`, transcript excerpts as `(session 12 @ 01:40:05)`, and pages as `(wiki: Name)`. If nothing relevant is found, the bot says **"That's not in the notes."** without asking the LLM, and the LLM is told to give the same answer when the sources don't contain one.

`!deep-question` also searches session transcripts, cut into chunks of about 1,500 characters that each start with a `[HH:MM:SS]` time. Transcript chunks are found by name and keyword only: embedding every chunk of every session would take hours of CPU.

The index is kept in SQLite (`search_docs`, `search_fts`) and is built from the pages, summaries and transcripts, so it never has to be edited by hand. It updates after each processed session and wiki change, and at startup.

## Prerequisites

- Python 3.11+ (`pip install -e .`) for running tests and the bot directly; `ffmpeg` and `libopus` if you run it outside a container.
- Docker with the compose plugin for building images and the local stack.
- An OpenAI-compatible LLM endpoint (see Configuration).

## Configuration

All configuration comes from environment variables (a `.env` file is optional and only for local development). Copy `.env.example` for the full list. Required:

- `DISCORD_BOT_TOKEN`
- `SCROLLKEEPER_STT_BASE_URL`: OpenAI-compatible speech-to-text server, including `/v1` (the bot calls `POST /audio/transcriptions` with `response_format=verbose_json` and word timestamps).
- `SCROLLKEEPER_LLM_BASE_URL` and `SCROLLKEEPER_LLM_MODEL`: OpenAI-compatible chat server (`POST /chat/completions`). `SCROLLKEEPER_LLM_API_KEY` is optional.

Optional:

- `SCROLLKEEPER_LLM_TIMEOUT_SECONDS=900`: read timeout for LLM calls. It is long on purpose because the LLM host may cold-start for several minutes.
- `SCROLLKEEPER_STT_TIMEOUT_SECONDS=7200`: read timeout for one speech-to-text call. Each call transcribes a **whole speaker track**, so it must cover the longest track: the bundled service runs about 9x realtime, so a 3-hour track takes roughly 20 minutes. Raise it for very long sessions or a busy node.
- `SCROLLKEEPER_SPOOL_DIR=`: where tracks are written while recording. Empty (the default) records straight into `data/sessions/`. Set it to a fast local disk when `SCROLLKEEPER_DATA_DIR` is on network storage; tracks move to the archive when processing starts. It must be **persistent** (for example a local-path volume, not an `emptyDir`), or a restart during a session loses its audio. Speech-to-text scratch files also go here (otherwise the system temp directory).
- `SCROLLKEEPER_AUDIO_RETENTION_DAYS=0`: delete a session's audio this many days after its summary was written (`0` keeps audio forever). Transcripts, summaries and the wiki are kept; `!reprocess-llm` still works, `!reprocess-session` does not.
- `SCROLLKEEPER_LLM_EXTRA_BODY=`: a JSON object merged into every chat request for server-specific options, such as a reasoning model's effort level. See [Recommended models and settings](#recommended-models-and-settings).
- `SCROLLKEEPER_WAIT_NOTICE_SECONDS=20`: if a request takes longer than this, the bot posts a "waking the inference box" notice in the Discord channel.
- `SCROLLKEEPER_HEALTH_PORT=8080`: serves `GET /healthz` for Kubernetes liveness probes (`0` disables). It returns 503 if the bot's event loop has stalled for over a minute.
- `SCROLLKEEPER_LLM_CONTEXT_TOKENS=`: the chat model's context length in tokens. Empty (the default) reads it from the server (`max_model_len` on vLLM's `GET /v1/models`), else assumes 32,768. The summary is written in one pass when the transcript fits (a quarter of the context, at least 8k tokens, stays free for the reply). A longer transcript is split into equal-sized parts, each cut at the longest pause near its target size so scenes stay whole; the parts are summarized and then combined. This replaces `SCROLLKEEPER_SUMMARY_SINGLE_PASS_MAX_CHARS` and `SCROLLKEEPER_SUMMARY_CHUNK_CHARS`, which are ignored now.
- `SCROLLKEEPER_SUMMARY_PROMPT_APPEND=` appends your own instructions to the summary system prompt.
- `SCROLLKEEPER_EXTRACT_CHUNK_CHARS=24000` sets the transcript chunk size for campaign fact extraction (each chunk is sent with the entity index).
- `SCROLLKEEPER_WIKI_EXPORT=1` writes the Obsidian-style export to `data/wiki/` after wiki changes (`0` disables).
- `SCROLLKEEPER_EMBED_MODEL=embeddinggemma-300m`: the in-process embedding model, `embeddinggemma-300m` or `qwen3-embedding-0.6b` (see [Embeddings](#embeddings-scrollkeeper_embed_model)). Changing it re-embeds everything at the next start.
- `SCROLLKEEPER_EMBED_MODEL_DIR=`: where model files are kept. Empty (the default) means `data/models/`.
- `SCROLLKEEPER_EMBED_THREADS=2`: ONNX Runtime threads for embeddings.

Summaries, fact extraction and page rewrites request schema-enforced JSON (`response_format: {"type": "json_schema"}`), which vLLM and other OpenAI-compatible servers support. A failed request (a transport error or unparseable reply) is retried once.

Speech-to-text often misspells campaign names. The summary prompt carries a spelling list (registered character names, plus wiki entity names and aliases; quest and mystery titles are left out) and asks the model to correct likely mis-transcriptions to those spellings. It holds names only, so summaries still come from the session's transcript alone. Fact extraction already sees the same names in its entity index and gets the same instruction.

Logs go to stdout. The bot never starts other containers and does not need the Docker socket.

## Recommended models and settings

This is what has been tested.

### Chat LLM (`SCROLLKEEPER_LLM_BASE_URL`, `SCROLLKEEPER_LLM_MODEL`)

Requirements:

- An OpenAI-compatible `POST /v1/chat/completions` that supports **`response_format: {"type": "json_schema"}`** (vLLM does). Summaries, fact extraction, entity-match checks and page rewrites depend on it.
- A context window of **at least 32k tokens**. Larger is better: a session that fits is summarized in one pass (a 32k context holds roughly 70,000 characters of transcript). Set `SCROLLKEEPER_LLM_CONTEXT_TOKENS` if the server doesn't report `max_model_len`.

Tested: a **Qwen3-family 27B reasoning model, 4-bit (W4A16), on vLLM** with a 150k-token context. On synthetic sessions it:

- extracted correct facts and aliases,
- kept claims as claims,
- matched misspellings and titles to the right entities,
- kept two characters with the same first name apart.

Smaller or non-reasoning models are untested. If you try one, check the change reports for invented facts and duplicate entities.

### Reasoning level (`SCROLLKEEPER_LLM_EXTRA_BODY`)

Reasoning models can think for minutes before answering. The tested model's chat template defaults to its highest effort level (`xhigh`), and a proxy in front of the LLM may cut long requests (a wake-on-LAN proxy with a 5-minute timeout did). Set the level explicitly:

```
SCROLLKEEPER_LLM_EXTRA_BODY={"chat_template_kwargs": {"reasoning_effort": "medium"}}
```

Timings for one fact-extraction call with the tested model:

| `chat_template_kwargs` | 15-line transcript | 19k-character chunk | Quality |
|---|---|---|---|
| default (`xhigh`) | 235 s | would exceed a 5-minute proxy timeout | good |
| `{"reasoning_effort": "medium"}` | 54 s | 54 s | **best**: most complete facts, claims kept as claims |
| `{"reasoning_effort": "low"}` | 30 s | 48 s | good, a few facts fewer |
| `{"enable_thinking": false}` | 17 s | n/a | states claims as facts, misattributes roles; not recommended |

Use **medium**, or **low** if processing takes too long. These kwargs belong to Qwen3-style chat templates on vLLM. Other models and servers use different options, so check what your model's chat template accepts (vLLM's `POST /tokenize` with `chat_template_kwargs` shows the rendered prompt).

Processing time at medium effort:

- About 1 minute per extraction chunk.
- 15–40 seconds per page rewrite, and about 5 seconds per entity-match check.
- About 4–7 minutes for a short test session.
- A long session that touches 40 or more entities can take 20–30 minutes, in the background.
- `!rebuild-pages` takes about 35–40 seconds per page.

### Embeddings (`SCROLLKEEPER_EMBED_MODEL`)

Embeddings run inside the bot on CPU with ONNX Runtime; no embeddings server is needed. On first start the bot downloads the model from Hugging Face at a pinned revision, checks each file's SHA-256, and keeps it in `data/models/`, so later starts need no download. Until the model is loaded, questions are answered from name and keyword search only.

| `SCROLLKEEPER_EMBED_MODEL` | Download | Peak RAM | 600-token page (2 threads, 4-core Skylake) | License |
|---|---|---|---|---|
| `embeddinggemma-300m` (default) | 310 MB | ~1.5 GB | ~1.3 s | [Gemma terms](https://ai.google.dev/gemma/terms) |
| `qwen3-embedding-0.6b` | 615 MB | ~2.4 GB | ~3 s | Apache-2.0 |

Both are int8 ONNX exports. They retrieved equally well on a small test set; EmbeddingGemma is the default because it is faster on long pages, uses less memory, and separates unrelated questions more clearly. Questions get each model's retrieval instruction; documents don't. Each stored vector records the model and dimension that made it, so changing the model re-embeds everything at the next start (a few minutes for a few hundred pages).

Allow for the peak RAM in the bot's memory limit. Embedding runs one document at a time on `SCROLLKEEPER_EMBED_THREADS` cores.

Without network access to huggingface.co, pre-download the model into a volume and point `SCROLLKEEPER_EMBED_MODEL_DIR` at it:

```bash
python -m scrollkeeper.embeddings /path/to/models embeddinggemma-300m
```

### Speech-to-text

Use the bundled Parakeet-TDT service (see [Speech-to-text service](#speech-to-text-service)).

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

## Recording

- **One track per speaker.** Each recorded player gets one Ogg Opus file per session (`audio/<discord-user-id>.ogg`). Discord's Opus packets are written as they arrive, without re-encoding. That is about 30 MB per speaker-hour, and any player (VLC, ffmpeg, Audacity) can open it.
- **Real timeline.** Packets are placed by their RTP timestamp (a 48 kHz clock that keeps running through silence), and silent gaps are filled in, so a point N seconds into a track is N seconds after that speaker's first packet. If RTP time and arrival time disagree by more than 2 seconds (a reconnect, a client restart), the track follows arrival time instead.
- **Crash-safe.** A writer thread does all file I/O off the event loop. It ends an Ogg page every second and fsyncs every 10 seconds, so a killed pod loses about a second of audio.
- **Restart recovery.** The processing queue is stored in SQLite. On startup, the bot queues any session that was still recording, keeps its audio, notes the gap in the transcript and the channel, and resumes sessions that were mid-processing. A session that kills the bot 3 times in a row is marked failed.
- **Transcription.** After the session, each track is decoded with ffmpeg to 16 kHz mono FLAC (soxr resampling) in a scratch directory, sent whole to the speech-to-text service, and the scratch file is deleted. The Opus tracks are the only audio kept.
- **Transcript.** Each track's words are split into utterances at pauses longer than 0.8 s. Utterances from all speakers are merged by start time, and a speaker's consecutive utterances are joined into one line. Every utterance keeps its own timestamp in the database. The summarizer sees `Speaker: text` lines; fact extraction sees `[HH:MM:SS]` offsets from the session start, which become citations.
- **Older recordings.** Sessions recorded before per-speaker tracks (many short `.skopus` clips) are converted to one Ogg track per speaker the first time `!reprocess-session` runs on them. The old clips are left in place, and you can delete them once the session has reprocessed successfully.

## Storage layout

- `data/scrollkeeper.db`: SQLite database
- `data/sessions/<session-id>/audio/<discord-user-id>.ogg`: one Ogg Opus track per recorded speaker
- `data/sessions/<session-id>/transcript.md`: finalized transcript
- `data/sessions/<session-id>/summary.md`: session notes + cinematic summary
- `data/models/`: downloaded embedding model files
- `data/wiki/<campaign-id>/<Type>/<Name>.md`: Obsidian-style wiki export, one file per entity with `aliases` frontmatter, `[[links]]` and session citations. It is regenerated from the database, so edits there are overwritten; use the commands above.

## Important implementation notes

- Discord voice receive in Python relies on `discord-ext-voice-recv`.
- The bot records one track per speaker and transcribes it after the session ends (see [Recording](#recording)). This is simpler and more reliable than trying to stream partial text live.
- `transcript.md` uses speaker-only lines (`Speaker: text`) without timestamps to reduce context-token overhead in summarization; per-utterance timestamps stay in the database.
- Session summaries are generated from the current session transcript only, so prior campaign notes are not used as summary source material.
- Wiki pages and session summaries are indexed for keyword and embedding search in SQLite; transcripts are searched by keyword with `!deep-question`.
- The wiki pipeline after each session: (1) summary from the transcript only; (2) fact extraction per timestamped transcript chunk, given the entity index, attaching facts to existing entities or proposing new ones; (3) rewrite of every page whose facts changed. Pinned facts are authoritative. Retracting or superseding a fact rebuilds the page from the remaining facts.
- If the voice connection drops mid-session, the bot will try to reconnect to the same channel and continue the session.
- `discord-ext-voice-recv` is pinned to a commit SHA in `pyproject.toml`; change it deliberately, in its own PR. If Python voice receive keeps breaking, the fallback is a small Node `@discordjs/voice` recorder feeding this pipeline.
- The bot calls the configured speech-to-text endpoint once per speaker track after the session ends.
- Summaries, wiki updates, and campaign Q&A go to the configured OpenAI-compatible LLM endpoint; embeddings are computed in-process.
- Long completion posts are split across multiple Discord messages automatically to avoid message-length truncation.
- `!end-session` now queues background processing so users can still run `!campaign-question` while transcription and note generation continue.

## Next improvements

- Incremental live transcript updates during the call
- Better diarization fallback when Discord user audio is unavailable
- Rich slash commands and admin-only maintenance commands
- Structured campaign schema tuning for your exact note taxonomy
