from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import requests

from .config import Settings
from .models import ENTITY_TYPES, QUEST_STATUSES, TimedText, TranscriptionResult
from .transcript import TranscriptLine, render_compact, split_at_pauses


log = logging.getLogger(__name__)

WaitNotifier = Callable[[str], Awaitable[None]]

CONNECT_TIMEOUT_SECONDS = 60
NOT_IN_NOTES_REPLY = "That's not in the notes."
# Context length assumed when neither the setting nor the server gives one.
DEFAULT_CONTEXT_TOKENS = 32768
# Tokens kept free for the summary reply (and a reasoning model's thinking): this, or a quarter
# of the context if that is more.
SUMMARY_REPLY_TOKENS = 8192
# Characters per token for estimating prompt size without the model's tokenizer. English chat
# is about 4; 3 leaves room for names, punctuation and other languages.
CHARS_PER_TOKEN = 3.0


class LocalAIService:
    """Client for OpenAI-compatible speech-to-text and chat endpoints (embeddings run in-process,
    see `embeddings.py`).

    The LLM host may cold-start for minutes, so reads use long timeouts and callers can pass
    `on_wait` to tell the Discord channel when a request is taking unusually long.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._context_tokens: int | None = None

    async def transcribe_track(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe one whole speaker track (can be hours long; see SCROLLKEEPER_STT_TIMEOUT_SECONDS).

        No wait notice: the speech-to-text service runs on CPU and doesn't cold-start; a long call
        just means a long track. The session manager reports per-track progress instead.
        """
        log.info("Submitting %s to the speech-to-text service", audio_path)
        result = await asyncio.to_thread(self._transcribe_track_sync, audio_path)
        log.info("Speech-to-text returned %s words for %s", len(result.words), audio_path)
        return result

    async def summarize_session(
        self,
        lines: list[TranscriptLine],
        note: str = "",
        glossary: Sequence[str] = (),
        on_wait: WaitNotifier | None = None,
    ) -> dict[str, Any]:
        """Session notes + cinematic recap, from the transcript only (no prior notes, see #7).

        `glossary` is a list of campaign names, used only to correct their spelling. A transcript
        too long for the model's context is summarized in parts cut at pauses, then combined.
        """
        return await self._run_blocking(
            self._summarize_session_sync,
            lines,
            note,
            glossary,
            on_wait=on_wait,
            wait_message="Waking the inference box, this can take a few minutes...",
        )

    async def extract_facts(
        self,
        entity_index: str,
        transcript_chunk: str,
        on_wait: WaitNotifier | None = None,
    ) -> dict[str, Any]:
        return await self._run_blocking(
            self._extract_facts_sync,
            entity_index,
            transcript_chunk,
            on_wait=on_wait,
            wait_message="Waking the inference box, this can take a few minutes...",
        )

    async def extract_journal_facts(
        self,
        entity_index: str,
        text: str,
        source: str,
        subject: str = "",
        on_wait: WaitNotifier | None = None,
    ) -> dict[str, Any]:
        """Facts from game-master text. `source` says what it is ("a dated session recap");
        `subject` is the `#<id> [Type] Name` the entry is about, if any. The result also has
        `subject_type`, the type the model thinks fits the subject ("" without a subject)."""
        return await self._run_blocking(
            self._extract_journal_facts_sync,
            entity_index,
            text,
            source,
            subject,
            on_wait=on_wait,
            wait_message="Waking the inference box, this can take a few minutes...",
        )

    async def match_entity(self, proposal: str, candidates: str, on_wait: WaitNotifier | None = None) -> dict[str, str]:
        """Ask whether a proposed new entity is one of the given existing candidates."""
        return await self._run_blocking(
            self._match_entity_sync,
            proposal,
            candidates,
            on_wait=on_wait,
            wait_message="Waking the inference box, this can take a few minutes...",
        )

    async def rewrite_page(
        self,
        entity_header: str,
        current_page: str,
        pinned_facts: str,
        new_facts: str,
        template: str = "",
        on_wait: WaitNotifier | None = None,
    ) -> dict[str, str]:
        return await self._run_blocking(
            self._rewrite_page_sync,
            entity_header,
            current_page,
            pinned_facts,
            new_facts,
            template,
            on_wait=on_wait,
            wait_message="Waking the inference box, this can take a few minutes...",
        )

    async def answer_question(self, question: str, sources: str, on_wait: WaitNotifier | None = None) -> str:
        return await self._run_blocking(
            self._answer_question_sync,
            question,
            sources,
            on_wait=on_wait,
            wait_message="Waking the inference box, this can take a few minutes...",
        )

    async def _run_blocking(self, func, *args, on_wait: WaitNotifier | None, wait_message: str):
        """Run a blocking call in a worker thread; post `wait_message` once if it runs long."""
        task = asyncio.ensure_future(asyncio.to_thread(func, *args))
        if on_wait is not None:
            done, _ = await asyncio.wait({task}, timeout=self.settings.wait_notice_seconds)
            if not done:
                try:
                    await on_wait(wait_message)
                except Exception:
                    log.warning("Could not post wait notice", exc_info=True)
        return await task

    def _timeout(self, read_seconds: int) -> tuple[int, int]:
        return (CONNECT_TIMEOUT_SECONDS, read_seconds)

    def _headers(self) -> dict[str, str]:
        if self.settings.llm_api_key:
            return {"Authorization": f"Bearer {self.settings.llm_api_key}"}
        return {}

    def _transcribe_track_sync(self, audio_path: Path) -> TranscriptionResult:
        fields = {
            "model": self.settings.stt_model,
            "response_format": "verbose_json",
            "timestamp_granularities[]": "word",
        }
        # The body is streamed from disk: a multi-hour track should not be built in memory.
        with MultipartFileBody(fields, "file", audio_path, "audio/flac") as body:
            response = requests.post(
                f"{self.settings.stt_base_url}/audio/transcriptions",
                data=body,
                headers={"Content-Type": body.content_type},
                timeout=self._timeout(self.settings.stt_timeout_seconds),
            )
        response.raise_for_status()
        return parse_transcription(response.json())

    def _summarize_session_sync(
        self,
        lines: list[TranscriptLine],
        note: str = "",
        glossary: Sequence[str] = (),
    ) -> dict[str, Any]:
        instructions = self._build_summary_instructions(glossary)
        transcript = render_compact(lines, note)
        budget = self._summary_budget_chars(instructions)
        if len(transcript) <= budget:
            return self._summarize_part(instructions, transcript, "single-pass")

        header = len(render_compact([], note))
        parts = split_at_pauses(lines, max(1000, budget - header))
        log.warning(
            "Transcript is %s chars (budget %s); summarizing it in %s parts cut at pauses",
            len(transcript),
            budget,
            len(parts),
        )
        part_payloads = [
            self._summarize_part(instructions, render_compact(part, note), f"part {index}/{len(parts)}")
            for index, part in enumerate(parts, start=1)
        ]
        sections = ["# Transcript", ""]
        for payload in part_payloads:
            sections.extend(
                [payload["session_notes_markdown"], "", payload["cinematic_summary_markdown"], ""]
            )
        combined = "\n".join(sections).strip() + "\n"
        final_instructions = instructions + """
Final synthesis requirements:
- The source is recap text written from consecutive parts of the same session, in order.
- Return one cohesive session-notes section and one cohesive cinematic summary.
- Do not structure output by phase/chunk/part/pass labels unless the players explicitly used those terms in-session.
"""
        return self._summarize_part(final_instructions, combined, "final-from-parts")

    def _summarize_part(self, instructions: str, transcript_markdown: str, stage_label: str) -> dict[str, Any]:
        payload = self._chat_schema_sync(
            instructions, f"Session transcript:\n{transcript_markdown}", "session_summary", SUMMARY_SCHEMA
        )
        normalized = self._normalize_summary_payload(payload)
        if not self._summary_has_content(normalized):
            raise RuntimeError(f"The LLM returned an empty summary ({stage_label}).")
        return normalized

    def _summary_budget_chars(self, instructions: str) -> int:
        """Transcript characters that fit in one summary request next to the instructions and reply."""
        context = self.context_tokens()
        reply = max(SUMMARY_REPLY_TOKENS, context // 4)
        prompt_tokens = len(instructions) / CHARS_PER_TOKEN + 100  # + chat template and framing
        return max(1000, int((context - reply - prompt_tokens) * CHARS_PER_TOKEN))

    def context_tokens(self) -> int:
        """The chat model's context length: the setting, else the server's `max_model_len` (vLLM),
        else a conservative default."""
        if self.settings.llm_context_tokens > 0:
            return self.settings.llm_context_tokens
        if self._context_tokens is None:
            try:
                self._context_tokens = self._fetch_context_tokens()
            except (requests.RequestException, ValueError) as exc:
                # Not cached: the server may just be cold-starting.
                log.warning("Could not read the model's context length from the server: %s", exc)
                return DEFAULT_CONTEXT_TOKENS
        return self._context_tokens

    def _fetch_context_tokens(self) -> int:
        response = requests.get(
            f"{self.settings.llm_base_url}/models",
            headers=self._headers(),
            timeout=self._timeout(self.settings.llm_timeout_seconds),
        )
        response.raise_for_status()
        for model in response.json().get("data", []):
            if isinstance(model, dict) and model.get("id") == self.settings.llm_model:
                length = model.get("max_model_len")
                if isinstance(length, int) and length > 0:
                    log.info("Model %s has a context length of %s tokens", self.settings.llm_model, length)
                    return length
        log.warning(
            "The server does not report a context length for %s; assuming %s tokens "
            "(set SCROLLKEEPER_LLM_CONTEXT_TOKENS)",
            self.settings.llm_model,
            DEFAULT_CONTEXT_TOKENS,
        )
        return DEFAULT_CONTEXT_TOKENS

    def _build_summary_instructions(self, glossary: Sequence[str] = ()) -> str:
        base = """
You are a campaign chronicler for a tabletop RPG.

Write two Markdown fields:
- `session_notes_markdown`: practical session notes (events, decisions, discoveries, loot, open threads).
- `cinematic_summary_markdown`: a cinematic narrative recap of the session.

Rules:
- Use only the provided session transcript as the source of truth.
- Do not invent facts.
- Never return placeholders like "No session notes available." or "No cinematic summary available.".
"""
        if glossary:
            names = "\n".join(f"- {name}" for name in glossary)
            base += f"""
The transcript comes from speech-to-text, which often misspells names. These are the correct
spellings of names in this campaign:
{names}

- When a word or phrase in the transcript is likely a mis-transcription of one of these names,
  write the name as spelled in the list.
- The list is only a spelling aid: do not mention a name unless the transcript refers to it.
"""
        prompt_append = os.getenv("SCROLLKEEPER_SUMMARY_PROMPT_APPEND", "").strip()
        if prompt_append:
            base += f"\n\nAdditional project instructions:\n{prompt_append}\n"
        return base

    def _normalize_summary_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_notes = str(payload.get("session_notes_markdown", "")).strip()
        cinematic = str(payload.get("cinematic_summary_markdown", "")).strip()
        return {
            "session_notes_markdown": session_notes,
            "cinematic_summary_markdown": cinematic,
        }

    def _summary_has_content(self, payload: dict[str, Any]) -> bool:
        sentinels = {
            "",
            "none",
            "n/a",
            "no session notes available.",
            "no cinematic summary available.",
        }
        notes = str(payload.get("session_notes_markdown", "")).strip()
        cinematic = str(payload.get("cinematic_summary_markdown", "")).strip()
        return notes.lower() not in sentinels and cinematic.lower() not in sentinels

    def _extract_facts_sync(self, entity_index: str, transcript_chunk: str) -> dict[str, Any]:
        instructions = f"""
You maintain a campaign wiki for a tabletop RPG. Extract durable facts from one part of a
session transcript. Lines start with [HH:MM:SS] (time into the session) and the speaker.

Known entities are listed as `#<id> [<type>] <name> (aka <aliases>): <description>`.

Rules:
- A fact is one short, self-contained statement about one entity that should still matter in later
  sessions: identity, role, relationships, allegiances, location, possessions, goals, secrets
  revealed, deaths, promises, and world-state changes. Skip table chatter, rules talk, jokes and
  blow-by-blow combat.
- Set `entity` to the `#<id>` of a known entity whenever the fact is about it, even when the
  transcript uses a nickname, title or misspelling. Only propose a new entity when nothing in the
  index matches; then set `entity` to the new entity's exact `name`.
- The transcript comes from speech-to-text, which often misspells names. When a word or phrase is
  likely a mis-transcription of a known entity's name or alias, treat it as that entity and use
  the index's spelling in fact text.
{_ENTITY_RULES}
- `timestamp` is the [HH:MM:SS] of the line the fact comes from.
- Use `alias_updates` when the transcript calls a known entity by a new name or title.
- Use `name_reveals` when the transcript reveals the real name of a known entity that is listed
  under a description or title (e.g. `#8 [Character] The Harbour Master` turns out to be called
  Aldous Penn): give `#8` and the revealed name, attach the facts to `#8`, and do not propose a
  new entity for that name.
- Players speak as their characters; the speaker name is the character's name, and player
  characters are in the index. The game master and the players themselves are not entities.
- Keep claims, rumours and lies as claims: "Sela says the ledger names the killer", not "the
  ledger names the killer". Do not turn a guess or an accusation into a fact.
- Do not give an entity a title or role that the transcript gives to someone else.
- Do not invent facts. If unsure, leave it out. Empty arrays are fine.
"""
        prompt = f"Known entities:\n{entity_index or '(none yet)'}\n\nTranscript part:\n{transcript_chunk}"
        payload = self._chat_schema_sync(instructions, prompt, "fact_extraction", FACT_EXTRACTION_SCHEMA)
        return normalize_extraction_payload(payload)

    def _extract_journal_facts_sync(self, entity_index: str, text: str, source: str, subject: str) -> dict[str, Any]:
        """Facts from game-master-written text (a journal entry or a session recap); see #15."""
        subject_rule = (
            f"""
- The entry is about {subject}. Attach facts about it to that `#<id>`, and set `subject_type` to
  the type that fits it best (usually its current type; e.g. a named ship listed as a Character
  is an Item)."""
            if subject
            else """
- Set `subject_type` to ""."""
        )
        instructions = f"""
You maintain a campaign wiki for a tabletop RPG. Extract durable facts from {source}, written by
the game master. It is prose, not a transcript, and it is authoritative campaign material.

Known entities are listed as `#<id> [<type>] <name> (aka <aliases>): <description>`.

Rules:
- A fact is one short, self-contained statement about one entity that should still matter later:
  identity, role, appearance, relationships, allegiances, location, possessions, goals, secrets,
  deaths, promises, and world-state changes. Skip blow-by-blow combat and game mechanics (stat
  blocks, hit points, DCs, damage dice), except what a unique item or place does.
- Set `entity` to the `#<id>` of a known entity whenever the fact is about it, even when the text
  uses a nickname, title or first name. Only propose a new entity when nothing in the index
  matches; then set `entity` to the new entity's exact `name`.
{_ENTITY_RULES}
- `timestamp` is always "".
- Use `alias_updates` when the text calls a known entity by a new name or title.
- Use `name_reveals` when the text reveals the real name of a known entity that is listed under a
  description or title: give its `#<id>` and the revealed name, and attach the facts to that id.
- Keep claims, rumours and lies as claims: "Sela says the ledger names the killer", not "the
  ledger names the killer". Do not turn a guess or an accusation into a fact.
- Do not give an entity a title or role that the text gives to someone else.
- Do not invent facts. If unsure, leave it out. Empty arrays are fine.{subject_rule}
"""
        prompt = f"Known entities:\n{entity_index or '(none yet)'}\n\nText:\n{text}"
        payload = self._chat_schema_sync(instructions, prompt, "journal_extraction", JOURNAL_EXTRACTION_SCHEMA)
        result = normalize_extraction_payload(payload)
        subject_type = str(payload.get("subject_type", "")).strip()
        result["subject_type"] = subject_type if subject_type in ENTITY_TYPES else ""
        return result

    def _match_entity_sync(self, proposal: str, candidates: str) -> dict[str, str]:
        instructions = """
You maintain the entity list of a tabletop RPG campaign wiki. A new entity was proposed from a
session transcript. Decide whether it is one of the existing candidate entities.

- `same`: the evidence clearly refers to that candidate (another name, a title, a nickname, a
  speech-to-text misspelling, or a description of the same person/place/thing). Set `entity` to
  its `#<id>`.
- `different`: it is clearly none of them, e.g. a different person who shares a first name or
  title. Set `entity` to "".
- `unsure`: the evidence is not enough to tell. Set `entity` to the most likely candidate's
  `#<id>`, or "".
Similar names alone are not enough for `same`; check that the facts fit together.
`reason` is one short sentence.
"""
        prompt = f"Proposed entity:\n{proposal}\n\nCandidates:\n{candidates}"
        payload = self._chat_schema_sync(instructions, prompt, "entity_match", ENTITY_MATCH_SCHEMA)
        decision = str(payload.get("decision", "")).strip()
        return {
            "decision": decision if decision in {"same", "different", "unsure"} else "unsure",
            "entity": str(payload.get("entity", "")).strip(),
            "reason": str(payload.get("reason", "")).strip(),
        }

    def _rewrite_page_sync(
        self,
        entity_header: str,
        current_page: str,
        pinned_facts: str,
        new_facts: str,
        template: str = "",
    ) -> dict[str, str]:
        instructions = """
You maintain one page of a tabletop RPG campaign wiki. Rewrite the page so it includes the new
facts, following the page layout below.

Rules:
- Use only the current page and the listed facts. Do not invent anything.
- Pinned facts are authoritative corrections from the game master: they override anything that
  conflicts with them, including the current page.
- Facts cite their session with its number and the date it was played (e.g. `session 7,
  2024-03-10`). When facts conflict, prefer the one from the later date and mention the change
  if it matters (e.g. "was an ally until session 7").
- Cite facts by id in square brackets after the statement they support, e.g. `[F12]` or `[F12, F15]`.
  Keep existing citations from the current page.
- Write Markdown without a top-level heading. Start with the opening paragraph described in the
  layout (no heading), then use exactly the layout's `##` headings, in that order. Leave out a
  section when no fact supports it; never write empty sections or add headings of your own.
- `short_description` is one line (under 100 characters) saying what the entity is.
- `status`: for a Quest, its current status (offered, active, completed, failed or abandoned);
  for any other entity, an empty string.
"""
        instructions += f"\nPage layout:\n{template}\n" if template else ""
        prompt = (
            f"Entity: {entity_header}\n\n"
            f"Current page:\n{current_page or '(empty)'}\n\n"
            f"Pinned facts:\n{pinned_facts or '(none)'}\n\n"
            f"New facts:\n{new_facts or '(none)'}"
        )
        payload = self._chat_schema_sync(instructions, prompt, "wiki_page", PAGE_SCHEMA)
        return {
            "short_description": str(payload.get("short_description", "")).strip(),
            "markdown": str(payload.get("markdown", "")).strip(),
            "status": str(payload.get("status", "")).strip().lower(),
        }

    def _answer_question_sync(self, question: str, sources: str) -> str:
        instructions = f"""
You answer questions about a tabletop RPG campaign for its players, using only the numbered
sources given (wiki pages, session summaries and transcript excerpts, each tagged like [S1]).

Rules:
- Use only the sources. Do not add outside knowledge, and do not guess.
- Cite every statement. Wiki pages carry citations in parentheses such as
  (session 12, 2024-03-10 @ 01:43:10); copy the one that supports the statement. Otherwise cite the
  source's tag, such as [S2] or [S1, S3].
- When sources disagree, prefer the one from the later date (sessions carry the date they were
  played) and say what changed.
- If the sources do not answer the question, reply exactly: {NOT_IN_NOTES_REPLY}
- Be concise: a few sentences or a short list.
"""
        return self._chat_sync(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": f"Question: {question}\n\nSources:\n{sources}"},
            ],
        )

    def _chat_sync(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            # Server-specific options first (e.g. disabling a reasoning model's thinking phase);
            # the fields below always win.
            **getattr(self.settings, "llm_extra_body", {}),
            "model": self.settings.llm_model,
            "stream": False,
            "messages": messages,
        }
        if response_format is not None:
            body["response_format"] = response_format
        response = requests.post(
            f"{self.settings.llm_base_url}/chat/completions",
            json=body,
            headers=self._headers(),
            timeout=self._timeout(self.settings.llm_timeout_seconds),
        )
        response.raise_for_status()
        choices = response.json().get("choices", [])
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content") or "").strip()

    def _chat_schema_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Chat with schema-enforced JSON output (`response_format: json_schema`, supported by vLLM).

        The server constrains decoding to the schema, so only transport or truncation errors are
        retried, once.
        """
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                payload = json.loads(self._chat_sync(messages, response_format=response_format))
                if isinstance(payload, dict):
                    return payload
                last_error = ValueError(f"{schema_name} response is not a JSON object")
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
            log.warning("%s attempt %s failed: %s", schema_name, attempt, last_error)
        raise RuntimeError(f"Could not get a valid {schema_name} response from the LLM.") from last_error


# Entity rules shared by transcript and journal fact extraction.
_ENTITY_RULES = f"""- A new entity's `name` is the fullest proper name used (e.g. "Varric Thane"), not a nickname
  or title; put nicknames and titles ("Old Varric", "Lord Thane") in `aliases`.
- New entity `type` is one of: {", ".join(ENTITY_TYPES)}. Record plot events as facts on the
  entities involved, not as entities.
- A Quest is a task, job or goal the party is offered or takes on (e.g. "Recover Varric's
  Ledger"); name it with a short imperative title. Record its giver, objective, reward and every
  status change (offered, accepted, progress, completed, failed, abandoned) as facts on the Quest
  only, naming every entity involved by its full name (e.g. "Varric Thane offered the party 200
  gold to recover his ledger from the Black Tide"). Other pages link to the quest from those
  names, so do not repeat quest progress on the other entities.
- Any other fact that matters to several entities is recorded once for each of them, phrased from
  that entity's side (a murder: "Varric Thane paid the Black Tide to murder Aldous Penn" on Varric,
  "Aldous Penn was murdered on Varric Thane's orders" on Aldous Penn)."""

FACT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["new_entities", "facts", "alias_updates", "name_reveals"],
    "properties": {
        "new_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "type", "aliases", "short_description"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": list(ENTITY_TYPES)},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                    "short_description": {"type": "string"},
                },
            },
        },
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["entity", "text", "timestamp"],
                "properties": {
                    "entity": {"type": "string"},
                    "text": {"type": "string"},
                    "timestamp": {"type": "string"},
                },
            },
        },
        "alias_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["entity", "alias"],
                "properties": {
                    "entity": {"type": "string"},
                    "alias": {"type": "string"},
                },
            },
        },
        "name_reveals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["entity", "name"],
                "properties": {
                    "entity": {"type": "string"},
                    "name": {"type": "string"},
                },
            },
        },
    },
}

# Journal extraction also says which type fits the entry's own entity (#15).
JOURNAL_EXTRACTION_SCHEMA: dict[str, Any] = {
    **FACT_EXTRACTION_SCHEMA,
    "required": [*FACT_EXTRACTION_SCHEMA["required"], "subject_type"],
    "properties": {
        **FACT_EXTRACTION_SCHEMA["properties"],
        "subject_type": {"type": "string", "enum": ["", *ENTITY_TYPES]},
    },
}

SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["session_notes_markdown", "cinematic_summary_markdown"],
    "properties": {
        "session_notes_markdown": {"type": "string"},
        "cinematic_summary_markdown": {"type": "string"},
    },
}

ENTITY_MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "entity", "reason"],
    "properties": {
        "decision": {"type": "string", "enum": ["same", "different", "unsure"]},
        "entity": {"type": "string"},
        "reason": {"type": "string"},
    },
}

PAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["short_description", "markdown", "status"],
    "properties": {
        "short_description": {"type": "string"},
        "markdown": {"type": "string"},
        "status": {"type": "string", "enum": ["", *QUEST_STATUSES]},
    },
}


def normalize_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop malformed items so callers can rely on the schema's shape."""

    def items(key: str) -> list[dict[str, Any]]:
        value = payload.get(key)
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    new_entities = []
    for item in items("new_entities"):
        name = str(item.get("name", "")).strip()
        entity_type = str(item.get("type", "")).strip()
        if not name or entity_type not in ENTITY_TYPES:
            continue
        aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
        new_entities.append(
            {
                "name": name,
                "type": entity_type,
                "aliases": [str(alias).strip() for alias in aliases if str(alias).strip()],
                "short_description": str(item.get("short_description", "")).strip(),
            }
        )
    facts = []
    for item in items("facts"):
        entity = str(item.get("entity", "")).strip()
        text = str(item.get("text", "")).strip()
        if entity and text:
            facts.append({"entity": entity, "text": text, "timestamp": str(item.get("timestamp", "")).strip()})
    alias_updates = []
    for item in items("alias_updates"):
        entity = str(item.get("entity", "")).strip()
        alias = str(item.get("alias", "")).strip()
        if entity and alias:
            alias_updates.append({"entity": entity, "alias": alias})
    name_reveals = []
    for item in items("name_reveals"):
        entity = str(item.get("entity", "")).strip()
        name = str(item.get("name", "")).strip()
        if entity and name:
            name_reveals.append({"entity": entity, "name": name})
    return {
        "new_entities": new_entities,
        "facts": facts,
        "alias_updates": alias_updates,
        "name_reveals": name_reveals,
    }


def split_transcript_chunks(transcript_markdown: str, max_chars: int) -> list[str]:
    if len(transcript_markdown) <= max_chars:
        return [transcript_markdown]
    lines = transcript_markdown.splitlines()
    # Preserve markdown heading while chunking the body.
    body_lines = lines[2:] if len(lines) > 2 else lines
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in body_lines:
        rendered = f"{line}\n"
        if current and current_len + len(rendered) > max_chars:
            chunk_text = "# Transcript\n\n" + "".join(current).strip() + "\n"
            chunks.append(chunk_text)
            current = []
            current_len = 0
        current.append(rendered)
        current_len += len(rendered)
    if current:
        chunk_text = "# Transcript\n\n" + "".join(current).strip() + "\n"
        chunks.append(chunk_text)
    return chunks


def parse_transcription(payload: dict[str, Any]) -> TranscriptionResult:
    """Read an OpenAI `verbose_json` transcription: text, plus words and segments with times if present."""

    def timed(items: Any, key: str) -> list[TimedText]:
        parsed = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                parsed.append(TimedText(str(item.get(key, "")).strip(), float(item["start"]), float(item["end"])))
            except (KeyError, TypeError, ValueError):
                continue
        return [entry for entry in parsed if entry.text]

    return TranscriptionResult(
        text=str(payload.get("text", "")).strip(),
        words=timed(payload.get("words"), "word"),
        segments=timed(payload.get("segments"), "text"),
    )


class MultipartFileBody:
    """A multipart/form-data body that streams one file from disk, with a known Content-Length.

    `requests` builds `files=` uploads in memory; passing this as `data=` sends the file in
    blocks instead.
    """

    BLOCK_SIZE = 1 << 20

    def __init__(self, fields: dict[str, str], file_field: str, path: Path, file_type: str) -> None:
        boundary = uuid.uuid4().hex
        self.content_type = f"multipart/form-data; boundary={boundary}"
        head = bytearray()
        for name, value in fields.items():
            head += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        head += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
            f"Content-Type: {file_type}\r\n\r\n"
        ).encode()
        self._head = bytes(head)
        self._tail = f"\r\n--{boundary}--\r\n".encode()
        self._file = path.open("rb")
        self._length = len(self._head) + path.stat().st_size + len(self._tail)
        self._stage = 0  # 0: head, 1: file, 2: tail, 3: done

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        while block := self.read(self.BLOCK_SIZE):
            yield block

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self._length
        out = bytearray()
        while len(out) < size and self._stage < 3:
            if self._stage == 0:
                out += self._head
                self._stage = 1
            elif self._stage == 1:
                block = self._file.read(size - len(out))
                if block:
                    out += block
                else:
                    self._stage = 2
            else:
                out += self._tail
                self._stage = 3
        return bytes(out)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "MultipartFileBody":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
