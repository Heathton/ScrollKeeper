from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import requests

from .config import Settings
from .models import ENTITY_TYPES


log = logging.getLogger(__name__)

WaitNotifier = Callable[[str], Awaitable[None]]

CONNECT_TIMEOUT_SECONDS = 60


class LocalAIService:
    """Client for OpenAI-compatible speech-to-text, chat, and embedding endpoints.

    The LLM host may cold-start for minutes, so reads use long timeouts and callers can pass
    `on_wait` to tell the Discord channel when a request is taking unusually long.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def transcribe_audio_segment(self, audio_path: Path, on_wait: WaitNotifier | None = None) -> str:
        log.info("Submitting %s to the speech-to-text service", audio_path)
        text = await self._run_blocking(
            self._transcribe_audio_segment_sync,
            audio_path,
            on_wait=on_wait,
            wait_message="Waiting on the speech-to-text service (it may be starting up)...",
        )
        log.info("Speech-to-text returned transcript for %s", audio_path)
        return text

    async def embed_text(self, text: str, on_wait: WaitNotifier | None = None) -> list[float]:
        return await self._run_blocking(
            self._embed_text_sync,
            text,
            on_wait=on_wait,
            wait_message="Waking the inference box, this can take a few minutes...",
        )

    async def summarize_session(
        self,
        transcript_markdown: str,
        on_wait: WaitNotifier | None = None,
    ) -> dict[str, Any]:
        """Session notes + cinematic recap, from the transcript only (no prior notes, see #7)."""
        return await self._run_blocking(
            self._summarize_session_sync,
            transcript_markdown,
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
        on_wait: WaitNotifier | None = None,
    ) -> dict[str, str]:
        return await self._run_blocking(
            self._rewrite_page_sync,
            entity_header,
            current_page,
            pinned_facts,
            new_facts,
            on_wait=on_wait,
            wait_message="Waking the inference box, this can take a few minutes...",
        )

    async def answer_question(self, question: str, note_context: str, on_wait: WaitNotifier | None = None) -> str:
        return await self._run_blocking(
            self._answer_question_sync,
            question,
            note_context,
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

    def _transcribe_audio_segment_sync(self, audio_path: Path) -> str:
        with audio_path.open("rb") as handle:
            response = requests.post(
                f"{self.settings.stt_base_url}/audio/transcriptions",
                files={"file": (audio_path.name, handle, "audio/wav")},
                data={
                    "model": self.settings.stt_model,
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
                timeout=self._timeout(self.settings.stt_timeout_seconds),
            )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text", "")).strip()

    def _embed_text_sync(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.settings.embed_base_url}/embeddings",
            json={"model": self.settings.embed_model, "input": text},
            headers=self._headers(),
            timeout=self._timeout(self.settings.llm_timeout_seconds),
        )
        response.raise_for_status()
        data = response.json().get("data", [])
        return list(data[0].get("embedding", [])) if data else []

    def _summarize_session_sync(self, transcript_markdown: str) -> dict[str, Any]:
        single_pass_max_chars = int(os.getenv("SCROLLKEEPER_SUMMARY_SINGLE_PASS_MAX_CHARS", "90000"))
        chunk_chars = int(os.getenv("SCROLLKEEPER_SUMMARY_CHUNK_CHARS", "45000"))
        if len(transcript_markdown) <= single_pass_max_chars:
            return self._summarize_with_retry(
                transcript_markdown,
                min_content_chars=80,
            )

        chunks = self._split_transcript_chunks(transcript_markdown, max_chars=chunk_chars)
        log.warning(
            "Transcript is %s chars; using chunked summarization (%s chunks, chunk size %s chars)",
            len(transcript_markdown),
            len(chunks),
            chunk_chars,
        )
        chunk_payloads: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_payload = self._summarize_with_retry(
                chunk,
                min_content_chars=40,
                stage_label=f"chunk {index}/{len(chunks)}",
            )
            chunk_payloads.append(chunk_payload)

        chunked_recap_sections = ["# Transcript", ""]
        for payload in chunk_payloads:
            chunked_recap_sections.extend(
                [
                    str(payload["session_notes_markdown"]).strip(),
                    "",
                    str(payload["cinematic_summary_markdown"]).strip(),
                    "",
                ]
            )
        combined_chunked_recap = "\n".join(chunked_recap_sections).strip() + "\n"
        return self._summarize_with_retry(
            combined_chunked_recap,
            min_content_chars=80,
            stage_label="final-from-chunks",
        )

    def _summarize_with_retry(
        self,
        transcript_markdown: str,
        min_content_chars: int,
        stage_label: str = "single-pass",
    ) -> dict[str, Any]:
        instructions = self._build_summary_instructions()
        prompt = f"""
Session transcript:
{transcript_markdown}
"""
        last_error: Exception | None = None
        for attempt in range(1, 4):
            attempt_instructions = instructions
            if attempt > 1:
                attempt_instructions += """
Additional retry requirements:
- The previous response was invalid or too empty.
- Provide substantive markdown in both summary fields.
- Ensure each field has concrete details grounded in the transcript.
"""
            if stage_label == "final-from-chunks":
                attempt_instructions += """
Final synthesis requirements:
- The source may be chunk-level recap text from the same session.
- Return one cohesive session-notes section and one cohesive cinematic summary.
- Do not structure output by phase/chunk/part/pass labels unless the players explicitly used those terms in-session.
"""
            try:
                payload = self._chat_json_sync(attempt_instructions, prompt)
                normalized = self._normalize_summary_payload(payload)
                if self._summary_has_content(normalized, min_content_chars=min_content_chars):
                    return normalized
                log.warning(
                    "Summary %s attempt %s returned low-content output; retrying",
                    stage_label,
                    attempt,
                )
            except Exception as exc:  # pragma: no cover - network/model failures are expected runtime paths
                last_error = exc
                log.warning("Summary %s attempt %s failed: %s", stage_label, attempt, exc)
        if last_error is not None:
            raise RuntimeError(f"Could not generate summary for {stage_label} after retries.") from last_error
        raise RuntimeError(f"Could not generate non-empty summary for {stage_label} after retries.")

    def _build_summary_instructions(self) -> str:
        base = """
You are a campaign chronicler for a tabletop RPG.

Return valid JSON only with this exact schema:
{
  "session_notes_markdown": "string",
  "cinematic_summary_markdown": "string"
}

Rules:
- Use only the provided session transcript as the source of truth.
- Produce practical session notes and a cinematic narrative recap.
- Do not invent facts.
- Do not wrap the JSON in markdown fences.
- Never return placeholders like "No session notes available." or "No cinematic summary available.".
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

    def _summary_has_content(self, payload: dict[str, Any], min_content_chars: int) -> bool:
        sentinels = {
            "",
            "none",
            "n/a",
            "no session notes available.",
            "no cinematic summary available.",
        }
        notes = str(payload.get("session_notes_markdown", "")).strip()
        cinematic = str(payload.get("cinematic_summary_markdown", "")).strip()
        if notes.lower() in sentinels or cinematic.lower() in sentinels:
            return False
        return len(notes) >= min_content_chars and len(cinematic) >= min_content_chars

    def _split_transcript_chunks(self, transcript_markdown: str, max_chars: int) -> list[str]:
        return split_transcript_chunks(transcript_markdown, max_chars)

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
- A new entity's `name` is the fullest proper name used (e.g. "Varric Thane"), not a nickname
  or title; put nicknames and titles ("Old Varric", "Lord Thane") in `aliases`.
- New entity `type` is one of: {", ".join(ENTITY_TYPES)}. Record plot events as facts on the
  entities involved, not as entities.
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
    ) -> dict[str, str]:
        instructions = """
You maintain one page of a tabletop RPG campaign wiki. Rewrite the page so it includes the new
facts.

Rules:
- Use only the current page and the listed facts. Do not invent anything.
- Pinned facts are authoritative corrections from the game master: they override anything that
  conflicts with them, including the current page.
- When other facts conflict, prefer the later session and mention the change if it matters
  (e.g. "was an ally until session 7").
- Cite facts by id in square brackets after the statement they support, e.g. `[F12]` or `[F12, F15]`.
  Keep existing citations from the current page.
- Write Markdown without a top-level heading: a short opening paragraph, then `##` sections only
  when there is enough material (for example Appearance, Relationships, History, Open Questions).
- `short_description` is one line (under 100 characters) saying what the entity is.
"""
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
        }

    def _answer_question_sync(self, question: str, note_context: str) -> str:
        instructions = """
You answer questions about a tabletop campaign using only the retrieved note context.
If the answer is uncertain or absent, say that clearly.
Be concise but useful.
"""
        return self._chat_sync(
            [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nRelevant campaign notes:\n{note_context}",
                },
            ],
        )

    def _chat_sync(
        self,
        messages: list[dict[str, str]],
        json_mode: bool = False,
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
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
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

    def _chat_json_sync(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        content = self._chat_sync(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            json_mode=True,
        )
        return json.loads(content)

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
    "required": ["short_description", "markdown"],
    "properties": {
        "short_description": {"type": "string"},
        "markdown": {"type": "string"},
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
