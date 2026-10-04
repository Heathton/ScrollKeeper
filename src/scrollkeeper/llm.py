from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

import requests

from .config import Settings


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
        existing_notes_context: str,
        on_wait: WaitNotifier | None = None,
    ) -> dict[str, Any]:
        return await self._run_blocking(
            self._summarize_session_sync,
            transcript_markdown,
            existing_notes_context,
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

    def _summarize_session_sync(self, transcript_markdown: str, existing_notes_context: str) -> dict[str, Any]:
        _ = existing_notes_context  # Intentionally ignored to prevent prior-note leakage into session summaries.
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
        final_payload = self._summarize_with_retry(
            combined_chunked_recap,
            min_content_chars=80,
            stage_label="final-from-chunks",
        )
        merged_updates = self._merge_note_updates(
            [update for payload in chunk_payloads for update in payload.get("note_updates", [])]
            + final_payload.get("note_updates", [])
        )
        final_payload["note_updates"] = merged_updates
        return final_payload

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
  "cinematic_summary_markdown": "string",
  "note_updates": [
    {
      "note_type": "string",
      "title": "string",
      "content": "string",
      "metadata": {}
    }
  ]
}

Rules:
- Use only the provided session transcript as the source of truth.
- Produce practical session notes and a cinematic narrative recap.
- For `note_updates`, include only durable world-state updates that should persist across sessions.
- Use only these `note_type` values: Character, Event, Faction, Location, Item, Mystery, PointOfInterest.
- Prefer updating existing concepts over creating near-duplicate titles.
- Do not invent facts; if uncertain, omit from `note_updates`.
- Avoid ephemeral chatter, jokes, and tactical minute-by-minute actions unless they permanently changed the world state.
- Write note `content` as factual bullet points with concrete details (who/what/where/when/why when available).
- If no durable updates exist, return an empty array for note_updates.
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
        normalized_updates: list[dict[str, Any]] = []
        raw_updates = payload.get("note_updates")
        if isinstance(raw_updates, list):
            for item in raw_updates:
                if not isinstance(item, dict):
                    continue
                note_type = str(item.get("note_type", "")).strip()
                title = str(item.get("title", "")).strip()
                content = str(item.get("content", "")).strip()
                metadata = item.get("metadata", {})
                if not note_type or not title or not content:
                    continue
                if not isinstance(metadata, dict):
                    metadata = {}
                normalized_updates.append(
                    {
                        "note_type": note_type,
                        "title": title,
                        "content": content,
                        "metadata": metadata,
                    }
                )
        return {
            "session_notes_markdown": session_notes,
            "cinematic_summary_markdown": cinematic,
            "note_updates": normalized_updates,
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

    def _merge_note_updates(self, updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        for item in updates:
            if not isinstance(item, dict):
                continue
            note_type = str(item.get("note_type", "")).strip()
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            metadata = item.get("metadata", {})
            if not note_type or not title or not content:
                continue
            if not isinstance(metadata, dict):
                metadata = {}
            key = (note_type.lower(), title.lower())
            candidate = {
                "note_type": note_type,
                "title": title,
                "content": content,
                "metadata": metadata,
            }
            existing = merged.get(key)
            if existing is None or len(content) > len(str(existing.get("content", ""))):
                merged[key] = candidate
        return list(merged.values())

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

    def _chat_sync(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        body: dict[str, Any] = {
            "model": self.settings.llm_model,
            "stream": False,
            "messages": messages,
        }
        if json_mode:
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
