from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import requests

from .service_manager import DockerServiceManager


log = logging.getLogger(__name__)


class LocalAIService:
    def __init__(self, services: DockerServiceManager) -> None:
        self.services = services

    async def transcribe_audio_segment(self, audio_path: Path) -> str:
        log.info("Submitting %s to Whisper", audio_path)
        await self.services.ensure_whisper_running()
        text = await asyncio.to_thread(self._transcribe_audio_segment_sync, audio_path)
        log.info("Whisper returned transcript for %s", audio_path)
        return text

    async def embed_text(self, text: str) -> list[float]:
        await self.services.ensure_ollama_running()
        embedding = await asyncio.to_thread(self._embed_text_sync, text)
        self.services.mark_ollama_used()
        return embedding

    async def summarize_session(self, transcript_markdown: str, existing_notes_context: str) -> dict[str, Any]:
        await self.services.ensure_ollama_running()
        summary = await asyncio.to_thread(
            self._summarize_session_sync,
            transcript_markdown,
            existing_notes_context,
        )
        self.services.mark_ollama_used()
        return summary

    async def answer_question(self, question: str, note_context: str) -> str:
        await self.services.ensure_ollama_running()
        answer = await asyncio.to_thread(self._answer_question_sync, question, note_context)
        self.services.mark_ollama_used()
        return answer

    def _transcribe_audio_segment_sync(self, audio_path: Path) -> str:
        with audio_path.open("rb") as handle:
            response = requests.post(
                f"{self.services.whisper_base_url}/transcribe",
                files={"file": (audio_path.name, handle, "audio/wav")},
                timeout=600,
            )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("text", "")).strip()

    def _embed_text_sync(self, text: str) -> list[float]:
        response = requests.post(
            f"{self.services.ollama_base_url}/api/embed",
            json={
                "model": self.services.settings.ollama_embed_model,
                "input": text,
            },
            timeout=180,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings", [])
        return list(embeddings[0]) if embeddings else []

    def _summarize_session_sync(self, transcript_markdown: str, existing_notes_context: str) -> dict[str, Any]:
        instructions = """
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
- Use the transcript as the source of truth.
- Produce practical session notes and a cinematic narrative recap.
- Extract durable note updates for Characters, Events, Factions, Locations, Items, Mysteries, and Points of Interest.
- If no durable updates exist, return an empty array for note_updates.
- Do not wrap the JSON in markdown fences.
"""
        prompt = f"""
Existing campaign note context:
{existing_notes_context}

Session transcript:
{transcript_markdown}
"""
        payload = self._chat_json_sync(instructions, prompt)
        return payload

    def _answer_question_sync(self, question: str, note_context: str) -> str:
        instructions = """
You answer questions about a tabletop campaign using only the retrieved note context.
If the answer is uncertain or absent, say that clearly.
Be concise but useful.
"""
        response = requests.post(
            f"{self.services.ollama_base_url}/api/chat",
            json={
                "model": self.services.settings.ollama_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": instructions},
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nRelevant campaign notes:\n{note_context}",
                    },
                ],
            },
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("message", {}).get("content", "")).strip()

    def _chat_json_sync(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = requests.post(
            f"{self.services.ollama_base_url}/api/chat",
            json={
                "model": self.services.settings.ollama_model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        content = str(payload.get("message", {}).get("content", "")).strip()
        return json.loads(content)
