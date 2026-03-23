from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scrollkeeper.llm import LocalAIService


class FakeServices:
    async def ensure_ollama_running(self) -> None:
        return

    def mark_ollama_used(self) -> None:
        return


class StubLocalAIService(LocalAIService):
    def __init__(self, responses: list[dict]) -> None:
        super().__init__(FakeServices())  # type: ignore[arg-type]
        self._responses = list(responses)

    def _chat_json_sync(self, _system_prompt: str, _user_prompt: str) -> dict:  # type: ignore[override]
        if not self._responses:
            raise RuntimeError("No stub responses remaining")
        return self._responses.pop(0)


class CapturingLocalAIService(LocalAIService):
    def __init__(self) -> None:
        super().__init__(FakeServices())  # type: ignore[arg-type]
        self.calls: list[tuple[str, str]] = []

    def _chat_json_sync(self, system_prompt: str, user_prompt: str) -> dict:  # type: ignore[override]
        self.calls.append((system_prompt, user_prompt))
        return {
            "session_notes_markdown": (
                "- Consolidated note content with concrete details across the whole encounter.\n"
                "- Party movement, combat beats, and key status effects are merged into one cohesive record."
            ),
            "cinematic_summary_markdown": (
                "A coherent cinematic recap grounded in the provided source text, with major turning points, "
                "threat escalation, and the party's response presented as one unified narrative."
            ),
            "note_updates": [],
        }


class SummaryValidationTests(unittest.TestCase):
    def test_summarize_retries_on_placeholder_output(self) -> None:
        service = StubLocalAIService(
            responses=[
                {
                    "session_notes_markdown": "No session notes available.",
                    "cinematic_summary_markdown": "No cinematic summary available.",
                    "note_updates": [],
                },
                {
                    "session_notes_markdown": (
                        "- The party breached the castle gate and coordinated from multiple flanks.\n"
                        "- They preserved civilians where possible and focused on disabling hostile guards."
                    ),
                    "cinematic_summary_markdown": (
                        "Steel rang against stone as the party forced open the gate and flooded the keep. "
                        "They fought through confusion, regrouped under pressure, and pressed into the throne room."
                    ),
                    "note_updates": [],
                },
            ]
        )
        payload = service._summarize_session_sync(
            transcript_markdown="# Transcript\n\nSpeaker: text\n",
            existing_notes_context="No prior campaign notes.",
        )
        self.assertNotEqual(payload["session_notes_markdown"], "No session notes available.")
        self.assertNotEqual(payload["cinematic_summary_markdown"], "No cinematic summary available.")
        self.assertGreater(len(payload["session_notes_markdown"]), 20)
        self.assertGreater(len(payload["cinematic_summary_markdown"]), 20)

    def test_chunked_synthesis_prompt_does_not_include_chunk_headers(self) -> None:
        service = CapturingLocalAIService()
        transcript = "# Transcript\n\n" + "\n".join(
            [f"Speaker: line {index}" for index in range(1, 25)]
        )
        with patch.dict(
            os.environ,
            {
                "SCROLLKEEPER_SUMMARY_SINGLE_PASS_MAX_CHARS": "40",
                "SCROLLKEEPER_SUMMARY_CHUNK_CHARS": "90",
            },
            clear=False,
        ):
            service._summarize_session_sync(
                transcript_markdown=transcript,
                existing_notes_context="No prior campaign notes.",
            )

        self.assertGreaterEqual(len(service.calls), 3)
        final_system_prompt, final_user_prompt = service.calls[-1]
        self.assertIn("# Transcript", final_user_prompt)
        self.assertNotIn("# Chunked Recap", final_user_prompt)
        self.assertNotIn("## Chunk", final_user_prompt)
        self.assertNotIn("### Session Notes", final_user_prompt)
        self.assertNotIn("### Cinematic Summary", final_user_prompt)
        self.assertIn("Do not structure output by phase/chunk/part/pass labels", final_system_prompt)


if __name__ == "__main__":
    unittest.main()
