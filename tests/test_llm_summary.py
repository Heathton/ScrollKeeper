from __future__ import annotations

import json
import os
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

from scrollkeeper.config import Settings
from scrollkeeper.llm import DEFAULT_CONTEXT_TOKENS, LocalAIService
from scrollkeeper.transcript import TranscriptLine


def fake_settings(**overrides) -> Settings:
    values = dict(
        discord_bot_token="token",
        command_prefix="!",
        data_dir=Path("/tmp/scrollkeeper-tests"),
        bot_name="ScrollKeeper",
        stt_base_url="http://stt/v1",
        stt_model="whisper-1",
        stt_timeout_seconds=600,
        llm_base_url="http://llm/v1",
        llm_model="test-model",
        llm_api_key="",
        llm_timeout_seconds=900,
        wait_notice_seconds=20,
        health_port=0,
        llm_context_tokens=32768,
    )
    values.update(overrides)
    return Settings(**values)


GOOD_SUMMARY = {
    "session_notes_markdown": "- The party breached the castle gate.",
    "cinematic_summary_markdown": "Steel rang against stone as the party forced open the gate.",
}


class ChatStubService(LocalAIService):
    """Records every chat request; replies come from `replies` (dicts are sent as JSON, exceptions raised)."""

    def __init__(self, replies: list | None = None, **settings) -> None:
        super().__init__(fake_settings(**settings))
        self.replies = list(replies or [])
        self.requests: list[tuple[list[dict], dict | None]] = []

    def _chat_sync(self, messages, response_format=None) -> str:  # type: ignore[override]
        self.requests.append((messages, response_format))
        reply = self.replies.pop(0) if self.replies else GOOD_SUMMARY
        if isinstance(reply, Exception):
            raise reply
        return json.dumps(reply)


def lines_with_gaps(gaps: list[float], text: str = "x" * 300) -> list[TranscriptLine]:
    """One line per gap: each starts `gap` seconds after the previous line ended and lasts 20 s."""
    lines, at = [], datetime(2026, 1, 1, 20, 0, 0)
    for index, gap in enumerate(gaps):
        at += timedelta(seconds=gap)
        lines.append(TranscriptLine(f"Speaker{index}", at, text, at + timedelta(seconds=20)))
        at += timedelta(seconds=20)
    return lines


class SummaryTests(unittest.TestCase):
    def test_single_pass_uses_the_summary_schema_and_glossary(self) -> None:
        service = ChatStubService()
        lines = lines_with_gaps([0, 1], text="We met Varik at the docks.")
        with patch.dict(os.environ, {"SCROLLKEEPER_SUMMARY_PROMPT_APPEND": "Write in British English."}):
            payload = service._summarize_session_sync(lines, "", ["Varric Thane", "Gullhaven"])

        self.assertEqual(payload, GOOD_SUMMARY)
        self.assertEqual(len(service.requests), 1)
        messages, response_format = service.requests[0]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertEqual(response_format["json_schema"]["name"], "session_summary")
        self.assertTrue(response_format["json_schema"]["strict"])
        system, user = messages[0]["content"], messages[1]["content"]
        self.assertIn("- Varric Thane\n- Gullhaven", system)
        self.assertIn("mis-transcription", system)
        self.assertIn("Write in British English.", system)
        self.assertIn("Speaker0: We met Varik at the docks.", user)

    def test_no_glossary_section_without_names(self) -> None:
        service = ChatStubService()
        service._summarize_session_sync(lines_with_gaps([0]), "", [])
        self.assertNotIn("correct\nspellings", service.requests[0][0][0]["content"])
        self.assertNotIn("mis-transcription", service.requests[0][0][0]["content"])

    def test_placeholder_summary_fails_without_content_retries(self) -> None:
        service = ChatStubService(
            [{"session_notes_markdown": "No session notes available.", "cinematic_summary_markdown": "Recap."}]
        )
        with self.assertRaises(RuntimeError):
            service._summarize_session_sync(lines_with_gaps([0]))
        self.assertEqual(len(service.requests), 1)

    def test_transport_error_is_retried_once(self) -> None:
        service = ChatStubService([requests.ConnectionError("reset"), GOOD_SUMMARY])
        self.assertEqual(service._summarize_session_sync(lines_with_gaps([0])), GOOD_SUMMARY)
        self.assertEqual(len(service.requests), 2)

        service = ChatStubService([requests.ConnectionError("reset"), requests.ConnectionError("reset")])
        with self.assertRaises(RuntimeError):
            service._summarize_session_sync(lines_with_gaps([0]))
        self.assertEqual(len(service.requests), 2)

    def test_long_transcript_is_split_at_the_longest_pause_then_combined(self) -> None:
        # 30 lines of ~313 chars with room for ~13 per request: three parts. The ideal cuts are
        # before lines 10 and 20; the long pauses before lines 9 and 21 are where scenes change.
        service = ChatStubService()
        service._summary_budget_chars = lambda _instructions: 4100
        gaps = [0] + [2] * 29
        gaps[9], gaps[21] = 300, 240
        service._summarize_session_sync(lines_with_gaps(gaps), "Interrupted at 01:00:00.")

        part_prompts = [messages[1]["content"] for messages, _ in service.requests[:-1]]
        self.assertEqual(len(part_prompts), 3)
        first_speakers = [prompt.split("\n\n")[2].split(":")[0] for prompt in part_prompts]
        self.assertEqual(first_speakers, ["Speaker0", "Speaker9", "Speaker21"])
        self.assertTrue(part_prompts[1].rstrip().endswith(f"Speaker20: {'x' * 300}"))
        self.assertTrue(all("_Interrupted at 01:00:00._" in prompt for prompt in part_prompts))
        final_system, final_user = (message["content"] for message in service.requests[-1][0])
        self.assertIn("Do not structure output by phase/chunk/part/pass labels", final_system)
        self.assertIn("# Transcript", final_user)
        self.assertEqual(final_user.count(GOOD_SUMMARY["session_notes_markdown"]), 3)


def fake_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class ContextLengthTests(unittest.TestCase):
    def test_setting_wins_without_asking_the_server(self) -> None:
        service = LocalAIService(fake_settings(llm_context_tokens=65536))
        with patch("scrollkeeper.llm.requests.get") as get:
            self.assertEqual(service.context_tokens(), 65536)
        get.assert_not_called()

    def test_reads_max_model_len_from_the_server_once(self) -> None:
        service = LocalAIService(fake_settings(llm_context_tokens=0, llm_api_key="secret"))
        models = {"data": [{"id": "other", "max_model_len": 4096}, {"id": "test-model", "max_model_len": 150000}]}
        with patch("scrollkeeper.llm.requests.get", return_value=fake_response(models)) as get:
            self.assertEqual(service.context_tokens(), 150000)
            self.assertEqual(service.context_tokens(), 150000)
        get.assert_called_once()
        self.assertEqual(get.call_args.args[0], "http://llm/v1/models")
        self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "Bearer secret"})

    def test_falls_back_and_retries_later_when_the_server_fails(self) -> None:
        service = LocalAIService(fake_settings(llm_context_tokens=0))
        with patch("scrollkeeper.llm.requests.get", side_effect=requests.ConnectionError("down")):
            self.assertEqual(service.context_tokens(), DEFAULT_CONTEXT_TOKENS)
        with patch("scrollkeeper.llm.requests.get", return_value=fake_response({"data": [{"id": "test-model", "max_model_len": 8192}]})):
            self.assertEqual(service.context_tokens(), 8192)

    def test_default_when_the_server_does_not_report_a_length(self) -> None:
        service = LocalAIService(fake_settings(llm_context_tokens=0))
        with patch("scrollkeeper.llm.requests.get", return_value=fake_response({"data": [{"id": "test-model"}]})):
            self.assertEqual(service.context_tokens(), DEFAULT_CONTEXT_TOKENS)

    def test_budget_grows_with_the_context(self) -> None:
        small = LocalAIService(fake_settings(llm_context_tokens=32768))
        large = LocalAIService(fake_settings(llm_context_tokens=150000))
        instructions = small._build_summary_instructions()
        # 32k: 32768 - 8192 reply tokens, minus the instructions, at 3 chars per token.
        self.assertTrue(60000 < small._summary_budget_chars(instructions) < 73728)
        self.assertGreater(large._summary_budget_chars(instructions), 300000)


if __name__ == "__main__":
    unittest.main()
