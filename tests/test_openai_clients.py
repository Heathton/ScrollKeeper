from __future__ import annotations

import asyncio
import email.parser
import email.policy
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

from scrollkeeper.config import Settings
from scrollkeeper.health import Heartbeat, start_health_server
from scrollkeeper.llm import LocalAIService, parse_transcription
from test_llm_summary import fake_settings


def fake_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class OpenAIClientTests(unittest.TestCase):
    def test_chat_json_uses_chat_completions_with_json_mode_and_auth(self) -> None:
        service = LocalAIService(fake_settings(llm_api_key="secret"))
        reply = {"choices": [{"message": {"content": '{"a": 1}'}}]}
        with patch("scrollkeeper.llm.requests.post", return_value=fake_response(reply)) as post:
            result = service._chat_json_sync("system", "user")
        self.assertEqual(result, {"a": 1})
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://llm/v1/chat/completions")
        self.assertEqual(kwargs["json"]["model"], "test-model")
        self.assertEqual(kwargs["json"]["response_format"], {"type": "json_object"})
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer secret"})
        self.assertEqual(kwargs["timeout"], (60, 900))

    def test_answer_question_omits_json_mode_and_auth_when_unset(self) -> None:
        service = LocalAIService(fake_settings())
        reply = {"choices": [{"message": {"content": " The crypt. "}}]}
        with patch("scrollkeeper.llm.requests.post", return_value=fake_response(reply)) as post:
            answer = service._answer_question_sync("Where?", "notes")
        self.assertEqual(answer, "The crypt.")
        kwargs = post.call_args.kwargs
        self.assertNotIn("response_format", kwargs["json"])
        self.assertEqual(kwargs["headers"], {})

    def test_embed_reads_openai_embedding_shape(self) -> None:
        service = LocalAIService(fake_settings())
        reply = {"data": [{"embedding": [0.1, 0.2]}]}
        with patch("scrollkeeper.llm.requests.post", return_value=fake_response(reply)) as post:
            vector = service._embed_text_sync("hello")
        self.assertEqual(vector, [0.1, 0.2])
        self.assertEqual(post.call_args.args[0], "http://llm/v1/embeddings")
        self.assertEqual(post.call_args.kwargs["json"], {"model": "test-embed", "input": "hello"})

    def test_transcribe_track_streams_a_multipart_upload_with_word_timestamps(self) -> None:
        service = LocalAIService(fake_settings())
        captured: dict = {}

        def fake_post(url, **kwargs):
            body = kwargs["data"]
            captured["length"] = len(body)
            captured["body"] = b"".join(iter(body))  # what requests would send
            captured["kwargs"] = kwargs
            captured["url"] = url
            return fake_response(
                {
                    "text": " Roll initiative. ",
                    "words": [{"word": "Roll", "start": 1.0, "end": 1.2}, {"word": "initiative.", "start": 1.3, "end": 1.9}],
                    "segments": [{"id": 0, "text": "Roll initiative.", "start": 1.0, "end": 1.9}],
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "track-7.flac"
            audio.write_bytes(b"fLaC" + bytes(range(256)) * 10)
            with patch("scrollkeeper.llm.requests.post", side_effect=fake_post):
                result = service._transcribe_track_sync(audio)

        self.assertEqual(captured["url"], "http://stt/v1/audio/transcriptions")
        self.assertEqual(captured["kwargs"]["timeout"], (60, 600))
        self.assertEqual(captured["length"], len(captured["body"]), "Content-Length matches the streamed body")
        content_type = captured["kwargs"]["headers"]["Content-Type"]
        message = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            f"Content-Type: {content_type}\r\n\r\n".encode() + captured["body"]
        )
        parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
        self.assertEqual(parts["response_format"].get_content().strip(), "verbose_json")
        self.assertEqual(parts["timestamp_granularities[]"].get_content().strip(), "word")
        self.assertEqual(parts["model"].get_content().strip(), "whisper-1")
        self.assertEqual(parts["file"].get_filename(), "track-7.flac")
        self.assertEqual(parts["file"].get_content(), b"fLaC" + bytes(range(256)) * 10)

        self.assertEqual(result.text, "Roll initiative.")
        self.assertEqual([(w.text, w.start, w.end) for w in result.words], [("Roll", 1.0, 1.2), ("initiative.", 1.3, 1.9)])
        self.assertEqual([s.text for s in result.segments], ["Roll initiative."])

    def test_parse_transcription_tolerates_missing_or_bad_timestamps(self) -> None:
        result = parse_transcription({"text": "Hi", "words": [{"word": "Hi"}, "junk", {"word": " ", "start": 0, "end": 1}]})
        self.assertEqual((result.text, result.words, result.segments), ("Hi", [], []))

    def test_stt_timeout_default_covers_whole_tracks(self) -> None:
        with patch.dict(os.environ, {"SCROLLKEEPER_STT_TIMEOUT_SECONDS": ""}, clear=False):
            os.environ.pop("SCROLLKEEPER_STT_TIMEOUT_SECONDS")
            with patch("scrollkeeper.config.load_dotenv"):
                settings = Settings.load()
        self.assertEqual(settings.stt_timeout_seconds, 7200)
        self.assertIsNone(settings.spool_dir)
        self.assertEqual(settings.audio_retention_days, 0)


class WaitNoticeTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_call_posts_notice_once(self) -> None:
        service = LocalAIService(fake_settings(wait_notice_seconds=0))
        notices: list[str] = []

        async def on_wait(message: str) -> None:
            notices.append(message)

        def slow() -> str:
            import time

            time.sleep(0.2)
            return "done"

        result = await service._run_blocking(slow, on_wait=on_wait, wait_message="waking")
        self.assertEqual(result, "done")
        self.assertEqual(notices, ["waking"])

    async def test_fast_call_posts_no_notice(self) -> None:
        service = LocalAIService(fake_settings(wait_notice_seconds=5))
        notices: list[str] = []

        async def on_wait(message: str) -> None:
            notices.append(message)

        result = await service._run_blocking(lambda: "fast", on_wait=on_wait, wait_message="waking")
        self.assertEqual(result, "fast")
        self.assertEqual(notices, [])


class ConfigTests(unittest.TestCase):
    def test_validate_lists_missing_service_settings(self) -> None:
        settings = fake_settings(llm_base_url="", llm_model="")
        with self.assertRaises(RuntimeError) as ctx:
            settings.validate()
        self.assertIn("SCROLLKEEPER_LLM_BASE_URL", str(ctx.exception))
        self.assertIn("SCROLLKEEPER_LLM_MODEL", str(ctx.exception))

    def test_load_reads_env_and_embed_url_defaults_to_llm_url(self) -> None:
        env = {
            "DISCORD_BOT_TOKEN": "t",
            "SCROLLKEEPER_DATA_DIR": tempfile.mkdtemp(),
            "SCROLLKEEPER_STT_BASE_URL": "http://stt:9000/v1/",
            "SCROLLKEEPER_LLM_BASE_URL": "http://llm:8000/v1/",
            "SCROLLKEEPER_LLM_MODEL": "m",
            "SCROLLKEEPER_EMBED_MODEL": "e",
        }
        with patch.dict("os.environ", env, clear=True), patch("scrollkeeper.config.load_dotenv"):
            settings = Settings.load()
        settings.validate()
        self.assertEqual(settings.stt_base_url, "http://stt:9000/v1")
        self.assertEqual(settings.embed_base_url, "http://llm:8000/v1")
        self.assertEqual(settings.llm_extra_body, {})

    def test_llm_extra_body_must_be_a_json_object(self) -> None:
        base = {"SCROLLKEEPER_DATA_DIR": tempfile.mkdtemp()}
        good = {**base, "SCROLLKEEPER_LLM_EXTRA_BODY": '{"chat_template_kwargs": {"enable_thinking": false}}'}
        with patch.dict("os.environ", good, clear=True), patch("scrollkeeper.config.load_dotenv"):
            self.assertEqual(
                Settings.load().llm_extra_body, {"chat_template_kwargs": {"enable_thinking": False}}
            )
        bad = {**base, "SCROLLKEEPER_LLM_EXTRA_BODY": "[1]"}
        with patch.dict("os.environ", bad, clear=True), patch("scrollkeeper.config.load_dotenv"):
            with self.assertRaises(RuntimeError):
                Settings.load()


class HealthTests(unittest.TestCase):
    def test_probe_turns_unhealthy_when_loop_stalls(self) -> None:
        heartbeat = Heartbeat(max_age_seconds=60)
        server = _start_on_free_port(heartbeat)
        port = server.server_address[1]
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz") as resp:
                self.assertEqual(resp.status, 200)
            heartbeat.max_age_seconds = -1
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz")
            self.assertEqual(ctx.exception.code, 503)
        finally:
            server.shutdown()
            server.server_close()

    def test_zero_port_disables_server(self) -> None:
        self.assertIsNone(start_health_server(Heartbeat(), 0))


def _start_on_free_port(heartbeat: Heartbeat):
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return start_health_server(heartbeat, port)
