from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from scrollkeeper.models import SpeakerSegment
from scrollkeeper.storage import Storage
from scrollkeeper.wiki import CampaignWiki

from test_llm_summary import fake_settings
from test_wiki import FakeLLM, extraction

DISCORD_IMPORT_ERROR: str | None = None
try:
    from scrollkeeper.session_manager import ActiveSession, SessionManager, format_offset
except ModuleNotFoundError as exc:
    SessionManager = None
    DISCORD_IMPORT_ERROR = str(exc)


class SummaryLLM(FakeLLM):
    async def summarize_session(self, transcript_markdown: str, on_wait=None) -> dict:
        self.summary_input = transcript_markdown
        return {"session_notes_markdown": "Notes.", "cinematic_summary_markdown": "Recap."}


@unittest.skipUnless(SessionManager is not None, f"discord dependency unavailable: {DISCORD_IMPORT_ERROR}")
class SessionWikiPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))
        self.llm = SummaryLLM()
        self.wiki = CampaignWiki(self.storage, self.llm, fake_settings())
        self.manager = SessionManager(self.storage, self.llm, self.wiki)
        self.session_id = self.storage.create_session(1, 2, 3, "Test")
        self.started = datetime.fromisoformat(self.storage.get_session(self.session_id)["started_at"])
        for offset, name, text in [(5, "Mira", "Varric owes us."), (3725, "Varric", "I do.")]:
            at = self.started + timedelta(seconds=offset)
            self.storage.add_transcript_segment(
                self.session_id,
                SpeakerSegment(1, name, name, at, at, Path(f"/nonexistent/{offset}.wav"), text),
            )
        base_dir = self.storage.sessions_dir / str(self.session_id)
        base_dir.mkdir(parents=True)
        self.session = ActiveSession(
            session_id=self.session_id,
            guild_id=1,
            voice_channel_id=2,
            text_channel_id=3,
            title="Test",
            base_dir=base_dir,
            audio_dir=base_dir / "audio",
            started_at=self.started,
            closed=True,
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def test_transcript_drives_summary_then_wiki(self) -> None:
        self.llm.extractions = [
            extraction(
                new_entities=[{"name": "Varric", "type": "Character", "aliases": [], "short_description": ""}],
                facts=[{"entity": "Varric", "text": "Varric owes the party.", "timestamp": "00:00:05"}],
            )
        ]
        artifacts = await self.manager._process_closed_session(1, self.session, transcribe_audio=False)

        self.assertNotIn("[00:", self.llm.summary_input, "the summary keeps the compact transcript")
        chunk = self.llm.extract_calls[0][1]
        self.assertIn("[00:00:05] Mira: Varric owes us.", chunk)
        self.assertIn("[01:02:05] Varric: I do.", chunk)
        self.assertEqual([e.canonical_name for e in artifacts.wiki_report.new_entities], ["Varric"])
        self.assertEqual(self.storage.get_session(self.session_id)["status"], "completed")

    async def test_wiki_failure_does_not_fail_the_session(self) -> None:
        async def broken(*_args, **_kwargs):
            raise RuntimeError("extraction exploded")

        self.llm.extract_facts = broken
        artifacts = await self.manager._process_closed_session(1, self.session, transcribe_audio=False)
        self.assertEqual(artifacts.wiki_report.error, "extraction exploded")
        self.assertEqual(artifacts.session_notes_markdown, "Notes.")
        self.assertEqual(self.storage.get_session(self.session_id)["status"], "completed")

    def test_format_offset(self) -> None:
        self.assertEqual(format_offset(timedelta(hours=2, minutes=3, seconds=4)), "02:03:04")
        self.assertEqual(format_offset(timedelta(seconds=-3)), "00:00:00")


if __name__ == "__main__":
    unittest.main()
