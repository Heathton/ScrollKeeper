from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scrollkeeper.llm import LocalAIService, normalize_extraction_payload
from scrollkeeper.models import Entity, Fact, WikiChangeReport
from scrollkeeper.storage import Storage
from scrollkeeper.wiki import (
    CampaignWiki,
    cited_fact_ids,
    duplicate_reason,
    format_change_report,
    format_entity_index,
    link_entity_names,
    render_citations,
)

from test_llm_summary import fake_settings


class FakeLLM:
    """Stands in for LocalAIService: canned extraction results, pages that cite every fact given."""

    def __init__(self, extractions: list[dict] | None = None) -> None:
        self.extractions = list(extractions or [])
        self.extract_calls: list[tuple[str, str]] = []
        self.rewrite_calls: list[dict[str, str]] = []

    async def extract_facts(self, entity_index: str, transcript_chunk: str, on_wait=None) -> dict:
        self.extract_calls.append((entity_index, transcript_chunk))
        return normalize_extraction_payload(self.extractions.pop(0) if self.extractions else {})

    async def rewrite_page(self, entity_header, current_page, pinned_facts, new_facts, on_wait=None) -> dict:
        self.rewrite_calls.append(
            {"entity": entity_header, "page": current_page, "pinned": pinned_facts, "new": new_facts}
        )
        lines = [line for line in (current_page.splitlines() if current_page else [])]
        for block in (pinned_facts, new_facts):
            for line in block.splitlines():
                fact_id, _, rest = line.partition(" ")
                lines.append(f"- {rest.split(') ', 1)[-1]} {fact_id}")
        return {"short_description": f"About {entity_header}", "markdown": "\n".join(dict.fromkeys(lines))}

    async def embed_text(self, text: str, on_wait=None) -> list[float]:
        return [float(len(text) % 7 + 1), 1.0]

    async def answer_question(self, question: str, note_context: str, on_wait=None) -> str:
        return note_context


def extraction(new_entities=(), facts=(), alias_updates=()) -> dict:
    return {"new_entities": list(new_entities), "facts": list(facts), "alias_updates": list(alias_updates)}


class WikiTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))
        self.llm = FakeLLM()
        self.wiki = CampaignWiki(self.storage, self.llm, fake_settings(extract_chunk_chars=24000))
        self.session_id = self.storage.create_session(1, 2, 3, "Session")

    def tearDown(self) -> None:
        self._tmp.cleanup()


class SessionPipelineTests(WikiTestCase):
    async def test_extraction_creates_entities_attaches_facts_and_writes_pages(self) -> None:
        varric = self.storage.create_entity(1, "Character", "Varric", short_description="Dock boss")
        self.llm.extractions = [
            extraction(
                new_entities=[
                    {"name": "Mira", "type": "Character", "aliases": ["Captain Mira"], "short_description": "Sailor"},
                    {"name": "Unused", "type": "Item", "aliases": [], "short_description": "never referenced"},
                    {"name": "varric", "type": "Character", "aliases": ["Old Varric"], "short_description": ""},
                ],
                facts=[
                    {"entity": f"#{varric}", "text": "Varric owes the party a favour.", "timestamp": "00:10:05"},
                    {"entity": "Mira", "text": "Mira captains the Gull.", "timestamp": "nonsense"},
                    {"entity": "Captain Mira", "text": "Mira lost an eye.", "timestamp": "01:00:00"},
                    {"entity": "Varric", "text": "Varric fears the Hand.", "timestamp": "00:11:00"},
                    {"entity": "Nobody", "text": "Dropped: no such entity.", "timestamp": "00:00:01"},
                ],
            )
        ]

        report = await self.wiki.process_session(1, self.session_id, "# Transcript\n\n[00:10:05] Varric: hi\n")

        self.assertEqual(report.facts_added, 4)
        self.assertEqual([e.canonical_name for e in report.new_entities], ["Mira"])
        self.assertEqual([e.canonical_name for e in report.updated_entities], ["Varric"])
        self.assertEqual(report.page_failures, [])
        # The proposal that matched an existing entity added its alias there instead of a duplicate.
        self.assertIn("Old Varric", self.storage.get_entity(1, varric).aliases)
        self.assertEqual(len(self.storage.list_entities(1)), 2)
        self.assertIn("#1 [Character] Varric: Dock boss", self.llm.extract_calls[0][0])

        mira = self.storage.find_entities_by_name(1, "Mira")[0]
        facts = self.storage.get_entity_facts(mira.id)
        self.assertEqual([f.transcript_ts for f in facts], [None, "01:00:00"])
        page = self.storage.get_page(mira.id)
        self.assertEqual(sorted(page.source_fact_ids), [f.id for f in facts])
        self.assertEqual(self.wiki.stale_entity_ids(1), [])
        self.assertEqual(self.storage.get_entity(1, mira.id).short_description, "About [Character] Mira (aka Captain Mira)")

    async def test_new_facts_are_folded_into_the_existing_page(self) -> None:
        entity_id = self.storage.create_entity(1, "Location", "Sharn")
        self.llm.extractions = [
            extraction(facts=[{"entity": f"#{entity_id}", "text": "Sharn has towers.", "timestamp": "00:00:01"}]),
            extraction(facts=[{"entity": "Sharn", "text": "Sharn floods.", "timestamp": "00:00:02"}]),
        ]
        await self.wiki.process_session(1, self.session_id, "# Transcript\n\n[00:00:01] A: x\n")
        second_session = self.storage.create_session(1, 2, 3, "Next")
        await self.wiki.process_session(1, second_session, "# Transcript\n\n[00:00:02] A: y\n")

        last = self.llm.rewrite_calls[-1]
        self.assertIn("Sharn has towers.", last["page"])
        self.assertNotIn("towers", last["new"])
        self.assertIn("Sharn floods.", last["new"])

    async def test_reprocessing_a_session_replaces_its_facts_and_rebuilds_pages(self) -> None:
        entity_id = self.storage.create_entity(1, "Character", "Varric")
        self.llm.extractions = [
            extraction(facts=[{"entity": f"#{entity_id}", "text": "Wrong fact.", "timestamp": "00:00:01"}]),
            extraction(facts=[{"entity": f"#{entity_id}", "text": "Right fact.", "timestamp": "00:00:01"}]),
        ]
        await self.wiki.process_session(1, self.session_id, "# Transcript\n\n[00:00:01] A: x\n")
        report = await self.wiki.process_session(1, self.session_id, "# Transcript\n\n[00:00:01] A: x\n")

        self.assertEqual(report.facts_retracted, 1)
        self.assertEqual([f.text for f in self.storage.get_entity_facts(entity_id)], ["Right fact."])
        last = self.llm.rewrite_calls[-1]
        self.assertEqual(last["page"], "", "a retracted source fact forces a rebuild from scratch")
        self.assertNotIn("Wrong", self.storage.get_page(entity_id).markdown)

    async def test_chunks_see_entities_created_by_earlier_chunks(self) -> None:
        self.wiki.settings.extract_chunk_chars = 40
        transcript = "# Transcript\n\n" + "\n".join(f"[00:00:0{i}] Speaker: line {i}" for i in range(1, 4))
        self.llm.extractions = [
            extraction(
                new_entities=[{"name": "Mira", "type": "Character", "aliases": [], "short_description": ""}],
                facts=[{"entity": "Mira", "text": "Mira appears.", "timestamp": "00:00:01"}],
            )
        ]
        await self.wiki.process_session(1, self.session_id, transcript)
        self.assertGreater(len(self.llm.extract_calls), 1)
        self.assertIn("Mira", self.llm.extract_calls[1][0])

    async def test_duplicate_candidates_are_reported(self) -> None:
        self.storage.create_entity(1, "Character", "Varric")
        self.llm.extractions = [
            extraction(
                new_entities=[{"name": "Lord Varric", "type": "Faction", "aliases": [], "short_description": ""}],
                facts=[{"entity": "Lord Varric", "text": "Lord Varric rules.", "timestamp": ""}],
            )
        ]
        report = await self.wiki.process_session(1, self.session_id, "# Transcript\n\nx\n")
        self.assertEqual(len(report.possible_duplicates), 1)
        pair = report.possible_duplicates[0]
        self.assertEqual({pair.first.canonical_name, pair.second.canonical_name}, {"Varric", "Lord Varric"})
        self.assertIn("Possible duplicates", format_change_report(report))

    async def test_failed_page_rewrite_is_reported_and_retried_next_time(self) -> None:
        entity_id = self.storage.create_entity(1, "Character", "Varric")
        self.llm.extractions = [extraction(facts=[{"entity": f"#{entity_id}", "text": "Fact.", "timestamp": ""}])]

        async def broken(*_args, **_kwargs):
            raise RuntimeError("LLM down")

        original = self.llm.rewrite_page
        self.llm.rewrite_page = broken
        report = await self.wiki.process_session(1, self.session_id, "# Transcript\n\nx\n")
        self.assertEqual(report.page_failures, ["Varric"])
        self.assertEqual(self.wiki.stale_entity_ids(1), [entity_id])

        self.llm.rewrite_page = original
        self.assertEqual(await self.wiki.refresh_stale_pages(1), [])
        self.assertIsNotNone(self.storage.get_page(entity_id))


class ReviewCommandTests(WikiTestCase):
    async def _entity_with_page(self, name: str, *facts: str) -> Entity:
        entity_id = self.storage.create_entity(1, "Character", name)
        for text in facts:
            self.storage.add_fact(1, entity_id, text, "observed", session_id=self.session_id, transcript_ts="00:01:00")
        await self.wiki.refresh_page(1, entity_id)
        return self.storage.get_entity(1, entity_id)

    async def test_pinned_facts_are_passed_as_authoritative(self) -> None:
        entity = await self._entity_with_page("Varric", "Varric is human.")
        self.storage.add_fact(1, entity.id, "Varric is a dwarf.", "pinned", created_by_user_id=7)
        await self.wiki.refresh_after_change(1, entity.id)
        self.assertIn("Varric is a dwarf.", self.llm.rewrite_calls[-1]["pinned"])
        self.assertEqual(self.llm.rewrite_calls[-1]["new"], "")

    async def test_correct_fact_supersedes_and_rebuilds(self) -> None:
        entity = await self._entity_with_page("Varric", "Varric is human.", "Varric has a crossbow.")
        wrong = self.storage.get_entity_facts(entity.id)[0]
        new_id = self.storage.add_fact(1, entity.id, "Varric is a dwarf.", "pinned", created_by_user_id=7)
        self.storage.supersede_fact(wrong.id, new_id)
        await self.wiki.refresh_after_change(1, entity.id)

        self.assertEqual(self.storage.get_fact(1, wrong.id).superseded_by, new_id)
        page = self.storage.get_page(entity.id).markdown
        self.assertNotIn("human", page)
        self.assertIn("crossbow", page)
        self.assertEqual(self.llm.rewrite_calls[-1]["page"], "")

    async def test_retracting_the_last_fact_removes_the_page(self) -> None:
        entity = await self._entity_with_page("Varric", "Varric is human.")
        fact = self.storage.get_entity_facts(entity.id)[0]
        self.storage.retract_fact(fact.id, "wrong")
        await self.wiki.refresh_after_change(1, entity.id)
        self.assertIsNone(self.storage.get_page(entity.id))

    async def test_merge_rebuilds_target_with_source_facts(self) -> None:
        source = await self._entity_with_page("Lord Varric", "Owns a tower.")
        target = await self._entity_with_page("Varric", "Runs the docks.")
        await self.wiki.merge(1, source, target)
        page = self.storage.get_page(target.id).markdown
        self.assertIn("Owns a tower.", page)
        self.assertIn("Runs the docks.", page)
        self.assertEqual([e.id for e in await self.wiki.resolve_entity(1, "Lord Varric")], [target.id])
        self.assertEqual([e.id for e in await self.wiki.resolve_entity(1, f"#{source.id}")], [target.id])

    async def test_render_entity_lists_sources(self) -> None:
        entity = await self._entity_with_page("Varric", "Runs the docks.")
        rendered = await self.wiki.render_entity(1, entity)
        fact = self.storage.get_entity_facts(entity.id)[0]
        self.assertIn(f"- F{fact.id} (session {self.session_id} @ 00:01:00): Runs the docks.", rendered)

    async def test_answer_question_renders_citations_as_sources(self) -> None:
        await self._entity_with_page("Varric", "Runs the docks.")
        context = await self.wiki.answer_question(1, "Who runs the docks?")
        self.assertIn(f"(session {self.session_id} @ 00:01:00)", context)
        self.assertNotIn("[F", context)

    async def test_export_writes_linked_markdown_files(self) -> None:
        await self._entity_with_page("Varric", "Varric trusts Mira.")
        mira = await self._entity_with_page("Mira", "Mira sails.")
        self.storage.add_alias(mira.id, 'The "Gull" Captain')
        target = self.wiki.export(1)

        varric_file = (target / "Character" / "Varric.md").read_text(encoding="utf-8")
        self.assertIn("# Varric", varric_file)
        self.assertIn("[[Mira|Mira]]", varric_file)
        self.assertIn(f"(session {self.session_id} @ 00:01:00)", varric_file)
        mira_file = (target / "Character" / "Mira.md").read_text(encoding="utf-8")
        self.assertIn('  - "The \\"Gull\\" Captain"', mira_file)


class HelperTests(unittest.TestCase):
    def test_entity_index_describes_only_mentioned_entities(self) -> None:
        entities = [
            Entity(1, 1, "Character", "Varric", ["Lord Varric"], "Dock boss"),
            Entity(2, 1, "Faction", "The Black Hand", [], "Thieves"),
        ]
        index = format_entity_index(entities, "[00:00:01] Mira: Lord Varric, again?")
        self.assertIn("#1 [Character] Varric (aka Lord Varric): Dock boss", index)
        self.assertIn("#2 [Faction] The Black Hand", index)
        self.assertNotIn("Thieves", index)

    def test_duplicate_reasons(self) -> None:
        def entity(name: str, *aliases: str) -> Entity:
            return Entity(0, 1, "Character", name, list(aliases))

        self.assertIsNotNone(duplicate_reason(entity("Varric"), entity("Lord Varric")))
        self.assertIsNotNone(duplicate_reason(entity("Thalrin"), entity("Thalren")))
        self.assertIsNotNone(duplicate_reason(entity("A", "The Hand"), entity("hand")))
        self.assertIsNone(duplicate_reason(entity("Varric"), entity("Mira")))
        self.assertIsNone(duplicate_reason(entity("Al"), entity("Al Bundy")), "short tokens are too common")

    def test_citations(self) -> None:
        facts = {
            3: Fact(3, 1, 1, "observed", "x", session_id=4, transcript_ts="00:01:02"),
            5: Fact(5, 1, 1, "pinned", "y"),
        }
        markdown = "Varric is a dwarf [F5]. He runs the docks [F3, F5, F99]."
        self.assertEqual(cited_fact_ids(markdown), [5, 3, 99])
        self.assertEqual(
            render_citations(markdown, facts),
            "Varric is a dwarf (pinned). He runs the docks (session 4 @ 00:01:02; pinned).",
        )

    def test_links_first_mention_of_other_entities_only(self) -> None:
        me = Entity(1, 1, "Character", "Varric")
        mira = Entity(2, 1, "Character", "Mira", ["Captain Mira"])
        text = link_entity_names("Varric met captain mira. Mira left.", me, [me, mira], {1: "Varric", 2: "Mira"})
        self.assertEqual(text, "Varric met [[Mira|captain mira]]. Mira left.")

    def test_change_report_shows_error(self) -> None:
        text = format_change_report(WikiChangeReport(error="boom"))
        self.assertIn("boom", text)
        self.assertIn("!reprocess-llm", text)


class SchemaRequestTests(unittest.TestCase):
    def _response(self, content: str):
        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"choices": [{"message": {"content": content}}]}

        return Response()

    def test_extraction_uses_json_schema_and_normalizes_output(self) -> None:
        service = LocalAIService(fake_settings())
        content = json.dumps(
            {
                "new_entities": [
                    {"name": "Mira", "type": "Character", "aliases": ["Cap"], "short_description": "x"},
                    {"name": "Bad", "type": "Event", "aliases": [], "short_description": ""},
                ],
                "facts": [{"entity": "Mira", "text": "Sails.", "timestamp": "00:00:01"}, {"entity": "", "text": "x"}],
                "alias_updates": [],
            }
        )
        with patch("scrollkeeper.llm.requests.post", return_value=self._response(content)) as post:
            payload = service._extract_facts_sync("(none)", "# Transcript\n\n[00:00:01] A: hi\n")

        body = post.call_args.kwargs["json"]
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertTrue(body["response_format"]["json_schema"]["strict"])
        self.assertEqual([e["name"] for e in payload["new_entities"]], ["Mira"])
        self.assertEqual(len(payload["facts"]), 1)

    def test_invalid_json_is_retried_once(self) -> None:
        service = LocalAIService(fake_settings())
        responses = [self._response("{not json"), self._response('{"short_description": "s", "markdown": "m"}')]
        with patch("scrollkeeper.llm.requests.post", side_effect=responses) as post:
            page = service._rewrite_page_sync("[Character] A", "", "", "[F1] (pinned) x")
        self.assertEqual(page, {"short_description": "s", "markdown": "m"})
        self.assertEqual(post.call_count, 2)

        with patch("scrollkeeper.llm.requests.post", side_effect=[self._response("{"), self._response("[")]):
            with self.assertRaises(RuntimeError):
                service._rewrite_page_sync("[Character] A", "", "", "")


if __name__ == "__main__":
    unittest.main()
