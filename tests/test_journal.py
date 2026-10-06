from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from scrollkeeper import storage as storage_module
from scrollkeeper.journal import (
    DEFAULT_RULES,
    MAX_IMPORT_ATTEMPTS,
    SHORT_ENTRY_CHARS,
    JournalEntry,
    JournalImporter,
    format_import_report,
    format_plan,
    html_to_text,
    match_rule,
    normalize_folder,
    parse_export,
    parse_rules,
    plan_import,
    recap_date,
    split_name,
    split_text,
)
from scrollkeeper.llm import JOURNAL_EXTRACTION_SCHEMA, LocalAIService, normalize_extraction_payload
from scrollkeeper.models import JOURNAL_SOURCE_PREFIX, Fact
from scrollkeeper.search import SearchIndex
from scrollkeeper.storage import MIGRATIONS, Storage
from scrollkeeper.wiki import CampaignWiki, spelling_glossary

from test_llm_summary import fake_settings
from test_wiki import FakeLLM, proposal


def item(entry_id: str, name: str, folder: str = "", notes: str = "", gmnotes: str = "", kind: str = "handout", **extra) -> dict:
    return {"type": kind, "id": entry_id, "name": name, "folder": folder, "order": 0, "archived": False,
            "notes": notes, "gmnotes": gmnotes, **extra}


def export(*items: dict) -> bytes:
    return json.dumps(list(items)).encode()


LONG_TEXT = "<p>" + "The keeper of the lighthouse tends the flame every night. " * 10 + "</p>"


class HtmlToTextTests(unittest.TestCase):
    def test_blocks_lists_tables_and_entities(self) -> None:
        html = (
            "<h3>Heading</h3><p>First&nbsp;line<br>second   line</p><ul><li>one</li><li>two</li></ul>"
            "<table><tr><th>d4</th><th>Effect</th></tr><tr><td>1</td><td>Air.</td></tr></table>"
            "<p><img src='x.png'></p><script>ignored()</script><p>Tom &amp; Jerry</p>"
        )
        self.assertEqual(
            html_to_text(html),
            "### Heading\n\nFirst line\nsecond line\n\n- one\n- two\n\nd4 | Effect\n1 | Air.\n\nTom & Jerry",
        )

    def test_empty_markup_is_empty(self) -> None:
        self.assertEqual(html_to_text("<p><br></p>"), "")
        self.assertEqual(html_to_text(""), "")


class RuleTests(unittest.TestCase):
    def test_folder_patterns_match_the_folder_and_what_is_inside_it(self) -> None:
        self.assertEqual(normalize_folder("NPC /  Harbour "), "NPC / Harbour")
        self.assertEqual(match_rule(DEFAULT_RULES, "Chapter 2 Gazetteer", "handout").action, "skip")
        self.assertEqual(match_rule(DEFAULT_RULES, "NPC / Harbour", "handout").entity_type, "Character")
        self.assertEqual(match_rule(DEFAULT_RULES, "session notes / 2024 / May", "handout").action, "session")
        # "Creatures" is a folder name, not a prefix of "Creatures & Monsters".
        self.assertEqual(match_rule(DEFAULT_RULES, "Creatures / Undead", "character").action, "name")
        self.assertIsNone(match_rule(DEFAULT_RULES, "Creatures & Monsters", "handout"))
        # Top-level character sheets are names; top-level handouts match nothing.
        self.assertEqual(match_rule(DEFAULT_RULES, "", "character").action, "name")
        self.assertIsNone(match_rule(DEFAULT_RULES, "", "handout"))

    def test_operator_rules_come_first(self) -> None:
        rules = parse_rules([
            {"folder": "Creatures & Monsters", "action": "name", "type": "Character"},
            {"folder": "Magic Items", "action": "skip"},
        ])
        self.assertEqual(match_rule(rules, "Creatures & Monsters / Generic NPCs", "handout").action, "name")
        self.assertEqual(match_rule(rules, "Magic Items", "handout").action, "skip")
        self.assertEqual(match_rule(rules, "Locations", "handout").action, "entity")

    def test_invalid_rules_are_rejected(self) -> None:
        for bad in ([{"folder": "X", "action": "import"}], [{"action": "skip"}], [{"folder": "X", "action": "entity", "type": "Ship"}]):
            with self.assertRaises(RuntimeError):
                parse_rules(bad)


class NameAndDateTests(unittest.TestCase):
    def test_nicknames_become_aliases(self) -> None:
        self.assertEqual(split_name('Pip "Stormcrow" Hale', "character"), ("Pip Hale", ["Stormcrow"], False))
        self.assertEqual(split_name("Mira", "handout"), ("Mira", [], False))

    def test_copies_of_character_sheets_map_to_the_base_name(self) -> None:
        self.assertEqual(split_name("Mira-Old", "character"), ("Mira", [], True))
        self.assertEqual(split_name("Copy of Varric Thane", "character"), ("Varric Thane", [], True))
        self.assertEqual(split_name("Old Mira Sheet", "character"), ("Mira", [], True))
        self.assertEqual(split_name("Dax 2", "character"), ("Dax", [], True))
        # Only character sheets; a numbered handout is its own entry.
        self.assertEqual(split_name("Letter 2", "handout"), ("Letter 2", [], False))

    def test_recap_dates(self) -> None:
        def dated(name: str, folder: str = "Session Notes / 2024 / May"):
            return recap_date(JournalEntry("x", "handout", name, folder))

        self.assertEqual(dated("05/12/2024"), (date(2024, 5, 12), ""))
        self.assertEqual(dated("05-26-2024 "), (date(2024, 5, 26), ""))
        found, note = dated("05-19-2025")
        self.assertEqual(found, date(2024, 5, 19))
        self.assertIn("year", note)
        self.assertEqual(dated("12/05/2024")[0], date(2024, 5, 12))  # Day first, per the folder's month.
        self.assertEqual(dated("Recap")[0], date(2024, 5, 1))
        self.assertEqual(dated("Recap", folder="Session Notes")[0], None)


class PlanTests(unittest.TestCase):
    def fixture(self) -> list[JournalEntry]:
        return parse_export(export(
            item("r2", "05/19/2024", "Session Notes / 2024 / May", "<p>Second recap.</p>"),
            item("r1", "05/12/2024", "Session Notes / 2024 / May", "<p>First recap.</p>"),
            item("n1", "Varric", "NPC / Docks", "<p>Runs the docks.</p>"),
            item("n2", "Varric", "NPC / Docks", kind="character"),
            item("n3", "Sela", "NPC / Docks"),
            item("l1", "Lair", "Monster Lairs", gmnotes="<p>Only GM notes.</p>"),
            item("d1", "A Letter", "Story Plothook", "<p>Dear Mira...</p>"),
            item("c1", "Bone Hound", "Creatures / Undead", "<p>stat block</p>", kind="character"),
            item("c2", "Mira-Old", "Characters", kind="character"),
            item("b1", "Rules", "Chapter 1 Character Creation", "<p>rules</p>"),
            item("u1", "Generic", "Creatures & Monsters", "<p>text</p>"),
            item("a1", "Gone", "NPC", "<p>x</p>", archived=True),
            {"type": "handout", "name": "no id"},
        ))

    def test_actions_skips_and_order(self) -> None:
        plan = plan_import(self.fixture())
        self.assertEqual(plan.total, 12)
        self.assertEqual(
            [(p.action, p.entry.journal_id) for p in plan.entries],
            [("name", "c1"), ("name", "n2"), ("name", "c2"), ("entity", "l1"), ("entity", "n1"),
             ("session", "r1"), ("session", "r2"), ("document", "d1")],
        )
        self.assertEqual(plan.excluded, {"Chapter 1 Character Creation": 1})
        self.assertEqual(plan.unmatched, {"Creatures & Monsters": 1})
        self.assertEqual((plan.empty, plan.archived), (1, 1))  # Sela: an empty handout.
        lair = next(p for p in plan.entries if p.entry.journal_id == "l1")
        self.assertEqual((lair.entity_type, lair.entry.body), ("PointOfInterest", "Only GM notes."))
        self.assertTrue(next(p for p in plan.entries if p.entry.journal_id == "c2").variant)

    def test_preview_lists_what_would_happen(self) -> None:
        text = format_plan(plan_import(self.fixture()))
        self.assertIn("2 session recaps** (2024-05-12 to 2024-05-19)", text)
        self.assertIn("`Creatures & Monsters` (1)", text)
        self.assertIn("'Mira-Old' looks like a copy", text)
        self.assertIn("2 entries are named 'Varric'", text)

    def test_not_an_export(self) -> None:
        for raw in (b"not json", b'{"a": 1}', b'[{"x": 1}]'):
            with self.assertRaises(ValueError):
                parse_export(raw)

    def test_split_text_keeps_paragraphs_whole(self) -> None:
        text = "\n\n".join(f"Paragraph {i} " + "x" * 30 for i in range(10))
        parts = split_text(text, 100)
        self.assertTrue(all(len(part) <= 100 for part in parts))
        self.assertEqual("\n\n".join(parts), text)
        self.assertEqual(len(split_text("y" * 250, 100)), 3)


class JournalLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.journal_payloads: dict[str, dict] = {}  # by a word of the text
        self.journal_calls: list[tuple[str, str, str, str]] = []
        self.fail_on: str | None = None

    async def extract_journal_facts(self, entity_index, text, source, subject="", on_wait=None) -> dict:
        self.journal_calls.append((entity_index, text, source, subject))
        if self.fail_on and self.fail_on in text:
            raise RuntimeError("the LLM is down")
        payload = next((p for key, p in self.journal_payloads.items() if key in text), {})
        result = normalize_extraction_payload(payload)
        result["subject_type"] = payload.get("subject_type", "")
        return result


class ImporterTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))
        self.campaign = self.storage.active_campaign(1)
        self.llm = JournalLLM()
        self.wiki = CampaignWiki(self.storage, self.llm, fake_settings(extract_chunk_chars=24000))
        self.importer = JournalImporter(self.storage, self.wiki)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def run_import(self, raw: bytes):
        return await self.importer.import_plan(self.campaign.id, plan_import(parse_export(raw)))

    def facts(self, name: str) -> list[Fact]:
        (entity,) = self.storage.find_entities_by_name(self.campaign.id, name)
        return self.storage.get_entity_facts(entity.id)


class ImporterTests(ImporterTestCase):
    async def test_imports_entities_recaps_documents_and_names(self) -> None:
        self.storage.register_character(self.campaign.id, 7, "Mira")
        self.assertGreater(len(html_to_text(LONG_TEXT)), SHORT_ENTRY_CHARS)
        self.llm.journal_payloads = {
            "lighthouse": {
                "subject_type": "PointOfInterest",
                "new_entities": [proposal("Old Tam", "Character", description="Lighthouse keeper")],
                "facts": [
                    {"entity": "#__SUBJECT__", "text": "The flame is lit every night.", "timestamp": ""},
                    {"entity": "Old Tam", "text": "Old Tam keeps the lighthouse.", "timestamp": ""},
                ],
            },
            "Docks recap": {"facts": [{"entity": "Varric", "text": "Varric hired the party.", "timestamp": ""}]},
            "Dear Mira": {"facts": [{"entity": "Mira", "text": "Mira received a letter from her mother.", "timestamp": ""}]},
        }
        raw = export(
            item("n1", "Varric", "NPC / Docks", "<p>Runs the docks.</p>"),
            item("n2", "Varric", "NPC", kind="character"),
            item("l1", "Gull Light", "Locations", LONG_TEXT),
            item("r1", "05/12/2024", "Session Notes / 2024 / May", "<p>Docks recap: the party met Varric.</p>"),
            item("d1", "A Letter", "Story Plothook", "<p>Dear Mira, come home.</p>"),
            item("c1", "Bone Hound", "Creatures / Undead", kind="character"),
            item("c2", "Mira-Old", "Characters", kind="character"),
        )
        # The subject's id isn't known before the import; resolve it when the LLM is called.
        original = self.llm.extract_journal_facts

        async def with_subject(entity_index, text, source, subject="", on_wait=None):
            payload = await original(entity_index, text, source, subject, on_wait)
            for fact in payload["facts"]:
                if fact["entity"] == "#__SUBJECT__":
                    fact["entity"] = subject.split()[0]
            return payload

        self.llm.extract_journal_facts = with_subject
        report = await self.run_import(raw)

        self.assertEqual(report.failures, [])
        self.assertEqual((report.entity_entries, report.sessions, report.documents, report.name_entries), (2, 1, 1, 3))
        # A short entry is one fact as written, with the journal id as provenance.
        (varric_fact, varric_recap) = self.facts("Varric")
        self.assertEqual((varric_fact.kind, varric_fact.text), ("imported", "Varric: Runs the docks."))
        self.assertEqual(varric_fact.source_ref, JOURNAL_SOURCE_PREFIX + "n1")
        self.assertEqual(varric_fact.source_label(), "journal")
        # The long entry went to the LLM with its entity as the subject, and its type was refined.
        (light,) = self.storage.find_entities_by_name(self.campaign.id, "Gull Light")
        self.assertEqual(light.type, "PointOfInterest")
        subject_call = next(call for call in self.llm.journal_calls if "lighthouse" in call[1])
        self.assertEqual(subject_call[3], f"#{light.id} [Location] Gull Light")
        self.assertEqual([f.text for f in self.facts("Old Tam")], ["Old Tam keeps the lighthouse."])
        # The recap is a dated session with a summary and no audio; its facts cite it.
        session = self.storage.get_session(varric_recap.session_id)
        self.assertEqual((session["journal_id"], session["status"], session["started_at"][:10]), ("r1", "completed", "2024-05-12"))
        self.assertIn("Docks recap", Path(session["summary_path"]).read_text())
        self.assertEqual(varric_recap.kind, "imported")
        self.assertIsNone(self.storage.get_latest_session(self.campaign.id))
        recap_call = next(call for call in self.llm.journal_calls if "Docks recap" in call[1])
        self.assertIn("2024-05-12", recap_call[2])
        self.assertEqual(recap_call[3], "")
        # The document's facts went to the registered player character; the copy of her sheet joined her.
        self.assertEqual([f.text for f in self.facts("Mira")], ["Mira received a letter from her mother."])
        self.assertEqual(len(self.storage.find_entities_by_name(self.campaign.id, "Mira")), 1)
        self.assertTrue(any("'Mira-Old' looks like a copy of" in line for line in report.review))
        self.assertTrue(any("2 journal entries were combined into" in line and "Varric" in line for line in report.review))
        # Names are entities without facts, and feed the spelling glossary.
        (hound,) = self.storage.find_entities_by_name(self.campaign.id, "Bone Hound")
        self.assertEqual(self.storage.get_entity_facts(hound.id), [])
        self.assertIn(
            "Bone Hound",
            spelling_glossary(self.storage.list_entities(self.campaign.id), self.storage.list_registered_characters(self.campaign.id)),
        )
        # Pages were written for every entity with facts.
        self.assertIsNotNone(self.storage.get_page(light.id))
        self.assertEqual(self.wiki.stale_entity_ids(self.campaign.id), [])
        text = format_import_report(report)
        self.assertIn("1 session recap(s), 2 entity entr(ies), 1 document(s) and 3 name(s) imported", text)

    async def test_reimport_skips_unchanged_and_replaces_changed_entries(self) -> None:
        first = export(
            item("n1", "Varric", "NPC", "<p>Runs the docks.</p>"),
            item("n2", "Sela", "NPC", "<p>A smuggler.</p>"),
        )
        await self.run_import(first)
        (varric,) = self.storage.find_entities_by_name(self.campaign.id, "Varric")
        pinned = self.storage.add_fact(self.campaign.id, varric.id, "Varric is the harbour master.", "pinned")
        rewrites = len(self.llm.rewrite_calls)

        report = await self.run_import(first)
        self.assertEqual((report.unchanged, report.facts_added), (2, 0))
        self.assertEqual(len(self.llm.rewrite_calls), rewrites + 1)  # Only the pin's page change.

        changed = export(
            item("n1", "Varric Thane", "NPC", "<p>Runs the docks and the smugglers.</p>"),
            item("n2", "Sela", "NPC", "<p>A smuggler.</p>"),
        )
        report = await self.run_import(changed)
        self.assertEqual((report.unchanged, report.facts_added, report.facts_retracted), (1, 1, 1))
        self.assertEqual(report.renamed, [("Varric", "Varric Thane")])
        entity = self.storage.get_entity(self.campaign.id, varric.id)
        self.assertEqual((entity.canonical_name, entity.aliases), ("Varric Thane", ["Varric"]))
        self.assertEqual(
            [(f.id == pinned, f.text) for f in self.storage.get_entity_facts(varric.id)],
            [(True, "Varric is the harbour master."), (False, "Varric Thane: Runs the docks and the smugglers.")],
        )

    async def test_short_entries_name_their_entity_once(self) -> None:
        await self.run_import(export(item("n1", "Sela", "NPC", "<p>Sela smuggles  wine.</p>")))
        self.assertEqual([f.text for f in self.facts("Sela")], ["Sela smuggles wine."])

    async def test_a_copied_sheet_joins_the_one_character_spelled_like_it(self) -> None:
        report = await self.run_import(export(
            item("c1", "Old Pip Hale Sheet", "Characters", kind="character"),
            item("c2", 'Pipp "Stormcrow" Hale', "Characters", kind="character"),
            item("c3", "Copy of Nobody Known", "Characters", kind="character"),
        ))
        (pipp,) = self.storage.find_entities_by_name(self.campaign.id, "Pipp Hale")
        self.assertEqual(pipp.aliases, ["Pip Hale", "Stormcrow"])
        self.assertTrue(any("'Old Pip Hale Sheet' looks like a copy of #" in line for line in report.review))
        self.assertEqual(len(self.storage.find_entities_by_name(self.campaign.id, "Nobody Known")), 1)

    async def test_a_failed_entry_is_reported_and_retried_next_time(self) -> None:
        raw = export(
            item("l1", "Gull Light", "Locations", LONG_TEXT),
            item("n1", "Varric", "NPC", "<p>Runs the docks.</p>"),
        )
        self.llm.fail_on = "lighthouse"
        report = await self.run_import(raw)
        self.assertEqual(len(report.failures), 1)
        self.assertIn("Gull Light", report.failures[0])
        self.assertIsNone(self.storage.get_journal_entry(self.campaign.id, "l1"))
        self.assertIn("Not imported", format_import_report(report))

        self.llm.fail_on = None
        report = await self.run_import(raw)
        self.assertEqual((report.failures, report.unchanged, report.entity_entries), ([], 1, 1))

    async def test_imported_recaps_are_indexed_as_session_summaries(self) -> None:
        await self.run_import(export(item("r1", "05/12/2024", "Session Notes / 2024 / May", "<p>Docks recap.</p>")))
        index = SearchIndex(self.storage, self.llm, None)
        index.sync_documents(self.campaign.id)
        docs = self.storage.search_doc_versions(self.campaign.id)
        self.assertEqual([kind for kind, _, _ in docs], ["summary"])
        (doc,) = self.storage.get_search_docs([doc_id for doc_id, _ in docs.values()]).values()
        self.assertIn("Journal recap (2024-05-12)", doc.title)
        self.assertIn("Docks recap.", doc.body)


class ImportJobTests(ImporterTestCase):
    async def test_job_survives_a_restart_and_cleans_up(self) -> None:
        posted: list[tuple[int, str]] = []

        async def notice(channel: int, message: str) -> None:
            posted.append((channel, message))

        raw = export(item("n1", "Varric", "NPC", "<p>Runs the docks.</p>"))
        # The job row and upload are stored before anything else happens; a "crash" leaves them.
        job_id = self.importer._store_upload(1, self.campaign.id, 55, raw)
        upload = Path(self.storage.running_journal_imports()[0]["path"])
        self.assertTrue(upload.exists())

        restarted = JournalImporter(self.storage, self.wiki)
        restarted.set_notice_handler(notice)
        await restarted.start()
        await restarted.tasks[1]

        self.assertEqual(self.storage.running_journal_imports(), [])
        self.assertFalse(upload.exists())
        self.assertEqual(posted[0], (55, f"Resuming journal import #{job_id} after a restart."))
        self.assertIn("### Journal import", posted[-1][1])
        self.assertIn("Resumed after a restart", posted[-1][1])
        self.assertEqual([f.text for f in self.facts("Varric")], ["Varric: Runs the docks."])
        self.assertIn("finished", restarted.status(1))

    async def test_a_job_that_keeps_crashing_is_given_up(self) -> None:
        posted: list[str] = []

        async def notice(channel: int, message: str) -> None:
            posted.append(message)

        self.importer._store_upload(1, self.campaign.id, 55, export(item("n1", "Varric", "NPC", "<p>x</p>")))
        job_id = int(self.storage.running_journal_imports()[0]["id"])
        for _ in range(MAX_IMPORT_ATTEMPTS):
            self.storage.begin_journal_import_attempt(job_id)
        self.importer.set_notice_handler(notice)
        await self.importer.start()
        await self.importer.tasks[1]
        self.assertEqual(self.storage.running_journal_imports(), [])
        self.assertIn("giving up", posted[-1])

    async def test_begin_refuses_an_export_with_nothing_to_import(self) -> None:
        with self.assertRaises(RuntimeError):
            await self.importer.begin(1, self.campaign.id, 55, export(item("b1", "Rules", "Chapter 1 Rules", "<p>x</p>")))
        with self.assertRaises(ValueError):
            await self.importer.begin(1, self.campaign.id, 55, b"nope")

    async def test_begin_runs_in_the_background_one_at_a_time(self) -> None:
        raw = export(item("n1", "Varric", "NPC", "<p>Runs the docks.</p>"))
        plan = await self.importer.begin(1, self.campaign.id, 55, raw)
        self.assertEqual(len(plan.entries), 1)
        with self.assertRaises(RuntimeError):
            await self.importer.begin(1, self.campaign.id, 55, raw)
        await self.importer.tasks[1]
        self.assertFalse(self.importer.is_running(1))


class JournalSessionStorageTests(unittest.TestCase):
    def test_migration_from_version_4_keeps_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(storage_module, "MIGRATIONS", MIGRATIONS[:4]):
                old = Storage(Path(tmp))
                campaign_id = old.active_campaign(1).id
                with old.connection() as conn:
                    session_id = conn.execute(
                        "INSERT INTO sessions (guild_id, campaign_id, voice_channel_id, text_channel_id, started_at, status)"
                        " VALUES (1, ?, 2, 3, '2026-01-01T20:00:00', 'completed')",
                        (campaign_id,),
                    ).lastrowid
            storage = Storage(Path(tmp))
            with storage.connection() as conn:
                self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], len(MIGRATIONS))
            campaign = storage.active_campaign(1)
            self.assertEqual(int(storage.get_latest_session(campaign.id)["id"]), session_id)
            self.assertIsNone(storage.get_session(session_id)["journal_id"])

    def test_retracting_a_source_leaves_pinned_and_other_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            storage = Storage(Path(tmp))
            campaign = storage.active_campaign(1)
            entity = storage.create_entity(campaign.id, "Character", "Varric")
            imported = storage.add_fact(campaign.id, entity, "a", "imported", source_ref="journal:x")
            storage.add_fact(campaign.id, entity, "b", "imported", source_ref="journal:y")
            storage.add_fact(campaign.id, entity, "c", "pinned", source_ref="journal:x")
            self.assertEqual(storage.retract_source_facts(campaign.id, "journal:x", "test"), (1, {entity}))
            self.assertEqual([f.text for f in storage.get_entity_facts(entity)], ["b", "c"])
            self.assertFalse(storage.get_fact(campaign.id, imported).active)


class JournalPromptTests(unittest.TestCase):
    def test_journal_extraction_asks_for_the_subject_type(self) -> None:
        self.assertIn("subject_type", JOURNAL_EXTRACTION_SCHEMA["required"])
        service = LocalAIService(fake_settings())
        captured = {}

        def fake_chat(system, user, name, schema):
            captured.update(system=system, user=user, name=name, schema=schema)
            return {"new_entities": [], "facts": [{"entity": "#3", "text": "Lit nightly.", "timestamp": ""}],
                    "alias_updates": [], "name_reveals": [], "subject_type": "PointOfInterest"}

        with patch.object(service, "_chat_schema_sync", side_effect=fake_chat):
            payload = service._extract_journal_facts_sync("#3 [Location] Gull Light", "Text", "a journal entry", "#3 [Location] Gull Light")
        self.assertEqual(payload["subject_type"], "PointOfInterest")
        self.assertEqual(payload["facts"][0]["text"], "Lit nightly.")
        self.assertIn("The entry is about #3 [Location] Gull Light", captured["system"])
        self.assertIn("not a transcript", captured["system"])
        self.assertIs(captured["schema"], JOURNAL_EXTRACTION_SCHEMA)


if __name__ == "__main__":
    unittest.main()
