from __future__ import annotations

import hashlib
import re
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scrollkeeper.embeddings import EMBEDDING_MODELS, LocalEmbedder, fetch_model, truncate_ids
from scrollkeeper.models import SpeakerSegment
from scrollkeeper.search import (
    NO_NOTES_YET,
    NOT_IN_NOTES,
    SearchIndex,
    fts_query,
    pack_vector,
    rank_by_similarity,
    reciprocal_rank_fusion,
    replace_source_tags,
    transcript_docs,
)
from scrollkeeper.storage import MIGRATIONS, Storage
from scrollkeeper.transcript import TranscriptLine
from scrollkeeper.wiki import CampaignWiki, format_change_report

from test_llm_summary import fake_settings
from test_wiki import FakeLLM, extraction


class FakeEmbedder:
    """Bag-of-words vectors: documents sharing words with the query score higher."""

    def __init__(self, name: str = "fake-embed", min_similarity: float = 0.2) -> None:
        self.spec = SimpleNamespace(name=name, min_similarity=min_similarity)
        self.ready = False
        self.fail_load = False
        self.documents: list[str] = []
        self.queries: list[str] = []

    def load(self) -> None:
        if self.fail_load:
            raise RuntimeError("model download failed")
        self.ready = True

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return _bag(text)

    def embed_document(self, title: str, text: str) -> list[float]:
        self.documents.append(title)
        return _bag(f"{title} {text}")


def _bag(text: str) -> list[float]:
    vector = [0.0] * 64
    for word in re.findall(r"\w+", text.casefold()):
        if len(word) > 3:
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
    return vector


class AnsweringLLM(FakeLLM):
    def __init__(self, reply: str = "Varric runs the docks [S1].") -> None:
        super().__init__()
        self.reply = reply
        self.questions: list[tuple[str, str]] = []

    async def answer_question(self, question: str, sources: str, on_wait=None) -> str:
        self.questions.append((question, sources))
        return self.reply


class SearchTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.storage = Storage(Path(self._tmp.name))
        self.llm = AnsweringLLM()
        self.embedder = FakeEmbedder()
        self.search = SearchIndex(self.storage, self.llm, self.embedder)
        self.wiki = CampaignWiki(self.storage, self.llm, fake_settings(wiki_export=False), search=self.search)
        self.session_id = self.storage.create_session(1, 2, 3, "The Docks")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    async def page(self, name: str, *facts: str, entity_type: str = "Character", aliases=()) -> int:
        entity_id = self.storage.create_entity(1, entity_type, name, aliases=list(aliases))
        for text in facts:
            self.storage.add_fact(1, entity_id, text, "observed", session_id=self.session_id, transcript_ts="00:01:00")
        await self.wiki.refresh_after_change(1, entity_id)
        return entity_id

    def finish_session(self, summary: str, lines: list[tuple[int, str, str]]) -> None:
        """Store utterances (seconds into the session, speaker, text) and a written summary."""
        session = self.storage.get_session(self.session_id)
        started = datetime.fromisoformat(session["started_at"])
        for offset, speaker, text in lines:
            at = started + timedelta(seconds=offset)
            self.storage.add_transcript_segment(
                self.session_id,
                SpeakerSegment(7, speaker, speaker, at, at + timedelta(seconds=2), Path("x.ogg"), text),
            )
        base = self.storage.sessions_dir / str(self.session_id)
        base.mkdir(parents=True, exist_ok=True)
        (base / "summary.md").write_text(summary, encoding="utf-8")
        self.storage.finalize_session(self.session_id, str(base / "transcript.md"), str(base / "summary.md"))

    def docs(self, kind: str) -> list[sqlite3.Row]:
        with self.storage.connection() as conn:
            return list(conn.execute("SELECT * FROM search_docs WHERE kind = ? ORDER BY part", (kind,)).fetchall())


class IndexTests(SearchTestCase):
    async def test_pages_summaries_and_transcripts_are_indexed(self) -> None:
        await self.page("Varric", "Varric runs the docks.", aliases=["Old Varric"])
        self.finish_session("# Session Notes\n\nThe party met Varric at the docks.", [(83, "Sera", "Where is the ledger?")])
        pending = await self.search.refresh(1)

        self.assertEqual(pending, 0)
        page = self.docs("page")[0]
        self.assertEqual(page["title"], "Varric, Old Varric")
        self.assertNotIn("[F", page["body"])
        self.assertEqual((page["embed_model"], page["embed_dim"]), ("fake-embed", 64))
        summary = self.docs("summary")[0]
        self.assertIn(f"session {self.session_id}: The Docks", summary["title"])
        self.assertIn("met Varric", summary["body"])
        transcript = self.docs("transcript")[0]
        self.assertEqual(transcript["start_ts"], "00:01:23")
        self.assertIn("[00:01:23] Sera: Where is the ledger?", transcript["body"])
        self.assertIsNone(transcript["embedding"], "transcript chunks are keyword-only")

    async def test_only_changed_documents_are_re_embedded(self) -> None:
        entity_id = await self.page("Varric", "Varric runs the docks.")
        await self.page("Mira", "Mira sails the Gull.")
        self.assertEqual(sorted(self.embedder.documents), ["Mira", "Varric"])

        self.embedder.documents.clear()
        await self.search.refresh(1)
        self.assertEqual(self.embedder.documents, [], "nothing changed")

        await self.wiki.add_alias(1, self.storage.get_entity(1, entity_id), "Old Varric")
        self.assertEqual(self.embedder.documents, ["Varric, Old Varric"])

    async def test_a_new_embedding_model_re_embeds_everything(self) -> None:
        await self.page("Varric", "Varric runs the docks.")
        other = FakeEmbedder(name="other-model")
        index = SearchIndex(self.storage, self.llm, other)
        await index.refresh(1)
        self.assertEqual(other.documents, ["Varric"])
        self.assertEqual(self.docs("page")[0]["embed_model"], "other-model")

    async def test_merged_and_deleted_pages_leave_the_index(self) -> None:
        source = await self.page("Lord Varric", "Owns a tower.")
        target = await self.page("Varric", "Runs the docks.")
        await self.wiki.merge(1, self.storage.get_entity(1, source), self.storage.get_entity(1, target))
        self.assertEqual([row["ref_id"] for row in self.docs("page")], [target])
        self.assertIn("Lord Varric", self.docs("page")[0]["title"])

    async def test_reindex_rebuilds_from_scratch(self) -> None:
        await self.page("Varric", "Runs the docks.")
        self.embedder.documents.clear()
        total, pending = await self.search.reindex(1)
        self.assertEqual((total, pending), (1, 0))
        self.assertEqual(self.embedder.documents, ["Varric"])

    async def test_without_a_model_pages_are_counted_and_still_found_by_keyword(self) -> None:
        self.embedder.fail_load = True
        entity_id = self.storage.create_entity(1, "Location", "Gullhaven")
        self.llm.extractions = [extraction(facts=[{"entity": f"#{entity_id}", "text": "A foggy port town.", "timestamp": ""}])]
        report = await self.wiki.process_session(1, self.session_id, "# Transcript\n\nx\n")

        self.assertEqual(report.pages_without_embedding, 1)
        self.assertIn("not in semantic search yet", format_change_report(report))
        result = await self.search.search(1, "Which town is foggy?")
        self.assertEqual([doc.ref_id for doc in result.docs], [entity_id])


class RetrievalTests(SearchTestCase):
    async def asyncSetUp(self) -> None:
        self.thalrin = await self.page(
            "Thalrin Vey", "Thalrin guards the Mirewood border.", aliases=["the Grey Warden"]
        )
        self.varric = await self.page("Varric Thane", "Varric paid the Black Tide to sink a ship.")
        self.finish_session(
            "# Session Notes\n\nThe party found a silver key under the lighthouse.",
            [(10, "Sera", "I pocket the silver key."), (4000, "Brom", "The lighthouse keeper whistles a strange tune.")],
        )
        await self.search.refresh(1)

    async def test_an_alias_in_the_question_finds_the_page_first(self) -> None:
        result = await self.search.search(1, "what does the grey warden want?")
        self.assertEqual((result.docs[0].kind, result.docs[0].ref_id), ("page", self.thalrin))

    async def test_a_misspelled_name_still_finds_the_page(self) -> None:
        result = await self.search.search(1, "Who is Thalren?")
        self.assertEqual(result.docs[0].ref_id, self.thalrin)

    async def test_keywords_find_session_summaries(self) -> None:
        result = await self.search.search(1, "Where was the silver key?")
        self.assertEqual(result.docs[0].kind, "summary")

    async def test_only_deep_search_reads_transcripts(self) -> None:
        normal = await self.search.search(1, "who whistles a tune?")
        deep = await self.search.search(1, "who whistles a tune?", deep=True)
        self.assertNotIn("transcript", [doc.kind for doc in normal.docs])
        self.assertEqual(deep.docs[0].kind, "transcript")
        self.assertIn("[01:06:40] Brom: The lighthouse keeper whistles", deep.docs[0].body)

    async def test_answer_cites_sources(self) -> None:
        answer = await self.search.answer(1, "What did Varric pay for?")
        question, sources = self.llm.questions[-1]
        self.assertTrue(sources.startswith("[S1] Wiki page [Character] Varric Thane"))
        self.assertIn(f"(session {self.session_id} @ 00:01:00)", sources)
        self.assertEqual(answer, "Varric runs the docks (wiki: Varric Thane).")

    async def test_unrelated_question_is_not_in_the_notes_without_asking_the_llm(self) -> None:
        answer = await self.search.answer(1, "What is the capital of France?")
        self.assertEqual(answer, NOT_IN_NOTES)
        self.assertEqual(self.llm.questions, [])

    async def test_empty_index_says_there_are_no_notes(self) -> None:
        self.assertEqual(await self.search.answer(2, "Who is Varric?"), NO_NOTES_YET)


class HelperTests(unittest.TestCase):
    def test_fts_query_drops_stopwords_and_adds_entity_names(self) -> None:
        self.assertEqual(
            fts_query("Who is the Grey Warden?", ["Thalrin Vey", "the Grey Warden"]),
            '"grey" OR "warden" OR "thalrin vey" OR "the grey warden"',
        )
        self.assertEqual(fts_query("what is it?"), "")

    def test_reciprocal_rank_fusion_rewards_agreement_and_weight(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([([1, 2], 1.0), ([2, 3], 1.0)]), [2, 1, 3])
        self.assertEqual(reciprocal_rank_fusion([([1], 1.0), ([5], 2.0)]), [5, 1])

    def test_rank_by_similarity(self) -> None:
        stored = [(1, pack_vector([1.0, 0.0])), (2, pack_vector([0.6, 0.8])), (3, pack_vector([0.0, 0.0]))]
        ranked = rank_by_similarity([0.0, 1.0], stored)
        self.assertEqual([doc_id for doc_id, _ in ranked], [2, 1, 3])
        self.assertAlmostEqual(ranked[0][1], 0.8, places=5)

    def test_transcript_chunks_keep_whole_lines_and_their_start_time(self) -> None:
        start = datetime(2026, 1, 1, 20, 0, 0)
        lines = [TranscriptLine("Sera", start + timedelta(seconds=60 * i), "x" * 50) for i in range(5)]
        docs = transcript_docs(9, start, lines, max_chars=150)
        self.assertEqual([doc.start_ts for doc in docs], ["00:00:00", "00:02:00", "00:04:00"])
        self.assertEqual([doc.part for doc in docs], [0, 1, 2])
        self.assertTrue(docs[0].body.startswith("[00:00:00] Sera: "))

    def test_source_tags_become_citations(self) -> None:
        labels = {"S1": "session 3 summary", "S2": "session 4 @ 01:43:10"}
        self.assertEqual(
            replace_source_tags("A [S1]. B [S1, S2]. C [S9].", labels),
            "A (session 3 summary). B (session 3 summary; session 4 @ 01:43:10). C .",
        )


class MigrationTests(unittest.TestCase):
    def test_version_2_database_keeps_pages_and_drops_old_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "scrollkeeper.db"
            conn = sqlite3.connect(db)
            conn.executescript(
                """
                CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, guild_id INTEGER NOT NULL,
                    voice_channel_id INTEGER NOT NULL, text_channel_id INTEGER NOT NULL, title TEXT,
                    started_at TEXT NOT NULL, ended_at TEXT, status TEXT NOT NULL, transcript_path TEXT,
                    summary_path TEXT);
                CREATE TABLE transcript_segments (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL, display_name TEXT NOT NULL, character_name TEXT NOT NULL,
                    started_at TEXT NOT NULL, ended_at TEXT NOT NULL, audio_path TEXT NOT NULL, transcript_text TEXT);
                """
            )
            for script in MIGRATIONS[:2]:
                conn.executescript(script)
            conn.execute("PRAGMA user_version = 2")
            conn.execute(
                "INSERT INTO entities (guild_id, type, canonical_name, name_norm, created_at, updated_at) "
                "VALUES (1, 'Character', 'Varric', 'varric', 'now', 'now')"
            )
            conn.execute("INSERT INTO pages VALUES (1, 'Runs the docks.', '[]', '[0.1, 0.2]', 'now')")
            conn.commit()
            conn.close()

            storage = Storage(Path(tmp))
            self.assertEqual(storage.get_page(1).markdown, "Runs the docks.")
            with storage.connection() as conn:
                columns = [row["name"] for row in conn.execute("PRAGMA table_info(pages)")]
                version = conn.execute("PRAGMA user_version").fetchone()[0]
            self.assertNotIn("embedding_json", columns)
            self.assertEqual(version, len(MIGRATIONS))


class FakeTokenizer:
    def encode(self, text: str) -> SimpleNamespace:
        self.last = text
        return SimpleNamespace(ids=list(range(1, len(text.split()) + 1)) + [99])


class FakeOnnxSession:
    def __init__(self, inputs: list[SimpleNamespace]) -> None:
        self._inputs = inputs
        self.feeds: list[dict] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return self._inputs

    def run(self, outputs: list[str], feed: dict) -> list:
        import numpy as np

        self.feeds.append(feed)
        length = feed["input_ids"].shape[1]
        if outputs == ["sentence_embedding"]:
            return [np.array([[3.0, 4.0]], dtype=np.float32)]
        hidden = np.zeros((1, length, 2), dtype=np.float32)
        hidden[0, -1] = [0.0, 2.0]
        return [hidden]


class EmbedderTests(unittest.TestCase):
    def embedder(self, model: str, inputs: list[SimpleNamespace]) -> tuple[LocalEmbedder, FakeTokenizer, FakeOnnxSession]:
        embedder = LocalEmbedder(EMBEDDING_MODELS[model], Path("/nonexistent"))
        tokenizer, session = FakeTokenizer(), FakeOnnxSession(inputs)
        embedder._tokenizer, embedder._session = tokenizer, session
        return embedder, tokenizer, session

    def test_queries_carry_the_instruction_and_documents_their_title(self) -> None:
        names = [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]
        embedder, tokenizer, _ = self.embedder("embeddinggemma-300m", names)
        self.assertEqual(embedder.embed_query("who is Varric?"), [0.6, 0.8])
        self.assertEqual(tokenizer.last, "task: search result | query: who is Varric?")
        embedder.embed_document("Varric", "Runs the docks.")
        self.assertEqual(tokenizer.last, "title: Varric | text: Runs the docks.")

    def test_decoder_models_pool_the_last_token_with_an_empty_cache(self) -> None:
        inputs = [
            SimpleNamespace(name="input_ids"),
            SimpleNamespace(name="attention_mask"),
            SimpleNamespace(name="position_ids"),
            SimpleNamespace(name="past_key_values.0.key", shape=["batch_size", 8, "past_sequence_length", 128]),
        ]
        embedder, tokenizer, session = self.embedder("qwen3-embedding-0.6b", inputs)
        self.assertEqual(embedder.embed_query("who?"), [0.0, 1.0])
        self.assertTrue(tokenizer.last.startswith("Instruct: "))
        self.assertTrue(tokenizer.last.endswith("\nQuery:who?"))
        feed = session.feeds[0]
        self.assertEqual(feed["past_key_values.0.key"].shape, (1, 8, 0, 128))
        self.assertEqual(feed["position_ids"].tolist(), [list(range(feed["input_ids"].shape[1]))])

    def test_truncation_keeps_the_final_token(self) -> None:
        self.assertEqual(truncate_ids([1, 2, 3, 4, 99], 3), [1, 2, 99])
        self.assertEqual(truncate_ids([1, 2], 3), [1, 2])


def fake_download(content: bytes) -> MagicMock:
    response = MagicMock()
    response.__enter__.return_value = response
    response.iter_content.return_value = [content]
    return response


class FetchModelTests(unittest.TestCase):
    def spec(self, content: bytes):
        return SimpleNamespace(
            repo="org/model", revision="abc", files={"onnx/model.onnx": hashlib.sha256(content).hexdigest()}
        )

    def test_downloads_and_verifies_each_file_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scrollkeeper.embeddings.requests.get", return_value=fake_download(b"weights")) as get:
                fetch_model(self.spec(b"weights"), Path(tmp))
                fetch_model(self.spec(b"weights"), Path(tmp))
            self.assertEqual(get.call_count, 1)
            self.assertEqual(get.call_args.args[0], "https://huggingface.co/org/model/resolve/abc/onnx/model.onnx")
            self.assertEqual((Path(tmp) / "onnx/model.onnx").read_bytes(), b"weights")

    def test_a_corrupt_download_is_discarded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("scrollkeeper.embeddings.requests.get", return_value=fake_download(b"tampered")):
                with self.assertRaises(RuntimeError):
                    fetch_model(self.spec(b"weights"), Path(tmp))
            self.assertEqual(list((Path(tmp) / "onnx").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
