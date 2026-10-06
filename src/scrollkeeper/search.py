"""Hybrid retrieval over the campaign record, and cited answers (#8).

The searchable corpus is derived from the database: one document per wiki page, one per session
summary, and (for deep search) timestamped transcript chunks. `refresh` brings it up to date
after any change; documents are re-embedded only when their text changes or the configured
embedding model does.

A question is matched three ways and the rankings are merged with reciprocal rank fusion:

1. entity names and aliases appearing in the question (their pages, boosted),
2. SQLite FTS5 keyword search (BM25) over titles and text,
3. vector similarity with the in-process embedding model (pages and summaries).

Answers cite their sources, and the bot says the answer is not in the notes when retrieval
finds nothing relevant instead of letting the LLM guess.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .llm import NOT_IN_NOTES_REPLY
from .models import SearchDoc, session_date, session_label
from .storage import Storage
from .transcript import format_offset, merge_lines
from .wiki import (
    cited_fact_ids,
    entity_header,
    linked_quests,
    mentioned_entity_ids,
    page_embedding_text,
    quest_list,
    render_citations,
)

if TYPE_CHECKING:
    from .embeddings import LocalEmbedder
    from .llm import LocalAIService


log = logging.getLogger(__name__)

WaitNotifier = Callable[[str], Awaitable[None]]

# Documents a normal question searches; deep search adds transcript chunks (keywords and names
# only: embedding every chunk of every session would cost hours of CPU).
SEARCH_KINDS = ("page", "summary")
DEEP_KINDS = ("page", "summary", "transcript")
EMBEDDED_KINDS = SEARCH_KINDS

TRANSCRIPT_CHUNK_CHARS = 1500
RRF_K = 60
NAME_WEIGHT = 2.0
CANDIDATES_PER_METHOD = 20
CONTEXT_DOCS = 8
DEEP_CONTEXT_DOCS = 12
DOC_CONTEXT_CHARS = 6000
CONTEXT_CHARS = 36000
# How long to wait before retrying a failed model download/load.
LOAD_RETRY_SECONDS = 600

NOT_IN_NOTES = NOT_IN_NOTES_REPLY
NO_NOTES_YET = "I don't have any campaign notes yet."
SOURCE_TAG_RE = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")

STOPWORDS = frozenset(
    """
    a about after again all also am an and any are as at be because been before being but by can
    could did do does doing done during each for from had has have having he her here hers him his
    how i if in into is it its just me more most my no nor not now of off on once only or other our
    out over own same she should so some such than that the their them then there these they this
    those through to too under until up very was we were what when where which while who whom whose
    why will with would you your yours tell know think anything something someone happened happen
    did does say said
    """.split()
)


@dataclass(slots=True)
class RetrievalResult:
    docs: list[SearchDoc] = field(default_factory=list)
    weak: bool = False
    best_similarity: float | None = None


class SearchIndex:
    def __init__(self, storage: Storage, llm: "LocalAIService", embedder: "LocalEmbedder | None") -> None:
        self.storage = storage
        self.llm = llm
        self.embedder = embedder
        self._lock = asyncio.Lock()
        self._load_failed_at: float | None = None
        self._start_task: asyncio.Task | None = None

    # --- Keeping the index current -------------------------------------------------------

    def start(self) -> None:
        """Load the model and bring every campaign's index up to date, in the background.

        A changed `SCROLLKEEPER_EMBED_MODEL` re-embeds everything here, since stored vectors
        record the model that made them.
        """
        if self._start_task is None:
            self._start_task = asyncio.create_task(self._refresh_all())

    async def _refresh_all(self) -> None:
        try:
            for campaign_id in await asyncio.to_thread(self.storage.search_campaign_ids):
                pending = await self.refresh(campaign_id)
                if pending:
                    log.warning("Campaign %s: %s document(s) are not embedded yet", campaign_id, pending)
        except Exception:
            log.exception("Could not refresh the search index at startup")

    async def refresh(self, campaign_id: int) -> int:
        """Index new or changed pages, summaries and transcripts and embed what needs it.

        Returns how many documents still have no embedding from the current model.
        """
        async with self._lock:
            await asyncio.to_thread(self.sync_documents, campaign_id)
            return await self._embed_pending(campaign_id)

    async def reindex(self, campaign_id: int) -> tuple[int, int]:
        """Rebuild a campaign's index from scratch. Returns (documents, documents without embedding)."""
        async with self._lock:
            await asyncio.to_thread(self.storage.clear_search_index, campaign_id)
            await asyncio.to_thread(self.sync_documents, campaign_id)
            pending = await self._embed_pending(campaign_id)
            total = len(await asyncio.to_thread(self.storage.search_doc_versions, campaign_id))
        return total, pending

    async def pending_count(self, campaign_id: int) -> int:
        model = self.embedder.spec.name if self.embedder else ""
        return await asyncio.to_thread(self.storage.count_unembedded_docs, campaign_id, model, EMBEDDED_KINDS)

    def sync_documents(self, campaign_id: int) -> None:
        """Make the stored documents match the current pages and summarized sessions (blocking)."""
        versions: dict[tuple[str, int], str] = {}
        for (kind, ref_id, _part), (_doc_id, version) in self.storage.search_doc_versions(campaign_id).items():
            versions[(kind, ref_id)] = version
        wanted: set[tuple[str, int]] = set()

        for row in self.storage.page_doc_sources(campaign_id):
            entity_id, version = int(row["entity_id"]), row["version"]
            wanted.add(("page", entity_id))
            if versions.get(("page", entity_id)) == version:
                continue
            entity = self.storage.get_entity(campaign_id, entity_id)
            page = self.storage.get_page(entity_id)
            if entity is None or page is None:
                continue
            doc = SearchDoc("page", entity_id, ", ".join(entity.names()), page_embedding_text(entity, page.markdown))
            self.storage.replace_search_docs(campaign_id, "page", entity_id, [doc], version)

        for session in self.storage.summarized_sessions(campaign_id):
            # The number is part of the version: titles carry it (and changed with #24).
            session_id, version = int(session["id"]), f"{session['ended_at'] or ''}|{session['number']}"
            wanted.update({("summary", session_id), ("transcript", session_id)})
            if versions.get(("summary", session_id)) != version:
                # The stored path is absolute; fall back to the usual place if the data dir moved.
                summary = _read_text(Path(session["summary_path"])) or _read_text(
                    self.storage.sessions_dir / str(session_id) / "summary.md"
                )
                docs = [summary_doc(session, summary)] if summary.strip() else []
                self.storage.replace_search_docs(campaign_id, "summary", session_id, docs, version)
            if versions.get(("transcript", session_id)) != version:
                started_at = datetime.fromisoformat(session["started_at"])
                lines = merge_lines(self.storage.get_session_segments(session_id))
                docs = transcript_docs(session_id, started_at, lines, number=session["number"])
                self.storage.replace_search_docs(campaign_id, "transcript", session_id, docs, version)

        for kind, ref_id in set(versions) - wanted:
            self.storage.delete_search_docs(campaign_id, kind, ref_id)

    async def _embed_pending(self, campaign_id: int) -> int:
        if not await self._ensure_embedder():
            return await self.pending_count(campaign_id)
        assert self.embedder is not None
        spec = self.embedder.spec
        docs = await asyncio.to_thread(self.storage.docs_needing_embedding, campaign_id, spec.name, EMBEDDED_KINDS)
        if docs:
            log.info("Campaign %s: embedding %s document(s) with %s", campaign_id, len(docs), spec.name)
        for doc in docs:
            try:
                vector = await asyncio.to_thread(self.embedder.embed_document, doc.title, doc.body)
            except Exception:
                log.exception("Could not embed search document %s", doc.id)
                continue
            await asyncio.to_thread(
                self.storage.set_doc_embedding, doc.id, doc.title, doc.body, spec.name, pack_vector(vector), len(vector)
            )
        return await self.pending_count(campaign_id)

    async def _ensure_embedder(self) -> bool:
        """Load the embedding model (downloading it the first time). A failure is retried later;
        meanwhile names and keywords still work."""
        if self.embedder is None:
            return False
        if self.embedder.ready:
            return True
        if self._load_failed_at is not None and time.monotonic() - self._load_failed_at < LOAD_RETRY_SECONDS:
            return False
        try:
            await asyncio.to_thread(self.embedder.load)
            self._load_failed_at = None
            return True
        except Exception:
            self._load_failed_at = time.monotonic()
            log.exception("Could not load the embedding model %s", self.embedder.spec.name)
            return False

    # --- Retrieval ------------------------------------------------------------------------

    async def search(self, campaign_id: int, question: str, deep: bool = False) -> RetrievalResult:
        kinds = DEEP_KINDS if deep else SEARCH_KINDS
        entities = await asyncio.to_thread(self.storage.list_entities, campaign_id)
        mentioned = [entity for entity in entities if entity.id in mentioned_entity_ids(entities, question)]
        page_ids = await asyncio.to_thread(self.storage.page_doc_ids, campaign_id, [entity.id for entity in mentioned])
        name_hits = [page_ids[entity.id] for entity in mentioned if entity.id in page_ids]

        match = fts_query(question, [name for entity in mentioned for name in entity.names()])
        keyword_hits = (
            await asyncio.to_thread(self.storage.keyword_search, campaign_id, match, kinds, CANDIDATES_PER_METHOD)
            if match
            else []
        )

        vector_hits: list[int] = []
        best: float | None = None
        # Never wait for a model download here; until it is loaded, names and keywords answer.
        if self.embedder is not None and self.embedder.ready:
            spec = self.embedder.spec
            query = await asyncio.to_thread(self.embedder.embed_query, question)
            stored = await asyncio.to_thread(
                self.storage.doc_embeddings, campaign_id, spec.name, len(query), tuple(k for k in kinds if k in EMBEDDED_KINDS)
            )
            scored = rank_by_similarity(query, stored)
            best = scored[0][1] if scored else None
            vector_hits = [doc_id for doc_id, score in scored[:CANDIDATES_PER_METHOD] if score >= spec.min_similarity]

        fused = reciprocal_rank_fusion([(name_hits, NAME_WEIGHT), (keyword_hits, 1.0), (vector_hits, 1.0)])
        limit = DEEP_CONTEXT_DOCS if deep else CONTEXT_DOCS
        docs_by_id = await asyncio.to_thread(self.storage.get_search_docs, fused[:limit])
        docs = [docs_by_id[doc_id] for doc_id in fused[:limit] if doc_id in docs_by_id]
        weak = not name_hits and not keyword_hits and not vector_hits
        return RetrievalResult(docs=docs, weak=weak, best_similarity=best)

    async def answer(
        self,
        campaign_id: int,
        question: str,
        deep: bool = False,
        on_wait: WaitNotifier | None = None,
    ) -> str:
        if not await asyncio.to_thread(self.storage.search_doc_versions, campaign_id):
            return NO_NOTES_YET
        result = await self.search(campaign_id, question, deep=deep)
        log.info(
            "Question %r: %s document(s), best similarity %s%s",
            question,
            len(result.docs),
            None if result.best_similarity is None else round(result.best_similarity, 3),
            " (weak)" if result.weak else "",
        )
        if result.weak or not result.docs:
            return NOT_IN_NOTES
        context, labels = await asyncio.to_thread(self.build_context, campaign_id, result.docs)
        answer = await self.llm.answer_question(question, context, on_wait=on_wait)
        return replace_source_tags(answer, labels) or NOT_IN_NOTES

    def build_context(self, campaign_id: int, docs: list[SearchDoc]) -> tuple[str, dict[str, str]]:
        """Numbered sources for the answer prompt, and each tag's citation label (blocking)."""
        links: dict[int, list[Any]] | None = None
        blocks: list[str] = []
        labels: dict[str, str] = {}
        used = 0
        for doc in docs:
            tag = f"S{len(blocks) + 1}"
            if doc.kind == "page":
                entity = self.storage.get_entity(campaign_id, doc.ref_id)
                page = self.storage.get_page(doc.ref_id)
                if entity is None or page is None:
                    continue
                if links is None:
                    links = linked_quests(self.storage, campaign_id)
                facts = self.storage.get_facts(campaign_id, cited_fact_ids(page.markdown))
                heading = f"Wiki page {entity_header(entity)}"
                text = "\n".join([render_citations(page.markdown, facts), *quest_list(links.get(entity.id, []))])
                labels[tag] = f"wiki: {entity.canonical_name}"
            elif doc.kind == "summary":
                heading = f"Summary of {doc.title}"
                text = doc.body
                labels[tag] = f"{self._session_label(doc.session_id)} summary"
            else:
                label = self._session_label(doc.session_id)
                heading = f"Transcript of {label} from {doc.start_ts}"
                text = doc.body
                labels[tag] = f"{label} @ {doc.start_ts}"
            text = text.strip()
            if len(text) > DOC_CONTEXT_CHARS:
                text = text[:DOC_CONTEXT_CHARS].rstrip() + "\n[...]"
            block = f"[{tag}] {heading}\n{text}"
            if blocks and used + len(block) > CONTEXT_CHARS:
                labels.pop(tag, None)
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks), labels


    def _session_label(self, session_id: int | None) -> str:
        """`session 12, 2024-03-10` for a source document's session (blocking)."""
        row = self.storage.get_session(session_id) if session_id is not None else None
        if row is None:
            return f"session {session_id}"
        return session_label(row["number"] or session_id, session_date(row["started_at"], row["journal_id"] is not None))


# --- Pure helpers (unit-tested) --------------------------------------------------------------


def summary_doc(session: Any, summary_markdown: str) -> SearchDoc:
    """A session summary titled `session 12: Title (2024-03-10)`: its number in the campaign and
    the date it was played."""
    session_id = int(session["id"])
    date = session_date(session["started_at"], session["journal_id"] is not None) or str(session["started_at"])[:10]
    title = f"session {session['number'] or session_id}"
    if session["title"]:
        title += f": {session['title']}"
    return SearchDoc("summary", session_id, f"{title} ({date})", summary_markdown.strip(), session_id=session_id)


def transcript_docs(
    session_id: int,
    started_at: datetime,
    lines: list[Any],
    max_chars: int = TRANSCRIPT_CHUNK_CHARS,
    number: int | None = None,
) -> list[SearchDoc]:
    """Cut a session transcript into chunks of whole `[HH:MM:SS] Speaker: text` lines."""
    docs: list[SearchDoc] = []
    current: list[str] = []
    start_ts = ""
    size = 0

    def flush() -> None:
        if current:
            docs.append(
                SearchDoc(
                    "transcript",
                    session_id,
                    f"session {number or session_id} transcript {start_ts}",
                    "\n".join(current),
                    part=len(docs),
                    session_id=session_id,
                    start_ts=start_ts,
                )
            )

    for line in lines:
        ts = format_offset(line.started_at - started_at)
        rendered = f"[{ts}] {line.speaker}: {line.text}"
        if current and size + len(rendered) > max_chars:
            flush()
            current, size = [], 0
        if not current:
            start_ts = ts
        current.append(rendered)
        size += len(rendered) + 1
    flush()
    return docs


def fts_query(question: str, names: list[str] = ()) -> str:
    """An FTS5 query matching any distinctive word of the question, or any name of an entity
    it mentions (so a nickname also finds summaries and transcripts that use the full name)."""
    terms: list[str] = []
    for word in re.findall(r"\w+", question.casefold()):
        if len(word) >= 2 and word not in STOPWORDS and f'"{word}"' not in terms:
            terms.append(f'"{word}"')
    for name in names:
        phrase = " ".join(re.findall(r"\w+", name.casefold()))
        if phrase and f'"{phrase}"' not in terms:
            terms.append(f'"{phrase}"')
    return " OR ".join(terms)


def reciprocal_rank_fusion(rankings: list[tuple[list[int], float]], k: int = RRF_K) -> list[int]:
    """Merge ranked id lists: each list adds weight / (k + rank) to an id's score."""
    scores: dict[int, float] = {}
    for ranking, weight in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))


def pack_vector(vector: list[float]) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype="<f4").tobytes()


def rank_by_similarity(query: list[float], stored: list[tuple[int, bytes]]) -> list[tuple[int, float]]:
    """Cosine similarity of the query to each stored vector, best first."""
    import numpy as np

    if not stored:
        return []
    matrix = np.stack([np.frombuffer(blob, dtype="<f4") for _, blob in stored])
    q = np.asarray(query, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) or 1.0)
    scores = matrix @ q / np.where(norms == 0, 1.0, norms)
    order = np.argsort(-scores, kind="stable")
    return [(stored[i][0], float(scores[i])) for i in order]


def replace_source_tags(answer: str, labels: dict[str, str]) -> str:
    """Turn `[S2]` / `[S1, S3]` tags into readable citations such as `(session 12 summary)`."""

    def replace(match: re.Match[str]) -> str:
        found = [labels[tag.strip()] for tag in match.group(1).split(",") if tag.strip() in labels]
        return f"({'; '.join(dict.fromkeys(found))})" if found else ""

    return SOURCE_TAG_RE.sub(replace, answer).strip()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""
