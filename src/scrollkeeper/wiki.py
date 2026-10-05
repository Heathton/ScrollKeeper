"""Campaign wiki built from sourced facts (#7).

Entities (with aliases) collect append-only facts, each with provenance (session + transcript
time, a pin by a user, or an import reference). Pages are rebuilt from facts by the LLM: a page
records which facts it was built from, so any entity whose active facts differ from its page's
sources is "stale" and gets rewritten. New facts are folded into the existing page; when a fact
the page was built from is retracted or superseded, the page is rebuilt from scratch.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import Settings
from .llm import LocalAIService, split_transcript_chunks
from .models import DuplicateCandidate, Entity, Fact, Page, WikiChangeReport, normalize_name
from .storage import Storage


log = logging.getLogger(__name__)

WaitNotifier = Callable[[str], Awaitable[None]]
ProgressCallback = Callable[[str], None]

# Facts per page-rewrite call are capped by size so a long history is folded in batches.
PAGE_FACT_BATCH_CHARS = 12000
DUPLICATE_SIMILARITY = 0.85
TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
ENTITY_ID_RE = re.compile(r"^#?(\d+)$")
CITATION_RE = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")


class CampaignWiki:
    def __init__(self, storage: Storage, llm: LocalAIService, settings: Settings) -> None:
        self.storage = storage
        self.llm = llm
        self.settings = settings
        self.export_root = storage.data_dir / "wiki"
        self._export_lock = threading.Lock()

    # --- Session pipeline ---------------------------------------------------------------

    async def process_session(
        self,
        guild_id: int,
        session_id: int,
        timed_transcript: str,
        on_wait: WaitNotifier | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> WikiChangeReport:
        """Extract facts from a session transcript, then rewrite every page they touched.

        Re-running for the same session retracts that session's earlier observed facts first, so
        `!reprocess-llm` replaces rather than duplicates them.
        """
        report = WikiChangeReport()
        retracted, _ = await asyncio.to_thread(
            self.storage.retract_session_facts, guild_id, session_id, "session reprocessed"
        )
        report.facts_retracted = retracted

        new_ids: list[int] = []
        touched_ids: set[int] = set()
        chunks = split_transcript_chunks(timed_transcript, self.settings.extract_chunk_chars)
        for index, chunk in enumerate(chunks, start=1):
            _progress(on_progress, f"Extracting campaign facts ({index}/{len(chunks)}).")
            entities = await asyncio.to_thread(self.storage.list_entities, guild_id)
            payload = await self.llm.extract_facts(format_entity_index(entities, chunk), chunk, on_wait=on_wait)
            created, touched, added = await asyncio.to_thread(
                self._apply_extraction, guild_id, session_id, payload
            )
            new_ids.extend(created)
            touched_ids |= touched
            report.facts_added += added

        report.page_failures = await self.refresh_stale_pages(guild_id, on_wait=on_wait, on_progress=on_progress)

        entities = await asyncio.to_thread(self.storage.list_entities, guild_id)
        by_id = {entity.id: entity for entity in entities}
        report.new_entities = [by_id[i] for i in new_ids if i in by_id]
        report.updated_entities = [
            by_id[i] for i in sorted(touched_ids) if i in by_id and i not in set(new_ids)
        ]
        report.possible_duplicates = find_possible_duplicates(entities, set(new_ids) | touched_ids)
        return report

    def _apply_extraction(
        self,
        guild_id: int,
        session_id: int | None,
        payload: dict[str, Any],
    ) -> tuple[list[int], set[int], int]:
        """Store one extraction result. Returns (created entity ids, touched entity ids, facts added).

        Proposed entities are created only when a fact refers to them, and a proposal whose name
        already matches a known entity is attached to that entity instead.
        """
        proposals = {normalize_name(item["name"]): item for item in payload.get("new_entities", [])}
        created: dict[str, int] = {}
        created_ids: list[int] = []
        touched: set[int] = set()

        def resolve(ref: str) -> int | None:
            match = ENTITY_ID_RE.match(ref.strip())
            if match:
                entity = self.storage.get_entity(guild_id, int(match.group(1)))
                return entity.id if entity else None
            norm = normalize_name(ref)
            if norm in created:
                return created[norm]
            matches = self.storage.find_entities_by_name(guild_id, ref)
            if matches:
                if len(matches) > 1:
                    log.info("Fact target %r is ambiguous (%s); using #%s", ref, len(matches), matches[0].id)
                for alias in proposals.get(norm, {}).get("aliases", []):
                    self.storage.add_alias(matches[0].id, alias)
                return matches[0].id
            proposal = proposals.get(norm)
            if proposal is None:
                return None
            entity_id = self.storage.create_entity(
                guild_id,
                proposal["type"],
                proposal["name"],
                aliases=proposal["aliases"],
                short_description=proposal["short_description"],
                created_session_id=session_id,
            )
            created_ids.append(entity_id)
            for name in [proposal["name"], *proposal["aliases"]]:
                created.setdefault(normalize_name(name), entity_id)
            return entity_id

        facts_added = 0
        for fact in payload.get("facts", []):
            entity_id = resolve(fact["entity"])
            if entity_id is None:
                log.info("Dropping fact for unknown entity %r: %s", fact["entity"], fact["text"])
                continue
            timestamp = fact.get("timestamp", "")
            self.storage.add_fact(
                guild_id,
                entity_id,
                fact["text"],
                "observed",
                session_id=session_id,
                transcript_ts=timestamp if TIMESTAMP_RE.match(timestamp) else None,
            )
            touched.add(entity_id)
            facts_added += 1

        for update in payload.get("alias_updates", []):
            match = ENTITY_ID_RE.match(update["entity"])
            entity = self.storage.get_entity(guild_id, int(match.group(1))) if match else None
            if entity is not None:
                self.storage.add_alias(entity.id, update["alias"])
        return created_ids, touched, facts_added

    # --- Pages --------------------------------------------------------------------------

    def stale_entity_ids(self, guild_id: int) -> list[int]:
        active = self.storage.active_fact_ids_by_entity(guild_id)
        pages = self.storage.page_source_ids_by_entity(guild_id)
        return sorted(
            entity_id
            for entity_id in set(active) | set(pages)
            if active.get(entity_id, set()) != pages.get(entity_id, set())
        )

    async def refresh_stale_pages(
        self,
        guild_id: int,
        on_wait: WaitNotifier | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> list[str]:
        """Rewrite every page whose facts changed. Returns names of entities whose rewrite failed."""
        stale = await asyncio.to_thread(self.stale_entity_ids, guild_id)
        failures: list[str] = []
        for index, entity_id in enumerate(stale, start=1):
            _progress(on_progress, f"Rewriting wiki pages ({index}/{len(stale)}).")
            try:
                await self.refresh_page(guild_id, entity_id, on_wait=on_wait)
            except Exception:
                log.exception("Could not rewrite the wiki page for entity %s", entity_id)
                entity = await asyncio.to_thread(self.storage.get_entity, guild_id, entity_id)
                failures.append(entity.canonical_name if entity else f"#{entity_id}")
        if self.settings.wiki_export:
            await asyncio.to_thread(self.export, guild_id)
        return failures

    async def refresh_page(self, guild_id: int, entity_id: int, on_wait: WaitNotifier | None = None) -> bool:
        """Bring one page up to date with its entity's active facts. Returns True if it changed."""
        entity = await asyncio.to_thread(self.storage.get_entity, guild_id, entity_id)
        if entity is None:
            return False
        facts = await asyncio.to_thread(self.storage.get_entity_facts, entity.id)
        page = await asyncio.to_thread(self.storage.get_page, entity.id)
        if not facts:
            if page is not None:
                await asyncio.to_thread(self.storage.delete_page, entity.id)
                return True
            return False

        active_ids = {fact.id for fact in facts}
        if page is not None and not set(page.source_fact_ids) <= active_ids:
            page = None  # A source fact was retracted, superseded or moved: rebuild from scratch.
        known = set(page.source_fact_ids) if page else set()
        new_facts = [fact for fact in facts if fact.id not in known]
        if page is not None and not new_facts:
            return False

        pinned = [fact for fact in facts if fact.kind == "pinned"]
        pinned_text = "\n".join(format_fact(fact) for fact in pinned)
        markdown = page.markdown if page else ""
        short_description = entity.short_description
        for batch in _batch_facts([fact for fact in new_facts if fact.kind != "pinned"] or [None]):
            result = await self.llm.rewrite_page(
                entity_header(entity),
                markdown,
                pinned_text,
                "\n".join(format_fact(fact) for fact in batch if fact is not None),
                on_wait=on_wait,
            )
            if not result["markdown"]:
                raise RuntimeError(f"The LLM returned an empty page for {entity.canonical_name}.")
            markdown = result["markdown"]
            short_description = _strip_citations(result["short_description"]) or short_description

        embedding = await self.llm.embed_text(page_embedding_text(entity, markdown), on_wait=on_wait)
        await asyncio.to_thread(self.storage.save_page, entity.id, markdown, sorted(active_ids), embedding)
        if short_description != entity.short_description:
            await asyncio.to_thread(self.storage.set_entity_short_description, entity.id, short_description)
        return True

    async def reembed_page(self, guild_id: int, entity_id: int, on_wait: WaitNotifier | None = None) -> None:
        """Refresh a page's embedding after a name or alias change (the text is unchanged)."""
        entity = await asyncio.to_thread(self.storage.get_entity, guild_id, entity_id)
        page = await asyncio.to_thread(self.storage.get_page, entity_id)
        if entity is None or page is None:
            return
        embedding = await self.llm.embed_text(page_embedding_text(entity, page.markdown), on_wait=on_wait)
        await asyncio.to_thread(self.storage.update_page_embedding, entity_id, embedding)

    # --- Review commands ----------------------------------------------------------------

    async def resolve_entity(self, guild_id: int, ref: str) -> list[Entity]:
        """Look up an entity by `#id`, id, canonical name or alias."""
        ref = ref.strip()
        match = ENTITY_ID_RE.match(ref)
        if match:
            entity = await asyncio.to_thread(self.storage.get_entity, guild_id, int(match.group(1)))
            if entity is not None:
                return [entity]
        return await asyncio.to_thread(self.storage.find_entities_by_name, guild_id, ref)

    async def merge(self, guild_id: int, source: Entity, target: Entity, on_wait: WaitNotifier | None = None) -> None:
        await asyncio.to_thread(self.storage.merge_entities, source.id, target.id)
        await self.refresh_after_change(guild_id, target.id, on_wait=on_wait)

    async def rename(self, guild_id: int, entity: Entity, new_name: str, on_wait: WaitNotifier | None = None) -> None:
        await asyncio.to_thread(self.storage.rename_entity, entity.id, new_name)
        await self.reembed_page(guild_id, entity.id, on_wait=on_wait)
        await self._export_if_enabled(guild_id)

    async def has_name(self, guild_id: int, entity: Entity, name: str) -> bool:
        return normalize_name(name) in {normalize_name(existing) for existing in entity.names()}

    async def add_alias(self, guild_id: int, entity: Entity, alias: str, on_wait: WaitNotifier | None = None) -> None:
        if await asyncio.to_thread(self.storage.add_alias, entity.id, alias):
            await self.reembed_page(guild_id, entity.id, on_wait=on_wait)
            await self._export_if_enabled(guild_id)

    async def refresh_after_change(self, guild_id: int, entity_id: int, on_wait: WaitNotifier | None = None) -> None:
        """Rewrite one entity's page after a manual fact change (pin, correction, retraction, merge)."""
        await self.refresh_page(guild_id, entity_id, on_wait=on_wait)
        await self._export_if_enabled(guild_id)

    async def render_entity(self, guild_id: int, entity: Entity) -> str:
        """Discord view of one entity: page (or raw facts) plus the sources it cites."""
        page = await asyncio.to_thread(self.storage.get_page, entity.id)
        facts = await asyncio.to_thread(self.storage.get_entity_facts, entity.id)
        lines = [f"## {entity.canonical_name} (#{entity.id}, {entity.type})"]
        if entity.aliases:
            lines.append(f"Also known as: {', '.join(entity.aliases)}")
        lines.append("")
        if page is not None:
            lines.append(page.markdown)
            cited = cited_fact_ids(page.markdown) or [fact.id for fact in facts]
        elif facts:
            lines.append("_No page has been written yet. Facts:_")
            cited = [fact.id for fact in facts]
        else:
            lines.append("_No facts recorded._")
            cited = []
        by_id = {fact.id: fact for fact in facts}
        sources = [by_id[fact_id] for fact_id in cited if fact_id in by_id]
        if sources:
            lines.extend(["", "**Sources**"])
            lines.extend(f"- F{fact.id} ({fact.source_label()}): {fact.text}" for fact in sources)
        return "\n".join(lines)

    async def answer_question(self, guild_id: int, question: str, on_wait: WaitNotifier | None = None) -> str:
        query_embedding = await self.llm.embed_text(question, on_wait=on_wait)
        results = await asyncio.to_thread(self.storage.semantic_search_pages, guild_id, query_embedding, 8)
        if not results:
            return "I do not have any campaign wiki pages yet."
        fact_ids = sorted({fact_id for _, page in results for fact_id in cited_fact_ids(page.markdown)})
        facts = await asyncio.to_thread(self.storage.get_facts, guild_id, fact_ids)
        context = "\n\n".join(
            f"{entity_header(entity)}\n{render_citations(page.markdown, facts)}" for entity, page in results
        )
        return await self.llm.answer_question(question, context, on_wait=on_wait)

    # --- Export -------------------------------------------------------------------------

    def export(self, guild_id: int) -> Path:
        """Write an Obsidian-style vault: one Markdown file per entity, `[[links]]`, cited sources.

        The directory is regenerated from the database each time; do not edit files in it.
        """
        with self._export_lock:
            return self._export_locked(guild_id)

    def _export_locked(self, guild_id: int) -> Path:
        target = self.export_root / str(guild_id)
        pages = self.storage.list_pages(guild_id)
        fact_ids = sorted({fact_id for _, page in pages for fact_id in cited_fact_ids(page.markdown)})
        facts = self.storage.get_facts(guild_id, fact_ids)
        entities = [entity for entity, _ in pages]
        file_names = _unique_file_names(entities)
        if target.exists():
            shutil.rmtree(target)
        for entity, page in pages:
            folder = target / entity.type
            folder.mkdir(parents=True, exist_ok=True)
            body = link_entity_names(render_citations(page.markdown, facts), entity, entities, file_names)
            frontmatter = ["---", f"type: {entity.type}", f"id: {entity.id}"]
            if entity.aliases:
                frontmatter.append("aliases:")
                frontmatter.extend(f'  - "{_yaml_escape(alias)}"' for alias in entity.aliases)
            frontmatter.append("---")
            content = "\n".join([*frontmatter, "", f"# {entity.canonical_name}", "", body.strip(), ""])
            (folder / f"{file_names[entity.id]}.md").write_text(content, encoding="utf-8")
        return target

    async def _export_if_enabled(self, guild_id: int) -> None:
        if self.settings.wiki_export:
            await asyncio.to_thread(self.export, guild_id)


# --- Formatting helpers (pure, unit-tested) ------------------------------------------------


def entity_header(entity: Entity) -> str:
    header = f"[{entity.type}] {entity.canonical_name}"
    if entity.aliases:
        header += f" (aka {', '.join(entity.aliases)})"
    return header


def page_embedding_text(entity: Entity, markdown: str) -> str:
    return f"{entity_header(entity)}\n{CITATION_RE.sub('', markdown)}"


def format_fact(fact: Fact) -> str:
    return f"[F{fact.id}] ({fact.source_label()}) {fact.text}"


def format_entity_index(entities: list[Entity], text: str) -> str:
    """Entity index for the extraction prompt.

    Every entity is listed by id, type and names so the model can match misspellings; only those
    whose name appears in `text` also carry their one-line description, to keep the prompt short.
    """
    haystack = f" {' '.join(_NAME_TOKENS.sub(' ', text.casefold()).split())} "
    lines = []
    for entity in entities:
        line = f"#{entity.id} {entity_header(entity)}"
        mentioned = any(f" {norm} " in haystack for norm in map(normalize_name, entity.names()) if norm)
        if mentioned and entity.short_description:
            line += f": {entity.short_description}"
        lines.append(line)
    return "\n".join(lines)


_NAME_TOKENS = re.compile(r"[^\w\s]")


def find_possible_duplicates(entities: list[Entity], focus_ids: set[int]) -> list[DuplicateCandidate]:
    """Pairs where at least one side is in `focus_ids` and the names look like the same thing."""
    candidates: list[DuplicateCandidate] = []
    seen: set[tuple[int, int]] = set()
    for first in entities:
        if first.id not in focus_ids:
            continue
        for second in entities:
            pair = (min(first.id, second.id), max(first.id, second.id))
            if first.id == second.id or pair in seen:
                continue
            reason = duplicate_reason(first, second)
            if reason:
                seen.add(pair)
                ordered = (first, second) if first.id < second.id else (second, first)
                candidates.append(DuplicateCandidate(ordered[0], ordered[1], reason))
    return candidates


def duplicate_reason(first: Entity, second: Entity) -> str | None:
    for a in {normalize_name(name) for name in first.names()} - {""}:
        for b in {normalize_name(name) for name in second.names()} - {""}:
            if a == b:
                return f"both are called '{a}'"
            tokens_a, tokens_b = set(a.split()), set(b.split())
            small, large = (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
            if small < large and any(len(token) >= 4 for token in small):
                return f"'{' '.join(sorted(small))}' is part of the other name"
            if min(len(a), len(b)) >= 4 and SequenceMatcher(None, a, b).ratio() >= DUPLICATE_SIMILARITY:
                return f"'{a}' and '{b}' are spelled alike"
    return None


def format_change_report(report: WikiChangeReport) -> str:
    lines = ["### Campaign wiki"]
    if report.error:
        lines.append(f"The wiki update failed: {report.error[:500]}. Run `!reprocess-llm` to retry.")
        return "\n".join(lines)
    summary = f"{report.facts_added} fact(s) recorded"
    if report.facts_retracted:
        summary += f" ({report.facts_retracted} from an earlier run of this session replaced)"
    lines.append(summary + ".")
    if report.new_entities:
        lines.append("**New:** " + ", ".join(f"#{e.id} {e.canonical_name} ({e.type})" for e in report.new_entities))
    if report.updated_entities:
        lines.append("**Updated:** " + ", ".join(f"#{e.id} {e.canonical_name}" for e in report.updated_entities))
    if report.possible_duplicates:
        lines.append("**Possible duplicates** (`!merge-entity <from> <into>` to merge):")
        lines.extend(
            f"- #{c.first.id} {c.first.canonical_name} ({c.first.type}) / "
            f"#{c.second.id} {c.second.canonical_name} ({c.second.type}): {c.reason}"
            for c in report.possible_duplicates
        )
    if report.page_failures:
        lines.append(
            "**Pages not updated** (retried on the next run): " + ", ".join(report.page_failures)
        )
    return "\n".join(lines)


def cited_fact_ids(markdown: str) -> list[int]:
    ids: list[int] = []
    for group in CITATION_RE.findall(markdown):
        for token in group.split(","):
            fact_id = int(token.strip()[1:])
            if fact_id not in ids:
                ids.append(fact_id)
    return ids


def render_citations(markdown: str, facts: dict[int, Fact]) -> str:
    """Replace `[F12, F15]` markers with readable sources such as `(session 3 @ 00:41:10; pinned)`."""

    def replace(match: re.Match[str]) -> str:
        labels: list[str] = []
        for token in match.group(1).split(","):
            fact = facts.get(int(token.strip()[1:]))
            label = fact.source_label() if fact else None
            if label and label not in labels:
                labels.append(label)
        return f"({'; '.join(labels)})" if labels else ""

    return CITATION_RE.sub(replace, markdown)


def link_entity_names(markdown: str, entity: Entity, entities: list[Entity], file_names: dict[int, str]) -> str:
    """Link the first mention of every other entity as `[[File|text]]` (Obsidian wiki links)."""
    by_name: dict[str, Entity] = {}
    for other in entities:
        if other.id == entity.id:
            continue
        for name in other.names():
            if len(name) >= 3:
                by_name.setdefault(name.casefold(), other)
    if not by_name:
        return markdown
    pattern = re.compile(
        r"(?<![\w\[])(" + "|".join(re.escape(name) for name in sorted(by_name, key=len, reverse=True)) + r")(?![\w\]])",
        re.IGNORECASE,
    )
    linked: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        other = by_name[match.group(1).casefold()]
        if other.id in linked:
            return match.group(0)
        linked.add(other.id)
        return f"[[{file_names[other.id]}|{match.group(1)}]]"

    return pattern.sub(replace, markdown)


def _unique_file_names(entities: list[Entity]) -> dict[int, str]:
    names: dict[int, str] = {}
    used: set[str] = set()
    for entity in entities:
        base = re.sub(r'[\\/:*?"<>|#^\[\]]', "", entity.canonical_name).strip() or f"Entity {entity.id}"
        name = base if base.casefold() not in used else f"{base} ({entity.id})"
        used.add(name.casefold())
        names[entity.id] = name
    return names


def _strip_citations(text: str) -> str:
    return re.sub(r"\s*" + CITATION_RE.pattern, "", text).strip()


def _yaml_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _batch_facts(facts: list[Fact | None]) -> list[list[Fact | None]]:
    batches: list[list[Fact | None]] = []
    current: list[Fact | None] = []
    size = 0
    for fact in facts:
        length = len(format_fact(fact)) if fact is not None else 0
        if current and size + length > PAGE_FACT_BATCH_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(fact)
        size += length
    if current:
        batches.append(current)
    return batches


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)
