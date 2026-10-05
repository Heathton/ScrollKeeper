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
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .config import Settings
from .llm import LocalAIService, split_transcript_chunks
from .models import DuplicateCandidate, Entity, Fact, Page, WikiChangeReport, normalize_name
from .storage import Storage

if TYPE_CHECKING:
    from .search import SearchIndex


log = logging.getLogger(__name__)

WaitNotifier = Callable[[str], Awaitable[None]]
ProgressCallback = Callable[[str], None]

# Facts per page-rewrite call are capped by size so a long history is folded in batches.
PAGE_FACT_BATCH_CHARS = 12000
DUPLICATE_SIMILARITY = 0.85
# Looser thresholds for picking candidates the LLM then checks (a wrong candidate costs one call).
CANDIDATE_SIMILARITY = 0.75
MENTION_SIMILARITY = 0.85
WORD_SIMILARITY = 0.7
MAX_MATCH_CANDIDATES = 5
# Words too common in names to suggest two names mean the same entity on their own.
TITLE_WORDS = frozenset(
    "lord lady sir dame captain king queen prince princess duke duchess baron master mistress "
    "father mother brother sister young old elder great house city town the of".split()
)
TIMESTAMP_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
ENTITY_ID_RE = re.compile(r"^#?(\d+)$")
CITATION_RE = re.compile(r"\[(F\d+(?:\s*,\s*F\d+)*)\]")
# Entity types whose names are descriptive titles ("Recover Varric's Ledger"), not words spoken at
# the table, so they are left out of the spelling glossary.
UNSPOKEN_NAME_TYPES = frozenset({"Quest", "Mystery"})


@dataclass(slots=True)
class ExtractionResult:
    created: list[int] = field(default_factory=list)
    touched: set[int] = field(default_factory=set)
    facts_added: int = 0
    flagged: list[DuplicateCandidate] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)


class CampaignWiki:
    def __init__(
        self,
        storage: Storage,
        llm: LocalAIService,
        settings: Settings,
        search: "SearchIndex | None" = None,
    ) -> None:
        self.storage = storage
        self.llm = llm
        self.settings = settings
        self.search = search
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
        await self.ensure_player_characters(guild_id)
        retracted, _ = await asyncio.to_thread(
            self.storage.retract_session_facts, guild_id, session_id, "session reprocessed"
        )
        report.facts_retracted = retracted

        result = ExtractionResult()
        chunks = split_transcript_chunks(timed_transcript, self.settings.extract_chunk_chars)
        for index, chunk in enumerate(chunks, start=1):
            _progress(on_progress, f"Extracting campaign facts ({index}/{len(chunks)}).")
            entities = await asyncio.to_thread(self.storage.list_entities, guild_id)
            index_text = await asyncio.to_thread(format_entity_index, entities, chunk)
            payload = await self.llm.extract_facts(index_text, chunk, on_wait=on_wait)
            await self._apply_extraction(guild_id, session_id, payload, chunk, result, on_wait=on_wait)
        report.facts_added = result.facts_added
        report.renamed = result.renamed

        report.page_failures = await self.refresh_stale_pages(guild_id, on_wait=on_wait, on_progress=on_progress)
        if self.search is not None:
            report.pages_without_embedding = await self.search.pending_count(guild_id)

        entities = await asyncio.to_thread(self.storage.list_entities, guild_id)
        by_id = {entity.id: entity for entity in entities}
        created = set(result.created)
        report.new_entities = [by_id[i] for i in result.created if i in by_id]
        report.updated_entities = [by_id[i] for i in sorted(result.touched) if i in by_id and i not in created]
        flagged = [
            DuplicateCandidate(by_id[c.first.id], by_id[c.second.id], c.reason)
            for c in result.flagged
            if c.first.id in by_id and c.second.id in by_id
        ]
        report.possible_duplicates = _combine_candidates(
            flagged, find_possible_duplicates(entities, created | result.touched)
        )
        return report

    async def ensure_player_characters(self, guild_id: int) -> list[int]:
        """Create a Character entity for every `!register-character` name that has none yet."""

        def ensure() -> list[int]:
            created = []
            for name in self.storage.list_registered_characters(guild_id):
                if not self.storage.find_entities_by_name(guild_id, name):
                    created.append(
                        self.storage.create_entity(guild_id, "Character", name, short_description="Player character")
                    )
            return created

        return await asyncio.to_thread(ensure)

    def spelling_glossary(self, guild_id: int) -> list[str]:
        """Names (blocking) the summarizer should spell correctly. Names only, no descriptions or
        facts, so a summary still comes from its own transcript alone."""
        return spelling_glossary(
            self.storage.list_entities(guild_id), self.storage.list_registered_characters(guild_id)
        )

    async def _apply_extraction(
        self,
        guild_id: int,
        session_id: int | None,
        payload: dict[str, Any],
        chunk: str,
        result: "ExtractionResult",
        on_wait: WaitNotifier | None = None,
    ) -> None:
        """Store one extraction result, resolving every fact's entity reference.

        Name reveals are applied first, so a proposal under the revealed name finds the renamed
        entity. Proposed entities are created only when a fact refers to them, and only after
        checking existing entities (see `_resolve_reference`).
        """
        for reveal in payload.get("name_reveals", []):
            await self._apply_name_reveal(guild_id, reveal, result)

        proposals: dict[str, dict[str, Any]] = {}
        for item in payload.get("new_entities", []):
            for name in [item["name"], *item["aliases"]]:
                proposals.setdefault(normalize_name(name), item)
        facts = payload.get("facts", [])
        resolved: dict[str, int | None] = {}
        for fact in facts:
            ref = fact["entity"].strip()
            key = ref if ENTITY_ID_RE.match(ref) else normalize_name(ref)
            if key not in resolved:
                proposal = proposals.get(normalize_name(ref))
                names = {normalize_name(n) for n in ([proposal["name"], *proposal["aliases"]] if proposal else [ref])}
                evidence = [f for f in facts if normalize_name(f["entity"]) in names]
                resolved[key] = await self._resolve_reference(
                    guild_id, session_id, ref, proposal, evidence, chunk, result, on_wait
                )
            entity_id = resolved[key]
            if entity_id is None:
                log.info("Dropping fact for unknown entity %r: %s", ref, fact["text"])
                continue
            timestamp = fact.get("timestamp", "")
            await asyncio.to_thread(
                self.storage.add_fact,
                guild_id,
                entity_id,
                fact["text"],
                "observed",
                session_id,
                timestamp if TIMESTAMP_RE.match(timestamp) else None,
            )
            result.touched.add(entity_id)
            result.facts_added += 1

        for update in payload.get("alias_updates", []):
            match = ENTITY_ID_RE.match(update["entity"])
            entity = await asyncio.to_thread(self.storage.get_entity, guild_id, int(match.group(1))) if match else None
            if entity is not None:
                await asyncio.to_thread(self.storage.add_alias, entity.id, update["alias"])

    async def _resolve_reference(
        self,
        guild_id: int,
        session_id: int | None,
        ref: str,
        proposal: dict[str, Any] | None,
        evidence: list[dict[str, str]],
        chunk: str,
        result: "ExtractionResult",
        on_wait: WaitNotifier | None,
    ) -> int | None:
        """Map a fact's entity reference to an entity id, creating the entity only when it is new.

        1. `#id` references are trusted.
        2. One existing entity with any of the proposed names: use it (and learn the new names).
        3. Several exact matches, or fuzzy candidates (similar spelling, a shared name word, the
           name appearing in a description): ask the LLM whether it is one of them.
        4. Otherwise, or when the LLM says different/unsure, create the proposed entity; an
           unsure or ambiguous outcome is flagged for review in the change report.
        """
        match = ENTITY_ID_RE.match(ref)
        if match:
            entity = await asyncio.to_thread(self.storage.get_entity, guild_id, int(match.group(1)))
            return entity.id if entity else None

        names = [proposal["name"], *proposal["aliases"]] if proposal else [ref]
        exact = await asyncio.to_thread(self._find_by_any_name, guild_id, names)
        if len(exact) == 1:
            await asyncio.to_thread(self._add_aliases, exact[0].id, names)
            return exact[0].id

        if exact:
            candidates = exact
        else:
            entities = await asyncio.to_thread(self.storage.list_entities, guild_id)
            candidates = match_candidates(entities, names)
        decision: dict[str, str] | None = None
        chosen: Entity | None = None
        if candidates:
            decision = await self._adjudicate(names, proposal, evidence, chunk, candidates, on_wait)
            chosen = next((c for c in candidates if f"#{c.id}" == decision["entity"].strip()), None)
            if decision["decision"] == "same" and chosen is not None:
                await asyncio.to_thread(self._add_aliases, chosen.id, names)
                return chosen.id

        if proposal is None:
            if not exact:
                return None
            # A bare name shared by several entities, and the LLM couldn't tell which one is meant.
            target = chosen or exact[0]
            other = next(c for c in exact if c.id != target.id)
            result.flagged.append(
                _candidate(target, other, f"'{ref}' names both; facts were attached to #{target.id}, check them")
            )
            return target.id

        entity_id = await asyncio.to_thread(
            self.storage.create_entity,
            guild_id,
            proposal["type"],
            proposal["name"],
            proposal["aliases"],
            proposal["short_description"],
            session_id,
        )
        result.created.append(entity_id)
        new_entity = await asyncio.to_thread(self.storage.get_entity, guild_id, entity_id)
        if decision is not None and (decision["decision"] == "unsure" or exact):
            other = chosen or candidates[0]
            label = "possibly the same" if decision["decision"] == "unsure" else "shares a name"
            result.flagged.append(_candidate(new_entity, other, f"{label}: {decision['reason'] or 'no reason given'}"))
        return entity_id

    async def _adjudicate(
        self,
        names: list[str],
        proposal: dict[str, Any] | None,
        evidence: list[dict[str, str]],
        chunk: str,
        candidates: list[Entity],
        on_wait: WaitNotifier | None,
    ) -> dict[str, str]:
        lines = [f"Names: {', '.join(names)}"]
        if proposal is not None:
            lines.append(f"Type: {proposal['type']}")
            if proposal["short_description"]:
                lines.append(f"Description: {proposal['short_description']}")
        lines.append("Facts from this session:")
        lines.extend(f"- [{fact.get('timestamp', '')}] {fact['text']}" for fact in evidence)
        excerpt = transcript_excerpt(chunk, [fact.get("timestamp", "") for fact in evidence])
        if excerpt:
            lines.extend(["Transcript lines:", excerpt])

        blocks = []
        for candidate in candidates:
            facts = await asyncio.to_thread(self.storage.get_entity_facts, candidate.id)
            block = f"#{candidate.id} {entity_header(candidate)}"
            if candidate.short_description:
                block += f": {candidate.short_description}"
            block += "".join(f"\n  - {fact.text}" for fact in facts[-8:])
            blocks.append(block)
        try:
            decision = await self.llm.match_entity("\n".join(lines), "\n\n".join(blocks), on_wait=on_wait)
        except Exception as exc:
            log.warning("Entity match check failed for %s: %s", names[0], exc)
            return {"decision": "unsure", "entity": "", "reason": "the match check failed"}
        log.info("Entity match for %r: %s %s (%s)", names[0], decision["decision"], decision["entity"], decision["reason"])
        return decision

    async def _apply_name_reveal(self, guild_id: int, reveal: dict[str, str], result: "ExtractionResult") -> None:
        match = ENTITY_ID_RE.match(reveal["entity"])
        entity = await asyncio.to_thread(self.storage.get_entity, guild_id, int(match.group(1))) if match else None
        new_name = reveal["name"].strip()
        if entity is None or normalize_name(new_name) == normalize_name(entity.canonical_name):
            return
        others = [
            other
            for other in await asyncio.to_thread(self.storage.find_entities_by_name, guild_id, new_name)
            if other.id != entity.id
        ]
        if others:
            result.flagged.append(
                _candidate(entity, others[0], f"#{entity.id} was revealed to be '{new_name}', which is #{others[0].id}")
            )
            return
        await asyncio.to_thread(self.storage.rename_entity, entity.id, new_name)
        result.renamed.append((entity.canonical_name, new_name))

    def _find_by_any_name(self, guild_id: int, names: list[str]) -> list[Entity]:
        found: dict[int, Entity] = {}
        for name in names:
            for entity in self.storage.find_entities_by_name(guild_id, name):
                found.setdefault(entity.id, entity)
        return list(found.values())

    def _add_aliases(self, entity_id: int, names: list[str]) -> None:
        for name in names:
            self.storage.add_alias(entity_id, name)

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
        await self._after_change(guild_id)
        return failures

    async def refresh_page(
        self,
        guild_id: int,
        entity_id: int,
        on_wait: WaitNotifier | None = None,
        force_full: bool = False,
    ) -> bool:
        """Bring one page up to date with its entity's active facts. Returns True if it changed.

        `force_full` rebuilds the page from all facts even if it is current (e.g. after a layout
        change).
        """
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
        if page is not None and (force_full or not set(page.source_fact_ids) <= active_ids):
            page = None  # A source fact was retracted, superseded or moved: rebuild from scratch.
        known = set(page.source_fact_ids) if page else set()
        new_facts = [fact for fact in facts if fact.id not in known]
        if page is not None and not new_facts:
            return False

        pinned = [fact for fact in facts if fact.kind == "pinned"]
        pinned_text = "\n".join(format_fact(fact) for fact in pinned)
        layout = page_layout(entity, await asyncio.to_thread(self._is_player_character, guild_id, entity))
        markdown = page.markdown if page else ""
        short_description = entity.short_description
        status = entity.status
        for batch in _batch_facts([fact for fact in new_facts if fact.kind != "pinned"] or [None]):
            result = await self.llm.rewrite_page(
                entity_header(entity),
                markdown,
                pinned_text,
                "\n".join(format_fact(fact) for fact in batch if fact is not None),
                render_layout(layout),
                on_wait=on_wait,
            )
            if not result["markdown"]:
                raise RuntimeError(f"The LLM returned an empty page for {entity.canonical_name}.")
            markdown = normalize_headings(result["markdown"], layout)
            status = result.get("status", "") if entity.type == "Quest" else ""
            short_description = _strip_citations(result["short_description"]) or short_description

        await asyncio.to_thread(self.storage.save_page, entity.id, markdown, sorted(active_ids))
        if short_description != entity.short_description:
            await asyncio.to_thread(self.storage.set_entity_short_description, entity.id, short_description)
        if status and status != entity.status:
            await asyncio.to_thread(self.storage.set_entity_status, entity.id, status)
        return True

    def linked_quests(self, guild_id: int) -> dict[int, list[Entity]]:
        return linked_quests(self.storage, guild_id)

    async def rebuild_all_pages(
        self,
        guild_id: int,
        on_wait: WaitNotifier | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[int, list[str]]:
        """Rebuild every page from its facts (after a layout change). Returns (rebuilt, failures)."""
        entity_ids = sorted(await asyncio.to_thread(self.storage.active_fact_ids_by_entity, guild_id))
        rebuilt, failures = 0, []
        for index, entity_id in enumerate(entity_ids, start=1):
            _progress(on_progress, f"Rebuilding wiki pages ({index}/{len(entity_ids)}).")
            try:
                await self.refresh_page(guild_id, entity_id, on_wait=on_wait, force_full=True)
                rebuilt += 1
            except Exception:
                log.exception("Could not rebuild the wiki page for entity %s", entity_id)
                entity = await asyncio.to_thread(self.storage.get_entity, guild_id, entity_id)
                failures.append(entity.canonical_name if entity else f"#{entity_id}")
        await self._after_change(guild_id)
        return rebuilt, failures

    def _is_player_character(self, guild_id: int, entity: Entity) -> bool:
        if entity.type != "Character":
            return False
        names = {normalize_name(name) for name in entity.names()}
        return any(normalize_name(name) in names for name in self.storage.list_registered_characters(guild_id))

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
        await self._after_change(guild_id)

    async def has_name(self, guild_id: int, entity: Entity, name: str) -> bool:
        return normalize_name(name) in {normalize_name(existing) for existing in entity.names()}

    async def add_alias(self, guild_id: int, entity: Entity, alias: str, on_wait: WaitNotifier | None = None) -> None:
        if await asyncio.to_thread(self.storage.add_alias, entity.id, alias):
            await self._after_change(guild_id)

    async def refresh_after_change(self, guild_id: int, entity_id: int, on_wait: WaitNotifier | None = None) -> None:
        """Rewrite one entity's page after a manual fact change (pin, correction, retraction, merge)."""
        await self.refresh_page(guild_id, entity_id, on_wait=on_wait)
        await self._after_change(guild_id)

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
        quests = (await asyncio.to_thread(self.linked_quests, guild_id)).get(entity.id, [])
        if quests:
            lines.extend(["", *quest_list(quests)])
        by_id = {fact.id: fact for fact in facts}
        sources = [by_id[fact_id] for fact_id in cited if fact_id in by_id]
        if sources:
            lines.extend(["", "**Sources**"])
            lines.extend(f"- F{fact.id} ({fact.source_label()}): {fact.text}" for fact in sources)
        return "\n".join(lines)

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
        links = self.linked_quests(guild_id)
        if target.exists():
            shutil.rmtree(target)
        for entity, page in pages:
            folder = target / entity.type
            folder.mkdir(parents=True, exist_ok=True)
            body = link_entity_names(render_citations(page.markdown, facts), entity, entities, file_names)
            quests = quest_list(links.get(entity.id, []), file_names)
            if quests:
                body = body.rstrip() + "\n\n" + "\n".join(quests)
            frontmatter = ["---", f"type: {entity.type}", f"id: {entity.id}"]
            if entity.aliases:
                frontmatter.append("aliases:")
                frontmatter.extend(f'  - "{_yaml_escape(alias)}"' for alias in entity.aliases)
            frontmatter.append("---")
            content = "\n".join([*frontmatter, "", f"# {entity.canonical_name}", "", body.strip(), ""])
            (folder / f"{file_names[entity.id]}.md").write_text(content, encoding="utf-8")
        return target

    async def _after_change(self, guild_id: int) -> None:
        """Export the vault and re-index changed pages for search. Search indexing failures are
        logged, not raised: the wiki change itself is saved."""
        if self.settings.wiki_export:
            await asyncio.to_thread(self.export, guild_id)
        if self.search is not None:
            try:
                await self.search.refresh(guild_id)
            except Exception:
                log.exception("Could not update the search index for guild %s", guild_id)


# --- Page layouts ----------------------------------------------------------------------------

# One layout per entity type: what the opening paragraph covers, then `##` sections (heading,
# guidance) in order, each written only when facts support it. Quest details live on the Quest
# page; other pages get a "Quests" list built from current data when shown (`linked_quests`). Changing a layout affects pages as
# they are next rewritten; `!rebuild-pages` applies it to every page at once.
Layout = tuple[str, list[tuple[str, str]]]

# Only Quest pages get Open Questions: a page sees only its own facts, so on other types the model
# asked questions other pages already answer. Mystery pages are built around their question.
_OPEN_QUESTIONS = (
    "Open Questions",
    "what is still unknown or left to do for this quest; not the party's own choices, and not "
    "questions the facts already answer",
)
_STATUS_RULE = "only what the facts state; leave out anything unknown"

PAGE_LAYOUTS: dict[str, Layout] = {
    "Character": (
        "who they are, their role, and where they are usually found",
        [
            ("Appearance & Personality", "how they look and behave"),
            ("Relationships", "with other characters, factions and the party"),
            ("History", "what happened involving them, in session order"),
            ("Status", f"alive, dead or missing; current location; {_STATUS_RULE}"),
        ],
    ),
    "Faction": (
        "what the faction is, its purpose, and where it is based",
        [
            ("Members & Leadership", "known members and who leads"),
            ("Allies & Enemies", "other factions and characters"),
            ("Activities", "what they do and have done"),
            ("Relationship with the Party", "how they treat the party and why"),
        ],
    ),
    "Location": (
        "what and where it is, and what larger region it belongs to",
        [
            ("Notable Places", "buildings and sites within it"),
            ("Notable People", "who lives or works there"),
            ("History", "what happened there, in session order"),
            ("Current State", _STATUS_RULE),
        ],
    ),
    "Item": (
        "what it is and who holds it now",
        [
            ("Properties & Effects", "what it does, including curses"),
            ("Provenance", "where it came from and who held it before"),
            ("History", "what happened involving it, in session order"),
        ],
    ),
    "Mystery": (
        "the unanswered question, stated plainly",
        [
            ("Clues", "each clue as a cited bullet, in the order the party found them"),
            ("Theories", "each labelled as a theory, with who proposed it; never state one as fact"),
            ("Status", "open or resolved, and how it was resolved"),
        ],
    ),
    "PointOfInterest": (
        "what it is and where it is",
        [
            ("Features", "what is there"),
            ("Dangers", "threats and hazards"),
            ("History", "what happened there, in session order"),
        ],
    ),
    "Quest": (
        "the objective, who gave the quest, and the promised reward",
        [
            ("Status", "offered, active, completed, failed or abandoned, with the session it changed"),
            ("Progress", "steps taken, in session order"),
            ("People & Places", "the entities involved and their part in it"),
            _OPEN_QUESTIONS,
        ],
    ),
}
PLAYER_CHARACTER_LAYOUT: Layout = (
    "who this player character is in the story so far",
    [
        ("Relationships", "with other characters, factions and the party"),
        ("History", "what happened involving them, in session order"),
    ],
)


def page_layout(entity: Entity, player_character: bool = False) -> Layout | None:
    return PLAYER_CHARACTER_LAYOUT if player_character else PAGE_LAYOUTS.get(entity.type)


def render_layout(layout: Layout | None) -> str:
    """Layout text for the page-rewrite prompt; headings are kept apart from their guidance."""
    if layout is None:
        return ""
    opening, sections = layout
    lines = [f"Opening paragraph (no heading): {opening}.", "Sections, in this order (heading: what goes in it):"]
    lines.extend(f"- `## {heading}`: {guidance}." for heading, guidance in sections)
    return "\n".join(lines)


def normalize_headings(markdown: str, layout: Layout | None) -> str:
    """Trim guidance text a model copied into a heading (`## Status: alive, dead...` -> `## Status`)
    and a leading `short_description:` line copied from the response fields."""
    # Some models echo the JSON field into the page body; drop such a line.
    markdown = re.sub(r"^\s*\**short_description\**\s*:.*\n+", "", markdown, flags=re.IGNORECASE)
    if layout is None:
        return markdown
    headings = sorted((heading for heading, _ in layout[1]), key=len, reverse=True)
    lines = []
    for line in markdown.splitlines():
        match = re.match(r"^(#{2,3})\s+(.*)$", line)
        if match:
            text = match.group(2).strip()
            for heading in headings:
                if text.lower().startswith(heading.lower()) and text[len(heading) :].strip()[:1] in {":", "(", "-", "–", "—", ""}:
                    line = f"{match.group(1)} {heading}"
                    break
        lines.append(line)
    return "\n".join(lines)


# --- Formatting helpers (pure, unit-tested) ------------------------------------------------


def entity_header(entity: Entity) -> str:
    header = f"[{entity.type}] {entity.canonical_name}"
    if entity.aliases:
        header += f" (aka {', '.join(entity.aliases)})"
    return header


def linked_quests(storage: Storage, guild_id: int) -> dict[int, list[Entity]]:
    """Quests each entity is involved in: those whose active facts name it (name or alias)."""
    entities = storage.list_entities(guild_id)
    quests = {entity.id: entity for entity in entities if entity.type == "Quest"}
    links: dict[int, list[Entity]] = {}
    for quest_id, texts in storage.active_facts_by_type(guild_id, "Quest").items():
        if quest_id not in quests:
            continue
        for entity_id in sorted(mentioned_entity_ids(entities, "\n".join(texts), fuzzy=False)):
            if entity_id != quest_id:
                links.setdefault(entity_id, []).append(quests[quest_id])
    return links


def page_embedding_text(entity: Entity, markdown: str) -> str:
    return f"{entity_header(entity)}\n{CITATION_RE.sub('', markdown)}"


def format_fact(fact: Fact) -> str:
    return f"[F{fact.id}] ({fact.source_label()}) {fact.text}"


def spelling_glossary(entities: list[Entity], registered_characters: list[str]) -> list[str]:
    """Registered character names plus entity names and aliases, deduplicated, sorted."""
    names: dict[str, str] = {}
    for name in registered_characters:
        names.setdefault(normalize_name(name), name.strip())
    for entity in entities:
        if entity.type in UNSPOKEN_NAME_TYPES:
            continue
        for name in entity.names():
            names.setdefault(normalize_name(name), name.strip())
    names.pop("", None)
    return sorted(names.values(), key=str.casefold)


def format_entity_index(entities: list[Entity], text: str) -> str:
    """Entity index for the extraction prompt.

    Every entity is listed by id, type and names so the model can match misspellings; only those
    whose name (or something spelled like it) appears in `text` also carry their one-line
    description, to keep the prompt short.
    """
    mentioned = mentioned_entity_ids(entities, text)
    lines = []
    for entity in entities:
        line = f"#{entity.id} {entity_header(entity)}"
        if entity.id in mentioned and entity.short_description:
            line += f": {entity.short_description}"
        lines.append(line)
    return "\n".join(lines)


def quest_list(quests: list[Entity], file_names: dict[int, str] | None = None) -> list[str]:
    """`## Quests` section listing linked quests and their current status (Obsidian links if given)."""
    if not quests:
        return []
    lines = ["## Quests"]
    for quest in quests:
        name = quest.canonical_name
        if file_names and quest.id in file_names:
            name = f"[[{file_names[quest.id]}|{name}]]"
        lines.append(f"- {name} (#{quest.id}): {quest.status or 'status unknown'}")
    return lines


def mentioned_entity_ids(entities: list[Entity], text: str, fuzzy: bool = True) -> set[int]:
    """Entities whose name or alias appears in `text`, exactly or (with `fuzzy`) with a near
    spelling (speech-to-text misspellings such as "Varic" for "Varric")."""
    tokens = normalize_name(text).split()
    grams: dict[int, set[str]] = {
        n: {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)} for n in range(1, 5)
    }
    buckets: dict[tuple[int, str], list[str]] = {}
    for n, values in grams.items():
        for gram in values:
            buckets.setdefault((n, gram[0]), []).append(gram)
    joined = f" {' '.join(tokens)} "
    mentioned: set[int] = set()
    for entity in entities:
        for name in {normalize_name(name) for name in entity.names()} - {""}:
            size = len(name.split())
            if f" {name} " in joined or f" {name}s " in joined:  # also the possessive ("Penn's" -> "penns")
                mentioned.add(entity.id)
                break
            if not fuzzy:
                continue
            # The whole name with a near spelling, or one distinctive word of a longer name
            # ("Thalren" for "Thalrin Vey").
            targets = [name] if size <= 4 and len(name) >= 4 else []
            if size > 1:
                targets += [word for word in _distinctive(name) if len(word) >= 5]
            if any(_near_mention(target, buckets) for target in targets):
                mentioned.add(entity.id)
                break
    return mentioned


def _near_mention(target: str, buckets: dict[tuple[int, str], list[str]]) -> bool:
    return any(
        abs(len(gram) - len(target)) <= 2 and SequenceMatcher(None, target, gram).ratio() >= MENTION_SIMILARITY
        for gram in buckets.get((len(target.split()), target[0]), [])
    )


def match_candidates(entities: list[Entity], names: list[str], limit: int = MAX_MATCH_CANDIDATES) -> list[Entity]:
    """Existing entities that might be the same as something called `names`, best first.

    Deliberately loose (the LLM makes the call): similar spelling of the name or of one of its
    words, a shared distinctive name word ("Lord Thane" / "Varric Thane"), or all of the name's distinctive words appearing in an
    entity's description ("the harbour master" / "former harbour master of Gullhaven").
    """
    wanted = {normalize_name(name) for name in names} - {""}
    scored: list[tuple[float, int, Entity]] = []
    for entity in entities:
        score = 0.0
        description = set(normalize_name(entity.short_description).split())
        for a in wanted:
            words_a = _distinctive(a)
            if words_a and words_a <= description:
                score = max(score, 0.6)
            for b in {normalize_name(name) for name in entity.names()} - {""}:
                if min(len(a), len(b)) >= 4:
                    ratio = SequenceMatcher(None, a, b).ratio()
                    if ratio >= CANDIDATE_SIMILARITY:
                        score = max(score, ratio)
                words_b = _distinctive(b)
                shared = words_a & words_b
                if shared:
                    score = max(score, 0.7 + 0.05 * len(shared))
                elif any(
                    SequenceMatcher(None, x, y).ratio() >= WORD_SIMILARITY for x in words_a for y in words_b
                ):
                    score = max(score, 0.65)  # A misspelled name word: "Varik" / "Varric Thane".
        if score > 0:
            scored.append((score, -entity.id, entity))
    scored.sort(reverse=True)
    return [entity for _, _, entity in scored[:limit]]


def _distinctive(name: str) -> set[str]:
    return {word for word in name.split() if len(word) >= 4 and word not in TITLE_WORDS}


def transcript_excerpt(chunk: str, timestamps: list[str], context: int = 1) -> str:
    """Transcript lines at the given `[HH:MM:SS]` times, with `context` lines before each."""
    lines = chunk.splitlines()
    wanted = {f"[{ts}]" for ts in timestamps if TIMESTAMP_RE.match(ts)}
    keep: set[int] = set()
    for index, line in enumerate(lines):
        if any(line.startswith(ts) for ts in wanted):
            keep.update(range(max(0, index - context), index + 1))
    return "\n".join(lines[i] for i in sorted(keep))


def _candidate(first: Entity, second: Entity, reason: str) -> DuplicateCandidate:
    return DuplicateCandidate(first, second, reason) if first.id < second.id else DuplicateCandidate(second, first, reason)


def _combine_candidates(*groups: list[DuplicateCandidate]) -> list[DuplicateCandidate]:
    combined: list[DuplicateCandidate] = []
    seen: set[tuple[int, int]] = set()
    for group in groups:
        for candidate in group:
            pair = (min(candidate.first.id, candidate.second.id), max(candidate.first.id, candidate.second.id))
            if pair not in seen:
                seen.add(pair)
                combined.append(candidate)
    return combined


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
    if report.renamed:
        lines.append("**Renamed:** " + ", ".join(f"{old} → {new}" for old, new in report.renamed))
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
    if report.pages_without_embedding:
        lines.append(
            f"**{report.pages_without_embedding} document(s) not in semantic search yet**: the embedding "
            "model is not available. Name and keyword search still find them; retried on the next change."
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
