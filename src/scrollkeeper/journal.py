"""Import a VTT journal export (Roll20-style JSON) as the campaign wiki's initial source (#15).

The export is one JSON array of journal entries (`type` handout or character, `id`, `name`,
`folder`, `order`, `archived`, and HTML `notes` / `gmnotes`). An ordered list of folder rules
decides what each entry becomes:

- `session`: a dated session recap. It becomes a session with the recap as its summary (no
  audio), and its facts are extracted like a session's, citing that session.
- `entity`: one wiki entity named after the entry, with the entry's text as `imported` facts
  (short entries as one fact, long ones split by the LLM, which may also attach facts to other
  entities it mentions).
- `document`: lore text (letters, plot hooks); facts are extracted onto the entities it is about.
- `name`: an entity with no facts (names for the spelling glossary and the entity index).
- `skip`: not imported (published setting and adventure text). Entries no rule matches are
  skipped too, and both are listed in the report so the operator can opt folders in.

`notes` and `gmnotes` are imported alike. Every imported fact keeps the journal entry id as
`source_ref` (`journal:<id>`), and `journal_entries` remembers each entry's content hash: a
re-import skips unchanged entries and replaces the facts of changed ones (pinned facts still win
on the page). The import runs as a job stored in SQLite, so it resumes after a restart.

Run `python -m scrollkeeper.journal <export.json>` to preview what an export would import.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from fnmatch import fnmatchcase
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Awaitable, Callable

from .models import ENTITY_TYPES, JOURNAL_SOURCE_PREFIX, DuplicateCandidate, Entity, normalize_name
from .storage import Storage
from .wiki import (
    CampaignWiki,
    ExtractionResult,
    duplicate_reason,
    entity_header,
    find_possible_duplicates,
    format_entity_index,
)


log = logging.getLogger(__name__)

Notifier = Callable[[str], Awaitable[None]]
ProgressCallback = Callable[[str], None]

ACTIONS = ("session", "entity", "document", "name", "skip")
ENTRY_TYPES = ("character", "handout")
# An entity entry at most this long becomes one fact as written; longer ones go to the LLM.
SHORT_ENTRY_CHARS = 400
# An import that crashed the bot this many times is marked failed instead of resumed again.
MAX_IMPORT_ATTEMPTS = 3
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
# Report lists longer than this are cut short in Discord.
REPORT_LIST_LIMIT = 40


@dataclass(slots=True, frozen=True)
class FolderRule:
    """What to do with entries in a folder. `folder` is a case-insensitive pattern (`*` matches
    anything) for the folder path, written with ` / ` between levels; it also matches everything
    inside that folder. `""` is the top level. `entry_type` limits the rule to `character` or
    `handout` entries."""

    folder: str
    action: str
    entity_type: str = ""
    entry_type: str = ""

    def matches(self, folder: str, entry_type: str) -> bool:
        if self.entry_type and self.entry_type != entry_type:
            return False
        pattern = normalize_folder(self.folder).casefold()
        path = folder.casefold()
        if not pattern:
            return not path
        return fnmatchcase(path, pattern) or fnmatchcase(path, f"{pattern} / *")


# The defaults, from a survey of a real export (#15): published setting, rules and adventure text
# and image handouts are skipped; campaign material is imported. Operator rules come first.
DEFAULT_RULES: tuple[FolderRule, ...] = (
    FolderRule("Chapter *", "skip"),
    FolderRule("Welcome to Eberron", "skip"),
    FolderRule("Supplemental Material", "skip"),
    FolderRule("Forgotten Relics Adventure", "skip"),
    FolderRule("Descent into Avernus Rollable Tables Add-on", "skip"),
    FolderRule("START HERE*", "skip"),
    FolderRule("Player Art Handouts", "skip"),
    FolderRule("Session Notes", "session"),
    FolderRule("NPC", "entity", "Character"),
    FolderRule("Locations", "entity", "Location"),
    FolderRule("Magic Items", "entity", "Item"),
    FolderRule("Rifts", "entity", "PointOfInterest"),
    FolderRule("Monster Lairs", "entity", "PointOfInterest"),
    FolderRule("Story Plothook", "document"),
    FolderRule("Creatures", "name", "Character"),
    FolderRule("Bosses and Avatars", "name", "Character"),
    FolderRule("Characters", "name", "Character"),
    FolderRule("", "name", "Character", entry_type="character"),
)


def parse_rules(raw: list[Any]) -> tuple[FolderRule, ...]:
    """Operator rules (`SCROLLKEEPER_JOURNAL_RULES`), checked before the defaults.

    Each is `{"folder": ..., "action": ..., "type": ..., "entries": "character" | "handout"}`;
    `type` and `entries` are optional.
    """
    rules = []
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("folder"), str):
            raise RuntimeError(f"SCROLLKEEPER_JOURNAL_RULES item {index} needs a \"folder\" string")
        action = str(item.get("action", "")).strip().lower()
        entity_type = str(item.get("type", "")).strip()
        entry_type = str(item.get("entries", "")).strip().lower()
        if action not in ACTIONS:
            raise RuntimeError(f"SCROLLKEEPER_JOURNAL_RULES item {index}: action must be one of {', '.join(ACTIONS)}")
        if entity_type and entity_type not in ENTITY_TYPES:
            raise RuntimeError(f"SCROLLKEEPER_JOURNAL_RULES item {index}: type must be one of {', '.join(ENTITY_TYPES)}")
        if entry_type and entry_type not in ENTRY_TYPES:
            raise RuntimeError(f"SCROLLKEEPER_JOURNAL_RULES item {index}: entries must be character or handout")
        rules.append(FolderRule(item["folder"], action, entity_type, entry_type))
    return (*rules, *DEFAULT_RULES)


def match_rule(rules: tuple[FolderRule, ...], folder: str, entry_type: str) -> FolderRule | None:
    return next((rule for rule in rules if rule.matches(folder, entry_type)), None)


# --- Parsing ---------------------------------------------------------------------------------


@dataclass(slots=True)
class JournalEntry:
    journal_id: str
    entry_type: str
    name: str
    folder: str
    order: int = 0
    archived: bool = False
    text: str = ""
    gm_text: str = ""

    @property
    def body(self) -> str:
        """The entry's text with its GM notes (imported alike, see #15)."""
        if self.text and self.gm_text:
            return f"{self.text}\n\nGM notes:\n{self.gm_text}"
        return self.text or self.gm_text


def parse_export(raw: bytes | str) -> list[JournalEntry]:
    """Read an export (blocking). Raises ValueError when it is not a journal export."""
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"The file is not JSON ({exc}).") from exc
    if not isinstance(data, list):
        raise ValueError("The file is not a journal export: expected a JSON array of entries.")
    entries = []
    for item in data:
        if not isinstance(item, dict) or not item.get("id") or not str(item.get("name", "")).strip():
            continue
        try:
            order = int(item.get("order") or 0)
        except (TypeError, ValueError):
            order = 0
        entries.append(
            JournalEntry(
                journal_id=str(item["id"]),
                entry_type=str(item.get("type", "")).strip().lower(),
                name=" ".join(str(item["name"]).split()),
                folder=normalize_folder(str(item.get("folder") or "")),
                order=order,
                archived=bool(item.get("archived")),
                text=html_to_text(str(item.get("notes") or "")),
                gm_text=html_to_text(str(item.get("gmnotes") or "")),
            )
        )
    if data and not entries:
        raise ValueError("The file is not a journal export: no entry has an id and a name.")
    return entries


def normalize_folder(folder: str) -> str:
    """`NPC /  Harbour ` -> `NPC / Harbour` (exports pad folder names with spaces)."""
    return " / ".join(" ".join(part.split()) for part in folder.split("/") if part.strip())


_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BLOCKS = frozenset("p div tr ul ol table thead tbody blockquote pre dl dt dd hr section article".split())


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self._pre = 0
        self._cells = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            self._skip += 1
        elif tag in _HEADINGS:
            self.parts.append("\n\n" + "#" * _HEADINGS[tag] + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in ("td", "th"):
            if self._cells:
                self.parts.append(" | ")
            self._cells += 1
        elif tag in _BLOCKS:
            if tag == "tr":
                self._cells = 0
            if tag == "pre":
                self._pre += 1
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif (tag in _HEADINGS or tag in _BLOCKS) and tag != "tr":  # Rows start their own line.
            if tag == "pre":
                self._pre = max(0, self._pre - 1)
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        data = data.replace("\xa0", " ")
        self.parts.append(data if self._pre else re.sub(r"\s+", " ", data))


def html_to_text(html: str) -> str:
    """Journal HTML as plain text with Markdown headings, list bullets and `a | b` table rows."""
    if not html.strip():
        return ""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = [line.strip() for line in "".join(parser.parts).splitlines()]
    text = "\n".join(re.sub(r"\s+\|", " |", line) for line in lines if line not in {"-", "|"})
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --- Names and dates ---------------------------------------------------------------------------

_NICKNAME = re.compile(r"\s*[\"“”]([^\"“”]+)[\"“”]\s*")
# Character sheets that are another copy of someone ("Mira-Old", "Copy of X", "X 2", "X Image").
_VARIANT_PATTERNS = (
    re.compile(r"^copy of\s+", re.IGNORECASE),
    re.compile(r"[\s-]+old$", re.IGNORECASE),
    re.compile(r"^old\s+(?=.+\s+sheet$)", re.IGNORECASE),
    re.compile(r"\s+sheet$", re.IGNORECASE),
    re.compile(r"\s+image$", re.IGNORECASE),
    re.compile(r"\s*\d+$"),
)
_DATE = re.compile(r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})")
_MONTHS = {
    name: number
    for number, name in enumerate(
        "january february march april may june july august september october november december".split(), start=1
    )
}


def split_name(name: str, entry_type: str) -> tuple[str, list[str], bool]:
    """(canonical name, aliases, looks like a copy of another sheet) for an entry name.

    A quoted nickname becomes an alias (`Pip "Stormcrow" Hale` -> `Pip Hale`, alias
    `Stormcrow`). Character sheets named like a copy (`Mira-Old`, `Copy of X`, `X 2`) map to
    the base name and are reported for review.
    """
    variant = False
    if entry_type == "character":
        base = name
        for pattern in _VARIANT_PATTERNS:
            base = pattern.sub("", base).strip()
        if base != name and re.search(r"[^\W\d_]", base):
            name, variant = base, True
    aliases = [match.strip() for match in _NICKNAME.findall(name) if match.strip()]
    canonical = " ".join(_NICKNAME.sub(" ", name).split()) if aliases else name
    if not normalize_name(canonical):
        return name, [], variant
    return canonical, aliases, variant


def recap_date(entry: JournalEntry) -> tuple[date | None, str]:
    """The date of a session recap, and a review note when its name and folder disagree.

    Names are US dates (`09/13/2026`, `09-30-2024`); the folder (`Session Notes / 2024 / May`)
    gives the year and month. When they disagree the folder's year wins (a typo in the name is
    more likely than a misfiled recap).
    """
    folder_year = next((int(part) for part in entry.folder.split(" / ") if re.fullmatch(r"\d{4}", part)), None)
    folder_month = next((_MONTHS[p.casefold()] for p in entry.folder.split(" / ") if p.casefold() in _MONTHS), None)
    match = _DATE.search(entry.name)
    if match is None:
        if folder_year is None:
            return None, f"'{entry.name}' ({entry.folder}): no date in the name or folder; not imported"
        found = date(folder_year, folder_month or 1, 1)
        return found, f"'{entry.name}' ({entry.folder}): no date in the name; dated {found.isoformat()}"
    month, day, year = (int(part) for part in match.groups())
    if folder_month is not None and month != folder_month and day == folder_month:
        month, day = day, month  # Written day first.
    note = ""
    if folder_year is not None and year != folder_year:
        year = folder_year
        note = "the year in the name does not match the folder"
    try:
        found = date(year, month, day)
    except ValueError:
        if folder_year is None:
            return None, f"'{entry.name}' ({entry.folder}): not a valid date; not imported"
        found = date(folder_year, folder_month or 1, 1)
        note = "not a valid date"
    if note:
        return found, f"'{entry.name}' ({entry.folder}): {note}; dated {found.isoformat()}"
    return found, ""


# --- Planning (pure; also the preview) ----------------------------------------------------------


@dataclass(slots=True)
class PlannedEntry:
    entry: JournalEntry
    action: str
    entity_type: str = ""
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    variant: bool = False
    date: date | None = None

    @property
    def content_hash(self) -> str:
        """Changes when the entry's content or what it is imported as changes."""
        e = self.entry
        key = [e.name, e.folder, e.entry_type, e.text, e.gm_text, self.action, self.entity_type, str(self.date or "")]
        return hashlib.sha256(json.dumps(key).encode()).hexdigest()


@dataclass(slots=True)
class ImportPlan:
    entries: list[PlannedEntry] = field(default_factory=list)
    total: int = 0
    excluded: Counter = field(default_factory=Counter)
    unmatched: Counter = field(default_factory=Counter)
    empty: int = 0
    archived: int = 0
    review: list[str] = field(default_factory=list)

    def count(self, action: str) -> int:
        return sum(1 for planned in self.entries if planned.action == action)


# Import order: names and entities first, so recaps and documents are extracted against a full
# entity index; then recaps by date; then documents.
_ACTION_ORDER = {"name": 0, "entity": 1, "session": 2, "document": 3}


def plan_import(entries: list[JournalEntry], rules: tuple[FolderRule, ...] = DEFAULT_RULES) -> ImportPlan:
    plan = ImportPlan(total=len(entries))
    for entry in entries:
        if entry.archived:
            plan.archived += 1
            continue
        rule = match_rule(rules, entry.folder, entry.entry_type)
        if rule is None:
            plan.unmatched[entry.folder or "(top level)"] += 1
            continue
        if rule.action == "skip":
            plan.excluded[entry.folder.split(" / ")[0] or "(top level)"] += 1
            continue
        action = rule.action
        if not entry.body and action != "name":
            if action == "entity" and entry.entry_type == "character":
                action = "name"  # A named character sheet with no notes still names someone.
            else:
                plan.empty += 1
                continue
        planned = PlannedEntry(entry, action, rule.entity_type or ("Character" if action in ("entity", "name") else ""))
        if action == "session":
            planned.date, note = recap_date(entry)
            if note:
                plan.review.append(f"Recap date: {note}")
            if planned.date is None:
                continue
        elif action in ("entity", "name"):
            planned.name, planned.aliases, planned.variant = split_name(entry.name, entry.entry_type)
        plan.entries.append(planned)
    # Copies of sheets go after the other names, so they can join the sheet they copy.
    plan.entries.sort(
        key=lambda p: (_ACTION_ORDER[p.action], p.variant, p.date or date.min, p.entry.folder.casefold(), -p.entry.order)
    )
    return plan


def estimate_llm_calls(plan: ImportPlan, chunk_chars: int) -> tuple[int, int]:
    """(fact-extraction calls, entities likely to get a page) for the preview."""
    extraction = 0
    page_entities: set[str] = set()
    for planned in plan.entries:
        body = planned.entry.body
        if planned.action == "entity":
            page_entities.add(normalize_name(planned.name))
            if len(body) > SHORT_ENTRY_CHARS:
                extraction += len(split_text(body, chunk_chars))
        elif planned.action in ("session", "document"):
            extraction += len(split_text(body, chunk_chars))
    return extraction, len(page_entities)


def format_plan(plan: ImportPlan, chunk_chars: int = 24000) -> str:
    """The dry-run report: what an import of this export would do, without doing it."""
    lines = [f"**Journal import preview**: {plan.total} entries in the export."]
    recaps = [p for p in plan.entries if p.action == "session"]
    if recaps:
        lines.append(
            f"- **{len(recaps)} session recaps** ({recaps[0].date.isoformat()} to {recaps[-1].date.isoformat()}) "
            "become dated sessions; facts are extracted from each."
        )
    entities = Counter(p.entity_type for p in plan.entries if p.action == "entity")
    if entities:
        long_entries = sum(
            1 for p in plan.entries if p.action == "entity" and len(p.entry.body) > SHORT_ENTRY_CHARS
        )
        lines.append(
            f"- **{sum(entities.values())} entries become entities with facts** ({_counts(entities)}); "
            f"{long_entries} are long enough to be split by the LLM."
        )
    if plan.count("document"):
        lines.append(f"- **{plan.count('document')} documents** (lore, letters): facts are extracted onto the entities they mention.")
    names = Counter(p.entity_type for p in plan.entries if p.action == "name")
    if names:
        lines.append(f"- **{sum(names.values())} names only** ({_counts(names)}): entities and spelling-glossary terms, no facts.")
    extraction, pages = estimate_llm_calls(plan, chunk_chars)
    lines.append(
        f"- About **{extraction} fact-extraction calls** and **{pages}+ page rewrites** (more as recaps name new entities)."
    )
    lines.append("")
    lines.append("**Skipped**")
    if plan.excluded:
        lines.append("- Excluded (published or reference material): " + _folder_counts(plan.excluded))
    if plan.unmatched:
        lines.append(
            "- No rule matches (add one to `SCROLLKEEPER_JOURNAL_RULES` to import): " + _folder_counts(plan.unmatched)
        )
    lines.append(f"- {plan.empty} empty, {plan.archived} archived.")
    variants = [p for p in plan.entries if p.variant]
    review = plan.review + [
        f"'{p.entry.name}' looks like a copy of a sheet; it is imported as '{p.name}' (joining an entity of that name)"
        for p in variants
    ]
    review += [f"{count} entries are named '{name}'; they become one entity" for name, count in _repeated_names(plan)]
    if review:
        lines.extend(["", "**Needs review**", *_capped(review)])
    return "\n".join(lines)


def _counts(counter: Counter) -> str:
    return ", ".join(f"{key} {value}" for key, value in counter.most_common())


def _folder_counts(counter: Counter) -> str:
    return ", ".join(f"`{folder}` ({count})" for folder, count in sorted(counter.items(), key=lambda kv: kv[0].casefold()))


def _capped(items: list[str], limit: int = REPORT_LIST_LIMIT) -> list[str]:
    lines = [f"- {item}" for item in items[:limit]]
    if len(items) > limit:
        lines.append(f"- ...and {len(items) - limit} more.")
    return lines


def _repeated_names(plan: ImportPlan) -> list[tuple[str, int]]:
    counts = Counter(normalize_name(p.name) for p in plan.entries if p.action in ("entity", "name") and not p.variant)
    first: dict[str, str] = {}
    for p in plan.entries:
        first.setdefault(normalize_name(p.name), p.name)
    return [(first[norm], count) for norm, count in counts.items() if count > 1]


def split_text(text: str, max_chars: int) -> list[str]:
    """Cut text into parts of at most `max_chars` at paragraph breaks (or lines, or hard cuts)."""
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    current = ""
    for block in re.split(r"(\n\n+)", text):
        while len(block) > max_chars:
            cut = block.rfind("\n", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            if current.strip():
                parts.append(current.strip())
                current = ""
            parts.append(block[:cut].strip())
            block = block[cut:]
        if current and len(current) + len(block) > max_chars:
            parts.append(current.strip())
            current = ""
        current += block
    if current.strip():
        parts.append(current.strip())
    return [part for part in parts if part]


# --- Running an import -------------------------------------------------------------------------


@dataclass(slots=True)
class JournalImportReport:
    plan: ImportPlan
    sessions: int = 0
    documents: int = 0
    entity_entries: int = 0
    name_entries: int = 0
    unchanged: int = 0
    facts_added: int = 0
    facts_retracted: int = 0
    new_entities: list[Entity] = field(default_factory=list)
    updated_entities: list[Entity] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)
    review: list[str] = field(default_factory=list)
    possible_duplicates: list[DuplicateCandidate] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    page_failures: list[str] = field(default_factory=list)
    resumed: bool = False


class JournalImporter:
    """Runs journal imports as jobs stored in SQLite (one at a time per server), resuming them
    after a restart. Entries already imported with the same content are skipped, so a resumed
    or repeated import only does the remaining work."""

    def __init__(self, storage: Storage, wiki: CampaignWiki, rules: tuple[FolderRule, ...] = DEFAULT_RULES) -> None:
        self.storage = storage
        self.wiki = wiki
        self.rules = rules
        self.tasks: dict[int, asyncio.Task] = {}  # by guild id
        self.statuses: dict[int, str] = {}  # by guild id
        self._notice: Callable[[int, str], Awaitable[None]] | None = None
        self._started = False

    def set_notice_handler(self, handler: Callable[[int, str], Awaitable[None]]) -> None:
        """Handler(text_channel_id, message) that posts import progress and the report."""
        self._notice = handler

    @property
    def imports_dir(self) -> Path:
        """Where uploaded exports wait while their import runs (deleted when it completes)."""
        return self.storage.data_dir / "imports"

    @property
    def chunk_chars(self) -> int:
        return self.wiki.settings.extract_chunk_chars

    def preview(self, raw: bytes) -> str:
        """The dry-run report for an export (blocking)."""
        return format_plan(plan_import(parse_export(raw), self.rules), self.chunk_chars)

    def is_running(self, guild_id: int) -> bool:
        task = self.tasks.get(guild_id)
        return task is not None and not task.done()

    def status(self, guild_id: int) -> str:
        if self.is_running(guild_id):
            return f"Journal import running: {self.statuses.get(guild_id, 'starting')}"
        return self.statuses.get(guild_id, "No journal import has run since the bot started.")

    async def start(self) -> None:
        """Resume imports a restart interrupted. Call once the bot is ready."""
        if self._started:
            return
        self._started = True
        for job in await asyncio.to_thread(self.storage.running_journal_imports):
            await self._notify(int(job["text_channel_id"]), f"Resuming journal import #{job['id']} after a restart.")
            self._spawn(job, resumed=True)

    async def begin(self, guild_id: int, campaign_id: int, text_channel_id: int, raw: bytes) -> ImportPlan:
        """Check the export, store it, and start importing it in the background."""
        if self.is_running(guild_id):
            raise RuntimeError("A journal import is already running in this server. `!import-status` shows its progress.")
        plan = await asyncio.to_thread(lambda: plan_import(parse_export(raw), self.rules))
        if not plan.entries:
            raise RuntimeError("Nothing in this export matches the import rules; see `!import-journal preview`.")
        job_id = await asyncio.to_thread(self._store_upload, guild_id, campaign_id, text_channel_id, raw)
        job = await asyncio.to_thread(self._job_row, job_id)
        self._spawn(job, resumed=False)
        return plan

    def _store_upload(self, guild_id: int, campaign_id: int, text_channel_id: int, raw: bytes) -> int:
        folder = self.imports_dir / str(campaign_id)
        folder.mkdir(parents=True, exist_ok=True)
        upload = folder / f"upload-{datetime.utcnow():%Y%m%dT%H%M%S%f}.json"
        upload.write_bytes(raw)
        return self.storage.create_journal_import(campaign_id, guild_id, text_channel_id, upload)

    def _job_row(self, job_id: int):
        return next(row for row in self.storage.running_journal_imports() if int(row["id"]) == job_id)

    def _spawn(self, job: Any, resumed: bool) -> None:
        guild_id = int(job["guild_id"])
        self.tasks[guild_id] = asyncio.create_task(self._run_job(job, resumed))

    async def _run_job(self, job: Any, resumed: bool) -> None:
        job_id, guild_id, channel = int(job["id"]), int(job["guild_id"]), int(job["text_channel_id"])
        campaign_id = int(job["campaign_id"])
        path = Path(job["path"])
        attempts = await asyncio.to_thread(self.storage.begin_journal_import_attempt, job_id)
        if attempts > MAX_IMPORT_ATTEMPTS:
            await asyncio.to_thread(self.storage.finish_journal_import, job_id, "failed")
            self.statuses[guild_id] = f"Journal import #{job_id} failed."
            await self._notify(
                channel,
                f"Journal import #{job_id} was interrupted {attempts - 1} times; giving up. "
                "Check the bot logs, then run `!import-journal` again (finished entries are skipped).",
            )
            return
        try:
            raw = await asyncio.to_thread(path.read_bytes)
            plan = await asyncio.to_thread(lambda: plan_import(parse_export(raw), self.rules))
            report = await self.import_plan(
                campaign_id,
                plan,
                notify=lambda message: self._notify(channel, message),
                on_progress=lambda message: self.statuses.__setitem__(guild_id, message),
            )
            report.resumed = resumed
        except Exception as exc:
            log.exception("Journal import #%s failed", job_id)
            await asyncio.to_thread(self.storage.finish_journal_import, job_id, "failed")
            self.statuses[guild_id] = f"Journal import #{job_id} failed: {exc}"
            await self._notify(channel, f"Journal import #{job_id} failed: {str(exc)[:1500]}")
            return
        await asyncio.to_thread(self.storage.finish_journal_import, job_id, "completed")
        await asyncio.to_thread(_unlink, path)
        self.statuses[guild_id] = f"Journal import #{job_id} finished."
        await self._notify(channel, format_import_report(report))

    async def _notify(self, text_channel_id: int, message: str) -> None:
        if self._notice is not None:
            try:
                await self._notice(text_channel_id, message)
            except Exception:
                log.warning("Could not post a journal import notice", exc_info=True)

    async def import_plan(
        self,
        campaign_id: int,
        plan: ImportPlan,
        notify: Notifier | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> JournalImportReport:
        """Import every planned entry, then rewrite the pages whose facts changed."""
        report = JournalImportReport(plan=plan, review=list(plan.review))
        result = ExtractionResult()
        lock = self.wiki.campaign_lock(campaign_id)
        on_wait = _once(notify)
        await self.wiki.ensure_player_characters(campaign_id)
        entries_per_entity: Counter = Counter()
        total = len(plan.entries)
        next_notice = 0.25
        for index, planned in enumerate(plan.entries, start=1):
            _progress(on_progress, f"importing entry {index}/{total} ({planned.action}: {planned.entry.name}).")
            if notify is not None and (index - 1) / total >= next_notice:
                await notify(f"Journal import: {index - 1}/{total} entries done.")
                next_notice += 0.25
            previous = await asyncio.to_thread(self.storage.get_journal_entry, campaign_id, planned.entry.journal_id)
            if previous is not None and previous["content_hash"] == planned.content_hash:
                report.unchanged += 1
                if previous["entity_id"] is not None:
                    entries_per_entity[int(previous["entity_id"])] += 1
                continue
            try:
                async with lock:
                    entity_id = await self._import_entry(campaign_id, planned, previous, report, result, on_wait)
                if entity_id is not None:
                    entries_per_entity[entity_id] += 1
            except Exception as exc:
                log.exception("Could not import journal entry %s (%s)", planned.entry.journal_id, planned.entry.name)
                report.failures.append(f"'{planned.entry.name}' ({planned.entry.folder}): {str(exc)[:200]}")

        _progress(on_progress, "rewriting wiki pages.")
        stale = await asyncio.to_thread(self.wiki.stale_entity_ids, campaign_id)
        if notify is not None and stale:
            await notify(f"Journal import: entries done; rewriting {len(stale)} wiki page(s) (one LLM call each).")
        report.page_failures = await self.wiki.refresh_stale_pages(
            campaign_id, on_wait=on_wait, on_progress=lambda m: _progress(on_progress, m), lock=lock
        )

        entities = await asyncio.to_thread(self.storage.list_entities, campaign_id)
        by_id = {entity.id: entity for entity in entities}
        created = set(result.created)
        report.new_entities = [by_id[i] for i in result.created if i in by_id]
        report.updated_entities = [by_id[i] for i in sorted(result.touched) if i in by_id and i not in created]
        report.facts_added += result.facts_added
        report.renamed.extend(result.renamed)
        for entity_id, count in entries_per_entity.items():
            if count > 1 and entity_id in by_id:
                report.review.append(f"{count} journal entries were combined into #{entity_id} {by_id[entity_id].canonical_name}")
        flagged = [
            DuplicateCandidate(by_id[c.first.id], by_id[c.second.id], c.reason)
            for c in result.flagged
            if c.first.id in by_id and c.second.id in by_id
        ]
        seen: set[tuple[int, int]] = set()
        for candidate in [*flagged, *find_possible_duplicates(entities, created | result.touched)]:
            pair = (min(candidate.first.id, candidate.second.id), max(candidate.first.id, candidate.second.id))
            if pair not in seen:
                seen.add(pair)
                report.possible_duplicates.append(candidate)
        return report

    async def _import_entry(
        self,
        campaign_id: int,
        planned: PlannedEntry,
        previous: Any,
        report: JournalImportReport,
        result: ExtractionResult,
        on_wait: Notifier | None,
    ) -> int | None:
        """Import one entry (replacing what an earlier import of it added). Returns its entity id."""
        entry = planned.entry
        source_ref = JOURNAL_SOURCE_PREFIX + entry.journal_id
        retracted, affected = await asyncio.to_thread(
            self.storage.retract_source_facts, campaign_id, source_ref, "journal entry re-imported"
        )
        report.facts_retracted += retracted
        result.touched.update(affected)
        entity_id: int | None = None
        session_id: int | None = None
        if planned.action in ("entity", "name"):
            entity_id, created = await self._entity_for(campaign_id, planned, previous, report, result)
            if planned.action == "entity":
                await self._entity_facts(campaign_id, planned, entity_id, created, source_ref, result, on_wait)
                report.entity_entries += 1
            else:
                report.name_entries += 1
        elif planned.action == "session":
            assert planned.date is not None
            session_id = await asyncio.to_thread(
                self.storage.upsert_journal_session,
                campaign_id,
                entry.journal_id,
                "Journal recap",
                datetime.combine(planned.date, datetime.min.time()),
                f"# Session Notes\n\n{entry.body}\n",
            )
            source = f"the game master's recap of the session played on {planned.date.isoformat()}"
            await self._extract(campaign_id, entry.body, source, "", session_id, source_ref, result, on_wait)
            report.sessions += 1
        else:
            source = f'the campaign journal entry "{entry.name}" (folder {entry.folder or "top level"})'
            await self._extract(campaign_id, entry.body, source, "", None, source_ref, result, on_wait)
            report.documents += 1
        await asyncio.to_thread(
            self.storage.record_journal_entry,
            campaign_id,
            entry.journal_id,
            entry.name,
            entry.folder,
            planned.action,
            planned.content_hash,
            entity_id,
            session_id,
        )
        return entity_id

    async def _entity_for(
        self,
        campaign_id: int,
        planned: PlannedEntry,
        previous: Any,
        report: JournalImportReport,
        result: ExtractionResult,
    ) -> tuple[int, bool]:
        """The entity an entity/name entry belongs to: the one an earlier import made, else one
        with the same name or alias, else a new one. Returns (entity id, created now)."""
        names = [planned.name, *planned.aliases]
        entity = None
        if previous is not None and previous["entity_id"] is not None:
            entity = await asyncio.to_thread(self.storage.get_entity, campaign_id, int(previous["entity_id"]))
            if entity is not None and normalize_name(planned.name) not in {normalize_name(n) for n in entity.names()}:
                # The entry was renamed in the journal: its name becomes canonical, the old one an alias.
                await asyncio.to_thread(self.storage.rename_entity, entity.id, planned.name)
                report.renamed.append((entity.canonical_name, planned.name))
        if entity is None:
            matches = await asyncio.to_thread(self._find_by_any_name, campaign_id, names)
            if len(matches) > 1:
                report.review.append(
                    f"'{planned.entry.name}' matches several entities ("
                    + ", ".join(f"#{m.id} {m.canonical_name}" for m in matches)
                    + f"); attached to #{matches[0].id}"
                )
            entity = matches[0] if matches else None
        if entity is None and planned.variant:
            entity = await asyncio.to_thread(self._similar_character, campaign_id, planned.name)
        if entity is not None:
            for alias in names:
                await asyncio.to_thread(self.storage.add_alias, entity.id, alias)
            if planned.variant:
                report.review.append(
                    f"'{planned.entry.name}' looks like a copy of #{entity.id} {entity.canonical_name}; attached to it"
                )
            result.touched.add(entity.id)
            return entity.id, False

        description = f"Journal: {planned.entry.folder}" if planned.action == "name" and planned.entry.folder else ""
        entity_id = await asyncio.to_thread(
            self.storage.create_entity, campaign_id, planned.entity_type, planned.name, planned.aliases, description
        )
        result.created.append(entity_id)
        if planned.variant:
            report.review.append(f"'{planned.entry.name}' looks like a copy of a sheet; imported as #{entity_id} {planned.name}")
        return entity_id, True

    def _similar_character(self, campaign_id: int, name: str) -> Entity | None:
        """The one Character spelled like `name` (`Pip Hale` for a copy of `Pipp Hale`'s
        sheet), or None when there are none or several."""
        probe = Entity(0, campaign_id, "Character", name)
        similar = [
            entity
            for entity in self.storage.list_entities(campaign_id)
            if entity.type == "Character" and duplicate_reason(probe, entity)
        ]
        return similar[0] if len(similar) == 1 else None

    def _find_by_any_name(self, campaign_id: int, names: list[str]) -> list[Entity]:
        found: dict[int, Entity] = {}
        for name in names:
            for entity in self.storage.find_entities_by_name(campaign_id, name):
                found.setdefault(entity.id, entity)
        return list(found.values())

    async def _entity_facts(
        self,
        campaign_id: int,
        planned: PlannedEntry,
        entity_id: int,
        created: bool,
        source_ref: str,
        result: ExtractionResult,
        on_wait: Notifier | None,
    ) -> None:
        """A short entry is one fact as written; a long one is split into facts by the LLM, which
        may also refine the type of an entity this entry created."""
        body = planned.entry.body
        if len(body) <= SHORT_ENTRY_CHARS:
            text = " ".join(body.split())
            if normalize_name(planned.name) not in normalize_name(text):
                text = f"{planned.name}: {text}"  # "Mira's grandmother" alone doesn't say whose fact it is.
            await asyncio.to_thread(
                self.storage.add_fact, campaign_id, entity_id, text, "imported", None, None, source_ref
            )
            result.touched.add(entity_id)
            result.facts_added += 1
            return
        entity = await asyncio.to_thread(self.storage.get_entity, campaign_id, entity_id)
        assert entity is not None
        source = f'the campaign journal entry "{planned.entry.name}" (folder {planned.entry.folder or "top level"})'
        subject = f"#{entity.id} {entity_header(entity)}"
        subject_type = await self._extract(campaign_id, body, source, subject, None, source_ref, result, on_wait)
        if created and subject_type and subject_type != entity.type:
            log.info("Journal entry %r: type %s -> %s", planned.entry.name, entity.type, subject_type)
            await asyncio.to_thread(self.storage.set_entity_type, entity.id, subject_type)
        result.touched.add(entity_id)

    async def _extract(
        self,
        campaign_id: int,
        text: str,
        source: str,
        subject: str,
        session_id: int | None,
        source_ref: str,
        result: ExtractionResult,
        on_wait: Notifier | None,
    ) -> str:
        """Extract and store facts from game-master text, part by part. Returns the subject type
        the first part suggested ("" without a subject)."""
        subject_type = ""
        for index, part in enumerate(split_text(text, self.chunk_chars)):
            entities = await asyncio.to_thread(self.storage.list_entities, campaign_id)
            index_text = await asyncio.to_thread(format_entity_index, entities, part)
            payload = await self.wiki.llm.extract_journal_facts(index_text, part, source, subject, on_wait=on_wait)
            if index == 0:
                subject_type = payload.get("subject_type", "")
            await self.wiki.apply_extraction(
                campaign_id, session_id, payload, part, result, on_wait=on_wait, kind="imported", source_ref=source_ref
            )
        return subject_type


def format_import_report(report: JournalImportReport) -> str:
    plan = report.plan
    lines = ["### Journal import"]
    if report.resumed:
        lines.append("_Resumed after a restart: entries finished before it count as unchanged._")
    lines.append(
        f"{report.sessions} session recap(s), {report.entity_entries} entity entr(ies), {report.documents} document(s) "
        f"and {report.name_entries} name(s) imported; {report.unchanged} unchanged since the last import."
    )
    summary = f"{report.facts_added} fact(s) recorded"
    if report.facts_retracted:
        summary += f" ({report.facts_retracted} from an earlier import of changed entries replaced)"
    lines.append(summary + ".")
    if report.new_entities:
        lines.append(f"**New entities:** {len(report.new_entities)} ({_counts(Counter(e.type for e in report.new_entities))}).")
    if report.updated_entities:
        lines.append(f"**Updated entities:** {len(report.updated_entities)}.")
    if report.renamed:
        lines.append("**Renamed:** " + ", ".join(f"{old} → {new}" for old, new in report.renamed))
    skipped = []
    if plan.excluded:
        skipped.append("excluded: " + _folder_counts(plan.excluded))
    if plan.unmatched:
        skipped.append("no matching rule (opt in with `SCROLLKEEPER_JOURNAL_RULES`): " + _folder_counts(plan.unmatched))
    skipped.append(f"{plan.empty} empty, {plan.archived} archived")
    lines.append("**Skipped:** " + "; ".join(skipped) + ".")
    if report.failures:
        lines.append("**Not imported** (retried by the next `!import-journal` with this export):")
        lines.extend(_capped(report.failures))
    if report.page_failures:
        lines.append("**Pages not updated** (retried on the next run): " + ", ".join(report.page_failures[:REPORT_LIST_LIMIT]))
    if report.review:
        lines.append("**Needs review:**")
        lines.extend(_capped(report.review))
    if report.possible_duplicates:
        lines.append("**Possible duplicates** (`!merge-entity <from> <into>` to merge):")
        lines.extend(
            _capped(
                [
                    f"#{c.first.id} {c.first.canonical_name} ({c.first.type}) / "
                    f"#{c.second.id} {c.second.canonical_name} ({c.second.type}): {c.reason}"
                    for c in report.possible_duplicates
                ]
            )
        )
    return "\n".join(lines)


def _once(notify: Notifier | None) -> Notifier | None:
    """A wait notifier that posts only the first time (an import makes hundreds of LLM calls)."""
    if notify is None:
        return None
    posted = False

    async def wrapper(message: str) -> None:
        nonlocal posted
        if not posted:
            posted = True
            await notify(message)

    return wrapper


def _progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    """`python -m scrollkeeper.journal <export.json>`: print the import preview."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m scrollkeeper.journal <journal-export.json>", file=sys.stderr)
        return 2
    raw_rules = os.getenv("SCROLLKEEPER_JOURNAL_RULES", "").strip()
    rules = parse_rules(json.loads(raw_rules)) if raw_rules else DEFAULT_RULES
    entries = parse_export(Path(args[0]).read_bytes())
    print(format_plan(plan_import(entries, rules)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
