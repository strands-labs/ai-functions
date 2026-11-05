"""Skill model, memory backend, on-disk store, and the ``Skill`` tool.

A *skill* is an ability: an inference function that improves with experience
because it depends on memory experience can change. An AI Function is one;
``@skillify`` improves it by growing its *procedural memory*, a library of
reusable procedures. Each procedure is stored as a directory containing a
``SKILL.md`` file -- YAML frontmatter (name, description, and friends) above a
Markdown body of instructions -- and loaded on demand through a ``Skill``
tool, which is why the API calls a stored procedure a skill and the library a
*skill library*. This module makes that library an optimizable object rather
than a static folder. Four parts, in dependency order:

  * **Skill model** -- ``Skill``/``SkillLibrary``, ``load_skill``,
    ``save_skill``, ``load_skill_library``: parsing and writing SKILL.md
    files.
  * **``SkillLibraryBackend``** -- the ``MemoryBackend`` that exposes a
    skill library for recall, search, save, delete, and consolidation.
  * **``LibraryStore``** -- every file ``@skillify`` keeps for one
    decorated function: versions, call records, round history and the
    round lock.
  * **The ``Skill`` tool** -- how an AI Function sees a skill library: an
    index of every skill's name and one-line description, plus a ``Skill``
    tool that returns one skill's full body.

See ``docs/skills.md``.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import pickle
import re
import shutil
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, Field, TypeAdapter, create_model
from strands import tool

from ai_functions.ai_thread.ai_function import AIFunction
from ai_functions.types.events import Event

from ...memory.base import MemoryBackend, ParameterMeta
from ...memory.json_backend import _consolidate_value, _query_value
from .observation import skill_event

if TYPE_CHECKING:
    from ...types.graph import GradFeedback

__all__ = [
    "INDEX",
    "CallRecord",
    "LibraryStore",
    "Skill",
    "SkillDeliveryError",
    "SkillLibrary",
    "SkillLibraryBackend",
    "SkillToolbox",
    "backend_from_dirs",
    "load_skill",
    "load_skill_library",
    "save_skill",
    "with_skills",
]

_logger = logging.getLogger(__name__)

# ── Skill model: Skill, SkillLibrary, and the SKILL.md parser ──────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Skill:
    """A single skill loaded from disk.

    Attributes:
        path: Path to the skill directory (containing SKILL.md).
        frontmatter: Parsed YAML frontmatter dict.
        body: Markdown body text (after the frontmatter block).
        raw: Full raw SKILL.md contents.
        resources: Filenames of additional bundled files in the skill dir.
    """

    path: Path
    frontmatter: dict
    body: str
    raw: str
    resources: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The skill's name: the frontmatter ``name`` field, or the directory name."""
        return str(self.frontmatter.get("name", self.path.name))

    @property
    def description(self) -> str:
        """The skill's description, from the frontmatter ``description`` field."""
        return str(self.frontmatter.get("description", ""))

    def section_headings(self) -> list[str]:
        """Return the body's level-2 ("## ...") heading texts, in order."""
        return [m.group(1).strip() for m in _H2_RE.finditer(self.body)]


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split SKILL.md text into ``(frontmatter dict, body)``.

    Text without a ``---``-delimited frontmatter block yields an empty dict
    and the text unchanged.

    Raises:
        ValueError: The frontmatter block does not parse as a YAML mapping.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    fm_text, body = m.group(1), m.group(2)
    fm = yaml.safe_load(fm_text) or {}
    if not isinstance(fm, dict):
        raise ValueError(f"Frontmatter is not a mapping: {fm!r}")
    return fm, body


def load_skill(path: str | Path) -> Skill:
    """Load a single skill from a directory or a SKILL.md file."""
    p = Path(path)
    if p.is_file() and p.name == "SKILL.md":
        skill_dir = p.parent
        skill_md = p
    elif p.is_dir():
        skill_dir = p
        skill_md = p / "SKILL.md"
    else:
        raise FileNotFoundError(f"No skill found at {path}")

    if not skill_md.exists():
        raise FileNotFoundError(f"Missing SKILL.md in {skill_dir}")

    raw = skill_md.read_text(encoding="utf-8")
    fm, body = _split_frontmatter(raw)
    resources = sorted(f.name for f in skill_dir.iterdir() if f.is_file() and f.name != "SKILL.md")
    return Skill(path=skill_dir, frontmatter=fm, body=body, raw=raw, resources=resources)


def save_skill(skill: Skill, root: str | Path, *, dirname: str | None = None, bump_version: bool = False) -> Path:
    """Write ``skill`` as ``<root>/<dirname>/SKILL.md`` (frontmatter + body).

    ``dirname`` defaults to ``skill.name`` (the frontmatter ``name`` field).
    Pass it explicitly to target a specific directory regardless of what the
    frontmatter says — e.g. a caller that indexes skills by directory slug
    must write back to that same slug even if frontmatter ``name`` disagrees
    with it, rather than silently creating a second directory.

    Args:
        skill: The skill to write. ``skill.frontmatter`` is not mutated —
            a copy is written, with ``version`` bumped in the copy when
            ``bump_version`` applies (see below).
        root: Directory the skill's own directory lives under.
        dirname: Target directory name; defaults to ``skill.name``.
        bump_version: When ``True`` *and* ``<root>/<dirname>/SKILL.md``
            already exists, the written frontmatter's ``version`` is set to
            one more than the *existing on-disk* skill's ``version`` (an
            ``int``, defaulting to ``1`` when that skill's frontmatter has
            no ``version`` key or it isn't a valid integer) — i.e. writing
            over an existing v1 skill produces v2, and so on. When ``False``
            (the default) or the target does not yet exist, ``version`` is
            left exactly as ``skill.frontmatter`` provides it (possibly
            absent) — existing callers that never pass this keyword see no
            change in behavior.

    Returns:
        The skill's directory (``<root>/<dirname>``).
    """
    directory = Path(root) / (dirname if dirname is not None else skill.name)
    frontmatter = dict(skill.frontmatter)
    existing_skill_md = directory / "SKILL.md"
    if bump_version and existing_skill_md.exists():
        try:
            existing_fm, _ = _split_frontmatter(existing_skill_md.read_text(encoding="utf-8"))
        except ValueError:
            existing_fm = {}
        try:
            existing_version = int(existing_fm.get("version", 1))
        except (TypeError, ValueError):
            existing_version = 1
        frontmatter["version"] = existing_version + 1
    directory.mkdir(parents=True, exist_ok=True)
    front = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    (directory / "SKILL.md").write_text(f"---\n{front}\n---\n\n{skill.body.strip()}\n")
    return directory


@dataclass
class SkillLibrary:
    """A collection of skills loaded from a root directory."""

    root: Path
    skills: list[Skill] = field(default_factory=list)

    def __iter__(self) -> Iterator[Skill]:
        """Iterate over the library's skills."""
        return iter(self.skills)

    def __len__(self) -> int:
        """Return the number of skills in the library."""
        return len(self.skills)

    def by_name(self, name: str) -> Skill | None:
        """Return the skill named ``name``, or ``None`` if not found."""
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def names(self) -> list[str]:
        """Return the names of every skill in the library."""
        return [s.name for s in self.skills]


def load_skill_library(root: str | Path) -> SkillLibrary:
    """Load every skill directory under ``root``.

    A skill directory is any immediate subdirectory containing a SKILL.md.
    """
    root_p = Path(root)
    if not root_p.is_dir():
        raise NotADirectoryError(f"Skill library root is not a directory: {root}")

    skills: list[Skill] = []
    for child in sorted(root_p.iterdir()):
        if child.is_dir() and (child / "SKILL.md").exists():
            skills.append(load_skill(child))
    return SkillLibrary(root=root_p, skills=skills)


# ── SkillLibraryBackend: a MemoryBackend over a directory of skills ────────
#
# Exposes a skill library (above) as a :class:`MemoryBackend` schema: one
# parameter per skill (its body text) plus a synthetic ``skill_index`` list
# parameter carrying one ``"<slug>: <description>"`` line per skill. Skill
# directory names ("slugs", e.g. ``csv-report``) are sanitized into valid
# Python identifiers for use as parameter names (e.g. ``csv_report``) — the
# schema field descriptions come from each skill's frontmatter
# ``description``.
#
# Supported: recall, search (BM25 over the index), save (body or index
# descriptions), delete, and single-entry consolidation (rewriting one
# skill's body, or the index descriptions of a targeted subset of skills).
# Minting new skills (adding a slug that didn't exist at construction time)
# is not -- ``_save(INDEX, ...)`` rejects unknown slugs -- and neither is
# cross-entry consolidation (merging or retiring skills); see
# ``_consolidate``'s docstring.

INDEX = "skill_index"


def _param_name(slug: str) -> str:
    """Sanitize a skill slug into a valid Python identifier for use as a parameter name."""
    return slug.replace("-", "_")


def _default_consolidate_fn(value: str, feedback: list[str], description: str = "") -> str:
    """Default ``consolidate_fn``: wire the module-level ``_consolidate_value`` prompt.

    Mirrors how ``JSONMemoryBackend.__init__`` wires the same prompt function
    for scalar consolidation, adapted to the plain-callable
    ``consolidate_fn(value, feedback, description) -> str`` contract so a
    caller-injected fake and the default share one call shape.
    """
    return _consolidate_value.run_sync(value=value, feedback=feedback, description=description)


class SkillLibraryBackend(MemoryBackend):
    """Memory backend over a directory of skills.

    Each skill directory under ``root`` (any immediate subdirectory containing
    a ``SKILL.md``) becomes a scalar parameter named after its sanitized slug,
    whose value is the skill's Markdown body. A synthetic ``skill_index`` list
    parameter carries one ``"<slug>: <description>"`` line per skill, searchable
    by BM25 — the entry point for discovering which skill to recall.

    With ``stage_root``, rewrites -- ``save`` of a skill body or of index
    descriptions, and so ``consolidate`` -- are *staged*: the edited skill goes
    to ``<stage_root>/<slug>/``, replacing any earlier staged copy. Reads still
    come from ``root``. ``delete``, the one other write, raises ``TypeError``,
    so ``root`` is never written. ``@skillify`` points ``stage_root`` at the
    library's ``pending/`` directory, so an optimizer's rewrite waits for the
    next round to gate it.
    """

    def __init__(
        self,
        root: str | Path,
        actor_id: str,
        *,
        consolidate_fn: Any = None,  # pyright: ignore[reportExplicitAny]
        query_fn: Any = None,  # pyright: ignore[reportExplicitAny]
        stage_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.stage_root = Path(stage_root) if stage_root is not None else None
        self._param_to_slug: dict[str, str] = {}
        self._scan_slugs()
        self._consolidate_fn = consolidate_fn or _default_consolidate_fn
        self._query_fn = query_fn or _query_value
        super().__init__(self._build_schema(), actor_id)

    # -- Slug/param mapping ------------------------------------------------------

    def param_for(self, slug: str) -> str:
        """Return the schema parameter name for a skill slug."""
        return _param_name(slug)

    def slug_for(self, param: str) -> str:
        """Return the skill slug for a schema parameter name."""
        return self._param_to_slug[param]

    # -- Schema construction ------------------------------------------------------

    def _scan_slugs(self) -> None:
        """Rebuild ``_param_to_slug`` from the skill dirs currently on disk.

        A skill dir is any immediate subdirectory of ``root`` containing a
        ``SKILL.md``.
        """
        self._param_to_slug = {}
        for slug in sorted(p.name for p in self.root.iterdir() if (p / "SKILL.md").is_file()):
            param = _param_name(slug)
            if param == INDEX:
                raise ValueError(
                    f"skill slug {slug!r} sanitizes to {param!r}, colliding with the reserved {INDEX!r} index parameter"
                )
            if param in self._param_to_slug:
                raise ValueError(f"slug collision after sanitization: {slug!r}")
            self._param_to_slug[param] = slug

    def _build_schema(self) -> type[BaseModel]:
        """Build the Pydantic schema: ``skill_index`` plus one field per skill."""
        fields: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
            INDEX: (
                list[str],
                Field(
                    default_factory=list,
                    description=(
                        "One line per skill: '<slug>: <description>'. Edit a line to "
                        "improve when that skill gets retrieved."
                    ),
                ),
            )
        }
        for param, slug in self._param_to_slug.items():
            skill = load_skill(self.root / slug)
            fields[param] = (str, Field(default="", description=skill.description))
        return create_model("SkillLibrarySchema", **fields)

    def _rebuild_schema(self) -> None:
        """Re-scan ``root`` for skill dirs, rebuild the slug map, and reassign ``self.schema``.

        Called after a save that edits ``INDEX`` descriptions and after a
        delete, so ``_param_to_slug`` and ``self.schema`` catch up with
        what's actually there.
        """
        self._scan_slugs()
        self.schema = self._build_schema()

    # -- Abstract storage contract ------------------------------------------------

    def close(self) -> None:
        """No-op: skills are written straight to disk on every ``_save``/``_delete``."""

    def _recall(self, name: str) -> tuple[Any, ParameterMeta]:  # pyright: ignore[reportExplicitAny]
        if name == INDEX:
            lines = [
                f"{slug}: {load_skill(self.root / slug).description}" for slug in sorted(self._param_to_slug.values())
            ]
            return lines, {}
        slug = self.slug_for(name)
        return load_skill(self.root / slug).body, {}

    def _query(self, name: str, query: str) -> tuple[str, ParameterMeta]:
        value, _ = self._recall(name)
        return self._query_fn.run_sync(value=value, query=query), {}

    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> tuple[Any, ParameterMeta]:  # pyright: ignore[reportExplicitAny]
        """Return the top-k index lines ranked by BM25 against ``query``.

        Only supported for ``INDEX`` — every other parameter is a scalar
        skill body, not a searchable list.

        Uses ``BM25Plus`` rather than ``JSONMemoryBackend``'s ``BM25Okapi``:
        Okapi's IDF is exactly zero for any term whose document frequency is 1
        out of a 2-document corpus (``log(1.5) - log(1.5) == 0``), which
        produces hard ties on small skill libraries — a common case here,
        unlike the larger memory lists ``JSONMemoryBackend`` ranks. BM25Plus's
        IDF floor avoids that degenerate case.
        """
        if name != INDEX:
            raise TypeError(f"Search is only supported for '{INDEX}', but '{name}' is not a list parameter.")
        lines, _ = self._recall(INDEX)
        slugs = sorted(self._param_to_slug.values())
        entries = dict(zip(slugs, lines, strict=True))
        if not entries:
            return [], {"results": {}}
        corpus = list(entries.items())

        from rank_bm25 import BM25Plus

        bm25 = BM25Plus([value.lower().split() for _, value in corpus])
        scores = bm25.get_scores(query.lower().split())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        ranked = sorted(zip(corpus, scores, strict=True), key=lambda pair: pair[1], reverse=True)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownLambdaType]
        top = [pair for pair, _ in ranked[:k]]  # pyright: ignore[reportUnknownVariableType]
        return [value for _, value in top], {"results": dict(top)}

    def _save(self, name: str, value: Any) -> None:  # pyright: ignore[reportExplicitAny]
        # `dirname=slug` on every save below: the directory slug — not
        # frontmatter `name` — is this backend's binding identity (it's what
        # `_recall`/`_search`/`_delete` address). A skill whose frontmatter
        # `name` disagrees with its directory must still be written back to
        # that same directory, not a second one derived from `name`.
        if name == INDEX:
            for line in value:  # pyright: ignore[reportAny]
                slug, _, description = line.partition(":")
                slug = slug.strip()
                if slug not in self._param_to_slug.values():
                    raise KeyError(f"Unknown skill slug in {INDEX!r} line: {slug!r}")
                skill = self._load_for_write(slug)
                if skill.description == description.strip():
                    continue  # unchanged: nothing to write, or to stage
                skill.frontmatter["description"] = description.strip()
                self._write(skill, slug)
            # Field descriptions (sourced from frontmatter `description`) are
            # now stale on `self.schema` — refresh so `_get_description()`
            # and the tool_provider docstrings reflect the edit.
            self._rebuild_schema()
            return
        slug = self.slug_for(name)
        skill = self._load_for_write(slug)
        skill.body = value
        self._write(skill, slug)

    def _refuse_if_staged(self, method: str) -> None:
        """Raise when this backend stages its writes: ``method`` would modify the published ``root``."""
        if self.stage_root is not None:
            raise TypeError(
                f"a staged SkillLibraryBackend is read-only except for staged rewrites: {method} would modify "
                f"a published version ({self.root})"
            )

    def _load_for_write(self, slug: str) -> Skill:
        """The skill a write starts from: its staged copy when there is one, else the published one."""
        if self.stage_root is not None and (self.stage_root / slug / "SKILL.md").is_file():
            return load_skill(self.stage_root / slug)
        return load_skill(self.root / slug)

    def _write(self, skill: Skill, slug: str) -> None:
        """Write ``skill`` to ``root``, or stage it under ``stage_root`` (replacing any staged copy)."""
        if self.stage_root is None:
            save_skill(skill, self.root, dirname=slug)
            return
        self.stage_root.mkdir(parents=True, exist_ok=True)
        tmp = self.stage_root / f".tmp-{slug}-{uuid.uuid4().hex[:8]}"
        shutil.copytree(skill.path, tmp)  # bundled resources travel with the revision
        save_skill(skill, tmp.parent, dirname=tmp.name)
        _replace_dir(tmp, self.stage_root / slug)
        skill_event("skill.staged", library=str(self.stage_root.parent.resolve()), skill_dirs=[self.stage_root / slug])

    def _consolidate(
        self,
        name: str,
        feedback: list[GradFeedback],
        retrieved: dict[str, str] | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        """Rewrite a skill body, or a targeted subset of index descriptions.

        A skill-body parameter is rewritten wholesale: ``consolidate_fn(body,
        [f.text for f in feedback], field_description)`` produces the new
        body, saved via ``_save`` (frontmatter-preserving, slug-dir-targeted).

        ``INDEX`` consolidation is scoped to the entries the forward pass
        actually retrieved: only lines whose slug appears in ``retrieved``
        are sent to ``consolidate_fn`` (all lines when ``retrieved`` is
        ``None``). The targeted lines are joined, rewritten in one call, and
        the returned ``"slug: description"`` lines are parsed back — any
        slug in that output that is not in the target set is **ignored**,
        so a hallucinated new slug or an edit to some other skill's line
        (cross-contamination) cannot leak into the store. Untargeted lines
        are left untouched.

        Cross-entry merge/retire — combining two skills' entries, minting a
        new skill, or deleting one during consolidation — is not supported;
        this method only ever rewrites the text of entries that already
        exist.
        """
        texts = [g.text for g in feedback]
        if name == INDEX:
            self._consolidate_index(texts, retrieved)
            return
        body, _ = self._recall(name)
        new_body = self._consolidate_fn(body, texts, self._get_description(name))
        self._save(name, new_body)

    def _consolidate_index(self, feedback: list[str], retrieved: dict[str, str] | None) -> None:
        """Consolidate helper for ``INDEX``: see ``_consolidate`` for the contract."""
        lines, _ = self._recall(INDEX)
        entries: dict[str, str] = {}
        for line in lines:  # pyright: ignore[reportAny]
            slug = line.partition(":")[0].strip()
            entries[slug] = line
        target_slugs = set(retrieved) if retrieved is not None else set(entries)
        target_lines = [entries[slug] for slug in sorted(entries) if slug in target_slugs]
        if not target_lines:
            return
        result = self._consolidate_fn("\n".join(target_lines), feedback, self._get_description(INDEX))
        new_lines = []
        for line in result.splitlines():
            line = line.strip()
            if not line:
                continue
            slug = line.partition(":")[0].strip()
            # Ignore any slug not in the target set: a hallucinated mint, or
            # an edit that drifted onto an untargeted/unknown skill.
            if slug in target_slugs:
                new_lines.append(line)
        if new_lines:
            self._save(INDEX, new_lines)

    def _delete(self, name: str) -> None:
        self._refuse_if_staged("delete")
        if name == INDEX:
            raise TypeError(f"Cannot delete '{INDEX}': it is not a single skill.")
        slug = self.slug_for(name)
        shutil.rmtree(self.root / slug)
        del self._param_to_slug[name]
        self._rebuild_schema()

    # -- Deleted-parameter graph reconstruction ----------------------------------
    #
    # `optimizer.build_graph_from_result` replays every `PARAMETER_RECALLED`
    # event on the graph against the *current* backend, including one
    # recorded for a skill that has since been `_delete`d. The base
    # `MemoryBackend.deserialize_value`/`_is_procedural` both resolve `name`
    # via `_resolve_field`, which raises `KeyError` for a name no longer on
    # `self.schema` — the two overrides below fall back to a schema-agnostic
    # result for such an unknown name instead, so reconstruction still
    # succeeds. Behavior for a name still on the schema is unchanged.

    def deserialize_value(self, name: str, raw: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Like the base implementation, but tolerant of a since-deleted parameter.

        Falls back to the raw payload, unvalidated, when ``name`` is no
        longer a schema field (see the section note above).
        """
        if name not in self.schema.model_fields:
            return raw
        return super().deserialize_value(name, raw)

    def _is_procedural(self, name: str) -> bool:
        """Like the base implementation, but ``False`` for a since-deleted parameter.

        See the section note above.
        """
        if name not in self.schema.model_fields:
            return False
        return super()._is_procedural(name)


def _replace_dir(src: Path, dest: Path) -> None:
    """Move ``src`` to ``dest``, replacing whatever ``dest`` held."""
    if dest.exists():
        shutil.rmtree(dest)
    src.rename(dest)


# ── LibraryStore: every file ``@skillify`` keeps for one decorated function ─
#
# Layout under ``root``::
#
#     CURRENT                  active version number ("0" or absent = empty)
#     versions/0003/<slug>/SKILL.md
#     versions.lock            held while CURRENT is read-then-written
#     records/calls.jsonl      one CallRecord per call, in call order
#     records/calls.lock       held exclusively while a record is appended,
#                               shared while records() reads
#     records/args/<id>.pkl    the call's (args, kwargs), for replay
#     records/events/<id>.json the call's event log, for mining
#     pending/<slug>/SKILL.md  a staged revision of an installed skill, gated
#                               (then removed) by the next round that reaches
#                               its gate
#     rounds/<k>/              a round's working files, removed when it ends
#     history.jsonl            one entry per round and per rollback
#     .lock                    held while a round runs
#
# A version directory is never modified after it is written. ``CURRENT``
# switches with an atomic rename under ``versions.lock``, so a call reading
# it sees either the old library or the new one, never a half-written one,
# and a publish and a rollback from two processes never race each other.

_EVENTS: TypeAdapter[list[Event]] = TypeAdapter(list[Event])


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CallRecord:
    """One recorded call of a decorated function.

    Attributes:
        call_id: Unique id; names the call's args and events files.
        seq: Position in call order, starting at 0.
        outcome: ``score`` or post-condition outcome in ``[0, 1]``;
            ``None`` when the call could not be scored.
        version: Library version the call ran with.
        skills_available: Slugs in the index the call was shown.
        skills_loaded: Slugs the agent loaded through the ``Skill`` tool.
        replayable: Whether the arguments were saved and can be replayed.
        error: Exception type name when the call raised, else ``None``.
        timestamp: ISO-8601 UTC time the record was written.
    """

    call_id: str
    seq: int
    outcome: float | None
    version: int
    skills_available: list[str]
    skills_loaded: list[str]
    replayable: bool
    error: str | None
    timestamp: str


class LibraryStore:
    """Versions, call records, round history and the round lock for one library.

    Args:
        root: The library directory; created if missing.
        keep_versions: Newest versions kept on disk after a publish. The
            version ``CURRENT`` names is never deleted.
    """

    def __init__(self, root: str | Path, *, keep_versions: int = 10) -> None:
        if keep_versions < 1:
            raise ValueError(f"keep_versions must be >= 1, got {keep_versions}")
        self.root = Path(root)
        self._keep = keep_versions
        for sub in ("versions", "records/args", "records/events", "rounds", "pending"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._calls = self.root / "records" / "calls.jsonl"
        self._history = self.root / "history.jsonl"
        self._current = self.root / "CURRENT"
        self._calls_lock = self.root / "records" / "calls.lock"
        self._versions_lock_path = self.root / "versions.lock"
        self._warned_unpicklable = False
        self._count_cache: tuple[int, int] | None = None

    # ── versions ──

    def current_version(self) -> int:
        """The active version; ``0`` is the empty library."""
        if not self._current.is_file():
            return 0
        return int(self._current.read_text().strip() or 0)

    def versions(self) -> list[int]:
        """Version numbers on disk, ascending."""
        out = []
        for path in (self.root / "versions").iterdir():
            if path.is_dir() and path.name.isdigit():
                out.append(int(path.name))
        return sorted(out)

    def version_dir(self, version: int) -> Path:
        """Directory of ``version`` (it may not exist)."""
        return self.root / "versions" / f"{version:04d}"

    def skill_dirs(self, version: int | None = None) -> list[Path]:
        """Skill directories of ``version`` (default: current), sorted by name."""
        version = self.current_version() if version is None else version
        if version == 0:
            return []
        directory = self.version_dir(version)
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.iterdir() if (p / "SKILL.md").is_file())

    @contextmanager
    def _version_lock(self) -> Iterator[None]:
        """Hold the exclusive lock guarding ``CURRENT``.

        ``publish``, ``publish_if_current`` and ``rollback`` all take this lock, so a
        version change from another process (e.g. a rollback) can never land between
        one of them reading ``current_version()`` and writing a new one.
        """
        handle = self._versions_lock_path.open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()

    def publish(self, skill_dirs: Sequence[Path]) -> int:
        """Write ``skill_dirs`` as a new version, switch ``CURRENT`` to it, prune old versions."""
        with self._version_lock():
            return self._publish_locked(skill_dirs)

    def publish_if_current(self, expected: int, skill_dirs: Sequence[Path]) -> int | None:
        """Publish ``skill_dirs`` only if ``current_version() == expected``.

        Checked and published under the same lock as ``publish`` and ``rollback``, so
        this never overwrites a version change (e.g. a rollback) that lands between a
        caller's own read of ``current_version()`` and this call. Writes nothing and
        returns ``None`` when ``expected`` is stale.
        """
        with self._version_lock():
            if self.current_version() != expected:
                return None
            return self._publish_locked(skill_dirs)

    def _publish_locked(self, skill_dirs: Sequence[Path]) -> int:
        new = max([*self.versions(), self.current_version()], default=0) + 1
        staging = self.root / "versions" / f".staging-{new}-{uuid.uuid4().hex[:8]}"
        staging.mkdir(parents=True)
        for directory in skill_dirs:
            shutil.copytree(Path(directory), staging / Path(directory).name)
        staging.rename(self.version_dir(new))
        self._set_current(new)
        skill_event("library.published", library=str(self.root.resolve()), version=new, skill_dirs=skill_dirs)
        self._prune()
        return new

    def rollback(self, to: int | None = None) -> int:
        """Switch ``CURRENT`` to ``to`` (default: the newest version older than current).

        Raises:
            ValueError: ``to`` is neither ``0`` nor a version on disk.
        """
        with self._version_lock():
            current = self.current_version()
            if to is None:
                older = [v for v in self.versions() if v < current]
                to = older[-1] if older else 0
            if to != 0 and to not in self.versions():
                raise ValueError(f"no version {to} on disk; available: {self.versions()}")
            self._set_current(to)
            self.append_history({"type": "rollback", "from": current, "to": to, "at": _now()})
            return to

    def _set_current(self, version: int) -> None:
        tmp = self.root / f"CURRENT.{uuid.uuid4().hex[:8]}"
        tmp.write_text(str(version))
        os.replace(tmp, self._current)

    def _prune(self) -> None:
        current = self.current_version()
        versions = self.versions()
        excess = len(versions) - self._keep
        for version in versions:
            if excess <= 0:
                break
            if version == current:
                continue
            shutil.rmtree(self.version_dir(version))
            excess -= 1

    # ── staged revisions ──

    @property
    def pending_dir(self) -> Path:
        """Where revisions of installed skills wait for the next round: ``<root>/pending``."""
        return self.root / "pending"

    def stage_revision(self, slug: str, skill_dir: Path) -> Path:
        """Copy ``skill_dir`` to ``pending/<slug>/``, replacing any revision staged earlier."""
        tmp = self.pending_dir / f".tmp-{slug}-{uuid.uuid4().hex[:8]}"
        shutil.copytree(Path(skill_dir), tmp)
        dest = self.pending_dir / slug
        _replace_dir(tmp, dest)
        skill_event("skill.staged", library=str(self.root.resolve()), skill_dirs=[dest])
        return dest

    def take_pending(self, dest: Path) -> list[tuple[str, Path]]:
        """Move every staged revision to ``dest/<slug>/``; ``(slug, path)`` each, by slug.

        ``pending/`` is empty afterwards. A round passes a directory inside its own.
        """
        dest_root = Path(dest)
        taken: list[tuple[str, Path]] = []
        for path in sorted(self.pending_dir.iterdir()):
            if path.name.startswith(".") or not (path / "SKILL.md").is_file():
                continue
            dest_root.mkdir(parents=True, exist_ok=True)
            target = dest_root / path.name
            if target.exists():
                shutil.rmtree(target)
            path.rename(target)
            taken.append((path.name, target))
        return taken

    # ── records ──

    def append_record(
        self,
        *,
        outcome: float | None,
        version: int,
        skills_available: Sequence[str],
        skills_loaded: Sequence[str],
        events: Sequence[Event] | None,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        error: str | None = None,
    ) -> CallRecord:
        """Persist one call. Unpicklable arguments make it non-replayable, not an error.

        Safe with several writers on one library: ``seq`` is assigned under a
        file lock from the records already written.
        """
        call_id = uuid.uuid4().hex
        replayable = True
        try:
            payload = pickle.dumps((tuple(args), dict(kwargs)))
        except Exception:  # noqa: BLE001 - any pickling failure means "cannot replay"
            replayable = False
            if not self._warned_unpicklable:
                _logger.warning("skillify: call arguments cannot be pickled; such calls are mined but never replayed")
                self._warned_unpicklable = True
        else:
            (self.root / "records" / "args" / f"{call_id}.pkl").write_bytes(payload)
        if events is not None:
            (self.root / "records" / "events" / f"{call_id}.json").write_bytes(_EVENTS.dump_json(list(events)))
        # ``seq`` is assigned under an exclusive lock from the lines already on
        # disk, not from a per-object counter: another store on this root (a
        # second process, or a second decoration of the same directory) may
        # have appended since this one was opened.
        with self._calls_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                record = CallRecord(
                    call_id=call_id,
                    seq=self._count_records(),
                    outcome=outcome,
                    version=version,
                    skills_available=list(skills_available),
                    skills_loaded=list(skills_loaded),
                    replayable=replayable,
                    error=error,
                    timestamp=_now(),
                )
                with self._calls.open("a") as handle:
                    handle.write(json.dumps(asdict(record)) + "\n")
                # We hold the lock and just wrote the line ourselves, so the new size
                # and count are known exactly -- no need to re-read what we wrote.
                self._count_cache = (self._calls.stat().st_size, record.seq + 1)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        skill_event("call.recorded", library=str(self.root.resolve()), record=record)
        return record

    def _count_records(self) -> int:
        """Number of records on disk. Called under ``self._calls_lock`` (``LOCK_EX``).

        Keeps a ``(file_size, line_count)`` cache: when the file's size matches the
        cache, its count is reused as-is; when the file only grew (another writer, or
        our own last append), only the bytes after the cached size are counted. A
        shrunk or missing file falls back to a full count.
        """
        if not self._calls.is_file():
            self._count_cache = None
            return 0
        size = self._calls.stat().st_size
        if self._count_cache is not None:
            cached_size, cached_count = self._count_cache
            if size == cached_size:
                return cached_count
            if size > cached_size:
                with self._calls.open("rb") as handle:
                    handle.seek(cached_size)
                    count = cached_count + sum(1 for line in handle if line.strip())
                self._count_cache = (size, count)
                return count
        with self._calls.open("rb") as handle:
            count = sum(1 for line in handle if line.strip())
        self._count_cache = (size, count)
        return count

    def records(self) -> list[CallRecord]:
        """Every record, in call order.

        Takes ``self._calls_lock`` with a shared lock while reading, so a call
        racing an in-progress ``append_record`` (which holds it exclusively) never
        sees a half-written last line.
        """
        with self._calls_lock.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            try:
                if not self._calls.is_file():
                    return []
                text = self._calls.read_text()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
        return [CallRecord(**json.loads(line)) for line in text.splitlines() if line.strip()]

    def load_args(self, call_id: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """The ``(args, kwargs)`` a replayable call was made with."""
        args, kwargs = pickle.loads((self.root / "records" / "args" / f"{call_id}.pkl").read_bytes())  # noqa: S301 - our own file
        return args, kwargs

    def load_events(self, call_id: str) -> list[Event] | None:
        """The call's event log, or ``None`` if none was saved."""
        path = self.root / "records" / "events" / f"{call_id}.json"
        if not path.is_file():
            return None
        return _EVENTS.validate_json(path.read_bytes())

    # ── history and rounds ──

    def append_history(self, entry: Mapping[str, Any]) -> None:
        """Append one JSON line to ``history.jsonl``."""
        with self._history.open("a") as handle:
            handle.write(json.dumps(dict(entry), default=str) + "\n")
        skill_event("history.appended", library=str(self.root.resolve()), entry=entry)

    def history(self) -> list[dict[str, Any]]:
        """Every history entry, oldest first."""
        if not self._history.is_file():
            return []
        return [json.loads(line) for line in self._history.read_text().splitlines() if line.strip()]

    def last_window_end(self) -> int:
        """Highest ``seq`` any round has consumed; ``-1`` before the first round."""
        ends = [int(e["window_end"]) for e in self.history() if e.get("type") == "round" and "window_end" in e]
        return max(ends, default=-1)

    def new_records(self) -> list[CallRecord]:
        """Records no round has consumed yet."""
        end = self.last_window_end()
        return [r for r in self.records() if r.seq > end]

    def next_round(self) -> int:
        """The number the next round will get, starting at 1."""
        return 1 + max((int(e["round"]) for e in self.history() if e.get("type") == "round"), default=0)

    def round_dir(self, round_number: int) -> Path:
        """Working directory for round ``round_number``."""
        return self.root / "rounds" / str(round_number)

    @contextmanager
    def round_lock(self) -> Iterator[bool]:
        """Hold the library's round lock; yields ``False`` if another holder has it."""
        handle = (self.root / ".lock").open("w")
        try:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


# ── The Skill tool: how an AI Function sees a skill library ────────────────
#
# An AI Function has no ``.claude/skills`` directory for a CLI to scan, so it
# is shown its skills the way Claude Code shows them: an **index** of every
# skill's name and one-line description, plus a ``Skill`` tool that returns
# one skill's full body. Nothing is preloaded; the agent decides whether a
# skill fits and when to read it.
#
# The tool is named ``Skill`` with a ``skill`` argument on purpose: that is
# Claude Code's own contract, so tooling that reads Claude Code's ``Skill``
# tool calls reads these runs unchanged.
#
# ``with_skills`` is the single place a with-skill variant of an AI Function
# is built. It refuses to build one that exposes nothing: a with-skill run
# given an empty library is a harness bug, and running it anyway would
# quietly compare no-skill against no-skill.


class SkillDeliveryError(RuntimeError):
    """A with-skill run was not given the skills it was supposed to get.

    Never retried: it means the harness is wrong, not that a service was
    busy, and retrying would turn a broken comparison into a slow one.
    """


class SkillToolbox:
    """The skill side of one with-skill run.

    Args:
        backend: The library this run may load from.

    Attributes:
        available: Every skill slug in the index, sorted.
        loaded: Slugs the agent loaded through the ``Skill`` tool, in call order.
    """

    def __init__(self, backend: SkillLibraryBackend) -> None:
        self._backend = backend
        self.loaded: list[str] = []
        self._descriptions: dict[str, str] = {}
        for param, model_field in backend.schema.model_fields.items():
            if param == INDEX:
                continue
            self._descriptions[backend.slug_for(param)] = " ".join((model_field.description or "").split())
        self.available: list[str] = sorted(self._descriptions)

    def index_text(self) -> str:
        """The prompt block listing every available skill."""
        lines = "\n".join(f"- {slug}: {self._descriptions[slug]}" for slug in self.available)
        return (
            "You have a library of skills learned from earlier runs of similar tasks. Each is a "
            "written procedure. None are loaded yet: if one looks relevant, call the `Skill` tool "
            "with its name to read it, then follow it where it applies. Ignore skills that do not fit "
            "this task.\n\nAvailable skills:\n" + lines
        )

    def make_tool(self) -> Any:  # noqa: ANN401 - strands DecoratedFunctionTool
        """A ``Skill`` tool bound to this library, recording every load."""
        backend, descriptions, loaded = self._backend, self._descriptions, self.loaded

        @tool(
            name="Skill",
            description=(
                "Load one skill from the library by name and return its full instructions. "
                "Call it with the exact name from the 'Available skills' list."
            ),
        )
        async def skill_tool(skill: str) -> str:
            name = skill.strip()
            if name not in descriptions:
                return f"No skill named {name!r}. Available: {', '.join(sorted(descriptions))}."
            loaded.append(name)
            view = await backend.recall(backend.param_for(name))
            return f"# {name}\n\n{view.value}"

        return skill_tool


def backend_from_dirs(dirs: Sequence[Path], root: Path, *, actor_id: str) -> SkillLibraryBackend | None:
    """Copy skill directories into ``root`` and open a backend over it.

    ``root`` is removed first if it exists, so the backend holds exactly
    ``dirs``: a root reused for another run never leaks that run's skills
    into this one.

    Returns ``None`` for an empty ``dirs`` -- "no skills", which the caller
    must be able to tell apart from "a library that happens to be empty".

    Raises:
        FileNotFoundError: A listed directory has no ``SKILL.md``.
    """
    dirs = [Path(d) for d in dirs]
    if not dirs:
        return None
    for directory in dirs:
        if not (directory / "SKILL.md").is_file():
            raise FileNotFoundError(f"skill directory {directory} has no SKILL.md")
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    for directory in dirs:
        shutil.copytree(directory, root / directory.name)
    return SkillLibraryBackend(root, actor_id=actor_id)


def with_skills[**P, T](
    fn: AIFunction[P, T], backend: SkillLibraryBackend | None
) -> tuple[AIFunction[P, T], SkillToolbox | None]:
    """Return ``fn`` with the skill index in its system prompt and the ``Skill`` tool added.

    ``backend=None`` returns ``fn`` itself: a no-skill call is exactly the
    undecorated function.

    Raises:
        SkillDeliveryError: ``backend`` holds no skills, or the index or the
            tool did not make it onto the variant.
    """
    if backend is None:
        return fn, None
    toolbox = SkillToolbox(backend)
    if not toolbox.available:
        raise SkillDeliveryError(f"skill library at {backend.root} has no skills; refusing a with-skill run")
    index = toolbox.index_text()
    base = fn.config.system_prompt
    system_prompt = index if not base else f"{base}\n\n{index}"
    variant = fn.replace(system_prompt=system_prompt, tools=(*fn.config.tools, toolbox.make_tool()))
    if index not in (variant.config.system_prompt or "") or len(variant.config.tools) != len(fn.config.tools) + 1:
        raise SkillDeliveryError(f"{fn.name}: skill index or Skill tool was not attached")
    return variant, toolbox
