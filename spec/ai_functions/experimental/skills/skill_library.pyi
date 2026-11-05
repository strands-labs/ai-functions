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
    files. This part is the on-disk layer only -- parse, load, write. It
    holds no policy about what a *good* skill looks like.

    Invariants:
        S1 — the *directory name* (the "slug") is a skill's identity to
        every consumer that indexes skills by directory, notably
        ``SkillLibraryBackend``. The frontmatter ``name`` field is only a
        label: ``Skill.name`` falls back to the directory name when it is
        absent, and the two are allowed to disagree. A caller that
        addresses skills by slug must therefore pass
        ``save_skill(..., dirname=slug)`` rather than let the write target
        be derived from ``name`` — see :func:`save_skill`.

        S2 — a round trip is body-and-frontmatter faithful, not
        byte-faithful. :func:`save_skill` re-emits the frontmatter through
        ``yaml.safe_dump`` (key order preserved, formatting not) and
        writes ``body.strip()``, so ``Skill.raw`` is authoritative only
        for a skill freshly loaded from disk.

        S3 — parsing never rejects a *missing* frontmatter block: text
        without a leading ``---``-delimited block is treated as all body
        with empty frontmatter. A block that is present but bad fails in
        one of two unrelated ways, and they are not one exception
        hierarchy: a syntactically malformed block raises
        ``yaml.YAMLError`` straight out of ``yaml.safe_load``, while a
        block that parses cleanly to something other than a mapping
        raises ``ValueError``. ``yaml.YAMLError`` is **not** a
        ``ValueError``, so a caller guarding only ``ValueError`` catches
        the rarer of the two.

  * **``SkillLibraryBackend``** -- the ``MemoryBackend`` that exposes a
    skill library for recall, search, save, delete, and consolidation.
    Because the schema *is* the directory listing, this backend is unusual
    among ``MemoryBackend`` implementations: the store is the filesystem,
    every write lands on disk immediately (``close()`` is a no-op), and
    the schema itself is rebuilt whenever the set of skill dirs changes.

    Invariants:
        B1 — ``INDEX`` (``"skill_index"``) is reserved. No skill may
        occupy it: the slug scan raises ``ValueError`` if any slug
        sanitizes to ``skill_index``, and likewise if two distinct slugs
        sanitize to the same parameter name. That scan re-runs on every
        schema rebuild (B3).

        B2 — the directory slug, not frontmatter ``name``, is the binding
        identity (S1). Every write goes through
        ``save_skill(..., dirname=slug)``.

        B3 — the schema is derived state. After any change to the set of
        skill dirs, or to a frontmatter ``description``, the slug map and
        ``self.schema`` are rebuilt from disk.

    Supported: recall, search (BM25 over the index), save (body or index
    descriptions), delete, and single-entry consolidation (rewriting one
    skill's body, or the index descriptions of a targeted subset of
    skills). Minting new skills (adding a slug that didn't exist at
    construction time) is not — saving an ``INDEX`` line for an unknown
    slug raises ``KeyError`` — and neither is cross-entry consolidation
    (merging or retiring skills). Creating skills is ``discovery``'s job,
    via ``save_skill`` plus a fresh backend.

  * **``LibraryStore``** -- every file ``@skillify`` keeps for one
    decorated function: versions, call records, round history and the
    round lock.

    Layout under ``root``::

        CURRENT                  active version number ("0" or absent = empty)
        versions/0003/<slug>/SKILL.md
        records/calls.jsonl      one CallRecord per call, in call order
        records/calls.lock       held while a record is appended
        records/args/<id>.pkl    the call's (args, kwargs), for replay
        records/events/<id>.json the call's event log, for mining
        rounds/<k>/              a round's working files, removed when it ends
        history.jsonl            one entry per round and per rollback
        .lock                    held while a round runs

    A version directory is never modified after it is written. ``CURRENT``
    switches with an atomic rename, so a call reading it sees either the
    old library or the new one, never a half-written one.

    Invariants:
        S1 — a version directory is never modified after ``publish``
        returns.
        S2 — ``CURRENT`` changes only by atomic rename under the version
        lock (``publish``, ``publish_if_current``, ``rollback``); the same
        lock makes the read-then-write in each of them atomic across
        processes, so a rollback can never be silently overwritten by a
        publish racing it (or vice versa).
        S3 — pruning never deletes the version ``CURRENT`` names.
        S4 — ``seq`` is dense and increasing across process restarts and
        across concurrent writers on one library.

  * **The ``Skill`` tool** -- how an AI Function sees a skill library: an
    index of every skill's name and one-line description, plus a
    ``Skill`` tool that returns one skill's full body.

    Invariants:
        T1 — ``with_skills(fn, None)`` returns ``fn`` itself.
        T2 — ``with_skills`` never returns a variant whose index is empty
        or whose ``Skill`` tool is missing; it raises
        ``SkillDeliveryError``.
        T3 — the ``Skill`` tool records a load only for a name in the
        index.
        T4 — ``backend_from_dirs(dirs, root)`` returns a backend holding
        exactly ``dirs``: whatever ``root`` held before is removed first.

See ``docs/skills.md``.
"""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_functions.ai_thread.ai_function import AIFunction
from ai_functions.types.events import Event

from ...memory.base import MemoryBackend

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

# ── Skill model ──────────────────────────────────────────────────────────────

@dataclass
class Skill:
    """A single skill loaded from disk.

    Mutable by design: ``SkillLibraryBackend`` rewrites ``body`` (or a
    ``frontmatter["description"]``) in place on a loaded instance and hands
    it straight back to :func:`save_skill`.

    Attributes:
        path: Path to the skill directory (containing SKILL.md).
        frontmatter: Parsed YAML frontmatter dict. Empty when the file has
            no frontmatter block (S3).
        body: Markdown body text (after the frontmatter block).
        raw: Full raw SKILL.md contents as read (S2).
        resources: Filenames (not paths) of the additional bundled files in
            the skill dir, sorted, excluding ``SKILL.md`` itself. Only
            immediate files are listed; subdirectories are not descended.
    """

    path: Path
    frontmatter: dict
    body: str
    raw: str
    resources: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The skill's name: the frontmatter ``name`` field, or the directory name.

        The fallback is what makes a frontmatter-less directory still
        addressable; it also means ``name`` is not a reliable key for a
        library whose identity is the slug (S1).
        """
        ...

    @property
    def description(self) -> str:
        """The skill's description, from the frontmatter ``description`` field.

        The empty string when the key is absent. This is the text
        ``SkillLibraryBackend`` publishes as the schema field description and
        as the skill's line in the ``skill_index``.
        """
        ...

    def section_headings(self) -> list[str]:
        """Return the body's level-2 ("## ...") heading texts, in order.

        Headings only — the surrounding text is not returned. Used by the
        Tier-1 structural metrics to check a skill against the expected
        section template.
        """
        ...

def load_skill(path: str | Path) -> Skill:
    """Load a single skill from a directory or a SKILL.md file.

    Args:
        path: Either the skill directory, or the ``SKILL.md`` file inside it
            (recognised by exact filename); in both cases ``Skill.path`` is
            set to the *directory*.

    Returns:
        The loaded skill, with ``resources`` populated from the sibling files.

    Raises:
        FileNotFoundError: ``path`` is neither a directory nor a ``SKILL.md``
            file, or the resolved directory contains no ``SKILL.md``.
        ValueError: The frontmatter block parses, but to something other than
            a YAML mapping (S3).
        yaml.YAMLError: The frontmatter block is not syntactically valid YAML
            — ``yaml.safe_load`` raises ``ParserError``/``ScannerError`` and
            nothing here converts it. Not a ``ValueError`` (S3), so it must be
            caught on its own.
    """
    ...

def save_skill(skill: Skill, root: str | Path, *, dirname: str | None = ..., bump_version: bool = ...) -> Path:
    """Write ``skill`` as ``<root>/<dirname>/SKILL.md`` (frontmatter + body).

    ``dirname`` defaults to ``skill.name`` (the frontmatter ``name`` field).
    Pass it explicitly to target a specific directory regardless of what the
    frontmatter says — e.g. a caller that indexes skills by directory slug
    must write back to that same slug even if frontmatter ``name`` disagrees
    with it, rather than silently creating a second directory (S1).

    Bundled ``resources`` are *not* written: this only ever creates or
    replaces ``SKILL.md``. Existing sibling files in the target directory are
    left untouched, and the directory (with parents) is created if absent.

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
            no ``version`` key, its value isn't a valid integer, or the
            block parses to a non-mapping — that last case is the
            ``ValueError`` of S3, and the only parse failure this function
            absorbs) — i.e. writing over an existing v1 skill produces v2,
            and so on. The bump reads the version from disk, never from
            ``skill.frontmatter``, so repeated saves of one in-memory
            ``Skill`` still advance monotonically. When ``False`` (the
            default) or the target does not yet exist, ``version`` is left
            exactly as ``skill.frontmatter`` provides it (possibly absent) —
            existing callers that never pass this keyword see no change in
            behavior.

    Returns:
        The skill's directory (``<root>/<dirname>``).

    Raises:
        yaml.YAMLError: ``bump_version=True`` and the *existing* file's
            frontmatter is syntactically malformed YAML (S3). Only S3's
            ``ValueError`` case is caught, so this does not degrade to
            ``version: 1`` — it aborts before anything is written, leaving
            the corrupt file in place. A caller overwriting a possibly
            corrupt skill must either guard this or save without the bump.
    """
    ...

@dataclass
class SkillLibrary:
    """A collection of skills loaded from a root directory.

    A snapshot, not a live view: the ``skills`` list reflects ``root`` as of
    the :func:`load_skill_library` call and does not track later writes.

    Attributes:
        root: The directory the skills were loaded from.
        skills: The loaded skills, in directory-name order.
    """

    root: Path
    skills: list[Skill] = field(default_factory=list)

    def __iter__(self) -> Iterator[Skill]:
        """Iterate over the library's skills, in ``skills`` order."""
        ...

    def __len__(self) -> int:
        """Return the number of skills in the library."""
        ...

    def by_name(self, name: str) -> Skill | None:
        """Return the skill named ``name``, or ``None`` if not found.

        Matches on ``Skill.name`` (frontmatter ``name``, else directory name),
        so this is *not* a slug lookup when the two disagree (S1). The first
        match wins if two skills share a name.
        """
        ...

    def names(self) -> list[str]:
        """Return the names of every skill in the library, in ``skills`` order."""
        ...

def load_skill_library(root: str | Path) -> SkillLibrary:
    """Load every skill directory under ``root``.

    A skill directory is any *immediate* subdirectory containing a SKILL.md;
    the scan is one level deep and never recurses, so nested skill trees are
    not discovered. Directories without a ``SKILL.md`` are skipped silently.
    An empty (or entirely skill-less) ``root`` yields a library of length 0
    rather than an error.

    Args:
        root: Directory holding the skill directories.

    Returns:
        The library, its ``skills`` sorted by directory name.

    Raises:
        NotADirectoryError: ``root`` is not an existing directory.
        FileNotFoundError: A discovered skill directory could not be loaded.
        ValueError: A discovered SKILL.md has frontmatter that parses to a
            non-mapping (S3).
        yaml.YAMLError: A discovered SKILL.md has a syntactically malformed
            frontmatter block (S3). Not a ``ValueError``: a caller loading an
            untrusted library must guard both, and either way the load fails
            whole — there is no per-skill skip.
    """
    ...

# ── SkillLibraryBackend ──────────────────────────────────────────────────────

INDEX: str
"""Name of the reserved synthetic list parameter (``"skill_index"``).

Its value is one ``"<slug>: <description>"`` line per skill, in slug order —
the entry point an agent searches to decide *which* skill to recall. Saving
to it rewrites the corresponding skills' frontmatter descriptions; it cannot
be deleted, and it is the only parameter ``search`` accepts (B1).
"""

class SkillLibraryBackend(MemoryBackend):
    """Memory backend over a directory of skills.

    Each skill directory under ``root`` (any immediate subdirectory containing
    a ``SKILL.md``) becomes a scalar parameter named after its sanitized slug,
    whose value is the skill's Markdown body. A synthetic ``skill_index`` list
    parameter carries one ``"<slug>: <description>"`` line per skill, searchable
    by BM25 — the entry point for discovering which skill to recall.

    Behaviour of the inherited public API, over and above
    :class:`~ai_functions.memory.base.MemoryBackend`:

    * ``recall(INDEX)`` returns the index lines (slug order); ``recall(param)``
      returns that skill's Markdown body, read fresh from disk.
    * ``search(INDEX, query, k=...)`` returns the top-``k`` index lines ranked
      by BM25, with ``{"results": {slug: line}}`` as the fetch meta — the
      ``{entry_id: value}`` shape the optimizer merges into the recall event
      so consolidation can target exactly the entries the forward pass
      retrieved. Ranking uses ``BM25Plus`` rather than ``JSONMemoryBackend``'s
      ``BM25Okapi``: Okapi's IDF is exactly zero for a term appearing in one
      of two documents, producing hard ties on the small libraries this
      backend typically holds. Searching any other parameter raises
      ``TypeError`` — every other parameter is a scalar skill body.
    * ``save(param, body)`` rewrites that skill's body, preserving its
      frontmatter. ``save(INDEX, lines)`` reads each ``"slug: description"``
      line and writes the description into that skill's frontmatter when it
      differs, then rebuilds the schema (B3); an unknown slug raises
      ``KeyError``.
    * ``delete(param)`` removes the skill *directory* outright (``rmtree``) —
      not a reset to a schema default, and not recoverable.
      ``delete(INDEX)`` raises ``TypeError``.
    * ``consolidate(param, feedback)`` rewrites a body wholesale through
      ``consolidate_fn``. ``consolidate(INDEX, feedback, retrieved)`` is
      scoped: only lines whose slug appears in ``retrieved`` (all lines when
      ``retrieved`` is ``None``) are rewritten, and any slug in the model's
      output that was not in that target set is discarded — a hallucinated
      new skill, or an edit that drifted onto an untargeted line, cannot
      reach the store. Untargeted lines are left byte-for-byte alone.
    * ``deserialize_value`` and procedural detection tolerate a parameter
      that has since been deleted: ``build_graph_from_result`` replays every
      recorded recall against the *current* backend, including one for a
      skill removed in the meantime, and a graph reconstruction must not
      fail because of it.

    Args:
        root: Directory holding the skill directories. Scanned at
            construction; must exist.
        actor_id: Namespaces this actor's values within the backend, as for
            any ``MemoryBackend``.
        consolidate_fn: Injection seam for the rewriting step, called as
            ``consolidate_fn(value, feedback_texts, description) -> str`` for
            both a skill body and the joined index lines. ``None`` uses the
            default, which wires the same scalar-consolidation prompt
            ``JSONMemoryBackend`` uses, adapted to this plain-callable shape
            so a test fake and the real thing share one call signature.
        query_fn: Injection seam for ``query``; an object with a
            ``run_sync(value=..., query=...) -> str`` method (i.e. an
            ``AIFunction``). ``None`` uses the shared default query function.
        stage_root: When set, rewrites -- ``save`` of a skill body or of
            index descriptions, and so ``consolidate`` -- are staged: the
            edited skill goes to ``<stage_root>/<slug>/``, replacing any
            earlier staged copy. Reads still come from ``root``. ``delete`` raises
            ``TypeError``, so ``root`` is never written. A staged ``INDEX`` save
            stages only the skills whose description it changes. ``@skillify`` passes the library's ``pending/``
            directory, so a rewrite waits for the next round's gate.

    Raises:
        ValueError: A slug sanitizes to the reserved ``INDEX`` name, or two
            slugs collide after sanitization (B1). Raised by the scan, not by
            construction as such — the same check re-runs on every rebuild,
            so it can also surface later from ``save(INDEX, ...)`` or
            ``delete``.
    """

    root: Path
    stage_root: Path | None

    def __init__(
        self,
        root: str | Path,
        actor_id: str,
        *,
        consolidate_fn: Any = ...,
        query_fn: Any = ...,
        stage_root: str | Path | None = ...,
    ) -> None:
        """Scan ``root`` for skill dirs and build the schema over them.

        Ensures:
            - ``self.schema`` has exactly one ``list[str]`` field named
              ``INDEX`` plus one ``str`` field per discovered skill, each
              described by that skill's frontmatter ``description``.
            - ``param_for`` / ``slug_for`` round-trip for every discovered
              skill.
        """
        ...

    def param_for(self, slug: str) -> str:
        """Return the schema parameter name for a skill slug.

        A pure string transform (``-`` → ``_``) with no existence check: it
        succeeds for a slug that has no skill (B1). Callers verifying a slug
        must follow up with :meth:`slug_for` (guarding ``KeyError``) rather
        than trust this result — checking ``self.schema.model_fields`` alone
        is not enough either, since ``skill-index`` sanitizes onto the
        reserved ``INDEX`` field, which is a schema field with no slug.
        """
        ...

    def slug_for(self, param: str) -> str:
        """Return the skill slug for a schema parameter name.

        Raises:
            KeyError: ``param`` is not a known skill parameter — including
                ``INDEX``, which is synthetic and maps to no skill.
        """
        ...

    def close(self) -> None:
        """No-op: skills are written straight to disk on every save or delete."""
        ...

    def deserialize_value(self, name: str, raw: Any) -> Any:
        """Like the base implementation, but tolerant of a since-deleted parameter.

        Falls back to returning ``raw`` unvalidated when ``name`` is no longer
        a field on ``self.schema``, so replaying a recall event recorded
        before a ``delete`` still reconstructs the graph. Behaviour for a
        live parameter is unchanged.
        """
        ...

# ── LibraryStore ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CallRecord:
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
    root: Path
    def __init__(self, root: str | Path, *, keep_versions: int = 10) -> None: ...
    def current_version(self) -> int: ...
    def versions(self) -> list[int]: ...
    def version_dir(self, version: int) -> Path: ...
    def skill_dirs(self, version: int | None = None) -> list[Path]: ...
    def publish(self, skill_dirs: Sequence[Path]) -> int: ...
    def publish_if_current(self, expected: int, skill_dirs: Sequence[Path]) -> int | None: ...
    def rollback(self, to: int | None = None) -> int: ...
    @property
    def pending_dir(self) -> Path:
        """Where revisions of installed skills wait for the next round: ``<root>/pending``."""
    def stage_revision(self, slug: str, skill_dir: Path) -> Path:
        """Copy ``skill_dir`` to ``pending/<slug>/``, replacing any revision staged earlier; returns that path."""
    def take_pending(self, dest: Path) -> list[tuple[str, Path]]:
        """Move every staged revision to ``dest/<slug>/``; ``(slug, path)`` each, sorted by slug.

        Ensures:
            - ``pending/`` holds no revision afterwards.
        """
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
    ) -> CallRecord: ...
    def records(self) -> list[CallRecord]: ...
    def load_args(self, call_id: str) -> tuple[tuple[Any, ...], dict[str, Any]]: ...
    def load_events(self, call_id: str) -> list[Event] | None: ...
    def append_history(self, entry: Mapping[str, Any]) -> None: ...
    def history(self) -> list[dict[str, Any]]: ...
    def last_window_end(self) -> int: ...
    def new_records(self) -> list[CallRecord]: ...
    def round_dir(self, round_number: int) -> Path: ...
    def next_round(self) -> int: ...
    @contextmanager
    def round_lock(self) -> Iterator[bool]: ...

# ── The Skill tool ───────────────────────────────────────────────────────────

class SkillDeliveryError(RuntimeError): ...

class SkillToolbox:
    loaded: list[str]
    available: list[str]
    def __init__(self, backend: SkillLibraryBackend) -> None: ...
    def index_text(self) -> str: ...
    def make_tool(self) -> Any: ...

def backend_from_dirs(dirs: Sequence[Path], root: Path, *, actor_id: str) -> SkillLibraryBackend | None: ...
def with_skills[**P, T](
    fn: AIFunction[P, T], backend: SkillLibraryBackend | None
) -> tuple[AIFunction[P, T], SkillToolbox | None]: ...
