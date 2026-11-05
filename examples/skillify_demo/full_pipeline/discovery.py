"""The demo's full discovery workflow and SKILL.md formatting.

FullDiscoveryMiner adapts scored calls to the five-stage pipeline selected by
DiscoveryPreset: segmentation, extraction, grouping, merging, and formatting.
Mining lives in mining.py; this module renders typed formatting results into
validated SKILL.md procedures and publishes the discovery output to disk.

None disables a preset's segmenter, grouper, or merge gate. None on an
extract/merge/format function selects the model-backed default. Construction
makes no model calls. Non-strict formatting drops failed format checks,
refusals, and duplicate slugs; invalid structured output and the reserved
skill-index slug always raise. Surviving skills are saved and reloaded, with
on-disk versions bumped when a later discovery run reproduces a slug.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ai_functions.ai_thread import ai_function
from ai_functions.ai_thread.postcondition import PostConditionResult
from ai_functions.experimental.skills import skill_event
from ai_functions.experimental.skills.learner import RecordedCall
from ai_functions.experimental.skills.skill_library import Skill, load_skill, save_skill

from .mining import (
    UNGROUPED_LABEL,
    Episode,
    EpisodeBuilder,
    ExtractFn,
    Grouper,
    LLMThematicGrouper,
    MergeFn,
    MergeGate,
    MetricGate,
    SkillDraft,
    SkillGroup,
    Subtask,
    Trajectory,
    extract_skills,
    merge_group,
    trajectories_from_events,
)
from .quality import ExpectedSkillFormat, FormatAlignmentMetric, Level1Input, MetricResult

# --- Format drafts as SKILL.md procedures ---

_NAME_RE_SUB = "abcdefghijklmnopqrstuvwxyz0123456789-"
_NAME_MAX_LEN = 64
"""Longest slug ``SkillFrontmatterSpec.name_pattern`` accepts (``quality.py``)."""


class FormattedSkillCandidate(BaseModel):
    """Typed LLM output: name/description plus one field per body section.

    Mirrors ``quality.py``'s ``SkillFrontmatterSpec`` (``name``,
    ``description``) and ``ExpectedSkillFormat`` (the three required
    sections as required fields; the three optional sections as optional
    fields) -- see this module's docstring for why a structured candidate,
    rather than one free-text body, is what guarantees heading presence.
    """

    name: str = Field(description="Short skill name (will be slugified to a valid frontmatter `name`).")
    description: str = Field(description="One-line skill description.")
    overview: str = Field(description="Body content for the 'Overview' section.")
    when_to_use: str = Field(description="Body content for the 'When to use this skill' section.")
    instructions: str = Field(description="Body content for the 'Instructions' section.")
    examples: str = Field(default="", description="Optional body content for an 'Examples' section.")
    resources: str = Field(default="", description="Optional body content for a 'Resources' section.")
    notes: str = Field(default="", description="Optional body content for a 'Notes' section.")


def _check_formatted_candidate(result: FormattedSkillCandidate) -> PostConditionResult:
    """Post-condition / defensive re-check: every required field carries real content.

    Wired onto the real template via ``ThreadConfig.post_conditions``
    (mirrors ``mining.py``'s ``_check_extracted_skill``,
    ``mining.py``'s ``_check_merged_candidate``), and re-run inside
    :func:`format_skill` against *any* ``format_fn`` result (real or fake)
    -- see this module's docstring on defensive validation. Only the three
    required sections plus name/description are checked; ``examples``/
    ``resources``/``notes`` are optional and never blank-checked.
    """
    required = (
        ("name", result.name),
        ("description", result.description),
        ("overview", result.overview),
        ("when_to_use", result.when_to_use),
        ("instructions", result.instructions),
    )
    blank = [field_name for field_name, value in required if not value.strip()]
    if blank:
        return PostConditionResult(
            passed=False,
            message=f"formatted skill candidate has blank field(s): {blank}",
        )
    return PostConditionResult(passed=True)


def _render_draft(draft: SkillDraft) -> str:
    """Render a ``SkillDraft`` as text for a formatting prompt."""
    return f"{draft.name}: {draft.description}\n\n{draft.body}"


def _render_guidance(draft_text: str, guidance: str | None) -> str:
    """Append caller-supplied formatting ``guidance`` to ``draft_text``.

    ``None``/blank returns ``draft_text`` unchanged. The extra context rides
    inside the one string argument :data:`FormatFn` already takes (same
    device as ``mining.py``'s ``_render_retry_input``), so the injection
    seam's signature is unchanged and an injected fake sees the guidance
    exactly as the real template does.
    """
    if guidance is None or not guidance.strip():
        return draft_text
    return f"{draft_text}\n\n## Additional formatting guidance\n{guidance.strip()}"


_EXPECTED_FORMAT = ExpectedSkillFormat()


@ai_function[FormattedSkillCandidate](post_conditions=[_check_formatted_candidate])
def _format_template(draft_text: str) -> str:
    """Build the prompt asking for a structured, format-conformant skill candidate."""
    return (
        "You are formatting a candidate SKILL for a shared skill library: a "
        "reusable, self-contained instruction another agent can follow.\n\n"
        "## Task\n"
        "Read the candidate skill below and produce a polished name, a "
        "one-line description, and content for each expected section, "
        "conforming to the template below.\n\n"
        "## Template\n"
        f"{_EXPECTED_FORMAT.as_prompt_block()}\n\n"
        f"## Candidate skill\n{draft_text}"
    )


FormatFn = Callable[[str], Awaitable[FormattedSkillCandidate]]
"""Call shape shared by the real template and any injected fake.

``(draft_text) -> FormattedSkillCandidate``, ``async``. See this module's
docstring on the injection seam and its defensive-validation consequence.
"""


async def _default_format(draft_text: str) -> FormattedSkillCandidate:
    """Default ``format_fn``: dispatch to the real module-level ``@ai_function`` template."""
    return await _format_template(draft_text)


class FormatAlignmentError(ValueError):
    """Raised by :func:`format_skill` when the rendered candidate fails ``format_alignment``.

    See this module's docstring ("The outbound gate -- mechanism and
    rationale") for why this is a hard failure rather than a retry or a
    silently-attached result. The message itself names the failing
    candidate's provenance (trajectory/constituent ids) -- matching the
    traceability bar ``mining.py``'s/``mining.py``'s own ``ValueError``
    messages already set (episode id, group label) -- so a caller reading
    just the exception text, without reaching for ``.draft``, can already
    tell which source material produced the rejected skill.

    Attributes:
        result: The failing ``FormatAlignmentMetric`` ``MetricResult`` --
            ``result.evidence`` names the concrete violation(s).
        draft: The ``SkillDraft`` that produced the failing candidate --
            ``draft.provenance`` is the same provenance folded into the
            message text, available here structurally for a caller that
            wants it without re-parsing the message.
        skill: The ``Skill`` that was rendered but failed the gate (its
            frontmatter/body are otherwise complete -- only
            ``format_alignment`` rejected it). Useful to a caller that
            wants to log or inspect the near-miss (e.g. its slug via
            ``skill.name``) without reformatting.
    """

    def __init__(self, message: str, *, result: MetricResult, draft: SkillDraft, skill: Skill) -> None:
        super().__init__(message)
        self.result = result
        self.draft = draft
        self.skill = skill


class SkillRefusalError(FormatAlignmentError):
    """Raised by :func:`format_skill` when the formatter's output is a refusal, not a skill.

    A formatting model that declines a draft does not necessarily fail
    structurally: it can fill every field of a well-formed candidate with
    its refusal. That candidate passes ``format_alignment`` -- every heading
    is present -- and then competes for retrieval like any real skill. A
    typical trigger is a task that legitimately allows something the
    formatter reads as unsafe (combusting a compound, say); the refusal can
    then be the most relevant-looking skill for exactly that task, telling
    every run not to do a legitimate part of it.

    Subclasses :class:`FormatAlignmentError` deliberately, so
    ``discover_skills(strict_format=False)`` drops a refusal with the same
    logged warning as any other non-conforming candidate, and
    ``strict_format=True`` still aborts on it.
    """


# Phrasings that only make sense when the text is talking about *itself* --
# the request, the candidate, the formatter -- rather than giving an
# instruction. Keyed that way on purpose: ordinary procedures say "dangerous"
# and "never" constantly ("never put plastic on an active stove"), so those
# words alone cannot mark a refusal.
_REFUSAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(this|the)\s+(skill|candidate|request|procedure|draft)\s+(is|was|has been)\s+refused\b",
        r"\b(this|the)\s+(skill|candidate|draft)\s+(cannot|can't|should not|must not)\s+be\s+"
        r"(formatted|added|created|provided|included)\b",
        r"\b(this|the)\s+skill\s+should\s+never\s+be\s+used\b",
        r"\bI\s+(cannot|can't|won't|will\s+not|am\s+unable\s+to|'m\s+unable\s+to|am\s+not\s+able\s+to)\s+"
        r"(help|assist|comply|provide|create|format|write)\b",
        r"\bI\s+(must|have\s+to)\s+(decline|refuse)\b",
    )
)


def _refusal_evidence(candidate: FormattedSkillCandidate) -> list[str]:
    """Every field/phrase pair where ``candidate`` reads as a refusal. Empty when it does not."""
    evidence: list[str] = []
    for field_name in ("name", "description", "overview", "when_to_use", "instructions", "examples", "notes"):
        text = getattr(candidate, field_name)
        for pattern in _REFUSAL_PATTERNS:
            match = pattern.search(text)
            if match:
                evidence.append(f"{field_name}: {match.group(0)!r}")
    return evidence


def _slugify_name(name: str) -> str:
    """Best-effort slug matching ``SkillFrontmatterSpec.name_pattern`` (``quality.py``).

    Same rule as ``mining.py``'s own ``_slugify_name`` (deliberately
    duplicated rather than imported -- a small, pure, module-local utility
    each formatting-adjacent module owns independently, matching this
    package's existing precedent of not reaching into another module's
    private helpers).

    The pattern caps a name at 64 characters (``[a-z0-9][a-z0-9-]{0,63}``),
    so a long draft name is truncated to :data:`_NAME_MAX_LEN` and any
    hyphen left at the truncation boundary is stripped -- otherwise a
    verbose draft name would produce a slug that silently fails
    ``FormatAlignmentMetric``'s frontmatter check (and, in
    :func:`format_skill`, raises :class:`FormatAlignmentError`).
    """
    lowered = name.strip().lower()
    slug = "".join(ch if ch in _NAME_RE_SUB else "-" for ch in lowered).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug[:_NAME_MAX_LEN].rstrip("-")
    return slug or "skill"


def _render_body(candidate: FormattedSkillCandidate) -> str:
    """Render ``candidate``'s sections as Markdown, guaranteeing every expected heading.

    The three expected headings (``Overview``, ``When to use this skill``,
    ``Instructions``) are always emitted, regardless of content -- this is
    what lets :mod:`format`'s outbound gate rely on section *presence* being
    unconditionally satisfied (see module docstring). The three optional
    headings are only emitted when the corresponding field is non-blank.
    """
    parts = [
        f"## Overview\n{candidate.overview.strip()}",
        f"## When to use this skill\n{candidate.when_to_use.strip()}",
        f"## Instructions\n{candidate.instructions.strip()}",
    ]
    if candidate.examples.strip():
        parts.append(f"## Examples\n{candidate.examples.strip()}")
    if candidate.resources.strip():
        parts.append(f"## Resources\n{candidate.resources.strip()}")
    if candidate.notes.strip():
        parts.append(f"## Notes\n{candidate.notes.strip()}")
    return "\n\n".join(parts) + "\n"


def _metadata_from_provenance(draft: SkillDraft) -> dict[str, object]:
    """Build the frontmatter ``metadata`` mapping from ``draft.provenance``."""
    prov = draft.provenance
    return {
        "trajectory_ids": list(prov.trajectory_ids),
        "constituent_ids": list(prov.constituent_ids),
        "strategy": prov.strategy,
        "group_label": prov.group_label,
    }


def _skill_from_candidate(candidate: FormattedSkillCandidate, draft: SkillDraft) -> Skill:
    """Build an in-memory ``Skill`` (no disk I/O) from a formatted candidate + its source draft."""
    slug = _slugify_name(candidate.name)
    body = _render_body(candidate)
    frontmatter: dict[str, object] = {
        "name": slug,
        "description": candidate.description.strip(),
        "version": 1,
        "metadata": _metadata_from_provenance(draft),
    }
    front = yaml.safe_dump(frontmatter, sort_keys=False).strip()
    raw = f"---\n{front}\n---\n\n{body.strip()}\n"
    return Skill(
        path=Path(f"<format-candidate:{slug}>"),
        frontmatter=frontmatter,
        body=body,
        raw=raw,
        resources=[],
    )


async def format_skill(
    draft: SkillDraft,
    *,
    format_fn: FormatFn | None = None,
    guidance: str | None = None,
) -> Skill:
    """Render ``draft`` into a disk-ready ``Skill``, gated by ``format_alignment``.

    Args:
        draft: The candidate skill draft to format (typically the output of
            extraction or merging).
        format_fn: Formatting callable; ``None`` (default) dispatches to the
            real module-level ``@ai_function`` template (:func:`_default_format`).
            Inject a fake matching :data:`FormatFn` in tests -- see this
            module's docstring on the injection seam.
        guidance: Optional extra formatting instructions (house style, a
            template-shaped hint, ...) appended to the text handed to
            ``format_fn`` -- reachable from a preset as
            ``DiscoveryPreset.format_guidance`` (see the package README).
            ``None``/blank (default) leaves the prompt input untouched.

    Returns:
        A ``Skill`` (no disk I/O -- ``path`` is a synthetic placeholder;
        callers that persist it use ``skill.py``'s ``save_skill``) whose
        frontmatter carries ``name``/``description``/``version``/``metadata``
        (the latter populated from ``draft.provenance``) and whose body
        has every expected H2 section, guaranteed to pass
        ``FormatAlignmentMetric`` -- see this module's docstring on the
        outbound gate.

    Raises:
        ValueError: ``format_fn`` (real or fake) returned a
            ``FormattedSkillCandidate`` failing
            :func:`_check_formatted_candidate` (defense-in-depth; see this
            module's docstring).
        SkillRefusalError: The formatter's output is a refusal rather than a
            procedure (a :class:`FormatAlignmentError` subclass, so it is
            handled wherever that is). Checked before ``format_alignment``,
            because a refusal can be structurally perfect.
        FormatAlignmentError: The rendered candidate fails
            ``FormatAlignmentMetric`` -- see this module's docstring on the
            outbound gate mechanism. ``exc.result``/``exc.draft``/
            ``exc.skill`` carry the failing ``MetricResult``, source draft,
            and rendered (but rejected) ``Skill`` respectively; the message
            itself already names the draft's provenance.
    """
    fn = format_fn if format_fn is not None else _default_format
    draft_text = _render_guidance(_render_draft(draft), guidance)
    candidate = await fn(draft_text)
    check = _check_formatted_candidate(candidate)
    if check is not None and not check.passed:
        raise ValueError(f"format_fn returned an invalid FormattedSkillCandidate: {check.message}")

    skill = _skill_from_candidate(candidate, draft)

    refusal = _refusal_evidence(candidate)
    if refusal:
        prov = draft.provenance
        raise SkillRefusalError(
            f"formatted skill {skill.name!r} (trajectory_ids={prov.trajectory_ids!r}, "
            f"constituent_ids={prov.constituent_ids!r}) is a refusal, not a skill: {refusal[0]}",
            result=MetricResult(
                score=0.0,
                passed=False,
                rationale="formatter output reads as a refusal rather than a procedure",
                evidence=refusal,
            ),
            draft=draft,
            skill=skill,
        )

    result = FormatAlignmentMetric().compute(Level1Input(skill_spec=skill))
    if not result.passed:
        prov = draft.provenance
        raise FormatAlignmentError(
            f"formatted skill {skill.name!r} (trajectory_ids={prov.trajectory_ids!r}, "
            f"constituent_ids={prov.constituent_ids!r}) failed format_alignment: {result.rationale}",
            result=result,
            draft=draft,
            skill=skill,
        )
    return skill


# --- Configure and run discovery ---

logger = logging.getLogger(__name__)

_INDEX = "skill_index"
"""The parameter name ``SkillLibraryBackend`` reserves for its skill index; no slug may sanitize to it."""


@dataclass(frozen=True, slots=True)
class DiscoveryPreset:
    """Configure each phase of :func:`discover_skills`.

    Args:
        episode_builder: Segments each trajectory into episodes. ``None``
            skips segmentation (whole-trajectory episodes, no LLM call).
        modes: Extraction modes forwarded to ``extract_skills``. Not a skip
            seam -- must be a non-empty subset of ``mining.MODES``.
        extract_fn: Forwarded to ``extract_skills``. ``None`` uses the real
            per-mode templates.
        grouper: Clusters drafts before merging. ``None`` skips grouping
            (every draft its own singleton group).
        gate: Forwarded to ``merge_group`` for every non-trivial group.
            ``None`` skips merging entirely.
        merge_fn: Forwarded to ``merge_group``. ``None`` uses the real merge
            template. Irrelevant when ``gate`` is ``None`` (merging never
            runs).
        format_fn: Forwarded to ``format_skill``. ``None`` uses the real
            format template.
        format_guidance: Extra instructions appended to the formatter's input.
            Forwarded as ``format_skill(..., guidance=...)``; ``None`` leaves
            the formatter prompt untouched.
    """

    episode_builder: EpisodeBuilder | None
    modes: tuple[str, ...]
    extract_fn: ExtractFn | None = None
    grouper: Grouper | None = None
    gate: MergeGate | None = None
    merge_fn: MergeFn | None = None
    format_fn: FormatFn | None = None
    format_guidance: str | None = None


DEFAULT_PRESET: DiscoveryPreset = DiscoveryPreset(
    episode_builder=Subtask(),
    # "repair" is in the default modes so that `failed_trajectory_ids` --
    # whose only purpose is to enable it -- is not a silent no-op. Including
    # it costs nothing on a run with no failure labels: `extract_skills`
    # gates repair per episode on `episode.trajectory_id in
    # failed_trajectory_ids`, so with the default empty set it runs for zero
    # episodes and makes zero model calls.
    modes=("preserve", "prune", "repair"),
    extract_fn=None,
    grouper=LLMThematicGrouper(),
    gate=MetricGate(),
    merge_fn=None,
    format_fn=None,
)
"""Subtask segmentation -> preserve/prune/repair extraction -> thematic grouping -> gated merging -> formatting.

Every seam left at its own real-template default (``extract_fn``/
``merge_fn``/``format_fn`` all ``None``); ``gate=MetricGate()`` uses a lazy
default judge and an empty comparison library (see ``mining.py``'s
``MetricGate`` docstring) -- constructing this preset does no I/O and
reaches no LLM/network until ``discover_skills`` actually runs a phase that
needs one.
"""


async def _build_episodes(trajectories: Sequence[Trajectory], episode_builder: EpisodeBuilder | None) -> list[Episode]:
    """Run phase 1 (episode building), honoring the ``episode_builder=None`` skip.

    With ``None``, each entire trajectory becomes one episode.
    """
    episodes: list[Episode] = []
    for trajectory in trajectories:
        if episode_builder is None:
            if trajectory.messages:
                episodes.append(Episode(trajectory_id=trajectory.id, start=0, end=len(trajectory.messages)))
            continue
        episodes.extend(await episode_builder.build(trajectory))
    return episodes


def _singleton_groups(drafts: list[SkillDraft]) -> list[SkillGroup]:
    """Wrap every draft in its own single-member group (the ``grouper=None`` skip)."""
    return [SkillGroup(label=UNGROUPED_LABEL, members=[draft]) for draft in drafts]


async def _merge_groups(groups: list[SkillGroup], gate: MergeGate | None, merge_fn: MergeFn | None) -> list[SkillDraft]:
    """Run phase 4 (merging), honoring the ``gate=None`` skip.

    With no merge gate, every draft passes through unchanged.
    """
    if gate is None:
        return [member for group in groups for member in group.members]
    merged: list[SkillDraft] = []
    for group in groups:
        merged.extend(await merge_group(group, gate=gate, merge_fn=merge_fn))
    return merged


def _provenance_str(draft: SkillDraft) -> str:
    prov = draft.provenance
    return f"trajectory_ids={prov.trajectory_ids!r}, constituent_ids={prov.constituent_ids!r}"


async def _format_drafts(
    drafts: list[SkillDraft],
    format_fn: FormatFn | None,
    *,
    strict_format: bool,
    guidance: str | None = None,
) -> list[tuple[SkillDraft, Skill]]:
    """Run phase 5 (formatting) over every draft, honoring ``strict_format``.

    Strict mode raises on a failed format check; non-strict mode logs and drops it.
    A defensive ``ValueError`` (contract-violating ``format_fn`` output --
    ``discovery.py``'s ``_check_formatted_candidate``) always propagates,
    regardless of ``strict_format`` -- it signals a broken ``format_fn``,
    not an ordinary content shortfall.

    Returns:
        One ``(draft, skill)`` pair per draft that formatted successfully,
        in ``drafts`` order -- a draft dropped under non-strict mode
        contributes nothing. The source ``draft`` is carried alongside its
        ``skill`` so a later phase (the slug-collision check) can report
        provenance without re-deriving it from the ``Skill``.
    """
    pairs: list[tuple[SkillDraft, Skill]] = []
    for draft in drafts:
        try:
            skill = await format_skill(draft, format_fn=format_fn, guidance=guidance)
        except FormatAlignmentError as exc:
            skill_event("candidate.excluded", skill=exc.skill, stage="format", reason=exc.result.rationale)
            if strict_format:
                raise
            logger.warning(
                "discover_skills: dropping draft %r (%s) - failed format_alignment: %s",
                exc.skill.name,
                _provenance_str(draft),
                exc.result.rationale,
            )
            continue
        pairs.append((draft, skill))
    return pairs


def _check_no_slug_collisions(
    pairs: list[tuple[SkillDraft, Skill]], *, strict: bool = True
) -> list[tuple[SkillDraft, Skill]]:
    """Resolve same-call slug collisions before writing skills.

    Strict mode raises on the first duplicate. Non-strict mode keeps the
    *first* skill with a given slug and drops each later one with a logged
    warning, returning the surviving pairs -- the same partial-success
    treatment non-strict mode already gives a draft that fails formatting.
    Either way nothing is overwritten and the result never names content
    that is not on disk.

    A slug that sanitizes to ``SkillLibraryBackend``'s reserved ``_INDEX``
    always raises, because that name would make the library unusable.

    Args:
        pairs: ``(draft, skill)`` pairs for this call's successfully
            formatted skills, in order -- see :func:`_format_drafts`.
        strict: ``True`` raises on a duplicate slug; ``False`` keeps the first
            and drops each later duplicate with a logged warning.

    Raises:
        ValueError: ``strict`` and two skills in ``pairs`` share
            ``skill.name``; or (in either mode) one of them sanitizes to
            ``SkillLibraryBackend``'s reserved ``INDEX`` parameter name --
            names the offending slug(s) and their source provenance.

    Returns:
        The pairs to keep, in their original order.
    """
    seen: dict[str, SkillDraft] = {}
    kept: list[tuple[SkillDraft, Skill]] = []
    for draft, skill in pairs:
        # `SkillLibraryBackend` sanitizes a slug into a parameter name by
        # mapping "-" to "_" (``backend._param_name``), so slug
        # "skill-index" would land on the reserved ``skill_index`` index
        # parameter and make the whole library unusable.
        if skill.name.replace("-", "_") == _INDEX:
            raise ValueError(
                f"discover_skills: draft ({_provenance_str(draft)}) formatted to slug {skill.name!r}, "
                f"which sanitizes to SkillLibraryBackend's reserved {_INDEX!r} index parameter -- "
                "rename the source draft or adjust format_fn. No skills from this call were "
                "written to disk."
            )
        prior_draft = seen.get(skill.name)
        if prior_draft is not None:
            if strict:
                raise ValueError(
                    f"discover_skills: two drafts formatted to the same slug {skill.name!r} within one "
                    f"call -- first draft ({_provenance_str(prior_draft)}) vs second draft "
                    f"({_provenance_str(draft)}); rename one of the source drafts or adjust format_fn to "
                    "disambiguate them. No skills from this call were written to disk."
                )
            logger.warning(
                "discover_skills: dropping draft (%s) -- formatted to slug %r, already taken by an "
                "earlier draft (%s) in this call; the first is kept",
                _provenance_str(draft),
                skill.name,
                _provenance_str(prior_draft),
            )
            skill_event("candidate.excluded", skill=skill, stage="duplicate", reason="Formatted slug already taken")
            continue
        seen[skill.name] = draft
        kept.append((draft, skill))
    return kept


async def discover_skills(
    trajectories: Sequence[Trajectory],
    *,
    preset: DiscoveryPreset = DEFAULT_PRESET,
    out_root: str | Path | None = None,
    failed_trajectory_ids: frozenset[str] = frozenset(),
    strict_format: bool = False,
) -> list[Skill]:
    """Run the full discovery pipeline: episodes -> extract -> group -> merge -> format -> (save).

    Args:
        trajectories: Source trajectories to mine for skills.
        preset: Names every phase's implementation/injection; see
            :class:`DiscoveryPreset` and this module's docstring for skip
            semantics. Defaults to :data:`DEFAULT_PRESET`.
        out_root: When given, every formatted skill is written under this
            root via ``save_skill(..., dirname=skill.name,
            bump_version=True)`` (see module docstring on version bumping),
            and the returned list holds the *reloaded* on-disk skills.
            ``None`` (default) performs no disk I/O; the returned list holds
            the in-memory candidates :func:`format_skill` produced
            (synthetic, non-existent ``path``s -- see that function's
            docstring).
        failed_trajectory_ids: Forwarded to ``extract_skills`` -- trajectory
            ids to treat as carrying a failure label, gating ``"repair"``
            mode when present in ``preset.modes``. See ``mining.py``'s
            ``extract_skills`` docstring for its own validation of this set.
        strict_format: ``False`` (default) drops a per-draft
            ``FormatAlignmentError`` with a logged warning and keeps going
            (partial success); ``True`` re-raises the first one immediately
            (abort-all). See module docstring's "Formatting failures"
            section -- in particular, the default mode's dropped drafts are
            visible only via the log, not the return value.

    Returns:
        One ``Skill`` per surviving draft after grouping/merging/formatting,
        in the order drafts were produced (episode order, then mode order,
        then one merged-or-fallback entry per group when merging ran) --
        ``[]`` if extraction produced no drafts at all, or if every draft
        was dropped by a non-strict formatting failure.

    Raises:
        ValueError: Propagated unchanged from ``extract_skills``/
            ``group()``/``merge_group`` on a contract violation (see each
            phase's own docstring), or raised by this function itself when
            two drafts in this call format to the same slug (see module
            docstring's "Same-call slug collisions" -- no skills from this
            call are written in that case).
        FormatAlignmentError: A formatted candidate fails
            ``format_alignment`` and ``strict_format=True`` -- see module
            docstring's "Formatting failures" section. Never raised when
            ``strict_format=False`` (the default); a defensive
            ``ValueError`` from a contract-violating ``format_fn`` result
            still propagates in either mode (see :func:`_format_drafts`).
    """
    skill_event("stage.started", stage="segment", source_ids=[t.id for t in trajectories])
    episodes = await _build_episodes(trajectories, preset.episode_builder)
    skill_event("stage.finished", stage="segment", episodes=episodes)

    skill_event("stage.started", stage="extract")
    drafts = await extract_skills(
        trajectories,
        episodes,
        modes=preset.modes,
        extract_fn=preset.extract_fn,
        failed_trajectory_ids=failed_trajectory_ids,
    )
    skill_event("stage.finished", stage="extract", drafts=drafts)
    if not drafts:
        return []

    skill_event("stage.started", stage="group")
    groups = _singleton_groups(drafts) if preset.grouper is None else await preset.grouper.group(drafts)
    skill_event("stage.finished", stage="group", groups=groups)

    skill_event("stage.started", stage="merge")
    merged_drafts = await _merge_groups(groups, preset.gate, preset.merge_fn)
    skill_event("stage.finished", stage="merge", drafts=merged_drafts)

    skill_event("stage.started", stage="format")
    pairs = await _format_drafts(
        merged_drafts,
        preset.format_fn,
        strict_format=strict_format,
        guidance=preset.format_guidance,
    )
    if not pairs:
        return []

    pairs = _check_no_slug_collisions(pairs, strict=strict_format)
    skills = [skill for _draft, skill in pairs]
    skill_event("stage.finished", stage="format", skills=skills)

    if out_root is None:
        return skills

    saved: list[Skill] = []
    for skill in skills:
        directory = save_skill(skill, out_root, dirname=skill.name, bump_version=True)
        saved.append(load_skill(directory))
    return saved


# --- FullDiscoveryMiner adapter ---


class FullDiscoveryMiner:
    """Episodes -> extract -> group -> gated merge -> format, over the round's scored calls."""

    def __init__(self, preset: DiscoveryPreset = DEFAULT_PRESET) -> None:
        self._preset = preset

    async def __call__(self, calls: Sequence[RecordedCall], out_dir: Path) -> list[Skill]:
        """Mine ``calls``; the ones with ``passed=False`` are labeled failed, which enables repair extraction."""
        trajectories = [trajectories_from_events(c.events, trajectory_id=c.call_id) for c in calls]
        failed = frozenset(c.call_id for c in calls if not c.passed)
        return await discover_skills(
            trajectories, preset=self._preset, out_root=out_dir, failed_trajectory_ids=failed, strict_format=False
        )
