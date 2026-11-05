"""The demo's quality metrics, model judge, and admission prefilter.

Format alignment checks SKILL.md structure; completeness and distinctiveness
use a shared model judge. tier1_score combines them for gated merging and
PrefilteredGate, which ranks surviving candidates before the caller's gate
measures completion gain. Revisions are compared with the replaced skill
removed from the quality baseline. Model prompts are preserved verbatim.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

from pydantic import BaseModel, Field

from ai_functions.ai_thread import AIFunction, ThreadConfig
from ai_functions.experimental.skills.evaluation import Candidate, Gate, Replay, Verdict
from ai_functions.experimental.skills.learner import RecordedCall
from ai_functions.experimental.skills.skill_library import Skill, SkillLibrary, load_skill
from ai_functions.runtime.worker import LocalWorker

# --- Shared model judge ---

if TYPE_CHECKING:
    from strands.models import Model

    from ai_functions.protocols import Coordinator

VerdictT = TypeVar("VerdictT", bound=BaseModel)


class JudgeError(Exception):
    """Raised when one ``AIFunctionJudge.score`` call fails to produce a verdict.

    Wraps whatever the underlying cycle raised (``AIFunctionError``,
    Strands' ``StructuredOutputException``, a throttling or credential
    error — anything short of ``BaseException``) so callers — typically a
    ``Metric`` — can catch one judge-specific exception type and map it to
    ``MetricResult(score=None, ...)`` without needing to know about
    ``ai_thread`` internals.

    This is the *recoverable* half of the judge's failure policy: it means
    "this one score is not available, carry on". Its deliberate counterpart
    is ``JudgeUnavailableError``, which does **not** subclass this type
    precisely so that an ``except JudgeError`` handler cannot swallow it.

    Args:
        message: Human-readable explanation.
        cause: The exception that triggered this failure, if any.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class JudgeUnavailableError(Exception):
    """Raised when a judge has failed ``max_consecutive_failures`` times in a row.

    Deliberately **not** a subclass of ``JudgeError``: every judged metric
    narrows its ``except`` clause to ``JudgeError`` and maps it onto
    ``MetricResult(score=None, ...)``, so a sibling type is what lets a
    persistent failure escape that ladder and stop the run. Grinding through
    hundreds of doomed model calls to emit a report full of ``None``s (the
    shape an expired credential or a wholly unreachable region produces) is
    worse than aborting.

    Args:
        message: Human-readable explanation, naming the consecutive-failure
            count and the threshold that was reached.
        cause: The last underlying exception — the one whose failure tripped
            the threshold — if any. Same attribute convention as
            ``JudgeError.cause``.
    """

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class Judge(Protocol):
    """LLM-as-judge contract every judged metric scores against.

    A single method: render a prompt against a pydantic verdict schema and
    get back a typed, structured verdict. Implementations own model/region
    selection and retry policy; callers only ever see the schema they asked
    for or a raised error.

    An implementation is expected to raise ``JudgeError`` for a *recoverable*
    failure — one score that is not available — rather than leaking its
    backend's exception types, so a metric can catch a single type. An
    implementation that concludes it is persistently unusable may instead
    raise ``JudgeUnavailableError``, which is not a ``JudgeError`` and is
    therefore *meant* to escape every metric's handler and stop the run.
    """

    async def score(self, prompt: str, schema: type[VerdictT], *, system_prompt: str | None = None) -> VerdictT:
        """Score ``prompt`` and return a ``schema``-typed verdict.

        Args:
            prompt: Fully composed prompt text.
            schema: Pydantic verdict type the model must return.
            system_prompt: Optional per-call system prompt (e.g. a metric's
                own judging-role framing). ``None`` (default) uses whatever
                system prompt the implementation falls back to.
        """
        ...


def _judge_prompt(prompt: str) -> str:
    """{prompt}"""  # noqa: D415
    # Return the composed string directly so template normalization cannot
    # strip newlines or otherwise change the judge's prompt.
    return prompt


class AIFunctionJudge:
    """``Judge`` backed by this repo's ``@ai_function`` / ``AIThread`` runtime.

    One conceptual template (verbatim prompt, ``max_attempts=2``)
    specialized per call via a fresh ``AIFunction`` sharing the same
    ``prompt_fn``/config but with ``output_type=schema`` — the composition
    ``AIFunction.replace`` cannot express, since it never touches
    ``output_type``.

    Args:
        model: Model id string or ``strands.models.Model`` instance, passed
            straight through to ``ThreadConfig.model``. ``None`` (default)
            means the judge runs on whatever model ``ai_functions`` is
            configured with — the judge does not pin one of its own. Because
            the judge model is what produces the verdicts, anyone comparing
            scores across runs should pass an explicit ``model`` so the
            instrument does not move between them; an explicit value always
            wins over the ambient default. A plain model-id string becomes a
            ``BedrockModel(model_id=...)`` inside ``strands.Agent``, which
            resolves its own region (honoring ``AWS_REGION``) — hence no
            separate region knob here.
        coordinator: When given, each ``score`` call spawns its specialized
            template on a fresh ``LocalWorker`` registered with this
            coordinator (so the judge's thread lands in the caller's event
            log/graph, e.g. alongside the metric run that invoked it).
            When ``None`` (default), each call runs as a throwaway one-shot
            (``AIFunction.__call__``'s own private coordinator, or the
            ambient thread scope if the judge is itself invoked from inside
            a running thread).
        max_consecutive_failures: Circuit-breaker threshold. After this many
            consecutive failed ``score`` calls, the failing call raises
            ``JudgeUnavailableError`` (which no metric catches) instead of
            ``JudgeError``, aborting the run rather than degrading every
            remaining score to ``None``. Any successful call resets the
            counter to zero. ``None`` disables the breaker entirely — every
            failure is wrapped as ``JudgeError``, forever. Must be >= 1 when
            given.

    Raises:
        ValueError: ``max_consecutive_failures`` is not ``None`` and is < 1
            (a threshold below one could only mean "trip before failing").

    Note:
        The failure counter lives on the *instance*. A metric handed
        ``judge=None`` builds its own ``AIFunctionJudge`` in
        ``__post_init__``, so each such metric gets its own independent
        counter and the breaker can only stop that one metric's repeated
        calls — a batch of fifteen metrics could still burn fifteen
        thresholds' worth of doomed calls. The path that actually protects a
        whole evaluation batch is a judge injected once and shared, e.g.
        ``default_registry(judge=AIFunctionJudge())``: there the count runs
        across every metric that shares the instance, so the third
        consecutive failure anywhere in the batch stops it. This is a real
        limitation, and a deliberate one — no global state is introduced to
        paper over it.
    """

    def __init__(
        self,
        model: str | Model | None = None,
        coordinator: Coordinator | None = None,
        *,
        max_consecutive_failures: int | None = 3,
    ) -> None:
        if max_consecutive_failures is not None and max_consecutive_failures < 1:
            raise ValueError(
                "max_consecutive_failures must be >= 1 (or None to disable the "
                f"circuit breaker), got {max_consecutive_failures!r}"
            )
        self._config: ThreadConfig = ThreadConfig(model=model, max_attempts=2)
        self._coordinator: Coordinator | None = coordinator
        self._max_consecutive_failures: int | None = max_consecutive_failures
        self._consecutive_failures: int = 0

    async def score(self, prompt: str, schema: type[VerdictT], *, system_prompt: str | None = None) -> VerdictT:
        """Run one judging call and return ``schema``-typed output.

        Args:
            prompt: Fully composed prompt text, executed verbatim as this
                call's docstring-templated (``"{prompt}"``) rendered prompt.
            schema: Pydantic verdict type the model must return.
            system_prompt: Optional per-call system prompt (e.g. a metric's
                own "you are a meticulous evaluator" framing). Routed to
                the underlying thread's real system-prompt surface
                (``ThreadConfig.system_prompt``, which ``ai_thread.py``
                passes straight through to ``strands.Agent(system_prompt=
                ...)``). ``None`` (default) leaves ``self._config``'s system prompt
                (i.e. ``ai_thread``'s own default) unchanged.

        Returns:
            The ``schema``-typed verdict.

        Raises:
            JudgeError: The underlying cycle failed to produce a verdict, and
                this was not the call that tripped the circuit breaker. Every
                ``Exception`` raised below this point is wrapped, and the
                original is available as ``JudgeError.cause``: an
                ``AIFunctionError`` (e.g. ``max_attempts`` exhausted), a
                ``strands.types.exceptions.StructuredOutputException``
                (Strands' own event loop forces one retry when the model
                doesn't call the verdict-schema tool; if the forced retry
                also comes back without it, Strands raises this directly,
                never wrapped as ``AIFunctionError`` — see
                ``tests/test_example_skillify_demo_quality.py::
                test_score_raises_judge_error_when_model_never_calls_the_verdict_tool``),
                a ``MaxTokensReachedException``, a throttling or other
                model-provider exception, or a ``botocore`` credential,
                permission, or network error.
            JudgeUnavailableError: This failure was the
                ``max_consecutive_failures``-th in a row. The tripping call
                raises this *instead of* ``JudgeError`` — it is one failure,
                reported once, at its most severe level — so no metric's
                ``except JudgeError`` swallows it and the evaluation run
                stops. The same underlying exception is carried as
                ``JudgeUnavailableError.cause``.

        Note:
            ``BaseException`` subclasses that are not ``Exception`` —
            ``asyncio.CancelledError``, ``KeyboardInterrupt``,
            ``SystemExit`` — are never caught, never wrapped, and never
            counted against the breaker; cancelling a run must stay
            cancellable.
        """
        config = (
            self._config if system_prompt is None else dataclasses.replace(self._config, system_prompt=system_prompt)
        )
        template: AIFunction[[str], VerdictT] = AIFunction(
            prompt_fn=_judge_prompt,
            output_type=schema,
            config=config,
        )
        try:
            if self._coordinator is not None:
                worker = LocalWorker(self._coordinator)
                try:
                    handle = await worker.spawn_locally(template, thread_name="ai_function_judge")
                    verdict = await handle.run(prompt)
                finally:
                    await worker.close()
            else:
                verdict = await template(prompt)
        except Exception as exc:
            # ``Exception``, never ``BaseException``: ``asyncio.CancelledError``,
            # ``KeyboardInterrupt`` and ``SystemExit`` must keep propagating.
            raise self._on_failure(schema, exc) from exc
        self._consecutive_failures = 0
        return verdict

    def _on_failure(self, schema: type[BaseModel], exc: Exception) -> JudgeError | JudgeUnavailableError:
        """Count one failed ``score`` call and build the exception it raises.

        Args:
            schema: The verdict type the failed call was scoring against,
                named in the message so a batch's logs identify the metric.
            exc: The exception the underlying cycle raised.

        Returns:
            A ``JudgeUnavailableError`` when this failure is the
            ``max_consecutive_failures``-th consecutive one (the tripping
            call reports at that severity *instead of* ``JudgeError``, so it
            is not swallowed by a metric), otherwise a ``JudgeError``. Both
            carry ``exc`` as ``cause``; the caller raises the result ``from
            exc``.
        """
        self._consecutive_failures += 1
        limit = self._max_consecutive_failures
        if limit is not None and self._consecutive_failures >= limit:
            return JudgeUnavailableError(
                f"AIFunctionJudge has failed {self._consecutive_failures} consecutive scoring "
                f"call(s), reaching its max_consecutive_failures threshold of {limit}; aborting "
                "the run instead of degrading every remaining score to None. Last failure was "
                f"against schema {schema.__name__!r}: {exc}",
                cause=exc,
            )
        return JudgeError(
            f"AIFunctionJudge failed to score prompt against schema {schema.__name__!r}: {exc}",
            cause=exc,
        )


# --- Format, completeness, and distinctiveness metrics ---

# ---------------------------------------------------------------------------
# Result and input shapes
# ---------------------------------------------------------------------------


class EvaluationLevel(int, Enum):
    """The level a metric scores at; the Tier-1 metrics score the skill specification."""

    SKILL_SPECIFICATION = 1


class EvaluationMethod(StrEnum):
    """How a metric's score is produced."""

    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"
    HYBRID = "hybrid"


class MetricSpec(BaseModel):
    """Serialisable description of a metric, set on each metric as ``spec``."""

    name: str = Field(..., description="Unique identifier.")
    level: EvaluationLevel = Field(..., description="Which evaluation level.")
    method: EvaluationMethod = Field(..., description="How the score is produced.")
    inputs_required: list[str] = Field(
        default_factory=list,
        description="Names of fields the metric reads from its input.",
    )
    description: str = Field(default="", description="Human-readable summary.")
    tags: list[str] = Field(default_factory=list, description="Optional free-form tags.")

    model_config = {"frozen": True}


@dataclass
class MetricResult:
    """One metric's output.

    Attributes:
        score: Primary numeric score in [0, 1], or ``None`` when the metric
            could not score (e.g. the judge failed).
        passed: Whether the metric considers this case a pass.
        rationale: Free-form explanation.
        components: Sub-scores keyed by component name.
        evidence: Optional evidence snippets.
        metadata: Free-form bag for metric-specific extras.
    """

    score: float | None
    passed: bool | None
    rationale: str = ""
    components: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Level1Input:
    """What a Tier-1 metric scores.

    Attributes:
        skill_spec: The skill under review.
        skill_library: Other skills the candidate is measured against
            (for distinctiveness).
    """

    skill_spec: Any
    skill_library: Any | None = None


@dataclass(frozen=True)
class SkillFrontmatterSpec:
    """Expected shape of a SKILL.md's YAML frontmatter."""

    required_keys: tuple[str, ...] = ("name", "description")
    optional_keys: tuple[str, ...] = ("license", "allowed-tools", "version")
    name_pattern: str = r"^[a-z0-9][a-z0-9-]{0,63}$"
    description_min_words: int = 5
    description_max_chars: int = 1024


@dataclass(frozen=True)
class ExpectedSkillFormat:
    """The structural template every skill is expected to conform to."""

    frontmatter: SkillFrontmatterSpec = field(default_factory=SkillFrontmatterSpec)

    expected_sections: tuple[str, ...] = (
        "Overview",
        "When to use this skill",
        "Instructions",
    )
    optional_sections: tuple[str, ...] = (
        "Examples",
        "Resources",
        "Notes",
    )

    body_min_words: int = 40

    def as_prompt_block(self) -> str:
        """Render the spec as text suitable for embedding in a judge prompt."""
        fm = self.frontmatter
        lines = [
            "Required frontmatter keys: " + ", ".join(fm.required_keys),
            "Optional frontmatter keys: " + ", ".join(fm.optional_keys),
            f"`name` must match regex: {fm.name_pattern}",
            f"`description` must be {fm.description_min_words}+ words and <= {fm.description_max_chars} characters.",
            "",
            "Expected H2 sections (in body):",
            *[f"  - {s}" for s in self.expected_sections],
            "Optional H2 sections:",
            *[f"  - {s}" for s in self.optional_sections],
            f"Body must be at least {self.body_min_words} words.",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared lexical helper (used by distinctiveness)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _tfidf_cosine(a: str, corpus: Iterable[str]) -> tuple[float, str | None]:
    """Return (max similarity in [0, 1], doc_id of nearest neighbour).

    Uses a simple TF-IDF weighting on word tokens, with cosine similarity.
    The corpus is expected to be the *other* docs the candidate should be
    distinct from. Hand-rolled (no sklearn) to keep this metric dependency-free.
    """
    docs = list(corpus)
    if not docs:
        return 0.0, None

    a_tokens = _tokens(a)
    if not a_tokens:
        return 0.0, None

    doc_tokens = [_tokens(d) for d in docs]
    for idx, tokens in enumerate(doc_tokens):
        if tokens == a_tokens:
            return 1.0, docs[idx][:60]

    n_docs = 1 + len(doc_tokens)
    df: Counter[str] = Counter()
    for d in [a_tokens, *doc_tokens]:
        df.update(set(d))

    def vec(toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        return {t: (count / len(toks)) * math.log(n_docs / df[t]) for t, count in tf.items() if df[t] > 0}

    a_vec = vec(a_tokens)

    best_sim = 0.0
    best_idx: int | None = None
    a_norm = math.sqrt(sum(v * v for v in a_vec.values()))
    if a_norm == 0.0:
        return 0.0, None
    for idx, toks in enumerate(doc_tokens):
        d_vec = vec(toks)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values()))
        if d_norm == 0.0:
            continue
        common = set(a_vec) & set(d_vec)
        dot = sum(a_vec[t] * d_vec[t] for t in common)
        sim = dot / (a_norm * d_norm)
        if sim > best_sim:
            best_sim = sim
            best_idx = idx

    nearest = None
    if best_idx is not None:
        nearest = docs[best_idx][:60]
    return float(min(max(best_sim, 0.0), 1.0)), nearest


# ---------------------------------------------------------------------------
# FormatAlignmentMetric (deterministic)
# ---------------------------------------------------------------------------


@dataclass
class FormatAlignmentMetric:
    """Deterministic structural check against ``ExpectedSkillFormat``.

    Verifies that a SKILL.md conforms to the structural template laid out in
    ``ExpectedSkillFormat``: required frontmatter keys, the ``name`` regex,
    description length, expected H2 sections, and body-length floor. No LLM
    in the loop. Pass ``expected=`` a custom ``ExpectedSkillFormat`` to adapt
    the template to a dataset.
    """

    expected: ExpectedSkillFormat = field(default_factory=ExpectedSkillFormat)
    pass_threshold: float = 1.0

    def __post_init__(self) -> None:
        """Register this metric's ``MetricSpec``."""
        self.spec = MetricSpec(
            name="format_alignment",
            level=EvaluationLevel.SKILL_SPECIFICATION,
            method=EvaluationMethod.DETERMINISTIC,
            inputs_required=["skill_spec"],
            description=("Structural conformance of a SKILL.md to the expected frontmatter + section template."),
            tags=["level1", "format"],
        )

    def compute(self, metric_input: Level1Input) -> MetricResult:
        """Score ``metric_input.skill_spec`` against ``self.expected``."""
        skill = metric_input.skill_spec
        if not isinstance(skill, Skill):
            raise TypeError(f"FormatAlignmentMetric expects skill_spec to be a Skill, got {type(skill).__name__}")

        violations: list[str] = []
        components: dict[str, float] = {}

        fm_score = self._check_frontmatter(skill, violations)
        components["frontmatter"] = fm_score

        body_score = self._check_body(skill, violations)
        components["body"] = body_score

        score = (fm_score + body_score) / 2.0
        passed = score >= self.pass_threshold and not violations
        rationale = "All checks passed." if not violations else "Violations: " + "; ".join(violations)
        return MetricResult(
            score=score,
            passed=passed,
            rationale=rationale,
            components=components,
            evidence=violations,
        )

    def _check_frontmatter(self, skill: Skill, violations: list[str]) -> float:
        fm_spec = self.expected.frontmatter
        fm = skill.frontmatter or {}
        checks = 0
        passes = 0

        for key in fm_spec.required_keys:
            checks += 1
            if key not in fm or fm[key] in (None, ""):
                violations.append(f"missing required frontmatter key {key!r}")
            else:
                passes += 1

        checks += 1
        name = str(fm.get("name", ""))
        if re.fullmatch(fm_spec.name_pattern, name):
            passes += 1
        else:
            violations.append(f"frontmatter name {name!r} does not match {fm_spec.name_pattern}")

        checks += 1
        desc = str(fm.get("description", ""))
        word_count = len(desc.split())
        if fm_spec.description_min_words <= word_count and len(desc) <= fm_spec.description_max_chars:
            passes += 1
        else:
            violations.append(
                f"description length {word_count} words / {len(desc)} chars "
                f"out of range "
                f"({fm_spec.description_min_words}+ words, "
                f"≤ {fm_spec.description_max_chars} chars)"
            )
        return passes / checks if checks else 1.0

    def _check_body(self, skill: Skill, violations: list[str]) -> float:
        sections = skill.section_headings()
        checks = 0
        passes = 0
        for section in self.expected.expected_sections:
            checks += 1
            if section in sections:
                passes += 1
            else:
                violations.append(f"missing expected section {section!r}")

        checks += 1
        body_words = len(skill.body.split())
        if body_words >= self.expected.body_min_words:
            passes += 1
        else:
            violations.append(f"body has {body_words} words, expected ≥ {self.expected.body_min_words}")
        return passes / checks if checks else 1.0


# ---------------------------------------------------------------------------
# CompletenessMetric (LLM judge)
# ---------------------------------------------------------------------------

_COMPLETENESS_SYSTEM_PROMPT = """\
You judge whether a SKILL.md is *complete enough* for an autonomous agent
to follow without asking the author for clarification. You ground every
judgement in concrete excerpts from the skill body.
"""

_COMPLETENESS_USER_PROMPT = """\
Below is a SKILL.md plus the structural template the team expects.

Score how *complete* the skill content is, on a 0.0–1.0 scale:
  - 1.0 = a competent agent can execute the skill end-to-end with no
    further information.
  - 0.5 = the skill describes the goal but skips concrete steps,
    examples, or boundaries.
  - 0.0 = the skill is essentially a stub.

Return scores for these dimensions and a final aggregate:
  * scope_clarity     — does the skill make its applicability obvious?
  * when_to_use       — are the trigger conditions stated unambiguously?
  * step_specificity  — are the steps concrete, ordered, and actionable?
  * examples          — is at least one worked example present?

Aggregate is the unweighted mean.

Expected template:
{expected}

SKILL.md:
---
{skill_text}
---
"""


class _CompletenessVerdict(BaseModel):
    scope_clarity: float = Field(..., ge=0.0, le=1.0)
    when_to_use: float = Field(..., ge=0.0, le=1.0)
    step_specificity: float = Field(..., ge=0.0, le=1.0)
    examples: float = Field(..., ge=0.0, le=1.0)
    aggregate: float = Field(..., ge=0.0, le=1.0)
    rationale: str


@dataclass
class CompletenessMetric:
    """LLM-judge metric grading content depth of a SKILL.md.

    Asks whether a SKILL.md contains the *content* an operator would
    actually need to follow it: clear scope, when-to-use criteria, concrete
    steps, and at least one example. Format is checked separately by
    ``FormatAlignmentMetric`` — this metric grades depth.
    """

    judge: Judge | None = None
    expected: ExpectedSkillFormat = field(default_factory=ExpectedSkillFormat)
    pass_threshold: float = 0.7

    def __post_init__(self) -> None:
        """Lazily build an ambient judge and register this metric's ``MetricSpec``."""
        if self.judge is None:
            self.judge = AIFunctionJudge()
        self.spec = MetricSpec(
            name="completeness",
            level=EvaluationLevel.SKILL_SPECIFICATION,
            method=EvaluationMethod.LLM_JUDGE,
            inputs_required=["skill_spec"],
            description=(
                "LLM-judge assessment of whether SKILL.md content is deep enough to execute the skill end-to-end."
            ),
            tags=["level1", "content"],
        )

    async def compute_async(self, metric_input: Level1Input) -> MetricResult:
        """Score ``metric_input.skill_spec``'s content depth via the judge."""
        skill = metric_input.skill_spec
        if not isinstance(skill, Skill):
            raise TypeError(f"CompletenessMetric expects a Skill, got {type(skill).__name__}")

        prompt = _COMPLETENESS_USER_PROMPT.format(
            expected=self.expected.as_prompt_block(),
            skill_text=skill.raw,
        )
        assert self.judge is not None  # set in __post_init__
        try:
            verdict = await self.judge.score(prompt, _CompletenessVerdict, system_prompt=_COMPLETENESS_SYSTEM_PROMPT)
        except JudgeError as exc:
            return MetricResult(
                score=None,
                passed=None,
                rationale=f"Judge failed to score completeness: {exc}",
                metadata={"judge": "ai_function/v1", "error": str(exc)},
            )
        components = {
            "scope_clarity": verdict.scope_clarity,
            "when_to_use": verdict.when_to_use,
            "step_specificity": verdict.step_specificity,
            "examples": verdict.examples,
        }
        return MetricResult(
            score=verdict.aggregate,
            passed=verdict.aggregate >= self.pass_threshold,
            rationale=verdict.rationale,
            components=components,
            metadata={"judge": "ai_function/v1"},
        )


# ---------------------------------------------------------------------------
# DistinctivenessMetric (hybrid)
# ---------------------------------------------------------------------------

_DISTINCT_SYSTEM_PROMPT = (
    "You judge whether a Skill duplicates another Skill in a "
    "library. You are concise and ground every verdict in "
    "concrete textual overlap."
)

_DISTINCT_JUDGE_PROMPT = """\
You are comparing one candidate Skill against the rest of a Skill
library. Decide whether the candidate is *semantically distinct* — i.e.
whether an agent could mistakenly invoke another skill in its place.

Return:
  * score in [0, 1] where 1.0 = clearly distinct, 0.0 = effectively a
    duplicate.
  * most_similar_skill: the name of the closest other skill, or "" if
    none feels close.
  * reasoning: 1-2 sentences citing the overlap (or lack thereof).

Candidate skill:
---
{candidate}
---

Other skills in the library:
{others}
"""


class _DistinctVerdict(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    most_similar_skill: str = ""
    reasoning: str


@dataclass
class DistinctivenessMetric:
    """Hybrid distinctiveness metric: lexical TF-IDF + LLM semantic check.

    Distinctiveness matters because near-duplicate skills make the agent's
    selection step harder. The metric measures the candidate against
    whatever ``skill_library`` the caller hands in via ``Level1Input``.
    Callers that want intra-task comparisons (e.g. a skill should be
    distinct from its sibling skills under the same task but is *expected*
    to differ from skills under unrelated tasks) should scope the library
    before invoking the metric.
    """

    judge: Judge | None = None
    pass_threshold: float = 0.6
    lexical_weight: float = 0.4
    semantic_weight: float = 0.6

    def __post_init__(self) -> None:
        """Lazily build an ambient judge and register this metric's ``MetricSpec``."""
        if self.judge is None:
            self.judge = AIFunctionJudge()
        self.spec = MetricSpec(
            name="distinctiveness",
            level=EvaluationLevel.SKILL_SPECIFICATION,
            method=EvaluationMethod.HYBRID,
            inputs_required=["skill_spec", "skill_library"],
            description=("Hybrid TF-IDF + LLM-judge measure of how distinct a skill is from the rest of the library."),
            tags=["level1", "library"],
        )

    async def compute_async(self, metric_input: Level1Input) -> MetricResult:
        """Score how lexically + semantically distinct the candidate is from its library."""
        candidate = metric_input.skill_spec
        library = metric_input.skill_library
        if not isinstance(candidate, Skill):
            raise TypeError("skill_spec must be a Skill")
        if not isinstance(library, SkillLibrary):
            raise TypeError("skill_library must be a SkillLibrary")

        others = [s for s in library.skills if s.path != candidate.path]
        if not others:
            return MetricResult(
                score=1.0,
                passed=True,
                rationale="Library has no other skills to compare against.",
                components={"lexical": 1.0, "semantic": 1.0},
            )

        sim, nearest = _tfidf_cosine(candidate.raw, [s.raw for s in others])
        lexical_distinct = 1.0 - sim

        listing = "\n".join(f"- {s.name}: {s.description}" for s in others)
        assert self.judge is not None  # set in __post_init__
        try:
            verdict = await self.judge.score(
                _DISTINCT_JUDGE_PROMPT.format(candidate=candidate.raw, others=listing),
                _DistinctVerdict,
                system_prompt=_DISTINCT_SYSTEM_PROMPT,
            )
        except JudgeError as exc:
            return MetricResult(
                score=None,
                passed=None,
                rationale=f"Judge failed to score distinctiveness: {exc}",
                components={"lexical": lexical_distinct},
                metadata={"judge": "ai_function/v1", "error": str(exc)},
            )

        score = self.lexical_weight * lexical_distinct + self.semantic_weight * verdict.score
        components = {
            "lexical": lexical_distinct,
            "semantic": verdict.score,
        }
        rationale = (
            f"lexical_max_sim={sim:.3f} (nearest snippet: {nearest!r}); "
            f"semantic={verdict.score:.3f} ({verdict.most_similar_skill!r}) — "
            f"{verdict.reasoning}"
        )
        return MetricResult(
            score=score,
            passed=score >= self.pass_threshold,
            rationale=rationale,
            components=components,
            evidence=[verdict.most_similar_skill] if verdict.most_similar_skill else [],
            metadata={"judge": "ai_function/v1"},
        )


# ---------------------------------------------------------------------------
# The Tier-1 aggregate
# ---------------------------------------------------------------------------


async def _tier1_aggregate(
    skill: Skill,
    library: SkillLibrary,
    judge: Judge | None,
) -> tuple[float | None, dict[str, float | None]]:
    """Score ``skill`` via Tier-1 completeness/distinctiveness/format_alignment.

    Args:
        skill: The skill to score.
        library: Library ``skill`` is measured for distinctiveness against.
        judge: Shared judge for the judged metrics; ``None`` lets each
            metric build its own lazy default (this function never eagerly
            constructs a judge itself).

    Returns:
        ``(aggregate, components)`` -- ``aggregate`` is the mean of
        whichever of the three metrics returned a non-``None`` score, or
        ``None`` if all three did (format_alignment is deterministic and
        always returns a score, so this is a defensive floor, not an
        expected case). ``components`` carries each metric's own score
        (possibly ``None``) keyed by metric name, for callers that want the
        breakdown.
    """
    metric_input = Level1Input(skill_spec=skill, skill_library=library)
    format_result = FormatAlignmentMetric().compute(metric_input)
    completeness_result = await CompletenessMetric(judge=judge).compute_async(metric_input)
    distinctiveness_result = await DistinctivenessMetric(judge=judge).compute_async(metric_input)
    components: dict[str, float | None] = {
        "completeness": completeness_result.score,
        "distinctiveness": distinctiveness_result.score,
        "format_alignment": format_result.score,
    }
    values = [v for v in components.values() if v is not None]
    aggregate = sum(values) / len(values) if values else None
    return aggregate, components


def _empty_library() -> SkillLibrary:
    return SkillLibrary(root=Path("."), skills=[])


async def tier1_score(skill: Skill, current_dirs: Sequence[Path], judge: Judge | None) -> float | None:
    """The Tier-1 aggregate of ``skill``, measured for distinctiveness against ``current_dirs``.

    ``current_dirs`` is the installed library a gate is handed; with none,
    the comparison library is empty and distinctiveness scores 1.0.
    """
    dirs = [Path(p) for p in current_dirs]
    library = SkillLibrary(root=dirs[0].parent, skills=[load_skill(p) for p in dirs]) if dirs else _empty_library()
    aggregate, _components = await _tier1_aggregate(skill, library, judge)
    return aggregate


# --- PrefilteredGate adapter ---


class PrefilteredGate:
    """Tier-1 quality prefilter, then ``inner`` on the survivors, highest Tier-1 first.

    Ordering the survivors is what makes ``inner``'s budget cut (from the end of the
    list) drop the lowest-quality candidates first. Every verdict carries its
    candidate's ``tier1`` score; a prefilter rejection has ``stage="prefilter"`` and
    is decided before anything ``inner`` decides. A revision's (``candidate.replaces``
    is not ``None``) Tier-1 is measured against ``current`` with the replaced skill
    removed, so it is not penalised for resembling its own previous version.

    Raises:
        RuntimeError: ``inner`` returns a verdict count other than the number of
            survivors it was handed, or a verdict about a candidate it was not
            handed -- mirroring the learner's own check on a custom ``Gate``.
    """

    def __init__(self, inner: Gate, *, min_tier1: float = 0.5, judge: Judge | None = None) -> None:
        self._inner, self._min_tier1, self._judge = inner, min_tier1, judge

    async def __call__(
        self,
        candidates: Sequence[Candidate],
        current: Sequence[Path],
        trials: Sequence[RecordedCall],
        replay: Replay,
    ) -> list[Verdict]:
        """One verdict per candidate, in order."""
        verdicts: dict[int, Verdict] = {}
        scored: list[tuple[float, Candidate]] = []
        for c in candidates:
            # A revision's distinctiveness is measured against the library with the
            # skill it replaces removed -- otherwise it reads as a near-duplicate of
            # its own previous version, which is expected and not a quality defect.
            baseline = [p for p in current if p.name != c.replaces] if c.replaces else current
            t1 = await tier1_score(c.skill, baseline, self._judge)
            from ai_functions.experimental.skills import skill_event

            skill_event("candidate.scored", skill=c.skill, stage="prefilter", score=t1, threshold=self._min_tier1)
            if t1 is None or t1 < self._min_tier1:
                scores: dict[str, Any] = {"tier1": t1, "stage": "prefilter", "order": float(len(verdicts))}
                reason = f"Tier-1 {t1!r} < min_tier1 {self._min_tier1!r}"
                verdicts[id(c)] = Verdict(c, False, None, reason, scores)
            else:
                scored.append((t1, c))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        tier1_of = {id(c): t for t, c in scored}
        tier1_by_cut = {f"candidate {c.skill.name}": f"candidate {c.skill.name} (tier1 {t:.2f})" for t, c in scored}
        decided_first = len(verdicts)
        if scored:
            survivors = [c for _, c in scored]
            inner_verdicts = await self._inner(survivors, current, trials, replay)
            if len(inner_verdicts) != len(survivors):
                raise RuntimeError(
                    f"inner gate returned {len(inner_verdicts)} verdicts for {len(survivors)} candidates"
                )
            if sorted(map(id, (v.candidate for v in inner_verdicts))) != sorted(map(id, survivors)):
                raise RuntimeError("inner gate returned verdicts that are not one per candidate it was given")
            for n, v in enumerate(inner_verdicts):
                scores = {"tier1": tier1_of[id(v.candidate)], **v.scores}
                scores["order"] = decided_first + float(scores.get("order", n))
                if isinstance(cuts := scores.get("budget_cuts"), list):
                    scores["budget_cuts"] = [tier1_by_cut.get(str(cut), str(cut)) for cut in cuts]
                verdicts[id(v.candidate)] = Verdict(v.candidate, v.admitted, v.gain, v.reason, scores)
        return [verdicts[id(c)] for c in candidates]
