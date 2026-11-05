"""Tests for the demo's judge, Tier-1 quality metrics and run types.

Exercises ``full_pipeline/quality.py`` and ``evaluation.py``. These tests came
from the library's eval suite and cover the metrics used by the demo.

``AIFunctionJudge.score`` is exercised end to end with
``ai_functions.testing.ScriptedModel`` (a real ``strands.models.Model`` that
plays back a scripted turn), and its failure policy by scripting
``AIFunction.__call__``: every ``Exception`` becomes a ``JudgeError``,
``BaseException`` subclasses such as ``asyncio.CancelledError`` pass through,
and the ``max_consecutive_failures``-th consecutive failure raises
``JudgeUnavailableError``, which escapes a judged metric's handler and stops
the run. The judged metrics run on ``StubJudge`` (``tests/_eval_stubs.py``) or
a local ``_FailingJudge``; ``trace_from_events`` runs on hand-built event logs.

Async code is driven via ``asyncio.run()`` inside plain ``def test_...``
functions. No LLM/network.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TypeVar

import pytest
from _eval_stubs import StubJudge
from pydantic import BaseModel
from strands.types.exceptions import StructuredOutputException

from ai_functions.ai_thread import AIFunction, AIFunctionError, ThreadConfig
from ai_functions.experimental.skills.skill_library import Skill, SkillLibrary, load_skill, load_skill_library
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types.events import (
    CompletedEvent,
    MessageAssistantCompleteEvent,
    ResultEvent,
    StartedEvent,
    ThreadId,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ai_functions.types.events import TokenUsage as RealTokenUsage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "skillify_demo"))

import evaluation as trace_module  # noqa: E402
from evaluation import (  # noqa: E402
    _ACTION_TYPE_BY_TOOL,
    OracleResult,
    _extract_skill_slug,
    trace_from_events,
)
from full_pipeline.quality import (  # noqa: E402
    _COMPLETENESS_SYSTEM_PROMPT,
    _COMPLETENESS_USER_PROMPT,
    _DISTINCT_JUDGE_PROMPT,
    _DISTINCT_SYSTEM_PROMPT,
    AIFunctionJudge,
    CompletenessMetric,
    DistinctivenessMetric,
    ExpectedSkillFormat,
    FormatAlignmentMetric,
    JudgeError,
    JudgeUnavailableError,
    Level1Input,
    _CompletenessVerdict,
    _DistinctVerdict,
    _judge_prompt,
    _tfidf_cosine,
)

# ===========================================================================
# quality.py
# ===========================================================================


class _Verdict(BaseModel):
    score: float
    rationale: str


class _OtherVerdict(BaseModel):
    passed: bool


# ── prompt parity: verbatim, no additions ──────────────────────────────────


def test_judge_template_docstring_is_bare_prompt_placeholder() -> None:
    """The template's docstring is exactly ``"{prompt}"`` -- no wrapper text."""
    assert _judge_prompt.__doc__ == "{prompt}"


def test_render_prompt_is_byte_identical_to_composed_prompt() -> None:
    """``AIFunction.render_prompt`` returns the composed prompt unchanged.

    This is the level at which "verbatim, no additions" parity holds: the
    ``ai_thread`` runtime separately appends a "use the X tool" instruction
    to the *rendered user turn* later in the cycle (shared by every
    ``@ai_function`` producing structured output, not something this module
    adds) -- ``render_prompt`` is upstream of that and is what a Judge
    implementation controls.
    """
    composed = "Evaluate this skill on clarity and correctness.\n\nRubric:\n- ...\n"
    template: AIFunction[[str], _Verdict] = AIFunction(
        prompt_fn=_judge_prompt,
        output_type=_Verdict,
        config=ThreadConfig(),
    )

    async def _run() -> str:
        return await template.render_prompt(composed)

    assert asyncio.run(_run()) == composed


# ── construction: model resolution ─────────────────────────────────────────


def test_explicit_model_arg_is_honored() -> None:
    judge = AIFunctionJudge(model="explicit-model-id")
    assert judge._config.model == "explicit-model-id"  # noqa: SLF001


def test_model_defaults_to_the_ambient_configuration() -> None:
    """No model argument leaves ``ThreadConfig.model`` unset.

    The judge inherits whatever model the surrounding ai-functions
    configuration resolves, rather than pinning one of its own. Because the
    judge model determines the verdicts, anyone comparing scores across runs
    should pass ``model=`` explicitly.
    """
    judge = AIFunctionJudge()
    assert judge._config.model is None  # noqa: SLF001


def test_max_attempts_is_two_per_the_failure_ladder() -> None:
    judge = AIFunctionJudge(model="whatever")
    assert judge._config.max_attempts == 2  # noqa: SLF001


# ── specialization wiring: fresh AIFunction per schema, not .replace ───────


def test_score_specializes_a_fresh_template_per_schema() -> None:
    """Two ``score`` calls with different schemas each get their own
    ``AIFunction`` with ``output_type=schema`` -- ``AIFunction.replace``
    cannot change ``output_type``, so this must be constructed fresh."""
    model = ScriptedModel(
        [
            Turn(tool_calls=(("_Verdict", {"score": 0.5, "rationale": "ok"}),)),
            Turn(tool_calls=(("_OtherVerdict", {"passed": True}),)),
        ]
    )
    judge = AIFunctionJudge(model=model)

    async def _run() -> tuple[_Verdict, _OtherVerdict]:
        first = await judge.score("prompt one", _Verdict)
        second = await judge.score("prompt two", _OtherVerdict)
        return first, second

    first, second = asyncio.run(_run())
    assert first == _Verdict(score=0.5, rationale="ok")
    assert second == _OtherVerdict(passed=True)


def _spy_on_model_stream(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Patch ``ScriptedModel.stream`` to record every ``system_prompt`` it's
    called with, then delegate to the original implementation.

    ``ScriptedModel`` is ``@final`` (can't subclass) and its own ``stream``
    ignores ``system_prompt`` entirely (it only plays back scripted turns),
    so patching the method is the seam to prove ``AIFunctionJudge.score``'s
    ``system_prompt`` actually reaches ``strands.Agent`` / the model's
    ``stream`` call -- not just that it's accepted and dropped.
    ``strands`` itself calls ``model.stream(messages, tool_specs,
    system_prompt, ...)`` positionally (``strands/event_loop/streaming.py``),
    so the replacement binds ``system_prompt`` by name (not ``*args``) to
    actually capture it.
    """
    seen: list[str | None] = []
    original_stream = ScriptedModel.stream

    def _spy_stream(
        self: ScriptedModel,
        messages: object,
        tool_specs: object = None,
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> object:
        seen.append(system_prompt)
        return original_stream(self, messages, tool_specs, system_prompt, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ScriptedModel, "stream", _spy_stream)
    return seen


def test_score_routes_system_prompt_to_the_underlying_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``system_prompt=`` isn't just accepted -- it reaches the real
    ``strands.Agent`` call (via ``ThreadConfig.system_prompt``,
    ``dataclasses.replace``d per call since ``ThreadConfig`` is frozen)."""
    seen = _spy_on_model_stream(monkeypatch)
    model = ScriptedModel([Turn(tool_calls=(("_Verdict", {"score": 0.5, "rationale": "ok"}),))])
    judge = AIFunctionJudge(model=model)

    asyncio.run(judge.score("prompt", _Verdict, system_prompt="You are a meticulous evaluator."))

    assert seen == ["You are a meticulous evaluator."]


def test_score_without_system_prompt_leaves_base_config_system_prompt_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting ``system_prompt`` must not perturb the judge's own default:
    ``self._config.system_prompt`` is ``None`` (unset), which ``ai_thread``
    resolves to its own framework default (``_DEFAULT_SYSTEM_PROMPT``), the
    same as before this amendment -- not the empty string, not ``None``
    reaching the model."""
    seen = _spy_on_model_stream(monkeypatch)
    model = ScriptedModel([Turn(tool_calls=(("_Verdict", {"score": 0.5, "rationale": "ok"}),))])
    judge = AIFunctionJudge(model=model)

    asyncio.run(judge.score("prompt", _Verdict))

    assert seen == ["You are an expert assistant who can solve any task"]


def test_score_with_explicit_coordinator_spawns_a_worker_on_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller-supplied ``coordinator`` isn't just accepted and ignored --
    ``score`` spawns the specialized template via a throwaway ``LocalWorker``
    registered on *that* coordinator (so the judge's thread lands in the
    caller's event log/graph), not a private one of its own. The worker is
    torn down (``close()``) before ``score`` returns, so this spies on
    construction rather than asserting post-hoc coordinator state."""
    import full_pipeline.quality as judge_module

    from ai_functions import InMemoryCoordinator
    from ai_functions.protocols import Coordinator

    coord = InMemoryCoordinator()
    seen_coordinators: list[Coordinator] = []
    real_local_worker = judge_module.LocalWorker

    class _SpyWorker(real_local_worker):  # type: ignore[misc, valid-type]
        def __init__(self, coordinator: Coordinator) -> None:
            seen_coordinators.append(coordinator)
            super().__init__(coordinator)

    monkeypatch.setattr(judge_module, "LocalWorker", _SpyWorker)
    model = ScriptedModel([Turn(tool_calls=(("_Verdict", {"score": 0.9, "rationale": "great"}),))])
    judge = AIFunctionJudge(model=model, coordinator=coord)

    verdict = asyncio.run(judge.score("prompt", _Verdict))

    assert verdict == _Verdict(score=0.9, rationale="great")
    assert seen_coordinators == [coord]


# ── live round trip via ScriptedModel ───────────────────────────────────────


def test_score_round_trip_returns_typed_verdict() -> None:
    model = ScriptedModel([Turn(tool_calls=(("_Verdict", {"score": 0.75, "rationale": "well argued"}),))])
    judge = AIFunctionJudge(model=model)

    verdict = asyncio.run(judge.score("Judge this skill spec.", _Verdict))

    assert isinstance(verdict, _Verdict)
    assert verdict.score == 0.75
    assert verdict.rationale == "well argued"


def test_score_raises_judge_error_when_model_never_calls_the_verdict_tool() -> None:
    """No structured-output tool call at all: Strands' own event loop forces
    one retry ("Force structured output tool call if LLM didn't use it
    automatically"), consuming a second scripted turn; when that also comes
    back without a tool call, Strands raises ``StructuredOutputException``
    directly (never wrapped as ``ai_thread``'s own ``AIFunctionError`` --
    see ``score``'s docstring). ``score`` must still surface *some* failure
    as a ``JudgeError``, wrapping whichever exception the cycle actually
    raised."""
    model = ScriptedModel(
        [
            Turn(text="I decline to use the tool."),
            Turn(text="Still declining."),
        ]
    )
    judge = AIFunctionJudge(model=model)

    with pytest.raises(JudgeError) as exc_info:
        asyncio.run(judge.score("Judge this skill spec.", _Verdict))

    assert isinstance(exc_info.value.cause, (AIFunctionError, StructuredOutputException))
    assert "_Verdict" in str(exc_info.value)


def test_score_wraps_ai_function_error_as_judge_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolates the ``AIFunctionError`` -> ``JudgeError`` mapping from the
    specific Strands forced-retry behavior covered above: whatever reason
    the underlying ``AIFunction`` cycle raises ``AIFunctionError`` for
    (post-condition validation exhausted, etc.), ``score`` must wrap it."""

    async def _boom(self: AIFunction, *args: object, **kwargs: object) -> object:
        del self, args, kwargs
        raise AIFunctionError("synthetic failure", function_name="judge")

    monkeypatch.setattr(AIFunction, "__call__", _boom)
    judge = AIFunctionJudge(model="unused")

    with pytest.raises(JudgeError) as exc_info:
        asyncio.run(judge.score("prompt", _Verdict))

    assert isinstance(exc_info.value.cause, AIFunctionError)
    assert "synthetic failure" in str(exc_info.value.cause)


# ── failure policy: wrap every Exception, then trip a circuit breaker ───────


class _FakeNoCredentialsError(Exception):
    """Stand-in for ``botocore.exceptions.NoCredentialsError``.

    ``botocore`` is not a declared dependency of this package's tests, and the
    point under test is *shape*, not identity: a credential failure is a plain
    ``Exception`` that is neither ``AIFunctionError`` nor
    ``StructuredOutputException``, and before the widened wrap it escaped
    ``score`` entirely and aborted the batch.
    """


def _script_calls(monkeypatch: pytest.MonkeyPatch, outcomes: list[object]) -> None:
    """Replace ``AIFunction.__call__`` with a scripted outcome sequence.

    Each entry is either an exception instance (raised on that call) or a
    value (returned). This is the same seam
    ``test_score_wraps_ai_function_error_as_judge_error`` uses, extended to a
    sequence so consecutive-failure behaviour can be driven without a model.
    """
    remaining = list(outcomes)

    async def _scripted(self: AIFunction, *args: object, **kwargs: object) -> object:
        del self, args, kwargs
        outcome = remaining.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(AIFunction, "__call__", _scripted)


def test_score_wraps_plain_runtime_error_as_judge_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure that is neither ``AIFunctionError`` nor
    ``StructuredOutputException`` is still wrapped. This is the case that used
    to escape ``score`` and abort the whole evaluation batch."""
    boom = RuntimeError("something the judge layer never anticipated")
    _script_calls(monkeypatch, [boom])
    judge = AIFunctionJudge(model="unused")

    with pytest.raises(JudgeError) as exc_info:
        asyncio.run(judge.score("prompt", _Verdict))

    assert exc_info.value.cause is boom
    assert exc_info.value.__cause__ is boom
    assert "_Verdict" in str(exc_info.value)


def test_score_wraps_credential_style_error_as_judge_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential/permission/network-shaped model-layer error degrades to a
    single missing score rather than killing the run (the breaker, not the
    wrap, is what stops a *persistent* one)."""
    creds = _FakeNoCredentialsError("Unable to locate credentials")
    _script_calls(monkeypatch, [creds])
    judge = AIFunctionJudge(model="unused")

    with pytest.raises(JudgeError) as exc_info:
        asyncio.run(judge.score("prompt", _Verdict))

    assert exc_info.value.cause is creds


def test_score_does_not_wrap_cancelled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """``asyncio.CancelledError`` is a ``BaseException``, not an
    ``Exception``: it must propagate untouched (cancelling a run has to stay
    possible) and must not count against the circuit breaker."""
    _script_calls(monkeypatch, [asyncio.CancelledError()])
    judge = AIFunctionJudge(model="unused")

    async def _run() -> asyncio.CancelledError:
        try:
            await judge.score("prompt", _Verdict)
        except asyncio.CancelledError as exc:
            return exc
        raise AssertionError("score returned instead of propagating CancelledError")

    raised = asyncio.run(_run())
    assert type(raised) is asyncio.CancelledError
    assert not isinstance(raised, (JudgeError, JudgeUnavailableError))
    assert judge._consecutive_failures == 0  # noqa: SLF001


def test_failures_below_the_threshold_raise_judge_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _script_calls(monkeypatch, [RuntimeError("1"), RuntimeError("2")])
    judge = AIFunctionJudge(model="unused", max_consecutive_failures=3)

    for _ in range(2):
        with pytest.raises(JudgeError):
            asyncio.run(judge.score("prompt", _Verdict))


def test_nth_consecutive_failure_raises_judge_unavailable_instead_of_judge_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins the decision: the call that *trips* the threshold reports at the
    breaker's severity instead of as a ``JudgeError`` -- one failure, one
    exception, at its most severe level."""
    last = RuntimeError("3")
    _script_calls(monkeypatch, [RuntimeError("1"), RuntimeError("2"), last])
    judge = AIFunctionJudge(model="unused", max_consecutive_failures=3)

    for _ in range(2):
        with pytest.raises(JudgeError):
            asyncio.run(judge.score("prompt", _Verdict))

    with pytest.raises(JudgeUnavailableError) as exc_info:
        asyncio.run(judge.score("prompt", _Verdict))

    assert not isinstance(exc_info.value, JudgeError)
    assert exc_info.value.cause is last
    assert exc_info.value.__cause__ is last
    message = str(exc_info.value)
    assert "3 consecutive" in message
    assert "threshold of 3" in message


def test_judge_unavailable_error_is_not_caught_by_except_judge_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The type relationship *is* the mechanism: every judged metric narrows
    its handler to ``JudgeError``, so ``JudgeUnavailableError`` must not be a
    subclass of it or the breaker would be silently swallowed."""
    assert not issubclass(JudgeUnavailableError, JudgeError)
    _script_calls(monkeypatch, [RuntimeError("boom")])
    judge = AIFunctionJudge(model="unused", max_consecutive_failures=1)

    async def _run() -> str:
        try:
            await judge.score("prompt", _Verdict)
        except JudgeError:
            return "swallowed"
        except JudgeUnavailableError:
            return "propagated"
        return "no error"

    assert asyncio.run(_run()) == "propagated"


def test_a_success_resets_the_consecutive_failure_counter(monkeypatch: pytest.MonkeyPatch) -> None:
    """fail, fail, succeed, fail, fail -> still ``JudgeError`` at threshold 3;
    only the *third consecutive* failure trips the breaker."""
    _script_calls(
        monkeypatch,
        [
            RuntimeError("1"),
            RuntimeError("2"),
            _Verdict(score=0.5, rationale="recovered"),
            RuntimeError("3"),
            RuntimeError("4"),
            RuntimeError("5"),
        ],
    )
    judge = AIFunctionJudge(model="unused")  # default threshold: 3

    for _ in range(2):
        with pytest.raises(JudgeError):
            asyncio.run(judge.score("prompt", _Verdict))

    assert asyncio.run(judge.score("prompt", _Verdict)) == _Verdict(score=0.5, rationale="recovered")
    assert judge._consecutive_failures == 0  # noqa: SLF001

    for _ in range(2):
        with pytest.raises(JudgeError) as exc_info:
            asyncio.run(judge.score("prompt", _Verdict))
        assert not isinstance(exc_info.value, JudgeUnavailableError)

    # ...and the counter really did restart from the success, so the third
    # post-reset failure is the one that trips it.
    with pytest.raises(JudgeUnavailableError):
        asyncio.run(judge.score("prompt", _Verdict))


def test_default_max_consecutive_failures_is_three() -> None:
    """Regression pin on the shipped default (documented in J4)."""
    judge = AIFunctionJudge(model="unused")
    assert judge._max_consecutive_failures == 3  # noqa: SLF001


def test_max_consecutive_failures_none_never_trips_the_breaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """``None`` restores the old never-abort behaviour: every failure, however
    many in a row, is a plain ``JudgeError``."""
    _script_calls(monkeypatch, [RuntimeError(str(i)) for i in range(25)])
    judge = AIFunctionJudge(model="unused", max_consecutive_failures=None)

    for _ in range(25):
        with pytest.raises(JudgeError) as exc_info:
            asyncio.run(judge.score("prompt", _Verdict))
        assert not isinstance(exc_info.value, JudgeUnavailableError)


def test_max_consecutive_failures_of_one_trips_on_the_first_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    _script_calls(monkeypatch, [RuntimeError("boom")])
    judge = AIFunctionJudge(model="unused", max_consecutive_failures=1)

    with pytest.raises(JudgeUnavailableError):
        asyncio.run(judge.score("prompt", _Verdict))


@pytest.mark.parametrize("threshold", [0, -1, -100])
def test_max_consecutive_failures_below_one_is_a_construction_error(threshold: int) -> None:
    """A threshold under one could only mean "trip before failing"; that is a
    caller error, caught at construction rather than mid-run."""
    with pytest.raises(ValueError, match="max_consecutive_failures"):
        AIFunctionJudge(model="unused", max_consecutive_failures=threshold)


def test_max_consecutive_failures_is_keyword_only() -> None:
    """It must not shift the two existing positional parameters
    (``model``, ``coordinator``) for callers that pass them positionally."""
    with pytest.raises(TypeError):
        AIFunctionJudge("unused", None, 3)  # type: ignore[misc]


# ── end to end: what a judged metric does with each of the two types ────────


class _RaisingJudge:
    """A ``Judge`` whose ``score`` always raises one preconfigured exception."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def score(self, prompt: str, schema: type[BaseModel], *, system_prompt: str | None = None) -> BaseModel:
        del prompt, schema, system_prompt
        raise self._exc


def _completeness_input() -> Level1Input:
    skill = Skill(
        path=Path("skills/example-skill"),
        frontmatter={"name": "example-skill", "description": "An example skill."},
        body="## Overview\nDo the thing.\n",
        raw="---\nname: example-skill\n---\n## Overview\nDo the thing.\n",
    )
    return Level1Input(skill_spec=skill)


def test_metric_degrades_to_none_on_a_one_off_judge_error() -> None:
    """The contract the widened wrap preserves: a single failure costs one
    score, not the run."""
    metric = CompletenessMetric(judge=_RaisingJudge(JudgeError("synthetic one-off failure")))

    result = asyncio.run(metric.compute_async(_completeness_input()))

    assert result.score is None
    assert result.passed is None
    assert "synthetic one-off failure" in result.metadata["error"]


def test_judge_unavailable_error_propagates_out_of_a_metric() -> None:
    """The point of the whole design: the breaker is not absorbed by the
    metric's ``except JudgeError`` ladder -- it leaves ``compute_async`` and
    stops the evaluation run."""
    metric = CompletenessMetric(judge=_RaisingJudge(JudgeUnavailableError("judge is unusable")))

    with pytest.raises(JudgeUnavailableError, match="judge is unusable"):
        asyncio.run(metric.compute_async(_completeness_input()))


# ── StubJudge ────────────────────────────────────────────────────────────────


def test_stub_judge_returns_canned_instance_for_registered_schema() -> None:
    canned = _Verdict(score=1.0, rationale="perfect")
    stub = StubJudge({_Verdict: canned})

    result = asyncio.run(stub.score("any prompt", _Verdict))

    assert result is canned


def test_stub_judge_calls_factory_fresh_each_time() -> None:
    calls = {"n": 0}

    def _factory() -> _Verdict:
        calls["n"] += 1
        return _Verdict(score=calls["n"] / 10, rationale=f"call {calls['n']}")

    stub = StubJudge({_Verdict: _factory})

    async def _run() -> tuple[_Verdict, _Verdict]:
        first = await stub.score("p1", _Verdict)
        second = await stub.score("p2", _Verdict)
        return first, second

    first, second = asyncio.run(_run())
    assert first.rationale == "call 1"
    assert second.rationale == "call 2"
    assert first is not second


def test_stub_judge_raises_key_error_for_unregistered_schema() -> None:
    stub = StubJudge({_Verdict: _Verdict(score=0.0, rationale="")})

    with pytest.raises(KeyError):
        asyncio.run(stub.score("prompt", _OtherVerdict))


def test_stub_judge_records_calls_in_order() -> None:
    stub = StubJudge({_Verdict: _Verdict(score=0.0, rationale=""), _OtherVerdict: _OtherVerdict(passed=False)})

    async def _run() -> None:
        await stub.score("first prompt", _Verdict)
        await stub.score("second prompt", _OtherVerdict, system_prompt="be concise")

    asyncio.run(_run())
    assert stub.calls == [
        ("first prompt", _Verdict, None),
        ("second prompt", _OtherVerdict, "be concise"),
    ]


def test_judge_reads_no_environment_variable_for_its_model() -> None:
    """The judge takes its model from its argument or the ambient config only.

    Pinning a model through an environment variable would make a score depend
    on ambient state that never appears in the run's own record, so there is
    deliberately no such knob: pass ``model=`` to fix the judge.
    """
    judge = AIFunctionJudge()
    assert judge._config.model is None  # noqa: SLF001


# ===========================================================================
# quality.py: the Tier-1 metrics
# ===========================================================================


_VerdictT = TypeVar("_VerdictT", bound=BaseModel)


class _FailingJudge:
    """A ``Judge`` that always raises ``JudgeError`` -- exercises the
    score=None / metadata-recorded-failure mapping without a real model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, type[BaseModel], str | None]] = []

    async def score(self, prompt: str, schema: type[_VerdictT], *, system_prompt: str | None = None) -> _VerdictT:
        self.calls.append((prompt, schema, system_prompt))
        raise JudgeError(f"synthetic judge failure for {schema.__name__}")


def _run(coro):  # noqa: ANN001, ANN201
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# FormatAlignmentMetric
# ---------------------------------------------------------------------------


def _score_format(skill: Skill):  # noqa: ANN201
    metric = FormatAlignmentMetric()
    return metric.compute(Level1Input(skill_spec=skill))


def test_well_formed_skill_passes(tmp_path: Path):
    good = tmp_path / "good-skill"
    good.mkdir()
    (good / "SKILL.md").write_text(
        "---\n"
        "name: good-skill\n"
        "description: A well-formed skill with all required sections for the format-alignment test.\n"
        "---\n"
        "## Overview\nA short overview.\n"
        "## When to use this skill\nWhen the test runs.\n"
        "## Instructions\n"
        "step one and step two and step three and step four and step five "
        "and step six and step seven and step eight and step nine and step "
        "ten and step eleven and step twelve and step thirteen and step "
        "fourteen and step fifteen and step sixteen and step seventeen and "
        "step eighteen and step nineteen and step twenty and step twentyone "
        "and step twentytwo and step twentythree.\n"
    )
    skill = load_skill(good)
    result = _score_format(skill)
    assert result.score == 1.0
    assert result.passed


def test_missing_required_section_fails(tmp_path: Path):
    bad = tmp_path / "broken-skill"
    bad.mkdir()
    (bad / "SKILL.md").write_text(
        "---\n"
        "name: broken-skill\n"
        "description: A broken skill missing required sections for testing.\n"
        "---\n"
        "## Overview\n"
        "Just an overview.\n"
    )
    skill = load_skill(bad)
    result = _score_format(skill)
    assert result.score is not None
    assert result.score < 1.0
    assert any("Instructions" in v for v in result.evidence)


def test_invalid_name_fails(tmp_path: Path):
    bad = tmp_path / "bad-name"
    bad.mkdir()
    (bad / "SKILL.md").write_text(
        "---\n"
        "name: BadName_With_Caps\n"
        "description: This description is at least five words long for the test.\n"
        "---\n"
        "## Overview\nx\n## When to use this skill\nx\n## Instructions\n"
        "step one and step two and step three and step four and step five "
        "and step six and step seven and step eight and step nine and step "
        "ten and step eleven and step twelve and step thirteen and step "
        "fourteen and step fifteen and step sixteen and step seventeen and "
        "step eighteen and step nineteen and step twenty and step twentyone "
        "and step twentytwo and step twentythree.\n"
    )
    skill = load_skill(bad)
    result = _score_format(skill)
    assert result.score is not None
    assert result.score < 1.0
    assert any("name" in v.lower() for v in result.evidence)


def test_format_alignment_rejects_non_skill_input():
    metric = FormatAlignmentMetric()
    with pytest.raises(TypeError):
        metric.compute(Level1Input(skill_spec={"not": "a skill"}))


def test_format_alignment_never_constructs_a_judge():
    """Deterministic metric -- no ``judge`` field/attribute at all."""
    metric = FormatAlignmentMetric()
    assert not hasattr(metric, "judge")


# ---------------------------------------------------------------------------
# _tfidf_cosine golden values
# ---------------------------------------------------------------------------


def test_identical_documents_have_full_lexical_similarity():
    similarity, _ = _tfidf_cosine("same words throughout", ["same words throughout"])
    assert similarity == pytest.approx(1.0)


def test_non_identical_documents_keep_legacy_idf_weighting():
    similarity, _ = _tfidf_cosine(
        "alpha beta gamma",
        ["alpha beta delta", "epsilon zeta"],
    )
    assert similarity == pytest.approx(0.214099491)


# ---------------------------------------------------------------------------
# DistinctivenessMetric -- library-backed cases use synthetic fixtures (see
# module docstring)
# ---------------------------------------------------------------------------

_CSV_ANALYZER_MD = (
    "---\n"
    "name: csv-analyzer\n"
    "description: Analyze CSV files and produce summary statistics reports for tabular datasets.\n"
    "---\n"
    "## Overview\nParses CSV files and computes summary statistics.\n"
    "## When to use this skill\nWhen a user provides a CSV file and asks for analysis.\n"
    "## Instructions\n"
    "Load the CSV with pandas, compute column-wise means, medians, and "
    "standard deviations, detect missing values, and render a summary "
    "table. Flag outliers using the interquartile range method and note "
    "any columns with high null rates for the user's attention before "
    "producing the final report.\n"
)

_INVOICE_PARSER_MD = (
    "---\n"
    "name: invoice-parser\n"
    "description: Extract structured line items and totals from scanned invoice PDF documents.\n"
    "---\n"
    "## Overview\nExtracts line items and totals from invoice PDFs.\n"
    "## When to use this skill\nWhen a user uploads an invoice and wants structured data.\n"
    "## Instructions\n"
    "Run OCR over the PDF pages, locate the vendor name and invoice date, "
    "extract each line item's description, quantity, and unit price, sum "
    "the subtotal and tax lines, and cross-check the extracted total "
    "against the printed grand total before returning the structured "
    "record to the user.\n"
)


def _distinct_library(tmp_path: Path) -> SkillLibrary:
    root = tmp_path / "library"
    root.mkdir()
    (root / "csv-analyzer").mkdir()
    (root / "csv-analyzer" / "SKILL.md").write_text(_CSV_ANALYZER_MD)
    (root / "invoice-parser").mkdir()
    (root / "invoice-parser" / "SKILL.md").write_text(_INVOICE_PARSER_MD)
    return load_skill_library(root)


def test_skill_compared_against_library_with_stub_high_distinctness(tmp_path: Path):
    library = _distinct_library(tmp_path)
    candidate = library.by_name("csv-analyzer")
    assert candidate is not None

    stub = StubJudge({_DistinctVerdict: _DistinctVerdict(score=1.0, most_similar_skill="", reasoning="stub")})
    metric = DistinctivenessMetric(judge=stub)
    result = _run(metric.compute_async(Level1Input(skill_spec=candidate, skill_library=library)))

    assert result.score is not None
    assert result.score > 0.5
    assert result.metadata == {"judge": "ai_function/v1"}


def test_duplicate_skill_scored_low(tmp_path: Path):
    library = _distinct_library(tmp_path)
    original = library.by_name("csv-analyzer")
    assert original is not None

    dup_dir = tmp_path / "skill-dup"
    dup_dir.mkdir()
    (dup_dir / "SKILL.md").write_text(original.raw)
    duplicate = load_skill(dup_dir)

    # Duplicate vs. the library (which still contains the original) gives
    # lexical similarity ~1.0; semantic=0.0 forces the combined score down.
    verdict = _DistinctVerdict(score=0.0, most_similar_skill="csv-analyzer", reasoning="dup")
    stub = StubJudge({_DistinctVerdict: verdict})
    metric = DistinctivenessMetric(judge=stub)
    result = _run(metric.compute_async(Level1Input(skill_spec=duplicate, skill_library=library)))

    assert result.score is not None
    assert result.score < 0.5


def test_distinctiveness_prompt_text_is_pinned(tmp_path: Path):
    """Both the composed user prompt (``_DISTINCT_JUDGE_PROMPT``) and the
    per-metric ``system_prompt`` (``_DISTINCT_SYSTEM_PROMPT``) reach the
    judge byte-for-byte unchanged. The prompt text is part of the
    measurement: rewording it changes every score it produces, so it is
    pinned deliberately rather than left free to drift."""
    library = _distinct_library(tmp_path)
    candidate = library.by_name("csv-analyzer")
    assert candidate is not None
    others = [s for s in library.skills if s.path != candidate.path]
    stub = StubJudge({_DistinctVerdict: _DistinctVerdict(score=1.0, most_similar_skill="", reasoning="stub")})
    metric = DistinctivenessMetric(judge=stub)

    _run(metric.compute_async(Level1Input(skill_spec=candidate, skill_library=library)))

    expected_listing = "\n".join(f"- {s.name}: {s.description}" for s in others)
    expected_prompt = _DISTINCT_JUDGE_PROMPT.format(candidate=candidate.raw, others=expected_listing)
    assert stub.calls == [(expected_prompt, _DistinctVerdict, _DISTINCT_SYSTEM_PROMPT)]


def test_distinctiveness_skips_judge_when_library_has_no_other_skills(tmp_path: Path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "csv-analyzer").mkdir()
    (root / "csv-analyzer" / "SKILL.md").write_text(_CSV_ANALYZER_MD)
    library = load_skill_library(root)
    candidate = library.by_name("csv-analyzer")
    assert candidate is not None

    stub = StubJudge({})  # no verdicts registered -- KeyError if ever called
    metric = DistinctivenessMetric(judge=stub)
    result = _run(metric.compute_async(Level1Input(skill_spec=candidate, skill_library=library)))

    assert result.score == 1.0
    assert result.passed is True
    assert stub.calls == []
    assert "judge" not in result.metadata


def test_distinctiveness_judge_error_maps_to_score_none(tmp_path: Path):
    library = _distinct_library(tmp_path)
    candidate = library.by_name("csv-analyzer")
    assert candidate is not None
    failing = _FailingJudge()
    metric = DistinctivenessMetric(judge=failing)

    result = _run(metric.compute_async(Level1Input(skill_spec=candidate, skill_library=library)))

    assert result.score is None
    assert result.passed is None
    assert result.metadata["judge"] == "ai_function/v1"
    assert "synthetic judge failure" in result.metadata["error"]
    assert failing.calls  # judge was actually invoked


def test_distinctiveness_rejects_non_skill_or_non_library_inputs(tmp_path: Path):
    library = _distinct_library(tmp_path)
    candidate = library.by_name("csv-analyzer")
    metric = DistinctivenessMetric(judge=StubJudge({}))

    with pytest.raises(TypeError):
        _run(metric.compute_async(Level1Input(skill_spec="not a skill", skill_library=library)))
    with pytest.raises(TypeError):
        _run(metric.compute_async(Level1Input(skill_spec=candidate, skill_library="not a library")))


def test_distinctiveness_lazily_builds_ambient_judge_when_omitted():
    metric = DistinctivenessMetric()
    assert isinstance(metric.judge, AIFunctionJudge)


# ---------------------------------------------------------------------------
# CompletenessMetric
# ---------------------------------------------------------------------------

_SIMPLE_SKILL_MD = (
    "---\n"
    "name: simple-skill\n"
    "description: A simple skill with enough content for judge-backed metric tests to exercise.\n"
    "---\n"
    "## Overview\nDoes a simple thing.\n"
    "## When to use this skill\nWhen you need the simple thing done.\n"
    "## Instructions\n"
    "Step one, step two, step three, step four, step five, step six, step "
    "seven, step eight, step nine, step ten, step eleven, step twelve, "
    "step thirteen, step fourteen, step fifteen, step sixteen, step "
    "seventeen, step eighteen, step nineteen, step twenty, step "
    "twentyone, step twentytwo, step twentythree, step twentyfour.\n"
)


def _simple_skill(tmp_path: Path) -> Skill:
    skill_dir = tmp_path / "simple-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(_SIMPLE_SKILL_MD)
    return load_skill(skill_dir)


def test_completeness_scores_via_stub_judge(tmp_path: Path):
    skill = _simple_skill(tmp_path)
    verdict = _CompletenessVerdict(
        scope_clarity=1.0, when_to_use=1.0, step_specificity=0.5, examples=0.0, aggregate=0.625, rationale="ok"
    )
    stub = StubJudge({_CompletenessVerdict: verdict})
    metric = CompletenessMetric(judge=stub)

    result = _run(metric.compute_async(Level1Input(skill_spec=skill)))

    assert result.score == 0.625
    assert result.passed is False  # default pass_threshold=0.7
    assert result.components == {
        "scope_clarity": 1.0,
        "when_to_use": 1.0,
        "step_specificity": 0.5,
        "examples": 0.0,
    }
    assert result.metadata == {"judge": "ai_function/v1"}


def test_completeness_prompt_text_is_pinned(tmp_path: Path):
    """Both the composed user prompt (``_COMPLETENESS_USER_PROMPT``) and the
    module-level ``_COMPLETENESS_SYSTEM_PROMPT`` reach the judge
    byte-for-byte unchanged. The prompt text is part of the measurement:
    rewording it changes every score it produces, so it is pinned
    deliberately rather than left free to drift."""
    skill = _simple_skill(tmp_path)
    verdict = _CompletenessVerdict(
        scope_clarity=1.0, when_to_use=1.0, step_specificity=1.0, examples=1.0, aggregate=1.0, rationale="ok"
    )
    stub = StubJudge({_CompletenessVerdict: verdict})
    metric = CompletenessMetric(judge=stub)

    _run(metric.compute_async(Level1Input(skill_spec=skill)))

    expected = _COMPLETENESS_USER_PROMPT.format(expected=ExpectedSkillFormat().as_prompt_block(), skill_text=skill.raw)
    assert stub.calls == [(expected, _CompletenessVerdict, _COMPLETENESS_SYSTEM_PROMPT)]


def test_completeness_judge_error_maps_to_score_none(tmp_path: Path):
    skill = _simple_skill(tmp_path)
    failing = _FailingJudge()
    metric = CompletenessMetric(judge=failing)

    result = _run(metric.compute_async(Level1Input(skill_spec=skill)))

    assert result.score is None
    assert result.passed is None
    assert result.metadata["judge"] == "ai_function/v1"
    assert "synthetic judge failure" in result.metadata["error"]


def test_completeness_lazily_builds_ambient_judge_when_omitted():
    metric = CompletenessMetric()
    assert isinstance(metric.judge, AIFunctionJudge)


def test_completeness_rejects_non_skill_input():
    metric = CompletenessMetric(judge=StubJudge({}))
    with pytest.raises(TypeError):
        _run(metric.compute_async(Level1Input(skill_spec=42)))


def test_completeness_compute_async_scores_a_complete_skill(tmp_path: Path):
    skill = _simple_skill(tmp_path)
    verdict = _CompletenessVerdict(
        scope_clarity=1.0, when_to_use=1.0, step_specificity=1.0, examples=1.0, aggregate=1.0, rationale="great"
    )
    stub = StubJudge({_CompletenessVerdict: verdict})
    metric = CompletenessMetric(judge=stub)

    async_result = _run(metric.compute_async(Level1Input(skill_spec=skill)))
    assert async_result.score == 1.0
    assert async_result.passed is True


# ===========================================================================
# evaluation.py: trace_from_events
# ===========================================================================


_TID = ThreadId("t-1")


def _tool_call(name: str, arguments: dict | None = None, *, ts: float = 1.0, tu: str = "tu-1"):
    return ToolCallEvent(thread_id=_TID, tool_use_id=tu, tool_name=name, arguments=arguments or {}, timestamp=ts)


def _tool_result(tu: str = "tu-1", *, status: str = "success", text: str = "ok", ts: float = 1.5):
    return ToolResultEvent(
        thread_id=_TID,
        tool_use_id=tu,
        status=status,
        content=[{"text": text}],
        timestamp=ts,
    )


def _assistant_complete(text: str, *, ts: float = 0.5):
    return MessageAssistantCompleteEvent(thread_id=_TID, content=[{"text": text}], timestamp=ts)


# ---------------------------------------------------------------------------
# Transcript order + entry types
# ---------------------------------------------------------------------------


def test_transcript_preserves_event_order_and_entry_types():
    events = [
        _assistant_complete("let me look at the file"),
        _tool_call("Read", {"file_path": "a.py"}, ts=1.0),
        _tool_result(text="file contents", ts=1.5),
        _tool_call("Bash", {"command": "rm a.py"}, ts=2.0, tu="tu-2"),
        _tool_result(tu="tu-2", status="error", text="permission denied", ts=2.5),
    ]
    trace = trace_from_events(events)

    assert [e.entry_type for e in trace.transcript] == [
        "thought",
        "action",
        "observation",
        "action",
        "error",
    ]
    assert trace.transcript[0].content == "let me look at the file"
    assert trace.transcript[0].action_type == "reason"
    assert trace.transcript[1].tool_name == "Read"
    assert trace.transcript[2].content == "file contents"
    assert trace.transcript[4].content == "permission denied"


def test_message_assistant_complete_yields_one_thought_per_text_block():
    event = MessageAssistantCompleteEvent(
        thread_id=_TID,
        content=[{"text": "first"}, {"text": "second"}],
        timestamp=0.0,
    )
    trace = trace_from_events([event])
    assert [e.content for e in trace.transcript] == ["first", "second"]
    assert all(e.entry_type == "thought" for e in trace.transcript)


def test_message_assistant_complete_reasoning_block_becomes_thought():
    event = MessageAssistantCompleteEvent(
        thread_id=_TID,
        content=[{"reasoningContent": {"reasoningText": {"text": "thinking it through"}}}],
        timestamp=0.0,
    )
    trace = trace_from_events([event])
    assert len(trace.transcript) == 1
    assert trace.transcript[0].entry_type == "thought"
    assert trace.transcript[0].content == "thinking it through"


# ---------------------------------------------------------------------------
# Token summation (both bases)
# ---------------------------------------------------------------------------


def test_token_counts_sum_across_multiple_usage_events():
    events = [
        TokenUsageEvent(
            thread_id=_TID,
            token_usage=RealTokenUsage(input_tokens=100, output_tokens=50, cache_read_tokens=10, cache_write_tokens=5),
        ),
        TokenUsageEvent(
            thread_id=_TID,
            token_usage=RealTokenUsage(input_tokens=200, output_tokens=25, cache_read_tokens=0, cache_write_tokens=15),
        ),
    ]
    trace = trace_from_events(events)

    # input_output basis
    assert trace.token_counts is not None
    assert trace.token_counts.input_tokens == 300
    assert trace.token_counts.output_tokens == 75

    # total (cache-inclusive) basis: cache_read 10+0=10, cache_write 5+15=20
    assert trace.metadata["total_tokens"] == 300 + 75 + 10 + 20


def test_token_counts_zero_when_no_usage_events():
    trace = trace_from_events([])
    assert trace.token_counts is not None
    assert trace.token_counts.input_tokens == 0
    assert trace.token_counts.output_tokens == 0
    assert trace.metadata["total_tokens"] == 0


# ---------------------------------------------------------------------------
# Latency arithmetic
# ---------------------------------------------------------------------------


def test_latency_seconds_from_started_and_completed():
    events = [
        StartedEvent(thread_id=_TID, timestamp=100.0),
        CompletedEvent(thread_id=_TID, timestamp=142.5),
    ]
    trace = trace_from_events(events)
    assert trace.metadata["latency_seconds"] == 42.5


def test_latency_seconds_none_when_started_missing():
    events = [CompletedEvent(thread_id=_TID, timestamp=142.5)]
    trace = trace_from_events(events)
    assert "latency_seconds" not in trace.metadata


def test_latency_seconds_none_when_completed_missing():
    events = [StartedEvent(thread_id=_TID, timestamp=100.0)]
    trace = trace_from_events(events)
    assert "latency_seconds" not in trace.metadata


def test_latency_seconds_uses_first_started_and_first_completed():
    events = [
        StartedEvent(thread_id=_TID, timestamp=100.0),
        CompletedEvent(thread_id=_TID, timestamp=110.0),
        StartedEvent(thread_id=_TID, timestamp=200.0),
        CompletedEvent(thread_id=_TID, timestamp=250.0),
    ]
    trace = trace_from_events(events)
    assert trace.metadata["latency_seconds"] == 10.0


def test_latency_seconds_none_when_lifecycle_events_are_out_of_order():
    """A leftover COMPLETED from a prior cycle precedes this cycle's STARTED:
    the first-COMPLETED/first-STARTED pairing would go negative. Negative
    means the log wasn't a single clean run, so latency is unknown -- not
    clamped to 0."""
    events = [
        CompletedEvent(thread_id=_TID, timestamp=50.0),
        StartedEvent(thread_id=_TID, timestamp=100.0),
    ]
    trace = trace_from_events(events)
    assert "latency_seconds" not in trace.metadata


# ---------------------------------------------------------------------------
# skills_used extraction
# ---------------------------------------------------------------------------


def test_skills_used_dedup_ordered_from_skill_tool_calls():
    events = [
        _tool_call("Skill", {"skill": "csv-report"}, tu="tu-1"),
        _tool_call("Skill", {"skill": "pdf-extract"}, tu="tu-2"),
        _tool_call("Skill", {"skill": "csv-report"}, tu="tu-3"),
    ]
    trace = trace_from_events(events)
    assert trace.metadata["skills_used"] == ["csv-report", "pdf-extract"]


def test_skills_used_skips_non_string_slug():
    events = [
        _tool_call("Skill", {"skill": 42}, tu="tu-1"),
        _tool_call("Skill", {}, tu="tu-2"),  # missing slug entirely
        _tool_call("Skill", {"skill": "real-skill"}, tu="tu-3"),
    ]
    trace = trace_from_events(events)
    assert trace.metadata["skills_used"] == ["real-skill"]


def test_extract_skill_slug_non_string_and_missing_return_none():
    assert _extract_skill_slug({"skill": 42}) is None
    assert _extract_skill_slug({}) is None
    assert _extract_skill_slug({"skill": "ok"}) == "ok"


def test_skill_tool_call_transcript_content_is_bare_slug():
    events = [_tool_call("Skill", {"skill": "csv-report"})]
    trace = trace_from_events(events)
    assert trace.transcript[0].content == "csv-report"


def test_skill_tool_call_with_bad_slug_falls_back_to_generic_content():
    events = [_tool_call("Skill", {"skill": 42})]
    trace = trace_from_events(events)
    # No crash, and the bad slug does not leak in as a bare-content match.
    assert trace.transcript[0].content != "42"
    assert trace.metadata["skills_used"] == []


# ---------------------------------------------------------------------------
# Action-type mapping (every tool in the map + an unknown one)
# ---------------------------------------------------------------------------


def test_action_type_mapping_matches_spec_table_exactly():
    assert _ACTION_TYPE_BY_TOOL == {
        "Read": "read",
        "LS": "read",
        "WebFetch": "read",
        "Glob": "search",
        "Grep": "search",
        "WebSearch": "search",
        "Write": "write",
        "Edit": "write",
        "MultiEdit": "write",
        "NotebookEdit": "write",
        "Bash": "execute",
        "Task": "communicate",
    }


def test_action_type_for_every_mapped_tool_and_one_unknown():
    tools_and_expected = [
        ("Read", "read"),
        ("LS", "read"),
        ("WebFetch", "read"),
        ("Glob", "search"),
        ("Grep", "search"),
        ("WebSearch", "search"),
        ("Write", "write"),
        ("Edit", "write"),
        ("MultiEdit", "write"),
        ("NotebookEdit", "write"),
        ("Bash", "execute"),
        ("Task", "communicate"),
        ("SomeUnknownTool", "other"),
        ("Skill", "other"),  # not in the map: falls through to "other"
    ]
    events = [_tool_call(name, tu=f"tu-{i}", ts=float(i)) for i, (name, _expected) in enumerate(tools_and_expected)]
    trace = trace_from_events(events)

    assert [e.action_type for e in trace.transcript] == [expected for _name, expected in tools_and_expected]


# ---------------------------------------------------------------------------
# Error-result mapping
# ---------------------------------------------------------------------------


def test_tool_result_error_status_maps_to_error_entry_type():
    events = [_tool_result(status="error", text="boom")]
    trace = trace_from_events(events)
    assert trace.transcript[0].entry_type == "error"


def test_tool_result_success_status_maps_to_observation_entry_type():
    events = [_tool_result(status="success", text="fine")]
    trace = trace_from_events(events)
    assert trace.transcript[0].entry_type == "observation"


# ---------------------------------------------------------------------------
# Deserializer application on RESULT payload
# ---------------------------------------------------------------------------


def test_output_defaults_to_identity_deserializer():
    events = [ResultEvent(thread_id=_TID, payload="raw-string-payload")]
    trace = trace_from_events(events)
    assert trace.output == "raw-string-payload"


def test_output_none_when_no_result_event():
    trace = trace_from_events([])
    assert trace.output is None


def test_output_applies_custom_deserializer():
    events = [ResultEvent(thread_id=_TID, payload='{"answer": 42}')]

    import json

    trace = trace_from_events(events, deserialize_result=json.loads)
    assert trace.output == {"answer": 42}


def test_output_uses_last_result_event_when_more_than_one():
    events = [
        ResultEvent(thread_id=_TID, payload="first"),
        ResultEvent(thread_id=_TID, payload="second"),
    ]
    trace = trace_from_events(events)
    assert trace.output == "second"


# ---------------------------------------------------------------------------
# Oracle passthrough
# ---------------------------------------------------------------------------


def test_oracle_passthrough():
    oracle = OracleResult(passed=True, score=1.0)
    assert trace_from_events([], oracle_result=oracle).oracle_result is oracle
    assert trace_from_events([]).oracle_result is None


# ---------------------------------------------------------------------------
# Alternate skill-tool names normalize onto "Skill"
# ---------------------------------------------------------------------------


def test_alternate_skill_tool_name_normalizes_to_canonical_skill(monkeypatch):
    """If _SKILL_TOOL_NAMES ever grows an alternate name, the emitted
    TraceEntry.tool_name is still the canonical "Skill"."""
    monkeypatch.setattr(trace_module, "_SKILL_TOOL_NAMES", ("Skill", "InvokeSkill"))

    events = [_tool_call("InvokeSkill", {"skill": "csv-report"}, ts=1.0)]
    trace = trace_from_events(events)

    assert trace.transcript[0].tool_name == "Skill"
    assert trace.metadata["skills_used"] == ["csv-report"]
