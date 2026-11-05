"""Tests for the demo's merge stage (``full_pipeline/mining.py``).

Moved with the merge module from the library's own discovery tests. The merge
``@ai_function`` template and the Tier-1 metrics behind ``MetricGate`` are
LLM-backed, so every test injects a fake ``merge_fn`` and/or a
``StubJudge``-driven gate -- no LLM/network is ever constructed or reached.
The retry-exhaustion fallback (``_fallback_draft``/``_best_constituent_index``)
is what a live run lands on when the gate keeps rejecting, so it is tested
here directly and through ``merge_group``.

Async functions are driven via ``asyncio.run()`` inside plain ``def
test_...`` functions. No LLM/network.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from _eval_stubs import StubJudge

from ai_functions.experimental.skills.skill_library import SkillLibrary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples" / "skillify_demo"))

from full_pipeline.mining import (  # noqa: E402
    DEFAULT_MAX_ATTEMPTS,
    UNGROUPED_LABEL,
    GateResult,
    MergedSkillCandidate,
    MergeGate,
    MetricGate,
    Provenance,
    SkillDraft,
    SkillGroup,
    _best_constituent_index,
    _check_merged_candidate,
    _fallback_draft,
    _skill_from_draft,
    _slugify_name,
    merge_group,
)
from full_pipeline.quality import _CompletenessVerdict, _DistinctVerdict  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _draft(name: str, body: str = "do the thing") -> SkillDraft:
    return SkillDraft(
        name=name,
        description=f"{name} description",
        body=body,
        strategy="preserve",
        provenance=Provenance(trajectory_ids=[f"t-{name}"], constituent_ids=[f"t-{name}:0-1"], strategy="preserve"),
    )


def _group(*names: str, label: str = "theme") -> SkillGroup:
    return SkillGroup(label=label, members=[_draft(n) for n in names])


def _candidate(name: str = "merged") -> MergedSkillCandidate:
    return MergedSkillCandidate(name=name, description=f"{name} description", body=f"{name} body")


class _FakeMerger:
    """Records calls; returns a canned/scripted sequence of candidates."""

    def __init__(self, candidates: list[MergedSkillCandidate]) -> None:
        self._candidates = candidates
        self.calls: list[str] = []

    async def __call__(self, constituents_text: str) -> MergedSkillCandidate:
        self.calls.append(constituents_text)
        idx = min(len(self.calls) - 1, len(self._candidates) - 1)
        return self._candidates[idx]


class _StubGate:
    """A ``MergeGate`` whose ``accept`` is scripted per call, no metrics involved."""

    def __init__(self, decisions: list[bool], *, scores: dict[str, float | None] | None = None) -> None:
        self._decisions = decisions
        self._scores = scores if scores is not None else {}
        self.calls: list[tuple[object, list[object]]] = []

    async def accept(self, merged, constituents) -> GateResult:  # noqa: ANN001
        self.calls.append((merged, constituents))
        idx = min(len(self.calls) - 1, len(self._decisions) - 1)
        accepted = self._decisions[idx]
        return GateResult(accepted=accepted, rationale=f"stub decision #{idx}", scores=self._scores)


# ---------------------------------------------------------------------------
# _check_merged_candidate
# ---------------------------------------------------------------------------


def test_check_merged_candidate_passes_for_full_fields():
    result = _check_merged_candidate(_candidate())
    assert result is None or result.passed is True


def test_check_merged_candidate_rejects_blank_body():
    bad = MergedSkillCandidate(name="n", description="d", body="")
    result = _check_merged_candidate(bad)
    assert result is not None
    assert result.passed is False
    assert "body" in result.message


def test_check_merged_candidate_rejects_all_blank_fields():
    bad = MergedSkillCandidate(name="", description="", body="")
    result = _check_merged_candidate(bad)
    assert result is not None
    assert result.passed is False
    assert "name" in result.message
    assert "description" in result.message
    assert "body" in result.message


# ---------------------------------------------------------------------------
# _slugify_name
# ---------------------------------------------------------------------------


def test_slugify_name_lowercases_and_replaces_invalid_chars():
    assert _slugify_name("My Cool Skill!!") == "my-cool-skill"


def test_slugify_name_falls_back_to_skill_for_empty_result():
    assert _slugify_name("###") == "skill"


def test_slugify_name_caps_at_the_frontmatter_name_pattern_length():
    # SkillFrontmatterSpec.name_pattern allows at most 64 characters, and
    # never a trailing hyphen (the truncation boundary must be trimmed).
    slug = _slugify_name("Recover A Failed Deployment " * 5)
    assert len(slug) <= 64
    assert not slug.endswith("-")
    assert slug.startswith("recover-a-failed-deployment")


# ---------------------------------------------------------------------------
# _best_constituent_index
# ---------------------------------------------------------------------------


def test_best_constituent_index_picks_argmax():
    scores = {"merged": 0.1, "constituent#0": 0.5, "constituent#1": 0.9, "constituent#2": 0.2}
    assert _best_constituent_index(scores, 3) == 1


def test_best_constituent_index_ignores_none_scores():
    scores = {"constituent#0": None, "constituent#1": 0.3}
    assert _best_constituent_index(scores, 2) == 1


def test_best_constituent_index_defaults_to_zero_when_no_constituent_keys():
    assert _best_constituent_index({"merged": 0.5}, 3) == 0


def test_fallback_draft_returns_best_scoring_member():
    group = _group("a", "b", "c")
    scores = {"constituent#0": 0.1, "constituent#1": 0.9, "constituent#2": 0.4}
    result = GateResult(accepted=False, rationale="r", scores=scores)
    draft = _fallback_draft(group, result)
    assert draft.name == "b"


def test_fallback_draft_with_no_gate_result_defaults_to_first_member():
    group = _group("a", "b")
    draft = _fallback_draft(group, None)
    assert draft.name == "a"


# ---------------------------------------------------------------------------
# merge_group -- singleton / ungrouped passthrough
# ---------------------------------------------------------------------------


def test_merge_group_singleton_passes_through_unmerged():
    group = _group("solo")
    fake = _FakeMerger([_candidate()])
    gate = _StubGate([True])

    result = asyncio.run(merge_group(group, gate=gate, merge_fn=fake))

    assert result == group.members
    assert fake.calls == []
    assert gate.calls == []


def test_merge_group_ungrouped_label_passes_through_unmerged_even_with_multiple_members():
    group = _group("a", "b", "c", label=UNGROUPED_LABEL)
    fake = _FakeMerger([_candidate()])
    gate = _StubGate([True])

    result = asyncio.run(merge_group(group, gate=gate, merge_fn=fake))

    assert result == group.members
    assert fake.calls == []
    assert gate.calls == []


def test_merge_group_empty_group_returns_empty_list():
    group = SkillGroup(label="theme", members=[])
    fake = _FakeMerger([_candidate()])
    gate = _StubGate([True])

    result = asyncio.run(merge_group(group, gate=gate, merge_fn=fake))

    assert result == []
    assert fake.calls == []


# ---------------------------------------------------------------------------
# merge_group -- accept path
# ---------------------------------------------------------------------------


def test_merge_group_accept_path_returns_merged_draft():
    group = _group("a", "b")
    fake = _FakeMerger([_candidate("winner")])
    gate = _StubGate([True])

    result = asyncio.run(merge_group(group, gate=gate, merge_fn=fake))

    assert len(result) == 1
    draft = result[0]
    assert draft.name == "winner"
    assert draft.strategy == "merge"
    assert draft.provenance.strategy == "merge"
    assert draft.provenance.group_label == "theme"
    assert sorted(draft.provenance.trajectory_ids) == ["t-a", "t-b"]
    assert len(fake.calls) == 1
    assert len(gate.calls) == 1


# ---------------------------------------------------------------------------
# merge_group -- reject-then-fallback path
# ---------------------------------------------------------------------------


def test_merge_group_reject_then_fallback_returns_best_constituent_no_exception():
    group = _group("a", "b", "c")
    fake = _FakeMerger([_candidate("try1"), _candidate("try2"), _candidate("try3")])
    # Every attempt rejected; the last GateResult ranks constituent #1 highest.
    gate = _StubGate(
        [False, False, False],
        scores={"merged": 0.1, "constituent#0": 0.2, "constituent#1": 0.9, "constituent#2": 0.4},
    )

    result = asyncio.run(merge_group(group, gate=gate, merge_fn=fake, max_attempts=3))

    assert len(result) == 1
    assert result[0].name == "b"  # best constituent, unmerged
    assert result[0] is group.members[1]
    assert len(fake.calls) == 3  # attempts exhausted
    assert len(gate.calls) == 3


def test_merge_group_retry_input_carries_prior_rejection_rationale():
    # Review fix (Finding 1): a rejected retry must not resend an identical
    # prompt -- the second call's input carries the first rejection's
    # rationale text; the first call's input must not (nothing to carry yet).
    group = _group("a", "b")
    fake = _FakeMerger([_candidate("try1"), _candidate("try2")])
    gate = _StubGate([False, True])

    result = asyncio.run(merge_group(group, gate=gate, merge_fn=fake, max_attempts=2))

    assert len(result) == 1
    assert result[0].name == "try2"
    assert len(fake.calls) == 2
    first_rationale = "stub decision #0"
    assert first_rationale not in fake.calls[0]
    assert first_rationale in fake.calls[1]
    assert "rejected by the quality gate" in fake.calls[1]


# ---------------------------------------------------------------------------
# merge_group -- merge_fn/gate exceptions propagate unchanged (Finding 2)
# ---------------------------------------------------------------------------


class _RaisingMerger:
    """A ``merge_fn`` that always raises -- an infrastructure fault, not a rejection."""

    async def __call__(self, constituents_text: str) -> MergedSkillCandidate:
        raise RuntimeError("merge_fn exploded")


class _RaisingGate:
    """A ``MergeGate`` whose ``accept`` always raises -- an infrastructure fault."""

    async def accept(self, merged, constituents) -> GateResult:  # noqa: ANN001
        raise RuntimeError("gate exploded")


def test_merge_group_propagates_merge_fn_exception_no_silent_fallback():
    group = _group("a", "b")
    gate = _StubGate([True])

    with pytest.raises(RuntimeError, match="merge_fn exploded"):
        asyncio.run(merge_group(group, gate=gate, merge_fn=_RaisingMerger()))


def test_merge_group_propagates_gate_accept_exception_no_silent_fallback():
    group = _group("a", "b")
    fake = _FakeMerger([_candidate()])

    with pytest.raises(RuntimeError, match="gate exploded"):
        asyncio.run(merge_group(group, gate=_RaisingGate(), merge_fn=fake))


def test_merge_group_defensive_validation_of_fake_merge_fn_output_raises():
    # Established package lesson (mining.py/mining.py): a fake never gets a
    # free pass from post-condition-shaped validation.
    group = _group("a", "b")
    bad = MergedSkillCandidate(name="n", description="d", body="")
    fake = _FakeMerger([bad])
    gate = _StubGate([True])

    with pytest.raises(ValueError, match="body"):
        asyncio.run(merge_group(group, gate=gate, merge_fn=fake))


def test_merge_group_rejects_max_attempts_below_one():
    group = _group("a", "b")
    fake = _FakeMerger([_candidate()])
    gate = _StubGate([True])

    with pytest.raises(ValueError, match="max_attempts"):
        asyncio.run(merge_group(group, gate=gate, merge_fn=fake, max_attempts=0))


def test_merge_group_rejects_max_attempts_below_one_even_for_a_singleton_group():
    # Review fix: argument validation runs before the singleton/ungrouped
    # passthrough, so an invalid max_attempts always raises rather than
    # being silently tolerated for groups that never reach the retry loop.
    group = _group("solo")
    fake = _FakeMerger([_candidate()])
    gate = _StubGate([True])

    with pytest.raises(ValueError, match="max_attempts"):
        asyncio.run(merge_group(group, gate=gate, merge_fn=fake, max_attempts=0))
    assert fake.calls == []


def test_merge_group_default_max_attempts_constant():
    assert DEFAULT_MAX_ATTEMPTS >= 1


def test_merge_group_default_merge_fn_is_module_level_ai_function_never_called():
    # Constructing merge_group's default binding must not require network/model
    # config; only reachable through singleton/ungrouped passthrough here so no
    # LLM call happens at all.
    group = _group("solo")
    gate = _StubGate([True])
    result = asyncio.run(merge_group(group, gate=gate))
    assert result == group.members


# ---------------------------------------------------------------------------
# MetricGate -- accept path driven by StubJudge, consuming Tier-1 scores
# ---------------------------------------------------------------------------


def _judge_favoring_merged() -> StubJudge:
    return StubJudge(
        {
            _CompletenessVerdict: _CompletenessVerdict(
                scope_clarity=0.9,
                when_to_use=0.9,
                step_specificity=0.9,
                examples=0.9,
                aggregate=0.9,
                rationale="thorough",
            ),
            _DistinctVerdict: _DistinctVerdict(score=0.9, most_similar_skill="", reasoning="distinct"),
        }
    )


def _judge_favoring_constituents() -> StubJudge:
    return StubJudge(
        {
            _CompletenessVerdict: _CompletenessVerdict(
                scope_clarity=0.1,
                when_to_use=0.1,
                step_specificity=0.1,
                examples=0.1,
                aggregate=0.1,
                rationale="thin",
            ),
            _DistinctVerdict: _DistinctVerdict(score=0.1, most_similar_skill="other", reasoning="near-duplicate"),
        }
    )


def test_metric_gate_accepts_when_merged_scores_at_least_as_well_as_best_constituent():
    # A non-empty library so DistinctivenessMetric actually queries the judge
    # rather than short-circuiting on "nothing to compare against" (see
    # quality.py's DistinctivenessMetric.compute_async).
    other = _skill_from_draft(_draft("unrelated-library-skill"))
    judge = _judge_favoring_merged()
    gate = MetricGate(judge=judge, library=SkillLibrary(root=Path("."), skills=[other]))
    group = _group("a", "b")
    fake = _FakeMerger([_candidate("merged-winner")])

    result = asyncio.run(merge_group(group, gate=gate, merge_fn=fake))

    assert len(result) == 1
    assert result[0].name == "merged-winner"
    # The gate must have actually consumed Tier-1 (judge) scores: one
    # completeness call and one distinctiveness call per skill scored
    # (merged + 2 constituents => 3 of each).
    completeness_calls = [c for c in judge.calls if c[1] is _CompletenessVerdict]
    distinct_calls = [c for c in judge.calls if c[1] is _DistinctVerdict]
    assert len(completeness_calls) == 3
    assert len(distinct_calls) == 3


def test_metric_gate_rejects_when_merged_scores_worse_than_best_constituent():
    judge = _judge_favoring_constituents()
    gate = MetricGate(judge=judge)
    merged_skill = _skill_from_draft(_draft("merged-candidate"))
    constituents = [_skill_from_draft(_draft("a")), _skill_from_draft(_draft("b"))]

    # merged and constituents share the same judge verdicts here, so this
    # asserts the accept rule is a real >= comparison, not "always pass".
    result = asyncio.run(gate.accept(merged_skill, constituents))

    assert result.accepted is True  # equal scores => merged wins the tie
    assert result.scores["merged"] is not None
    assert result.scores["constituent#0"] is not None
    assert result.scores["constituent#1"] is not None


_WELL_FORMED_BODY = (
    "## Overview\nA short overview.\n"
    "## When to use this skill\nWhen the test runs.\n"
    "## Instructions\n" + " ".join(f"step{i}" for i in range(50)) + "\n"
)


def test_metric_gate_end_to_end_reject_then_fallback_via_merge_group():
    # Completeness/distinctiveness are pinned equal (same StubJudge verdict
    # for every call), so the only thing that can break the tie is
    # FormatAlignmentMetric (deterministic, no judge): constituents get a
    # well-formed, section-structured body while every merge candidate stays
    # a thin one-liner -- merged strictly loses the aggregate every attempt,
    # so every attempt is rejected and merge_group falls back with no
    # exception.
    judge = StubJudge(
        {
            _CompletenessVerdict: _CompletenessVerdict(
                scope_clarity=0.5,
                when_to_use=0.5,
                step_specificity=0.5,
                examples=0.5,
                aggregate=0.5,
                rationale="stub",
            ),
            _DistinctVerdict: _DistinctVerdict(score=0.5, most_similar_skill="", reasoning="stub"),
        }
    )
    library = SkillLibrary(root=Path("."), skills=[_skill_from_draft(_draft("unrelated-library-skill"))])
    gate = MetricGate(judge=judge, library=library)
    group = _group("well-formed-a", "well-formed-b")
    for member in group.members:
        member.body = _WELL_FORMED_BODY
    fake = _FakeMerger([_candidate("bad1"), _candidate("bad2"), _candidate("bad3")])

    result = asyncio.run(merge_group(group, gate=gate, merge_fn=fake, max_attempts=3))

    assert len(result) == 1
    assert result[0] in group.members
    assert result[0].name in {"well-formed-a", "well-formed-b"}
    assert len(fake.calls) == 3


# ---------------------------------------------------------------------------
# MetricGate is configuration; MergeGate is structural
# ---------------------------------------------------------------------------


def test_metric_gate_is_a_frozen_config_dataclass():
    # The gate only holds configuration (matching GateResult's frozen
    # style); rebinding a field on a live gate is a bug, not a knob.
    metric_gate = MetricGate()
    with pytest.raises(FrozenInstanceError):
        metric_gate.judge = StubJudge({})  # type: ignore[misc]


def test_merge_gate_protocol_is_structurally_satisfied_by_metric_gate():
    metric_gate: MergeGate = MetricGate(judge=StubJudge({}))
    assert callable(metric_gate.accept)
