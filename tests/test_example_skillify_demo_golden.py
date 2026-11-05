"""Pins what the demo's full pipeline produces on a fixed recorded window.

The snapshots were captured from the in-library pipeline before it moved into
examples/skillify_demo/full_pipeline (``skillify_golden.json`` at commit
17a82b5's pipeline, ``skillify_golden_prefilter_budget.json`` just before the
move); ``FullDiscoveryMiner`` + ``PrefilteredGate(CompletionGate(...))`` must
reproduce them. Offline: discovery runs on the deterministic fakes the
discovery tests use, the Tier-1 prefilter on a stub judge, and the replay on a
scripted outcome table.

The snapshots are the guarantee that the published pipeline did not change:
a mismatch means the code changed behaviour, never that a snapshot is stale.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from _eval_stubs import StubJudge

from ai_functions import ai_function
from ai_functions.experimental.skills.evaluation import Candidate, CompletionGate
from ai_functions.experimental.skills.learner import RecordedCall
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types.events import Event

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "skillify_demo"
sys.path.insert(0, str(EXAMPLE_DIR))

from full_pipeline import FullDiscoveryMiner, PrefilteredGate  # noqa: E402
from full_pipeline.discovery import (  # noqa: E402
    DiscoveryPreset,
    FormattedSkillCandidate,
)
from full_pipeline.mining import (  # noqa: E402
    CutPoints,
    ExtractedSkill,
    GateResult,
    MergedSkillCandidate,
    Subtask,
)
from full_pipeline.quality import _CompletenessVerdict, _DistinctVerdict  # noqa: E402

GOLDEN = Path(__file__).parent / "fixtures" / "skillify_golden.json"

# ── discovery fakes (from tests/test_example_skillify_demo_pipeline.py) ──


async def _segment_fn_whole_trajectory(summary: str, message_count: int) -> CutPoints:
    return CutPoints(cuts=[])


async def _fake_extract_fn(episode_text: str, mode: str) -> ExtractedSkill:
    # Tagged by the episode's last line (the call's answer), so each call yields a distinct draft.
    tag = episode_text.strip().splitlines()[-1].strip().replace(" ", "-") or "ep"
    return ExtractedSkill(
        worth_extracting=True,
        name=f"{tag}-{mode}",
        description=f"Description for {tag} {mode}.",
        body=f"Body content for {tag} {mode} with enough words to be real.",
    )


class _FakeMergerAllAccept:
    async def __call__(self, constituents_text: str) -> MergedSkillCandidate:
        return MergedSkillCandidate(
            name="merged-skill",
            description="A merged skill from multiple constituents.",
            body="Merged body content describing the combined procedure end to end.",
        )


class _AcceptAllGate:
    async def accept(self, merged, constituents) -> GateResult:  # noqa: ANN001
        return GateResult(accepted=True, rationale="stub accept", scores={"merged": 1.0})


_GOOD_INSTRUCTIONS = " ".join(f"step{i}" for i in range(45))


def _good_candidate(name: str, overview: str = "A short overview of the skill.") -> FormattedSkillCandidate:
    return FormattedSkillCandidate(
        name=name,
        description="A well formed description with enough words.",
        overview=overview,
        when_to_use="Use this when the described situation arises.",
        instructions=_GOOD_INSTRUCTIONS,
    )


class _FakeFormatter:
    """Names drafts from ``names`` in order; the overview carries the draft's first line."""

    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.calls: list[str] = []

    async def __call__(self, draft_text: str) -> FormattedSkillCandidate:
        self.calls.append(draft_text)
        name = self._names[min(len(self.calls) - 1, len(self._names) - 1)]
        return _good_candidate(name, overview=draft_text.strip().splitlines()[0])


def _end_to_end_preset(*, format_fn) -> DiscoveryPreset:  # noqa: ANN001
    return DiscoveryPreset(
        episode_builder=Subtask(segment_fn=_segment_fn_whole_trajectory),
        modes=("preserve", "repair"),
        extract_fn=_fake_extract_fn,
        grouper=None,
        gate=_AcceptAllGate(),
        merge_fn=_FakeMergerAllAccept(),
        format_fn=format_fn,
    )


def _preset() -> DiscoveryPreset:
    return _end_to_end_preset(format_fn=_FakeFormatter(["skill-a", "skill-b", "skill-c", "skill-d"]))


# ── Tier-1 judge stub ──


def _judge(score: float) -> StubJudge:
    return StubJudge(
        {
            _CompletenessVerdict: _CompletenessVerdict(
                scope_clarity=score,
                when_to_use=score,
                step_specificity=score,
                examples=score,
                aggregate=score,
                rationale="stub",
            ),
            _DistinctVerdict: _DistinctVerdict(score=score, most_similar_skill="", reasoning="stub"),
        }
    )


# ── the fixed window ──


@ai_function[str](structured_output=False)
def _echo(task: str) -> str:
    """Echo: {task}"""


CALLS: list[tuple[str, float, str]] = [
    ("call-0", 1.0, "sorted the list"),
    ("call-1", 0.0, "gave up on the parser"),
    ("call-2", 1.0, "fixed the off by one"),
]
"""``(call_id, outcome, the model's answer)``; ``call-1`` failed."""

INSTALLED = "installed-skill"

OUTCOMES: dict[frozenset[str], dict[str, float | None]] = {
    frozenset({INSTALLED}): {"call-0": 1.0, "call-1": 0.0, "call-2": 0.0},
    frozenset({INSTALLED, "skill-a"}): {"call-0": 1.0, "call-1": 1.0, "call-2": 1.0},
    frozenset({INSTALLED, "skill-b"}): {"call-0": 1.0, "call-1": 0.0, "call-2": 0.0},
    frozenset({INSTALLED, "skill-c"}): {"call-0": 1.0, "call-1": 0.0, "call-2": None},
    frozenset({INSTALLED, "skill-d"}): {"call-0": 1.0, "call-1": 1.0, "call-2": 0.0},
}
"""Replay outcome by (installed skill-dir names, call id), the same for every repeat."""


async def _events(answer: str) -> list[Event]:
    result = await _echo.replace(model=ScriptedModel([Turn(text=answer)])).trace(task=answer)
    return list(await result.coordinator.get_events(result.thread_id))


def _installed(root: Path) -> Path:
    directory = root / INSTALLED
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\nname: {INSTALLED}\ndescription: already here\n---\nStep 1. Do it.\n")
    return directory


async def _replay(skill_dirs: Sequence[Path], calls: Sequence[RecordedCall], repeats: int) -> list[float | None]:
    table = OUTCOMES[frozenset(p.name for p in skill_dirs)]
    return [table[c.call_id] for c in calls for _ in range(repeats)]


async def test_the_pipeline_output_matches_the_golden_snapshot(tmp_path: Path):
    calls = [
        RecordedCall(call_id=call_id, outcome=outcome, passed=outcome >= 1.0, events=await _events(answer))
        for call_id, outcome, answer in CALLS
    ]

    skills = await FullDiscoveryMiner(preset=_preset())(calls, tmp_path / "candidates")
    gate = PrefilteredGate(CompletionGate(min_gain=0.02, repeats=2), min_tier1=0.0, judge=_judge(0.9))
    verdicts = await gate([Candidate(s) for s in skills], [_installed(tmp_path / "current")], calls, _replay)

    actual = {
        "skills": [{"name": s.name, "body": s.body} for s in skills],
        "verdicts": [{"name": v.candidate.skill.name, "admitted": v.admitted, "gain": v.gain} for v in verdicts],
    }
    assert actual == json.loads(GOLDEN.read_text())


# ── case 2: a prefilter rejection and a budget cut ──

GOLDEN_PREFILTER_BUDGET = Path(__file__).parent / "fixtures" / "skillify_golden_prefilter_budget.json"

TIER1_JUDGE_SCORES = {"skill-a": 0.9, "skill-b": 0.0, "skill-c": 0.6, "skill-d": 0.8}
"""Per-candidate judge score: ``skill-b`` falls below ``min_tier1``; the rest survive, ``a`` > ``d`` > ``c``."""


class _PerSkillJudge:
    """A stub judge whose Tier-1 verdicts depend on which candidate the prompt shows."""

    async def score(self, prompt: str, schema, *, system_prompt: str | None = None):  # noqa: ANN001, ANN201
        [name] = [n for n in TIER1_JUDGE_SCORES if f"name: {n}\n" in prompt]
        return _judge(TIER1_JUDGE_SCORES[name])._verdicts[schema]


def _round_view(verdicts: Sequence) -> dict:  # noqa: ANN001
    """What the round records of the verdicts: per-verdict outcome, decision order, report fields."""
    from ai_functions.experimental.skills.learner import _in_decision_order, _verdict_json

    def rendered(v) -> dict:  # noqa: ANN001
        return {k: val for k, val in _verdict_json(v).items() if k != "rationale"}

    return {
        "verdicts": [
            {
                "name": v.candidate.skill.name,
                "admitted": v.admitted,
                "gain": v.gain,
                "stage": v.scores.get("stage"),
            }
            for v in verdicts
        ],
        "cut": [v.candidate.skill.name for v in verdicts if v.scores.get("stage") == "budget"],
        "decision_order": [v.candidate.skill.name for v in _in_decision_order(verdicts)],
        "budget_cuts": [str(c) for v in verdicts for c in (v.scores.get("budget_cuts") or [])],
        "extra_calls": sum(int(v.scores.get("extra_calls") or 0) for v in verdicts),
        "rendered": {v.candidate.skill.name: rendered(v) for v in verdicts},
    }


async def test_a_prefilter_rejection_and_a_budget_cut_match_the_golden_snapshot(tmp_path: Path):
    # 3 trial calls x 2 repeats x (1 + 3 survivors) = 24 > 10: the two lowest-Tier-1
    # survivors are cut, then a trial input (3 x 2 x 2 = 12 > 10 -> 2 x 2 x 2 = 8).
    calls = [
        RecordedCall(call_id=call_id, outcome=outcome, passed=outcome >= 1.0, events=await _events(answer))
        for call_id, outcome, answer in CALLS
    ]

    skills = await FullDiscoveryMiner(preset=_preset())(calls, tmp_path / "candidates")
    gate = PrefilteredGate(
        CompletionGate(min_gain=0.02, repeats=2, budget_calls=10), min_tier1=0.5, judge=_PerSkillJudge()
    )
    verdicts = await gate([Candidate(s) for s in skills], [_installed(tmp_path / "current")], calls, _replay)

    actual = _round_view(verdicts)
    assert {v["stage"] for v in actual["verdicts"]} == {"prefilter", "budget", "trial"}
    assert actual == json.loads(GOLDEN_PREFILTER_BUDGET.read_text())
