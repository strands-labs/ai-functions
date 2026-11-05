"""``Candidate`` and ``Verdict`` -- what a ``Gate`` is handed and what it returns."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ai_functions.experimental.skills.evaluation import Candidate, CompletionGate, Verdict
from ai_functions.experimental.skills.learner import RecordedCall


def _skill(tmp_path: Path, name: str):  # noqa: ANN202
    from ai_functions.experimental.skills.skill_library import load_skill

    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: d\n---\nbody\n")
    return load_skill(d)


def test_candidate_defaults_to_a_new_skill(tmp_path: Path):
    c = Candidate(_skill(tmp_path, "a"))
    assert c.replaces is None


def test_verdict_carries_its_candidate(tmp_path: Path):
    c = Candidate(_skill(tmp_path, "a"), replaces="a")
    v = Verdict(candidate=c, admitted=True, gain=0.5, reason="ok", scores={"with": 1.0, "without": 0.5})
    assert v.candidate.replaces == "a" and v.admitted


# ── CompletionGate ──


def _calls(n: int) -> list[RecordedCall]:
    return [RecordedCall(call_id=f"c{i}", outcome=0.0, passed=False, events=[]) for i in range(n)]


def _replay_by_names(table):  # noqa: ANN001, ANN202
    """Replay fake: outcome looked up by the set of skill-dir names installed; records each arm."""
    seen = []

    async def replay(skill_dirs, calls, repeats):  # noqa: ANN001, ANN202
        key = frozenset(p.name for p in skill_dirs)
        seen.append(key)
        return [table[key]] * (len(calls) * repeats)

    replay.seen = seen
    return replay


def test_new_skill_admitted_when_it_raises_completion(tmp_path: Path):
    cand = Candidate(_skill(tmp_path, "new"))
    replay = _replay_by_names({frozenset({"old"}): 0.0, frozenset({"old", "new"}): 1.0})
    [v] = asyncio.run(CompletionGate(min_gain=0.02, repeats=2)([cand], [Path("/lib/old")], _calls(2), replay))
    assert v.admitted and v.gain == 1.0 and v.scores["stage"] == "trial"


def test_baseline_arm_runs_once_for_many_candidates(tmp_path: Path):
    a, b = Candidate(_skill(tmp_path, "a")), Candidate(_skill(tmp_path, "b"))
    replay = _replay_by_names({frozenset({"old"}): 0.0, frozenset({"old", "a"}): 1.0, frozenset({"old", "b"}): 0.0})
    vs = asyncio.run(CompletionGate()([a, b], [Path("/lib/old")], _calls(2), replay))
    assert [v.admitted for v in vs] == [True, False]
    assert replay.seen.count(frozenset({"old"})) == 1


def test_a_worse_revision_is_rejected_by_the_replace_arm(tmp_path: Path):
    # Review Focus 5. The revision lives under pending/; both arms install a dir named "x",
    # so arms are told apart by path: only the with-arm contains the pending copy.
    rev = Candidate(_skill(tmp_path / "pending", "x"), replaces="x")
    arms = []

    async def replay(skill_dirs, calls, repeats):  # noqa: ANN001, ANN202
        is_revision_arm = any("pending" in str(p) for p in skill_dirs)
        arms.append(sorted(str(p) for p in skill_dirs))
        return [0.0 if is_revision_arm else 1.0] * (len(calls) * repeats)

    [v] = asyncio.run(CompletionGate()([rev], [Path("/lib/x")], _calls(2), replay))
    assert not v.admitted and v.gain == -1.0
    assert arms[1] == [str(rev.skill.path)]  # old "x" swapped out, not kept alongside


def test_budget_cuts_candidates_from_the_end_then_trials(tmp_path: Path):
    cands = [Candidate(_skill(tmp_path, n)) for n in ("a", "b", "c")]
    replay = _replay_by_names({frozenset(): 0.0, **{frozenset({n}): 1.0 for n in "abc"}})
    # 2 trials x 1 repeat x (1 + 3) = 8 > 6  ->  drop "c":  2 x 1 x (1 + 2) = 6
    vs = asyncio.run(CompletionGate(repeats=1, budget_calls=6)(cands, [], _calls(2), replay))
    assert [v.admitted for v in vs] == [True, True, False]
    assert vs[2].scores["stage"] == "budget"


def test_pass_at_decides_what_counts_as_completed(tmp_path: Path):
    cand = Candidate(_skill(tmp_path, "a"))
    replay = _replay_by_names({frozenset(): 0.4, frozenset({"a"}): 0.6})
    [v] = asyncio.run(CompletionGate(pass_at=0.5)([cand], [], _calls(1), replay))
    assert v.admitted and v.gain == 1.0


def test_no_trials_rejects_rather_than_admitting_unmeasured(tmp_path: Path):
    [v] = asyncio.run(CompletionGate()([Candidate(_skill(tmp_path, "a"))], [], [], _replay_by_names({})))
    assert not v.admitted


def test_the_round_report_learns_the_cuts_the_replays_and_the_decision_order(tmp_path: Path):
    cands = [Candidate(_skill(tmp_path, n)) for n in ("a", "b", "c")]
    replay = _replay_by_names({frozenset(): 0.0, **{frozenset({n}): 1.0 for n in "abc"}})
    # 3 x 1 x 4 = 12 > 5: cut "c", then "b" (3 x 1 x 2 = 6 > 5), then trial input c2 (2 x 1 x 2 = 4).
    vs = asyncio.run(CompletionGate(repeats=1, budget_calls=5)(cands, [], _calls(3), replay))
    assert vs[0].scores["budget_cuts"] == ["candidate c", "candidate b", "trial input c2"]
    assert sum(v.scores.get("extra_calls") or 0 for v in vs) == 4  # baseline + "a", 2 trials each
    assert [v.scores["order"] for v in vs] == [2.0, 1.0, 0.0]  # cuts are decided before trials
    assert vs[0].scores["with_skill"] == 1.0 and vs[0].scores["without_skill"] == 0.0
