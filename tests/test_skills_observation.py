"""Observation must preserve execution, isolate callbacks, and expose the learning lifecycle."""

import asyncio
from pathlib import Path

import pytest

from ai_functions import ai_function
from ai_functions.ai_thread import AIFunctionError, PostConditionResult
from ai_functions.experimental.skills import CompletionGate, observe_skills, skill_event, skillify
from ai_functions.experimental.skills.observation import trace_observed
from ai_functions.experimental.skills.skill_library import load_skill
from ai_functions.testing import ScriptedModel, Turn


@ai_function[str](structured_output=False)
def _answer(task: str) -> str:
    """Answer {task}."""


def _skill(root: Path, name: str):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Use a concrete procedure\n---\n\n## Instructions\n1. Press the button.\n"
    )
    return load_skill(directory)


async def test_observed_round_preserves_gating_and_captures_discarded_source(tmp_path):
    async def mine(calls, out):
        return [_skill(out, "good"), _skill(out, "bad")]

    fn = skillify(
        library=tmp_path / "library",
        replay=True,
        every=1000,
        trial_inputs=1,
        score=lambda args, value: float(value.strip() == "yes"),
        miner=mine,
        gate=CompletionGate(repeats=1),
    )(_answer.replace(model=ScriptedModel([Turn(text=t) for t in ["no", "yes", "no", "no", "yes", "no"]])))
    events = []
    with observe_skills(events.append):
        assert [(await fn(task=str(i))).strip() for i in range(3)] == ["no", "yes", "no"]
        report = await fn.refresh_skills()
    assert list(report.admitted) == ["good"] and list(report.rejected) == ["bad"]
    kinds = [e["kind"] for e in events]
    assert kinds.index("round.started") < kinds.index("mining.finished") < kinds.index("gate.started")
    assert kinds.index("gate.finished") < kinds.index("library.published") < kinds.index("history.appended")
    mined = next(e for e in events if e["kind"] == "mining.finished")
    assert "Press the button" in mined["skills"][1]["raw"]
    assert not Path(mined["skills"][1]["path"]).exists(), "the rejected candidate was archived before cleanup"
    calls = [e for e in events if e["kind"] == "run.started" and e["phase"] == "call"]
    replays = [e for e in events if e["kind"] == "run.started" and e["phase"] == "replay"]
    assert len(calls) == 3 and len(replays) == 3
    assert {e["run_id"] for e in calls}.isdisjoint(e["run_id"] for e in replays)
    assert all(e["round_id"] for e in replays)
    assert not mined.get("run_id"), "learning cannot inherit the triggering call's execution identity"


async def test_observer_is_isolated_from_state_and_failures_do_not_affect_calls(tmp_path):
    skill = _skill(tmp_path, "alpha")

    def mutate(event):
        event["skill"]["frontmatter"]["name"] = "tampered"

    with observe_skills(mutate):
        skill_event("candidate", skill=skill)
    assert skill.name == "alpha"

    def fail(event):
        raise RuntimeError("viewer unavailable")

    fn = skillify(library=tmp_path / "lib", replay=True, score=lambda a, r: 1.0)(
        _answer.replace(model=ScriptedModel([Turn(text="yes")]))
    )
    with observe_skills(fail):
        result = await fn.trace(task="t")
    assert result.value.strip() == "yes"
    assert len(fn.learner.store.records()) == 1
    assert not result.coordinator._subscribers


async def test_concurrent_scopes_do_not_mix_and_closed_capture_stops_child_tasks(tmp_path):
    events = []
    release = asyncio.Event()

    async def late():
        await release.wait()
        skill_event("late")

    async def call(label):
        fn = skillify(library=tmp_path / label, replay=True, score=lambda a, r: 1.0)(
            _answer.replace(model=ScriptedModel([Turn(text=label)]))
        )
        return await fn(task=label)

    with observe_skills(events.append):
        task = asyncio.create_task(late())
        assert [value.strip() for value in await asyncio.gather(call("a"), call("b"))] == ["a", "b"]
    release.set()
    await task
    assert not any(e["kind"] == "late" for e in events)
    started = [e for e in events if e["kind"] == "run.started"]
    for start in started:
        same_run = [e for e in events if e.get("run_id") == start["run_id"]]
        assert all(e["library"] == start["library"] for e in same_run)
        assert any(e["kind"] == "run.event" and e["event"]["kind"] == "tool_result" for e in same_run) is False
        assert any(e["kind"] == "run.event" and e["event"]["kind"] == "message_assistant_complete" for e in same_run)


async def test_real_skill_load_is_visible_without_changing_the_skill_tool(tmp_path):
    fn = skillify(library=tmp_path / "lib", replay=True, score=lambda a, r: 1.0)(
        _answer.replace(model=ScriptedModel([Turn(tool_calls=(("Skill", {"skill": "alpha"}),)), Turn(text="yes")]))
    )
    skill = _skill(tmp_path / "source", "alpha")
    fn.learner.store.publish([skill.path])
    events = []
    with observe_skills(events.append):
        assert (await fn(task="t")).strip() == "yes"
    tool_results = [e["event"] for e in events if e["kind"] == "run.event" and e["event"]["kind"] == "tool_result"]
    assert any(e["content"][0]["text"] == f"# alpha\n\n{skill.body}" for e in tool_results)
    record = next(e["record"] for e in events if e["kind"] == "call.recorded")
    assert record["skills_loaded"] == ["alpha"]


async def test_driver_trace_keeps_existing_config_hook_and_failed_events():
    hooks, events = [], []

    def original(ctx):
        hooks.append(ctx)
        return {"system_prompt": "Original configured prompt"}

    def check(value):
        return PostConditionResult(passed=False, message="expected failure")

    fn = _answer.replace(
        model=ScriptedModel([Turn(text="no")]),
        config_hook=original,
        post_conditions=[check],
        max_attempts=0,
    )
    with observe_skills(events.append, run_id="failed-trial", phase="replay"):
        with pytest.raises(AIFunctionError):
            await trace_observed(fn, task="t")
    assert len(hooks) == 1
    assert any(e["kind"] == "run.event" and e["event"]["kind"] == "failed" for e in events)
    assert not hooks[0].coordinator._subscribers
