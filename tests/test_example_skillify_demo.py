"""Offline tests for ``examples/skillify_demo`` -- the parts that need no JVM and no model.

Covers the fixed task set and how the count flags cut it down, the
ScienceWorld episode factory's drift guard (against a fake session), the
session's JVM release, and the demo's flag presets, round settling and
``--evaluate-only`` path. Nothing here starts ScienceWorld or calls a model.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai_functions.experimental.skills.skill_library import LibraryStore

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "skillify_demo"
TASK = "measure-melting-point-unknown-substance"


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """The example's modules, imported by their sibling names and removed again afterwards."""
    monkeypatch.syspath_prepend(str(EXAMPLE_DIR))
    before = set(sys.modules)
    names = ("scienceworld_env", "evaluation", "main")
    modules = {name: importlib.import_module(name) for name in names}
    yield SimpleNamespace(**modules)
    for name in set(sys.modules) - before:
        del sys.modules[name]


def _fake_loader(example: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    """Build ``Variation``s for the requested ids without ScienceWorld."""

    def load(task: str, ids: list[int]) -> list:
        return [example.scienceworld_env.Variation(task=task, variation=n, description=f"task {n}") for n in ids]

    monkeypatch.setattr(example.main, "load_variations", load)


def test_the_fixed_task_set_is_two_disjoint_splits(example: SimpleNamespace) -> None:
    traffic, evaluation = example.evaluation.TRAFFIC_VARIATIONS, example.evaluation.EVAL_VARIATIONS
    assert example.evaluation.TASK == TASK
    assert len(traffic) == 12 and len(evaluation) == 15
    assert not set(traffic) & set(evaluation)
    assert len(set(traffic)) == len(traffic) and len(set(evaluation)) == len(evaluation)
    # The train ids keep the pilot's alternating failure-mode order, not a
    # sorted one: --traffic-tasks takes a prefix, and a prefix must carry both.
    assert traffic != sorted(traffic)


def test_select_tasks_takes_a_prefix_of_the_fixed_ids(example: SimpleNamespace, monkeypatch) -> None:
    _fake_loader(example, monkeypatch)
    traffic, evaluation, variations = example.main._select_tasks(example.main._parse_args([]))
    assert [v.variation for v in traffic] == example.evaluation.TRAFFIC_VARIATIONS
    assert [v.variation for v in evaluation] == example.evaluation.EVAL_VARIATIONS
    assert set(variations) == {v.task_id for v in (*traffic, *evaluation)}
    assert all(v.description for v in variations.values())

    args = example.main._parse_args(["--traffic-tasks", "3", "--eval-tasks", "2"])
    traffic, evaluation, variations = example.main._select_tasks(args)
    assert [v.variation for v in traffic] == example.evaluation.TRAFFIC_VARIATIONS[:3]
    assert [v.variation for v in evaluation] == example.evaluation.EVAL_VARIATIONS[:2]
    assert len(variations) == 5


def test_evaluate_only_loads_no_traffic_variations(example: SimpleNamespace, monkeypatch) -> None:
    _fake_loader(example, monkeypatch)
    args = example.main._parse_args(["--evaluate-only", "/some/library"])
    traffic, evaluation, variations = example.main._select_tasks(args)
    assert traffic == []
    assert [v.variation for v in evaluation] == example.evaluation.EVAL_VARIATIONS
    assert len(variations) == len(example.evaluation.EVAL_VARIATIONS)


def test_no_pilot_file_or_machine_specific_path_is_referenced(example: SimpleNamespace) -> None:
    """The demo runs from a clean clone: no baselines file, no cache, no absolute path."""
    flags = vars(example.main._parse_args([]))
    assert not [name for name in flags if "pilot" in name or "cache" in name]
    source = "".join((EXAMPLE_DIR / name).read_text() for name in ("main.py", "evaluation.py", "scienceworld_env.py"))
    assert "pilot_baselines" not in source
    assert "/home/ubuntu" not in source


class _FakeSession:
    """Stands in for ``ScienceWorldSession``: no JVM, a fixed opening, a settable score."""

    instances: list[_FakeSession] = []
    opening = "You are in the kitchen."

    def __init__(self, variation: object, step_limit: int) -> None:
        self.variation, self.step_limit = variation, step_limit
        self.score, self.steps, self.closed = 0.0, 0, 0
        _FakeSession.instances.append(self)

    def start(self) -> str:
        return self.opening

    def close(self) -> None:
        self.closed += 1

    def make_tools(self) -> list[object]:
        return [f"sw_act<{id(self)}>"]


@pytest.fixture
def fake_sessions(example: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> type[_FakeSession]:
    _FakeSession.instances = []
    monkeypatch.setattr(example.scienceworld_env, "ScienceWorldSession", _FakeSession)
    return _FakeSession


def _factory(example: SimpleNamespace, built: list | None = None):
    v = example.scienceworld_env.Variation(task=TASK, variation=7, description="melt it")
    return example.scienceworld_env.make_episode_factory(
        {v.task_id: v}, 12, on_episode=built.append if built is not None else None
    ), v


async def test_episode_factory_builds_a_fresh_session_per_run(example, fake_sessions) -> None:
    built: list = []
    factory, v = _factory(example, built)
    args = {"variation_id": v.task_id, "task_description": "melt it", "opening_observation": fake_sessions.opening}
    first, second = await factory(args), await factory(args)
    assert first.session is not second.session and len(fake_sessions.instances) == 2
    assert first.tools != second.tools and built == [first, second]
    assert first.session.step_limit == 12 and first.session.variation is v


async def test_episode_outcome_is_completion_and_close_releases(example, fake_sessions) -> None:
    factory, v = _factory(example)
    episode = await factory({"variation_id": v.task_id, "opening_observation": fake_sessions.opening})
    for score, expected in ((100.0, 1.0), (99.0, 0.0), (79.0, 0.0), (-100.0, 0.0)):
        episode.session.score = score
        assert episode.outcome("whatever the model said") == expected
        assert episode.raw_score == score
    episode.close()
    assert episode.session.closed == 1


async def test_drift_guard_refuses_a_different_environment(example, fake_sessions) -> None:
    built: list = []
    factory, v = _factory(example, built)
    with pytest.raises(example.scienceworld_env.ReplayDriftError):
        await factory({"variation_id": v.task_id, "opening_observation": "You are in the greenhouse."})
    assert built == []
    assert fake_sessions.instances[0].closed == 1


def test_session_close_releases_the_jvm_once_off_the_calling_thread(example) -> None:
    closed_on: list[str] = []

    class _Env:
        def close(self) -> None:
            closed_on.append(threading.current_thread().name)

    env = _Env()
    session = example.scienceworld_env.ScienceWorldSession(
        example.scienceworld_env.Variation(task=TASK, variation=1, description="")
    )
    session._env = env
    session.close()
    session.close()  # idempotent
    for thread in threading.enumerate():
        if thread.name == "scienceworld-close":
            thread.join(timeout=5)
    assert closed_on == ["scienceworld-close"]
    env.close()  # what ScienceWorldEnv.__del__ does later: now a no-op
    assert closed_on == ["scienceworld-close"]


def test_skillify_prompt_is_the_evaluation_prompt_without_the_skills_slot(example) -> None:
    d = example.evaluation
    assert d.SKILLS_SLOT in d.PLAY_PROMPT
    prompt = d.PLAY_PROMPT.replace(d.SKILLS_SLOT, "")
    assert "{skills}" not in prompt and "{task_description}" in prompt and "{opening_observation}" in prompt
    assert prompt.count("\n\n") == d.PLAY_PROMPT.count("\n\n") - 1


def test_flag_defaults_and_smoke_preset(example) -> None:
    demo = example.main
    full = demo._parse_args([])
    assert (full.traffic_tasks, full.traffic_repeats, full.every, full.trial_inputs, full.trial_repeats) == (
        12,
        2,
        12,
        4,
        2,
    )
    assert (full.eval_tasks, full.repeats, full.step_limit, full.min_gain, full.concurrency) == (15, 2, 60, 0.02, 10)
    assert full.model == "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    smoke = demo._parse_args(["--smoke"])
    assert {k: getattr(smoke, k) for k in demo.SMOKE} == demo.SMOKE
    assert demo._parse_args(["--smoke", "--every", "3"]).every == 3


def test_make_play_wires_the_environment(example, tmp_path: Path) -> None:
    demo = example.main
    args = demo._parse_args(["--smoke", "--out", str(tmp_path)])
    play = demo.make_play({}, args, demo.EpisodeStats())
    cfg = play.config
    assert cfg.environment is not None and cfg.score is None and cfg.replay is True
    assert (cfg.every, cfg.trial_inputs, cfg.max_concurrency) == (4, 2, 10)
    assert cfg.library == tmp_path / "library"


def test_make_play_plugs_in_the_full_pipeline(example, tmp_path: Path) -> None:
    from full_pipeline import FullDiscoveryMiner, PrefilteredGate

    from ai_functions.experimental.skills.evaluation import CompletionGate

    demo = example.main
    args = demo._parse_args(["--smoke", "--min-gain", "0.1", "--budget-calls", "50", "--out", str(tmp_path)])
    cfg = demo.make_play({}, args, demo.EpisodeStats()).config
    assert isinstance(cfg.miner, FullDiscoveryMiner)
    assert isinstance(cfg.gate, PrefilteredGate) and cfg.gate._min_tier1 == 0.5  # noqa: SLF001
    inner = cfg.gate._inner  # noqa: SLF001
    assert isinstance(inner, CompletionGate)
    assert (inner.min_gain, inner.repeats, inner.budget_calls) == (0.1, 1, 50)


def _fake_play(pending: int, task: asyncio.Task | None) -> SimpleNamespace:
    refreshed: list[bool] = []

    async def refresh_skills() -> None:
        refreshed.append(True)

    return SimpleNamespace(
        learner=SimpleNamespace(_task=task, pending=pending), refresh_skills=refresh_skills, refreshed=refreshed
    )


async def test_settle_rounds_awaits_the_in_flight_round_and_does_not_rerun_a_consumed_window(example) -> None:
    finished: list[bool] = []

    async def round_() -> None:
        await asyncio.sleep(0.01)
        finished.append(True)

    play = _fake_play(0, asyncio.create_task(round_()))
    await example.main.settle_rounds(play, every=4)
    assert finished == [True] and play.refreshed == []


async def test_settle_rounds_runs_a_window_left_pending(example) -> None:
    play = _fake_play(4, None)
    await example.main.settle_rounds(play, every=4)
    assert play.refreshed == [True]
    play = _fake_play(3, None)
    await example.main.settle_rounds(play, every=4)
    assert play.refreshed == []


# --- --evaluate-only ---------------------------------------------------------


def _publish_one_skill(library: Path, tmp_path: Path, *, name: str = "do-thing") -> LibraryStore:
    """A real ``LibraryStore`` at ``library`` with one published skill, current version 1."""
    store = LibraryStore(library)
    skill_src = tmp_path / "skill-src" / name
    skill_src.mkdir(parents=True)
    (skill_src / "SKILL.md").write_text(f"---\nname: {name}\ndescription: does the thing\n---\nBody.\n")
    store.publish([skill_src])
    return store


def test_open_library_store_rejects_a_non_library_path(example: SimpleNamespace, tmp_path: Path) -> None:
    demo = example.main
    plain_dir = tmp_path / "not-a-library"
    plain_dir.mkdir()
    for bad in (plain_dir, tmp_path / "does-not-exist-at-all"):
        with pytest.raises(SystemExit, match="not a skill library"):
            demo._open_library_store(bad)


async def test_run_evaluate_only_rejects_a_library_whose_current_version_has_no_skills(
    example: SimpleNamespace, tmp_path: Path
) -> None:
    demo = example.main
    library = tmp_path / "library"
    LibraryStore(library)  # a real library, never published: current version 0, no skills
    args = SimpleNamespace(evaluate_only=str(library))
    with pytest.raises(SystemExit, match="has no skills"):
        await demo.run_evaluate_only(args, tmp_path / "out", {}, [], 0.0)


class _FakeEvalDriver:
    """Stands in for ``AIFunctionEvalDriver``: records the specs it was run with, no JVM/model."""

    instances: list[_FakeEvalDriver] = []

    def __init__(self, variations, **kwargs) -> None:
        self.variations = variations
        self.kwargs = kwargs
        self.specs: list = []
        _FakeEvalDriver.instances.append(self)

    async def run(self, specs):
        self.specs = list(specs)
        results = []
        for spec in specs:
            for arm in spec.arms:
                for repeat in range(spec.repeats):
                    score = 100.0 if arm == "skill_on" else 50.0
                    trace = SimpleNamespace(
                        oracle_result=SimpleNamespace(
                            details={
                                "raw_score": score,
                                "steps": 10,
                                "skills_available": ["do-thing"],
                                "skills_loaded": ["do-thing"] if arm == "skill_on" else [],
                            }
                        ),
                        token_counts=SimpleNamespace(input_tokens=1, output_tokens=1),
                    )
                    results.append(SimpleNamespace(task_id=spec.task_id, arm=arm, repeat=repeat, trace=trace))
        return results


async def test_evaluate_only_selects_the_same_tasks_as_the_normal_path_and_writes_outputs(
    example: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    demo = example.main
    _fake_loader(example, monkeypatch)
    library = tmp_path / "library"
    _publish_one_skill(library, tmp_path)

    common = ["--eval-tasks", "3"]
    normal_args = demo._parse_args([*common, "--out", str(tmp_path / "out-normal")])
    _, evaluation_normal, _ = demo._select_tasks(normal_args)
    assert evaluation_normal, "the fixed id list must produce evaluation tasks"

    monkeypatch.setattr(demo, "AIFunctionEvalDriver", _FakeEvalDriver)
    out_dir = tmp_path / "out-eval-only"
    eval_only_args = demo._parse_args([*common, "--evaluate-only", str(library), "--out", str(out_dir)])
    await demo.main(eval_only_args)

    driver = _FakeEvalDriver.instances[-1]
    assert [s.task_id for s in driver.specs] == [v.task_id for v in evaluation_normal]
    assert all(list(s.skills) == list(driver.specs[0].skills) for s in driver.specs)

    summary = json.loads((out_dir / "skillify_summary.json").read_text())
    assert summary["mode"] == "evaluate_only"
    assert summary["library_path"] == str(library.resolve())
    assert summary["final_version"] == 1
    assert summary["final_skills"] == ["do-thing"]
    assert summary["evaluation_tasks"] == [v.task_id for v in evaluation_normal]
    assert (out_dir / "eval_results.json").exists()
    assert (out_dir / "analysis.json").exists()


def test_more_tasks_than_there_are_fixed_ids_is_refused(example: SimpleNamespace) -> None:
    """Truncating silently would report fewer tasks than the caller asked for."""
    demo = example.main
    for flag, ids in (("--traffic-tasks", demo.TRAFFIC_VARIATIONS), ("--eval-tasks", demo.EVAL_VARIATIONS)):
        with pytest.raises(SystemExit):
            demo._parse_args([flag, str(len(ids) + 1)])
        with pytest.raises(SystemExit):
            demo._parse_args([flag, "-1"])
        assert demo._parse_args([flag, str(len(ids))])  # the whole list is fine


def test_evaluate_only_flag_default_and_value(example: SimpleNamespace) -> None:
    demo = example.main
    assert demo._parse_args([]).evaluate_only is None
    assert demo._parse_args(["--evaluate-only", "/tmp/some/library"]).evaluate_only == "/tmp/some/library"
