"""``ReplayDriver`` and ``SkillLearner`` -- replaying recorded calls, and the rounds behind ``@skillify``.

``ReplayDriver``: re-run recorded calls, with and without skills. Uses
``ScriptedModel`` so every run is deterministic and offline.

``SkillLearner``: when a round runs, and what it does to the library. Every
test uses a real ``LibraryStore`` in ``tmp_path`` with a fake miner and the
default gate with stubbed stages: no model, no network.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult
from ai_functions.experimental.skills.evaluation import Candidate, CompletionGate, Verdict
from ai_functions.experimental.skills.learner import (
    ReplayDriver,
    RoundReport,
    SkillifyConfig,
    SkillLearner,
    bind_arguments,
    outcome_of,
)
from ai_functions.experimental.skills.learner import _RunResult as RunResult
from ai_functions.experimental.skills.learner import _RunSpec as RunSpec
from ai_functions.experimental.skills.skill_library import LibraryStore, load_skill
from ai_functions.runtime import InMemoryCoordinator
from ai_functions.testing import ScriptedModel, Turn
from ai_functions.types import ThreadId, current_thread_scope, thread_scope


def _write_skill(root: Path, name: str, description: str = "does x", body: str = "Step 1. Do x.") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    return directory


# ── ReplayDriver ─────────────────────────────────────────────────────────────


def _require_yes(result, **kwargs):  # noqa: ANN001, ANN003, ANN202
    if "yes" not in result:
        return PostConditionResult(passed=False, message="must contain yes")
    return None


@ai_function[str](structured_output=False, post_conditions=[_require_yes], max_attempts=0)
def _solve(task: str) -> str:
    """Solve: {task}"""


def _spec(call_id: str, skills: list[Path], arm: str) -> RunSpec:
    return RunSpec(task_id=call_id, skills=skills, arms=(arm,))


def _store_with_call(tmp_path: Path) -> tuple[LibraryStore, str]:
    store = LibraryStore(tmp_path / "lib")
    record = store.append_record(
        outcome=0.0, version=0, skills_available=[], skills_loaded=[], events=[], args=("t",), kwargs={}
    )
    return store, record.call_id


def test_bind_arguments_names_positional_arguments():
    assert bind_arguments(_solve, ("t",), {}) == {"task": "t"}


def test_outcome_without_score_is_one_and_a_raising_score_is_unscored():
    assert outcome_of(_solve, None, ("t",), {}, "yes") == 1.0
    assert outcome_of(_solve, lambda a, r: 0.25, ("t",), {}, "x") == 0.25
    assert outcome_of(_solve, lambda a, r: 1 / 0, ("t",), {}, "x") is None
    assert outcome_of(_solve, lambda a, r: 3.0, ("t",), {}, "x") is None


async def test_a_passing_replay_reports_success(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    fn = _solve.replace(model=ScriptedModel([Turn(text="yes indeed")]))
    driver = ReplayDriver(fn, store, score=None, pass_at=1.0, work_root=tmp_path / "runs", max_attempts=1)

    [result] = await driver.run([_spec(call_id, [], "skill_off")])

    assert result == RunResult(outcome=1.0, passed=True)


async def test_a_post_condition_failure_is_outcome_zero_not_an_error(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    fn = _solve.replace(model=ScriptedModel([Turn(text="no")]))
    driver = ReplayDriver(fn, store, score=None, pass_at=1.0, work_root=tmp_path / "runs", max_attempts=1)

    [result] = await driver.run([_spec(call_id, [], "skill_off")])

    assert result == RunResult(outcome=0.0, passed=False)


class _SeeingModel(ScriptedModel):
    """A ``ScriptedModel`` that also keeps each request's system prompt and messages."""

    def __init__(self, turns: list[Turn]) -> None:
        super().__init__(turns)
        self.requests: list[tuple[str | None, str]] = []

    def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        self.requests.append((system_prompt, repr(messages)))
        return super().stream(messages, tool_specs, system_prompt, **kwargs)


async def test_skill_on_exposes_the_skills_and_the_skill_tool_loads_them(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    alpha = _write_skill(tmp_path / "src", "alpha", body="Say yes.")
    model = _SeeingModel([Turn(tool_calls=(("Skill", {"skill": "alpha"}),)), Turn(text="yes")])
    driver = ReplayDriver(
        _solve.replace(model=model), store, score=None, pass_at=1.0, work_root=tmp_path / "runs", max_attempts=1
    )

    [result] = await driver.run([_spec(call_id, [alpha], "skill_on")])

    assert result.passed is True
    (first_prompt, _), (_, second_messages) = model.requests
    assert "- alpha: does x" in (first_prompt or "")
    assert "Say yes." in second_messages


async def test_score_override_and_pass_at(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    fn = _solve.replace(model=ScriptedModel([Turn(text="yes")]))
    driver = ReplayDriver(fn, store, score=lambda a, r: 0.5, pass_at=0.5, work_root=tmp_path / "runs", max_attempts=1)

    [result] = await driver.run([_spec(call_id, [], "skill_off")])

    assert result == RunResult(outcome=0.5, passed=True)


async def test_an_infrastructure_error_is_raised_after_the_last_attempt(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    fn = _solve.replace(model=ScriptedModel([]))  # ScriptExhausted on first use
    driver = ReplayDriver(
        fn, store, score=None, pass_at=1.0, work_root=tmp_path / "runs", max_attempts=2, retry_backoff=0.0
    )

    with pytest.raises(Exception):  # noqa: B017 - the scripted model's exhaustion error
        await driver.run([_spec(call_id, [], "skill_off")])


async def test_every_run_gets_its_own_run_dir_even_under_the_same_label(tmp_path: Path):
    """Two batches with the same (call, arm, repeat) -- a baseline and a candidate -- must not share files."""
    store, call_id = _store_with_call(tmp_path)
    alpha = _write_skill(tmp_path / "src", "alpha")
    beta = _write_skill(tmp_path / "src", "beta")
    fn = _solve.replace(model=ScriptedModel([Turn(text="yes"), Turn(text="yes")]))
    driver = ReplayDriver(fn, store, score=None, pass_at=1.0, work_root=tmp_path / "runs", max_attempts=1)

    await driver.run([_spec(call_id, [alpha], "skill_on")])
    await driver.run([_spec(call_id, [beta], "skill_on")])

    run_dirs = sorted((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 2
    assert sorted(p.name for d in run_dirs for p in (d / "skills-1").iterdir()) == ["alpha", "beta"]


# ── environment= ──


class _PressModel(ScriptedModel):
    """Presses the button once, then answers "yes"; stateless, so concurrent runs cannot interleave turns."""

    def __init__(self) -> None:
        super().__init__([])

    def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
        from ai_functions.testing.scripted_model import _stream_turn

        results = sum("toolResult" in b for m in messages for b in m.get("content", []))
        has_press = any(spec.get("name") == "press" for spec in (tool_specs or []))
        turn = Turn(tool_calls=(("press", {}),)) if has_press and not results else Turn(text="yes")
        return _stream_turn(turn)


class _ButtonEpisode:
    """Per-episode state: a button press counter. Outcome is 1.0 iff pressed exactly once in this episode."""

    def __init__(self, arguments) -> None:  # noqa: ANN001
        from strands import tool

        self.arguments = dict(arguments)
        self.presses = 0
        self.closed = 0

        @tool(name="press", description="Press the button.")
        def press() -> str:
            self.presses += 1
            return f"pressed {self.presses}"

        self.tools = [press]

    def outcome(self, result) -> float:  # noqa: ANN001
        return 1.0 if self.presses == 1 else 0.0

    def close(self) -> None:
        self.closed += 1


class _ButtonEnvironment:
    def __init__(self, fail_times: int = 0) -> None:
        self.episodes: list[_ButtonEpisode] = []
        self.calls = 0
        self._fail_times = fail_times

    def __call__(self, arguments):  # noqa: ANN001, ANN204
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("environment could not start")
        episode = _ButtonEpisode(arguments)
        self.episodes.append(episode)
        return episode


async def test_every_replay_run_gets_a_fresh_episode_from_the_recorded_arguments(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    env = _ButtonEnvironment()
    fn = _solve.replace(model=_PressModel())
    driver = ReplayDriver(
        fn, store, score=None, pass_at=1.0, work_root=tmp_path / "runs", max_attempts=1, environment=env
    )
    alpha = _write_skill(tmp_path / "src", "alpha")
    spec = RunSpec(task_id=call_id, skills=[alpha], arms=("skill_off", "skill_on"), repeats=3)

    results = await driver.run([spec])

    assert env.calls == 6 and len(env.episodes) == 6
    assert all(e.arguments == {"task": "t"} for e in env.episodes)
    # State never leaks: each episode saw exactly its own run's single press.
    assert [e.presses for e in env.episodes] == [1] * 6
    assert all(r.outcome == 1.0 for r in results)
    assert [e.closed for e in env.episodes] == [1] * 6


async def test_the_replay_outcome_comes_from_the_episode_not_the_return_value(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    env = _ButtonEnvironment()
    # Answers "yes" (post-condition passes) without ever pressing: the episode says 0.0.
    fn = _solve.replace(model=ScriptedModel([Turn(text="yes")]))
    driver = ReplayDriver(
        fn, store, score=None, pass_at=1.0, work_root=tmp_path / "runs", max_attempts=1, environment=env
    )

    [result] = await driver.run([_spec(call_id, [], "skill_off")])

    assert result == RunResult(outcome=0.0, passed=False)
    assert env.episodes[0].closed == 1


async def test_a_replay_closes_its_episode_on_post_condition_failure_and_on_infrastructure_errors(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    env = _ButtonEnvironment()
    failing = ReplayDriver(
        _solve.replace(model=ScriptedModel([Turn(text="no")])),
        store,
        score=None,
        pass_at=1.0,
        work_root=tmp_path / "runs",
        max_attempts=1,
        environment=env,
    )
    [result] = await failing.run([_spec(call_id, [], "skill_off")])
    assert result.outcome == 0.0

    broken = ReplayDriver(
        _solve.replace(model=ScriptedModel([])),
        store,
        score=None,
        pass_at=1.0,
        work_root=tmp_path / "runs2",
        max_attempts=2,
        retry_backoff=0.0,
        environment=env,
    )
    with pytest.raises(Exception):  # noqa: B017 - the scripted model's exhaustion error
        await broken.run([_spec(call_id, [], "skill_off")])

    assert len(env.episodes) == 3
    assert [e.closed for e in env.episodes] == [1, 1, 1]


async def test_a_raising_environment_factory_on_replay_is_retried_then_raised(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    env = _ButtonEnvironment(fail_times=100)
    driver = ReplayDriver(
        _solve.replace(model=_PressModel()),
        store,
        score=None,
        pass_at=1.0,
        work_root=tmp_path / "runs",
        max_attempts=3,
        retry_backoff=0.0,
        environment=env,
    )

    with pytest.raises(RuntimeError, match="could not start"):
        await driver.run([_spec(call_id, [], "skill_off")])

    assert env.calls == 3


async def test_a_factory_failure_on_replay_recovers_on_retry(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    env = _ButtonEnvironment(fail_times=1)
    driver = ReplayDriver(
        _solve.replace(model=_PressModel()),
        store,
        score=None,
        pass_at=1.0,
        work_root=tmp_path / "runs",
        max_attempts=2,
        retry_backoff=0.0,
        environment=env,
    )

    [result] = await driver.run([_spec(call_id, [], "skill_off")])

    assert env.calls == 2
    assert result.outcome == 1.0


async def test_an_async_environment_factory_is_awaited(tmp_path: Path):
    store, call_id = _store_with_call(tmp_path)
    env = _ButtonEnvironment()

    async def factory(arguments):  # noqa: ANN001, ANN202
        return env(arguments)

    driver = ReplayDriver(
        _solve.replace(model=_PressModel()),
        store,
        score=None,
        pass_at=1.0,
        work_root=tmp_path / "runs",
        max_attempts=1,
        environment=factory,
    )

    [result] = await driver.run([_spec(call_id, [], "skill_off")])

    assert result.outcome == 1.0
    assert env.episodes[0].closed == 1


# ── SkillLearner ─────────────────────────────────────────────────────────────


def _config(tmp_path: Path, **overrides) -> SkillifyConfig:  # noqa: ANN003
    """A small config; ``miner=``/``gate=`` pass through (a custom gate keeps the default gate settings)."""
    values = {"library": tmp_path / "lib", "replay": True, "every": 3, "trial_inputs": 2}
    if overrides.get("gate") is None:
        values["trial_repeats"] = 1
    values.update(overrides)
    return SkillifyConfig(**values)


def _record(learner: SkillLearner, outcome: float | None, replayable: bool = True):  # noqa: ANN202
    args = ("t",) if replayable else (lambda: 1,)
    return learner.record(
        outcome=outcome,
        version=learner.store.current_version(),
        skills_available=[],
        skills_loaded=[],
        events=[],
        args=args,
        kwargs={},
    )


class _Discover:
    """A ``Miner``: records ``(number of calls, ids of failed calls)`` and mines ``names``."""

    def __init__(self, names: list[str], fail: bool = False) -> None:
        self.names, self.fail = names, fail
        self.calls: list[tuple[int, frozenset[str]]] = []

    async def __call__(self, calls, out_dir):  # noqa: ANN001, ANN204
        self.calls.append((len(calls), frozenset(c.call_id for c in calls if not c.passed)))
        if self.fail:
            raise RuntimeError("discovery blew up")
        return [load_skill(_write_skill(out_dir, n)) for n in self.names]


class _Gate:
    """A fake ``Gate``: admits candidates named in ``gains`` whose gain >= 0.02, with no real replay.

    Records the trial calls and the current library it was asked to decide over, for tests
    that inspect what the round handed the gate.
    """

    def __init__(self, gains: dict[str, float]) -> None:
        self.gains = gains
        self.trial_task_ids: list[str] = []
        self.baseline: list[str] = []

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN202
        self.trial_task_ids = [t.call_id for t in trials]
        self.baseline = [p.name for p in current]
        verdicts = []
        for c in candidates:
            gain = self.gains.get(c.skill.name, 0.0)
            verdicts.append(Verdict(c, gain >= 0.02, gain, "fake", {"stage": "trial", "gain": gain}))
        return verdicts


class _HangingDiscover:
    """Marks ``out_root`` on disk, signals ``started``, then hangs until cancelled."""

    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def __call__(self, calls, out_dir):  # noqa: ANN001, ANN204
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "marker").write_text("x")
        self.started.set()
        await asyncio.Event().wait()  # never set; hangs until the caller cancels
        return []  # pragma: no cover - unreachable


class _SlowDiscover:
    """Yields to the event loop once (so a concurrent ``run_round`` can interleave)."""

    def __init__(self, names: list[str]) -> None:
        self.names = names

    async def __call__(self, calls, out_dir):  # noqa: ANN001, ANN204
        await asyncio.sleep(0)
        return [load_skill(_write_skill(out_dir, n)) for n in self.names]


def _learner(tmp_path: Path, discover, gate: _Gate, **overrides) -> SkillLearner:  # noqa: ANN001, ANN003
    """A learner mining with ``discover``, gating with ``gate`` set up from the config's gate settings."""
    config = _config(tmp_path, miner=discover, **overrides)
    store = LibraryStore(config.library, keep_versions=config.keep_versions)
    learner = SkillLearner(store, config, driver_factory=lambda root: None)
    gate.min_gain, gate.repeats, gate.budget_calls = config.min_gain, config.trial_repeats, config.budget_calls
    learner._gate = gate
    return learner


def test_config_requires_replay_true(tmp_path: Path):
    with pytest.raises(ValueError, match="replay"):
        SkillifyConfig(library=tmp_path)
    with pytest.raises(ValueError, match="replay"):
        SkillifyConfig(library=tmp_path, replay=False)


def test_a_window_with_no_signal_is_declined(tmp_path: Path):
    discover = _Discover(["alpha"])
    learner = _learner(tmp_path, discover, _Gate({}))
    for _ in range(3):
        _record(learner, 1.0)

    report = asyncio.run(learner.run_round())

    assert report.status == "declined"
    assert discover.calls == []
    assert learner.pending == 0
    assert RoundReport.from_json(learner.store.history()[-1]).status == "declined"


def test_admitted_candidates_become_the_new_version(tmp_path: Path):
    learner = _learner(tmp_path, _Discover(["alpha", "beta"]), _Gate({"alpha": 0.5}))
    for outcome in (1.0, 0.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.status == "admitted"
    assert list(report.admitted) == ["alpha"] and list(report.rejected) == ["beta"]
    assert (report.version_before, report.version_after) == (0, 1)
    assert [d.name for d in learner.store.skill_dirs()] == ["alpha"]
    assert not learner.store.round_dir(report.round).exists()


def test_failed_calls_are_forwarded_as_failed_trajectory_ids(tmp_path: Path):
    discover = _Discover([])
    learner = _learner(tmp_path, discover, _Gate({}))
    failed = _record(learner, 0.0)
    _record(learner, 1.0)
    _record(learner, 1.0)

    asyncio.run(learner.run_round())

    assert discover.calls == [(3, frozenset({failed.call_id}))]


def test_unscored_calls_are_never_mined(tmp_path: Path):
    """An unscored call (outcome=None) is a real, pending call but is neither mined nor a trial input."""
    discover = _Discover([])
    learner = _learner(tmp_path, discover, _Gate({}))
    failed = _record(learner, 0.0)
    _record(learner, 1.0)
    _record(learner, 1.0)
    _record(learner, None)
    _record(learner, None)

    report = asyncio.run(learner.run_round())

    assert discover.calls == [(3, frozenset({failed.call_id}))]
    assert report.mined_from == 3
    assert report.window_calls == 5


def test_first_round_trials_overlap_the_window_and_later_rounds_do_not(tmp_path: Path):
    gate = _Gate({"alpha": 0.5, "beta": 0.5})
    learner = _learner(tmp_path, _Discover(["alpha"]), gate)
    first = [_record(learner, o) for o in (0.0, 1.0, 1.0)]

    report = asyncio.run(learner.run_round())
    assert report.trials_overlap_mining is True
    assert set(gate.trial_task_ids) <= {r.call_id for r in first}

    learner = _learner(tmp_path, _Discover(["beta"]), gate)
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)
    report = asyncio.run(learner.run_round())

    assert report.trials_overlap_mining is False
    assert set(gate.trial_task_ids) <= {r.call_id for r in first}
    # The second round is gated against the library the first one installed.
    assert gate.baseline == ["alpha"]


def test_trial_inputs_favor_failed_calls(tmp_path: Path):
    gate = _Gate({})
    learner = _learner(tmp_path, _Discover(["alpha"]), gate, trial_inputs=1)
    _record(learner, 1.0)
    failed = _record(learner, 0.0)
    _record(learner, 1.0)

    asyncio.run(learner.run_round())

    assert gate.trial_task_ids == [failed.call_id]


def test_a_candidate_named_like_an_installed_skill_is_dropped(tmp_path: Path):
    learner = _learner(tmp_path, _Discover(["alpha"]), _Gate({"alpha": 0.5}))
    learner.store.publish([_write_skill(tmp_path / "old", "alpha")])
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.dropped_duplicates == ["alpha"]
    assert report.status == "none_admitted"
    assert learner.store.current_version() == 1


def test_a_reserved_or_colliding_candidate_is_dropped_and_the_round_goes_on(tmp_path: Path):
    """``skill-index`` is the backend's index parameter; ``foo_bar`` and ``foo-bar`` share one parameter name."""
    gate = _Gate({"beta": 0.5, "gamma-x": 0.5})
    learner = _learner(tmp_path, _Discover(["skill-index", "foo_bar", "beta", "gamma-x", "gamma_x"]), gate)
    learner.store.publish([_write_skill(tmp_path / "old", "foo-bar")])
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.rejected["skill-index"]["stage"] == "invalid"
    assert report.dropped_duplicates == ["foo_bar", "gamma_x"]
    assert report.status == "admitted" and sorted(report.admitted) == ["beta", "gamma-x"]
    assert [d.name for d in learner.store.skill_dirs()] == ["beta", "foo-bar", "gamma-x"]


def test_budget_cuts_candidates_from_the_end_of_the_list_before_trial_inputs(tmp_path: Path):
    """``CompletionGate`` cuts from the end of the candidate list it is given, not by any quality score."""
    recorded: list[int] = []

    async def fake_replay(skill_dirs, calls, repeats):  # noqa: ANN001, ANN202
        recorded.append(len(calls))
        return [0.0] * (len(calls) * repeats)

    # 2 trial inputs x 1 repeat x (1 + 3 candidates) = 8 > 6  ->  drop the last candidate "c": 2 x (1 + 2) = 6
    learner = _learner(tmp_path, _Discover(["a", "b", "c"]), CompletionGate(), budget_calls=6)
    learner._replay = fake_replay
    for outcome in (0.0, 0.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.budget_cuts == ["candidate c"]
    assert report.rejected["c"]["stage"] == "budget"
    assert recorded and all(n == 2 for n in recorded)
    assert report.extra_calls == 6


def test_budget_then_cuts_trial_inputs(tmp_path: Path):
    recorded: list[int] = []

    async def fake_replay(skill_dirs, calls, repeats):  # noqa: ANN001, ANN202
        recorded.append(len(calls))
        return [0.0] * (len(calls) * repeats)

    # 2 x 1 x (1 + 2) = 6 > 3 -> drop "b": 2 x (1 + 1) = 4 > 3 -> drop a trial input: 1 x (1 + 1) = 2
    learner = _learner(tmp_path, _Discover(["a", "b"]), CompletionGate(), budget_calls=3)
    learner._replay = fake_replay
    for outcome in (0.0, 0.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.budget_cuts[0] == "candidate b"
    assert report.budget_cuts[1].startswith("trial input ")
    assert recorded and all(n == 1 for n in recorded)


def test_max_skills_keeps_the_highest_gain(tmp_path: Path):
    learner = _learner(tmp_path, _Discover(["low", "high"]), _Gate({"low": 0.1, "high": 0.4}), max_skills=1)
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert list(report.admitted) == ["high"]
    assert report.rejected["low"]["stage"] == "max_skills"


def test_a_failed_round_leaves_the_library_unchanged_and_resets_the_count(tmp_path: Path):
    learner = _learner(tmp_path, _Discover([], fail=True), _Gate({}))
    learner.store.publish([_write_skill(tmp_path / "old", "alpha")])
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.status == "failed" and "discovery blew up" in (report.error or "")
    assert learner.store.current_version() == 1
    assert learner.pending == 0


def test_a_round_starts_only_after_every_calls_and_never_twice(tmp_path: Path):
    async def scenario() -> None:
        learner = _learner(tmp_path, _Discover([]), _Gate({}))
        _record(learner, 0.0)
        _record(learner, 1.0)
        assert learner.maybe_start_round() is None
        _record(learner, 1.0)
        task = learner.maybe_start_round()
        assert task is not None
        assert learner.maybe_start_round() is None  # one round at a time
        report = await task
        assert report.status == "none_admitted"

    asyncio.run(scenario())


def test_a_cancelled_round_keeps_its_window_and_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    async def scenario() -> None:
        discover = _HangingDiscover()
        learner = _learner(tmp_path, discover, _Gate({}))
        for outcome in (0.0, 1.0, 1.0):
            _record(learner, outcome)
        pending_before = learner.pending

        task = asyncio.create_task(learner.run_round())
        await discover.started.wait()
        assert learner.store.round_dir(1).exists()  # proves there is something to clean up

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert learner.store.history() == []
        assert learner.pending == pending_before
        assert not learner.store.round_dir(1).exists()
        assert any(r.levelname == "WARNING" and "cancelled" in r.getMessage() for r in caplog.records)

    asyncio.run(scenario())


def test_concurrent_run_round_calls_yield_exactly_one_history_entry(tmp_path: Path):
    async def scenario() -> None:
        learner = _learner(tmp_path, _SlowDiscover([]), _Gate({}))
        for outcome in (0.0, 1.0, 1.0):
            _record(learner, outcome)

        first, second = await asyncio.gather(learner.run_round(), learner.run_round())

        assert sorted([first.status, second.status]) == ["failed", "none_admitted"]
        busy = first if first.status == "failed" else second
        assert busy.error == "a round is already running in this process"
        assert len(learner.store.history()) == 1

    asyncio.run(scenario())


def test_pending_survives_a_restart(tmp_path: Path):
    learner = _learner(tmp_path, _Discover([]), _Gate({}))
    _record(learner, 1.0)
    _record(learner, 0.0)

    restarted = _learner(tmp_path, _Discover([]), _Gate({}))

    assert restarted.pending == 2


class _ScopeCheckingDiscover:
    """Fails the round unless it sees no ambient thread scope."""

    async def __call__(self, calls, out_dir):  # noqa: ANN001, ANN204
        assert current_thread_scope() is None, "a round must not run under the triggering call's scope"
        return []


def test_a_background_round_does_not_inherit_the_triggering_call_s_thread_scope(tmp_path: Path):
    """``maybe_start_round`` must not hand its task the caller's ambient scope.

    A task copies the context it is created under; without ``no_thread_scope()``
    around the ``create_task`` call, a round triggered from inside a call's own
    cycle would inherit that call's ``(coordinator, thread_id)`` -- which is torn
    down as soon as the call returns, breaking anything the round later spawns.
    """

    async def scenario() -> None:
        learner = _learner(tmp_path, _ScopeCheckingDiscover(), _Gate({}))
        for _ in range(3):
            _record(learner, 0.0)

        coord = InMemoryCoordinator()
        with thread_scope(coord, ThreadId("caller-thread")):
            task = learner.maybe_start_round()
        assert task is not None
        report = await task

        assert report.status == "none_admitted"  # discovery ran, mined nothing

    asyncio.run(scenario())


def test_refresh_does_not_inherit_an_ambient_thread_scope(tmp_path: Path):
    """``refresh`` may be called from inside a running thread (e.g. as a tool)."""

    async def scenario() -> None:
        learner = _learner(tmp_path, _ScopeCheckingDiscover(), _Gate({}))
        for _ in range(3):
            _record(learner, 0.0)

        coord = InMemoryCoordinator()
        with thread_scope(coord, ThreadId("caller-thread")):
            report = await learner.refresh()

        assert report.status == "none_admitted"

    asyncio.run(scenario())


def test_refresh_always_runs_its_own_round_after_an_in_flight_one(tmp_path: Path):
    """``refresh`` awaits a round already in flight, then still runs a fresh one of its own.

    Returning the in-flight round's own report instead would make ``refresh``'s
    result depend on timing: it would be right if the in-flight round had just
    finished on the current window, but wrong the moment new calls land while
    it runs, or it finished earlier on an older window. Always running (and
    returning) a fresh round makes the result reflect the window as it stands
    when ``refresh`` returns, regardless of what triggered the earlier round.
    """

    async def scenario() -> None:
        learner = _learner(tmp_path, _Discover(["alpha"]), _Gate({"alpha": 1.0}))
        for _ in range(3):
            _record(learner, 0.0)

        task = learner.maybe_start_round()
        assert task is not None

        report = await learner.refresh()

        # The in-flight round mined and admitted "alpha", consuming the window...
        assert (await task).status == "admitted"
        # ...so refresh's own round -- run after it, over what's left -- finds nothing to do.
        assert report.status == "declined"
        assert report is not await task
        assert learner.pending == 0

    asyncio.run(scenario())


class _RollingBackGate(_Gate):
    """Admits like ``_Gate``, but rolls the library back while deciding."""

    def __init__(self, gains: dict[str, float], store: LibraryStore) -> None:
        super().__init__(gains)
        self.store = store

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN202
        self.store.rollback()
        return await super().__call__(candidates, current, trials, replay)


def test_a_rollback_during_a_round_is_not_overwritten(tmp_path: Path):
    """The round publishes current + admitted as it saw them at the start; a rollback since then must win."""
    config = _config(tmp_path, miner=_Discover(["alpha"]))
    store = LibraryStore(config.library)
    store.publish([_write_skill(tmp_path / "src", "base")])
    gate = _RollingBackGate({"alpha": 1.0}, store)
    learner = SkillLearner(store, config, driver_factory=lambda root: None)
    learner._gate = gate
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.status == "failed"
    assert "library changed during the round" in (report.error or "")
    assert report.version_before == 1 and report.version_after == 0
    assert report.admitted == {}
    assert report.rejected["alpha"]["stage"] == "stale"
    assert store.current_version() == 0
    assert store.versions() == [1]
    assert [e["type"] for e in store.history()] == ["rollback", "round"]


class _GatedDiscover:
    """Signals ``started``, then waits for ``release`` before mining ``names``."""

    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, calls, out_dir):  # noqa: ANN001, ANN204
        self.started.set()
        await self.release.wait()
        return [load_skill(_write_skill(out_dir, n)) for n in self.names]


def test_refresh_waits_for_a_directly_awaited_round_too(tmp_path: Path):
    """A round run by awaiting ``run_round()`` directly (not the background task) must also be waited for."""

    async def scenario() -> None:
        discover = _GatedDiscover(["alpha"])
        learner = _learner(tmp_path, discover, _Gate({"alpha": 1.0}))
        for _ in range(3):
            _record(learner, 0.0)

        first_task = asyncio.create_task(learner.run_round())
        await discover.started.wait()
        refresh_task = asyncio.create_task(learner.refresh())
        await asyncio.sleep(0.1)
        assert not refresh_task.done()
        discover.release.set()

        first, second = await first_task, await refresh_task

        assert first.status == "admitted"
        assert second.status != "failed", second
        assert [e["round"] for e in learner.store.history()] == [1, 2]

    asyncio.run(scenario())


class _WrongCountGate:
    """A custom ``Gate`` that returns no verdicts at all."""

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN204
        return []


def test_gate_returning_wrong_verdict_count_fails_the_round(tmp_path: Path):
    config = _config(tmp_path, miner=_Discover(["alpha", "beta"]), gate=_WrongCountGate())
    store = LibraryStore(config.library)
    store.publish([_write_skill(tmp_path / "src", "base")])
    learner = SkillLearner(store, config, driver_factory=lambda root: None)
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.status == "failed"
    assert "verdicts" in (report.error or "")
    assert store.current_version() == 1
    assert [d.name for d in store.skill_dirs()] == ["base"]


class _ForeignVerdictGate:
    """A custom ``Gate`` that returns the right number of verdicts, about candidates it was not given."""

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN202
        return [Verdict(Candidate(c.skill), True, 1.0, "copied", {}) for c in candidates]


def test_gate_returning_verdicts_for_other_candidates_fails_the_round(tmp_path: Path):
    config = _config(tmp_path, miner=_Discover(["alpha", "beta"]), gate=_ForeignVerdictGate())
    store = LibraryStore(config.library)
    store.publish([_write_skill(tmp_path / "src", "base")])
    learner = SkillLearner(store, config, driver_factory=lambda root: None)
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.status == "failed"
    assert "one per candidate" in (report.error or "")
    assert [d.name for d in store.skill_dirs()] == ["base"]


def test_skillify_rejects_gate_with_gate_settings(tmp_path: Path):
    from ai_functions.experimental.skills import skillify

    with pytest.raises(ValueError, match="gate="):
        skillify(library=tmp_path / "lib", replay=True, gate=_WrongCountGate(), min_gain=0.5)


class _ReplayDriverFake:
    """An ``EvalDriver`` recording each spec; every run scores ``score``."""

    def __init__(self, score: float | None) -> None:
        self.score = score
        self.specs: list = []

    async def run(self, specs):  # noqa: ANN001, ANN201
        self.specs.extend(specs)
        passed = self.score is not None and self.score >= 1.0
        return [RunResult(self.score, passed) for s in specs for _ in s.arms for _ in range(s.repeats)]


@pytest.mark.parametrize("score", [0.5, None])
def test_replay_runs_one_spec_per_call_and_returns_one_outcome_per_run(tmp_path: Path, score):  # noqa: ANN001
    from ai_functions.experimental.skills.learner import RecordedCall

    driver = _ReplayDriverFake(score)
    config = _config(tmp_path)
    learner = SkillLearner(LibraryStore(config.library), config, driver_factory=lambda root: driver)
    calls = [RecordedCall(call_id=f"c{i}", outcome=0.0, passed=False, events=[]) for i in range(2)]

    with_skill = asyncio.run(learner._replay([tmp_path / "s"], calls, 3))
    without = asyncio.run(learner._replay([], calls, 3))

    assert with_skill == without == [score] * 6
    assert [(s.task_id, s.arms, s.repeats) for s in driver.specs] == [
        ("c0", ("skill_on",), 3),
        ("c1", ("skill_on",), 3),
        ("c0", ("skill_off",), 3),
        ("c1", ("skill_off",), 3),
    ]
    assert driver.specs[0].skills == [tmp_path / "s"]


# ── _should_induce -- the default trigger ──
#
# Moved from ``test_skills_implicit_induce.py`` (deleted with the rest of
# that module) when ``default_should_induce`` moved into ``learner.py`` as
# ``_should_induce``.


def _results(*scores: float, passed: bool | None = None):  # noqa: ANN201
    return [RunResult(score, (score > 0.0) if passed is None else passed) for score in scores]


def test_default_trigger_declines_at_ceiling_with_no_spread():
    from ai_functions.experimental.skills.learner import _should_induce

    assert _should_induce(_results(1.0, 1.0, 1.0)) is False


def test_default_trigger_fires_when_a_run_failed():
    from ai_functions.experimental.skills.learner import _should_induce

    assert _should_induce(_results(1.0, 1.0, 0.0)) is True


def test_default_trigger_fires_on_spread_between_passing_runs():
    from ai_functions.experimental.skills.learner import _should_induce

    assert _should_induce(_results(1.0, 0.8, 0.6)) is True


def test_default_trigger_declines_below_three_runs():
    from ai_functions.experimental.skills.learner import _should_induce

    assert _should_induce(_results(1.0, 0.0)) is False


# ── Staged revisions ─────────────────────────────────────────────────────────


class _NoMiner:
    async def __call__(self, calls, out_dir):  # noqa: ANN001, ANN204
        return []


class _AdmitRevisions:
    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN204
        return [Verdict(c, c.replaces is not None, 1.0, "fake", {"stage": "trial", "gain": 1.0}) for c in candidates]


def _revision_learner(tmp_path: Path, gate) -> SkillLearner:  # noqa: ANN001
    config = _config(tmp_path, miner=_NoMiner(), gate=gate)
    store = LibraryStore(config.library, keep_versions=config.keep_versions)
    return SkillLearner(store, config, driver_factory=lambda root: None)


def _stage(learner: SkillLearner, slug: str, body: str) -> None:
    src = _write_skill(learner.store.root.parent / "staged-src" / body.replace(" ", "-"), slug, body=body)
    learner.store.stage_revision(slug, src)


def _replay_where_pending_scores(pending_value: float, *, otherwise: float):  # noqa: ANN202
    async def replay(skill_dirs, calls, repeats):  # noqa: ANN001, ANN202
        hit = any("pending" in str(p) or "rounds" in str(p) for p in skill_dirs)
        return [pending_value if hit else otherwise] * (len(calls) * repeats)

    return replay


def test_a_pending_revision_that_helps_is_published(tmp_path: Path):
    learner = _revision_learner(tmp_path, _AdmitRevisions())
    learner.store.publish([_write_skill(tmp_path / "v1", "alpha", body="old text")])
    _stage(learner, "alpha", "new text")
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)
    report = asyncio.run(learner.run_round())
    assert report.status == "admitted" and list(report.revised) == ["alpha"]
    assert "new text" in (learner.store.version_dir(2) / "alpha" / "SKILL.md").read_text()
    assert "old text" in (learner.store.version_dir(1) / "alpha" / "SKILL.md").read_text()
    assert learner.store.take_pending(tmp_path / "taken") == []


def test_a_worse_revision_is_rejected_and_the_old_text_stays(tmp_path: Path):
    learner = _revision_learner(tmp_path, CompletionGate(repeats=1))
    learner._replay = _replay_where_pending_scores(0.0, otherwise=1.0)
    learner.store.publish([_write_skill(tmp_path / "v1", "alpha", body="old text")])
    _stage(learner, "alpha", "worse text")
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)
    report = asyncio.run(learner.run_round())
    assert report.rejected["alpha"]["stage"] == "trial" and report.revised == {}
    assert learner.store.current_version() == 1
    assert learner.store.take_pending(tmp_path / "taken") == []


def test_a_revision_of_a_skill_no_longer_installed_is_discarded(tmp_path: Path):
    learner = _revision_learner(tmp_path, _AdmitRevisions())
    learner.store.publish([_write_skill(tmp_path / "v1", "alpha", body="old text")])
    _stage(learner, "alpha", "new text")
    learner.store.rollback(to=0)
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)
    report = asyncio.run(learner.run_round())
    assert report.status != "failed"
    assert report.rejected["alpha"]["stage"] == "stale" and "alpha" in report.rejected["alpha"]["rationale"]


def test_a_staged_revision_does_not_trigger_a_round(tmp_path: Path):
    async def scenario() -> None:
        learner = _revision_learner(tmp_path, _AdmitRevisions())
        learner.store.publish([_write_skill(tmp_path / "v1", "alpha", body="old text")])
        _stage(learner, "alpha", "new text")
        assert learner.maybe_start_round() is None

    asyncio.run(scenario())


class _VerdictOnRevisions:
    """Admits (or rejects) every revision; records the candidates it was given."""

    def __init__(self, admit: bool) -> None:
        self.admit = admit
        self.seen: list[str | None] = []

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN204
        self.seen = [c.replaces for c in candidates]
        return [Verdict(c, self.admit, 1.0 if self.admit else -1.0, "fake", {"stage": "trial"}) for c in candidates]


@pytest.mark.parametrize("admit", [True, False])
def test_a_round_with_no_signal_still_gates_staged_revisions(tmp_path: Path, admit: bool):
    gate = _VerdictOnRevisions(admit)
    miner = _Discover(["alpha-new"])
    config = _config(tmp_path, miner=miner, gate=gate)
    learner = SkillLearner(LibraryStore(config.library), config, driver_factory=lambda root: None)
    learner.store.publish([_write_skill(tmp_path / "v1", "alpha", body="old text")])
    _stage(learner, "alpha", "new text")
    for _ in range(3):
        _record(learner, 1.0)  # no failure, no spread -> nothing to mine

    report = asyncio.run(learner.run_round())

    assert miner.calls == [] and report.mined == []
    assert gate.seen == ["alpha"]
    if admit:
        assert report.status == "admitted" and list(report.revised) == ["alpha"]
        assert "new text" in (learner.store.version_dir(2) / "alpha" / "SKILL.md").read_text()
    else:
        assert report.status == "none_admitted" and report.rejected["alpha"]["stage"] == "trial"
        assert learner.store.current_version() == 1
    assert learner.store.take_pending(tmp_path / "taken") == []


class _RollingBackRevisionGate:
    """Admits every revision, but rolls the library back while deciding (optionally staging a newer one first)."""

    def __init__(self, learner: SkillLearner, newer: bool) -> None:
        self.learner, self.newer = learner, newer

    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN204
        if self.newer:
            _stage(self.learner, "alpha", "newer text")
        self.learner.store.rollback()
        return [Verdict(c, True, 1.0, "fake", {"stage": "trial"}) for c in candidates]


@pytest.mark.parametrize("newer", [False, True])
def test_a_stale_publish_puts_the_revisions_back(tmp_path: Path, newer: bool):
    learner = _revision_learner(tmp_path, None)
    learner._gate = _RollingBackRevisionGate(learner, newer)
    learner.store.publish([_write_skill(tmp_path / "v1", "alpha", body="old text")])
    learner.store.publish([_write_skill(tmp_path / "v2", "alpha", body="old text")])
    _stage(learner, "alpha", "new text")
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)

    report = asyncio.run(learner.run_round())

    assert report.status == "failed" and "library changed during the round" in (report.error or "")
    assert learner.store.current_version() == 1
    [(slug, path)] = learner.store.take_pending(tmp_path / "taken")
    assert slug == "alpha"
    assert ("newer text" if newer else "new text") in (path / "SKILL.md").read_text()


class _RaisingGate:
    async def __call__(self, candidates, current, trials, replay):  # noqa: ANN001, ANN204
        raise RuntimeError("gate blew up")


def test_a_round_that_fails_before_gating_puts_its_revisions_back(tmp_path: Path):
    learner = _revision_learner(tmp_path, _RaisingGate())
    learner.store.publish([_write_skill(tmp_path / "v1", "alpha", body="old text")])
    _stage(learner, "alpha", "new text")
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)
    report = asyncio.run(learner.run_round())
    assert report.status == "failed" and learner.store.current_version() == 1
    [(slug, path)] = learner.store.take_pending(tmp_path / "taken")
    assert slug == "alpha" and "new text" in (path / "SKILL.md").read_text()


def test_a_malformed_revision_is_rejected_and_the_valid_ones_are_still_gated(tmp_path: Path):
    learner = _revision_learner(tmp_path, _AdmitRevisions())
    learner.store.publish(
        [_write_skill(tmp_path / "v1", "alpha", body="old text"), _write_skill(tmp_path / "v1", "beta", body="old")]
    )
    _stage(learner, "alpha", "new text")
    broken = tmp_path / "broken" / "beta"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text("---\n- not: a mapping\n---\nbody\n")
    learner.store.stage_revision("beta", broken)
    for outcome in (0.0, 1.0, 1.0):
        _record(learner, outcome)
    report = asyncio.run(learner.run_round())
    assert report.status == "admitted" and list(report.revised) == ["alpha"]
    assert report.rejected["beta"]["stage"] == "invalid" and "mapping" in report.rejected["beta"]["rationale"]
    assert "new text" in (learner.store.version_dir(2) / "alpha" / "SKILL.md").read_text()
    assert "old" in (learner.store.version_dir(2) / "beta" / "SKILL.md").read_text()
    assert learner.store.take_pending(tmp_path / "taken") == []
