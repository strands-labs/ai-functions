"""``@skillify`` -- the decorator end to end over scripted models."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import pytest

from ai_functions import ai_function, scope
from ai_functions.ai_thread import AIFunctionError, PostConditionResult
from ai_functions.experimental.economics import PricedModel, Prices, routed
from ai_functions.experimental.skills import SkillifiedFunction, skillify
from ai_functions.experimental.skills.skill_library import SkillDeliveryError
from ai_functions.runtime import InMemoryCoordinator
from ai_functions.runtime.worker import LocalWorker
from ai_functions.testing import ScriptedModel, ScriptExhausted, Turn


def _write_skill(root: Path, name: str, description: str = "does x", body: str = "Step 1. Do x.") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\n{body}\n")
    return directory


def _require_yes(result, **kwargs):  # noqa: ANN001, ANN003, ANN202
    if "yes" not in result:
        return PostConditionResult(passed=False, message="must contain yes")
    return None


@ai_function[str](structured_output=False, post_conditions=[_require_yes], max_attempts=0)
def _solve(task: str) -> str:
    """Solve: {task}"""


def _decorate(tmp_path: Path, turns: list[Turn], **kwargs) -> SkillifiedFunction:  # noqa: ANN003
    options = {"library": tmp_path / "lib", "replay": True, "every": 1000}
    options.update(kwargs)
    return skillify(**options)(_solve.replace(model=ScriptedModel(turns)))


def test_replay_must_be_explicit(tmp_path: Path):
    with pytest.raises(ValueError, match="replay"):
        skillify(library=tmp_path / "lib")(_solve)


def test_warns_when_nothing_can_fail(tmp_path: Path):
    @ai_function[str](structured_output=False)
    def plain(task: str) -> str:
        """Do: {task}"""

    with pytest.warns(UserWarning, match="always decline"):
        skillify(library=tmp_path / "lib", replay=True)(plain)


def test_the_default_miner_runs_on_the_decorated_function_s_model(tmp_path: Path):
    model = ScriptedModel([])
    fn = skillify(library=tmp_path / "lib", replay=True)(_solve.replace(model=model))

    assert fn.learner._miner.model is model


async def test_empty_library_call_is_the_plain_function_and_is_recorded(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="yes")])

    assert (await fn(task="t")).strip() == "yes"  # ScriptedModel ends text with a newline

    [record] = fn.learner.store.records()
    assert record.outcome == 1.0 and record.version == 0 and record.skills_available == []
    assert fn.learner.store.load_events(record.call_id)
    assert fn.learner.store.load_args(record.call_id) == ((), {"task": "t"})


async def test_post_condition_failure_is_recorded_as_zero_and_re_raised(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="no")])

    with pytest.raises(AIFunctionError) as info:
        await fn(task="t")

    assert type(info.value) is AIFunctionError  # K3: the runtime's exception, unchanged
    [record] = fn.learner.store.records()
    assert record.outcome == 0.0 and record.error is not None


async def test_an_infrastructure_error_is_recorded_unscored(tmp_path: Path):
    fn = _decorate(tmp_path, [])  # ScriptExhausted

    with pytest.raises(ScriptExhausted):  # K3: the model's own error, unchanged
        await fn(task="t")

    [record] = fn.learner.store.records()
    assert record.outcome is None and record.error == "ScriptExhausted"


async def test_score_overrides_post_conditions(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="yes")], score=lambda a, r: 0.25 if a["task"] == "t" else 1.0)

    await fn(task="t")

    assert fn.learner.store.records()[0].outcome == 0.25


async def test_installed_skills_are_shown_and_loads_are_recorded(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(tool_calls=(("Skill", {"skill": "alpha"}),)), Turn(text="yes")])
    fn.learner.store.publish([_write_skill(tmp_path / "src", "alpha", body="Say yes.")])

    assert (await fn(task="t")).strip() == "yes"  # ScriptedModel ends text with a newline

    [record] = fn.learner.store.records()
    assert record.version == 1
    assert record.skills_available == ["alpha"]
    assert record.skills_loaded == ["alpha"]
    assert [s.name for s in fn.skills] == ["alpha"]


async def test_a_library_that_cannot_be_loaded_is_a_delivery_error_and_is_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The package re-exports the `skillify` function, which shadows the submodule attribute.
    module = importlib.import_module("ai_functions.experimental.skills.skillify")

    fn = _decorate(tmp_path, [Turn(text="yes")])
    fn.learner.store.publish([_write_skill(tmp_path / "src", "alpha")])

    def _corrupt(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("corrupt library")

    monkeypatch.setattr(module, "SkillLibraryBackend", _corrupt)

    with pytest.raises(SkillDeliveryError, match="could not be loaded"):
        await fn(task="t")

    assert fn.learner.store.records() == []


async def test_skill_backend_is_none_for_an_empty_library_and_staged_otherwise(tmp_path: Path):
    fn = _decorate(tmp_path, [])
    assert fn.skill_backend is None
    fn.learner.store.publish([_write_skill(tmp_path / "src", "alpha")])
    backend = fn.skill_backend
    assert backend is not None and backend.root == fn.learner.store.version_dir(1)
    assert backend.stage_root == fn.learner.store.pending_dir


async def test_trace_returns_a_result_node(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="yes")])

    result = await fn.trace(task="t")

    assert result.value.strip() == "yes"
    assert result.thread_id is not None


@pytest.mark.parametrize("traced", [False, True], ids=["call", "trace"])
async def test_calls_use_the_ambient_coordinator_scope(tmp_path: Path, traced: bool):
    fn = _decorate(tmp_path, [Turn(tool_calls=(("Skill", {"skill": "alpha"}),)), Turn(text="yes")])
    fn.learner.store.publish([_write_skill(tmp_path / "src", "alpha", body="Say yes.")])
    events = []

    async with scope(on_event=events.append) as coordinator:
        if traced:
            result = await fn.trace(task="t")
            assert result.coordinator is coordinator
            value = result.value
        else:
            value = await fn(task="t")

        assert value.strip() == "yes"
        [record] = fn.learner.store.records()
        assert record.outcome == 1.0 and record.skills_loaded == ["alpha"]
        assert any(event.kind == "tool_call" for event in events)

    assert await coordinator.list_threads() == []


async def test_rollback_and_history(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="yes")])
    fn.learner.store.publish([_write_skill(tmp_path / "a", "alpha")])

    assert fn.rollback_skills() == 0
    assert fn.skills == []
    assert fn.history()[-1]["type"] == "rollback"


async def test_refresh_skills_runs_a_round_now(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="yes")])
    await fn(task="t")

    report = await fn.refresh_skills()

    # One passing call is below the three-call floor, so the round declines.
    assert report.status == "declined"
    assert fn.learner.pending == 0


async def test_routed_stacks_on_skillify(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="unused")])
    cheap = PricedModel(
        model=ScriptedModel([Turn(text="yes", input_tokens=1, output_tokens=1)]),
        prices=Prices(input=1.0, output=1.0),
        label="cheap",
    )

    stacked = routed(value=1.0, models=[cheap])(fn)
    result = await stacked(task="t")

    assert "yes" in result
    assert len(fn.learner.store.records()) == 1


async def test_a_recording_failure_is_logged_and_never_breaks_the_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    fn = _decorate(tmp_path, [Turn(text="yes")])

    def _broken(**kwargs):  # noqa: ANN003, ANN202
        raise OSError("disk full")

    monkeypatch.setattr(fn.learner, "record", _broken)

    with caplog.at_level(logging.ERROR, logger="ai_functions.experimental.skills.skillify"):
        assert (await fn(task="t")).strip() == "yes"

    assert any("recording" in r.getMessage() and r.exc_info for r in caplog.records)


async def test_a_nested_call_tears_down_its_child_thread(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="yes")])
    coord = InMemoryCoordinator()
    await LocalWorker(coord).register()

    handle = await coord.spawn(fn)
    assert (await handle.run(task="t")).strip() == "yes"

    # Only the skillified thread itself is still registered; its per-call child is gone.
    assert [info.thread_id for info in await coord.list_threads()] == [handle.id]
    await handle.terminate_now()


def test_run_sync_records_the_call_but_starts_no_round(tmp_path: Path):
    fn = _decorate(tmp_path, [Turn(text="yes")], every=1)

    assert fn.run_sync(task="t").strip() == "yes"

    assert len(fn.learner.store.records()) == 1
    assert fn.learner.pending == 1
    assert fn.history() == []


def _scripted_round(tmp_path: Path, model: ScriptedModel, discover, environment=None):  # noqa: ANN001, ANN202
    """A decorated function whose rounds replay over ``model`` with real gate trials.

    The default gate (``CompletionGate``, no LLM judge involved) replays through the
    learner's own replay; the miner is ``discover``; ``environment`` goes to both the
    config and the replay driver, as ``@skillify`` wires it.
    """
    from ai_functions.experimental.skills.learner import ReplayDriver, SkillifyConfig, SkillLearner
    from ai_functions.experimental.skills.skill_library import LibraryStore

    fn = _solve.replace(model=model)
    config = SkillifyConfig(
        library=tmp_path / "lib",
        replay=True,
        every=1000,
        trial_inputs=2,
        trial_repeats=1,
        environment=environment,
        miner=discover,
    )
    store = LibraryStore(config.library)

    def driver(root: Path) -> ReplayDriver:
        return ReplayDriver(fn, store, score=None, pass_at=1.0, work_root=root, max_attempts=1, environment=environment)

    learner = SkillLearner(store, config, driver_factory=driver)
    return SkillifiedFunction(fn, config, learner=learner), store


async def test_a_full_round_admits_a_skill_that_turns_failures_into_passes(tmp_path: Path):
    """Calls fail until the agent loads a mined skill; the round must find and install it.

    The scripted model answers "no" unless the Skill tool is available and
    it has loaded "say-yes". Discovery is faked to return that skill. The
    replayed gate trial must then show: baseline (no skills) fails, with the
    candidate passes, so the candidate is admitted and published.
    """
    from ai_functions.experimental.skills.skill_library import load_skill
    from ai_functions.testing.scripted_model import _stream_turn

    class _Model(ScriptedModel):
        """Says yes only after a Skill tool result is in the conversation.

        With the Skill tool offered and nothing loaded yet, it loads
        "say-yes". Without the tool it answers "no", failing the post-condition.
        """

        def __init__(self) -> None:
            super().__init__([])

        def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
            has_tool = any(spec.get("name") == "Skill" for spec in (tool_specs or []))
            loaded = any("toolResult" in block for message in messages for block in message.get("content", []))
            if has_tool and not loaded:
                turn = Turn(tool_calls=(("Skill", {"skill": "say-yes"}),))
            else:
                turn = Turn(text="yes" if loaded else "no")
            return _stream_turn(turn)

    async def discover(calls, out_dir):  # noqa: ANN001, ANN202
        assert len(calls) == 3 and len([c for c in calls if not c.passed]) == 3
        return [load_skill(_write_skill(out_dir, "say-yes", "answer yes", "Always answer yes."))]

    decorated, store = _scripted_round(tmp_path, _Model(), discover)

    for _ in range(3):
        with pytest.raises(Exception):  # noqa: B017 - post-condition failure
            await decorated(task="t")

    report = await decorated.refresh_skills()

    assert report.status == "admitted", report
    assert list(report.admitted) == ["say-yes"]
    assert report.admitted["say-yes"]["scores"]["gain"] == 1.0
    assert [s.name for s in decorated.skills] == ["say-yes"]
    # The gate's replays were not recorded as calls.
    assert len(store.records()) == 3
    # The next call gets the skill and passes.
    assert (await decorated(task="t")).strip() == "yes"  # ScriptedModel ends text with a newline


async def test_each_candidate_is_trialled_without_the_candidates_before_it(tmp_path: Path):
    """A useless candidate must not be admitted by riding on an earlier, useful one.

    The model always tries to load "say-yes" and passes only when that
    skill's body actually comes back. "say-yes" is trialled first, "noise"
    second: if the second trial's library still held "say-yes" from the
    first (same replay label, same run directory), "noise" would score a
    gain of 1.0 it did nothing to earn.
    """
    from ai_functions.experimental.skills.skill_library import load_skill
    from ai_functions.testing.scripted_model import _stream_turn

    class _Model(ScriptedModel):
        def __init__(self) -> None:
            super().__init__([])

        def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
            has_tool = any(spec.get("name") == "Skill" for spec in (tool_specs or []))
            results = [str(b) for m in messages for b in m.get("content", []) if "toolResult" in b]
            if has_tool and not results:
                turn = Turn(tool_calls=(("Skill", {"skill": "say-yes"}),))
            else:
                turn = Turn(text="yes" if any("Always answer yes" in r for r in results) else "no")
            return _stream_turn(turn)

    async def discover(calls, out_dir):  # noqa: ANN001, ANN202
        return [
            load_skill(_write_skill(out_dir, "say-yes", "answer yes", "Always answer yes.")),
            load_skill(_write_skill(out_dir, "noise", "unrelated", "Do nothing useful.")),
        ]

    decorated, _ = _scripted_round(tmp_path, _Model(), discover)
    for _ in range(3):
        with pytest.raises(Exception):  # noqa: B017 - post-condition failure
            await decorated(task="t")

    report = await decorated.refresh_skills()

    assert list(report.admitted) == ["say-yes"], report
    assert report.rejected["noise"]["scores"]["gain"] == 0.0


# ── environment= ──


class _ButtonEpisode:
    """Per-episode state: a button press counter. Outcome is 1.0 iff the button was pressed in this episode."""

    def __init__(self, arguments, *, outcome=None, close_raises: bool = False) -> None:  # noqa: ANN001
        from strands import tool

        self.arguments = dict(arguments)
        self.presses = 0
        self.closed = 0
        self._outcome = outcome
        self._close_raises = close_raises

        @tool(name="press", description="Press the button.")
        def press() -> str:
            self.presses += 1
            return f"pressed {self.presses}"

        self.tools = [press]

    def outcome(self, result) -> float:  # noqa: ANN001
        if self._outcome is not None:
            return self._outcome(result)
        return 1.0 if self.presses == 1 else 0.0

    def close(self) -> None:
        self.closed += 1
        if self._close_raises:
            raise OSError("close failed")


class _ButtonEnvironment:
    def __init__(self, **episode_options) -> None:  # noqa: ANN003
        self.episodes: list[_ButtonEpisode] = []
        self._options = episode_options

    def __call__(self, arguments):  # noqa: ANN001, ANN204
        episode = _ButtonEpisode(arguments, **self._options)
        self.episodes.append(episode)
        return episode


_PRESS_THEN_YES = [Turn(tool_calls=(("press", {}),)), Turn(text="yes")]


async def test_a_live_call_is_scored_by_its_environment(tmp_path: Path):
    env = _ButtonEnvironment()
    fn = _decorate(tmp_path, [*_PRESS_THEN_YES, Turn(text="yes")], environment=env)

    assert (await fn(task="t")).strip() == "yes"
    assert (await fn(task="u")).strip() == "yes"  # returns, post-condition passes, but never presses

    first, second = fn.learner.store.records()
    assert first.outcome == 1.0
    assert second.outcome == 0.0
    assert [e.arguments for e in env.episodes] == [{"task": "t"}, {"task": "u"}]
    assert [e.presses for e in env.episodes] == [1, 0]
    # The episode's tools applied to that run only: the base function is untouched.
    assert fn._fn.config.tools == _solve.config.tools


async def test_an_async_environment_factory_is_awaited_on_a_live_call(tmp_path: Path):
    env = _ButtonEnvironment()

    async def factory(arguments):  # noqa: ANN001, ANN202
        return env(arguments)

    fn = _decorate(tmp_path, _PRESS_THEN_YES, environment=factory)

    await fn(task="t")

    assert fn.learner.store.records()[0].outcome == 1.0
    assert env.episodes[0].closed == 1


async def test_close_runs_on_success_on_post_condition_failure_and_on_other_errors(tmp_path: Path):
    env = _ButtonEnvironment()
    ok = _decorate(tmp_path / "a", _PRESS_THEN_YES, environment=env)
    await ok(task="t")

    failing = _decorate(tmp_path / "b", [Turn(text="no")], environment=env)
    with pytest.raises(AIFunctionError):
        await failing(task="t")

    broken = _decorate(tmp_path / "c", [], environment=env)
    with pytest.raises(ScriptExhausted):
        await broken(task="t")

    assert [e.closed for e in env.episodes] == [1, 1, 1]
    assert failing.learner.store.records()[0].outcome == 0.0
    assert broken.learner.store.records()[0].outcome is None


async def test_a_raising_close_is_logged_and_masks_neither_result_nor_exception(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    env = _ButtonEnvironment(close_raises=True)
    ok = _decorate(tmp_path / "a", _PRESS_THEN_YES, environment=env)
    failing = _decorate(tmp_path / "b", [Turn(text="no")], environment=env)

    with caplog.at_level(logging.WARNING, logger="ai_functions.experimental.skills"):
        assert (await ok(task="t")).strip() == "yes"
        with pytest.raises(AIFunctionError):
            await failing(task="t")

    assert ok.learner.store.records()[0].outcome == 1.0
    assert sum("close" in r.getMessage() for r in caplog.records) == 2


def test_score_and_environment_together_are_rejected(tmp_path: Path):
    from ai_functions.experimental.skills import SkillifyConfig

    env = _ButtonEnvironment()
    with pytest.raises(ValueError, match="environment"):
        skillify(library=tmp_path / "lib", replay=True, score=lambda a, r: 1.0, environment=env)
    with pytest.raises(ValueError, match="environment"):
        SkillifyConfig(library=tmp_path / "lib", replay=True, score=lambda a, r: 1.0, environment=env)


def test_config_rejects_environment_and_score(tmp_path: Path):
    from ai_functions.experimental.skills import SkillifyConfig

    config = SkillifyConfig(library=tmp_path / "lib", replay=True)
    with pytest.raises(ValueError, match="environment"):
        skillify(config=config, environment=_ButtonEnvironment())
    with pytest.raises(ValueError, match="score"):
        skillify(config=config, score=lambda a, r: 1.0)


def test_an_environment_counts_as_an_outcome_source_for_the_warning(tmp_path: Path):
    import warnings

    @ai_function[str](structured_output=False)
    def plain(task: str) -> str:
        """Do: {task}"""

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        skillify(library=tmp_path / "lib", replay=True, environment=_ButtonEnvironment())(plain)


@pytest.mark.parametrize("outcome", [lambda r: 1 / 0, lambda r: 1.5, lambda r: -0.1])
async def test_a_raising_or_out_of_range_environment_outcome_is_unscored(tmp_path: Path, outcome):  # noqa: ANN001
    env = _ButtonEnvironment(outcome=outcome)
    fn = _decorate(tmp_path, _PRESS_THEN_YES, environment=env)

    assert (await fn(task="t")).strip() == "yes"

    [record] = fn.learner.store.records()
    assert record.outcome is None
    assert env.episodes[0].closed == 1


async def test_a_raising_environment_factory_reaches_the_caller_and_nothing_is_recorded(tmp_path: Path):
    class _EnvDown(RuntimeError):
        pass

    def factory(arguments):  # noqa: ANN001, ANN202
        raise _EnvDown("environment could not start")

    fn = _decorate(tmp_path, _PRESS_THEN_YES, environment=factory)

    with pytest.raises(_EnvDown):
        await fn(task="t")

    assert fn.learner.store.records() == []
    assert fn.learner.pending == 0


async def test_a_round_admits_a_skill_whose_effect_only_the_environment_can_see(tmp_path: Path):
    """The answer is always "yes"; only the environment knows whether the button was pressed.

    Without skills the model answers straight away, so every live call and every
    baseline replay scores 0.0 in its episode. With the "press-first" skill loaded,
    it presses, then answers. The candidate is admitted only if replays rebuild the
    environment and read its outcome: judged by the return value, both arms would
    pass and the gain would be 0.
    """
    from ai_functions.experimental.skills.skill_library import load_skill
    from ai_functions.testing.scripted_model import _stream_turn

    class _Model(ScriptedModel):
        def __init__(self) -> None:
            super().__init__([])

        def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):  # noqa: ANN001, ANN003, ANN201
            has_skill_tool = any(spec.get("name") == "Skill" for spec in (tool_specs or []))
            results = sum("toolResult" in b for m in messages for b in m.get("content", []))
            if has_skill_tool and results == 0:
                turn = Turn(tool_calls=(("Skill", {"skill": "press-first"}),))
            elif has_skill_tool and results == 1:
                turn = Turn(tool_calls=(("press", {}),))
            else:
                turn = Turn(text="yes")
            return _stream_turn(turn)

    async def discover(calls, out_dir):  # noqa: ANN001, ANN202
        assert len([c for c in calls if not c.passed]) == 3
        return [load_skill(_write_skill(out_dir, "press-first", "press the button", "Press the button first."))]

    env = _ButtonEnvironment()
    decorated, store = _scripted_round(tmp_path, _Model(), discover, environment=env)

    for _ in range(3):
        assert (await decorated(task="t")).strip() == "yes"
    assert [r.outcome for r in store.records()] == [0.0, 0.0, 0.0]

    report = await decorated.refresh_skills()

    assert report.status == "admitted", report
    assert report.admitted["press-first"]["scores"]["gain"] == 1.0
    # 3 live calls + 2 trial inputs x (baseline + candidate) x 1 repeat, each its own episode.
    assert len(env.episodes) == 3 + 2 * 2
    assert all(e.closed == 1 for e in env.episodes)
    assert all(e.presses <= 1 for e in env.episodes)

    await decorated(task="t")
    assert store.records()[-1].outcome == 1.0


# ── Skills are graph parameters ──────────────────────────────────────────────


class _FeedbackOnEveryInput:
    """Offline backward stand-in: one feedback item for every target the optimizer offers."""

    def replace(self, **kwargs: object) -> _FeedbackOnEveryInput:
        del kwargs
        return self

    def run_sync(self, *, inputs: str, **kwargs: object) -> object:
        import yaml

        from ai_functions.optimizer.textgrad import Feedback, Feedbacks

        del kwargs
        rendered = yaml.safe_load(inputs) or {}
        return Feedbacks(feedbacks=[Feedback(node_id=nid, feedback="say it better", score=0.5) for nid in rendered])


async def test_a_loaded_skill_is_a_graph_parameter_and_optimizing_it_stages_a_revision(tmp_path: Path):
    """Skills are parameters: the Skill tool's recall becomes a graph edge, and an
    optimizer step over that edge stages -- never publishes -- a rewrite."""
    from ai_functions.optimizer import TextGradOptimizer, build_graph_from_result

    fn = _decorate(tmp_path, [Turn(tool_calls=(("Skill", {"skill": "alpha"}),)), Turn(text="yes")])
    store = fn.learner.store
    store.publish([_write_skill(tmp_path / "src", "alpha", body="Say yes.")])
    published = (store.version_dir(1) / "alpha" / "SKILL.md").read_text()

    result = await fn.trace(task="t")

    backend = fn.skill_backend
    assert backend is not None
    graph = await build_graph_from_result(result, [backend])
    params = [p for node in _walk(graph) for p in node.parameters if p.name == backend.param_for("alpha")]
    assert params, "the Skill tool's recall of 'alpha' is not a parameter node of the traced call"
    assert params[0].requires_grad and params[0].backend is backend

    backend._consolidate_fn = lambda value, feedback, description="": "rewritten"  # noqa: SLF001
    opt = TextGradOptimizer()
    opt._backward_fn = _FeedbackOnEveryInput()  # noqa: SLF001
    await opt.step(result, "the answer should explain itself", backends=[backend])

    assert "rewritten" in (store.pending_dir / "alpha" / "SKILL.md").read_text()
    assert (store.version_dir(1) / "alpha" / "SKILL.md").read_text() == published
    assert store.current_version() == 1
    assert [s.body.strip() for s in fn.skills] == ["Say yes."]


def _walk(node):  # noqa: ANN001, ANN202
    yield node
    for child in node.child_threads:
        yield from _walk(child)
