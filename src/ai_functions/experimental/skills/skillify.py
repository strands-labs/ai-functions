"""``@skillify`` -- an AI Function that improves as it is called.

The function is the skill: an ability that improves with experience.
``@skillify`` improves it by growing its procedural memory, a library of
reusable procedures mined from its own calls; each procedure is stored as a
``SKILL.md`` file and loaded through a ``Skill`` tool, which is why this API
calls a stored procedure a skill.

Stacks on ``@ai_function`` the way ``@routed`` does::

    @skillify(library="skills/solve", replay=True)
    @ai_function(tools=[...], post_conditions=[...])
    async def solve(task: str) -> Report: ...

Every call runs the function with the current library's index and ``Skill``
tool (or unchanged, while the library is empty), then records the call's
outcome, arguments and event log. Every ``every`` calls a background round
(:class:`~.learner.SkillLearner`) decides whether those calls have anything
to teach, mines candidate skills, keeps the ones that improve on the current
library when recorded calls are replayed, and publishes a new library
version. Callers never wait for a round.

Outcome of a call: ``score(arguments, result)`` if given, else ``1.0`` when
it returned and ``0.0`` when it raised ``AIFunctionError`` (post-conditions
exhausted). Any other exception leaves the call unscored. Every exception
reaches the caller unchanged; the learner never breaks a call.

With ``environment=``, each call first builds an episode from its own
arguments (``environment(arguments)``, awaited if awaitable), runs with the
episode's tools added for that call only, and is scored by
``episode.outcome(result)``; ``AIFunctionError`` is still 0.0 and any other
exception unscored. ``episode.close()`` always runs; a close that raises is
logged. A factory that raises reaches the caller and the call is not
recorded. Every gate replay rebuilds its own episode from the recorded
arguments the same way.

``replay=True`` is required: gating re-runs recorded calls, so the
function's tools must be safe to call again.

Stored procedures are parameters: a skill the agent loads through the
``Skill`` tool is a parameter node of the traced call's graph -- procedural
memory that an optimizer step over ``trace()``'s result can rewrite. That rewrite is staged in the
library's ``pending/`` directory, never published directly; the next round
gates it like any candidate (``RoundReport.revised``).

Background rounds need an event loop that outlives the call. Under
``run_sync`` the call is recorded but no round is started; the pending count
is on disk, so the next async call or ``refresh_skills()`` runs it.

Gate replays run the undecorated base function -- the one passed to
``skillify``, with the model it was configured with -- including when
``@routed`` is stacked on top: variants built by ``replace`` share the
learner, so a round measures skills on that base function, not on whichever
model a router would pick.
"""

from __future__ import annotations

import asyncio
import logging
import warnings
from collections.abc import Callable
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ai_functions.ai_thread.ai_function import AIFunction
from ai_functions.ai_thread.errors import AIFunctionError

from .discovery import Miner
from .evaluation import Gate
from .learner import (
    Environment,
    Episode,
    ReplayDriver,
    RoundReport,
    Score,
    SkillifyConfig,
    SkillLearner,
    bind_arguments,
    close_episode,
    episode_outcome,
    open_episode,
    outcome_of,
    with_episode_tools,
)
from .skill_library import (
    LibraryStore,
    Skill,
    SkillDeliveryError,
    SkillLibraryBackend,
    SkillToolbox,
    load_skill,
    with_skills,
)

if TYPE_CHECKING:
    from ai_functions.handle import ThreadHandle
    from ai_functions.protocols import ThreadContext
    from ai_functions.types import InputShape, ThreadId
    from ai_functions.types.graph import Result

__all__ = ["RoundReport", "SkillifiedFunction", "SkillifiedThread", "SkillifyConfig", "skillify"]

_logger = logging.getLogger(__name__)

_SYNC_CALL: ContextVar[bool] = ContextVar("skillify_sync_call", default=False)
"""Set while ``run_sync`` runs: its event loop ends with the call, so no round is started on it."""


class SkillifiedThread[**P, T]:
    """Live thread for one call: pick the library version, run, record."""

    def __init__(self, fn: SkillifiedFunction[P, T]) -> None:
        self._fn = fn

    @property
    def name(self) -> str:
        """The wrapped function's name."""
        return self._fn.name

    async def execute(self, ctx: ThreadContext, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run the wrapped function as a child thread with the current skills; record the call."""
        from .observation import _scope

        with _scope(
            function=self.name,
            library=str(self._fn.learner.store.root.resolve()),
            run_id=str(ctx.thread_id),
            phase="call",
            round_id=None,
        ):
            return await self._execute(ctx, *args, **kwargs)

    async def _execute(self, ctx: ThreadContext, *args: P.args, **kwargs: P.kwargs) -> T:
        fn = self._fn
        version = fn.learner.store.current_version()
        backend = fn._backend(version)
        episode = None
        if fn.config.environment is not None:
            # Raises to the caller before anything runs or is recorded.
            episode = await open_episode(fn.config.environment, bind_arguments(fn._fn, args, kwargs))
        try:
            return await self._run(ctx, version, backend, episode, args, kwargs)
        finally:
            if episode is not None:
                close_episode(episode)

    async def _run(
        self,
        ctx: ThreadContext,
        version: int,
        backend: SkillLibraryBackend | None,
        episode: Episode | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> T:
        fn = self._fn
        variant, toolbox = with_skills(with_episode_tools(fn._fn, episode), backend)
        info = await ctx.coordinator.get_thread_info(ctx.thread_id)
        handle = await ctx.coordinator.spawn(variant, worker_id=info.worker_id, parent_id=ctx.thread_id)
        from .observation import _watch_thread, skill_event

        skill_event(
            "run.started",
            version=version,
            thread_id=str(handle.id),
            inputs={"args": args, "kwargs": kwargs},
            skill_root=fn.learner.store.version_dir(version),
        )
        try:
            try:
                with _watch_thread(ctx.coordinator, handle.id):
                    result = await handle.run(*args, **kwargs)
            except asyncio.CancelledError:
                skill_event("run.finished", status="cancelled")
                raise
            except AIFunctionError as exc:
                await fn._record(ctx, handle.id, version, toolbox, args, kwargs, 0.0, type(exc).__name__)
                raise
            except Exception as exc:
                await fn._record(ctx, handle.id, version, toolbox, args, kwargs, None, type(exc).__name__)
                raise
            if episode is not None:
                outcome = episode_outcome(episode, result)
            else:
                outcome = outcome_of(fn._fn, fn.config.score, args, kwargs, result)
            await fn._record(ctx, handle.id, version, toolbox, args, kwargs, outcome, None)
            return result
        finally:
            # After _record has read the child's events: a finished call leaves no live child thread.
            try:
                await handle.terminate_now()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                _logger.debug("skillify: failed to terminate child thread %s", handle.id, exc_info=True)

    async def notify(self, text: str) -> None:
        """No-op; the call's child thread owns the conversation."""
        del text

    def serialize_result(self, result: T) -> str:
        """Serialize via the wrapped function's thread."""
        return self._fn._fn.to_thread().serialize_result(result)

    def deserialize_result(self, payload: str) -> T:
        """Deserialize via the wrapped function's thread."""
        return self._fn._fn.to_thread().deserialize_result(payload)

    async def fork(self) -> SkillifiedFunction[P, T]:
        """Return the decorated function as a spawnable."""
        return self._fn

    async def teardown(self) -> None:
        """No-op teardown."""


class SkillifiedFunction[**P, T]:
    """An AI Function plus the learner that grows its skill library.

    Construct with :func:`skillify`. ``replace`` returns a function sharing
    this one's learner and library, which is what lets ``@routed`` stack on
    top: every model variant it builds records into the same library.
    """

    def __init__(self, fn: AIFunction[P, T], config: SkillifyConfig, *, learner: SkillLearner | None = None) -> None:
        self._fn = fn
        self.config = config
        if learner is None:
            store = LibraryStore(config.library, keep_versions=config.keep_versions)
            learner = SkillLearner(
                store,
                config,
                driver_factory=lambda root: ReplayDriver(
                    fn,
                    store,
                    score=config.score,
                    pass_at=config.pass_at,
                    work_root=root,
                    max_concurrency=config.max_concurrency,
                    environment=config.environment,
                ),
                model=fn.config.model,
            )
        self._learner = learner
        self._backends: dict[int, SkillLibraryBackend] = {}

    # ── Spawnable ──

    def to_thread(self) -> SkillifiedThread[P, T]:
        """A fresh live thread that runs one recorded call."""
        return SkillifiedThread(self)

    @property
    def input_shape(self) -> InputShape:
        """The wrapped function's input shape."""
        return self._fn.input_shape

    @property
    def name(self) -> str:
        """The wrapped function's name."""
        return self._fn.name

    def replace(self, **kwargs: Any) -> SkillifiedFunction[P, T]:  # noqa: ANN401
        """``fn.replace(**kwargs)``, sharing this function's learner and library."""
        return SkillifiedFunction(self._fn.replace(**kwargs), self.config, learner=self._learner)

    async def render_prompt(self, *args: Any, **kwargs: Any) -> str:  # noqa: ANN401
        """The wrapped function's rendered prompt (``@routed`` estimators read it)."""
        return await self._fn.render_prompt(*args, **kwargs)

    # ── Calling ──

    async def spawn(self) -> ThreadHandle[P, T]:
        """Spawn on a private in-process worker and return the handle."""
        from ai_functions.runtime import InMemoryCoordinator
        from ai_functions.runtime.worker import LocalWorker

        coord = InMemoryCoordinator()
        worker = LocalWorker(coord)
        await worker.register()
        return await coord.spawn(self)

    async def _spawn_in_context(self) -> ThreadHandle[P, T]:
        """Spawn for a one-shot call, reusing the ambient thread scope if set."""
        from ai_functions.types import current_thread_scope

        scope = current_thread_scope()
        if scope is None:
            return await self.spawn()
        if scope.thread_id is None:
            return await scope.coordinator.spawn(self)
        caller = await scope.coordinator.get_thread_info(scope.thread_id)
        return await scope.coordinator.spawn(self, worker_id=caller.worker_id, parent_id=scope.thread_id)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run one call with the current skills and record it."""
        handle = await self._spawn_in_context()
        try:
            return await handle.run(*args, **kwargs)
        finally:
            await handle.terminate_now()

    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Blocking wrapper around ``__call__``.

        The call is recorded, but no round is started: the loop ``run_blocking``
        creates ends with the call and would cancel it. The pending count is on
        disk, so the next async call or ``refresh_skills()`` runs the round.
        """
        from ai_functions.utils import run_blocking

        # Set in the calling context, so the copied context of run_blocking's task sees it.
        token = _SYNC_CALL.set(True)
        try:
            return run_blocking(lambda: self(*args, **kwargs))
        finally:
            _SYNC_CALL.reset(token)

    async def trace(self, *args: Any, **kwargs: Any) -> Result[T]:  # noqa: ANN401
        """Run like ``__call__`` and return a traced ``Result`` node."""
        from ai_functions.types.graph import ParameterView, Result, collect_nodes

        inputs = collect_nodes((args, kwargs))
        handle = await self._spawn_in_context()
        try:
            for node in inputs:
                if isinstance(node, ParameterView):
                    await node.backend.emit_recall(node, handle.coordinator, handle.id)
            value = await handle.run(*args, **kwargs)
        finally:
            await handle.terminate_now()
        return Result(value=value, coordinator=handle.coordinator, thread_id=handle.id, inputs=inputs)

    # ── The library ──

    @property
    def learner(self) -> SkillLearner:
        """The learner that records calls and runs rounds."""
        return self._learner

    @property
    def skills(self) -> list[Skill]:
        """Skills in the current version."""
        return [load_skill(d) for d in self._learner.store.skill_dirs()]

    @property
    def skill_backend(self) -> SkillLibraryBackend | None:
        """The current version as a staged ``SkillLibraryBackend``; ``None`` while the library is empty.

        Pass it to an optimizer (e.g. ``TextGradOptimizer.step(result, feedback,
        backends=[fn.skill_backend])``): its rewrites are staged in the library's
        ``pending/`` directory for the next round to gate, never published directly.

        Raises:
            SkillDeliveryError: The current version could not be loaded.
        """
        return self._backend(self._learner.store.current_version())

    def history(self) -> list[dict[str, Any]]:
        """Every round report and rollback, oldest first."""
        return self._learner.store.history()

    def rollback_skills(self, to: int | None = None) -> int:
        """Switch to the previous version (or ``to``); returns the version now current."""
        return self._learner.store.rollback(to)

    async def refresh_skills(self) -> RoundReport:
        """Run a round now (after any running one) and return its report."""
        return await self._learner.refresh()

    # ── internals ──

    def _backend(self, version: int) -> SkillLibraryBackend | None:
        if version == 0 or not self._learner.store.skill_dirs(version):
            return None
        if version not in self._backends:
            try:
                # Writes through the backend (an optimizer's rewrite) are staged in
                # pending/ for the next round to gate; a published version never changes.
                store = self._learner.store
                self._backends[version] = SkillLibraryBackend(
                    store.version_dir(version), actor_id=f"skillify:{self.name}", stage_root=store.pending_dir
                )
            except Exception as exc:
                raise SkillDeliveryError(f"library version {version} could not be loaded") from exc
        return self._backends[version]

    async def _record(
        self,
        ctx: ThreadContext,
        child_id: ThreadId,
        version: int,
        toolbox: SkillToolbox | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        outcome: float | None,
        error: str | None,
    ) -> None:
        try:
            events = await ctx.coordinator.get_events(child_id)
            from .observation import skill_event

            skill_event("run.events", events=events)
            skill_event("run.finished", status="failed" if error else "completed", outcome=outcome, error=error)
            self._learner.record(
                outcome=outcome,
                version=version,
                skills_available=list(toolbox.available) if toolbox else [],
                skills_loaded=list(toolbox.loaded) if toolbox else [],
                events=list(events),
                args=tuple(args),
                kwargs=dict(kwargs),
                error=error,
            )
            if not _SYNC_CALL.get():
                self._learner.maybe_start_round()
        except Exception:  # noqa: BLE001 - recording never breaks a call
            _logger.exception("skillify: recording %s failed; the call's result is unaffected", self.name)


def skillify[**P, T](
    *,
    library: str | Path | None = None,
    replay: bool | None = None,
    every: int = 20,
    score: Score | None = None,
    pass_at: float = 1.0,
    trial_inputs: int = 8,
    trial_repeats: int = 2,
    min_gain: float = 0.02,
    max_skills: int = 20,
    budget_calls: int = 200,
    keep_versions: int = 10,
    max_concurrency: int = 4,
    environment: Environment | None = None,
    miner: Miner | None = None,
    gate: Gate | None = None,
    config: SkillifyConfig | None = None,
) -> Callable[[AIFunction[P, T]], SkillifiedFunction[P, T]]:
    """Make an AI Function learn skills from its own calls.

    Args:
        library: Directory holding skills, versions, call records and history.
        replay: Must be ``True``: confirms the function's tools may be re-run.
        every: New recorded calls between rounds.
        score: ``(arguments, result) -> float in [0, 1]``; default is
            post-condition pass/fail.
        pass_at: Outcome at or above which a call counts as a success.
        trial_inputs: Recorded calls replayed per gate trial.
        trial_repeats: Runs per arm per trial input (default gate only).
        min_gain: Completion-rate gain a skill must add to the current library
            (default gate only).
        max_skills: Cap on the installed library.
        budget_calls: Cap on gate replay runs per round (default gate only).
        keep_versions: Library versions kept on disk.
        max_concurrency: Replays in flight during a round.
        environment: ``arguments -> Episode`` (or an awaitable of one): a
            fresh stateful environment per call and per replay, whose tools
            are added to that run and whose ``outcome(result)`` scores it.
            Exclusive with ``score``.
        miner: Where candidate skills come from; ``None`` uses a
            ``PromptMiner`` on the decorated function's model.
        gate: Which candidates join the library; ``None`` uses a
            ``CompletionGate`` built from ``min_gain``/``trial_repeats``/
            ``budget_calls``/``pass_at``. Exclusive with non-default
            ``min_gain``/``trial_repeats``/``budget_calls``. A gate you
            pass gets none of these settings: give a ``CompletionGate``
            its own ``pass_at`` when it should not be ``1.0``.
        config: All of the above at once; exclusive with ``library``/``replay``.

    Raises:
        ValueError: ``replay`` is not ``True``; ``library`` missing; or
            ``config`` combined with ``library``/``replay``, with
            ``score``/``environment`` or with ``miner``/``gate``; both
            ``score`` and ``environment``; or ``gate`` with a non-default
            ``min_gain``/``trial_repeats``/``budget_calls``.
        TypeError: The decorated object is not an ``AIFunction``.
    """
    if config is None:
        if library is None:
            raise ValueError("skillify requires library=")
        config = SkillifyConfig(
            library=Path(library),
            replay=replay,
            every=every,
            score=score,
            pass_at=pass_at,
            trial_inputs=trial_inputs,
            trial_repeats=trial_repeats,
            min_gain=min_gain,
            max_skills=max_skills,
            budget_calls=budget_calls,
            keep_versions=keep_versions,
            max_concurrency=max_concurrency,
            environment=environment,
            miner=miner,
            gate=gate,
        )
    else:
        if library is not None or replay is not None:
            raise ValueError("pass either config= or library=/replay=, not both")
        if score is not None or environment is not None:
            raise ValueError("pass either config= or score=/environment=, not both")
        if miner is not None or gate is not None:
            raise ValueError("pass either config= or miner=/gate=, not both")
    resolved = config

    def _decorate(fn: AIFunction[P, T]) -> SkillifiedFunction[P, T]:
        if not isinstance(fn, AIFunction):
            raise TypeError(f"skillify decorates an @ai_function, got {type(fn).__name__}")
        if not fn.config.post_conditions and resolved.score is None and resolved.environment is None:
            warnings.warn(
                f"{fn.name}: no post_conditions, no score and no environment, so every call scores 1.0 and rounds "
                "will always decline. Add post-conditions or pass score= or environment=.",
                UserWarning,
                stacklevel=2,
            )
        return SkillifiedFunction(fn, resolved)

    return _decorate
