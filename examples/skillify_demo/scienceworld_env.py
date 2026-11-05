"""ScienceWorld as tools an AI Function can call, and the tasks it is called on.

The agent under evaluation here is an ``@ai_function``, not an external
agent runtime, so the environment needs no bridge: a session object holds a
``ScienceWorldEnv`` and two ``@tool``-decorated closures hand it to the
model directly. (The Claude Code path needs an HTTP server and a shell
script per run, because an external agent acts through bash.)

``load_variations`` builds the tasks the demo runs from a fixed list of
variation ids. A ScienceWorld variation is deterministic given its task
name and number, so a list of ids reproduces exactly the same tasks for
every reader, with no ranking pass and nothing cached. Which ids, and why
those, is recorded in ``evaluation.py``.

ScienceWorld's own ``train``/``dev``/``test`` variation splits do the
train/test separation: the demo's calls are made on ``train`` variations and
the library is measured on ``test`` ones, so no skill is ever mined from a
task it is later scored on.

``make_episode_factory`` turns a variation into an ``@skillify``
``environment=``: every call, and every replay of a recorded call, gets a
fresh session rebuilt from the call's arguments, and a replay refuses to run
against an environment that differs from the one the call saw.

``scienceworld`` is imported where an environment is built rather than at
module import, so the selection and episode code can be imported (and
tested) without it.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands import tool

if TYPE_CHECKING:
    from scienceworld import ScienceWorldEnv

_logger = logging.getLogger(__name__)

STEP_LIMIT = 150
"""Environment step cap. ScienceWorld ends the episode when it is hit."""


@dataclass
class Variation:
    """One ScienceWorld task variation.

    ``gold_length`` -- the length of the environment's own gold action
    sequence -- is the difficulty proxy the one-off pilot ranked candidates
    by. The demo runs a fixed set of variations and never ranks, so it is
    optional and normally unset.
    """

    task: str
    variation: int
    description: str
    gold_length: int | None = None

    @property
    def task_id(self) -> str:
        return f"{self.task}-{self.variation}"


def load_variations(task: str, variations: Sequence[int]) -> list[Variation]:
    """The named variations of ``task``, in the order given.

    One environment and one ``load`` per id, no gold paths: a variation's
    task description is deterministic given ``(task, id)``, which is what
    lets a fixed list of ids stand in for a selection pass. Takes a few
    seconds and one JVM; nothing is cached because nothing is expensive.
    """
    from scienceworld import ScienceWorldEnv

    env = ScienceWorldEnv("", envStepLimit=STEP_LIMIT)
    try:
        out = []
        for number in variations:
            env.load(task, number, "")
            out.append(Variation(task=task, variation=number, description=env.get_task_description()))
    finally:
        _shutdown(env)
    _logger.info("loaded %d variations of %s", len(out), task)
    return out


def _shutdown(env: ScienceWorldEnv) -> None:
    """Shut one environment's JVM down, exactly once."""
    try:
        env.close()
    except Exception:  # noqa: BLE001 - teardown is best-effort
        _logger.debug("scienceworld: environment close failed", exc_info=True)
    # ``ScienceWorldEnv.__del__`` calls ``close()`` again, which would log a
    # traceback from the already-closed gateway.
    env.close = lambda: None  # type: ignore[method-assign]


@dataclass
class ScienceWorldSession:
    """One episode's environment, exposed to a model as two tools.

    Each run gets its own session, so parallel runs never share environment
    state. ``score`` is read after the fact by the oracle -- the model is
    never told its score mid-episode, which would turn the task into
    score-hacking rather than problem-solving.
    """

    variation: Variation
    step_limit: int = STEP_LIMIT
    _env: ScienceWorldEnv | None = field(default=None, init=False, repr=False)
    score: float = field(default=0.0, init=False)
    done: bool = field(default=False, init=False)
    steps: int = field(default=0, init=False)

    def start(self) -> str:
        """Load the variation and return the opening observation. Blocking: launches a JVM."""
        from scienceworld import ScienceWorldEnv

        self._env = ScienceWorldEnv("", envStepLimit=self.step_limit)
        self._env.load(self.variation.task, self.variation.variation, "")
        observation, _info = self._env.reset()
        return observation

    @property
    def task_description(self) -> str:
        return self.variation.description

    def step(self, action: str) -> str:
        """Take one action; return the observation, with progress bookkeeping."""
        if self._env is None:
            raise RuntimeError("session not started")
        if self.done:
            return "The episode has already finished. No further actions are possible."
        observation, _reward, done, info = self._env.step(action)
        self.score = float(info.get("score", 0.0))
        self.done = bool(done)
        self.steps += 1
        if self.steps >= self.step_limit:
            self.done = True
        return f"{observation}\n\n(step {self.steps}/{self.step_limit})"

    def close(self) -> None:
        """Release the environment's JVM. Returns at once; idempotent.

        Each ``ScienceWorldEnv`` runs its own JVM behind a py4j gateway. The
        shutdown talks to that JVM over a socket, so it runs on a daemon
        thread: ``close`` is called on the event loop (``Episode.close`` is
        synchronous), and the loop must not wait on it.
        """
        env, self._env = self._env, None
        if env is not None:
            threading.Thread(target=_shutdown, args=(env,), name="scienceworld-close", daemon=True).start()

    def make_tools(self) -> list[object]:
        """One tool bound to this session.

        Deliberately one and not two: a zero-argument "look around" tool made
        the model emit an empty tool-input payload, which strands cannot parse
        (it warns and substitutes ``{}``) -- and ``sw_act("look around")``
        already does the same thing. Every affordance the environment has is
        reachable through a single action string.
        """

        @tool(
            name="sw_act",
            description=(
                "Take one action in the ScienceWorld environment and return the "
                "resulting observation. Actions are short imperative phrases, e.g. "
                "'open door to kitchen', 'pick up thermometer', 'move metal pot to stove', "
                "'activate stove', 'focus on steam'. Use 'look around' to re-read the "
                "current room. Returns the observation text."
            ),
        )
        def sw_act(action: str) -> str:
            return self.step(action)

        return [sw_act]


class ReplayDriftError(RuntimeError):
    """A rebuilt environment does not match the one the recorded call saw.

    A replay must never quietly run against a different environment than the
    call it replays: its outcome would then measure a different task.
    """


@dataclass
class ScienceWorldEpisode:
    """One run's environment, in the shape ``@skillify``'s ``Episode`` protocol asks for.

    Attributes:
        session: The environment this run acts in.
        tools: ``sw_act`` bound to ``session``.
    """

    session: ScienceWorldSession
    tools: list[object]

    @property
    def raw_score(self) -> float:
        """ScienceWorld's own score, -100 to 100."""
        return self.session.score

    @property
    def steps(self) -> int:
        """Environment steps taken."""
        return self.session.steps

    def outcome(self, result: Any) -> float:  # noqa: ANN401 - the model's answer is not what is scored
        """Completion: 1.0 when the task was finished (score 100), else 0.0.

        ``result`` is ignored: the environment, not the model's summary, says
        whether the task was done. Completion rather than partial credit,
        because this family pays 79 of 100 for the mechanics alone.
        """
        del result
        return 1.0 if self.session.score >= 100.0 else 0.0

    def close(self) -> None:
        """Release the environment (off the event loop)."""
        self.session.close()


def make_episode_factory(
    variations: Mapping[str, Variation],
    step_limit: int,
    *,
    on_episode: Callable[[ScienceWorldEpisode], None] | None = None,
) -> Callable[[Mapping[str, Any]], Any]:
    """An ``environment=`` for ``@skillify``: a fresh ScienceWorld episode per run.

    Args:
        variations: ``variation_id`` argument -> the variation it names.
        step_limit: Environment steps per episode.
        on_episode: Called with every episode built, for reporting.

    Returns:
        ``async environment(arguments) -> ScienceWorldEpisode``. It reads the
        call's ``variation_id`` and ``opening_observation`` arguments and
        starts the session on a worker thread (starting launches a JVM).

    Raises (from the returned factory):
        ReplayDriftError: The fresh opening observation differs from the
            call's ``opening_observation``.
    """

    async def environment(arguments: Mapping[str, Any]) -> ScienceWorldEpisode:
        variation = variations[arguments["variation_id"]]
        session = ScienceWorldSession(variation=variation, step_limit=step_limit)
        try:
            opening = await asyncio.to_thread(session.start)
        except BaseException:
            session.close()
            raise
        if opening != arguments["opening_observation"]:
            session.close()
            raise ReplayDriftError(
                f"{variation.task_id}: the rebuilt environment's opening observation differs from the "
                "recorded call's; refusing to run against a different environment"
            )
        episode = ScienceWorldEpisode(session=session, tools=session.make_tools())
        if on_episode is not None:
            on_episode(episode)
        return episode

    return environment


async def opening_observation(variation: Variation, step_limit: int) -> str:
    """Start a session only to read its opening observation, then release it."""
    session = ScienceWorldSession(variation=variation, step_limit=step_limit)
    try:
        return await asyncio.to_thread(session.start)
    finally:
        session.close()


def live_jvms() -> int:
    """Live ``java`` processes started by this process -- one per open environment."""
    count = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            stat = (proc / "stat").read_text()
            comm = stat[stat.index("(") + 1 : stat.rindex(")")]
            ppid = int(stat[stat.rindex(")") + 2 :].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        if ppid == os.getpid() and comm == "java":
            count += 1
    return count
