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

Background rounds need an event loop that outlives the call. Under
``run_sync`` the call is recorded but no round is started; the pending count
is on disk, so the next async call or ``refresh_skills()`` runs it.

Gate replays run the undecorated base function -- the one passed to
``skillify``, with the model it was configured with -- including when
``@routed`` is stacked on top: variants built by ``replace`` share the
learner, so a round measures skills on that base function, not on whichever
model a router would pick.

Invariants:
    K1 — with an empty library a call is exactly the undecorated function.
    K2 — every call is recorded; recording failures are logged, never raised.
    K3 — the wrapped function's exceptions reach the caller unchanged.
    K4 — a non-empty library that cannot be shown raises SkillDeliveryError.
    K5 — replace() shares the learner and library.
    K6 — with an environment, ``episode.close()`` always runs, whether the call returned or
    raised; a raising close is logged and never masks the call's result or exception.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ai_functions.ai_thread.ai_function import AIFunction
from ai_functions.handle import ThreadHandle
from ai_functions.protocols import ThreadContext
from ai_functions.types import InputShape
from ai_functions.types.graph import Result

from .discovery import Miner
from .evaluation import Gate
from .learner import Environment, Score, SkillLearner
from .learner import RoundReport as RoundReport
from .learner import SkillifyConfig as SkillifyConfig
from .skill_library import Skill, SkillLibraryBackend

__all__ = ["RoundReport", "SkillifiedFunction", "SkillifiedThread", "SkillifyConfig", "skillify"]

class SkillifiedThread[**P, T]:
    def __init__(self, fn: SkillifiedFunction[P, T]) -> None: ...
    @property
    def name(self) -> str: ...
    async def execute(self, ctx: ThreadContext, *args: P.args, **kwargs: P.kwargs) -> T: ...
    async def notify(self, text: str) -> None: ...
    def serialize_result(self, result: T) -> str: ...
    def deserialize_result(self, payload: str) -> T: ...
    async def fork(self) -> SkillifiedFunction[P, T]: ...
    async def teardown(self) -> None: ...

class SkillifiedFunction[**P, T]:
    config: SkillifyConfig
    def __init__(
        self, fn: AIFunction[P, T], config: SkillifyConfig, *, learner: SkillLearner | None = None
    ) -> None: ...
    def to_thread(self) -> SkillifiedThread[P, T]: ...
    @property
    def input_shape(self) -> InputShape: ...
    @property
    def name(self) -> str: ...
    def replace(self, **kwargs: Any) -> SkillifiedFunction[P, T]: ...
    async def render_prompt(self, *args: Any, **kwargs: Any) -> str: ...
    async def spawn(self) -> ThreadHandle[P, T]: ...
    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...
    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T: ...
    async def trace(self, *args: Any, **kwargs: Any) -> Result[T]: ...
    @property
    def learner(self) -> SkillLearner: ...
    @property
    def skills(self) -> list[Skill]: ...
    @property
    def skill_backend(self) -> SkillLibraryBackend | None:
        """The current version as a staged ``SkillLibraryBackend``; ``None`` while the library is empty.

        Pass it to an optimizer: its rewrites are staged in the library's ``pending/``
        directory for the next round to gate, never published directly.
        """
    def history(self) -> list[dict[str, Any]]: ...
    def rollback_skills(self, to: int | None = None) -> int: ...
    async def refresh_skills(self) -> RoundReport: ...

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
) -> Callable[[AIFunction[P, T]], SkillifiedFunction[P, T]]: ...
