"""What a round asks of a miner: candidate skills from recorded calls.

A :class:`Miner` receives the window's scored calls -- each with its event log
and whether it passed -- and a directory to write into, and returns the skills it
wrote there. Pass your own to ``@skillify(miner=...)`` to change where candidate
skills come from; every candidate it returns is still gated before it is published.

:class:`PromptMiner` is the library's small default: one model call per round.

Invariants:
    M1 — a miner sees scored calls only; failed calls are the ones with ``passed=False``.
    M2 — ``PromptMiner`` drops, and logs, a draft with no name, no description, no body, a
         refusal-shaped body, a name duplicating one already accepted in the same round, or
         the name reserved for the skill index; a dropped draft writes nothing to disk.
    M3 — when the model's structured output fails altogether (``AIFunctionError``),
         ``PromptMiner`` logs it and returns no drafts; the round goes on.
"""

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from .learner import RecordedCall
from .skill_library import Skill

if TYPE_CHECKING:
    from strands.models import Model

__all__ = ["Draft", "MineFn", "Miner", "PromptMiner"]

class Miner(Protocol):
    async def __call__(self, calls: Sequence[RecordedCall], out_dir: Path) -> list[Skill]: ...

class Draft(BaseModel):
    name: str
    description: str
    body: str

MineFn = Callable[[str, int], Awaitable[list[Draft]]]

class PromptMiner:
    model: str | Model | None
    max_drafts: int
    def __init__(
        self, model: str | Model | None = None, max_drafts: int = 5, mine_fn: MineFn | None = None
    ) -> None: ...
    @staticmethod
    def render(calls: Sequence[RecordedCall]) -> str: ...
    async def __call__(self, calls: Sequence[RecordedCall], out_dir: Path) -> list[Skill]: ...
