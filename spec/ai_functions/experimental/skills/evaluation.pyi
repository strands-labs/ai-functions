"""What a round asks of a gate: which candidate skills earn a place in the library.

A :class:`Gate` receives the candidates, the current library's skill directories,
the recorded calls to try them on and a :data:`Replay` that re-runs those calls
with a given set of skills; it returns one :class:`Verdict` per candidate.
Pass your own to ``@skillify(gate=...)`` to change how skills are judged; the
learner only publishes what the gate admits.

Invariants:
    E1 — a gate returns exactly one verdict per candidate, in candidate order.
    E2 — ``replay(skill_dirs, calls, repeats)`` returns one outcome per (call, repeat), in that order.
    E3 — ``CompletionGate`` admits a candidate only when its with-arm completion rate exceeds the
         without-arm's by at least ``min_gain``; an unmeasured candidate (no trial calls, or no
         scored run in an arm) is rejected.
    E4 — ``CompletionGate`` measures the baseline arm once per gate call and shares it with every
         candidate, and over ``budget_calls`` cuts
         candidates from the end of the list before trial inputs.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .learner import RecordedCall
from .skill_library import Skill

__all__ = ["Candidate", "CompletionGate", "Gate", "Replay", "Verdict"]

@dataclass(frozen=True)
class Candidate:
    skill: Skill
    replaces: str | None = None

@dataclass(frozen=True)
class Verdict:
    candidate: Candidate
    admitted: bool
    gain: float | None
    reason: str
    scores: dict[str, float | None]

Replay = Callable[[Sequence[Path], Sequence[RecordedCall], int], Awaitable[list[float | None]]]

class Gate(Protocol):
    async def __call__(
        self,
        candidates: Sequence[Candidate],
        current: Sequence[Path],
        trials: Sequence[RecordedCall],
        replay: Replay,
    ) -> list[Verdict]: ...

class CompletionGate:
    min_gain: float
    repeats: int
    budget_calls: int
    pass_at: float
    def __init__(
        self, min_gain: float = 0.02, repeats: int = 2, budget_calls: int = 200, pass_at: float = 1.0
    ) -> None: ...
    async def __call__(
        self,
        candidates: Sequence[Candidate],
        current: Sequence[Path],
        trials: Sequence[RecordedCall],
        replay: Replay,
    ) -> list[Verdict]: ...
