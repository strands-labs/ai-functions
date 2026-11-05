"""What a round asks of a gate: which candidate skills earn a place in the library.

A :class:`Gate` receives the candidates, the current library's skill directories,
the recorded calls to try them on and a :data:`Replay` that re-runs those calls
with a given set of skills; it returns one :class:`Verdict` per candidate.
Pass your own to ``@skillify(gate=...)`` to change how skills are judged; the
learner only publishes what the gate admits.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from .observation import skill_event
from .skill_library import Skill

if TYPE_CHECKING:
    from .learner import RecordedCall

__all__ = ["Candidate", "CompletionGate", "Gate", "Replay", "Verdict"]


@dataclass(frozen=True)
class Candidate:
    """A skill proposed for the library.

    Attributes:
        skill: The proposed skill, on disk.
        replaces: Name of the installed skill this revises; ``None`` for a new skill.
    """

    skill: Skill
    replaces: str | None = None


@dataclass(frozen=True)
class Verdict:
    """A gate's decision on one candidate.

    Attributes:
        candidate: The candidate decided.
        admitted: Whether it joins the library.
        gain: How much it improved on the current library, when measured.
        reason: Human-readable explanation.
        scores: The numbers behind the decision.
    """

    candidate: Candidate
    admitted: bool
    gain: float | None
    reason: str
    scores: dict[str, float | None]


Replay = Callable[[Sequence[Path], Sequence["RecordedCall"], int], Awaitable[list[float | None]]]
"""``replay(skill_dirs, calls, repeats)``: one outcome per (call, repeat), in that order; ``None`` when unscored."""


class Gate(Protocol):
    """Decides which candidates join the library."""

    async def __call__(
        self,
        candidates: Sequence[Candidate],
        current: Sequence[Path],
        trials: Sequence[RecordedCall],
        replay: Replay,
    ) -> list[Verdict]:
        """One verdict per candidate, in order."""
        ...


class CompletionGate:
    """Admit a candidate only if replaying recorded calls with it completes more of them.

    The *without* arm is the current library; the *with* arm is the current library
    plus the candidate -- or, for a revision, with the skill it ``replaces`` swapped
    for it. A run counts as completed when its outcome is at least ``pass_at``; an
    unscored run is left out. The baseline arm is measured once per gate call and
    shared by every candidate. Over ``budget_calls`` replays, candidates are cut from
    the end of the list, then trial inputs: order the list to choose what survives.

    Besides ``with_skill``/``without_skill``/``gain``, each verdict's scores tell the
    round report its ``stage`` (``"budget"`` or ``"trial"``), its ``order`` of
    decision and the ``extra_calls`` it cost; the first verdict also lists the
    ``budget_cuts``.
    """

    def __init__(self, min_gain: float = 0.02, repeats: int = 2, budget_calls: int = 200, pass_at: float = 1.0) -> None:
        self.min_gain, self.repeats, self.budget_calls, self.pass_at = min_gain, repeats, budget_calls, pass_at

    def _rate(self, outcomes: Sequence[float | None]) -> float | None:
        scored = [o for o in outcomes if o is not None]
        return sum(o >= self.pass_at for o in scored) / len(scored) if scored else None

    def _cost(self, n_trials: int, n_candidates: int) -> int:
        return n_trials * self.repeats * (1 + n_candidates)

    async def __call__(
        self,
        candidates: Sequence[Candidate],
        current: Sequence[Path],
        trials: Sequence[RecordedCall],
        replay: Replay,
    ) -> list[Verdict]:
        """One verdict per candidate, in order."""
        trials, kept = list(trials), list(candidates)
        cut: list[Candidate] = []
        cuts: list[str] = []
        while len(kept) > 1 and self._cost(len(trials), len(kept)) > self.budget_calls:
            cut.append(kept.pop())
            cuts.append(f"candidate {cut[-1].skill.name}")
        while len(trials) > 1 and self._cost(len(trials), len(kept)) > self.budget_calls:
            cuts.append(f"trial input {trials.pop().call_id}")
        if kept and self._cost(len(trials), len(kept)) > self.budget_calls:
            cut += kept
            cuts += [f"candidate {c.skill.name}" for c in kept]
            kept = []
        decided: list[Verdict] = [Verdict(c, False, None, "over budget_calls", {"stage": "budget"}) for c in cut]
        if kept and not trials:
            unmeasured: dict[str, Any] = {"stage": "trial", "with_skill": None, "without_skill": None, "gain": None}
            decided += [Verdict(c, False, None, "no trial calls to measure against", dict(unmeasured)) for c in kept]
            kept = []
        if kept:
            current = list(current)
            base = self._rate(await replay(current, trials, self.repeats))
            for n, c in enumerate(kept):
                arm = [p for p in current if p.name != c.replaces] if c.replaces else list(current)
                with_ = self._rate(await replay([*arm, c.skill.path], trials, self.repeats))
                gain = None if with_ is None or base is None else with_ - base
                admitted = gain is not None and gain >= self.min_gain
                runs = len(trials) * self.repeats * (2 if n == 0 else 1)  # the first also carries the baseline arm
                scores: dict[str, Any] = {
                    "stage": "trial",
                    "with_skill": with_,
                    "without_skill": base,
                    "gain": gain,
                    "extra_calls": float(runs),
                }
                reason = f"completion {with_!r} vs {base!r} over {len(trials)} call(s) x {self.repeats}"
                decided.append(Verdict(c, admitted, gain, reason, scores))
                skill_event(
                    "candidate.evaluated", skill=c.skill, admitted=admitted, gain=gain, reason=reason, scores=scores
                )
        by_candidate = {id(v.candidate): (n, v) for n, v in enumerate(decided)}
        verdicts: list[Verdict] = []
        for i, c in enumerate(candidates):
            n, v = by_candidate[id(c)]
            report: dict[str, Any] = {"order": float(n), **({"budget_cuts": cuts} if i == 0 else {})}
            verdicts.append(Verdict(c, v.admitted, v.gain, v.reason, {**v.scores, **report}))
        return verdicts
