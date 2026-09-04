"""Keep the best report: sample either model while a better attempt is worth its cost.

Two models review a buggy C module. Each attempt is independent and graded by
F1 against the planted defects — a score in [0, 1] the decorator prices at
``VALUE`` (reward = VALUE * F1) — and the search keeps whichever report scores
highest. It keeps sampling while some model's *index* (reservation price)
exceeds the best reward in hand, i.e. while the expected gain from one more
attempt still covers its cost.

This is the graded form of ``@routed``: the score is a continuous
[0, 1] grader rather than the default binary pass/fail, and the default
``ReservationPricePolicy`` (Weitzman) drives sampling ordering and stopping.
The per-model indices come from a calibration phase: try each model a few times
outside the search, grade each report, and keep the empirical reward
distribution and average cost as that model's Pandora "box".
"""

import asyncio
import logging
from collections import Counter

from _buggy_c import BUGGY_C, PLANTED
from _economics_utils import HAIKU, SONNET
from _utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult
from ai_functions.experimental.economics import (
    AttemptRecord,
    Beliefs,
    Candidate,
    TaskView,
    attempts,
    decisions,
    routed,
)
from ai_functions.experimental.economics.search import Categorical, Estimate
from ai_functions.runtime.usage import subtree_token_usage

VALUE = 0.10  # a perfect report (all planted defects, nothing else) is worth 10 cents
NUM_PLANTED = len(PLANTED)
CALIBRATION_TRIALS = 5  # number of calibration trials per model before the search
MAX_TRIES = 2  # maximum attempts allowed per model


class Defect(BaseModel):
    function: str = Field(description="Name of the function containing the defect")
    kind: str = Field(description="e.g. data race, use-after-free, integer overflow, OOB access")
    description: str = Field(description="What the defect is and how it manifests")


class Report(BaseModel):
    defects: list[Defect]


def at_least_one(result: Report) -> PostConditionResult | None:
    if not result.defects:
        return PostConditionResult(passed=False, message="Report at least one defect")
    return None


def f1_score(report: Report) -> float:
    """Score of one report in [0, 1]: its F1 against the planted defects.

    F1 is the harmonic mean of precision (fraction of the report's claims
    that are real) and recall (fraction of the planted defects it found),
    so every false positive and every miss lowers the score — only a report
    naming exactly the planted defects scores 1.0. The decorator prices this
    at ``VALUE``: reward = VALUE * f1_score.
    """
    hits = len({d.function for d in report.defects} & set(PLANTED))
    if hits == 0:
        return 0.0
    precision = hits / len(report.defects)
    recall = hits / NUM_PLANTED
    return 2 * precision * recall / (precision + recall)


def worth(report: Report) -> float:
    """Dollar worth of one report: ``VALUE * f1_score`` — the reward the search books."""
    return VALUE * f1_score(report)


class CalibratedRewards(Beliefs):
    """A separate reward distribution and attempt cost per model, from its own calibration.

    One shared estimator holding, for each model, a histogram of the rewards
    its calibration attempts earned and the mean dollar cost of one attempt
    — together, that model's "box" in Pandora terms: what an attempt might
    pay, and what opening the box costs. Starts empty; the calibration
    phase fills it via ``observe``. The boxes stay fixed for the whole
    search — ``estimate`` ignores the search's history.

    This is a graded counterpart of ``EmpiricalBeliefs``: per-model observed
    statistics, but keeping the full histogram of rewards rather than a
    pass rate.
    """

    def __init__(self) -> None:
        self._table: dict[str, tuple[Categorical, float]] = {}

    def observe(self, label: str, rewards: list[float], mean_cost: float) -> None:
        """Record one model's calibration: its graded rewards and mean cost.

        The box is the frequency histogram of the rewards — each distinct
        reward becomes one entry whose probability is how often it was
        observed: rewards {0.04, 0.06, 0.04, 0.06, 0.06} become
        {$0.04: 40%, $0.06: 60%}.
        """
        counts = Counter(rewards)
        values = tuple(sorted(counts))
        probs = tuple(counts[v] / len(rewards) for v in values)
        self._table[label] = (Categorical(values=values, probs=probs), mean_cost)

    def box(self, label: str) -> str:
        """Display one model's box: each reward with its probability, and the cost per attempt."""
        dist, cost = self._table[label]
        entries = ", ".join(f"${v:.4f}: {p:.0%}" for v, p in zip(dist.values, dist.probs, strict=True))
        return f"{{{entries}}} at ${cost:.4f}/attempt"

    async def estimate(
        self, task: TaskView, candidates: list[Candidate], value: float | None, history: list[AttemptRecord]
    ) -> dict[str, Estimate]:
        del task, value, history
        missing = [c.label for c in candidates if c.label not in self._table]
        if missing:
            raise RuntimeError(f"models not calibrated yet: {missing}")
        return {c.label: Estimate(dist=self._table[c.label][0], cost=self._table[c.label][1]) for c in candidates}


BOXES = CalibratedRewards()  # empty until phase 1 runs


# Graded best-of-N with adaptive ordering and stopping: each attempt is scored
# by F1 (in [0, 1]) and priced at VALUE, and the search keeps the best result
# while some model's index (reservation price) beats the best reward in hand —
# Weitzman's Pandora rule, the default policy. Fixed calibrated boxes, so no
# re-estimation is needed.
@routed(
    models=[HAIKU, SONNET],
    value=VALUE,  # dollars a perfect (F1 = 1.0) report is worth
    scorer=f1_score,  # score in [0, 1]; reward = VALUE * f1_score
    budget=0.10,  # backstop; the indices are what should stop the search
    beliefs=BOXES,
    max_tries=MAX_TRIES,
)
@ai_function[Report](post_conditions=[at_least_one])
def review(source: str):
    """You are reviewing this C module. Report any real bugs you find: for each,
    name the function and explain what goes wrong. Report only bugs you are
    confident are real, not style issues or hypothetical concerns.

    ```c
    {source}
    ```"""


async def calibrate(candidate: Candidate) -> tuple[list[float], float]:
    """Try one model directly a few times; return (graded rewards, mean cost).

    ``review.candidates`` carries each model's plain function (model already
    swapped in) and its prices, so calibration needs nothing the decorator
    doesn't already have.
    """
    rewards: list[float] = []
    costs: list[float] = []
    for _ in range(CALIBRATION_TRIALS):
        run = await candidate.fn.trace(source=BUGGY_C)
        rewards.append(worth(run.value))
        usage = await subtree_token_usage(run.coordinator, run.thread_id)
        costs.append(candidate.prices.cost_of(usage))
    return rewards, sum(costs) / len(costs)


async def main():
    logging.basicConfig(level=logging.WARNING)

    rule("Phase 1 — calibrate: try each model, grade each report, measure each cost")

    lines = []
    for label, candidate in review.candidates.items():
        rewards, mean_cost = await calibrate(candidate)
        BOXES.observe(label, rewards, mean_cost)
        graded = "  ".join(f"${w:.4f}" for w in rewards)
        lines.append(f"{label:<8} rewards: {graded}   mean cost ${mean_cost:.4f}")
    display("Calibration", "\n".join(lines), lang="text")

    rule("Phase 2 — search: keep-best reward, Weitzman indices based on calibration data")

    run = await review.trace(source=BUGGY_C)
    records = await attempts(run)
    (ranking,) = await decisions(run)

    lines = ["indices (g per model, solving E[(R - g)+] = cost for each model):"]
    for r in ranking:
        lines.append(f"  {r.label:<8} {BOXES.box(r.label)}  ->  g=${r.reservation_price:.4f}")
    lines.append("")
    best_reward = 0.0
    for i, r in enumerate(records, 1):
        # Booked rewards are improvements over the incumbent, so the new
        # best is the running sum of the improvements so far.
        new_best = best_reward + r.reward
        lines.append(
            f"attempt {i}  {r.candidate:<8} current_best=${best_reward:.4f}  +${r.reward:.4f} improvement"
            f"  ->  new_best=${new_best:.4f}   (cost ${r.cost:.4f})"
        )
        best_reward = new_best
    lines.append("")
    # Why it stopped: with fixed boxes, each model is out either because its
    # tries are spent or because its index no longer beats the best in hand.
    tries = Counter(r.candidate for r in records)
    reasons = ", ".join(
        f"{r.label} {'at its cap' if tries[r.label] >= MAX_TRIES else f'g=${r.reservation_price:.4f} <= best'}"
        for r in ranking
    )
    lines.append(f"stopped: {reasons}")
    kept: Report = run.value
    hits = {d.function for d in kept.defects} & set(PLANTED)
    false_positives = [d for d in kept.defects if d.function not in PLANTED]
    lines.append(
        f"best ${best_reward:.4f} · kept report worth ${worth(kept):.4f} "
        f"({len(hits)}/{NUM_PLANTED} planted found, {len(false_positives)} false positives) "
        f"· spent ${sum(r.cost for r in records):.4f}"
    )
    display("Search decisions", "\n".join(lines), lang="text")


if __name__ == "__main__":
    asyncio.run(main())
