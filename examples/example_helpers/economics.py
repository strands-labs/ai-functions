"""Shared utilities for the economics examples: priced models and 3-SAT tasks.

**Priced models.** The library deliberately ships no price table — what a
token costs depends on your contract, region, and provisioning. The prices
below are ILLUSTRATIVE EXAMPLES ONLY and do not reflect any real price list:
define your own ``PricedModel`` collection with the rates you actually pay.
The models are built with an explicit ``max_tokens`` cap, bounding each
attempt's cost and latency — and capping what a failure costs: when the
cheap model misses a hard instance (a wrong answer, or output truncated at
the cap), the failure is cheap, and the search can escalate to the stronger
model.

**Planted 3-SAT.** Instances are generated with a hidden planted assignment,
so every instance is guaranteed satisfiable. Verification is pure Python —
the post-condition checks every clause. Difficulty is governed by the
clause/variable ratio: low ratios leave many satisfying assignments; near
~4.3 — the random 3-SAT satisfiability phase transition — solutions become
scarce and instances get hard, even though planting keeps them satisfiable.
"""

from __future__ import annotations

import random

from strands.models import BedrockModel

from ai_functions.experimental.economics import PricedModel, Prices

# ── Priced models ─────────────────────────────────────────────────

# Cap generation so a rambling cheap model fails fast (and cheap) rather than
# burning tens of thousands of tokens on one hard instance.
MAX_TOKENS = 8_000

haiku = PricedModel(
    model=BedrockModel(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        max_tokens=MAX_TOKENS,
    ),
    prices=Prices(input=1.00, output=5.00, cache_read=0.10, cache_write=1.25),
    label="haiku",
    description="Fast and cheap; handles straightforward tasks well",
)

sonnet = PricedModel(
    model=BedrockModel(
        model_id="global.anthropic.claude-sonnet-4-6",
        max_tokens=MAX_TOKENS,
    ),
    prices=Prices(input=3.00, output=15.00, cache_read=0.30, cache_write=3.75),
    label="sonnet",
    description="Balanced; strong at multi-step reasoning and constraints",
)

# ── Planted 3-SAT instances ───────────────────────────────────────

# A clause is a list of (variable_index, is_positive) literals.
Clause = list[tuple[int, bool]]


def make_instance(n_vars: int, seed: int, ratio: float) -> list[Clause]:
    """Generate a satisfiable random 3-SAT instance with ``n_vars`` variables.

    A hidden assignment is drawn first; every generated clause is patched to
    contain at least one literal satisfied by it, guaranteeing satisfiability
    without making the instance trivial.
    """
    rng = random.Random(seed)
    hidden = [rng.random() < 0.5 for _ in range(n_vars)]
    clauses: list[Clause] = []
    for _ in range(round(n_vars * ratio)):
        idxs = rng.sample(range(n_vars), 3)
        clause: Clause = [(i, rng.random() < 0.5) for i in idxs]
        if not any(hidden[i] == positive for i, positive in clause):
            j = rng.randrange(3)
            i, positive = clause[j]
            clause[j] = (i, not positive)
        clauses.append(clause)
    return clauses


def format_clause(clause: Clause) -> str:
    """Render one clause as ``(x1 OR NOT x3 OR x5)``."""
    lits = [f"x{i + 1}" if positive else f"NOT x{i + 1}" for i, positive in clause]
    return "(" + " OR ".join(lits) + ")"


def format_instance(clauses: list[Clause]) -> str:
    """Render the whole instance, one clause per line."""
    return "\n".join(format_clause(c) for c in clauses)


def violated_clauses(clauses: list[Clause], values: list[bool]) -> list[Clause]:
    """Return the clauses that ``values`` does not satisfy."""
    return [c for c in clauses if not any(values[i] == positive for i, positive in c)]


def parse_instance(text: str) -> list[Clause]:
    """Parse :func:`format_instance` output back into structured clauses.

    Lets a post-condition receive the formula as the same string the prompt
    interpolates — so one module-level checker serves every instance, keyed
    off the call's own arguments rather than a per-instance closure.
    """
    clauses: list[Clause] = []
    for line in text.strip().splitlines():
        clause: Clause = []
        for lit in line.strip("() ").split(" OR "):
            positive = not lit.startswith("NOT ")
            clause.append((int(lit.removeprefix("NOT ").strip().lstrip("x")) - 1, positive))
        clauses.append(clause)
    return clauses


def check_sat(result, clauses: str, n_vars: int, **kwargs):
    """Module-level post-condition: the assignment satisfies every clause.

    Receives ``clauses`` and ``n_vars`` from the call's bound arguments (the
    ``PostCondition`` contract forwards matching argument names), so it needs
    no per-instance construction.
    """
    from ai_functions.ai_thread import PostConditionResult

    if len(result.values) != n_vars:
        return PostConditionResult(passed=False, message=f"Need {n_vars} values, got {len(result.values)}")
    bad = violated_clauses(parse_instance(clauses), result.values)
    if bad:
        return PostConditionResult(passed=False, message=f"{len(bad)} clauses violated")
    return None


def make_sat_checker(n_vars: int, clauses: list[Clause]):
    """Build a post-condition that accepts an ``Assignment`` iff it satisfies every clause.

    Returned as a factory (not an inline closure) so it binds ``n_vars`` and
    ``clauses`` explicitly — safe to build inside a loop over instances.
    """
    from ai_functions.ai_thread import PostConditionResult

    def check_assignment(result, **kwargs) -> PostConditionResult | None:
        if len(result.values) != n_vars:
            return PostConditionResult(passed=False, message=f"Need {n_vars} values, got {len(result.values)}")
        bad = violated_clauses(clauses, result.values)
        if bad:
            return PostConditionResult(passed=False, message=f"{len(bad)} clauses violated")
        return None

    return check_assignment
