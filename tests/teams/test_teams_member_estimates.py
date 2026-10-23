"""Every shipped member declares what its next ask costs: its declaration, then its measured asks (members EA1, EA2).

The claims are read off the budget a real ``Team.run`` reserves against and off the member's own ``ask_costs``,
never inferred from ``TeamRun`` alone. Members run on scripted models whose turns spend tokens; a model that names
an id the price table knows has its dollars measured too.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from strands.models import Model

from ai_functions.experimental.teams import FunctionMember, MethodMember, Team, WiringError
from ai_functions.experimental.teams.members import ASK_HISTORY
from ai_functions.runtime.budget import Budget, Cost
from ai_functions.runtime.prices import PRICES
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import TokenUsage

from .teams_methods_wire import Analyst
from .teams_wire import Counting, call_member, read, ruling, scripted_lead


def _reading(input_tokens: int, output_tokens: int) -> Turn:
    return Turn(
        tool_calls=(("Reading", {"source": "left", "detail": "seen"}),),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class _Watching(Model):
    """A member model that records what the budget held open when each of its calls began."""

    def __init__(self, inner: Counting, budget: Budget) -> None:
        super().__init__()
        self._inner = inner
        self._budget = budget
        self.reserved: list[Cost] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> dict[str, object]:
        return {}

    def structured_output(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream(self, messages: Any, tool_specs: Any = None, *args: Any, **kwargs: Any) -> Any:
        self.reserved.append(self._budget.reserved)
        return self._inner.stream(messages, tool_specs, *args, **kwargs)


PRICED = "claude-opus-5-5"


class _Named(ScriptedModel):
    """A scripted model that reports a ``model_id``, so the usage its turns spend is priced."""

    def __init__(self, turns: list[Turn], model_id: str | None = PRICED) -> None:
        super().__init__(turns)
        self.model_id = model_id

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "model_id": self.model_id} if self.model_id else super().get_config()


def _spent(input_tokens: int, output_tokens: int, **kwargs: Any) -> Turn:
    return Turn(
        tool_calls=(("Reading", {"source": "left", "detail": "seen"}),),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        **kwargs,
    )


# ── EA1: the declaration, and the refusal of anything that is not a Cost ──


def _every_shape(**kwargs: Any) -> list[Any]:
    return [FunctionMember(read, name="fn", **kwargs), MethodMember(Analyst("left"), "read", **kwargs)]


def test_every_shipped_member_answers_its_declaration_before_it_has_measured_anything() -> None:
    """EA1: ``ask_estimate`` is the answer until an ask is measured; ``None`` by default."""
    declared = Cost(tokens=400, dollars=0.25)
    assert [member.estimate_ask() for member in _every_shape(ask_estimate=declared)] == [declared] * 2
    assert [member.estimate_ask() for member in _every_shape()] == [None] * 2


def test_an_estimate_that_is_not_a_cost_is_refused_at_construction() -> None:
    """EA1: a wiring fault, named, before anything spawns."""
    with pytest.raises(WiringError, match="ask_estimate"):
        FunctionMember(read, name="fn", ask_estimate=400)  # type: ignore[arg-type]
    with pytest.raises(WiringError, match="ask_estimate"):
        MethodMember(Analyst("left"), "read", ask_estimate={"tokens": 400})  # type: ignore[arg-type]


async def test_a_declared_estimate_lets_a_refusing_budget_admit_the_member() -> None:
    """C10 with EA1: under ``on_unknown="refuse"`` a member that declares an estimate is admitted and reserved at it."""
    budget = Budget(Cost(tokens=1000), name="strict")
    watching = _Watching(Counting([_reading(30, 20)]), budget)
    member = FunctionMember(read, name="left", model=watching, ask_estimate=Cost(tokens=300))
    lead, _ = scripted_lead([call_member("left"), ruling()])
    async with RuntimeHarness() as h:
        run = await Team(lead, [member]).run("go", h.coordinator, budget=budget)
    assert run.answer.admitted is True
    assert watching.reserved == [Cost(tokens=300)], "the declaration was the hold while the ask ran"


# ── EA1/EA2: measured history ──


async def test_a_function_member_estimates_from_its_costliest_measured_ask() -> None:
    """EA2: each ask's own tokens are recorded; EA1: the estimate is the costliest of them."""
    model = Counting([_reading(30, 20), _reading(80, 20), _reading(10, 5)])
    member = FunctionMember(read, name="left", model=model, ask_estimate=Cost(tokens=999, dollars=0.5))
    async with RuntimeHarness() as h:
        await member.spawn(h.coordinator)
        try:
            for focus in ("a", "b", "c"):
                await member.ask(focus)
        finally:
            await member.retire()
    assert member.ask_costs == (Cost(tokens=50), Cost(tokens=100), Cost(tokens=15))
    assert member.estimate_ask() == Cost(tokens=100, dollars=0.5), (
        "the costliest measured tokens; the dollars the scripted model's usage could not be priced at come from "
        "the declaration"
    )


async def test_a_budgeted_run_reserves_each_ask_at_the_estimate_as_it_stands_then() -> None:
    """C10 reads ``estimate_ask`` before every ask, so the second ask is held at what the first one measured."""
    budget = Budget(Cost(tokens=10_000), name="run")
    watching = _Watching(Counting([_reading(30, 20), _reading(40, 20)]), budget)
    member = FunctionMember(read, name="left", model=watching, ask_estimate=Cost(tokens=500))
    lead, _ = scripted_lead([call_member("left", "first"), call_member("left", "second"), ruling()])
    async with RuntimeHarness() as h:
        await Team(lead, [member]).run("go", h.coordinator, budget=budget)
    assert watching.reserved == [Cost(tokens=500), Cost(tokens=50)]
    assert member.ask_costs == (Cost(tokens=50), Cost(tokens=60))
    assert budget.reserved == Cost(), "nothing is left held after the run"


async def test_the_history_carries_into_the_next_run_on_the_same_member() -> None:
    """EA2: the history lives on the member object, so a second run starts from what the first one measured."""
    budget = Budget(Cost(tokens=10_000), name="run")
    watching = _Watching(Counting([_reading(30, 20), _reading(30, 20)]), budget)
    member = FunctionMember(read, name="left", model=watching)
    async with RuntimeHarness() as h:
        for _ in range(2):
            lead, _ = scripted_lead([call_member("left"), ruling()])
            await Team(lead, [member]).run(
                "go",
                h.coordinator,
                budget=budget.child(Cost(tokens=5000), on_unknown="estimate", default_estimate=Cost(tokens=700)),
            )
    assert watching.reserved == [Cost(tokens=700), Cost(tokens=50)], (
        "run one fell back to the budget's default; run two reserved at run one's measured ask"
    )


async def test_a_member_measures_its_asks_in_tokens_and_priced_dollars() -> None:
    """EA2: each ask's usage is priced by the coordinator's table, by the model it names."""
    member = FunctionMember(read, name="left", model=_Named([_spent(1000, 100), _spent(3000, 100)]))
    async with RuntimeHarness() as h:
        await member.spawn(h.coordinator)
        try:
            await member.ask("one")
            await member.ask("two")
        finally:
            await member.retire()
    first = PRICES.dollars(TokenUsage(input_tokens=1000, output_tokens=100), PRICED)
    second = PRICES.dollars(TokenUsage(input_tokens=3000, output_tokens=100), PRICED)
    assert first is not None and second is not None
    assert member.ask_costs == (Cost(tokens=1100, dollars=first), Cost(tokens=3100, dollars=second))
    assert member.estimate_ask() == Cost(tokens=3100, dollars=second)


async def test_an_unpriced_ask_keeps_the_declared_dollars() -> None:
    """EA1: an axis no measured ask carried comes from the declaration."""
    member = MethodMember(
        Analyst("left"), "read", model=_Named([_spent(10, 5)], model_id="unlisted"), ask_estimate=Cost(dollars=2.0)
    )
    async with RuntimeHarness() as h:
        await member.spawn(h.coordinator)
        try:
            await member.ask("one")
        finally:
            await member.retire()
    assert member.ask_costs == (Cost(tokens=15),)
    assert member.estimate_ask() == Cost(tokens=15, dollars=2.0)


async def test_an_ask_that_reports_no_usage_is_not_recorded() -> None:
    """EA2: an ask whose model reports no usage leaves the member on its declaration."""
    member = FunctionMember(read, name="left", model=Counting([_reading(0, 0)]), ask_estimate=Cost(tokens=250))
    async with RuntimeHarness() as h:
        await member.spawn(h.coordinator)
        try:
            await member.ask("one")
        finally:
            await member.retire()
    assert member.ask_costs == ()
    assert member.estimate_ask() == Cost(tokens=250)


async def test_overlapping_asks_on_one_member_are_not_recorded() -> None:
    """EA2: two asks in flight on one member share one window, so neither is attributed."""
    model = _Named([_spent(10, 5, await_before="gate"), _spent(20, 5), _spent(30, 5)], model_id=None)
    member = FunctionMember(read, name="left", model=model)
    async with RuntimeHarness() as h:
        await member.spawn(h.coordinator)
        try:
            both = asyncio.gather(member.ask("one"), member.ask("two"))
            await asyncio.sleep(0.05)
            h.release("gate")
            await both
            await member.ask("three")
        finally:
            await member.retire()
    assert member.ask_costs == (Cost(tokens=35),), "the overlapping pair recorded nothing; the third ask did"


async def test_the_history_keeps_the_latest_asks_only() -> None:
    """EA1: an old outlier ages out after ``ASK_HISTORY`` newer measured asks."""
    turns = [_spent(5000, 0)] + [_spent(10, 0)] * ASK_HISTORY
    member = FunctionMember(read, name="left", model=_Named(turns, model_id=None))
    async with RuntimeHarness() as h:
        await member.spawn(h.coordinator)
        try:
            for index in range(len(turns)):
                await member.ask(f"turn {index}")
        finally:
            await member.retire()
    assert len(member.ask_costs) == ASK_HISTORY
    assert member.estimate_ask() == Cost(tokens=10)
