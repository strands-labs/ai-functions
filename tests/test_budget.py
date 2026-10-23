"""Hierarchical cost reservations: ``Cost``, ``Budget`` and ``Reservation``.

- a ``Cost`` is a frozen value that round-trips through JSON; a limit constrains at least one axis (BU1);
- ``reserve`` is all-or-nothing up the chain and a refusal names the path of the budget that refused (BU2);
- ``settle`` replaces the estimate with the actual on every budget in the chain and never raises (BU3);
- unknown cost is a policy: ``refuse`` raises ``UnknownCost``, ``estimate`` holds the budget's own
  ``default_estimate``, and ``estimate`` without one is refused at construction (BU4);
- ``charge`` records a spend that already happened, past the limit included (BU5).
"""

from __future__ import annotations

import pytest

from ai_functions.runtime.budget import Budget, BudgetError, BudgetExhausted, Cost, Reservation, UnknownCost
from ai_functions.types import TokenUsage

# ── Cost ─────────────────────────────────────────────────────────────────


class TestCost:
    def test_a_cost_is_frozen_and_round_trips(self) -> None:
        cost = Cost(tokens=10, dollars=0.5)
        assert Cost.model_validate_json(cost.model_dump_json()) == cost
        with pytest.raises(Exception):  # noqa: B017, PT011 — pydantic's frozen error type is its own business
            cost.tokens = 3  # type: ignore[misc]

    def test_sum_treats_an_unmeasured_axis_as_zero_when_the_other_side_measured_it(self) -> None:
        assert Cost(tokens=3) + Cost(tokens=4, dollars=1.0) == Cost(tokens=7, dollars=1.0)
        assert Cost() + Cost() == Cost()
        assert (Cost(dollars=0.25) + Cost(dollars=0.25)).dollars == pytest.approx(0.5)

    def test_from_usage_sums_all_four_counters(self) -> None:
        usage = TokenUsage(input_tokens=1, output_tokens=2, cache_read_tokens=3, cache_write_tokens=4)
        assert Cost.from_usage(usage) == Cost(tokens=10)

    def test_negative_quantities_are_refused(self) -> None:
        with pytest.raises(ValueError):
            Cost(tokens=-1)
        with pytest.raises(ValueError):
            Cost(dollars=-0.1)


# ── Budget ───────────────────────────────────────────────────────────────


class TestBudget:
    def test_a_limit_with_no_constrained_axis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            Budget(Cost())

    def test_reserve_then_settle_moves_the_hold_from_estimate_to_actual(self) -> None:
        budget = Budget(Cost(tokens=100))
        reservation = budget.reserve(Cost(tokens=40))
        assert isinstance(reservation, Reservation)
        assert budget.reserved == Cost(tokens=40) and budget.spent == Cost() and budget.committed.tokens == 40
        assert budget.remaining == Cost(tokens=60)
        reservation.settle(Cost(tokens=25))
        assert reservation.settled and reservation.actual == Cost(tokens=25)
        assert budget.spent == Cost(tokens=25) and budget.reserved == Cost()
        assert budget.remaining == Cost(tokens=75)
        assert not budget.exhausted

    def test_reserve_past_the_limit_raises_naming_the_path_and_reserves_nothing(self) -> None:
        budget = Budget(Cost(tokens=100), name="run")
        _ = budget.reserve(Cost(tokens=80))
        with pytest.raises(BudgetExhausted) as info:
            budget.reserve(Cost(tokens=30))
        assert isinstance(info.value, BudgetError)
        assert info.value.path == "run" and "run" in str(info.value)
        assert budget.committed == Cost(tokens=80), "the refused reservation holds nothing"

    def test_exactly_the_limit_is_admitted_and_exhausts_the_budget(self) -> None:
        budget = Budget(Cost(tokens=100))
        _ = budget.reserve(Cost(tokens=100))
        assert budget.exhausted
        with pytest.raises(BudgetExhausted):
            budget.reserve(Cost())

    def test_a_settle_above_the_estimate_exhausts_without_raising(self) -> None:
        """The spend is a fact; settle records it and exhaustion is a state, read via ``exhausted`` (BU3)."""
        budget = Budget(Cost(tokens=100))
        reservation = budget.reserve(Cost(tokens=10))
        reservation.settle(Cost(tokens=150))
        assert budget.exhausted and budget.spent == Cost(tokens=150)
        assert budget.remaining == Cost(tokens=0), "remaining never goes negative"

    def test_dollars_axis_is_bounded_independently(self) -> None:
        budget = Budget(Cost(dollars=1.0))
        _ = budget.reserve(Cost(tokens=10_000, dollars=0.6))
        assert not budget.exhausted, "tokens are unconstrained on a dollar budget"
        with pytest.raises(BudgetExhausted):
            budget.reserve(Cost(dollars=0.5))

    def test_settle_may_be_repeated_with_a_running_total(self) -> None:
        budget = Budget(Cost(tokens=100))
        reservation = budget.reserve(Cost(tokens=50))
        reservation.settle(Cost(tokens=10))
        reservation.settle(Cost(tokens=30))
        assert budget.spent == Cost(tokens=30), "each settle replaces the figure; the caller owns the total"


class TestHierarchy:
    def test_child_reservations_hold_on_every_ancestor_and_settle_propagates_up(self) -> None:
        root = Budget(Cost(tokens=1000), name="fleet")
        team = root.child(Cost(tokens=300), name="team")
        member = team.child(Cost(tokens=100), name="left")
        assert member.path == "fleet/team/left" and member.parent is team and team.parent is root
        reservation = member.reserve(Cost(tokens=60))
        assert (member.reserved.tokens, team.reserved.tokens, root.reserved.tokens) == (60, 60, 60)
        reservation.settle(Cost(tokens=45))
        assert (member.spent.tokens, team.spent.tokens, root.spent.tokens) == (45, 45, 45)
        assert (member.reserved, team.reserved, root.reserved) == (Cost(), Cost(), Cost())

    def test_a_refusal_up_the_chain_names_the_ancestor_and_holds_nothing_anywhere(self) -> None:
        root = Budget(Cost(tokens=100), name="fleet")
        child = root.child(Cost(tokens=1000), name="greedy")
        _ = root.reserve(Cost(tokens=90))
        with pytest.raises(BudgetExhausted) as info:
            child.reserve(Cost(tokens=50))
        assert info.value.path == "fleet"
        assert child.committed == Cost() and root.committed == Cost(tokens=90)

    def test_children_inherit_the_unknown_cost_policy_unless_told_otherwise(self) -> None:
        root = Budget(Cost(tokens=100), on_unknown="estimate", default_estimate=Cost(tokens=5))
        child = root.child(Cost(tokens=50))
        assert child.on_unknown == "estimate" and child.default_estimate == Cost(tokens=5)
        strict = root.child(Cost(tokens=50), on_unknown="refuse")
        with pytest.raises(UnknownCost):
            strict.reserve(None)

    def test_default_names_still_make_a_readable_path(self) -> None:
        root = Budget(Cost(tokens=10))
        child = root.child(Cost(tokens=5))
        assert root.path == "budget" and child.path.startswith("budget/")


class TestUnknownCost:
    def test_refuse_is_the_default_and_names_the_path(self) -> None:
        budget = Budget(Cost(tokens=100), name="strict")
        assert budget.on_unknown == "refuse"
        with pytest.raises(UnknownCost) as info:
            budget.reserve(None)
        assert isinstance(info.value, BudgetError) and info.value.path == "strict"
        assert "estimate" in str(info.value), "the error names the fix"

    def test_estimate_holds_the_default_estimate_for_an_unknown_request(self) -> None:
        budget = Budget(Cost(tokens=100), on_unknown="estimate", default_estimate=Cost(tokens=30))
        reservation = budget.reserve(None)
        assert reservation.estimate == Cost(tokens=30) and budget.reserved == Cost(tokens=30)
        known = budget.reserve(Cost(tokens=1))
        assert known.estimate == Cost(tokens=1), "a known estimate is never replaced by the default"

    def test_estimate_without_a_default_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="default_estimate"):
            Budget(Cost(tokens=100), on_unknown="estimate")

    def test_an_unknown_policy_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError, match="on_unknown"):
            Budget(Cost(tokens=100), on_unknown="guess")  # type: ignore[arg-type]


class TestCharge:
    """BU5: a charge records a spend that already happened, up the chain, and nothing refuses it."""

    def test_a_charge_is_settled_on_every_ancestor_and_may_be_settled_again(self) -> None:
        root = Budget(Cost(tokens=100), name="fleet")
        team = root.child(Cost(tokens=50), name="team")
        charge = team.charge(Cost())
        assert charge.settled and not team.spent.tokens and root.reserved == Cost()
        charge.settle(Cost(tokens=30))
        assert (team.spent, root.spent) == (Cost(tokens=30), Cost(tokens=30))

    def test_an_exhausted_budget_still_takes_a_charge_past_its_limit(self) -> None:
        budget = Budget(Cost(tokens=10))
        _ = budget.reserve(Cost(tokens=10))
        assert budget.exhausted
        _ = budget.charge(Cost(tokens=25))
        assert budget.spent == Cost(tokens=25) and budget.committed == Cost(tokens=35)
