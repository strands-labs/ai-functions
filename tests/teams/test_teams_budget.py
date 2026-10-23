"""Teams wiring for the admission contract: ``Team(ask_timeout_s=)`` and ``Team.run(budget=)``.

Every claim is checked on the wire (``Counting``) or on the budget itself, never inferred from the
``TeamRun`` alone. Zero-argument defaults keep today's behavior: a team built without
``ask_timeout_s`` and run without ``budget`` reserves nothing and times nothing out.

- C9: the team's ``ask_timeout_s`` is the deadline for every member whose own is ``None``; a member
  past it is retired and the lead reads the failure as text;
- C10: with ``budget=`` every member ask is a reservation at the member's ``estimate_ask()`` (or the
  budget's default estimate), settled from the member thread's ``TOKEN_USAGE``; under ``refuse`` a
  member with no estimate is refused before anything spawns, by name; an exhausted budget reaches the
  lead as text.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_functions.experimental.teams import FunctionMember as TeamMember
from ai_functions.experimental.teams import MemberCall, Team, WiringError
from ai_functions.runtime.budget import Budget, Cost
from ai_functions.testing import RuntimeHarness, Turn

from .teams_wire import Counting, Spy, call_member, read, ruling, scripted_lead


def _reading(*, input_tokens: int = 30, output_tokens: int = 20) -> Turn:
    """A member turn that spends tokens, so the member thread emits a ``TOKEN_USAGE`` the budget can settle."""
    return Turn(
        tool_calls=(("Reading", {"source": "left", "detail": "seen"}),),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class _Hanging(Spy):
    """A fake member that may carry an ``ask_timeout_s`` of its own, and answers late or never."""

    def __init__(self, name: str, *, ask_timeout_s: float | None = None, delay_s: float | None = None) -> None:
        super().__init__(name)
        self.ask_timeout_s = ask_timeout_s
        self.delay_s = delay_s

    async def ask(self, request: str) -> Any:
        self.events.append(f"ask:{request}")
        if self.delay_s is None:
            await asyncio.Event().wait()
        else:
            await asyncio.sleep(self.delay_s)
        return self.answer


class _Estimating(Spy):
    """A member that declares what one ask costs and records what the budget held while it ran."""

    def __init__(self, name: str, *, estimate: Cost | None, budget: Budget) -> None:
        super().__init__(name)
        self._estimate = estimate
        self._budget = budget
        self.reserved_during_ask: list[Cost] = []

    def estimate_ask(self) -> Cost | None:
        return self._estimate

    async def ask(self, request: str) -> Any:
        self.reserved_during_ask.append(self._budget.reserved)
        return await super().ask(request)


# ── C9: the team-wide ask deadline ──────────────────────────────────────


class TestAskTimeout:
    async def test_a_member_past_the_team_deadline_is_retired_and_the_lead_reads_text(self) -> None:
        member = _Hanging("slow", ask_timeout_s=None)
        lead, lead_model = scripted_lead([call_member("slow", "take your time"), ruling(admitted=False)])
        async with RuntimeHarness() as h:
            run = await Team(lead, [member], ask_timeout_s=0.05).run("go", h.coordinator)

        assert member.retirements >= 1 and member.events[:2] == ["spawn", "ask:take your time"]
        assert member.events[2] == "retire", "retired at the deadline, before the run's own unwind"
        results = lead_model.tool_results(1)
        assert len(results) == 1 and "did not answer within 0.05s" in results[0] and "slow" in results[0]
        calls = [e for e in run.transcript if isinstance(e, MemberCall)]
        assert len(calls) == 1 and "did not answer" in calls[0].answer, "the transcript records the failure (C6)"

    async def test_a_member_with_its_own_deadline_keeps_it(self) -> None:
        """The guard, from the other side: the team default applies only where the member's own is None."""
        member = _Hanging("own", ask_timeout_s=5.0, delay_s=0.1)
        lead, lead_model = scripted_lead([call_member("own"), ruling()])
        async with RuntimeHarness() as h:
            run = await Team(lead, [member], ask_timeout_s=0.02).run("go", h.coordinator)
        assert lead_model.tool_results(1) == ["ok"], "the member answered under its own, longer deadline"
        assert run.answer.admitted is True

    async def test_a_retired_member_is_not_asked_again_and_both_calls_are_recorded(self) -> None:
        """C9: past the deadline the member is retired, so a second call to its tool asks nothing and says why;
        both calls are on the transcript (C6)."""
        member = _Hanging("slow")
        lead, lead_model = scripted_lead([call_member("slow", "first"), call_member("slow", "second"), ruling()])
        async with RuntimeHarness() as h:
            run = await Team(lead, [member], ask_timeout_s=0.05).run("go", h.coordinator)

        assert [e for e in member.events if e.startswith("ask:")] == ["ask:first"], "the retired member is not asked"
        second = lead_model.tool_results(2)[-1]
        assert second.startswith("error: slow was retired") and "can't be asked again" in second, second
        assert "Ask again" not in lead_model.tool_results(1)[0], "the timeout text never invites a call that fails"
        calls = [e for e in run.transcript if isinstance(e, MemberCall)]
        assert [c.request for c in calls] == ["first", "second"]

    async def test_a_members_own_deadline_is_enforced_without_a_team_deadline(self) -> None:
        """C9: a member's own ``ask_timeout_s`` is held by the core, not just a reason to skip the team's."""
        member = _Hanging("own", ask_timeout_s=0.05)
        lead, lead_model = scripted_lead([call_member("own"), ruling()])
        async with RuntimeHarness() as h:
            await Team(lead, [member]).run("go", h.coordinator)
        assert "did not answer within 0.05s" in lead_model.tool_results(1)[0]
        assert member.retirements >= 1

    async def test_no_team_deadline_means_no_timeout(self) -> None:
        member = _Hanging("patient", delay_s=0.1)
        lead, lead_model = scripted_lead([call_member("patient"), ruling()])
        async with RuntimeHarness() as h:
            await Team(lead, [member]).run("go", h.coordinator)
        assert lead_model.tool_results(1) == ["ok"]

    def test_a_non_positive_team_deadline_is_refused_at_construction(self) -> None:
        lead, _ = scripted_lead([ruling()])
        with pytest.raises(WiringError, match="ask_timeout_s"):
            Team(lead, [], ask_timeout_s=0)


# ── C10: budgeted member asks ────────────────────────────────────────────


class TestTeamBudget:
    async def test_each_ask_is_reserved_at_the_default_estimate_and_settled_from_usage(self) -> None:
        budget = Budget(Cost(tokens=1000), on_unknown="estimate", default_estimate=Cost(tokens=200), name="run")
        async with RuntimeHarness() as h:
            member = TeamMember(read, name="left", model=Counting([_reading()]))
            lead, lead_model = scripted_lead([call_member("left"), ruling()])
            run = await Team(lead, [member]).run("go", h.coordinator, budget=budget)

        assert run.answer.admitted is True
        assert "seen" in lead_model.tool_results(1)[0], "the member answered normally on the wire"
        assert budget.spent == Cost(tokens=50), "settled from the member thread's TOKEN_USAGE (30 in + 20 out)"
        assert budget.reserved == Cost(), "nothing is left held after the ask"

    async def test_a_member_estimate_is_the_reservation_while_it_runs(self) -> None:
        budget = Budget(Cost(tokens=1000), name="run")
        member = _Estimating("est", estimate=Cost(tokens=40), budget=budget)
        lead, _ = scripted_lead([call_member("est"), ruling()])
        async with RuntimeHarness() as h:
            await Team(lead, [member]).run("go", h.coordinator, budget=budget)
        assert member.reserved_during_ask == [Cost(tokens=40)]
        assert budget.reserved == Cost() and budget.spent == Cost(tokens=40), (
            "a member with no thread of its own settles at its estimate, the honest figure"
        )

    async def test_refuse_policy_refuses_a_member_without_an_estimate_before_anything_spawns(self) -> None:
        budget = Budget(Cost(tokens=1000), name="strict")
        member = _Estimating("nameless", estimate=None, budget=budget)
        lead, lead_model = scripted_lead([ruling()])
        async with RuntimeHarness() as h:
            with pytest.raises(WiringError) as info:
                await Team(lead, [member]).run("go", h.coordinator, budget=budget)
        assert "nameless" in str(info.value) and "estimate_ask" in str(info.value) and "strict" in str(info.value)
        assert member.events == [], "refused before spawn: the member never spawned"
        assert lead_model.contexts == [], "and the lead never ran"

    async def test_an_exhausted_budget_reaches_the_lead_as_text_and_the_member_is_not_asked(self) -> None:
        budget = Budget(Cost(tokens=10), on_unknown="estimate", default_estimate=Cost(tokens=100), name="tiny")
        async with RuntimeHarness() as h:
            member = TeamMember(read, name="left", model=Counting([]))
            lead, lead_model = scripted_lead([call_member("left"), ruling(admitted=False)])
            run = await Team(lead, [member]).run("go", h.coordinator, budget=budget)
        results = lead_model.tool_results(1)
        assert len(results) == 1 and results[0].startswith("error:") and "tiny" in results[0]
        assert budget.spent == Cost() and not budget.exhausted, "a refused reservation spends nothing"
        calls = [e for e in run.transcript if isinstance(e, MemberCall)]
        assert len(calls) == 1 and calls[0].answer.startswith("error:")

    async def test_no_budget_reserves_nothing(self) -> None:
        member = _Estimating("free", estimate=Cost(tokens=40), budget=Budget(Cost(tokens=1)))
        lead, lead_model = scripted_lead([call_member("free"), ruling()])
        async with RuntimeHarness() as h:
            await Team(lead, [member]).run("go", h.coordinator)
        assert lead_model.tool_results(1) == ["ok"]
        assert member.reserved_during_ask == [Cost()], "without budget= the estimate is never consulted"
