"""C13 and C10 at the end of a run: no cycle after the answer, a total taken once everything stopped, a full charge.

None of these reach a thread through ``send_message``, which no managed thread carries (C12). The late work is what
a hook or a background submit can still start:

- a hook that asks the lead for more once its answer is taken is refused by the runtime, and the lead runs no
  second cycle;
- a member cycle nobody awaited that ends during teardown is in ``TeamRun.cost`` and in the budget;
- a member cycle still running at the retire is cancelled on the record, and the tokens its cleanup reports count;
- the budget is charged the lead's tokens and a hook's ask, and a later ask is refused once the run has spent the
  budget;
- a budget that bounds dollars is charged the run's priced spend and refuses the ask past it, and usage the price
  table cannot price refuses every later ask under it, naming the model, while a tokens-only budget ignores it.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from ai_functions import ai_function
from ai_functions.experimental.teams import FunctionMember, MemberCall, Team
from ai_functions.experimental.teams.contracts import Workspace
from ai_functions.runtime import ThreadNotFoundError
from ai_functions.runtime.budget import Budget, Cost
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import EventKind, InputShape, ThreadId, ThreadSpawnedEvent, TokenUsage, TokenUsageEvent


@ai_function[str](structured_output=False)
def lead_fn(request: str) -> str | None:
    """Lead the team on: {request}"""


@ai_function[str](structured_output=False)
def worker_fn(request: str) -> str | None:
    """Do: {request}"""


def text(value: str, tokens: int = 0, **kwargs: Any) -> Turn:
    return Turn(text=value, input_tokens=tokens, **kwargs)


def ask(member: str, request: str, tokens: int = 0) -> Turn:
    return Turn(tool_calls=((member, {"request": request}),), input_tokens=tokens)


async def logged_tokens(coordinator: Any, *thread_ids: str) -> int:
    total = 0
    for thread_id in thread_ids:
        for event in await coordinator.get_events(ThreadId(thread_id), kinds=[EventKind.TOKEN_USAGE]):
            total += event.token_usage.input_tokens + event.token_usage.output_tokens
    return total


async def children(coordinator: Any, lead_id: str | None) -> list[str]:
    """The threads the lead's log records spawning, in order: the cast, then anything spawned under the lead."""
    assert lead_id is not None
    events = await coordinator.get_events(ThreadId(lead_id), kinds=[EventKind.THREAD_SPAWNED])
    return [str(e.child_thread_id) for e in events if isinstance(e, ThreadSpawnedEvent)]


class TemplateMember:
    """A minimal ``Member`` over any spawnable template, so a test can drive a thread of its own making."""

    def __init__(self, name: str, template: Any) -> None:
        self.name = name
        self.template = template
        self.handle: Any = None

    async def spawn(self, coordinator: Any, *, parent_id: ThreadId | None = None) -> Any:
        self.handle = await coordinator.spawn(self.template, thread_name=self.name, parent_id=parent_id)
        return self.handle

    async def ask(self, request: str) -> Any:
        return await self.handle.run(request)

    async def retire(self) -> None:
        handle, self.handle = self.handle, None
        if handle is not None:
            try:
                await handle.terminate_now()
            except ThreadNotFoundError:
                pass


class AskTheLeadAgain:
    """At teardown, asks the lead for one more cycle and keeps what the runtime answered."""

    def __init__(self) -> None:
        self.refused: BaseException | None = None

    async def on_teardown(self, work: Workspace) -> None:
        assert work.lead is not None
        try:
            await work.lead.run("one more thing")
        except ThreadNotFoundError as error:
            self.refused = error


class Background:
    """Starts one cycle on a member at assemble that nobody awaits; ``finish`` runs at teardown."""

    def __init__(self, name: str, finish: Any = None) -> None:
        self.name = name
        self.finish = finish
        self.pending: asyncio.Future[Any] | None = None

    def on_assemble(self, work: Workspace) -> None:
        member = next(m for m in work.members if m.name == self.name)
        self.pending = member.handle.run("background job")  # type: ignore[attr-defined]

    async def on_teardown(self, work: Workspace) -> None:
        if self.finish is not None:
            await self.finish(self)


# ══════════════════════════════════════════════════════════════════
# C13: nothing runs on the lead once its answer is taken
# ══════════════════════════════════════════════════════════════════


async def test_a_hook_that_asks_the_lead_after_its_answer_is_taken_is_refused() -> None:
    late = AskTheLeadAgain()
    team = Team(lead_fn.replace(model=ScriptedModel([text("LEAD-FINAL"), text("LEAD-AGAIN")])), [], hooks=[late])
    async with RuntimeHarness() as h:
        run = await asyncio.wait_for(team.run("go", h.coordinator), 5)
        assert run.lead_id is not None
        log = await h.coordinator.get_events(ThreadId(run.lead_id))

    assert run.answer.strip() == "LEAD-FINAL"
    assert isinstance(late.refused, ThreadNotFoundError), late.refused
    assert [e.kind for e in log].count(EventKind.STARTED) == 1, "the lead ran no cycle after its answer"


# ══════════════════════════════════════════════════════════════════
# C13: the run is totaled once every thread has stopped
# ══════════════════════════════════════════════════════════════════


async def test_a_member_cycle_that_ends_during_teardown_is_in_the_cost_and_the_budget() -> None:
    budget = Budget(Cost(tokens=1_000_000), on_unknown="estimate", default_estimate=Cost(tokens=5))
    worker_model = ScriptedModel([text("BG-DONE", 1000, await_before="background")])

    async with RuntimeHarness() as h:

        async def finish(hook: Background) -> None:
            h.release("background")
            assert hook.pending is not None
            await hook.pending

        background = Background("B", finish)
        team = Team(
            lead_fn.replace(model=ScriptedModel([text("LEAD-FINAL", 10)])),
            [FunctionMember(worker_fn.replace(model=worker_model), name="B")],
            hooks=[background],
        )
        run = await asyncio.wait_for(team.run("go", h.coordinator, budget=budget), 5)
        logged = await logged_tokens(h.coordinator, str(run.lead_id), *await children(h.coordinator, run.lead_id))

    assert logged == 1010
    assert run.cost == Cost(tokens=1010), "the cycle that ended during teardown is counted"
    assert budget.spent == Cost(tokens=1010), "and charged"


async def test_a_member_cycle_still_running_at_the_retire_is_cancelled_on_the_record_and_its_cleanup_counts() -> None:
    entered = asyncio.Event()

    class Backend:
        name = "cleanup"

        async def execute(self, ctx: Any, prompt: str) -> str:
            entered.set()
            try:
                await asyncio.Future()
            finally:
                ctx.on_event(TokenUsageEvent(thread_id=ctx.thread_id, token_usage=TokenUsage(input_tokens=17)))
            return prompt

        async def teardown(self) -> None:
            pass

    class Template:
        input_shape = InputShape.STR_PROMPT

        def to_thread(self) -> Backend:
            return Backend()

    async def started(hook: Background) -> None:
        await asyncio.wait_for(entered.wait(), 5)

    team = Team(
        lead_fn.replace(model=ScriptedModel([text("LEAD-FINAL")])),
        # The fake thread implements execute and teardown, the part of Thread this retire drives.
        [TemplateMember("cleanup", Template())],
        hooks=[Background("cleanup", started)],
    )
    async with RuntimeHarness() as h:
        run = await asyncio.wait_for(team.run("go", h.coordinator), 5)
        (member_id,) = await children(h.coordinator, run.lead_id)
        member_log = await h.coordinator.get_events(ThreadId(member_id))

    kinds = [e.kind for e in member_log]
    assert EventKind.CANCELLED in kinds, kinds
    assert run.cost == Cost(tokens=17), "tokens reported while the thread was torn down are counted"


# ══════════════════════════════════════════════════════════════════
# C10: every token the run spends is charged
# ══════════════════════════════════════════════════════════════════


class AskOnAnswer:
    """A reviewing hook that asks a member once through ``work.ask`` and accepts."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def on_answer(self, work: Workspace, answer: Any) -> Any:
        from ai_functions.experimental.teams.types import Accept

        member = next(m for m in work.members if m.name == self.name)
        _ = await work.ask(member, "check the answer")
        return Accept()


async def test_the_budget_is_charged_the_leads_tokens_and_a_hooks_ask() -> None:
    budget = Budget(Cost(tokens=10_000), on_unknown="estimate", default_estimate=Cost(tokens=100))
    team = Team(
        lead_fn.replace(model=ScriptedModel([ask("A", "part A", 7), text("LEAD-FINAL", 3)])),
        [FunctionMember(worker_fn.replace(model=ScriptedModel([text("A-DONE", 50), text("A-CHECKED", 20)])), name="A")],
        hooks=[AskOnAnswer("A")],
    )
    async with RuntimeHarness() as h:
        run = await asyncio.wait_for(team.run("go", h.coordinator, budget=budget), 5)
        logged = await logged_tokens(h.coordinator, str(run.lead_id), *await children(h.coordinator, run.lead_id))

    assert logged == 80
    assert run.cost == Cost(tokens=80)
    assert budget.spent == Cost(tokens=80) and budget.reserved == Cost(), "charged every token, holding nothing"


class ReviseOnce:
    """Sends the lead back once, so its second cycle runs after the first one's usage is on the log."""

    def __init__(self) -> None:
        self.rounds = 0

    def on_answer(self, work: Workspace, answer: Any) -> Any:
        from ai_functions.experimental.teams.types import Accept, Revise

        self.rounds += 1
        return Revise(feedback="ask A", cap=1) if self.rounds == 1 else Accept()


async def test_an_ask_is_refused_once_the_leads_own_spend_has_used_the_budget() -> None:
    budget = Budget(Cost(tokens=100), on_unknown="estimate", default_estimate=Cost(tokens=10))
    team = Team(
        lead_fn.replace(model=ScriptedModel([text("DRAFT", 95), ask("A", "part A"), text("LEAD-FINAL")])),
        [FunctionMember(worker_fn.replace(model=ScriptedModel([text("A-DONE")])), name="A")],
        hooks=[ReviseOnce()],
    )
    async with RuntimeHarness() as h:
        run = await asyncio.wait_for(team.run("go", h.coordinator, budget=budget), 5)

    calls = [e for e in run.transcript if isinstance(e, MemberCall) and e.member == "A"]
    assert len(calls) == 1 and calls[0].answer.startswith("error: A was not asked"), calls
    assert budget.spent == Cost(tokens=95)


# ══════════════════════════════════════════════════════════════════
# C10: a budget that bounds dollars is enforced on the run's priced spend
# ══════════════════════════════════════════════════════════════════

OPUS = "global.anthropic.claude-opus-4-6-v1"
"""Priced at the list rate, $5 per million input tokens."""


class Priced(ScriptedModel):
    """A scripted model that names its model, so its usage is priced."""

    def __init__(self, turns: list[Turn], model_id: str | None = OPUS) -> None:
        super().__init__(turns)
        self.model_id = model_id

    def get_config(self) -> dict[str, object]:
        return {**super().get_config(), "model_id": self.model_id}


async def test_a_dollar_budget_is_charged_the_runs_priced_spend_and_refuses_the_ask_past_it() -> None:
    budget = Budget(Cost(dollars=1.0), on_unknown="estimate", default_estimate=Cost(dollars=0.01), name="dollars")
    lead = Priced([ask("A", "one", 1_000), ask("A", "two", 1_000), text("LEAD-FINAL", 1_000)])
    member = Priced([text("A-ONE", 300_000), text("A-TWO")])
    team = Team(lead_fn.replace(model=lead), [FunctionMember(worker_fn.replace(model=member), name="A")])
    async with RuntimeHarness() as h:
        run = await asyncio.wait_for(team.run("go", h.coordinator, budget=budget), 5)

    answers = [e.answer for e in run.transcript if isinstance(e, MemberCall) and e.member == "A"]
    assert answers[0].strip() == "A-ONE"
    assert answers[1].startswith("error: A was not asked") and "'dollars'" in answers[1], answers
    assert run.cost is not None and run.cost.tokens == 303_000
    assert run.cost.dollars == pytest.approx(1.515), "$5 per million input tokens on every event"
    assert budget.spent == run.cost and budget.exhausted


async def test_usage_a_dollar_budget_cannot_price_refuses_every_later_ask_by_model() -> None:
    budget = Budget(Cost(dollars=100.0), on_unknown="estimate", default_estimate=Cost(dollars=0.01), name="dollars")
    lead = Priced([ask("A", "one"), ask("A", "two"), ask("A", "three"), text("LEAD-FINAL")])
    member = Priced([text("A-ONE", 10), text("A-TWO")], model_id="house-model-7")
    team = Team(lead_fn.replace(model=lead), [FunctionMember(worker_fn.replace(model=member), name="A")])
    async with RuntimeHarness() as h:
        run = await asyncio.wait_for(team.run("go", h.coordinator, budget=budget), 5)

    answers = [e.answer for e in run.transcript if isinstance(e, MemberCall) and e.member == "A"]
    assert answers[0].strip() == "A-ONE"
    assert len(answers) == 3 and all("'house-model-7'" in a and "'dollars'" in a for a in answers[1:]), answers
    assert run.cost == Cost(tokens=10, dollars=0.0), "the lead priced at nothing, the house model not at all"


async def test_a_tokens_only_budget_is_not_refused_by_usage_it_cannot_price() -> None:
    budget = Budget(Cost(tokens=1_000), on_unknown="estimate", default_estimate=Cost(tokens=1))
    lead = Priced([ask("A", "one", 10), text("LEAD-FINAL")], model_id="house-model-7")
    team = Team(
        lead_fn.replace(model=lead), [FunctionMember(worker_fn.replace(model=Priced([text("A-ONE", 5)])), name="A")]
    )
    async with RuntimeHarness() as h:
        run = await asyncio.wait_for(team.run("go", h.coordinator, budget=budget), 5)

    assert [e.answer.strip() for e in run.transcript if isinstance(e, MemberCall) and e.member == "A"] == ["A-ONE"]
    assert run.cost is not None and run.cost.tokens == 15
