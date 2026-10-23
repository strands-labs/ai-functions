"""Offline tests for stopping a live run: ``Team.cancel`` and ``StopRun`` (C13, C14).

Each claim is read off the wire where it can be: the lead's recorded model calls and the coordinator's registry,
not the returned ``TeamRun`` alone.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
from typing import Any

from ai_functions.experimental.teams import FunctionMember, Member, StopRun, Team, Workspace
from ai_functions.testing import RuntimeHarness

from .teams_wire import Counting, call_member, read, reading, ruling, scripted_lead


class Halt:
    """Stops the run at the first member ask."""

    def __init__(self, reason: str = "halted") -> None:
        self.reason = reason

    def before_ask(self, work: Workspace, member: Member) -> None:
        raise StopRun(self.reason)


class CancelOnAssemble:
    """Cancels its own team while the run assembles; ``reasons`` are passed to ``cancel`` in order."""

    def __init__(self, *reasons: str, runs: int = 1) -> None:
        self.reasons = reasons
        self.runs = runs
        self.returned: list[bool] = []

    def on_assemble(self, work: Workspace) -> None:
        if self.runs <= 0:
            return
        self.runs -= 1
        self.returned.extend(work.team.cancel(reason) for reason in self.reasons)


class Closing:
    """Records what ``on_teardown`` and ``on_closed`` saw."""

    def __init__(self) -> None:
        self.torn_down = 0
        self.closed: list[tuple[bool, str | None, BaseException | None]] = []

    def on_teardown(self, work: Workspace) -> None:
        self.torn_down += 1

    def on_closed(self, work: Workspace) -> None:
        self.closed.append((work.cancelled, work.cancel_reason, work.fault))


class Stalling(Counting):
    """A lead model that blocks inside its first call until cancelled, so the run is live when the test cancels it."""

    def __init__(self, turns: list[Any]) -> None:
        super().__init__(turns)
        self.started = asyncio.Event()

    async def _stall(self, inner: AsyncIterable[Any]) -> AsyncIterable[Any]:
        self.started.set()
        await asyncio.Event().wait()
        async for chunk in inner:
            yield chunk

    def stream(self, *args: Any, **kwargs: Any) -> AsyncIterable[Any]:
        return self._stall(super().stream(*args, **kwargs))


class TestStopRun:
    async def test_a_before_ask_stop_cancels_the_run_asks_nobody_and_tears_down(self):
        """A ``StopRun`` from ``before_ask`` stops the run before the ask it was checking; teardown still runs, nothing
        stays live, and ``on_closed`` reads a cancel, not a fault (C14)."""
        member_model = Counting([reading()])
        closing = Closing()
        lead, _ = scripted_lead([call_member("left"), ruling(cites=["never"])])
        async with RuntimeHarness() as h:
            team = Team(
                lead, [FunctionMember(read, name="left", model=member_model)], hooks=[Halt("stop here"), closing]
            )
            run = await team.run("go", h.coordinator)
            assert await h.coordinator.list_threads() == []

        assert run.cancelled is True and run.cancel_reason == "stop here"
        assert run.answer is None
        assert member_model.contexts == [], "the member was never asked"
        assert closing.torn_down == 1
        assert closing.closed == [(True, "stop here", None)]

    async def test_the_lead_makes_no_model_call_after_the_stop(self):
        """C13: the stopped ask ends the lead's cycle at the tool boundary. Without that, the lead's model loop reads
        the missing tool result and calls the model a second time before the run tears the lead down."""
        lead, lead_model = scripted_lead([call_member("left"), ruling(cites=["never"])])
        async with RuntimeHarness() as h:
            member = FunctionMember(read, name="left", model=Counting([reading()]))
            await Team(lead, [member], hooks=[Halt()]).run("go", h.coordinator)
        assert len(lead_model.contexts) == 1, "the lead's model was called again after the stop"


class TestCancel:
    async def test_cancel_from_a_hook_stops_the_run_before_the_lead_runs(self):
        """``cancel`` on a live run returns true, and a run cancelled while it assembles never runs its lead."""
        hook = CancelOnAssemble("halted")
        lead, lead_model = scripted_lead([ruling()])
        async with RuntimeHarness() as h:
            run = await Team(lead, [], hooks=[hook]).run("go", h.coordinator)
            assert await h.coordinator.list_threads() == []
        assert hook.returned == [True]
        assert run.cancelled is True and run.cancel_reason == "halted"
        assert lead_model.contexts == []

    def test_cancel_with_no_live_run_returns_false(self):
        """Nothing to stop, nothing done."""
        lead, _ = scripted_lead([ruling()])
        assert Team(lead, []).cancel("nothing") is False

    async def test_the_first_reason_wins(self):
        """A second ``cancel`` on a run already stopping keeps the first reason."""
        hook = CancelOnAssemble("first", "second")
        lead, _ = scripted_lead([ruling()])
        async with RuntimeHarness() as h:
            run = await Team(lead, [], hooks=[hook]).run("go", h.coordinator)
        assert hook.returned == [True, True]
        assert run.cancel_reason == "first"

    async def test_a_cancelled_run_does_not_stop_the_next_run_on_the_instance(self):
        """The stop belongs to one run: the same team answers its next run."""
        hook = CancelOnAssemble("only once")
        lead, _ = scripted_lead([ruling(cites=["second"])])
        team = Team(lead, [], hooks=[hook])
        async with RuntimeHarness() as h:
            first = await team.run("one", h.coordinator)
            second = await team.run("two", h.coordinator)
        assert first.cancelled is True
        assert second.cancelled is False and second.answer.cites == ["second"]

    async def test_cancel_from_another_task_interrupts_the_lead_mid_cycle(self):
        """The lead is interrupted in the middle of a model call, every member is retired, and the run reports it."""
        lead_model = Stalling([ruling()])
        lead, _ = scripted_lead([])
        closing = Closing()
        team = Team(
            lead.replace(model=lead_model),
            [FunctionMember(read, name="left", model=Counting([]))],
            hooks=[closing],
        )
        async with RuntimeHarness() as h:
            running = asyncio.create_task(team.run("go", h.coordinator))
            await asyncio.wait_for(lead_model.started.wait(), 5)
            assert team.cancel("from outside") is True
            run = await asyncio.wait_for(running, 5)
            assert await h.coordinator.list_threads() == []
        assert run.cancelled is True and run.cancel_reason == "from outside"
        assert closing.closed == [(True, "from outside", None)]
