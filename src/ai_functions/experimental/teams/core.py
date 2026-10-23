"""``Team`` — a lead over live members, with hooks for everything that is not that.

The core owns exactly what every team needs — spawn the members, run the lead
with the members as typed tools, retire everybody — and everything else arrives
as a :class:`~ai_functions.experimental.teams.TeamHook`. A hook can contribute
tools to the lead or the members, rewrite the request, and review the answer
through a bounded Accept/Revise loop the core drives. The bare team is the
whole API for the common case::

    team = Team(
        lead=decide,
        members=[FunctionMember(read, name="left"), FunctionMember(read, name="right")],
    )
    run = await team.run("who is right")
    print(run.answer)

**The one hard mechanical constraint.** The runtime resolves exactly one
``config_hook`` per cycle and its ``tools`` patch *replaces* the compiled
tools. ``AIThread._run_cycle`` calls ``self._config.config_hook(ctx)`` once at
the top of the cycle, pops the ``config_hook`` key out of the patch and applies
the rest with ``dataclasses.replace``, so a ``tools`` key stands in for the
compiled ``tools=`` wholesale; ``ThreadConfig.config_hook`` documents that
replace-not-merge semantics; and the same method appends the coordinator tools
*after* the patch, only when the cycle's ``coordinator_tools_enabled`` holds,
which the core turns off on every thread it spawns (C12).
So the core owns the single hook on each thread and folds every contribution
into it: the lead's own hook and tools are recomposed first, then the member
tools, then every ``tools_for_lead``; a member's own ``tools=`` override is
recomposed by ``FunctionMember.spawn``'s wrapper. Nothing else may set
``config_hook`` on a thread the core manages — ``FunctionMember.equip`` refuses a
member that carries its own.

Invariants:
    C1 — one run is one subtree, torn down on every path. The lead's thread
    exists first, every member spawns as its child, and the ``finally`` runs
    every ``on_teardown`` and then retires everybody even when a hook or the
    lead raised; a teardown hook's own error resurfaces only when nothing
    else is already propagating.

    C2 — one ``config_hook`` per managed thread. The core composes every tool
    source into that one hook and never stacks a second; the lead's own hook
    and ``tools=`` survive the fold, and so do a member's own tools. A member
    without the ``Equippable`` seam is skipped by the fold, not refused, and
    the transcript names it in an ``unequipped`` note with the hooks whose
    tools it goes without.

    C3 — the answer loop terminates. Every restart spends one round of some
    hook's budget, so the loop is bounded by the sum of the caps in play;
    budgets persist across restarts, the cap is read off the latest verdict,
    and exhaustion passes the answer through and records ``revise_cap`` once
    per hook.

    C4 — a verdict that is neither ``Accept`` nor ``Revise`` is a wiring bug
    and raises :class:`WiringError` naming the hook, never a silent pass.

    C5 — member names are unique on the lead's wire; a collision is refused
    at construction, before anything spawns.

    C6 — the transcript is written by the wire that carried the event, as
    the typed records of :mod:`~ai_functions.experimental.teams.types`, and
    always under the member's real name whatever the tool had to be called.
    A member's answer is recorded as the text the lead read: a string as is,
    anything else formatted as a Strands tool formats its result (a model as
    its JSON).

    C7 — NoUnreviewedShip. After any ``Revise`` by hook k the walk restarts
    at hook 1 and re-consults hooks 1..k-1 on the revised answer before any
    ``Accept`` can ship it; an answer ships only after one full uninterrupted
    pass of ``Accept`` and cap-exhausted pass-throughs. In the code this is
    the ``break`` out of the ``for`` and the ``while revised`` walking again
    in ``_answer_loop``. The rejected alternative — per-hook loops rather
    than a restart chain — ships an accepted-then-mutated answer, and
    ``tests/teams/test_teams_invariants.py`` pins the code to the restart
    chain. A cap-exhausted pass-through in that final pass is still a hook
    that did not approve; whether the run may call itself a success anyway
    is C8's question.

    C8 — an exhausted mandatory review is a blocked result, never ordinary
    success. When the final pass carries a cap-exhausted pass-through whose
    latest verdict said ``mandatory=True``, ``Team.run`` returns
    ``TeamRun(blocked=True, blocked_by=(<label>, ...))`` with the last answer
    still attached, the labels in hook order, and the transcript unchanged
    (``revise_cap`` once per hook as under C3). The block is a fact about the
    final pass: a mandatory hook that exhausted on a draft and then accepted
    the revised answer approved what ships, so it blocks nothing. A verdict
    that leaves ``mandatory`` off keeps the C3 pass-through, so a team with
    no mandatory reviewer never sees the field.

    C9 — one deadline per ask, the member's own first. A member that carries
    an ``ask_timeout_s`` of its own is held to it; every other member is held
    to ``Team(ask_timeout_s=)``. The core applies the deadline in the member
    tool, around ``Member.ask``: past it the member is retired, so no orphan
    survives, and the lead reads the failure as ``error:`` text, recorded on
    the transcript under the member's real name (C6). A retired member is not
    asked again in the run: a later call to its tool asks nothing and answers
    the lead with ``error:`` text. ``None`` (the default) bounds nothing; a
    non-positive team deadline is refused at construction.

    C10 — a budgeted run is charged everything it spends and gates every lead ask.
    With ``Team.run(budget=)`` the run holds one charge on the
    :class:`~ai_functions.runtime.budget.Budget` (``Budget.charge``, BU5),
    settled at the run's measured spend, the same fold ``TeamRun.cost``
    reports: its tokens, and the dollars the coordinator's ``PriceTable``
    prices each usage event at by the model it names. It is refreshed around
    every ask and closed at the final fold (C13), so the lead's tokens, a
    hook's ask and a nested team's are charged along with the cast's. A
    hook's own ask (``Workspace.ask``) is charged but reserves nothing. A
    budget that bounds dollars is enforced like one that bounds tokens, and
    once the run has spent usage the table cannot price, every later ask
    under a budget whose chain bounds dollars is refused as text naming the
    budget and the model. Every ask the lead makes of a member is also one reservation at the
    estimate the member declares through the optional ``estimate_ask()``
    method (``None`` or no method means unknown, which the budget's
    ``on_unknown`` policy decides), taken once the charge is refreshed, so a
    budget the run has spent refuses the next ask. Under
    ``on_unknown="refuse"`` a cast member with no estimate is refused with a
    :class:`WiringError` naming the member, before the lead or any member
    spawns. A reservation the budget refuses reaches the lead as ``error:``
    text and asks the member nothing. After the ask the reservation settles at
    nothing when the run's measured total moved during it (those tokens are in
    the charge), and at its estimate when nothing moved, the honest figure for
    a member whose work this coordinator cannot measure (a remote agent, a
    fake). ``budget=None`` (the default) reserves nothing and never consults
    ``estimate_ask``.

    C11 — one live run per ``Team`` instance. ``run`` holds an
    ``asyncio.Lock`` for the whole run; a second call while the first is in
    flight is a ``WiringError`` naming the fix (two ``Team`` instances),
    raised before anything spawns. Members keep mutable handle slots and hooks
    key state by workspace, which is why. Sequential runs are unaffected.

    C12 — the team is the only wire between its threads. No thread the team
    spawns carries the runtime's ``list_threads`` / ``send_message``: the lead
    spawns with ``coordinator_tools_enabled=False`` and its composed hook pins
    it off whatever the lead's own hook returns, and every member adapter the
    package ships spawns its thread the same way (TB1 in
    :mod:`~ai_functions.experimental.teams.members`). The lead reaches a
    member only through the member's tool, so every ask it makes is a
    ``MemberCall`` under ``before_ask``, the deadline and the budget; members share only
    through what a hook hands them, so no member can make another member
    work. A ``Member`` written outside this package owns its own spawn and
    should turn the tools off the same way; a thread spawned outside a team
    keeps the runtime's default.

    C13 — a run's work ends with its answer, and its cost is totaled once
    everything has stopped. When the answer loop hands the answer back, or the
    run faults or is cancelled, the core tears the lead's thread down before
    any ``on_teardown`` runs, so no cycle starts on the lead after its answer
    is taken. After the retire gather, anything still live in the lead's
    subtree, a thread a hook or member spawned and nobody retired, is torn
    down; only then is the subtree's usage folded into ``Workspace.usage`` and
    ``TeamRun.cost`` and the budget's charge closed at the same figure (C10).
    A cycle that ended during teardown, and tokens a thread reported while it
    was being torn down, are therefore counted. ``on_closed`` hooks read the
    final figure. A member ask that stops the run (a ``StopRun`` from a
    ``before_ask`` or from the member) also ends the lead's in-flight cycle at
    that tool boundary, so the lead's model is not called again with the
    stopped ask's missing result.

    C14 — ``on_closed`` knows how the run ends. Before the first
    ``on_closed``, ``Workspace.fault`` is set to what the run is going to
    raise: the fault propagating out of the run, else the first error an
    ``on_teardown``, the retire or the usage fold raised (collected, and
    raised only once every ``on_closed`` has run), and an ``on_closed`` that
    raises is named to the hooks after it. A cancellation is not a fault
    (``Workspace.cancelled`` says it), so a cancelled run's ``fault`` is
    ``None`` and a run that answered and tore down cleanly has none either.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from pydantic_core import PydanticSerializationError
from strands.tools.decorator import tool as strands_tool

from ai_functions.ai_thread import AIFunction
from ai_functions.runtime import InMemoryCoordinator, LocalWorker, ThreadNotFoundError, last_event_id
from ai_functions.runtime.budget import BudgetError, Cost
from ai_functions.runtime.prices import PRICES, PriceTable
from ai_functions.types import ThreadSpawnedEvent, TokenUsage, TokenUsageEvent

from .contracts import Described, Equippable, Member, PartialTeamHook, Workspace
from .types import Accept, MemberCall, Note, Revise, Revision, RevisionCap, RunId, TeamError, TranscriptEntry

if TYPE_CHECKING:
    from strands.types.tools import AgentTool

    from ai_functions.ai_thread.config import ThreadKwargs
    from ai_functions.handle import ThreadHandle
    from ai_functions.protocols import Coordinator
    from ai_functions.runtime.budget import Budget, Reservation
    from ai_functions.types import EventId, ThreadContext, ThreadId


# ── The ask in flight ──


@dataclass(frozen=True)
class CallContext:
    """Which member ask is in flight: the run, the member and the ordinal of this ask to it (T9).

    Set by the core around every member ask and read back with
    :func:`current_call`, so a member that opens work of its own (a team run
    inside its ask) can name that work after the ask that caused it without
    any new argument on the ``Member`` protocol.
    """

    run_id: str
    member: str
    index: int
    _check: Callable[[], Awaitable[None]] | None = field(default=None, repr=False, compare=False, kw_only=True)

    async def check_active(self) -> None:
        """Re-run the originating ask's ``before_ask`` checks; raise ``StopRun`` if that run has stopped."""
        if self._check is not None:
            await self._check()

    @property
    def call_id(self) -> str:
        """The transcript's ``MemberCall.call_id`` for this ask: ``run_id:member:index``."""
        return f"{self.run_id}:{self.member}:{self.index}"


_CURRENT_CALL: ContextVar[CallContext | None] = ContextVar("ai_functions_teams_current_call", default=None)
"""The ask in flight on this task, or ``None`` outside a member ask."""


def current_call() -> CallContext | None:
    """The member ask in flight on the current task, or ``None`` when no team is asking anything here (T9).

    Returns:
        The :class:`CallContext` the core set around the ask, or ``None``.
    """
    return _CURRENT_CALL.get()


# ── Errors ──


class WiringError(TeamError):
    """The team was assembled wrong: a fault in the calling code, never in a model.

    Raised for the mistakes a run cannot recover from and must not paper over —
    two members colliding on the lead's wire, a hook whose ``on_answer`` returned
    something that is not a verdict, a member asked to take a second
    ``config_hook``. ``function_name`` names the lead or member concerned.
    """


class StopRun(asyncio.CancelledError):
    """Stop a live team at a work boundary and return a cancelled run record.

    Hooks may raise this from ``before_ask``. It inherits cancellation rather
    than ``Exception`` so model tool adapters cannot turn it into retry feedback.
    Use ``Team.cancel(reason)`` to stop a run from another task.
    """


# ── The report ──


class TeamRun(BaseModel):
    """Everything one run produced: the answer, the audit trail, and what hooks recorded.

    ``answer`` is ``Any`` because the lead's output type is the caller's choice.
    ``transcript`` carries the core's own record — one
    :class:`~ai_functions.experimental.teams.MemberCall` per member call, one
    :class:`~ai_functions.experimental.teams.Revision` per revise round, one
    :class:`~ai_functions.experimental.teams.RevisionCap` per exhausted budget —
    plus whatever :class:`~ai_functions.experimental.teams.Note` entries hooks
    appended. ``hooks_data`` is ``Workspace.data`` verbatim. ``blocked`` and
    ``blocked_by`` say whether a mandatory reviewer ran out of revisions without
    approving the answer, and which (C8); the two always agree, and a blocked
    run still carries the answer so the caller can read what was refused.
    ``run_id`` and ``lead_id`` name the run and the root of its event log.
    Every optional key serializes away when empty, false or at its default, so
    a bare run's artifact stays small and a hook-laden, cancelled or blocked
    run's says exactly what happened; ``model_dump_json()`` round-trips through
    ``model_validate_json()``. Frozen, because a report is evidence of what
    happened.
    """

    model_config = ConfigDict(frozen=True)

    answer: Any
    """What the lead returned after the answer loop; the lead's own output type."""
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    """The audit trail, in wire order: the core's records and the hooks' notes."""
    hooks_data: dict[str, Any] = Field(default_factory=dict)
    """``Workspace.data`` verbatim: whatever each hook published under its own key."""
    blocked: bool = False
    """True when a mandatory reviewer's cap ran out on the final pass: reviewed, never approved (C8)."""
    blocked_by: tuple[str, ...] = ()
    """Budget-stable labels of the mandatory reviewers that blocked the run, in hook order; empty when not blocked."""
    cancelled: bool = False
    """The run was stopped while live (``Team.cancel``, a ``StopRun`` from a hook, or the caller's task); no answer
    was approved."""
    cancel_reason: str | None = None
    """Why a live run stopped, when known."""
    cost: Cost | None = None
    """Measured cost of this run's subtree, totalled once every thread of it had stopped (C13): the tokens, and the
    dollars the coordinator's ``PriceTable`` put on each usage event by its model (C10), ``None`` when no event was
    priced, and without whatever an unpriced event cost."""
    run_id: str | None = None
    """The run's identity, the caller's or minted; ``None`` only on a report built by hand."""
    lead_id: str | None = None
    """The lead's thread id; ``coordinator.get_events(lead_id)`` reads the root of the run's event log."""

    @model_validator(mode="after")
    def _blocked_and_blocked_by_agree(self) -> TeamRun:
        if self.blocked and not self.blocked_by:
            raise ValueError(
                "TeamRun(blocked=True) names nobody in blocked_by; a blocked result says which "
                "mandatory reviewers blocked it. Pass blocked_by=(<hook label>, ...)."
            )
        if self.blocked_by and not self.blocked:
            raise ValueError(
                f"TeamRun(blocked_by={self.blocked_by!r}) names blockers but blocked is False; "
                f"pass blocked=True alongside blocked_by, or leave both at their defaults."
            )
        return self

    @model_serializer(mode="wrap")
    def _without_empty_keys(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if not self.transcript:
            data.pop("transcript", None)
        if not self.hooks_data:
            data.pop("hooks_data", None)
        if not self.blocked:
            data.pop("blocked", None)
            data.pop("blocked_by", None)
        if not self.cancelled:
            data.pop("cancelled", None)
        if self.cancel_reason is None:
            data.pop("cancel_reason", None)
        for key in ("run_id", "lead_id", "cost"):
            if getattr(self, key) is None:
                data.pop(key, None)
        return data


UNEQUIPPED = "unequipped"
"""The ``Note.kind`` naming a member that takes none of the hooks' member tools: it is not ``Equippable`` (C2)."""


def _unequipped(member: Member, contributors: Sequence[object]) -> Note:
    """The transcript entry for a member the ``tools_for_member`` fold skips: who, which hooks, and why (C2)."""
    return Note(
        kind=UNEQUIPPED,
        data={
            "member": member.name,
            "hooks": [type(h).__name__ for h in contributors],
            "reason": f"{type(member).__name__} is not Equippable, so it takes none of the member tools these "
            f"hooks contribute",
        },
    )


# ── The core ──


class Team:
    """A lead ``AIFunction`` over live members, extended by hooks and nothing else.

    Args:
        lead: The compiled function that answers the request. It runs on a live
            thread with every member available as a typed tool; its own
            ``tools=`` and ``config_hook`` are recomposed into the one hook the
            core installs, so a lead that already carries either loses nothing.
        members: The cast. Each is spawned as a child of the lead's thread
            before any hook runs and retired unconditionally in a ``finally``.
            Names must be unique — each becomes a tool named after it, and two
            tools sharing a name shadow silently.
        hooks: Objects implementing any of ``TeamHook``'s methods (``PartialTeamHook``), consulted in
            order everywhere order matters (request folding, tool contribution, the answer loop).
        ask_timeout_s: The deadline, in seconds, for one ask of any member whose
            own ``ask_timeout_s`` is ``None`` (C9). A member past it is retired,
            the lead reads ``error:`` text, and the member isn't asked again in
            the run. ``None`` (the default) bounds nothing.

    Raises:
        WiringError: Two members collide on the lead's wire (C5), or
            ``ask_timeout_s`` is not positive (C9).
    """

    def __init__(
        self,
        lead: AIFunction[..., Any],
        members: Sequence[Member],
        hooks: Sequence[PartialTeamHook] = (),
        *,
        ask_timeout_s: float | None = None,
    ) -> None:
        if ask_timeout_s is not None and ask_timeout_s <= 0:
            raise WiringError(
                f"Team(ask_timeout_s={ask_timeout_s}) is not positive; pass None for no team deadline, or a positive "
                f"number of seconds one member ask may take",
                function_name=lead.name,
            )
        self.lead: AIFunction[..., Any] = lead
        self.members: list[Member] = list(members)
        self.hooks: list[PartialTeamHook] = list(hooks)
        self.ask_timeout_s: float | None = ask_timeout_s
        self._check_no_duplicate_member_names(self.members)
        self._lock = asyncio.Lock()
        self._active: Workspace | None = None
        self._stop = asyncio.Event()
        self._boundary_error: Exception | None = None
        self._ledger: _AskLedger | None = None
        self._lost: set[str] = set()

    def cancel(self, reason: str = "cancelled") -> bool:
        """Request cancellation of this instance's live run, on its event loop.

        Returns false when no run is live. The first reason wins. The lead is
        interrupted, all members are retired, teardown hooks run, and the
        awaiting caller receives ``TeamRun(cancelled=True)``. Cancellation of
        the caller's own task still propagates normally.
        """
        if self._active is None:
            return False
        if not self._stop.is_set():
            self._active.cancel_reason = reason
            self._stop.set()
        return True

    async def run(
        self,
        request: str,
        coordinator: Coordinator | None = None,
        *,
        parent_id: ThreadId | None = None,
        run_id: str | None = None,
        budget: Budget | None = None,
    ) -> TeamRun:
        """One full run: spawn, hook, lead, answer loop, teardown — always all of it.

        Args:
            request: What the team is asked. Hooks may rewrite it (``on_request``)
                before the lead sees it; the lead's first positional parameter is
                where it lands, which binds for a ``STRUCTURED`` lead as much as
                for a ``STR_PROMPT`` one.
            coordinator: The coordinator every thread spawns onto. ``None``
                stands up a private ``InMemoryCoordinator`` + ``LocalWorker``
                pair for this run and closes it in a ``finally`` — the
                convenience path for scripts; anything embedded in a larger
                runtime passes its own so the run lands in the caller's
                registry and log.
            parent_id: Thread to attribute the lead to. The members are always
                children of the lead, so the whole run is one subtree whatever
                the caller passes.
            run_id: The run's identity when the caller owns it; ``None`` mints a
                fresh one (T4). Every ``MemberCall.call_id`` starts with it.
            budget: The :class:`~ai_functions.runtime.budget.Budget` the run
                is charged its whole measured spend on, tokens and priced
                dollars, and every member ask reserves against (C10).
                ``None`` (the default) reserves nothing.

        Returns:
            The answer, the transcript, and what hooks recorded. ``blocked`` is
            true, and ``blocked_by`` names the hooks, when a mandatory reviewer
            ran out of revisions without approving the final answer (C8);
            ``cancelled`` is true when the run was stopped while live.

        Raises:
            WiringError: A hook's ``on_answer`` returned something that is not a
                verdict (C4), a member could not take the hook slot (C2), a run
                is already in flight on this instance (C11), or ``budget``
                refuses unknown cost and a member declares no estimate (C10).
        """
        if self._lock.locked():
            raise WiringError(
                f"team {self.lead.name!r}: a run is already in flight on this Team instance, and a concurrent second "
                f"run would share the members' live handles and the hooks' per-run state; construct two Team "
                f"instances over the same lead and members, one per concurrent run",
                function_name=self.lead.name,
            )
        async with self._lock:
            if coordinator is None:
                private = InMemoryCoordinator()
                worker = LocalWorker(private)
                await worker.register()
                try:
                    return await self._run(
                        request,
                        private,
                        parent_id=parent_id,
                        run_id=run_id,
                        budget=budget,
                    )
                finally:
                    await worker.close()
            return await self._run(
                request,
                coordinator,
                parent_id=parent_id,
                run_id=run_id,
                budget=budget,
            )

    async def _run(self, request: str, coordinator: Coordinator, **kwargs: Any) -> TeamRun:
        """Clear the cancellation handle even if resource cleanup itself fails."""
        try:
            return await self._execute(request, coordinator, **kwargs)
        finally:
            self._active = None
            self._ledger = None

    async def _execute(
        self,
        request: str,
        coordinator: Coordinator,
        *,
        parent_id: ThreadId | None,
        run_id: str | None,
        budget: Budget | None,
    ) -> TeamRun:
        """The run proper, under :meth:`run`'s lock and on a coordinator that exists."""
        identity: dict[str, Any] = {} if run_id is None else {"run_id": RunId(run_id)}
        work = Workspace(
            team=self,
            request=request,
            coordinator=coordinator,
            members=self.members,
            parent_call=current_call(),
            **identity,
        )
        # Before anything spawns: a budget that refuses unknown cost refuses a member with no
        # estimate here, by name, rather than from the middle of the lead's first cycle (C10).
        ledger = _AskLedger(budget, self.members) if budget is not None else None
        self._ledger = ledger
        lead_handle: ThreadHandle[..., Any] | None = None
        baselines: dict[ThreadId, EventId | None] = {}
        usage_marker: EventId | None = None
        spend: _RunSpend | None = None
        self._active = work
        self._stop = asyncio.Event()
        self._boundary_error = None
        self._lost = set()
        outcome: tuple[Any, tuple[str, ...]] = (None, ())
        try:
            if work.parent_call is not None:
                await work.parent_call.check_active()

            # The lead's thread exists first — not running, just registered — so every member
            # spawns as its child and the run is one subtree in one event log. The lead's one
            # config_hook is composed before the spawn, closing over `work` so per-cycle tool
            # contributions see the same state every other hook method does.
            lead_fn = self.lead.replace(config_hook=self._lead_hook(work), coordinator_tools_enabled=False)
            lead_handle = await coordinator.spawn(lead_fn, thread_name=f"{self.lead.name}-lead", parent_id=parent_id)
            work.lead = lead_handle
            usage_marker = await last_event_id(coordinator, lead_handle.id)
            if ledger is not None:
                lead_id, marker = lead_handle.id, usage_marker
                ledger.attach(lambda: _run_spend(coordinator, lead_id, marker, baselines))

            self._equip_members(work)
            for member in self.members:
                spawned = await member.spawn(coordinator, parent_id=lead_handle.id)
                baselines[spawned.id] = await _baseline(coordinator, spawned.id)

            for hook in self.hooks:
                on_assemble = getattr(hook, "on_assemble", None)
                if on_assemble is not None:
                    await _maybe_await(on_assemble(work))

            for hook in self.hooks:
                on_request = getattr(hook, "on_request", None)
                if on_request is not None:
                    request = str(await _maybe_await(on_request(work, request)))

            answer = await self._run_lead(work, lead_handle, request)
            outcome = await self._answer_loop(work, lead_handle, answer)
            if self._stop.is_set():
                raise StopRun(work.cancel_reason)
        except StopRun as error:
            if self._boundary_error is not None:
                raise self._boundary_error from error
            work.cancelled = True
            work.cancel_reason = work.cancel_reason or str(error) or "cancelled"
        finally:
            # Teardown hooks run even on a mid-run fault, and the retire runs even when a
            # teardown hook raises — the registry must be empty on every path. One hook's
            # raise must not silence another's teardown, so each is guarded and the first
            # error resurfaces only when nothing else is already propagating.
            errors: list[BaseException] = []
            if isinstance(sys.exception(), asyncio.CancelledError):
                work.cancelled = True
                work.cancel_reason = work.cancel_reason or "caller task cancelled"
            if lead_handle is not None:
                try:
                    if work.cancelled:
                        await self._stop_subtree(coordinator, lead_handle.id)
                    # The lead's answer is taken, or the run is over: nothing more runs on the lead (C13).
                    await _retire_thread(lead_handle)
                except BaseException as error:
                    errors.append(error)
            for hook in self.hooks:
                on_teardown = getattr(hook, "on_teardown", None)
                if on_teardown is None:
                    continue
                try:
                    await _maybe_await(on_teardown(work))
                except BaseException as error:  # noqa: BLE001 — collected, re-raised below
                    errors.append(error)
            retired = await asyncio.gather(
                *(member.retire() for member in self.members),
                *([_retire_thread(lead_handle)] if lead_handle is not None else []),
                return_exceptions=True,
            )
            # A retire that raised is collected like a teardown error (C14); a cancellation is not a fault.
            errors.extend(result for result in retired if isinstance(result, Exception))
            # Every thread of the subtree has stopped before the run is totaled (C13): whatever a hook or a
            # member spawned and nobody retired is torn down first, so no cycle can land after the fold, and
            # the budget is charged the same total the report carries.
            if lead_handle is not None:
                try:
                    await self._stop_subtree(coordinator, lead_handle.id)
                    spend = await _run_spend(coordinator, lead_handle.id, usage_marker, baselines)
                    work.usage = spend.usage
                    if ledger is not None:
                        ledger.close(spend)
                except BaseException as error:
                    errors.append(error)
            # Members gone: same guarded-collect shape as on_teardown, into the same list, so a teardown
            # error still wins and one hook's raise never silences another's close. Each close is told
            # what the run is going to raise (C14), a teardown error it cannot see through sys.exception()
            # included.
            work.fault = _fault_of(sys.exception(), errors)
            for hook in self.hooks:
                on_closed = getattr(hook, "on_closed", None)
                if on_closed is None:
                    continue
                try:
                    await _maybe_await(on_closed(work))
                except BaseException as error:  # noqa: BLE001 — collected, re-raised below
                    errors.append(error)
                    work.fault = work.fault or _fault_of(None, [error])
            self._active = None
            if errors and sys.exception() is None:
                raise errors[0]
        answer, blocked_by = outcome
        return TeamRun(
            answer=answer,
            transcript=work.transcript,
            hooks_data=work.data,
            blocked=bool(blocked_by),
            blocked_by=blocked_by,
            cancelled=work.cancelled,
            cancel_reason=work.cancel_reason,
            cost=spend.cost if spend is not None else None,
            run_id=str(work.run_id),
            lead_id=str(lead_handle.id) if lead_handle is not None else None,
        )

    async def _stop_subtree(self, coordinator: Coordinator, root: ThreadId) -> None:
        """Tear down every live thread whose parent chain reaches ``root``, ``root`` included (C13).

        Lineage alone decides, read off each thread's ``ThreadInfo.parent_id``, so a thread spawned under the lead
        and not yet run is stopped with the rest. A ``THREAD_SPAWNED`` edge is never read: one that arrived in a
        copied log names a child of the thread the history came from, which this run does not own.
        """
        live = await coordinator.list_threads()
        parents = {info.thread_id: info.parent_id for info in live}
        under: dict[ThreadId, bool] = {root: True}

        async def reaches_root(thread_id: ThreadId) -> bool:
            chain: list[ThreadId] = []
            current: ThreadId | None = thread_id
            while current is not None and current not in under and current not in chain:
                chain.append(current)
                if current in parents:
                    current = parents[current]
                    continue
                try:
                    current = (await coordinator.get_thread_info(current)).parent_id
                except ThreadNotFoundError:
                    current = None
            found = current is not None and under.get(current, False)
            for link in chain:
                under[link] = found
            return found

        doomed = [info.thread_id for info in live if await reaches_root(info.thread_id)]
        await asyncio.gather(*(coordinator.terminate_now(tid) for tid in doomed), return_exceptions=True)

    async def _run_lead(self, work: Workspace, lead: ThreadHandle[..., Any], request: str) -> Any:
        """Race each lead cycle against explicit cancellation; never leave a worker running."""
        if self._stop.is_set():
            raise StopRun(work.cancel_reason)
        running = asyncio.ensure_future(lead.run(request))
        stopping = asyncio.create_task(self._stop.wait())
        try:
            await asyncio.wait((running, stopping), return_when=asyncio.FIRST_COMPLETED)
            if self._stop.is_set():
                await lead.terminate_now()
                raise StopRun(work.cancel_reason)
            return await running
        finally:
            for task in (running, stopping):
                if not task.done():
                    task.cancel()
            await asyncio.gather(running, stopping, return_exceptions=True)

    async def _before_ask(self, work: Workspace, member: Member) -> None:
        """Run every ``before_ask`` check before an ask, the lead's or a hook's, and the enclosing ask's checks."""
        try:
            if self._stop.is_set():
                raise StopRun(work.cancel_reason)
            if work.parent_call is not None:
                await work.parent_call.check_active()
            for hook in self.hooks:
                check = getattr(hook, "before_ask", None)
                if check is not None:
                    await _maybe_await(check(work, member))
        except StopRun as error:
            self.cancel(str(error) or "cancelled")
            raise
        except Exception as error:
            # Tool adapters may turn ordinary exceptions into feedback. A
            # failing check is an infrastructure fault, never a retryable ask.
            self._boundary_error = error
            self.cancel(f"ask guard failed: {type(error).__name__}")
            raise StopRun(self._active.cancel_reason if self._active else str(error)) from error

    async def _hook_ask(self, work: Workspace, member: Member, request: str) -> Any:
        """Typed asks made by hooks carry the same ``before_ask`` checks and cancellation as lead tool calls."""
        index = work.calls.get(member.name, 0) + 1
        work.calls[member.name] = index
        context = CallContext(str(work.run_id), member.name, index, _check=lambda: self._before_ask(work, member))
        token = _CURRENT_CALL.set(context)
        running = stopping = None
        try:
            await self._before_ask(work, member)
            running = asyncio.ensure_future(member.ask(request))
            stopping = asyncio.create_task(self._stop.wait())
            await asyncio.wait((running, stopping), return_when=asyncio.FIRST_COMPLETED)
            if self._stop.is_set():
                raise StopRun(work.cancel_reason)
            return await running
        except StopRun as error:
            self.cancel(str(error) or "cancelled")
            raise
        finally:
            tasks = [task for task in (running, stopping) if task is not None]
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            _CURRENT_CALL.reset(token)

    # ── The answer loop ──

    async def _answer_loop(
        self, work: Workspace, lead_handle: ThreadHandle[..., Any], answer: Any
    ) -> tuple[Any, tuple[str, ...]]:
        """Every reviewing hook grades in order; a ``Revise`` re-runs the lead and RESTARTS the walk.

        The walk restarts from the first reviewing hook, with per-hook budgets that persist.
        Restart-chain, not per-hook loops, because an earlier hook's ``Accept`` graded the
        answer as it stood *then*: when a later hook's ``Revise`` re-runs the lead, the mutated
        answer must go back through every prior gate or an accepted-then-mutated answer ships
        unreviewed. So the loop ends only when one full uninterrupted pass yields nothing but
        ``Accept`` and cap-exhausted pass-throughs (C7, NoUnreviewedShip; the rejected
        alternative, per-hook loops, ships an accepted-then-mutated answer).

        The rounds counter is per hook and persists across restarts — a restart re-consults
        a hook, it does not refill its budget — and the cap is read off the *latest* verdict,
        so a hook may lower (or keep raising, spending its own budget) mid-loop. Termination:
        every restart increments some hook's rounds, bounded by the sum of the caps in play.
        Cap exhaustion passes the answer through and records ``revise_cap`` once per hook; a
        verdict that is neither ``Accept`` nor ``Revise`` is a wiring bug and raises naming the
        hook, because a ``None`` silently treated as accept would grade nothing while looking
        like a review happened.

        The second value returned is who blocked the run (C8): the labels of the pass-throughs
        in the final pass whose latest verdict was mandatory, in hook order, empty when nobody
        did. It is rebuilt on every pass, because a pass that ends in a restart is not the pass
        that ships, and a hook exhausted on one draft may well approve the next.

        Returns:
            The answer as it stands after the final pass, and the blocking labels.
        """
        reviewers = [
            (label, on_answer)
            for label, hook in self._reviewer_labels()
            if (on_answer := getattr(hook, "on_answer", None)) is not None
        ]
        rounds: dict[str, int] = dict.fromkeys((label for label, _ in reviewers), 0)
        cap_recorded: set[str] = set()
        blocked_by: list[str] = []
        revised = True
        while revised:
            revised = False
            blocked_by = []
            for label, on_answer in reviewers:
                verdict = await _maybe_await(on_answer(work, answer))
                if isinstance(verdict, Accept):
                    continue
                if not isinstance(verdict, Revise):
                    raise WiringError(
                        f"{label}.on_answer returned {verdict!r}; it must return Accept() or "
                        f"Revise(feedback=...) — anything else silently reviewed nothing.",
                        function_name=self.lead.name,
                    )
                if rounds[label] >= verdict.cap:
                    if label not in cap_recorded:
                        cap_recorded.add(label)
                        work.transcript.append(RevisionCap(hook=label, rounds=rounds[label]))
                    if verdict.mandatory:
                        blocked_by.append(label)
                    continue
                rounds[label] += 1
                work.transcript.append(Revision(hook=label, round=rounds[label], feedback=verdict.feedback))
                answer = await self._run_lead(
                    work, lead_handle, f"Your answer was reviewed and needs revision.\n\nFeedback: {verdict.feedback}"
                )
                revised = True
                break
        return answer, tuple(blocked_by)

    def _reviewer_labels(self) -> list[tuple[str, PartialTeamHook]]:
        """Each hook under a budget-stable label: the type name, index-suffixed on collision.

        Two hooks of one class would otherwise share a label and merge their revise budgets.
        Single instances keep the clean type name so transcripts read as they always did;
        only genuine collisions grow a ``#n`` suffix (1-based, in hook order).
        """
        counts: dict[str, int] = {}
        for hook in self.hooks:
            name = type(hook).__name__
            counts[name] = counts.get(name, 0) + 1
        seen: dict[str, int] = {}
        labeled: list[tuple[str, PartialTeamHook]] = []
        for hook in self.hooks:
            name = type(hook).__name__
            seen[name] = seen.get(name, 0) + 1
            label = name if counts[name] == 1 else f"{name}#{seen[name]}"
            labeled.append((label, hook))
        return labeled

    # ── The tool composer ──

    def _lead_hook(self, work: Workspace) -> Callable[[ThreadContext], ThreadKwargs]:
        """The lead's one ``config_hook``, folding every tool source into one patch.

        The runtime calls exactly one hook per cycle and its ``tools`` patch replaces the
        compiled tools (``AIThread._run_cycle``; ``ThreadConfig.config_hook``; see the module
        docstring), so composition happens here or not at all. Order per cycle: the lead's own
        hook runs first and its full patch is honored — its ``tools``, when it sets them, stand
        in for the compiled ``tools=`` rather than stacking on top (the replace semantics the
        lead's author already wrote against); then the members-as-tools; then each hook's
        ``tools_for_lead``, rebuilt against that cycle's ``ctx``. The patch always turns the
        runtime's ``list_threads`` / ``send_message`` off, whatever the lead's own hook said (C12).
        """
        own_hook = self.lead.config.config_hook
        compiled_tools: list[Any] = list(self.lead.config.tools)

        def hook(ctx: ThreadContext) -> ThreadKwargs:
            patch: dict[str, Any] = {}
            tools = compiled_tools
            if own_hook is not None:
                patch = dict(own_hook(ctx))
                if "tools" in patch:
                    tools = list(patch["tools"] or [])
            contributed: list[AgentTool] = []
            for h in self.hooks:
                tools_for_lead = getattr(h, "tools_for_lead", None)
                if tools_for_lead is not None:
                    contributed.extend(tools_for_lead(work, ctx))
            patch["tools"] = [*tools, *self._member_tools(work), *contributed]
            patch["coordinator_tools_enabled"] = False
            return cast("ThreadKwargs", patch)

        return hook

    def _member_tools(self, work: Workspace) -> list[AgentTool]:
        """One tool per member, targeting the member's *live* thread.

        Named after the member, one ``request: str`` parameter — the whole of what
        ``Member.ask`` guarantees, so any member shape joins the lead's wire the same way. The
        call reaches the live thread, not a throwaway compile, so a member's history accrues
        across the lead's calls to it. Each call is recorded on the transcript by the wire that
        carried it, not reported by the model — and always under the member's *real* name,
        whatever the tool had to be called (C6).
        """
        return [self._member_tool(work, member) for member in self.members]

    def _member_tool(self, work: Workspace, member: Member) -> AgentTool:
        description = (
            f"Ask your team member {member.name} to do one piece of work and return its "
            f"answer. It does not see your conversation, so state everything it needs."
        )
        # A member that says what it is for exposes the ``Described`` seam (T8); a member without it, or with
        # ``None`` or blank, keeps the generic text byte for byte.
        if isinstance(member, Described):
            about = member.description
            if isinstance(about, str) and about.strip():
                description = f"{description} About {member.name}: {about.strip()}"

        @strands_tool(name=_tool_name(member.name), description=description)
        async def call(request: str) -> str:
            return await self._call_member(work, member, request)

        return cast("AgentTool", call)

    async def _call_member(self, work: Workspace, member: Member, request: str) -> str:
        """One checked, budgeted ask under a call context: what every member tool runs."""
        index = work.calls.get(member.name, 0) + 1
        work.calls[member.name] = index
        context = CallContext(str(work.run_id), member.name, index, _check=lambda: self._before_ask(work, member))
        call_id = context.call_id
        token = _CURRENT_CALL.set(context)
        try:
            await self._before_ask(work, member)
            answer = await self._ask(member, request)
        except StopRun as error:
            self.cancel(str(error) or "cancelled")
            await self._end_lead_cycle(work)
            raise
        except Exception as error:
            # The tool raises to the lead as a failed tool call; the transcript still records the ask (C6).
            failed = f"error: {type(error).__name__}: {error}"
            work.transcript.append(MemberCall(member=member.name, request=request, answer=failed, call_id=call_id))
            raise
        finally:
            _CURRENT_CALL.reset(token)
        work.transcript.append(MemberCall(member=member.name, request=request, answer=answer, call_id=call_id))
        return answer

    async def _end_lead_cycle(self, work: Workspace) -> None:
        """End the lead's in-flight cycle at this tool boundary, so it makes no model call after a stop (C13).

        A tool that raises hands the lead's model loop no result for its call, and the loop starts its next
        model call before the run's race in :meth:`_run_lead` tears the lead down. The runtime's cooperative
        cancel stops the cycle at that boundary instead; the teardown still runs as for any stop.
        """
        if work.lead is None:
            return
        with contextlib.suppress(ThreadNotFoundError):
            await work.lead.cancel()

    async def _ask(self, member: Member, request: str) -> str:
        """One member ask under the team deadline (C9) and the run's budget (C10).

        The member's own ``ask_timeout_s`` wins when it carries one: the team's applies only
        where that is ``None`` or absent. A refused reservation asks the member nothing; a
        deadline that passes retires the member first, so no orphan survives, then answers the
        lead with what happened, and a retired member is not asked again in the run. All three
        are model-fixable failures (the lead can wrap up with what it has, or ask for less), so
        they return as ``error:`` text rather than raising.
        """
        if member.name in self._lost:
            return (
                f"error: {member.name} was retired after it missed its deadline and can't be asked again in this "
                f"run. Finish with what the other members gave you."
            )
        ledger = self._ledger
        reservation: Reservation | None = None
        before = TokenUsage()
        if ledger is not None:
            try:
                reservation, before = await ledger.reserve(member)
            except BudgetError as error:
                return f"error: {member.name} was not asked; {error}"
        try:
            own = getattr(member, "ask_timeout_s", None)
            timeout = self.ask_timeout_s if own is None else own
            ask: Awaitable[Any] = member.ask(request)
            if timeout is not None:
                ask = asyncio.wait_for(ask, timeout)
            try:
                return _answer_text(await ask)
            except TimeoutError:
                if timeout is None:
                    raise
                await member.retire()
                self._lost.add(member.name)
                return (
                    f"error: {member.name} did not answer within {timeout:g}s and was retired; it can't be asked "
                    f"again in this run. Finish with what the other members gave you."
                )
        finally:
            if ledger is not None and reservation is not None:
                await ledger.settle(reservation, before)

    def _equip_members(self, work: Workspace) -> None:
        """Fold every ``tools_for_member`` into one equipped hook per member — before spawn.

        Skipped entirely when no hook contributes member tools, so the bare path spawns
        members exactly as their adapters would alone (and a member carrying its own
        ``config_hook=`` override is only refused when something actually needs the slot).
        Members without the :class:`~ai_functions.experimental.teams.Equippable` seam are
        skipped, not refused: a mixed cast is legitimate, and a member that cannot take tools
        is a fact rather than a fault. The skip is written down: each one is an ``unequipped``
        note on the transcript naming the member and the hooks whose tools it goes without.
        """
        contributors = [h for h in self.hooks if getattr(h, "tools_for_member", None) is not None]
        if not contributors:
            return
        for member in self.members:
            if not isinstance(member, Equippable):
                work.transcript.append(_unequipped(member, contributors))
                continue

            def hook(ctx: ThreadContext, member: Member = member) -> ThreadKwargs:
                tools: list[AgentTool] = []
                for h in contributors:
                    tools_for_member = getattr(h, "tools_for_member", None)
                    if tools_for_member is not None:
                        tools.extend(tools_for_member(work, member, ctx))
                return cast("ThreadKwargs", {"tools": tools})

            member.equip(hook)

    # ── Guards ──

    def _check_no_duplicate_member_names(self, members: Sequence[Member]) -> None:
        """Refuse two members under one name, at construction.

        Each member becomes a lead tool named ``member.name``, and two tools sharing a name
        shadow silently — the lead would reach one member believing it reached either. Refused
        before anything spawns (C5).
        """
        seen: dict[str, str] = {}
        duplicates: set[str] = set()
        for member in members:
            wire = _tool_name(member.name)
            if wire in seen:
                duplicates.add(f"{seen[wire]!r}/{member.name!r}")
            seen[wire] = member.name
        if duplicates:
            raise WiringError(
                f"team: members {sorted(duplicates)} collide on the lead's wire; members join "
                f"the lead as tools named after them (dots mapped to underscores for the tool "
                f"registry), so the later would shadow the earlier silently. Give them "
                f"distinct names.",
                function_name=self.lead.name,
            )

    def __repr__(self) -> str:
        """The lead's name, the members' names, and the hooks' type names."""
        return (
            f"<Team lead={self.lead.name!r} members={[m.name for m in self.members]!r} "
            f"hooks={[type(h).__name__ for h in self.hooks]!r}>"
        )


class _AskLedger:
    """The run's budget bookkeeping: one charge for everything the run measured, one hold per ask (C10).

    Built by ``Team.run`` when ``budget=`` is given. Construction is the refusal
    point for a budget that refuses unknown cost: every cast member is asked for
    its estimate, and one that declares none is a :class:`WiringError` naming
    it, raised before anything spawns. Every ask then reads the member's
    estimate again, so a member whose estimate follows its measured asks is
    held at its latest.

    The charge is settled at the run's measured spend, tokens and priced
    dollars, the fold ``TeamRun.cost`` reports, so every token any thread of
    the subtree spends is charged, the lead's own, a hook's ask and a nested
    team's included. It is refreshed around every ask, so a later ask is gated
    on what the run has spent so far on every axis the budget bounds, and
    closed at the final fold once every thread has stopped. Under a budget
    whose chain bounds dollars, usage the price table could not price refuses
    every later ask, since a dollar bound that cannot be measured is not
    enforced by carrying on.
    An ask's own reservation is the admission gate: it holds the member's
    estimate while the member works, then settles at nothing when the run's
    measured total moved during the ask (the tokens are in the charge), or at
    the estimate when nothing moved, the honest figure for a member whose work
    this coordinator cannot measure (a remote agent, a fake). Two asks in
    flight at once each see the other's tokens, so an unmeasurable one among
    them settles at nothing rather than its estimate.
    """

    def __init__(self, budget: Budget, members: Sequence[Member]) -> None:
        self.budget: Budget = budget
        self.estimates: dict[str, Cost | None] = {}
        for member in members:
            estimate = _estimate_of(member)
            if estimate is None and budget.on_unknown == "refuse":
                raise WiringError(
                    f"{member.name}: declares no ask estimate and budget {budget.path!r} refuses unknown cost "
                    f"(on_unknown='refuse'). Give the member an estimate_ask() returning a Cost, or build the "
                    f"budget with on_unknown='estimate' and a default_estimate.",
                    function_name=member.name,
                )
            self.estimates[member.name] = estimate
        self.charge: Reservation = budget.charge(Cost())
        self._measure: Callable[[], Awaitable[_RunSpend]] | None = None
        self._measured: _RunSpend = _RunSpend()

    def attach(self, measure: Callable[[], Awaitable[_RunSpend]]) -> None:
        """Measure the run with ``measure`` from here on; called once the lead's thread exists."""
        self._measure = measure

    async def refresh(self) -> TokenUsage:
        """Re-measure the run and settle the charge at its spend; the token total, for an ask to compare."""
        if self._measure is not None:
            self._measured = await self._measure()
            self.charge.settle(self._measured.charge)
        return self._measured.usage

    async def reserve(self, member: Member) -> tuple[Reservation, TokenUsage]:
        """One hold for one ask at the member's estimate, and the run's total before it.

        Raises:
            BudgetExhausted: The budget, counting what the run has spent so far, cannot hold the estimate.
            UnknownCost: The member declares no estimate and the budget refuses unknown cost.
            BudgetError: The budget's chain bounds dollars and the run spent usage the price table cannot price.
        """
        before = await self.refresh()
        bound = _dollars_bound(self.budget)
        if bound is not None and self._measured.unpriced:
            models = ", ".join(repr(model) for model in self._measured.unpriced)
            raise BudgetError(
                bound.path,
                f"budget {bound.path!r} bounds dollars, and this run spent usage the price table cannot price "
                f"(model {models}). Give the model a price (InMemoryCoordinator(prices=PRICES.with_prices(...))), "
                f"or bound the budget in tokens.",
            )
        # Read before every ask: a member's estimate may follow what its own asks have measured (members EA1).
        self.estimates[member.name] = _estimate_of(member)
        return self.budget.reserve(self.estimates[member.name]), before

    async def settle(self, reservation: Reservation, before: TokenUsage) -> None:
        """Release the ask's hold into the charge when the run's total moved; settle at the estimate otherwise."""
        after = await self.refresh()
        reservation.settle(Cost() if after != before else reservation.estimate)

    def close(self, spend: _RunSpend) -> None:
        """Settle the charge at the run's final fold, taken once every thread of the subtree has stopped (C13)."""
        self._measured = spend
        self.charge.settle(spend.charge)


def _dollars_bound(budget: Budget) -> Budget | None:
    """The nearest budget, from ``budget`` up to the root, whose limit sets dollars; ``None`` when none does."""
    current: Budget | None = budget
    while current is not None:
        if current.limit.dollars is not None:
            return current
        current = current.parent
    return None


def _estimate_of(member: Member) -> Cost | None:
    """The member's declared ask cost through the optional ``estimate_ask()`` seam, or ``None``."""
    estimate_ask = getattr(member, "estimate_ask", None)
    if estimate_ask is None:
        return None
    estimate = estimate_ask()
    if estimate is not None and not isinstance(estimate, Cost):
        raise WiringError(
            f"{member.name}.estimate_ask() returned {estimate!r}; it must return a Cost or None",
            function_name=member.name,
        )
    return estimate


@dataclass(frozen=True)
class _RunSpend:
    """What a run spent: its tokens, their priced dollars, and the models that had no price (C10)."""

    usage: TokenUsage = field(default_factory=TokenUsage)
    """The componentwise token total."""
    dollars: float | None = None
    """The sum of every priced event's dollars; ``None`` when no event was priced."""
    unpriced: tuple[str | None, ...] = ()
    """Each model, ``None`` for an event that named none, whose usage the price table could not price, once."""

    @property
    def cost(self) -> Cost:
        """The spend as ``TeamRun.cost`` reports it: the tokens, always measured, and the priced dollars."""
        return Cost.from_usage(self.usage) + Cost(dollars=self.dollars)

    @property
    def charge(self) -> Cost:
        """The spend as the budget is charged it: ``Cost()`` when nothing was measured, as a run that spent nothing."""
        return Cost() if self.usage == TokenUsage() and self.dollars is None else self.cost


async def _run_spend(
    coordinator: Coordinator,
    lead_id: ThreadId,
    marker: EventId | None,
    baselines: Mapping[ThreadId, EventId | None],
) -> _RunSpend:
    """This run's spend: the lead's log after ``marker`` and every cast thread's log after its own baseline.

    The fold descends ``THREAD_SPAWNED`` edges as
    :func:`~ai_functions.runtime.subtree_token_usage` does. A cast member is
    counted from the baseline captured right after its spawn, as the lead is
    from ``marker``, so anything a member's thread held before the run is
    behind the cursor; and every cast thread is folded whether or not an edge
    names it, so a member that spawned its thread somewhere other than under
    the lead is still counted.

    Each ``TOKEN_USAGE`` event is priced by the model it names through the
    coordinator's ``PriceTable`` (``InMemoryCoordinator.prices``, or
    :data:`~ai_functions.runtime.prices.PRICES` for a coordinator that carries
    none); an event the table cannot price adds no dollars and names its model
    in ``unpriced``.

    Args:
        coordinator: The coordinator holding the run's logs.
        lead_id: The lead's thread, the root of the run.
        marker: The lead's log position right after its spawn; ``None`` counts it whole.
        baselines: Each cast thread's log position right after its spawn.

    Returns:
        The run's tokens, priced dollars and unpriced models.
    """
    declared: object = getattr(coordinator, "prices", None)
    prices = declared if isinstance(declared, PriceTable) else PRICES
    total = TokenUsage()
    dollars: float | None = None
    unpriced: list[str | None] = []
    visited: set[ThreadId] = set()

    async def fold(thread_id: ThreadId, cursor: EventId | None) -> None:
        nonlocal total, dollars
        if thread_id in visited:
            return
        visited.add(thread_id)
        events = await coordinator.get_events(thread_id, since_id=cursor)
        for event in events:
            if isinstance(event, TokenUsageEvent):
                total = total + event.token_usage
                priced = prices.dollars(event.token_usage, event.model)
                if priced is not None:
                    dollars = priced if dollars is None else dollars + priced
                elif event.model not in unpriced:
                    unpriced.append(event.model)
            elif isinstance(event, ThreadSpawnedEvent):
                child = event.child_thread_id
                await fold(child, baselines.get(child))

    await fold(lead_id, marker)
    for thread_id, cursor in baselines.items():
        await fold(thread_id, cursor)
    return _RunSpend(usage=total, dollars=dollars, unpriced=tuple(unpriced))


async def _baseline(coordinator: Coordinator, thread_id: ThreadId) -> EventId | None:
    """The newest event on ``thread_id``'s log right after its spawn; ``None`` for an id that is no thread here.

    A member's ``spawn`` may return an id the coordinator never registered (a
    synthetic id for a member with no single thread, a fake); such a member
    has no log of its own to fold.
    """
    try:
        return await last_event_id(coordinator, thread_id)
    except ThreadNotFoundError:
        return None


def _fault_of(propagating: BaseException | None, errors: Sequence[BaseException]) -> BaseException | None:
    """The first of ``propagating`` and ``errors`` that is a fault rather than a cancellation (C14)."""
    for error in (propagating, *errors):
        if error is not None and not isinstance(error, asyncio.CancelledError):
            return error
    return None


async def _retire_thread(handle: ThreadHandle[..., Any]) -> None:
    """Tear ``handle``'s thread down now; one already gone is already retired (C13)."""
    try:
        await handle.terminate_now()
    except ThreadNotFoundError:
        pass


def _answer_text(answer: object) -> str:
    """What the lead reads for a member's answer: text as is, anything else as a Strands tool result formats it.

    A model is its JSON and any other value its ``json.dumps``, falling back to ``str`` when
    neither serializes, so a typed member reads to the lead as an ``AIFunction`` used as a tool
    does, and the transcript records the same text (C6).
    """
    if isinstance(answer, str):
        return answer
    if isinstance(answer, BaseModel):
        try:
            return answer.model_dump_json()
        except PydanticSerializationError:
            return str(answer)
    try:
        return json.dumps(answer, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(answer)


def _tool_name(name: str) -> str:
    r"""A member's name as a wire-legal tool name.

    Strands validates tool names against ``^[a-zA-Z0-9_\-]{1,}$`` and 64 chars, and a name
    that fails is *dropped from the registry with only a warning logged*, so the lead silently
    loses the member. Dotted names are the natural spelling for ``owner.method`` members, so the
    dot is mapped, not refused. The transcript keeps the real name; only the wire sees this one.
    """
    mapped = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
    return mapped[:64] or "member"


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` when a hook method chose to be async; pass it through otherwise."""
    if inspect.isawaitable(value):
        return await value
    return value
