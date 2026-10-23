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
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ...ai_thread import AIFunction
from ...runtime.budget import Budget, Cost
from .contracts import PartialTeamHook, Member
from .types import TeamError, TranscriptEntry

if TYPE_CHECKING:
    from ...protocols import Coordinator
    from ...types import ThreadId

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
        ...
    @property
    def call_id(self) -> str:
        """The transcript's ``MemberCall.call_id`` for this ask: ``run_id:member:index``."""
        ...

def current_call() -> CallContext | None:
    """The member ask in flight on the current task, or ``None`` when no team is asking anything here (T9).

    Returns:
        The :class:`CallContext` the core set around the ask, or ``None``.
    """
    ...

# ── Errors ──

class StopRun(asyncio.CancelledError):
    """Stop a live team at a work boundary and return a cancelled run record.

    Hooks may raise this from ``before_ask``. It inherits cancellation rather
    than ``Exception`` so model tool adapters cannot turn it into retry feedback.
    Use ``Team.cancel(reason)`` to stop a run from another task.
    """

class WiringError(TeamError):
    """The team was assembled wrong: a fault in the calling code, never in a model.

    Raised for the mistakes a run cannot recover from and must not paper over —
    two members colliding on the lead's wire, a hook whose ``on_answer`` returned
    something that is not a verdict, a member asked to take a second
    ``config_hook``. ``function_name`` names the lead or member concerned.
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
    transcript: list[TranscriptEntry] = ...
    """The audit trail, in wire order: the core's records and the hooks' notes."""
    hooks_data: dict[str, Any] = ...
    """``Workspace.data`` verbatim: whatever each hook published under its own key."""
    cancelled: bool = False
    cancel_reason: str | None = None
    cost: Cost | None = None
    blocked: bool = False
    """True when a mandatory reviewer's cap ran out on the final pass: reviewed, never approved (C8)."""
    blocked_by: tuple[str, ...] = ()
    """Budget-stable labels of the mandatory reviewers that blocked the run, in hook order; empty when not blocked."""
    run_id: str | None = None
    """The run's identity, the caller's or minted; ``None`` only on a report built by hand."""
    lead_id: str | None = None
    """The lead's thread id; ``coordinator.get_events(lead_id)`` reads the root of the run's event log."""

UNEQUIPPED: str
"""The ``Note.kind`` naming a member that takes none of the hooks' member tools: it is not ``Equippable`` (C2)."""

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

    lead: AIFunction[..., Any]
    members: list[Member]
    hooks: list[PartialTeamHook]
    ask_timeout_s: float | None

    def __init__(
        self,
        lead: AIFunction[..., Any],
        members: Sequence[Member],
        hooks: Sequence[PartialTeamHook] = ...,
        *,
        ask_timeout_s: float | None = None,
    ) -> None: ...
    def cancel(self, reason: str = "cancelled") -> bool:
        """Request cancellation of this instance's live run, on its event loop.

        Returns false when no run is live. The first reason wins. The lead is
        interrupted, all members are retired, teardown hooks run, and the
        awaiting caller receives ``TeamRun(cancelled=True)``. Cancellation of
        the caller's own task still propagates normally.
        """
        ...

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
        ...

    def __repr__(self) -> str:
        """The lead's name, the members' names, and the hooks' type names."""
        ...
