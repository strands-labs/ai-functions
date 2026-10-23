"""The member shapes a team accepts: an ``AIFunction`` or a ``MethodAgent`` method adapted to ``Member``.

The ``Member`` protocol lives in
:mod:`~ai_functions.experimental.teams.contracts`; the adapters live here.
:class:`FunctionMember` adapts an :class:`~ai_functions.AIFunction`, the typed,
compiled function. :class:`MethodMember` adapts one ``@ai_method`` on a
:class:`~ai_functions.experimental.methods.MethodAgent`, so one capability of
a larger agent joins as a member. This module lives beside ``core.py`` rather
than inside it because it answers a different question: ``core.py`` says what a
team *does* with a member, this module says what shapes may *be* one.

Both adapters make the same promises, and the same conformance suite runs
against both (``tests/teams/test_teams_members_conformance.py``).

Invariants:
    M1 — a member's name is the caller's to choose. It defaults to the
    function's name (``FunctionMember``) or the compiled tool name ``{owner}_{method}``
    (``MethodMember``) and is overridable, because two members compiled from
    one function are legitimate and must not collide on the lead's wire.

    M2 — ``ask`` routes the whole request to exactly one named parameter,
    fixed at construction; a function with nowhere to put it, or a named
    parameter it does not have, is refused before anything spawns.

    M3 — one ``config_hook`` per member thread. ``equip`` refuses a member
    that already carries one, and ``spawn`` recomposes the member's own tools
    ahead of the equipped hook's, so a hook the member never asked about
    cannot cost it a tool.

    M4 — ``retire`` is idempotent, including against a thread somebody else
    already tore down; an unwind loop must not crash mid-unwind.

    M5 — a ``MethodMember`` spawns through its agent's own ``spawn`` (so a
    test that rebinds ``compiled`` on the instance scripts the live thread
    too), under the coordinator and parent the core passes, and the thread it
    holds is a ``MethodThread``; M1 to M4 hold for it exactly as for
    ``FunctionMember``.

    M6 — teams imports methods; methods never imports teams. This module is
    the only teams module that reaches into ``experimental/methods``.

    EA1 — every shipped member declares what its next ask costs through
    ``estimate_ask()``, the optional seam a budgeted ``Team.run`` reserves each
    ask at (C10 on the core, which reads it before every ask). Until the member
    has measured an ask it answers its ``ask_estimate`` (``None`` by default,
    which leaves the ask to the budget's ``on_unknown`` policy); once it has,
    it answers per axis the costliest of its last ``ASK_HISTORY`` measured
    asks, an axis none of them carried (dollars the price table could not
    price) taken from ``ask_estimate``. The costliest rather than the mean,
    because the reservation is an admission gate: a budget admits the next ask
    only when it could fund the costliest recent one, and a member's turns
    grow as its context does. A non-``Cost`` ``ask_estimate`` is refused at
    construction.

    EA2 — an ask is measured on the member's own thread: every
    ``TOKEN_USAGE`` its log gained during the ask and in the threads it
    spawned there (the fold ``TeamRun`` uses, C10), tokens summed and dollars
    priced by the coordinator's ``PriceTable``, the dollars left unmeasured
    when any event could not be priced. An ask that raised or timed out, that
    measured no usage (a scripted fake), or that overlapped another ask on the
    same member (its window holds the other's spend) is not recorded. The
    history lives on the member object, across its spawns and the runs it is
    cast in, and ``ask_costs`` shows it.

    TB1 — a member's thread carries no runtime peer tools. Both adapters here
    spawn their thread without the runtime's ``list_threads`` /
    ``send_message`` (C12 on the core): they spawn with
    ``coordinator_tools_enabled=False`` and pin it off in every patch the
    thread's one ``config_hook`` returns, whether the hook is the member's own
    or the equipped one. An override ``coordinator_tools_enabled=True`` is
    refused at construction, because a member reached around the lead's typed
    tool is a member the team cannot see, check, bound or charge.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ...ai_thread import AIFunction
from ...runtime.budget import Cost
from ..methods import MethodAgent, MethodThread

if TYPE_CHECKING:
    from ...ai_thread.config import ThreadKwargs
    from ...handle import ThreadHandle
    from ...protocols import Coordinator
    from ...types import ThreadContext, ThreadId

ASK_HISTORY: int
"""How many of a member's latest measured asks ``estimate_ask`` reads (EA1)."""

class FunctionMember:
    """An ``AIFunction`` as a ``Member``: the typed member, adapted once.

    Everything about a compiled function already works — ``coordinator.spawn``
    puts it on a live thread, ``ThreadHandle.run`` forwards typed keywords,
    ``terminate_now`` takes it down. The only thing missing is ``Member``'s
    ``ask(request: str)``, because a typed signature is not one string. So this
    adapter names *which* keyword the request arrives as, and that is the whole
    of it.

    Args:
        function: The compiled function that joins the team. Its own
            ``tools=`` and ``model=`` are honored; ``overrides`` win where
            they overlap.
        name: The member's name on the team, and its tool name on the lead's
            wire. Defaults to ``function.name``; pass one when two members
            share a function (M1).
        parameter: The keyword ``ask`` passes the request as. Defaults to the
            function's first positional parameter, which is right for the
            ordinary shape and explicit for the rest (M2).
        ask_estimate: What one ask is expected to cost, the reservation a
            budgeted run holds for it until this member has measured an ask of
            its own (EA1); ``None`` leaves an unmeasured member to the budget's
            ``on_unknown`` policy.
        overrides: ``ThreadConfig`` overrides for the spawn — ``model=`` is how
            a test scripts a member offline.

    ``notify()`` is the member's inbound side channel, and this adapter
    deliberately does not wrap it — a caller wanting it holds the
    :attr:`handle` this exposes.

    The thread spawns without the runtime's ``list_threads`` /
    ``send_message`` (TB1): the lead reaches it through its tool, and it
    reaches its teammates only through the tools a hook hands it.

    Raises:
        WiringError: ``function`` has no positional parameter to brief, or
            ``parameter`` names one it does not have (M2); ``overrides`` turns
            ``coordinator_tools_enabled`` on (TB1).
    """

    function: AIFunction[..., Any]
    name: str

    def __init__(
        self,
        function: AIFunction[..., Any],
        *,
        name: str | None = None,
        parameter: str | None = None,
        ask_estimate: Cost | None = None,
        **overrides: Any,
    ) -> None: ...
    def estimate_ask(self) -> Cost | None:
        """What the next ask is expected to cost: the costliest recent measured ask, else ``ask_estimate`` (EA1)."""
        ...
    @property
    def ask_costs(self) -> tuple[Cost, ...]:
        """What this member's latest measured asks spent, oldest first; what :meth:`estimate_ask` reads (EA2)."""
        ...

    @property
    def handle(self) -> ThreadHandle[..., Any]:
        """The live thread's handle, for the runtime operations this adapter does not wrap.

        Raises:
            WiringError: The member has not been spawned, or has been retired.
        """
        ...

    def equip(self, config_hook: Callable[[ThreadContext], ThreadKwargs]) -> None:
        """Attach a per-cycle ``config_hook``, applied at the next ``spawn``.

        The seam hook-contributed member tools ride in on: the core equips each
        member with one composed hook between construction and spawn, without
        the caller writing any wiring. Refused when the member already carries
        a ``config_hook`` — as a construction override or on the function's own
        config — because the runtime calls exactly one hook per cycle
        (``AIThread._run_cycle``; executable:
        ``tests/teams/test_teams_ai_thread_contract.py``), so silently dropping
        either side costs tools invisibly (M3). A *previously equipped* hook is
        replaced rather than refused: the equipped slot is team-owned, and a
        second run on the same member re-equips the same cast.

        The hook's ``tools`` patch replaces the compiled tools for the cycle
        (the merge semantics ``ThreadConfig.config_hook`` documents), so
        :meth:`spawn` composes the member's own tools back in ahead of whatever
        the hook adds — a member that carried tools must not lose them to a
        hook it never asked about.

        Args:
            config_hook: Rebuilt every cycle against that cycle's context; its
                ``tools`` patch is what the runtime installs for the cycle.

        Raises:
            WiringError: The member already carries its own ``config_hook``.
        """
        ...

    async def spawn(self, coordinator: Coordinator, *, parent_id: ThreadId | None = None) -> ThreadHandle[..., Any]:
        """Put the function on a live thread as a child of ``parent_id``.

        The thread spawns with ``coordinator_tools_enabled=False``, and its one
        ``config_hook``, when it has one, pins that off in every patch (TB1).

        Args:
            coordinator: The coordinator every thread of this run spawns onto.
            parent_id: The lead's thread, so the run is one subtree in one event log.

        Returns:
            The live thread's handle; the core reads only ``.id`` off it.
        """
        ...

    async def ask(self, request: str) -> Any:
        """Run one cycle with ``request`` in the named parameter and return its result.

        Args:
            request: The whole of what the lead tells the member; it never sees the lead's conversation.

        Returns:
            The function's typed result.

        Raises:
            WiringError: The member has not been spawned.
        """
        ...

    async def retire(self) -> None:
        """Tear the thread down; idempotent, quiet about a thread already gone (M4)."""
        ...

    def __repr__(self) -> str:
        """The member's name and the function it wraps."""
        ...

class MethodMember:
    """One ``@ai_method`` on a ``MethodAgent`` as a ``Member``: one capability of a larger agent, adapted once.

    The methods package already does everything but one thing — ``agent.spawn``
    puts a capability on a live :class:`~ai_functions.experimental.methods.MethodThread`,
    ``MethodThread.run`` forwards typed keywords, ``MethodThread.retire`` is
    idempotent against the runtime. What is missing is ``Member``'s
    ``ask(request: str)``, because a typed signature is not one string. So this
    adapter names *which* keyword the request arrives as, exactly as
    :class:`FunctionMember` does for a function, and equips the one ``config_hook`` the
    core composes (M3).

    Args:
        agent: The agent whose capability joins the team. One agent may
            contribute several members, because a thread hosts exactly one
            signature (``MethodThread``'s docstring).
        method: Which ``@ai_method`` to put on a thread.
        name: The member's name on the team, and its tool name on the lead's
            wire. Defaults to the compiled tool name — ``{owner}_{method}``, or
            an ``@ai_method(..., name=...)`` rename — so the lead's tool, the
            coordinator's ``thread_name`` and every error agree (M1).
        parameter: The keyword ``ask`` passes the request as. Defaults to the
            method's first positional parameter (M2).
        ask_estimate: What one ask is expected to cost until this member has
            measured an ask of its own (EA1); ``None`` leaves an unmeasured
            member to the budget's ``on_unknown`` policy.
        overrides: ``ThreadConfig`` overrides for the compilation — ``model=``
            is how a test scripts a member offline.

    The method is compiled once here, through ``agent.compiled``, and kept as
    :attr:`function`: that is what M2's parameter check reads, what
    :meth:`equip` reads the existing ``config_hook`` off, and what surfaces a
    name that is not an ``@ai_method`` before anything spawns. :meth:`spawn`
    compiles again through ``agent.spawn`` so a test that rebinds ``compiled``
    on the instance scripts the live thread too.

    ``notify()`` is the member's inbound side channel, and this adapter
    deliberately does not wrap it — a caller wanting it holds the
    :attr:`thread` this exposes.

    The thread spawns without the runtime's ``list_threads`` /
    ``send_message`` (TB1), whatever input shape the method compiles to.

    Raises:
        WiringError: The method has no positional parameter to brief, or
            ``parameter`` names one it does not have (M2); ``overrides`` turns
            ``coordinator_tools_enabled`` on (TB1).
        CompilationError: ``method`` is not an ``@ai_method`` on the agent.
    """

    agent: MethodAgent
    method: str
    function: AIFunction[..., Any]
    name: str

    def __init__(
        self,
        agent: MethodAgent,
        method: str,
        *,
        name: str | None = None,
        parameter: str | None = None,
        ask_estimate: Cost | None = None,
        **overrides: Any,
    ) -> None: ...
    def estimate_ask(self) -> Cost | None:
        """What the next ask is expected to cost: the costliest recent measured ask, else ``ask_estimate`` (EA1)."""
        ...
    @property
    def ask_costs(self) -> tuple[Cost, ...]:
        """What this member's latest measured asks spent, oldest first; what :meth:`estimate_ask` reads (EA2)."""
        ...

    @property
    def thread(self) -> MethodThread:
        """The live ``MethodThread``, for the runtime operations this adapter does not wrap.

        Raises:
            WiringError: The member has not been spawned, or has been retired.
        """
        ...

    @property
    def handle(self) -> ThreadHandle[..., Any]:
        """The live thread's raw handle: the same surface :attr:`FunctionMember.handle` exposes.

        Raises:
            WiringError: The member has not been spawned, or has been retired.
        """
        ...

    def equip(self, config_hook: Callable[[ThreadContext], ThreadKwargs]) -> None:
        """Attach a per-cycle ``config_hook``, applied at the next ``spawn``.

        The same seam :meth:`FunctionMember.equip` is, with the same refusal: the
        runtime calls exactly one hook per cycle (``AIThread._run_cycle``),
        so a member whose compiled config already carries one — from the
        decorator's config or from a construction override, both of which the
        compiled :attr:`function` reflects — is refused rather than silently
        losing a side (M3). A previously equipped hook is replaced: the
        equipped slot is team-owned.

        Args:
            config_hook: Rebuilt every cycle against that cycle's context; its
                ``tools`` patch is what the runtime installs for the cycle.

        Raises:
            WiringError: The member already carries its own ``config_hook``.
        """
        ...

    async def spawn(self, coordinator: Coordinator, *, parent_id: ThreadId | None = None) -> MethodThread:
        """Put the method on a live thread as a child of ``parent_id`` (M5).

        The hook's ``tools`` patch replaces the compiled tools for the cycle
        (``ThreadConfig.config_hook``), so the member's own tools — the
        decorator's or the construction override's, as compiled — are composed
        back in ahead of whatever the equipped hook adds (M3). The thread
        spawns with ``coordinator_tools_enabled=False``, and its one
        ``config_hook``, when it has one, pins that off in every patch (TB1).

        Args:
            coordinator: The coordinator every thread of this run spawns onto.
            parent_id: The lead's thread, so the run is one subtree in one event log.

        Returns:
            The live ``MethodThread``; the core reads only ``.id`` off it.
        """
        ...

    async def ask(self, request: str) -> Any:
        """Run one cycle with ``request`` in the named parameter and return its result.

        Args:
            request: The whole of what the lead tells the member; it never sees the lead's conversation.

        Returns:
            The method's typed result.

        Raises:
            WiringError: The member has not been spawned.
        """
        ...

    async def retire(self) -> None:
        """Tear the thread down. Idempotent, and quiet about a thread already gone (M4).

        ``MethodThread.retire`` already swallows a thread the runtime tore down
        first; this clears the reference so a retired member reads as
        unspawned, exactly like :class:`FunctionMember`.
        """
        ...

    def __repr__(self) -> str:
        """The member's name, its agent's name and the method it wraps."""
        ...
