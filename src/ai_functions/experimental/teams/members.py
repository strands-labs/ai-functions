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

import inspect
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from ai_functions.ai_thread import AIFunction
from ai_functions.experimental.methods import MethodAgent, MethodThread
from ai_functions.runtime import ThreadNotFoundError
from ai_functions.runtime.budget import Cost

from .core import WiringError, _run_spend

if TYPE_CHECKING:
    from ai_functions.ai_thread.config import ThreadKwargs
    from ai_functions.handle import ThreadHandle
    from ai_functions.protocols import Coordinator
    from ai_functions.types import EventId, ThreadContext, ThreadId


ASK_HISTORY: int = 8
"""How many of a member's latest measured asks ``estimate_ask`` reads (EA1)."""


class _AskMeter:
    """A member's declared ask cost and its measured ask history: what ``estimate_ask`` answers from (EA1, EA2).

    The history lives on the member object, so it spans the member's spawns and the runs it is cast in; a new
    member object starts from its declaration.
    """

    __slots__ = ("_costs", "_cursor", "_in_flight", "_serial", "_tainted", "_thread", "declared")

    def __init__(self, declared: Cost | None, name: str) -> None:
        if declared is not None and not isinstance(declared, Cost):
            raise WiringError(
                f"{name}: ask_estimate={declared!r} is not a Cost; pass Cost(tokens=..., dollars=...) for what one ask "
                f"is expected to spend, or None to leave it to the budget's on_unknown policy",
                function_name=name,
            )
        self.declared: Cost | None = declared
        self._costs: deque[Cost] = deque(maxlen=ASK_HISTORY)
        self._thread: ThreadId | None = None
        self._cursor: EventId | None = None
        self._serial: int = 0
        self._in_flight: set[int] = set()
        self._tainted: set[int] = set()

    @property
    def costs(self) -> tuple[Cost, ...]:
        """The measured asks the estimate reads, oldest first."""
        return tuple(self._costs)

    def estimate(self) -> Cost | None:
        """Per axis the costliest measured ask, an axis none measured from the declaration (EA1).

        The declaration alone until an ask has been measured.
        """
        if not self._costs:
            return self.declared
        tokens = max((cost.tokens for cost in self._costs if cost.tokens is not None), default=None)
        dollars = max((cost.dollars for cost in self._costs if cost.dollars is not None), default=None)
        if self.declared is not None:
            tokens = self.declared.tokens if tokens is None else tokens
            dollars = self.declared.dollars if dollars is None else dollars
        return Cost(tokens=tokens, dollars=dollars)

    async def measure[T](
        self, coordinator: Coordinator | None, thread_id: ThreadId, run: Callable[[], Awaitable[T]]
    ) -> T:
        """Run one ask and record what it spent on ``thread_id``'s subtree, when that can be told apart (EA2).

        Nothing is recorded for an ask that raised, measured no usage, or overlapped another ask on this member,
        since its window would hold the other's spend too. A coordinator that no longer knows the thread records
        nothing either; the ask's own answer or error is never touched.
        """
        if coordinator is None:
            return await run()
        start = await self._mark(coordinator, thread_id)
        self._serial += 1
        token = self._serial
        if self._in_flight:
            self._tainted.update((*self._in_flight, token))
        self._in_flight.add(token)
        try:
            answer = await run()
        except BaseException:
            self._tainted.discard(token)
            raise
        finally:
            self._in_flight.discard(token)
        if token in self._tainted:
            self._tainted.discard(token)
            return answer
        try:
            spend = await _run_spend(coordinator, thread_id, start, {})
        except ThreadNotFoundError:
            return answer
        measured = Cost.from_usage(spend.usage)
        if measured.tokens:
            self._costs.append(measured + Cost(dollars=None if spend.unpriced else spend.dollars))
        return answer

    async def _mark(self, coordinator: Coordinator, thread_id: ThreadId) -> EventId | None:
        """The newest event id on ``thread_id``'s log, read from the last mark on (a new thread is read whole once)."""
        if thread_id != self._thread:
            self._thread, self._cursor = thread_id, None
        try:
            events = await coordinator.get_events(thread_id, since_id=self._cursor)
        except ThreadNotFoundError:
            return self._cursor
        for event in reversed(events):
            event_id: EventId | None = getattr(event, "id", None)
            if event_id is not None:
                self._cursor = event_id
                break
        return self._cursor


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

    def __init__(
        self,
        function: AIFunction[..., Any],
        *,
        name: str | None = None,
        parameter: str | None = None,
        ask_estimate: Cost | None = None,
        **overrides: Any,
    ) -> None:
        self.function: AIFunction[..., Any] = function
        self.name: str = name or function.name
        self._parameter: str = _request_parameter(function, parameter, self.name)
        _refuse_peer_tools(overrides, self.name)
        self._overrides: dict[str, Any] = dict(overrides)
        self._equipped: Callable[[ThreadContext], ThreadKwargs] | None = None
        self._handle: ThreadHandle[..., Any] | None = None
        self._coordinator: Coordinator | None = None
        self._meter: _AskMeter = _AskMeter(ask_estimate, self.name)

    def estimate_ask(self) -> Cost | None:
        """What the next ask is expected to cost: the costliest recent measured ask, else ``ask_estimate`` (EA1)."""
        return self._meter.estimate()

    @property
    def ask_costs(self) -> tuple[Cost, ...]:
        """What this member's latest measured asks spent, oldest first; what :meth:`estimate_ask` reads (EA2)."""
        return self._meter.costs

    @property
    def handle(self) -> ThreadHandle[..., Any]:
        """The live thread's handle, for the runtime operations this adapter does not wrap.

        Raises:
            WiringError: The member has not been spawned, or has been retired.
        """
        if self._handle is None:
            raise WiringError(f"{self.name}: not spawned yet", function_name=self.name)
        return self._handle

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
        if self._overrides.get("config_hook") is not None or self.function.config.config_hook is not None:
            raise WiringError(
                f"{self.name}: this member already carries a config_hook, and the runtime "
                f"calls exactly one hook per cycle — compose them into one hook instead",
                function_name=self.name,
            )
        self._equipped = config_hook

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
        overrides = dict(self._overrides)
        if self._equipped is not None:
            own = overrides.get("tools")
            own_tools: list[Any] = list(own) if own is not None else list(self.function.config.tools)
            equipped = self._equipped

            def hook(ctx: ThreadContext) -> ThreadKwargs:
                patch: dict[str, Any] = dict(equipped(ctx))
                patch["tools"] = [*own_tools, *(patch.get("tools") or [])]
                return cast("ThreadKwargs", patch)

            overrides["config_hook"] = hook
        own_hook = overrides.get("config_hook", self.function.config.config_hook)
        if own_hook is not None:
            overrides["config_hook"] = _without_peer_tools(own_hook)
        target = self.function.replace(**overrides, coordinator_tools_enabled=False)
        self._handle = await coordinator.spawn(target, thread_name=self.name, parent_id=parent_id)
        self._coordinator = coordinator
        return self._handle

    async def ask(self, request: str) -> Any:
        """Run one cycle with ``request`` in the named parameter and return its result.

        Args:
            request: The whole of what the lead tells the member; it never sees the lead's conversation.

        Returns:
            The function's typed result.

        Raises:
            WiringError: The member has not been spawned.
        """
        handle = self.handle
        return await self._meter.measure(self._coordinator, handle.id, lambda: handle.run(**{self._parameter: request}))

    async def retire(self) -> None:
        """Tear the thread down; idempotent, quiet about a thread already gone (M4)."""
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            await handle.terminate_now()
        except ThreadNotFoundError:
            pass

    def __repr__(self) -> str:
        """The member's name and the function it wraps."""
        return f"<FunctionMember {self.name!r} function={self.function.name!r}>"


def _refuse_peer_tools(overrides: dict[str, Any], name: str) -> None:
    """Refuse an override that would hand a member's thread the runtime's peer tools (TB1).

    Raises:
        WiringError: ``overrides`` sets ``coordinator_tools_enabled`` to anything but ``False``.
    """
    if overrides.get("coordinator_tools_enabled", False) is not False:
        raise WiringError(
            f"{name}: coordinator_tools_enabled={overrides['coordinator_tools_enabled']!r} would hand this member "
            f"send_message and list_threads, and a team member is reached only through the lead's tool for it; "
            f"drop the override, and share between members through a team hook",
            function_name=name,
        )


def _without_peer_tools(hook: Callable[[ThreadContext], ThreadKwargs]) -> Callable[[ThreadContext], ThreadKwargs]:
    """``hook`` with ``coordinator_tools_enabled`` pinned off in every patch it returns (TB1).

    The runtime applies the one hook's patch over the thread's config each
    cycle, so a hook that switched the peer tools back on would undo the
    spawn-time setting; the pin makes the member's own hook unable to.
    """

    def pinned(ctx: ThreadContext) -> ThreadKwargs:
        patch = dict(hook(ctx))
        patch["coordinator_tools_enabled"] = False
        return cast("ThreadKwargs", patch)

    return pinned


def _request_parameter(function: AIFunction[..., Any], parameter: str | None, name: str) -> str:
    """The keyword a request lands in, or a refusal naming the fix.

    A function with no positional parameters cannot be briefed at all, and the
    failure without this check arrives as a ``TypeError`` from the signature
    bind *after* the team has been assembled — a wiring mistake reported from
    the middle of a run. A named parameter the function lacks fails the same
    way, so it is checked here too (M2).
    """
    parameters = inspect.signature(function.prompt_fn).parameters
    if parameter is not None:
        if parameter not in parameters:
            raise WiringError(
                f"{name}: parameter={parameter!r} is not a parameter of {function.name!r} "
                f"(it has {list(parameters)!r}); name one it has",
                function_name=name,
            )
        return parameter
    positional = [
        p.name
        for p in parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not positional:
        raise WiringError(
            f"{name}: {function.name!r} takes no positional parameter, so a request has nowhere to go; "
            f"give the function a parameter or pass parameter= naming a keyword one",
            function_name=name,
        )
    return positional[0]


# ── The MethodAgent-backed member ──


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

    def __init__(
        self,
        agent: MethodAgent,
        method: str,
        *,
        name: str | None = None,
        parameter: str | None = None,
        ask_estimate: Cost | None = None,
        **overrides: Any,
    ) -> None:
        self.agent: MethodAgent = agent
        self.method: str = method
        _refuse_peer_tools(overrides, name or method)
        self._overrides: dict[str, Any] = dict(overrides)
        self.function: AIFunction[..., Any] = agent.compiled(method, **self._overrides)
        self.name: str = name or self.function.config.name or f"{agent.name}_{method}"
        self._parameter: str = _request_parameter(self.function, parameter, self.name)
        self._equipped: Callable[[ThreadContext], ThreadKwargs] | None = None
        self._thread: MethodThread | None = None
        self._coordinator: Coordinator | None = None
        self._meter: _AskMeter = _AskMeter(ask_estimate, self.name)

    def estimate_ask(self) -> Cost | None:
        """What the next ask is expected to cost: the costliest recent measured ask, else ``ask_estimate`` (EA1)."""
        return self._meter.estimate()

    @property
    def ask_costs(self) -> tuple[Cost, ...]:
        """What this member's latest measured asks spent, oldest first; what :meth:`estimate_ask` reads (EA2)."""
        return self._meter.costs

    @property
    def thread(self) -> MethodThread:
        """The live ``MethodThread``, for the runtime operations this adapter does not wrap.

        Raises:
            WiringError: The member has not been spawned, or has been retired.
        """
        if self._thread is None:
            raise WiringError(f"{self.name}: not spawned yet", function_name=self.name)
        return self._thread

    @property
    def handle(self) -> ThreadHandle[..., Any]:
        """The live thread's raw handle: the same surface :attr:`FunctionMember.handle` exposes.

        Raises:
            WiringError: The member has not been spawned, or has been retired.
        """
        return self.thread.handle

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
        if self.function.config.config_hook is not None:
            raise WiringError(
                f"{self.name}: this member already carries a config_hook, and the runtime "
                f"calls exactly one hook per cycle — compose them into one hook instead",
                function_name=self.name,
            )
        self._equipped = config_hook

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
        overrides = dict(self._overrides)
        if self._equipped is not None:
            own_tools: list[Any] = list(self.function.config.tools)
            equipped = self._equipped

            def hook(ctx: ThreadContext) -> ThreadKwargs:
                patch: dict[str, Any] = dict(equipped(ctx))
                patch["tools"] = [*own_tools, *(patch.get("tools") or [])]
                return cast("ThreadKwargs", patch)

            overrides["config_hook"] = hook
        own_hook = overrides.get("config_hook", self.function.config.config_hook)
        if own_hook is not None:
            overrides["config_hook"] = _without_peer_tools(own_hook)
        overrides["coordinator_tools_enabled"] = False
        self._thread = await self.agent.spawn(
            self.method, coordinator, parent_id=parent_id, thread_name=self.name, **overrides
        )
        self._coordinator = coordinator
        return self._thread

    async def ask(self, request: str) -> Any:
        """Run one cycle with ``request`` in the named parameter and return its result.

        Args:
            request: The whole of what the lead tells the member; it never sees the lead's conversation.

        Returns:
            The method's typed result.

        Raises:
            WiringError: The member has not been spawned.
        """
        thread = self.thread
        return await self._meter.measure(self._coordinator, thread.id, lambda: thread.run(**{self._parameter: request}))

    async def retire(self) -> None:
        """Tear the thread down. Idempotent, and quiet about a thread already gone (M4).

        ``MethodThread.retire`` already swallows a thread the runtime tore down
        first; this clears the reference so a retired member reads as
        unspawned, exactly like :class:`FunctionMember`.
        """
        thread, self._thread = self._thread, None
        if thread is None:
            return
        await thread.retire()

    def __repr__(self) -> str:
        """The member's name, its agent's name and the method it wraps."""
        return f"<MethodMember {self.name!r} agent={self.agent.name!r} method={self.method!r}>"
