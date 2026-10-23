"""The seams of a team: what a member promises, what a hook may implement, what a run shares.

Deliberately protocols rather than base classes: the members worth having
already exist and already differ, and a hook implements the two methods it
needs and nothing else. The core (:mod:`.core`) says what a team *does* with a
member and a hook; this module says what shapes may *be* one. Only
:class:`Workspace` is concrete, because it is state, not behavior — the one
argument every hook method receives.

Everything here binds to this repository's runtime. A :class:`Member` spawns
onto an :class:`~ai_functions.protocols.Coordinator` under a
:class:`~ai_functions.types.ThreadId` parent and hands back a
:class:`Spawned`; the workspace's ``lead`` is a
:class:`~ai_functions.handle.ThreadHandle`; hook tools are rebuilt per cycle
against a :class:`~ai_functions.types.ThreadContext`.

Invariants:
    T1 — ``Member`` is a name and three verbs; ``retire`` is idempotent.

    T4 — ``Workspace`` has identity, not value equality, and a fresh
    ``run_id`` per construction. ``Workspace.calls`` counts each member's
    asks so the core can mint deterministic call ids.

    T6 — every ``TeamHook`` method is optional; the protocol is not runtime
    checkable on purpose, so ``isinstance`` cannot be used where ``getattr``
    is meant.

    T8 (the seam half) — ``Described`` is the one seam the lead's member-tool
    text reads: a member that says what it is for exposes ``description``, and
    the core appends it to the generic tool text; a member without the seam, or
    with ``None`` or blank, keeps the generic text byte for byte. The core reads
    the seam by ``isinstance``, never a bare attribute name.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from ai_functions.types import TokenUsage

from .types import RunId, TranscriptEntry, Verdict

if TYPE_CHECKING:
    from strands.types.tools import AgentTool

    from ai_functions.ai_thread.config import ThreadKwargs
    from ai_functions.handle import ThreadHandle
    from ai_functions.protocols import Coordinator
    from ai_functions.types import ThreadContext, ThreadId

    from .core import CallContext


# ── The member shape ──


@runtime_checkable
class Spawned(Protocol):
    """Synchronous methods and properties; calls finish before returning.

    What ``Member.spawn`` hands back: something the core can name by ``id``.

    The core reads only ``.id`` off a spawned member — to make it the parent
    of nothing and the child of the lead — so this is the whole of the
    promise. A :class:`~ai_functions.handle.ThreadHandle` satisfies it as
    written; so does a synthetic handle for a member with no single thread of its own.
    """

    @property
    def id(self) -> ThreadId:
        """The thread this member now lives on, or a synthetic id when no single thread does."""
        ...


@runtime_checkable
class Member(Protocol):
    """Async methods; await each operation. Sync providers use ``as_async``.

    A member: something with a name that spawns, answers once, and retires.

    Three verbs and nothing else, because that is the whole of what the core
    does to a member — it stands one up, asks it one question, and takes it
    down in the ``finally``. Anything richer would be a contract the library
    cannot honor for every member shape it wants to accept (T1).

    Deliberately a protocol rather than a base class: the members worth
    having already exist and already differ. A typed method on an agent
    adapts by holding the thread its own ``spawn`` returns and routing ``ask``
    to one parameter; a member of your own can run a whole team per ask. Each
    joins the lead's wire the same way — one ``request: str``, one answer.

    ``spawn`` returns a :class:`Spawned` rather than a concrete handle: the
    core reads only ``.id`` off it, and demanding a
    :class:`~ai_functions.handle.ThreadHandle` would exclude a member shape
    that has no single thread.

    The thread a member spawns should carry none of the runtime's
    ``list_threads`` / ``send_message``, because a member reachable around the
    lead's typed tool is work the team never sees (C12): the adapters in
    :mod:`~ai_functions.experimental.teams.members` spawn with them off (TB1).
    """

    name: str
    """Unique within a team; each member becomes a lead tool named after it."""

    async def spawn(self, coordinator: Coordinator, *, parent_id: ThreadId | None = None) -> Spawned:
        """Put this member on a live thread as a child of ``parent_id``.

        Args:
            coordinator: The coordinator every thread of this run spawns onto.
            parent_id: The lead's thread, so the run is one subtree in one event log.

        Returns:
            Something with an ``id``; the core reads nothing else off it.
        """
        ...

    async def ask(self, request: str) -> Any:
        """Run one cycle with ``request`` and return its result.

        Args:
            request: The whole of what the lead tells the member; it never sees the lead's conversation.

        Returns:
            Whatever the member produced; the lead reads text as is and anything else as its JSON (C6).
        """
        ...

    async def retire(self) -> None:
        """Tear the thread down. Idempotent — an unwind loop must not crash mid-unwind."""
        ...


@runtime_checkable
class Equippable(Protocol):
    """Synchronous methods and properties; calls finish before returning.

    The optional seam hook-contributed member tools ride in on.

    The core equips each member that exposes this with one composed
    ``config_hook`` between construction and spawn, without the caller
    writing any wiring. Members without the seam are skipped, not refused:
    a mixed cast is legitimate, and a member that cannot take tools is a fact
    rather than a fault. The fact is recorded: the run's transcript names each
    skipped member in an ``unequipped`` note (C2).
    """

    def equip(self, config_hook: Callable[[ThreadContext], ThreadKwargs]) -> None:
        """Attach a per-cycle ``config_hook``, applied at the next ``spawn``.

        Args:
            config_hook: Rebuilt every cycle against that cycle's context; its
                ``tools`` patch is what the runtime installs for the cycle.
        """
        ...


@runtime_checkable
class Described(Protocol):
    """Synchronous methods and properties; calls finish before returning.

    The optional seam the lead's member-tool text rides in on.

    A member that can say what it is for exposes ``description``; the core
    appends it to the generic "Ask your team member ..." text of the tool the
    lead calls it through, so the lead can pick between two members by what
    they do rather than by name alone. ``None`` or blank keeps the generic
    text unchanged. Members without the seam are skipped, not refused, like
    ``Equippable``: an in-process typed member is described by its signature
    and docstring already (T8).
    """

    description: str | None
    """What the lead reads about this member on its tool; ``None`` keeps the generic text."""


# ── What hooks see ──


@dataclass(eq=False, kw_only=True)
class Workspace:
    """One run's shared state, handed to every hook method.

    A single argument rather than a growing signature, so a new field here
    breaks no existing hook. ``data`` is the cross-hook scratch space the
    run's report publishes; ``transcript`` is the audit trail the core itself
    writes member calls and revise rounds into, and hooks may append their
    own :class:`~ai_functions.experimental.teams.types.Note` entries.
    ``members`` are live once ``on_assemble`` runs — spawning precedes every
    hook call. ``lead`` is the lead's live thread handle, set as soon as the
    lead's thread exists — before any hook runs — because a hook that opens a
    delivery channel to the lead must open it before the lead's first cycle.
    The lead's thread is registered but *not running* until after the
    ``on_request`` fold.

    Identity, not value: two workspaces are never equal unless they are the
    same object, so per-run hook state may key by the workspace itself, and
    every construction mints a fresh ``run_id`` for the places a reference
    cannot reach (T4).
    """

    team: Any
    """The composition that opened this run. Typed loosely because the state plane must not import it."""
    request: str
    """What the team was asked, before any ``on_request`` rewrite."""
    coordinator: Coordinator
    """The coordinator every thread of this run lives on."""
    members: Sequence[Member]
    """The cast, in team order; live from ``on_assemble`` onward."""
    run_id: RunId = field(default_factory=lambda: RunId(uuid4().hex))
    """Fresh per construction; the identity of this run in any journal."""
    data: dict[str, Any] = field(default_factory=dict)
    """Cross-hook scratch space; published verbatim as the run's ``hooks_data``."""
    transcript: list[TranscriptEntry] = field(default_factory=list)
    """The audit trail: what the wire carried, in order."""
    lead: ThreadHandle[..., Any] | None = None
    """The lead's live handle; set before any hook runs, ``None`` only before that."""
    calls: dict[str, int] = field(default_factory=dict)
    """How many times each member has been asked this run; what ``MemberCall.call_id`` counts from (T4)."""
    parent_call: CallContext | None = None
    """The enclosing team ask when this run was started inside one; its ``before_ask`` checks rerun before each ask."""
    cancelled: bool = False
    """Whether this run stopped through cancellation."""
    cancel_reason: str | None = None
    """First cancellation reason, available to teardown hooks."""
    usage: TokenUsage | None = None
    """Usage of this run's subtree; folded once every thread of it has stopped, after the retire gather and before
    ``on_closed`` (C13), so ``None`` while ``on_teardown`` runs."""
    fault: BaseException | None = None
    """The error the run is going to raise, set before ``on_closed`` (C14): what faulted the run, else the first
    teardown error, else an earlier hook's ``on_closed`` error; ``None`` for a run that answered or was cancelled.

    A hook settling a record at ``on_closed`` reads the outcome here: a teardown error does not propagate until
    every ``on_closed`` has run, so ``sys.exception()`` there cannot see it.
    """

    async def ask(self, member: Member, request: str) -> Any:
        """Ask from a hook through the run's ``before_ask`` checks and cancellation, preserving the typed answer.

        Hooks use this method for their own member asks. A direct ``member.ask``
        bypasses the team's ``before_ask`` checks and cancellation. The hook owns its transcript records.
        """
        return await self.team._hook_ask(self, member, request)


class TeamHook(Protocol):
    """What a hook may implement. Every method is optional — absence means skip.

    Checked with ``getattr(hook, name, None)`` rather than ``isinstance``,
    deliberately: a hook implements the two methods it needs and nothing
    else, and an introspection-safe absence check is what lets a debugger's
    ``hasattr`` probe never detonate a guard. This protocol is therefore *not*
    ``runtime_checkable`` — an ``isinstance`` against it raises, which is the
    point (T6). A protocol cannot mark a method optional, so a two-method hook
    does not satisfy this one; ``Team(hooks=...)`` takes :data:`PartialTeamHook`,
    one protocol per method here, and this class documents the full shape.

    ``tools_for_lead`` and ``tools_for_member`` are **synchronous**, because
    the runtime calls the one ``config_hook`` synchronously inside a cycle and
    documents "must not block". The other six methods (``on_assemble``,
    ``on_request``, ``on_answer``, ``on_teardown``, ``on_closed``,
    ``before_ask``) may be sync or async and the core awaits whichever it
    finds.

    Three conventions a concrete hook should keep: per-run state resets on
    ``Workspace`` identity, so one hook instance may serve many runs; a
    model-fixable failure returns as ``"error: ..."`` text through the
    ordinary channel, never raises, and never reads as approval; a
    persistence failure raises.
    """

    def on_assemble(self, work: Workspace) -> Any:
        """Members are live; the lead has not run. Hooks that brief members start here."""
        ...

    def on_request(self, work: Workspace, request: str) -> str | Awaitable[str]:
        """Rewrite the request the lead will be asked. Folded left across hooks in order."""
        ...

    def tools_for_lead(self, work: Workspace, ctx: ThreadContext) -> Sequence[AgentTool]:
        """Extra tools for the lead's cycle, rebuilt every cycle against ``ctx``. Sync."""
        ...

    def tools_for_member(self, work: Workspace, member: Member, ctx: ThreadContext) -> Sequence[AgentTool]:
        """Extra tools for one member's cycle, rebuilt every cycle against ``ctx``. Sync."""
        ...

    def on_answer(self, work: Workspace, answer: Any) -> Verdict | Awaitable[Verdict]:
        """Review the lead's answer. ``Accept`` passes it on; ``Revise`` re-runs the lead."""
        ...

    def on_teardown(self, work: Workspace) -> Any:
        """The run is over (successfully or not); the cast is still live. Clean up."""
        ...

    def on_closed(self, work: Workspace) -> Any:
        """Every member and the lead have been retired; nothing of the run is live. Release what they used.

        The step after ``on_teardown``: a hook that closes a resource a member
        held open does it here and not in ``on_teardown``, where the member is
        still alive. ``work.usage`` holds the run's final total and
        ``work.fault`` what the run is going to raise (C13, C14).
        """
        ...

    def before_ask(self, work: Workspace, member: Member) -> Any:
        """Sync or awaitable check before every member ask, the lead's and a hook's.

        Raise ``StopRun`` to end the live run with a cancelled report.
        Persistence errors must propagate; they never authorize an ask.
        """
        ...


# ── One protocol per hook method: what ``Team(hooks=...)`` accepts ──
#
# A protocol cannot mark a member optional, so a class implementing two of TeamHook's methods is not a
# TeamHook to a type checker. Each method below is its own protocol, and ``PartialTeamHook`` is their union: a
# hook is any object with at least one of them. The checker verifies the one method that makes the object match,
# not the hook's other methods.


class _OnAssemble(Protocol):
    def on_assemble(self, work: Workspace) -> Any: ...


class _OnRequest(Protocol):
    def on_request(self, work: Workspace, request: str) -> str | Awaitable[str]: ...


class _ToolsForLead(Protocol):
    def tools_for_lead(self, work: Workspace, ctx: ThreadContext) -> Sequence[AgentTool]: ...


class _ToolsForMember(Protocol):
    def tools_for_member(self, work: Workspace, member: Member, ctx: ThreadContext) -> Sequence[AgentTool]: ...


class _OnAnswer(Protocol):
    def on_answer(self, work: Workspace, answer: Any) -> Verdict | Awaitable[Verdict]: ...


class _OnTeardown(Protocol):
    def on_teardown(self, work: Workspace) -> Any: ...


class _OnClosed(Protocol):
    def on_closed(self, work: Workspace) -> Any: ...


class _BeforeAsk(Protocol):
    def before_ask(self, work: Workspace, member: Member) -> Any: ...


PartialTeamHook = (
    _OnAssemble | _OnRequest | _ToolsForLead | _ToolsForMember | _OnAnswer | _OnTeardown | _OnClosed | _BeforeAsk
)
"""What a team's ``hooks`` holds: an object implementing at least one :class:`TeamHook` method.

The core reaches every method with ``getattr`` and skips an absent one (T6), so a hook implements the methods it needs.
A union is satisfied by any one member: a type checker accepts the object once one of its methods matches the signature
:class:`TeamHook` documents and does not check its other methods, so a mistyped second method fails only at run time. An
object with none of the methods is rejected.
"""
