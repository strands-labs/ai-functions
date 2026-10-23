"""Teams: a lead over live members, with hooks for everything that is not that.

The core (:mod:`.core`) owns exactly what every team needs — spawn the
members, run the lead with the members as typed tools, review the answer
through a bounded loop, retire everybody — and every other capability arrives
as a hook. The contracts layer (:mod:`.contracts`, :mod:`.types`) is the
vocabulary those parties exchange and nothing else; :class:`Team` and its
:class:`TeamRun` report are the composition built on it.
:class:`FunctionMember` is the ``AIFunction``-backed member shape and
:class:`MethodMember` the ``MethodAgent``-backed one.

Four protocols and one state plane. :class:`Member` is what a member
promises: a name, and the three verbs the core performs on it — ``spawn``,
``ask``, ``retire``. :class:`TeamHook` is what a hook may implement; every
method is optional and absence means skip, so a team's ``hooks`` are typed
:data:`PartialTeamHook`, any object with at least one of those methods.
:class:`Equippable` is the optional seam a member exposes so hook-contributed
tools can ride into its cycles, and :class:`Described` the optional seam a
member's ``description`` reaches the lead through, as part of the tool the
lead calls it by. :class:`Workspace` is one run's shared state, handed to
every hook method as one argument so a new field breaks no existing hook.

The answer loop speaks in two verdicts. :class:`Accept` passes the answer on;
:class:`Revise` re-runs the lead with feedback under a cap the hook itself
chooses. What the loop and the members did is written to the transcript as
frozen records — a :class:`MemberCall` per question the lead asked a member,
a :class:`Revision` per re-run, a :class:`RevisionCap` when a hook's budget ran
out, a :class:`Note` for anything a hook wants remembered — each recorded by
the wire that carried the event, never reported by the model.

The protocols compose with this repository's runtime: ``spawn`` takes an
:class:`~ai_functions.protocols.Coordinator` and a
:class:`~ai_functions.types.ThreadId` parent, and the lead's handle on the
workspace is a :class:`~ai_functions.handle.ThreadHandle`. A run can be
bounded by a :class:`~ai_functions.runtime.budget.Budget` in tokens, dollars
or both.

Invariants:
    T1 — a member is a name and three verbs. ``spawn`` puts it on a live
    thread as a child of the given parent and returns something with an
    ``id``; ``ask`` runs one cycle and returns its result; ``retire`` tears the
    thread down and is idempotent, because an unwind loop must not crash
    mid-unwind.

    T2 — a verdict is exactly one of ``Accept`` or ``Revise``. ``Accept``
    carries nothing and refuses extra fields; ``Revise`` carries feedback and
    a cap. Anything else returned from ``on_answer`` is a wiring bug for the
    core to raise on, never a silent pass.

    T3 — ``Revise.cap`` is the hook's own budget for this finding, read off
    the latest verdict. A negative cap is a construction error because it
    would silently behave as zero. Exhaustion is not an error: the answer
    passes on and the transcript records that the cap, not the hook, ended
    the loop. ``Revise.mandatory`` (default off) makes that exhaustion a
    blocked ``TeamRun`` instead of ordinary success (core C8).

    T4 — a ``Workspace`` is one run and has identity, not value equality: two
    runs never share one, two workspaces built from the same inputs are not
    equal, and per-run state may key by the workspace itself or its
    ``run_id``. ``Workspace.calls`` counts each member's asks, so
    ``MemberCall.call_id`` is ``run_id:member:n``.

    T5 — transcript entries are frozen records with a ``kind``. The core owns
    ``member``, ``revise``, and ``revise_cap``; a hook's ``Note`` may not reuse
    them, so a reader filtering on kind never mistakes a hook's entry for the
    core's.

    T6 — every ``TeamHook`` method is optional, and presence is checked with
    ``getattr``, never ``isinstance``. ``tools_for_lead`` and
    ``tools_for_member`` are synchronous; the lifecycle methods may be sync or
    async and the core awaits whichever it finds.

    T7 — the contracts and core modules import nothing from any other
    experimental package; :mod:`.members` reaches sideways only into
    ``experimental/methods``. Methods never imports teams.

    T8 — a member's description reaches the lead through one seam.
    ``Described`` is the seam (``description: str | None``); the core reads it
    by ``isinstance`` and appends the text to the member tool's generic
    description, and a ``None`` or blank description, or a member without the
    seam, leaves the generic text byte for byte.

    T9 — the ask in flight is readable. The core sets a :class:`CallContext`
    around every member ask and :func:`current_call` reads it back on the
    same task, so a member that opens work of its own names that work after
    the ask that caused it without a new argument on ``Member``.
"""

from __future__ import annotations

from .contracts import Described, Equippable, Member, PartialTeamHook, Spawned, TeamHook, Workspace
from .core import CallContext, StopRun, Team, TeamRun, WiringError, current_call
from .members import FunctionMember, MethodMember
from .types import (
    CORE_TRANSCRIPT_KINDS,
    Accept,
    MemberCall,
    Note,
    Revise,
    Revision,
    RevisionCap,
    RunId,
    TeamError,
    TranscriptEntry,
    Verdict,
)

__all__ = [
    "Accept",
    "CORE_TRANSCRIPT_KINDS",
    "CallContext",
    "Described",
    "Equippable",
    "FunctionMember",
    "Member",
    "MemberCall",
    "MethodMember",
    "Note",
    "PartialTeamHook",
    "Revise",
    "Revision",
    "RevisionCap",
    "RunId",
    "Spawned",
    "StopRun",
    "Team",
    "TeamError",
    "TeamHook",
    "TeamRun",
    "TranscriptEntry",
    "Verdict",
    "WiringError",
    "Workspace",
    "current_call",
]
