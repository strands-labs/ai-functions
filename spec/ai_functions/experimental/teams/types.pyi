"""Value types the teams contracts exchange: ids, verdicts, and transcript records.

The answer loop and the transcript are the two places a team's parties hand
each other values rather than calls. The loop's currency is a verdict — a hook
looks at the lead's answer and returns :class:`Accept` or :class:`Revise`, and
the core does the rest. The transcript's currency is a record — one
:class:`MemberCall` per question the lead put to a member, one
:class:`Revision` per re-run a hook demanded, one :class:`RevisionCap` when a
hook's budget ran out, and a :class:`Note` for whatever a hook wants
remembered. Every value here is frozen: a verdict or a record is evidence of
what happened, and evidence does not change after the fact.

``kind`` rides on every record so a reader can filter without ``isinstance``
and so the records survive a trip through JSON unambiguously. The core's three
kinds are ``member``, ``revise`` and ``revise_cap``.

Invariants:
    T2 — ``Accept`` and ``Revise`` are disjoint and exhaustive. ``Accept``
    refuses extra fields, so feedback cannot be smuggled into a pass;
    ``Revise`` requires ``feedback``.

    T3 — ``Revise.cap`` is never negative. Zero is legal and means "this
    feedback buys no rounds"; the guard exists because a negative cap would
    silently behave as zero. ``Revise.mandatory`` defaults off; a verdict
    that leaves it off exhausts its cap as a pass-through, one that sets it
    exhausts its cap as a block (core C8).

    T5 — a ``Note`` may not carry one of ``CORE_TRANSCRIPT_KINDS``; a
    ``Revision`` is round one or later; a ``RevisionCap`` records zero or more
    rounds spent. A ``MemberCall`` carries an optional ``call_id`` (the core's
    key for one member turn), which serializes away when unset.
"""

from __future__ import annotations

from typing import Any, Literal, NewType

from pydantic import BaseModel, ConfigDict

from ...ai_thread.errors import AIFunctionError

RunId = NewType("RunId", str)
"""Identity of one team run; what per-run state keys by when the workspace itself is out of reach."""

CORE_TRANSCRIPT_KINDS: frozenset[str]
"""The transcript kinds the core writes. A hook's ``Note`` may not reuse them (T5)."""

# ── The answer-loop verdicts ──

class Accept(BaseModel):
    """A hook's verdict: the answer passes; the loop moves to the next hook.

    Carries nothing and refuses anything offered — ``Accept(feedback=...)`` is
    a construction error, because a pass that quietly held feedback would look
    like a review that happened and change nothing (T2).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["accept"] = "accept"
    """Discriminator; lets a serialized verdict come back as the right type."""

class Revise(BaseModel):
    """A hook's verdict: re-run the lead with ``feedback``, at most ``cap`` times for this hook.

    ``cap`` rides on the verdict rather than on the hook because the hook is
    the party that knows how much revision this particular finding is worth —
    and a hook that returns ``Revise`` forever must still terminate, so the
    core reads the cap off the *latest* verdict and stops when the rounds
    already spent reach it. Exhaustion is not an error: the last answer passes
    on and the transcript records that the cap, not the hook, ended the loop
    (T3). ``mandatory`` changes what that pass-through means to the run, never
    whether it happens: a mandatory reviewer whose cap ran out marks the whole
    run blocked (``TeamRun.blocked``, core C8), so a result no mandatory
    reviewer ever approved is never presented as ordinary success. Like
    ``cap`` it is read off the latest verdict, and it defaults off, so a hook
    that says nothing about it keeps the pass-through behavior.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["revise"] = "revise"
    """Discriminator; lets a serialized verdict come back as the right type."""
    feedback: str
    """What the lead is told before it answers again."""
    cap: int = 2
    """How many re-runs this hook's findings may buy in total; read off the latest verdict."""
    mandatory: bool = False
    """Whether exhausting ``cap`` blocks the run instead of passing the answer through; read off the latest verdict."""

Verdict = Accept | Revise
"""What ``TeamHook.on_answer`` returns: exactly one of the two (T2)."""

# ── The transcript records ──

class MemberCall(BaseModel):
    """One question the lead put to a member, and the answer that came back.

    Written by the wire that carried the call, not reported by the model, and
    always under the member's *real* name whatever the tool had to be called.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["member"] = "member"
    """Discriminator; the string the transcript always used for member calls."""
    member: str
    """``Member.name`` of the member asked."""
    request: str
    """What the lead asked, verbatim."""
    answer: str
    """What the lead read: the member's text as is, a typed answer as its JSON, or ``error:`` and the exception
    when the ask raised."""
    call_id: str | None = None
    """The core's key for this turn: ``run_id:member:n``, ``n`` counting that member's asks from one (T5)."""

class Revision(BaseModel):
    """One re-run of the lead demanded by a hook's ``Revise`` verdict."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["revise"] = "revise"
    """Discriminator; the string the transcript always used for revise rounds."""
    hook: str
    """Budget-stable label of the hook that demanded the re-run."""
    round: int
    """Which of this hook's rounds this was, counting from one."""
    feedback: str
    """The ``Revise.feedback`` the lead was given."""

class RevisionCap(BaseModel):
    """A hook's revise budget ran out; the answer passed on unreviewed by it from here.

    Recorded once per hook per run, so a transcript says which party ended
    the loop — the hook accepting or the cap exhausting (T3).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["revise_cap"] = "revise_cap"
    """Discriminator; the string the transcript always used for cap exhaustion."""
    hook: str
    """Budget-stable label of the hook whose cap was reached."""
    rounds: int
    """Rounds the hook had spent when the cap stopped it."""

class Note(BaseModel):
    """A hook-authored transcript entry: any ``kind`` the core does not own, plus a payload.

    Hooks may append their own entries to the audit trail. They may not
    borrow the core's kinds, so a reader filtering on ``member`` or ``revise``
    sees only what the core wrote (T5).
    """

    model_config = ConfigDict(frozen=True)

    kind: str
    """The hook's own label for this entry; never one of ``CORE_TRANSCRIPT_KINDS``."""
    data: dict[str, Any] = ...
    """Whatever the hook wants remembered; refs and summaries, not raw model output."""

TranscriptEntry = MemberCall | Revision | RevisionCap | Note
"""What ``Workspace.transcript`` holds: the core's three records or a hook's ``Note`` (T5)."""

# ── Errors ──

class TeamError(AIFunctionError):
    """Base of the teams stack's failure modes; the core and its compositions add the leaves."""
