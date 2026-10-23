"""Offline tests for mandatory review: ``Revise.mandatory`` and the blocked ``TeamRun`` (core C8).

Without ``mandatory`` an exhausted revision cap ends the loop as ordinary success: the transcript says
``revise_cap`` and the ``TeamRun`` looks like an approved run. These tests pin the rule for a *mandatory*
review: an exhausted one is a blocked result. ``TeamRun.blocked`` is true and ``blocked_by`` names every
mandatory reviewer whose cap-exhausted pass-through is part of the final pass. Non-mandatory reviewers keep
the pass-through, pinned here too, so the default path is unchanged.

Every delivery claim is checked on the wire (``Counting``), as in the neighboring files: the lead model's
call count says how many revisions actually ran, never the returned report alone.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_functions.experimental.teams import Accept, Revise, Revision, RevisionCap, Team, TeamRun, Workspace
from ai_functions.testing import RuntimeHarness

from .teams_wire import ruling, scripted_lead

# ── Hooks ──


class Blocking:
    """Revises every answer it sees, mandatorily, with a fixed cap: the blocked-result probe."""

    def __init__(self, feedback: str = "not good enough", cap: int = 2) -> None:
        self.feedback = feedback
        self.cap = cap

    def on_answer(self, work: Workspace, answer: Any) -> Revise:
        return Revise(feedback=self.feedback, cap=self.cap, mandatory=True)


class Advisory:
    """The same script without ``mandatory``: the default behavior, pinned."""

    def __init__(self, feedback: str = "could be better", cap: int = 2) -> None:
        self.feedback = feedback
        self.cap = cap

    def on_answer(self, work: Workspace, answer: Any) -> Revise:
        return Revise(feedback=self.feedback, cap=self.cap)


class ReviseThenAccept:
    """Mandatory on the first answer it sees, then accepts."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def on_answer(self, work: Workspace, answer: Any) -> Accept | Revise:
        self.seen.append(answer)
        return Revise(feedback="cite somebody", mandatory=True) if len(self.seen) == 1 else Accept()


# ══════════════════════════════════════════════════════════════════
# 1. The report fields
# ══════════════════════════════════════════════════════════════════


class TestTeamRunBlockedFields:
    def test_a_bare_run_is_not_blocked(self):
        """``TeamRun`` carries ``blocked`` and ``blocked_by``, with defaults that mean 'not blocked'."""
        run = TeamRun(answer="x")
        assert run.blocked is False
        assert run.blocked_by == ()

    def test_not_blocked_serializes_away_like_the_other_empty_keys(self):
        """A bare run's artifact stays one key; a blocked run's says who blocked it. Both round-trip through JSON."""
        assert set(TeamRun(answer="x").model_dump()) == {"answer"}
        blocked = TeamRun(answer="x", blocked=True, blocked_by=("Critic",))
        assert blocked.model_dump() == {"answer": "x", "blocked": True, "blocked_by": ("Critic",)}
        assert TeamRun.model_validate_json(blocked.model_dump_json()) == blocked

    def test_blocked_and_blocked_by_must_agree(self):
        """The guard, broken deliberately both ways: a blocked result names who blocked it, and a named blocker
        means blocked. Either half alone would be a report that contradicts itself."""
        with pytest.raises(ValueError, match="blocked_by"):
            TeamRun(answer="x", blocked=True)
        with pytest.raises(ValueError, match="blocked=True"):
            TeamRun(answer="x", blocked_by=("Critic",))


# ══════════════════════════════════════════════════════════════════
# 2. The answer loop
# ══════════════════════════════════════════════════════════════════


class TestMandatoryReview:
    async def test_an_exhausted_mandatory_reviewer_blocks_the_run_with_the_last_answer_attached(self):
        """C8: mandatory, always revising, cap 2. Exactly two revisions run (three lead calls), the transcript holds
        two ``Revision`` entries and one ``RevisionCap``, and the run comes back ``blocked=True`` naming the hook,
        with the LAST answer still attached so an orchestrator can read what was refused."""
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["a"]), ruling(cites=["b"]), ruling(cites=["c"])])
            run = await Team(lead, [], hooks=[Blocking(cap=2)]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 3, "draft + exactly cap revisions, the block buys no extra cycle"
        assert run.blocked is True
        assert run.blocked_by == ("Blocking",)
        assert run.answer.cites == ["c"], "the last answer rides on the blocked result"
        assert [e.kind for e in run.transcript] == ["revise", "revise", "revise_cap"]
        assert run.transcript[0] == Revision(hook="Blocking", round=1, feedback="not good enough")
        assert run.transcript[1] == Revision(hook="Blocking", round=2, feedback="not good enough")
        assert run.transcript[2] == RevisionCap(hook="Blocking", rounds=2)

    async def test_a_non_mandatory_reviewer_under_the_same_script_is_ordinary_success(self):
        """Pins the default behavior (C3, T3): the same script without ``mandatory`` exhausts its cap, records it, and
        the run is not blocked. The default path is unchanged."""
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["a"]), ruling(cites=["b"]), ruling(cites=["c"])])
            run = await Team(lead, [], hooks=[Advisory(cap=2)]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 3
        assert run.blocked is False
        assert run.blocked_by == ()
        assert [e.kind for e in run.transcript] == ["revise", "revise", "revise_cap"]
        assert "blocked" not in run.model_dump()

    async def test_two_exhausted_mandatory_reviewers_are_both_named_in_hook_order(self):
        """Two mandatory hooks of one class, cap 1 each: both exhaust, both are named under their budget-stable
        labels, in hook order."""
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["a"]), ruling(cites=["b"]), ruling(cites=["c"])])
            hooks = [Blocking("A objects", cap=1), Blocking("B objects", cap=1)]
            run = await Team(lead, [], hooks=hooks).run("go", h.coordinator)

        assert len(lead_model.contexts) == 3, "one revision apiece"
        assert run.blocked is True
        assert run.blocked_by == ("Blocking#1", "Blocking#2")
        assert [e.kind for e in run.transcript].count("revise_cap") == 2

    async def test_an_accept_after_a_revision_leaves_the_run_unblocked(self):
        """A mandatory reviewer that revises once and then accepts reviewed the final answer positively: nothing is
        exhausted, nothing is blocked, no ``RevisionCap`` is written."""
        async with RuntimeHarness() as h:
            hook = ReviseThenAccept()
            lead, lead_model = scripted_lead([ruling(cites=[]), ruling(cites=["left"])])
            run = await Team(lead, [], hooks=[hook]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 2
        assert run.blocked is False and run.blocked_by == ()
        assert run.answer.cites == ["left"]
        assert [e.kind for e in run.transcript] == ["revise"]

    async def test_a_mandatory_reviewer_with_cap_zero_blocks_on_sight(self):
        """cap=0 buys no rounds: one lead call, no ``Revision``, one ``RevisionCap(rounds=0)``, and the run is
        blocked because the reviewer never approved anything."""
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["a"])])
            run = await Team(lead, [], hooks=[Blocking(cap=0)]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 1
        assert run.blocked is True and run.blocked_by == ("Blocking",)
        assert run.transcript == [RevisionCap(hook="Blocking", rounds=0)]

    async def test_a_block_belongs_to_the_final_pass_so_a_later_accept_clears_it(self):
        """The block is a fact about the final pass (C7 and C8 together). Hook 1 is mandatory and exhausts on the
        draft; hook 2 revises; the walk restarts and hook 1, re-consulted on the revised answer, accepts it. The
        final answer was positively reviewed by every mandatory hook, so the run is not blocked, though the
        transcript still shows hook 1's cap ran out once."""

        class ExhaustThenAccept:
            def __init__(self) -> None:
                self.seen: list[Any] = []

            def on_answer(self, work: Workspace, answer: Any) -> Accept | Revise:
                self.seen.append(answer)
                if len(self.seen) == 1:
                    return Revise(feedback="no budget", cap=0, mandatory=True)
                return Accept()

        class ReviseOnce:
            def __init__(self) -> None:
                self.calls = 0

            def on_answer(self, work: Workspace, answer: Any) -> Accept | Revise:
                self.calls += 1
                return Revise(feedback="tighten it") if self.calls == 1 else Accept()

        async with RuntimeHarness() as h:
            first, second = ExhaustThenAccept(), ReviseOnce()
            lead, lead_model = scripted_lead([ruling(cites=["a"]), ruling(cites=["b"])])
            run = await Team(lead, [], hooks=[first, second]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 2
        assert [a.cites for a in first.seen] == [["a"], ["b"]], "re-consulted on the revised answer (C7)"
        assert run.blocked is False and run.blocked_by == ()
        assert [e.kind for e in run.transcript] == ["revise_cap", "revise"]

    async def test_mandatory_is_read_off_the_latest_verdict_like_the_cap(self):
        """A hook that turns mandatory mid-loop blocks; one that drops it does not. The core never caches the flag."""

        class TurnsMandatory:
            def __init__(self, *, first: bool, then: bool) -> None:
                self.flags = [first, then]
                self.calls = 0

            def on_answer(self, work: Workspace, answer: Any) -> Revise:
                flag = self.flags[min(self.calls, 1)]
                self.calls += 1
                return Revise(feedback="again", cap=1, mandatory=flag)

        async with RuntimeHarness() as h:
            lead, _ = scripted_lead([ruling(cites=["a"]), ruling(cites=["b"])])
            became = await Team(lead, [], hooks=[TurnsMandatory(first=False, then=True)]).run("go", h.coordinator)
            lead, _ = scripted_lead([ruling(cites=["a"]), ruling(cites=["b"])])
            dropped = await Team(lead, [], hooks=[TurnsMandatory(first=True, then=False)]).run("go", h.coordinator)

        assert became.blocked is True and became.blocked_by == ("TurnsMandatory",)
        assert dropped.blocked is False and dropped.blocked_by == ()
