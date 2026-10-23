"""NoUnreviewedShip on teams core: invariant C7 of ``teams/core.py``.

``NoUnreviewedShip`` says a shipped answer is one every hook cleared at its current version,
and the rejected alternative — per-hook loops rather than a restart chain — ships an
accepted-then-mutated answer. In the code's terms: after any
``Revise`` by hook k, ``_answer_loop`` restarts the walk at hook 1 and re-consults hooks
1..k-1 on the revised answer before any ``Accept`` can ship it.
These tests pin the code to that on the hooks' own records of what they were shown.

The two-hook shape (A accepts, B revises, A re-sees) is pinned in ``test_teams_core.py``.
This file drives the three-hook shapes: a Revise from hook 3 that must
send the answer back through hooks 1 and 2, and a Revise from hook 2 that must re-consult
hook 1 while hook 3, which never graded the draft, sees only the revised answer.
"""

from __future__ import annotations

from typing import Any

from ai_functions.experimental.teams import Accept, Revise, Revision, Team, Workspace
from ai_functions.testing import RuntimeHarness

from .teams_wire import ruling, scripted_lead


class Witness:
    """Accepts everything and records every answer it was shown."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def on_answer(self, work: Workspace, answer: Any) -> Accept:
        self.seen.append(answer)
        return Accept()


class ReviseOnce:
    """Revises the first answer it sees, then accepts."""

    def __init__(self, feedback: str = "objection") -> None:
        self.feedback = feedback
        self.seen: list[Any] = []

    def on_answer(self, work: Workspace, answer: Any) -> Accept | Revise:
        self.seen.append(answer)
        return Revise(feedback=self.feedback) if len(self.seen) == 1 else Accept()


class TestNoUnreviewedShip:
    async def test_a_revise_from_hook_three_sends_the_revised_answer_back_through_hooks_one_and_two(self):
        """C7 NoUnreviewedShip, k = 3: hooks 1 and 2 accepted the draft; hook 3's Revise must
        not let the revised answer ship without hooks 1 and 2 grading it too."""
        one, two, three = Witness(), Witness(), ReviseOnce("hook three objects")
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["draft"]), ruling(cites=["revised"])])
            run = await Team(lead, [], hooks=[one, two, three]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 2, "one draft, one revision"
        assert any("hook three objects" in p for p in lead_model.prompts(1)), "the feedback reached the lead"
        assert run.answer.cites == ["revised"]
        assert [a.cites for a in one.seen] == [["draft"], ["revised"]], "hook 1 re-consulted on the revised answer"
        assert [a.cites for a in two.seen] == [["draft"], ["revised"]], "hook 2 re-consulted on the revised answer"
        assert [a.cites for a in three.seen] == [["draft"], ["revised"]], "hook 3 accepted what it asked for"
        revisions = [e for e in run.transcript if isinstance(e, Revision)]
        assert [(r.hook, r.round) for r in revisions] == [("ReviseOnce", 1)], "one restart, attributed to hook 3"

    async def test_a_revise_from_hook_two_re_consults_hook_one_and_hook_three_grades_only_the_revised_answer(self):
        """C7 NoUnreviewedShip, k = 2: hook 1 accepted the draft and hook 2 sent it back, so
        hook 1 must re-see the revised answer; hook 3 was never reached on the first pass (the walk
        breaks at the Revise), so it grades exactly one answer, the revised one, and never the draft
        an earlier gate already rejected."""
        one, two, three = Witness(), ReviseOnce("hook two objects"), Witness()
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["draft"]), ruling(cites=["revised"])])
            run = await Team(lead, [], hooks=[one, two, three]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 2, "one draft, one revision"
        assert run.answer.cites == ["revised"]
        assert [a.cites for a in one.seen] == [["draft"], ["revised"]], "hook 1 re-consulted on the revised answer"
        assert [a.cites for a in two.seen] == [["draft"], ["revised"]], "hook 2 accepted what it asked for"
        assert [a.cites for a in three.seen] == [["revised"]], "hook 3 saw only the answer that shipped"
        revisions = [e for e in run.transcript if isinstance(e, Revision)]
        assert [(r.hook, r.round) for r in revisions] == [("ReviseOnce", 1)], "one restart, attributed to hook 2"
