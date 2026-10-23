"""Offline tests for teams/core.py: the bare team, the composer, the answer loop, the unwind.

The lead is an ``AIFunction``, the members are ``FunctionMember`` (an ``AIFunction`` adapter) or a
``Spy``, and the transcript is typed records. Every delivery claim is asserted on the wire
(``Counting``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from strands.tools.decorator import tool as strands_tool

from ai_functions.experimental.teams import (
    Accept,
    Member,
    MemberCall,
    Note,
    Revise,
    Revision,
    RevisionCap,
    Team,
    TeamError,
    TeamRun,
    WiringError,
    Workspace,
)
from ai_functions.experimental.teams import FunctionMember as TeamMember
from ai_functions.handle import ThreadHandle
from ai_functions.testing import RuntimeHarness

from .teams_wire import Counting, Spy, call_member, read, reading, ruling, scripted_lead

# ══════════════════════════════════════════════════════════════════
# 1. The bare team
# ══════════════════════════════════════════════════════════════════


class TestBareTeam:
    async def test_lead_reaches_a_member_as_a_typed_tool_and_the_answer_is_ungraded(self):
        """The member is on the lead's wire under its own name, the lead calls it,
        the member's model runs, and the answer comes back with NO grading anywhere (C6)."""
        async with RuntimeHarness() as h:
            left_model = Counting([reading("left")])
            member = TeamMember(read, name="left-analyst.read", model=left_model)
            turns = [call_member("left-analyst_read", "what does left show"), ruling(admitted=False)]
            lead, lead_model = scripted_lead(turns)
            run = await Team(lead, [member]).run("who is right", h.coordinator)

        assert isinstance(run, TeamRun)
        assert run.answer.admitted is False, "returned ungraded: no oracle refused it"
        assert "left-analyst_read" in lead_model.tool_specs[0], "the member is a tool on the wire (dot mapped)"
        assert len(left_model.contexts) == 1, "the lead's tool call reached the member's model"
        assert any("what does left show" in p for p in left_model.prompts(0))
        member_calls = [e for e in run.transcript if e.kind == "member"]
        assert len(member_calls) == 1
        call = member_calls[0]
        assert isinstance(call, MemberCall), "the transcript holds typed records, not dicts"
        assert call.member == "left-analyst.read", "recorded under the real name, not the wire name"
        assert call.request == "what does left show"
        assert "seen" in call.answer

    async def test_a_typed_answer_reaches_the_lead_as_json_and_text_as_is(self):
        """A member's model answer is what any Strands tool returns for a model, its JSON, on the lead's wire
        and in the transcript alike; a text answer passes through unchanged (C6)."""
        async with RuntimeHarness() as h:
            member = TeamMember(read, name="left", model=Counting([reading("left", "seen")]))
            turns = [call_member("left"), call_member("plain"), ruling()]
            lead, lead_model = scripted_lead(turns)
            run = await Team(lead, [member, Spy("plain", answer="just text")]).run("who is right", h.coordinator)

        typed, plain = [e for e in run.transcript if isinstance(e, MemberCall)]
        assert json.loads(typed.answer) == {"source": "left", "detail": "seen"}, typed.answer
        assert plain.answer == "just text"
        assert lead_model.tool_results(2) == [typed.answer, "just text"], "the lead read what the transcript holds"

    async def test_no_hooks_costs_exactly_one_lead_cycle_and_no_member_cycles(self):
        """hooks=() adds zero model cycles; counted, not inferred. Empty keys serialize away, so a bare run's dump
        is its answer, its identity and its cost."""
        async with RuntimeHarness() as h:
            left_model = Counting([])
            member = TeamMember(read, name="left", model=left_model)
            lead, lead_model = scripted_lead([ruling()])
            run = await Team(lead, [member]).run("go", h.coordinator)

        assert run.answer.admitted is True
        assert len(lead_model.contexts) == 1, "one lead cycle, nothing else"
        assert left_model.contexts == [], "an unconsulted member never touched its model"
        assert run.transcript == [] and run.hooks_data == {}
        assert set(run.model_dump()) == {"answer", "run_id", "lead_id", "cost"}, "TeamRun._without_empty_keys"

    async def test_the_convenience_path_spawns_and_closes_its_own_coordinator(self):
        """Team.run(coordinator=None): a private InMemoryCoordinator + LocalWorker, closed in a finally."""
        lead, lead_model = scripted_lead([ruling()])
        run = await Team(lead, [Spy("watcher")]).run("go")
        assert run.answer.admitted is True
        assert len(lead_model.contexts) == 1

    async def test_a_second_run_of_the_same_team_composes_fresh(self):
        """Team.run composes the lead hook per run via AIFunction.replace: the template is never mutated,
        so the same Team runs twice and the lead's own config still carries no config_hook."""
        lead, lead_model = scripted_lead([ruling(cites=["one"]), ruling(cites=["two"])])
        team = Team(lead, [])
        async with RuntimeHarness() as h:
            first = await team.run("go", h.coordinator)
            second = await team.run("again", h.coordinator)
        assert (first.answer.cites, second.answer.cites) == (["one"], ["two"])
        assert lead.config.config_hook is None
        assert len(lead_model.contexts) == 2

    def test_repr_names_the_lead_the_members_and_the_hook_types(self):
        """Team.__repr__ names the lead, the members and the hook types."""
        lead, _ = scripted_lead([])

        class Quiet:
            def on_teardown(self, work: Workspace) -> None:
                pass

        assert repr(Team(lead, [Spy("a"), Spy("b")], hooks=[Quiet()])) == (
            "<Team lead='decide' members=['a', 'b'] hooks=['Quiet']>"
        )


# ══════════════════════════════════════════════════════════════════
# 2. Hooks: request folding, assembly order, the workspace
# ══════════════════════════════════════════════════════════════════


class TestHooks:
    async def test_on_request_folds_in_hook_order_and_the_result_reaches_the_lead_on_the_wire(self):
        """Two rewriting hooks compose left to right; the folded text is what the lead's
        model was actually asked."""

        class Prefixer:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def on_request(self, work: Workspace, request: str) -> str:
                return f"[{self.tag}] {request}"

        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling()])
            await Team(lead, [], hooks=[Prefixer("first"), Prefixer("second")]).run("THE-QUESTION", h.coordinator)

        assert any("[second] [first] THE-QUESTION" in p for p in lead_model.prompts(0))

    async def test_on_assemble_sees_live_members_and_runs_before_the_lead(self):
        """Members are spawned when on_assemble fires, the lead's model has not been called,
        and members are children of the lead's thread (C1)."""
        journal: list[str] = []

        class Observer:
            async def on_assemble(self, work: Workspace) -> None:
                watcher = work.members[0]
                assert isinstance(watcher, Spy)
                journal.append(f"assemble:{watcher.events[-1]}")
                journal.append(f"asked:{await watcher.ask('probe')}")

            def on_request(self, work: Workspace, request: str) -> str:
                journal.append("request")
                return request

        async with RuntimeHarness() as h:
            spy = Spy("watcher", answer="here")
            lead, _ = scripted_lead([ruling()])
            await Team(lead, [spy], hooks=[Observer()]).run("go", h.coordinator)

        assert journal == ["assemble:spawn", "asked:here", "request"]
        assert spy.parent_id is not None, "members are children of the lead's thread"
        assert spy.retirements == 1

    async def test_workspace_carries_the_team_the_live_lead_handle_and_the_subtree(self):
        """Workspace docstring: lead is set before any hook runs and is the lead's live ThreadHandle; team is
        the composition; the lead's ThreadInfo carries the caller's parent_id and each member's the lead's id."""
        seen: dict[str, Any] = {}

        class Witness:
            async def on_assemble(self, work: Workspace) -> None:
                assert work.lead is not None
                seen["lead"] = work.lead
                seen["team"] = work.team
                seen["lead_info"] = await work.coordinator.get_thread_info(work.lead.id)
                seen["members"] = list(work.members)

        async with RuntimeHarness() as h:
            spy = Spy("watcher")
            lead, _ = scripted_lead([ruling()])
            team = Team(lead, [spy], hooks=[Witness()])
            parent = await h.spawn(lead)
            await team.run("go", h.coordinator, parent_id=parent.id)

        assert isinstance(seen["lead"], ThreadHandle)
        assert seen["team"] is team
        assert seen["lead_info"].parent_id == parent.id
        assert seen["lead_info"].thread_name == "decide-lead"
        assert spy.parent_id == seen["lead"].id, "members spawn as children of the lead's thread"
        assert seen["members"] == [spy]

    async def test_async_and_sync_hook_methods_are_both_awaited(self):
        """The four lifecycle methods may be sync or async; the core awaits whichever it finds."""
        order: list[str] = []

        class Mixed:
            async def on_assemble(self, work: Workspace) -> None:
                order.append("assemble")

            async def on_request(self, work: Workspace, request: str) -> str:
                order.append("request")
                return request

            async def on_answer(self, work: Workspace, answer: Any) -> Accept:
                order.append("answer")
                return Accept()

            def on_teardown(self, work: Workspace) -> None:
                order.append("teardown")

        async with RuntimeHarness() as h:
            lead, _ = scripted_lead([ruling()])
            await Team(lead, [], hooks=[Mixed()]).run("go", h.coordinator)
        assert order == ["assemble", "request", "answer", "teardown"]

    async def test_a_hooks_note_and_data_survive_into_the_run(self):
        """Workspace: hooks append their own transcript entries and publish under data; both reach TeamRun."""

        class Journal:
            def on_answer(self, work: Workspace, answer: Any) -> Accept:
                work.transcript.append(Note(kind="journal", data={"cites": list(answer.cites)}))
                work.data["journal"] = {"reviewed": True}
                return Accept()

        async with RuntimeHarness() as h:
            lead, _ = scripted_lead([ruling(cites=["left"])])
            run = await Team(lead, [], hooks=[Journal()]).run("go", h.coordinator)

        assert run.transcript == [Note(kind="journal", data={"cites": ["left"]})]
        assert run.hooks_data == {"journal": {"reviewed": True}}
        assert set(run.model_dump()) == {"answer", "transcript", "hooks_data", "run_id", "lead_id", "cost"}


# ══════════════════════════════════════════════════════════════════
# 3. The composer: one hook, everything folded
# ══════════════════════════════════════════════════════════════════


class TestComposer:
    async def test_folds_lead_tools_member_tools_and_two_hooks_on_the_wire(self):
        """C2: the lead carries its own tools= AND its own config_hook, two hooks each
        contribute a tool, one member joins — every one is in the tool_specs the lead's model received on one cycle.
        The runtime calls exactly one hook (``AIThread._run_cycle``) and its patch REPLACES tools
        (``ThreadConfig.config_hook``), so this passing means the core's fold really is the one hook."""

        @strands_tool(name="own_tool", description="the lead's own compiled tool")
        def own_tool(x: str) -> str:
            return x

        @strands_tool(name="hooked_tool", description="from the lead's own config_hook")
        def hooked_tool(x: str) -> str:
            return x

        def own_hook(ctx: Any) -> dict[str, Any]:
            return {"tools": [own_tool, hooked_tool]}

        class Contributor:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            def tools_for_lead(self, work: Workspace, ctx: Any) -> list[Any]:
                @strands_tool(name=f"{self.tag}_tool", description=f"from hook {self.tag}")
                def contributed(x: str) -> str:
                    return x

                return [contributed]

        async with RuntimeHarness() as h:
            member = TeamMember(read, name="left-analyst.read", model=Counting([]))
            lead, lead_model = scripted_lead([ruling()], config_hook=own_hook)
            await Team(lead, [member], hooks=[Contributor("alpha"), Contributor("beta")]).run("go", h.coordinator)

        specs = lead_model.tool_specs[0]
        for expected in ("own_tool", "hooked_tool", "left-analyst_read", "alpha_tool", "beta_tool"):
            assert expected in specs, f"{expected} missing from the lead's wire: {specs}"
        for runtime_tool in ("list_threads", "send_message"):
            assert runtime_tool not in specs, "C12: the lead reaches members only through their tools"

    async def test_the_leads_compiled_tools_survive_when_it_has_no_hook_of_its_own(self):
        """_lead_hook: with no own hook, the compiled tools= are the base the members and hooks stack on."""

        @strands_tool(name="compiled_tool", description="on the lead's config")
        def compiled_tool(x: str) -> str:
            return x

        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling()], tools=[compiled_tool])
            await Team(lead, [Spy("watcher")]).run("go", h.coordinator)

        assert {"compiled_tool", "watcher"} <= set(lead_model.tool_specs[0])

    async def test_recomposes_a_members_own_tools_with_hook_contributions(self):
        """M3: a hook's tools patch REPLACES the member's compiled tools, so the member's
        own tools= must be recomposed in ahead of what hooks add. Both are asserted from the member model's wire."""

        @strands_tool(name="members_own", description="the member carried this itself")
        def members_own(x: str) -> str:
            return x

        class MemberContributor:
            def tools_for_member(self, work: Workspace, member: Member, ctx: Any) -> list[Any]:
                @strands_tool(name="hook_gave", description="a hook contributed this")
                def hook_gave(x: str) -> str:
                    return x

                return [hook_gave]

        async with RuntimeHarness() as h:
            member_model = Counting([reading("left")])
            member = TeamMember(read, name="left", model=member_model, tools=[members_own])
            lead, _ = scripted_lead([call_member("left"), ruling()])
            await Team(lead, [member], hooks=[MemberContributor()]).run("go", h.coordinator)

        specs = member_model.tool_specs[0]
        assert "members_own" in specs, f"the member's own tool was lost to the hook: {specs}"
        assert "hook_gave" in specs, f"the hook's tool never reached the member: {specs}"

    async def test_the_lead_carries_no_runtime_tools_even_when_its_own_hook_turns_them_on(self):
        """C12: the composed hook pins ``coordinator_tools_enabled`` off over whatever the lead's own hook returns."""

        def own_hook(ctx: Any) -> dict[str, Any]:
            return {"coordinator_tools_enabled": True}

        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling()], config_hook=own_hook)
            await Team(lead, [Spy("watcher")]).run("go", h.coordinator)

        specs = lead_model.tool_specs[0]
        assert "watcher" in specs
        assert {"list_threads", "send_message"}.isdisjoint(specs), specs

    async def test_a_thread_spawned_outside_the_team_keeps_the_runtime_tools(self):
        """C12 reaches only the threads a team manages: a plain thread on the same coordinator keeps both tools."""
        outside_model = Counting([reading()])
        member_model = Counting([reading()])
        async with RuntimeHarness() as h:
            outside = await h.coordinator.spawn(read.replace(model=outside_model), thread_name="outside")
            lead, _ = scripted_lead([call_member("left"), ruling()])
            await Team(lead, [TeamMember(read, name="left", model=member_model)]).run("go", h.coordinator)
            await outside.run(focus="the horizon")

        assert {"list_threads", "send_message"} <= set(outside_model.tool_specs[0])
        assert {"list_threads", "send_message"}.isdisjoint(member_model.tool_specs[0])

    async def test_a_member_without_the_equip_seam_is_skipped_not_refused(self):
        """_equip_members / B1 Equippable: a mixed cast is legitimate; the Spy has no equip, the run passes."""

        class MemberContributor:
            def tools_for_member(self, work: Workspace, member: Member, ctx: Any) -> list[Any]:
                return []

        async with RuntimeHarness() as h:
            lead, _ = scripted_lead([ruling()])
            run = await Team(lead, [Spy("plain")], hooks=[MemberContributor()]).run("go", h.coordinator)
        assert run.answer.admitted is True


# ══════════════════════════════════════════════════════════════════
# 4. The answer loop
# ══════════════════════════════════════════════════════════════════


class OneRevision:
    """Revises once with fixed feedback, then accepts whatever comes back."""

    def __init__(self, feedback: str = "cite somebody") -> None:
        self.feedback = feedback
        self.seen: list[Any] = []

    def on_answer(self, work: Workspace, answer: Any) -> Accept | Revise:
        self.seen.append(answer)
        if len(self.seen) == 1:
            return Revise(feedback=self.feedback)
        return Accept()


class AcceptingWitness:
    """Accepts everything, recording every answer it was shown."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def on_answer(self, work: Workspace, answer: Any) -> Accept:
        self.seen.append(answer)
        return Accept()


class AlwaysRevise:
    """Revises every answer it sees, with a fixed cap — the budget-exhaustion probe."""

    def __init__(self, feedback: str, cap: int) -> None:
        self.feedback = feedback
        self.cap = cap

    def on_answer(self, work: Workspace, answer: Any) -> Revise:
        return Revise(feedback=self.feedback, cap=self.cap)


class TestAnswerLoop:
    async def test_revise_feedback_reaches_the_leads_next_context_on_the_wire(self):
        """Revise(feedback) re-runs the lead and the feedback text is IN the second call's
        context; the transcript holds one typed Revision."""
        async with RuntimeHarness() as h:
            hook = OneRevision("cite somebody, anybody")
            lead, lead_model = scripted_lead([ruling(cites=[]), ruling(cites=["left"])])
            run = await Team(lead, [], hooks=[hook]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 2, "one draft, one revision"
        assert any("cite somebody, anybody" in p for p in lead_model.prompts(1))
        assert run.answer.cites == ["left"], "the revised answer is the one returned"
        assert run.transcript == [Revision(hook="OneRevision", round=1, feedback="cite somebody, anybody")]
        assert hook.seen[0].cites == [] and hook.seen[1].cites == ["left"]

    async def test_accept_passes_straight_through_with_no_extra_lead_cycle(self):
        """Accept costs nothing and records nothing."""
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling()])
            run = await Team(lead, [], hooks=[AcceptingWitness()]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 1
        assert run.transcript == []

    async def test_cap_exhaustion_passes_the_last_answer_and_records_it(self):
        """C3 / T3: a hook that revises forever is bounded by its own cap: cap=2 spends
        exactly two revision cycles (three lead calls), the transcript says the cap ended the loop, and the LAST
        answer is the one returned."""
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["a"]), ruling(cites=["b"]), ruling(cites=["c"])])
            run = await Team(lead, [], hooks=[AlwaysRevise("still wrong", cap=2)]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 3, "draft + exactly cap revisions"
        assert run.answer.cites == ["c"], "cap exhaustion passes the last answer on"
        assert [e.kind for e in run.transcript] == ["revise", "revise", "revise_cap"]
        assert run.transcript[-1] == RevisionCap(hook="AlwaysRevise", rounds=2)
        assert run.transcript[0] == Revision(hook="AlwaysRevise", round=1, feedback="still wrong")
        assert run.transcript[1] == Revision(hook="AlwaysRevise", round=2, feedback="still wrong")

    async def test_two_answer_hooks_each_get_their_own_budget_in_order(self):
        """Hook A revises once and accepts, then hook B sees A's accepted answer and revises
        once itself."""
        a, b = OneRevision("A says revise"), OneRevision("B says revise")
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead(
                [ruling(cites=["draft"]), ruling(cites=["afterA"]), ruling(cites=["afterB"])]
            )
            run = await Team(lead, [], hooks=[a, b]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 3
        assert any("A says revise" in p for p in lead_model.prompts(1))
        assert any("B says revise" in p for p in lead_model.prompts(2))
        assert b.seen[0].cites == ["afterA"], "B reviews what A accepted"
        assert run.answer.cites == ["afterB"]

    async def test_a_later_hooks_revision_goes_back_through_the_earlier_gate(self):
        """Restart-chain: hook A accepts the draft, hook B revises it — and A is consulted
        AGAIN on the revised answer before the run returns it."""
        a, b = AcceptingWitness(), OneRevision("B wants a citation")
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["draft"]), ruling(cites=["revised"])])
            run = await Team(lead, [], hooks=[a, b]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 2, "one draft, one revision"
        assert any("B wants a citation" in p for p in lead_model.prompts(1))
        assert run.answer.cites == ["revised"], "the revised answer is the one returned"
        assert len(a.seen) == 2, "A re-reviewed after B's revision — the gate the old loop skipped"
        assert a.seen[0].cites == ["draft"] and a.seen[1].cites == ["revised"]
        assert b.seen[-1].cites == ["revised"], "B accepted the answer it asked for"

    async def test_budgets_persist_across_restarts_and_cap_records_after_own_spend(self):
        """C3: two ever-revising hooks with cap=1 each buy exactly one revision apiece
        across the whole loop, each hook's revise_cap lands after its own spend, and two instances of one class get
        #n labels so budgets and entries stay per-instance (``_reviewer_labels``)."""
        a, b = AlwaysRevise("A objects", cap=1), AlwaysRevise("B objects", cap=1)
        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead(
                [ruling(cites=["draft"]), ruling(cites=["afterA"]), ruling(cites=["afterB"])]
            )
            run = await Team(lead, [], hooks=[a, b]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 3, "draft + at most one re-run per hook's budget"
        assert any("A objects" in p for p in lead_model.prompts(1))
        assert any("B objects" in p for p in lead_model.prompts(2))
        assert run.answer.cites == ["afterB"]
        assert [(e.kind, e.hook) for e in run.transcript] == [  # type: ignore[union-attr]
            ("revise", "AlwaysRevise#1"),
            ("revise_cap", "AlwaysRevise#1"),
            ("revise", "AlwaysRevise#2"),
            ("revise_cap", "AlwaysRevise#2"),
        ], "each cap entry follows its own hook's spend, budgets never refilled by restarts"

    async def test_a_cap_exhausted_hook_records_revise_cap_once_across_later_passes(self):
        """A hook whose budget is spent (cap=0, exhausted on sight) is revisited on every
        restart another hook causes — twice here — and still records revise_cap exactly once."""

        class Broke:
            def on_answer(self, work: Workspace, answer: Any) -> Revise:
                return Revise(feedback="no budget for this", cap=0)

        class TwoRevisions:
            def __init__(self) -> None:
                self.seen: list[Any] = []

            def on_answer(self, work: Workspace, answer: Any) -> Accept | Revise:
                self.seen.append(answer)
                if len(self.seen) <= 2:
                    return Revise(feedback="go again")
                return Accept()

        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["draft"]), ruling(cites=["v2"]), ruling(cites=["v3"])])
            run = await Team(lead, [], hooks=[Broke(), TwoRevisions()]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 3, "Broke's cap=0 bought no lead cycles"
        assert run.answer.cites == ["v3"]
        cap_entries = [e for e in run.transcript if e.kind == "revise_cap"]
        assert cap_entries == [RevisionCap(hook="Broke", rounds=0)], "revisited on every restart, recorded once"

    async def test_the_cap_is_read_off_the_latest_verdict(self):
        """Revise docstring / T3: a hook may lower its cap mid-loop; the loop stops when rounds spent reach
        the LATEST cap, not the first."""

        class Shrinking:
            def __init__(self) -> None:
                self.calls = 0

            def on_answer(self, work: Workspace, answer: Any) -> Revise:
                self.calls += 1
                return Revise(feedback="more", cap=5 if self.calls == 1 else 1)

        async with RuntimeHarness() as h:
            lead, lead_model = scripted_lead([ruling(cites=["draft"]), ruling(cites=["v2"])])
            run = await Team(lead, [], hooks=[Shrinking()]).run("go", h.coordinator)

        assert len(lead_model.contexts) == 2, "first verdict bought one round; the second verdict's cap=1 ended it"
        assert run.answer.cites == ["v2"]
        assert run.transcript[-1] == RevisionCap(hook="Shrinking", rounds=1)

    async def test_a_verdict_that_is_neither_accept_nor_revise_raises_naming_the_hook(self):
        """T2 / C4: a None silently treated as accept would grade nothing while looking like
        a review happened; the raise is a WiringError (a TeamError) naming the hook."""

        class Confused:
            def on_answer(self, work: Workspace, answer: Any) -> None:
                return None

        async with RuntimeHarness() as h:
            lead, _ = scripted_lead([ruling()])
            with pytest.raises(WiringError, match="Confused.on_answer returned None") as raised:
                # Confused returns None where on_answer promises a Verdict, on purpose.
                await Team(lead, [], hooks=[Confused()]).run("go", h.coordinator)  # pyright: ignore[reportArgumentType]
        assert isinstance(raised.value, TeamError)
        assert raised.value.function_name == "decide"


# ══════════════════════════════════════════════════════════════════
# 5. Teardown
# ══════════════════════════════════════════════════════════════════


class _RaisingAsk(Spy):
    async def ask(self, request: str) -> Any:
        self.events.append(f"ask:{request}")
        raise RuntimeError("the member broke")


class _RaisingRetire(Spy):
    async def retire(self) -> None:
        await super().retire()
        raise RuntimeError("retire failed")


class _SeesFault:
    def __init__(self) -> None:
        self.faults: list[BaseException | None] = []

    def on_closed(self, work: Workspace) -> None:
        self.faults.append(work.fault)


class TestTeardown:
    async def test_an_ask_that_raises_is_recorded_and_the_lead_reads_a_failed_tool_call(self):
        """C6: the transcript records an ask whose member raised, with ``error:`` and the exception."""
        lead, lead_model = scripted_lead([call_member("broken", "try"), ruling()])
        async with RuntimeHarness() as h:
            run = await Team(lead, [_RaisingAsk("broken")]).run("go", h.coordinator)
        (call,) = [e for e in run.transcript if isinstance(e, MemberCall)]
        assert call.request == "try" and call.answer == "error: RuntimeError: the member broke"
        assert any("the member broke" in r for r in lead_model.tool_results(1)), "the lead saw the failure"

    async def test_a_retire_that_raises_is_collected_for_on_closed_and_raised_after_it(self):
        """C14: a member's retire error is the run's fault: ``on_closed`` reads it, then the run raises it."""
        seen = _SeesFault()
        lead, _ = scripted_lead([ruling()])
        async with RuntimeHarness() as h:
            with pytest.raises(RuntimeError, match="retire failed"):
                await Team(lead, [_RaisingRetire("brittle")], hooks=[seen]).run("go", h.coordinator)
        assert len(seen.faults) == 1 and isinstance(seen.faults[0], RuntimeError)

    async def test_everything_is_retired_even_when_a_hook_raises_and_the_error_survives(self):
        """C1: a raising on_teardown neither leaks a thread nor eats the fault: members and
        the lead are retired (the coordinator registry holds nothing of the run's), and the hook's error is the one
        the caller sees."""

        class Bomb:
            def on_teardown(self, work: Workspace) -> None:
                raise ConnectionError("teardown hiccup")

        async with RuntimeHarness() as h:
            spies = [Spy("one"), Spy("two")]
            lead, _ = scripted_lead([ruling()])
            with pytest.raises(ConnectionError, match="teardown hiccup"):
                await Team(lead, spies, hooks=[Bomb()]).run("go", h.coordinator)

            assert all(s.retirements == 1 for s in spies), "the raise did not stop the unwind"
            assert await h.coordinator.list_threads() == [], "threads left live after teardown"

    async def test_a_mid_run_fault_still_retires_everybody_and_runs_teardown_hooks(self):
        """C1: the lead's model exhausts its script mid-run; the members are retired anyway
        and on_teardown fired — the unconditional-finally claim, from the objects that know."""

        class Recorder:
            def __init__(self) -> None:
                self.torn_down = 0

            def on_teardown(self, work: Workspace) -> None:
                self.torn_down += 1

        recorder = Recorder()
        async with RuntimeHarness() as h:
            spies = [Spy("one"), Spy("two")]
            lead, _ = scripted_lead([])  # ScriptExhausted on the first lead cycle
            with pytest.raises(Exception, match="ScriptExhausted|script has only"):
                await Team(lead, spies, hooks=[recorder]).run("go", h.coordinator)
            assert await h.coordinator.list_threads() == []

        assert all(s.retirements == 1 for s in spies)
        assert recorder.torn_down == 1

    async def test_the_original_fault_wins_over_a_teardown_hooks_error(self):
        """Team.run finally: a teardown hook's raise resurfaces only when nothing else is already propagating."""

        class Bomb:
            def on_teardown(self, work: Workspace) -> None:
                raise ConnectionError("teardown hiccup")

        async with RuntimeHarness() as h:
            lead, _ = scripted_lead([])
            with pytest.raises(Exception, match="ScriptExhausted|script has only"):
                await Team(lead, [], hooks=[Bomb()]).run("go", h.coordinator)


# ══════════════════════════════════════════════════════════════════
# 6. Guards
# ══════════════════════════════════════════════════════════════════


class TestGuards:
    def test_two_members_colliding_on_the_wire_are_refused_at_construction(self):
        """_check_no_duplicate_member_names / C5: both the same-name case and the sneaky one — a.b vs a_b map
        to one wire name."""
        lead, _ = scripted_lead([ruling()])
        with pytest.raises(WiringError, match="collide on the lead's wire"):
            Team(lead, [Spy("twin"), Spy("twin")])
        with pytest.raises(WiringError, match="collide on the lead's wire"):
            Team(lead, [Spy("a.b"), Spy("a_b")])

    async def test_member_own_config_hook_refused_only_when_a_hook_needs_the_slot(self):
        """C2 / M3: the one-hook-per-cycle conflict is refused loudly when real (a hook
        contributes member tools onto a member already carrying its own hook), and NOT refused when no hook needs
        the slot — the bare path must not police what it does not use."""

        def own(ctx: Any) -> dict[str, Any]:
            return {}

        class MemberContributor:
            def tools_for_member(self, work: Workspace, member: Member, ctx: Any) -> list[Any]:
                return []

        async with RuntimeHarness() as h:
            member = TeamMember(read, name="left", model=Counting([]), config_hook=own)
            lead, _ = scripted_lead([ruling()])
            run = await Team(lead, [member]).run("go", h.coordinator)
            assert run.answer.admitted is True

            member2 = TeamMember(read, name="right", model=Counting([]), config_hook=own)
            lead2, lead2_model = scripted_lead([ruling()])
            with pytest.raises(WiringError, match="already carries a config_hook"):
                await Team(lead2, [member2], hooks=[MemberContributor()]).run("go", h.coordinator)
            assert lead2_model.contexts == [], "refused before the lead spent anything"
            assert await h.coordinator.list_threads() == [], "and the lead's thread was still torn down"


# ══════════════════════════════════════════════════════════════════
# 7. The report
# ══════════════════════════════════════════════════════════════════


class TestTeamRun:
    def test_report_is_frozen(self):
        """Data carriers are frozen pydantic models; a report is evidence."""
        run = TeamRun(answer="x")
        with pytest.raises(Exception, match="frozen"):
            run.answer = "y"  # type: ignore[misc]

    def test_dump_matches_the_shape_transcripts_always_had(self):
        """TeamRun: kind strings member/revise/revise_cap and plain dicts on the way out."""
        run = TeamRun(
            answer={"admitted": True},
            transcript=[
                MemberCall(member="left.read", request="q", answer="a"),
                Revision(hook="Critic", round=1, feedback="f"),
                RevisionCap(hook="Critic", rounds=1),
            ],
            hooks_data={"critic": 1},
        )
        assert run.model_dump() == {
            "answer": {"admitted": True},
            "transcript": [
                {"kind": "member", "member": "left.read", "request": "q", "answer": "a"},
                {"kind": "revise", "hook": "Critic", "round": 1, "feedback": "f"},
                {"kind": "revise_cap", "hook": "Critic", "rounds": 1},
            ],
            "hooks_data": {"critic": 1},
        }
