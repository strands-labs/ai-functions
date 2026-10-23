"""Contract tests for ai_functions.experimental.teams: Member, TeamHook, Accept/Revise, Workspace, records."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import TypeAdapter, ValidationError

import ai_functions.experimental.teams as teams
from ai_functions import InMemoryCoordinator
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.experimental.teams import (
    CORE_TRANSCRIPT_KINDS,
    Accept,
    Equippable,
    Member,
    MemberCall,
    Note,
    Revise,
    Revision,
    RevisionCap,
    RunId,
    Spawned,
    TeamError,
    TeamHook,
    TranscriptEntry,
    Verdict,
    Workspace,
)
from ai_functions.handle import ThreadHandle
from ai_functions.types import ThreadId

# ── Fakes ─────────────────────────────────────────────────────────

HOOK_METHODS = {
    "on_assemble",
    "on_request",
    "tools_for_lead",
    "tools_for_member",
    "on_answer",
    "on_teardown",
    "on_closed",
    "before_ask",
}


class _Handle:
    """The minimal spawn handle: the core reads only ``.id`` (a nested team's handle is one)."""

    def __init__(self, ident: str) -> None:
        self.id = ThreadId(ident)


class FakeMember:
    """A three-verb member that records its lifecycle (the ``Member`` contract, as the scripted members do)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.spawned_on: Any = None
        self.parent_id: ThreadId | None = None
        self.asked: list[str] = []
        self.retired = 0

    async def spawn(self, coordinator: Any, *, parent_id: ThreadId | None = None) -> _Handle:
        self.spawned_on = coordinator
        self.parent_id = parent_id
        return _Handle(f"fake:{self.name}")

    async def ask(self, request: str) -> str:
        self.asked.append(request)
        return f"{self.name} says: {request}"

    async def retire(self) -> None:
        self.retired += 1


class FakeEquippableMember(FakeMember):
    """A member that also takes hook-contributed tools (the ``FunctionMember.equip`` seam)."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.equipped: Any = None

    def equip(self, config_hook: Any) -> None:
        self.equipped = config_hook


class PartialHook:
    """A hook implementing exactly two methods: the shape TeamHook exists to admit."""

    def __init__(self) -> None:
        self.saw: list[str] = []

    def on_assemble(self, work: Workspace) -> None:
        self.saw.append("assemble")
        work.data["partial"] = {"members": [m.name for m in work.members]}

    def on_answer(self, work: Workspace, answer: Any) -> Verdict:
        self.saw.append("answer")
        return Accept()


class FullHook:
    """A hook implementing every TeamHook method, half of them async, as the core allows."""

    def __init__(self, verdicts: list[Verdict]) -> None:
        self.verdicts = list(verdicts)

    async def on_assemble(self, work: Workspace) -> None:
        work.data["full"] = "assembled"

    def on_request(self, work: Workspace, request: str) -> str:
        return f"{request} (rewritten)"

    def tools_for_lead(self, work: Workspace, ctx: Any) -> list[Any]:
        return []

    def tools_for_member(self, work: Workspace, member: Member, ctx: Any) -> list[Any]:
        return []

    async def on_answer(self, work: Workspace, answer: Any) -> Verdict:
        return self.verdicts.pop(0)

    def on_teardown(self, work: Workspace) -> None:
        work.data["full"] = "torn down"


def _workspace(*members: Member, request: str = "who is right", team: Any = None) -> Workspace:
    return Workspace(team=team, request=request, coordinator=InMemoryCoordinator(), members=list(members))


def _entry(entry: TranscriptEntry) -> TranscriptEntry:
    return TypeAdapter(TranscriptEntry).validate_python(entry.model_dump())


# ══════════════════════════════════════════════════════════════════
# Member: name and three verbs
# ══════════════════════════════════════════════════════════════════


class TestMember:
    def test_three_verbs_and_a_name_satisfy_the_protocol(self):
        """T1 / ``Member``: name, spawn, ask, retire is the whole contract."""
        assert isinstance(FakeMember("left"), Member)

    @pytest.mark.parametrize("missing", ["name", "spawn", "ask", "retire"])
    def test_each_member_of_the_contract_is_required(self, missing):
        """T1: drop any one of the four and the shape is no longer a member."""
        member = FakeMember("left")
        if missing == "name":
            del member.name
            assert not isinstance(member, Member)
        else:
            verbs = {m: getattr(FakeMember, m) for m in ("spawn", "ask", "retire") if m != missing}
            almost = type("Almost", (), {"name": "almost", **verbs})()
            assert not isinstance(almost, Member)

    async def test_lifecycle_spawn_ask_retire(self):
        """T1 / the ``Member`` contract: stand one up, ask it one question, take it down."""
        member = FakeMember("left")
        coordinator = InMemoryCoordinator()
        handle = await member.spawn(coordinator, parent_id=ThreadId("lead-1"))
        assert isinstance(handle, Spawned)
        assert handle.id == "fake:left"
        assert member.spawned_on is coordinator
        assert member.parent_id == "lead-1"
        assert await member.ask("read this") == "left says: read this"
        await member.retire()
        assert member.retired == 1

    async def test_retire_is_idempotent(self):
        """T1 / Member.retire: an unwind loop must not crash mid-unwind."""
        member = FakeMember("left")
        await member.spawn(InMemoryCoordinator())
        await member.retire()
        await member.retire()
        assert member.retired == 2

    def test_spawn_parent_id_is_keyword_only_and_optional(self):
        """T1 / Member.spawn signature: ``spawn(coordinator, *, parent_id=None)``."""
        parameters = inspect.signature(Member.spawn).parameters
        assert list(parameters) == ["self", "coordinator", "parent_id"]
        assert parameters["parent_id"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["parent_id"].default is None


class TestSpawned:
    def test_a_real_thread_handle_is_spawned(self):
        """Composition with this repo: ``ThreadHandle.id`` is the whole of what the core reads off a spawn."""
        handle = ThreadHandle(ThreadId("t-1"), InMemoryCoordinator())
        assert isinstance(handle, Spawned)
        assert handle.id == ThreadId("t-1")

    def test_a_synthetic_handle_is_spawned(self):
        """A nested team's handle carries only an id and still qualifies."""
        assert isinstance(_Handle("synthetic:inner"), Spawned)

    def test_something_without_an_id_is_not_spawned(self):
        """T1: the promise is exactly ``.id``; an object without it fails the check."""
        assert not isinstance(object(), Spawned)


class TestEquippable:
    def test_a_member_with_equip_is_equippable(self):
        """FunctionMember.equip: the seam hook-contributed member tools ride in on."""
        member = FakeEquippableMember("right")
        assert isinstance(member, Equippable)
        assert isinstance(member, Member)
        member.equip(lambda ctx: {"tools": []})
        assert member.equipped is not None

    def test_a_plain_member_is_not_equippable(self):
        """core._equip_members: members without the seam are skipped, not refused."""
        assert not isinstance(FakeMember("left"), Equippable)


# ══════════════════════════════════════════════════════════════════
# TeamHook: every method optional
# ══════════════════════════════════════════════════════════════════


class TestTeamHook:
    def test_the_protocol_names_exactly_eight_methods(self):
        """T6: the six lifecycle methods (assemble, request, tools_for_lead, tools_for_member, answer, teardown), plus
        on_closed once nothing of the run is live and before_ask ahead of every member ask, all optional."""
        # CPython sets __protocol_attrs__ on every Protocol class; typeshed does not declare it.
        assert TeamHook.__protocol_attrs__ == HOOK_METHODS  # pyright: ignore[reportAttributeAccessIssue]

    def test_isinstance_against_the_hook_protocol_raises(self):
        """T6 / the ``TeamHook`` contract: checked with getattr, never isinstance."""
        with pytest.raises(TypeError):
            # TeamHook is not runtime-checkable, on purpose: isinstance against it must raise.
            isinstance(PartialHook(), TeamHook)  # pyright: ignore[reportArgumentType]

    def test_partial_hook_absent_methods_read_as_none(self):
        """T6: a hook implements the two methods it needs; absence means skip."""
        hook = PartialHook()
        present = {name for name in HOOK_METHODS if getattr(hook, name, None) is not None}
        assert present == {"on_assemble", "on_answer"}

    def test_tool_methods_are_declared_synchronous(self):
        """T6 / the ``TeamHook`` contract: tools_for_* run inside the sync config_hook and must not block."""
        assert not inspect.iscoroutinefunction(TeamHook.tools_for_lead)
        assert not inspect.iscoroutinefunction(TeamHook.tools_for_member)

    async def test_full_hook_walks_the_lifecycle_against_one_workspace(self):
        """Team.run order: assemble, request fold, answer loop, teardown, all over one Workspace."""
        hook = FullHook([Revise(feedback="tighten it", cap=1), Accept()])
        work = _workspace(FakeMember("left"))
        await hook.on_assemble(work)
        assert work.data["full"] == "assembled"
        assert hook.on_request(work, "who is right") == "who is right (rewritten)"
        assert hook.tools_for_lead(work, ctx=None) == []
        assert hook.tools_for_member(work, work.members[0], ctx=None) == []
        first = await hook.on_answer(work, "draft")
        assert isinstance(first, Revise) and first.cap == 1
        second = await hook.on_answer(work, "draft 2")
        assert isinstance(second, Accept)
        hook.on_teardown(work)
        assert work.data["full"] == "torn down"

    def test_partial_hook_sees_live_members_on_assemble(self):
        """Workspace docstring: members are live once on_assemble runs."""
        hook = PartialHook()
        work = _workspace(FakeMember("left"), FakeMember("right"))
        hook.on_assemble(work)
        assert work.data["partial"] == {"members": ["left", "right"]}


# ══════════════════════════════════════════════════════════════════
# Accept / Revise: the two verdicts
# ══════════════════════════════════════════════════════════════════


class TestVerdicts:
    def test_accept_carries_nothing(self):
        """T2 / Accept: the answer passes; the loop moves to the next hook."""
        assert Accept() == Accept()
        assert Accept().model_dump() == {"kind": "accept"}

    def test_accept_refuses_feedback(self):
        """T2: feedback cannot be smuggled into a pass."""
        with pytest.raises(ValidationError):
            Accept.model_validate({"feedback": "actually, no"})

    def test_revise_requires_feedback_and_defaults_cap_to_two(self):
        """Revise: ``feedback: str``, ``cap: int = 2``."""
        verdict = Revise(feedback="tighten it")
        assert verdict.feedback == "tighten it"
        assert verdict.cap == 2
        with pytest.raises(ValidationError):
            Revise.model_validate({})

    def test_revise_refuses_unknown_fields(self):
        """T2: a verdict is exactly its declared fields."""
        with pytest.raises(ValidationError):
            Revise.model_validate({"feedback": "x", "rounds": 3})

    def test_negative_cap_is_a_construction_error(self):
        """T3 / Revise validation: a negative cap would silently behave as 0."""
        with pytest.raises(ValidationError, match="negative"):
            Revise(feedback="x", cap=-1)

    @given(cap=st.integers(min_value=0, max_value=10_000))
    def test_any_non_negative_cap_is_accepted(self, cap):
        """T3: zero is legal (feedback buys no rounds); every positive cap is legal."""
        assert Revise(feedback="x", cap=cap).cap == cap

    @given(cap=st.integers(max_value=-1))
    def test_every_negative_cap_is_rejected(self, cap):
        """T3: the guard fires for the whole negative range, not just -1."""
        with pytest.raises(ValidationError):
            Revise(feedback="x", cap=cap)

    def test_verdicts_are_frozen(self):
        """T2: a verdict is evidence of a review and does not change after the fact."""
        with pytest.raises(ValidationError):
            # "revise" is not the kind an Accept declares, on purpose: the frozen model must refuse the change.
            Accept().kind = "revise"  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(ValidationError):
            Revise(feedback="x").cap = 5

    @pytest.mark.parametrize("verdict", [Accept(), Revise(feedback="tighten it", cap=0)])
    def test_a_verdict_is_exactly_one_of_the_two(self, verdict):
        """T2: Accept and Revise are disjoint; Verdict is their union and nothing else."""
        assert isinstance(verdict, Accept) != isinstance(verdict, Revise)
        assert set(Verdict.__args__) == {Accept, Revise}

    @pytest.mark.parametrize("verdict", [Accept(), Revise(feedback="tighten it", cap=3)])
    def test_verdicts_round_trip_through_the_union(self, verdict):
        """T2: ``kind`` makes a serialized verdict come back as the right type."""
        assert TypeAdapter(Verdict).validate_python(verdict.model_dump()) == verdict
        assert TypeAdapter(Verdict).validate_json(verdict.model_dump_json()) == verdict


# ══════════════════════════════════════════════════════════════════
# Transcript records
# ══════════════════════════════════════════════════════════════════


class TestTranscriptRecords:
    def test_core_kinds_are_the_strings_the_transcript_always_used(self):
        """T5: kind="member", kind="revise", kind="revise_cap"."""
        assert MemberCall(member="left", request="q", answer="a").kind == "member"
        assert Revision(hook="Critic", round=1, feedback="f").kind == "revise"
        assert RevisionCap(hook="Critic", rounds=2).kind == "revise_cap"
        assert CORE_TRANSCRIPT_KINDS == {"member", "revise", "revise_cap"}

    def test_member_call_records_the_wire_under_the_real_name(self):
        """core._member_tool: request and answer recorded by the wire, under the member's real name."""
        call = MemberCall(member="chair.decide", request="who is right", answer="left")
        assert call.model_dump() == {
            "kind": "member",
            "member": "chair.decide",
            "request": "who is right",
            "answer": "left",
        }

    def test_revision_round_counts_from_one(self):
        """T5 / _answer_loop: rounds[label] += 1 before the entry is written, so round 0 never exists."""
        assert Revision(hook="Critic", round=1, feedback="f").round == 1
        with pytest.raises(ValidationError):
            Revision(hook="Critic", round=0, feedback="f")

    def test_revision_cap_records_zero_or_more_rounds(self):
        """T5 / _answer_loop: a cap of 0 exhausts at rounds=0; negative rounds cannot have been spent."""
        assert RevisionCap(hook="Critic", rounds=0).rounds == 0
        with pytest.raises(ValidationError):
            RevisionCap(hook="Critic", rounds=-1)

    @pytest.mark.parametrize("kind", sorted(CORE_TRANSCRIPT_KINDS))
    def test_note_may_not_reuse_a_core_kind(self, kind):
        """T5: a hook's entry never masquerades as the core's."""
        with pytest.raises(ValidationError, match="reuses a transcript kind"):
            Note(kind=kind)

    def test_note_needs_a_kind(self):
        """T5: an unlabeled entry cannot be filtered for."""
        with pytest.raises(ValidationError):
            Note(kind="")
        assert Note(kind="worklog").data == {}

    @given(kind=st.text(min_size=1).filter(lambda k: k not in CORE_TRANSCRIPT_KINDS))
    def test_any_other_kind_is_a_valid_note(self, kind):
        """T5: the only forbidden kinds are the core's three."""
        assert Note(kind=kind, data={"n": 1}).kind == kind

    @pytest.mark.parametrize(
        "entry",
        [
            MemberCall(member="left", request="q", answer="a"),
            Revision(hook="Critic", round=2, feedback="f"),
            RevisionCap(hook="Critic", rounds=2),
            Note(kind="worklog", data={"posted": 3}),
        ],
    )
    def test_entries_round_trip_through_the_union_as_their_own_type(self, entry):
        """T5: ``kind`` plus the Note guard make TranscriptEntry unambiguous on the way back from JSON."""
        rehydrated = _entry(entry)
        assert type(rehydrated) is type(entry)
        assert rehydrated == entry

    def test_entries_are_frozen(self):
        """T5: transcript records are evidence and do not change after the fact."""
        call = MemberCall(member="left", request="q", answer="a")
        with pytest.raises(ValidationError):
            call.answer = "b"
        with pytest.raises(ValidationError):
            Note(kind="worklog").kind = "member"


# ══════════════════════════════════════════════════════════════════
# Workspace: one run's shared state
# ══════════════════════════════════════════════════════════════════


class TestWorkspace:
    def test_fields_match_the_documented_set_plus_run_id(self):
        """Workspace: team, request, coordinator, members, data, transcript, lead, plus run_id and calls (T4), the
        enclosing ask, how the run stopped, its usage once totaled (C13) and its fault for on_closed (C14)."""
        work = _workspace(FakeMember("left"))
        assert set(work.__dataclass_fields__) == {
            "team",
            "request",
            "coordinator",
            "members",
            "run_id",
            "data",
            "transcript",
            "lead",
            "calls",
            "parent_call",
            "cancelled",
            "cancel_reason",
            "usage",
            "fault",
        }
        assert work.data == {}
        assert work.fault is None
        assert work.transcript == []
        assert work.lead is None
        assert work.calls == {}

    def test_two_runs_from_the_same_inputs_are_distinct(self):
        """T4: identity, not value equality; a fresh run_id per construction."""
        members = [FakeMember("left")]
        coordinator = InMemoryCoordinator()
        first = Workspace(team=None, request="q", coordinator=coordinator, members=members)
        second = Workspace(team=None, request="q", coordinator=coordinator, members=members)
        assert first != second
        assert first == first
        assert first.run_id != second.run_id
        assert isinstance(first.run_id, str) and len(first.run_id) == 32

    def test_workspace_keys_per_run_hook_state(self):
        """T4: hooks carry per-run state keyed by workspace identity."""
        first = _workspace()
        second = _workspace()
        per_run = {first: "a", second: "b"}
        assert per_run[first] == "a" and per_run[second] == "b"

    def test_data_and_transcript_are_shared_across_hooks(self):
        """Workspace docstring: data is the cross-hook scratch space; transcript is the audit trail."""
        work = _workspace(FakeMember("left"))
        PartialHook().on_assemble(work)
        work.transcript.append(MemberCall(member="left", request="q", answer="a"))
        work.transcript.append(Note(kind="worklog", data={"posted": 1}))
        assert "partial" in work.data
        assert [entry.kind for entry in work.transcript] == ["member", "worklog"]

    def test_lead_is_set_after_construction_before_hooks_run(self):
        """Workspace docstring: lead is the live handle, set as soon as the lead's thread exists."""
        work = _workspace(FakeMember("left"))
        handle = ThreadHandle(ThreadId("lead-1"), work.coordinator)
        work.lead = handle
        assert work.lead is handle
        assert work.lead.id == ThreadId("lead-1")

    def test_explicit_run_id_is_honored(self):
        """T4: a journal replaying a run may pin the id instead of minting one."""
        work = Workspace(team=None, request="q", coordinator=InMemoryCoordinator(), members=[], run_id=RunId("run-7"))
        assert work.run_id == "run-7"

    def test_workspace_is_keyword_only(self):
        """Team.run builds Workspace by keyword; positional construction is refused."""
        with pytest.raises(TypeError):
            Workspace(None, "q", InMemoryCoordinator(), [])  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════
# Package surface
# ══════════════════════════════════════════════════════════════════


class TestPackageSurface:
    def test_team_error_is_an_ai_function_error(self):
        """One module error base: ``TeamError`` subclasses ``AIFunctionError``; the core and its compositions add
        the leaves."""
        error = TeamError("boom", function_name="chair")
        assert isinstance(error, AIFunctionError)
        assert error.function_name == "chair"

    def test_public_names_have_stub_coverage(self):
        """Every runtime module has a stub and the stub package exports the same __all__."""
        runtime_dir = Path(teams.__file__).parent
        spec_dir = runtime_dir.parents[3] / "spec" / "ai_functions" / "experimental" / "teams"
        for module in runtime_dir.glob("*.py"):
            assert (spec_dir / f"{module.stem}.pyi").exists(), module.name
        stub_tree = ast.parse((spec_dir / "__init__.pyi").read_text())
        stub_all = next(
            ast.literal_eval(node.value)
            for node in stub_tree.body
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "__all__" for t in node.targets)
        )
        assert stub_all == teams.__all__
        assert teams.__all__ == sorted(teams.__all__)

    def test_no_import_from_other_experimental_packages(self):
        """T7: teams imports nothing from any other experimental package; ``members.py`` alone reaches sideways
        into ``experimental/methods`` for the ``MethodAgent``-backed member (M6; methods never imports teams, pinned
        in tests/methods/test_methods_boundary.py).

        The scan is recursive, so any subpackage counts, and relative imports are resolved against the importing
        module's package."""
        root = Path(teams.__file__).parent
        own = "ai_functions.experimental.teams"
        sideways = {"members.py": "ai_functions.experimental.methods"}
        assert all((root / path).is_file() for path in sideways), sideways
        scanned: list[str] = []
        for module in sorted(root.rglob("*.py")):
            rel = module.relative_to(root).as_posix()
            scanned.append(rel)
            package = [*own.split("."), *module.relative_to(root).parent.parts]
            allowed = (own, sideways[rel]) if rel in sideways else (own,)
            for node in ast.walk(ast.parse(module.read_text())):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom):
                    base = package[: len(package) - node.level + 1] if node.level else []
                    names = [".".join([*base, node.module] if node.module else base)]
                for name in names:
                    if name.startswith("ai_functions.experimental"):
                        assert name.startswith(allowed), (rel, name)
        assert {"contracts.py", "core.py", "members.py", "types.py"} <= set(scanned)
