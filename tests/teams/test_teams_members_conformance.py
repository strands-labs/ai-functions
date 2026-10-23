"""One conformance suite, two member kinds: ``FunctionMember`` (over an ``AIFunction``) and ``MethodMember`` (over a
``MethodAgent`` method) answer the same assertions.

Both satisfy ``Member``, and this suite runs the same assertions against both: the ones
tests/teams/test_teams_members.py pins for ``FunctionMember`` alone (spawn under the lead, ask on the wire, retire
idempotent, the one-config_hook seam), each run through a ``kind`` fixture that builds the same ``left-analyst.read``
member from either shape. Delivery claims are read off the member model's wire; lifecycle claims off the coordinator's
registry.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

import pytest

from ai_functions.ai_thread.config import ThreadKwargs
from ai_functions.experimental.teams import Equippable, Member, MemberCall, Spawned, Team, WiringError
from ai_functions.experimental.teams import FunctionMember as TeamMember
from ai_functions.experimental.teams import MethodMember as TeamMethodMember
from ai_functions.testing import RuntimeHarness
from ai_functions.types import ThreadContext

from .teams_methods_wire import (
    Analyst,
    Briefed,
    Silent,
    ToolCarrier,
    brief_fn,
    equipped_gave,
    first_gave,
    members_own,
    nothing_fn,
)
from .teams_wire import Counting, Reading, call_member, read, reading, ruling, scripted_lead

NAME = "left-analyst_read"
WIRE = "left-analyst_read"


def _own(ctx: ThreadContext) -> ThreadKwargs:
    return {}


type AnyMember = TeamMember | TeamMethodMember
"""The two member kinds this suite runs one set of assertions against."""


@dataclass(frozen=True)
class Kind:
    """How to build each shape of the same member, so one assertion runs against both."""

    label: str
    make: Callable[..., AnyMember]
    """``make(model, **overrides)``: the ``left-analyst.read`` member."""
    carrier: Callable[..., AnyMember]
    """``carrier(model)``: the same member whose *compiled* config (not an override) carries ``members_own``."""
    hooked: Callable[[], AnyMember]
    """A member whose compiled config carries its own ``config_hook``."""
    briefed: Callable[..., AnyMember]
    """``briefed(model)``: ``parameter="request"`` on a shape whose first positional is ``context``."""
    silent: Callable[[], AnyMember]
    """A capability with no positional parameter: refused at construction (M2)."""


KINDS = [
    Kind(
        label="function",
        make=lambda model, **ov: TeamMember(read, name=NAME, model=model, **ov),
        carrier=lambda model: TeamMember(read.replace(tools=[members_own]), name=NAME, model=model),
        hooked=lambda: TeamMember(read.replace(config_hook=_own), name=NAME),
        briefed=lambda model: TeamMember(brief_fn, parameter="request", model=model),
        silent=lambda: TeamMember(nothing_fn),
    ),
    Kind(
        label="method",
        make=lambda model, **ov: TeamMethodMember(Analyst("left"), "read", model=model, **ov),
        carrier=lambda model: TeamMethodMember(ToolCarrier(), "read", name=NAME, model=model),
        hooked=lambda: TeamMethodMember(Analyst("left"), "read", config_hook=_own),
        briefed=lambda model: TeamMethodMember(Briefed(), "brief", parameter="request", model=model),
        silent=lambda: TeamMethodMember(Silent(), "report"),
    ),
]


@pytest.fixture(params=KINDS, ids=[k.label for k in KINDS])
def kind(request: pytest.FixtureRequest) -> Kind:
    return request.param


# ══════════════════════════════════════════════════════════════════
# Protocols and construction
# ══════════════════════════════════════════════════════════════════


class TestShape:
    def test_both_kinds_are_members_and_equippable(self, kind: Kind):
        """T1 / Equippable: the core spawns either and can equip either (test_teams_members
        test_a_function_member_is_a_member_and_is_equippable, both kinds)."""
        member = kind.make(Counting([]))
        assert isinstance(member, Member)
        assert isinstance(member, Equippable)

    def test_both_kinds_carry_the_same_name(self, kind: Kind):
        """M1: ``FunctionMember`` from ``name=``, ``MethodMember`` from ``{owner}_{method}``; the lead's wire sees one
        name."""
        assert kind.make(Counting([])).name == NAME

    def test_parameter_override_is_honored_by_both(self, kind: Kind):
        """M2 / test_teams_members test_parameter_defaults_to_the_first_positional_and_is_overridable."""
        assert kind.briefed(Counting([]))._parameter == "request"

    def test_a_shape_with_no_positional_parameter_is_refused_by_both(self, kind: Kind):
        """M2 / test_teams_members test_a_function_with_no_positional_parameter_is_refused_at_construction."""
        with pytest.raises(WiringError, match="takes no positional parameter"):
            kind.silent()

    def test_handle_before_spawn_is_a_wiring_error_for_both(self, kind: Kind):
        """test_teams_members test_handle_before_spawn_is_a_wiring_error; ``MethodMember`` exposes the same handle."""
        with pytest.raises(WiringError, match="not spawned yet"):
            _ = kind.make(Counting([])).handle


# ══════════════════════════════════════════════════════════════════
# Lifecycle
# ══════════════════════════════════════════════════════════════════


class TestLifecycle:
    async def test_spawn_puts_the_member_on_a_child_thread_named_after_it(self, kind: Kind):
        """T1 / test_teams_members test_spawn_puts_the_function_on_a_child_thread_named_after_the_member: a live
        thread, child of parent_id, named ``member.name``, on the given coordinator; the return is ``Spawned``."""
        async with RuntimeHarness() as h:
            parent = await h.spawn(read.replace(model=Counting([])))
            member = kind.make(Counting([]))
            spawned = await member.spawn(h.coordinator, parent_id=parent.id)
            try:
                assert isinstance(spawned, Spawned)
                info = await h.coordinator.get_thread_info(spawned.id)
                assert info.parent_id == parent.id
                assert info.thread_name == NAME
                assert member.handle.id == spawned.id
            finally:
                await member.retire()

    async def test_ask_routes_the_request_to_the_named_parameter_on_the_wire(self, kind: Kind):
        """T1 / test_teams_members test_ask_routes_the_request_to_the_named_parameter_and_returns_the_typed_result."""
        model = Counting([reading("left", "LEFT-EVIDENCE")])
        async with RuntimeHarness() as h:
            member = kind.briefed(model)
            await member.spawn(h.coordinator)
            try:
                result = await member.ask("DO-THIS")
            finally:
                await member.retire()
        assert isinstance(result, Reading) and result.detail == "LEFT-EVIDENCE"
        prompt = "\n".join(model.prompts(0))
        assert "Request: DO-THIS" in prompt and "Context: none" in prompt

    async def test_history_accrues_across_asks(self, kind: Kind):
        """test_teams_members test_a_members_history_accrues_across_asks: the call reaches the live thread."""
        model = Counting([reading("left", "one"), reading("left", "two")])
        async with RuntimeHarness() as h:
            member = kind.make(model)
            await member.spawn(h.coordinator)
            try:
                await member.ask("first")
                await member.ask("second")
            finally:
                await member.retire()
        assert any("first" in p for p in model.prompts(1))

    async def test_retire_removes_the_thread_and_is_idempotent(self, kind: Kind):
        """T1 / M4 / test_teams_members test_retire_removes_the_thread_and_is_idempotent."""
        async with RuntimeHarness() as h:
            member = kind.make(Counting([]))
            spawned = await member.spawn(h.coordinator)
            await member.retire()
            await asyncio.sleep(0)  # LocalWorker._run_teardown deregisters the thread a tick later
            assert all(i.thread_id != spawned.id for i in await h.coordinator.list_threads())
            await member.retire()
            with pytest.raises(WiringError, match="not spawned yet"):
                _ = member.handle

    async def test_retire_before_spawn_is_a_no_op(self, kind: Kind):
        """M4 / test_teams_members test_retire_before_spawn_is_a_no_op."""
        await kind.make(Counting([])).retire()

    async def test_retire_is_quiet_when_somebody_else_already_tore_the_thread_down(self, kind: Kind):
        """M4 / test_teams_members test_retire_is_quiet_when_somebody_else_already_tore_the_thread_down."""
        async with RuntimeHarness() as h:
            member = kind.make(Counting([]))
            spawned = await member.spawn(h.coordinator)
            await h.coordinator.terminate_now(spawned.id)
            await member.retire()

    async def test_a_member_can_be_spawned_again_after_retire(self, kind: Kind):
        """test_teams_members test_a_member_can_be_spawned_again_after_retire: a fresh thread, no history."""
        model = Counting([reading(), reading()])
        async with RuntimeHarness() as h:
            member = kind.make(model)
            first = await member.spawn(h.coordinator)
            await member.ask("one")
            await member.retire()
            second = await member.spawn(h.coordinator)
            try:
                await member.ask("two")
            finally:
                await member.retire()
        assert first.id != second.id
        assert not any("one" in p for p in model.prompts(1))


# ══════════════════════════════════════════════════════════════════
# Equip: the one-config_hook seam
# ══════════════════════════════════════════════════════════════════


class TestEquip:
    async def test_an_equipped_member_keeps_its_own_override_tools_alongside_the_hooks(self, kind: Kind):
        """M3 / test_teams_members test_an_equipped_member_keeps_its_own_override_tools_alongside_the_hooks."""
        model = Counting([reading()])
        async with RuntimeHarness() as h:
            member = kind.make(model, tools=[members_own])
            member.equip(lambda ctx: {"tools": [equipped_gave]})
            await member.spawn(h.coordinator)
            try:
                await member.ask("the horizon")
            finally:
                await member.retire()
        specs = model.tool_specs[0]
        assert "members_own" in specs, f"the member's own tool was lost to the hook: {specs}"
        assert "equipped_gave" in specs, f"the equipped hook's tool never arrived: {specs}"

    async def test_an_equipped_member_keeps_its_compiled_tools_too(self, kind: Kind):
        """M3 / test_teams_members test_an_equipped_member_keeps_the_functions_compiled_tools_too: the tool lives on
        the function's config or the decorator's config, not in an override, and still survives the hook's patch."""
        model = Counting([reading()])
        async with RuntimeHarness() as h:
            member = kind.carrier(model)
            member.equip(lambda ctx: {"tools": [equipped_gave]})
            await member.spawn(h.coordinator)
            try:
                await member.ask("the horizon")
            finally:
                await member.retire()
        assert {"members_own", "equipped_gave"} <= set(model.tool_specs[0])

    async def test_equip_replaces_a_previously_equipped_hook(self, kind: Kind):
        """test_teams_members test_equip_replaces_a_previously_equipped_hook: the equipped slot is team-owned."""
        model = Counting([reading()])
        async with RuntimeHarness() as h:
            member = kind.make(model)
            member.equip(lambda ctx: {"tools": [first_gave]})
            member.equip(lambda ctx: {"tools": [equipped_gave]})
            await member.spawn(h.coordinator)
            try:
                await member.ask("the horizon")
            finally:
                await member.retire()
        specs = model.tool_specs[0]
        assert "equipped_gave" in specs and "first_gave" not in specs

    def test_equip_refuses_a_member_carrying_its_own_config_hook_override(self, kind: Kind):
        """M3 / test_teams_members test_equip_refuses_a_member_carrying_its_own_config_hook_override."""
        member = kind.make(Counting([]), config_hook=_own)
        with pytest.raises(WiringError, match="already carries a config_hook") as raised:
            member.equip(lambda ctx: {"tools": []})
        assert raised.value.function_name == NAME

    def test_equip_refuses_a_compiled_config_that_carries_a_config_hook(self, kind: Kind):
        """M3 / test_teams_members test_equip_refuses_a_function_whose_own_config_carries_a_config_hook."""
        with pytest.raises(WiringError, match="already carries a config_hook"):
            kind.hooked().equip(lambda ctx: {"tools": []})

    async def test_an_unequipped_member_spawns_exactly_as_it_would_alone(self, kind: Kind):
        """test_teams_members test_an_unequipped_member_spawns_exactly_as_its_function_would."""
        model = Counting([reading()])
        async with RuntimeHarness() as h:
            member = kind.carrier(model)
            await member.spawn(h.coordinator)
            try:
                await member.ask("x")
            finally:
                await member.retire()
        specs = model.tool_specs[0]
        assert "members_own" in specs and "equipped_gave" not in specs


# ══════════════════════════════════════════════════════════════════
# In a team
# ══════════════════════════════════════════════════════════════════


class TestInATeam:
    async def test_the_lead_reaches_either_kind_as_a_typed_tool_and_the_transcript_names_it(self, kind: Kind):
        """C6 / test_teams_core test_lead_reaches_a_member_as_a_typed_tool_and_the_answer_is_ungraded, both kinds: the
        member is on the lead's wire under its own (dot-mapped) name, the request reaches the member's model, and
        the transcript records the call under the real name."""
        async with RuntimeHarness() as h:
            model = Counting([reading("left", "LEFT-SAW")])
            member = kind.make(model)
            lead, lead_model = scripted_lead([call_member(WIRE, "what does left show"), ruling(cites=[NAME])])
            run = await Team(lead, [member]).run("who is right", h.coordinator)
            remaining = {str(i.thread_id) for i in await h.coordinator.list_threads()}

        assert WIRE in lead_model.tool_specs[0]
        assert any("what does left show" in p for p in model.prompts(0))
        calls = [e for e in run.transcript if isinstance(e, MemberCall)]
        assert [c.member for c in calls] == [NAME]
        assert "LEFT-SAW" in calls[0].answer
        assert run.answer.admitted is True
        assert remaining == set(), f"the run's subtree must be gone after teardown: {remaining}"
        with pytest.raises(WiringError, match="not spawned yet"):
            _ = member.handle
