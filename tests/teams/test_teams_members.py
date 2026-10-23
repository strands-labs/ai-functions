"""Offline tests for teams/members.py: ``FunctionMember``, an ``AIFunction`` adapted to ``Member``.

``FunctionMember`` adapts an ``AIFunction``. Delivery claims are read off the member model's wire;
lifecycle claims off the coordinator's registry.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from strands.tools.decorator import tool as strands_tool

from ai_functions import ai_function
from ai_functions.ai_thread.config import ThreadKwargs
from ai_functions.experimental.methods import MethodAgent, ai_method
from ai_functions.experimental.teams import Equippable, Member, MethodMember, TeamError, WiringError
from ai_functions.experimental.teams import FunctionMember as TeamMember
from ai_functions.handle import ThreadHandle
from ai_functions.testing import RuntimeHarness, Turn
from ai_functions.types import ThreadContext

from .teams_methods_wire import answered
from .teams_wire import Counting, Reading, read, reading


@ai_function[Reading]
def brief(context: str = "none", request: str = "") -> str | None:
    """Context: {context}. Request: {request}."""


@ai_function[Reading]
def nothing() -> str | None:
    """Report."""


# ══════════════════════════════════════════════════════════════════
# Construction
# ══════════════════════════════════════════════════════════════════


class TestConstruction:
    def test_name_defaults_to_the_functions_name_and_is_overridable(self):
        """M1 / FunctionMember.__init__: the name defaults to the function's name, or is the caller's."""
        assert TeamMember(read).name == "read"
        assert TeamMember(read, name="left-analyst.read").name == "left-analyst.read"

    def test_parameter_defaults_to_the_first_positional_and_is_overridable(self):
        """M2 / ``_request_parameter``: the keyword a briefing lands in."""
        assert TeamMember(read)._parameter == "focus"
        assert TeamMember(brief, parameter="request")._parameter == "request"

    def test_a_function_with_no_positional_parameter_is_refused_at_construction(self):
        """M2 / ``_request_parameter``: a capability with no parameters cannot be briefed at all; refused before the
        team is assembled, not as a TypeError mid-run."""
        with pytest.raises(WiringError, match="takes no positional parameter"):
            TeamMember(nothing)

    def test_a_named_parameter_the_function_lacks_is_refused_at_construction(self):
        """M2: a typo in parameter= would otherwise surface as a TypeError from the signature bind mid-run."""
        with pytest.raises(WiringError, match="not a parameter of 'read'") as raised:
            TeamMember(read, parameter="nope")
        assert raised.value.function_name == "read"

    def test_a_function_member_is_a_member_and_is_equippable(self):
        """T1 / B1 Equippable: FunctionMember satisfies both protocols, so the core spawns it and can equip it."""
        member = TeamMember(read)
        assert isinstance(member, Member)
        assert isinstance(member, Equippable)

    def test_handle_before_spawn_is_a_wiring_error(self):
        """FunctionMember.handle before a spawn: ``not spawned yet``."""
        with pytest.raises(WiringError, match="not spawned yet") as raised:
            _ = TeamMember(read).handle
        assert isinstance(raised.value, TeamError)

    def test_repr(self):
        """FunctionMember.__repr__ names the member and the function it wraps."""
        assert repr(TeamMember(read, name="left")) == "<FunctionMember 'left' function='read'>"


# ══════════════════════════════════════════════════════════════════
# Lifecycle: spawn, ask, retire
# ══════════════════════════════════════════════════════════════════


class TestLifecycle:
    async def test_spawn_puts_the_function_on_a_child_thread_named_after_the_member(self):
        """T1 / FunctionMember.spawn: a live thread, child of parent_id, on the given coordinator."""
        async with RuntimeHarness() as h:
            parent = await h.spawn(read.replace(model=Counting([])))
            member = TeamMember(read, name="left", model=Counting([]))
            handle = await member.spawn(h.coordinator, parent_id=parent.id)
            try:
                assert isinstance(handle, ThreadHandle)
                assert member.handle is handle
                info = await h.coordinator.get_thread_info(handle.id)
                assert info.parent_id == parent.id
                assert info.thread_name == "left"
            finally:
                await member.retire()

    async def test_ask_routes_the_request_to_the_named_parameter_and_returns_the_typed_result(self):
        """T1 / FunctionMember.ask: ``thread.run(**{parameter: request})``; asserted from the member model's own
        context, and the result is the function's typed output."""
        model = Counting([reading("left", "LEFT-EVIDENCE")])
        async with RuntimeHarness() as h:
            member = TeamMember(brief, parameter="request", model=model)
            await member.spawn(h.coordinator)
            try:
                result = await member.ask("DO-THIS")
            finally:
                await member.retire()

        assert isinstance(result, Reading) and result.detail == "LEFT-EVIDENCE"
        prompt = "\n".join(model.prompts(0))
        assert "Request: DO-THIS" in prompt and "Context: none" in prompt

    async def test_a_members_history_accrues_across_asks(self):
        """_member_tools docstring: the call reaches the live thread, not a throwaway compile, so history
        accrues across the lead's calls to it."""
        model = Counting([reading("left", "one"), reading("left", "two")])
        async with RuntimeHarness() as h:
            member = TeamMember(read, model=model)
            await member.spawn(h.coordinator)
            try:
                await member.ask("first")
                await member.ask("second")
            finally:
                await member.retire()
        assert any("first" in p for p in model.prompts(1)), "the second cycle still sees the first request"

    async def test_retire_removes_the_thread_and_is_idempotent(self):
        """T1 / M4 / Member.retire: an unwind loop must not crash mid-unwind."""
        async with RuntimeHarness() as h:
            member = TeamMember(read, model=Counting([]))
            handle = await member.spawn(h.coordinator)
            await member.retire()
            await asyncio.sleep(0)  # LocalWorker._run_teardown deregisters it a tick later
            assert all(i.thread_id != handle.id for i in await h.coordinator.list_threads())
            await member.retire()
            with pytest.raises(WiringError, match="not spawned yet"):
                _ = member.handle

    async def test_retire_before_spawn_is_a_no_op(self):
        """M4: the core retires every member in its finally, including one whose spawn never happened."""
        await TeamMember(read).retire()

    async def test_retire_is_quiet_when_somebody_else_already_tore_the_thread_down(self):
        """M4: ``terminate_now`` on a gone thread raises ThreadNotFoundError; retire swallows exactly that."""
        async with RuntimeHarness() as h:
            member = TeamMember(read, model=Counting([]))
            handle = await member.spawn(h.coordinator)
            await h.coordinator.terminate_now(handle.id)
            await member.retire()

    async def test_a_member_can_be_spawned_again_after_retire(self):
        """FunctionMember.equip docstring: a second run on the same member re-uses the same cast."""
        model = Counting([reading(), reading()])
        async with RuntimeHarness() as h:
            member = TeamMember(read, model=model)
            first = await member.spawn(h.coordinator)
            await member.ask("one")
            await member.retire()
            second = await member.spawn(h.coordinator)
            try:
                await member.ask("two")
            finally:
                await member.retire()
        assert first.id != second.id
        assert not any("one" in p for p in model.prompts(1)), "a fresh thread carries no history"


# ══════════════════════════════════════════════════════════════════
# Equip: the one-config_hook seam
# ══════════════════════════════════════════════════════════════════


@strands_tool(name="members_own", description="the member carried this itself")
def members_own(x: str) -> str:
    return x


@strands_tool(name="equipped_gave", description="the equipped hook contributed this")
def equipped_gave(x: str) -> str:
    return x


@strands_tool(name="first_gave", description="the first equipped hook contributed this")
def first_gave(x: str) -> str:
    return x


class TestEquip:
    async def test_an_equipped_member_keeps_its_own_override_tools_alongside_the_hooks(self):
        """M3: equip composes, spawn recomposes the member's own tools= ahead
        of the hook's — both tools on the member's wire."""
        model = Counting([reading()])
        async with RuntimeHarness() as h:
            member = TeamMember(read, model=model, tools=[members_own])
            member.equip(lambda ctx: {"tools": [equipped_gave]})
            await member.spawn(h.coordinator)
            try:
                await member.ask("the horizon")
            finally:
                await member.retire()

        specs = model.tool_specs[0]
        assert "members_own" in specs, f"the member's own tool was lost to the hook: {specs}"
        assert "equipped_gave" in specs, f"the equipped hook's tool never arrived: {specs}"

    async def test_an_equipped_member_keeps_the_functions_compiled_tools_too(self):
        """M3: with no tools= override, the function's own compiled tools are what must survive the hook's patch
        (``AIThread._run_cycle`` applies the one ``ThreadConfig.config_hook``, whose patch replaces the tools)."""
        model = Counting([reading()])
        async with RuntimeHarness() as h:
            member = TeamMember(read.replace(tools=[members_own]), model=model)
            member.equip(lambda ctx: {"tools": [equipped_gave]})
            await member.spawn(h.coordinator)
            try:
                await member.ask("the horizon")
            finally:
                await member.retire()

        assert {"members_own", "equipped_gave"} <= set(model.tool_specs[0])

    async def test_equip_replaces_a_previously_equipped_hook(self):
        """FunctionMember.equip: the equipped slot is team-owned; a second equip replaces rather than refuses."""
        model = Counting([reading()])
        async with RuntimeHarness() as h:
            member = TeamMember(read, model=model)
            member.equip(lambda ctx: {"tools": [first_gave]})
            member.equip(lambda ctx: {"tools": [equipped_gave]})
            await member.spawn(h.coordinator)
            try:
                await member.ask("the horizon")
            finally:
                await member.retire()

        specs = model.tool_specs[0]
        assert "equipped_gave" in specs and "first_gave" not in specs

    def test_equip_refuses_a_member_carrying_its_own_config_hook_override(self):
        """FunctionMember.equip / M3: the runtime calls exactly one hook per cycle; two would drop one silently."""

        def own(ctx: ThreadContext) -> ThreadKwargs:
            return {}

        member = TeamMember(read, config_hook=own)
        with pytest.raises(WiringError, match="already carries a config_hook") as raised:
            member.equip(lambda ctx: {"tools": []})
        assert raised.value.function_name == "read"

    def test_equip_refuses_a_function_whose_own_config_carries_a_config_hook(self):
        """M3: the hook may live on the compiled function rather than in the overrides; same conflict, same refusal."""

        def own(ctx: ThreadContext) -> ThreadKwargs:
            return {}

        member = TeamMember(read.replace(config_hook=own))
        with pytest.raises(WiringError, match="already carries a config_hook"):
            member.equip(lambda ctx: {"tools": []})

    async def test_an_unequipped_member_spawns_exactly_as_its_function_would(self):
        """_equip_members docstring: the bare path spawns members exactly as their adapters would alone."""
        model = Counting([Turn(tool_calls=(("Reading", {"source": "s", "detail": "d"}),))])
        async with RuntimeHarness() as h:
            member = TeamMember(read.replace(tools=[members_own]), model=model)
            await member.spawn(h.coordinator)
            try:
                await member.ask("x")
            finally:
                await member.retire()
        specs = model.tool_specs[0]
        assert "members_own" in specs and "equipped_gave" not in specs


# ══════════════════════════════════════════════════════════════════
# The runtime's peer tools (TB1)
# ══════════════════════════════════════════════════════════════════

RUNTIME_TOOLS = {"list_threads", "send_message"}


class Lookout(MethodAgent):
    name = "lookout"

    @ai_method(str, description="Look at one target.")
    def look(self, target: str) -> str | None:
        """Look at {target}."""


class TestNoPeerTools:
    async def _offered(self, member: Any, model: Counting) -> set[str]:
        async with RuntimeHarness() as h:
            await member.spawn(h.coordinator)
            try:
                await member.ask("the horizon")
            finally:
                await member.retire()
        return set(model.tool_specs[0])

    async def test_a_members_thread_is_offered_neither_runtime_tool(self):
        """TB1: an unequipped ``FunctionMember`` with no hook of its own spawns with the peer tools off."""
        model = Counting([reading()])
        assert RUNTIME_TOOLS.isdisjoint(await self._offered(TeamMember(read, model=model), model))

    async def test_a_members_own_hook_cannot_turn_them_back_on(self):
        """TB1: the member's own ``config_hook`` patch is pinned, so returning ``True`` changes nothing."""
        model = Counting([reading()])
        member = TeamMember(read, model=model, config_hook=lambda ctx: {"coordinator_tools_enabled": True})
        assert RUNTIME_TOOLS.isdisjoint(await self._offered(member, model))

    async def test_an_equipped_members_hook_cannot_turn_them_back_on(self):
        """TB1: the composed, equipped hook is pinned the same way."""
        model = Counting([reading()])
        member = TeamMember(read, model=model)
        member.equip(lambda ctx: {"tools": [equipped_gave], "coordinator_tools_enabled": True})
        offered = await self._offered(member, model)
        assert "equipped_gave" in offered
        assert RUNTIME_TOOLS.isdisjoint(offered)

    async def test_a_method_members_str_prompt_thread_is_offered_neither(self):
        """TB1: a method whose only parameter is a ``str`` compiles to ``STR_PROMPT``; its thread is sealed anyway."""
        model = Counting([answered("seen")])
        member = MethodMember(Lookout(), "look", name="lookout", model=model)
        assert RUNTIME_TOOLS.isdisjoint(await self._offered(member, model))

    def test_an_override_that_turns_them_on_is_refused_at_construction(self):
        """TB1: a member reached around the lead's typed tool is a member the team cannot see; ``False`` is fine."""
        with pytest.raises(WiringError, match="coordinator_tools_enabled"):
            TeamMember(read, coordinator_tools_enabled=True)
        with pytest.raises(WiringError, match="coordinator_tools_enabled"):
            MethodMember(Lookout(), "look", coordinator_tools_enabled=True)
        assert TeamMember(read, coordinator_tools_enabled=False).name == "read"
