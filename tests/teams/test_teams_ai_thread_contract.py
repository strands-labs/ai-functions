"""Executable pins for the ``ai_functions`` runtime behaviors the teams core relies on.

``teams/core.py`` and ``teams/members.py`` describe runtime behaviors in their docstrings that tool
composition depends on. Prose can't catch a runtime change that moves those semantics, so each test
here is the executable version of one of them, asserted on the wire (what a recording model
actually received) or against the runtime object itself. A semantic change fails THIS file loudly.

What is pinned, and which teams code relies on it:

- ``AIThread._run_cycle`` resolves exactly one ``config_hook`` per cycle and applies its patch with
  ``dataclasses.replace``, so a ``tools`` key REPLACES the compiled tools wholesale (no merge), and
  the ``config_hook`` key itself is popped. Relied on by ``Team._lead_hook`` (the composer),
  ``FunctionMember.equip`` and ``FunctionMember.spawn``.
- When ``coordinator_tools_enabled`` holds, ``AIThread._run_cycle`` APPENDS ``list_threads`` and
  ``send_message`` AFTER the hook's replacement, so a replacing hook can't cost a thread its runtime
  tools. The team turns them off on every thread it spawns.
- ``ThreadKwargs`` declares the keys the core passes as overrides and hook patches (``model``,
  ``tools``, ``config_hook``); ``ThreadConfig.config_hook`` documents the replace semantics the
  first pin measures.
- ``AIFunction.replace`` returns a new template and leaves the original untouched. Relied on by
  ``Team.run``, which composes a fresh lead hook per run.
- ``InMemoryCoordinator.spawn`` records ``parent_id`` and ``thread_name`` on the ``ThreadInfo``.
  Relied on by ``Team.run`` (one run is one subtree) and ``FunctionMember.spawn``.
- ``InMemoryCoordinator.terminate_now`` routes to ``LocalWorker._teardown``, which drops the
  worker's record synchronously (a second call raises ``ThreadNotFoundError``) and schedules
  ``deregister_thread`` as a task, so the registry is empty one loop tick later. Relied on by
  ``FunctionMember.retire`` (idempotent, catches exactly that) and by ``Team.run``'s teardown claim
  that the registry is empty after its ``gather``.
- ``ThreadHandle.run(**kwargs)`` forwards keywords to the prompt function. Relied on by
  ``FunctionMember.ask`` (``run(**{parameter: request})``).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import BaseModel, Field
from strands.tools.decorator import tool as strands_tool

from ai_functions import ai_function
from ai_functions.ai_thread.config import ThreadKwargs
from ai_functions.runtime import ThreadNotFoundError
from ai_functions.testing import RuntimeHarness, Turn

from .teams_wire import Counting

# ── Output type and probes, module level ──


class Sighting(BaseModel):
    target: str = Field(description="What was looked at")


@ai_function[Sighting]
def probe(question: str) -> str | None:
    """Answer {question}."""


@ai_function[Sighting]
def look(context: str = "none", target: str = "") -> str | None:
    """Context {context}. Look at {target}."""


@strands_tool(name="tool_a", description="the tool the compiled config carries")
def tool_a(x: str) -> str:
    return x


@strands_tool(name="tool_b", description="the tool the hook's patch carries")
def tool_b(x: str) -> str:
    return x


def sighting() -> Turn:
    return Turn(tool_calls=(("Sighting", {"target": "seen"}),))


# ── 1+2. config_hook: replacement semantics, one call per cycle, key popped ──


async def test_a_hooks_tools_patch_replaces_compiled_tools_and_coordinator_tools_append() -> None:
    """AIThread._run_cycle: the hook's tools REPLACE the compiled tools wholesale — tool_b is on the wire and
    tool_a is NOT. And the runtime appends list_threads/send_message AFTER that
    replacement, so they survive it."""
    model = Counting([sighting()])

    def hook(ctx: Any) -> ThreadKwargs:
        return {"tools": [tool_b]}

    async with RuntimeHarness() as h:
        handle = await h.spawn(probe.replace(model=model, tools=[tool_a], config_hook=hook))
        await handle.run("anything")

    specs = model.tool_specs[0]
    assert "tool_b" in specs, f"the hook's tool never reached the wire: {specs}"
    assert "tool_a" not in specs, f"REPLACE became merge — compiled tools survived: {specs}"
    for runtime_tool in ("list_threads", "send_message"):
        assert runtime_tool in specs, f"{runtime_tool} missing: coordinator tools no longer append after the hook"


async def test_the_config_hook_is_called_exactly_once_per_cycle_and_cannot_replace_itself() -> None:
    """AIThread._run_cycle: one hook resolution per cycle — two run() cycles on one thread are exactly two
    calls. And a config_hook key in the patch is popped, so a hook smuggling in a replacement hook
    changes nothing — the decoy is never called and the original still fires on the next cycle."""
    calls = {"hook": 0, "decoy": 0}

    def decoy(ctx: Any) -> ThreadKwargs:
        calls["decoy"] += 1
        return {}

    def hook(ctx: Any) -> ThreadKwargs:
        calls["hook"] += 1
        return {"config_hook": decoy}  # popped by the runtime, a no-op

    model = Counting([sighting(), sighting()])

    async with RuntimeHarness() as h:
        handle = await h.spawn(probe.replace(model=model, config_hook=hook))
        await handle.run("first")
        await handle.run("second")

    assert calls["hook"] == 2, f"expected one hook call per cycle, got {calls['hook']}"
    assert calls["decoy"] == 0, "the patch's config_hook key was applied instead of popped"
    assert len(model.contexts) == 2


# ── 3. ThreadKwargs surface ──


def test_thread_kwargs_declares_every_key_the_teams_core_passes() -> None:
    """ThreadKwargs: the keys teams uses as overrides and hook patches all exist on ThreadKwargs — a renamed or
    dropped key breaks composition silently because TypedDict keys are not checked at runtime."""
    declared = ThreadKwargs.__required_keys__ | ThreadKwargs.__optional_keys__
    assert {"model", "tools", "config_hook", "coordinator_tools_enabled"} <= declared


# ── 4. AIFunction.replace leaves the template alone ──


def test_replace_returns_a_new_template_and_leaves_the_original_untouched() -> None:
    """AIFunction.replace: Team.run composes ``lead.replace(config_hook=...)`` per run, so the caller's
    template must keep its own config — including its own config_hook — for the next run to recompose from."""

    def own(ctx: Any) -> ThreadKwargs:
        return {}

    base = probe.replace(config_hook=own, tools=[tool_a])
    derived = base.replace(config_hook=lambda ctx: {"tools": []})
    assert derived is not base
    assert base.config.config_hook is own, "replace mutated the original's config_hook"
    assert tuple(base.config.tools) == (tool_a,)
    assert probe.config.config_hook is None and probe.config.tools == ()


# ── 5+6. Coordinator: spawn records the subtree; terminate_now deregisters ──


async def test_spawn_records_parent_id_and_thread_name_on_the_thread_info() -> None:
    """InMemoryCoordinator.spawn: the ThreadInfo carries the caller's parent_id and thread_name — the facts Team.run's
    ``{lead}-lead`` naming and members-as-children rely on."""
    async with RuntimeHarness() as h:
        parent = await h.spawn(probe.replace(model=Counting([])))
        child = await h.coordinator.spawn(probe.replace(model=Counting([])), thread_name="x-lead", parent_id=parent.id)
        try:
            info = await h.coordinator.get_thread_info(child.id)
            assert info.parent_id == parent.id
            assert info.thread_name == "x-lead"
        finally:
            await child.terminate_now()


async def test_terminate_now_drops_the_record_at_once_and_the_registry_catches_up_within_a_tick() -> None:
    """InMemoryCoordinator.terminate_now routes to ``LocalWorker._teardown``, which drops the worker's record
    synchronously but SCHEDULES ``coordinator.deregister_thread`` as a task. So a second terminate_now raises
    ThreadNotFoundError immediately — the exception FunctionMember.retire catches to stay idempotent — while
    list_threads still shows the thread until the loop runs once. Team.run's ``finally`` awaits the retires through
    ``asyncio.gather`` (tasks), so its own next line already sees the registry empty; a caller awaiting
    ``FunctionMember.retire`` directly must yield a tick first."""
    async with RuntimeHarness() as h:
        handle = await h.coordinator.spawn(probe.replace(model=Counting([])))
        await handle.terminate_now()
        with pytest.raises(ThreadNotFoundError):
            await handle.terminate_now()
        await asyncio.sleep(0)
        assert all(info.thread_id != handle.id for info in await h.coordinator.list_threads())


# ── 7. ThreadHandle.run forwards keywords ──


async def test_run_forwards_a_keyword_argument_to_the_prompt_functions_parameter() -> None:
    """ThreadHandle.run: ``run(**{name: value})`` lands in the named prompt parameter, which is the whole of how
    FunctionMember.ask routes a request. Asserted from the model's own context."""
    model = Counting([sighting()])
    async with RuntimeHarness() as h:
        handle = await h.spawn(look.replace(model=model))
        result = await handle.run(target="the horizon")

    assert result.target == "seen"
    prompt = "\n".join(model.prompts(0))
    assert "Look at the horizon" in prompt and "Context none" in prompt
