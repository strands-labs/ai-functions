"""Offline tests for versions and the conflict policy on a ``Local`` parameter.

Two learners on one parameter is the case this file exists for: both read version 0, both consolidate, and
what happens to the second write is the declared policy, never silence. The concurrent case is driven
through ``TextGradOptimizer.consolidate`` itself so the version the graph read is what the optimizer hands
back, not something the test threads through by hand.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from ai_functions import TextGradOptimizer, build_graph_from_result
from ai_functions.experimental.params import (
    Bind,
    Local,
    Param,
    ParamsError,
    ParamWiringError,
    VersionConflictError,
)
from ai_functions.types.graph import GradFeedback, ParameterNode, ThreadNode, VersionedParameterHost

from .params_wire import EchoBackward, research, scripted, write


def fb(*texts: str) -> list[GradFeedback]:
    return [GradFeedback(text=t) for t in texts]


# ── Versions ──


def test_load_returns_the_value_and_its_version_and_consolidate_bumps_by_one() -> None:
    local = Local(instructions="base")
    assert local.load("instructions") == ("base", 0)
    local.consolidate("instructions", fb("a"))
    assert local.load("instructions") == ("base\na", 1)
    local.consolidate("instructions", fb("b"))
    assert local.load("instructions") == ("base\na\nb", 2)
    assert local.param("instructions").version == 2


def test_history_keeps_one_value_per_version() -> None:
    local = Local(instructions="base")
    local.consolidate("instructions", fb("a"))
    local.consolidate("instructions", fb("b"))
    assert local.history("instructions") == ("base", "base\na", "base\na\nb")


def test_a_duplicate_only_consolidate_leaves_the_value_and_the_version_alone() -> None:
    """A batch the reducer folds to the same value is not a new version, so a repeat gradient never bumps."""
    local = Local(p="Rules:\n\n- be brief")
    local.consolidate("p", fb("- be brief"))
    assert local.load("p") == ("Rules:\n\n- be brief", 0)
    assert local.history("p") == ("Rules:\n\n- be brief",)
    local.consolidate("p", fb("- cite"), expected_version=0)
    assert local.load("p") == ("Rules:\n\n- be brief\n- cite", 1)
    local.consolidate("p", fb("- cite"), expected_version=1)
    assert local.load("p") == ("Rules:\n\n- be brief\n- cite", 1)


def test_a_consolidate_without_an_expected_version_applies_to_the_current_value() -> None:
    """The plain ``ParameterHost`` call carries no version; it merges onto whatever is current."""
    local = Local(instructions="base")
    local.consolidate("instructions", fb("a"))
    local.consolidate("instructions", fb("b"), None)
    assert local.load("instructions") == ("base\na\nb", 2)


def test_a_matching_expected_version_applies_under_every_policy() -> None:
    for conflict in ("refuse", "merge", "last_writer_wins"):
        local = Local()
        local.declare("p", "base", conflict=cast("Any", conflict))
        local.consolidate("p", fb("a"), expected_version=0)
        assert local.load("p") == ("base\na", 1), conflict


def test_an_expected_version_the_host_never_had_is_refused_under_every_policy() -> None:
    for conflict in ("refuse", "merge", "last_writer_wins"):
        local = Local()
        local.declare("p", "base", conflict=cast("Any", conflict))
        with pytest.raises(VersionConflictError, match=r"'p'.*version 7.*version 0"):
            local.consolidate("p", fb("a"), expected_version=7)


# ── The three policies, on a stale write ──


def test_refuse_is_the_default_and_names_the_parameter_and_both_versions() -> None:
    local = Local(instructions="base")
    local.consolidate("instructions", fb("a"), expected_version=0)
    with pytest.raises(VersionConflictError) as raised:
        local.consolidate("instructions", fb("b"), expected_version=0)
    message = str(raised.value)
    assert "'instructions'" in message
    assert "version 0" in message
    assert "version 1" in message
    assert raised.value.name == "instructions"
    assert raised.value.expected == 0
    assert raised.value.current == 1
    assert local.load("instructions") == ("base\na", 1), "a refused write changes nothing"
    assert issubclass(VersionConflictError, ParamsError)
    assert not issubclass(VersionConflictError, ParamWiringError)


def test_merge_folds_the_stale_write_onto_the_current_value() -> None:
    local = Local()
    local.declare("instructions", "base", conflict="merge")
    local.consolidate("instructions", fb("a"), expected_version=0)
    local.consolidate("instructions", fb("b"), expected_version=0)
    assert local.load("instructions") == ("base\na\nb", 2)


def test_last_writer_wins_rebuilds_from_the_version_the_writer_read() -> None:
    """The second learner's result is computed from what it saw; the first learner's contribution is dropped."""
    local = Local()
    local.declare("instructions", "base", conflict="last_writer_wins")
    local.consolidate("instructions", fb("a"), expected_version=0)
    local.consolidate("instructions", fb("b"), expected_version=0)
    assert local.load("instructions") == ("base\nb", 2)
    assert local.history("instructions") == ("base", "base\na", "base\nb")


def test_an_unknown_conflict_policy_is_refused_at_declaration() -> None:
    with pytest.raises(ParamWiringError, match=r"'sometimes'.*last_writer_wins.*merge.*refuse"):
        Local().declare("p", "v", conflict=cast("Any", "sometimes"))


# ── Two concurrent learners, through the optimizer ──


async def _graph(bound: Bind, feedback: str, **kwargs: str) -> ThreadNode:
    """Trace once, reconstruct, and run the offline backward so the parameter node carries ``feedback``."""
    result = await bound.trace(**kwargs, overrides={"model": scripted()})
    graph = await build_graph_from_result(result, [bound.local])
    opt = TextGradOptimizer()
    opt._backward_fn = cast("Any", EchoBackward())  # noqa: SLF001
    opt.backward(graph, feedback)
    return graph


async def test_two_learners_on_one_parameter_the_second_is_refused_with_both_versions() -> None:
    """Both graphs read version 0. The first consolidate lands; the second is stale and refused."""
    bound = Bind(research)
    first = await _graph(bound, "Cite sources.", query="q")
    second = await _graph(bound, "Be brief.", query="q")
    assert first.parameters[0].meta["version"] == 0
    assert second.parameters[0].meta["version"] == 0

    TextGradOptimizer().consolidate(first)
    assert bound.local.load("instructions") == ("Find relevant facts.\nCite sources.", 1)

    with pytest.raises(VersionConflictError, match=r"'instructions'.*version 0.*version 1"):
        TextGradOptimizer().consolidate(second)
    assert bound.local.load("instructions") == ("Find relevant facts.\nCite sources.", 1)


async def test_two_learners_under_merge_both_land() -> None:
    """``write`` declares ``style`` with ``conflict="merge"`` on its marker, so the stale second write folds in."""
    bound = Bind(write)
    first = await _graph(bound, "Shorter.", facts="f")
    second = await _graph(bound, "Headings.", facts="f")
    TextGradOptimizer().consolidate(first)
    TextGradOptimizer().consolidate(second)
    assert bound.local.load("style") == ("Plain prose.\nShorter.\nHeadings.", 2)


async def test_a_graph_that_read_the_current_version_consolidates_cleanly_after_another_write() -> None:
    """Fresh reads are not conflicts: only a write against a version the host has moved past is."""
    bound = Bind(research)
    first = await _graph(bound, "Cite sources.", query="q")
    TextGradOptimizer().consolidate(first)
    second = await _graph(bound, "Be brief.", query="q")
    assert second.parameters[0].meta["version"] == 1
    TextGradOptimizer().consolidate(second)
    assert bound.local.load("instructions") == ("Find relevant facts.\nCite sources.\nBe brief.", 2)


def test_the_optimizer_passes_no_version_to_a_host_that_is_not_versioned() -> None:
    """Breaking the seam deliberately: a plain ``ParameterHost`` must be called with today's signature."""
    calls: list[tuple[Any, ...]] = []

    class Plain:
        backend_id = "plain:1"

        def deserialize_value(self, name: str, raw: Any) -> Any:
            return raw

        def _is_procedural(self, name: str) -> bool:
            return False

        def consolidate(self, name: str, feedback: list[GradFeedback], retrieved: dict[str, str] | None = None) -> None:
            calls.append((name, [f.text for f in feedback], retrieved))

    host = Plain()
    assert not isinstance(host, VersionedParameterHost)
    node = ParameterNode(node_id="p", name="p", backend=cast("Any", host), meta={"version": 0})
    node.gradients.append(GradFeedback(text="x"))
    TextGradOptimizer().consolidate(ThreadNode(node_id="t", thread_id="t", parameters=[node]))
    assert calls == [("p", ["x"], None)]


def test_the_optimizer_passes_the_oldest_version_the_graph_read_to_a_versioned_host() -> None:
    """A parameter read at two versions inside one graph consolidates against the older read, so the policy sees
    the change that happened under the graph."""
    local = Local(p="base")
    assert isinstance(local, VersionedParameterHost)
    seen: list[int | None] = []
    real = local.consolidate

    def spy(
        name: str, feedback: list[GradFeedback], retrieved: Any = None, *, expected_version: int | None = None
    ) -> None:
        seen.append(expected_version)
        real(name, feedback, retrieved, expected_version=expected_version)

    local.consolidate = spy  # type: ignore[method-assign]
    real("p", fb("first"))  # the host moves to version 1 under the graph, without the spy seeing it
    assert local.load("p") == ("base\nfirst", 1)
    a = ParameterNode(node_id="a", name="p", backend=local, meta={"version": 0}, gradients=[GradFeedback(text="x")])
    b = ParameterNode(node_id="b", name="p", backend=local, meta={"version": 1}, gradients=[GradFeedback(text="y")])
    child = ThreadNode(node_id="c", thread_id="c", parameters=[b])
    root = ThreadNode(node_id="r", thread_id="r", parameters=[a], child_threads=[child])
    with pytest.raises(VersionConflictError):
        TextGradOptimizer().consolidate(root)
    assert seen == [0]


# ── Candidates: the optimize bridge bumps versions too ──


def test_apply_replaces_values_and_bumps_each_touched_version() -> None:
    local = Local(instructions="base", weights=[1.0, 2.0])
    local.apply({"instructions": "rewritten"})
    assert local.load("instructions") == ("rewritten", 1)
    assert local.load("weights") == ([1.0, 2.0], 0)
    local.apply({"weights": "[3, 4]"})
    assert local.load("weights") == ([3.0, 4.0], 1)


def test_apply_refuses_an_unknown_name_and_changes_nothing() -> None:
    local = Local(instructions="base")
    with pytest.raises(ParamWiringError, match=r"'nope'"):
        local.apply({"instructions": "x", "nope": "y"})
    assert local.load("instructions") == ("base", 0)


def test_apply_with_one_malformed_value_changes_nothing_and_names_the_parameter() -> None:
    """All or nothing: the first parameter is not written when a later one fails to parse."""
    local = Local(instructions="base", weights=[1.0, 2.0])
    with pytest.raises(ParamWiringError, match=r"'weights'.*'not json'") as raised:
        local.apply({"instructions": "rewritten", "weights": "not json"})
    assert isinstance(raised.value, ParamsError)
    assert local.load("instructions") == ("base", 0)
    assert local.load("weights") == ([1.0, 2.0], 0)
    with pytest.raises(ParamWiringError, match=r"(?s)'weights'.*Nothing was written"):
        local.apply({"instructions": "rewritten", "weights": '{"a": 1}'})  # valid JSON, not a list
    assert local.load("instructions") == ("base", 0)
    assert local.load("weights") == ([1.0, 2.0], 0)


# ── Snapshot and restore: the record round-trips through the host too ──


def test_restore_rebuilds_a_host_at_the_stored_value_and_version() -> None:
    local = Local(instructions="base", weights=[1.0, 2.0])
    local.consolidate("instructions", fb("a"))
    local.consolidate("instructions", fb("b"))
    local.consolidate("weights", fb("[1, 1]"))
    stored = {name: Param.model_validate_json(p.model_dump_json()) for name, p in local.snapshot().items()}

    rebuilt = Local(instructions="base", weights=[1.0, 2.0])
    rebuilt.restore(stored)
    assert rebuilt.load("instructions") == local.load("instructions") == ("base\na\nb", 2)
    assert rebuilt.load("weights") == local.load("weights") == ([2.0, 3.0], 1)
    assert rebuilt.snapshot() == local.snapshot()
    assert rebuilt.history("instructions") == ("base\na\nb",)

    rebuilt.consolidate("instructions", fb("c"), expected_version=2)
    assert rebuilt.load("instructions") == ("base\na\nb\nc", 3)


async def test_a_graph_traced_before_a_restart_consolidates_into_the_restored_host() -> None:
    """The version the graph read is the version the restored host holds, so the default policy applies it."""
    bound = Bind(research)
    bound.local.consolidate("instructions", fb("Cite sources."))
    graph = await _graph(bound, "Be brief.", query="q")
    assert graph.parameters[0].meta["version"] == 1
    stored = bound.local.snapshot()

    restarted = Bind(research)
    assert restarted.local.load("instructions") == ("Find relevant facts.", 0)
    restarted.local.restore(stored)
    for node in graph.parameters:
        node.backend = restarted.local
    TextGradOptimizer().consolidate(graph)
    assert restarted.local.load("instructions") == ("Find relevant facts.\nCite sources.\nBe brief.", 2)


def test_restore_declares_nothing_and_refuses_an_unknown_name_before_anything_changes() -> None:
    source = Local(a="x", b="y")
    source.consolidate("a", fb("more"))
    target = Local(a="x")
    with pytest.raises(ParamWiringError, match=r"'b'"):
        target.restore(source.snapshot())
    assert target.load("a") == ("x", 0)


def test_restore_refuses_a_record_whose_reducer_or_policy_disagrees_with_the_declaration() -> None:
    source = Local(p="x")
    other_reducer = Local()
    other_reducer.declare("p", 1.0)
    with pytest.raises(ParamWiringError, match=r"'p'.*'numeric_add'.*'ordered'"):
        other_reducer.restore(source.snapshot())
    other_policy = Local()
    other_policy.declare("p", "x", conflict="merge")
    with pytest.raises(ParamWiringError, match=r"'p'.*'merge'.*'refuse'"):
        other_policy.restore(source.snapshot())
    assert other_policy.load("p") == ("x", 0)


def test_restore_refuses_a_host_that_has_already_been_written() -> None:
    source = Local(p="x")
    source.consolidate("p", fb("a"))
    target = Local(p="x")
    target.consolidate("p", fb("b"))
    with pytest.raises(ParamWiringError, match=r"'p'.*version 1"):
        target.restore(source.snapshot())
    assert target.load("p") == ("x\nb", 1)


def test_history_before_the_restored_version_is_unavailable_so_last_writer_wins_refuses_there() -> None:
    source = Local()
    source.declare("p", "base", conflict="last_writer_wins")
    source.consolidate("p", fb("a"))
    source.consolidate("p", fb("b"))
    target = Local()
    target.declare("p", "base", conflict="last_writer_wins")
    target.restore(source.snapshot())
    with pytest.raises(VersionConflictError, match=r"'p'.*version 0.*version 2"):
        target.consolidate("p", fb("c"), expected_version=0)
    assert target.load("p") == ("base\na\nb", 2)
    target.consolidate("p", fb("c"), expected_version=2)
    assert target.load("p") == ("base\na\nb\nc", 3)


def test_a_restored_host_still_accepts_the_same_declaration_as_one_parameter() -> None:
    """LP8 compares against the declared initial, not the restored value, so a second Bind on the host is fine."""
    source = Bind(research)
    source.local.consolidate("instructions", fb("Cite sources."))
    restarted = Bind(research)
    restarted.local.restore(source.local.snapshot())
    again = Bind(research, restarted.local)
    assert again.local is restarted.local
    assert again.local.load("instructions") == ("Find relevant facts.\nCite sources.", 1)
