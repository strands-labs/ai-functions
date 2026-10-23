"""Offline tests for ``params/local.py``: the marker, the host, the binder, and the injection.

Three kinds of claim, checked three ways. The *marker* is introspection, so ``trainable_params`` is called
directly. The *injection* is a claim about the optimizer's graph, so it is asserted from
``build_graph_from_result`` over the traced thread's event log, and then from the TextGrad backward pass
itself through the same offline backward seam ``tests/test_trace_dataflow.py`` uses. The *guards* are checked
by counting model calls: every guard test scripts a model with zero turns, so a call that reached the
runtime raises ``ScriptExhausted`` instead of the expected error.
"""

from __future__ import annotations

import typing
from typing import Annotated, Any, cast

import pytest

from ai_functions import TextGradOptimizer, build_graph_from_result
from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.experimental.methods import MethodAgent, Recalled, ai_method
from ai_functions.experimental.params import (
    ORDERED,
    Bind,
    Local,
    Param,
    ParamsError,
    ParamWiringError,
    Trainable,
    trainable_params,
)
from ai_functions.testing import RuntimeHarness
from ai_functions.types.graph import ParameterHost, ParameterView, collect_nodes

from .params_wire import Choice, EchoBackward, Navigator, choosing, plain, research, scripted, texting, weigh, write

# ── The marker ──


def test_the_bare_alias_and_the_configured_marker_are_one_marker() -> None:
    """``Trainable[str]`` is ``Annotated[str, Trainable()]``: one name, one default, every knob optional."""
    alias = Trainable[str]
    assert typing.get_origin(alias) is Annotated
    assert typing.get_args(alias)[0] is str
    assert typing.get_args(alias)[1] == Trainable()
    assert Trainable().conflict == "refuse"
    assert Trainable().reducer is None


def test_the_marker_is_found_on_functions_and_bound_methods_and_unmarked_params_are_ignored() -> None:
    """Introspection is the binder's whole input, so it has to see exactly the marked parameters."""
    assert trainable_params(research.prompt_fn) == {"instructions": Trainable()}
    assert trainable_params(write.prompt_fn) == {
        "style": Trainable(conflict="merge", description="How the report should read.")
    }
    assert trainable_params(plain.prompt_fn) == {}
    assert trainable_params(Navigator().choose) == {"playbook": Trainable()}
    assert "self" not in trainable_params(Navigator().choose)
    assert trainable_params(Navigator().guess) == {}


def test_two_distinct_markers_on_one_annotation_are_refused_and_a_repeat_is_fine() -> None:
    """Which reducer a parameter learns under must not depend on annotation order."""

    def conflicted(x: Annotated[str, Trainable(), Trainable(conflict="merge")] = "v") -> None: ...

    with pytest.raises(ParamWiringError, match=r"2 distinct `Trainable` markers"):
        trainable_params(conflicted)

    def repeated(x: Annotated[str, Trainable(), Trainable()] = "v") -> None: ...

    assert trainable_params(repeated) == {"x": Trainable()}


def test_a_parameter_cannot_be_both_recalled_and_trainable() -> None:
    """Memory-backed and local are two spellings of where a value comes from; one parameter picks one."""

    def both(x: Annotated[str, Trainable(), Recalled("guidance")] = "v") -> None: ...

    with pytest.raises(ParamWiringError, match=r"Recalled"):
        trainable_params(both)


def test_an_unresolvable_annotation_raises_rather_than_reading_as_unmarked() -> None:
    """Swallowing here would mean no injection: an agent that trains nothing forever, with no error."""

    def broken(x: NoSuchType = "v") -> None: ...  # noqa: F821  # pyright: ignore[reportUndefinedVariable] (the typo under test)

    with pytest.raises(NameError):
        trainable_params(broken)


# ── The host ──


def test_the_host_declares_from_keywords_with_sensible_defaults() -> None:
    """``Local(instructions="...")`` is a trainable parameter at version 0 under the ORDERED reducer."""
    local = Local(instructions="Find facts.")
    assert local.names == ("instructions",)
    assert local.load("instructions") == ("Find facts.", 0)
    param = local.param("instructions")
    assert isinstance(param, Param)
    assert param.reducer == ORDERED.name
    assert param.conflict == "refuse"
    assert param.version == 0
    assert local.backend_id.startswith("Local:")


def test_the_param_record_round_trips_through_json() -> None:
    """A frozen pydantic record an orchestrator can store."""
    local = Local(weights=[1.0, 2.0])
    param = local.param("weights")
    assert Param.model_validate_json(param.model_dump_json()) == param
    with pytest.raises(Exception, match=r"frozen"):
        param.version = 3  # type: ignore[misc]


def test_redeclaring_an_identical_parameter_is_a_no_op_and_a_different_one_is_refused() -> None:
    """Two functions sharing one parameter declare it twice; two functions disagreeing is a wiring bug."""
    local = Local()
    first = local.declare("instructions", "Find facts.")
    assert local.declare("instructions", "Find facts.") == first
    with pytest.raises(ParamWiringError, match=r"'instructions'.*'Find facts\.'.*'Find more\.'"):
        local.declare("instructions", "Find more.")


def test_an_unknown_name_is_refused_naming_the_declared_ones() -> None:
    local = Local(instructions="x", style="y")
    with pytest.raises(ParamWiringError, match=r"'nope'.*\['instructions', 'style'\]"):
        local.load("nope")


def test_the_host_satisfies_the_parameter_host_protocol_without_a_memory_backend() -> None:
    """The whole point: a trainable value with no memory backend anywhere."""
    local = Local(instructions="x")
    assert isinstance(local, ParameterHost)
    assert local._is_procedural("instructions") is False  # noqa: SLF001
    assert local.deserialize_value("instructions", "raw") == "raw"
    weights = Local(weights=[1.0, 2.0])
    assert weights.deserialize_value("weights", [3, 4]) == [3.0, 4.0]


def test_a_view_is_fresh_per_call_and_carries_version_and_reducer_in_its_meta() -> None:
    local = Local(instructions="x")
    a, b = local.view("instructions"), local.view("instructions")
    assert isinstance(a, ParameterView)
    assert a is not b
    assert a.value == "x"
    assert a.name == "instructions"
    assert a.backend is local
    assert a.emitted is False
    assert a.requires_grad is True
    assert a.meta["version"] == 0
    assert a.meta["reducer"] == ORDERED.name
    assert ORDERED.guidance in a.description


# ── The binder: wiring ──


def test_bind_reads_the_default_as_the_initial_value_on_a_function_and_on_an_agent() -> None:
    """One annotation, one default, one trainable parameter: nothing else to write."""
    bound = Bind(research)
    assert bound.bound() == {"instructions": Trainable()}
    assert bound.local.load("instructions") == ("Find relevant facts.", 0)

    agent = Bind(Navigator())
    assert agent.bound("choose") == {"playbook": Trainable()}
    assert agent.bound("guess") == {}
    assert agent.local.load("playbook") == ("Prefer the shortest path.", 0)


def test_bind_honours_the_markers_knobs() -> None:
    bound = Bind(write)
    param = bound.local.param("style")
    assert param.conflict == "merge"
    assert param.description == "How the report should read."
    assert Bind(weigh).local.param("weights").reducer == "numeric_add"


def test_two_binders_can_share_one_host() -> None:
    """A pipeline's parameters live in one ``Local``; each stage binds against it."""
    local = Local()
    a = Bind(research, local)
    b = Bind(write, local)
    assert a.local is b.local is local
    assert local.names == ("instructions", "style")


def test_a_trainable_parameter_without_a_default_is_refused_at_bind() -> None:
    """The default is the initial value; without one there is nothing to train."""

    def nodefault(x: Trainable[str]) -> None: ...

    with pytest.raises(ParamWiringError, match=r"'x'.*default"):
        Bind(cast("Any", _fn(nodefault)))


def test_a_positional_only_trainable_parameter_is_refused_at_bind() -> None:
    def cornered(x: Annotated[str, Trainable()] = "v", /) -> None: ...

    with pytest.raises(ParamWiringError, match=r"positional-only"):
        Bind(cast("Any", _fn(cornered)))


def test_a_parameter_named_overrides_is_refused_at_bind() -> None:
    def shadow(overrides: str = "", x: Annotated[str, Trainable()] = "v") -> None: ...

    with pytest.raises(ParamWiringError, match=r"'overrides'"):
        Bind(cast("Any", _fn(shadow)))


def test_a_target_that_is_neither_a_function_nor_an_agent_is_refused() -> None:
    with pytest.raises(ParamWiringError, match=r"AIFunction.*MethodAgent"):
        Bind(cast("Any", object()))


def _fn(prompt_fn: Any) -> Any:
    """Wrap a plain function as an ``AIFunction`` the way the decorator does, for bind-time guards."""
    from ai_functions import ai_function

    return ai_function[str](structured_output=False)(prompt_fn)


# ── The injection ──


async def test_the_bound_value_reaches_the_graph_as_a_parameter_node_on_a_function() -> None:
    """Asserted from the graph, not from what ``trace`` was handed."""
    bound = Bind(research)
    model = texting(1)
    result = await bound.trace(query="cats", overrides={"model": model})

    graph = await build_graph_from_result(result, [bound.local])
    assert [p.name for p in graph.parameters] == ["instructions"]
    node = graph.parameters[0]
    assert node.backend is bound.local
    assert node.value == "Find relevant facts."
    assert node.requires_grad is True
    assert node.meta["version"] == 0
    assert any("Find relevant facts." in p for p in model.prompts(0))


async def test_the_bound_value_reaches_the_graph_on_an_agent_method() -> None:
    async with RuntimeHarness():
        bound = Bind(Navigator())
        model = choosing(1)
        result = await bound.trace("choose", state="S0", overrides={"model": model})

        assert result.value.transition == "T0"
        graph = await build_graph_from_result(result, [bound.local])
        assert [p.name for p in graph.parameters] == ["playbook"]
        assert graph.parameters[0].backend is bound.local
        assert any("shortest path" in p for p in model.prompts(0))


async def test_a_second_trace_gets_a_fresh_node_because_nothing_is_cached() -> None:
    """A view is single-use; a binder that stored one would yield a node once and never again."""
    bound = Bind(research)
    names: list[list[str]] = []
    for _ in range(2):
        result = await bound.trace(query="q", overrides={"model": scripted()})
        graph = await build_graph_from_result(result, [bound.local])
        names.append([p.name for p in graph.parameters])
    assert names == [["instructions"], ["instructions"]]


async def test_the_view_enters_the_traced_call_as_a_keyword_argument() -> None:
    """The invariant named at its boundary: ``collect_nodes((args, kwargs))`` finds exactly the view."""
    seen: dict[str, Any] = {}
    fn = research.replace(model=scripted())
    real = fn.trace

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen["nodes"] = collect_nodes((args, kwargs))
        seen["kwargs"] = dict(kwargs)
        seen["args"] = args
        return await real(*args, **kwargs)

    fn.trace = spy  # type: ignore[method-assign]
    bound = Bind(fn)
    result = await bound.trace(query="q")

    views = [n for n in seen["nodes"] if isinstance(n, ParameterView)]
    assert len(views) == 1
    assert seen["kwargs"]["instructions"] is views[0]
    assert seen["args"] == ()
    assert result.inputs == [views[0]]


async def test_an_explicitly_supplied_value_passes_through_and_is_not_a_parameter_node() -> None:
    """Explicit beats implicit: a peer calling the function as a tool supplies the value itself."""
    bound = Bind(research)
    model = texting(1)
    result = await bound.trace(query="q", instructions="the caller's own", overrides={"model": model})
    graph = await build_graph_from_result(result, [bound.local])
    assert graph.parameters == []
    assert any("the caller's own" in p for p in model.prompts(0))


async def test_an_unmarked_function_traces_with_no_injection() -> None:
    bound = Bind(plain)
    result = await bound.trace(query="q", overrides={"model": scripted()})
    assert result.inputs == []
    assert bound.local.names == ()


async def test_a_positional_argument_that_would_land_on_a_trainable_parameter_is_refused_before_the_model() -> None:
    """Positionals bind left to right and cannot skip a slot."""

    class Fronted(MethodAgent):
        name = "fronted"

        @ai_method(Choice, description="Choose, marked parameter first")
        def choose(self, playbook: Annotated[str, Trainable()] = "p", state: str = "S") -> Choice:
            """Playbook: {playbook}. State: {state}."""
            ...

    async with RuntimeHarness():
        model = choosing(0)
        with pytest.raises(ParamWiringError, match=r"left to right"):
            await Bind(Fronted()).trace("choose", "S0", overrides={"model": model})
        assert model.contexts == []


async def test_an_agent_trace_without_a_method_name_is_refused_before_the_model() -> None:
    async with RuntimeHarness():
        model = choosing(0)
        with pytest.raises(ParamWiringError, match=r"method name"):
            await Bind(Navigator()).trace(state="S0", overrides={"model": model})
        with pytest.raises(ParamWiringError, match=r"'nope'"):
            await Bind(Navigator()).trace("nope", state="S0", overrides={"model": model})
        assert model.contexts == []


# ── The gradient target: spied through TextGrad's own backward and consolidate ──


async def test_the_bound_value_is_a_gradient_target_and_consolidates_back_into_the_host() -> None:
    """The backward stub routes the feedback onto the Local's node; ``consolidate`` applies ORDERED and bumps
    the version, with no memory backend and no model call for the update."""
    bound = Bind(research)
    result = await bound.trace(query="cats", overrides={"model": scripted()})
    graph = await build_graph_from_result(result, [bound.local])

    opt = TextGradOptimizer()
    stub = EchoBackward()
    opt._backward_fn = cast("Any", stub)  # noqa: SLF001 -- offline stand-in for the model call
    opt.backward(graph, "Cite sources.")

    assert [g.text for g in graph.parameters[0].gradients] == ["Cite sources."]
    assert stub.calls == 1
    assert "instructions" in stub.inputs[0]

    opt.consolidate(graph)
    assert bound.local.load("instructions") == ("Find relevant facts.\nCite sources.", 1)


async def test_step_drives_the_whole_dance_over_a_local_host() -> None:
    """``TextGradOptimizer.step(result, feedback, backends=[local])``: the one-call form works for a Local."""
    bound = Bind(write)
    result = await bound.trace(facts="f", overrides={"model": scripted()})
    opt = TextGradOptimizer()
    opt._backward_fn = cast("Any", EchoBackward())  # noqa: SLF001
    await opt.step(result, "Shorter sentences.", backends=[bound.local])
    assert bound.local.load("style") == ("Plain prose.\nShorter sentences.", 1)


# ── Errors ──


def test_params_errors_are_ai_function_errors() -> None:
    assert issubclass(ParamsError, AIFunctionError)
    assert issubclass(ParamWiringError, ParamsError)
