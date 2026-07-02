"""Offline tests for memory + optimizer: recall emission, graph build, grouping.

None of these require a live model. They exercise the parts of the port that
are pure logic: ``recall`` event emission, ``build_graph`` reconstruction
(dedup, value, tool pairing, backend matching), the optimizer's
``topological_sort`` / ``consolidate`` grouping, and frozen / procedural
schema introspection.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest
from pydantic import BaseModel, Field, create_model
from strands import ToolContext, tool
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics

from ai_functions import (
    AgentCoreMemoryBackend,
    Frozen,
    JSONMemoryBackend,
    MemoryBackend,
    Procedural,
    TextGradOptimizer,
    ai_function,
    build_graph,
)
from ai_functions.optimizer._graph import topological_sort
from ai_functions.runtime import InMemoryCoordinator
from ai_functions.types import ParameterRecalledEvent, ThreadId
from ai_functions.types.events import (
    EventKind,
    MessageAssistantCompleteEvent,
    MessageUserEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ai_functions.types.graph import ParameterNode, ThreadNode


@tool(context=True)
def _request_state_writer(value: str, tool_context: ToolContext) -> str:
    """Test tool: write ``value`` into request_state and request loop stop."""
    rs = tool_context.invocation_state["request_state"]
    rs["python_executor_result"] = f"CAPTURED:{value}"
    rs["stop_event_loop"] = True
    return "ok"


# ── Schema ────────────────────────────────────────────────────────────────


class WritingMemory(BaseModel):
    joke_guidelines: str = Field("No specific guidelines yet.", description="Guidelines to write a good joke")
    formatting_guidelines: str = Field("No specific guidelines yet.", description="Email layout guidelines.")
    brand_voice: Frozen[str] = Field("Friendly, concise.", description="Fixed house style.")
    helpers: Procedural = Field(description="Helper functions available to the agent.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")


def _backend(tmp_path: Path) -> JSONMemoryBackend:
    return JSONMemoryBackend(WritingMemory, actor_id="w1", path=tmp_path / "mem.json")


# ── Schema introspection (frozen / procedural / description) ────────────────


def test_frozen_field_detected(tmp_path: Path) -> None:
    """A ``Frozen[T]`` field is recognized as frozen; plain fields are not."""
    mem = _backend(tmp_path)
    assert mem._is_frozen("brand_voice") is True  # noqa: SLF001
    assert mem._is_frozen("joke_guidelines") is False  # noqa: SLF001


def test_procedural_field_detected(tmp_path: Path) -> None:
    """A ``Procedural`` field is recognized; plain fields are not."""
    mem = _backend(tmp_path)
    assert mem._is_procedural("helpers") is True  # noqa: SLF001
    assert mem._is_procedural("joke_guidelines") is False  # noqa: SLF001


def test_description_introspection(tmp_path: Path) -> None:
    """Field descriptions are read from the schema."""
    mem = _backend(tmp_path)
    assert mem._get_description("joke_guidelines") == "Guidelines to write a good joke"  # noqa: SLF001


def test_backend_id_format(tmp_path: Path) -> None:
    """``backend_id`` is ``ClassName:actor_id``."""
    assert _backend(tmp_path).backend_id == "JSONMemoryBackend:w1"


def test_is_list_field(tmp_path: Path) -> None:
    """List fields are detected; scalar fields are not."""
    mem = _backend(tmp_path)
    assert mem._is_list_field("tags") is True  # noqa: SLF001
    assert mem._is_list_field("joke_guidelines") is False  # noqa: SLF001


# ── tool_provider: schema-driven tool generation ───────────────────────────


def test_tool_provider_names_scoped_by_field_type(tmp_path: Path) -> None:
    """search_* is list-only; save_*/delete_* are scalar-only; recall_*/query_* always."""
    mem = _backend(tmp_path)
    provider = mem.tool_provider("joke_guidelines", "tags")
    names = {t.tool_name for t in asyncio.run(provider.load_tools())}
    assert names == {
        "recall_joke_guidelines",
        "query_joke_guidelines",
        "save_joke_guidelines",
        "delete_joke_guidelines",
        "recall_tags",
        "query_tags",
        "search_tags",
    }


def test_tool_provider_operations_filter(tmp_path: Path) -> None:
    """``operations`` restricts which tools are generated (read-only here)."""
    mem = _backend(tmp_path)
    provider = mem.tool_provider("joke_guidelines", operations={"recall"})
    names = {t.tool_name for t in asyncio.run(provider.load_tools())}
    assert names == {"recall_joke_guidelines"}


def test_delete_resets_scalar_to_default(tmp_path: Path) -> None:
    """``delete`` restores a scalar field to its schema default."""
    mem = _backend(tmp_path)
    mem.save("joke_guidelines", "changed")
    assert mem._recall("joke_guidelines") == "changed"  # noqa: SLF001
    mem.delete("joke_guidelines")
    assert mem._recall("joke_guidelines") == "No specific guidelines yet."  # noqa: SLF001


def test_delete_required_field_raises(tmp_path: Path) -> None:
    """Deleting a required field with no default is an explicit error."""
    import json

    class _Required(BaseModel):
        needed: str = Field(description="A required field with no default.")

    # Seed the file so the backend can hydrate the required field on load.
    path = tmp_path / "req.json"
    path.write_text(json.dumps({"r1": {"needed": "value"}}))
    mem = JSONMemoryBackend(_Required, actor_id="r1", path=path)
    with pytest.raises(ValueError, match="no schema default"):
        mem.delete("needed")


class _Inner(BaseModel):
    val: str = Field("x", description="inner val")


class _NestedSchema(BaseModel):
    plain: _Inner = Field(default_factory=_Inner)
    optional: _Inner | None = Field(default=None)


def test_resolve_nested_field_through_plain_and_optional(tmp_path: Path) -> None:
    """Nested a/b paths resolve through both plain and Optional[Model] intermediates.

    An Optional/Union intermediate annotation (``_Inner | None``) has no
    ``model_fields``; the resolver must unwrap the union to the model member.
    """
    mem = JSONMemoryBackend(_NestedSchema, actor_id="n1", path=tmp_path / "n.json")
    assert mem._get_description("plain/val") == "inner val"  # noqa: SLF001
    assert mem._get_description("optional/val") == "inner val"  # noqa: SLF001


def test_resolve_nested_field_through_non_model_raises(tmp_path: Path) -> None:
    """A nested path through a non-model intermediate raises a clear error."""
    mem = _backend(tmp_path)
    with pytest.raises(TypeError, match="not a Pydantic model"):
        mem._get_description("joke_guidelines/nope")  # noqa: SLF001 -- str field, can't nest


class _DocItem(BaseModel):
    id: int = 0
    text: str = ""


class _ModelListSchema(BaseModel):
    items: list[_DocItem] = Field(default_factory=list)


class _NestedListSchema(BaseModel):
    groups: list[list[_DocItem]] = Field(default_factory=list)


def test_consolidate_rejects_non_string_list(tmp_path: Path) -> None:
    """Consolidating a list[BaseModel] field raises rather than storing list[str].

    The list consolidator is typed list[str]; routing a model list through it
    would silently corrupt the field (fails to reload). It must be rejected.
    """
    mem = JSONMemoryBackend(_ModelListSchema, actor_id="m1", path=tmp_path / "m.json")
    mem.save("items", [_DocItem(id=1, text="a")])
    with pytest.raises(NotImplementedError, match="non-string list"):
        mem._consolidate("items", ["feedback"])  # noqa: SLF001


def test_deserialize_value_symmetric_for_nested_models(tmp_path: Path) -> None:
    """deserialize_value rehydrates arbitrarily nested model shapes (symmetric).

    list[list[Model]] must come back with inner elements as models, not dicts.
    """
    mem = JSONMemoryBackend(_NestedListSchema, actor_id="n1", path=tmp_path / "n.json")
    out = mem.deserialize_value("groups", [[{"id": 1, "text": "x"}]])
    assert isinstance(out[0][0], _DocItem)
    assert out[0][0].id == 1


# ── recall(): pure fetch vs. immediate emission ─────────────────────────────


async def test_recall_without_coordinator_emits_nothing(tmp_path: Path) -> None:
    """A bare recall is a pure fetch and returns the stored value unchanged."""
    mem = _backend(tmp_path)
    value = await mem.recall("joke_guidelines")
    assert value == "No specific guidelines yet."


async def test_recall_emits_event_immediately(tmp_path: Path) -> None:
    """recall(coordinator, thread_id) appends one ParameterRecalledEvent now."""
    mem = _backend(tmp_path)
    coord = InMemoryCoordinator()
    tid = ThreadId("joke-1")

    value = await mem.recall("joke_guidelines", coordinator=coord, thread_id=tid)

    assert value == "No specific guidelines yet."
    events = coord._events[tid]  # noqa: SLF001 -- inspecting the log directly
    assert len(events) == 1
    evt = events[0]
    assert isinstance(evt, ParameterRecalledEvent)
    assert evt.name == "joke_guidelines"
    assert evt.derivation == "full"
    assert evt.requires_grad is True
    assert evt.backend_id == "JSONMemoryBackend:w1"
    assert evt.thread_id == tid


async def test_recall_before_thread_exists(tmp_path: Path) -> None:
    """The event log is created on demand: recall works before any thread spawns."""
    mem = _backend(tmp_path)
    coord = InMemoryCoordinator()
    tid = ThreadId("not-yet-spawned")

    await mem.recall("joke_guidelines", coordinator=coord, thread_id=tid)

    assert tid in coord._events  # noqa: SLF001


async def test_recall_frozen_defaults_requires_grad_false(tmp_path: Path) -> None:
    """A frozen field recalls with requires_grad=False by default."""
    mem = _backend(tmp_path)
    coord = InMemoryCoordinator()
    tid = ThreadId("t")

    await mem.recall("brand_voice", coordinator=coord, thread_id=tid)

    assert coord._events[tid][0].requires_grad is False  # noqa: SLF001


async def test_recall_requires_grad_override(tmp_path: Path) -> None:
    """An explicit requires_grad overrides the frozen default."""
    mem = _backend(tmp_path)
    coord = InMemoryCoordinator()
    tid = ThreadId("t")

    await mem.recall("brand_voice", coordinator=coord, thread_id=tid, requires_grad=True)

    assert coord._events[tid][0].requires_grad is True  # noqa: SLF001


async def test_recall_confirms_deferred_append_before_returning(tmp_path: Path) -> None:
    """recall awaits durability when append_event defers the write (I8).

    Simulates a network-backed coordinator: ``append_event`` schedules the
    write on the event loop (like ``CoordinatorClient``) instead of applying
    it synchronously. ``recall`` must not return until ``get_events`` shows the
    event — otherwise a later ``build_graph`` read could overtake the write.
    """

    class _DeferredCoordinator(InMemoryCoordinator):
        def append_event(self, event: object) -> None:  # type: ignore[override]
            async def _later() -> None:
                await asyncio.sleep(0)  # land only after the loop cycles
                super(_DeferredCoordinator, self).append_event(event)  # type: ignore[arg-type]

            _ = asyncio.create_task(_later())

    coord = _DeferredCoordinator()
    tid = ThreadId("deferred-1")
    mem = _backend(tmp_path)

    await mem.recall("joke_guidelines", coordinator=coord, thread_id=tid)

    # By the time recall returns, the deferred append must be visible.
    events = await coord.get_events(tid, kinds=[EventKind.PARAMETER_RECALLED])
    assert len(events) == 1


# ── build_graph: reconstruction from a hand-built event log ─────────────────


def _recall_event(backend: MemoryBackend, name: str, tid: str, **over: object) -> ParameterRecalledEvent:
    return ParameterRecalledEvent(
        thread_id=ThreadId(tid),
        thread_name=over.get("thread_name", "joke_writer"),  # type: ignore[arg-type]
        name=name,
        value=over.get("value", "some guidelines"),
        derivation=over.get("derivation", "full"),  # type: ignore[arg-type]
        requires_grad=over.get("requires_grad", True),  # type: ignore[arg-type]
        backend_id=backend.backend_id,
        description=over.get("description", ""),  # type: ignore[arg-type]
    )


def test_build_graph_single_parameter(tmp_path: Path) -> None:
    """A recall event becomes a ParameterNode referencing the live backend."""
    mem = _backend(tmp_path)
    events = [
        _recall_event(mem, "joke_guidelines", "joke-1"),
        MessageUserEvent(thread_id=ThreadId("joke-1"), text="write a joke"),
        MessageAssistantCompleteEvent(
            thread_id=ThreadId("joke-1"),
            content=[{"text": "Why did the cat..."}],
        ),
    ]

    node = build_graph(events, [mem])

    assert node.thread_id == "joke-1"
    assert node.func_name == "joke_writer"
    assert len(node.parameters) == 1
    p = node.parameters[0]
    assert p.name == "joke_guidelines"
    assert p.backend is mem
    assert p.requires_grad is True
    assert node.value == "Why did the cat..."
    assert node.child_threads == []


def test_build_graph_dedup_last_write_wins(tmp_path: Path) -> None:
    """The same (backend, name) recalled twice yields one node; latest wins."""
    mem = _backend(tmp_path)
    events = [
        _recall_event(mem, "joke_guidelines", "joke-1", value="old", requires_grad=True),
        _recall_event(mem, "joke_guidelines", "joke-1", value="new", requires_grad=False),
    ]

    node = build_graph(events, [mem])

    assert len(node.parameters) == 1
    p = node.parameters[0]
    assert p.value == "new"
    assert p.requires_grad is False


def test_build_graph_unknown_backend_skipped(tmp_path: Path) -> None:
    """A recall event whose backend_id matches no backend is skipped."""
    mem = _backend(tmp_path)
    events = [
        ParameterRecalledEvent(
            thread_id=ThreadId("t"),
            name="joke_guidelines",
            value="x",
            backend_id="JSONMemoryBackend:someone-else",
        )
    ]

    node = build_graph(events, [mem])

    assert node.parameters == []


def test_build_graph_tool_calls_paired(tmp_path: Path) -> None:
    """ToolCall + ToolResult events pair by tool_use_id into one ToolCallNode."""
    mem = _backend(tmp_path)
    events = [
        ToolCallEvent(
            thread_id=ThreadId("t"),
            tool_use_id="tu-1",
            tool_name="search",
            arguments={"q": "cats"},
        ),
        ToolResultEvent(
            thread_id=ThreadId("t"),
            tool_use_id="tu-1",
            status="success",
            content=[{"text": "result text"}],
        ),
    ]

    node = build_graph(events, [mem])

    assert len(node.tool_calls) == 1
    tc = node.tool_calls[0]
    assert tc.tool_name == "search"
    assert tc.arguments == {"q": "cats"}
    assert tc.result == "result text"
    assert tc.status == "success"


def test_build_graph_value_none_without_assistant_turn(tmp_path: Path) -> None:
    """With no assistant turn, the node value is None."""
    mem = _backend(tmp_path)
    node = build_graph([_recall_event(mem, "joke_guidelines", "t")], [mem])
    assert node.value is None


# ── topological_sort ────────────────────────────────────────────────────────


def _grad_node(name: str) -> ThreadNode:
    return ThreadNode(
        node_id=name,
        thread_id=name,
        parameters=[ParameterNode(node_id=f"{name}-p", requires_grad=True)],
    )


def test_topological_sort_root_first_then_children() -> None:
    """Root is visited before its children (feedback flows root -> children)."""
    root = ThreadNode(node_id="root", thread_id="root")
    c1, c2 = _grad_node("c1"), _grad_node("c2")
    root.child_threads = [c1, c2]

    order = topological_sort(root)

    assert order[0] is root
    assert {id(n) for n in order[1:]} == {id(c1), id(c2)}


def test_topological_sort_prunes_gradless_subtrees() -> None:
    """A child subtree with no grad-enabled parameter is pruned."""
    root = _grad_node("root")
    gradless = ThreadNode(node_id="c", thread_id="c")  # no parameters
    root.child_threads = [gradless]

    order = topological_sort(root)

    assert gradless not in order
    assert root in order


def test_topological_sort_diamond_visits_shared_child_once() -> None:
    """A child reached via two parents appears exactly once (no double-visit)."""
    root = ThreadNode(node_id="root", thread_id="root")
    left, right = _grad_node("left"), _grad_node("right")
    shared = _grad_node("shared")
    left.child_threads = [shared]
    right.child_threads = [shared]
    root.child_threads = [left, right]

    order = topological_sort(root)

    assert sum(1 for n in order if n is shared) == 1


# ── optimizer.consolidate grouping (no model needed) ────────────────────────


class _RecordingBackend(MemoryBackend):
    """A backend that records consolidate() calls instead of running a model."""

    def __init__(self, schema: type[BaseModel], actor_id: str) -> None:
        super().__init__(schema, actor_id)
        self.calls: list[tuple[str, list[str]]] = []

    def close(self) -> None: ...
    def _save(self, name: str, value: object) -> None: ...
    def _recall(self, name: str) -> object:
        return ""

    def _query(self, name: str, query: str) -> str:
        return ""

    def _search(self, name: str, query: str, k: int = 5, **kwargs: object) -> object:
        return []

    def _consolidate(self, name: str, feedback: list[str], **kwargs: object) -> None:
        self.calls.append((name, feedback))

    def _delete(self, name: str) -> None: ...


def test_consolidate_groups_by_backend_and_name() -> None:
    """One consolidate call per (backend, name), merging gradients across nodes."""
    backend = _RecordingBackend(WritingMemory, "w1")
    optimizer = TextGradOptimizer()

    # Same parameter referenced from two nodes (root + child) with gradients.
    p_root = ParameterNode(node_id="r-p", name="joke_guidelines", backend=backend, gradients=["fb-a"])
    p_child = ParameterNode(node_id="c-p", name="joke_guidelines", backend=backend, gradients=["fb-b"])
    root = ThreadNode(node_id="root", thread_id="root", parameters=[p_root])
    child = ThreadNode(node_id="c", thread_id="c", parameters=[p_child])
    root.child_threads = [child]

    optimizer.consolidate(root)

    assert len(backend.calls) == 1
    name, feedback = backend.calls[0]
    assert name == "joke_guidelines"
    assert set(feedback) == {"fb-a", "fb-b"}


def test_consolidate_skips_parameters_without_gradients() -> None:
    """Parameters with no accumulated gradients are not consolidated."""
    backend = _RecordingBackend(WritingMemory, "w1")
    optimizer = TextGradOptimizer()
    p = ParameterNode(node_id="p", name="joke_guidelines", backend=backend, gradients=[])
    root = ThreadNode(node_id="root", thread_id="root", parameters=[p])

    optimizer.consolidate(root)

    assert backend.calls == []


def test_zero_grad_clears_all_gradients() -> None:
    """zero_grad empties node and parameter gradients across the graph."""
    backend = _RecordingBackend(WritingMemory, "w1")
    optimizer = TextGradOptimizer()
    p = ParameterNode(node_id="p", name="joke_guidelines", backend=backend, gradients=["x"])
    root = ThreadNode(node_id="root", thread_id="root", parameters=[p], gradients=["y"])

    optimizer.zero_grad(root)

    assert root.gradients == []
    assert p.gradients == []


# ── procedural validation ───────────────────────────────────────────────────


def test_validate_procedural_accepts_valid_code() -> None:
    """Parseable code passes through unchanged."""
    from ai_functions.memory.procedural import validate_procedural

    code = "def f(x):\n    return x + 1\n"
    assert validate_procedural(code) == code


def test_validate_procedural_rejects_invalid_code() -> None:
    """Unparseable code raises SyntaxError."""
    from ai_functions.memory.procedural import validate_procedural

    with pytest.raises(SyntaxError):
        validate_procedural("def f(:\n")


def test_validate_procedural_allows_empty() -> None:
    """Empty / whitespace-only code is returned without parsing."""
    from ai_functions.memory.procedural import validate_procedural

    assert validate_procedural("   ") == "   "


# ── JSON backend persistence ────────────────────────────────────────────────


async def test_json_backend_save_and_persist(tmp_path: Path) -> None:
    """save() then close() persists values; a fresh backend reloads them."""
    path = tmp_path / "mem.json"
    mem = JSONMemoryBackend(WritingMemory, actor_id="w1", path=path)
    mem.save("joke_guidelines", "Always about cats.")
    mem.close()

    reloaded = JSONMemoryBackend(WritingMemory, actor_id="w1", path=path)
    assert await reloaded.recall("joke_guidelines") == "Always about cats."


def test_json_backend_str_renders_multiline_as_literal_block(tmp_path: Path) -> None:
    """str(memory) renders multi-line code as a YAML literal block, not escaped."""
    mem = _backend(tmp_path)
    mem.save("helpers", "def greet(name):\n    return f'Hello, {name}'\n")

    rendered = str(mem)

    assert "helpers: |" in rendered  # literal block, not a quoted scalar
    assert "\\n" not in rendered  # no escaped newlines


async def test_json_backend_namespaces_actors(tmp_path: Path) -> None:
    """Two actors share one file without clobbering each other."""
    path = tmp_path / "mem.json"
    a = JSONMemoryBackend(WritingMemory, actor_id="a", path=path)
    a.save("joke_guidelines", "actor-a value")
    a.close()
    b = JSONMemoryBackend(WritingMemory, actor_id="b", path=path)
    b.save("joke_guidelines", "actor-b value")
    b.close()

    assert await JSONMemoryBackend(WritingMemory, "a", path).recall("joke_guidelines") == "actor-a value"
    assert await JSONMemoryBackend(WritingMemory, "b", path).recall("joke_guidelines") == "actor-b value"


async def test_json_backend_concurrent_open_no_clobber(tmp_path: Path) -> None:
    """Two backends opened on one file before either closes must not clobber.

    Both open the (empty) file, so each holds a snapshot missing the other.
    close() must re-read + merge its own key, not write the stale snapshot —
    otherwise the later close() destroys the earlier actor's data.
    """
    path = tmp_path / "mem.json"
    a = JSONMemoryBackend(WritingMemory, actor_id="a", path=path)
    b = JSONMemoryBackend(WritingMemory, actor_id="b", path=path)
    a.save("joke_guidelines", "actor-a value")
    b.save("joke_guidelines", "actor-b value")

    b.close()
    a.close()  # would previously overwrite the file and drop actor "b"

    assert await JSONMemoryBackend(WritingMemory, "a", path).recall("joke_guidelines") == "actor-a value"
    assert await JSONMemoryBackend(WritingMemory, "b", path).recall("joke_guidelines") == "actor-b value"


async def test_json_backend_search_empty_list(tmp_path: Path) -> None:
    """Searching an empty list parameter returns no results."""
    mem = _backend(tmp_path)
    assert await mem.search("tags", "anything") == []


async def test_json_backend_search_rejects_scalar(tmp_path: Path) -> None:
    """Search is only valid on list parameters; a scalar field raises TypeError."""
    mem = _backend(tmp_path)
    with pytest.raises(TypeError, match="list parameters"):
        await mem.search("joke_guidelines", "query")


async def test_json_backend_search_ranks_by_bm25(tmp_path: Path) -> None:
    """BM25 ranks the most query-relevant entries first and honours k."""
    mem = _backend(tmp_path)
    mem.save(
        "tags",
        [
            "season pasta water generously with salt",
            "let meat rest before slicing",
            "add a pinch of sugar to tomato sauces",
            "toast spices in a dry pan",
        ],
    )

    top = await mem.search("tags", "tomato sauce sugar", k=2)

    assert len(top) == 2
    assert top[0] == "add a pinch of sugar to tomato sauces"


async def test_json_backend_search_emits_event(tmp_path: Path) -> None:
    """A tracked search emits one ParameterRecalledEvent with derivation='search'."""
    mem = _backend(tmp_path)
    mem.save("tags", ["alpha beta", "gamma delta"])
    coord = InMemoryCoordinator()
    tid = ThreadId("s-1")

    await mem.search("tags", "alpha", k=1, coordinator=coord, thread_id=tid)

    events = coord._events[tid]  # noqa: SLF001
    assert len(events) == 1
    assert events[0].derivation == "search"
    assert events[0].meta["query"] == "alpha"
    assert events[0].meta["top_k"] == 1


# ── AgentCore backend (offline parts only) ──────────────────────────────────


_HAS_AGENTCORE = importlib.util.find_spec("bedrock_agentcore") is not None


class _ProceduralSchema(BaseModel):
    code: Procedural = Field(description="Some helper code.")


class _PlainSchema(BaseModel):
    note: str = Field("", description="A plain text note.")


@pytest.mark.skipif(_HAS_AGENTCORE, reason="exercises the missing-dependency path only")
def test_agentcore_requires_dependency() -> None:
    """Constructing the backend without bedrock-agentcore raises a clear ImportError."""
    with pytest.raises(ImportError, match="bedrock-agentcore"):
        AgentCoreMemoryBackend(WritingMemory, actor_id="a", memory_name="m")


@pytest.mark.skipif(not _HAS_AGENTCORE, reason="requires bedrock-agentcore installed")
def test_agentcore_rejects_procedural_fields() -> None:
    """A schema with a Procedural field is rejected before any AWS call."""
    with pytest.raises(ValueError, match="Procedural"):
        AgentCoreMemoryBackend(_ProceduralSchema, actor_id="a", memory_name="m")


@pytest.mark.skipif(not _HAS_AGENTCORE, reason="requires bedrock-agentcore installed")
def test_agentcore_requires_one_of_memory_id_or_name() -> None:
    """Exactly one of memory_id / memory_name must be provided."""
    with pytest.raises(ValueError, match="Either memory_id or memory_name"):
        AgentCoreMemoryBackend(_PlainSchema, actor_id="a")


@pytest.mark.skipif(not _HAS_AGENTCORE, reason="requires bedrock-agentcore installed")
def test_agentcore_rejects_both_memory_id_and_name() -> None:
    """Providing both memory_id and memory_name is rejected."""
    with pytest.raises(ValueError, match="Cannot provide both"):
        AgentCoreMemoryBackend(_PlainSchema, actor_id="a", memory_id="m-1", memory_name="m")


# ── Procedural code execution (offline parts only) ──────────────────────────

_HAS_SMOLAGENTS = importlib.util.find_spec("smolagents") is not None


@pytest.mark.skipif(_HAS_SMOLAGENTS, reason="exercises the missing-dependency path only")
def test_executor_requires_smolagents() -> None:
    """Constructing the executor without smolagents raises a clear ImportError."""
    from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

    answer_model = create_model("FinalAnswer", answer=(str, Field(...)))
    with pytest.raises(ImportError, match="smolagents"):
        LocalPythonExecutorTool(output_type=answer_model)


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="requires smolagents installed")
def test_executor_runs_code_and_captures_final_answer() -> None:
    """The sandboxed executor runs code and captures the final_answer payload."""
    from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

    answer_model = create_model("FinalAnswer", answer=(str, Field(...)))
    tool = LocalPythonExecutorTool(output_type=answer_model)

    result = tool._execute_code("x = 6 * 7\nfinal_answer(str(x))")  # noqa: SLF001

    assert result.success is True
    assert result.final_answer == {"answer": "42"}


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="requires smolagents installed")
def test_executor_defines_procedural_code() -> None:
    """Code passed via initial_code is DEFINED in the namespace and callable."""
    from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

    answer_model = create_model("FinalAnswer", answer=(str, Field(...)))
    # initial_code runs at setup so `shout` becomes a callable in the namespace.
    tool = LocalPythonExecutorTool(
        output_type=answer_model,
        initial_code=["def shout(s):\n    return s.upper()\n"],
    )

    result = tool._execute_code("final_answer(shout('hi'))")  # noqa: SLF001

    assert result.success is True
    assert result.final_answer == {"answer": "HI"}


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="requires smolagents installed")
def test_executor_reports_errors() -> None:
    """A runtime error in executed code is captured, not raised."""
    from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

    answer_model = create_model("FinalAnswer", answer=(str, Field(...)))
    tool = LocalPythonExecutorTool(output_type=answer_model)

    result = tool._execute_code("1 / 0")  # noqa: SLF001

    assert result.success is False
    assert result.error is not None


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="requires smolagents installed")
def test_executor_raises_on_bad_initial_code() -> None:
    """Malformed/erroring procedural setup code surfaces, not silently dropped."""
    from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

    answer_model = create_model("FinalAnswer", answer=(str, Field(...)))
    with pytest.raises(ValueError, match="Failed to load procedural code"):
        # references an undefined name at setup -> execution error
        LocalPythonExecutorTool(output_type=answer_model, initial_code=["undefined_name_xyz()"])


async def test_strands_request_state_surfaces_as_agent_result_state() -> None:
    """Lock in the Strands contract the executor result path depends on.

    The whole code-execution result path assumes a tool writing into
    ``invocation_state["request_state"]`` surfaces as ``AgentResult.state`` and
    that ``stop_event_loop`` halts the loop. This is a Strands-internal contract;
    a Strands upgrade could break result extraction with no other failing test.
    This pins it directly (offline, via a scripted model that calls the tool).
    """
    from strands import Agent

    from ai_functions.testing import ScriptedModel, Turn

    # Two turns: the tool call, then a benign end_turn in case the loop checks
    # stop_event_loop only at the next boundary.
    model = ScriptedModel(
        [
            Turn(tool_calls=(("_request_state_writer", {"value": "hi"}),)),
            Turn(text="done"),
        ]
    )
    agent = Agent(model=model, tools=[_request_state_writer])
    result = await agent.invoke_async("call writer")

    state = result.state or {}
    assert state.get("python_executor_result") == "CAPTURED:hi"
    assert state.get("stop_event_loop") is True


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="requires smolagents installed")
def test_executor_empty_final_answer_is_distinguished_from_no_call() -> None:
    """final_answer() with no args yields {} (a real call), not None (no call).

    The tool must treat {} as an attempted-but-invalid answer (construction
    fails on the missing required field) rather than dropping it like a missing
    final_answer.
    """
    from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

    answer_model = create_model("FinalAnswer", answer=(str, Field(...)))
    tool = LocalPythonExecutorTool(output_type=answer_model)

    # No final_answer call at all -> None.
    no_call = tool._execute_code("x = 1")  # noqa: SLF001
    assert no_call.final_answer is None

    # final_answer() with no args -> {} (called, but missing required 'answer').
    empty = tool._execute_code("final_answer()")  # noqa: SLF001
    assert empty.final_answer == {}
    # Constructing the output from {} must fail (missing required field), which
    # the tool surfaces to the model rather than silently dropping.
    with pytest.raises(Exception):  # noqa: B017, PT011 -- pydantic ValidationError
        answer_model(**empty.final_answer)


def test_code_execution_mode_defaults_disabled() -> None:
    """code_execution_mode defaults to DISABLED (opt-in)."""
    from ai_functions.ai_thread.config import CodeExecutionMode, ThreadConfig

    assert ThreadConfig().code_execution_mode == CodeExecutionMode.DISABLED


def test_ai_function_accepts_code_execution_mode() -> None:
    """@ai_function(code_execution_mode='local') sets the config field."""
    from ai_functions.ai_thread.config import CodeExecutionMode

    @ai_function(str, code_execution_mode="local")
    def _fn(helpers: str):
        """Use {helpers}."""

    assert _fn.config.code_execution_mode == CodeExecutionMode.LOCAL


def test_procedural_param_detection() -> None:
    """_procedural_param_names detects Procedural-typed params, ignores plain str.

    This is the central wiring: only params the thread reports as procedural get
    their code DEFINED in the executor namespace (via initial_code). A plain str
    param would land in initial_state as an inert string the sandbox can't exec.
    """

    @ai_function(str, code_execution_mode="local")
    def proc_fn(helper_functions: Procedural, topic: str):
        """Use {helper_functions} for {topic}."""

    @ai_function(str, code_execution_mode="local")
    def str_fn(helper_functions: str):
        """Use {helper_functions}."""

    assert proc_fn.to_thread()._procedural_param_names() == {"helper_functions"}  # noqa: SLF001
    assert str_fn.to_thread()._procedural_param_names() == set()  # noqa: SLF001


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="requires smolagents installed")
def test_with_python_executor_is_fresh_per_attempt() -> None:
    """Each _with_python_executor call yields an independent sandbox.

    The smolagents namespace persists across calls on one executor, so retries
    must rebuild it. This asserts a fresh executor re-defines the Procedural
    helper (callable) but does NOT carry ad-hoc state from a prior attempt.
    """
    from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

    def _build() -> LocalPythonExecutorTool:
        answer_model = create_model("FinalAnswer", answer=(str, Field(...)))
        return LocalPythonExecutorTool(
            output_type=answer_model,
            initial_code=["def helper():\n    return 'OK'\n"],
        )

    attempt1 = _build()
    attempt1._execute_code("leaked = 'STALE'")  # noqa: SLF001 -- failed attempt's ad-hoc state

    attempt2 = _build()  # a fresh attempt
    # The helper is re-defined (callable)...
    ok = attempt2._execute_code("final_answer(helper())")  # noqa: SLF001
    assert ok.final_answer == {"answer": "OK"}
    # ...but the prior attempt's ad-hoc variable is gone.
    leaked = attempt2._execute_code("final_answer(leaked)")  # noqa: SLF001
    assert leaked.success is False


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="requires smolagents installed")
def test_procedural_param_code_becomes_callable() -> None:
    """End-to-end wiring (offline): a Procedural param's recalled code is callable.

    Replicates the split _run_cycle performs — procedural-typed args go to
    initial_code (defined at setup), the rest to initial_state — and proves the
    helper is then callable. This covers the integration path that the live
    example exercises, without a model.
    """
    from ai_functions.tools.local_python_executor import LocalPythonExecutorTool

    @ai_function(str, code_execution_mode="local")
    def proc_fn(helper_functions: Procedural, topic: str):
        """Use {helper_functions} for {topic}."""

    thread = proc_fn.to_thread()
    bound = thread._bind_args(helper_functions="def shout(s):\n    return s.upper()\n", topic="x")  # noqa: SLF001
    procedural = thread._procedural_param_names()  # noqa: SLF001
    initial_code = [str(v) for k, v in bound.items() if k in procedural and isinstance(v, str)]
    initial_state = {k: v for k, v in bound.items() if k not in procedural}

    answer_model = create_model("FinalAnswer", answer=(str, Field(...)))
    tool = LocalPythonExecutorTool(output_type=answer_model, initial_state=initial_state, initial_code=initial_code)

    result = tool._execute_code("final_answer(shout('hi'))")  # noqa: SLF001

    assert result.success is True
    assert result.final_answer == {"answer": "HI"}


# ── _with_python_executor: mode gating + structured-output guard ─────────────


def _agent_result(
    structured: BaseModel | None = None,
    state: dict[str, object] | None = None,
) -> AgentResult:
    """Build a minimal Strands ``AgentResult`` for offline _extract_result tests."""
    return AgentResult(
        stop_reason="end_turn",
        message={"role": "assistant", "content": [{"text": "x"}]},
        metrics=EventLoopMetrics(),
        state=state or {},
        structured_output=structured,
    )


def test_with_python_executor_noop_when_disabled() -> None:
    """code_execution_mode != LOCAL returns the config unchanged (no executor tool)."""

    @ai_function(str)
    def fn(helpers: str):
        """Use {helpers}."""

    thread = fn.to_thread()
    cfg = thread.config
    # DISABLED is the default -> identical config object, no tool appended.
    assert thread._with_python_executor(cfg) is cfg  # noqa: SLF001


def test_with_python_executor_rejects_plain_str_return() -> None:
    """code_execution_mode=LOCAL with a plain str return raises a clear error.

    The python_executor's final_answer needs a typed model; a bare str return
    has none (structured_output_model is None), so the guard must fail fast.
    """
    from ai_functions.ai_thread.errors import AIFunctionError

    @ai_function(str, code_execution_mode="local", structured_output=False)
    def fn(helpers: str):
        """Use {helpers}."""

    thread = fn.to_thread()
    with pytest.raises(AIFunctionError, match="plain str return type"):
        thread._with_python_executor(thread.config)  # noqa: SLF001


def test_non_json_serializable_return_disables_structured_output() -> None:
    """A non-JSON-serializable return type is wrapped but not structured.

    Mirrors the predecessor: is_json_serializable_type gates structured output.
    The wrapper model is still built (the executor needs it for final_answer),
    but structured output is off so Strands is never asked for a JSON schema.
    """
    from ai_functions.ai_thread.ai_thread import OutputSpec

    class _Unserializable:
        pass

    spec = OutputSpec.from_type(_Unserializable, is_structured=True)
    assert spec.is_structured is False  # not JSON-serializable
    assert spec.is_wrapped is True
    assert spec.structured_output_model is not None  # kept for final_answer


def test_serialize_result_degrades_for_non_serializable() -> None:
    """serialize_result falls back to str() for a non-JSON-serializable result."""

    class _Thing:
        def __str__(self) -> str:
            return "THING-REPR"

    @ai_function(_Thing, code_execution_mode="local")
    def fn(x: str):
        """Make a thing from {x}."""

    thread = fn.to_thread()
    # Never raises; records a best-effort string for the ResultEvent.
    assert thread.serialize_result(_Thing()) == "THING-REPR"


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="requires smolagents installed")
def test_with_python_executor_appends_executor_tool() -> None:
    """LOCAL mode appends exactly one python_executor tool to the cycle config."""

    @ai_function(str, code_execution_mode="local")
    def fn(helpers: Procedural, topic: str):
        """Use {helpers} for {topic}."""

    thread = fn.to_thread()
    thread._bound_args = thread._bind_args(  # noqa: SLF001
        helpers="def shout(s):\n    return s.upper()\n", topic="x"
    )
    new_cfg = thread._with_python_executor(thread.config)  # noqa: SLF001

    assert len(new_cfg.tools) == len(thread.config.tools) + 1
    appended = new_cfg.tools[-1]
    assert getattr(appended, "tool_name", None) == "python_executor"
    # The source config is untouched (a fresh config per attempt).
    assert len(thread.config.tools) == len(new_cfg.tools) - 1


# ── _extract_result: structured / executor / error branches ─────────────────


def test_extract_result_local_prefers_executor_over_structured() -> None:
    """In LOCAL mode an executor final_answer shadows a stray structured output.

    A late structured tool call must not override the answer the model
    explicitly committed to via final_answer(...).
    """

    class Answer(BaseModel):
        answer: str = Field(...)

    @ai_function(Answer, code_execution_mode="local")
    def fn(topic: str):
        """About {topic}."""

    thread = fn.to_thread()
    response = _agent_result(
        structured=Answer(answer="from-structured"),
        state={"python_executor_result": Answer(answer="from-executor")},
    )
    result = thread._extract_result(response, response.state)  # noqa: SLF001
    assert result.answer == "from-executor"


def test_extract_result_non_local_uses_structured_output() -> None:
    """Without LOCAL mode the structured output wins over any executor state."""

    class Answer(BaseModel):
        answer: str = Field(...)

    @ai_function(Answer)
    def fn(topic: str):
        """About {topic}."""

    thread = fn.to_thread()
    response = _agent_result(
        structured=Answer(answer="from-structured"),
        state={"python_executor_result": Answer(answer="from-executor")},
    )
    result = thread._extract_result(response, response.state)  # noqa: SLF001
    assert result.answer == "from-structured"


def test_extract_result_falls_back_to_executor_when_structured_missing() -> None:
    """When structured output is absent the executor result is used as fallback."""

    class Answer(BaseModel):
        answer: str = Field(...)

    @ai_function(Answer)
    def fn(topic: str):
        """About {topic}."""

    thread = fn.to_thread()
    response = _agent_result(
        structured=None,
        state={"python_executor_result": Answer(answer="from-executor")},
    )
    result = thread._extract_result(response, response.state)  # noqa: SLF001
    assert result.answer == "from-executor"


def test_extract_result_unwraps_wrapped_executor_answer() -> None:
    """A wrapped (FinalAnswer) executor result is unwrapped to the bare value.

    Non-pydantic output types are wrapped in a FinalAnswer model; the extracted
    result must be the inner ``answer`` value, not the wrapper.
    """

    @ai_function(int, code_execution_mode="local")
    def fn(topic: str):
        """About {topic}."""

    thread = fn.to_thread()
    wrapped_model = thread._output_spec.structured_output_model  # noqa: SLF001
    assert wrapped_model is not None
    assert thread._output_spec.is_wrapped is True  # noqa: SLF001
    response = _agent_result(
        structured=None,
        state={"python_executor_result": wrapped_model(answer=42)},
    )
    assert thread._extract_result(response, response.state) == 42  # noqa: SLF001


def test_extract_result_raises_when_neither_present() -> None:
    """No structured output and no executor result is a hard error, not None."""
    from ai_functions.ai_thread.errors import AIFunctionError

    class Answer(BaseModel):
        answer: str = Field(...)

    @ai_function(Answer, code_execution_mode="local")
    def fn(topic: str):
        """About {topic}."""

    thread = fn.to_thread()
    response = _agent_result(structured=None, state={})
    with pytest.raises(AIFunctionError, match="neither a structured"):
        thread._extract_result(response, response.state)  # noqa: SLF001


# ── Targeted regression tests for review findings ───────────────────────────


def test_agentcore_memory_id_matches_hyphenated_name() -> None:
    """_memory_id_matches handles AgentCore's '{name}-{hash}' id format."""
    from ai_functions.memory.agentcore_backend import _memory_id_matches

    assert _memory_id_matches("writing-memory-abc123", "writing-memory") is True
    assert _memory_id_matches("writing-memory", "writing-memory") is True
    # Distinct names sharing a first token must NOT collide.
    assert _memory_id_matches("writing-a-x", "writing-b") is False
    assert _memory_id_matches("other-mem-x", "writing-memory") is False


def test_bind_args_keeps_positional_inputs() -> None:
    """_bind_args names positional args by param even on the fallback path."""

    @ai_function(str)
    def fn(helper_functions: str, topic: str):
        """Use {helper_functions} for {topic}."""

    thread = fn.to_thread()
    bound = thread._bind_args("code-here", "the-topic")  # noqa: SLF001 -- positional
    assert bound["helper_functions"] == "code-here"
    assert bound["topic"] == "the-topic"


def test_render_messages_skips_non_dict_blocks() -> None:
    """render_messages tolerates non-dict content blocks (no crash)."""
    from ai_functions.optimizer.rendering import render_messages

    messages = [
        {"role": "assistant", "content": [None, "stray", {"text": "real"}]},
    ]
    out = render_messages(messages)  # type: ignore[arg-type] -- intentionally malformed
    assert "real" in out


def test_backward_forwards_feedback_to_child_params() -> None:
    """backward forwards a parent's gradients to children, which refine to their params.

    Deterministic via a stubbed gradient fn that refines whatever feedback it
    receives onto the visited node's own parameter. Proves feedback reaches a
    depth-2 child's parameter (the multi-hop path).
    """
    from ai_functions.optimizer import textgrad as tg

    backend = _RecordingBackend(WritingMemory, "w1")
    child_param = ParameterNode(node_id="cp", name="joke_guidelines", backend=backend, requires_grad=True)
    child = ThreadNode(node_id="child", thread_id="child", parameters=[child_param])
    root = ThreadNode(node_id="root", thread_id="root", child_threads=[child])

    # Fake gradient fn: refine the incoming feedback onto every grad param it
    # is given (matched by node_id, which is what the real fn returns).
    class _FakeFn:
        def run_sync(self, *, parameters: str, trace: str, output: str, feedback: list[str]) -> tg.Feedbacks:
            del parameters, trace, output
            return tg.Feedbacks(feedbacks=[tg.Feedback(node_name="cp", feedback="refined-for-param")])

    opt = TextGradOptimizer()
    opt._backward_fn = _FakeFn()  # type: ignore[assignment]  # noqa: SLF001
    opt.backward(root, "top-level feedback")

    # Root (no grad params) forwarded its gradients to the child; the child
    # then refined them onto its own parameter.
    assert "top-level feedback" in child.gradients
    assert "refined-for-param" in child_param.gradients
