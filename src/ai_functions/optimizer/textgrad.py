"""TextGrad-style optimizer over a reconstructed computation graph.

Walks a ``ThreadNode`` graph in reverse topological order. At each node an
internal AI function distributes the node's accumulated feedback across its
routable targets: its grad-enabled ``ParameterNode`` inputs *and* its child
``ThreadNode`` s that lead to a grad-enabled parameter. Feedback routed to a
parameter accumulates for consolidation; feedback routed to a child thread is
*refined* feedback appended to that child, which re-distributes it to its own
targets when it is visited later in the walk. Parameter gradients are finally
consolidated directly into the memory backends referenced by each
``ParameterNode``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field
from strands.models import Model

from ..ai_thread import ai_function
from ..ai_thread.errors import AIFunctionError
from ..ai_thread.postcondition import PostConditionResult
from ..types.context import no_thread_scope
from ..types.graph import GradFeedback, Node, ParameterNode, ThreadNode, VersionedParameterHost
from ._graph import build_graph_from_result, leads_to_grad_parameter, topological_sort
from .rendering import render_inputs, render_messages

if TYPE_CHECKING:
    from ..memory.base import MemoryBackend
    from ..types.graph import Result

logger = logging.getLogger(__name__)


OPTIMIZE_TOOLS_PROMPT = (
    "You can also provide feedback to function calls that appears in the trace. "
    "However try to minimize the number of function calls to which you provide feedback. "
    "You cannot provide feedback to tool calls."
)

DO_NOT_OPTIMIZE_TOOLS_PROMPT = "Do not provide any feedback to tool calls or function calls."


class Feedback(BaseModel):
    """A single feedback entry targeting a specific input node."""

    node_id: str = Field(..., description="The id of the node or tool call to which you are providing feedback.")
    feedback: str = Field(
        ...,
        description="How the input node should change. "
        "The feedback MUST be relevant to the node description, if any. "
        "Feedback MUST be general and applicable to different future inputs.",
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="A number in [0, 1] rating how well this input's VALUE actually served "
        "the agent's output — 1.0 = fully useful, 0.0 = useless. Rate the value as it was "
        "consumed, independently of the improvements you suggest in `feedback`. An input's "
        "score_note, when present, says how the score will be used.",
    )


class Feedbacks(BaseModel):
    """Collection of feedback entries produced by the optimizer."""

    feedbacks: list[Feedback]


# coordinator_tools_enabled=False: the backward model is internal machinery
# operating on a *finished* trace — it must not list or message live threads.
@ai_function[Feedbacks](coordinator_tools_enabled=False)
def _compute_gradients(
    inputs: str,
    trace: str,
    output: str,
    feedback: list[str],
    optimize_tools: bool,
) -> str:
    """Build the backward prompt that distributes feedback across a node's inputs."""
    issues = "\n".join(f"- {f}" for f in feedback)
    tools_rule = OPTIMIZE_TOOLS_PROMPT if optimize_tools else DO_NOT_OPTIMIZE_TOOLS_PROMPT
    return (
        "You are an optimization agent. You analyze conversation traces to determine how\n"
        "parameters and inputs to an agent should be updated. You will be provided with\n"
        "input results, parameters (with their current values) and the execution\n"
        "trace of an agent using this information.\n\n"
        "# Inputs to the agent\n"
        f"{inputs}\n\n"
        "## Conversation trace\n"
        f"{trace}\n\n"
        "## Agent Output\n"
        f"{output}\n\n"
        "## Issues\n"
        f"{issues}\n\n"
        "## Rules\n"
        "Analyze the trace and produce per-input feedback.\n\n"
        "1. If the input has parameter type, provide feedback using the following bullet point format:\n"
        "```\n"
        "- add: <text> — information to add to the input\n"
        "- update: <text> — information to change in the input\n"
        "- delete: <text> — information to remove from the input\n"
        "```\n\n"
        "2. If the input has result type, provide feedback using the following format:\n"
        "```\n"
        "- improve: <text> — concrete feedback of how this specific result should change to resolve the issues\n"
        "```\n\n"
        "3. ONLY provide feedback that is relevant to the parameter's description. It may be that some of the user\n"
        "  feedback is not relevant to any of the input nodes. It is fine to ignore this feedback.\n\n"
        "4. The feedback you provide should always be general and applicable to future inputs.\n\n"
        f"{tools_rule}"
    )


class TextGradOptimizer:
    """Propagate feedback through a ThreadNode graph and consolidate into memory.

    One-call usage over a traced result::

        result = await email_writer.trace(jokes=cat, formatting_guidelines=fmt)
        optimizer = TextGradOptimizer(model=model)
        graph = await optimizer.step(result, "The email needs joke titles.", backends=[memory])

    Step-by-step usage over a reconstructed graph::

        graph = await build_graph(coord, thread_id, [memory])
        optimizer.backward(graph, "The output should be more concise.")
        optimizer.consolidate(graph)

    Args:
        optimize_tools: Whether the backward model may also target function
            calls that appear in the trace.
        model: LLM model for computing gradients.

    Attributes:
        last_dropped_feedback: Node ids from the most recent ``backward`` for
            which the backward model returned feedback that matched no routable
            target *after retries were exhausted* (that feedback is dropped).
            Empty when nothing was lost.
    """

    def __init__(
        self,
        optimize_tools: bool = False,
        model: Model | str | None = None,
    ) -> None:
        self.optimize_tools = optimize_tools
        self._backward_fn = _compute_gradients.replace(model=model)
        self.last_dropped_feedback: list[str] = []

    def backward(self, root: ThreadNode, feedback: str) -> None:
        """Propagate ``feedback`` from ``root`` through the graph.

        Seeds ``root`` with ``feedback`` and distributes it across each node's
        routable targets in reverse topological order. Idempotent: intermediate
        node gradients are cleared first, so calling ``backward`` twice does not
        double-count. Parameter gradients deliberately accumulate and are reset
        only by :meth:`zero_grad`.
        """
        sorted_nodes = topological_sort(root)

        # Idempotent backward: reset intermediate node gradients (parameter
        # gradients persist and accumulate; they are cleared by zero_grad).
        for node in sorted_nodes:
            node.gradients.clear()

        root.gradients.append(GradFeedback(text=feedback))
        self.last_dropped_feedback = []

        for node in sorted_nodes:
            if not node.gradients:
                continue

            grad_params = [p for p in node.parameters if p.requires_grad]
            grad_children = [c for c in node.child_threads if leads_to_grad_parameter(c)]
            targets: list[Node] = [*grad_params, *grad_children]

            # No routable target: forward raw gradients to children (pass-through).
            if not targets:
                for child in node.child_threads:
                    child.gradients.extend(node.gradients)
                continue

            self._distribute(node, targets)

        if self.last_dropped_feedback:
            logger.warning(
                "Backward: %d feedback item(s) matched no target and were dropped: %s. "
                "Inspect TextGradOptimizer.last_dropped_feedback.",
                len(self.last_dropped_feedback),
                ", ".join(self.last_dropped_feedback),
            )

    def _distribute(self, node: ThreadNode, targets: list[Node]) -> None:
        """Run the backward model for one node and route feedback to its targets.

        ``targets`` are the node's grad-enabled parameters and grad-reaching
        child threads, keyed by ``node_id``. A post-condition asserts every
        returned id is a valid target, so an invalid id triggers an automatic
        retry; if retries are exhausted the last attempt is salvaged and
        unroutable ids are recorded in ``last_dropped_feedback`` rather than
        crashing the whole backward pass.
        """
        target_map: dict[str, Node] = {t.node_id: t for t in targets}
        valid_ids = set(target_map)
        # Capture each attempt's result so the safety net can salvage the last
        # one if the post-condition never passes within the retry budget.
        last_result: list[Feedbacks] = []

        def _node_ids_valid(res: Feedbacks) -> PostConditionResult | None:
            last_result.append(res)
            invalid = [fb.node_id for fb in res.feedbacks if fb.node_id not in valid_ids]
            if invalid:
                return PostConditionResult(
                    passed=False,
                    message=(
                        f"Feedback references unknown node id(s) {invalid}. "
                        f"Valid ids: {sorted(valid_ids)}. Only use ids from the listed inputs."
                    ),
                )
            return None

        # The backward model call must not attribute to any ambient thread
        # scope — it would spawn as a child of the user's thread and pollute the
        # event log the graph was built from.
        try:
            with no_thread_scope():
                result = self._backward_fn.replace(post_conditions=[_node_ids_valid]).run_sync(
                    inputs=render_inputs(targets),
                    trace=render_messages(node.messages, {}),
                    output=str(node.value or ""),
                    feedback=[g.text for g in node.gradients],
                    optimize_tools=self.optimize_tools,
                )
        except AIFunctionError:
            # Retries exhausted without valid ids. Salvage the last attempt as a
            # best effort rather than aborting the whole graph.
            if not last_result:
                logger.warning(
                    "Backward: model produced no usable feedback for node %s after retries.",
                    node.thread_id,
                )
                return
            result = last_result[-1]

        for fb in result.feedbacks:
            target = target_map.get(fb.node_id)
            if target is None:
                # Only reachable on the salvage path above; a passing
                # post-condition guarantees every id is valid.
                self.last_dropped_feedback.append(fb.node_id)
                logger.warning(
                    "Backward: feedback for '%s' but no such target in node %s (dropped).",
                    fb.node_id,
                    node.thread_id,
                )
            else:
                # Route the refined feedback and its score together. A
                # ParameterNode's host reads the text (and, for a score-learning
                # host like the economics beliefs adapter, the score); a child
                # ThreadNode re-distributes the text to its own targets when
                # visited and carries the score to its own parameter hosts.
                target.gradients.append(GradFeedback(text=fb.feedback, score=fb.score))

    def consolidate(self, root: ThreadNode) -> None:
        """Consolidate accumulated parameter gradients into their memory backends.

        Groups gradients by ``(backend, name)`` so a parameter recalled in
        several threads is consolidated once, via the node's direct backend ref.
        Search-derived retrieval context (``meta["results"]``, the
        ``{entry_id: value}`` mapping of the entries the forward pass actually
        retrieved) is merged across the group and passed along, so a backend
        can target consolidation at those entries instead of the full value.

        A host that is a :class:`~ai_functions.types.graph.VersionedParameterHost`
        (it has ``load``) also receives ``expected_version``: the oldest
        ``meta["version"]`` the graph read for that parameter, so the host's own
        conflict policy sees any write that happened under the graph. A host
        without ``load`` is called with the plain ``ParameterHost`` signature.
        """
        grouped: dict[tuple[int, str], tuple[ParameterNode, list[GradFeedback], dict[str, str], list[int]]] = {}
        for node in topological_sort(root):
            for p in node.parameters:
                if not p.gradients or p.backend is None:
                    continue
                key = (id(p.backend), p.name)
                if key not in grouped:
                    grouped[key] = (p, [], {}, [])
                grouped[key][1].extend(p.gradients)
                results = p.meta.get("results")  # pyright: ignore[reportAny]
                if isinstance(results, dict):
                    grouped[key][2].update({str(k): str(v) for k, v in results.items()})  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType]
                version = p.meta.get("version")  # pyright: ignore[reportAny]
                if isinstance(version, int) and not isinstance(version, bool):
                    grouped[key][3].append(version)

        # One consolidate per (backend, parameter). A text-rewriting backend
        # reads each gradient's text; a score-learning host (the economics
        # beliefs adapter) reads the scores and settles its records; a versioned
        # host (a local parameter) also checks the version the graph read. All
        # are ``consolidate`` calls matched by ``backend_id`` — no separate hook.
        for (_, param_name), (param, feedbacks, retrieved, versions) in grouped.items():
            host = param.backend
            assert host is not None
            if versions and isinstance(host, VersionedParameterHost):
                host.consolidate(param_name, feedbacks, retrieved=retrieved or None, expected_version=min(versions))
            else:
                host.consolidate(param_name, feedbacks, retrieved=retrieved or None)

    def zero_grad(self, root: ThreadNode) -> None:
        """Clear all node and parameter gradients in the graph."""
        for node in topological_sort(root):
            node.gradients.clear()
            for p in node.parameters:
                p.gradients.clear()

    async def step(
        self,
        result: Result[Any],  # pyright: ignore[reportExplicitAny]
        feedback: str,
        backends: list[MemoryBackend],
    ) -> ThreadNode:
        """Build the graph from a traced result, backpropagate, and consolidate.

        The whole optimization dance in one call: reconstructs the graph from
        ``result`` (spawned children from events, sibling edges from
        ``Result.inputs``), runs :meth:`backward` with ``feedback``, then
        :meth:`consolidate` — on the **same** graph object, preserving the key
        invariant that gradients accumulate and are consolidated on nodes
        built exactly once.

        ``backward`` and ``consolidate`` make blocking model calls, so they
        run in a worker thread to keep the event loop responsive.

        Args:
            result: The root ``Result`` returned by ``AIFunction.trace``.
            feedback: Natural-language feedback on the root's output.
            backends: Live memory backends, matched by ``backend_id``.

        Returns:
            The graph, for inspection (gradients, structure) after the step.
        """
        graph = await build_graph_from_result(result, backends)
        await asyncio.to_thread(self.backward, graph, feedback)
        await asyncio.to_thread(self.consolidate, graph)
        return graph
