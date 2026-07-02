"""TextGrad-style optimizer over a reconstructed computation graph.

Walks a ``ThreadNode`` graph in reverse topological order, using an internal AI
function to distribute feedback from each node to its grad-enabled parameter
inputs, then consolidates feedback directly into the memory backends referenced
by each ``ParameterNode``.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from strands.models import Model

from ..ai_thread import ai_function
from ..types.graph import ParameterNode, ThreadNode
from ._graph import topological_sort
from .rendering import render_inputs, render_messages

logger = logging.getLogger(__name__)


class Feedback(BaseModel):
    node_name: str = Field(..., description="The parameter node_id to provide feedback to.")
    feedback: str = Field(..., description="How the parameter should change.")


class Feedbacks(BaseModel):
    feedbacks: list[Feedback]


@ai_function(Feedbacks)
def _compute_gradients(
    parameters: str,
    trace: str,
    output: str,
    feedback: list[str],
) -> str:
    """Build the backward prompt that distributes feedback to parameters."""
    issues = "\n".join(f"- {f}" for f in feedback)
    return (
        "An agent received the following parameter inputs:\n"
        f"<parameters>\n{parameters}\n</parameters>\n\n"
        "The agent produced the following execution trace:\n"
        f"<trace>\n{trace}\n</trace>\n\n"
        "At the end, it produced the following output:\n"
        f"<output>\n{output}\n</output>\n\n"
        "The following issues need to be fixed:\n"
        f"<issues>\n{issues}\n</issues>\n\n"
        "Return feedback for each parameter that needs to change.\n"
        "Rules:\n"
        "- Only provide feedback relevant to the parameter's description.\n"
        "- Feedback must be general and applicable to different future inputs.\n"
        "- If a parameter doesn't need changes, omit it."
    )


class TextGradOptimizer:
    """Propagate feedback through a ThreadNode graph and consolidate into memory.

    Usage::

        graph = build_graph(await coord.get_events("t1"), [memory])
        optimizer = TextGradOptimizer(model=model)
        optimizer.backward(graph, "The output should be more concise.")
        optimizer.consolidate(graph)
    """

    def __init__(
        self,
        model: Model | str | None = None,
        quiet: bool = True,
    ) -> None:
        self.quiet = quiet
        self._backward_fn = _compute_gradients.replace(model=model)

    def backward(self, root: ThreadNode, feedback: str) -> None:
        """Propagate feedback through the graph.

        Appends feedback to root, then for each ThreadNode with gradients and
        grad-enabled parameters, runs the backward AI function to distribute
        feedback to individual parameters. A node's gradients are forwarded to
        its child threads, which re-refine them against their own parameters
        when visited — this carries feedback through a multi-level graph.
        """
        root.gradients.append(feedback)

        for node in topological_sort(root):
            if not node.gradients:
                continue
            grad_params = [p for p in node.parameters if p.requires_grad]
            if not grad_params:
                for child in node.child_threads:
                    child.gradients.extend(node.gradients)
                continue

            result = self._backward_fn.run_sync(
                parameters=render_inputs(grad_params),
                trace=render_messages(node.messages, {}),
                output=str(node.value or ""),
                feedback=node.gradients,
            )

            param_map = {p.node_id: p for p in grad_params}
            for fb in result.feedbacks:
                if fb.node_name in param_map:
                    param_map[fb.node_name].gradients.append(fb.feedback)
                else:
                    logger.warning(
                        "Backward: feedback for '%s' but no such parameter in node %s",
                        fb.node_name,
                        node.thread_id,
                    )

            for child in node.child_threads:
                child.gradients.extend(node.gradients)

    def consolidate(self, root: ThreadNode) -> None:
        """Consolidate accumulated parameter gradients into their memory backends.

        Groups gradients by ``(backend, name)`` so a parameter recalled in
        several threads is consolidated once, via the node's direct backend ref.
        """
        grouped: dict[tuple[int, str], tuple[ParameterNode, list[str]]] = {}
        for node in topological_sort(root):
            for p in node.parameters:
                if not p.gradients or p.backend is None:
                    continue
                key = (id(p.backend), p.name)
                if key not in grouped:
                    grouped[key] = (p, [])
                grouped[key][1].extend(p.gradients)

        for (_, param_name), (param, feedbacks) in grouped.items():
            assert param.backend is not None
            param.backend.consolidate(param_name, feedbacks)

    def zero_grad(self, root: ThreadNode) -> None:
        """Clear all node and parameter gradients in the graph."""
        for node in topological_sort(root):
            node.gradients.clear()
            for p in node.parameters:
                p.gradients.clear()
