"""TextGrad-style optimizer over a reconstructed computation graph.

Walks a :class:`ThreadNode` graph in reverse topological order, using an
internal AI function to distribute natural-language feedback from each node to
its grad-enabled parameter inputs, then consolidates that feedback directly
into the memory backends referenced by each :class:`ParameterNode`.

``backward`` / ``consolidate`` are pure consumers of an already-built graph:
they never read the event log or touch a running thread. ``step`` is the
one-call form over a traced :class:`Result`: it builds the graph (spawned
children from events, sibling edges from ``Result.inputs``), backpropagates,
and consolidates — on the same graph object, built exactly once. The autograd
analogy is exact — ``backward`` is ``loss.backward()`` (textual gradients
instead of numeric), ``consolidate`` is ``optimizer.step()`` (writes
improvements back into memory).
"""

from __future__ import annotations

from typing import Any

from strands.models import Model

from ..memory.base import MemoryBackend
from ..types.graph import Result, ThreadNode

OPTIMIZE_TOOLS_PROMPT: str
"""Backward-prompt rule allowing feedback on function calls in the trace."""

DO_NOT_OPTIMIZE_TOOLS_PROMPT: str
"""Backward-prompt rule forbidding feedback on tool and function calls."""


class TextGradOptimizer:
    """Propagate feedback through a ``ThreadNode`` graph and consolidate into memory.

    One-call usage over a traced result::

        result = await email_writer.trace(jokes=cat, formatting_guidelines=fmt)
        optimizer = TextGradOptimizer(model=model)
        graph = await optimizer.step(result, "The email needs joke titles.", backends=[memory])

    Step-by-step usage over a reconstructed graph::

        graph = await build_graph(coord, thread_id, [memory])
        optimizer.backward(graph, "The output should be more concise.")
        optimizer.consolidate(graph)
    """

    last_dropped_feedback: list[str]
    """Parameter ids from the most recent ``backward`` whose feedback matched
    no parameter and was dropped. Empty when nothing was lost."""

    optimize_tools: bool
    """Whether the backward model may also target function calls in the trace."""

    def __init__(
        self,
        optimize_tools: bool = False,
        model: Model | str | None = None,
    ) -> None:
        """Build an optimizer whose internal gradient function uses ``model``.

        Args:
            optimize_tools: Whether the backward model may also target function
                calls that appear in the trace.
            model: Model (or model id) the internal feedback-distribution AI
                function runs on. ``None`` uses the library default provider.
        """
        ...

    def backward(self, root: ThreadNode, feedback: str) -> None:
        """Propagate feedback from ``root`` through the graph into parameter gradients.

        Seeds ``root.gradients`` with ``feedback``, then walks every
        ``ThreadNode`` parent-before-child (reverse topological order). For each
        node with grad-enabled parameters, an internal AI function refines the
        node's accumulated gradients into per-parameter feedback (matched by
        ``node_id``). The node's gradients are then forwarded to its child
        threads, which re-refine them against their own parameters when visited
        — the child describes its own parameters far better than a parent's
        gradient call could, so this is what carries feedback through a
        multi-level graph. The internal model calls run with the ambient thread
        scope cleared, so they never pollute a running thread's event log.

        Args:
            root: Root of the reconstructed graph (typically the final
                thread whose output the feedback is about).
            feedback: Natural-language description of what should change.

        Ensures:
            - ``feedback`` is appended to ``root.gradients`` as a
              :class:`~ai_functions.types.graph.GradFeedback`.
            - For every grad-enabled parameter the model deems relevant, a
              refined ``GradFeedback`` (text plus the model's ``[0, 1]`` score)
              is appended to its ``gradients``. A text-rewriting host uses the
              text; a score-learning host (the economics beliefs adapter) uses
              the score.
            - Each node's gradients are forwarded to its child threads.
            - Parameters with ``requires_grad=False`` receive no gradients.
            - Model feedback matching no parameter is dropped with a warning
              and recorded in ``last_dropped_feedback`` (reset on each call).
        """
        ...

    def consolidate(self, root: ThreadNode) -> None:
        """Consolidate accumulated parameter gradients into their memory backends.

        Groups gradients by ``(backend, parameter name)`` so a parameter
        recalled in several threads is consolidated once, then calls
        ``param.backend.consolidate(name, feedbacks, retrieved=...)`` through
        each node's direct backend reference. No external backend lookup table
        is used. Search-derived retrieval context (``meta["results"]``, the
        ``{entry_id: value}`` mapping of the entries the forward pass actually
        retrieved) is merged across the group and passed along, so a backend
        can target consolidation at those entries instead of the full value.

        Args:
            root: Root of the graph whose gradients should be written back.

        Ensures:
            - Each ``(backend, name)`` group triggers exactly one
              ``backend.consolidate`` call carrying all of that parameter's
              gradients across the graph, plus the merged retrieval context
              (``None`` when no grouped node carries ``meta["results"]``).
            - A host that is a ``VersionedParameterHost`` (it has ``load``)
              also receives ``expected_version``, the oldest ``meta["version"]``
              the graph read for that parameter; a host without ``load`` is
              called with the plain ``ParameterHost`` signature.
            - Parameters with no gradients are skipped.
        """
        ...

    def zero_grad(self, root: ThreadNode) -> None:
        """Clear all node and parameter gradients in the graph.

        Args:
            root: Root of the graph to reset.

        Ensures:
            - Every reachable ``ThreadNode.gradients`` and
              ``ParameterNode.gradients`` is emptied.
        """
        ...

    async def step(
        self,
        result: Result[Any],
        feedback: str,
        backends: list[MemoryBackend],
    ) -> ThreadNode:
        """Build the graph from a traced result, backpropagate, and consolidate.

        The whole optimization dance in one call: reconstructs the graph from
        ``result`` via :func:`build_graph_from_result`, runs :meth:`backward`
        with ``feedback``, then :meth:`consolidate` — on the **same** graph
        object, preserving the invariant that gradients accumulate and are
        consolidated on nodes built exactly once. The blocking model calls run
        in a worker thread, keeping the event loop responsive.

        Args:
            result: The root ``Result`` returned by ``AIFunction.trace``.
            feedback: Natural-language feedback on the root's output.
            backends: Live memory backends, matched by ``backend_id``.

        Returns:
            The graph, for inspection (gradients, structure) after the step.
        """
        ...
