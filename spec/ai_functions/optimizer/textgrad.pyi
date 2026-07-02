"""TextGrad-style optimizer over a reconstructed computation graph.

Walks a :class:`ThreadNode` graph in reverse topological order, using an
internal AI function to distribute natural-language feedback from each node to
its grad-enabled parameter inputs, then consolidates that feedback directly
into the memory backends referenced by each :class:`ParameterNode`.

The optimizer is a pure consumer of an already-built graph: it never reads the
event log or touches a running thread. The autograd analogy is exact —
``backward`` is ``loss.backward()`` (textual gradients instead of numeric),
``consolidate`` is ``optimizer.step()`` (writes improvements back into memory).
"""

from __future__ import annotations

from strands.models import Model

from ..types.graph import ThreadNode


class TextGradOptimizer:
    """Propagate feedback through a ``ThreadNode`` graph and consolidate into memory.

    Usage::

        graph = build_graph(await coord.get_events("t1"), [memory])
        optimizer = TextGradOptimizer(model=model)
        optimizer.backward(graph, "The output should be more concise.")
        optimizer.consolidate(graph)
    """

    quiet: bool

    def __init__(
        self,
        model: Model | str | None = None,
        quiet: bool = True,
    ) -> None:
        """Build an optimizer whose internal gradient function uses ``model``.

        Args:
            model: Model (or model id) the internal feedback-distribution AI
                function runs on. ``None`` uses the library default provider.
            quiet: Suppress the internal AI function's callback output.
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
        multi-level graph.

        Args:
            root: Root of the reconstructed graph (typically the final
                thread whose output the feedback is about).
            feedback: Natural-language description of what should change.

        Ensures:
            - ``feedback`` is appended to ``root.gradients``.
            - For every grad-enabled parameter the model deems relevant, a
              refined feedback string is appended to its ``gradients``.
            - Each node's gradients are forwarded to its child threads.
            - Parameters with ``requires_grad=False`` receive no gradients.
        """
        ...

    def consolidate(self, root: ThreadNode) -> None:
        """Consolidate accumulated parameter gradients into their memory backends.

        Groups gradients by ``(backend, parameter name)`` so a parameter
        recalled in several threads is consolidated once, then calls
        ``param.backend.consolidate(name, feedbacks)`` through each node's
        direct backend reference. No external backend lookup table is used.

        Args:
            root: Root of the graph whose gradients should be written back.

        Ensures:
            - Each ``(backend, name)`` group triggers exactly one
              ``backend.consolidate`` call carrying all of that parameter's
              gradients across the graph.
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
