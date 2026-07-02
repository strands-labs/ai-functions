"""Abstract memory backend with parameter-recall tracking.

A ``MemoryBackend`` exposes named, typed parameters over a Pydantic schema and
is, at its core, plain storage. When a ``recall`` / ``query`` / ``search`` is
given a ``coordinator`` and ``thread_id``, it **immediately** emits a
``ParameterRecalledEvent`` into that thread's log so the computation graph can
be reconstructed post-hoc for optimization; without them it is a pure fetch
that emits nothing.

Emission happens at call time and works from anywhere — including outside an
``@ai_function`` body — because ``Coordinator.append_event`` only requires the
event's ``thread_id`` to be set and creates the thread's log on demand. The
recalled value is returned unchanged (a plain ``str`` / ``list`` / model), so
it interpolates into prompts and f-strings normally; the tracking is a side
effect on the log, never a decoration on the return value.

The backend holds no reference to a coordinator or thread between calls. It is
identified by a stable ``backend_id`` that :func:`build_graph` uses to match
recall events back to the live backend at consolidation time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Self

from pydantic import BaseModel
from strands.tools import ToolProvider
from strands.types.tools import AgentTool

from ..protocols import Coordinator
from ..types.ids import ThreadId

# Legacy alias kept for backward compatibility with JSONMemoryBackend.
ValueType = str | list[str]


class MemoryBackend(ABC):
    """Abstract memory backend over a Pydantic schema.

    Subclasses implement storage, retrieval, and consolidation through the
    abstract ``_*`` methods. The base class owns the public API, schema
    introspection (frozen / procedural markers, descriptions), value
    serialization, and recall-event emission.
    """

    actor_id: str
    schema: type[BaseModel]

    def __init__(self, schema: type[BaseModel], actor_id: str) -> None:
        """Bind a schema and actor namespace to this backend.

        Args:
            schema: Pydantic model describing the memory parameters.
            actor_id: Namespaces this actor's values within the backend.
        """
        ...

    @property
    def backend_id(self) -> str:
        """Stable identifier used to match recall events back to this backend.

        Format ``"ClassName:actor_id"``. ``build_graph`` keys on this to
        reconnect a reconstructed ``ParameterNode`` to the live backend so
        ``consolidate`` can route feedback. Subclasses may override.
        """
        ...

    # ── Abstract storage contract (subclass implements) ──

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this backend (e.g. flush to disk)."""
        ...

    @abstractmethod
    def _save(self, name: str, value: Any) -> None:
        """Replace the stored value of parameter ``name``."""
        ...

    @abstractmethod
    def _recall(self, name: str) -> Any:
        """Return parameter ``name``'s current value from storage."""
        ...

    @abstractmethod
    def _query(self, name: str, query: str) -> str:
        """Answer ``query`` given the content of parameter ``name``."""
        ...

    @abstractmethod
    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> Any:
        """Return the top-``k`` matches from parameter ``name``'s value."""
        ...

    @abstractmethod
    def _consolidate(self, name: str, feedback: list[str], **kwargs: Any) -> None:
        """Incorporate ``feedback`` into the stored value of parameter ``name``."""
        ...

    @abstractmethod
    def _delete(self, name: str) -> None:
        """Reset parameter ``name`` to its schema default."""
        ...

    # ── Value (de)serialization ──

    def deserialize_value(self, name: str, raw: Any) -> Any:
        """Reconstruct a parameter value from its event representation.

        The inverse of value serialization: validates ``raw`` against the
        field's full type annotation via ``TypeAdapter``, so arbitrarily nested
        shapes (``list[Model]``, ``list[list[Model]]``, ``dict[str, Model]``, a
        bare ``BaseModel``) rehydrate to models rather than raw dicts; plain
        types pass through. Falls back to the raw value if validation fails.

        Args:
            name: Parameter name (supports nested ``a/b/c`` paths).
            raw: The JSON-shaped value taken from a recall event.

        Returns:
            The deserialized value.
        """
        ...

    # ── Public API (async; each read emits a ParameterRecalledEvent) ──
    #
    # These are async so emission is durable before the call returns, even on
    # a network-backed coordinator whose ``append_event`` defers the write to
    # the event loop. After appending, the method awaits ``get_events`` until
    # the event is visible (read-after-write coherence), bounded so a stalled
    # connection degrades to best-effort rather than hanging. On an in-process
    # coordinator the append is already durable, so the confirmation returns
    # on its first read.

    async def recall(
        self,
        name: str,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Recall a parameter's full value, emitting a recall event.

        Args:
            name: Parameter name (supports nested ``a/b/c`` paths).
            coordinator: Coordinator to append the recall event to. When
                ``None``, the read is a pure fetch and no event is emitted.
            thread_id: Thread the recalled value is being fed into; stamped
                onto the event. Required for emission alongside ``coordinator``.
            requires_grad: Override gradient tracking for this read; defaults
                to ``False`` for ``Frozen`` fields and ``True`` otherwise.
            kwargs: Forwarded to the backend's ``_recall``.

        Returns:
            The stored value, unchanged.

        Ensures:
            - When both ``coordinator`` and ``thread_id`` are given, one
              ``ParameterRecalledEvent`` with ``derivation="full"`` is appended
              to ``thread_id``'s log and confirmed durable (visible via
              ``get_events``) before this call returns, subject to a bounded
              number of confirmation reads.
            - When either is ``None``, no event is emitted (pure fetch).
            - The returned value is identical to a direct storage read.
        """
        ...

    async def query(
        self,
        name: str,
        query: str,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
        **kwargs: Any,
    ) -> str:
        """Answer a question over a parameter's value, emitting a recall event.

        Args:
            name: Parameter name.
            query: Natural-language question answered over the value.
            coordinator: Coordinator to append the recall event to; ``None``
                makes this a pure fetch.
            thread_id: Thread the answer is fed into; stamped onto the event.
            requires_grad: Override gradient tracking; default as in
                :meth:`recall`.
            kwargs: Forwarded to the backend's ``_query``.

        Returns:
            The model's answer.

        Ensures:
            - When both ``coordinator`` and ``thread_id`` are given, one
              ``ParameterRecalledEvent`` with ``derivation="query"`` and the
              query in its ``meta`` is appended and confirmed durable before
              returning.
        """
        ...

    async def search(
        self,
        name: str,
        query: str,
        k: int = 5,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return top-``k`` matches from a collection parameter, emitting a recall event.

        Args:
            name: Parameter name (typically a collection-valued field).
            query: Retrieval query.
            k: Maximum number of matches.
            coordinator: Coordinator to append the recall event to; ``None``
                makes this a pure fetch.
            thread_id: Thread the matches are fed into; stamped onto the event.
            requires_grad: Override gradient tracking; default as in
                :meth:`recall`.
            kwargs: Backend-specific search options, forwarded to ``_search``
                and recorded in the event ``meta``.

        Returns:
            The top-``k`` results.

        Ensures:
            - When both ``coordinator`` and ``thread_id`` are given, one
              ``ParameterRecalledEvent`` with ``derivation="search"`` carrying
              ``query``, ``top_k``, and any extra options in its ``meta`` is
              appended and confirmed durable before returning.
        """
        ...

    def consolidate(self, name: str, feedback: list[str]) -> None:
        """Incorporate ``feedback`` into parameter ``name``.

        Args:
            name: Parameter name.
            feedback: Feedback strings to merge into the stored value.
        """
        ...

    def save(self, name: str, value: Any) -> None:
        """Store a parameter's value directly, without consolidation.

        Args:
            name: Parameter name.
            value: New value to store.
        """
        ...

    def delete(self, name: str) -> None:
        """Reset a parameter to its schema default.

        Args:
            name: Parameter name.
        """
        ...

    # ── Scoped tool provider ──

    def tool_provider(self, *names: str, operations: set[str] | None = None) -> DynamicToolProvider:
        """Return a ``ToolProvider`` with tools scoped to the given parameter names.

        For each parameter, generates uniquely-named tools (``recall_<name>``,
        ``query_<name>``, and — by field type — ``search_<name>`` for list
        parameters or ``save_<name>`` / ``delete_<name>`` for scalar ones) whose
        descriptions come from the schema, so an agent can read and write memory
        during a cycle.

        The generated tools are pure fetches/writes: they thread no
        coordinator/thread through, so they do **not** emit
        ``ParameterRecalledEvent`` s and do not feed the optimization graph.

        Args:
            names: One or more parameter names (slash-separated for nested fields).
            operations: Restrict to this subset of ``{"recall", "query",
                "search", "save", "delete"}``; all applicable tools if ``None``.

        Returns:
            A ``DynamicToolProvider`` holding the generated tools.
        """
        ...

    # ── Context manager ──

    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


class DynamicToolProvider(ToolProvider):
    """A ``ToolProvider`` holding a pre-built list of memory tools.

    Returned by :meth:`MemoryBackend.tool_provider`. The tools are already
    materialized, so consumer tracking is bookkeeping only.
    """

    def __init__(self, tools: list[AgentTool]) -> None:
        """Hold the pre-built ``tools``."""
        ...

    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Return the pre-built tools."""
        ...

    def add_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Register a consumer (bookkeeping only)."""
        ...

    def remove_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Deregister a consumer (bookkeeping only)."""
        ...
