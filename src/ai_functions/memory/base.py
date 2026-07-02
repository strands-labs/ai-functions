"""Abstract memory backend with parameter-recall tracking.

A ``MemoryBackend`` exposes named, typed parameters over a Pydantic schema and
is, at its core, plain storage. When a ``recall`` / ``query`` / ``search`` is
given a ``coordinator`` and ``thread_id``, it immediately emits a
``ParameterRecalledEvent`` into that thread's log so the computation graph can
be reconstructed post-hoc for optimization; without them it is a pure fetch
that emits nothing.

Emission works from anywhere — including outside an ``@ai_function`` body —
because ``Coordinator.append_event`` only requires the event's ``thread_id`` to
be set and creates the thread's log on demand. The recalled value is returned
unchanged, so it interpolates into prompts and f-strings normally.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, get_args, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo
from strands.tools import ToolProvider
from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]

from ..types.events import EventKind, ParameterRecalledEvent
from .frozen import FrozenMarker
from .procedural import ProceduralMarker

if TYPE_CHECKING:
    from strands.types.tools import AgentTool

    from ..protocols import Coordinator
    from ..types.ids import ThreadId

logger = logging.getLogger(__name__)

# Legacy alias kept for backward compatibility with JSONMemoryBackend.
ValueType = str | list[str]


def _model_from_annotation(annotation: Any) -> type[BaseModel] | None:  # pyright: ignore[reportExplicitAny]
    """Extract the ``BaseModel`` subclass from a (possibly wrapped) annotation.

    Handles a bare model, ``Annotated[Model, ...]``, and ``Optional[Model]`` /
    ``Model | None`` (and other unions) by returning the single ``BaseModel``
    member. Returns ``None`` if no model is present (e.g. a plain ``str`` or a
    union of non-models), so callers can raise a clear error.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    # Unwrap Annotated[...] / Optional[...] / Union[...] — find a model member.
    for arg in get_args(annotation):
        model = _model_from_annotation(arg)
        if model is not None:
            return model
    return None


# Bounded read-after-write confirmation: how many times ``get_events`` is
# polled for an appended recall event before degrading to best-effort. An
# in-process coordinator confirms on the first read; a network-backed one may
# need the event loop to run its deferred append task (and a round-trip).
_CONFIRM_MAX_READS = 50

# Backoff between confirmation reads. Starts at 0 (one loop tick, enough for an
# in-process deferred task) and grows so a genuine network round-trip has time
# to land within the bounded number of reads rather than spinning instantly.
_CONFIRM_BACKOFF_STEP = 0.01
_CONFIRM_BACKOFF_MAX = 0.2


class MemoryBackend(ABC):
    """Abstract memory backend over a Pydantic schema.

    Subclasses implement storage, retrieval, and consolidation through the
    abstract ``_*`` methods. The base class owns the public API, schema
    introspection, value serialization, and recall-event emission.
    """

    def __init__(self, schema: type[BaseModel], actor_id: str) -> None:
        self.actor_id = actor_id
        self.schema = schema

    # -- Identification --------------------------------------------------------

    @property
    def backend_id(self) -> str:
        """Stable identifier used to match recall events back to this backend.

        Format ``"ClassName:actor_id"``. Subclasses may override.
        """
        return f"{type(self).__name__}:{self.actor_id}"

    # -- Schema introspection --------------------------------------------------

    def _resolve_field(self, name: str) -> FieldInfo:
        parts = name.split("/")
        current_model = self.schema
        for part in parts[:-1]:
            annotation = current_model.model_fields[part].annotation
            current_model = _model_from_annotation(annotation)
            if current_model is None:
                raise TypeError(
                    f"Cannot resolve nested parameter {name!r}: intermediate field {part!r} "
                    f"is not a Pydantic model (annotation: {annotation!r})."
                )
        return current_model.model_fields[parts[-1]]

    def _get_description(self, name: str) -> str:
        return self._resolve_field(name).description or ""

    def _is_procedural(self, name: str) -> bool:
        return any(isinstance(m, ProceduralMarker) for m in self._resolve_field(name).metadata)

    def _is_frozen(self, name: str) -> bool:
        return any(isinstance(m, FrozenMarker) for m in self._resolve_field(name).metadata)

    def _is_list_field(self, name: str) -> bool:
        """Return whether parameter ``name`` is a list-valued field."""
        return get_origin(self._resolve_field(name).annotation) is list

    # -- Abstract storage contract (subclass implements) -----------------------

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this backend."""

    @abstractmethod
    def _save(self, name: str, value: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Save the value of a parameter, replacing whatever is stored."""

    @abstractmethod
    def _recall(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        """Return a parameter's current value from storage."""

    @abstractmethod
    def _query(self, name: str, query: str) -> str:
        """Answer a question given the content of parameter name."""

    @abstractmethod
    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Return top-k matches from the parameter's stored value."""

    @abstractmethod
    def _consolidate(self, name: str, feedback: list[str], **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Incorporate feedback into the stored value of parameter name."""

    @abstractmethod
    def _delete(self, name: str) -> None:
        """Reset a parameter to its schema default."""

    # -- Value (de)serialization -----------------------------------------------

    def _serialize_value(self, name: str, value: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Convert a parameter value to a JSON-serializable form."""
        annotation = self._resolve_field(name).annotation
        if annotation is None:
            return value
        try:
            return TypeAdapter(annotation).dump_python(value, mode="json")
        except Exception:  # noqa: BLE001 — fall back to the raw value
            return value

    def deserialize_value(self, name: str, raw: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Reconstruct a parameter value from its event representation.

        The inverse of :meth:`_serialize_value`: validates ``raw`` against the
        field's full type annotation via ``TypeAdapter`` (so arbitrarily nested
        shapes — ``list[Model]``, ``list[list[Model]]``, ``dict[str, Model]``,
        a bare ``BaseModel`` — rehydrate to models, not raw dicts). Falls back
        to the raw value if validation is impossible.
        """
        annotation = self._resolve_field(name).annotation
        if annotation is None:
            return raw
        try:
            return TypeAdapter(annotation).validate_python(raw)
        except Exception:  # noqa: BLE001 — fall back to the raw value (mirrors _serialize_value)
            return raw

    # -- Event emission --------------------------------------------------------

    async def _emit_parameter_event(
        self,
        coordinator: Coordinator | None,
        thread_id: ThreadId | None,
        name: str,
        value: Any,  # pyright: ignore[reportExplicitAny]
        derivation: str,
        requires_grad: bool,
        description: str = "",
        meta: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        """Append a ``ParameterRecalledEvent`` and confirm it is durable.

        No-op when ``coordinator`` or ``thread_id`` is ``None`` (pure fetch).

        ``Coordinator.append_event`` is synchronous, but a network-backed
        coordinator defers the actual write to the event loop and returns
        before it lands. Per I8 (read-after-write coherence), this method then
        awaits ``get_events`` until the appended event is visible, so a
        subsequent ``build_graph`` read cannot overtake the write. An
        in-process coordinator confirms on the first read. The poll uses a
        growing backoff and is bounded by ``_CONFIRM_MAX_READS``; if the event
        is still not visible, it logs a warning (the I8 guarantee is unmet for
        that event) rather than hanging or failing silently.
        """
        if coordinator is None or thread_id is None:
            return
        event = ParameterRecalledEvent(
            thread_id=thread_id,
            name=name,
            value=self._serialize_value(name, value),
            derivation=derivation,  # pyright: ignore[reportArgumentType]
            requires_grad=requires_grad,
            backend_id=self.backend_id,
            description=description,
            meta=meta or {},
        )
        coordinator.append_event(event)

        delay = 0.0
        for _ in range(_CONFIRM_MAX_READS):
            stored = await coordinator.get_events(thread_id, kinds=[EventKind.PARAMETER_RECALLED])
            if any(e.id == event.id for e in stored):
                return
            await asyncio.sleep(delay)
            delay = min(delay + _CONFIRM_BACKOFF_STEP, _CONFIRM_BACKOFF_MAX)

        # Never confirmed within the bound. The I8 read-after-write guarantee is
        # not met for this event; warn rather than fail silently so a dropped
        # parameter recall (and the resulting under-optimization) is diagnosable.
        logger.warning(
            "Parameter recall event for %r (thread %s) was not confirmed durable after %d reads; "
            "the optimization graph may miss this parameter.",
            name,
            thread_id,
            _CONFIRM_MAX_READS,
        )

    # -- Public API ------------------------------------------------------------

    async def recall(
        self,
        name: str,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> Any:  # pyright: ignore[reportExplicitAny]
        """Recall a parameter's full value, emitting a recall event."""
        if requires_grad is None:
            requires_grad = not self._is_frozen(name)
        value = self._recall(name, **kwargs)
        await self._emit_parameter_event(
            coordinator,
            thread_id,
            name,
            value,
            "full",
            requires_grad,
            description=self._get_description(name),
        )
        return value

    async def query(
        self,
        name: str,
        query: str,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> str:
        """Answer a question over a parameter's value, emitting a recall event."""
        if requires_grad is None:
            requires_grad = not self._is_frozen(name)
        value = self._query(name, query, **kwargs)
        await self._emit_parameter_event(
            coordinator,
            thread_id,
            name,
            value,
            "query",
            requires_grad,
            description=self._get_description(name),
            meta={"query": query},
        )
        return value

    async def search(
        self,
        name: str,
        query: str,
        k: int = 5,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> Any:  # pyright: ignore[reportExplicitAny]
        """Return top-k matches from a collection parameter, emitting a recall event."""
        if requires_grad is None:
            requires_grad = not self._is_frozen(name)
        value = self._search(name, query, k, **kwargs)
        meta: dict[str, Any] = {"query": query, "top_k": k}  # pyright: ignore[reportExplicitAny]
        if kwargs:
            meta.update(kwargs)
        await self._emit_parameter_event(
            coordinator,
            thread_id,
            name,
            value,
            "search",
            requires_grad,
            description=self._get_description(name),
            meta=meta,
        )
        return value

    def consolidate(self, name: str, feedback: list[str]) -> None:
        """Incorporate feedback into a parameter."""
        self._consolidate(name, feedback)

    def save(self, name: str, value: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Store a parameter's value directly, without consolidation."""
        self._save(name, value)

    def delete(self, name: str) -> None:
        """Reset a parameter to its schema default."""
        self._delete(name)

    # -- Scoped tool provider --------------------------------------------------

    def tool_provider(self, *names: str, operations: set[str] | None = None) -> DynamicToolProvider:
        """Return a ``ToolProvider`` with tools scoped to the given parameter names.

        For each parameter, generates uniquely-named tools (e.g. ``recall_facts``,
        ``search_visited``) whose descriptions are derived from the schema so an
        agent can read and write memory directly during a cycle. ``search_<name>``
        is generated only for list parameters; ``save_<name>`` / ``delete_<name>``
        only for scalar parameters.

        The generated tools are pure fetches/writes: they do not emit
        ``ParameterRecalledEvent`` s (no coordinator/thread is threaded through),
        so they drive in-cycle memory use, not the optimization graph.

        Args:
            names: One or more parameter names (slash-separated for nested fields).
            operations: Restrict to this subset of ``{"recall", "query",
                "search", "save", "delete"}``; all applicable tools if ``None``.

        Returns:
            A ``DynamicToolProvider`` holding the generated tools.
        """
        ops = operations or {"recall", "query", "search", "save", "delete"}
        tools: list[AgentTool] = []
        for name in names:
            desc = self._get_description(name) or name
            safe = name.replace("/", "_")
            is_list = self._is_list_field(name)
            if "recall" in ops:
                tools.append(
                    _strands_tool(name=f"recall_{safe}", description=f"Retrieve the full content of: {desc}")(
                        self._make_recall_tool(name)
                    )
                )
            if "query" in ops:
                tools.append(
                    _strands_tool(name=f"query_{safe}", description=f"Ask a natural-language question about: {desc}")(
                        self._make_query_tool(name)
                    )
                )
            if "search" in ops and is_list:
                tools.append(
                    _strands_tool(name=f"search_{safe}", description=f"Search for relevant entries in: {desc}")(
                        self._make_search_tool(name)
                    )
                )
            if "save" in ops and not is_list:
                tools.append(
                    _strands_tool(name=f"save_{safe}", description=f"Overwrite the content of: {desc}")(
                        self._make_save_tool(name)
                    )
                )
            if "delete" in ops and not is_list:
                tools.append(
                    _strands_tool(name=f"delete_{safe}", description=f"Reset to default: {desc}")(
                        self._make_delete_tool(name)
                    )
                )
        return DynamicToolProvider(tools)

    # -- Tool factories (capture the parameter name via closure) ---------------

    def _make_recall_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        async def _recall() -> Any:  # pyright: ignore[reportExplicitAny]
            """Retrieve the full content of this memory parameter."""
            return await self.recall(name)

        return _recall

    def _make_query_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        async def _query(query: str) -> str:
            """Ask a question about this memory parameter.

            Args:
                query: The natural-language question to answer.
            """
            return await self.query(name, query)

        return _query

    def _make_search_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        async def _search(query: str, k: int = 5) -> list[str]:
            """Search for relevant entries in this memory parameter.

            Args:
                query: Keywords or phrase to match against entries.
                k: Maximum number of results to return.
            """
            return await self.search(name, query, k)

        return _search

    def _make_save_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        def _save(value: str) -> str:
            """Overwrite this memory parameter with a new value.

            Args:
                value: The new value to store.
            """
            self.save(name, value)
            return "Saved"

        return _save

    def _make_delete_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        def _delete() -> str:
            """Reset this memory parameter to its default value."""
            self.delete(name)
            return "Deleted"

        return _delete

    # -- Context manager -------------------------------------------------------

    def __enter__(self) -> Self:
        """Enter the context, returning this backend."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit the context, closing the backend."""
        self.close()


class DynamicToolProvider(ToolProvider):
    """A ``ToolProvider`` holding a pre-built list of memory tools.

    Returned by :meth:`MemoryBackend.tool_provider`. Consumer tracking is a
    no-op set kept only to satisfy the Strands ``ToolProvider`` contract; the
    tools are already materialized so nothing is loaded lazily per consumer.
    """

    def __init__(self, tools: list[AgentTool]) -> None:
        self._tools = tools
        self._consumers: set[object] = set()

    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Return the pre-built tools."""
        return self._tools

    def add_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Register a consumer (no-op beyond bookkeeping)."""
        self._consumers.add(consumer_id)

    def remove_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Deregister a consumer (no-op beyond bookkeeping)."""
        self._consumers.discard(consumer_id)
