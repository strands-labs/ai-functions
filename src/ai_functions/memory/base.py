"""Base memory backend with scoped ToolProvider support."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any, Self, TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefinedType
from strands import tool
from strands.tools import ToolProvider
from strands.types.tools import AgentTool

from ..types.graph import Derivation, ParameterGradient, ParameterMeta, ParameterRef, ParameterView
from ..utils._formatting import unique_name
from .frozen import FrozenMarker
from .procedural import ProceduralMarker
from .utils import flatten_schema

ValueType = str | list[str]

T = TypeVar("T", bound=ValueType)


class MemoryBackend(ABC):
    """Abstract memory backend for parameters and conversations."""

    def __init__(self, schema: type[BaseModel], actor_id: str) -> None:
        """Initialize the memory backend."""
        missing = [
            path
            for path, fi in flatten_schema(schema)
            if isinstance(fi.default, PydanticUndefinedType) and fi.default_factory is None
        ]
        if missing:
            raise ValueError(
                f"All fields in the memory schema must have a default value or default_factory. "
                f"Missing defaults for: {missing}"
            )
        self.actor_id = actor_id
        self.schema = schema
        self._refs: dict[str, ParameterRef] = {}

    # -- Schema introspection ---------------------------------------------------

    def _resolve_field(self, name: str) -> FieldInfo:
        """Resolve a hierarchical parameter name to its FieldInfo."""
        parts = name.split("/")
        current_model: Any = self.schema
        for part in parts[:-1]:
            field = current_model.model_fields[part]
            current_model = field.annotation
        result: FieldInfo = current_model.model_fields[parts[-1]]
        return result

    def _get_description(self, name: str) -> str:
        """Return parameter description from the schema."""
        return self._resolve_field(name).description or ""

    def _is_procedural(self, name: str) -> bool:
        """Check if a parameter name corresponds to a Procedural-typed field."""
        field_info = self._resolve_field(name)
        for metadata in field_info.metadata:
            if isinstance(metadata, ProceduralMarker):
                return True
        return False

    def _is_frozen(self, name: str) -> bool:
        """Check if a parameter name corresponds to a Frozen-typed field."""
        field_info = self._resolve_field(name)
        for metadata in field_info.metadata:
            if isinstance(metadata, FrozenMarker):
                return True
        return False

    # -- Abstract storage methods (implemented by subclasses) ------------------

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this backend."""

    @abstractmethod
    def _save(self, name: str, value: ValueType) -> None:
        """Save the value of a parameter replacing whatever is currently stored."""

    @abstractmethod
    def delete(self, name: str) -> None:
        """Delete a parameter."""

    @abstractmethod
    def _recall(self, name: str) -> tuple[ValueType, ParameterMeta]:
        """Return (value, meta) for a parameter from storage."""

    @abstractmethod
    def _query(self, name: str, query: str) -> tuple[str, ParameterMeta]:
        """Return (answer, meta) for a query against parameter *name*."""

    @abstractmethod
    def _search(self, name: str, query: str, k: int = 5) -> tuple[list[str], ParameterMeta]:
        """Return (top_k_values, meta) for a search against a list parameter."""

    @abstractmethod
    def _consolidate(self, name: str, feedback: list[ParameterGradient]) -> None:
        """Use the feedback to update/consolidate the value of parameter name on the backend."""

    # -- Parameter graph operations --------------------------------------------

    def _parameter_actor(self, name: str) -> str:
        return f"{self.actor_id}/{name}"

    def _get_or_create_ref(self, name: str) -> ParameterRef:
        """Return the unique ParameterRef for *name*, creating it on first access."""
        if name not in self._refs:
            self._refs[name] = ParameterRef(
                name=name,
                memory=self,
                description=self._get_description(name),
                procedural=self._is_procedural(name),
            )
        return self._refs[name]

    def _create_and_register_view(
        self,
        name: str,
        value: T,
        derivation: Derivation,
        requires_grad: bool | None = None,
    ) -> ParameterView[T]:
        """Build a ParameterView for *name* and register it in the current trace context."""
        if requires_grad is None:
            requires_grad = not self._is_frozen(name)
        ref = self._get_or_create_ref(name)
        view = ParameterView(
            value=value,
            name=unique_name(name),
            source=ref,
            derivation=derivation,
            requires_grad=requires_grad,
        )
        from ..trace_context import get_context

        ctx = get_context()
        ctx.inputs.append(view)
        return view

    def save(self, parameter: ParameterView) -> None:
        """Store a parameter's current value."""
        self._save(parameter.source.name, parameter.value)

    def recall(self, name: str, requires_grad: bool | None = None) -> ParameterView[ValueType]:
        """Recall a parameter by name, returning a ParameterView bound to this memory."""
        value, meta = self._recall(name)
        derivation = Derivation(kind="full", meta=meta)
        return self._create_and_register_view(name, value, derivation, requires_grad=requires_grad)

    def query(self, name: str, query: str, requires_grad: bool | None = None) -> ParameterView[str]:
        """Query a parameter with a natural-language question."""
        value, meta = self._query(name, query)
        derivation = Derivation(kind="query", meta={"query": query, **meta})
        return self._create_and_register_view(name, value, derivation, requires_grad=requires_grad)

    def search(self, name: str, query: str, k: int = 5, requires_grad: bool | None = None) -> ParameterView[list[str]]:
        """Search a list parameter and return the top-k matching entries."""
        values, meta = self._search(name, query, k)
        derivation = Derivation(kind="search", meta={"query": query, "top_k": k, **meta})
        return self._create_and_register_view(name, values, derivation, requires_grad=requires_grad)

    def consolidate(self, ref: ParameterRef) -> None:
        """Incorporate feedback from a ParameterRef into storage."""
        self._consolidate(ref.name, ref.gradients)

    # -- Scoped ToolProvider ---------------------------------------------------

    def tool_provider(self, *names: str, operations: set[str] | None = None) -> "DynamicToolProvider":
        """Return a ToolProvider with tools scoped to the given parameter names.

        For each parameter, generates uniquely-named tools (e.g. ``recall_facts``,
        ``search_config_rules``) with descriptions derived from the schema.
        ``search_<name>`` is only generated for list parameters.
        ``save_<name>`` and ``delete_<name>`` are only generated for scalar parameters.

        Args:
            *names: One or more parameter names (slash-separated for nested fields).
            operations: Optional set of tool types to generate. Valid values:
                ``recall``, ``query``, ``search``, ``save``, ``delete``.
                If None, all applicable tools are generated.

        Example::

            # All tools for two parameters
            memory.tool_provider("facts", "config/rules")

            # Read-only access
            memory.tool_provider("facts", operations={"recall", "search", "query"})
        """
        from .utils import is_list_field

        ops = operations or {"recall", "query", "search", "save", "delete"}
        tools: list[AgentTool] = []
        for name in names:
            desc = self._get_description(name) or name
            safe = name.replace("/", "_")
            field_info = self._resolve_field(name)
            is_list = is_list_field(field_info)

            if "recall" in ops:
                tools.append(
                    tool(name=f"recall_{safe}", description=f"Retrieve the full content of: {desc}")(
                        self._make_recall(name)
                    )
                )
            if "query" in ops:
                tools.append(
                    tool(name=f"query_{safe}", description=f"Ask a natural-language question about: {desc}")(
                        self._make_query(name)
                    )
                )
            if "search" in ops and is_list:
                tools.append(
                    tool(name=f"search_{safe}", description=f"Search for relevant entries in: {desc}")(
                        self._make_search(name)
                    )
                )
            if "save" in ops and not is_list:
                tools.append(
                    tool(name=f"save_{safe}", description=f"Overwrite the content of: {desc}")(self._make_save(name))
                )
            if "delete" in ops and not is_list:
                tools.append(
                    tool(name=f"delete_{safe}", description=f"Reset to default: {desc}")(self._make_delete(name))
                )

        return DynamicToolProvider(tools)

    # -- Tool factory methods (capture name via closure) -----------------------

    def _make_recall(self, param_name: str) -> Callable[[], ValueType]:
        def _recall() -> ValueType:
            """Retrieve the full content of this memory parameter."""
            return self.recall(param_name).value

        return _recall

    def _make_query(self, param_name: str) -> Callable[[str], str]:
        def _query(query: str) -> str:
            """Ask a question about this memory parameter.

            Args:
                query: The natural-language question to answer.
            """
            return self.query(param_name, query).value

        return _query

    def _make_search(self, param_name: str) -> Callable[..., list[str]]:
        def _search(query: str, k: int = 5) -> list[str]:
            """Search for relevant entries in this memory parameter.

            Args:
                query: Keywords or phrase to match against entries.
                k: Maximum number of results to return.
            """
            return self.search(param_name, query, k).value

        return _search

    def _make_save(self, param_name: str) -> Callable[[str], str]:
        def _save(value: str) -> str:
            """Overwrite this memory parameter with a new value.

            Args:
                value: The new value to store.
            """
            self._save(param_name, value)
            return "Saved"

        return _save

    def _make_delete(self, param_name: str) -> Callable[[], str]:
        def _delete() -> str:
            """Reset this memory parameter to its default value."""
            self.delete(param_name)
            return "Deleted"

        return _delete

    # -- Context manager -------------------------------------------------------

    def __enter__(self) -> Self:
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit the context manager and close resources."""
        self.close()


class DynamicToolProvider(ToolProvider):
    """ToolProvider that holds a pre-built list of dynamically generated tools."""

    def __init__(self, tools: list[AgentTool]) -> None:
        """Initialize with a pre-built list of tools."""
        self._tools = tools
        self._consumers: set[Any] = set()

    async def load_tools(self, **kwargs: Any) -> Sequence[AgentTool]:
        """Return the generated tools."""
        return self._tools

    def add_consumer(self, consumer_id: Any, **kwargs: Any) -> None:
        """Add a consumer."""
        self._consumers.add(consumer_id)

    def remove_consumer(self, consumer_id: Any, **kwargs: Any) -> None:
        """Remove a consumer."""
        self._consumers.discard(consumer_id)
