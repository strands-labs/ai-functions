"""AWS Bedrock AgentCore memory backend for AI Function parameters.

Stores each parameter as conversation turns in an AgentCore Memory resource.
AgentCore's semantic-memory strategy extracts and consolidates feedback into
long-term memory automatically, so ``_consolidate`` simply appends feedback as
turns rather than running an explicit merge AI function (as ``JSONMemoryBackend``
does).

``bedrock-agentcore`` is an optional dependency: importing this module is cheap,
but constructing :class:`AgentCoreMemoryBackend` raises ``ImportError`` if the
package is not installed (``pip install strands-ai-functions[agentcore]``).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from strands.models import Model

from .base import MemoryBackend, ValueType
from .json_backend import _query_value

logger = logging.getLogger(__name__)

# Maximum number of memory records to retrieve in a single operation.
MAX_MEMORY_RECORDS = 100

if TYPE_CHECKING:
    from bedrock_agentcore.memory import MemorySession


def _require_agentcore() -> Any:  # pyright: ignore[reportExplicitAny]
    """Import and return the ``bedrock_agentcore.memory`` module, or raise.

    Raises:
        ImportError: ``bedrock-agentcore`` is not installed.
    """
    try:
        import bedrock_agentcore.memory as acm
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "AgentCoreMemoryBackend requires the 'bedrock-agentcore' package. "
            "Install it with: pip install strands-ai-functions[agentcore]"
        ) from exc
    return acm


def _extract_event_texts(events: list[Any]) -> list[str]:  # pyright: ignore[reportExplicitAny]
    """Extract text content from short-term-memory events."""
    return [
        text
        for event in events
        if isinstance(event.get("payload", []), list)
        for item in event.get("payload", [])
        if "conversational" in item
        for text in [item["conversational"].get("content", {}).get("text", "")]
        if text
    ]


def _extract_record_texts(records: list[Any]) -> list[str]:  # pyright: ignore[reportExplicitAny]
    """Extract text content from long-term-memory records."""
    texts: list[str] = []
    for record in records:
        content = record.get("content", {})
        text = content.get("text", "").strip() if isinstance(content, dict) else str(content).strip()
        if text:
            texts.append(text)
    return texts


def create_memory(name: str, region_name: str = "us-east-1") -> str:
    """Create an AgentCore Memory resource and return its memory id."""
    acm = _require_agentcore()
    client = acm.MemoryClient(region_name=region_name)
    memory = client.create_memory_and_wait(
        name=name,
        description="AI Function memory (parameters, gradients, conversations)",
        strategies=[
            {
                "semanticMemoryStrategy": {
                    "name": "SemanticExtractor",
                    "description": "Extract reusable knowledge from all memory events",
                    "namespaces": ["/{actorId}/"],
                }
            }
        ],
    )
    memory_id: str = memory["id"]
    logger.info("Created memory '%s' with id: %s", name, memory_id)
    return memory_id


def _get_memory_id(name: str, region_name: str = "us-east-1") -> str:
    """Return the id of an existing memory named ``name``, creating it if absent."""
    acm = _require_agentcore()
    client = acm.MemoryClient(region_name=region_name)
    for m in client.list_memories():
        mid: str = m["memoryId"]
        if _memory_id_matches(mid, name):
            logger.info("Found existing memory '%s' with id: %s", name, mid)
            return mid
    logger.info("Memory '%s' does not exist. Creating it now.", name)
    return create_memory(name, region_name)


def _memory_id_matches(memory_id: str, name: str) -> bool:
    """Whether an AgentCore ``memory_id`` corresponds to memory ``name``.

    AgentCore ids have the form ``"{name}-{hash}"``. Match the exact name or
    that prefix — NOT ``split("-")[0]``, which compares only the first token
    and so never matches a hyphenated name and collides distinct names sharing
    a first token (e.g. ``writing-a`` vs ``writing-b``).
    """
    return memory_id == name or memory_id.startswith(f"{name}-")


class AgentCoreMemoryBackend(MemoryBackend):
    """AWS Bedrock AgentCore-backed memory for parameters.

    Each parameter is stored under its own AgentCore actor namespace; recall
    concatenates the actor's short-term events and long-term records. Procedural
    fields are not supported — use :class:`JSONMemoryBackend` for those.
    """

    def __init__(
        self,
        schema: type[BaseModel],
        actor_id: str,
        memory_id: str | None = None,
        memory_name: str | None = None,
        session_id: str | None = None,
        region_name: str = "us-east-1",
        model: Model | str | None = None,
    ) -> None:
        """Open an AgentCore-backed memory store for one actor.

        Args:
            schema: Pydantic model describing the memory parameters.
            actor_id: Unique identifier for this actor.
            memory_id: AgentCore memory id, if already known.
            memory_name: Memory name to get-or-create (alternative to ``memory_id``).
            session_id: Optional session id; auto-generated when omitted.
            region_name: AWS region name.
            model: Model (or id) the internal ``query`` AI function runs on.

        Raises:
            ImportError: ``bedrock-agentcore`` is not installed.
            ValueError: The schema contains a Procedural field, or neither /
                both of ``memory_id`` and ``memory_name`` were provided.
        """
        super().__init__(schema, actor_id)
        acm = _require_agentcore()

        self._validate_no_procedural_fields()

        if memory_id is None and memory_name is None:
            raise ValueError("Either memory_id or memory_name must be provided")
        if memory_id is not None and memory_name is not None:
            raise ValueError("Cannot provide both memory_id and memory_name")

        if memory_name is not None:
            self.memory_id = _get_memory_id(memory_name, region_name=region_name)
        else:
            assert memory_id is not None  # guaranteed by validation above
            self.memory_id = memory_id

        self.region_name = region_name
        self.session_id = session_id or f"session_{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        self.manager = acm.MemorySessionManager(memory_id=self.memory_id, region_name=region_name)
        self._sessions: dict[str, MemorySession] = {}
        self._query_value_fn = _query_value.replace(model=model)

    def _validate_no_procedural_fields(self) -> None:
        """Raise if the schema contains any Procedural field (unsupported here)."""
        for field_name in self.schema.model_fields:
            if self._is_procedural(field_name):
                raise ValueError(
                    f"AgentCoreMemoryBackend does not support Procedural fields. "
                    f"Field '{field_name}' in schema '{self.schema.__name__}' is marked as Procedural. "
                    f"Use JSONMemoryBackend for schemas with Procedural parameters."
                )

    def _parameter_actor(self, name: str) -> str:
        """Return the AgentCore actor namespace for parameter ``name``."""
        return f"{self.actor_id}/{name}"

    def _get_session(self, actor_id: str) -> MemorySession:
        """Get or create a memory session for an actor."""
        if actor_id not in self._sessions:
            self._sessions[actor_id] = self.manager.create_memory_session(
                actor_id=actor_id,
                session_id=self.session_id,
            )
        return self._sessions[actor_id]

    def _retrieve_raw(
        self, actor_id: str, query: str | None = None, top_k: int = MAX_MEMORY_RECORDS
    ) -> tuple[list[Any], list[Any]]:  # pyright: ignore[reportExplicitAny]
        """Fetch raw STM events and LTM records for an actor, balanced to ``top_k``."""
        session = self._get_session(actor_id)
        ns = f"/{actor_id}/"
        half_k = top_k // 2

        try:
            stm_events = session.list_events(include_payload=True, max_results=top_k)[::-1]
        except Exception as e:  # noqa: BLE001 - external service; degrade gracefully
            logger.warning("retrieve STM error for '%s': %s: %s", actor_id, type(e).__name__, e)
            stm_events = []

        ltm_quota = top_k - min(len(stm_events), half_k)
        try:
            ltm_records = (
                session.search_long_term_memories(query=query, namespace_prefix=ns, top_k=ltm_quota)
                if query is not None
                else self.manager.list_long_term_memory_records(namespace_prefix=ns, max_results=ltm_quota)
            )
        except Exception as e:  # noqa: BLE001 - external service; degrade gracefully
            logger.warning("retrieve LTM error for '%s': %s: %s", actor_id, type(e).__name__, e)
            ltm_records = []

        stm_take = min(len(stm_events), half_k)
        ltm_take = min(len(ltm_records), top_k - stm_take)
        return stm_events[:stm_take], ltm_records[:ltm_take]

    def _save(self, name: str, value: ValueType) -> None:
        """Replace the stored value of a parameter."""
        acm = _require_agentcore()
        actor = self._parameter_actor(name)
        self._delete_records(name, wait=False)

        text = "\n\n".join(value) if isinstance(value, list) else str(value)
        session = self._get_session(actor)
        session.add_turns(
            messages=[acm.constants.ConversationalMessage(text, acm.constants.MessageRole.USER)],
            metadata={"type": {"stringValue": "parameter"}, "name": {"stringValue": name}},
            event_timestamp=datetime.now(UTC),
        )

    def _recall(self, name: str) -> ValueType:
        """Return a parameter's value, concatenated from STM events and LTM records."""
        actor = self._parameter_actor(name)
        events, records = self._retrieve_raw(actor, query=None, top_k=MAX_MEMORY_RECORDS)
        all_texts = _extract_event_texts(events) + _extract_record_texts(records)

        instance = self.schema()
        default_value = getattr(instance, name)
        if not all_texts:
            return default_value

        concatenated = "\n\n".join(all_texts)
        if isinstance(default_value, list):
            return [item.strip() for item in concatenated.split("\n\n") if item.strip()]
        return concatenated

    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> list[str]:  # pyright: ignore[reportExplicitAny]
        """Return the top-k texts most relevant to ``query`` for a parameter."""
        actor = self._parameter_actor(name)
        events, records = self._retrieve_raw(actor, query=query, top_k=k)
        return _extract_event_texts(events) + _extract_record_texts(records)

    def _query(self, name: str, query: str) -> str:
        """Answer ``query`` over the most relevant content for a parameter."""
        relevant_texts = self._search(name, query, k=10)
        if not relevant_texts:
            return ""
        content = "\n\n".join(relevant_texts)
        return self._query_value_fn.run_sync(value=content, query=query)

    def _consolidate(self, name: str, feedback: list[str], **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Append feedback as conversation turns for AgentCore to consolidate.

        Unlike :class:`JSONMemoryBackend` (which runs an explicit merge AI
        function), AgentCore's semantic-memory strategy extracts and
        consolidates these turns into long-term memory on its own.
        """
        if not feedback:
            return
        acm = _require_agentcore()
        actor = self._parameter_actor(name)
        session = self._get_session(actor)
        for item in feedback:
            session.add_turns(
                messages=[acm.constants.ConversationalMessage(item, acm.constants.MessageRole.USER)],
                metadata={"type": {"stringValue": "feedback"}, "name": {"stringValue": name}},
                event_timestamp=datetime.now(UTC),
            )

    # -- Record management -----------------------------------------------------

    def record_counts(self, name: str | None = None) -> tuple[int, int]:
        """Return ``(stm_count, ltm_count)`` for a parameter, or all fields if ``None``."""
        if name is not None:
            actor = self._parameter_actor(name)
            events, records = self._retrieve_raw(actor, query=None, top_k=MAX_MEMORY_RECORDS)
            return len(events), len(records)

        total_stm, total_ltm = 0, 0
        for field_name in self.schema.model_fields:
            stm, ltm = self.record_counts(field_name)
            total_stm += stm
            total_ltm += ltm
        return total_stm, total_ltm

    def delete(self, name: str) -> None:
        """Delete a parameter, removing all its STM events and LTM records."""
        self._delete_records(name, wait=True)

    def _delete(self, name: str) -> None:
        """Reset a parameter, removing all its STM events and LTM records."""
        self._delete_records(name, wait=True)

    def delete_all(self, wait: bool = False) -> None:
        """Delete every field's memories for this actor (fire-and-forget by default)."""
        for field_name in self.schema.model_fields:
            self._delete_records(field_name, wait=False)
        if wait:
            for field_name in self.schema.model_fields:
                self._wait_until_empty(field_name)

    def _delete_records(self, name: str, wait: bool = True) -> None:
        """Delete all STM events and LTM records for a parameter."""
        actor = self._parameter_actor(name)
        ns = f"/{actor}/"
        events, records = self._retrieve_raw(actor, query=None, top_k=MAX_MEMORY_RECORDS)
        if not events and not records:
            return

        for eid in (e.get("eventId") for e in events):
            if eid:
                try:
                    self.manager.delete_event(actor_id=actor, session_id=self.session_id, event_id=eid)
                except Exception as e:  # noqa: BLE001 - best-effort cleanup
                    logger.warning("Failed to delete STM event '%s': %s", eid, e)

        if records:
            try:
                self.manager.delete_all_long_term_memories_in_namespace(namespace=ns)
            except Exception as e:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Failed to bulk delete LTM records in '%s': %s", ns, e)

        if wait:
            self._wait_until_empty(name)

    def _wait_until_empty(self, name: str, max_wait: int = 180, poll_interval: int = 5) -> None:
        """Poll until all STM and LTM records for a parameter are gone."""
        import time

        elapsed = 0
        while elapsed < max_wait:
            stm, ltm = self.record_counts(name)
            if stm == 0 and ltm == 0:
                return
            time.sleep(poll_interval)
            elapsed += poll_interval
        logger.warning("Timed out waiting for records to be deleted for '%s'", name)

    def close(self) -> None:
        """Release resources held by this backend."""
        logger.info("AgentCoreMemoryBackend closed (memory_id: %s)", self.memory_id)

    def __str__(self) -> str:
        """Return a YAML representation of the current memory state."""
        from ..optimizer._formatting import to_yaml

        data: dict[str, ValueType] = {field_name: self._recall(field_name) for field_name in self.schema.model_fields}
        return to_yaml(self.schema(**data).model_dump())
