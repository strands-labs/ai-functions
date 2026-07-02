"""AWS Bedrock AgentCore memory backend for AI Function parameters.

Stores each parameter as conversation turns in an AgentCore Memory resource.
AgentCore's semantic-memory strategy extracts and consolidates feedback into
long-term memory automatically, so ``consolidate`` appends feedback as turns
rather than running an explicit merge AI function (as ``JSONMemoryBackend``
does).

``bedrock-agentcore`` is an optional dependency: importing this module is cheap,
but constructing :class:`AgentCoreMemoryBackend` raises ``ImportError`` if the
package is not installed (``pip install strands-ai-functions[agentcore]``).
"""

from __future__ import annotations

from pydantic import BaseModel
from strands.models import Model

from .base import MemoryBackend

MAX_MEMORY_RECORDS: int

def create_memory(name: str, region_name: str = ...) -> str:
    """Create an AgentCore Memory resource and return its memory id.

    Args:
        name: Human-readable name for the memory resource.
        region_name: AWS region to create the resource in.

    Returns:
        The created memory's id.

    Raises:
        ImportError: ``bedrock-agentcore`` is not installed.
    """
    ...

class AgentCoreMemoryBackend(MemoryBackend):
    """AWS Bedrock AgentCore-backed memory for parameters.

    Each parameter is stored under its own AgentCore actor namespace; recall
    concatenates the actor's short-term events and long-term records.
    Procedural fields are not supported — use :class:`JSONMemoryBackend` for
    those.
    """

    memory_id: str
    region_name: str
    session_id: str

    def __init__(
        self,
        schema: type[BaseModel],
        actor_id: str,
        memory_id: str | None = None,
        memory_name: str | None = None,
        session_id: str | None = None,
        region_name: str = ...,
        model: Model | str | None = None,
    ) -> None:
        """Open an AgentCore-backed memory store for one actor.

        Args:
            schema: Pydantic model describing the memory parameters.
            actor_id: Unique identifier for this actor.
            memory_id: AgentCore memory id, if already known.
            memory_name: Memory name to get-or-create (alternative to
                ``memory_id``).
            session_id: Optional session id; auto-generated when omitted.
            region_name: AWS region name.
            model: Model (or id) the internal ``query`` AI function runs on.

        Raises:
            ImportError: ``bedrock-agentcore`` is not installed.
            ValueError: The schema contains a Procedural field, or neither /
                both of ``memory_id`` and ``memory_name`` were provided.
        """
        ...

    def record_counts(self, name: str | None = None) -> tuple[int, int]:
        """Return ``(stm_count, ltm_count)`` for a parameter, or all fields if ``None``.

        Args:
            name: Parameter to count, or ``None`` to sum across every field.

        Returns:
            A ``(short_term_count, long_term_count)`` pair.
        """
        ...

    def delete(self, name: str) -> None:
        """Delete a parameter, removing all its STM events and LTM records.

        Args:
            name: Parameter to delete.

        Ensures:
            - Blocks until the parameter's records are confirmed removed.
        """
        ...

    def delete_all(self, wait: bool = False) -> None:
        """Delete every field's memories for this actor.

        Args:
            wait: When ``True``, block until every field is confirmed empty;
                otherwise fire the deletes and return immediately.
        """
        ...

    def close(self) -> None:
        """Release resources held by this backend."""
        ...

    def __str__(self) -> str:
        """Return a YAML representation of the current memory state."""
        ...
