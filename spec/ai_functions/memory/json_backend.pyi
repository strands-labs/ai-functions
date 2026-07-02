"""JSON-file-backed memory backend.

Stores a memory schema as a Pydantic model serialized to a JSON file,
namespaced per ``actor_id``. Consolidation (merging feedback into values) is
delegated to internal AI functions: value, list, and procedural variants.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel
from strands.models import Model

from .base import MemoryBackend


class JSONMemoryBackend(MemoryBackend):
    """File-backed memory using JSON serialization.

    The file holds a mapping ``{actor_id: schema-dump}``; multiple actors may
    share one file. Construction loads this actor's slice (or an empty schema
    instance); :meth:`close` writes it back.
    """

    path: Path
    quiet: bool

    def __init__(
        self,
        schema: type[BaseModel],
        actor_id: str,
        path: Path | str,
        model: Model | str | None = None,
        quiet: bool = True,
    ) -> None:
        """Open a JSON-backed memory store for one actor.

        Args:
            schema: Pydantic model describing the memory parameters.
            actor_id: Namespaces this actor's values within ``path``.
            path: JSON file backing the store; created on :meth:`close` if
                absent.
            model: Model (or model id) the consolidation / query AI functions
                run on. ``None`` uses the library default provider.
            quiet: Suppress internal AI function callback output.

        Ensures:
            - If ``path`` exists and contains ``actor_id``, the stored values
              are loaded and validated against ``schema``.
            - Otherwise the store starts from a default ``schema()`` instance.
        """
        ...

    def close(self) -> None:
        """Persist this actor's current values back to ``path``.

        Ensures:
            - ``path`` contains this actor's serialized schema under
              ``actor_id``, preserving any other actors already in the file.
        """
        ...

    def dump(self) -> BaseModel:
        """Return the underlying Pydantic model holding current values."""
        ...

    def __str__(self) -> str:
        """Return a human-readable YAML dump of the current values."""
        ...
