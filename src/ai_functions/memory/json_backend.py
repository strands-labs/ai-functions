"""JSON-file-backed memory backend.

Stores a memory schema as a Pydantic model serialized to a JSON file,
namespaced per ``actor_id``. Consolidation (merging feedback into values) uses
internal AI functions: value, list, and procedural variants.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from strands.models import Model

from ..ai_thread import ai_function
from ..ai_thread.postcondition import PostConditionResult
from .base import MemoryBackend, ValueType
from .procedural import validate_procedural


def _bullet_points(values: list[str]) -> str:
    return "\n".join(f"- {v}" for v in values)


def _check_valid_python(response: str) -> PostConditionResult:
    """Post-condition: consolidated procedural code must parse as Python."""
    try:
        validate_procedural(response)
    except SyntaxError as exc:
        return PostConditionResult(passed=False, message=f"Code is not valid Python: {exc}")
    return PostConditionResult(passed=True)


@ai_function(list[str])
def _consolidate_list(value: list[str], feedback: list[str]) -> str:
    """Build the prompt to merge feedback into a list-valued parameter."""
    return (
        "Update the following list with the feedback provided.\n"
        "Return the new full list containing all updated information without redundancy.\n\n"
        f"<values>\n{_bullet_points(value)}\n</values>\n\n"
        f"<feedback>\n{_bullet_points(feedback)}\n</feedback>"
    )


@ai_function(str)
def _consolidate_value(value: str, feedback: list[str]) -> str:
    """Build the prompt to merge feedback into a scalar-valued parameter."""
    return (
        "Update the following value with the feedback provided.\n"
        "Return the updated value.\n\n"
        f"<value>\n{value}\n</value>\n\n"
        f"<feedback>\n{_bullet_points(feedback)}\n</feedback>"
    )


@ai_function(str, post_conditions=[_check_valid_python])
def _consolidate_procedural(value: str, feedback: list[str]) -> str:
    """Build the prompt to merge feedback into a code-valued parameter."""
    return (
        "Update the following code with the feedback provided.\n"
        "Return the complete updated code as a string.\n\n"
        f"<code>\n{value}\n</code>\n\n"
        f"<feedback>\n{_bullet_points(feedback)}\n</feedback>"
    )


@ai_function(str)
def _query_value(value: str, query: str) -> str:
    """Build the prompt to answer a question over a parameter's value."""
    return (
        "Based on the content below, answer the following question:\n"
        f"<question>{query}</question>\n"
        f"<content>{value}</content>"
    )


def _get_nested_attr(obj: Any, path: str) -> Any:  # pyright: ignore[reportExplicitAny]
    for part in path.split("/"):
        obj = getattr(obj, part)
    return obj


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:  # pyright: ignore[reportExplicitAny]
    parts = path.split("/")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _is_str_list_field(annotation: Any) -> bool:  # pyright: ignore[reportExplicitAny]
    """Return whether ``annotation`` is ``list[str]`` (possibly Optional-wrapped)."""
    import typing

    origin = typing.get_origin(annotation)
    if origin is list:
        args = typing.get_args(annotation)
        return bool(args) and args[0] is str
    # Unwrap Optional[list[str]] / Annotated[...] and re-check the members.
    for arg in typing.get_args(annotation):
        if _is_str_list_field(arg):
            return True
    return False


class JSONMemoryBackend(MemoryBackend):
    """File-backed memory using JSON serialization."""

    def __init__(
        self,
        schema: type[BaseModel],
        actor_id: str,
        path: Path | str,
        model: Model | str | None = None,
        quiet: bool = True,
    ) -> None:
        super().__init__(schema, actor_id)
        self.path = Path(path)
        self.quiet = quiet

        self._consolidate_value_fn = _consolidate_value.replace(model=model)
        self._consolidate_procedural_fn = _consolidate_procedural.replace(model=model)
        self._consolidate_list_fn = _consolidate_list.replace(model=model)
        self._query_value_fn = _query_value.replace(model=model)

        self._all_actors: dict[str, dict[str, Any]] = {}  # pyright: ignore[reportExplicitAny]
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open() as f:
                self._all_actors = json.load(f)

        if actor_id in self._all_actors:
            self._model = schema.model_validate(self._all_actors[actor_id])
        else:
            self._model = schema()

    def _save(self, name: str, value: ValueType) -> None:
        _set_nested_attr(self._model, name, value)

    def _consolidate(self, name: str, feedback: list[str], **kwargs: Any) -> None:  # pyright: ignore[reportExplicitAny]
        value = _get_nested_attr(self._model, name)
        if self._is_procedural(name):
            value = self._consolidate_procedural_fn.run_sync(value=value, feedback=feedback)
        elif isinstance(value, list):
            # The list consolidator is typed list[str]; routing a list[BaseModel]
            # through it would silently store strings back into a model field and
            # corrupt the schema (fails to reload). Reject it explicitly.
            if not _is_str_list_field(self._resolve_field(name).annotation):
                raise NotImplementedError(
                    f"JSONMemoryBackend cannot consolidate non-string list parameter '{name}'. "
                    f"Only list[str] (and str / Procedural) fields are supported."
                )
            value = self._consolidate_list_fn.run_sync(value=value, feedback=feedback)
        else:
            value = self._consolidate_value_fn.run_sync(value=value, feedback=feedback)
        _set_nested_attr(self._model, name, value)

    def _delete(self, name: str) -> None:
        """Reset a parameter to its schema default."""
        field_info = self._resolve_field(name)
        # A required field has no default to reset to; surface it clearly.
        if field_info.is_required():
            raise ValueError(f"Cannot delete required parameter '{name}': it has no schema default.")
        _set_nested_attr(self._model, name, field_info.get_default(call_default_factory=True))

    def _recall(self, name: str) -> ValueType:
        return _get_nested_attr(self._model, name)

    def _query(self, name: str, query: str) -> str:
        value = _get_nested_attr(self._model, name)
        return self._query_value_fn.run_sync(value=value, query=query)

    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> list[str]:  # pyright: ignore[reportExplicitAny]
        """Return the top-k entries of a list parameter, ranked by BM25 against query.

        Only supported for ``list[str]`` parameters. Requires the optional
        ``rank-bm25`` dependency (``pip install strands-ai-functions[search]``).
        """
        value = _get_nested_attr(self._model, name)
        if not isinstance(value, list):
            raise TypeError(f"search() is only supported for list parameters, but '{name}' is {type(value).__name__}")
        corpus = [str(item) for item in value]
        if not corpus:
            return []

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "JSONMemoryBackend.search requires the 'rank-bm25' package. "
                "Install it with: pip install strands-ai-functions[search]"
            ) from exc

        bm25 = BM25Okapi([doc.lower().split() for doc in corpus])
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(corpus, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [doc for doc, _ in ranked[:k]]

    def close(self) -> None:
        """Persist this actor's current values back to the JSON file.

        Re-reads the file and merges in only this actor's key, so concurrent
        backends sharing one file do not clobber each other (the in-memory
        snapshot taken at construction may be stale). Writes atomically
        (temp-file + rename) so a crash mid-write cannot corrupt the file for
        other actors.
        """
        on_disk: dict[str, dict[str, Any]] = {}  # pyright: ignore[reportExplicitAny]
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open() as f:
                on_disk = json.load(f)

        on_disk[self.actor_id] = self._model.model_dump()
        self._all_actors = on_disk

        # Atomic write: temp file in the same dir, then os.replace (matches the
        # pattern in discovery.py / session.py). A crash mid-write leaves the
        # prior file intact for all actors.
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(on_disk, fh)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def dump(self) -> BaseModel:
        """Return the underlying Pydantic model."""
        return self._model

    def __str__(self) -> str:
        """Return a human-readable YAML dump of the current values.

        Uses ``to_yaml`` so multi-line values (e.g. ``Procedural`` code) render
        as unquoted literal blocks rather than escaped one-line scalars.
        """
        from ..optimizer._formatting import to_yaml

        return to_yaml(self._model.model_dump())
