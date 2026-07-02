"""Type introspection helpers (ported from the ai-functions mainline).

Internal utilities shared by the output-spec logic (``ai_thread``) and the
sandboxed executor (``tools``): detecting pydantic models, checking JSON
serializability (which gates structured output), and rendering a pydantic
model's constructor as a ``final_answer(...)`` signature for the code executor.
"""

from __future__ import annotations

import inspect
import typing
from typing import get_args, get_origin

from pydantic import BaseModel, TypeAdapter


def is_pydantic_model(type_: type) -> bool:
    """Return whether ``type_`` is a Pydantic ``BaseModel`` subclass."""
    return isinstance(type_, type) and issubclass(type_, BaseModel)


def is_json_serializable_type(type_: type) -> bool:
    """Return whether ``type_`` can produce a JSON schema via Pydantic.

    Pydantic models are always serializable; other types are tested by
    attempting to build a serialization JSON schema through ``TypeAdapter``.
    Used to decide whether structured output is possible for a return type.
    """
    if is_pydantic_model(type_):
        return True
    try:
        adapter: TypeAdapter[type] = TypeAdapter(type_)
        adapter.json_schema(mode="serialization")
        return True
    except Exception:  # noqa: BLE001 — any failure means "not JSON-serializable"
        return False


def _simplify_annotation(annotation: type | None) -> type:
    """Replace Pydantic model types with ``dict`` in a type annotation.

    Applied recursively so generic aliases like ``list[MyModel]`` become
    ``list[dict]`` — used to render a readable ``final_answer`` signature.
    """
    if annotation is None:
        return type(None)
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        new_args = tuple(_simplify_annotation(a) for a in args)
        if origin is typing.Union:
            return typing.Union[new_args]  # type: ignore[no-any-return]  # noqa: UP007
        try:
            return origin[new_args]  # type: ignore[no-any-return]
        except TypeError:
            return annotation
    if is_pydantic_model(annotation):
        return dict
    return annotation


def generate_signature_from_model(model: type[BaseModel], func_name: str = "final_answer") -> str:
    """Render a pydantic model's constructor as a ``func_name(...)`` signature string."""
    params: list[inspect.Parameter] = []
    for field_name, field_info in model.model_fields.items():
        annotation = _simplify_annotation(field_info.annotation)
        if field_info.is_required():
            params.append(inspect.Parameter(field_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation))
        else:
            params.append(
                inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=field_info.default,
                    annotation=annotation,
                )
            )
    # Required (no default) params must precede optional ones.
    params.sort(key=lambda p: p.default is not inspect.Parameter.empty)
    return f"{func_name}{inspect.Signature(params)}"
