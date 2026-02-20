"""Type utility functions for AI Functions.

This module provides helper functions for type introspection and validation.
"""

import inspect

from pydantic import BaseModel, TypeAdapter


def is_pydantic_model(type_: type) -> bool:
    """Check if a type is a Pydantic model.

    Args:
        type_: The type to check

    Returns:
        True if the type is a Pydantic BaseModel subclass
    """
    return isinstance(type_, type) and issubclass(type_, BaseModel)


def is_json_serializable_type(type_: type) -> bool:
    """Check if a type can be serialized to/from JSON using Pydantic's TypeAdapter.

    Args:
        type_: The type to check

    Returns:
        True if the type is JSON-serializable, False otherwise
    """
    # Pydantic models are always JSON-serializable
    if is_pydantic_model(type_):
        return True

    # Use Pydantic's TypeAdapter as the authoritative check
    try:
        adapter: TypeAdapter[type] = TypeAdapter(type_)
        adapter.json_schema(mode="serialization")
        return True
    except Exception:
        return False


def generate_signature_from_model(model: type[BaseModel], func_name: str = "final_answer") -> str:
    """Generate function signature corresponding to the constructor of a pydantic model."""
    # Create parameter list
    params = []
    for field_name, field_info in model.model_fields.items():
        annotation = field_info.annotation
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

    # Create signature
    sig = inspect.Signature(params)
    return f"{func_name}{sig}"
