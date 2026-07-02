"""Internal formatting helpers for optimizer rendering (YAML / truncation)."""

from __future__ import annotations

import json
import uuid
from typing import Any

import yaml

_TRUNCATION_MARKER = " [...truncated...] "


def unique_name(name: str) -> str:
    """Append a 4-character random hex suffix to ``name`` (e.g. ``foo`` -> ``foo_a1b2``)."""
    return f"{name}_{uuid.uuid4().hex[:4]}"


def truncate(value: Any, max_length: int = 500) -> str:  # pyright: ignore[reportExplicitAny]
    """Truncate a value to at most ``max_length`` characters (JSON-encoding non-strings)."""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    else:
        value = str(value).strip()

    if len(value) <= max_length:
        return value

    available = max_length - len(_TRUNCATION_MARKER)
    if available <= 0:
        return value[:max_length]
    prefix_len = available // 2
    suffix_len = available - prefix_len
    return value[:prefix_len] + _TRUNCATION_MARKER + value[len(value) - suffix_len :]


def _str_representer(dumper: yaml.Dumper, data: str) -> yaml.ScalarNode:
    """Represent multi-line strings as unquoted literal blocks (``|``)."""
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data.rstrip("\n") + "\n", style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="")


class _LiteralDumper(yaml.Dumper):
    """YAML dumper that avoids quoting / escaping string values where possible."""

    def choose_scalar_style(self) -> str:  # type: ignore[override]
        if self.event.style == "|":  # type: ignore[union-attr]
            return "|"
        return super().choose_scalar_style()


_LiteralDumper.add_representer(str, _str_representer)


def to_yaml(obj: Any) -> str:  # pyright: ignore[reportExplicitAny]
    """Convert ``obj`` to a human-readable YAML string (literal blocks for multi-line)."""
    return yaml.dump(
        obj,
        Dumper=_LiteralDumper,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    ).rstrip()
