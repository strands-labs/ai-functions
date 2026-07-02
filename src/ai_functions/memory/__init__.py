"""Typed, optimizable memory backends for AI Functions.

Memory backends store named parameters (strings, lists, code) over a Pydantic
schema and emit a ``ParameterRecalledEvent`` on each read (when given a
coordinator + thread_id), so the computation graph can be reconstructed
post-hoc and optimized.
"""

from .agentcore_backend import AgentCoreMemoryBackend
from .base import MemoryBackend
from .frozen import Frozen
from .json_backend import JSONMemoryBackend
from .procedural import Procedural

__all__ = [
    "AgentCoreMemoryBackend",
    "Frozen",
    "JSONMemoryBackend",
    "MemoryBackend",
    "Procedural",
]
