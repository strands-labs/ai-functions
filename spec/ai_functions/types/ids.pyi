"""Semantic ``NewType`` aliases for opaque string identifiers."""

from __future__ import annotations

from typing import NewType

ThreadId = NewType("ThreadId", str)
"""Runtime-assigned identifier for a registered thread."""

EventId = NewType("EventId", str)
"""Stable event id; never reused; used as the ``since_id`` cursor."""

MessageId = NewType("MessageId", str)
"""Ties the fragments of a single assistant turn together."""

WorkerId = NewType("WorkerId", str)
"""Server-assigned identifier for a ``WorkerProcess``."""
