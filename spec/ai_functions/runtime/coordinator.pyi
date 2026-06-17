"""InMemoryCoordinator — default single-process ``Coordinator`` implementation.

Invariants:
    I2, I3.
"""

from __future__ import annotations

from ..protocols import Coordinator


class InMemoryCoordinator(Coordinator):
    """Single-process ``Coordinator``; events in per-thread lists, callbacks in dicts.

    Implements:
        Coordinator.

    Invariants:
        - I2 — sole sink for every event this coordinator serves.
        - I3 — per-thread pause signals are driven by ``TOKEN_USAGE``
          events appended here.
    """
