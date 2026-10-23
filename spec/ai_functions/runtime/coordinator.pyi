"""InMemoryCoordinator — default single-process ``Coordinator`` implementation.

Invariants:
    I2, I3.
"""

from __future__ import annotations

from ..protocols import Coordinator
from .prices import PriceTable


class InMemoryCoordinator(Coordinator):
    """Single-process ``Coordinator``; events in per-thread lists, callbacks in dicts.

    Implements:
        Coordinator.

    Args:
        prices: The table that prices each ``TOKEN_USAGE`` event by the model
            it names, for a caller that folds a run's usage into dollars (a
            team totaling its run). :data:`~ai_functions.runtime.prices.PRICES`
            by default; pass ``PRICES.with_prices({...})`` to add or correct a
            model.

    Invariants:
        - I2 — sole sink for every event this coordinator serves.
        - I3 — per-thread pause signals are driven by ``TOKEN_USAGE``
          events appended here.
    """

    def __init__(self, *, prices: PriceTable = ...) -> None: ...
    @property
    def prices(self) -> PriceTable:
        """The table this coordinator prices ``TOKEN_USAGE`` events with.

        Read by a caller that folds usage itself and must price it the same way
        every time, such as a team totaling its run.
        """
        ...
