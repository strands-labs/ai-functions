"""Optional JSON-compatible skill lifecycle observations; no viewer dependencies.

Observers are synchronous and isolated from live state. Their exceptions are
logged, never propagated. Await outstanding rounds before closing a capture.
Core facts cover runs, induction, candidates, gates, publication and history.
Custom miners/gates can add facts with ``skill_event`` under the same scope.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

Observer = Callable[[dict[str, Any]], None]
__all__ = ["Observer", "observe_skills", "skill_event", "trace_observed"]

@contextmanager
def observe_skills(observer: Observer | None = None, **context: Any) -> Iterator[None]:
    """Observe calls/rounds in this context; without a sink, inherit it and add labels."""
    ...

def skill_event(kind: str, **data: Any) -> None:
    """Emit a copied fact with a timestamp and context labels; no-op without an observer."""
    ...

async def trace_observed(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Trace an AI Function in a labeled scope, retaining live and failed-run events."""
    ...
