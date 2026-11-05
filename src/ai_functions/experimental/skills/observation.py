"""Optional, isolated observation of skill learning and execution.

``with observe_skills(sink): ...`` sends JSON-compatible facts to a synchronous
callback. It adds no storage or UI dependency and never changes prompts, tools,
or admission decisions. Await outstanding rounds before leaving the context;
callbacks inherited by background tasks stop when that context closes.

Custom miners and gates can report their internal stages with
``skill_event("stage.started", stage="extract", ...)``. Library, function, run,
and round labels are supplied by the enclosing skillify operation.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from typing import Any

Observer = Callable[[dict[str, Any]], None]
__all__ = ["Observer", "observe_skills", "skill_event", "trace_observed"]
_sink: ContextVar[list[Observer] | None] = ContextVar("skill_observer", default=None)
_labels: ContextVar[dict[str, Any] | None] = ContextVar("skill_observation_labels", default=None)
_logger = logging.getLogger(__name__)


def _json(value: Any) -> Any:  # noqa: ANN401
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


def _deliver(sinks: list[Observer] | None, labels: dict, kind: str, data: dict) -> None:
    if not sinks:
        return
    try:
        # Copy before dispatch: a viewer cannot mutate live skills, reports, or events.
        event = json.loads(
            json.dumps({"schema_version": 1, "timestamp": time.time(), **labels, **data, "kind": kind}, default=_json)
        )
        sinks[0](event)
    except Exception:  # noqa: BLE001 - observability must never affect the task
        _logger.warning("skill observer failed for %s", kind, exc_info=True)


def skill_event(kind: str, **data: Any) -> None:
    """Report an observation fact; a missing or failing observer never interrupts work."""
    _deliver(_sink.get(), _labels.get() or {}, kind, data)


@contextmanager
def observe_skills(observer: Observer | None = None, **context: Any) -> Iterator[None]:
    """Observe skillify calls and rounds in this async context, including child tasks.

    The callback must return promptly. Exceptions are logged and suppressed.
    Payloads may contain skill text, prompts, and tool results: choose the sink
    and retention policy appropriate for the application. With no callback,
    inherit the current observer and add labels for a custom driver or stage.
    """
    sinks = [observer] if observer is not None else _sink.get()
    token = _sink.set(sinks)
    labels = _labels.set({**(_labels.get() or {}), **context})
    try:
        yield
    finally:
        if observer is not None and sinks is not None:
            sinks.clear()
        _sink.reset(token)
        _labels.reset(labels)


@contextmanager
def _scope(**labels: Any) -> Iterator[None]:
    token = _labels.set({**(_labels.get() or {}), **labels})
    try:
        yield
    finally:
        _labels.reset(token)


@contextmanager
def _watch_thread(coordinator: Any, thread_id: Any) -> Iterator[None]:
    sinks, labels = _sink.get(), dict(_labels.get() or {})
    if not sinks:
        yield
        return

    def receive(event: Any) -> None:
        # Complete messages retain the text; per-token writes would impede streaming.
        if event.kind not in ("message_assistant_token", "message_assistant_thinking"):
            _deliver(sinks, labels, "run.event", {"event": event})

    subscription = None
    try:
        subscription = coordinator.on(receive, thread_id=thread_id)
    except Exception:  # noqa: BLE001 - a remote coordinator may not offer live subscriptions
        _logger.warning("skill observer could not subscribe; completed trace remains available", exc_info=True)
    try:
        yield
    finally:
        if subscription is not None:
            try:
                subscription.unsubscribe()
            except Exception:  # noqa: BLE001
                _logger.warning("skill observer could not unsubscribe", exc_info=True)


async def _trace(result: Any) -> None:
    if _sink.get():
        try:
            skill_event("run.events", events=await result.coordinator.get_events(result.thread_id))
        except Exception:  # noqa: BLE001
            _logger.warning("skill observer could not read the completed trace", exc_info=True)


async def trace_observed(fn: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Trace an AI Function in a driver's labeled run scope, including failed/live events.

    Composes the function's existing config hook unchanged. The subscription is
    installed when the actual execution context becomes available and is always
    released, including on cancellation or post-condition failure.
    """
    if not _sink.get():
        return await fn.trace(*args, **kwargs)
    original, seen = fn.config.config_hook, set()
    with ExitStack() as subscriptions:

        def config_hook(ctx: Any) -> dict:
            if ctx.thread_id not in seen:
                seen.add(ctx.thread_id)
                subscriptions.enter_context(_watch_thread(ctx.coordinator, ctx.thread_id))
            return original(ctx) if original else {}

        result = await fn.replace(config_hook=config_hook).trace(*args, **kwargs)
        await _trace(result)
        return result
