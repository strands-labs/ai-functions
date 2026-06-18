"""Non-TUI ``ai_functions`` subcommands — ``ps``, ``logs``, ``notify``, ``submit``, ``kill``, ``run``.

Each function is a Typer command body, callable directly from tests
via ``typer.testing.CliRunner`` on the top-level
:data:`ai_functions.cli.app` app. Every command follows the same skeleton:

1. Resolve the coordinator URL (``--url`` > env var > runtime file).
2. Open a short-lived :class:`CoordinatorClient`.
3. Issue the relevant RPC(s).
4. Close the client; print one line per record or emit a formatted
   :class:`RenderableType` for each event.

``run_cmd`` is the odd one out: it loads a user script, locates the
main :class:`Spawnable`, and hands it to :func:`ai_functions.serve`. It does
not open its own client — :func:`ai_functions.serve` does.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import cast

import typer
from pydantic import BaseModel
from rich.console import Console

from ..connect import connect
from ..discovery import NoCoordinatorError
from ..handle import ThreadHandle
from ..network import ConnectionClosedError
from ..protocols import Spawnable
from ..runtime.errors import ThreadNotFoundError
from ..serve import aserve
from ..types import Event, EventId, InputShape, ThreadId, ThreadInfo, TokenUsage, TokenUsageEvent
from .events import filter_events, format_event


def ps() -> int:
    """Implementation of ``ai-functions ps``.

    Prints one line per registered thread: ``thread_id`` (short form),
    ``status``, ``input_shape``, ``thread_name``, ``worker_id``. Output
    format mimics ``docker ps``: a header row followed by one row per
    thread, padded for terminal readability.

    Returns:
        Exit code; ``0`` if the list was fetched (even when empty),
        ``2`` if no coordinator was discovered.
    """

    async def _run() -> int:
        try:
            async with connect() as client:
                threads = await client.list_threads()
        except NoCoordinatorError as exc:
            typer.echo(f"error: {exc}", err=True)
            return 2
        except OSError as exc:
            typer.echo(f"error: could not reach coordinator: {exc}", err=True)
            return 2

        if not threads:
            typer.echo("no threads registered")
            return 0

        header = f"{'THREAD ID':<24} {'STATUS':<12} {'SHAPE':<12} {'NAME':<20} WORKER"
        typer.echo(header)
        for info in threads:
            tid = str(info.thread_id)
            short = tid if len(tid) <= 24 else tid[:21] + "..."
            typer.echo(
                f"{short:<24} {info.status.value:<12} "
                f"{info.input_shape.value:<12} {(info.thread_name or '-'):<20} "
                f"{info.worker_id}",
            )
        return 0

    return asyncio.run(_run())


def logs(thread_id: ThreadId, *, follow: bool = False, since: str | None = None) -> int:
    """Implementation of ``ai-functions logs <thread-id>``.

    Replays every stored event for ``thread_id`` using
    :meth:`Coordinator.get_events`, pretty-printing each through
    :func:`ai_functions.cli.events.format_event`. With ``--follow``, the
    command then subscribes via :meth:`Coordinator.on` and streams new
    events until Ctrl-C.

    Args:
        thread_id: Thread whose events to dump.
        follow: Keep the stream open after replay and tail new events.
        since: If set, an event-id string; only events strictly after
            this id are returned. Passed through to
            ``get_events(since_id=...)``.

    Returns:
        ``0`` on normal exit, ``1`` if the thread was not found,
        ``130`` if interrupted during ``--follow``.
    """
    console = Console()

    async def _run() -> int:
        try:
            async with connect() as client:
                since_id: EventId | None = None if since is None else EventId(since)
                try:
                    events = await client.get_events(thread_id, since_id=since_id)
                except ThreadNotFoundError:
                    typer.echo(f"error: thread '{thread_id}' not found", err=True)
                    return 1

                for event in events:
                    if filter_events(event):
                        console.print(format_event(event))

                if not follow:
                    return 0

                queue: asyncio.Queue[Event] = asyncio.Queue()
                loop = asyncio.get_running_loop()

                def _emit(event: Event) -> None:
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    except RuntimeError:
                        pass

                sub = client.on(_emit, thread_id=thread_id)
                try:
                    while True:
                        event = await queue.get()
                        if filter_events(event):
                            console.print(format_event(event))
                finally:
                    sub.unsubscribe()
        except NoCoordinatorError as exc:
            typer.echo(f"error: {exc}", err=True)
            return 2
        except OSError as exc:
            typer.echo(f"error: could not reach coordinator: {exc}", err=True)
            return 2

    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


def notify(thread_id: ThreadId, text: str) -> int:
    """Implementation of ``ai-functions notify <thread-id> <text>``.

    Calls :meth:`Coordinator.notify` exactly once. Does NOT
    start a cycle — the message is delivered side-channel. To run a
    chat-shaped thread with a new prompt and wait for its result, use
    ``ai-functions submit`` instead.

    Args:
        thread_id: Target thread.
        text: Message body.

    Returns:
        ``0`` on success, ``1`` if the thread was not found.
    """

    async def _run() -> int:
        try:
            async with connect() as client:
                try:
                    await client.notify(thread_id, text)
                except ThreadNotFoundError:
                    typer.echo(f"error: thread '{thread_id}' not found", err=True)
                    return 1
        except NoCoordinatorError as exc:
            typer.echo(f"error: {exc}", err=True)
            return 2
        except OSError as exc:
            typer.echo(f"error: could not reach coordinator: {exc}", err=True)
            return 2
        return 0

    return asyncio.run(_run())


def submit(thread_id: ThreadId, text: str, *, as_json: bool = False) -> int:
    """Implementation of ``ai-functions submit <thread-id> <text>``.

    Calls :meth:`Coordinator.submit` with ``text`` as the single
    positional argument, starting one cycle, and blocks until the cycle
    resolves. By default the typed result is printed to stdout (strings
    verbatim, other types via ``repr``).

    With ``as_json``, a single JSON object is printed instead, carrying
    the result alongside cycle metadata: the resolved ``status``, the
    summed ``token_usage`` across the cycle's ``TokenUsageEvent`` events,
    and ``timing`` derived from event timestamps. The token events are read
    back via :meth:`Coordinator.get_events` after the cycle resolves —
    both backends emit token usage synchronously inside ``execute`` (so
    it is already persisted by the time ``submit`` resolves), and the
    cycle's events are scoped with a ``since_id`` watermark captured
    before submitting.

    Only ``STR_PROMPT`` threads accept a single freeform string from the
    command line. For ``STRUCTURED`` / ``NO_ARGS`` threads the command
    refuses to guess at the arguments and exits with a clear error —
    those threads must be driven from a client script via
    :func:`ai_functions.connect`. This holds with ``as_json`` too: the flag
    enriches output, not input.

    Ctrl-C while the cycle is in flight cancels it via
    :meth:`Coordinator.cancel` and exits ``130``.

    Args:
        thread_id: Target thread.
        text: Prompt forwarded as the cycle's single positional
            argument.
        as_json: Emit a JSON object (result + token usage + timing)
            instead of the bare result.

    Returns:
        ``0`` on a completed cycle, ``1`` if the thread was not found or
        does not accept a string prompt, ``2`` if no coordinator was
        discovered, ``130`` if interrupted.
    """

    async def _run() -> int:
        try:
            async with connect() as client:
                try:
                    info = await client.get_thread_info(thread_id)
                except ThreadNotFoundError:
                    typer.echo(f"error: thread '{thread_id}' not found", err=True)
                    return 1
                if info.input_shape is not InputShape.STR_PROMPT:
                    typer.echo(
                        f"error: thread '{thread_id}' has input shape "
                        f"'{info.input_shape.value}'; 'ai-functions submit' only supports "
                        "'str_prompt' threads — drive this one from a client script",
                        err=True,
                    )
                    return 1
                # Watermark the log before submitting so the post-cycle
                # replay can be scoped to this cycle's events. get_events
                # is oldest-first, so the last event with an id is the
                # newest currently stored.
                watermark = _latest_event_id(await client.get_events(thread_id))
                try:
                    result = await client.submit(thread_id, text)
                except ThreadNotFoundError:
                    typer.echo(f"error: thread '{thread_id}' not found", err=True)
                    return 1
                except asyncio.CancelledError:
                    await client.cancel(thread_id)
                    raise
                except OSError as exc:
                    typer.echo(f"error: could not reach coordinator: {exc}", err=True)
                    return 2
                except Exception as exc:  # noqa: BLE001
                    # The cycle itself raised (the thread's ``execute``
                    # failed); the rehydrated exception — or a
                    # ``RemoteError`` — propagates through the awaitable.
                    # Report it cleanly rather than dumping a traceback.
                    typer.echo(f"error: cycle failed: {exc}", err=True)
                    return 1
                if not as_json:
                    typer.echo(result if isinstance(result, str) else repr(result))
                    return 0
                events = await client.get_events(thread_id, since_id=watermark)
                payload = _build_submit_json(info, result, events)
                typer.echo(json.dumps(payload, indent=2))
        except NoCoordinatorError as exc:
            typer.echo(f"error: {exc}", err=True)
            return 2
        except OSError as exc:
            typer.echo(f"error: could not reach coordinator: {exc}", err=True)
            return 2
        return 0

    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 130


def _latest_event_id(events: list[Event]) -> EventId | None:
    """Return the id of the newest event, or ``None`` if there is none.

    ``get_events`` is oldest-first, so the last event with an ``id``
    attribute is the watermark. ``CustomEvent`` is not a ``BaseEvent``
    and carries no ``id``, so it is skipped.
    """
    for event in reversed(events):
        event_id = getattr(event, "id", None)
        if event_id is not None:
            return cast("EventId", event_id)
    return None


def _jsonify_result(result: object) -> object:
    """Coerce a cycle result into a JSON-safe value.

    Strings, numbers, booleans, ``None``, and JSON-safe containers pass
    through; pydantic models are dumped in JSON mode; anything else is
    rendered via ``repr`` so the output stays valid and lossless-ish.
    """
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump(mode="json")
    try:
        json.dumps(result)
    except (TypeError, ValueError):
        return repr(result)
    return result


def _build_submit_json(info: ThreadInfo, result: object, events: list[Event]) -> dict[str, object]:
    """Fold a completed cycle's result and events into the JSON shape.

    Token usage is summed across every ``TokenUsageEvent`` in ``events``;
    timing is derived from the earliest and latest event timestamps. The
    cycle is known to have completed (``submit`` resolved normally), so
    ``status`` is ``"completed"``.

    Args:
        info: The target thread's static snapshot.
        result: The value ``submit`` resolved with.
        events: This cycle's events (scoped via the ``since_id``
            watermark), oldest-first.

    Returns:
        A JSON-serialisable dict with ``thread_id``, ``thread_name``,
        ``status``, ``result``, ``token_usage``, and ``timing``.
    """
    usage = TokenUsage()
    for event in events:
        if isinstance(event, TokenUsageEvent):
            usage = usage + event.token_usage

    timestamps = [ts for ts in (getattr(e, "timestamp", None) for e in events) if ts is not None]
    timing: dict[str, float] | None = None
    if timestamps:
        started_at, completed_at = min(timestamps), max(timestamps)
        timing = {
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": completed_at - started_at,
        }

    return {
        "thread_id": str(info.thread_id),
        "thread_name": info.thread_name,
        "status": "completed",
        "result": _jsonify_result(result),
        "token_usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "total_tokens": (
                usage.input_tokens + usage.output_tokens + usage.cache_read_tokens + usage.cache_write_tokens
            ),
        },
        "timing": timing,
    }


def kill(thread_id: ThreadId, *, now: bool = False) -> int:
    """Implementation of ``ai-functions kill <thread-id>``.

    Calls :meth:`Coordinator.terminate` by default, or
    :meth:`Coordinator.terminate_now` when ``--now`` is set.

    Args:
        thread_id: Thread to stop.
        now: Use ``terminate_now`` (hard stop) instead of
            ``terminate`` (graceful).

    Returns:
        ``0`` on success, ``1`` if the thread was not found.
    """

    async def _run() -> int:
        try:
            async with connect() as client:
                try:
                    if now:
                        await client.terminate_now(thread_id)
                    else:
                        await client.terminate(thread_id)
                except ThreadNotFoundError:
                    typer.echo(f"error: thread '{thread_id}' not found", err=True)
                    return 1
        except NoCoordinatorError as exc:
            typer.echo(f"error: {exc}", err=True)
            return 2
        except OSError as exc:
            typer.echo(f"error: could not reach coordinator: {exc}", err=True)
            return 2
        return 0

    return asyncio.run(_run())


def run_cmd(script: Path, *, attr: str = "main") -> int:
    """Implementation of ``ai-functions run <script.py>``.

    Loads ``script`` as a standalone module (via ``importlib`` with a
    synthetic module name derived from the path), looks up the
    attribute named ``attr`` (default ``"main"``), and hands it to
    :func:`ai_functions.serve`. The attribute must be a
    :class:`~ai_functions.protocols.Spawnable` — typically a function
    decorated with :func:`~ai_functions.ai_function`, but any object
    satisfying the protocol works.

    The resulting thread is hosted by a :class:`LocalWorker` in this
    process and stays alive until the user presses Ctrl-C or the
    thread reaches a terminal status — the full
    :func:`ai_functions.serve` contract. No initial cycle is started; peers
    must drive the agent via ``ai-functions submit`` / ``ai-functions attach`` / a
    client script using :func:`ai_functions.connect`.

    Scripts that want to run their own ``asyncio.run`` with an
    initial cycle should skip this subcommand and call
    :func:`ai_functions.serve` (or build the ``LocalWorker`` dance
    themselves with :func:`ai_functions.connect`) in their ``__main__``
    block.

    Args:
        script: Path to the user's ``.py`` file.
        attr: Module attribute to treat as the spawnable; defaults to
            ``"main"``.

    Returns:
        ``0`` on clean shutdown; ``1`` if the script does not expose
        the named attribute, the attribute is not a spawnable, or the
        spawnable raised; ``2`` if no coordinator was discovered;
        ``130`` if interrupted by Ctrl-C.
    """
    if not script.is_file():
        typer.echo(f"error: script not found: {script}", err=True)
        return 1

    module_name = f"_ai_functions_user_script_{script.stem}"
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        typer.echo(f"error: could not load script: {script}", err=True)
        return 1

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"error: script raised on import: {exc}", err=True)
        return 1

    target = getattr(module, attr, None)
    if target is None:
        typer.echo(f"error: script has no attribute '{attr}'", err=True)
        return 1
    if not isinstance(target, Spawnable):
        typer.echo(f"error: attribute '{attr}' is not a Spawnable", err=True)
        return 1

    def _announce(handle: ThreadHandle[..., object]) -> None:
        """Print the hosted thread's id and how to drive it, once registered."""
        tid = handle.id
        typer.echo(f"hosting '{attr}' as {tid}")
        typer.echo(f'  submit:  ai-functions submit {tid} "<prompt>"')
        typer.echo(f"  attach:  ai-functions attach {tid}")
        typer.echo("  (Ctrl-C to stop)")

    try:
        _ = asyncio.run(aserve(cast("Spawnable[..., object]", target), thread_name=attr, on_ready=_announce))
    except NoCoordinatorError as exc:
        typer.echo(f"error: {exc}", err=True)
        return 2
    except ConnectionClosedError as exc:
        # The coordinator connection dropped while hosting. Surface it loudly
        # rather than exiting silently, and report it as a coordinator-reach
        # failure (exit 2) so a supervising harness sees a non-clean stop.
        typer.echo(f"error: lost connection to coordinator while serving '{attr}': {exc}", err=True)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"error: {exc}", err=True)
        return 1
    return 0
