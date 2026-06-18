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

from pathlib import Path

from ..types import ThreadId


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
    ...


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
    ...


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
    ...


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
    ...


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
    ...


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
    ...
