"""``ai-functions attach <thread-id>`` — Textual TUI for one live thread.

Layout::

    ┌─ ai-functions attach thread-abc123 (researcher) ────────────────┐
    │  ▶ researcher started                                    │
    │  ▷ user: Research quantum computing                      │
    │  ◁ assistant: Quantum computing uses…                    │
    │    ⚙ tool call: web_search(query='quantum…')             │
    │    ⚙ tool result: web_search [ok]                        │
    │  Σ tokens: in=1240 out=312                               │
    ├──────────────────────────────────────────────────────────┤
    │ > █                                                      │
    └──────────────────────────────────────────────────────────┘
      [enter] submit  [alt+enter] inject  [ctrl+o] toggle view  [ctrl+c] detach

Behaviour
---------
- The top pane is a Textual ``RichLog`` fed by:
  1. an initial replay of :meth:`Coordinator.get_events`, and
  2. a live :meth:`Coordinator.on` subscription scoped to
     ``thread_id=thread_id``.
- Every replayed and live event is buffered in memory; the ``RichLog``
  is a display of that buffer, not the store. ``Ctrl+O`` toggles between
  two views and re-renders the full buffer through the active one, so no
  event is lost when switching:
  - **conversation** (default) — only user / assistant / tool-call /
    tool-result events, rendered through
    :func:`ai_functions.cli.events.format_event_full`: user and assistant
    turns in full, tool activity abbreviated.
  - **all events** — every event, one line each, rendered through
    :func:`ai_functions.cli.events.format_event` (long content is truncated to
    a preview).
- The bottom pane is conditional on
  :attr:`ThreadInfo.input_shape`:
  - ``STR_PROMPT`` — show a multi-line input widget.
    ``Enter`` calls :meth:`Coordinator.submit` (starts a new cycle);
    ``Alt+Enter`` calls :meth:`Coordinator.notify` (side-
    channel, no new cycle).
  - ``STRUCTURED`` / ``NO_ARGS`` — show a banner explaining that this
    thread doesn't accept freeform input; the log remains visible and
    interactive (scroll / copy).
- ``Ctrl+C`` detaches (closes the TUI + the client, leaves the thread
  running). No command currently *terminates* the thread from within
  the TUI — users run ``ai-functions kill`` explicitly. This is deliberate:
  ``attach`` is read-mostly and the destructive action is a separate
  verb.

Implementation module lives at ``src/ai_functions/cli/attach.py`` and depends
on ``textual``; the Typer command body in
:mod:`ai_functions.cli.commands` imports it lazily so ``ai-functions ps`` / ``ai_functions
logs`` do not pay the Textual import cost.
"""

from __future__ import annotations

from ..types import ThreadId


def attach(thread_id: ThreadId) -> int:
    """Implementation of ``ai-functions attach <thread-id>``.

    Discovers the coordinator, opens a
    :class:`~ai_functions.network.CoordinatorClient`, fetches the target's
    :class:`ThreadInfo` (to pick the bottom-pane variant), and runs
    the Textual app until the user detaches.

    Args:
        thread_id: Thread to attach to.

    Returns:
        ``0`` on clean detach, ``1`` if the thread was not found,
        ``2`` if no coordinator was discovered.

    Ensures:
        - The subscription is unsubscribed and the client is closed
          before this function returns, regardless of how the TUI was
          exited (normal quit, exception, Ctrl+C).
        - No events are printed to stdout — the TUI owns the terminal
          for the lifetime of the call.
    """
    ...
