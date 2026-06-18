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

import asyncio
from typing import ClassVar

import typer
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.widgets import Footer, Header, Input, RichLog, Static

from ..connect import connect
from ..discovery import NoCoordinatorError
from ..network import CoordinatorClient
from ..runtime.errors import ThreadNotFoundError
from ..types import Event, InputShape, ThreadId, ThreadInfo, ToolCallEvent, ToolResultEvent
from .events import (
    STRUCTURED_OUTPUT_TOOL,
    filter_events,
    filter_events_full,
    format_event,
    format_event_full,
)


class _AttachApp(App[None]):
    """Textual application body for ``ai-functions attach``."""

    CSS = """
    Screen { layout: vertical; }
    #log { height: 1fr; border: round $primary; }
    #banner { height: 3; padding: 1; color: $warning; }
    #input  { dock: bottom; height: 3; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "quit", "Detach"),
        Binding("alt+enter", "inject", "Inject message"),
        Binding("ctrl+o", "toggle_view", "Toggle view (all events / conversation only)"),
    ]

    def __init__(self, client: CoordinatorClient, info: ThreadInfo) -> None:
        self._client = client
        self._info = info
        self._log_widget: RichLog | None = None
        self._inflight: set[asyncio.Task[object]] = set()
        # Every replayed and live event is buffered here so a view toggle
        # can re-render the full history non-destructively. The RichLog is
        # a display sink, not a store; this list is the source of truth.
        self._events: list[Event] = []
        # ``True``  -> "conversation" (messages + tool activity, full; the
        #              default — the cleanest read of a thread).
        # ``False`` -> "all events" (one-line, truncated firehose).
        self._conversation_view: bool = True
        # ``tool_use_id``s of structured-output (``FinalAnswer``) calls.
        # Their result events carry no tool name, only this id, so the
        # conversation view correlates results back to the call to hide
        # both (the answer is already surfaced as the turn's content).
        self._structured_output_ids: set[str] = set()
        super().__init__()
        # ``title`` is a Textual reactive attribute, not something to
        # override with a property (that breaks Textual's own assignment
        # in ``App.__init__``). Set it after init instead.
        self._label = info.thread_name or "thread"
        self._refresh_title()

    def _refresh_title(self) -> None:
        """Update the header title to reflect the active view mode."""
        mode = "conversation" if self._conversation_view else "all events"
        self.title = f"ai-functions attach {self._info.thread_id} ({self._label}) — {mode}"

    def compose(self) -> ComposeResult:
        """Build the widget tree — header, log, optional input, footer."""
        yield Header()
        with Vertical():
            yield RichLog(id="log", wrap=True, highlight=False, markup=False)
            if self._info.input_shape is InputShape.STR_PROMPT:
                yield Input(placeholder="Enter to submit • Alt+Enter to inject", id="input")
            else:
                yield Static(
                    f"this thread does not accept freeform input (shape={self._info.input_shape.value}); "
                    "use 'ai-functions notify' or a client script",
                    id="banner",
                )
        yield Footer()

    async def on_mount(self) -> None:
        """Replay stored events and subscribe to new ones."""
        self._log_widget = self.query_one("#log", RichLog)
        # Both RichLog and Input are focusable and the log is mounted
        # first, so Textual auto-focuses the log and keystrokes never
        # reach the input. Move focus to the input box explicitly when
        # this thread accepts freeform prompts.
        if self._info.input_shape is InputShape.STR_PROMPT:
            self.query_one("#input", Input).focus()
        try:
            events = await self._client.get_events(self._info.thread_id)
        except ThreadNotFoundError:
            self._log_widget.write(Text("thread not found", style="red bold"))
            return
        for ev in events:
            self._events.append(ev)
            self._render_event(ev)

        loop = asyncio.get_running_loop()

        def _emit(event: Event) -> None:
            try:
                loop.call_soon_threadsafe(self._append_event, event)
            except RuntimeError:
                pass

        sub = self._client.on(_emit, thread_id=self._info.thread_id)
        self._subscription = sub

    async def on_unmount(self) -> None:
        """Tear down the event subscription and any pending work tasks."""
        sub = getattr(self, "_subscription", None)
        if sub is not None:
            sub.unsubscribe()
        for task in list(self._inflight):
            if not task.done():
                _ = task.cancel()

    def _append_event(self, event: Event) -> None:
        # Always buffer; render only if it passes the active view filter.
        self._events.append(event)
        self._render_event(event)

    def _render_event(self, event: Event) -> None:
        """Write one event to the log if it belongs in the active view.

        Each view pairs a filter with a renderer: "all events" uses
        :func:`filter_events` + the compact, truncated :func:`format_event`;
        "conversation" uses :func:`filter_events_full` + the untruncated
        :func:`format_event_full`. An event the active filter rejects is
        hidden in the current view but remains buffered for the other view.

        The structured-output (``FinalAnswer``) tool call and its result
        are additionally suppressed in the conversation view: the call is
        rejected by :func:`filter_events_full`, and the matching result —
        which carries no tool name, only a ``tool_use_id`` — is dropped
        here by correlating the id recorded when the call was seen.
        """
        if self._log_widget is None:
            return
        if isinstance(event, ToolCallEvent) and event.tool_name == STRUCTURED_OUTPUT_TOOL:
            self._structured_output_ids.add(event.tool_use_id)
        if self._conversation_view:
            if isinstance(event, ToolResultEvent) and event.tool_use_id in self._structured_output_ids:
                return
            if not filter_events_full(event):
                return
            self._log_widget.write(format_event_full(event))
        else:
            if not filter_events(event):
                return
            self._log_widget.write(format_event(event))

    def action_toggle_view(self) -> None:
        """Toggle between the "all events" and "conversation" views.

        Re-renders the full buffered history through the newly active
        view. Nothing is lost: events hidden in one view are still
        buffered and reappear when toggled back.
        """
        self._conversation_view = not self._conversation_view
        self._refresh_title()
        if self._log_widget is None:
            return
        self._log_widget.clear()
        for event in self._events:
            self._render_event(event)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter / Alt+Enter in the input box."""
        text = event.value
        if not text:
            return
        event.input.value = ""
        # ``Input.Submitted`` does not carry modifier state; the Alt+Enter
        # shortcut is bound via a key action below.
        task = asyncio.create_task(self._submit(text))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)

    async def _submit(self, text: str) -> None:
        try:
            _ = await self._client.submit(self._info.thread_id, text)
        except ThreadNotFoundError:
            if self._log_widget is not None:
                self._log_widget.write(Text("thread not found", style="red bold"))
        except Exception as exc:  # noqa: BLE001
            if self._log_widget is not None:
                self._log_widget.write(Text(f"submit error: {exc}", style="red"))

    async def _inject(self, text: str) -> None:
        try:
            await self._client.notify(self._info.thread_id, text)
        except ThreadNotFoundError:
            if self._log_widget is not None:
                self._log_widget.write(Text("thread not found", style="red bold"))
        except Exception as exc:  # noqa: BLE001
            if self._log_widget is not None:
                self._log_widget.write(Text(f"inject error: {exc}", style="red"))

    async def action_inject(self) -> None:
        """Key action for Alt+Enter: inject the current input as a message."""
        if self._info.input_shape is not InputShape.STR_PROMPT:
            return
        inp = self.query_one("#input", Input)
        text = inp.value
        if not text:
            return
        inp.value = ""
        task = asyncio.create_task(self._inject(text))
        self._inflight.add(task)
        task.add_done_callback(self._inflight.discard)


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

    async def _run() -> int:
        try:
            async with connect() as client:
                try:
                    info = await client.get_thread_info(thread_id)
                except ThreadNotFoundError:
                    typer.echo(f"error: thread '{thread_id}' not found", err=True)
                    return 1
                app = _AttachApp(client, info)
                await app.run_async()
        except NoCoordinatorError as exc:
            typer.echo(f"error: {exc}", err=True)
            return 2
        return 0

    return asyncio.run(_run())
