"""View recorded workflow graphs and install their companion skill.

The ``view`` command opens a read-only Textual app over a folder of recordings
(``.aif/`` by default) with three panes: runs and their threads on the left, the
selected run's graph in the middle, and the selected node's prompt, condition
tickets and tool calls on the right. The app rescans the folder on an interval
and rebuilds only when the scan result changes. The ``skills`` sub-command is
defined in :mod:`skills`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, ClassVar

import typer
from rich.console import Group, RenderableType
from rich.markdown import Markdown
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, ScrollableContainer, Vertical, VerticalScroll
from textual.geometry import Size
from textual.widget import Widget
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane

from layout import GAP, NODE_WIDTH, START, components, connectors, layout
from recordings import Board, StepSpec, default_root
from skills import skills_app

DEFAULT_INTERVAL = 1.5
"""Seconds between folder scans. Cards appear within one interval of a run's first event."""

app = typer.Typer(help="Experimental workflow graphs.", no_args_is_help=True)
app.add_typer(skills_app, name="skills")

_GLYPH: dict[str, tuple[str, str]] = {
    "passed": ("✓", "green"),
    "failed": ("✗", "red"),
    "running": ("●", "yellow"),
    "blocked": ("○", "red dim"),
    "skipped": ("○", "dim"),
    "cancelled": ("○", "dim"),
    "error": ("✗", "red"),
    "pending": ("○", "dim"),
    "unrun": ("○", "dim"),
}
"""The mark and colour shown for each run or thread status."""


def _glyph(status: str) -> Text:
    """Return a one-character status mark, coloured by the fixed status palette.

    Args:
        status: A run or thread status; unknown values render as a dim circle.

    Returns:
        The styled mark.
    """
    mark, style = _GLYPH.get(status, ("○", "dim"))
    return Text(mark, style=style)


# ── Widgets ───────────────────────────────────────────────────────────────────


class Row(Static, can_focus=True):
    """One focusable line inside a card. Focus is selection; arrows move within the pane.

    Every selectable item in the app is a ``Row`` subclass. Each carries a stable
    ``key`` used to restore focus after a rebuild, and the ``payload`` dict that the
    app reads when the row gains focus.
    """

    DEFAULT_CSS = """
    Row { height: auto; padding: 0 1; }
    Row:focus { background: $primary 35%; }
    Row.-selected { text-style: bold; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "row(-1)", "Up", show=False),
        Binding("down", "row(1)", "Down", show=False),
        Binding("left", "row(-1)", "Left", show=False),
        Binding("right", "row(1)", "Right", show=False),
    ]

    def __init__(self, label: Text, *, key: str, payload: dict[str, Any]) -> None:
        """Create a row.

        Args:
            label: What the row displays.
            key: Stable identity, unique within the app, used to restore focus.
            payload: The data the app acts on when the row is focused.
        """
        self.key = key
        self.payload = payload
        self.shown: RenderableType = label
        super().__init__(label)

    def show(self, renderable: RenderableType) -> None:
        """Replace the row's content, remembering it so it can be read back.

        Args:
            renderable: The new content.
        """
        self.shown = renderable
        self.update(renderable)

    def action_row(self, delta: int) -> None:
        """Move focus to the neighbouring row in the same pane, if there is one.

        Bound to the arrow keys. Movement stays within the nearest ``pane-list``
        ancestor and stops at either end rather than wrapping.

        Args:
            delta: ``-1`` for the previous row, ``1`` for the next.
        """
        pane = next((a for a in self.ancestors if a.has_class("pane-list")), None)
        if pane is None:
            return
        # Rows in document order across every card in the pane.
        rows = list(pane.query(Row))
        index = rows.index(self) + delta
        if 0 <= index < len(rows):
            rows[index].focus()
            rows[index].scroll_visible()


class ThreadRow(Row):
    """A thread of a run, in the left pane."""


class TicketRow(Row):
    """A condition ticket of a thread, in the middle pane."""


class PromptRow(Row):
    """The prompt a thread was given, boxed at the top of the AI-Thread tab; focusing it opens the full text below."""

    DEFAULT_CSS = """
    PromptRow { border: round $panel-lighten-2; padding: 0 1; margin: 0 0 1 0; text-style: italic; }
    PromptRow:focus { border: round $primary; background: $primary 20%; }
    """

    def __init__(self, label: Text, *, key: str, payload: dict[str, Any]) -> None:
        """Create the prompt box.

        Args:
            label: The prompt's short note.
            key: Stable identity for focus restoration.
            payload: The thread card, whose ``prompt`` is shown in full on focus.
        """
        super().__init__(label, key=key, payload=payload)
        self.border_title = "prompt"


class TaskRow(Row):
    """The run's task, boxed at the top of the graph; focusing it opens the task and objective in full."""

    DEFAULT_CSS = """
    TaskRow { border: round $primary; padding: 0 1; margin: 0 0 1 0; width: 1fr; height: 3; }
    TaskRow:focus { background: $primary 20%; }
    """

    def __init__(self, label: Text, *, key: str, payload: dict[str, Any]) -> None:
        """Create the task box.

        Args:
            label: The task's one-line note.
            key: Stable identity for focus restoration.
            payload: ``slug``, ``task`` and ``objective``, shown in full on focus.
        """
        super().__init__(label, key=key, payload=payload)
        self.border_title = "task"


class ToolRow(Row):
    """A tool call in the right pane; focusing it expands its arguments and output in place."""

    DEFAULT_CSS = """
    ToolRow.-expanded { background: $primary 15%; }
    """

    expanded = False

    def expand(self, expanded: bool) -> None:
        """Show or hide the call's detail beneath its one-line summary.

        Args:
            expanded: Whether the detail should be visible.
        """
        if expanded == self.expanded:
            return
        self.expanded = expanded
        self.set_class(expanded, "-expanded")
        line = _tool_line(self.payload)
        self.show(Group(line, _tool_detail(self.payload)) if expanded else line)


class NodeCard(Row):
    """One thread as a node of the graph: a fixed-width bordered box, border coloured by status."""

    DEFAULT_CSS = f"""
    NodeCard {{ width: {NODE_WIDTH}; height: auto; margin: 0 {GAP} 0 0; border: round $panel-lighten-2; padding: 0 1; }}
    NodeCard.-passed {{ border: round $success; }}
    NodeCard.-failed, NodeCard.-blocked, NodeCard.-error {{ border: round $error; }}
    NodeCard.-running {{ border: round $warning; }}
    NodeCard:focus {{ background: $primary 35%; }}
    NodeCard.-selected {{ border: double $primary; }}
    """

    def __init__(self, label: Text, *, key: str, payload: dict[str, Any]) -> None:
        """Create a node.

        Args:
            label: The node's lines, from :func:`_node_label`.
            key: The thread id, which is also the selection key.
            payload: The thread card; its ``status`` picks the border colour.
        """
        super().__init__(label, key=key, payload=payload)
        self.add_class(f"-{payload['status']}")


class PassThrough(Static):
    """A slot an edge passes through on its way to a lower row; draws as a vertical line."""

    DEFAULT_CSS = f"""
    PassThrough {{ width: {NODE_WIDTH}; height: auto; margin: 0 {GAP} 0 0; color: $text-muted; }}
    """

    def __init__(self) -> None:
        """Create a pass-through, initially one line tall."""
        super().__init__(Text(" " * (NODE_WIDTH // 2) + "│"))

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        """Match sibling nodes' natural height without depending on the viewport height.

        Args:
            container: Size of the containing row.
            viewport: Size of the viewport.
            width: Width available to this widget.

        Returns:
            The tallest sibling node's outer height, or 1 in a row without nodes.
        """
        nodes = (child for child in self.parent.children if isinstance(child, NodeCard)) if self.parent else ()
        return max(
            (
                node.get_content_height(container, viewport, width - node.styles.gutter.width)
                + node.styles.gutter.height
                for node in nodes
            ),
            default=1,
        )

    def on_resize(self, event: events.Resize) -> None:
        """Draw the edge through every line of its node row.

        Args:
            event: The resize, carrying the new height to fill.
        """
        self.update(Text("\n".join([" " * (NODE_WIDTH // 2) + "│"] * event.size.height)))


class Terminal(Static):
    """The start or end marker of the graph: a dot at the slot's centre, labelled."""

    DEFAULT_CSS = f"""
    Terminal {{ width: {NODE_WIDTH}; height: 1; margin: 0 {GAP} 0 0; color: $text-muted; }}
    """

    def __init__(self, name: str) -> None:
        """Create a marker.

        Args:
            name: ``START`` for the start marker; anything else draws the end marker.
        """
        label = "start" if name == START else "end"
        super().__init__(Text.assemble(" " * (NODE_WIDTH // 2), ("●", "bold"), f" {label}"))


class GraphRow(Horizontal):
    """One wave of the graph: its node slots side by side, as wide as they need to be."""

    DEFAULT_CSS = """
    GraphRow { width: auto; height: auto; }
    """


class Connector(Static):
    """The box-drawing lines between two rows of the graph, as wide as the wider row."""

    DEFAULT_CSS = """
    Connector { width: auto; height: auto; color: $text-muted; }
    """


class GraphComponent(Vertical):
    """One independent graph with its own start, end, and naturally sized rows.

    Built from :func:`layout.layout`: each layout row becomes a :class:`GraphRow`
    of node cards, pass-throughs and terminal markers, with a :class:`Connector`
    between consecutive rows.
    """

    DEFAULT_CSS = """
    GraphComponent { width: auto; height: auto; }
    """

    def __init__(self, steps: list[StepSpec], cards: dict[str, dict[str, Any]]) -> None:
        """Lay out one connected component and build its widgets.

        Args:
            steps: The component's steps, as the layout needs them.
            cards: Thread cards by step name, for the node labels and payloads.
        """
        graph = layout(steps, terminals=True)
        widgets: list[Widget] = []
        for index, row in enumerate(graph.rows):
            # Each slot becomes a marker, a pass-through line, or a node card.
            slots: list[Widget] = [
                Terminal(slot.name)
                if slot.terminal
                else PassThrough()
                if slot.passthrough
                else NodeCard(
                    _node_label(cards[slot.name]), key=cards[slot.name]["thread_id"], payload=cards[slot.name]
                )
                for slot in row
            ]
            widgets.append(GraphRow(*slots))
            # Lines down to the next row; the last row has none below it.
            if index < len(graph.rows) - 1:
                widgets.append(Connector(Text("\n".join(connectors(graph, index)))))
        super().__init__(*widgets)


class GraphComponents(Horizontal):
    """Independent graph columns, each wide enough to contain all its connectors."""

    DEFAULT_CSS = """
    GraphComponents { width: auto; height: auto; }
    """


class Card(Vertical):
    """A bordered block: one run on the left, one attempt on the right."""

    DEFAULT_CSS = """
    Card { height: auto; border: round $panel-lighten-2; margin: 0 0 1 0; }
    Card.-passed { border: round $success; }
    Card.-failed, Card.-blocked, Card.-error { border: round $error; }
    Card.-running { border: round $warning; }
    Card > Static.note { color: $text-muted; padding: 0 1; }
    """

    def __init__(self, title: Text, *children: Widget, status: str, subtitle: str = "") -> None:
        """Create a card.

        Args:
            title: Shown in the top border.
            *children: The card's rows.
            status: Picks the border colour through a ``-<status>`` CSS class.
            subtitle: Shown in the bottom border; omitted when empty.
        """
        super().__init__(*children, classes=f"-{status}")
        self.border_title = title
        if subtitle:
            self.border_subtitle = subtitle


class DetailPanel(Static):
    """A panel under a list that remembers what it shows, so it can be read back."""

    shown: RenderableType | str = ""

    def show(self, renderable: RenderableType | str) -> None:
        """Replace the panel's content and scroll its container back to the top.

        Args:
            renderable: The new content; an empty string clears the panel.
        """
        self.shown = renderable
        self.update(renderable)
        if isinstance(self.parent, VerticalScroll):
            self.parent.scroll_home(animate=False)


class Splitter(Static):
    """A one-cell drag handle that resizes ``target`` as the mouse moves.

    Vertical by default, resizing ``target``'s width; ``horizontal`` resizes its
    height instead. ``inverse`` is for a handle on the target's leading edge, where
    dragging towards the target should grow it.
    """

    DEFAULT_CSS = """
    Splitter { width: 1; height: 1fr; background: $panel-lighten-2; color: $text-muted; }
    Splitter:hover, Splitter.-dragging { background: $accent; }
    Splitter.-horizontal { width: 1fr; height: 1; }
    """

    def __init__(
        self, target: str, *, horizontal: bool = False, inverse: bool = False, minimum: int = 8, classes: str = ""
    ) -> None:
        """Create a handle.

        Args:
            target: CSS selector of the widget to resize.
            horizontal: Resize height by vertical drags instead of width.
            inverse: Grow the target when dragging left or up.
            minimum: Smallest size, in cells, the target may be dragged to.
            classes: Extra CSS classes, e.g. ``-hidden`` for a handle shown later.
        """
        super().__init__("", classes=f"{'-horizontal' if horizontal else ''} {classes}".strip())
        self.target = target
        self.horizontal = horizontal
        self.inverse = inverse
        self.minimum = minimum
        self._dragging = False

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Start following the mouse.

        Captures the mouse so the drag continues when the pointer leaves the handle.
        """
        del event
        self._dragging = True
        self.add_class("-dragging")
        self.capture_mouse()

    def on_mouse_up(self, event: events.MouseUp) -> None:
        """Stop following the mouse and release the capture."""
        del event
        self._dragging = False
        self.remove_class("-dragging")
        self.release_mouse()

    def on_mouse_move(self, event: events.MouseMove) -> None:
        """Resize the target by the distance moved, while a drag is in progress.

        Args:
            event: The move, whose ``delta_x`` or ``delta_y`` is the distance.
        """
        if self._dragging:
            self.drag(event.delta_y if self.horizontal else event.delta_x)

    def drag(self, delta: int) -> int:
        """Resize the target by ``delta`` cells; return the new size in cells.

        Args:
            delta: Cells moved; positive is right or down.

        Returns:
            The target's new width or height, never below ``minimum``.
        """
        target = self.screen.query_one(self.target)
        # Styles size the outer box (border and padding included), so measure that.
        current = target.outer_size.height if self.horizontal else target.outer_size.width
        size = max(self.minimum, current + (-delta if self.inverse else delta))
        if self.horizontal:
            target.styles.height = size
        else:
            target.styles.width = size
        return size


# ── The app ───────────────────────────────────────────────────────────────────


class WorkflowsApp(App[None]):
    """Textual application body for ``uv run python graphs.py view``.

    State is a snapshot of :meth:`recordings.Board.workflows` plus the selected run and
    thread. Each scan fingerprints the snapshot and rebuilds the panes only when it
    changes, so an idle poll never steals focus. Focus is selection: focusing a
    thread row or a node selects it, and the graph and node panes follow.
    """

    CSS = """
    Screen { layout: vertical; }
    #panes { height: 1fr; }
    #workflows { width: 34; min-width: 24; border: round $primary; padding: 0 1; }
    #middle { width: 1fr; min-width: 30; }
    #graph { height: 1fr; border: round $primary; padding: 0 1; overflow-x: auto; }
    #graph_detail { height: 40%; min-height: 5; border: round $primary; padding: 0 1; }
    .-hidden { display: none; }
    #right { width: 1fr; min-width: 30; border: round $primary; padding: 0 1; }
    #node_tabs { height: 1fr; }
    #conditions, #tools { height: 1fr; padding: 0 1; }
    #detail { height: 40%; min-height: 5; border: round $secondary; padding: 0 1; }
    .empty { color: $text-muted; padding: 1; }
    Footer FooterKey .footer-key--key { text-style: none; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Rescan"),
        Binding("tab", "focus_next", "Next pane", show=True),
    ]

    def __init__(self, root: Path | None = None, *, interval: float = DEFAULT_INTERVAL) -> None:
        """Create the app over a recordings folder.

        Args:
            root: Folder holding ``<slug>/runs/<run-id>/``; defaults to
                :func:`recordings.default_root`.
            interval: Seconds between folder scans.
        """
        self._board = Board(root=root or default_root())
        self._interval = interval
        # The last scan result, kept so a poll that changes nothing does not
        # rebuild the panes and lose the user's focus.
        self._snapshot: list[dict[str, Any]] = []
        self._fingerprint: str = ""
        self._selected_run: str | None = None
        self._selected_thread: str | None = None
        self._graph_for: str | None = None
        super().__init__()
        self.title = f"Graphs — {self._board.root}"

    @property
    def root(self) -> Path:
        """The folder being scanned."""
        return self._board.root

    def compose(self) -> ComposeResult:
        """Build the widget tree — header, runs | graph | node panes, footer.

        Panes start empty; :meth:`refresh_board` fills them once mounted. The task
        window under the graph starts hidden and opens when the task row is focused.
        """
        yield Header()
        with Horizontal(id="panes"):
            # Left: one card per run.
            workflows = VerticalScroll(id="workflows", classes="pane-list")
            workflows.border_title = "WORKFLOWS"
            yield workflows
            yield Splitter("#workflows")
            # Middle: the graph, with the task window beneath it.
            with Vertical(id="middle"):
                graph = ScrollableContainer(id="graph", classes="pane-list")
                graph.border_title = "GRAPH"
                yield graph
                yield Splitter("#graph_detail", horizontal=True, inverse=True, minimum=5, classes="-hidden")
                task = VerticalScroll(DetailPanel(id="graph_detail_body"), id="graph_detail", classes="-hidden")
                task.border_title = "TASK"
                yield task
            yield Splitter("#right", inverse=True, minimum=30)
            # Right: the selected node's tabs, with the detail panel beneath them.
            with Vertical(id="right"):
                with TabbedContent(id="node_tabs"):
                    with TabPane("AI-Thread", id="tab-conditions"):
                        yield VerticalScroll(id="conditions", classes="pane-list")
                    with TabPane("Tool calls", id="tab-tools"):
                        yield VerticalScroll(id="tools", classes="pane-list")
                yield Splitter("#detail", horizontal=True, inverse=True, minimum=5)
                detail = VerticalScroll(DetailPanel(id="detail_body"), id="detail")
                detail.border_title = "DETAIL"
                yield detail
        yield Footer()

    async def on_mount(self) -> None:
        """Scan once, then keep scanning on the interval."""
        await self.refresh_board()
        self.set_interval(self._interval, self.refresh_board)

    async def action_refresh(self) -> None:
        """Rescan the folder now."""
        await self.refresh_board()

    # ── Scanning ─────────────────────────────────────────────────────────────

    async def refresh_board(self) -> None:
        """Rescan the folder; re-render only when something changed.

        Called on mount, on every interval tick, and by ``r``. A change invalidates
        the drawn graph so the selected run's graph is redrawn too.
        """
        snapshot = self._board.workflows()
        # Compare by content so an unchanged folder costs no rebuild and keeps focus.
        fingerprint = json.dumps(snapshot, sort_keys=True, default=str)
        if fingerprint == self._fingerprint:
            return
        self._snapshot, self._fingerprint = snapshot, fingerprint
        self._graph_for = None
        await self._rebuild_workflows()
        await self._render_selection()

    async def _rebuild_workflows(self) -> None:
        """Rebuild the left pane from the snapshot, keeping the current selection when it still exists.

        A selected run that has disappeared falls back to the first run in the
        snapshot; a selected thread that has disappeared is cleared. Keyboard focus
        returns to the same thread row when that row still exists.
        """
        pane = self.query_one("#workflows", VerticalScroll)
        # Remember focus by key: the row widgets are about to be replaced.
        focused_key = self.focused.key if isinstance(self.focused, ThreadRow) else None
        await pane.remove_children()
        if not self._snapshot:
            await pane.mount(Static(f"no runs yet under {self.root}", classes="empty"))
            self._selected_run = self._selected_thread = None
            return
        await pane.mount(*(_workflow_card(group) for group in self._snapshot))

        # Repair a selection whose run or thread is no longer on disk.
        known = {row.key for row in pane.query(ThreadRow)}
        if self._selected_run not in {g["run"]["run_id"] for g in self._snapshot}:
            self._selected_run, self._selected_thread = self._snapshot[0]["run"]["run_id"], None
        elif self._selected_thread is not None and f"{self._selected_run}|{self._selected_thread}" not in known:
            self._selected_thread = None
        self._mark_selected()
        if focused_key in known:
            next(row for row in pane.query(ThreadRow) if row.key == focused_key).focus()

    def _mark_selected(self) -> None:
        """Mark the rows and node standing for the current selection.

        Selection is drawn with the ``-selected`` class, separately from focus, so it
        stays visible while focus is in another pane.
        """
        for row in self.query(ThreadRow):
            selected = row.payload["run_id"] == self._selected_run and row.payload["thread_id"] == self._selected_thread
            row.set_class(selected, "-selected")
        for node in self.query(NodeCard):
            node.set_class(node.payload["thread_id"] == self._selected_thread, "-selected")

    # ── Selection ────────────────────────────────────────────────────────────

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        """Focusing a row is selecting it; the other panes follow.

        Thread rows and nodes change the selection and redraw the graph and node
        panes; tickets, prompts and tool calls fill the detail panel; the task row
        opens the task window.

        Args:
            event: The focus change, naming the newly focused widget.
        """
        widget = event.widget
        # A thread: select it, then redraw in a worker so focus handling stays quick.
        if isinstance(widget, ThreadRow | NodeCard):
            self._selected_run = widget.payload["run_id"]
            self._selected_thread = widget.payload["thread_id"]
            self._mark_selected()
            self._hide_task()
            self.run_worker(self._render_selection(), exclusive=True, group="render")
        elif isinstance(widget, TicketRow):
            self._show_detail(f"{widget.payload['stage'].upper()}-CONDITION", _ticket_detail(widget.payload))
        elif isinstance(widget, PromptRow):
            self._show_detail(f"PROMPT · {widget.payload['name']}", Markdown(widget.payload["prompt"]))
        elif isinstance(widget, TaskRow):
            self._show_task(widget.payload)
        elif isinstance(widget, ToolRow):
            # Only one tool call is expanded at a time.
            for row in self.query(ToolRow):
                row.expand(row is widget)
            self._show_detail(f"TOOL · {_one_line(widget.payload['tool'], 40)}", _tool_detail(widget.payload))

    def _show_task(self, payload: dict[str, Any]) -> None:
        """Open the resizable, scrollable window under the graph with the task and objective in full.

        Args:
            payload: The task row's ``slug``, ``task`` and ``objective``.
        """
        window = self.query_one("#graph_detail", VerticalScroll)
        window.border_title = f"TASK · {payload['slug']}"
        self.query_one("#graph_detail_body", DetailPanel).show(_task_detail(payload))
        # Show the window together with its drag handle.
        for widget in (window, *[s for s in self.query(Splitter) if s.target == "#graph_detail"]):
            widget.remove_class("-hidden")

    def _hide_task(self) -> None:
        """Close the task window under the graph, and its drag handle."""
        for widget in (
            self.query_one("#graph_detail"),
            *[s for s in self.query(Splitter) if s.target == "#graph_detail"],
        ):
            widget.add_class("-hidden")

    def _show_detail(self, title: str, renderable: RenderableType) -> None:
        """Put ``renderable`` in the panel under the right pane, titled ``title``.

        Args:
            title: The panel's border title.
            renderable: What to show; an empty string clears the panel.
        """
        self.query_one("#detail", VerticalScroll).border_title = title
        self.query_one("#detail_body", DetailPanel).show(renderable)

    def _group(self) -> dict[str, Any] | None:
        """The snapshot entry for the selected run, if it is still on disk.

        Returns:
            The run's group from :meth:`recordings.Board.workflows`, or ``None``.
        """
        return next((g for g in self._snapshot if g["run"]["run_id"] == self._selected_run), None)

    async def _render_selection(self) -> None:
        """Draw the selected run's graph in the middle and the selected node on the right.

        The graph is redrawn only when the selected run differs from the one drawn,
        so selecting another node of the same run doesn't rebuild it.
        """
        group = self._group()
        self.sub_title = group["run"]["slug"] if group else ""
        if group is None or self._graph_for != self._selected_run:
            await self._render_graph(group)
        self._mark_selected()
        await self._render_node(group)

    async def _render_graph(self, group: dict[str, Any] | None) -> None:
        """Fill the middle pane with the run's thread graph: rows of node cards joined by connectors.

        Each connected component is drawn as its own column with its own start and
        end markers, below a task row when the run recorded a task or objective.

        Args:
            group: The selected run's snapshot entry; ``None`` empties the pane.
        """
        pane = self.query_one("#graph", ScrollableContainer)
        await pane.remove_children()
        if group is None:
            self._graph_for = None
            return
        self._hide_task()
        # Split the run's threads into independent graphs, drawn side by side.
        cards = {card["name"]: card for card in group["threads"]}
        groups = components([_spec(card) for card in group["threads"]])
        widgets: list[Widget] = []
        if group["run"]["task"] or group["objective"]:
            payload = {"slug": group["run"]["slug"], "task": group["run"]["task"], "objective": group["objective"]}
            widgets.append(TaskRow(_task_note(payload), key=f"{group['run']['run_id']}::task", payload=payload))
        if groups:
            widgets.append(GraphComponents(*(GraphComponent(steps, cards) for steps in groups)))
        if not widgets:
            widgets.append(Static("no threads yet", classes="empty"))
        await pane.mount(*widgets)
        self._graph_for = self._selected_run

    async def _render_node(self, group: dict[str, Any] | None) -> None:
        """Fill the right pane's tabs for the selected node, and clear the detail panel.

        The AI-Thread tab holds the prompt box and one row per condition ticket; the
        Tool calls tab holds the thread's tool calls, grouped by attempt.

        Args:
            group: The selected run's snapshot entry, or ``None``.
        """
        conditions = self.query_one("#conditions", VerticalScroll)
        tools = self.query_one("#tools", VerticalScroll)
        await conditions.remove_children()
        await tools.remove_children()
        self._show_detail("DETAIL", "")
        pane_right = self.query_one("#right")
        # Find the selected thread's card; without one, show placeholders.
        card = None
        if group is not None:
            card = next((t for t in group["threads"] if t["thread_id"] == self._selected_thread), None)
        if card is None:
            pane_right.border_title = "NODE"
            await conditions.mount(Static("select a node to see its prompt and conditions", classes="empty"))
            await tools.mount(Static("select a node to see its tool calls", classes="empty"))
            return
        pane_right.border_title = f"NODE · {card['name']}"
        # AI-Thread tab: the prompt, then pre- and post-condition tickets.
        rows: list[Widget] = []
        if card["prompt"]:
            rows.append(PromptRow(_prompt_note(card["prompt"]), key=f"{card['thread_id']}::prompt", payload=card))
        rows += [
            TicketRow(
                _ticket_row(ticket), key=f"{card['thread_id']}::{ticket['stage']}::{ticket['name']}", payload=ticket
            )
            for ticket in card["tickets"]
        ]
        if not card["tickets"]:
            rows.append(Static("no conditions declared", classes="note"))
        await conditions.mount(*rows)
        await tools.mount(*self._tool_cards(card))

    def _tool_cards(self, card: dict[str, Any]) -> list[Widget]:
        """One card per post-condition attempt, each holding the tool calls made during it.

        Tool calls are not part of the snapshot, so they are read from the run's
        parsed view, which the board caches until the log grows.

        Args:
            card: The selected thread's card.

        Returns:
            Attempt cards in attempt order, or a placeholder when the thread made no
            recorded tool calls.
        """
        found = self._board.view(card["run_id"])
        thread = None
        if found is not None:
            _, view = found
            thread = next((t for t in view.threads if t.thread_id == card["thread_id"]), None)
        if thread is None or not thread.tools:
            return [Static("no tool calls recorded", classes="empty")]
        # Group calls by the attempt they were made in.
        by_attempt: dict[int, list[ToolRow]] = {}
        for index, activity in enumerate(thread.tools):
            record = activity.model_dump()
            by_attempt.setdefault(record["attempt"], []).append(
                ToolRow(_tool_line(record), key=f"{card['thread_id']}::tool-{index}", payload=record)
            )
        return [
            Card(Text(f"ATTEMPT {attempt}", style="bold"), *rows, status="attempt")
            for attempt, rows in by_attempt.items()
        ]


# ── Builders ──────────────────────────────────────────────────────────────────


def _workflow_card(group: dict[str, Any]) -> Card:
    """One run as a card: status-coloured border, slug in the title, one row per thread.

    Args:
        group: One entry of :meth:`recordings.Board.workflows`.

    Returns:
        The card, titled with the run's status, slug and passed/total threads, and
        subtitled with its run id, backend and session (or a stale-runtime flag).
    """
    run = group["run"]
    threads = group["threads"]
    passed = sum(1 for t in threads if t["status"] == "passed")
    title = Text.assemble(_glyph(run["status"]), " ", (run["slug"], "bold"), " ", (f"{passed}/{len(threads)}", "dim"))
    # Subtitle parts, skipping any that are empty; an unrun package has no run id.
    subtitle = " · ".join(
        part
        for part in (
            run["run_id"] if run["status"] != "unrun" else "",
            run.get("backend") or "",
            "not run yet"
            if run["status"] == "unrun"
            else (f"stale runtime v{run['runtime_version']}" if group["stale_runtime"] else (run["session"] or "")),
        )
        if part
    )
    rows: list[Widget] = [
        ThreadRow(
            _thread_label(card),
            key=f"{run['run_id']}|{card['thread_id']}",
            payload={"run_id": run["run_id"], "thread_id": card["thread_id"]},
        )
        for card in threads
    ]
    if not threads:
        rows.append(Static("written, not run yet — steps appear on the first run", classes="note"))
    return Card(title, *rows, status=run["status"], subtitle=subtitle)


def _spec(card: dict[str, Any]) -> StepSpec:
    """What the layout needs to know about a thread: its name, dependencies and output type.

    Args:
        card: A thread card from the snapshot.

    Returns:
        A minimal step spec for :func:`layout.layout` and :func:`layout.components`.
    """
    return StepSpec(name=card["name"], depends_on=card["depends_on"], output_type=card["output_type"] or "str")


def _node_label(card: dict[str, Any]) -> Text:
    """Render a node's lines: status and name; type contract and backend; one line per condition with its tick.

    Args:
        card: The thread's card.

    Returns:
        The node's label, every line cut to fit inside a ``NODE_WIDTH`` box.
    """
    inner = NODE_WIDTH - 4  # border and padding on both sides
    # Line 1: status mark and name, with the attempt count when the step retried.
    retries = f" ×{card['attempts']}" if card["attempts"] > 1 else ""
    label = Text.assemble(_glyph(card["status"]), " ", (_one_line(card["name"], inner - 2 - len(retries)), "bold"))
    if retries:
        label.append(retries, style="yellow")
    contract = f"{card['input_type'] or 'str'} → {card['output_type'] or 'str'}"
    if card["backend"]:
        contract += f" · {card['backend']}"
    # Line 2: what goes in and out, and which backend ran it.
    label.append("\n" + _one_line(contract, inner), style="dim")
    # Then one line per condition, PRE before POST, the stage named on its first line only.
    for stage in ("pre", "post"):
        for index, ticket in enumerate(t for t in card["tickets"] if t["stage"] == stage):
            tag = stage.upper() if index == 0 else ""
            label.append(f"\n{tag:<4} ", style="dim")
            label.append_text(_tick(ticket))
            label.append(" " + _one_line(ticket["name"], inner - 7), style="dim")
    return label


def _tick(ticket: dict[str, Any]) -> Text:
    """Return a condition's verdict as one coloured character.

    Args:
        ticket: A condition ticket.

    Returns:
        ``○`` when never evaluated, otherwise ``✓`` or ``✗`` for its latest result.
    """
    if ticket["attempts"] == 0:
        return Text("○", style="dim")
    return Text("✓", style="green") if ticket["passed"] else Text("✗", style="red")


def _task_note(payload: dict[str, Any]) -> Text:
    """Render the task box's single line: the task, cut to fit, with a hint that it opens.

    Args:
        payload: The task row's payload.

    Returns:
        The one-line note.
    """
    note = Text(_one_line(payload["task"] or "(no task recorded)", 140))
    note.append("  ▸ open", style="dim")
    return note


def _task_detail(payload: dict[str, Any]) -> RenderableType:
    """Render the task and the objective in full, each under its own rule.

    Args:
        payload: The task row's payload.

    Returns:
        The task, followed by the objective when one was recorded.
    """
    parts: list[RenderableType] = [Text(payload["task"] or "(no task recorded)")]
    if payload["objective"]:
        parts += [Rule("what success means", style="dim", align="left"), Text(payload["objective"], style="italic")]
    return Group(*parts)


def _prompt_note(prompt: str, limit: int = 240) -> Text:
    """Render a step's prompt as a short italic note: the first paragraph, truncated.

    Args:
        prompt: The step's full prompt.
        limit: Maximum characters before truncation.

    Returns:
        The note; the full prompt opens in the detail panel.
    """
    first_paragraph = prompt.strip().split("\n\n", 1)[0]
    return Text(_one_line(first_paragraph, limit), style="italic")


def _thread_label(card: dict[str, Any]) -> Text:
    """Render a thread row: status, name, passed/total tickets, and the attempt count when it retried.

    Args:
        card: The thread's card.

    Returns:
        The row's label.
    """
    tickets = card["tickets"]
    ok = sum(1 for t in tickets if t["passed"])
    label = Text.assemble(_glyph(card["status"]), " ", card["name"])
    label.append(f"  {ok}/{len(tickets)}", style="dim")
    if card["attempts"] > 1:
        label.append(f"  ×{card['attempts']}", style="yellow")
    return label


def _ticket_row(ticket: dict[str, Any]) -> Text:
    """Render one condition ticket as a line: verdict, name, stage and attempt count.

    Args:
        ticket: A condition ticket.

    Returns:
        The ticket's line, tagged ``NOT RUN`` when it was never evaluated.
    """
    tag = "NOT RUN" if ticket["attempts"] == 0 else ticket["stage"].upper()
    if ticket["attempts"] > 1:
        tag = f"{tag} · {ticket['attempts']} ATTEMPTS"
    line = Text.assemble(_tick(ticket), " ", ticket["name"])
    line.append(f"  {tag}", style="dim")
    return line


def _ticket_detail(ticket: dict[str, Any]) -> RenderableType:
    """Render a ticket's detail: the condition's source as it ran, and nothing else.

    Falls back to the failure message, then to the signature or name, for
    conditions whose source could not be recorded.

    Args:
        ticket: A condition ticket.

    Returns:
        Highlighted source, or the best available fallback.
    """
    if ticket["source"]:
        return Syntax(ticket["source"], "python", theme="ansi_dark", word_wrap=True, background_color="default")
    if ticket["message"]:
        return Text(ticket["message"], style="red")
    return Text(ticket["signature"] or ticket["name"], style="dim")


def _tool_line(activity: dict[str, Any]) -> Text:
    """Render one tool call as a line: name, purpose or first argument, status and duration.

    Args:
        activity: A dumped :class:`recordings.ToolActivity`.

    Returns:
        The call's one-line summary; a non-success status is shown in red.
    """
    # Kiro's tool name can be a whole shell command; keep the line a line.
    line = Text.assemble(("▪ ", "cyan"), (_one_line(activity["tool"], 60), "bold"))
    detail = activity["purpose"] or _first_argument(activity["arguments"])
    if detail:
        line.append(f"  {_one_line(detail, 100)}", style="dim")
    status = activity["status"]
    if status not in {"success", "pending"}:
        line.append(f"  [{status}]", style="red")
    if activity["duration"] is not None:
        line.append(f"  {activity['duration']:.1f}s", style="dim")
    return line


def _tool_detail(activity: dict[str, Any]) -> RenderableType:
    """Render what sits under an expanded tool row: purpose, arguments and output.

    Args:
        activity: A dumped :class:`recordings.ToolActivity`.

    Returns:
        The parts that were recorded, each under its own rule.
    """
    parts: list[RenderableType] = []
    if activity["purpose"]:
        parts.append(Text(activity["purpose"], style="italic"))
    if activity["arguments"]:
        rendered = json.dumps(activity["arguments"], indent=2, default=str)[:2000]
        parts += [
            Rule("arguments", style="dim", align="left"),
            Syntax(rendered, "json", theme="ansi_dark", word_wrap=True, background_color="default"),
        ]
    if activity["output"]:
        parts += [Rule("output", style="dim", align="left"), Text(activity["output"][:2000])]
    if not parts:
        parts.append(Text("no arguments or output recorded", style="dim"))
    return Group(*parts)


def _one_line(text: str, limit: int) -> str:
    """Collapse whitespace and truncate, so a multi-line value fits one list row.

    Args:
        text: Any text, possibly multi-line.
        limit: Maximum length of the result, ellipsis included.

    Returns:
        The text on one line, ending in ``…`` when it was cut.
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _first_argument(arguments: dict[str, Any]) -> str:
    """Return the most telling argument of a tool call, as a short string.

    Args:
        arguments: The call's arguments.

    Returns:
        A path, command, pattern or query when present, else the first argument;
        at most 80 characters, and empty when there are no arguments.
    """
    for key in ("path", "command", "pattern", "query"):
        if key in arguments:
            return str(arguments[key])[:80]
    return next((str(v)[:80] for v in arguments.values()), "")


@app.command("view")
def workflows(
    root: Annotated[Path | None, typer.Argument(help="Recorded runs folder; defaults to $AIF_HOME or ./.aif.")] = None,
    *,
    interval: Annotated[float, typer.Option(min=0.1, help="Seconds between folder scans.")] = DEFAULT_INTERVAL,
) -> int:
    """Open the read-only terminal UI over recorded workflow runs."""
    WorkflowsApp(root, interval=interval).run()
    return 0


__all__ = [
    "Card",
    "Connector",
    "DEFAULT_INTERVAL",
    "DetailPanel",
    "GraphComponent",
    "GraphRow",
    "NodeCard",
    "PassThrough",
    "PromptRow",
    "Row",
    "Splitter",
    "TaskRow",
    "Terminal",
    "ThreadRow",
    "TicketRow",
    "ToolRow",
    "WorkflowsApp",
    "workflows",
]


if __name__ == "__main__":
    app()
