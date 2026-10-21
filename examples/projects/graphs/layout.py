"""Lay out dependency graphs and render their terminal connectors.

The viewer draws a workflow top to bottom. :func:`layout` assigns every step to a
row (its depth along the longest dependency path) and fills the rows with slots;
:func:`connectors` renders the box-drawing lines between two adjacent rows. Both
work on fixed-width slots, so a slot's column index alone determines where its
centre falls on screen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from recordings import StepSpec

NODE_WIDTH = 26
"""Columns a node slot occupies, border included. Connector maths assumes every slot is this wide."""

GAP = 2
"""Columns between neighbouring slots."""

ARROW = "▼"

START = "⟨start⟩"
"""Name of the virtual slot above the first wave when a layout is built with terminals."""

END = "⟨end⟩"
"""Name of the virtual slot below the last wave when a layout is built with terminals."""


@dataclass(frozen=True)
class Edge:
    """A dependency, drawn from the upstream node down to the one that waits for it."""

    source: str
    target: str
    label: str = "str"
    """What flows along the edge: the upstream step's output type."""


@dataclass(frozen=True)
class Slot:
    """One position in a row: a real node, or a pass-through carrying a longer edge."""

    name: str
    """The node's step name, ``"<source>→<target>"`` for a pass-through, or ``START`` / ``END``."""
    passthrough: bool = False
    edge: Edge | None = None
    """For a pass-through, the edge it carries."""

    @property
    def terminal(self) -> bool:
        """Whether this is the start or end marker rather than a node."""
        return self.name in (START, END)


@dataclass
class Layout:
    """Rows of slots plus the edges between them; the whole picture the viewer draws."""

    rows: list[list[Slot]] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def row_of(self, name: str) -> int:
        """Return the row index holding the slot ``name``.

        Args:
            name: A slot name: a step, a pass-through, or a terminal marker.

        Returns:
            The index into :attr:`rows` of the row containing the slot.

        Raises:
            KeyError: No row holds a slot with that name.
        """
        for index, row in enumerate(self.rows):
            if any(slot.name == name for slot in row):
                return index
        raise KeyError(name)

    def column_of(self, name: str) -> int:
        """Return the column index of the slot ``name`` within its row.

        Args:
            name: A slot name present in the layout.

        Returns:
            The slot's position in its row, counted from the left.

        Raises:
            KeyError: No row holds a slot with that name.
        """
        row = self.rows[self.row_of(name)]
        return next(i for i, slot in enumerate(row) if slot.name == name)

    def incoming(self, name: str) -> list[Edge]:
        """Return the edges into ``name``, in the order its dependencies were declared.

        Args:
            name: The target slot's name.

        Returns:
            Every edge whose target is ``name``; empty for a root.
        """
        return [edge for edge in self.edges if edge.target == name]


def centre(column: int) -> int:
    """Return the x coordinate of a slot's centre, given its column index.

    Args:
        column: The slot's index within its row.

    Returns:
        The character column, from the row's left edge, on which the slot is centred.
    """
    return column * (NODE_WIDTH + GAP) + NODE_WIDTH // 2


def components(steps: Sequence[StepSpec]) -> list[list[StepSpec]]:
    """Group connected steps, ignoring dependency direction and preserving declaration order.

    The viewer draws each group as its own graph with its own start and end
    markers, so independent pipelines are never joined by shared terminals.
    Dependencies on names outside ``steps`` are ignored.

    Args:
        steps: The workflow's steps, in declaration order.

    Returns:
        One list per connected component, ordered by each component's first
        declared step; steps keep their declaration order within a component.
    """
    # Treat each dependency as an undirected edge between two known steps.
    neighbours: dict[str, set[str]] = {step.name: set() for step in steps}
    for step in steps:
        for dependency in step.depends_on:
            if dependency in neighbours:
                neighbours[step.name].add(dependency)
                neighbours[dependency].add(step.name)
    # Flood-fill from each step not yet assigned; each fill is one component.
    groups: list[list[StepSpec]] = []
    component_of: dict[str, int] = {}
    for step in steps:
        if step.name in component_of:
            continue
        index = len(groups)
        groups.append([])
        pending = [step.name]
        while pending:
            name = pending.pop()
            if name not in component_of:
                component_of[name] = index
                pending.extend(neighbours[name] - component_of.keys())
    # Fill the groups in a second pass so each keeps declaration order.
    for step in steps:
        groups[component_of[step.name]].append(step)
    return groups


def layout(steps: Sequence[StepSpec], *, terminals: bool = False) -> Layout:
    """Layer ``steps`` into rows by longest path and insert pass-throughs for long edges.

    A step's row is one more than the deepest row among its dependencies, so
    every edge points downward and roots share the top row. An edge that spans
    more than one row gets a pass-through slot in each row it crosses, which is
    what lets :func:`connectors` draw only between adjacent rows. Steps caught in
    a cycle are placed together in one final row. With ``terminals``, a start
    marker feeds every root and every leaf feeds an end marker; this is only done
    for a single connected component (see :func:`components`).

    Args:
        steps: The workflow's steps, in declaration order.
        terminals: Add ``START`` and ``END`` marker rows around the graph.

    Returns:
        The rows of slots and the edges between them, terminal edges included.
    """
    terminals = terminals and len(components(steps)) == 1  # Shared markers would join independent components.
    by_name = {step.name: step for step in steps}
    depth: dict[str, int] = {}
    remaining = [step.name for step in steps]
    offset = 1 if terminals else 0
    # Place steps whose dependencies all have a depth, until none remain.
    while remaining:
        placed = [name for name in remaining if all(dep in depth for dep in by_name[name].depends_on if dep in by_name)]
        if not placed:  # a cycle: put what is left together in one final row
            last = max(depth.values(), default=offset - 1) + 1
            for name in remaining:
                depth[name] = last
            break
        for name in placed:
            deps = [depth[dep] for dep in by_name[name].depends_on if dep in by_name]
            depth[name] = max(deps, default=offset - 1) + 1
        remaining = [name for name in remaining if name not in placed]

    # Build the rows, leaving room for the end marker when terminals are on.
    last_row = max(depth.values(), default=offset - 1) + (1 if terminals else 0)
    rows: list[list[Slot]] = [[] for _ in range(last_row + 1)]
    for step in steps:  # declaration order within a row
        rows[depth[step.name]].append(Slot(step.name))

    # One edge per known dependency, labelled with what the upstream step returns.
    edges: list[Edge] = []
    for step in steps:
        for dep in step.depends_on:
            if dep in by_name:
                edges.append(Edge(dep, step.name, by_name[dep].output_type or "str"))
    # Terminals: START feeds every step with no dependency; every leaf feeds END.
    if terminals and steps:
        rows[0].append(Slot(START))
        rows[-1].append(Slot(END))
        depth[START], depth[END] = 0, last_row
        targets = {edge.target for edge in edges}
        sources = {edge.source for edge in edges}
        edges = (
            [Edge(START, s.name, s.input_type or "str") for s in steps if s.name not in targets]
            + edges
            + [Edge(s.name, END, s.output_type or "str") for s in steps if s.name not in sources]
        )
    for edge in edges:
        # A long edge gets one pass-through slot per skipped row, placed after the
        # row's nodes so it draws as a straight line where it can.
        for row in range(depth[edge.source] + 1, depth[edge.target]):
            rows[row].append(Slot(f"{edge.source}→{edge.target}", passthrough=True, edge=edge))
    return Layout(rows=rows, edges=edges)


def _segments(graph: Layout, upper_index: int) -> list[tuple[str, str, str]]:
    """Return ``(source_slot, target_slot, label)`` for every line drawn between two adjacent rows.

    An edge crossing the gap starts at its source node or at its pass-through in
    the upper row, and ends at its target node or at its pass-through in the
    lower row.

    Args:
        graph: The full layout.
        upper_index: Index of the upper of the two rows.

    Returns:
        One tuple per edge that crosses the gap below ``upper_index``.
    """
    upper = graph.rows[upper_index]
    lower = graph.rows[upper_index + 1]
    lower_names = {slot.name for slot in lower}
    segments: list[tuple[str, str, str]] = []
    for edge in graph.edges:
        source_row, target_row = graph.row_of(edge.source), graph.row_of(edge.target)
        # Skip edges entirely above or below this gap.
        if source_row > upper_index or target_row <= upper_index:
            continue
        # Where does this edge sit in the upper row, and where does it land in the lower?
        start = edge.source if source_row == upper_index else f"{edge.source}→{edge.target}"
        end = edge.target if target_row == upper_index + 1 else f"{edge.source}→{edge.target}"
        if end in lower_names and any(slot.name == start for slot in upper):
            segments.append((start, end, edge.label))
    return segments


def connectors(graph: Layout, upper_index: int) -> list[str]:
    """Render the box-drawing lines between row ``upper_index`` and the row below it.

    Each target slot in the lower row owns one horizontal *bus* spanning its own
    centre and the centres of all its sources. Sources drop from the top line onto
    their target's bus, and the target continues down to an arrow labelled with
    the incoming types. Buses whose spans do not overlap share a line, so the
    connector is only as tall as the overlaps require. A segment ending at a
    pass-through continues straight through instead of ending in an arrow.

    Args:
        graph: The full layout.
        upper_index: Index of the upper of the two rows; must not be the last row.

    Returns:
        The connector's lines, each as wide as the wider of the two rows; a single
        blank line when no edge crosses the gap.
    """
    segments = _segments(graph, upper_index)
    upper, lower = graph.rows[upper_index], graph.rows[upper_index + 1]
    width = max(len(upper), len(lower)) * (NODE_WIDTH + GAP)
    if not segments:
        return [" " * width]

    source_columns = {slot.name: i for i, slot in enumerate(upper)}
    target_columns = {slot.name: i for i, slot in enumerate(lower)}
    continuations = {slot.name for slot in lower if slot.passthrough}
    # Group by target: each target owns one horizontal bus joining its sources to it.
    targets: dict[str, list[tuple[int, str]]] = {}
    for start, end, label in segments:
        targets.setdefault(end, []).append((centre(source_columns[start]), label))
    spans = {
        end: (
            min([centre(target_columns[end]), *(x for x, _ in sources)]),
            max([centre(target_columns[end]), *(x for x, _ in sources)]),
        )
        for end, sources in targets.items()
    }
    # Pack buses onto lines: two targets share a line only when their spans are disjoint.
    # A straight edge (span of one column) draws no horizontal and reserves nothing.
    bus_of: dict[str, int] = {}
    lines_used: list[list[tuple[int, int]]] = []
    for end, (lo, hi) in sorted(spans.items(), key=lambda item: item[1]):
        if lo == hi:
            bus_of[end] = 0
            continue
        for index, taken in enumerate(lines_used):
            if all(hi < a - 1 or lo > b + 1 for a, b in taken):
                taken.append((lo, hi))
                bus_of[end] = index
                break
        else:
            lines_used.append([(lo, hi)])
            bus_of[end] = len(lines_used) - 1

    # Draw: a top line for sources, one line per bus, then the arrow line.
    height = 1 + max(1, len(lines_used)) + 1
    grid = _Grid(width, height)
    for end, sources in targets.items():
        bus = 1 + bus_of[end]
        tx = centre(target_columns[end])
        lo, hi = spans[end]
        for x in range(lo, hi + 1):  # the bus itself
            if x > lo:
                grid.add(bus, x, "L")
            if x < hi:
                grid.add(bus, x, "R")
        for sx, _ in sources:  # each source drops from the top line onto the bus
            for y in range(0, bus):
                grid.add(y, sx, "D")
                if y > 0:
                    grid.add(y, sx, "U")
            grid.add(bus, sx, "U")
        for y in range(bus, height - 1):  # the target continues down to the arrow line
            grid.add(y, tx, "D")
            if y > bus:
                grid.add(y, tx, "U")
        # A pass-through carries the line on into the next row; a node gets an arrow.
        if end in continuations:
            grid.add(height - 1, tx, "U")
            grid.add(height - 1, tx, "D")
        else:
            labels = sorted({label for _, label in sources})
            grid.arrow(height - 1, tx, ", ".join(labels))
    return grid.render()


_GLYPHS = {
    frozenset("U"): "│",
    frozenset("D"): "│",
    frozenset("UD"): "│",
    frozenset("L"): "─",
    frozenset("R"): "─",
    frozenset("LR"): "─",
    frozenset("UR"): "╰",
    frozenset("UL"): "╯",
    frozenset("DR"): "╭",
    frozenset("DL"): "╮",
    frozenset("UDR"): "├",
    frozenset("UDL"): "┤",
    frozenset("ULR"): "┴",
    frozenset("DLR"): "┬",
    frozenset("UDLR"): "┼",
}
"""The box-drawing character for each set of directions a line leaves a cell in."""


class _Grid:
    """A character grid where each cell collects the directions a line leaves it in.

    Lines are drawn by marking directions rather than characters, so crossings and
    junctions resolve to the right glyph (``┬``, ``┼``, …) however many lines meet.
    Text such as arrowheads and labels overrides the glyph in its cell.
    """

    def __init__(self, width: int, height: int) -> None:
        """Create an empty grid.

        Args:
            width: Columns in the grid; marks outside it are dropped.
            height: Rows in the grid.
        """
        self.width = width
        self.cells: list[list[set[str]]] = [[set() for _ in range(width)] for _ in range(height)]
        self.text: dict[tuple[int, int], str] = {}

    def add(self, y: int, x: int, direction: str) -> None:
        """Mark that a line leaves cell ``(y, x)`` in ``direction`` (one of U D L R).

        Args:
            y: Row of the cell.
            x: Column of the cell; ignored when outside the grid.
            direction: ``"U"``, ``"D"``, ``"L"`` or ``"R"``.
        """
        if 0 <= x < self.width:
            self.cells[y][x].add(direction)

    def arrow(self, y: int, x: int, label: str) -> None:
        """Place an arrowhead at ``(y, x)`` with ``label`` to its right.

        Args:
            y: Row of the arrowhead.
            x: Column of the arrowhead.
            label: Text written after one space; truncated at the grid's edge.
        """
        self.text[(y, x)] = ARROW
        for offset, char in enumerate(f" {label}", start=1):
            if x + offset < self.width:
                self.text[(y, x + offset)] = char

    def render(self) -> list[str]:
        """Return one string per row.

        Returns:
            The grid's rows, with text taking precedence over line glyphs.
        """
        lines: list[str] = []
        for y, row in enumerate(self.cells):
            chars = [self.text.get((y, x)) or _GLYPHS.get(frozenset(cell), " ") for x, cell in enumerate(row)]
            lines.append("".join(chars))
        return lines


__all__ = [
    "ARROW",
    "END",
    "GAP",
    "NODE_WIDTH",
    "START",
    "Edge",
    "Layout",
    "Slot",
    "centre",
    "components",
    "connectors",
    "layout",
]
