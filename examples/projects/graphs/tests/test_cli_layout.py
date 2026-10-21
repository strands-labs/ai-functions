"""Layered layout and box-drawing connectors for a workflow's thread graph."""

from __future__ import annotations

import pytest

from layout import ARROW, END, NODE_WIDTH, START, centre, components, connectors, layout
from recordings import StepSpec


def _step(name: str, *deps: str, output_type: str = "str") -> StepSpec:
    return StepSpec(name=name, depends_on=list(deps), output_type=output_type)


def _names(graph) -> list[list[str]]:  # type: ignore[no-untyped-def]
    return [[slot.name for slot in row] for row in graph.rows]


# ── Layering ──────────────────────────────────────────────────────────────────


def test_a_chain_is_one_node_per_row() -> None:
    graph = layout([_step("a"), _step("b", "a"), _step("c", "b")])
    assert _names(graph) == [["a"], ["b"], ["c"]]
    assert [(e.source, e.target) for e in graph.edges] == [("a", "b"), ("b", "c")]


def test_independent_steps_share_a_row_and_a_fan_in_sits_below() -> None:
    graph = layout([_step("north"), _step("south"), _step("east"), _step("totals", "north", "south", "east")])
    assert _names(graph) == [["north", "south", "east"], ["totals"]]
    assert len(graph.edges) == 3
    assert [e.source for e in graph.incoming("totals")] == ["north", "south", "east"]


def test_a_skip_edge_gets_a_pass_through_in_the_skipped_row() -> None:
    graph = layout([_step("a"), _step("b", "a"), _step("c", "a", "b")])
    assert _names(graph) == [["a"], ["b", "a→c"], ["c"]]
    (passthrough,) = [s for s in graph.rows[1] if s.passthrough]
    assert passthrough.edge is not None and (passthrough.edge.source, passthrough.edge.target) == ("a", "c")


def test_edge_labels_carry_the_upstream_output_type() -> None:
    graph = layout(
        [
            _step("extract", output_type="str"),
            _step("triage", "extract", output_type="TriageReport"),
            _step("report", "triage"),
        ]
    )
    assert {(e.target, e.label) for e in graph.edges} == {("triage", "str"), ("report", "TriageReport")}


def test_unknown_dependencies_are_ignored_and_cycles_do_not_hang() -> None:
    graph = layout([_step("a", "ghost"), _step("b", "c"), _step("c", "b")])
    assert _names(graph)[0] == ["a"]
    assert sorted(_names(graph)[1]) == ["b", "c"]


# ── Connectors ────────────────────────────────────────────────────────────────


def test_a_straight_edge_is_a_vertical_line_ending_in_an_arrow() -> None:
    graph = layout([_step("a"), _step("b", "a")])
    lines = connectors(graph, 0)
    x = centre(0)
    assert [line[x] for line in lines] == ["│", "│", ARROW]
    assert lines[-1][x + 2 : x + 5] == "str"


def test_a_fan_in_joins_three_sources_on_one_bus() -> None:
    graph = layout([_step("north"), _step("south"), _step("east"), _step("totals", "north", "south", "east")])
    top, bus, arrows = connectors(graph, 0)
    assert [top[centre(i)] for i in range(3)] == ["│", "│", "│"]
    assert bus[centre(0)] == "├"  # north continues straight down to totals, the bus joins from the right
    assert bus[centre(1)] == "┴"  # south meets the bus from above
    assert bus[centre(2)] == "╯"  # east turns inward
    assert bus[centre(0) + 1 : centre(2)].replace("┴", "─") == "─" * (centre(2) - centre(0) - 1)
    assert arrows[centre(0)] == ARROW  # totals sits in column 0 of its row


def test_a_fan_out_shares_one_bus_and_two_arrows() -> None:
    graph = layout([_step("split"), _step("left", "split"), _step("right", "split")])
    top, bus, arrows = connectors(graph, 0)
    assert top[centre(0)] == "│"
    assert bus[centre(0)] == "├" and bus[centre(1)] == "╮"  # straight down to left, sideways to right
    assert arrows[centre(0)] == ARROW and arrows[centre(1)] == ARROW


def test_disjoint_edges_share_a_bus_line_and_overlapping_ones_stack() -> None:
    disjoint = layout([_step("a"), _step("b"), _step("c", "a"), _step("d", "b")])
    assert len(connectors(disjoint, 0)) == 3  # top, one bus, arrows
    crossing = layout([_step("a"), _step("b"), _step("c"), _step("d", "a", "b"), _step("e", "b", "c")])
    assert len(connectors(crossing, 0)) == 4  # two buses: d's span and e's span both touch b
    straight_plus_bus = layout([_step("a"), _step("b"), _step("c", "a", "b"), _step("d", "b")])
    assert len(connectors(straight_plus_bus, 0)) == 3  # d sits under b: a straight drop reserves no bus line


def test_a_pass_through_carries_its_edge_straight_down() -> None:
    graph = layout([_step("a"), _step("b", "a"), _step("c", "a", "b")])
    first = connectors(graph, 0)  # a → b, and a → (a→c) pass-through in column 1
    assert first[0][centre(0)] == "│"
    assert first[-1][centre(0)] == ARROW and first[-1][centre(1)] == "│"
    assert first[-1].count("str") == 1
    second = connectors(graph, 1)  # b → c and (a→c) → c meet on c
    assert second[-1][centre(0)] == ARROW
    assert second[1][centre(1)] == "╯"


@pytest.mark.parametrize("widens", [False, True])
def test_a_long_edge_stays_connected_when_its_column_changes(widens: bool) -> None:
    steps = (
        [_step("source"), _step("work", "source"), _step("left", "work"), _step("right", "work")]
        if widens
        else [_step("source"), _step("left", "source"), _step("right", "source"), _step("work", "left", "right")]
    )
    steps.append(_step("finish", "source", *("left", "right") if widens else ("work",)))
    graph = layout(steps)
    upper_column, lower_column = (1, 2) if widens else (2, 1)
    assert graph.rows[1][upper_column].name == graph.rows[2][lower_column].name == "source→finish"
    lines = connectors(graph, 1)
    assert lines[0][centre(upper_column)] == "│"
    assert lines[-1][centre(lower_column)] == "│"
    assert not lines[-1][centre(lower_column) + 1 :].strip()
    left, right = sorted((centre(upper_column), centre(lower_column)))
    assert any(all(char in "─┬┴┼" for char in line[left + 1 : right]) for line in lines[1:-1])


def test_long_edges_show_type_labels_only_at_destinations() -> None:
    graph = layout(
        [
            _step("source", output_type="Payload"),
            _step("work", "source"),
            _step("review", "work"),
            _step("finish", "source", "review"),
        ]
    )
    for index in range(2):
        lines = connectors(graph, index)
        assert lines[-1][centre(1)] == "│"
        assert not lines[-1][centre(1) + 1 :].strip()
        assert lines[-1].count(ARROW) == 1
    assert "Payload, str" in connectors(graph, 2)[-1]


def test_rows_with_no_edges_between_them_render_a_blank_line() -> None:
    graph = layout([_step("a"), _step("b")])
    graph.rows.append([])  # nothing below the single row
    assert connectors(graph, 0) == [" " * (2 * (NODE_WIDTH + 2))]


def test_connector_lines_are_equal_width() -> None:
    graph = layout([_step("north"), _step("south"), _step("east"), _step("totals", "north", "south", "east")])
    lines = connectors(graph, 0)
    assert len({len(line) for line in lines}) == 1


# ── Terminals ─────────────────────────────────────────────────────────────────


def test_terminals_fan_out_to_roots_and_in_from_leaves() -> None:
    graph = layout(
        [
            _step("stats"),
            _step("report"),
            _step("merge", "stats", "report"),
            _step("left", "merge"),
            _step("right", "merge"),
        ],
        terminals=True,
    )
    assert _names(graph) == [[START], ["stats", "report"], ["merge"], ["left", "right"], [END]]
    assert graph.rows[0][0].terminal and graph.rows[-1][0].terminal
    assert [(e.source, e.target) for e in graph.edges] == [
        (START, "stats"),
        (START, "report"),
        ("stats", "merge"),
        ("report", "merge"),
        ("merge", "left"),
        ("merge", "right"),
        ("left", END),
        ("right", END),
    ]
    top, bus, arrows = connectors(graph, 0)
    assert top[centre(0)] == "│"
    assert bus[centre(0)] == "├" and bus[centre(1)] == "╮"  # one dot fans out to two boxes
    assert arrows[centre(0)] == ARROW and arrows[centre(1)] == ARROW
    _, bus2, arrows2 = connectors(graph, 3)
    assert bus2[centre(0)] == "├" and bus2[centre(1)] == "╯"  # and both fan back in
    assert arrows2[centre(0)] == ARROW


def test_terminal_edges_carry_input_and_output_types() -> None:
    steps = [StepSpec(name="triage", input_type="str", output_type="TriageReport")]
    graph = layout(steps, terminals=True)
    assert {(e.source, e.target, e.label) for e in graph.edges} == {
        (START, "triage", "str"),
        ("triage", END, "TriageReport"),
    }


def test_a_leaf_in_an_early_row_reaches_the_end_through_pass_throughs() -> None:
    graph = layout([_step("a"), _step("b", "a"), _step("side", "a"), _step("c", "b")], terminals=True)
    assert _names(graph) == [[START], ["a"], ["b", "side"], ["c", f"side→{END}"], [END]]


def test_disconnected_components_keep_only_their_real_dependencies() -> None:
    graph = layout(
        [
            _step("a_start"),
            _step("a_work", "a_start"),
            _step("a_finish", "a_work"),
            _step("b_start"),
            _step("b_work", "b_start"),
            _step("b_finish", "b_work"),
            _step("independent"),
        ],
        terminals=True,
    )
    assert _names(graph) == [["a_start", "b_start", "independent"], ["a_work", "b_work"], ["a_finish", "b_finish"]]
    assert [(edge.source, edge.target) for edge in graph.edges] == [
        ("a_start", "a_work"),
        ("a_work", "a_finish"),
        ("b_start", "b_work"),
        ("b_work", "b_finish"),
    ]
    for index in range(2):
        for line in connectors(graph, index):
            assert not line[centre(0) + 6 : centre(1)].strip()


def test_disconnected_cycles_and_unknown_dependencies_do_not_gain_shared_terminals() -> None:
    graph = layout([_step("a", "ghost"), _step("b", "c"), _step("c", "b")], terminals=True)
    assert _names(graph) == [["a"], ["b", "c"]]
    assert _names(layout([_step("a"), _step("b")], terminals=True)) == [["a", "b"]]
    assert layout([], terminals=True).rows == []


def test_without_terminals_nothing_changes() -> None:
    graph = layout([_step("a"), _step("b", "a")])
    assert _names(graph) == [["a"], ["b"]] and not any(s.terminal for row in graph.rows for s in row)


def test_components_preserve_order_and_have_their_own_terminals() -> None:
    steps = [_step("a_end", "a_start"), _step("alone"), _step("b_start"), _step("a_start"), _step("b_end", "b_start")]
    groups = components(steps)
    assert [[step.name for step in group] for group in groups] == [
        ["a_end", "a_start"],
        ["alone"],
        ["b_start", "b_end"],
    ]
    for group in groups:
        graph = layout(group, terminals=True)
        assert graph.rows[0][0].name == START
        assert graph.rows[-1][0].name == END
        names = {step.name for step in group} | {START, END}
        assert all(edge.source in names and edge.target in names for edge in graph.edges)
    assert _names(layout(groups[1], terminals=True)) == [[START], ["alone"], [END]]


def test_components_handle_cycles_unknown_dependencies_and_empty_graphs() -> None:
    groups = components([_step("a", "ghost"), _step("b", "c"), _step("c", "b")])
    assert [[step.name for step in group] for group in groups] == [["a"], ["b", "c"]]
    assert components([]) == []
