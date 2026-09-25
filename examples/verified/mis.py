"""Reasoning solver that uses the structure of the input to find a solution quickly.

The problem is to find a maximum independent set on a graph, which generally takes
exponential time. However, using the particular structure of the graph a solution can
be found more quickly.

The agent is provided with a generic solver that is certified to find the correct solution,
but only works on small graphs. The agent has to discover the particular structure of
the input graph to splits the graph into pieces the solver accepts, piece together a
global solution, and proves its correctness for the whole graph.

Demonstrates:
- An agent that explores the input structure to simplify a computationally intractable problem
- A tool whose results carry guarantees (`Certified`)
- Tools creating auditable trust boundaries
"""

import asyncio
from pathlib import Path

import networkx as nx
from example_helpers import models

from ai_functions import scope
from ai_functions.cli import print_event
from ai_functions.experimental import verified
from ai_functions.experimental.verified.function import Certified
from ai_functions.experimental.verified.lean import LeanProject
from ai_functions.experimental.verified.lean.types import encode

project = LeanProject(Path(__file__).parent / "lean", imports=["MIS"])
MIS = project.symbols.MIS

SOLVER_CAP = 15
"""Largest graph the solver accepts. The instance below is deliberately bigger."""

Graph = tuple[list[int], list[tuple[int, int]]]


def exact_mis(verts: list[int], edges: list[tuple[int, int]]) -> int:
    """Exact maximum independent set size: the maximum clique of the complement graph."""
    graph = nx.Graph()
    graph.add_nodes_from(verts)
    graph.add_edges_from(edges)
    _, size = nx.max_weight_clique(nx.complement(graph), weight=None)
    return int(size)


@verified.tool(MIS.misSolver, stem="mis")
def mis_solver(graph: Graph) -> Certified:
    """Exact maximum-independent-set solver for graphs of at most 15 vertices.

    Returns the size of a maximum independent set of the graph, with the
    guarantee that it is optimal for that graph. The graph is `(verts, edges)`:
    distinct natural vertices, edges as `(u, v)` pairs with both endpoints in
    `verts`, no self-loops. Refuses larger or malformed graphs; a refused call
    records nothing.
    """
    verts, edges = graph
    if len(verts) > SOLVER_CAP:
        raise ValueError(
            f"solver cap exceeded: {len(verts)} vertices, at most {SOLVER_CAP} accepted. "
            "Reduce the graph first and solve the pieces."
        )
    if len(set(verts)) != len(verts):
        raise ValueError("`verts` contains duplicates")
    for u, v in edges:
        if u not in verts or v not in verts:
            raise ValueError(f"edge {(u, v)!r} has an endpoint outside `verts`")
        if u == v:
            raise ValueError(f"self-loop {(u, v)!r} not accepted")
    size = exact_mis(verts, edges)
    # The claim, made where it is earned: this size is optimal for exactly the
    # graph the solver was just run on. `value` is the returned size as Lean source.
    lean_graph = encode(graph, MIS.misSolver.info.parameters[0].type)
    return Certified(size, guarantees=lambda value: f"MIS.MaxIndependentSize {lean_graph} {value}")


# The contract `MIS.Contract size verts edges` is about the whole instance, a
# graph the solver refuses. There is no strategy guidance: the reduction
# theorems are in the project source, and finding the pendant-then-split route
# is the task.
@verified.ai_function(
    contract=MIS.Contract,
    tools=[mis_solver],
    model=models.large,
    max_attempts=12,
)
def max_independent_set(verts: list[int], edges: list[tuple[int, int]]) -> int:
    """Find the size of a maximum independent set of an undirected graph.

    Vertices: {verts}
    Edges: {edges}

    Return the optimal size, and prove both that it is achieved by some
    independent set and that no independent set is larger.
    """


def instance() -> Graph:
    """29 vertices, connected, above the solver cap.

    Two chorded cycles (1–12 and 13–26) glued through a hub vertex 0, a leaf 28
    tied to the hub, and a pendant vertex 27 hanging off it. Everything routes
    through vertex 0, so the pendant reduction, which deletes 0, shatters the
    graph into solver-sized pieces.
    """
    cycle_a = [(i, i + 1) for i in range(1, 12)] + [(1, 12)]
    cycle_b = [(i, i + 1) for i in range(13, 26)] + [(13, 26)]
    chords = [(2, 8), (4, 10), (14, 21), (17, 24)]
    hub = [(0, 1), (0, 6), (0, 13), (0, 20), (0, 28), (0, 27)]
    return list(range(29)), cycle_a + cycle_b + chords + hub


async def main() -> None:
    print("Preparing Lean and certifying the maximum independent set...", flush=True)
    verts, edges = instance()
    async with scope(on_event=print_event):
        best, certificate = await max_independent_set.with_certificate(verts, edges)
    print(f"max_independent_set(|V|={len(verts)}, |E|={len(edges)}) = {best}")
    certificate.summary()
    path = certificate.write(Path(__file__).parent / "data" / "mis.lean")
    print(f"Certificate: {path}")


if __name__ == "__main__":
    asyncio.run(main())
