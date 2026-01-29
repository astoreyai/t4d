"""Tests for flow analysis engine."""

from ww.diagrams.flow_analysis import (
    betweenness_centrality,
    degree_analysis,
    detect_cycles,
    find_bottlenecks,
    subgraph_coupling_matrix,
)
from ww.diagrams.schema import GraphEdge, GraphNode, UnifiedGraph


def _diamond_graph() -> UnifiedGraph:
    """A->B, A->C, B->D, C->D (diamond shape)."""
    g = UnifiedGraph(
        nodes=[
            GraphNode(id="A", label="A"),
            GraphNode(id="B", label="B"),
            GraphNode(id="C", label="C"),
            GraphNode(id="D", label="D"),
        ],
        edges=[
            GraphEdge(source="A", target="B"),
            GraphEdge(source="A", target="C"),
            GraphEdge(source="B", target="D"),
            GraphEdge(source="C", target="D"),
        ],
    )
    g.update_metadata()
    return g


def test_betweenness_known_graph():
    g = _diamond_graph()
    bc = betweenness_centrality(g)
    # B and C are on shortest paths from A to D
    assert bc["B"] > 0
    assert bc["C"] > 0
    # A and D are endpoints, lower centrality
    assert bc["A"] <= bc["B"] or bc["A"] <= bc["C"]


def test_degree_analysis():
    g = _diamond_graph()
    deg = degree_analysis(g)
    assert deg["A"]["out"] == 2
    assert deg["A"]["in"] == 0
    assert deg["D"]["in"] == 2
    assert deg["D"]["out"] == 0


def test_find_bottlenecks():
    g = _diamond_graph()
    bots = find_bottlenecks(g, top_k=2)
    assert len(bots) == 2
    assert bots[0]["id"] in ("B", "C")


def test_detect_cycles():
    g = UnifiedGraph(
        nodes=[GraphNode(id="A"), GraphNode(id="B"), GraphNode(id="C")],
        edges=[
            GraphEdge(source="A", target="B"),
            GraphEdge(source="B", target="C"),
            GraphEdge(source="C", target="A"),
        ],
    )
    g.update_metadata()
    cycles = detect_cycles(g)
    assert len(cycles) >= 1


def test_no_cycles_in_dag():
    g = _diamond_graph()
    cycles = detect_cycles(g)
    assert len(cycles) == 0


def test_coupling_matrix():
    g = UnifiedGraph(
        nodes=[
            GraphNode(id="A", subgraph="sg1"),
            GraphNode(id="B", subgraph="sg2"),
        ],
        edges=[GraphEdge(source="A", target="B")],
    )
    g.update_metadata()
    matrix = subgraph_coupling_matrix(g)
    assert matrix["sg1"]["sg2"] == 1
    assert matrix["sg2"]["sg1"] == 1  # Symmetric
