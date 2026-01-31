"""Tests for graph merger."""

from pathlib import Path

import pytest

from t4dm.diagrams.graph_merger import merge_from_directory
from t4dm.diagrams.schema import GraphEdge, GraphNode, Subgraph, UnifiedGraph
from t4dm.diagrams.graph_merger import merge_graphs

DIAGRAMS_DIR = Path(__file__).resolve().parents[3] / "docs" / "diagrams"


def test_merge_simple():
    g1 = UnifiedGraph(
        nodes=[GraphNode(id="A", label="A"), GraphNode(id="B", label="B")],
        edges=[GraphEdge(source="A", target="B")],
    )
    g2 = UnifiedGraph(
        nodes=[GraphNode(id="A", label="Node A"), GraphNode(id="C", label="C")],
        edges=[GraphEdge(source="A", target="C")],
    )
    merged = merge_graphs({"g1": g1, "g2": g2})
    # A should be deduplicated
    assert len(merged.nodes) == 3
    assert len(merged.edges) == 2
    # A should have cross_diagram metadata
    a_node = merged.node_by_id("A")
    assert a_node.metadata.get("cross_diagram") is True
    # Longer label should win
    assert a_node.label == "Node A"


def test_merge_preserves_subgraphs():
    g1 = UnifiedGraph(
        nodes=[GraphNode(id="X")],
        subgraphs=[Subgraph(id="sg1", label="Group")],
    )
    g2 = UnifiedGraph(
        nodes=[GraphNode(id="Y")],
        subgraphs=[Subgraph(id="sg2", label="Other")],
    )
    merged = merge_graphs({"g1": g1, "g2": g2})
    assert len(merged.subgraphs) == 2


@pytest.mark.skipif(not DIAGRAMS_DIR.exists(), reason="Diagrams dir not found")
def test_merge_all():
    merged = merge_from_directory(DIAGRAMS_DIR)
    assert merged.metadata.node_count > 200
    assert merged.metadata.edge_count > 150
    # No duplicate node IDs
    ids = [n.id for n in merged.nodes]
    assert len(ids) == len(set(ids))


@pytest.mark.skipif(not DIAGRAMS_DIR.exists(), reason="Diagrams dir not found")
def test_cross_diagram_nodes_exist():
    merged = merge_from_directory(DIAGRAMS_DIR)
    cross = [n for n in merged.nodes if n.metadata.get("cross_diagram")]
    # Should have some shared nodes across diagrams
    assert len(cross) > 0
