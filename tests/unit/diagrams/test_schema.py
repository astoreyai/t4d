"""Tests for the unified graph schema."""

from t4dm.diagrams.schema import (
    DiagramType,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    Subgraph,
    UnifiedGraph,
)


def test_graph_node_defaults():
    node = GraphNode(id="test")
    assert node.label == ""
    assert node.node_type == NodeType.PROCESS
    assert node.subgraph is None
    assert node.style == {}


def test_graph_edge_defaults():
    edge = GraphEdge(source="a", target="b")
    assert edge.edge_type == EdgeType.SOLID
    assert edge.weight == 1.0
    assert edge.label == ""


def test_subgraph_nesting():
    sg = Subgraph(id="child", label="Child", parent="parent", children=["n1", "n2"])
    assert sg.parent == "parent"
    assert len(sg.children) == 2


def test_unified_graph_construction():
    nodes = [
        GraphNode(id="a", label="Node A"),
        GraphNode(id="b", label="Node B"),
    ]
    edges = [GraphEdge(source="a", target="b", label="connects")]
    sg = Subgraph(id="sg1", label="Group 1", children=["a", "b"])

    graph = UnifiedGraph(nodes=nodes, edges=edges, subgraphs=[sg])
    graph.update_metadata()

    assert graph.metadata.node_count == 2
    assert graph.metadata.edge_count == 1
    assert graph.metadata.subgraph_count == 1


def test_unified_graph_serialization():
    graph = UnifiedGraph(
        nodes=[GraphNode(id="x", label="X", node_type=NodeType.DECISION)],
        edges=[GraphEdge(source="x", target="x", edge_type=EdgeType.DOTTED, weight=0.5)],
    )
    graph.update_metadata()

    data = graph.model_dump()
    assert data["nodes"][0]["node_type"] == "decision"
    assert data["edges"][0]["edge_type"] == "dotted"

    # Roundtrip
    restored = UnifiedGraph.model_validate(data)
    assert restored.nodes[0].id == "x"
    assert restored.edges[0].weight == 0.5


def test_node_by_id():
    graph = UnifiedGraph(nodes=[GraphNode(id="a"), GraphNode(id="b")])
    assert graph.node_by_id("a") is not None
    assert graph.node_by_id("a").id == "a"
    assert graph.node_by_id("missing") is None
