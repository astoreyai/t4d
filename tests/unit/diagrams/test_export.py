"""Tests for graph export formats."""

import json
from xml.etree import ElementTree

from t4dm.diagrams.export import to_gexf, to_graphml, to_json
from t4dm.diagrams.schema import GraphEdge, GraphNode, UnifiedGraph


def _sample_graph() -> UnifiedGraph:
    g = UnifiedGraph(
        nodes=[
            GraphNode(id="a", label="A", style={"fill": "#ff0000"}),
            GraphNode(id="b", label="B"),
            GraphNode(id="c", label="C"),
        ],
        edges=[
            GraphEdge(source="a", target="b", label="connects"),
            GraphEdge(source="b", target="c"),
        ],
    )
    g.update_metadata()
    return g


def test_graphml_valid_xml():
    xml_str = to_graphml(_sample_graph())
    root = ElementTree.fromstring(xml_str)
    nodes = root.findall(".//{http://graphml.graphstruct.org/xmlns}node")
    edges = root.findall(".//{http://graphml.graphstruct.org/xmlns}edge")
    assert len(nodes) == 3
    assert len(edges) == 2


def test_gexf_valid_xml():
    xml_str = to_gexf(_sample_graph())
    root = ElementTree.fromstring(xml_str)
    nodes = root.findall(".//{http://gexf.net/1.3}node")
    edges = root.findall(".//{http://gexf.net/1.3}edge")
    assert len(nodes) == 3
    assert len(edges) == 2


def test_json_roundtrip():
    graph = _sample_graph()
    json_str = to_json(graph)
    data = json.loads(json_str)
    restored = UnifiedGraph.model_validate(data)
    assert len(restored.nodes) == len(graph.nodes)
    assert len(restored.edges) == len(graph.edges)
    assert restored.nodes[0].id == graph.nodes[0].id
