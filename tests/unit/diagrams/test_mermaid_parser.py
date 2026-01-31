"""Tests for Mermaid diagram parser."""

from pathlib import Path

import pytest

from t4dm.diagrams.mermaid_parser import (
    detect_diagram_type,
    parse_directory,
    parse_file,
    parse_flowchart,
    parse_sequence_diagram,
    parse_state_diagram,
)
from t4dm.diagrams.schema import DiagramType, EdgeType

DIAGRAMS_DIR = Path(__file__).resolve().parents[3] / "docs" / "diagrams"


def test_detect_flowchart():
    assert detect_diagram_type("flowchart TB\n  A --> B") == DiagramType.FLOWCHART
    assert detect_diagram_type("graph TB\n  A --> B") == DiagramType.FLOWCHART


def test_detect_state():
    assert detect_diagram_type("stateDiagram-v2\n  [*] --> A") == DiagramType.STATE


def test_detect_sequence():
    assert detect_diagram_type("sequenceDiagram\n  A->>B: msg") == DiagramType.SEQUENCE


def test_detect_with_init_header():
    content = '%%{init: {"theme": "base"}}%%\nflowchart TB\n  A --> B'
    assert detect_diagram_type(content) == DiagramType.FLOWCHART


def test_parse_simple_flowchart():
    content = """flowchart TB
    A[Node A] --> B[Node B]
    B -.-> C{Decision}
    C ==> D([Store])
    """
    graph = parse_flowchart(content, "test.mmd")
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 3

    # Check edge types
    etypes = [e.edge_type for e in graph.edges]
    assert EdgeType.SOLID in etypes
    assert EdgeType.DOTTED in etypes
    assert EdgeType.THICK in etypes


def test_parse_subgraphs():
    content = """flowchart TB
    subgraph Outer["Outer Group"]
        subgraph Inner["Inner"]
            A[Node A]
        end
        B[Node B]
    end
    A --> B
    """
    graph = parse_flowchart(content, "test.mmd")
    assert len(graph.subgraphs) == 2
    # Inner is child of Outer
    inner = [sg for sg in graph.subgraphs if sg.id == "Inner"][0]
    assert inner.parent == "Outer"


def test_parse_edge_labels():
    content = """flowchart TB
    A -->|"excites"| B
    C -- inhibits --> D
    """
    graph = parse_flowchart(content, "test.mmd")
    labels = {e.label for e in graph.edges}
    assert "excites" in labels
    assert "inhibits" in labels


def test_parse_style_directives():
    content = """flowchart TB
    A[Node]
    style A fill:#ff0000,stroke:#000
    """
    graph = parse_flowchart(content, "test.mmd")
    node = graph.node_by_id("A")
    assert node is not None
    assert node.style.get("fill") == "#ff0000"


def test_parse_html_entities():
    content = """flowchart TB
    A[Node<br/>Label]
    """
    graph = parse_flowchart(content, "test.mmd")
    node = graph.node_by_id("A")
    assert "<br/>" not in node.label


@pytest.mark.skipif(not DIAGRAMS_DIR.exists(), reason="Diagrams dir not found")
def test_parse_nca_module_map():
    graph = parse_file(DIAGRAMS_DIR / "nca_module_map.mermaid")
    assert len(graph.nodes) > 20
    assert len(graph.edges) > 15
    assert len(graph.subgraphs) >= 6


@pytest.mark.skipif(not DIAGRAMS_DIR.exists(), reason="Diagrams dir not found")
def test_parse_vta_circuit():
    graph = parse_file(DIAGRAMS_DIR / "vta_circuit.mermaid")
    assert len(graph.nodes) > 10
    # Should have both solid and dotted edges
    etypes = {e.edge_type for e in graph.edges}
    assert EdgeType.SOLID in etypes
    assert EdgeType.DOTTED in etypes


@pytest.mark.skipif(not DIAGRAMS_DIR.exists(), reason="Diagrams dir not found")
def test_parse_state_circuit_breaker():
    graph = parse_file(DIAGRAMS_DIR / "31_state_circuit_breaker.mmd")
    assert len(graph.nodes) > 5
    assert len(graph.edges) > 5
    assert len(graph.subgraphs) >= 3


@pytest.mark.skipif(not DIAGRAMS_DIR.exists(), reason="Diagrams dir not found")
def test_parse_sequence_store_memory():
    graph = parse_file(DIAGRAMS_DIR / "41_seq_store_memory.mmd")
    assert len(graph.nodes) >= 5  # participants
    assert len(graph.edges) > 10  # messages


@pytest.mark.skipif(not DIAGRAMS_DIR.exists(), reason="Diagrams dir not found")
def test_parse_all_files_no_exceptions():
    """All diagram files should parse without raising exceptions."""
    results = parse_directory(DIAGRAMS_DIR)
    assert len(results) > 20
    for name, graph in results.items():
        assert len(graph.nodes) >= 0, f"Failed: {name}"
