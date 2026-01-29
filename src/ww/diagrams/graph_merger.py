"""Merge multiple parsed diagrams into a single UnifiedGraph."""

from __future__ import annotations

from pathlib import Path

from ww.diagrams.mermaid_parser import parse_directory
from ww.diagrams.schema import GraphEdge, GraphNode, Subgraph, UnifiedGraph


def merge_graphs(graphs: dict[str, UnifiedGraph]) -> UnifiedGraph:
    """Merge multiple UnifiedGraphs, deduplicating nodes by ID.

    Nodes with the same ID across diagrams are merged (metadata combined).
    Cross-diagram edges are created for shared nodes.
    """
    merged_nodes: dict[str, GraphNode] = {}
    merged_edges: list[GraphEdge] = []
    merged_subgraphs: dict[str, Subgraph] = {}

    # Track which diagrams each node appears in
    node_sources: dict[str, list[str]] = {}

    for filename, graph in graphs.items():
        for node in graph.nodes:
            if node.id in merged_nodes:
                # Merge: keep existing, add source info
                existing = merged_nodes[node.id]
                existing.metadata.setdefault("also_in", [])
                existing.metadata["also_in"].append(filename)
                # Prefer longer labels
                if len(node.label) > len(existing.label):
                    existing.label = node.label
                # Merge styles
                existing.style.update(node.style)
            else:
                merged_nodes[node.id] = node.model_copy()

            node_sources.setdefault(node.id, []).append(filename)

        for edge in graph.edges:
            merged_edges.append(edge)

        for sg in graph.subgraphs:
            sg_key = f"{filename}::{sg.id}"
            if sg_key not in merged_subgraphs:
                sg_copy = sg.model_copy()
                sg_copy.id = sg_key
                merged_subgraphs[sg_key] = sg_copy

    # Mark cross-diagram nodes
    for node_id, sources in node_sources.items():
        if len(sources) > 1:
            merged_nodes[node_id].metadata["cross_diagram"] = True
            merged_nodes[node_id].metadata["diagram_count"] = len(sources)
            merged_nodes[node_id].metadata["diagrams"] = sources

    result = UnifiedGraph(
        nodes=list(merged_nodes.values()),
        edges=merged_edges,
        subgraphs=list(merged_subgraphs.values()),
    )
    result.update_metadata()
    result.metadata.diagram_count = len(graphs)
    return result


def merge_from_directory(input_dir: Path) -> UnifiedGraph:
    """Parse all diagrams in a directory and merge into one graph."""
    graphs = parse_directory(input_dir)
    return merge_graphs(graphs)
