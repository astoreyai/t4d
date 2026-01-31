"""Diagram graph API endpoints."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Query

from t4dm.diagrams.flow_analysis import compute_all_metrics
from t4dm.diagrams.graph_merger import merge_from_directory
from t4dm.diagrams.schema import UnifiedGraph

router = APIRouter()

# Default diagrams directory (relative to project root)
_DIAGRAMS_DIR = Path(__file__).resolve().parents[3] / "docs" / "diagrams"


@functools.lru_cache(maxsize=1)
def _get_cached_graph() -> UnifiedGraph:
    """Parse and cache the unified graph."""
    return merge_from_directory(_DIAGRAMS_DIR)


@router.get("/graph")
async def get_diagram_graph(
    subgraph: str | None = Query(None, description="Filter by subgraph name"),
    diagram_type: str | None = Query(None, description="Filter by diagram type"),
) -> dict[str, Any]:
    """Return the unified diagram graph as JSON."""
    graph = _get_cached_graph()

    nodes = graph.nodes
    if subgraph:
        nodes = [n for n in nodes if n.subgraph and subgraph.lower() in n.subgraph.lower()]
    if diagram_type:
        nodes = [n for n in nodes if n.diagram_type.value == diagram_type]

    node_ids = {n.id for n in nodes}
    edges = [e for e in graph.edges if e.source in node_ids and e.target in node_ids]

    filtered = UnifiedGraph(nodes=nodes, edges=edges, subgraphs=graph.subgraphs)
    filtered.update_metadata()
    return filtered.model_dump()


@router.get("/graph/metrics")
async def get_diagram_metrics() -> dict[str, Any]:
    """Return flow analysis metrics for the diagram graph."""
    graph = _get_cached_graph()
    return compute_all_metrics(graph)
