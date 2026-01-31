"""Graph analysis engine for bottleneck and flow detection."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from t4dm.diagrams.schema import UnifiedGraph


def betweenness_centrality(graph: UnifiedGraph) -> dict[str, float]:
    """Compute betweenness centrality using Brandes' algorithm."""
    # Build adjacency list
    adj: dict[str, list[str]] = defaultdict(list)
    node_ids = {n.id for n in graph.nodes}
    for e in graph.edges:
        if e.source in node_ids and e.target in node_ids:
            adj[e.source].append(e.target)

    cb: dict[str, float] = {n.id: 0.0 for n in graph.nodes}

    for s in node_ids:
        # BFS
        stack: list[str] = []
        pred: dict[str, list[str]] = {v: [] for v in node_ids}
        sigma: dict[str, int] = {v: 0 for v in node_ids}
        sigma[s] = 1
        dist: dict[str, int] = {v: -1 for v in node_ids}
        dist[s] = 0
        queue = [s]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in adj.get(v, []):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta: dict[str, float] = {v: 0.0 for v in node_ids}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w]) if sigma[w] > 0 else 0
            if w != s:
                cb[w] += delta[w]

    # Normalize
    n = len(node_ids)
    if n > 2:
        norm = 1.0 / ((n - 1) * (n - 2))
        cb = {k: v * norm for k, v in cb.items()}

    return cb


def degree_analysis(graph: UnifiedGraph) -> dict[str, dict[str, int]]:
    """Compute in-degree and out-degree for each node."""
    node_ids = {n.id for n in graph.nodes}
    degrees: dict[str, dict[str, int]] = {
        nid: {"in": 0, "out": 0, "total": 0} for nid in node_ids
    }
    for e in graph.edges:
        if e.source in degrees:
            degrees[e.source]["out"] += 1
            degrees[e.source]["total"] += 1
        if e.target in degrees:
            degrees[e.target]["in"] += 1
            degrees[e.target]["total"] += 1
    return degrees


def find_bottlenecks(graph: UnifiedGraph, top_k: int = 10) -> list[dict[str, Any]]:
    """Find top-k bottleneck nodes (high betweenness + high degree)."""
    bc = betweenness_centrality(graph)
    deg = degree_analysis(graph)

    scored = []
    for node in graph.nodes:
        b = bc.get(node.id, 0)
        d = deg.get(node.id, {})
        score = b * (1 + d.get("total", 0))
        scored.append({
            "id": node.id,
            "label": node.label,
            "betweenness": round(b, 6),
            "in_degree": d.get("in", 0),
            "out_degree": d.get("out", 0),
            "score": round(score, 6),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def detect_cycles(graph: UnifiedGraph) -> list[list[str]]:
    """Detect cycles in the graph using DFS."""
    adj: dict[str, list[str]] = defaultdict(list)
    node_ids = {n.id for n in graph.nodes}
    for e in graph.edges:
        if e.source in node_ids and e.target in node_ids:
            adj[e.source].append(e.target)

    cycles: list[list[str]] = []
    visited: set[str] = set()
    rec_stack: set[str] = set()
    path: list[str] = []

    def dfs(v: str) -> None:
        visited.add(v)
        rec_stack.add(v)
        path.append(v)

        for w in adj.get(v, []):
            if w not in visited:
                dfs(w)
            elif w in rec_stack:
                # Found cycle
                idx = path.index(w)
                cycles.append(path[idx:] + [w])

        path.pop()
        rec_stack.discard(v)

    for nid in node_ids:
        if nid not in visited:
            dfs(nid)

    return cycles


def subgraph_coupling_matrix(graph: UnifiedGraph) -> dict[str, dict[str, int]]:
    """Compute edge count between each pair of subgraphs."""
    # Map node -> subgraph
    node_sg: dict[str, str] = {}
    for n in graph.nodes:
        node_sg[n.id] = n.subgraph or "ungrouped"

    sg_names = sorted(set(node_sg.values()))
    matrix: dict[str, dict[str, int]] = {a: {b: 0 for b in sg_names} for a in sg_names}

    for e in graph.edges:
        src_sg = node_sg.get(e.source, "ungrouped")
        tgt_sg = node_sg.get(e.target, "ungrouped")
        if src_sg != tgt_sg:
            matrix[src_sg][tgt_sg] += 1
            matrix[tgt_sg][src_sg] += 1

    return matrix


def compute_all_metrics(graph: UnifiedGraph) -> dict[str, Any]:
    """Compute all flow analysis metrics."""
    bc = betweenness_centrality(graph)
    deg = degree_analysis(graph)
    bottlenecks = find_bottlenecks(graph)
    cycles = detect_cycles(graph)
    coupling = subgraph_coupling_matrix(graph)

    return {
        "betweenness": bc,
        "degrees": deg,
        "bottlenecks": bottlenecks,
        "cycles": [c for c in cycles[:20]],  # Cap at 20
        "coupling_matrix": coupling,
        "summary": {
            "total_nodes": len(graph.nodes),
            "total_edges": len(graph.edges),
            "cycle_count": len(cycles),
            "max_betweenness": max(bc.values()) if bc else 0,
            "avg_degree": sum(d["total"] for d in deg.values()) / len(deg) if deg else 0,
        },
    }
