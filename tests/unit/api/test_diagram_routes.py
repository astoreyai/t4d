"""Tests for diagram API endpoints."""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ww.api.routes.diagrams import router
from ww.diagrams.schema import GraphEdge, GraphNode, UnifiedGraph, GraphMetadata


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/diagrams")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _mock_graph() -> UnifiedGraph:
    g = UnifiedGraph(
        nodes=[
            GraphNode(id="A", label="Node A", subgraph="sg1"),
            GraphNode(id="B", label="Node B", subgraph="sg2"),
            GraphNode(id="C", label="Node C", subgraph="sg1"),
        ],
        edges=[
            GraphEdge(source="A", target="B"),
            GraphEdge(source="B", target="C"),
        ],
        metadata=GraphMetadata(node_count=3, edge_count=2),
    )
    return g


def test_get_diagram_graph(client):
    with patch("ww.api.routes.diagrams._get_cached_graph", return_value=_mock_graph()):
        resp = client.get("/api/v1/diagrams/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 3
    assert len(data["edges"]) == 2


def test_get_diagram_graph_filter_subgraph(client):
    with patch("ww.api.routes.diagrams._get_cached_graph", return_value=_mock_graph()):
        resp = client.get("/api/v1/diagrams/graph?subgraph=sg1")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["nodes"]) == 2  # A and C in sg1


def test_get_diagram_metrics(client):
    with patch("ww.api.routes.diagrams._get_cached_graph", return_value=_mock_graph()):
        resp = client.get("/api/v1/diagrams/graph/metrics")
    assert resp.status_code == 200
    data = resp.json()
    assert "betweenness" in data
    assert "bottlenecks" in data
    assert "coupling_matrix" in data
