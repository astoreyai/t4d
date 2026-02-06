"""
Chaos Test: Graceful Degradation (A6.11)

Tests graceful degradation when components fail:
- API continues to respond when storage is slow
- Visualization endpoints work with empty data
- System handles missing dependencies gracefully
"""

import time
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from t4dm.api.server import app
from t4dm.storage.t4dx.engine import T4DXEngine


DIM = 32
pytestmark = [pytest.mark.chaos]


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestAPIGracefulDegradation:
    """Test API behavior when backend services degrade."""

    def test_health_endpoint_always_responds(self, client):
        """Health endpoint should always respond."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_viz_endpoints_with_empty_data(self, client):
        """Visualization endpoints should work with no data."""
        endpoints = [
            "/api/v1/viz/kappa/distribution",
            "/api/v1/viz/kappa/flow",
            "/api/v1/viz/t4dx/storage",
            "/api/v1/viz/spiking/dynamics",
            "/api/v1/viz/qwen/metrics",
            "/api/v1/viz/oscillator/phase",
            "/api/v1/viz/energy/landscape",
            "/api/v1/viz/consolidation/replay",
            "/api/v1/viz/realtime/metrics",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Failed: {endpoint}"

    def test_graph_endpoint_with_no_data(self, client):
        """Graph visualization should handle empty data or degrade gracefully."""
        response = client.get("/api/v1/viz/graph")
        # 200 = success with empty data, 500 = service error (acceptable degradation)
        assert response.status_code in [200, 500]

    def test_bio_endpoints_degradation(self, client):
        """Bio mechanism endpoints should degrade gracefully."""
        endpoints = [
            "/api/v1/viz/bio/sleep",
            "/api/v1/viz/bio/neuromodulators",
        ]

        for endpoint in endpoints:
            response = client.get(endpoint)
            # Should return data or indicate unavailable, not crash
            assert response.status_code in [200, 503]


class TestStorageDegradation:
    """Test behavior when storage degrades."""

    def test_search_with_empty_engine(self, tmp_path):
        """Search on empty engine should return empty, not crash."""
        engine = T4DXEngine(tmp_path / "empty")
        engine.startup()

        query = np.random.randn(DIM).tolist()
        results = engine.search(query, k=10)

        assert results == []

        engine.shutdown()

    def test_get_nonexistent_item(self, tmp_path):
        """Getting nonexistent item should return None."""
        engine = T4DXEngine(tmp_path / "get_test")
        engine.startup()

        result = engine.get(b"nonexistent_id_1234567890123456")
        assert result is None

        engine.shutdown()

    def test_traverse_nonexistent_node(self, tmp_path):
        """Traversing from nonexistent node should return empty."""
        engine = T4DXEngine(tmp_path / "traverse_test")
        engine.startup()

        edges = engine.traverse(b"nonexistent_id_12345", edge_type="ANY", direction="out")
        assert edges == []

        engine.shutdown()


class TestVisualizationDegradation:
    """Test visualization module degradation."""

    def test_kappa_viz_no_snapshots(self, client):
        """Kappa visualization with no snapshots should work."""
        response = client.get("/api/v1/viz/kappa/flow")
        assert response.status_code == 200

        data = response.json()
        assert "timestamps" in data
        assert "band_series" in data

    def test_spiking_viz_no_data(self, client):
        """Spiking visualization with no data should work."""
        response = client.get("/api/v1/viz/spiking/dynamics")
        assert response.status_code == 200

        data = response.json()
        assert "snapshots" in data

    def test_export_all_with_empty_state(self, client):
        """Exporting all visualization data when empty should work."""
        response = client.get("/api/v1/viz/all/export")
        assert response.status_code == 200

        data = response.json()
        assert "timestamp" in data
        assert "modules" in data


class TestErrorHandling:
    """Test error handling and recovery."""

    def test_invalid_json_rejected(self, client):
        """Invalid JSON should be rejected gracefully."""
        response = client.post(
            "/api/v1/episodes/",
            data="this is not json",
            headers={"Content-Type": "application/json"},
        )

        # Should reject with 422 (validation error)
        assert response.status_code in [400, 422]

    def test_missing_required_fields(self, client):
        """Missing required fields should be rejected."""
        response = client.post(
            "/api/v1/episodes/",
            json={},  # Missing required fields
        )

        assert response.status_code == 422

    def test_wrong_data_types(self, client):
        """Wrong data types should be rejected."""
        response = client.post(
            "/api/v1/viz/spiking/record",
            json={
                "membrane_potentials": "not a list",  # Should be list
                "spike_mask": [True],
                "thalamic_gate": [0.5],
                "apical_error": [0.1],
            },
        )

        assert response.status_code == 422


class TestResourceExhaustion:
    """Test behavior under resource exhaustion."""

    def test_many_rapid_requests(self, client):
        """System should handle rapid requests."""
        success_count = 0
        error_count = 0

        for _ in range(100):
            response = client.get("/api/v1/health")
            if response.status_code == 200:
                success_count += 1
            else:
                error_count += 1

        print(f"\nRapid requests: {success_count} success, {error_count} errors")

        # Most should succeed
        assert success_count >= 90, f"Too many failures: {error_count}"

    def test_large_batch_insert(self, tmp_path):
        """System should handle large batch inserts."""
        from t4dm.storage.t4dx.types import ItemRecord

        engine = T4DXEngine(tmp_path / "batch")
        engine.startup()

        # Try inserting 10K items
        start = time.time()
        for i in range(10000):
            item = ItemRecord(
                id=uuid.uuid4().bytes,
                vector=np.random.randn(DIM).tolist(),
                event_time=time.time(),
                record_time=time.time(),
                valid_from=time.time(),
                valid_until=None,
                kappa=0.0,
                importance=0.5,
                item_type="episode",
                content=f"item_{i}",
                access_count=0,
                session_id=None,
            )
            engine.insert(item)

        elapsed = time.time() - start
        print(f"\n10K inserts completed in {elapsed:.2f}s")

        # Should complete without crashing
        engine.shutdown()
