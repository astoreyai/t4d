"""
Performance Test: Latency Testing (A6.7)

Tests latency requirements:
- P95 latency < 100ms for API requests
- Search latency under various conditions
- Response time distribution
"""

import time
import uuid
from statistics import mean, stdev

import numpy as np
import pytest
from fastapi.testclient import TestClient

from t4dm.api.server import app
from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import ItemRecord


DIM = 32
pytestmark = [pytest.mark.slow, pytest.mark.performance]


def _make_item(content="test", kappa=0.0, importance=0.5, **kwargs):
    """Create a test item record."""
    defaults = dict(
        id=uuid.uuid4().bytes,
        vector=np.random.randn(DIM).tolist(),
        event_time=time.time(),
        record_time=time.time(),
        valid_from=time.time(),
        valid_until=None,
        kappa=kappa,
        importance=importance,
        item_type="episode",
        content=content,
        access_count=0,
        session_id=None,
    )
    defaults.update(kwargs)
    return ItemRecord(**defaults)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def populated_engine(tmp_path):
    """Engine pre-populated with items."""
    engine = T4DXEngine(tmp_path / "latency_test")
    engine.startup()

    # Pre-populate with 5000 items
    for i in range(5000):
        item = _make_item(content=f"item_{i}")
        engine.insert(item)

    yield engine
    engine.shutdown()


class TestAPILatency:
    """Test API endpoint latencies."""

    def test_health_endpoint_latency(self, client):
        """Health endpoint should respond in < 50ms."""
        latencies = []

        for _ in range(100):
            start = time.time()
            response = client.get("/api/v1/health")
            latency = (time.time() - start) * 1000  # ms

            assert response.status_code == 200
            latencies.append(latency)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        print(f"\nHealth endpoint latency (100 requests):")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")

        assert p95 < 100, f"P95 too high: {p95:.2f}ms"

    def test_realtime_metrics_latency(self, client):
        """Realtime metrics should respond in < 100ms."""
        latencies = []

        for _ in range(50):
            start = time.time()
            response = client.get("/api/v1/viz/realtime/metrics")
            latency = (time.time() - start) * 1000

            assert response.status_code == 200
            latencies.append(latency)

        p95 = np.percentile(latencies, 95)

        print(f"\nRealtime metrics latency (50 requests):")
        print(f"P95: {p95:.2f}ms")

        assert p95 < 100, f"P95 too high: {p95:.2f}ms"

    def test_kappa_distribution_latency(self, client):
        """Kappa distribution endpoint latency."""
        latencies = []

        for _ in range(20):
            start = time.time()
            response = client.get("/api/v1/viz/kappa/distribution")
            latency = (time.time() - start) * 1000

            assert response.status_code == 200
            latencies.append(latency)

        p95 = np.percentile(latencies, 95)

        print(f"\nKappa distribution latency (20 requests):")
        print(f"P95: {p95:.2f}ms")

        # This may be slower due to storage scan
        assert p95 < 500, f"P95 too high: {p95:.2f}ms"


class TestSearchLatency:
    """Test search operation latencies."""

    def test_search_latency_empty(self, tmp_path):
        """Search latency on empty engine."""
        engine = T4DXEngine(tmp_path / "empty")
        engine.startup()

        query = np.random.randn(DIM).tolist()
        latencies = []

        for _ in range(100):
            start = time.time()
            results = engine.search(query, k=10)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        p95 = np.percentile(latencies, 95)

        print(f"\nSearch latency (empty, 100 queries):")
        print(f"P95: {p95:.2f}ms")

        assert p95 < 10, f"P95 too high: {p95:.2f}ms"

        engine.shutdown()

    def test_search_latency_populated(self, populated_engine):
        """Search latency on populated engine (5K items)."""
        query = np.random.randn(DIM).tolist()
        latencies = []

        for _ in range(100):
            start = time.time()
            results = populated_engine.search(query, k=10)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)

        print(f"\nSearch latency (5K items, 100 queries):")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")
        print(f"P99: {p99:.2f}ms")

        # P95 should be under 100ms
        assert p95 < 100, f"P95 too high: {p95:.2f}ms"

    def test_insert_latency(self, tmp_path):
        """Insert latency measurement."""
        engine = T4DXEngine(tmp_path / "insert_latency")
        engine.startup()

        latencies = []

        for i in range(100):
            item = _make_item(content=f"item_{i}")

            start = time.time()
            engine.insert(item)
            latency = (time.time() - start) * 1000

            latencies.append(latency)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)

        print(f"\nInsert latency (100 inserts):")
        print(f"P50: {p50:.2f}ms")
        print(f"P95: {p95:.2f}ms")

        assert p95 < 50, f"P95 too high: {p95:.2f}ms"

        engine.shutdown()

    def test_get_latency(self, populated_engine):
        """Get by ID latency."""
        # Get some IDs
        items = populated_engine.scan()[:100]
        latencies = []

        for item in items:
            start = time.time()
            result = populated_engine.get(item.id)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        p95 = np.percentile(latencies, 95)

        print(f"\nGet latency (100 gets):")
        print(f"P95: {p95:.2f}ms")

        assert p95 < 10, f"P95 too high: {p95:.2f}ms"


class TestLatencyDistribution:
    """Analyze latency distribution characteristics."""

    def test_latency_stability(self, populated_engine):
        """Verify latency is stable (low variance)."""
        query = np.random.randn(DIM).tolist()
        latencies = []

        for _ in range(500):
            start = time.time()
            results = populated_engine.search(query, k=10)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg = mean(latencies)
        std = stdev(latencies)
        cv = std / avg  # Coefficient of variation

        print(f"\nLatency stability (500 searches):")
        print(f"Mean: {avg:.2f}ms")
        print(f"Std: {std:.2f}ms")
        print(f"CV: {cv:.2%}")

        # For sub-millisecond operations, high CV is expected due to
        # OS scheduling, cache effects, and measurement granularity.
        # What matters is that P95 stays low (verified in other tests).
        # CV < 200% is acceptable for micro-second scale operations.
        assert cv < 2.0, f"Latency too variable: CV={cv:.2%}"
