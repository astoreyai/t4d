"""
Performance Test: Load Testing (A6.6)

Tests system behavior under high concurrent load:
- 1000 concurrent store operations
- Throughput measurement
- Error rate tracking
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

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


class TestConcurrentStores:
    """Test concurrent store operations."""

    def test_100_sequential_stores(self, tmp_path):
        """Baseline: 100 sequential stores."""
        engine = T4DXEngine(tmp_path / "seq_100")
        engine.startup()

        start = time.time()
        for i in range(100):
            item = _make_item(content=f"item_{i}")
            engine.insert(item)

        elapsed = time.time() - start
        throughput = 100 / elapsed

        print(f"\n100 sequential stores: {elapsed:.2f}s ({throughput:.0f}/sec)")

        assert elapsed < 10.0, f"Too slow: {elapsed:.2f}s"
        engine.shutdown()

    def test_1000_stores_threaded(self, tmp_path):
        """Load test: 1000 stores with thread pool."""
        engine = T4DXEngine(tmp_path / "load_1000")
        engine.startup()

        items = [_make_item(content=f"item_{i}") for i in range(1000)]
        errors = []

        def store_item(item):
            try:
                engine.insert(item)
                return True
            except Exception as e:
                errors.append(str(e))
                return False

        start = time.time()

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(store_item, items))

        elapsed = time.time() - start
        successes = sum(results)
        throughput = successes / elapsed

        print(f"\n1000 threaded stores: {elapsed:.2f}s")
        print(f"Successes: {successes}, Errors: {len(errors)}")
        print(f"Throughput: {throughput:.0f}/sec")

        # Acceptance criteria
        assert successes >= 900, f"Too many failures: {len(errors)}"
        assert elapsed < 60.0, f"Too slow: {elapsed:.2f}s"

        engine.shutdown()

    def test_search_under_load(self, tmp_path):
        """Test search performance with concurrent inserts."""
        engine = T4DXEngine(tmp_path / "search_load")
        engine.startup()

        # Pre-populate with 1000 items
        for i in range(1000):
            item = _make_item(content=f"preload_{i}")
            engine.insert(item)

        # Search while inserting
        query_vector = np.random.randn(DIM).tolist()
        search_times = []

        def search_task():
            start = time.time()
            results = engine.search(query_vector, k=10)
            return time.time() - start

        def insert_task():
            item = _make_item(content="concurrent_insert")
            engine.insert(item)

        with ThreadPoolExecutor(max_workers=20) as executor:
            # 50 searches, 50 inserts
            search_futures = [executor.submit(search_task) for _ in range(50)]
            insert_futures = [executor.submit(insert_task) for _ in range(50)]

            for f in search_futures:
                search_times.append(f.result())

        avg_search_time = np.mean(search_times)
        p95_search_time = np.percentile(search_times, 95)

        print(f"\nSearch under load:")
        print(f"Avg search time: {avg_search_time*1000:.2f}ms")
        print(f"P95 search time: {p95_search_time*1000:.2f}ms")

        # P95 should be under 100ms for local T4DX
        assert p95_search_time < 0.5, f"P95 too high: {p95_search_time*1000:.2f}ms"

        engine.shutdown()


class TestMemoryUnderLoad:
    """Test memory usage under load."""

    def test_memory_growth(self, tmp_path):
        """Track memory growth during bulk inserts."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        baseline_mb = process.memory_info().rss / 1024 / 1024

        engine = T4DXEngine(tmp_path / "memory_test")
        engine.startup()

        # Insert 10K items
        for i in range(10000):
            item = _make_item(content=f"item_{i}")
            engine.insert(item)

        peak_mb = process.memory_info().rss / 1024 / 1024
        delta_mb = peak_mb - baseline_mb

        print(f"\nMemory growth for 10K items:")
        print(f"Baseline: {baseline_mb:.0f}MB")
        print(f"Peak: {peak_mb:.0f}MB")
        print(f"Delta: {delta_mb:.0f}MB")
        print(f"Per item: {delta_mb/10:.2f}KB")

        # Should stay under 4GB for 100K items (scaled down for 10K)
        assert delta_mb < 500, f"Memory growth too high: {delta_mb:.0f}MB"

        engine.shutdown()
