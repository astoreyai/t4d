"""Search performance tests at various scales."""

from __future__ import annotations

import time
import uuid

import numpy as np
import pytest

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import ItemRecord


def _make_item(dim: int = 32) -> ItemRecord:
    now = time.time()
    return ItemRecord(
        id=uuid.uuid4().bytes,
        vector=np.random.randn(dim).tolist(),
        kappa=np.random.uniform(0.0, 1.0),
        importance=0.5,
        event_time=now,
        record_time=now,
        valid_from=now,
        valid_until=None,
        item_type="episodic",
        content="scale test",
        access_count=0,
        session_id="perf",
    )


@pytest.fixture
def engine(tmp_path):
    eng = T4DXEngine(tmp_path / "perf_data", flush_threshold=100000)
    eng.startup()
    yield eng
    if eng._started:
        eng.shutdown()


@pytest.mark.parametrize("n_items", [100, 1000, 5000])
def test_search_returns_correct_topk(engine, n_items):
    """Search returns correct number of results at various scales."""
    np.random.seed(42)
    items = [_make_item() for _ in range(n_items)]
    for item in items:
        engine.insert(item)

    query = np.random.randn(32).tolist()
    k = min(10, n_items)
    results = engine.search(query, k=k)

    assert len(results) == k
    # Results should be (id_bytes, score) tuples; verify they are ordered
    scores = [s for _, s in results]
    # Scores may be ascending (distance) or descending (similarity)
    is_ascending = scores == sorted(scores)
    is_descending = scores == sorted(scores, reverse=True)
    assert is_ascending or is_descending, "Results should be consistently ordered by score"


@pytest.mark.parametrize("n_items", [100, 1000, 5000])
def test_search_latency_under_one_second(engine, n_items):
    """Search should complete in under 1 second for up to 5000 items."""
    np.random.seed(123)
    items = [_make_item() for _ in range(n_items)]
    for item in items:
        engine.insert(item)

    query = np.random.randn(32).tolist()

    start = time.perf_counter()
    for _ in range(10):
        engine.search(query, k=10)
    elapsed = (time.perf_counter() - start) / 10

    assert elapsed < 1.0, f"Average search took {elapsed:.3f}s for {n_items} items"
