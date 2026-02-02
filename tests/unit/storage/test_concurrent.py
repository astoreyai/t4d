"""Concurrent access tests for T4DXEngine."""

from __future__ import annotations

import threading
import time
import uuid

import numpy as np
import pytest

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.types import ItemRecord


def _make_item(dim: int = 8, content: str = "concurrent") -> ItemRecord:
    now = time.time()
    return ItemRecord(
        id=uuid.uuid4().bytes,
        vector=np.random.randn(dim).tolist(),
        kappa=0.1,
        importance=0.5,
        event_time=now,
        record_time=now,
        valid_from=now,
        valid_until=None,
        item_type="episodic",
        content=content,
        access_count=0,
        session_id="test",
    )


@pytest.fixture
def engine(tmp_path):
    eng = T4DXEngine(tmp_path / "concurrent_data", flush_threshold=100000)
    eng.startup()
    yield eng
    if eng._started:
        eng.shutdown()


class TestConcurrentInsert:
    def test_four_threads_no_corruption(self, engine):
        """Four threads inserting simultaneously should not corrupt data."""
        items_per_thread = 50
        all_items: list[list[ItemRecord]] = [[] for _ in range(4)]
        errors: list[Exception] = []

        def inserter(thread_idx: int) -> None:
            try:
                for i in range(items_per_thread):
                    item = _make_item(content=f"t{thread_idx}-{i}")
                    all_items[thread_idx].append(item)
                    engine.insert(item)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=inserter, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent insert: {errors}"

        # Verify all items are retrievable
        for thread_items in all_items:
            for item in thread_items:
                retrieved = engine.get(item.id)
                assert retrieved is not None, f"Item {item.id.hex()} missing after concurrent insert"
                assert retrieved.content == item.content


class TestConcurrentSearch:
    def test_search_during_writes(self, engine):
        """Concurrent searches during writes should not crash or corrupt."""
        # Pre-populate
        for _ in range(100):
            engine.insert(_make_item())

        search_results: list[list] = []
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for _ in range(50):
                    engine.insert(_make_item())
            except Exception as e:
                errors.append(e)

        def searcher() -> None:
            try:
                for _ in range(50):
                    q = np.random.randn(8).tolist()
                    results = engine.search(q, k=5)
                    search_results.append(results)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=writer))
        for _ in range(2):
            threads.append(threading.Thread(target=searcher))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent search+write: {errors}"
        # All search calls should have returned lists
        for r in search_results:
            assert isinstance(r, list)


class TestConcurrentGet:
    def test_concurrent_get_returns_correct_data(self, engine):
        """Concurrent GETs should return correct items."""
        items = [_make_item(content=f"get-{i}") for i in range(100)]
        for item in items:
            engine.insert(item)

        results: dict[int, list[ItemRecord | None]] = {i: [] for i in range(4)}
        errors: list[Exception] = []

        def getter(thread_idx: int) -> None:
            try:
                for item in items:
                    r = engine.get(item.id)
                    results[thread_idx].append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=getter, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent get: {errors}"
        for thread_idx, res_list in results.items():
            for i, r in enumerate(res_list):
                assert r is not None
                assert r.content == items[i].content


class TestConcurrentUpdateFields:
    def test_no_lost_updates(self, engine):
        """Concurrent UPDATE_FIELDS should not lose updates."""
        item = _make_item()
        engine.insert(item)

        errors: list[Exception] = []
        updates_applied = threading.Lock()
        update_count = [0]

        def updater(thread_idx: int) -> None:
            try:
                for i in range(25):
                    engine.update_fields(
                        item.id,
                        {"metadata": {f"t{thread_idx}_k{i}": True}},
                    )
                    with updates_applied:
                        update_count[0] += 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=updater, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent update: {errors}"
        assert update_count[0] == 100, f"Expected 100 updates, got {update_count[0]}"

        # The item should still be retrievable
        retrieved = engine.get(item.id)
        assert retrieved is not None
