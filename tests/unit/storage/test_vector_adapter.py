"""Tests for T4DX VectorStore adapter."""

import uuid

import numpy as np
import pytest

from tests.unit.storage.conftest import make_item
from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.vector_adapter import T4DXVectorStore


@pytest.fixture
def store(tmp_path):
    engine = T4DXEngine(tmp_path / "data", flush_threshold=100)
    engine.startup()
    vs = T4DXVectorStore(engine)
    yield vs
    engine.shutdown()


class TestT4DXVectorStore:
    @pytest.mark.asyncio
    async def test_create_collection(self, store):
        await store.create_collection("test", 4)
        assert "test" in store._collections

    @pytest.mark.asyncio
    async def test_add_and_get(self, store):
        await store.create_collection("episodic", 4)
        uid = str(uuid.uuid4())
        await store.add(
            "episodic",
            [uid],
            [[1.0, 0.0, 0.0, 0.0]],
            [{"content": "hello", "item_type": "episodic"}],
        )
        results = await store.get("episodic", [uid])
        assert len(results) == 1
        assert results[0][0] == uid

    @pytest.mark.asyncio
    async def test_search(self, store):
        await store.create_collection("episodic", 4)
        uid = str(uuid.uuid4())
        await store.add(
            "episodic",
            [uid],
            [[1.0, 0.0, 0.0, 0.0]],
            [{"content": "hello", "item_type": "episodic"}],
        )
        results = await store.search("episodic", [1.0, 0.0, 0.0, 0.0], limit=1)
        assert len(results) == 1
        assert results[0][0] == uid

    @pytest.mark.asyncio
    async def test_delete(self, store):
        uid = str(uuid.uuid4())
        await store.add(
            "test",
            [uid],
            [[1.0, 0.0, 0.0, 0.0]],
            [{"content": "bye"}],
        )
        await store.delete("test", [uid])
        results = await store.get("test", [uid])
        assert len(results) == 0
