"""Tests for T4DX GraphStore adapter."""

import uuid

import pytest

from t4dm.storage.t4dx.engine import T4DXEngine
from t4dm.storage.t4dx.graph_adapter import T4DXGraphStore


@pytest.fixture
def store(tmp_path):
    engine = T4DXEngine(tmp_path / "data", flush_threshold=100)
    engine.startup()
    gs = T4DXGraphStore(engine)
    yield gs
    engine.shutdown()


class TestT4DXGraphStore:
    @pytest.mark.asyncio
    async def test_create_and_get_node(self, store):
        uid = str(uuid.uuid4())
        nid = await store.create_node("episodic", {"id": uid, "content": "hello"})
        assert nid == uid

        node = await store.get_node(uid, "episodic")
        assert node is not None
        assert node["content"] == "hello"

    @pytest.mark.asyncio
    async def test_update_node(self, store):
        uid = str(uuid.uuid4())
        await store.create_node("episodic", {"id": uid, "content": "v1"})
        await store.update_node(uid, {"kappa": 0.9})
        node = await store.get_node(uid)
        assert node is not None

    @pytest.mark.asyncio
    async def test_delete_node(self, store):
        uid = str(uuid.uuid4())
        await store.create_node("episodic", {"id": uid, "content": "bye"})
        await store.delete_node(uid)
        assert await store.get_node(uid) is None

    @pytest.mark.asyncio
    async def test_create_and_get_relationships(self, store):
        u1 = str(uuid.uuid4())
        u2 = str(uuid.uuid4())
        await store.create_node("episodic", {"id": u1, "content": "a"})
        await store.create_node("episodic", {"id": u2, "content": "b"})

        await store.create_relationship(u1, u2, "USES", {"weight": 0.7})
        rels = await store.get_relationships(u1, direction="out")
        assert len(rels) == 1
        assert rels[0]["type"] == "USES"

    @pytest.mark.asyncio
    async def test_update_relationship(self, store):
        u1 = str(uuid.uuid4())
        u2 = str(uuid.uuid4())
        await store.create_node("episodic", {"id": u1, "content": "a"})
        await store.create_node("episodic", {"id": u2, "content": "b"})
        await store.create_relationship(u1, u2, "USES", {"weight": 0.5})

        await store.update_relationship(u1, u2, "USES", {"weight": 0.9})
        rels = await store.get_relationships(u1, direction="out")
        assert rels[0]["weight"] == pytest.approx(0.9, abs=0.05)

    @pytest.mark.asyncio
    async def test_query_raises(self, store):
        with pytest.raises(NotImplementedError):
            await store.query({"match": "all"})

    @pytest.mark.asyncio
    async def test_find_path(self, store):
        u1 = str(uuid.uuid4())
        u2 = str(uuid.uuid4())
        u3 = str(uuid.uuid4())
        await store.create_node("episodic", {"id": u1, "content": "a"})
        await store.create_node("episodic", {"id": u2, "content": "b"})
        await store.create_node("episodic", {"id": u3, "content": "c"})
        await store.create_relationship(u1, u2, "USES", {})
        await store.create_relationship(u2, u3, "USES", {})

        path = await store.find_path(u1, u3, max_depth=3)
        assert path is not None
        assert path[0] == u1
        assert path[-1] == u3

    @pytest.mark.asyncio
    async def test_find_path_no_route(self, store):
        u1 = str(uuid.uuid4())
        u2 = str(uuid.uuid4())
        await store.create_node("episodic", {"id": u1, "content": "a"})
        await store.create_node("episodic", {"id": u2, "content": "b"})
        path = await store.find_path(u1, u2)
        assert path is None
