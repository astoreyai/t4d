"""Tests for Mem0-compatible REST API routes."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from t4dm.api.routes.compat import router, InMemoryBackend, _backend


@pytest.fixture(autouse=True)
def reset_backend():
    """Reset to in-memory backend before each test."""
    import t4dm.api.routes.compat as compat_module
    # Use fresh in-memory backend for tests
    compat_module._backend = InMemoryBackend()
    yield
    # Clear after test
    if hasattr(compat_module._backend, '_memories'):
        compat_module._backend._memories.clear()


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestMemoryCreate:
    def test_create_memory(self, client):
        resp = client.post(
            "/v1/memories/",
            json={"content": "test memory", "user_id": "u1"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["memory"] == "test memory"
        assert data["user_id"] == "u1"
        assert "id" in data
        assert "created_at" in data

    def test_create_with_metadata(self, client):
        resp = client.post(
            "/v1/memories/",
            json={"content": "meta test", "metadata": {"key": "val"}},
        )
        assert resp.status_code == 201
        assert resp.json()["metadata"]["key"] == "val"


class TestMemoryGet:
    def test_get_memory(self, client):
        create = client.post("/v1/memories/", json={"content": "hello"})
        mid = create.json()["id"]
        resp = client.get(f"/v1/memories/{mid}")
        assert resp.status_code == 200
        assert resp.json()["memory"] == "hello"

    def test_get_not_found(self, client):
        resp = client.get("/v1/memories/nonexistent")
        assert resp.status_code == 404


class TestMemoryList:
    def test_list_all(self, client):
        client.post("/v1/memories/", json={"content": "a", "user_id": "u1"})
        client.post("/v1/memories/", json={"content": "b", "user_id": "u2"})
        resp = client.get("/v1/memories/")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_list_filter_user(self, client):
        client.post("/v1/memories/", json={"content": "a", "user_id": "u1"})
        client.post("/v1/memories/", json={"content": "b", "user_id": "u2"})
        resp = client.get("/v1/memories/?user_id=u1")
        assert len(resp.json()) == 1
        assert resp.json()[0]["user_id"] == "u1"


class TestMemorySearch:
    def test_search(self, client):
        client.post("/v1/memories/", json={"content": "python decorators"})
        client.post("/v1/memories/", json={"content": "java generics"})
        resp = client.get("/v1/memories/search/?query=python")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 1
        assert "python" in results[0]["memory"].lower()

    def test_search_no_match(self, client):
        client.post("/v1/memories/", json={"content": "hello"})
        resp = client.get("/v1/memories/search/?query=nonexistent")
        assert resp.json() == []


class TestMemoryUpdate:
    def test_update_content(self, client):
        create = client.post("/v1/memories/", json={"content": "old"})
        mid = create.json()["id"]
        resp = client.put(f"/v1/memories/{mid}", json={"content": "new"})
        assert resp.status_code == 200
        assert resp.json()["memory"] == "new"

    def test_update_not_found(self, client):
        resp = client.put("/v1/memories/bad", json={"content": "x"})
        assert resp.status_code == 404


class TestMemoryDelete:
    def test_delete(self, client):
        create = client.post("/v1/memories/", json={"content": "bye"})
        mid = create.json()["id"]
        resp = client.delete(f"/v1/memories/{mid}")
        assert resp.status_code == 204
        assert client.get(f"/v1/memories/{mid}").status_code == 404

    def test_delete_not_found(self, client):
        resp = client.delete("/v1/memories/bad")
        assert resp.status_code == 404
