"""Tests for T4DM LlamaIndex adapter."""

from unittest.mock import MagicMock, patch

import pytest

# Patch the llamaindex check so tests run without llama_index installed
@pytest.fixture(autouse=True)
def _skip_llamaindex_check():
    with patch("t4dm.adapters.llamaindex._check_llamaindex"):
        yield


class TestT4DMVectorStore:
    """Test LlamaIndex VectorStore adapter."""

    @pytest.fixture
    def mock_t4dm(self):
        t = MagicMock()
        t.add.return_value = "new-id"
        t.delete.return_value = True
        t.search.return_value = [
            {"id": "1", "content": "Python decorators", "score": 0.9, "timestamp": "2025-01-01"},
        ]
        return t

    @pytest.fixture
    def store(self, mock_t4dm):
        from t4dm.adapters.llamaindex import T4DMVectorStore
        return T4DMVectorStore(t4dm=mock_t4dm)

    def test_add_nodes(self, store, mock_t4dm):
        node = MagicMock()
        node.get_content.return_value = "test content"
        ids = store.add([node])
        assert ids == ["new-id"]
        mock_t4dm.add.assert_called_once_with("test content")

    def test_delete(self, store, mock_t4dm):
        store.delete("doc-123")
        mock_t4dm.delete.assert_called_once_with("doc-123")

    def test_query(self, store, mock_t4dm):
        from t4dm.adapters.llamaindex import VectorStoreQuery
        q = VectorStoreQuery(query_str="decorators", similarity_top_k=3)
        result = store.query(q)
        assert len(result.nodes) == 1
        assert result.similarities == [0.9]
        assert result.ids == ["1"]
        mock_t4dm.search.assert_called_once_with("decorators", k=3)

    def test_query_empty(self, mock_t4dm):
        mock_t4dm.search.return_value = []
        from t4dm.adapters.llamaindex import T4DMVectorStore, VectorStoreQuery
        store = T4DMVectorStore(t4dm=mock_t4dm)
        q = VectorStoreQuery(query_str="nothing", similarity_top_k=5)
        result = store.query(q)
        assert result.nodes == []

    def test_class_name(self, store):
        assert store.class_name() == "T4DMVectorStore"
