"""Tests for T4DM AutoGen adapter."""

from unittest.mock import MagicMock

import pytest

from t4dm.adapters.autogen import T4DMAutoGenMemory


class TestT4DMAutoGenMemory:
    """Test AutoGen memory adapter."""

    @pytest.fixture
    def mock_t4dm(self):
        t = MagicMock()
        t.search.return_value = [
            {"id": "1", "content": "Python decorators", "score": 0.9, "timestamp": "2025-01-01"},
        ]
        t.add.return_value = "new-id"
        t.close.return_value = None
        return t

    @pytest.fixture
    def memory(self, mock_t4dm):
        return T4DMAutoGenMemory(t4dm=mock_t4dm, k=5)

    def test_query(self, memory, mock_t4dm):
        result = memory.query("decorators")
        assert "Python decorators" in result
        mock_t4dm.search.assert_called_once_with("decorators", k=5)

    def test_query_empty(self, mock_t4dm):
        mock_t4dm.search.return_value = []
        mem = T4DMAutoGenMemory(t4dm=mock_t4dm)
        assert mem.query("nothing") == ""

    def test_query_none_t4dm(self):
        mem = T4DMAutoGenMemory(t4dm=None)
        assert mem.query("anything") == ""

    def test_update_context(self, memory, mock_t4dm):
        ctx = {"input": "decorators", "other": "data"}
        result = memory.update_context(ctx)
        assert "memory" in result
        assert "Python decorators" in result["memory"]
        assert result["memory_ids"] == ["1"]

    def test_update_context_no_query(self, memory):
        ctx = {"unrelated_key": 123}
        result = memory.update_context(ctx)
        assert "memory" not in result

    def test_add(self, memory, mock_t4dm):
        memory.add("new memory", importance=0.8)
        mock_t4dm.add.assert_called_once_with("new memory", importance=0.8)

    def test_add_none_t4dm(self):
        mem = T4DMAutoGenMemory(t4dm=None)
        mem.add("test")  # Should not raise

    def test_clear_is_noop(self, memory):
        memory.clear()  # Should not raise

    def test_close(self, memory, mock_t4dm):
        memory.close()
        mock_t4dm.close.assert_called_once()
