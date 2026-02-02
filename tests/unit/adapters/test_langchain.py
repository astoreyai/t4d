"""Tests for T4DM LangChain adapter."""

from unittest.mock import MagicMock, patch

import pytest

# Patch the langchain check so tests run without langchain installed
@pytest.fixture(autouse=True)
def _skip_langchain_check():
    with patch("t4dm.adapters.langchain._check_langchain"):
        yield


class TestT4DMMemory:
    """Test LangChain BaseMemory adapter."""

    @pytest.fixture
    def mock_t4dm(self):
        t = MagicMock()
        t.search.return_value = [
            {"id": "1", "content": "Python decorators", "score": 0.9, "timestamp": "2025-01-01"},
            {"id": "2", "content": "Function wrappers", "score": 0.8, "timestamp": "2025-01-02"},
        ]
        t.add.return_value = "new-id"
        return t

    @pytest.fixture
    def memory(self, mock_t4dm):
        from t4dm.adapters.langchain import T4DMMemory
        return T4DMMemory(t4dm=mock_t4dm, memory_key="history", k=3)

    def test_memory_variables(self, memory):
        assert memory.memory_variables == ["history"]

    def test_load_memory_variables(self, memory, mock_t4dm):
        result = memory.load_memory_variables({"input": "decorators"})
        assert "history" in result
        assert "Python decorators" in result["history"]
        mock_t4dm.search.assert_called_once_with("decorators", k=3)

    def test_load_memory_empty_input(self, memory):
        result = memory.load_memory_variables({})
        assert result["history"] == ""

    def test_save_context(self, memory, mock_t4dm):
        memory.save_context(
            {"input": "What are decorators?"},
            {"output": "They modify functions."},
        )
        mock_t4dm.add.assert_called_once()
        call_arg = mock_t4dm.add.call_args[0][0]
        assert "What are decorators?" in call_arg
        assert "They modify functions." in call_arg

    def test_clear_is_noop(self, memory):
        memory.clear()  # Should not raise


class TestT4DMRetriever:
    """Test LangChain BaseRetriever adapter."""

    @pytest.fixture
    def mock_t4dm(self):
        t = MagicMock()
        t.search.return_value = [
            {"id": "1", "content": "Python decorators", "score": 0.9, "timestamp": "2025-01-01"},
        ]
        return t

    @pytest.fixture
    def retriever(self, mock_t4dm):
        from t4dm.adapters.langchain import T4DMRetriever
        return T4DMRetriever(t4dm=mock_t4dm, k=5)

    def test_get_relevant_documents(self, retriever, mock_t4dm):
        docs = retriever._get_relevant_documents("decorators")
        assert len(docs) == 1
        assert docs[0].page_content == "Python decorators"
        assert docs[0].metadata["score"] == 0.9

    def test_empty_results(self, mock_t4dm):
        mock_t4dm.search.return_value = []
        from t4dm.adapters.langchain import T4DMRetriever
        retriever = T4DMRetriever(t4dm=mock_t4dm)
        docs = retriever._get_relevant_documents("nothing")
        assert docs == []
