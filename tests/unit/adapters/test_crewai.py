"""Tests for CrewAI adapter."""

from unittest.mock import MagicMock, patch

import pytest


class TestT4DMCrewMemory:
    """Test the CrewAI memory adapter with mocked T4DM."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mocked T4DM backend."""
        mock_t4dm = MagicMock()
        with patch("t4dm.sdk.simple.T4DM", return_value=mock_t4dm):
            from t4dm.adapters.crewai import T4DMCrewMemory
            a = T4DMCrewMemory(url="http://test:8765", api_key="key123")
        # Replace t4dm with mock (in case it was constructed before patch)
        a.t4dm = mock_t4dm
        yield a, mock_t4dm

    def test_save_basic(self, adapter):
        a, mock = adapter
        a.save("hello world")
        mock.add.assert_called_once()
        assert mock.add.call_args.kwargs["content"] == "hello world"

    def test_save_with_metadata_and_agent(self, adapter):
        a, mock = adapter
        a.save("test", metadata={"topic": "ai"}, agent="researcher")
        assert mock.add.call_args.kwargs["metadata"]["agent"] == "researcher"
        assert mock.add.call_args.kwargs["metadata"]["topic"] == "ai"

    def test_search_returns_filtered_results(self, adapter):
        a, mock = adapter
        mock.search.return_value = [
            {"content": "good match", "score": 0.9},
            {"content": "bad match", "score": 0.3},
        ]
        results = a.search("query", limit=5, score_threshold=0.5)
        assert len(results) == 1
        assert results[0]["memory"] == "good match"
        assert results[0]["score"] == 0.9

    def test_search_empty(self, adapter):
        a, mock = adapter
        mock.search.return_value = []
        results = a.search("nothing")
        assert results == []

    def test_reset(self, adapter):
        a, mock = adapter
        a.reset()
        mock.forget.assert_called_once_with("")
