"""Tests for T4DM simple API."""

from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest


class TestT4DMSimple:
    """Test the ultra-simple T4DM API with mocked HTTP client."""

    @pytest.fixture
    def mock_client(self):
        with patch("t4dm.sdk.client.WorldWeaverClient") as MockClient:
            instance = MockClient.return_value
            instance.connect = MagicMock()
            instance.close = MagicMock()
            yield instance

    @pytest.fixture
    def t4dm(self, mock_client):
        from t4dm.sdk.simple import T4DM
        return T4DM(url="http://test:8765", user_id="test-user")

    def test_init_connects(self, mock_client):
        from t4dm.sdk.simple import T4DM
        T4DM(url="http://test:8765")
        mock_client.connect.assert_called_once()

    def test_add_returns_id(self, t4dm, mock_client):
        ep = MagicMock()
        ep.id = "abc-123"
        mock_client.create_episode.return_value = ep

        result = t4dm.add("hello world")
        assert result == "abc-123"
        mock_client.create_episode.assert_called_once()

    def test_add_with_tags(self, t4dm, mock_client):
        ep = MagicMock()
        ep.id = "abc-123"
        mock_client.create_episode.return_value = ep

        t4dm.add("hello", tags=["python", "decorators"])
        call_kwargs = mock_client.create_episode.call_args
        assert call_kwargs.kwargs.get("project") == "python,decorators"

    def test_add_with_importance(self, t4dm, mock_client):
        ep = MagicMock()
        ep.id = "abc-123"
        mock_client.create_episode.return_value = ep

        t4dm.add("hello", importance=0.9)
        call_kwargs = mock_client.create_episode.call_args
        assert call_kwargs.kwargs.get("emotional_valence") == 0.9

    def test_search_returns_list(self, t4dm, mock_client):
        ep = MagicMock()
        ep.id = "abc-123"
        ep.content = "hello world"
        ep.timestamp = datetime(2025, 1, 1)

        recall = MagicMock()
        recall.episodes = [ep]
        recall.scores = [0.95]
        mock_client.recall_episodes.return_value = recall

        results = t4dm.search("hello")
        assert len(results) == 1
        assert results[0]["id"] == "abc-123"
        assert results[0]["score"] == 0.95

    def test_search_with_k(self, t4dm, mock_client):
        recall = MagicMock()
        recall.episodes = []
        recall.scores = []
        mock_client.recall_episodes.return_value = recall

        t4dm.search("hello", k=3)
        mock_client.recall_episodes.assert_called_with(query="hello", limit=3)

    def test_get(self, t4dm, mock_client):
        mock_client._request.return_value = {
            "id": "abc-123",
            "content": "hello",
            "timestamp": "2025-01-01T00:00:00",
            "outcome": "neutral",
            "emotional_valence": 0.5,
        }

        result = t4dm.get("abc-123")
        assert result["id"] == "abc-123"
        assert result["content"] == "hello"

    def test_get_all(self, t4dm, mock_client):
        mock_client._request.return_value = {
            "episodes": [
                {
                    "id": "abc-123",
                    "content": "hello",
                    "timestamp": "2025-01-01T00:00:00",
                    "outcome": "neutral",
                    "emotional_valence": 0.5,
                }
            ]
        }

        results = t4dm.get_all(limit=50)
        assert len(results) == 1

    def test_delete_success(self, t4dm, mock_client):
        mock_client._request.return_value = {}
        assert t4dm.delete("abc-123") is True

    def test_delete_failure(self, t4dm, mock_client):
        mock_client._request.side_effect = Exception("not found")
        assert t4dm.delete("abc-123") is False

    def test_forget(self, t4dm, mock_client):
        ep = MagicMock()
        ep.id = "abc-123"
        ep.content = "hello"
        ep.timestamp = datetime(2025, 1, 1)

        recall = MagicMock()
        recall.episodes = [ep]
        recall.scores = [0.9]
        mock_client.recall_episodes.return_value = recall
        mock_client._request.return_value = {}

        count = t4dm.forget("hello")
        assert count == 1

    def test_context_manager(self):
        with patch("t4dm.sdk.client.WorldWeaverClient") as MockClient:
            instance = MockClient.return_value
            instance.connect = MagicMock()
            instance.close = MagicMock()
            from t4dm.sdk.simple import T4DM
            with T4DM(url="http://test:8765") as m:
                pass
            instance.close.assert_called()
