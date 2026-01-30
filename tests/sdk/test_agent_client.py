"""
Tests for AgentMemoryClient (Phase 10).

Tests the outcome-based learning integration with Claude Agent SDK.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from ww.sdk.agent_client import (
    AgentMemoryClient,
    CreditAssignmentResult,
    RetrievalContext,
    ScoredMemory,
    create_agent_memory_client,
)
from ww.sdk.models import Episode, EpisodeContext


@pytest.fixture
def mock_episode():
    """Create a mock episode."""
    return Episode(
        id=uuid4(),
        session_id="test-session",
        content="Test episode content about Python decorators",
        timestamp=datetime.now(),
        outcome="success",
        emotional_valence=0.7,
        context=EpisodeContext(project="test-project"),
        access_count=1,
        stability=0.5,
    )


@pytest.fixture
def mock_recall_result(mock_episode):
    """Create a mock recall result."""
    from ww.sdk.models import RecallResult
    return RecallResult(
        query="Python decorators",
        episodes=[mock_episode],
        scores=[0.85],
    )


class TestAgentMemoryClientInit:
    """Tests for AgentMemoryClient initialization."""

    def test_default_init(self):
        """Test default initialization."""
        client = AgentMemoryClient()
        assert client.base_url == "http://localhost:8765"
        assert client.session_id.startswith("agent-")
        assert client.timeout == 30.0
        assert client.base_learning_rate == 0.01

    def test_custom_init(self):
        """Test custom initialization."""
        client = AgentMemoryClient(
            base_url="http://custom:9000",
            session_id="my-session",
            api_key="test-key",
            timeout=60.0,
            base_learning_rate=0.05,
        )
        assert client.base_url == "http://custom:9000"
        assert client.session_id == "my-session"
        assert client.api_key == "test-key"
        assert client.timeout == 60.0
        assert client.base_learning_rate == 0.05


class TestRetrievalContext:
    """Tests for RetrievalContext."""

    def test_creation(self):
        """Test retrieval context creation."""
        ctx = RetrievalContext(
            task_id="task-123",
            query="test query",
            memory_ids=["mem-1", "mem-2"],
            scores=[0.9, 0.8],
        )
        assert ctx.task_id == "task-123"
        assert ctx.query == "test query"
        assert len(ctx.memory_ids) == 2
        assert len(ctx.scores) == 2
        assert ctx.timestamp is not None


class TestScoredMemory:
    """Tests for ScoredMemory."""

    def test_combined_score_calculation(self, mock_episode):
        """Test combined score is weighted average."""
        scored = ScoredMemory(
            episode=mock_episode,
            similarity_score=0.8,
            ff_relevance_score=0.6,
        )
        # 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
        assert scored.combined_score == pytest.approx(0.72, rel=0.01)

    def test_default_ff_score(self, mock_episode):
        """Test default FF score is 0."""
        scored = ScoredMemory(
            episode=mock_episode,
            similarity_score=0.9,
        )
        # 0.6 * 0.9 + 0.4 * 0 = 0.54
        assert scored.combined_score == pytest.approx(0.54, rel=0.01)


class TestCreditAssignmentResult:
    """Tests for CreditAssignmentResult."""

    def test_creation(self):
        """Test credit assignment result creation."""
        result = CreditAssignmentResult(
            credited=5,
            reconsolidated=["mem-1", "mem-2"],
            total_lr_applied=0.03,
        )
        assert result.credited == 5
        assert len(result.reconsolidated) == 2
        assert result.total_lr_applied == 0.03

    def test_default_values(self):
        """Test default values."""
        result = CreditAssignmentResult(credited=0)
        assert result.credited == 0
        assert result.signals == {}
        assert result.reconsolidated == []
        assert result.total_lr_applied == 0.0


class TestAgentMemoryClientRetrieval:
    """Tests for retrieval with eligibility tracking."""

    @pytest.mark.asyncio
    async def test_retrieve_for_task_marks_eligibility(self, mock_recall_result):
        """Test that retrieval marks memories as active in eligibility traces."""
        with patch("ww.sdk.agent_client.AsyncWorldWeaverClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.recall_episodes = AsyncMock(return_value=mock_recall_result)
            MockClient.return_value = mock_client_instance

            client = AgentMemoryClient(session_id="test")
            client._client = mock_client_instance

            memories = await client.retrieve_for_task(
                task_id="task-1",
                query="Python decorators",
                limit=5,
            )

            assert len(memories) == 1
            assert "task-1" in client._active_retrievals
            assert client._total_retrievals == 1

    @pytest.mark.asyncio
    async def test_retrieve_stores_context(self, mock_recall_result):
        """Test that retrieval stores context for credit assignment."""
        with patch("ww.sdk.agent_client.AsyncWorldWeaverClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.recall_episodes = AsyncMock(return_value=mock_recall_result)
            MockClient.return_value = mock_client_instance

            client = AgentMemoryClient(session_id="test")
            client._client = mock_client_instance

            await client.retrieve_for_task(
                task_id="task-2",
                query="test query",
            )

            ctx = client._active_retrievals.get("task-2")
            assert ctx is not None
            assert ctx.query == "test query"
            assert len(ctx.memory_ids) == 1


class TestAgentMemoryClientOutcomes:
    """Tests for outcome reporting and credit assignment."""

    @pytest.mark.asyncio
    async def test_report_outcome_success(self, mock_recall_result):
        """Test successful outcome reporting."""
        with patch("ww.sdk.agent_client.AsyncWorldWeaverClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.recall_episodes = AsyncMock(return_value=mock_recall_result)
            mock_client_instance._request = AsyncMock(return_value={})
            MockClient.return_value = mock_client_instance

            client = AgentMemoryClient(session_id="test")
            client._client = mock_client_instance

            # First retrieve
            await client.retrieve_for_task(task_id="task-3", query="test")

            # Then report outcome
            result = await client.report_task_outcome(
                task_id="task-3",
                success=True,
            )

            assert result.credited >= 0
            assert "task-3" not in client._active_retrievals
            assert client._total_outcomes == 1
            assert client._successful_tasks == 1

    @pytest.mark.asyncio
    async def test_report_outcome_failure(self, mock_recall_result):
        """Test failure outcome reporting."""
        with patch("ww.sdk.agent_client.AsyncWorldWeaverClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.recall_episodes = AsyncMock(return_value=mock_recall_result)
            mock_client_instance._request = AsyncMock(return_value={})
            MockClient.return_value = mock_client_instance

            client = AgentMemoryClient(session_id="test")
            client._client = mock_client_instance

            await client.retrieve_for_task(task_id="task-4", query="test")
            result = await client.report_task_outcome(
                task_id="task-4",
                success=False,
            )

            assert client._successful_tasks == 0
            assert client._total_outcomes == 1

    @pytest.mark.asyncio
    async def test_report_outcome_partial_credit(self, mock_recall_result):
        """Test partial credit reporting."""
        with patch("ww.sdk.agent_client.AsyncWorldWeaverClient") as MockClient:
            mock_client_instance = AsyncMock()
            mock_client_instance.recall_episodes = AsyncMock(return_value=mock_recall_result)
            mock_client_instance._request = AsyncMock(return_value={})
            MockClient.return_value = mock_client_instance

            client = AgentMemoryClient(session_id="test")
            client._client = mock_client_instance

            await client.retrieve_for_task(task_id="task-5", query="test")
            result = await client.report_task_outcome(
                task_id="task-5",
                partial_credit=0.7,
            )

            assert client._total_outcomes == 1

    @pytest.mark.asyncio
    async def test_report_outcome_unknown_task(self):
        """Test reporting outcome for unknown task."""
        client = AgentMemoryClient(session_id="test")
        client._client = AsyncMock()

        result = await client.report_task_outcome(
            task_id="unknown-task",
            success=True,
        )

        assert result.credited == 0


class TestAgentMemoryClientStats:
    """Tests for statistics."""

    def test_get_stats_initial(self):
        """Test initial statistics."""
        client = AgentMemoryClient(session_id="test-session")
        stats = client.get_stats()

        assert stats["session_id"] == "test-session"
        assert stats["total_retrievals"] == 0
        assert stats["total_outcomes"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_pending_tasks_empty(self):
        """Test pending tasks when empty."""
        client = AgentMemoryClient()
        assert client.get_pending_tasks() == []


class TestFFRetrievalScorerIntegration:
    """Tests for FFRetrievalScorer integration (TODO fix)."""

    def test_ff_scorer_parameter(self):
        """Test that FFRetrievalScorer can be passed to client."""
        from ww.bridges.ff_retrieval_scorer import FFRetrievalScorer, FFRetrievalConfig
        from ww.nca.forward_forward import ForwardForwardLayer, ForwardForwardConfig

        ff_config = ForwardForwardConfig(input_dim=64, hidden_dim=32)
        ff_layer = ForwardForwardLayer(ff_config)
        scorer = FFRetrievalScorer(ff_layer=ff_layer)

        client = AgentMemoryClient(ff_scorer=scorer)
        assert client._ff_scorer is scorer

    def test_compute_ff_relevance_with_scorer(self, mock_episode):
        """Test FF relevance computation with actual scorer."""
        import numpy as np
        from ww.bridges.ff_retrieval_scorer import FFRetrievalScorer, FFRetrievalConfig
        from ww.nca.forward_forward import ForwardForwardLayer, ForwardForwardConfig

        ff_config = ForwardForwardConfig(input_dim=64, hidden_dim=32)
        ff_layer = ForwardForwardLayer(ff_config)
        scorer = FFRetrievalScorer(ff_layer=ff_layer)

        client = AgentMemoryClient(ff_scorer=scorer)

        query_emb = np.random.randn(64).astype(np.float32)
        episode_emb = np.random.randn(64).astype(np.float32)

        score = client._compute_ff_relevance(
            query="test",
            episode=mock_episode,
            query_embedding=query_emb,
            episode_embedding=episode_emb,
        )

        # Score should be between 0 and 1
        assert 0.0 <= score <= 1.0

    def test_compute_ff_relevance_fallback_no_embeddings(self, mock_episode):
        """Test FF relevance falls back when no embeddings provided."""
        from ww.bridges.ff_retrieval_scorer import FFRetrievalScorer
        from ww.nca.forward_forward import ForwardForwardLayer, ForwardForwardConfig

        ff_config = ForwardForwardConfig(input_dim=64, hidden_dim=32)
        ff_layer = ForwardForwardLayer(ff_config)
        scorer = FFRetrievalScorer(ff_layer=ff_layer)

        client = AgentMemoryClient(ff_scorer=scorer)

        # No embeddings provided - should use text heuristic
        score = client._compute_ff_relevance(
            query="Python decorators",
            episode=mock_episode,  # content: "Test episode content about Python decorators"
        )

        # Should have positive overlap due to "Python decorators" in both
        assert score > 0.0

    def test_compute_ff_relevance_fallback_no_scorer(self, mock_episode):
        """Test FF relevance uses text heuristic when no scorer."""
        import numpy as np

        client = AgentMemoryClient()  # No scorer

        # Even with embeddings, should use text heuristic
        score = client._compute_ff_relevance(
            query="Python",
            episode=mock_episode,
        )

        # Should work without scorer
        assert 0.0 <= score <= 1.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_agent_memory_client(self):
        """Test client creation convenience function."""
        client = create_agent_memory_client(
            base_url="http://test:8000",
            session_id="my-session",
        )
        assert isinstance(client, AgentMemoryClient)
        assert client.base_url == "http://test:8000"
        assert client.session_id == "my-session"

    def test_create_agent_memory_client_with_ff_scorer(self):
        """Test client creation with FF scorer."""
        from ww.bridges.ff_retrieval_scorer import FFRetrievalScorer
        from ww.nca.forward_forward import ForwardForwardLayer, ForwardForwardConfig

        ff_config = ForwardForwardConfig(input_dim=128, hidden_dim=64)
        ff_layer = ForwardForwardLayer(ff_config)
        scorer = FFRetrievalScorer(ff_layer=ff_layer)

        client = create_agent_memory_client(
            ff_scorer=scorer,
        )
        assert client._ff_scorer is scorer
