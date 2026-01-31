"""
Tests for WWAgent (Phase 10).

Tests the Claude Agent SDK integration wrapper.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from t4dm.sdk.agent import AgentConfig, AgentPhase, WWAgent
from t4dm.sdk.agent_client import AgentMemoryClient, ScoredMemory
from t4dm.sdk.models import Episode, EpisodeContext


@pytest.fixture
def agent_config():
    """Create a test agent config."""
    return AgentConfig(
        name="test-agent",
        model="claude-sonnet-4-5-20250929",
        memory_enabled=True,
        consolidation_interval=5,
    )


@pytest.fixture
def mock_memory_client():
    """Create a mock memory client."""
    client = AsyncMock(spec=AgentMemoryClient)
    client.session_id = "test-session"
    client.get_stats.return_value = {
        "session_id": "test-session",
        "total_retrievals": 0,
        "total_outcomes": 0,
    }
    return client


@pytest.fixture
def mock_episode():
    """Create a mock episode."""
    return Episode(
        id=uuid4(),
        session_id="test-session",
        content="Test memory content",
        timestamp=datetime.now(),
        outcome="success",
        emotional_valence=0.7,
        context=EpisodeContext(),
        access_count=1,
        stability=0.5,
    )


class TestAgentConfig:
    """Tests for AgentConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentConfig(name="test")
        assert config.name == "test"
        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.memory_enabled is True
        assert config.consolidation_interval == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentConfig(
            name="custom",
            model="claude-opus-4-5-20251101",
            system_prompt="You are a helpful assistant.",
            memory_enabled=False,
            consolidation_interval=20,
        )
        assert config.name == "custom"
        assert config.model == "claude-opus-4-5-20251101"
        assert config.system_prompt == "You are a helpful assistant."
        assert config.memory_enabled is False
        assert config.consolidation_interval == 20


class TestAgentPhase:
    """Tests for AgentPhase enum."""

    def test_phases_exist(self):
        """Test all phases are defined."""
        assert AgentPhase.IDLE.value == "idle"
        assert AgentPhase.ENCODING.value == "encoding"
        assert AgentPhase.RETRIEVAL.value == "retrieval"
        assert AgentPhase.EXECUTING.value == "executing"
        assert AgentPhase.CONSOLIDATING.value == "consolidating"


class TestWWAgentInit:
    """Tests for WWAgent initialization."""

    def test_init_with_config(self, agent_config):
        """Test initialization with config."""
        agent = WWAgent(config=agent_config)
        assert agent.config == agent_config
        assert agent._memory is None
        assert agent._context is None

    def test_init_with_memory_client(self, agent_config, mock_memory_client):
        """Test initialization with pre-configured memory client."""
        agent = WWAgent(config=agent_config, memory_client=mock_memory_client)
        assert agent._memory == mock_memory_client


class TestWWAgentSessionLifecycle:
    """Tests for session lifecycle management."""

    @pytest.mark.asyncio
    async def test_start_session(self, agent_config):
        """Test starting a session."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            agent = WWAgent(config=agent_config)
            await agent.start_session(session_id="test-123")

            assert agent._context is not None
            assert agent._context.session_id == "test-123"
            assert agent._context.phase == AgentPhase.IDLE
            assert agent._context.message_count == 0

    @pytest.mark.asyncio
    async def test_start_session_generates_id(self, agent_config):
        """Test session ID generation."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            agent = WWAgent(config=agent_config)
            await agent.start_session()

            assert agent._context.session_id.startswith("test-agent-")

    @pytest.mark.asyncio
    async def test_end_session(self, agent_config):
        """Test ending a session."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            with patch.object(AgentMemoryClient, "close", new_callable=AsyncMock):
                with patch.object(AgentMemoryClient, "trigger_consolidation", new_callable=AsyncMock):
                    agent = WWAgent(config=agent_config)
                    await agent.start_session(session_id="test-123")
                    await agent.end_session()

                    assert agent._context is None
                    assert agent._memory is None

    @pytest.mark.asyncio
    async def test_context_manager(self, agent_config):
        """Test async context manager."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            with patch.object(AgentMemoryClient, "close", new_callable=AsyncMock):
                with patch.object(AgentMemoryClient, "trigger_consolidation", new_callable=AsyncMock):
                    async with WWAgent(config=agent_config) as agent:
                        assert agent._context is not None

                    assert agent._context is None


class TestWWAgentExecution:
    """Tests for task execution."""

    @pytest.mark.asyncio
    async def test_execute_without_session(self, agent_config):
        """Test execution without active session raises error."""
        agent = WWAgent(config=agent_config)

        with pytest.raises(RuntimeError, match="Session not started"):
            await agent.execute(
                messages=[{"role": "user", "content": "Hello"}]
            )

    @pytest.mark.asyncio
    async def test_execute_with_memory_context(self, agent_config, mock_episode):
        """Test execution retrieves and injects memory context."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            with patch.object(AgentMemoryClient, "retrieve_for_task", new_callable=AsyncMock) as mock_retrieve:
                mock_retrieve.return_value = [
                    ScoredMemory(
                        episode=mock_episode,
                        similarity_score=0.9,
                    )
                ]

                agent = WWAgent(config=agent_config)
                await agent.start_session()

                result = await agent.execute(
                    messages=[{"role": "user", "content": "How do I fix bugs?"}],
                    include_memory_context=True,
                )

                mock_retrieve.assert_called_once()
                assert result["memory_context_used"] is True

    @pytest.mark.asyncio
    async def test_execute_increments_message_count(self, agent_config):
        """Test that execution increments message count."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            with patch.object(AgentMemoryClient, "retrieve_for_task", new_callable=AsyncMock) as mock_retrieve:
                mock_retrieve.return_value = []

                agent = WWAgent(config=agent_config)
                await agent.start_session()

                await agent.execute(messages=[{"role": "user", "content": "Test"}])
                assert agent._context.message_count == 1

                await agent.execute(messages=[{"role": "user", "content": "Test 2"}])
                assert agent._context.message_count == 2


class TestWWAgentOutcomes:
    """Tests for outcome reporting."""

    @pytest.mark.asyncio
    async def test_report_outcome(self, agent_config):
        """Test reporting task outcome."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            with patch.object(AgentMemoryClient, "retrieve_for_task", new_callable=AsyncMock) as mock_retrieve:
                mock_retrieve.return_value = []
                with patch.object(AgentMemoryClient, "report_task_outcome", new_callable=AsyncMock) as mock_report:
                    mock_report.return_value = MagicMock(
                        credited=1,
                        reconsolidated=["mem-1"],
                        total_lr_applied=0.01,
                    )
                    with patch.object(AgentMemoryClient, "store_experience", new_callable=AsyncMock):
                        agent = WWAgent(config=agent_config)
                        await agent.start_session()

                        # Execute a task
                        result = await agent.execute(
                            messages=[{"role": "user", "content": "Test task"}],
                            task_id="task-123",
                        )

                        # Report outcome
                        outcome = await agent.report_outcome(
                            task_id="task-123",
                            success=True,
                        )

                        assert outcome["credited"] == 1


class TestWWAgentMemoryAccess:
    """Tests for direct memory access."""

    @pytest.mark.asyncio
    async def test_store_memory(self, agent_config, mock_episode):
        """Test storing memory directly."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            with patch.object(AgentMemoryClient, "store_experience", new_callable=AsyncMock) as mock_store:
                mock_store.return_value = mock_episode

                agent = WWAgent(config=agent_config)
                await agent.start_session()

                episode = await agent.store_memory(
                    content="Important learning",
                    outcome="success",
                    importance=0.8,
                )

                assert episode is not None
                mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_recall_memories(self, agent_config, mock_episode):
        """Test recalling memories directly."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            with patch.object(AgentMemoryClient, "retrieve_for_task", new_callable=AsyncMock) as mock_retrieve:
                mock_retrieve.return_value = [
                    ScoredMemory(episode=mock_episode, similarity_score=0.9)
                ]

                agent = WWAgent(config=agent_config)
                await agent.start_session()

                episodes = await agent.recall_memories(
                    query="test query",
                    limit=5,
                )

                assert len(episodes) == 1
                assert episodes[0] == mock_episode


class TestWWAgentStats:
    """Tests for agent statistics."""

    @pytest.mark.asyncio
    async def test_get_stats(self, agent_config):
        """Test getting agent statistics."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            agent = WWAgent(config=agent_config)
            await agent.start_session(session_id="stats-test")

            stats = agent.get_stats()

            assert stats["name"] == "test-agent"
            assert stats["session_id"] == "stats-test"
            assert stats["phase"] == "idle"
            assert stats["message_count"] == 0


class TestWWAgentHooks:
    """Tests for lifecycle hooks."""

    @pytest.mark.asyncio
    async def test_register_hook(self, agent_config):
        """Test registering a lifecycle hook."""
        with patch.object(AgentMemoryClient, "connect", new_callable=AsyncMock):
            hook_called = False

            async def my_hook(**kwargs):
                nonlocal hook_called
                hook_called = True

            agent = WWAgent(config=agent_config)
            agent.on("session_start", my_hook)

            await agent.start_session()

            assert hook_called

    @pytest.mark.asyncio
    async def test_invalid_hook_event(self, agent_config):
        """Test registering hook for invalid event raises error."""
        agent = WWAgent(config=agent_config)

        with pytest.raises(ValueError, match="Unknown event"):
            agent.on("invalid_event", AsyncMock())
