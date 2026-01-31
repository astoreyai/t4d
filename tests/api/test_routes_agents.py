"""
Tests for Agent API Routes (Phase 10).

Tests the REST API endpoints for agent management.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from t4dm.api.routes.agents import (
    AgentCreateRequest,
    AgentInfo,
    ConsolidateRequest,
    ExecuteRequest,
    ExecuteResponse,
    OutcomeRequest,
    OutcomeResponse,
    SessionInfo,
    _agents,
    _sessions,
    router,
)
from t4dm.sdk.agent import AgentConfig, AgentPhase, WWAgent


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear agent registries before each test."""
    _agents.clear()
    _sessions.clear()
    yield
    _agents.clear()
    _sessions.clear()


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock(spec=WWAgent)
    agent.config = AgentConfig(name="test-agent")
    agent._context = None
    agent._memory = None
    agent.get_stats.return_value = {
        "name": "test-agent",
        "session_id": None,
        "phase": "idle",
        "message_count": 0,
    }
    return agent


class TestAgentCreateRequest:
    """Tests for AgentCreateRequest model."""

    def test_default_values(self):
        """Test default values."""
        req = AgentCreateRequest(name="test")
        assert req.model == "claude-sonnet-4-5-20250929"
        assert req.memory_enabled is True
        assert req.consolidation_interval == 10

    def test_custom_values(self):
        """Test custom values."""
        req = AgentCreateRequest(
            name="custom",
            model="claude-opus-4-5-20251101",
            system_prompt="Be helpful",
            memory_enabled=False,
            consolidation_interval=5,
        )
        assert req.name == "custom"
        assert req.model == "claude-opus-4-5-20251101"
        assert req.system_prompt == "Be helpful"
        assert req.memory_enabled is False


class TestExecuteRequest:
    """Tests for ExecuteRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        req = ExecuteRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert len(req.messages) == 1
        assert req.include_memory_context is True

    def test_optional_fields(self):
        """Test optional fields."""
        req = ExecuteRequest(
            messages=[],
            task_id="task-123",
            include_memory_context=False,
            session_id="session-456",
        )
        assert req.task_id == "task-123"
        assert req.include_memory_context is False
        assert req.session_id == "session-456"


class TestOutcomeRequest:
    """Tests for OutcomeRequest model."""

    def test_required_fields(self):
        """Test required fields."""
        req = OutcomeRequest(task_id="task-123")
        assert req.task_id == "task-123"
        assert req.success is None

    def test_success_outcome(self):
        """Test success outcome."""
        req = OutcomeRequest(
            task_id="task-123",
            success=True,
            feedback="Good job",
        )
        assert req.success is True
        assert req.feedback == "Good job"

    def test_partial_credit(self):
        """Test partial credit."""
        req = OutcomeRequest(
            task_id="task-123",
            partial_credit=0.7,
        )
        assert req.partial_credit == 0.7


class TestAgentInfo:
    """Tests for AgentInfo model."""

    def test_all_fields(self):
        """Test all fields populated."""
        info = AgentInfo(
            id="agent-123",
            name="test",
            model="claude-sonnet-4-5-20250929",
            memory_enabled=True,
            session_id="session-456",
            phase="executing",
            message_count=5,
            created_at="2024-01-01T00:00:00",
        )
        assert info.id == "agent-123"
        assert info.phase == "executing"


class TestSessionInfo:
    """Tests for SessionInfo model."""

    def test_all_fields(self):
        """Test all fields populated."""
        info = SessionInfo(
            session_id="session-123",
            agent_id="agent-456",
            phase="encoding",
            message_count=10,
            start_time="2024-01-01T00:00:00",
            pending_outcomes=2,
            memory_stats={"total": 100},
        )
        assert info.session_id == "session-123"
        assert info.pending_outcomes == 2


class TestAgentManagementEndpoints:
    """Tests for agent management endpoints."""

    @pytest.mark.asyncio
    async def test_create_agent(self):
        """Test creating an agent."""
        from t4dm.api.routes.agents import create_agent

        request = AgentCreateRequest(
            name="new-agent",
            model="claude-sonnet-4-5-20250929",
            memory_enabled=True,
        )

        result = await create_agent(request)

        assert result.name == "new-agent"
        assert result.model == "claude-sonnet-4-5-20250929"
        assert result.memory_enabled is True
        assert result.phase == "idle"
        assert len(_agents) == 1

    @pytest.mark.asyncio
    async def test_list_agents_empty(self):
        """Test listing agents when empty."""
        from t4dm.api.routes.agents import list_agents

        result = await list_agents()

        assert result == []

    @pytest.mark.asyncio
    async def test_list_agents_with_agents(self, mock_agent):
        """Test listing agents."""
        from t4dm.api.routes.agents import list_agents

        _agents["agent-1"] = mock_agent
        _agents["agent-2"] = mock_agent

        result = await list_agents()

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_agent_found(self, mock_agent):
        """Test getting existing agent."""
        from t4dm.api.routes.agents import get_agent

        _agents["agent-123"] = mock_agent

        result = await get_agent("agent-123")

        assert result.name == "test-agent"

    @pytest.mark.asyncio
    async def test_get_agent_not_found(self):
        """Test getting non-existent agent."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import get_agent

        with pytest.raises(HTTPException) as exc_info:
            await get_agent("unknown")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_agent(self, mock_agent):
        """Test deleting agent."""
        from t4dm.api.routes.agents import delete_agent

        mock_agent._context = None
        _agents["agent-del"] = mock_agent

        result = await delete_agent("agent-del")

        assert result["status"] == "deleted"
        assert "agent-del" not in _agents

    @pytest.mark.asyncio
    async def test_delete_agent_with_session(self, mock_agent):
        """Test deleting agent with active session."""
        from t4dm.api.routes.agents import delete_agent

        mock_agent._context = MagicMock()
        mock_agent._context.session_id = "session-123"
        mock_agent.end_session = AsyncMock()
        _agents["agent-del"] = mock_agent

        result = await delete_agent("agent-del")

        mock_agent.end_session.assert_called_once()
        assert result["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_agent_not_found(self):
        """Test deleting non-existent agent."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import delete_agent

        with pytest.raises(HTTPException) as exc_info:
            await delete_agent("unknown")

        assert exc_info.value.status_code == 404


class TestSessionManagementEndpoints:
    """Tests for session management endpoints."""

    @pytest.mark.asyncio
    async def test_start_session(self, mock_agent):
        """Test starting a session."""
        from t4dm.api.routes.agents import start_session

        context = MagicMock()
        context.session_id = "new-session"

        async def mock_start(session_id=None):
            mock_agent._context = context

        mock_agent.start_session = mock_start
        _agents["agent-1"] = mock_agent

        result = await start_session("agent-1")

        assert result.session_id == "new-session"
        assert result.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_start_session_agent_not_found(self):
        """Test starting session for non-existent agent."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import start_session

        with pytest.raises(HTTPException) as exc_info:
            await start_session("unknown")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_start_session_already_active(self, mock_agent):
        """Test starting session when one is already active."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import start_session

        mock_agent._context = MagicMock()
        mock_agent._context.session_id = "existing"
        _agents["agent-1"] = mock_agent

        with pytest.raises(HTTPException) as exc_info:
            await start_session("agent-1")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_end_session(self, mock_agent):
        """Test ending a session."""
        from t4dm.api.routes.agents import end_session

        mock_agent._context = MagicMock()
        mock_agent._context.session_id = "session-123"
        mock_agent.end_session = AsyncMock()
        _agents["agent-1"] = mock_agent

        result = await end_session("agent-1", "session-123")

        assert result["status"] == "ended"
        mock_agent.end_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_session_wrong_id(self, mock_agent):
        """Test ending wrong session."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import end_session

        mock_agent._context = MagicMock()
        mock_agent._context.session_id = "session-123"
        _agents["agent-1"] = mock_agent

        with pytest.raises(HTTPException) as exc_info:
            await end_session("agent-1", "wrong-session")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_session(self, mock_agent):
        """Test getting session info."""
        from t4dm.api.routes.agents import get_session

        mock_agent._context = MagicMock()
        mock_agent._context.session_id = "session-123"
        _agents["agent-1"] = mock_agent
        _sessions["agent-1"] = {"created_at": datetime.now().isoformat()}

        result = await get_session("agent-1", "session-123")

        assert result.session_id == "session-123"


class TestExecutionEndpoints:
    """Tests for execution endpoints."""

    @pytest.mark.asyncio
    async def test_execute_agent(self, mock_agent):
        """Test executing agent task."""
        from t4dm.api.routes.agents import execute_agent

        mock_agent._context = MagicMock()
        mock_agent._context.session_id = "session-123"
        mock_agent._context.retrieved_memories = []
        mock_agent.execute = AsyncMock(return_value={
            "task_id": "task-123",
            "session_id": "session-123",
            "message_count": 1,
            "memory_context_used": True,
            "status": "completed",
        })
        _agents["agent-1"] = mock_agent

        request = ExecuteRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )

        result = await execute_agent("agent-1", request)

        assert result.task_id == "task-123"
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_execute_agent_auto_start_session(self, mock_agent):
        """Test execution auto-starts session."""
        from t4dm.api.routes.agents import execute_agent

        context = MagicMock()
        context.session_id = "auto-session"
        context.retrieved_memories = []

        async def mock_start(session_id=None):
            mock_agent._context = context

        mock_agent._context = None
        mock_agent.start_session = mock_start
        mock_agent.execute = AsyncMock(return_value={
            "task_id": "task-123",
            "session_id": "auto-session",
            "message_count": 1,
        })
        _agents["agent-1"] = mock_agent

        request = ExecuteRequest(
            messages=[{"role": "user", "content": "Hello"}],
        )

        result = await execute_agent("agent-1", request)

        assert result.session_id == "auto-session"

    @pytest.mark.asyncio
    async def test_execute_agent_not_found(self):
        """Test execution for non-existent agent."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import execute_agent

        request = ExecuteRequest(messages=[])

        with pytest.raises(HTTPException) as exc_info:
            await execute_agent("unknown", request)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_report_outcome(self, mock_agent):
        """Test reporting task outcome."""
        from t4dm.api.routes.agents import report_outcome

        mock_agent._context = MagicMock()
        mock_agent.report_outcome = AsyncMock(return_value={
            "credited": 3,
            "reconsolidated": ["mem-1", "mem-2"],
            "total_lr": 0.03,
        })
        _agents["agent-1"] = mock_agent

        request = OutcomeRequest(
            task_id="task-123",
            success=True,
        )

        result = await report_outcome("agent-1", request)

        assert result.task_id == "task-123"
        assert result.memories_credited == 3

    @pytest.mark.asyncio
    async def test_report_outcome_no_session(self, mock_agent):
        """Test outcome reporting without session."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import report_outcome

        mock_agent._context = None
        _agents["agent-1"] = mock_agent

        request = OutcomeRequest(task_id="task-123")

        with pytest.raises(HTTPException) as exc_info:
            await report_outcome("agent-1", request)

        assert exc_info.value.status_code == 400


class TestConsolidationEndpoints:
    """Tests for consolidation endpoints."""

    @pytest.mark.asyncio
    async def test_consolidate(self, mock_agent):
        """Test triggering consolidation."""
        from t4dm.api.routes.agents import consolidate

        mock_agent._context = MagicMock()
        mock_agent._context.session_id = "session-123"
        mock_agent._memory = MagicMock()
        mock_agent._memory.trigger_consolidation = AsyncMock(return_value={"status": "done"})
        _agents["agent-1"] = mock_agent

        request = ConsolidateRequest(mode="deep")

        result = await consolidate("agent-1", request)

        assert result.mode == "deep"
        mock_agent._memory.trigger_consolidation.assert_called_with(mode="deep")

    @pytest.mark.asyncio
    async def test_consolidate_no_memory(self, mock_agent):
        """Test consolidation without memory client."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import consolidate

        mock_agent._memory = None
        _agents["agent-1"] = mock_agent

        request = ConsolidateRequest()

        with pytest.raises(HTTPException) as exc_info:
            await consolidate("agent-1", request)

        assert exc_info.value.status_code == 400


class TestMemoryAccessEndpoints:
    """Tests for memory access endpoints."""

    @pytest.mark.asyncio
    async def test_store_memory(self, mock_agent):
        """Test storing memory."""
        from t4dm.api.routes.agents import store_memory

        episode = MagicMock()
        episode.id = uuid4()
        mock_agent.store_memory = AsyncMock(return_value=episode)
        _agents["agent-1"] = mock_agent

        result = await store_memory(
            "agent-1",
            content="Important thing",
            outcome="success",
            importance=0.9,
        )

        assert "id" in result
        assert result["content"] == "Important thing"

    @pytest.mark.asyncio
    async def test_store_memory_disabled(self, mock_agent):
        """Test storing memory when disabled."""
        from fastapi import HTTPException

        from t4dm.api.routes.agents import store_memory

        mock_agent.store_memory = AsyncMock(return_value=None)
        _agents["agent-1"] = mock_agent

        with pytest.raises(HTTPException) as exc_info:
            await store_memory("agent-1", content="test")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_recall_memories(self, mock_agent):
        """Test recalling memories."""
        from t4dm.api.routes.agents import recall_memories

        episode = MagicMock()
        episode.id = uuid4()
        episode.content = "Test content"
        episode.outcome = "success"
        episode.timestamp = datetime.now()

        mock_agent.recall_memories = AsyncMock(return_value=[episode])
        _agents["agent-1"] = mock_agent

        result = await recall_memories("agent-1", query="test", limit=5)

        assert result["count"] == 1
        assert len(result["memories"]) == 1

    @pytest.mark.asyncio
    async def test_get_agent_stats(self, mock_agent):
        """Test getting agent stats."""
        from t4dm.api.routes.agents import get_agent_stats

        mock_agent.get_stats.return_value = {
            "name": "test",
            "session_id": "session-123",
            "message_count": 42,
        }
        _agents["agent-1"] = mock_agent

        result = await get_agent_stats("agent-1")

        assert result["name"] == "test"
        assert result["message_count"] == 42
