"""
Tests for WorldWeaverMCPServer (Phase 10).

Tests the MCP server implementation for Claude Code/Desktop integration.
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from t4dm.mcp.server import (
    WorldWeaverMCPServer,
    create_mcp_server,
    handle_request,
)
from t4dm.mcp.tools import MEMORY_PROMPTS, MEMORY_TOOLS, ToolResult
from t4dm.sdk.agent_client import ScoredMemory
from t4dm.sdk.models import Episode, EpisodeContext


@pytest.fixture
def mock_episode():
    """Create a mock episode."""
    return Episode(
        id=uuid4(),
        session_id="test-session",
        content="Test memory about Python patterns",
        timestamp=datetime.now(),
        outcome="success",
        emotional_valence=0.7,
        context=EpisodeContext(),
        access_count=1,
        stability=0.5,
    )


@pytest.fixture
def mcp_server():
    """Create an MCP server instance."""
    return WorldWeaverMCPServer(
        api_url="http://localhost:8765",
        session_id="test-session",
    )


class TestMCPServerInit:
    """Tests for MCP server initialization."""

    def test_default_init(self):
        """Test default initialization."""
        server = WorldWeaverMCPServer()
        assert server.api_url == "http://localhost:8765"
        assert server.session_id.startswith("mcp-")
        assert server.api_key is None
        assert server._initialized is False

    def test_custom_init(self):
        """Test custom initialization."""
        server = WorldWeaverMCPServer(
            api_url="http://custom:9000",
            session_id="my-session",
            api_key="test-key",
        )
        assert server.api_url == "http://custom:9000"
        assert server.session_id == "my-session"
        assert server.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_initialize_creates_client(self, mcp_server):
        """Test initialization creates memory client."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            assert mcp_server._initialized is True
            MockClient.assert_called_once()
            mock_instance.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mcp_server):
        """Test initialization is idempotent."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await mcp_server.initialize()
            await mcp_server.initialize()

            # Should only create client once
            assert MockClient.call_count == 1

    @pytest.mark.asyncio
    async def test_shutdown(self, mcp_server):
        """Test shutdown cleans up resources."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await mcp_server.initialize()
            await mcp_server.shutdown()

            assert mcp_server._initialized is False
            assert mcp_server._memory is None
            mock_instance.close.assert_called_once()


class TestMCPProtocolHandlers:
    """Tests for MCP protocol handlers."""

    @pytest.mark.asyncio
    async def test_handle_initialize(self, mcp_server):
        """Test initialize handler."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            result = await mcp_server.handle_initialize({})

            assert result["protocolVersion"] == "2024-11-05"
            assert "tools" in result["capabilities"]
            assert "resources" in result["capabilities"]
            assert "prompts" in result["capabilities"]
            assert result["serverInfo"]["name"] == "world-weaver"

    @pytest.mark.asyncio
    async def test_handle_list_tools(self, mcp_server):
        """Test list tools handler."""
        result = await mcp_server.handle_list_tools()

        assert "tools" in result
        assert len(result["tools"]) == len(MEMORY_TOOLS)

    @pytest.mark.asyncio
    async def test_handle_list_resources(self, mcp_server):
        """Test list resources handler."""
        result = await mcp_server.handle_list_resources()

        assert "resources" in result
        assert len(result["resources"]) == 1
        assert mcp_server.session_id in result["resources"][0]["uri"]

    @pytest.mark.asyncio
    async def test_handle_read_resource_session(self, mcp_server):
        """Test read session resource."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get_stats.return_value = {"session_id": "test"}
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_read_resource(
                f"ww://sessions/{mcp_server.session_id}"
            )

            assert "contents" in result
            assert len(result["contents"]) == 1

    @pytest.mark.asyncio
    async def test_handle_list_prompts(self, mcp_server):
        """Test list prompts handler."""
        result = await mcp_server.handle_list_prompts()

        assert "prompts" in result
        assert len(result["prompts"]) == len(MEMORY_PROMPTS)

    @pytest.mark.asyncio
    async def test_handle_get_prompt(self, mcp_server):
        """Test get prompt handler."""
        result = await mcp_server.handle_get_prompt(
            "session_start",
            {"project": "test-project"},
        )

        assert "messages" in result
        if result["messages"]:
            assert result["messages"][0]["role"] == "user"


class TestMCPToolHandlers:
    """Tests for MCP tool handlers."""

    @pytest.mark.asyncio
    async def test_handle_call_tool_unknown(self, mcp_server):
        """Test calling unknown tool returns error."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            result = await mcp_server.handle_call_tool("unknown_tool", {})

            assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handle_store(self, mcp_server, mock_episode):
        """Test ww_store tool."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.store_experience.return_value = mock_episode
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_call_tool(
                "ww_store",
                {"content": "Test content", "outcome": "success"},
            )

            assert "isError" not in result  # Only present on errors
            assert "content" in result
            mock_instance.store_experience.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_store_missing_content(self, mcp_server):
        """Test ww_store requires content."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_call_tool("ww_store", {})

            assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handle_recall(self, mcp_server, mock_episode):
        """Test ww_recall tool."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.retrieve_for_task.return_value = [
                ScoredMemory(episode=mock_episode, similarity_score=0.9)
            ]
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_call_tool(
                "ww_recall",
                {"query": "Python patterns"},
            )

            assert "isError" not in result  # Only present on errors
            content = result["content"][0]["text"]
            assert "task_id" in content

    @pytest.mark.asyncio
    async def test_handle_recall_missing_query(self, mcp_server):
        """Test ww_recall requires query."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_call_tool("ww_recall", {})

            assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handle_learn_outcome(self, mcp_server):
        """Test ww_learn_outcome tool."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.report_task_outcome.return_value = MagicMock(
                credited=3,
                reconsolidated=["mem-1", "mem-2"],
                total_lr_applied=0.03,
            )
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_call_tool(
                "ww_learn_outcome",
                {"task_id": "task-123", "success": True},
            )

            assert "isError" not in result  # Only present on errors

    @pytest.mark.asyncio
    async def test_handle_learn_outcome_missing_task_id(self, mcp_server):
        """Test ww_learn_outcome requires task_id."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_call_tool(
                "ww_learn_outcome",
                {"success": True},
            )

            assert result["isError"] is True

    @pytest.mark.asyncio
    async def test_handle_consolidate(self, mcp_server):
        """Test ww_consolidate tool."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.trigger_consolidation.return_value = {"status": "complete"}
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_call_tool(
                "ww_consolidate",
                {"mode": "deep"},
            )

            assert "isError" not in result  # Only present on errors
            mock_instance.trigger_consolidation.assert_called_with(mode="deep")

    @pytest.mark.asyncio
    async def test_handle_get_context(self, mcp_server, mock_episode):
        """Test ww_get_context tool."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.get_stats.return_value = {"total_retrievals": 5}
            mock_instance.retrieve_for_task.return_value = [
                ScoredMemory(episode=mock_episode, similarity_score=0.8)
            ]
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            result = await mcp_server.handle_call_tool(
                "ww_get_context",
                {"include_stats": True, "include_recent": True},
            )

            assert "isError" not in result  # Only present on errors


class TestToolResult:
    """Tests for ToolResult helper class."""

    def test_success_result(self):
        """Test successful tool result."""
        result = ToolResult(
            success=True,
            data={"key": "value"},
        )
        mcp_response = result.to_mcp_response()

        assert "isError" not in mcp_response  # Only present on errors
        assert "content" in mcp_response
        assert len(mcp_response["content"]) == 1
        assert mcp_response["content"][0]["type"] == "text"

    def test_error_result(self):
        """Test error tool result."""
        result = ToolResult(
            success=False,
            error="Something went wrong",
        )
        mcp_response = result.to_mcp_response()

        assert mcp_response["isError"] is True
        assert "content" in mcp_response
        assert "Something went wrong" in mcp_response["content"][0]["text"]


class TestHandleRequest:
    """Tests for request routing."""

    @pytest.mark.asyncio
    async def test_handle_initialize_request(self, mcp_server):
        """Test handling initialize request."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance

            request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {},
            }

            response = await handle_request(mcp_server, request)

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == 1
            assert "result" in response
            assert "error" not in response

    @pytest.mark.asyncio
    async def test_handle_tools_list_request(self, mcp_server):
        """Test handling tools/list request."""
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        response = await handle_request(mcp_server, request)

        assert response["id"] == 2
        assert "tools" in response["result"]

    @pytest.mark.asyncio
    async def test_handle_tools_call_request(self, mcp_server, mock_episode):
        """Test handling tools/call request."""
        with patch("t4dm.mcp.server.AgentMemoryClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.store_experience.return_value = mock_episode
            MockClient.return_value = mock_instance

            await mcp_server.initialize()

            request = {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "ww_store",
                    "arguments": {"content": "Test"},
                },
            }

            response = await handle_request(mcp_server, request)

            assert response["id"] == 3
            assert "result" in response

    @pytest.mark.asyncio
    async def test_handle_unknown_method(self, mcp_server):
        """Test handling unknown method."""
        request = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "unknown/method",
            "params": {},
        }

        response = await handle_request(mcp_server, request)

        assert response["id"] == 4
        assert "error" in response
        assert response["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_handle_notification(self, mcp_server):
        """Test handling notification (no id)."""
        request = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }

        response = await handle_request(mcp_server, request)

        # Notifications return empty response
        assert response == {}


class TestCreateMCPServer:
    """Tests for server creation."""

    def test_create_with_defaults(self):
        """Test creating server with defaults."""
        server = create_mcp_server()
        assert server.api_url == "http://localhost:8765"

    def test_create_with_custom_values(self):
        """Test creating server with custom values."""
        server = create_mcp_server(
            api_url="http://custom:9000",
            session_id="custom-session",
            api_key="my-key",
        )
        assert server.api_url == "http://custom:9000"
        assert server.session_id == "custom-session"
        assert server.api_key == "my-key"

    def test_create_from_env(self, monkeypatch):
        """Test creating server from environment variables."""
        monkeypatch.setenv("T4DM_API_URL", "http://env:8000")
        monkeypatch.setenv("T4DM_SESSION_ID", "env-session")
        monkeypatch.setenv("T4DM_API_KEY", "env-key")

        server = create_mcp_server()

        assert server.api_url == "http://env:8000"
        assert server.session_id == "env-session"
        assert server.api_key == "env-key"
