"""
Tests for T4DM FastMCP server.

Tests the MCP server tool implementations.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from t4dm.mcp.server import mcp, _get_client, t4dm_store, t4dm_search
from t4dm.mcp.tools import MEMORY_TOOLS, ToolResult


class TestMCPServerSetup:
    """Tests for FastMCP server configuration."""

    def test_server_name(self):
        assert mcp.name == "t4dm-memory"

    def test_server_has_instructions(self):
        assert "T4DM" in (mcp.instructions or "")

    def test_tools_registered(self):
        """Verify tools are defined in MEMORY_TOOLS."""
        tool_names = [t["name"] for t in MEMORY_TOOLS]
        assert "t4dm_store" in tool_names
        assert "t4dm_search" in tool_names
        assert "t4dm_learn" in tool_names
        assert "t4dm_consolidate" in tool_names
        assert "t4dm_context" in tool_names


class TestToolResult:
    """Tests for ToolResult helper class."""

    def test_success_result(self):
        result = ToolResult(success=True, data={"key": "value"})
        mcp_response = result.to_mcp_response()
        assert "isError" not in mcp_response
        assert "content" in mcp_response
        assert len(mcp_response["content"]) == 1
        assert mcp_response["content"][0]["type"] == "text"

    def test_error_result(self):
        result = ToolResult(success=False, error="Something went wrong")
        mcp_response = result.to_mcp_response()
        assert mcp_response["isError"] is True
        assert "Something went wrong" in mcp_response["content"][0]["text"]


class TestToolNames:
    """Verify all MCP tool names have t4dm_ prefix."""

    def test_all_tools_have_t4dm_prefix(self):
        for tool in MEMORY_TOOLS:
            assert tool["name"].startswith("t4dm_"), f"Tool '{tool['name']}' missing t4dm_ prefix"

    def test_no_ww_prefix_tools(self):
        for tool in MEMORY_TOOLS:
            assert not tool["name"].startswith("ww_"), f"Tool '{tool['name']}' has legacy ww_ prefix"


class TestClientInit:
    """Tests for lazy client initialization."""

    def test_get_client_default_url(self, monkeypatch):
        """Test client uses default URL."""
        import t4dm.mcp.server as srv
        srv._client = None
        monkeypatch.delenv("T4DM_API_URL", raising=False)
        monkeypatch.delenv("T4DM_SESSION_ID", raising=False)

        with patch("t4dm.sdk.agent_client.AgentMemoryClient", autospec=True) as MockClient:
            client = _get_client()
            MockClient.assert_called_once()
            call_kwargs = MockClient.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:8765"

        srv._client = None  # reset

    def test_get_client_from_env(self, monkeypatch):
        """Test client reads env vars."""
        import t4dm.mcp.server as srv
        srv._client = None
        monkeypatch.setenv("T4DM_API_URL", "http://custom:9000")
        monkeypatch.setenv("T4DM_SESSION_ID", "env-session")
        monkeypatch.setenv("T4DM_API_KEY", "env-key")

        with patch("t4dm.sdk.agent_client.AgentMemoryClient", autospec=True) as MockClient:
            client = _get_client()
            call_kwargs = MockClient.call_args[1]
            assert call_kwargs["base_url"] == "http://custom:9000"
            assert call_kwargs["session_id"] == "env-session"
            assert call_kwargs["api_key"] == "env-key"

        srv._client = None
