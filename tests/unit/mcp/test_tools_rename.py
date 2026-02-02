"""Tests verifying MCP tool names have been renamed from ww_* to t4dm_*."""

import pytest

from t4dm.mcp.tools import MEMORY_TOOLS


class TestToolNamesRenamed:
    """Verify all MCP tool names start with t4dm_ prefix."""

    def test_all_tools_have_t4dm_prefix(self):
        for tool in MEMORY_TOOLS:
            name = tool["name"]
            assert name.startswith("t4dm_"), f"Tool '{name}' does not start with 't4dm_'"

    def test_no_ww_prefix_tools(self):
        for tool in MEMORY_TOOLS:
            name = tool["name"]
            assert not name.startswith("ww_"), f"Tool '{name}' still has legacy 'ww_' prefix"

    def test_expected_tools_exist(self):
        names = {t["name"] for t in MEMORY_TOOLS}
        expected = {
            "t4dm_store",
            "t4dm_search",
            "t4dm_learn",
            "t4dm_consolidate",
            "t4dm_context",
            "t4dm_entity",
            "t4dm_skill",
        }
        assert expected.issubset(names), f"Missing tools: {expected - names}"

    def test_tool_count(self):
        assert len(MEMORY_TOOLS) >= 7
