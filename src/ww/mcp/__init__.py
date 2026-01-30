"""
World Weaver MCP Server.

Exposes World Weaver memory system as an MCP server for Claude Code/Desktop.
"""

from ww.mcp.server import create_mcp_server, run_mcp_server
from ww.mcp.tools import MEMORY_TOOLS

__all__ = [
    "MEMORY_TOOLS",
    "create_mcp_server",
    "run_mcp_server",
]
