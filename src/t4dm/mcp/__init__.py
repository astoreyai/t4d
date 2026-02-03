"""
T4DM MCP Server (FastMCP).

Exposes T4DM memory system as an MCP server for Claude Code/Desktop.
"""

from t4dm.mcp.server import mcp, main

__all__ = [
    "mcp",
    "main",
]
