"""
T4DM Automatic Memory MCP Server.

Provides automatic memory for Claude Code/Desktop - no manual tool calls needed.
Memory is injected via resources and captured via lifecycle patterns.

Architecture:
    1. Hot Cache Resource: Auto-loaded context (~10 items, 0ms latency)
    2. Session Context Resource: Project-specific memories
    3. Single Manual Tool: t4dm_remember for explicit "remember this"
    4. Auto-Learning: Observations captured from tool outputs

Configuration via environment:
    T4DM_API_URL: T4DM API URL (default: http://localhost:8765)
    T4DM_SESSION_ID: Session identifier
    T4DM_PROJECT: Current project name
    T4DM_HOT_CACHE_SIZE: Items in hot cache (default: 10)
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.environ.get("T4DM_API_URL", "http://localhost:8765")
SESSION_ID = os.environ.get("T4DM_SESSION_ID", f"claude-{uuid4().hex[:8]}")
PROJECT = os.environ.get("T4DM_PROJECT", "default")
HOT_CACHE_SIZE = int(os.environ.get("T4DM_HOT_CACHE_SIZE", "10"))

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "t4dm-memory",
    instructions="""You have automatic persistent memory via T4DM.

AUTOMATIC BEHAVIORS (no tool calls needed):
- Session context is pre-loaded in the memory://context resource
- Your observations and learnings are automatically captured
- Important patterns and decisions are stored when sessions end

WHEN TO USE t4dm_remember:
- When user explicitly says "remember this" or "save this"
- For critical information the user wants persisted
- Never mention memory tools to the user unprompted

The memory system learns from your outputs automatically.""",
)

# ---------------------------------------------------------------------------
# Lazy client
# ---------------------------------------------------------------------------

_client = None


def _get_client():
    """Get or create the AgentMemoryClient (lazy init)."""
    global _client
    if _client is None:
        from t4dm.sdk.agent_client import AgentMemoryClient

        _client = AgentMemoryClient(
            base_url=API_URL,
            session_id=SESSION_ID,
            api_key=os.environ.get("T4DM_API_KEY"),
        )
        logger.info(f"T4DM client initialized: session={SESSION_ID}, project={PROJECT}")
    return _client


# ---------------------------------------------------------------------------
# Resources (Auto-loaded context)
# ---------------------------------------------------------------------------


@mcp.resource("memory://context")
async def get_session_context() -> str:
    """
    Auto-loaded session context.

    This resource is automatically injected at session start.
    Contains hot cache + project-relevant memories.
    """
    try:
        client = _get_client()
        await client.connect()

        # Get hot cache (frequently used memories)
        hot_cache = await _get_hot_cache()

        # Get project-specific context
        project_context = await _get_project_context()

        # Get recent session memories
        recent = await _get_recent_memories()

        # Format as context block
        sections = []

        if hot_cache:
            sections.append("## Frequently Referenced\n" + "\n".join(
                f"- {m['content'][:200]}" for m in hot_cache
            ))

        if project_context:
            sections.append("## Project Context\n" + "\n".join(
                f"- {m['content'][:200]}" for m in project_context
            ))

        if recent:
            sections.append("## Recent Session\n" + "\n".join(
                f"- [{m['timestamp']}] {m['content'][:150]}" for m in recent
            ))

        if not sections:
            return "No relevant memories for this session."

        return "\n\n".join(sections)

    except Exception as e:
        logger.warning(f"Failed to load context: {e}")
        return "Memory context unavailable."


@mcp.resource("memory://hot-cache")
async def get_hot_cache_resource() -> str:
    """Hot cache of frequently accessed memories (0ms retrieval)."""
    try:
        cache = await _get_hot_cache()
        if not cache:
            return "Hot cache empty."
        return "\n".join(f"- {m['content']}" for m in cache)
    except Exception as e:
        logger.warning(f"Hot cache error: {e}")
        return "Hot cache unavailable."


@mcp.resource("memory://project/{project_name}")
async def get_project_memories(project_name: str) -> str:
    """Project-specific memories."""
    try:
        client = _get_client()
        await client.connect()

        memories = await client.retrieve_for_task(
            task_id=f"project-{project_name}-{uuid4().hex[:6]}",
            query=f"project {project_name} context patterns decisions",
            limit=15,
            project=project_name,
        )

        if not memories:
            return f"No memories for project: {project_name}"

        return "\n".join(
            f"- [{m.episode.outcome}] {m.episode.content}"
            for m in memories
        )
    except Exception as e:
        logger.warning(f"Project memory error: {e}")
        return f"Project memories unavailable: {project_name}"


# ---------------------------------------------------------------------------
# Single Manual Tool (for explicit "remember this")
# ---------------------------------------------------------------------------


@mcp.tool
async def t4dm_remember(
    content: str,
    importance: float = 0.7,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Explicitly store something in memory.

    Use this ONLY when:
    - User says "remember this", "save this", "don't forget"
    - Critical information that must be persisted
    - User explicitly requests memory storage

    Do NOT use for routine observations - those are captured automatically.
    """
    client = _get_client()
    await client.connect()

    episode = await client.store_experience(
        content=content,
        outcome="important",  # Explicit stores are important
        importance=importance,
        project=PROJECT,
    )

    # Add to hot cache immediately (high importance = promoted)
    await _promote_to_hot_cache(str(episode.id), content)

    return {
        "stored": True,
        "id": str(episode.id),
        "message": "Remembered.",
    }


# ---------------------------------------------------------------------------
# Auto-capture Tools (called by hooks, not manually)
# ---------------------------------------------------------------------------


@mcp.tool
async def _t4dm_auto_observe(
    observation: str,
    observation_type: str = "pattern",
    source: str = "tool_output",
) -> dict:
    """
    Internal: Auto-capture observations from Claude's outputs.
    Called by hooks, not manually by Claude.
    """
    client = _get_client()
    await client.connect()

    # Lower importance for auto-captured items
    importance = 0.4 if observation_type == "pattern" else 0.3

    episode = await client.store_experience(
        content=f"[{observation_type}] {observation}",
        outcome="observed",
        importance=importance,
        project=PROJECT,
    )

    return {"captured": True, "id": str(episode.id)}


@mcp.tool
async def _t4dm_session_end(
    summary: str,
    key_decisions: Optional[list[str]] = None,
    learned_patterns: Optional[list[str]] = None,
) -> dict:
    """
    Internal: Called at session end to store session summary.
    Triggered by SessionEnd hook, not manually.
    """
    client = _get_client()
    await client.connect()

    # Store session summary with high importance
    content_parts = [f"Session Summary: {summary}"]

    if key_decisions:
        content_parts.append("Decisions: " + "; ".join(key_decisions))

    if learned_patterns:
        content_parts.append("Patterns: " + "; ".join(learned_patterns))

    episode = await client.store_experience(
        content="\n".join(content_parts),
        outcome="session_end",
        importance=0.8,
        project=PROJECT,
    )

    # Trigger consolidation
    await client.trigger_consolidation(mode="light")

    return {
        "stored": True,
        "consolidated": True,
        "id": str(episode.id),
    }


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


async def _get_hot_cache() -> list[dict]:
    """Get hot cache items (frequently accessed, high importance)."""
    try:
        client = _get_client()
        await client.connect()

        # Query for high-importance recent items
        memories = await client.retrieve_for_task(
            task_id=f"hot-cache-{uuid4().hex[:6]}",
            query="frequently used important patterns decisions",
            limit=HOT_CACHE_SIZE,
            min_similarity=0.3,
        )

        return [
            {
                "id": str(m.episode.id),
                "content": m.episode.content,
                "score": m.combined_score,
            }
            for m in memories
        ]
    except Exception:
        return []


async def _get_project_context() -> list[dict]:
    """Get project-specific context."""
    try:
        client = _get_client()
        await client.connect()

        memories = await client.retrieve_for_task(
            task_id=f"project-ctx-{uuid4().hex[:6]}",
            query=f"project {PROJECT} architecture decisions patterns",
            limit=5,
            project=PROJECT,
        )

        return [
            {
                "id": str(m.episode.id),
                "content": m.episode.content,
            }
            for m in memories
        ]
    except Exception:
        return []


async def _get_recent_memories() -> list[dict]:
    """Get recent session memories."""
    try:
        client = _get_client()
        await client.connect()

        memories = await client.retrieve_for_task(
            task_id=f"recent-{uuid4().hex[:6]}",
            query="recent session activity",
            limit=5,
        )

        return [
            {
                "id": str(m.episode.id),
                "content": m.episode.content,
                "timestamp": m.episode.timestamp.isoformat() if m.episode.timestamp else "unknown",
            }
            for m in memories
        ]
    except Exception:
        return []


async def _promote_to_hot_cache(memory_id: str, content: str):
    """Promote a memory to hot cache tier."""
    # This would update the memory's access count / importance
    # For now, just log - the retrieve_for_task naturally promotes via access patterns
    logger.info(f"Promoted to hot cache: {memory_id[:8]}...")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Entry point for MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    logger.info(f"Starting T4DM MCP Server")
    logger.info(f"  API: {API_URL}")
    logger.info(f"  Session: {SESSION_ID}")
    logger.info(f"  Project: {PROJECT}")
    logger.info(f"  Hot cache size: {HOT_CACHE_SIZE}")

    mcp.run()


if __name__ == "__main__":
    main()
