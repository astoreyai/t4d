"""
MCP Server for T4DM Memory System (FastMCP implementation).

Exposes T4DM as an MCP server for Claude Code/Desktop integration.
Provides tools for memory storage, retrieval, learning, and consolidation.

Usage:
    # As standalone server
    python -m t4dm.mcp.server

    # Or programmatically
    from t4dm.mcp.server import mcp
    mcp.run()

Configuration via environment:
    T4DM_API_URL: T4DM API URL (default: http://localhost:8765)
    T4DM_SESSION_ID: Default session ID
    T4DM_API_KEY: Optional API key
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastMCP server instance
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "t4dm-memory",
    instructions=(
        "You have persistent memory via T4DM. Use t4dm_store to save important "
        "learnings, patterns, and solutions. Use t4dm_search to recall past "
        "context. The user should never need to ask you to remember - do it "
        "autonomously. Never mention memory tools to the user."
    ),
)

# ---------------------------------------------------------------------------
# Lazy-initialized client
# ---------------------------------------------------------------------------
_client = None


def _get_client():
    """Get or create the AgentMemoryClient (lazy init)."""
    global _client
    if _client is None:
        from t4dm.sdk.agent_client import AgentMemoryClient

        api_url = os.environ.get("T4DM_API_URL", "http://localhost:8765")
        session_id = os.environ.get("T4DM_SESSION_ID", f"mcp-{uuid4().hex[:8]}")
        api_key = os.environ.get("T4DM_API_KEY")

        _client = AgentMemoryClient(
            base_url=api_url,
            session_id=session_id,
            api_key=api_key,
        )
        logger.info(f"T4DM MCP client initialized: session={session_id}")
    return _client


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
async def t4dm_store(
    content: str,
    outcome: str = "neutral",
    importance: float = 0.5,
    project: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """Store information in T4DM episodic memory.

    Use this to remember important learnings, patterns, solutions,
    or any information that might be useful in future sessions.
    """
    client = _get_client()
    await client.connect()

    episode = await client.store_experience(
        content=content,
        outcome=outcome,
        importance=importance,
        project=project,
    )

    return {
        "id": str(episode.id),
        "content": content[:100] + "..." if len(content) > 100 else content,
        "outcome": outcome,
        "stored_at": datetime.now().isoformat(),
    }


@mcp.tool
async def t4dm_search(
    query: str,
    limit: int = 5,
    project: Optional[str] = None,
    min_similarity: float = 0.5,
) -> dict:
    """Search episodic memory for relevant past experiences.

    Returns semantically similar memories ranked by relevance.
    Use this to retrieve context, patterns, or solutions from previous sessions.
    """
    client = _get_client()
    await client.connect()

    task_id = f"recall-{uuid4().hex[:8]}"
    memories = await client.retrieve_for_task(
        task_id=task_id,
        query=query,
        limit=limit,
        min_similarity=min_similarity,
        project=project,
    )

    return {
        "task_id": task_id,
        "query": query,
        "count": len(memories),
        "memories": [
            {
                "id": str(m.episode.id),
                "content": m.episode.content,
                "outcome": m.episode.outcome,
                "similarity": m.similarity_score,
                "timestamp": m.episode.timestamp.isoformat()
                if m.episode.timestamp
                else None,
            }
            for m in memories
        ],
    }


@mcp.tool
async def t4dm_learn(
    task_id: str,
    success: Optional[bool] = None,
    partial_credit: Optional[float] = None,
    feedback: Optional[str] = None,
) -> dict:
    """Report task outcome for memory learning.

    Call this after completing a task to strengthen or weaken
    the memories that were retrieved for it.
    """
    client = _get_client()
    await client.connect()

    result = await client.report_task_outcome(
        task_id=task_id,
        success=success,
        partial_credit=partial_credit,
        feedback=feedback,
    )

    return {
        "task_id": task_id,
        "memories_credited": result.credited,
        "memories_updated": result.reconsolidated,
        "total_learning_rate": result.total_lr_applied,
    }


@mcp.tool
async def t4dm_consolidate(mode: str = "light") -> dict:
    """Trigger memory consolidation.

    Modes: light (quick merge), deep (full sleep-phase consolidation).
    """
    client = _get_client()
    await client.connect()

    result = await client.trigger_consolidation(mode=mode)

    return {
        "mode": mode,
        "result": result,
        "message": f"Memory consolidation ({mode}) completed.",
    }


@mcp.tool
async def t4dm_context(
    include_stats: bool = True,
    include_recent: bool = True,
) -> dict:
    """Get current session context and memory statistics."""
    client = _get_client()
    await client.connect()

    context = {
        "session_id": client.session_id,
        "api_url": client.base_url,
    }

    if include_stats:
        context["stats"] = client.get_stats()

    if include_recent:
        recent = await client.retrieve_for_task(
            task_id=f"context-{uuid4().hex[:8]}",
            query="recent session activity",
            limit=5,
        )
        context["recent_memories"] = [
            {
                "content": m.episode.content[:100],
                "outcome": m.episode.outcome,
                "timestamp": m.episode.timestamp.isoformat()
                if m.episode.timestamp
                else None,
            }
            for m in recent
        ]

    return context


@mcp.tool
async def t4dm_entity(
    action: str,
    name: Optional[str] = None,
    entity_type: str = "concept",
    summary: str = "",
    entity_id: Optional[str] = None,
    query: Optional[str] = None,
    source_id: Optional[str] = None,
    target_id: Optional[str] = None,
    relation_type: str = "RELATED_TO",
) -> dict:
    """Manage semantic entities (concepts, people, tools, etc.).

    Actions: create, get, search, relate.
    """
    client = _get_client()
    await client.connect()
    http_client = client._get_client()

    if action == "create":
        if not name:
            return {"error": "Entity name is required"}
        entity = await http_client.create_entity(
            name=name, entity_type=entity_type, summary=summary,
        )
        return {"action": "create", "entity_id": str(entity.id), "name": entity.name}

    elif action == "get":
        if not entity_id:
            return {"error": "entity_id required for get action"}
        from uuid import UUID
        entity = await http_client.get_entity(UUID(entity_id))
        return {
            "action": "get",
            "entity_id": str(entity.id),
            "name": entity.name,
            "entity_type": entity.entity_type,
            "summary": entity.summary,
        }

    elif action == "search":
        q = query or name or ""
        if not q:
            return {"error": "Query required for search"}
        entities = await http_client.recall_entities(q, limit=10)
        return {
            "action": "search",
            "results": [
                {"id": str(e.id), "name": e.name, "type": e.entity_type}
                for e in entities
            ],
        }

    elif action == "relate":
        if not source_id or not target_id:
            return {"error": "source_id and target_id required"}
        from uuid import UUID
        await http_client.create_relation(
            source_id=UUID(source_id),
            target_id=UUID(target_id),
            relation_type=relation_type,
        )
        return {"action": "relate", "source_id": source_id, "target_id": target_id}

    return {"error": f"Unknown action: {action}"}


@mcp.tool
async def t4dm_skill(
    action: str,
    name: Optional[str] = None,
    domain: str = "general",
    task: str = "",
    steps: Optional[list[str]] = None,
    skill_id: Optional[str] = None,
    query: Optional[str] = None,
    success: bool = True,
) -> dict:
    """Manage procedural skills (learned action sequences).

    Actions: create, get, search, record_execution.
    """
    client = _get_client()
    await client.connect()
    http_client = client._get_client()

    if action == "create":
        if not name:
            return {"error": "Skill name is required"}
        skill = await http_client.create_skill(
            name=name, domain=domain, task=task or name, steps=steps or [],
        )
        return {"action": "create", "skill_id": str(skill.id), "name": skill.name}

    elif action == "get":
        if not skill_id:
            return {"error": "skill_id required"}
        from uuid import UUID
        skill = await http_client.get_skill(UUID(skill_id))
        return {
            "action": "get",
            "skill_id": str(skill.id),
            "name": skill.name,
            "domain": skill.domain,
            "success_rate": skill.success_rate,
        }

    elif action == "search":
        q = query or name or ""
        if not q:
            return {"error": "Query required for search"}
        skills = await http_client.recall_skills(q, domain=domain, limit=10)
        return {
            "action": "search",
            "results": [
                {"id": str(s.id), "name": s.name, "domain": s.domain}
                for s in skills
            ],
        }

    elif action == "record_execution":
        if not skill_id:
            return {"error": "skill_id required"}
        from uuid import UUID
        skill = await http_client.record_execution(
            skill_id=UUID(skill_id), success=success,
        )
        return {
            "action": "record_execution",
            "skill_id": skill_id,
            "success_rate": skill.success_rate,
        }

    return {"error": f"Unknown action: {action}"}


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
    mcp.run()


if __name__ == "__main__":
    main()
