"""
Agent Management API Routes.

Provides REST API endpoints for managing World Weaver agents:
- Create/configure agents
- Execute with memory context
- Report task outcomes
- Manage agent sessions

These endpoints integrate with the Claude Agent SDK pattern.
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from t4dm.sdk.agent import AgentConfig, AgentPhase, WWAgent
from t4dm.sdk.agent_client import AgentMemoryClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# In-memory agent registry (would be persistent in production)
_agents: dict[str, WWAgent] = {}
_sessions: dict[str, dict[str, Any]] = {}


# =========================================================================
# Request/Response Models
# =========================================================================


class AgentCreateRequest(BaseModel):
    """Request to create a new agent."""

    name: str = Field(..., description="Agent name")
    model: str = Field(default="claude-sonnet-4-5-20250929", description="Model to use")
    system_prompt: str | None = Field(default=None, description="Custom system prompt")
    memory_enabled: bool = Field(default=True, description="Enable memory integration")
    consolidation_interval: int = Field(default=10, description="Messages between consolidations")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "code-assistant",
                "model": "claude-sonnet-4-5-20250929",
                "memory_enabled": True,
                "consolidation_interval": 10,
            }
        }


class AgentInfo(BaseModel):
    """Agent information response."""

    id: str
    name: str
    model: str
    memory_enabled: bool
    session_id: str | None
    phase: str
    message_count: int
    created_at: str


class ExecuteRequest(BaseModel):
    """Request to execute agent task."""

    messages: list[dict[str, Any]] = Field(..., description="Conversation messages")
    task_id: str | None = Field(default=None, description="Optional task identifier")
    include_memory_context: bool = Field(default=True, description="Include retrieved memories")
    session_id: str | None = Field(default=None, description="Session ID to use/resume")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "How do I fix authentication bugs?"}
                ],
                "include_memory_context": True,
            }
        }


class ExecuteResponse(BaseModel):
    """Response from agent execution."""

    task_id: str
    session_id: str
    message_count: int
    memory_context_used: bool
    memories_retrieved: int
    status: str
    response: dict[str, Any] | None = None


class OutcomeRequest(BaseModel):
    """Request to report task outcome."""

    task_id: str = Field(..., description="Task identifier from execute")
    success: bool | None = Field(default=None, description="Task success/failure")
    partial_credit: float | None = Field(default=None, ge=0, le=1, description="Partial success score")
    feedback: str | None = Field(default=None, description="Optional feedback")

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "task-abc123",
                "success": True,
                "feedback": "Solution worked perfectly",
            }
        }


class OutcomeResponse(BaseModel):
    """Response from outcome reporting."""

    task_id: str
    memories_credited: int
    memories_updated: list[str]
    total_learning_rate: float
    message: str


class SessionInfo(BaseModel):
    """Session information."""

    session_id: str
    agent_id: str
    phase: str
    message_count: int
    start_time: str
    pending_outcomes: int
    memory_stats: dict[str, Any] | None = None


class ConsolidateRequest(BaseModel):
    """Request to trigger consolidation."""

    mode: str = Field(default="light", description="Consolidation mode")
    session_id: str | None = Field(default=None, description="Session to consolidate")


class ConsolidateResponse(BaseModel):
    """Response from consolidation."""

    mode: str
    session_id: str
    result: dict[str, Any]


# =========================================================================
# Agent Management Endpoints
# =========================================================================


@router.post("", response_model=AgentInfo)
async def create_agent(request: AgentCreateRequest):
    """
    Create a new agent with memory integration.

    The agent will be initialized but not started.
    Use POST /agents/{agent_id}/sessions to start a session.
    """
    agent_id = f"{request.name}-{uuid4().hex[:8]}"

    config = AgentConfig(
        name=request.name,
        model=request.model,
        system_prompt=request.system_prompt,
        memory_enabled=request.memory_enabled,
        consolidation_interval=request.consolidation_interval,
    )

    agent = WWAgent(config=config)
    _agents[agent_id] = agent

    logger.info(f"Agent created: {agent_id}")

    return AgentInfo(
        id=agent_id,
        name=config.name,
        model=config.model,
        memory_enabled=config.memory_enabled,
        session_id=None,
        phase=AgentPhase.IDLE.value,
        message_count=0,
        created_at=datetime.now().isoformat(),
    )


@router.get("", response_model=list[AgentInfo])
async def list_agents():
    """List all registered agents."""
    agents = []
    for agent_id, agent in _agents.items():
        stats = agent.get_stats()
        agents.append(AgentInfo(
            id=agent_id,
            name=agent.config.name,
            model=agent.config.model,
            memory_enabled=agent.config.memory_enabled,
            session_id=stats.get("session_id"),
            phase=stats.get("phase", "idle"),
            message_count=stats.get("message_count", 0),
            created_at=_sessions.get(agent_id, {}).get("created_at", datetime.now().isoformat()),
        ))
    return agents


@router.get("/{agent_id}", response_model=AgentInfo)
async def get_agent(agent_id: str):
    """Get agent information."""
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    stats = agent.get_stats()
    return AgentInfo(
        id=agent_id,
        name=agent.config.name,
        model=agent.config.model,
        memory_enabled=agent.config.memory_enabled,
        session_id=stats.get("session_id"),
        phase=stats.get("phase", "idle"),
        message_count=stats.get("message_count", 0),
        created_at=_sessions.get(agent_id, {}).get("created_at", datetime.now().isoformat()),
    )


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """Delete an agent."""
    agent = _agents.pop(agent_id, None)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    # End session if active
    if agent._context:
        await agent.end_session()

    logger.info(f"Agent deleted: {agent_id}")
    return {"status": "deleted", "agent_id": agent_id}


# =========================================================================
# Session Management Endpoints
# =========================================================================


@router.post("/{agent_id}/sessions", response_model=SessionInfo)
async def start_session(agent_id: str, session_id: str | None = None):
    """
    Start a new session for an agent.

    This initializes the memory client and sets encoding mode.
    """
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    if agent._context:
        raise HTTPException(
            status_code=400,
            detail=f"Agent already has active session: {agent._context.session_id}",
        )

    await agent.start_session(session_id=session_id)

    _sessions[agent_id] = {
        "created_at": datetime.now().isoformat(),
        "session_id": agent._context.session_id,
    }

    stats = agent.get_stats()

    return SessionInfo(
        session_id=agent._context.session_id,
        agent_id=agent_id,
        phase=stats.get("phase", "idle"),
        message_count=0,
        start_time=datetime.now().isoformat(),
        pending_outcomes=0,
        memory_stats=stats.get("memory"),
    )


@router.delete("/{agent_id}/sessions/{session_id}")
async def end_session(agent_id: str, session_id: str):
    """
    End an agent session.

    Triggers consolidation and closes memory client.
    """
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    if not agent._context or agent._context.session_id != session_id:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    await agent.end_session()
    _sessions.pop(agent_id, None)

    return {"status": "ended", "session_id": session_id}


@router.get("/{agent_id}/sessions/{session_id}", response_model=SessionInfo)
async def get_session(agent_id: str, session_id: str):
    """Get session information."""
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    if not agent._context or agent._context.session_id != session_id:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    stats = agent.get_stats()

    return SessionInfo(
        session_id=session_id,
        agent_id=agent_id,
        phase=stats.get("phase", "idle"),
        message_count=stats.get("message_count", 0),
        start_time=_sessions.get(agent_id, {}).get("created_at", ""),
        pending_outcomes=stats.get("pending_outcomes", 0),
        memory_stats=stats.get("memory"),
    )


# =========================================================================
# Execution Endpoints
# =========================================================================


@router.post("/{agent_id}/execute", response_model=ExecuteResponse)
async def execute_agent(agent_id: str, request: ExecuteRequest):
    """
    Execute agent with memory-augmented context.

    This:
    1. Retrieves relevant memories for the query
    2. Injects memory context into system prompt
    3. Executes with Claude (placeholder in current impl)
    4. Tracks task for outcome reporting
    """
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    # Auto-start session if needed
    if not agent._context:
        await agent.start_session(session_id=request.session_id)
        _sessions[agent_id] = {
            "created_at": datetime.now().isoformat(),
            "session_id": agent._context.session_id,
        }

    # Execute
    result = await agent.execute(
        messages=request.messages,
        task_id=request.task_id,
        include_memory_context=request.include_memory_context,
    )

    memories_retrieved = len(agent._context.retrieved_memories) if agent._context else 0

    return ExecuteResponse(
        task_id=result.get("task_id", ""),
        session_id=result.get("session_id", ""),
        message_count=result.get("message_count", 0),
        memory_context_used=result.get("memory_context_used", False),
        memories_retrieved=memories_retrieved,
        status=result.get("status", "completed"),
        response=result,
    )


@router.post("/{agent_id}/outcome", response_model=OutcomeResponse)
async def report_outcome(agent_id: str, request: OutcomeRequest):
    """
    Report task outcome for learning.

    This triggers credit assignment to memories that were retrieved
    for this task, enabling the agent to learn from experience.
    """
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    if not agent._context:
        raise HTTPException(status_code=400, detail="No active session")

    result = await agent.report_outcome(
        task_id=request.task_id,
        success=request.success,
        partial_credit=request.partial_credit,
        feedback=request.feedback,
    )

    return OutcomeResponse(
        task_id=request.task_id,
        memories_credited=result.get("credited", 0),
        memories_updated=result.get("reconsolidated", []),
        total_learning_rate=result.get("total_lr", 0.0),
        message=f"Credit assigned to {result.get('credited', 0)} memories",
    )


# =========================================================================
# Consolidation Endpoints
# =========================================================================


@router.post("/{agent_id}/consolidate", response_model=ConsolidateResponse)
async def consolidate(agent_id: str, request: ConsolidateRequest):
    """
    Trigger memory consolidation for an agent.

    Modes:
    - light: Quick NREM-like replay (use between tasks)
    - deep: Full consolidation (use at session end)
    - full: Complete consolidation with REM (use overnight)
    """
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    if not agent._memory:
        raise HTTPException(status_code=400, detail="Agent has no memory client")

    result = await agent._memory.trigger_consolidation(mode=request.mode)

    session_id = agent._context.session_id if agent._context else "none"

    return ConsolidateResponse(
        mode=request.mode,
        session_id=session_id,
        result=result,
    )


# =========================================================================
# Memory Access Endpoints
# =========================================================================


@router.post("/{agent_id}/memory/store")
async def store_memory(
    agent_id: str,
    content: str,
    outcome: str = "neutral",
    importance: float = 0.5,
    project: str | None = None,
):
    """
    Store a memory directly for an agent.

    This bypasses the normal task flow and stores directly.
    """
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    episode = await agent.store_memory(
        content=content,
        outcome=outcome,
        importance=importance,
        project=project,
    )

    if not episode:
        raise HTTPException(status_code=400, detail="Memory storage disabled or failed")

    return {
        "id": str(episode.id),
        "content": content[:100],
        "outcome": outcome,
        "importance": importance,
    }


@router.post("/{agent_id}/memory/recall")
async def recall_memories(
    agent_id: str,
    query: str,
    limit: int = Query(default=5, ge=1, le=20),
):
    """
    Recall memories directly for an agent.

    This bypasses the normal task flow and retrieves directly.
    """
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    episodes = await agent.recall_memories(query=query, limit=limit)

    return {
        "query": query,
        "count": len(episodes),
        "memories": [
            {
                "id": str(ep.id),
                "content": ep.content,
                "outcome": ep.outcome,
                "timestamp": ep.timestamp.isoformat(),
            }
            for ep in episodes
        ],
    }


@router.get("/{agent_id}/stats")
async def get_agent_stats(agent_id: str):
    """Get detailed agent statistics."""
    agent = _agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent not found: {agent_id}")

    return agent.get_stats()
