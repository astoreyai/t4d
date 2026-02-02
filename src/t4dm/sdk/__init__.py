"""
T4DM Python SDK.

Programmatic access to the tripartite memory system via REST API.
Includes Claude Agent SDK integration for learning agents.
"""

from t4dm.sdk.agent import AgentConfig, AgentPhase, WWAgent
from t4dm.sdk.agent_client import (
    AgentMemoryClient,
    CreditAssignmentResult,
    RetrievalContext,
    ScoredMemory,
    create_agent_memory_client,
)
from t4dm.sdk.client import AsyncWorldWeaverClient, WorldWeaverClient
from t4dm.sdk.models import (
    ActivationResult,
    Entity,
    Episode,
    RecallResult,
    Skill,
)

__all__ = [
    # Core SDK
    "ActivationResult",
    "AsyncWorldWeaverClient",
    "Entity",
    "Episode",
    "RecallResult",
    "Skill",
    "WorldWeaverClient",
    # Agent SDK integration
    "AgentConfig",
    "AgentMemoryClient",
    "AgentPhase",
    "CreditAssignmentResult",
    "RetrievalContext",
    "ScoredMemory",
    "WWAgent",
    "create_agent_memory_client",
]
