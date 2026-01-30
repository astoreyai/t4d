"""
World Weaver Python SDK.

Programmatic access to the tripartite memory system via REST API.
Includes Claude Agent SDK integration for learning agents.
"""

from ww.sdk.agent import AgentConfig, AgentPhase, WWAgent
from ww.sdk.agent_client import (
    AgentMemoryClient,
    CreditAssignmentResult,
    RetrievalContext,
    ScoredMemory,
    create_agent_memory_client,
)
from ww.sdk.client import AsyncWorldWeaverClient, WorldWeaverClient
from ww.sdk.models import (
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
