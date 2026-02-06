"""
T4DM - Biologically-inspired temporal memory for AI.

A framework implementing tripartite neural memory (episodic, semantic,
procedural) with cognitive science foundations, spiking cortical blocks,
and T4DX embedded spatiotemporal storage.

Quick Start:
    from t4dm import memory

    # Store content
    await memory.store("User discussed Python decorators")

    # Recall similar memories
    results = await memory.recall("decorators")

    # Use session context
    async with memory.session("my-project") as m:
        await m.store("Project-specific knowledge")
"""

__version__ = "2.0.0"
__author__ = "Aaron Storey"

from t4dm.core.config import Settings, get_settings
from t4dm.core.types import (
    Domain,
    Entity,
    EntityType,
    Episode,
    Outcome,
    Procedure,
    RelationType,
)

# Import simplified memory API
from t4dm.memory_api import (
    Memory,
    MemoryResult,
    get_recent,
    memory,
    recall,
    recall_entities,
    recall_episodes,
    recall_skills,
    store,
    store_entity,
    store_episode,
    store_skill,
)

# Import lite API for quick prototyping
from t4dm import lite
from t4dm.lite import Memory as LiteMemory

__all__ = [
    # Core types
    "Domain",
    "Entity",
    "EntityType",
    "Episode",
    "Outcome",
    "Procedure",
    "RelationType",
    # Config
    "Settings",
    "get_settings",
    # Simplified memory API
    "Memory",
    "MemoryResult",
    "memory",
    "store",
    "recall",
    "store_episode",
    "store_entity",
    "store_skill",
    "recall_episodes",
    "recall_entities",
    "recall_skills",
    "get_recent",
    # Lite API (quick prototyping)
    "lite",
    "LiteMemory",
]
