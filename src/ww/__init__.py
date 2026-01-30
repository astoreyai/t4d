"""
World Weaver - Biologically-inspired memory for AI.

A framework implementing tripartite neural memory (episodic, semantic,
procedural) with cognitive science foundations.

Quick Start:
    from ww import memory

    # Store content
    await memory.store("User discussed Python decorators")

    # Recall similar memories
    results = await memory.recall("decorators")

    # Use session context
    async with memory.session("my-project") as m:
        await m.store("Project-specific knowledge")
"""

__version__ = "0.5.0"
__author__ = "Aaron Storey"

from ww.core.config import Settings, get_settings
from ww.core.types import (
    Domain,
    Entity,
    EntityType,
    Episode,
    Outcome,
    Procedure,
    RelationType,
)

# Import simplified memory API
from ww.memory_api import (
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
]
