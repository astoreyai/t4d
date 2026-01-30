"""Memory services for World Weaver tripartite memory system."""

from ww.memory.episodic import EpisodicMemory
from ww.memory.forgetting import (
    ActiveForgettingSystem,
    ForgettingCandidate,
    ForgettingResult,
    ForgettingStrategy,
    RetentionPolicy,
    get_forgetting_system,
    reset_forgetting_system,
)
from ww.memory.pattern_separation import (
    DentateGyrus,
    # Modern Hopfield (P3.2)
    HopfieldConfig,
    HopfieldMode,
    HopfieldResult,
    PatternCompletion,
    SeparationResult,
    attention_entropy,
    benchmark_hopfield_capacity,
    create_dentate_gyrus,
    create_pattern_completion,
    hopfield_energy,
    modern_hopfield_update,
    sparse_hopfield_update,
)
from ww.memory.procedural import ProceduralMemory
from ww.memory.semantic import SemanticMemory
from ww.memory.working_memory import (
    AttentionalBlink,
    EvictionEvent,
    ItemState,
    WorkingMemory,
    WorkingMemoryItem,
    create_working_memory,
)

__all__ = [
    # PO-2: Active Forgetting
    "ActiveForgettingSystem",
    "ForgettingCandidate",
    "ForgettingResult",
    "ForgettingStrategy",
    "RetentionPolicy",
    "get_forgetting_system",
    "reset_forgetting_system",
    # Memory types
    "AttentionalBlink",
    "DentateGyrus",
    "EpisodicMemory",
    "EvictionEvent",
    "ItemState",
    "PatternCompletion",
    "ProceduralMemory",
    "SemanticMemory",
    "SeparationResult",
    "WorkingMemory",
    "WorkingMemoryItem",
    "create_dentate_gyrus",
    "create_working_memory",
    # Modern Hopfield (P3.2)
    "HopfieldConfig",
    "HopfieldMode",
    "HopfieldResult",
    "modern_hopfield_update",
    "sparse_hopfield_update",
    "hopfield_energy",
    "attention_entropy",
    "create_pattern_completion",
    "benchmark_hopfield_capacity",
]
