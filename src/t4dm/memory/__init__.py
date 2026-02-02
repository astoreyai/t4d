"""Memory services for T4DM tripartite memory system."""

from t4dm.memory.episodic import EpisodicMemory
from t4dm.memory.forgetting import (
    ActiveForgettingSystem,
    ForgettingCandidate,
    ForgettingResult,
    ForgettingStrategy,
    RetentionPolicy,
    get_forgetting_system,
    reset_forgetting_system,
)
from t4dm.memory.pattern_separation import (
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
from t4dm.memory.procedural import ProceduralMemory
from t4dm.memory.semantic import SemanticMemory
from t4dm.memory.working_memory import (
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
