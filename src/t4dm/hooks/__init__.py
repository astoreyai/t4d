"""
World Weaver Lifecycle Hooks System.

Provides extensible hook interfaces for all modules with:
- Registration and execution mechanisms
- Execution order guarantees
- Error isolation and handling
- Built-in hooks for observability, caching, auditing
"""

from t4dm.hooks.base import (
    Hook,
    HookContext,
    HookError,
    HookPhase,
    HookPriority,
    HookRegistry,
)
from t4dm.hooks.consolidation import (
    ClusterFormHook,
    ConsolidationHook,
    DuplicateFoundHook,
    EntityExtractedHook,
)
from t4dm.hooks.core import (
    ConfigChangeHook,
    CoreHook,
    HealthCheckHook,
    InitHook,
    ShutdownHook,
)
from t4dm.hooks.memory import (
    AccessHook,
    CreateHook,
    DecayHook,
    MemoryHook,
    RecallHook,
    UpdateHook,
)
from t4dm.hooks.registry import get_global_registry
from t4dm.hooks.session_lifecycle import (
    IdleConsolidationHook,
    SessionContext,
    SessionEndHook,
    SessionStartHook,
    TaskOutcomeHook,
    create_session_hooks,
)
from t4dm.hooks.storage import (
    ConnectionHook,
    ErrorHook,
    QueryHook,
    RetryHook,
    StorageHook,
)

__all__ = [
    # Base
    "Hook",
    "HookRegistry",
    "HookContext",
    "HookPhase",
    "HookPriority",
    "HookError",
    # Core
    "CoreHook",
    "InitHook",
    "ShutdownHook",
    "HealthCheckHook",
    "ConfigChangeHook",
    # Memory
    "MemoryHook",
    "CreateHook",
    "RecallHook",
    "UpdateHook",
    "AccessHook",
    "DecayHook",
    # Storage
    "StorageHook",
    "ConnectionHook",
    "QueryHook",
    "ErrorHook",
    "RetryHook",
    # Consolidation
    "ConsolidationHook",
    "ClusterFormHook",
    "DuplicateFoundHook",
    "EntityExtractedHook",
    # Registry
    "get_global_registry",
    # Session Lifecycle (Claude Agent SDK)
    "SessionStartHook",
    "SessionEndHook",
    "TaskOutcomeHook",
    "IdleConsolidationHook",
    "SessionContext",
    "create_session_hooks",
]
