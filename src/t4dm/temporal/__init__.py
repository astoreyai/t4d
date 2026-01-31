"""Temporal dynamics for World Weaver memory system."""

from t4dm.temporal.dynamics import (
    TemporalConfig,
    TemporalDynamics,
    TemporalPhase,
    TemporalState,
    create_temporal_dynamics,
)
from t4dm.temporal.integration import (
    LearnedSalienceProvider,
    PlasticityConfig,
    PlasticityCoordinator,
    SalienceProvider,
    adapt_orchestra_state,
    create_plasticity_coordinator,
    get_consolidation_state,
    get_pattern_separation_state,
    get_sleep_replay_state,
)
from t4dm.temporal.lifecycle import (
    LifecycleConfig,
    LifecyclePhase,
    LifecycleState,
    MemoryLifecycleManager,
    clear_lifecycle_managers,
    get_lifecycle_manager,
)
from t4dm.temporal.session import (
    SessionContext,
    SessionManager,
    get_session_manager,
)

__all__ = [
    # Dynamics
    "TemporalPhase",
    "TemporalState",
    "TemporalConfig",
    "TemporalDynamics",
    "create_temporal_dynamics",
    # Session management
    "SessionContext",
    "SessionManager",
    "get_session_manager",
    # Integration
    "adapt_orchestra_state",
    "SalienceProvider",
    "LearnedSalienceProvider",
    "get_consolidation_state",
    "get_sleep_replay_state",
    "get_pattern_separation_state",
    "PlasticityConfig",
    "PlasticityCoordinator",
    "create_plasticity_coordinator",
    # P7.2: Lifecycle
    "LifecyclePhase",
    "LifecycleState",
    "LifecycleConfig",
    "MemoryLifecycleManager",
    "get_lifecycle_manager",
    "clear_lifecycle_managers",
]
