"""Memory consolidation services for T4DM."""

from t4dm.consolidation.lability import (
    DEFAULT_LABILITY_WINDOW_HOURS,
    LabilityConfig,
    LabilityManager,
    LabilityPhase,
    LabilityState,
    compute_reconsolidation_strength,
    get_lability_manager,
    get_reconsolidation_learning_rate,
    is_reconsolidation_eligible,
    reset_lability_manager,
)
from t4dm.consolidation.parallel import (
    ParallelConfig,
    ParallelExecutor,
    ParallelStats,
    cluster_embeddings_hdbscan,
    get_parallel_executor,
    reset_parallel_executor,
)
from t4dm.consolidation.service import (
    HDBSCAN_AVAILABLE,
    # P3.3: Automatic consolidation triggering
    ConsolidationScheduler,
    ConsolidationService,
    ConsolidationTrigger,
    SchedulerState,
    TriggerReason,
    get_consolidation_scheduler,
    get_consolidation_service,
    reset_consolidation_scheduler,
)
from t4dm.consolidation.sleep import (
    AbstractionEvent,
    ReplayDirection,
    ReplayEvent,
    SharpWaveRipple,
    SleepConsolidation,
    SleepCycleResult,
    SleepPhase,
    run_sleep_cycle,
)
from t4dm.consolidation.adaptive_trigger import (
    AdaptiveConsolidationConfig,
    AdaptiveConsolidationTrigger,
    ConsolidationTriggerResult,
)
from t4dm.consolidation.generalization import (
    Cluster,
    GeneralizationQualityScorer,
    GeneralizationResult,
)
from t4dm.consolidation.variational import (
    ClusterAssignment,
    ClusterPrototype,
    VariationalConsolidation,
    VariationalState,
    VariationalStep,
)

__all__ = [
    "HDBSCAN_AVAILABLE",
    "AbstractionEvent",
    "ConsolidationService",
    # Phase 7: Lability window (protein synthesis gate)
    "DEFAULT_LABILITY_WINDOW_HOURS",
    "LabilityConfig",
    "LabilityManager",
    "LabilityPhase",
    "LabilityState",
    "compute_reconsolidation_strength",
    "get_lability_manager",
    "get_reconsolidation_learning_rate",
    "is_reconsolidation_eligible",
    "reset_lability_manager",
    # P3.3: Automatic consolidation triggering
    "ConsolidationScheduler",
    "ConsolidationTrigger",
    "SchedulerState",
    "TriggerReason",
    # W1-04: Adaptive Consolidation Trigger (CLS-based)
    "AdaptiveConsolidationConfig",
    "AdaptiveConsolidationTrigger",
    "ConsolidationTriggerResult",
    # PO-1: Parallel consolidation
    "ParallelConfig",
    "ParallelExecutor",
    "ParallelStats",
    "cluster_embeddings_hdbscan",
    "get_parallel_executor",
    "reset_parallel_executor",
    # Sleep consolidation
    "ReplayDirection",
    "ReplayEvent",
    "SharpWaveRipple",
    "SleepConsolidation",
    "SleepCycleResult",
    "SleepPhase",
    "get_consolidation_scheduler",
    "get_consolidation_service",
    "reset_consolidation_scheduler",
    "run_sleep_cycle",
    # W3-03: Generalization Quality Scoring (O'Reilly)
    "Cluster",
    "GeneralizationQualityScorer",
    "GeneralizationResult",
    # W4-01: Variational Consolidation Framing (Friston)
    "ClusterAssignment",
    "ClusterPrototype",
    "VariationalConsolidation",
    "VariationalState",
    "VariationalStep",
]
