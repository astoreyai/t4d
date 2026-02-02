"""
Visualization modules for T4DM neurocomputational system.

This package provides comprehensive visualization tools for analyzing
and understanding the neural dynamics of the memory system:

- Pattern separation and completion (DG/CA3)
- Synaptic plasticity traces (LTP/LTD, homeostatic scaling)
- Neuromodulator state (DA/NE/ACh/5-HT/GABA)
- Memory consolidation dynamics (SWR replay)
- Embedding projections (t-SNE/UMAP)
- Persistence state (WAL, checkpoints, recovery)

Each module provides both static (matplotlib) and interactive (plotly)
visualizations with export capabilities.
"""

from t4dm.visualization.activation_heatmap import (
    ActivationHeatmap,
    plot_activation_heatmap,
    plot_activation_timeline,
)
from t4dm.visualization.capsule_visualizer import (
    CapsuleSnapshot,
    CapsuleVisualizer,
    RoutingEvent,
    create_capsule_dashboard,
    plot_entity_probabilities,
    plot_pose_vectors,
    plot_routing_entropy,
    plot_routing_heatmap,
)
from t4dm.visualization.consolidation_replay import (
    ConsolidationVisualizer,
    plot_replay_priority,
    plot_swr_sequence,
)
from t4dm.visualization.coupling_dynamics import (
    CouplingDynamicsVisualizer,
    CouplingSnapshot,
    LearningEvent,
    create_coupling_dashboard,
    plot_coupling_heatmap,
    plot_ei_balance_timeline,
    plot_eligibility_heatmap,
    plot_spectral_radius_timeline,
)
from t4dm.visualization.coupling_dynamics import (
    plot_eigenvalue_spectrum as plot_coupling_eigenvalue_spectrum,
)
from t4dm.visualization.da_telemetry import (
    DARampEvent,
    DASignalType,
    DASnapshot,
    DAStatistics,
    DATelemetry,
    create_da_dashboard,
    plot_da_timeline,
    plot_rpe_distribution,
)
from t4dm.visualization.embedding_projections import (
    EmbeddingProjector,
    plot_tsne_projection,
    plot_umap_projection,
)
from t4dm.visualization.energy_landscape import (
    EnergyLandscapeVisualizer,
    EnergySnapshot,
    TrajectoryPoint,
    create_energy_dashboard,
    plot_basin_occupancy,
    plot_energy_contour,
    plot_energy_timeline,
)
from t4dm.visualization.ff_visualizer import (
    FFSnapshot,
    FFTrainingEvent,
    ForwardForwardVisualizer,
    create_ff_dashboard,
    plot_goodness_bars,
    plot_goodness_timeline,
    plot_margin_evolution,
    plot_phase_comparison,
)
from t4dm.visualization.glymphatic_visualizer import (
    ClearanceEvent,
    GlymphaticSnapshot,
    GlymphaticVisualizer,
    SleepStage,
    create_glymphatic_dashboard,
    plot_clearance_and_waste,
    plot_clearance_by_stage,
    plot_sleep_stage_timeline,
    plot_stage_pie,
)
from t4dm.visualization.neuromodulator_state import (
    NeuromodulatorDashboard,
    plot_neuromodulator_radar,
    plot_neuromodulator_traces,
)
from t4dm.visualization.nt_state_dashboard import (
    HOMEOSTATIC_SETPOINTS,
    NT_COLORS,
    NT_LABELS,
    RECEPTOR_KM,
    NTSnapshot,
    NTStateDashboard,
    NTStatistics,
    create_nt_dashboard,
    plot_autocorrelation,
    plot_correlation_matrix,
    plot_deviation_heatmap,
    plot_nt_channels,
    plot_opponent_processes,
    plot_saturation_curves,
)
from t4dm.visualization.pac_telemetry import (
    PACSnapshot,
    PACTelemetry,
)
from t4dm.visualization.pattern_separation import (
    PatternSeparationVisualizer,
    plot_separation_comparison,
    plot_sparsity_distribution,
)
from t4dm.visualization.neuromod_layers import (
    NT_COLORS as NEUROMOD_NT_COLORS,
    NT_LAYER_MAPPING,
    NT_NAMES as NEUROMOD_NT_NAMES,
    NeuromodLayerSnapshot,
    NeuromodLayerVisualizer,
)
from t4dm.visualization.oscillator_injection import (
    OSC_COLORS,
    OSC_NAMES,
    OscillatorInjectionVisualizer,
    OscillatorSnapshot,
)
from t4dm.visualization.persistence_state import (
    CheckpointInfo,
    PersistenceMetrics,
    PersistenceVisualizer,
    T4DXPersistenceInfo,
    WALSegmentInfo,
    plot_checkpoint_history,
    plot_durability_dashboard,
    plot_t4dx_state,
    plot_wal_timeline,
)
from t4dm.visualization.plasticity_traces import (
    PlasticityTracer,
    plot_bcm_curve,
    plot_ltp_ltd_distribution,
    plot_weight_changes,
)
from t4dm.visualization.stability_monitor import (
    BifurcationEvent,
    StabilityMonitor,
    StabilitySnapshot,
    StabilityType,
    create_stability_dashboard,
    plot_bifurcation_diagram,
    plot_eigenvalue_evolution,
    plot_eigenvalue_spectrum,
    plot_lyapunov_timeline,
    plot_oscillation_metrics,
    plot_stability_timeline,
)
from t4dm.visualization.kappa_gradient import (
    KappaGradientVisualizer,
    KappaSnapshot,
    KAPPA_BANDS,
    KAPPA_COLORS,
)
from t4dm.visualization.qwen_metrics import (
    QwenMetricsVisualizer,
    QwenSnapshot,
)
from t4dm.visualization.spiking_dynamics import (
    SpikingDynamicsVisualizer,
    SpikingSnapshot,
)
from t4dm.visualization.swr_telemetry import (
    SWRTelemetry,
    SWRTelemetryEvent,
)
from t4dm.visualization.t4dx_metrics import (
    CompactionEvent,
    CompactionType,
    T4DXMetricsVisualizer,
    T4DXSnapshot,
)
from t4dm.visualization.telemetry_hub import (
    CrossScaleEvent,
    SystemHealth,
    TelemetryConfig,
    TelemetryHub,
    TimeScale,
    create_telemetry_hub,
)
from t4dm.visualization.validation import (
    BIOLOGICAL_RANGES,
    BiologicalValidator,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    quick_validation,
    validate_telemetry_hub,
)

__all__ = [
    # Activation visualization
    "ActivationHeatmap",
    "plot_activation_heatmap",
    "plot_activation_timeline",
    # Plasticity visualization
    "PlasticityTracer",
    "plot_bcm_curve",
    "plot_weight_changes",
    "plot_ltp_ltd_distribution",
    # Neuromodulator visualization
    "NeuromodulatorDashboard",
    "plot_neuromodulator_traces",
    "plot_neuromodulator_radar",
    # Pattern separation visualization
    "PatternSeparationVisualizer",
    "plot_separation_comparison",
    "plot_sparsity_distribution",
    # Consolidation visualization
    "ConsolidationVisualizer",
    "plot_swr_sequence",
    "plot_replay_priority",
    # Embedding projection
    "EmbeddingProjector",
    "plot_tsne_projection",
    "plot_umap_projection",
    # Neuromod layers visualization
    "NeuromodLayerVisualizer",
    "NeuromodLayerSnapshot",
    "NEUROMOD_NT_NAMES",
    "NEUROMOD_NT_COLORS",
    "NT_LAYER_MAPPING",
    # Oscillator injection visualization
    "OscillatorInjectionVisualizer",
    "OscillatorSnapshot",
    "OSC_NAMES",
    "OSC_COLORS",
    # Persistence visualization
    "PersistenceVisualizer",
    "PersistenceMetrics",
    "T4DXPersistenceInfo",
    "WALSegmentInfo",
    "CheckpointInfo",
    "plot_wal_timeline",
    "plot_durability_dashboard",
    "plot_t4dx_state",
    "plot_checkpoint_history",
    # Energy landscape visualization
    "EnergyLandscapeVisualizer",
    "EnergySnapshot",
    "TrajectoryPoint",
    "plot_energy_contour",
    "plot_energy_timeline",
    "plot_basin_occupancy",
    "create_energy_dashboard",
    # Coupling dynamics visualization
    "CouplingDynamicsVisualizer",
    "CouplingSnapshot",
    "LearningEvent",
    "plot_coupling_heatmap",
    "plot_eligibility_heatmap",
    "plot_spectral_radius_timeline",
    "plot_ei_balance_timeline",
    "plot_coupling_eigenvalue_spectrum",
    "create_coupling_dashboard",
    # NT state dashboard visualization
    "NTStateDashboard",
    "NTSnapshot",
    "NTStatistics",
    "NT_LABELS",
    "NT_COLORS",
    "HOMEOSTATIC_SETPOINTS",
    "RECEPTOR_KM",
    "plot_nt_channels",
    "plot_deviation_heatmap",
    "plot_saturation_curves",
    "plot_correlation_matrix",
    "plot_autocorrelation",
    "plot_opponent_processes",
    "create_nt_dashboard",
    # Stability monitor visualization
    "StabilityMonitor",
    "StabilitySnapshot",
    "StabilityType",
    "BifurcationEvent",
    "plot_eigenvalue_spectrum",
    "plot_stability_timeline",
    "plot_lyapunov_timeline",
    "plot_eigenvalue_evolution",
    "plot_bifurcation_diagram",
    "plot_oscillation_metrics",
    "create_stability_dashboard",
    # SWR telemetry
    "SWRTelemetry",
    "SWRTelemetryEvent",
    # PAC telemetry
    "PACTelemetry",
    "PACSnapshot",
    # DA telemetry
    "DATelemetry",
    "DASnapshot",
    "DASignalType",
    "DARampEvent",
    "DAStatistics",
    "create_da_dashboard",
    "plot_da_timeline",
    "plot_rpe_distribution",
    # Telemetry Hub
    "TelemetryHub",
    "TelemetryConfig",
    "TimeScale",
    "CrossScaleEvent",
    "SystemHealth",
    "create_telemetry_hub",
    # Biological Validation
    "BiologicalValidator",
    "ValidationResult",
    "ValidationReport",
    "ValidationSeverity",
    "BIOLOGICAL_RANGES",
    "validate_telemetry_hub",
    "quick_validation",
    # Forward-Forward visualization (Phase 4)
    "ForwardForwardVisualizer",
    "FFSnapshot",
    "FFTrainingEvent",
    "plot_goodness_bars",
    "plot_goodness_timeline",
    "plot_margin_evolution",
    "plot_phase_comparison",
    "create_ff_dashboard",
    # Capsule network visualization (Phase 4)
    "CapsuleVisualizer",
    "CapsuleSnapshot",
    "RoutingEvent",
    "plot_entity_probabilities",
    "plot_routing_heatmap",
    "plot_pose_vectors",
    "plot_routing_entropy",
    "create_capsule_dashboard",
    # Glymphatic visualization (Phase 4)
    "GlymphaticVisualizer",
    "GlymphaticSnapshot",
    "ClearanceEvent",
    "SleepStage",
    "plot_sleep_stage_timeline",
    "plot_clearance_and_waste",
    "plot_clearance_by_stage",
    "plot_stage_pie",
    "create_glymphatic_dashboard",
    # Kappa gradient visualization
    "KappaGradientVisualizer",
    "KappaSnapshot",
    "KAPPA_BANDS",
    "KAPPA_COLORS",
    # T4DX metrics visualization
    "T4DXMetricsVisualizer",
    "T4DXSnapshot",
    "CompactionEvent",
    "CompactionType",
    # Spiking dynamics visualization
    "SpikingDynamicsVisualizer",
    "SpikingSnapshot",
    # Qwen metrics visualization
    "QwenMetricsVisualizer",
    "QwenSnapshot",
]
