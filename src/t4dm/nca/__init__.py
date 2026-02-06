"""
Neural Cellular Automata (NCA) module for T4DM.

Implements KATIE-inspired neural field dynamics with learnable coupling,
integrated with WW's existing learning mechanisms.

Architecture:
- neural_field.py: 3D spatiotemporal PDE solver for NT dynamics
- coupling.py: Learnable coupling matrix with biological bounds
- attractors.py: Cognitive state attractor basins
- energy.py: Hopfield-inspired energy landscape with contrastive learning

Integration:
- Connects to ww.learning for dopamine RPE, eligibility traces
- Connects to ww.memory for encoding/retrieval modulation
- Connects to bridge/ for memory-NCA binding
"""

from t4dm.nca.adenosine import (
    AdenosineConfig,
    AdenosineDynamics,
    AdenosineState,
    SleepPressureIntegrator,
    SleepWakeState,
)
from t4dm.nca.astrocyte import (
    AstrocyteConfig,
    AstrocyteLayer,
    AstrocyteLayerState,
    AstrocyteState,
    compute_tripartite_synapse,
)
from t4dm.nca.attractors import (
    AttractorBasin,
    CognitiveState,
    StateTransitionManager,
)
from t4dm.nca.cerebellum import (
    CerebellarConfig,
    CerebellarModule,
    CerebellarState,
    DeepCerebellarNuclei,
    GranuleCellLayer,
    PurkinjeCellLayer,
    create_cerebellar_module,
)
from t4dm.nca.capsules import (
    CapsuleConfig,
    CapsuleLayer,
    CapsuleNetwork,
    CapsuleState,
    RoutingType,
    SquashType,
    create_capsule_layer,
    create_capsule_network,
)
from t4dm.nca.connectome import (
    BrainRegion,
    Connectome,
    ConnectomeConfig,
    ConnectomeIntegrator,
    NTSystem,
    ProjectionPathway,
    RegionType,
    create_default_connectome,
    create_minimal_connectome,
    get_pathway_summary,
)
from t4dm.nca.coupling import (
    BiologicalBounds,
    CouplingConfig,
    LearnableCoupling,
)
from t4dm.nca.delays import (
    CircularDelayBuffer,
    DelayConfig,
    DelayDifferentialOperator,
    DelayState,
    DistanceMatrix,
    FiberType,
    TransmissionDelaySystem,
)
from t4dm.nca.dopamine_integration import (
    DopamineIntegration,
    DopamineIntegrationConfig,
    IntegratedDAState,
    create_dopamine_integration,
)
from t4dm.nca.energy import (
    ContrastiveState,
    EnergyBasedLearner,
    EnergyConfig,
    EnergyLandscape,
    HopfieldIntegration,
    LearningPhase,
)
from t4dm.nca.forward_forward import (
    FFPhase,
    ForwardForwardConfig,
    ForwardForwardLayer,
    ForwardForwardNetwork,
    ForwardForwardState,
    create_ff_layer,
    create_ff_network,
)
from t4dm.nca.adaptive_threshold import (
    AdaptiveThreshold,
    AdaptiveThresholdConfig,
    AdaptiveThresholdManager,
)
from t4dm.nca.glutamate_signaling import (
    GlutamateConfig,
    GlutamatePool,
    GlutamateSignaling,
    GlutamateState,
    NMDASubtype,
    PlasticityDirection,
    create_glutamate_signaling,
)
from t4dm.nca.glymphatic import (
    ClearanceEvent,
    GlymphaticConfig,
    GlymphaticSystem,
    WasteCategory,
    WasteState,
    WasteTracker,
    create_glymphatic_system,
)
from t4dm.nca.hippocampus import (
    CA1Layer,
    CA3Layer,
    DentateGyrusLayer,
    HippocampalCircuit,
    HippocampalConfig,
    HippocampalMode,
    HippocampalState,
    create_hippocampal_circuit,
)
from t4dm.nca.locus_coeruleus import (
    LCConfig,
    LCFiringMode,
    LCState,
    LocusCoeruleus,
    SurpriseConfig,
    # Phase 2: Surprise Model
    SurpriseModel,
    SurpriseState,
    create_locus_coeruleus,
)
from t4dm.nca.neural_field import (
    NeuralFieldConfig,
    NeuralFieldSolver,
    NeurotransmitterState,
)
from t4dm.nca.oscillators import (
    AlphaOscillator,
    CognitivePhase,
    DeltaOscillator,
    FrequencyBandGenerator,
    OscillationBand,
    OscillatorConfig,
    OscillatorState,
    PhaseAmplitudeCoupling,
    SleepState,
)
from t4dm.nca.pose import (
    PoseConfig,
    PoseMatrix,
    PoseState,
    SemanticDimension,
    create_identity_pose,
    create_random_pose,
)
from t4dm.nca.pose_learner import (
    PoseDimensionDiscovery,
    PoseLearnerConfig,
    PoseLearnerState,
    PoseLearningMixin,
    create_learnable_capsule_system,
    create_pose_learner,
)
from t4dm.nca.raphe import (
    PatienceConfig,
    # Phase 2: Patience Model
    PatienceModel,
    PatienceState,
    RapheConfig,
    RapheNucleus,
    RapheNucleusState,
    RapheState,
    create_raphe_nucleus,
)
from t4dm.nca.sleep_spindles import (
    SleepSpindleGenerator,
    SpindleConfig,
    SpindleDeltaCoupler,
    SpindleEvent,
    SpindleState,
)
from t4dm.nca.spatial_cells import (
    GridModule,
    PlaceCell,
    Position2D,
    SpatialCellSystem,
    SpatialConfig,
    SpatialState,
)
from t4dm.nca.stability import (
    StabilityAnalyzer,
    StabilityConfig,
    StabilityResult,
    StabilityType,
    check_energy_stability,
)
from t4dm.nca.striatal_coupling import (
    DAACHCoupling,
    DAACHCouplingConfig,
    DAACHState,
)
from t4dm.nca.striatal_msn import (
    ActionState,
    MSNConfig,
    MSNPopulationState,
    StriatalMSN,
    create_striatal_msn,
)
from t4dm.nca.swr_coupling import (
    RIPPLE_FREQ_MAX,
    RIPPLE_FREQ_MIN,
    RIPPLE_FREQ_OPTIMAL,
    SWRConfig,
    SWRCouplingState,
    SWREvent,
    SWRNeuralFieldCoupling,
    SWRPhase,
    WakeSleepMode,
    create_swr_coupling,
)
from t4dm.nca.theta_gamma_integration import (
    ThetaGammaConfig,
    ThetaGammaIntegration,
    ThetaGammaState,
    WMSlot,
)
from t4dm.nca.vta import (
    VTACircuit,
    VTAConfig,
    VTAFiringMode,
    VTAState,
    create_vta_circuit,
)

__all__ = [
    # Neural Field
    "NeuralFieldConfig",
    "NeuralFieldSolver",
    "NeurotransmitterState",
    # Coupling
    "LearnableCoupling",
    "BiologicalBounds",
    "CouplingConfig",
    # Attractors
    "CognitiveState",
    "AttractorBasin",
    "StateTransitionManager",
    # Energy
    "EnergyLandscape",
    "HopfieldIntegration",
    "EnergyConfig",
    "EnergyBasedLearner",
    "LearningPhase",
    "ContrastiveState",
    # Striatal Coupling
    "DAACHCoupling",
    "DAACHCouplingConfig",
    "DAACHState",
    # Stability Analysis
    "StabilityAnalyzer",
    "StabilityConfig",
    "StabilityResult",
    "StabilityType",
    "check_energy_stability",
    # Astrocyte Layer
    "AstrocyteLayer",
    "AstrocyteConfig",
    "AstrocyteLayerState",
    "AstrocyteState",
    "compute_tripartite_synapse",
    # Oscillators
    "FrequencyBandGenerator",
    "OscillatorConfig",
    "OscillatorState",
    "OscillationBand",
    "CognitivePhase",
    "PhaseAmplitudeCoupling",
    "AlphaOscillator",
    "DeltaOscillator",
    "SleepState",
    # Sleep Spindles
    "SleepSpindleGenerator",
    "SpindleConfig",
    "SpindleEvent",
    "SpindleState",
    "SpindleDeltaCoupler",
    # Adenosine/Sleep
    "AdenosineDynamics",
    "AdenosineConfig",
    "AdenosineState",
    "SleepWakeState",
    "SleepPressureIntegrator",
    # Transmission Delays
    "TransmissionDelaySystem",
    "DelayConfig",
    "DelayState",
    "CircularDelayBuffer",
    "DistanceMatrix",
    "DelayDifferentialOperator",
    "FiberType",
    # Connectome
    "Connectome",
    "ConnectomeConfig",
    "ConnectomeIntegrator",
    "BrainRegion",
    "ProjectionPathway",
    "RegionType",
    "NTSystem",
    "create_default_connectome",
    "create_minimal_connectome",
    "get_pathway_summary",
    # Hippocampus
    "HippocampalCircuit",
    "HippocampalConfig",
    "HippocampalState",
    "HippocampalMode",
    "DentateGyrusLayer",
    "CA3Layer",
    "CA1Layer",
    "create_hippocampal_circuit",
    # VTA Dopamine Circuit
    "VTACircuit",
    "VTAConfig",
    "VTAState",
    "VTAFiringMode",
    "create_vta_circuit",
    # Dopamine Integration
    "DopamineIntegration",
    "DopamineIntegrationConfig",
    "IntegratedDAState",
    "create_dopamine_integration",
    # Raphe Nucleus
    "RapheNucleus",
    "RapheConfig",
    "RapheNucleusState",
    "RapheState",
    "create_raphe_nucleus",
    # Phase 2: Patience Model (Temporal Discounting)
    "PatienceModel",
    "PatienceConfig",
    "PatienceState",
    # SWR Coupling
    "SWRNeuralFieldCoupling",
    "SWRConfig",
    "SWRCouplingState",
    "SWREvent",
    "SWRPhase",
    "WakeSleepMode",
    "RIPPLE_FREQ_MIN",
    "RIPPLE_FREQ_MAX",
    "RIPPLE_FREQ_OPTIMAL",
    "create_swr_coupling",
    # Striatal MSN Populations
    "StriatalMSN",
    "MSNConfig",
    "MSNPopulationState",
    "ActionState",
    "create_striatal_msn",
    # Locus Coeruleus
    "LocusCoeruleus",
    "LCConfig",
    "LCState",
    "LCFiringMode",
    "create_locus_coeruleus",
    # Phase 2: Surprise Model (Uncertainty Signaling)
    "SurpriseModel",
    "SurpriseConfig",
    "SurpriseState",
    # Glutamate Signaling
    "GlutamateSignaling",
    "GlutamateConfig",
    "GlutamateState",
    "GlutamatePool",
    "NMDASubtype",
    "PlasticityDirection",
    "create_glutamate_signaling",
    # Theta-Gamma Integration (P4-4)
    "ThetaGammaIntegration",
    "ThetaGammaConfig",
    "ThetaGammaState",
    "WMSlot",
    # Spatial Cells (P4-3)
    "SpatialCellSystem",
    "SpatialConfig",
    "SpatialState",
    "Position2D",
    "PlaceCell",
    "GridModule",
    # Phase 3: Forward-Forward Algorithm (Hinton 2022)
    "ForwardForwardLayer",
    "ForwardForwardNetwork",
    "ForwardForwardConfig",
    "ForwardForwardState",
    "FFPhase",
    "create_ff_layer",
    "create_ff_network",
    # W1-02: Adaptive Threshold (Hinton extension)
    "AdaptiveThreshold",
    "AdaptiveThresholdConfig",
    "AdaptiveThresholdManager",
    # Phase 4: Pose Matrix (H9)
    "PoseConfig",
    "PoseState",
    "PoseMatrix",
    "SemanticDimension",
    "create_identity_pose",
    "create_random_pose",
    # Phase 4: Capsule Networks (H8)
    "CapsuleConfig",
    "CapsuleState",
    "CapsuleLayer",
    "CapsuleNetwork",
    "SquashType",
    "RoutingType",
    "create_capsule_layer",
    "create_capsule_network",
    # Phase 4B: Emergent Pose Learning (H8-H9)
    "PoseDimensionDiscovery",
    "PoseLearnerConfig",
    "PoseLearnerState",
    "PoseLearningMixin",
    "create_pose_learner",
    "create_learnable_capsule_system",
    # Cerebellum
    "CerebellarModule",
    "CerebellarConfig",
    "CerebellarState",
    "GranuleCellLayer",
    "PurkinjeCellLayer",
    "DeepCerebellarNuclei",
    "create_cerebellar_module",
    # Phase 4: Glymphatic System (B8)
    "GlymphaticConfig",
    "GlymphaticSystem",
    "WasteState",
    "WasteTracker",
    "WasteCategory",
    "ClearanceEvent",
    "create_glymphatic_system",
]
