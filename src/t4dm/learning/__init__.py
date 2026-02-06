"""
T4DM Learning System.

Implements adaptive learning for memory retrieval based on outcomes.

Modules:
- events: Learning event data structures and representation formats
- collector: Event collection and SQLite storage
- neuro_symbolic: Neuro-symbolic graph representation
- dopamine: Reward prediction error for surprise-driven learning
- norepinephrine: Arousal/attention modulation
- acetylcholine: Encoding/retrieval mode switching
- serotonin: Long-term credit assignment
- inhibition: GABA-like competitive dynamics
- neuromodulators: Integrated neuromodulator orchestra
- credit_flow: Connect neuromodulator signals to weight updates
- three_factor: Biologically-plausible three-factor learning rule
- self_supervised: Credit estimation without explicit outcomes (P1-2)
- retrieval_feedback: Implicit feedback collection from retrieval outcomes (P1C)
- feedback_signals: Convert feedback to learning signals (P1C)
- unified_signals: Deep integration of all learning systems (P6B)

Key Concepts:
1. RetrievalEvent: Captures what memories were retrieved and their scores
2. OutcomeEvent: Records success/failure of tasks using retrieved memories
3. Experience: Combined retrieval+outcome for training
4. NeuroSymbolicMemory: Memory with both neural (embedding) and symbolic (triples) representation

Neuromodulator Systems:
1. Dopamine: delta = actual - expected (reward prediction error)
2. Norepinephrine: Arousal, novelty detection, exploration/exploitation
3. Acetylcholine: Encoding vs retrieval mode switching
4. Serotonin: Long-term credit assignment, patience, mood
5. GABA/Glutamate: Lateral inhibition, winner-take-all, sparsity

Three-Factor Learning:
- effective_lr = base_lr * eligibility * neuromod_gate * dopamine_surprise
- Eligibility: Which synapses were active (temporal credit assignment)
- Neuromodulator gate: Should we learn now (encoding mode, arousal)
- Dopamine surprise: How surprising was this (prediction error)

Credit Flow:
- CreditFlowEngine: Applies neuromodulator signals to actual memory updates
- Bridges process_outcome() results to reconsolidation and Hebbian updates

Retrieval Feedback Loop (P1C):
- RetrievalFeedbackCollector: Collect implicit feedback (clicks, dwell times)
- FeedbackSignalProcessor: Convert feedback to three-factor compatible signals
- Enables continuous learning from user interactions without explicit labels

Unified Learning Signals (P6B):
- UnifiedLearningSignal: Combines all learning signals
- Three-factor base * (FF goodness + capsule agreement) = weight delta
- Neurogenesis signals for structural updates
- Deep integration of all learning systems

Representation Formats:
- FullJSON: Complete fidelity for storage
- ToonJSON: 50-60% token reduction for LLM context
- NeuroSymbolicTriples: Graph-based for reasoning
"""

from t4dm.learning.acetylcholine import (
    AcetylcholineState,
    AcetylcholineSystem,
    CognitiveMode,
)
from t4dm.learning.causal_discovery import (
    CausalAttribution,
    CausalAttributor,
    CausalDiscoveryConfig,
    CausalEdge,
    CausalGraph,
    CausalLearner,
    CausalRelationType,
)
from t4dm.learning.cold_start import (
    ColdStartManager,
    ContextLoader,
    ContextSignals,
    PopulationPrior,
)
from t4dm.learning.collector import (
    CollectorConfig,
    EventCollector,
    EventStore,
    get_collector,
)
from t4dm.learning.credit_flow import (
    CreditFlowEngine,
)
from t4dm.learning.dopamine import (
    DopamineSystem,
    RewardPredictionError,
    compute_rpe,
)
from t4dm.learning.eligibility import (
    EligibilityConfig,
    EligibilityTrace,
    LayeredEligibilityTrace,
)
from t4dm.learning.events import (
    Experience,
    FeedbackSignal,
    FullJSON,
    MemoryType,
    NeuroSymbolicTriples,
    OutcomeEvent,
    OutcomeType,
    RetrievalEvent,
    ToonJSON,
    get_representation,
)
from t4dm.learning.feedback_signals import (
    AdapterTrainingSignal,
    FeedbackSignalProcessor,
    FeedbackToAdapterBridge,
    LearningSignal,
)
from t4dm.learning.fsrs import (
    FSRS,
    FSRSMemoryTracker,
    FSRSParameters,
    MemoryState,
    Rating,
    SchedulingInfo,
    create_fsrs,
)
from t4dm.learning.generative_replay import (
    GeneratedSample,
    GenerativeReplayConfig,
    GenerativeReplaySystem,
    ReplayPhase,
    ReplayStats,
    create_generative_replay,
)
from t4dm.learning.homeostatic import (
    HomeostaticPlasticity,
    HomeostaticState,
    apply_homeostatic_bounds,
)
from t4dm.learning.hooks import (
    RetrievalHookMixin,
    emit_retrieval_event,
    emit_unified_retrieval_event,
    get_learning_collector,
    learning_retrieval,
)
from t4dm.learning.inhibition import (
    InhibitionResult,
    InhibitoryNetwork,
    SparseRetrieval,
)
from t4dm.learning.neuro_symbolic import (
    NeuroSymbolicMemory,
    NeuroSymbolicReasoner,
    PredicateType,
    Triple,
    TripleSet,
)
from t4dm.learning.neuromodulators import (
    NeuromodulatorOrchestra,
    NeuromodulatorState,
    create_neuromodulator_orchestra,
)
from t4dm.learning.norepinephrine import (
    ArousalState,
    NorepinephrineSystem,
)
from t4dm.learning.persistence import (
    LearnedGateState,
    ScorerState,
    StatePersister,
)
from t4dm.learning.persistence import (
    NeuromodulatorState as PersistedNeuromodulatorState,
)
from t4dm.learning.reconsolidation import (
    DopamineModulatedReconsolidation,
    ReconsolidationEngine,
    ReconsolidationUpdate,
    reconsolidate,
)
from t4dm.learning.retrieval_feedback import (
    RetrievalFeedback,
    RetrievalFeedbackCollector,
    RetrievalOutcome,
)
from t4dm.learning.scorer import (
    LearnedRetrievalScorer,
    ListMLELoss,
    PrioritizedReplayBuffer,
    ReplayItem,
    ScorerTrainer,
    TrainerConfig,
    create_scorer,
    create_trainer,
)
from t4dm.learning.self_supervised import (
    ImplicitCredit,
    SelfSupervisedCredit,
)
from t4dm.learning.serotonin import (
    SerotoninSystem,
    TemporalContext,
)
from t4dm.learning.three_factor import (
    ThreeFactorLearningRule,
    ThreeFactorReconsolidation,
    ThreeFactorSignal,
    create_three_factor_rule,
)
from t4dm.learning.unified_signals import (
    LearningContext,
    LearningUpdate,
    SignalSource,
    StructuralUpdate,
    UnifiedLearningSignal,
    UnifiedSignalConfig,
    UpdateType,
    create_fully_integrated_signal,
    create_unified_signal,
)
from t4dm.learning.vae_generator import (
    VAEConfig,
    VAEGenerator,
    VAEState,
    create_vae_generator,
)
from t4dm.learning.free_energy import (
    FreeEnergyConfig,
    FreeEnergyMinimizer,
    FreeEnergyResult,
    FreeEnergyTracker,
)

__all__ = [
    # Events
    "OutcomeType",
    "FeedbackSignal",
    "MemoryType",
    "RetrievalEvent",
    "OutcomeEvent",
    "Experience",
    # Representations
    "FullJSON",
    "ToonJSON",
    "NeuroSymbolicTriples",
    "get_representation",
    # Collector
    "EventStore",
    "CollectorConfig",
    "EventCollector",
    "get_collector",
    # Neuro-Symbolic
    "PredicateType",
    "Triple",
    "TripleSet",
    "NeuroSymbolicMemory",
    "NeuroSymbolicReasoner",
    # Hooks
    "get_learning_collector",
    "emit_retrieval_event",
    "emit_unified_retrieval_event",
    "learning_retrieval",
    "RetrievalHookMixin",
    # Scorer
    "LearnedRetrievalScorer",
    "PrioritizedReplayBuffer",
    "ReplayItem",
    "ListMLELoss",
    "ScorerTrainer",
    "TrainerConfig",
    "create_scorer",
    "create_trainer",
    # Dopamine
    "RewardPredictionError",
    "DopamineSystem",
    "compute_rpe",
    # Norepinephrine
    "ArousalState",
    "NorepinephrineSystem",
    # Acetylcholine
    "CognitiveMode",
    "AcetylcholineState",
    "AcetylcholineSystem",
    # Serotonin
    "TemporalContext",
    "SerotoninSystem",
    # Eligibility Traces
    "EligibilityTrace",
    "EligibilityConfig",
    "LayeredEligibilityTrace",
    # Inhibition
    "InhibitionResult",
    "InhibitoryNetwork",
    "SparseRetrieval",
    # Neuromodulators
    "NeuromodulatorState",
    "NeuromodulatorOrchestra",
    "create_neuromodulator_orchestra",
    # Credit Flow
    "CreditFlowEngine",
    # Persistence
    "LearnedGateState",
    "ScorerState",
    "PersistedNeuromodulatorState",
    "StatePersister",
    # Cold Start
    "ContextSignals",
    "ContextLoader",
    "PopulationPrior",
    "ColdStartManager",
    # Reconsolidation
    "ReconsolidationUpdate",
    "ReconsolidationEngine",
    "DopamineModulatedReconsolidation",
    "reconsolidate",
    # Homeostatic Plasticity
    "HomeostaticState",
    "HomeostaticPlasticity",
    "apply_homeostatic_bounds",
    # Three-Factor Learning
    "ThreeFactorSignal",
    "ThreeFactorLearningRule",
    "ThreeFactorReconsolidation",
    "create_three_factor_rule",
    # FSRS Spaced Repetition
    "Rating",
    "FSRSParameters",
    "MemoryState",
    "SchedulingInfo",
    "FSRS",
    "FSRSMemoryTracker",
    "create_fsrs",
    # P1-2: Self-Supervised Credit
    "ImplicitCredit",
    "SelfSupervisedCredit",
    # P1C: Retrieval Feedback Loop
    "RetrievalFeedback",
    "RetrievalOutcome",
    "RetrievalFeedbackCollector",
    "LearningSignal",
    "FeedbackSignalProcessor",
    "AdapterTrainingSignal",
    "FeedbackToAdapterBridge",
    # P4-2: Causal Discovery
    "CausalLearner",
    "CausalGraph",
    "CausalAttributor",
    "CausalAttribution",
    "CausalEdge",
    "CausalRelationType",
    "CausalDiscoveryConfig",
    # Generative Replay (Hinton Wake-Sleep)
    "ReplayPhase",
    "GenerativeReplayConfig",
    "GeneratedSample",
    "ReplayStats",
    "GenerativeReplaySystem",
    "create_generative_replay",
    # P6.1: VAE Generator
    "VAEConfig",
    "VAEState",
    "VAEGenerator",
    "create_vae_generator",
    # P6B: Unified Learning Signals
    "UnifiedLearningSignal",
    "UnifiedSignalConfig",
    "LearningContext",
    "LearningUpdate",
    "StructuralUpdate",
    "UpdateType",
    "SignalSource",
    "create_unified_signal",
    "create_fully_integrated_signal",
    # W1-01: Free Energy Objective (Friston)
    "FreeEnergyConfig",
    "FreeEnergyMinimizer",
    "FreeEnergyResult",
    "FreeEnergyTracker",
]
