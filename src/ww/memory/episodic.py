"""
Episodic Memory Service for World Weaver.

Implements autobiographical memory with bi-temporal versioning and FSRS decay.
"""

import logging
import math
from datetime import datetime
from uuid import UUID

import numpy as np
from opentelemetry.trace import SpanKind

# P7.1: Bridge container for NCA subsystem integration
from ww.core.bridge_container import get_bridge_container
from ww.core.config import get_settings
from ww.core.types import Episode, EpisodeContext, Outcome, ScoredResult
from ww.embedding.bge_m3 import get_embedding_provider
from ww.encoding.ff_encoder import FFEncoder, FFEncoderConfig, get_ff_encoder
from ww.nca.capsules import CapsuleConfig, CapsuleLayer
# Phase 2A: FF-Capsule Bridge for unified goodness + routing agreement
from ww.bridges.ff_capsule_bridge import FFCapsuleBridge, FFCapsuleBridgeConfig
from ww.learning.neuro_symbolic import NeuroSymbolicReasoner
from ww.learning.reconsolidation import ReconsolidationEngine
from ww.learning.three_factor import ThreeFactorLearningRule
from ww.memory.cluster_index import ClusterIndex
from ww.memory.feature_aligner import FeatureAligner
from ww.memory.learned_sparse_index import LearnedSparseIndex
from ww.observability.tracing import add_span_attribute, traced
from ww.storage.neo4j_store import get_neo4j_store
from ww.storage.qdrant_store import get_qdrant_store
from ww.storage.saga import Saga, SagaState

# Phase 1A: Decomposed modules for focused functionality
from ww.memory.episodic_fusion import LearnedFusionWeights, LearnedReranker
from ww.memory.episodic_saga import EpisodicSagaCoordinator
from ww.memory.episodic_storage import EpisodicStorageOps, _validate_uuid as _validate_uuid_impl
from ww.memory.episodic_learning import EpisodicLearningOps
from ww.memory.episodic_retrieval import EpisodicRetrievalOps

logger = logging.getLogger(__name__)


def _validate_uuid(value: any, param_name: str) -> UUID:
    """
    DATA-006 FIX: Validate UUID parameter type.

    Phase 1A: Now delegates to episodic_storage module implementation.
    """
    return _validate_uuid_impl(value, param_name)


class EpisodicMemory:
    """
    Episodic memory service.

    Stores autobiographical events with temporal-spatial context.
    Implements FSRS-based decay and recency-weighted retrieval.
    Supports hybrid search (dense + sparse vectors) for improved recall.
    """

    def __init__(self, session_id: str | None = None):
        """
        Initialize episodic memory service.

        Args:
            session_id: Instance namespace for multi-Claude isolation
        """
        settings = get_settings()
        self.session_id = session_id or settings.session_id

        self.embedding = get_embedding_provider()
        self.vector_store = get_qdrant_store(self.session_id)
        self.graph_store = get_neo4j_store(self.session_id)

        # Retrieval weights
        self.semantic_weight = settings.episodic_weight_semantic
        self.recency_weight = settings.episodic_weight_recency
        self.outcome_weight = settings.episodic_weight_outcome
        self.importance_weight = settings.episodic_weight_importance

        # FSRS parameters
        self.default_stability = settings.fsrs_default_stability
        self.decay_factor = settings.fsrs_decay_factor
        self.recency_decay = settings.fsrs_recency_decay

        # Hybrid search settings
        self.hybrid_enabled = settings.embedding_hybrid_enabled
        self._hybrid_initialized = False

        # P5.2: Temporal sequencing within session
        self._last_episode_id: UUID | None = None
        self._sequence_counter: int = 0

        # P5.4: Query-memory encoder separation
        from ww.embedding.query_memory_separation import get_query_memory_separator
        self._query_memory_separator = get_query_memory_separator()
        self._query_memory_separation_enabled = getattr(settings, "query_memory_separation_enabled", True)

        # Three-factor learning rule for biologically-plausible credit assignment
        # Combines: eligibility traces, neuromodulator gating, dopamine surprise
        # PHASE-2 FIX: This was previously computed but not wired to reconsolidation
        self.three_factor = ThreeFactorLearningRule(
            ach_weight=0.4,   # ACh encoding/retrieval mode
            ne_weight=0.35,   # NE arousal contribution
            serotonin_weight=0.25,  # 5-HT mood contribution
            min_eligibility_threshold=0.01,
            min_effective_lr=0.1,
            max_effective_lr=3.0,
        )

        # Reconsolidation engine for embedding updates based on retrieval outcomes
        # Per Hinton critique: memories should update when retrieved, not remain frozen
        # CLS-001 FIX: Hippocampal (episodic) learning should be 10-100x faster than semantic
        # PHASE-2 FIX: Now wired to three-factor learning rule for proper credit assignment
        self.reconsolidation = ReconsolidationEngine(
            base_learning_rate=0.1,  # CLS-001: Fast hippocampal learning rate
            max_update_magnitude=0.2,  # Allow larger updates for one-shot learning
            cooldown_hours=0.5,  # Shorter cooldown for rapid plasticity
            three_factor=self.three_factor,  # PHASE-2: Wire three-factor learning
        )
        self._reconsolidation_enabled = getattr(settings, "reconsolidation_enabled", True)

        # Pattern separation for reducing interference between similar memories
        # Per Hinton critique: DG-like orthogonalization needed for similar inputs
        from ww.memory.pattern_separation import DentateGyrus
        self.pattern_separator = DentateGyrus(
            embedding_provider=self.embedding,
            vector_store=self.vector_store,
            collection_name=self.vector_store.episodes_collection,
            similarity_threshold=0.75,  # Only separate very similar items
            max_separation=0.2,  # Conservative separation
            use_sparse_coding=False  # Keep dense for compatibility
        )
        self._pattern_separation_enabled = getattr(settings, "pattern_separation_enabled", True)

        # Dopamine-like reward prediction error system
        # Learning scales with surprise (δ = actual - expected), not raw outcomes
        # Unexpected successes/failures drive adaptation, expected outcomes don't
        from ww.learning.dopamine import DopamineSystem
        self.dopamine = DopamineSystem(
            default_expected=0.5,
            value_learning_rate=0.1,
            surprise_threshold=0.05
        )

        # Neuro-symbolic reasoner for learned fusion
        # Per Hinton critique: fusion weights should be query-dependent, not fixed
        self.reasoner = NeuroSymbolicReasoner(
            use_learned_fusion=True,
            embed_dim=settings.embedding_dimension
        )

        # Pattern completion for reconstructing partial queries
        # Complementary to pattern separation: DG separates, CA3 completes
        from ww.memory.pattern_separation import PatternCompletion
        self.pattern_completion = PatternCompletion(
            embedding_dim=settings.embedding_dimension,
            num_attractors=1000,  # Store recent patterns as attractors
            convergence_threshold=0.01,
            max_iterations=5
        )
        self._pattern_completion_enabled = getattr(settings, "pattern_completion_enabled", True)

        # Neuromodulator orchestra for brain-like coordination
        # Integrates: DA (surprise), NE (arousal), ACh (mode), 5-HT (long-term), GABA (inhibition)
        from ww.learning.neuromodulators import NeuromodulatorOrchestra
        self.orchestra = NeuromodulatorOrchestra(
            dopamine=self.dopamine  # Share dopamine system
        )
        self._neuromodulation_enabled = getattr(settings, "neuromodulation_enabled", True)

        # Homeostatic plasticity for embedding norm regulation
        # Prevents runaway potentiation via BCM-like sliding threshold
        from ww.learning.homeostatic import HomeostaticPlasticity
        self.homeostatic = HomeostaticPlasticity(
            target_norm=getattr(settings, "homeostatic_target_norm", 1.0),
            norm_tolerance=getattr(settings, "homeostatic_norm_tolerance", 0.2),
            ema_alpha=getattr(settings, "homeostatic_ema_alpha", 0.01),
            decorrelation_strength=getattr(settings, "homeostatic_decorrelation", 0.01),
        )
        self._homeostatic_enabled = getattr(settings, "homeostatic_enabled", True)

        # Learned Memory Gate for "learning what to remember"
        # Per Hinton critique: storage decisions should be learned, not heuristic
        # Uses Bayesian logistic regression + Thompson sampling for exploration
        from ww.core.learned_gate import LearnedMemoryGate
        from ww.core.memory_gate import MemoryGate
        self.learned_gate = LearnedMemoryGate(
            neuromod_orchestra=self.orchestra,
            store_threshold=0.6,
            buffer_threshold=0.3,
            cold_start_threshold=100,
            fallback_gate=MemoryGate()  # Heuristic fallback during cold start
        )
        self._gating_enabled = getattr(settings, "memory_gating_enabled", False)
        self._gate_context = None  # Track context for gate decisions

        # BufferManager for CA1-like temporary storage
        # Per Hinton: Items with BUFFER decisions need evidence accumulation
        # before promotion or discard - BUFFER != "delayed STORE"
        from ww.memory.buffer_manager import BufferManager
        self.buffer_manager = BufferManager(
            promotion_threshold=0.65,  # Higher than gate's 0.6 (need more evidence)
            discard_threshold=0.25,    # Below gate's 0.3 buffer threshold
            learned_gate=self.learned_gate,  # For training on promotion/discard
            stagger_limit=5,  # Prevent catastrophic forgetting
        )
        self._buffering_enabled = getattr(settings, "memory_buffering_enabled", True)

        # R1: Learned fusion weights for query-dependent scoring
        # Replaces fixed weights (0.4/0.3/0.2/0.1) with adaptive weights
        self.learned_fusion = LearnedFusionWeights(
            embed_dim=1024,
            hidden_dim=32,
            learning_rate=0.01,
        )
        self._learned_fusion_enabled = getattr(settings, "learned_fusion_enabled", True)

        # P0c: Learned re-ranking for retrieval results
        # Second-pass scoring using cross-component interactions
        self.learned_reranker = LearnedReranker(
            embed_dim=1024,
            learning_rate=0.005,
        )
        self._reranking_enabled = getattr(settings, "reranking_enabled", True)

        # Phase 1: Hierarchical ClusterIndex for O(log n) retrieval
        # CA3-like semantic clustering - clusters registered during sleep consolidation
        self.cluster_index = ClusterIndex(
            embedding_dim=1024,
            default_k=5,
            similarity_threshold=0.3,
        )
        self._hierarchical_search_enabled = getattr(settings, "hierarchical_search_enabled", True)
        self._min_clusters_for_hierarchical = 3  # Need enough clusters to benefit

        # Phase 2: Learned sparse addressing for adaptive feature/cluster attention
        # Replaces fixed 10% sparsity with query-dependent addressing
        self.sparse_index = LearnedSparseIndex(
            embed_dim=1024,
            hidden_dim=256,
            max_clusters=500,
            learning_rate=0.005,
        )
        self._sparse_addressing_enabled = getattr(settings, "sparse_addressing_enabled", True)

        # Phase 3: Feature aligner for joint gate-retrieval optimization
        # Projects gate features to retrieval space for consistency loss
        self.feature_aligner = FeatureAligner(
            gate_dim=247,  # After P0a content projection
            retrieval_dim=4,  # semantic, recency, outcome, importance
            hidden_dim=32,
            learning_rate=0.01,
        )
        self._joint_optimization_enabled = getattr(settings, "joint_optimization_enabled", True)

        # P7.1: Bridge container for FF encoding bridge integration
        # FF bridge provides novelty detection to modulate encoding strength
        self._bridge_container = get_bridge_container(self.session_id)
        self._ff_encoding_enabled = getattr(settings, "ff_encoding_enabled", True)

        # P6.4: FF retrieval scoring - uses goodness to boost confident matches
        # High FF goodness = strong pattern match = more reliable memory
        self._ff_retrieval_enabled = getattr(settings, "ff_retrieval_enabled", True)

        # P6.2: Capsule retrieval scoring - uses routing agreement for composition
        # Strong agreement between query and candidates = better hierarchical fit
        self._capsule_retrieval_enabled = getattr(settings, "capsule_retrieval_enabled", True)

        # Phase 5: Learnable FF Encoder - THE LEARNING GAP FIX
        # This sits between frozen embedder and storage, allowing system to LEARN
        # representations through use, not just store frozen embeddings.
        # Three-factor learning: eligibility x neuromod x dopamine -> weight updates
        self._ff_encoder_enabled = getattr(settings, "ff_encoder_enabled", True)
        if self._ff_encoder_enabled:
            self._ff_encoder = get_ff_encoder(
                FFEncoderConfig(
                    input_dim=settings.embedding_dimension,
                    hidden_dims=(512, 256),
                    output_dim=settings.embedding_dimension,
                    learning_rate=0.03,
                    use_residual=True,
                    use_neuromod_gating=True,
                )
            )
            logger.info("Phase 5: FFEncoder initialized for learnable representation encoding")
        else:
            self._ff_encoder = None

        # Phase 6: Capsule Layer for pose-based representation
        # Poses now emerge from routing-by-agreement, not hand-setting
        # Capsule activations encode entity existence, poses encode configuration
        self._capsule_layer_enabled = getattr(settings, "capsule_layer_enabled", True)
        if self._capsule_layer_enabled:
            self._capsule_layer = CapsuleLayer(
                CapsuleConfig(
                    input_dim=settings.embedding_dimension,
                    num_capsules=32,
                    capsule_dim=16,
                    pose_dim=4,
                    routing_iterations=3,
                    learning_rate=0.01,
                    use_ff_learning=True,
                )
            )
            logger.info("Phase 6: CapsuleLayer initialized for pose-based encoding")
        else:
            self._capsule_layer = None

        # Phase 2A: FF-Capsule Bridge for unified encoding confidence
        # Combines FF goodness (familiarity) with capsule routing agreement (composition)
        # Per Hinton: Both measure pattern coherence, but in complementary ways
        self._ff_capsule_bridge_enabled = getattr(settings, "ff_capsule_bridge_enabled", True)
        if self._ff_capsule_bridge_enabled:
            self._ff_capsule_bridge = FFCapsuleBridge(
                ff_encoder=self._ff_encoder if self._ff_encoder_enabled else None,
                capsule_layer=self._capsule_layer if self._capsule_layer_enabled else None,
                config=FFCapsuleBridgeConfig(
                    ff_weight=0.6,  # Balance FF goodness vs routing agreement
                    goodness_threshold=2.0,
                    agreement_threshold=0.3,
                    joint_learning=True,
                    ff_learning_rate=0.03,
                    capsule_learning_rate=0.01,
                    track_history=True,
                ),
            )
            logger.info("Phase 2A: FFCapsuleBridge initialized for unified encoding confidence")
        else:
            self._ff_capsule_bridge = None

    async def initialize(self) -> None:
        """Initialize storage backends and bridge container."""
        await self.vector_store.initialize()
        await self.graph_store.initialize()

        # Initialize hybrid collection if enabled
        if self.hybrid_enabled and not self._hybrid_initialized:
            try:
                await self.vector_store.ensure_hybrid_collection(
                    self.vector_store.episodes_collection + "_hybrid"
                )
                self._hybrid_initialized = True
                logger.info("Hybrid episode collection initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid collection: {e}. Falling back to dense-only.")

        # P7.1 Phase 2B: Initialize bridge container with NCA components
        self._bridge_container = get_bridge_container(self.session_id)

        # Wire NCA components to bridges if they exist
        if self._ff_encoder_enabled and self._ff_encoder is not None:
            # FFEncoder contains the ForwardForwardLayer
            ff_layer = getattr(self._ff_encoder, 'ff_layer', None)
            if ff_layer is not None:
                self._bridge_container.set_ff_layer(ff_layer)
                logger.info("P7.1 Phase 2B: FF layer wired to bridge container")

        if self._capsule_layer_enabled and self._capsule_layer is not None:
            self._bridge_container.set_capsule_layer(self._capsule_layer)
            logger.info("P7.1 Phase 2B: Capsule layer wired to bridge container")

        if self.dopamine is not None:
            self._bridge_container.set_dopamine_system(self.dopamine)
            logger.info("P7.1 Phase 2B: Dopamine system wired to bridge container")

        # Trigger lazy initialization of bridges
        if self._bridge_container.config.lazy_init:
            # Access bridges to trigger their creation
            _ = self._bridge_container.get_ff_bridge()
            _ = self._bridge_container.get_capsule_bridge()
            _ = self._bridge_container.get_dopamine_bridge()
            logger.info("P7.1 Phase 2B: Bridge container lazy initialization triggered")

    @traced("episodic.create", kind=SpanKind.INTERNAL)
    async def create(
        self,
        content: str,
        context: dict | None = None,
        outcome: str = "neutral",
        valence: float = 0.5,
    ) -> Episode | None:
        """
        Store a new autobiographical event with learned gating.

        Per Hinton critique: Storage decisions should be learned, not heuristic.
        The LearnedMemoryGate uses Bayesian logistic regression + Thompson sampling
        to predict P(memory will be useful) at encoding time.

        Args:
            content: Full interaction text
            context: Spatial context (project, file, tool, etc.)
            outcome: success/failure/partial/neutral
            valence: Importance signal [0,1]

        Returns:
            Created episode with embedding, or None if gated out
        """
        # Generate embedding with optional pattern separation
        # Per Hinton: DG-like orthogonalization reduces interference
        if self._pattern_separation_enabled:
            try:
                embedding = await self.pattern_separator.encode(content)
                embedding = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            except Exception as e:
                logger.warning(f"Pattern separation failed, using standard embedding: {e}")
                embedding = await self.embedding.embed_query(content)
        else:
            embedding = await self.embedding.embed_query(content)

        # Phase 5: Learnable FF Encoding - THE LEARNING GAP FIX
        # Transform frozen embedding through learnable FF layers before storage
        # This enables the system to LEARN representations through use
        if self._ff_encoder_enabled and self._ff_encoder is not None:
            try:
                embedding_np = np.array(embedding)
                encoded_embedding = self._ff_encoder.encode(embedding_np, training=False)
                embedding = encoded_embedding.tolist()

                add_span_attribute("ff_encoder.enabled", True)
                add_span_attribute("ff_encoder.goodness", self._ff_encoder.get_goodness(embedding_np))

                logger.debug(
                    f"Phase 5: Embedding encoded through learnable FF layers "
                    f"(goodness={self._ff_encoder.state.mean_goodness:.3f})"
                )
            except Exception as e:
                logger.warning(f"Phase 5: FF encoding failed, using raw embedding: {e}")

        # Phase 6: Capsule encoding for pose-based representation
        # Use forward_with_routing() so poses emerge from routing agreement
        # rather than direct linear transform. This is THE KEY Phase 6 change.
        capsule_activations = None
        capsule_poses = None
        capsule_routing_stats = None
        if self._capsule_layer_enabled and self._capsule_layer is not None:
            try:
                embedding_np = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding

                # Phase 6: Use routing-based encoding instead of direct forward
                # Poses now emerge from routing-by-agreement among capsules
                capsule_activations, capsule_poses, capsule_routing_stats = (
                    self._capsule_layer.forward_with_routing(
                        embedding_np,
                        learn_poses=True,  # Enable pose weight learning from routing
                    )
                )

                add_span_attribute("capsule.enabled", True)
                add_span_attribute("capsule.mean_activation", float(np.mean(capsule_activations)))
                add_span_attribute("capsule.sparsity", float(self._capsule_layer.state.sparsity))
                add_span_attribute("capsule.mean_agreement", capsule_routing_stats.get('mean_agreement', 0.0))
                add_span_attribute("capsule.pose_change", capsule_routing_stats.get('pose_change', 0.0))

                logger.debug(
                    f"Phase 6: Capsule routing encoding computed "
                    f"(mean_activation={np.mean(capsule_activations):.3f}, "
                    f"sparsity={self._capsule_layer.state.sparsity:.3f}, "
                    f"agreement={capsule_routing_stats.get('mean_agreement', 0.0):.3f}, "
                    f"pose_change={capsule_routing_stats.get('pose_change', 0.0):.3f})"
                )
            except Exception as e:
                logger.warning(f"Phase 6: Capsule routing encoding failed: {e}")
                capsule_activations = None
                capsule_poses = None
                capsule_routing_stats = None

        # Phase 2A: Process through FF-Capsule Bridge for unified confidence
        # Combines FF goodness (have we seen this?) with routing agreement (is it coherent?)
        ff_capsule_confidence = None
        ff_capsule_novelty = None
        if self._ff_capsule_bridge_enabled and self._ff_capsule_bridge is not None:
            try:
                embedding_np = np.array(embedding) if not isinstance(embedding, np.ndarray) else embedding

                # Forward through bridge to get combined FF+Capsule confidence
                ff_output, bridge_capsule_state, ff_capsule_confidence = (
                    self._ff_capsule_bridge.forward(embedding_np, learn_poses=True)
                )

                # Novelty = inverse of familiarity (low goodness = high novelty)
                # Used to modulate encoding strength
                ff_goodness = self._ff_capsule_bridge.state.last_ff_goodness
                ff_capsule_novelty = 1.0 - self._ff_capsule_bridge._normalize_goodness(ff_goodness)

                add_span_attribute("ff_capsule.confidence", float(ff_capsule_confidence))
                add_span_attribute("ff_capsule.novelty", float(ff_capsule_novelty))
                add_span_attribute("ff_capsule.goodness", float(ff_goodness))
                add_span_attribute("ff_capsule.agreement", float(bridge_capsule_state.agreement))

                logger.debug(
                    f"Phase 2A: FF-Capsule bridge processed - "
                    f"confidence={ff_capsule_confidence:.3f}, novelty={ff_capsule_novelty:.3f}, "
                    f"goodness={ff_goodness:.3f}, agreement={bridge_capsule_state.agreement:.3f}"
                )
            except Exception as e:
                logger.warning(f"Phase 2A: FF-Capsule bridge processing failed: {e}")
                ff_capsule_confidence = None
                ff_capsule_novelty = None

        # P7.1: FF Encoding Bridge for novelty detection
        # Uses Forward-Forward goodness to detect novel patterns
        # Novel patterns → enhanced encoding strength, higher priority
        ff_guidance = None
        encoding_multiplier = 1.0
        if self._ff_encoding_enabled:
            try:
                ff_bridge = self._bridge_container.get_ff_bridge()
                if ff_bridge is not None:
                    ff_guidance = ff_bridge.process(np.array(embedding))
                    encoding_multiplier = ff_guidance.encoding_multiplier

                    # Log novelty detection
                    add_span_attribute("ff.is_novel", ff_guidance.is_novel)
                    add_span_attribute("ff.goodness", ff_guidance.goodness)
                    add_span_attribute("ff.encoding_multiplier", encoding_multiplier)
                    add_span_attribute("ff.priority", ff_guidance.priority)

                    if ff_guidance.is_novel:
                        logger.debug(
                            f"P7.1: Novel pattern detected - "
                            f"goodness={ff_guidance.goodness:.3f}, "
                            f"encoding_boost={encoding_multiplier:.3f}"
                        )
            except Exception as e:
                logger.warning(f"FF encoding bridge failed: {e}")
                ff_guidance = None

        # Learned Memory Gating: Should we store this memory?
        # Per Hinton: "Storage should be as learned as retrieval"
        gate_decision = None
        if self._gating_enabled:
            try:
                from ww.core.memory_gate import GateContext

                # Build gate context from episode context
                ctx = context or {}
                gate_context = GateContext(
                    session_id=self.session_id,
                    project=ctx.get("project"),
                    cwd=ctx.get("cwd"),
                    recent_entities=ctx.get("recent_entities", []),
                    last_store_time=self._gate_context.get("last_store_time") if self._gate_context else None,
                    message_count_since_store=self._gate_context.get("msg_count", 0) if self._gate_context else 0,
                    current_task=ctx.get("tool"),  # Use tool as current task
                    is_voice=ctx.get("is_voice", False),
                )

                # Get neuromodulator state for gate decision
                # Process as statement (encoding) not question (retrieval)
                neuromod_state = self.orchestra.process_query(
                    query_embedding=np.array(embedding),
                    is_question=False,  # This is a storage operation
                    explicit_importance=valence if valence != 0.5 else None
                )

                # Predict storage decision with Thompson sampling
                gate_decision = self.learned_gate.predict(
                    content_embedding=np.array(embedding),
                    context=gate_context,
                    neuromod_state=neuromod_state,
                    explore=True  # Use Thompson sampling for exploration
                )

                # Track gate context for next call
                self._gate_context = {
                    "last_store_time": datetime.now() if gate_decision.action.value == "store" else (
                        self._gate_context.get("last_store_time") if self._gate_context else None
                    ),
                    "msg_count": 0 if gate_decision.action.value == "store" else (
                        (self._gate_context.get("msg_count", 0) if self._gate_context else 0) + 1
                    ),
                }

                # If gated out, log and return None
                if gate_decision.action.value == "skip":
                    logger.info(
                        f"Memory gated out: p={gate_decision.probability:.3f}, "
                        f"threshold={self.learned_gate.θ_store:.2f}, "
                        f"exploration_boost={gate_decision.exploration_boost:.2f}"
                    )
                    add_span_attribute("gate.decision", "skip")
                    add_span_attribute("gate.probability", gate_decision.probability)
                    return None

                # If BUFFER, add to buffer manager for evidence accumulation
                # Per Hinton: BUFFER != "delayed STORE" - it means "candidate under observation"
                # The buffer participates in retrieval, gathering implicit evidence
                if gate_decision.action.value == "buffer" and self._buffering_enabled:
                    try:
                        buffer_id = self.buffer_manager.add(
                            content=content,
                            embedding=np.array(embedding),
                            features=gate_decision.features,
                            context=context or {},
                            outcome=outcome,
                            valence=valence,
                        )
                        logger.info(
                            f"Memory buffered: p={gate_decision.probability:.3f}, "
                            f"buffer_id={buffer_id}, buffer_size={self.buffer_manager.size}"
                        )
                        add_span_attribute("gate.decision", "buffer")
                        add_span_attribute("gate.probability", gate_decision.probability)
                        add_span_attribute("buffer.id", str(buffer_id))
                        add_span_attribute("buffer.size", self.buffer_manager.size)
                        return None  # Not stored yet - waiting for evidence
                    except Exception as e:
                        logger.warning(f"Buffering failed, falling back to store: {e}")
                        # Fall through to store

                add_span_attribute("gate.decision", gate_decision.action.value)
                add_span_attribute("gate.probability", gate_decision.probability)

            except Exception as e:
                logger.warning(f"Memory gating failed, storing anyway: {e}")
                gate_decision = None

        # Create episode object with P5.2 temporal sequencing
        # P7.1: Modulate emotional_valence by FF priority for novel patterns
        # Novel patterns → higher importance → better consolidation priority
        effective_valence = valence
        effective_stability = self.default_stability
        if ff_guidance is not None:
            # Boost valence for novel patterns (they're more important to remember)
            effective_valence = min(1.0, valence + ff_guidance.priority * 0.2)
            # Novel patterns get higher initial stability (encoding strength boost)
            effective_stability = self.default_stability * encoding_multiplier

        episode = Episode(
            session_id=self.session_id,
            content=content,
            embedding=embedding,
            context=EpisodeContext(**(context or {})),
            outcome=Outcome(outcome),
            emotional_valence=effective_valence,
            stability=effective_stability,
            # P5.2: Temporal structure
            previous_episode_id=self._last_episode_id,
            sequence_position=self._sequence_counter,
        )

        # Wrap in saga for atomicity across dual stores
        saga = Saga(f"create_episode_{episode.id}")

        # Build payload with Phase 6 capsule data
        episode_payload = self._to_payload(episode)
        if capsule_activations is not None:
            # Store capsule activations as list (for retrieval ranking)
            episode_payload["capsule_activations"] = capsule_activations.tolist()
            episode_payload["capsule_mean_activation"] = float(np.mean(capsule_activations))
        if capsule_poses is not None:
            # Store flattened poses for retrieval (32 capsules × 4×4 = 512 values)
            episode_payload["capsule_poses"] = capsule_poses.reshape(-1).tolist()
        # Phase 2C: Store routing agreement from capsule layer for confidence scoring
        # Per Hinton: Higher routing agreement = more compositionally coherent memory
        if capsule_routing_stats is not None and "mean_agreement" in capsule_routing_stats:
            episode_payload["capsule_routing_agreement"] = float(
                capsule_routing_stats["mean_agreement"]
            )

        # Phase 2A: Add FF-Capsule bridge metadata
        if ff_capsule_confidence is not None:
            episode_payload["ff_capsule_confidence"] = float(ff_capsule_confidence)
            episode_payload["ff_capsule_novelty"] = float(ff_capsule_novelty) if ff_capsule_novelty else 0.0
            episode_payload["ff_capsule_goodness"] = float(self._ff_capsule_bridge.state.last_ff_goodness) if self._ff_capsule_bridge else 0.0
            episode_payload["ff_capsule_agreement"] = float(self._ff_capsule_bridge.state.last_routing_agreement) if self._ff_capsule_bridge else 0.0

        # Step 1: Add to vector store
        saga.add_step(
            name="add_vector",
            action=lambda: self.vector_store.add(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
                vectors=[embedding],
                payloads=[episode_payload],
            ),
            compensate=lambda: self.vector_store.delete(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
            ),
        )

        # Step 2: Create graph node
        saga.add_step(
            name="create_node",
            action=lambda: self.graph_store.create_node(
                label="Episode",
                properties=self._to_graph_props(episode),
            ),
            compensate=lambda: self.graph_store.delete_node(
                node_id=str(episode.id),
                label="Episode",
            ),
        )

        # Execute saga
        result = await saga.execute()

        # Check saga result and raise on failure
        if result.state not in (SagaState.COMMITTED,):
            raise RuntimeError(
                f"Episode creation failed: {result.error} "
                f"(saga: {result.saga_id}, state: {result.state.value})"
            )

        logger.info(
            f"Created episode {episode.id} in session {self.session_id} "
            f"(saga: {result.saga_id}, state: {result.state.value})"
        )

        # P5.2: Update temporal sequencing
        # Link previous episode to this one (bidirectional linking)
        if self._last_episode_id is not None:
            try:
                await self._link_episodes(self._last_episode_id, episode.id)
            except Exception as e:
                logger.warning(f"Failed to link episodes: {e}")

        # Update tracking for next episode
        self._last_episode_id = episode.id
        self._sequence_counter += 1

        # P3.3: Notify consolidation scheduler of new memory
        try:
            from ww.consolidation.service import get_consolidation_scheduler

            scheduler = get_consolidation_scheduler()
            scheduler.record_memory_created()
        except Exception as e:
            # Don't fail episode creation if scheduler notification fails
            logger.debug(f"Scheduler notification skipped: {e}")

        # Also store in hybrid collection if enabled
        if self.hybrid_enabled and self._hybrid_initialized:
            try:
                await self._store_hybrid(episode, content)
            except Exception as e:
                logger.warning(f"Failed to store hybrid vectors for {episode.id}: {e}")

        # Add embedding as attractor for pattern completion
        # This allows partial queries to be completed toward stored memories
        if self._pattern_completion_enabled:
            try:
                self.pattern_completion.add_attractor(np.array(embedding))
            except Exception as e:
                logger.warning(f"Failed to add attractor for {episode.id}: {e}")

        # Register pending label for learned gate training
        # When outcome arrives via learn_from_outcome(), gate learns from this sample
        if self._gating_enabled and gate_decision is not None:
            try:
                # P0a: Pass raw embedding for content projection learning
                # P1: Pass neuromod_state for three-factor learning
                raw_emb = np.array(embedding) if embedding is not None else None
                self.learned_gate.register_pending(
                    episode.id,
                    gate_decision.features,
                    raw_content_embedding=raw_emb,
                    neuromod_state=gate_decision.neuromod_state
                )
                logger.debug(
                    f"Registered pending gate label for {episode.id}: "
                    f"p={gate_decision.probability:.3f}"
                )
            except Exception as e:
                logger.warning(f"Failed to register pending gate label: {e}")

        return episode

    async def _store_hybrid(self, episode: Episode, content: str) -> None:
        """
        Store episode in hybrid collection with dense + sparse vectors.

        Args:
            episode: Episode to store
            content: Original content for embedding
        """
        # Generate hybrid embeddings
        dense_vecs, sparse_vecs = await self.embedding.embed_hybrid([content])

        collection = self.vector_store.episodes_collection + "_hybrid"
        await self.vector_store.add_hybrid(
            collection=collection,
            ids=[str(episode.id)],
            dense_vectors=dense_vecs,
            sparse_vectors=sparse_vecs,
            payloads=[self._to_payload(episode)],
        )
        logger.debug(f"Stored hybrid vectors for episode {episode.id}")

    async def _link_episodes(self, prev_id: UUID, next_id: UUID) -> None:
        """
        P5.2: Create bidirectional temporal link between episodes.

        Updates the previous episode's next_episode_id and creates a
        SEQUENCE temporal link in the graph store.

        Args:
            prev_id: Previous episode UUID
            next_id: Next episode UUID
        """
        from ww.core.types import TemporalLinkType

        # Update previous episode's next_episode_id in vector store
        try:
            await self.vector_store.update_payload(
                collection=self.vector_store.episodes_collection,
                id=str(prev_id),
                payload={"next_episode_id": str(next_id)},
            )
        except Exception as e:
            logger.debug(f"Vector store update skipped (may not support partial update): {e}")

        # Create temporal link relationship in graph store
        try:
            await self.graph_store.create_relationship(
                source_id=str(prev_id),
                target_id=str(next_id),
                relation_type="TEMPORAL_SEQUENCE",
                properties={
                    "link_type": TemporalLinkType.SEQUENCE.value,
                    "strength": 1.0,
                    "created_at": datetime.now().isoformat(),
                },
            )
            logger.debug(f"Created temporal link: {prev_id} -> {next_id}")
        except Exception as e:
            logger.warning(f"Failed to create temporal link in graph: {e}")

    def _to_payload(self, episode: Episode) -> dict:
        """Convert episode to Qdrant payload."""
        now = datetime.now()
        payload = {
            "content": episode.content,
            "session_id": episode.session_id,
            "timestamp": episode.timestamp.isoformat() if episode.timestamp else now.isoformat(),
            "ingested_at": now.isoformat(),
            "emotional_valence": episode.emotional_valence,
            "outcome": episode.outcome.value,
            "stability": episode.stability,
            "access_count": 1,
            "last_accessed": now.isoformat(),
            "context": {
                "project": episode.context.project,
                "file": episode.context.file,
                "tool": episode.context.tool,
            },
        }
        # P5.2: Add temporal fields if present
        if episode.previous_episode_id:
            payload["previous_episode_id"] = str(episode.previous_episode_id)
        if episode.next_episode_id:
            payload["next_episode_id"] = str(episode.next_episode_id)
        if episode.sequence_position is not None:
            payload["sequence_position"] = episode.sequence_position
        if episode.duration_ms is not None:
            payload["duration_ms"] = episode.duration_ms
        if episode.end_timestamp:
            payload["end_timestamp"] = episode.end_timestamp.isoformat()
        return payload

    def _to_graph_props(self, episode: Episode) -> dict:
        """Convert episode to graph properties."""
        return {
            "episode_id": str(episode.id),
            "sessionId": episode.session_id,  # camelCase for Neo4j
            "content": episode.content,
            "timestamp": episode.timestamp.isoformat() if episode.timestamp else None,
            "outcome": episode.outcome.value,
            "valence": episode.emotional_valence,
        }

    def _from_payload(self, episode_id: str, payload: dict) -> Episode:
        """Reconstruct episode from Qdrant payload."""
        timestamp_str = payload.get("timestamp")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else None

        # P5.2: Parse temporal fields
        prev_id_str = payload.get("previous_episode_id")
        next_id_str = payload.get("next_episode_id")
        end_ts_str = payload.get("end_timestamp")

        return Episode(
            id=UUID(episode_id),
            session_id=payload.get("session_id", "default"),
            content=payload.get("content", ""),
            embedding=None,  # Not loaded from payload
            context=EpisodeContext(**payload.get("context", {})),
            outcome=Outcome(payload.get("outcome", "neutral")),
            emotional_valence=payload.get("emotional_valence", 0.5),
            stability=payload.get("stability", 1.0),
            timestamp=timestamp,
            # P5.2: Temporal structure
            previous_episode_id=UUID(prev_id_str) if prev_id_str else None,
            next_episode_id=UUID(next_id_str) if next_id_str else None,
            sequence_position=payload.get("sequence_position"),
            duration_ms=payload.get("duration_ms"),
            end_timestamp=datetime.fromisoformat(end_ts_str) if end_ts_str else None,
        )

    @traced("episodic.recall", kind=SpanKind.INTERNAL)
    async def recall(
        self,
        query: str,
        limit: int = 10,
        session_filter: str | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
        use_pattern_completion: bool = True,
        is_question: bool = True,
        explicit_importance: float | None = None,
    ) -> list[ScoredResult]:
        """
        Retrieve episodes by semantic similarity.

        Neuromodulation integration:
        - NE arousal modulates retrieval threshold (high arousal = broader search)
        - ACh mode determines encoding/retrieval balance
        - GABA inhibition sharpens final results (winner-take-all dynamics)

        Args:
            query: Search query
            limit: Maximum results to return
            session_filter: Optional session filter
            time_start: Optional start time filter
            time_end: Optional end time filter
            use_pattern_completion: Whether to complete partial queries (CA3-like)
            is_question: Whether the query is a question (affects ACh mode)
            explicit_importance: User-indicated importance (affects ACh encoding demand)

        Returns:
            List of scored episodes
        """
        # Generate query embedding
        query_emb = await self.embedding.embed_query(query)
        query_emb_np = np.array(query_emb)

        # Phase 5: Encode query through learnable FF layers
        # This ensures query and stored embeddings are in the same learned space
        if self._ff_encoder_enabled and self._ff_encoder is not None:
            try:
                encoded_query = self._ff_encoder.encode(query_emb_np, training=False)
                query_emb = encoded_query.tolist()
                query_emb_np = encoded_query
                logger.debug(
                    f"Phase 5: Query encoded through FF layers "
                    f"(goodness={self._ff_encoder.get_goodness(encoded_query):.3f})"
                )
            except Exception as e:
                logger.warning(f"Phase 5: Query FF encoding failed: {e}")

        # Process query through neuromodulator orchestra
        # This determines: arousal (NE), encoding/retrieval mode (ACh), threshold modulation
        neuro_state = None
        if self._neuromodulation_enabled:
            try:
                neuro_state = self.orchestra.process_query(
                    query_embedding=query_emb_np,
                    is_question=is_question,
                    explicit_importance=explicit_importance
                )
                logger.debug(
                    f"Neuromodulation: mode={neuro_state.acetylcholine_mode}, "
                    f"gain={neuro_state.norepinephrine_gain:.2f}, "
                    f"explore={neuro_state.exploration_exploitation_balance:.2f}"
                )
            except Exception as e:
                logger.warning(f"Neuromodulation query processing failed: {e}")

        # Apply pattern completion to query embedding if enabled
        # This mimics CA3 pattern completion: partial cues are completed
        # toward stored attractors, improving recall for vague queries
        #
        # P2.2 ACh-CA3 Connection (Hasselmo 2006):
        # - High ACh (encoding mode): Reduce pattern completion, favor separation
        # - Low ACh (retrieval mode): Enhance pattern completion
        completion_iterations = 0
        if use_pattern_completion and self._pattern_completion_enabled:
            try:
                query_arr = np.array(query_emb)
                completed, completion_iterations = self.pattern_completion.complete(query_arr)
                if completion_iterations > 0:
                    # ACh-modulated completion strength (P2.2)
                    # encoding → 0.2 (less completion, favor pattern separation)
                    # balanced → 0.3 (default)
                    # retrieval → 0.6 (stronger pattern completion)
                    if neuro_state is not None:
                        if neuro_state.acetylcholine_mode == "encoding":
                            blend_factor = 0.2
                        elif neuro_state.acetylcholine_mode == "retrieval":
                            blend_factor = 0.6
                        else:
                            blend_factor = 0.3
                    else:
                        blend_factor = 0.3  # Default when neuromodulation disabled

                    query_emb = (
                        (1 - blend_factor) * query_arr +
                        blend_factor * completed
                    ).tolist()
                    logger.debug(
                        f"Pattern completion: {completion_iterations} iterations, "
                        f"blend_factor={blend_factor}, ach_mode={neuro_state.acetylcholine_mode if neuro_state else 'disabled'}"
                    )
            except Exception as e:
                logger.warning(f"Pattern completion failed: {e}")

        add_span_attribute("recall.pattern_completion_iterations", completion_iterations)

        # P5.4: Apply query-memory separation projection
        # Query embeddings are projected differently than memory embeddings
        # This helps disambiguate search intent from stored content
        if self._query_memory_separation_enabled:
            try:
                query_emb_np = np.array(query_emb)
                projected_query = self._query_memory_separator.project_query(query_emb_np)
                query_emb = projected_query.tolist()
                logger.debug("P5.4: Applied query projection for encoder separation")
            except Exception as e:
                logger.warning(f"Query-memory separation failed: {e}")

        # Build filter dict
        filter_dict = {}

        # Determine session filter to use
        session_id = session_filter or (self.session_id if self.session_id != "default" else None)
        if session_id:
            filter_dict["session_id"] = session_id

        # Determine retrieval threshold (NE arousal modulates this)
        # High arousal = lower threshold = broader search (exploration)
        # Low arousal = higher threshold = focused search (exploitation)
        base_threshold = 0.5
        if self._neuromodulation_enabled and neuro_state:
            modulated_threshold = self.orchestra.get_retrieval_threshold(base_threshold)
        else:
            modulated_threshold = base_threshold

        # Phase 1: Hierarchical cluster-based search for O(log n) retrieval
        # Only enabled when we have enough clusters registered
        selected_clusters = []
        cluster_member_ids = []
        use_hierarchical = (
            self._hierarchical_search_enabled and
            self.cluster_index.n_clusters >= self._min_clusters_for_hierarchical
        )

        if use_hierarchical:
            try:
                # Get NE arousal and ACh mode for cluster selection
                ne_gain = neuro_state.norepinephrine_gain if neuro_state else 1.0
                ach_mode = neuro_state.acetylcholine_mode if neuro_state else "retrieval"

                # Select relevant clusters (CA3-like routing)
                selected_clusters = self.cluster_index.select_clusters(
                    query_embedding=query_emb_np,
                    ne_gain=ne_gain,
                    ach_mode=ach_mode,
                )

                # Collect member IDs from selected clusters
                for cluster_id, sim in selected_clusters:
                    members = self.cluster_index.get_cluster_members(cluster_id)
                    cluster_member_ids.extend(str(m) for m in members)

                logger.debug(
                    f"Hierarchical search: {len(selected_clusters)} clusters, "
                    f"{len(cluster_member_ids)} candidates (ne_gain={ne_gain:.2f})"
                )
                add_span_attribute("hierarchical.clusters_selected", len(selected_clusters))
                add_span_attribute("hierarchical.candidate_count", len(cluster_member_ids))
            except Exception as e:
                logger.warning(f"Hierarchical search failed, falling back to flat: {e}")
                use_hierarchical = False
                cluster_member_ids = []

        # Search in Qdrant (with cluster filter if hierarchical)
        if use_hierarchical and cluster_member_ids:
            # Two-stage search: filter to cluster members first
            search_filter = filter_dict.copy() if filter_dict else {}
            # Add cluster member filter (Qdrant "should" filter for UUID match)
            # Note: This assumes episodes are indexed by their UUID string
            results = await self.vector_store.search(
                collection=self.vector_store.episodes_collection,
                vector=query_emb,
                limit=min(limit * 2, len(cluster_member_ids)),  # Over-fetch then filter
                score_threshold=modulated_threshold,
                filter=search_filter if search_filter else None,
            )
            # Filter to cluster members (secondary filter in case Qdrant doesn't support ID filter)
            cluster_member_set = set(cluster_member_ids)
            results = [
                (rid, score, payload) for rid, score, payload in results
                if rid in cluster_member_set
            ][:limit]
        else:
            # Fall back to flat search
            results = await self.vector_store.search(
                collection=self.vector_store.episodes_collection,
                vector=query_emb,
                limit=limit,
                score_threshold=modulated_threshold,
                filter=filter_dict if filter_dict else None,
            )

        # Convert results to Episode objects with recency scoring
        scored_results = []
        episode_ids = []
        now = datetime.now()

        # R1: Compute query-dependent fusion weights
        if self._learned_fusion_enabled:
            try:
                fusion_weights = self.learned_fusion.compute_weights(query_emb_np)
                semantic_w, recency_w, outcome_w, importance_w = fusion_weights
                add_span_attribute("fusion.semantic_w", float(semantic_w))
                add_span_attribute("fusion.recency_w", float(recency_w))
                logger.debug(
                    f"Learned fusion: sem={semantic_w:.2f}, rec={recency_w:.2f}, "
                    f"out={outcome_w:.2f}, imp={importance_w:.2f}"
                )
            except Exception as e:
                logger.warning(f"Learned fusion failed, using defaults: {e}")
                semantic_w, recency_w = self.semantic_weight, self.recency_weight
                outcome_w, importance_w = self.outcome_weight, self.importance_weight
        else:
            # Use fixed weights
            semantic_w, recency_w = self.semantic_weight, self.recency_weight
            outcome_w, importance_w = self.outcome_weight, self.importance_weight

        for result_id, semantic_score, payload in results:
            if not payload:
                continue

            episode = self._from_payload(result_id, payload)

            # Apply time range filter if specified
            if time_start and episode.timestamp and episode.timestamp < time_start:
                continue
            if time_end and episode.timestamp and episode.timestamp > time_end:
                continue

            # Calculate recency score
            recency_score = 1.0
            if episode.timestamp:
                time_diff = (now - episode.timestamp).total_seconds()
                time_diff_days = time_diff / 86400.0  # Convert to days
                recency_score = math.exp(-self.recency_decay * time_diff_days)

            # Calculate outcome score
            outcome_score = {
                Outcome.SUCCESS: 1.0,
                Outcome.PARTIAL: 0.5,
                Outcome.NEUTRAL: 0.3,
                Outcome.FAILURE: 0.1,
            }.get(episode.outcome, 0.3)

            # Calculate importance score (emotional valence)
            importance_score = episode.emotional_valence

            # R1: Combined weighted score with learned weights
            combined_score = (
                semantic_w * semantic_score +
                recency_w * recency_score +
                outcome_w * outcome_score +
                importance_w * importance_score
            )

            # Track component scores for transparency and training
            components = {
                "semantic": semantic_score,
                "recency": recency_score,
                "outcome": outcome_score,
                "importance": importance_score,
            }

            scored_results.append(
                ScoredResult(item=episode, score=combined_score, components=components)
            )
            episode_ids.append(episode.id)

        # Sort by combined score (descending)
        scored_results.sort(key=lambda x: x.score, reverse=True)

        # P6.4: Apply FF retrieval scoring for pattern confidence boost
        # High FF goodness = confident pattern match = more reliable memory
        if self._ff_retrieval_enabled and scored_results:
            try:
                ff_retrieval_scorer = self._bridge_container.get_ff_retrieval_scorer()
                if ff_retrieval_scorer is not None:
                    # Score using query embedding to compute goodness
                    # This provides a query-aware confidence boost
                    query_score = ff_retrieval_scorer.score_candidate(
                        embedding=query_emb_np,
                        memory_id="query",
                        use_cache=False,
                    )

                    # Apply boost based on query confidence
                    # High query goodness = familiar query type = boost similar retrievals
                    if query_score.is_confident:
                        query_boost = query_score.boost * 0.5  # Scale for retrievals
                        for result in scored_results:
                            result.score += query_boost
                            result.components["ff_confidence"] = query_score.confidence
                            result.components["ff_boost"] = query_boost

                        logger.debug(
                            f"P6.4: FF retrieval boost applied - "
                            f"query_confidence={query_score.confidence:.3f}, "
                            f"boost={query_boost:.3f}"
                        )
                        add_span_attribute("ff_retrieval.confidence", query_score.confidence)
                        add_span_attribute("ff_retrieval.boost", query_boost)
            except Exception as e:
                logger.warning(f"P6.4: FF retrieval scoring failed: {e}")

        # P6.2 + Phase 6: Capsule pose agreement scoring
        # Phase 6 enhancement: Use pose agreement between query and stored memories
        # Strong pose agreement = consistent part-whole relationship = better match
        if self._capsule_retrieval_enabled and scored_results:
            try:
                # First try the new Phase 6 local capsule layer
                if self._capsule_layer_enabled and self._capsule_layer is not None:
                    # Phase 6: Compute query capsule representation with routing
                    # learn_poses=False for retrieval (don't modify weights on queries)
                    query_activations, query_poses, _ = self._capsule_layer.forward_with_routing(
                        query_emb_np,
                        learn_poses=False,  # Don't learn during retrieval
                    )

                    for result in scored_results:
                        # Get stored capsule data from payload if available
                        stored_activations = None
                        stored_poses = None

                        # Try to get capsule data from the raw result payload
                        if hasattr(result, 'item') and hasattr(result.item, '__dict__'):
                            payload = getattr(result.item, '_payload', None) or {}
                        elif hasattr(result, 'payload'):
                            payload = result.payload or {}
                        else:
                            payload = {}

                        if "capsule_activations" in payload:
                            stored_activations = np.array(payload["capsule_activations"])
                        if "capsule_poses" in payload:
                            # Reshape from flat list back to [num_caps, pose_dim, pose_dim]
                            stored_poses_flat = np.array(payload["capsule_poses"])
                            num_caps = self._capsule_layer.config.num_capsules
                            pose_dim = self._capsule_layer.config.pose_dim
                            if len(stored_poses_flat) == num_caps * pose_dim * pose_dim:
                                stored_poses = stored_poses_flat.reshape(num_caps, pose_dim, pose_dim)

                        # Compute agreement if we have stored data
                        pose_agreement = 0.0
                        activation_sim = 0.0

                        if stored_activations is not None:
                            # Activation similarity (cosine)
                            norm_q = np.linalg.norm(query_activations) + 1e-8
                            norm_s = np.linalg.norm(stored_activations) + 1e-8
                            activation_sim = float(np.dot(query_activations, stored_activations) / (norm_q * norm_s))

                        if stored_poses is not None and query_poses is not None:
                            # Pose agreement: average agreement across capsules
                            # Uses Frobenius distance between pose matrices
                            diff = query_poses - stored_poses
                            distances = np.linalg.norm(diff.reshape(len(query_poses), -1), axis=1)
                            max_distance = 2 * self._capsule_layer.config.pose_dim
                            agreements = 1.0 - np.minimum(distances / max_distance, 1.0)
                            pose_agreement = float(np.mean(agreements))

                        # Combined capsule boost (activation similarity + pose agreement)
                        # Weight: 40% activation, 60% pose (poses are more semantically meaningful)
                        combined_capsule_score = 0.4 * max(activation_sim, 0) + 0.6 * pose_agreement
                        capsule_boost = min(combined_capsule_score * 0.2, 0.15)  # Cap boost at 0.15

                        # Phase 2C: Use stored routing agreement for confidence scoring
                        # Per Hinton: Higher routing agreement = more coherent memory = higher confidence
                        stored_routing_agreement = payload.get("capsule_routing_agreement")
                        if stored_routing_agreement is not None:
                            # Confidence multiplier: 0.7 + 0.3 * agreement (range: 0.7 to 1.0)
                            # High agreement boosts confidence, low agreement reduces it
                            confidence_multiplier = 0.7 + 0.3 * min(max(stored_routing_agreement, 0.0), 1.0)
                            capsule_boost *= confidence_multiplier
                            result.components["routing_agreement"] = stored_routing_agreement
                            result.components["confidence_multiplier"] = confidence_multiplier

                        result.score += capsule_boost
                        result.components["capsule_activation_sim"] = activation_sim
                        result.components["capsule_pose_agreement"] = pose_agreement
                        result.components["capsule_boost"] = capsule_boost

                    # Log summary
                    mean_agreement = np.mean([
                        r.components.get("capsule_pose_agreement", 0) for r in scored_results
                    ])
                    logger.debug(
                        f"Phase 6: Capsule pose agreement applied - "
                        f"mean_agreement={mean_agreement:.3f}"
                    )
                    add_span_attribute("capsule.mean_pose_agreement", mean_agreement)

                # Fallback to bridge-based scoring if no local capsule layer
                else:
                    capsule_bridge = self._bridge_container.get_capsule_bridge()
                    if capsule_bridge is not None:
                        query_repr = capsule_bridge.compute_capsule_representation(query_emb_np)
                        if query_repr is not None and query_repr.activations is not None:
                            mean_activation = float(np.mean(query_repr.activations))
                            capsule_boost = min(mean_activation * capsule_bridge.config.max_boost, 0.2)
                            for result in scored_results:
                                result.score += capsule_boost
                                result.components["capsule_activation"] = mean_activation
                                result.components["capsule_boost"] = capsule_boost
                            logger.debug(
                                f"P6.2: Capsule bridge boost applied - "
                                f"mean_activation={mean_activation:.3f}"
                            )
            except Exception as e:
                logger.warning(f"Phase 6/P6.2: Capsule retrieval scoring failed: {e}")

        # Re-sort after FF/Capsule boosts
        if (self._ff_retrieval_enabled or self._capsule_retrieval_enabled) and scored_results:
            scored_results.sort(key=lambda x: x.score, reverse=True)

        # P0c: Apply learned re-ranking if enabled
        # Re-ranks results using cross-component interactions and query context
        if self._reranking_enabled and scored_results:
            try:
                scored_results = self.learned_reranker.rerank(
                    scored_results=scored_results,
                    query_embedding=query_emb_np,
                )
                if self.learned_reranker.n_updates >= self.learned_reranker.cold_start_threshold:
                    logger.debug(
                        f"Reranking applied: n_updates={self.learned_reranker.n_updates}, "
                        f"residual_weight={self.learned_reranker.residual_weight:.2f}"
                    )
                    add_span_attribute("rerank.enabled", True)
                    add_span_attribute("rerank.n_updates", self.learned_reranker.n_updates)
            except Exception as e:
                logger.warning(f"Learned reranking failed: {e}")

        # Apply limit after sorting
        scored_results = scored_results[:limit]

        # Probe buffer for matching items (gathers implicit evidence)
        # Per Hinton: Buffer items that match recall queries are proving utility
        # This is how evidence accumulates without explicit feedback
        if self._buffering_enabled and self.buffer_manager.size > 0:
            try:
                buffer_matches = self.buffer_manager.probe(
                    query_embedding=query_emb_np,
                    threshold=0.6,
                    limit=5
                )
                # Note: probe() automatically accumulates evidence for matches
                # We could optionally include buffer items in results (with flag)
                # For now, just log that we found matches
                if buffer_matches:
                    logger.debug(
                        f"Buffer probe found {len(buffer_matches)} matches, "
                        f"evidence accumulated"
                    )
                    add_span_attribute("buffer.probe_matches", len(buffer_matches))
            except Exception as e:
                logger.warning(f"Buffer probe failed: {e}")

        # Apply inhibitory dynamics through neuromodulator orchestra
        # GABA-like lateral inhibition sharpens the score distribution
        # Winner-take-all dynamics: top memories suppress weaker ones
        if self._neuromodulation_enabled and scored_results:
            try:
                # Prepare scores for inhibition
                scores_dict = {str(r.item.id): r.score for r in scored_results}
                retrieved_ids = [r.item.id for r in scored_results]

                # Process through orchestra (adds eligibility traces, applies inhibition)
                sharpened_scores = self.orchestra.process_retrieval(
                    retrieved_ids=retrieved_ids,
                    scores=scores_dict,
                    embeddings=None  # Could add embeddings for similarity-based inhibition
                )

                # Update scores with inhibited values
                for result in scored_results:
                    id_str = str(result.item.id)
                    if id_str in sharpened_scores:
                        # Track original and sharpened
                        original = result.score
                        result.score = sharpened_scores[id_str]
                        result.components["inhibited"] = True
                        result.components["pre_inhibition"] = original

                # Re-sort after inhibition
                scored_results.sort(key=lambda x: x.score, reverse=True)

                logger.debug(
                    f"Inhibitory dynamics applied: "
                    f"sparsity={self.orchestra.get_current_state().inhibition_sparsity:.2f}"
                    if self.orchestra.get_current_state() else "state unavailable"
                )
            except Exception as e:
                logger.warning(f"Inhibitory dynamics failed: {e}")

        # Update access tracking for recalled episodes (only those returned)
        episode_ids_to_update = [r.item.id for r in scored_results]
        if episode_ids_to_update:
            await self._batch_update_access(episode_ids_to_update, success=True)

            # PHASE-2 FIX: Mark retrieved memories as active for eligibility traces
            # This enables temporal credit assignment when outcomes arrive later
            # Activity level scales with retrieval score (more relevant = more active)
            for result in scored_results:
                memory_id_str = str(result.item.id)
                # Scale activity by retrieval score (0.5-1.0 range)
                activity = 0.5 + 0.5 * result.score
                self.three_factor.mark_active(memory_id_str, activity=activity)

            # Trigger lability for retrieved memories (reconsolidation window opens)
            for eid in episode_ids_to_update:
                self.reconsolidation.trigger_lability(eid)

            # Implicit retrieval feedback for learned gate
            # Per Hinton/AGI analysis: retrieval = revealed preference = positive signal
            # This closes the implicit feedback loop without waiting for explicit outcomes
            if self._gating_enabled:
                try:
                    implicit_utility = 0.6  # Moderate positive (retrieval = somewhat useful)
                    implicit_updated = 0
                    for eid in episode_ids_to_update:
                        # Only update if still in pending labels (was gated in)
                        if eid in self.learned_gate.pending_labels:
                            self.learned_gate.update(eid, implicit_utility)
                            implicit_updated += 1
                    if implicit_updated > 0:
                        logger.debug(
                            f"Applied implicit gate feedback for {implicit_updated} retrievals"
                        )
                except Exception as e:
                    logger.debug(f"Implicit gate feedback failed: {e}")

        add_span_attribute("recall.neuromodulation_enabled", self._neuromodulation_enabled)
        if neuro_state:
            add_span_attribute("recall.ach_mode", neuro_state.acetylcholine_mode)
            add_span_attribute("recall.ne_gain", neuro_state.norepinephrine_gain)

        return scored_results

    @traced("episodic.recall_hybrid", kind=SpanKind.INTERNAL)
    async def recall_hybrid(
        self,
        query: str,
        limit: int = 10,
        session_filter: str | None = None,
        time_start: datetime | None = None,
        time_end: datetime | None = None,
    ) -> list[ScoredResult]:
        """
        Retrieve episodes using hybrid search (dense + sparse vectors).

        Uses Qdrant's native RRF fusion for combining semantic similarity
        with lexical matching. Falls back to dense-only if hybrid not available.

        Args:
            query: Search query
            limit: Maximum results to return
            session_filter: Optional session filter
            time_start: Optional start time filter
            time_end: Optional end time filter

        Returns:
            List of scored episodes
        """
        # Fall back to dense-only recall if hybrid not initialized
        if not self._hybrid_initialized:
            logger.debug("Hybrid not initialized, falling back to dense recall")
            return await self.recall(query, limit, session_filter, time_start, time_end)

        # Generate hybrid query embedding (dense + sparse)
        dense_emb, sparse_emb = await self.embedding.embed_query_hybrid(query)

        # Build filter dict
        filter_dict = {}

        # Determine session filter to use
        session_id = session_filter or (self.session_id if self.session_id != "default" else None)
        if session_id:
            filter_dict["session_id"] = session_id

        # Add time range filter if specified
        if time_start:
            filter_dict["timestamp"] = filter_dict.get("timestamp", {})
            filter_dict["timestamp"]["gte"] = time_start.timestamp()
        if time_end:
            filter_dict["timestamp"] = filter_dict.get("timestamp", {})
            filter_dict["timestamp"]["lte"] = time_end.timestamp()

        # Search in hybrid collection using RRF fusion
        collection = self.vector_store.episodes_collection + "_hybrid"
        results = await self.vector_store.search_hybrid(
            collection=collection,
            dense_vector=dense_emb,
            sparse_vector=sparse_emb,
            limit=limit,
            filter=filter_dict if filter_dict else None,
            score_threshold=0.1,  # Lower threshold for RRF scores
        )

        # Convert results to Episode objects with recency scoring
        scored_results = []
        episode_ids = []
        now = datetime.now()

        for result_id, rrf_score, payload in results:
            if not payload:
                continue

            episode = self._from_payload(result_id, payload)

            # Calculate recency score
            recency_score = 1.0
            if episode.timestamp:
                time_diff = (now - episode.timestamp).total_seconds()
                time_diff_days = time_diff / 86400.0
                recency_score = math.exp(-self.recency_decay * time_diff_days)

            # Calculate outcome score
            outcome_score = {
                Outcome.SUCCESS: 1.0,
                Outcome.PARTIAL: 0.5,
                Outcome.NEUTRAL: 0.3,
                Outcome.FAILURE: 0.1,
            }.get(episode.outcome, 0.3)

            # Calculate importance score
            importance_score = episode.emotional_valence

            # Combined weighted score (RRF score replaces pure semantic)
            combined_score = (
                self.semantic_weight * rrf_score +
                self.recency_weight * recency_score +
                self.outcome_weight * outcome_score +
                self.importance_weight * importance_score
            )

            # Track component scores for transparency
            components = {
                "rrf": rrf_score,  # Combined dense+sparse via RRF
                "recency": recency_score,
                "outcome": outcome_score,
                "importance": importance_score,
            }

            scored_results.append(
                ScoredResult(item=episode, score=combined_score, components=components)
            )
            episode_ids.append(episode.id)

        # Sort by combined score (descending)
        scored_results.sort(key=lambda x: x.score, reverse=True)

        # Apply limit after sorting
        scored_results = scored_results[:limit]

        # Update access tracking
        episode_ids_to_update = [r.item.id for r in scored_results]
        if episode_ids_to_update:
            await self._batch_update_access(episode_ids_to_update, success=True)

            # Implicit retrieval feedback for learned gate (same as recall())
            if self._gating_enabled:
                try:
                    implicit_utility = 0.6
                    for eid in episode_ids_to_update:
                        if eid in self.learned_gate.pending_labels:
                            self.learned_gate.update(eid, implicit_utility)
                except Exception as e:
                    logger.debug(f"Implicit gate feedback failed: {e}")

        logger.debug(f"Hybrid recall returned {len(scored_results)} results")
        return scored_results

    async def recall_by_timerange(
        self,
        start_time: datetime,
        end_time: datetime,
        page_size: int = 100,
        cursor: str | None = None,
        session_filter: str | None = None,
    ) -> tuple[list[Episode], str | None]:
        """
        Retrieve episodes within a time range with pagination.

        More efficient than embedding-based recall for bulk operations
        like consolidation that need all episodes in a time window.

        Args:
            start_time: Start of time window (inclusive)
            end_time: End of time window (inclusive)
            page_size: Number of episodes per page (max 500)
            cursor: Pagination cursor from previous call
            session_filter: Optional session ID filter

        Returns:
            Tuple of (episodes, next_cursor). next_cursor is None if no more pages.
        """
        page_size = min(page_size, 500)  # Cap page size

        # Parse cursor to get offset
        offset = 0
        if cursor:
            try:
                offset = int(cursor)
            except ValueError:
                logger.warning(f"Invalid cursor: {cursor}, starting from 0")
                offset = 0

        # Convert datetime to Unix timestamps for Qdrant Range filter
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()

        # Build scroll filter for Qdrant
        filter_conditions = {
            "timestamp": {
                "gte": start_timestamp,
                "lte": end_timestamp,
            }
        }

        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            # Default to current session
            filter_conditions["session_id"] = self.session_id

        # Use Qdrant scroll for pagination
        try:
            results, next_offset = await self.vector_store.scroll(
                collection=self.vector_store.episodes_collection,
                scroll_filter=filter_conditions,
                limit=page_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't need vectors for consolidation
            )

            episodes = []
            for id_str, payload, _ in results:
                episode = self._from_payload(id_str, payload)
                episodes.append(episode)

            # Determine next cursor
            next_cursor = None
            if next_offset and len(episodes) == page_size:
                next_cursor = str(next_offset)

            logger.debug(
                f"Retrieved {len(episodes)} episodes for timerange "
                f"{start_time.isoformat()[:10]} to {end_time.isoformat()[:10]}, "
                f"offset={offset}"
            )

            return episodes, next_cursor

        except Exception as e:
            logger.error(f"Error in recall_by_timerange: {e}")
            raise

    async def count_by_timerange(
        self,
        start_time: datetime,
        end_time: datetime,
        session_filter: str | None = None,
    ) -> int:
        """
        Count episodes in a time range without fetching them.

        Args:
            start_time: Start of time window (inclusive)
            end_time: End of time window (inclusive)
            session_filter: Optional session ID filter

        Returns:
            Count of matching episodes
        """
        # Convert datetime to Unix timestamps for Qdrant Range filter
        start_timestamp = start_time.timestamp()
        end_timestamp = end_time.timestamp()

        filter_conditions = {
            "timestamp": {
                "gte": start_timestamp,
                "lte": end_timestamp,
            }
        }

        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            filter_conditions["session_id"] = self.session_id

        return await self.vector_store.count(
            collection=self.vector_store.episodes_collection,
            count_filter=filter_conditions,
        )

    async def recent(
        self,
        limit: int = 20,
        session_filter: str | None = None,
    ) -> list[Episode]:
        """
        Get most recent episodes ordered by timestamp descending.

        Args:
            limit: Maximum number of episodes to return (default 20, max 500)
            session_filter: Optional session ID filter

        Returns:
            List of episodes ordered by most recent first
        """
        limit = min(limit, 500)  # Cap limit

        # Build filter for current session
        filter_conditions = {}
        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            filter_conditions["session_id"] = self.session_id

        try:
            # Use scroll to get episodes, then sort by timestamp
            results, _ = await self.vector_store.scroll(
                collection=self.vector_store.episodes_collection,
                scroll_filter=filter_conditions if filter_conditions else None,
                limit=limit,
                offset=0,
                with_payload=True,
                with_vectors=False,
            )

            episodes = []
            for id_str, payload, _ in results:
                episode = self._from_payload(id_str, payload)
                episodes.append(episode)

            # Sort by timestamp descending (most recent first)
            episodes.sort(key=lambda e: e.timestamp, reverse=True)

            logger.debug(f"Retrieved {len(episodes)} recent episodes")
            return episodes[:limit]

        except Exception as e:
            logger.error(f"Error in recent: {e}")
            raise

    async def get(self, episode_id: UUID) -> Episode | None:
        """
        Get episode by ID.

        Args:
            episode_id: Episode UUID

        Returns:
            Episode or None

        Raises:
            TypeError: If episode_id is not a valid UUID
        """
        # DATA-006 FIX: Validate type
        episode_id = _validate_uuid(episode_id, "episode_id")

        results = await self.vector_store.get(
            collection=self.vector_store.episodes_collection,
            ids=[str(episode_id)],
        )

        if results:
            id_str, payload = results[0]
            return self._from_payload(id_str, payload)

        return None

    async def get_episode_sequence(
        self,
        episode_id: UUID,
        before: int = 5,
        after: int = 5,
    ) -> list[Episode]:
        """
        P5.2: Get episodes in sequence around a given episode.

        Retrieves the temporal context of an episode by walking the
        sequence links forward and backward.

        Args:
            episode_id: Central episode UUID
            before: Number of preceding episodes to retrieve
            after: Number of following episodes to retrieve

        Returns:
            List of episodes in temporal order (earliest first)
        """
        episode_id = _validate_uuid(episode_id, "episode_id")

        result = []
        center_episode = await self.get(episode_id)
        if not center_episode:
            return result

        # Walk backward to get preceding episodes
        preceding = []
        current = center_episode
        for _ in range(before):
            if current.previous_episode_id is None:
                break
            prev_ep = await self.get(current.previous_episode_id)
            if prev_ep is None:
                break
            preceding.append(prev_ep)
            current = prev_ep

        # Walk forward to get following episodes
        following = []
        current = center_episode
        for _ in range(after):
            if current.next_episode_id is None:
                break
            next_ep = await self.get(current.next_episode_id)
            if next_ep is None:
                break
            following.append(next_ep)
            current = next_ep

        # Combine in temporal order: preceding (reversed) + center + following
        result = list(reversed(preceding)) + [center_episode] + following
        return result

    async def get_session_timeline(
        self,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Episode]:
        """
        P5.2: Get all episodes in a session in temporal order.

        Uses sequence_position for ordering when available, falls back
        to timestamp ordering.

        Args:
            session_id: Session to retrieve (defaults to current session)
            limit: Maximum episodes to return

        Returns:
            List of episodes in temporal order (earliest first)
        """
        session_id = session_id or self.session_id
        limit = min(limit, 1000)

        # Build filter for session
        filter_conditions = {"session_id": session_id}

        try:
            results, _ = await self.vector_store.scroll(
                collection=self.vector_store.episodes_collection,
                scroll_filter=filter_conditions,
                limit=limit,
                offset=0,
                with_payload=True,
                with_vectors=False,
            )

            episodes = []
            for id_str, payload, _ in results:
                episode = self._from_payload(id_str, payload)
                episodes.append(episode)

            # Sort by sequence_position if available, else by timestamp
            def sort_key(e: Episode):
                if e.sequence_position is not None:
                    return (0, e.sequence_position)
                return (1, e.timestamp)

            episodes.sort(key=sort_key)
            return episodes

        except Exception as e:
            logger.error(f"Error getting session timeline: {e}")
            raise

    async def mark_important(
        self,
        episode_id: UUID,
        new_valence: float | None = None,
    ) -> Episode:
        """
        Mark an episode as important by increasing its emotional valence.

        Uses saga pattern to ensure atomicity across dual stores.

        Args:
            episode_id: Episode UUID to mark
            new_valence: Explicit valence value [0,1], or None to increase by 0.2

        Returns:
            Updated Episode

        Raises:
            TypeError: If episode_id is not a valid UUID
            ValueError: If episode not found
            RuntimeError: If update fails (after rollback)
        """
        # DATA-006 FIX: Validate type
        episode_id = _validate_uuid(episode_id, "episode_id")

        # Fetch current episode
        results = await self.vector_store.get(
            collection=self.vector_store.episodes_collection,
            ids=[str(episode_id)],
        )

        if not results:
            raise ValueError(f"Episode {episode_id} not found")

        id_str, payload = results[0]

        # Store original payload for potential rollback
        original_payload = dict(payload)

        # Calculate new valence
        current_valence = payload.get("emotional_valence", 0.5)
        if new_valence is not None:
            updated_valence = new_valence
        else:
            updated_valence = current_valence + 0.2

        # Clamp to [0, 1]
        updated_valence = max(0.0, min(1.0, updated_valence))

        # Update payload
        payload["emotional_valence"] = updated_valence

        # Step 1: Update in vector store
        await self.vector_store.update_payload(
            collection=self.vector_store.episodes_collection,
            id=id_str,
            payload=payload,
        )

        # Step 2: Update in graph store (with compensation on failure)
        try:
            await self.graph_store.update_node(
                node_id=id_str,
                properties={"valence": updated_valence},
            )
        except Exception as e:
            # Compensate: revert Qdrant to original payload
            logger.warning(f"Graph update failed for {episode_id}, rolling back: {e}")
            await self.vector_store.update_payload(
                collection=self.vector_store.episodes_collection,
                id=id_str,
                payload=original_payload,
            )
            raise RuntimeError(f"mark_important failed: {e}") from e

        return self._from_payload(id_str, payload)

    @traced("episodic.query_at_time", kind=SpanKind.INTERNAL)
    async def query_at_time(
        self,
        query: str,
        point_in_time: datetime,
        limit: int = 10,
        recency_weight: float = 0.3,
    ) -> list[ScoredResult]:
        """
        Query episodes as they existed at a specific point in time.

        Returns episodes that existed at or before the given point in time,
        with recency weighting based on temporal distance from that time.

        Args:
            query: Search query
            point_in_time: The historical point in time to query
            limit: Maximum results to return
            recency_weight: Weight for recency scoring (0.0-1.0)

        Returns:
            List of scored episodes with semantic and recency components
        """
        # Generate query embedding
        query_emb = await self.embedding.embed_query(query)

        # Determine session filter to use
        session_id = self.session_id if self.session_id != "default" else None

        # Search in Qdrant with time filter (only episodes that existed at that time)
        results = await self.vector_store.search(
            collection=self.vector_store.episodes_collection,
            vector=query_emb,
            limit=limit * 2,  # Fetch more since we'll filter by time
            score_threshold=0.5,
            session_id=session_id,
        )

        # Convert results to Episode objects and calculate temporal scores
        scored_results = []

        for result_id, semantic_score, payload in results:
            if not payload:
                continue

            # Parse episode timestamp
            timestamp_str = payload.get("timestamp")
            if not timestamp_str:
                continue

            episode_timestamp = datetime.fromisoformat(timestamp_str)

            # Filter out episodes created after the point_in_time
            if episode_timestamp > point_in_time:
                continue

            # Calculate recency score based on distance from point_in_time
            time_diff = (point_in_time - episode_timestamp).total_seconds()
            time_diff_days = time_diff / 86400.0  # Convert to days

            # Exponential decay: closer to point_in_time = higher score
            # Using similar decay pattern to FSRS but relative to query time
            recency_score = math.exp(-self.recency_decay * time_diff_days)

            # Combined score
            combined_score = (
                (1.0 - recency_weight) * semantic_score +
                recency_weight * recency_score
            )

            # Reconstruct episode
            episode = self._from_payload(result_id, payload)

            # Track component scores for transparency
            components = {
                "semantic": semantic_score,
                "recency": recency_score,
            }

            scored_results.append(
                ScoredResult(item=episode, score=combined_score, components=components)
            )

        # Sort by combined score (descending)
        scored_results.sort(key=lambda x: x.score, reverse=True)

        return scored_results[:limit]

    async def _batch_update_access(
        self,
        episode_ids: list[UUID],
        success: bool = True,
    ) -> int:
        """
        Update access tracking for multiple episodes in batch.

        Updates stability, access_count, and last_accessed for recalled episodes.
        Used by recall methods to track memory reinforcement via FSRS.

        Args:
            episode_ids: List of episode UUIDs to update
            success: Whether recall was successful (affects stability)

        Returns:
            Number of episodes updated
        """
        if not episode_ids:
            return 0

        try:
            # Fetch current episode payloads
            id_strings = [str(eid) for eid in episode_ids]
            results = await self.vector_store.get(
                collection=self.vector_store.episodes_collection,
                ids=id_strings,
            )

            if not results:
                return 0

            # Prepare batch updates
            now = datetime.now()
            updates = []

            for id_str, payload in results:
                # Calculate new stability
                current_stability = payload.get("stability", 1.0)
                if success:
                    # Increase stability on successful recall (bounded growth)
                    new_stability = current_stability + 0.1 * (2.0 - current_stability)
                else:
                    # Decrease stability on failed recall
                    new_stability = current_stability * 0.8

                # Prepare updated payload
                updated_payload = {
                    **payload,
                    "stability": new_stability,
                    "access_count": payload.get("access_count", 0) + 1,
                    "last_accessed": now.isoformat(),
                }

                updates.append((id_str, updated_payload))

            # Batch update payloads
            updated_count = await self.vector_store.batch_update_payloads(
                collection=self.vector_store.episodes_collection,
                updates=updates,
            )

            logger.debug(f"Batch updated {updated_count} episodes (success={success})")
            return updated_count

        except Exception as e:
            logger.warning(f"Error in batch access update: {e}")
            return 0

    @traced("episodic.apply_reconsolidation", kind=SpanKind.INTERNAL)
    async def apply_reconsolidation(
        self,
        episode_ids: list[UUID],
        query: str,
        outcome_score: float,
        per_memory_rewards: dict[str, float] | None = None,
        use_dopamine: bool = True,
    ) -> int:
        """
        Apply reconsolidation to retrieved memories based on retrieval outcome.

        Per Hinton critique: Embeddings should update based on retrieval outcomes,
        not remain frozen after initial encoding. Positive outcomes pull the
        memory embedding toward the query; negative outcomes push it away.

        This implements biological reconsolidation: retrieved memories become
        labile and are re-encoded with new contextual information.

        Dopamine integration (use_dopamine=True):
        - Computes reward prediction error (δ = actual - expected)
        - Modulates learning rate by surprise magnitude
        - Updates value estimates for future predictions
        - Unexpected outcomes drive more learning than expected ones

        Args:
            episode_ids: List of episode UUIDs that were retrieved
            query: The query that retrieved these memories
            outcome_score: Success score [0, 1] (0.5 = neutral)
            per_memory_rewards: Optional per-memory reward overrides
            use_dopamine: Whether to use dopamine RPE modulation

        Returns:
            Number of episodes that were reconsolidated
        """
        if not self._reconsolidation_enabled:
            return 0

        if not episode_ids:
            return 0

        try:
            # Generate query embedding
            query_emb = await self.embedding.embed_query(query)
            query_emb_np = np.array(query_emb)

            # Fetch current embeddings for the episodes
            id_strings = [str(eid) for eid in episode_ids]
            results = await self.vector_store.get_with_vectors(
                collection=self.vector_store.episodes_collection,
                ids=id_strings,
            )

            if not results:
                return 0

            # Prepare batch of memories for reconsolidation
            # Also compute importance from stability and emotional_valence
            # Higher importance = smaller updates (catastrophic forgetting protection)
            memories = []
            per_memory_importance = {}
            per_memory_lr_modulation = {}  # Dopamine surprise modulation

            for id_str, payload, vector in results:
                if vector is not None:
                    mem_uuid = UUID(id_str)
                    memories.append((mem_uuid, np.array(vector)))

                    # Importance = stability * (0.5 + valence)
                    # High stability + high valence = very important = small updates
                    stability = payload.get("stability", 1.0)
                    valence = payload.get("emotional_valence", 0.5)
                    importance = stability * (0.5 + valence)
                    per_memory_importance[id_str] = importance

                    # Dopamine RPE modulation: surprise drives learning
                    if use_dopamine:
                        actual = per_memory_rewards.get(id_str, outcome_score) if per_memory_rewards else outcome_score
                        rpe = self.dopamine.compute_rpe(mem_uuid, actual)
                        # Surprise modulation: |δ| scales learning rate
                        # Higher surprise = more learning needed
                        per_memory_lr_modulation[id_str] = rpe.surprise_magnitude
                        # Update expectations for next time
                        self.dopamine.update_expectations(mem_uuid, actual)
                    else:
                        per_memory_lr_modulation[id_str] = 1.0

            # Apply reconsolidation with importance-weighted and surprise-modulated learning rates
            updates = self.reconsolidation.batch_reconsolidate(
                memories=memories,
                query_embedding=query_emb_np,
                outcome_score=outcome_score,
                per_memory_rewards=per_memory_rewards,
                per_memory_importance=per_memory_importance,
                per_memory_lr_modulation=per_memory_lr_modulation,
            )

            if not updates:
                return 0

            # Persist updated embeddings to Qdrant
            vector_updates = [
                (str(mem_id), new_emb.tolist())
                for mem_id, new_emb in updates.items()
            ]

            updated_count = await self.vector_store.batch_update_vectors(
                collection=self.vector_store.episodes_collection,
                updates=vector_updates,
            )

            logger.info(
                f"Reconsolidated {updated_count} episodes: "
                f"outcome={outcome_score:.2f}, query='{query[:50]}...'"
            )

            add_span_attribute("reconsolidation.count", updated_count)
            add_span_attribute("reconsolidation.outcome", outcome_score)

            return updated_count

        except Exception as e:
            logger.warning(f"Error in reconsolidation: {e}")
            return 0

    def get_reconsolidation_stats(self) -> dict:
        """Get reconsolidation statistics for monitoring."""
        return self.reconsolidation.get_stats()

    def get_dopamine_stats(self) -> dict:
        """Get dopamine system statistics for monitoring."""
        return self.dopamine.get_stats()

    @traced("episodic.learn_from_outcome", kind=SpanKind.INTERNAL)
    async def learn_from_outcome(
        self,
        episode_ids: list[UUID],
        query: str,
        outcome_score: float,
        neural_scores: dict[str, float] | None = None,
        symbolic_scores: dict[str, float] | None = None,
        recency_scores: dict[str, float] | None = None,
        per_memory_rewards: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Complete learning loop: reconsolidation + fusion training.

        This is the critical missing piece per Hinton critique:
        "The system has all the components for online learning but
        the loop is not closed... train_fusion_step() exists but
        nothing calls it."

        This method closes the loop by:
        1. Applying reconsolidation with dopamine RPE-modulated learning
        2. Training fusion weights with RPE-based targets (not raw outcomes)

        Why RPE not raw outcomes?
        - δ = actual - expected = surprise signal
        - Expected outcomes (δ≈0) don't require model updates
        - Surprising outcomes (|δ|>0) indicate model error → learn from these
        - This is how biological dopamine guides learning

        Args:
            episode_ids: List of episode UUIDs that were retrieved
            query: The query that retrieved these memories
            outcome_score: Overall success score [0, 1]
            neural_scores: Per-memory neural similarity scores (for fusion training)
            symbolic_scores: Per-memory symbolic relevance scores (for fusion training)
            recency_scores: Per-memory recency scores (for fusion training)
            per_memory_rewards: Optional per-memory reward overrides

        Returns:
            Dict with learning statistics:
            - reconsolidated_count: Number of embeddings updated
            - fusion_loss: Training loss (if fusion trained)
            - avg_rpe: Average reward prediction error
            - positive_surprises: Count of better-than-expected
            - negative_surprises: Count of worse-than-expected
        """
        stats = {
            "reconsolidated_count": 0,
            "fusion_loss": 0.0,
            "avg_rpe": 0.0,
            "positive_surprises": 0,
            "negative_surprises": 0,
        }

        if not episode_ids:
            return stats

        # Step 1: Apply reconsolidation with dopamine RPE modulation
        # This updates embeddings based on outcome, scaled by surprise
        reconsolidated = await self.apply_reconsolidation(
            episode_ids=episode_ids,
            query=query,
            outcome_score=outcome_score,
            per_memory_rewards=per_memory_rewards,
            use_dopamine=True
        )
        stats["reconsolidated_count"] = reconsolidated

        # Step 1.5: Process outcome through neuromodulator orchestra
        # This coordinates: dopamine (immediate RPE), serotonin (long-term credit)
        # Serotonin distributes credit across time via eligibility traces
        combined_signals: dict[str, float] = {}  # Initialize for gate training below
        if self._neuromodulation_enabled:
            try:
                memory_outcomes = {}
                for eid in episode_ids:
                    id_str = str(eid)
                    if per_memory_rewards and id_str in per_memory_rewards:
                        memory_outcomes[id_str] = per_memory_rewards[id_str]
                    else:
                        memory_outcomes[id_str] = outcome_score

                # Orchestra combines DA (immediate) and 5-HT (long-term) signals
                combined_signals = self.orchestra.process_outcome(
                    memory_outcomes=memory_outcomes,
                    session_outcome=outcome_score  # Overall session outcome for 5-HT
                )

                # Track combined learning signals
                stats["serotonin_credits"] = len(combined_signals)
                stats["orchestra_learning_signals"] = combined_signals

                logger.debug(
                    f"Orchestra outcome processing: "
                    f"{len(combined_signals)} memories received credit"
                )
            except Exception as e:
                logger.warning(f"Orchestra outcome processing failed: {e}")

        # Step 1.6: Update learned memory gate with outcomes
        # This closes the "what to remember" learning loop:
        # Storage decision → memory stored → memory retrieved → outcome observed
        # The gate learns: P(useful | features) from (features, utility) pairs
        #
        # Per Hinton/AGI analysis: Use combined DA+5HT signals instead of raw outcomes
        # - DA RPE = surprise signal (what was unexpected)
        # - 5-HT credit = long-term value (what contributed to session success)
        # - Combined = 0.7*DA + 0.3*5-HT (already computed in orchestra)
        # - Transform from [-1,1] RPE range to [0,1] utility for gate
        if self._gating_enabled:
            try:
                gate_updated = 0
                for eid in episode_ids:
                    id_str = str(eid)

                    # Prefer combined DA+5HT signals from orchestra (surprise-based learning)
                    if combined_signals and id_str in combined_signals:
                        # Combined signal is in [-1, 1] range (RPE-based)
                        # Transform to [0, 1] utility: utility = 0.5 + 0.5 * signal
                        raw_signal = combined_signals[id_str]
                        utility = 0.5 + 0.5 * np.clip(raw_signal, -1.0, 1.0)
                    elif per_memory_rewards and id_str in per_memory_rewards:
                        # Fallback to per-memory reward if specified
                        utility = per_memory_rewards[id_str]
                    else:
                        # Final fallback to raw outcome
                        utility = outcome_score

                    # Update gate model
                    self.learned_gate.update(eid, utility)
                    gate_updated += 1

                stats["gate_updated"] = gate_updated
                logger.debug(
                    f"Updated learned gate with {gate_updated} samples "
                    f"(neuromod-enhanced), n_observations={self.learned_gate.n_observations}"
                )
            except Exception as e:
                logger.warning(f"Learned gate update failed: {e}")

        # Step 1.7: Propagate outcome signals to buffer items
        # Per Hinton: Buffer items semantically related to outcome context
        # should also receive evidence (helps with implicit feedback)
        if self._buffering_enabled and self.buffer_manager.size > 0:
            try:
                # Get query embedding for similarity check
                query_emb = await self.embedding.embed_query(query)
                query_emb_np = np.array(query_emb)

                # Compute average combined signal for propagation
                if combined_signals:
                    avg_signal = float(np.mean(list(combined_signals.values())))
                else:
                    # Fallback: convert outcome_score to signal range
                    avg_signal = 2 * outcome_score - 1  # [0,1] -> [-1,1]

                # Propagate to semantically related buffer items
                self.buffer_manager.accumulate_from_outcome(
                    query_embedding=query_emb_np,
                    combined_signal=avg_signal,
                    similarity_threshold=0.5
                )
                logger.debug(
                    f"Propagated outcome signal to buffer: "
                    f"signal={avg_signal:.3f}, buffer_size={self.buffer_manager.size}"
                )
            except Exception as e:
                logger.warning(f"Buffer outcome propagation failed: {e}")

        # Step 1.8: R1 - Train learned fusion weights from outcome
        # Updates query-dependent scoring weights based on what worked
        if self._learned_fusion_enabled and (neural_scores or recency_scores):
            try:
                # Compute average component scores across retrieved memories
                avg_components = {
                    "semantic": 0.0,
                    "recency": 0.0,
                    "outcome": 0.0,
                    "importance": 0.0,
                }

                n_memories = len(episode_ids)
                for id_str in [str(eid) for eid in episode_ids]:
                    if neural_scores and id_str in neural_scores:
                        avg_components["semantic"] += neural_scores[id_str]
                    if recency_scores and id_str in recency_scores:
                        avg_components["recency"] += recency_scores[id_str]

                if n_memories > 0:
                    for k in avg_components:
                        avg_components[k] /= n_memories

                # Get query embedding for training
                query_emb = await self.embedding.embed_query(query)
                query_emb_np = np.array(query_emb)

                # Update fusion weights
                self.learned_fusion.update(
                    query_embedding=query_emb_np,
                    component_scores=avg_components,
                    outcome_utility=outcome_score
                )

                stats["fusion_updated"] = True
                stats["fusion_n_updates"] = self.learned_fusion.n_updates
                logger.debug(
                    f"Updated learned fusion weights: "
                    f"n_updates={self.learned_fusion.n_updates}, outcome={outcome_score:.2f}"
                )
            except Exception as e:
                logger.warning(f"Learned fusion update failed: {e}")

        # Step 1.9: P0c - Train learned reranker from outcome
        # Updates cross-component scoring based on what worked
        if self._reranking_enabled and (neural_scores or recency_scores):
            try:
                # Build component scores list for each retrieved memory
                component_scores_list = []
                outcome_utilities = []

                for id_str in [str(eid) for eid in episode_ids]:
                    components = {
                        "semantic": neural_scores.get(id_str, 0.5) if neural_scores else 0.5,
                        "recency": recency_scores.get(id_str, 0.5) if recency_scores else 0.5,
                        "outcome": 0.5,  # Default
                        "importance": 0.5,  # Default
                    }
                    component_scores_list.append(components)

                    # Determine utility for this memory
                    if per_memory_rewards and id_str in per_memory_rewards:
                        utility = per_memory_rewards[id_str]
                    elif combined_signals and id_str in combined_signals:
                        # Transform RPE to utility
                        utility = 0.5 + 0.5 * np.clip(combined_signals[id_str], -1.0, 1.0)
                    else:
                        utility = outcome_score
                    outcome_utilities.append(utility)

                # Get query embedding for training
                query_emb = await self.embedding.embed_query(query)
                query_emb_np = np.array(query_emb)

                # Update reranker
                self.learned_reranker.update(
                    query_embedding=query_emb_np,
                    component_scores_list=component_scores_list,
                    outcome_utilities=outcome_utilities,
                )

                stats["reranker_updated"] = True
                stats["reranker_n_updates"] = self.learned_reranker.n_updates
                logger.debug(
                    f"Updated learned reranker: "
                    f"n_updates={self.learned_reranker.n_updates}, "
                    f"residual_weight={self.learned_reranker.residual_weight:.2f}"
                )
            except Exception as e:
                logger.warning(f"Learned reranker update failed: {e}")

        # Step 1.10: Phase 1 - Update cluster statistics from retrieval outcome
        # Records success/failure for clusters that were searched
        if self._hierarchical_search_enabled and self.cluster_index.n_clusters > 0:
            try:
                # Determine which clusters were involved (find clusters containing retrieved episodes)
                involved_clusters = set()
                for eid in episode_ids:
                    for cid, cluster in self.cluster_index.clusters.items():
                        if eid in cluster.member_ids:
                            involved_clusters.add(cid)
                            break

                if involved_clusters:
                    # Record outcome for involved clusters
                    successful = outcome_score > 0.5  # Above-average outcome = success
                    self.cluster_index.record_retrieval_outcome(
                        cluster_ids=list(involved_clusters),
                        successful=successful,
                        retrieved_scores=dict.fromkeys(involved_clusters, outcome_score),
                    )

                    stats["cluster_feedback_sent"] = True
                    stats["clusters_updated"] = len(involved_clusters)
                    logger.debug(
                        f"Updated {len(involved_clusters)} cluster statistics: "
                        f"successful={successful}"
                    )
            except Exception as e:
                logger.warning(f"Cluster feedback update failed: {e}")

        # Step 1.11: Phase 2 - Update sparse index from retrieval outcome
        # Trains cluster and feature attention based on what worked
        if self._sparse_addressing_enabled and len(self.sparse_index._pending_updates) > 0:
            try:
                # Find any pending sparse index updates and apply feedback
                pending_ids = list(self.sparse_index._pending_updates.keys())
                for query_id in pending_ids[:5]:  # Process up to 5 pending
                    # Build cluster rewards from episode clusters
                    cluster_rewards = {}
                    for idx, cid in enumerate(self.cluster_index._centroid_ids):
                        cluster = self.cluster_index.clusters.get(cid)
                        if cluster:
                            # Reward based on whether cluster members were retrieved
                            for eid in episode_ids:
                                if eid in cluster.member_ids:
                                    cluster_rewards[idx] = outcome_score
                                    break

                    if cluster_rewards:
                        self.sparse_index.update(
                            query_id=query_id,
                            cluster_rewards=cluster_rewards,
                            overall_success=outcome_score > 0.5,
                        )

                stats["sparse_index_updated"] = True
                stats["sparse_index_n_updates"] = self.sparse_index.n_updates
                logger.debug(
                    f"Updated sparse index: n_updates={self.sparse_index.n_updates}, "
                    f"avg_sparsity={self.sparse_index.avg_sparsity:.3f}"
                )
            except Exception as e:
                logger.warning(f"Sparse index update failed: {e}")

        # Step 1.12: P6.4 - Train FF retrieval scorer from outcome
        # Updates FF layer to increase goodness for patterns that led to success
        # and decrease goodness for patterns that led to failure
        if self._ff_retrieval_enabled:
            try:
                ff_retrieval_scorer = self._bridge_container.get_ff_retrieval_scorer()
                if ff_retrieval_scorer is not None:
                    # Get query embedding for training
                    query_emb = await self.embedding.embed_query(query)
                    query_emb_np = np.array(query_emb)

                    # Train FF from outcome - query embedding represents the pattern
                    # that was used to retrieve these memories
                    ff_stats = ff_retrieval_scorer.learn_from_outcome(
                        embeddings=[query_emb_np],  # Query pattern
                        memory_ids=["query_" + str(episode_ids[0])],
                        outcome_score=outcome_score,
                    )

                    stats["ff_retrieval_updated"] = True
                    stats["ff_positive_learning"] = ff_stats.get("positive_learning", 0)
                    stats["ff_negative_learning"] = ff_stats.get("negative_learning", 0)

                    logger.debug(
                        f"P6.4: FF retrieval learning from outcome: "
                        f"score={outcome_score:.2f}, "
                        f"positive={ff_stats.get('positive_learning', 0)}, "
                        f"negative={ff_stats.get('negative_learning', 0)}"
                    )
            except Exception as e:
                logger.warning(f"P6.4: FF retrieval learning failed: {e}")

        # Step 1.13: Phase 5 - Train FFEncoder from outcome with three-factor learning
        # This is THE LEARNING GAP FIX - the encoder now learns from outcomes
        # Three-factor rule: eligibility x neuromod x dopamine -> weight updates
        # CRITICAL FIX: Learn on MEMORY embeddings, not query embedding (Hinton critique)
        if self._ff_encoder_enabled and self._ff_encoder is not None:
            try:
                # Fetch memory embeddings for the retrieved episodes
                # This is what we should learn on - the actual memory representations
                id_strings = [str(eid) for eid in episode_ids]
                results = await self.vector_store.get_with_vectors(
                    collection=self.vector_store.episodes_collection,
                    ids=id_strings,
                )

                if not results:
                    logger.debug("Phase 5: No embeddings found for FFEncoder learning")
                else:
                    # Build map of memory_id -> embedding
                    memory_embeddings = {}
                    for id_str, _payload, vector in results:
                        if vector is not None:
                            memory_embeddings[id_str] = np.array(vector)

                    # Prepare outcome mapping for batch computation
                    batch_outcomes = {}
                    for eid in episode_ids:
                        id_str = str(eid)
                        if id_str in memory_embeddings:
                            # Mark memory as active (updates eligibility trace)
                            self.three_factor.mark_active(id_str, activity=1.0)
                            # Get per-memory outcome if available
                            batch_outcomes[id_str] = (
                                per_memory_rewards.get(id_str, outcome_score)
                                if per_memory_rewards else outcome_score
                            )

                    # Batch compute three-factor signals for all memories
                    # This is more efficient than computing one at a time
                    valid_ids = [eid for eid in episode_ids if str(eid) in memory_embeddings]
                    three_factor_signals = self.three_factor.batch_compute(
                        memory_ids=valid_ids,
                        base_lr=0.03,  # FFEncoder base learning rate
                        outcomes=batch_outcomes,
                    )

                    # Apply learning for each memory using precomputed signals
                    ff_updates_count = 0
                    for eid in valid_ids:
                        id_str = str(eid)
                        memory_emb = memory_embeddings[id_str]
                        three_factor_signal = three_factor_signals[id_str]
                        mem_outcome = batch_outcomes[id_str]

                        # Learn from outcome on THIS MEMORY'S embedding (not query)
                        # The memory representation is what needs to be reinforced/suppressed
                        ff_learning_stats = self._ff_encoder.learn_from_outcome(
                            embedding=memory_emb,
                            outcome_score=mem_outcome,
                            three_factor_signal=three_factor_signal,
                        )
                        ff_updates_count += 1

                        logger.debug(
                            f"Phase 5: FFEncoder learned from outcome "
                            f"(memory={id_str[:8]}, outcome={mem_outcome:.2f}, "
                            f"eligibility={three_factor_signal.eligibility:.3f}, "
                            f"effective_lr={three_factor_signal.effective_lr_multiplier:.3f})"
                        )

                    stats["ff_encoder_updated"] = ff_updates_count > 0
                    stats["ff_encoder_memories_trained"] = ff_updates_count
                    stats["ff_encoder_total_updates"] = self._ff_encoder.state.total_positive_updates + \
                                                        self._ff_encoder.state.total_negative_updates

            except Exception as e:
                logger.warning(f"Phase 5: FFEncoder learning failed: {e}")

        # Step 1.14: Phase 6 - Train CapsuleLayer from outcome
        # Capsule poses and activations learn from routing agreement
        # Positive outcomes reinforce capsule configurations, negative outcomes suppress them
        if self._capsule_layer_enabled and self._capsule_layer is not None:
            try:
                # Get query embedding
                query_emb = await self.embedding.embed_query(query)
                query_emb_np = np.array(query_emb)

                # Get capsule activations and poses for query
                capsule_activations, capsule_poses = self._capsule_layer.forward(query_emb_np)

                # Learn from outcome: positive = increase goodness, negative = decrease
                if outcome_score > 0.5:
                    # Positive outcome: reinforce this capsule configuration
                    capsule_stats = self._capsule_layer.learn_positive(
                        x=query_emb_np,
                        activations=capsule_activations,
                        poses=capsule_poses,
                        learning_rate=self._capsule_layer.config.learning_rate * outcome_score,
                    )
                else:
                    # Negative outcome: suppress this capsule configuration
                    capsule_stats = self._capsule_layer.learn_negative(
                        x=query_emb_np,
                        activations=capsule_activations,
                        poses=capsule_poses,
                        learning_rate=self._capsule_layer.config.learning_rate * (1 - outcome_score),
                    )

                stats["capsule_updated"] = True
                stats["capsule_goodness"] = capsule_stats.get("goodness", 0.0)
                stats["capsule_pose_update_norm"] = capsule_stats.get("pose_update_norm", 0.0)

                logger.debug(
                    f"Phase 6: CapsuleLayer learned from outcome "
                    f"(outcome={outcome_score:.2f}, phase={capsule_stats.get('phase')}, "
                    f"pose_update_norm={capsule_stats.get('pose_update_norm', 0):.4f})"
                )

            except Exception as e:
                logger.warning(f"Phase 6: CapsuleLayer learning failed: {e}")

        # Step 1.15: Phase 2A - Train FF-Capsule Bridge from outcome
        # Joint learning enables both FF and capsule systems to improve together
        # Positive outcomes reinforce both goodness and routing, negative suppresses
        if self._ff_capsule_bridge_enabled and self._ff_capsule_bridge is not None:
            try:
                # Get query embedding for bridge learning
                query_emb = await self.embedding.embed_query(query)
                query_emb_np = np.array(query_emb)

                # Forward through bridge first (updates internal state)
                self._ff_capsule_bridge.forward(query_emb_np, learn_poses=False)

                # Learn from outcome - propagates to both FF encoder and capsule layer
                bridge_learn_stats = self._ff_capsule_bridge.learn(
                    outcome=outcome_score,
                    embedding=query_emb_np,
                )

                stats["ff_capsule_bridge_updated"] = True
                stats["ff_capsule_bridge_total_learns"] = self._ff_capsule_bridge.state.total_learn_calls
                stats["ff_capsule_bridge_positive"] = self._ff_capsule_bridge.state.total_positive_outcomes
                stats["ff_capsule_bridge_negative"] = self._ff_capsule_bridge.state.total_negative_outcomes

                logger.debug(
                    f"Phase 2A: FF-Capsule Bridge learned from outcome "
                    f"(outcome={outcome_score:.2f}, "
                    f"total_learns={self._ff_capsule_bridge.state.total_learn_calls}, "
                    f"mean_outcome={self._ff_capsule_bridge.state.mean_outcome_score:.3f})"
                )

            except Exception as e:
                logger.warning(f"Phase 2A: FF-Capsule Bridge learning failed: {e}")

        # Step 2: Compute RPE-based training targets for fusion
        # Instead of training on raw outcomes, use prediction errors
        id_strings = [str(eid) for eid in episode_ids]
        memory_outcomes = {}
        for id_str in id_strings:
            if per_memory_rewards and id_str in per_memory_rewards:
                memory_outcomes[id_str] = per_memory_rewards[id_str]
            else:
                memory_outcomes[id_str] = outcome_score

        # Get RPE targets: shifts δ from [-1,1] to [0,1] for ranking
        rpe_targets = self.dopamine.get_rpe_for_fusion_training(memory_outcomes)

        # Compute RPE statistics
        rpes = self.dopamine.batch_compute_rpe(memory_outcomes)
        if rpes:
            stats["avg_rpe"] = float(np.mean([r.rpe for r in rpes.values()]))
            stats["positive_surprises"] = sum(1 for r in rpes.values() if r.is_positive_surprise)
            stats["negative_surprises"] = sum(1 for r in rpes.values() if r.is_negative_surprise)

        # Step 3: Train fusion weights if we have the required scores
        if neural_scores and symbolic_scores and len(episode_ids) >= 2:
            try:
                # Generate query embedding for fusion training
                query_emb = await self.embedding.embed_query(query)
                query_emb_np = np.array(query_emb)

                # Default recency to 0.5 if not provided
                if recency_scores is None:
                    recency_scores = dict.fromkeys(id_strings, 0.5)

                # Outcome history scores (use dopamine value estimates)
                outcome_scores = {
                    id_str: self.dopamine.get_expected_value(UUID(id_str))
                    for id_str in id_strings
                }

                # Train fusion with RPE targets (not raw outcomes!)
                loss = self.reasoner.train_fusion_step(
                    query_embedding=query_emb_np,
                    neural_scores=neural_scores,
                    symbolic_scores=symbolic_scores,
                    recency_scores=recency_scores,
                    outcome_scores=outcome_scores,
                    target_rewards=rpe_targets  # RPE-based targets!
                )
                stats["fusion_loss"] = loss

                logger.info(
                    f"Fusion training: loss={loss:.4f}, "
                    f"avg_rpe={stats['avg_rpe']:.3f}, "
                    f"surprises=+{stats['positive_surprises']}/-{stats['negative_surprises']}"
                )

            except Exception as e:
                logger.warning(f"Fusion training failed: {e}")

        add_span_attribute("learning.reconsolidated", stats["reconsolidated_count"])
        add_span_attribute("learning.fusion_loss", stats["fusion_loss"])
        add_span_attribute("learning.avg_rpe", stats["avg_rpe"])

        return stats

    def get_fusion_training_stats(self) -> dict:
        """Get learned fusion training statistics."""
        return self.reasoner.get_fusion_training_stats()

    def get_orchestra_stats(self) -> dict:
        """Get neuromodulator orchestra statistics."""
        return self.orchestra.get_stats()

    def get_learned_gate_stats(self) -> dict:
        """
        Get learned memory gate statistics.

        Returns:
            Dict with:
            - enabled: Whether gating is enabled
            - n_observations: Total training samples
            - store_rate/skip_rate/buffer_rate: Decision distribution
            - cold_start_progress: Progress through cold start phase
            - calibration_ece: Expected calibration error
        """
        if not self._gating_enabled:
            return {"enabled": False}

        stats = self.learned_gate.get_stats()
        stats["enabled"] = True
        return stats

    def get_buffer_stats(self) -> dict:
        """
        Get buffer manager statistics.

        Returns:
            Dict with buffer size, pressure, promotion rate, etc.
        """
        if not self._buffering_enabled:
            return {"enabled": False}

        stats = self.buffer_manager.get_stats()
        stats["enabled"] = True
        return stats

    @traced("episodic.tick_buffer", kind=SpanKind.INTERNAL)
    async def tick_buffer(self) -> list[Episode]:
        """
        Evaluate buffer items for promotion/discard.

        Should be called periodically (e.g., after each interaction or
        every 30 seconds). Promoted items are stored to long-term memory.

        Returns:
            List of episodes that were promoted and stored
        """
        if not self._buffering_enabled or self.buffer_manager.size == 0:
            return []

        promoted_episodes = []

        try:
            # Get current neuromodulator state for threshold adjustment
            neuro_state = self.orchestra.get_current_state() if self._neuromodulation_enabled else None

            # Evaluate all buffer items
            decisions = self.buffer_manager.tick(neuro_state=neuro_state)

            # Process promotions by storing to long-term memory
            for decision in decisions:
                if decision.action.value == "promote":
                    item = self.buffer_manager.get_item(decision.item_id)
                    if item is not None:
                        # Store the promoted item
                        episode = await self._store_promoted_item(item)
                        if episode:
                            promoted_episodes.append(episode)

            if decisions:
                promote_count = sum(1 for d in decisions if d.action.value == "promote")
                discard_count = sum(1 for d in decisions if d.action.value == "discard")
                logger.info(
                    f"Buffer tick: {promote_count} promoted, {discard_count} discarded, "
                    f"{self.buffer_manager.size} remaining"
                )
                add_span_attribute("buffer.tick_promoted", promote_count)
                add_span_attribute("buffer.tick_discarded", discard_count)

        except Exception as e:
            logger.warning(f"Buffer tick failed: {e}")

        return promoted_episodes

    # =========================================================================
    # P3.4: Interleaved Replay Support Methods
    # =========================================================================

    async def get_recent(
        self,
        hours: int = 24,
        limit: int = 100,
        session_filter: str | None = None,
    ) -> list[Episode]:
        """
        Get recent episodes within time window (for sleep consolidation).

        Used by SleepConsolidation.nrem_phase() for replay selection.
        Implements the EpisodicMemory protocol expected by sleep.py.

        Args:
            hours: Hours to look back (default: 24)
            limit: Maximum episodes to return (default: 100)
            session_filter: Optional session ID filter

        Returns:
            List of recent episodes ordered by timestamp descending
        """
        from datetime import timedelta

        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        # Use recall_by_timerange for efficient retrieval
        episodes = []
        cursor = None

        while len(episodes) < limit:
            page_size = min(100, limit - len(episodes))
            page, cursor = await self.recall_by_timerange(
                start_time=start_time,
                end_time=end_time,
                page_size=page_size,
                cursor=cursor,
                session_filter=session_filter,
            )
            episodes.extend(page)

            if cursor is None or len(page) == 0:
                break

        # Sort by timestamp descending (most recent first)
        episodes.sort(key=lambda e: e.timestamp, reverse=True)

        logger.debug(f"get_recent: retrieved {len(episodes)} episodes from last {hours}h")
        return episodes[:limit]

    async def sample_random(
        self,
        limit: int = 50,
        session_filter: str | None = None,
        exclude_hours: int = 24,
    ) -> list[Episode]:
        """
        Sample random episodes from memory (for interleaved replay).

        Used by CLS (Complementary Learning Systems) theory for mixing
        recent and older memories during replay to prevent catastrophic forgetting.

        Args:
            limit: Number of episodes to sample
            session_filter: Optional session ID filter
            exclude_hours: Exclude episodes from this recent time window
                          (default: 24, to avoid overlap with get_recent)

        Returns:
            List of randomly sampled episodes (older than exclude_hours)
        """
        import random
        from datetime import timedelta

        # Build filter for older episodes
        filter_conditions = {}
        if session_filter:
            filter_conditions["session_id"] = session_filter
        elif self.session_id != "default":
            filter_conditions["session_id"] = self.session_id

        # Exclude recent episodes
        if exclude_hours > 0:
            cutoff_time = datetime.now() - timedelta(hours=exclude_hours)
            filter_conditions["timestamp"] = {"lt": cutoff_time.timestamp()}

        try:
            # Get count of available episodes
            total_count = await self.vector_store.count(
                collection=self.vector_store.episodes_collection,
                count_filter=filter_conditions if filter_conditions else None,
            )

            if total_count == 0:
                logger.debug("sample_random: no older episodes available")
                return []

            # If we have fewer episodes than limit, get all of them
            if total_count <= limit:
                results, _ = await self.vector_store.scroll(
                    collection=self.vector_store.episodes_collection,
                    scroll_filter=filter_conditions if filter_conditions else None,
                    limit=total_count,
                    offset=0,
                    with_payload=True,
                    with_vectors=True,  # Need vectors for embedding attribute
                )
                episodes = [
                    self._from_payload(id_str, payload)
                    for id_str, payload, _ in results
                ]
                random.shuffle(episodes)
                logger.debug(f"sample_random: retrieved all {len(episodes)} older episodes")
                return episodes

            # For larger collections, use random offset sampling
            # This is more efficient than fetching all and shuffling
            sample_size = min(limit * 3, total_count)  # Oversample to get variety
            random_offsets = random.sample(range(total_count), sample_size)
            random_offsets.sort()  # Sort for sequential access efficiency

            episodes = []
            seen_ids = set()

            # Batch fetch in chunks of ~100
            chunk_size = 100
            for i in range(0, len(random_offsets), chunk_size):
                chunk_offsets = random_offsets[i:i + chunk_size]
                if not chunk_offsets:
                    continue

                # Use first offset in chunk
                offset = chunk_offsets[0]
                results, _ = await self.vector_store.scroll(
                    collection=self.vector_store.episodes_collection,
                    scroll_filter=filter_conditions if filter_conditions else None,
                    limit=chunk_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )

                for id_str, payload, _ in results:
                    if id_str not in seen_ids:
                        episode = self._from_payload(id_str, payload)
                        episodes.append(episode)
                        seen_ids.add(id_str)

                        if len(episodes) >= limit:
                            break

                if len(episodes) >= limit:
                    break

            random.shuffle(episodes)
            logger.debug(f"sample_random: sampled {len(episodes)} episodes from {total_count} total")
            return episodes[:limit]

        except Exception as e:
            logger.error(f"Error in sample_random: {e}")
            return []

    async def _store_promoted_item(self, item) -> Episode | None:
        """
        Store a promoted buffer item to long-term memory.

        Args:
            item: BufferedItem to store

        Returns:
            Created episode if successful
        """
        from ww.memory.buffer_manager import BufferedItem

        if not isinstance(item, BufferedItem):
            logger.warning(f"Invalid item type for promotion: {type(item)}")
            return None

        # Create episode object
        episode = Episode(
            session_id=self.session_id,
            content=item.content,
            embedding=item.embedding.tolist() if hasattr(item.embedding, "tolist") else list(item.embedding),
            context=EpisodeContext(**(item.context or {})),
            outcome=Outcome(item.outcome),
            emotional_valence=item.valence,
            stability=self.default_stability,
        )

        # Wrap in saga for atomicity
        saga = Saga(f"promote_buffered_{episode.id}")

        # Step 1: Add to vector store
        saga.add_step(
            name="add_vector",
            action=lambda: self.vector_store.add(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
                vectors=[episode.embedding],
                payloads=[self._to_payload(episode)],
            ),
            compensate=lambda: self.vector_store.delete(
                collection=self.vector_store.episodes_collection,
                ids=[str(episode.id)],
            ),
        )

        # Step 2: Create graph node
        saga.add_step(
            name="create_node",
            action=lambda: self.graph_store.create_node(
                label="Episode",
                properties=self._to_graph_props(episode),
            ),
            compensate=lambda: self.graph_store.delete_node(
                node_id=str(episode.id),
                label="Episode",
            ),
        )

        # Execute saga
        result = await saga.execute()

        if result.state not in (SagaState.COMMITTED,):
            logger.warning(
                f"Promoted episode storage failed: {result.error} "
                f"(saga: {result.saga_id})"
            )
            return None

        logger.info(
            f"Stored promoted episode {episode.id} "
            f"(evidence={item.evidence_score:.3f}, hits={item.retrieval_hits})"
        )

        # Add to hybrid collection if enabled
        if self.hybrid_enabled and self._hybrid_initialized:
            try:
                await self._store_hybrid(episode, item.content)
            except Exception as e:
                logger.warning(f"Failed to store hybrid for promoted {episode.id}: {e}")

        # Add as pattern completion attractor
        if self._pattern_completion_enabled:
            try:
                self.pattern_completion.add_attractor(np.array(episode.embedding))
            except Exception as e:
                logger.warning(f"Failed to add attractor for promoted {episode.id}: {e}")

        return episode

    def clear_buffer(self) -> None:
        """Clear all buffered items (discards with training signal)."""
        if self._buffering_enabled:
            self.buffer_manager.clear()
            logger.info("Buffer cleared")

    def get_current_neuromodulator_state(self) -> dict | None:
        """Get current neuromodulator state for monitoring."""
        state = self.orchestra.get_current_state()
        return state.to_dict() if state else None

    def start_session(self, goal: str | None = None) -> None:
        """
        Start a new session for neuromodulator tracking.

        Args:
            goal: Optional goal description for this session
        """
        if self._neuromodulation_enabled:
            self.orchestra.start_session(self.session_id, goal)
            logger.info(f"Started neuromodulator session: {self.session_id}")

    def end_session(self, outcome: float) -> dict[str, float]:
        """
        End current session and distribute serotonin credit.

        Args:
            outcome: Final session outcome [0, 1]

        Returns:
            Memory ID -> credit assigned
        """
        if self._neuromodulation_enabled:
            credits = self.orchestra.end_session(self.session_id, outcome)
            logger.info(
                f"Ended session {self.session_id}: "
                f"outcome={outcome:.2f}, {len(credits)} memories credited"
            )
            return credits
        return {}

    def get_long_term_value(self, episode_id: UUID) -> float:
        """
        Get long-term value estimate for an episode (serotonin).

        This reflects how often this memory has led to positive
        long-term outcomes across sessions.

        Args:
            episode_id: Episode to check

        Returns:
            Long-term value [0, 1]
        """
        return self.orchestra.get_long_term_value(episode_id)

    def get_expected_value(self, episode_id: UUID) -> float:
        """
        Get expected value for an episode (dopamine).

        This reflects the expected immediate outcome when using this memory.

        Args:
            episode_id: Episode to check

        Returns:
            Expected value [0, 1]
        """
        return self.orchestra.get_expected_value(episode_id)

    def should_encode_now(self) -> bool:
        """
        Check if system should prioritize encoding (ACh mode).

        Returns True when the neuromodulator orchestra indicates
        encoding mode (novel context, high NE arousal, etc.)
        """
        if not self._neuromodulation_enabled:
            return True  # Default to encoding when disabled
        return self.orchestra.should_encode()

    def should_retrieve_now(self) -> bool:
        """
        Check if system should prioritize retrieval (ACh mode).

        Returns True when the neuromodulator orchestra indicates
        retrieval mode (familiar context, question asked, etc.)
        """
        if not self._neuromodulation_enabled:
            return True  # Default to retrieval when disabled
        return self.orchestra.should_retrieve()

    # Phase 1: ClusterIndex management methods

    def register_cluster(
        self,
        cluster_id: str,
        centroid: np.ndarray,
        member_ids: list[UUID],
        variance: float = 0.0,
        coherence: float = 1.0,
    ) -> None:
        """
        Register a semantic cluster (called during sleep consolidation).

        This enables hierarchical O(log n) retrieval by grouping similar
        episodes into clusters with CA3-like routing.

        Args:
            cluster_id: Unique cluster identifier
            centroid: Mean embedding of cluster members
            member_ids: UUIDs of episodes in this cluster
            variance: Intra-cluster variance (lower = tighter)
            coherence: Semantic coherence score (higher = better)
        """
        self.cluster_index.register_cluster(
            cluster_id=cluster_id,
            centroid=centroid,
            member_ids=member_ids,
            variance=variance,
            coherence=coherence,
        )

    def get_cluster_statistics(self) -> dict:
        """
        Get cluster index statistics.

        Returns:
            Dictionary with cluster counts, sizes, and performance metrics
        """
        return self.cluster_index.get_statistics()

    def prune_stale_clusters(
        self,
        max_age_days: float = 30.0,
        min_success_rate: float = 0.1,
    ) -> list[str]:
        """
        Remove stale or low-performing clusters.

        Args:
            max_age_days: Maximum age before pruning consideration
            min_success_rate: Minimum success rate to retain

        Returns:
            List of pruned cluster IDs
        """
        return self.cluster_index.prune_stale_clusters(
            max_age_days=max_age_days,
            min_success_rate=min_success_rate,
        )


def get_episodic_memory(session_id: str | None = None) -> EpisodicMemory:
    """Get or create episodic memory service."""
    return EpisodicMemory(session_id)
