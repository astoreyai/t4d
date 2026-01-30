# World Weaver Learning Improvement Plan

**Created**: 2026-01-07
**Author**: Geoffrey Hinton AI Architect Agent
**Status**: Draft for Review
**Baseline Score**: 6.5/10 (with current learning assessed at 5/10 for integration)

---

## Executive Summary

The current World Weaver implementation has sophisticated neural architecture components (Forward-Forward, Capsules, VAE) but lacks deep integration that would make representations actually improve through use. This plan addresses five core limitations:

1. **Frozen Embeddings**: BGE-M3 produces static vectors; FF only refines downstream
2. **Hand-Set Poses**: Capsule pose dimensions are manually defined, not emergent
3. **Underutilized Routing**: Capsule routing-by-agreement rarely exercised in practice
4. **Generative Replay Gap**: VAE exists but not integrated with consolidation
5. **No Retrieval Feedback Loop**: System cannot learn from retrieval successes/failures

The plan is structured as four phases that can run partially in parallel with ongoing CompBio and Architecture work.

---

## Phase 1: Embedding Adaptation Pipeline (Sprint 13-14)

### Problem Statement

BGE-M3 embeddings are frozen. The LoRA adapter exists (`lora_adapter.py`) but:
- Not integrated into the retrieval pipeline
- No automatic feedback from retrieval outcomes
- Training requires manual invocation

The core question: **Should we fine-tune BGE-M3 or build a more powerful FF stack?**

**Answer**: Neither extreme. Fine-tuning BGE-M3 is expensive and risks catastrophic forgetting. Pure FF stack from scratch would lose BGE-M3's massive pretraining. Instead:
- Use BGE-M3 as frozen backbone (captures general semantics)
- Train lightweight adapters using retrieval feedback (task-specific adaptation)
- Use FF layers for online refinement during session

### Implementation Approach

#### 1.1 Retrieval Feedback Collector

**File**: `/mnt/projects/ww/src/ww/learning/retrieval_feedback.py` (new)

```python
@dataclass
class RetrievalFeedback:
    """Feedback signal from retrieval outcomes."""
    query_embedding: np.ndarray
    retrieved_ids: list[UUID]
    retrieved_embeddings: list[np.ndarray]
    relevance_signals: list[float]  # 0.0-1.0 from user actions
    implicit_negatives: list[np.ndarray]  # Retrieved but not used
    timestamp: datetime
    session_id: str

class RetrievalFeedbackCollector:
    """
    Collects implicit and explicit feedback from retrieval.

    Implicit signals:
    - User clicked/expanded a result (positive)
    - User scrolled past (weak negative)
    - User reformulated query after retrieval (negative for old results)

    Explicit signals:
    - User marked as relevant/irrelevant
    - User edited or corrected retrieved content
    """
```

#### 1.2 Online Adapter Training

**File**: `/mnt/projects/ww/src/ww/learning/online_adapter.py` (new)

```python
class OnlineAdapterTrainer:
    """
    Trains LoRA adapter from retrieval feedback in background.

    Training schedule:
    - Batch feedback into groups of 32
    - Train every 5 minutes or 100 feedback signals
    - Use EWC to prevent forgetting old adaptations
    """

    def __init__(
        self,
        lora_adapter: LoRAEmbeddingAdapter,
        feedback_collector: RetrievalFeedbackCollector,
        train_interval_seconds: int = 300,
        min_batch_size: int = 32,
    ):
        ...

    async def process_feedback_batch(self, feedback_batch: list[RetrievalFeedback]) -> dict:
        """
        Convert feedback to training signal.

        Key insight: Retrieval feedback provides natural contrastive pairs:
        - Query + clicked result = positive pair
        - Query + skipped result = negative pair
        """
```

#### 1.3 Integration with Memory Pipeline

**Files to modify**:
- `/mnt/projects/ww/src/ww/memory/episodic.py`: Add feedback hooks
- `/mnt/projects/ww/src/ww/embedding/adapter.py`: Connect to OnlineAdapterTrainer
- `/mnt/projects/ww/src/ww/api/routes/episodes.py`: Expose feedback endpoints

### Validation Strategy

**How do we validate that retrieval quality improves over time?**

1. **Retrieval Quality Metrics** (continuous monitoring):
   - Mean Reciprocal Rank (MRR) of clicked results
   - Precision@K for top-K retrieved
   - Query reformulation rate (lower = better)

2. **A/B Testing Framework**:
   - Split sessions: 50% use adapted embeddings, 50% use frozen
   - Compare click-through rates and session success

3. **Representation Analysis**:
   - Track embedding space clustering quality over time
   - Measure semantic coherence of retrieved sets

**File**: `/mnt/projects/ww/src/ww/observability/retrieval_metrics.py` (new)

```python
class RetrievalQualityTracker:
    """
    Tracks retrieval quality metrics over time.

    Key metrics:
    - mrr_history: List of MRR values per session
    - precision_at_k: Dict mapping k -> precision history
    - embedding_cluster_quality: Silhouette score of embeddings
    """

    def log_retrieval(
        self,
        query_id: str,
        retrieved_ids: list[str],
        clicked_ids: list[str],
        session_id: str,
    ) -> None:
        ...

    def get_improvement_trend(self, window_days: int = 7) -> dict:
        """
        Compute improvement trend over time window.

        Returns:
            {
                "mrr_trend": float,  # Positive = improving
                "precision_trend": float,
                "significant": bool,  # Statistical significance
            }
        """
```

### Testing Requirements

1. **Unit Tests** (`tests/learning/test_retrieval_feedback.py`):
   - Feedback collection from various signal types
   - Batch formation and training trigger logic
   - EWC regularization prevents forgetting

2. **Integration Tests** (`tests/integration/test_online_adaptation.py`):
   - End-to-end retrieval -> feedback -> training -> improved retrieval
   - Concurrent session handling
   - Graceful degradation when training fails

3. **Benchmark Tests** (`tests/benchmarks/test_adaptation_quality.py`):
   - Measure retrieval improvement on held-out queries
   - Compare adapted vs frozen embedding performance

### Documentation Updates

- Update `/mnt/projects/ww/docs/CLAUDE_CODE_INTEGRATION.md` with feedback API
- Add `/mnt/projects/ww/docs/EMBEDDING_ADAPTATION.md` explaining the approach
- Update `/mnt/projects/ww/docs/architecture.md` with new components

### Estimated Effort

- Implementation: 5-7 days
- Testing: 3-4 days
- Documentation: 1 day
- **Total: ~10-12 days**

### Dependencies

- Existing LoRA adapter (`lora_adapter.py`) - DONE
- EWC regularizer in plasticity module - DONE
- Session management infrastructure - DONE

---

## Phase 2: Emergent Pose Learning (Sprint 15-16)

### Problem Statement

Current pose dimensions are hand-set in `pose.py`:
- TEMPORAL, CAUSAL, SEMANTIC_ROLE, CERTAINTY
- Values assigned manually via `set_temporal()`, `set_causal()`, etc.
- No learning from data about what pose dimensions should emerge

**How do we make poses emergent vs hand-set?**

The key insight from Hinton's capsule work: poses should emerge from routing agreement, not be predefined. When multiple lower-level capsules consistently vote for similar transformations, those transformations should strengthen.

### Implementation Approach

#### 2.1 Pose Dimension Discovery

**File**: `/mnt/projects/ww/src/ww/nca/pose_learner.py` (new)

```python
class PoseDimensionDiscovery:
    """
    Discovers pose dimensions from data distribution.

    Approach:
    1. Collect pose transformation patterns from routing
    2. Apply PCA/ICA to find principal variation axes
    3. Name dimensions via clustering with interpretable features

    The 4 semantic dimensions (temporal, causal, role, certainty)
    become initialization hints, not fixed structure.
    """

    def __init__(
        self,
        initial_dim_hints: list[str] = ["temporal", "causal", "role", "certainty"],
        max_discovered_dims: int = 8,
        discovery_threshold: float = 0.1,  # Min variance to keep dimension
    ):
        ...

    def update_from_routing(
        self,
        routing_coefficients: np.ndarray,
        pose_predictions: np.ndarray,
        consensus_poses: np.ndarray,
    ) -> dict:
        """
        Update dimension discovery from routing iteration.

        Key signal: When routing converges quickly, poses are well-aligned.
        When routing oscillates, poses capture conflicting information.
        """
```

#### 2.2 Agreement-Based Pose Update

The existing `learn_pose_from_routing()` in `capsules.py` (lines 592-688) is a good start but needs enhancement:

**File**: `/mnt/projects/ww/src/ww/nca/capsules.py` (modify)

```python
def learn_pose_from_routing_v2(
    self,
    lower_poses: np.ndarray,
    predictions: np.ndarray,
    consensus_poses: np.ndarray,
    agreement_scores: np.ndarray,
    learning_rate: float | None = None,
    use_hebbian: bool = True,  # NEW: Use Hebbian-style update
) -> dict:
    """
    Enhanced pose learning with Hebbian update rule.

    Hinton's key insight: Learning should be local.
    Instead of gradient-based update, use:

    delta_W_ij = eta * a_i * a_j * (consensus_j - prediction_ij)

    Where:
    - a_i = activation of lower capsule
    - a_j = activation of higher capsule
    - consensus_j = agreed pose
    - prediction_ij = this capsule's prediction

    This is local: only uses info available at the synapse.
    """
```

#### 2.3 Pose Initialization from Data

**File**: `/mnt/projects/ww/src/ww/nca/pose_initializer.py` (new)

```python
class DataDrivenPoseInitializer:
    """
    Initialize pose dimensions from memory corpus.

    Instead of hand-setting temporal/causal/etc, analyze:
    1. Co-occurrence patterns (what memories appear together)
    2. Temporal patterns (what sequences exist)
    3. Semantic clusters (what topics emerge)

    These become the initial pose dimensions.
    """

    def analyze_corpus(
        self,
        memories: list[MemoryItem],
        embeddings: np.ndarray,
    ) -> PoseConfig:
        """
        Analyze memory corpus to derive pose dimensions.

        Returns configured PoseConfig with discovered dimensions.
        """
```

### Validation Strategy

1. **Pose Interpretability**:
   - Track what each learned pose dimension captures
   - Visualize pose space with example memories
   - Compare with hand-set baselines

2. **Routing Convergence**:
   - Measure iterations to routing convergence
   - Compare with fixed-pose baseline

3. **Memory Organization**:
   - Do similar memories cluster in pose space?
   - Does pose similarity predict semantic similarity?

**File**: `/mnt/projects/ww/src/ww/interfaces/pose_inspector.py` (new)

```python
class PoseInspector:
    """
    Interactive inspection of learned pose dimensions.

    For Claude Code integration: provides human-readable
    description of what each pose dimension has learned.
    """

    def describe_dimensions(self) -> list[dict]:
        """
        Return interpretable descriptions of pose dimensions.

        Example output:
        [
            {"dim": 0, "name": "discovered_temporal",
             "examples": ["yesterday", "last week", "will happen"],
             "variance_explained": 0.23},
            {"dim": 1, "name": "discovered_causal",
             "examples": ["because", "therefore", "leads to"],
             "variance_explained": 0.18},
        ]
        """
```

### Testing Requirements

1. **Unit Tests** (`tests/nca/test_pose_learner.py`):
   - Dimension discovery from synthetic data
   - Hebbian update rule correctness
   - Pose initialization from corpus

2. **Integration Tests** (`tests/nca/test_emergent_poses.py`):
   - End-to-end pose learning during memory encoding
   - Pose quality improves with more data
   - Backward compatibility with hand-set poses

3. **Interpretability Tests** (`tests/nca/test_pose_interpretability.py`):
   - Learned dimensions correlate with semantic features
   - Pose space visualization produces meaningful clusters

### Documentation Updates

- Update `/mnt/projects/ww/docs/concepts/capsules.md`
- Add `/mnt/projects/ww/docs/POSE_LEARNING.md`
- Update `/mnt/projects/ww/src/ww/nca/PARAMETERS.md`

### Estimated Effort

- Implementation: 6-8 days
- Testing: 3-4 days
- Documentation: 1-2 days
- **Total: ~11-14 days**

### Dependencies

- Capsule implementation (`capsules.py`) - DONE
- Pose matrix (`pose.py`) - DONE
- Memory corpus access - DONE

---

## Phase 3: Integrated Generative Replay (Sprint 17-18)

### Problem Statement

The VAE generator (`vae_generator.py`) and generative replay system (`generative_replay.py`) exist but:
- VAE trains independently, not connected to memory pipeline
- Replay uses stored patterns as fallback (not true generation)
- No integration with sleep consolidation

**What's the right generative model for replay?**

The VAE is a good choice because:
1. Learns compressed latent space of memory representations
2. Can generate novel combinations (not just replay exact memories)
3. Allows controlled variation via latent sampling

But needs proper integration with the consolidation pipeline.

### Implementation Approach

#### 3.1 VAE Training Pipeline

**File**: `/mnt/projects/ww/src/ww/learning/vae_training.py` (new)

```python
class VAETrainingPipeline:
    """
    Continuous training of VAE on memory embeddings.

    Training schedule:
    - During wake: Buffer new memories
    - During NREM: Train VAE on buffered + old samples
    - During REM: Generate novel combinations for replay
    """

    def __init__(
        self,
        vae: VAEGenerator,
        memory_buffer_size: int = 1000,
        train_on_wake: bool = False,  # Only train during sleep
        interleave_ratio: float = 0.5,  # Old vs new samples
    ):
        ...

    async def train_epoch(self, new_embeddings: list[np.ndarray]) -> dict:
        """
        Train VAE for one epoch.

        Uses interleaved sampling:
        - 50% newly stored memories
        - 50% generated from VAE (prevents forgetting)
        """
```

#### 3.2 Sleep Consolidation Integration

**File**: `/mnt/projects/ww/src/ww/consolidation/sleep.py` (modify)

The existing `SleepConsolidation` class needs to integrate with VAE:

```python
class SleepConsolidation:
    # ... existing code ...

    async def nrem_consolidation(self) -> dict:
        """
        NREM phase: Train VAE and replay important memories.

        Enhanced with:
        1. VAE training on recent memories
        2. Generative replay to strengthen old memories
        3. Sharp-wave ripple timing for replay
        """
        # Train VAE
        vae_stats = await self._vae_pipeline.train_epoch(self._recent_embeddings)

        # Generate replay samples
        replay_samples = self._replay_system.generate_samples(
            n_samples=100,
            temperature=0.8,  # Moderate diversity
        )

        # Replay through memory system
        for sample in replay_samples:
            await self._memory.consolidate_embedding(sample)

        return {
            "vae_loss": vae_stats["total_loss"],
            "replay_count": len(replay_samples),
            "phase": "NREM",
        }
```

#### 3.3 Quality-Filtered Replay

**File**: `/mnt/projects/ww/src/ww/learning/quality_replay.py` (new)

```python
class QualityFilteredReplay:
    """
    Filter generated samples by quality before replay.

    Quality criteria:
    1. Reconstruction confidence (VAE can round-trip)
    2. Semantic coherence (embedding is in valid region)
    3. Novelty (not too close to existing memories)
    """

    def filter_samples(
        self,
        generated: list[np.ndarray],
        existing_embeddings: np.ndarray,
        min_confidence: float = 0.7,
        min_novelty: float = 0.1,
    ) -> list[np.ndarray]:
        """
        Filter generated samples for replay.

        Returns only high-quality, sufficiently novel samples.
        """
```

### Validation Strategy

1. **Generation Quality**:
   - Track VAE reconstruction loss over time
   - Measure embedding coherence of generated samples
   - Visualize generated vs real in embedding space

2. **Replay Effectiveness**:
   - Measure memory retention with/without replay
   - Track forgetting rate on old memories
   - A/B test consolidation with/without generative replay

3. **Coverage Metrics**:
   - Does replay cover the embedding space uniformly?
   - Are rare/important memories replayed appropriately?

**File**: `/mnt/projects/ww/tests/integration/test_generative_consolidation.py` (new)

```python
async def test_replay_prevents_forgetting():
    """
    Verify generative replay prevents catastrophic forgetting.

    Approach:
    1. Store 100 memories
    2. Store 100 more (potentially interfering)
    3. Run consolidation with replay
    4. Verify original 100 still retrievable
    """
```

### Testing Requirements

1. **Unit Tests** (`tests/learning/test_vae_training.py`):
   - VAE training loop correctness
   - Interleaved sampling works
   - Quality filtering logic

2. **Integration Tests** (`tests/consolidation/test_sleep_vae.py`):
   - Sleep phases trigger VAE training
   - Generated samples flow through memory pipeline
   - Consolidation metrics tracked

3. **Stress Tests** (`tests/stress/test_replay_scale.py`):
   - Replay scales to 10K+ memories
   - VAE doesn't degrade with large corpus
   - Memory doesn't grow unbounded

### Documentation Updates

- Update `/mnt/projects/ww/docs/concepts/forward-forward.md` with replay integration
- Update `/mnt/projects/ww/src/ww/consolidation/README.md`
- Add `/mnt/projects/ww/docs/GENERATIVE_REPLAY.md`

### Estimated Effort

- Implementation: 5-6 days
- Testing: 3-4 days
- Documentation: 1 day
- **Total: ~9-11 days**

### Dependencies

- VAE generator (`vae_generator.py`) - DONE
- Generative replay (`generative_replay.py`) - DONE
- Sleep consolidation (`sleep.py`) - DONE

---

## Phase 4: FF-Capsule-NCA Deep Integration (Sprint 19-20)

### Problem Statement

The system has three powerful neural components:
1. Forward-Forward layers (local learning)
2. Capsule networks (part-whole composition)
3. NCA neuromodulation (biological signals)

But integration is shallow:
- FF processes embeddings, capsules process embeddings, but they don't talk
- NCA modulates parameters but doesn't drive learning decisions
- Routing-by-agreement rarely exercised in practice

### Implementation Approach

#### 4.1 FF-Capsule Bridge

**File**: `/mnt/projects/ww/src/ww/bridges/ff_capsule_bridge.py` (new)

```python
class FFCapsuleBridge:
    """
    Bridge between Forward-Forward and Capsule layers.

    Architecture:
        Input embedding
            |
        FF Layer 1 (goodness-based learning)
            |
        Capsule Layer (routing-by-agreement)
            |
        FF Layer 2 (refines capsule output)
            |
        Output embedding

    Key insight: FF goodness can guide capsule routing.
    High goodness = confident classification = trust routing.
    Low goodness = uncertain = explore alternative routes.
    """

    def __init__(
        self,
        ff_config: ForwardForwardConfig,
        capsule_config: CapsuleConfig,
        use_goodness_routing: bool = True,
    ):
        self.ff_layer1 = ForwardForwardLayer(ff_config)
        self.capsule_layer = CapsuleLayer(capsule_config)
        self.ff_layer2 = ForwardForwardLayer(ff_config)
        self.use_goodness_routing = use_goodness_routing

    def forward(self, x: np.ndarray, training: bool = False) -> tuple[np.ndarray, dict]:
        """
        Integrated forward pass.

        Returns:
            output: Final embedding
            stats: Dictionary with goodness, routing, and learning statistics
        """
        # FF Layer 1
        h1 = self.ff_layer1.forward(x, training=training)
        goodness1 = self.ff_layer1.state.goodness

        # Capsule with goodness-modulated routing
        if self.use_goodness_routing:
            # High goodness = more routing iterations (confident, refine)
            # Low goodness = fewer iterations (uncertain, don't over-commit)
            routing_iters = self._goodness_to_iterations(goodness1)
        else:
            routing_iters = self.capsule_layer.config.routing_iterations

        caps_act, caps_pose, routing_stats = self.capsule_layer.forward_with_routing(
            h1,
            routing_iterations=routing_iters,
            learn_poses=training,
        )

        # Flatten capsule output for FF Layer 2
        caps_flat = self._flatten_capsules(caps_act, caps_pose)

        # FF Layer 2
        h2 = self.ff_layer2.forward(caps_flat, training=training)

        return h2, {
            "goodness_1": goodness1,
            "goodness_2": self.ff_layer2.state.goodness,
            "routing_iters": routing_iters,
            "mean_agreement": routing_stats["mean_agreement"],
        }
```

#### 4.2 NCA-Driven Learning Signals

**File**: `/mnt/projects/ww/src/ww/nca/learning_signals.py` (new)

```python
class NCALearningSignalGenerator:
    """
    Generate learning signals from NCA state.

    Maps biological signals to learning parameters:
    - Dopamine (surprise) -> learning rate boost
    - ACh (attention) -> encoding vs retrieval mode
    - NE (arousal) -> classification threshold
    - Adenosine (fatigue) -> consolidation trigger
    """

    def __init__(
        self,
        dopamine_tracker: DopamineIntegration,
        ach_tracker: Any,  # ACh module
        ne_tracker: Any,   # NE module
        adenosine_tracker: AdenosineHomeostasis,
    ):
        ...

    def get_ff_modulation(self) -> dict:
        """
        Get FF learning modulation from NCA state.

        Returns:
            {
                "learning_rate_scale": float,  # 0.5-2.0
                "threshold_shift": float,      # -1.0 to +1.0
                "phase": "encoding" | "retrieval",
            }
        """

    def get_capsule_modulation(self) -> dict:
        """
        Get capsule routing modulation from NCA state.

        Returns:
            {
                "routing_temperature": float,
                "pose_learning_rate": float,
                "agreement_threshold": float,
            }
        """
```

#### 4.3 Routing Exercise During Retrieval

Currently capsule routing is rarely exercised. Integration with retrieval changes this:

**File**: `/mnt/projects/ww/src/ww/memory/episodic.py` (modify)

```python
async def retrieve(
    self,
    query: str,
    limit: int = 10,
    use_capsule_routing: bool = True,  # NEW
) -> list[Episode]:
    """
    Retrieve episodes with optional capsule routing.

    When use_capsule_routing=True:
    1. Get candidate embeddings from vector store
    2. Route through capsule layer
    3. Re-rank by capsule agreement
    4. Return top-K by combined score

    This exercises routing on every retrieval, not just encoding.
    """
```

### Validation Strategy

1. **Integration Coherence**:
   - FF goodness correlates with capsule agreement
   - NCA signals produce expected learning changes
   - End-to-end flow produces valid embeddings

2. **Learning Dynamics**:
   - Track learning over extended sessions
   - Compare integrated vs independent components
   - Measure representation quality improvement

3. **Routing Exercise Frequency**:
   - Count routing operations per session
   - Measure routing convergence improvement
   - Track pose learning from routing

### Testing Requirements

1. **Unit Tests** (`tests/bridges/test_ff_capsule_bridge.py`):
   - Bridge forward pass correctness
   - Goodness-routing coupling
   - Learning updates propagate

2. **Integration Tests** (`tests/integration/test_deep_integration.py`):
   - Full pipeline: input -> FF -> Capsule -> FF -> output
   - NCA modulation affects learning
   - Retrieval exercises routing

3. **Learning Dynamics Tests** (`tests/learning/test_integrated_learning.py`):
   - Representations improve over simulated sessions
   - No catastrophic interference
   - Stable training dynamics

### Documentation Updates

- Update `/mnt/projects/ww/docs/architecture.md` with integration diagram
- Add `/mnt/projects/ww/docs/DEEP_INTEGRATION.md`
- Update `/mnt/projects/ww/docs/HINTON_DESIGN_RATIONALE.md`

### Estimated Effort

- Implementation: 7-9 days
- Testing: 4-5 days
- Documentation: 2 days
- **Total: ~13-16 days**

### Dependencies

- Forward-Forward (`forward_forward.py`) - DONE
- Capsules (`capsules.py`) - DONE
- NCA modules - DONE
- Phase 2 (emergent poses) - Should complete first

---

## Cross-Phase Integration Points

### Shared Infrastructure

1. **Learning Metrics Dashboard**:
   - All phases feed into unified metrics
   - Track improvement across all learning systems
   - Alert on degradation

2. **Experiment Framework**:
   - A/B testing infrastructure
   - Holdout set for quality measurement
   - Statistical significance testing

3. **Persistence Layer**:
   - Save/load all learned parameters
   - Checkpoint during training
   - Rollback capability

### Parallel Execution Strategy

```
Sprint 13-14: Phase 1 (Embedding Adaptation)
Sprint 15-16: Phase 2 (Emergent Poses) + Phase 1 testing
Sprint 17-18: Phase 3 (Generative Replay) + Phase 2 testing
Sprint 19-20: Phase 4 (Deep Integration) + Phase 3 testing
Sprint 21:    Full integration testing + Documentation
```

Phase 1 and Phase 3 can start in parallel (independent components).
Phase 2 should precede Phase 4 (poses needed for integration).

---

## Success Criteria

### Phase 1: Embedding Adaptation
- [ ] MRR improves 10%+ after 1 week of use
- [ ] Adapter training runs without manual intervention
- [ ] No regression in cold-start retrieval quality

### Phase 2: Emergent Poses
- [ ] Discovered dimensions correlate with semantic features
- [ ] Routing convergence improves 20%+ vs hand-set
- [ ] Pose inspector produces interpretable descriptions

### Phase 3: Generative Replay
- [ ] VAE reconstruction loss < 0.1 after training
- [ ] Memory retention improves 15%+ with replay
- [ ] Generated samples pass quality filter at 70%+ rate

### Phase 4: Deep Integration
- [ ] Goodness-routing correlation > 0.5
- [ ] Routing exercised on 80%+ of retrievals
- [ ] Learning metrics show steady improvement over 30 days

### Overall Target
- Raise learning integration score from 5/10 to 8/10
- Raise overall Hinton assessment from 6.5/10 to 7.5/10

---

## Risk Assessment

### Technical Risks

1. **Adaptation Instability**:
   - Risk: LoRA training diverges or oscillates
   - Mitigation: EWC regularization, learning rate warmup, gradient clipping

2. **Pose Degeneracy**:
   - Risk: All poses collapse to same values
   - Mitigation: Diversity loss, periodic reset of low-variance dimensions

3. **Replay Quality**:
   - Risk: VAE generates incoherent samples
   - Mitigation: Quality filtering, temperature scheduling, reconstruction monitoring

4. **Integration Complexity**:
   - Risk: Too many interacting components cause instability
   - Mitigation: Incremental integration, extensive testing, feature flags

### Resource Risks

1. **Memory Growth**:
   - Risk: Feedback buffers and VAE training data grow unbounded
   - Mitigation: Fixed-size buffers, LRU eviction, periodic cleanup

2. **Compute Cost**:
   - Risk: Training overhead impacts retrieval latency
   - Mitigation: Background training, batch processing, early stopping

---

## Next Steps

1. Review this plan with stakeholders
2. Prioritize phases based on impact vs effort
3. Create detailed tickets for Sprint 13
4. Set up metrics infrastructure before implementation
5. Establish baseline measurements for success criteria

---

## References

- Hinton (2022): The Forward-Forward Algorithm
- Sabour et al. (2017): Dynamic Routing Between Capsules
- Hu et al. (2021): LoRA: Low-Rank Adaptation
- Shin et al. (2017): Continual Learning with Deep Generative Replay
- McClelland et al. (1995): Why There Are Complementary Learning Systems
