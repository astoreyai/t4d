# CompBio Neural Memory Integration Plan

**Date**: 2025-12-06
**Status**: Planning Complete - Ready for Implementation
**WW Version**: 0.2.0 (1,259 tests, 79% coverage)
**Source**: `/home/aaron/mem/NEURAL_MEMORY_VS_T4DM_COMPARISON.md`

---

## Executive Summary

This document provides a detailed, actionable implementation plan for integrating CompBio neural memory components into T4DM. The integration is structured in phases, prioritizing high-impact, low-risk enhancements that preserve WW's existing strengths (symbolic reasoning, production readiness, tripartite memory) while adding critical scaling and learning capabilities.

**Timeline**: 8-12 weeks
**Priority**: Critical for scaling to millions of memories
**Risk**: Low-Medium (incremental, additive changes)

---

## Gap Analysis Summary

### Critical Gaps (Must Have)
1. **Hierarchical Sparse Addressing** - 67x speedup, O(log n) vs O(n)
2. **Forward-Forward Learning** - Immediate Hebbian plasticity during encoding

### Important Gaps (Should Have)
3. **Hopfield-Fenchel-Young Memory** - Pattern completion, exponential capacity
4. **Pattern Separation** - DG-style orthogonalization

### Research Gaps (Could Have)
5. **Complementary Learning Systems with Replay**
6. **Dendritic Two-Compartment Neurons**

---

## Phase 0: Hierarchical Sparse Addressing (Week 1-2)

**Goal**: Implement O(log n) retrieval to replace flat O(n) k-NN scan

### Priority: CRITICAL (1/5)
**Impact**: Enables scaling from thousands to millions of memories
**Complexity**: Medium
**Risk**: Low (additive, preserves existing fallback)
**Timeline**: 2 weeks

---

### Task 0.1: Implement ClusterIndex

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/cluster_index.py`

**Purpose**: Hierarchical memory clustering for fast candidate generation

**Dependencies**:
- `numpy`, `scikit-learn` (already in project)
- `t4dm.storage.t4dx_vector_adapter` (existing)
- `t4dm.embedding.bge_m3` (existing)

**Implementation Details**:

```python
"""
Hierarchical Cluster Index for fast episodic retrieval.

Algorithm:
1. Periodically cluster memory embeddings (K-means or HDBSCAN)
2. Store cluster centroids in memory
3. At query time:
   - Compute distance to all centroids (cheap, ~100 clusters)
   - Select top-k nearest clusters
   - Search only within those clusters (67x reduction)
4. Fall back to flat search if cluster index stale

Key Parameters:
- n_clusters: 50-200 (adaptive based on total memories)
- rebuild_threshold: Rebuild when 10% new memories added
- max_cluster_size: 5000 (prevent mega-clusters)
"""

class ClusterIndex:
    def __init__(
        self,
        t4dx_vector_adapter,
        n_clusters: int = 100,
        rebuild_threshold: float = 0.1,
        min_memories_for_clustering: int = 500,
    ):
        ...

    async def rebuild_index(self, session_id: Optional[str] = None):
        """Rebuild cluster centroids from current memories."""
        ...

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        max_clusters: int = 3,
        session_id: Optional[str] = None,
    ) -> List[ScoredResult]:
        """
        Hierarchical search:
        1. Find top max_clusters nearest centroids
        2. Get all memories in those clusters
        3. Re-rank by exact similarity
        4. Return top-k
        """
        ...

    def needs_rebuild(self) -> bool:
        """Check if index is stale."""
        ...
```

**Acceptance Criteria**:
- [ ] `ClusterIndex` class with rebuild, search, needs_rebuild methods
- [ ] Automatic index rebuilding when 10% new memories added
- [ ] Falls back to flat search if index not initialized
- [ ] Unit tests: clustering, search accuracy, speed
- [ ] Integration test: EpisodicMemory.recall() uses ClusterIndex
- [ ] Benchmark: 10-67x speedup vs flat k-NN (depending on data size)

**Testing Requirements**:
- Unit tests: `/mnt/projects/t4d/t4dm/tests/unit/test_cluster_index.py`
  - Test clustering with 1K, 10K, 100K synthetic memories
  - Test search accuracy (recall@10 >= 95% vs flat)
  - Test rebuild logic
  - Test fallback to flat search
- Integration test: `/mnt/projects/t4d/t4dm/tests/integration/test_episodic_cluster.py`
  - Test end-to-end episodic recall with ClusterIndex
  - Verify saga consistency (no partial states)

**Success Metrics**:
- Search time: <50ms for 100K memories (vs 500ms flat)
- Recall accuracy: >=95% of flat k-NN results
- No breaking changes to existing API

**Risk Mitigation**:
- Maintain flat k-NN as fallback (low risk)
- Feature flag for gradual rollout
- Extensive benchmarking before production

**Effort**: Small-Medium (1-2 days implementation + 1 day testing)

---

### Task 0.2: Implement LearnedSparseIndex

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/learned_sparse_index.py`

**Purpose**: Learn optimal per-query sparsity (not fixed 10%)

**Dependencies**:
- `ClusterIndex` (Task 0.1)
- `t4dm.learning.neuromodulators` (existing)

**Implementation Details**:

```python
"""
Learned Sparse Index with adaptive sparsity.

Instead of fixed sparsity (always search 10% of clusters),
learn query-dependent sparsity from retrieval outcomes.

Training Signal:
- High utility retrieved → increase sparsity (search fewer clusters)
- Low utility retrieved → decrease sparsity (search more clusters)

Uses simple online gradient descent on sparsity parameter.
"""

class LearnedSparseIndex:
    def __init__(
        self,
        cluster_index: ClusterIndex,
        min_sparsity: float = 0.05,  # At least 5% of clusters
        max_sparsity: float = 0.5,   # At most 50% of clusters
        learning_rate: float = 0.01,
    ):
        ...

    def predict_sparsity(
        self,
        query_embedding: np.ndarray,
        neuromod_state: NeuromodulatorState,
    ) -> float:
        """
        Predict optimal sparsity for this query.

        Features:
        - Query embedding (via simple MLP)
        - NE arousal (high arousal → lower sparsity for exploration)
        - ACh mode (encoding → lower sparsity, retrieval → higher)
        """
        ...

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        neuromod_state: Optional[NeuromodulatorState] = None,
        session_id: Optional[str] = None,
    ) -> List[ScoredResult]:
        """
        Search with learned sparsity:
        1. Predict optimal cluster fraction
        2. Compute max_clusters = sparsity * total_clusters
        3. Delegate to ClusterIndex.search()
        """
        ...

    def update(
        self,
        query_embedding: np.ndarray,
        predicted_sparsity: float,
        retrieval_utility: float,  # 0-1, from outcome
    ):
        """
        Update sparsity predictor:
        - If utility high → increase sparsity (reward)
        - If utility low → decrease sparsity (penalty)
        """
        ...
```

**Acceptance Criteria**:
- [ ] `LearnedSparseIndex` wraps `ClusterIndex` with adaptive sparsity
- [ ] Simple 2-layer MLP predicts sparsity from query + neuromod state
- [ ] Online updates from retrieval feedback
- [ ] Unit tests: sparsity prediction, learning, bounds
- [ ] Ablation study: learned vs fixed sparsity

**Testing Requirements**:
- Unit tests: `/mnt/projects/t4d/t4dm/tests/unit/test_learned_sparse_index.py`
  - Test sparsity prediction (bounded to [min, max])
  - Test learning (utility high → sparsity increases)
  - Test integration with NeuromodulatorState
- Integration test: Compare learned vs fixed sparsity on synthetic workload

**Success Metrics**:
- Learned sparsity achieves >=98% utility of fixed 10% sparsity
- Reduces search time by 20-50% on average
- Adapts to query difficulty (hard queries → lower sparsity)

**Risk Mitigation**:
- Start with conservative bounds (5-50% sparsity)
- Monitor for catastrophic failures (sparsity → 0 or → 100%)
- Feature flag for gradual rollout

**Effort**: Small (1-2 days implementation + 1 day testing)

---

### Task 0.3: Implement FeatureAligner

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/feature_aligner.py`

**Purpose**: Ensure gate and retrieval scorer use same feature space

**Dependencies**:
- `t4dm.learning.learned_gate` (if exists, else create stub)
- `LearnedFusionWeights` (existing in episodic.py)

**Implementation Details**:

```python
"""
Feature Alignment between LearnedMemoryGate and LearnedFusionWeights.

Problem: Gate uses features F_gate(x), scorer uses features F_scorer(x).
If these feature spaces diverge, the gate's predictions become useless
for the scorer's ranking.

Solution: Shared projection layer that both gate and scorer use.

Architecture:
    Input (1024-dim BGE-M3)
         → Shared Projection (1024 → 128)
         → Gate Branch (128 → 1, binary decision)
         → Scorer Branch (128 → 4, fusion weights)

Joint Training:
- Gate loss: Binary cross-entropy (store or not)
- Scorer loss: Ranking loss (utility-weighted)
- Shared projection gets gradients from both
"""

class FeatureAligner:
    def __init__(
        self,
        input_dim: int = 1024,
        shared_dim: int = 128,
        learning_rate: float = 0.001,
    ):
        # Shared projection (input → shared)
        self.W_shared = ...
        self.b_shared = ...

        # Gate branch (shared → 1)
        self.W_gate = ...
        self.b_gate = ...

        # Scorer branch (shared → 4 for fusion weights)
        self.W_scorer = ...
        self.b_scorer = ...

    def forward_gate(self, embedding: np.ndarray) -> float:
        """
        Predict storage probability.

        Returns:
            p_store ∈ [0, 1]
        """
        ...

    def forward_scorer(self, embedding: np.ndarray) -> np.ndarray:
        """
        Predict fusion weights.

        Returns:
            weights ∈ [0, 1]^4, sum = 1
        """
        ...

    def update_gate(
        self,
        embedding: np.ndarray,
        should_store: bool,
        learning_rate: Optional[float] = None,
    ):
        """Update gate branch from storage feedback."""
        ...

    def update_scorer(
        self,
        embedding: np.ndarray,
        retrieval_utility: float,
        current_weights: np.ndarray,
    ):
        """Update scorer branch from retrieval utility."""
        ...
```

**Acceptance Criteria**:
- [ ] Shared projection layer for gate and scorer
- [ ] Separate update methods for gate vs scorer
- [ ] Joint gradient flow through shared layer
- [ ] Unit tests: forward pass, updates, alignment
- [ ] Integration test: gate predictions correlate with scorer utilities

**Testing Requirements**:
- Unit tests: `/mnt/projects/t4d/t4dm/tests/unit/test_feature_aligner.py`
  - Test forward passes (gate, scorer)
  - Test separate updates
  - Test gradient flow to shared layer
  - Test alignment: cosine similarity of learned representations
- Integration test: End-to-end with LearnedMemoryGate + LearnedFusionWeights

**Success Metrics**:
- Gate predictions and scorer utilities have correlation >= 0.6
- Shared projection improves both gate and scorer performance
- No degradation in either component vs standalone

**Risk Mitigation**:
- Can fall back to separate feature spaces if alignment hurts
- Monitor for gradient conflicts (gate vs scorer)
- Use separate learning rates for gate/scorer branches

**Effort**: Medium (2-3 days implementation + 1-2 days testing)

---

### Task 0.4: Integration with EpisodicMemory

**Files Modified**:
- `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py`

**Purpose**: Wire hierarchical sparse addressing into existing recall flow

**Implementation Details**:

```python
class EpisodicMemory:
    def __init__(self, ...):
        # ... existing initialization ...

        # NEW: Hierarchical retrieval components
        self.cluster_index = ClusterIndex(
            t4dx_vector_adapter=self.t4dx_vector_adapter,
            n_clusters=get_settings().cluster_index_clusters,
        )
        self.learned_sparse_index = LearnedSparseIndex(
            cluster_index=self.cluster_index,
        )

        # Optional: Feature alignment for gate + scorer
        if get_settings().enable_feature_alignment:
            self.feature_aligner = FeatureAligner()
            # Replace LearnedFusionWeights with aligner's scorer

        # Feature flags
        self.use_hierarchical_retrieval = get_settings().use_hierarchical_retrieval

    async def recall(
        self,
        query: str,
        k: int = 10,
        session_id: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> List[Episode]:
        # ... existing code for embedding, neuromod state ...

        # NEW: Route to hierarchical or flat retrieval
        if self.use_hierarchical_retrieval and self.cluster_index.needs_rebuild():
            await self.cluster_index.rebuild_index(session_id)

        if self.use_hierarchical_retrieval:
            # Use learned sparse index
            candidates = await self.learned_sparse_index.search(
                query_embedding=query_emb,
                k=k * 3,  # Get more candidates for re-ranking
                neuromod_state=neuromod_state,
                session_id=session_id,
            )
        else:
            # Fall back to flat k-NN (existing)
            candidates = await self.t4dx_vector_adapter.search(...)

        # ... existing re-ranking, scoring, return ...
```

**Acceptance Criteria**:
- [ ] EpisodicMemory initializes hierarchical components
- [ ] Feature flag controls hierarchical vs flat retrieval
- [ ] Automatic index rebuilding when stale
- [ ] No breaking changes to recall() API
- [ ] All existing tests pass

**Testing Requirements**:
- Integration tests: `/mnt/projects/t4d/t4dm/tests/integration/test_episodic_hierarchical.py`
  - Test recall with hierarchical retrieval enabled
  - Test fallback to flat when disabled
  - Test index rebuilding
  - Test feature alignment (if enabled)
- Regression tests: All existing episodic tests must pass

**Success Metrics**:
- Zero breaking changes
- Recall results identical (or 95%+ overlap) vs flat
- 10-67x speedup on large datasets

**Risk Mitigation**:
- Feature flag allows instant rollback
- Extensive regression testing
- Gradual rollout (dev → staging → production)

**Effort**: Small-Medium (1-2 days integration + 1 day testing)

---

### Task 0.5: Benchmarking and Documentation

**Files Created**:
- `/mnt/projects/t4d/t4dm/benchmarks/hierarchical_retrieval_benchmark.py`
- `/mnt/projects/t4d/t4dm/docs/HIERARCHICAL_RETRIEVAL.md`

**Purpose**: Validate speedup claims and document usage

**Benchmark Scenarios**:
1. **Scaling Test**: 1K, 10K, 100K, 1M memories
   - Measure: search time (flat vs hierarchical)
   - Measure: recall accuracy (% of flat k-NN results retrieved)
   - Expected: ~10x at 10K, ~67x at 1M memories

2. **Sparsity Ablation**: Fixed 5%, 10%, 20%, 50% vs learned
   - Measure: search time, utility, adaptation speed
   - Expected: Learned adapts to query difficulty

3. **Cluster Quality**: K-means vs HDBSCAN vs random
   - Measure: cluster coherence, retrieval accuracy
   - Expected: HDBSCAN best for irregular distributions

**Documentation Topics**:
- When to use hierarchical retrieval (>10K memories)
- How to tune cluster count
- Interpreting learned sparsity
- Fallback behavior
- Performance characteristics

**Acceptance Criteria**:
- [ ] Benchmark script with 3 scenarios
- [ ] Results table showing speedup vs memory count
- [ ] Documentation with usage guide
- [ ] Added to main README.md

**Effort**: Small (1 day benchmarking + 0.5 day docs)

---

### Phase 0 Summary

**Total Effort**: 1.5-2 weeks
**Files Created**: 3 new modules + 3 test files + 1 benchmark
**Files Modified**: `episodic.py`, `README.md`
**Risk**: Low (additive, feature-flagged)
**Deliverable**: 10-67x faster retrieval, production-ready

**Handoff Checklist**:
- [ ] All 5 tasks complete
- [ ] 1,259+ tests passing (no regressions)
- [ ] Benchmark results documented
- [ ] Feature flag = ON in dev, OFF in prod initially
- [ ] PR reviewed and merged

---

## Phase 1: Forward-Forward Learning (Week 3-4)

**Goal**: Add immediate Hebbian plasticity during encoding/retrieval

### Priority: CRITICAL (2/5)
**Impact**: Enables learning during forward pass (no backprop delay)
**Complexity**: Medium
**Risk**: Medium (changes learning dynamics)
**Timeline**: 2 weeks

---

### Background: Forward-Forward Learning

**Problem**: WW's current learning is DELAYED
- LearnedMemoryGate updates from outcome feedback (minutes to days later)
- LearnedFusionWeights updates from retrieval utility (after re-ranking)
- No learning during the encoding/retrieval pass itself

**Solution**: Forward-Forward (FF) learning
- Two passes: positive (real data) and negative (synthetic/corrupted)
- Goodness function: G(h) = Σh_i² - θ (layer activation magnitude)
- Update: Maximize G(positive) - G(negative)
- No backpropagation needed - local layer-wise learning

**Biological Motivation**: Hebbian plasticity happens DURING neural activity, not after reward feedback. FF gives both:
- **Immediate learning**: During encoding/retrieval (Hebbian)
- **Delayed learning**: From outcomes (dopaminergic) - KEEP EXISTING

---

### Task 1.1: Implement Forward-Forward Base Module

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/learning/forward_forward.py`

**Purpose**: Generic FF goodness functions and update rules

**Implementation Details**:

```python
"""
Forward-Forward Learning Module.

Implements layer-local learning without backpropagation.
Each layer has a "goodness" function that measures how well
it responds to data. Learning maximizes goodness for positive
data and minimizes it for negative data.

Reference: Hinton (2022) "The Forward-Forward Algorithm"
"""

import numpy as np
from typing import Tuple, Optional

class ForwardForwardLayer:
    """
    Single layer with FF learning.

    Goodness function: G(h) = Σh_i² - θ
    Where:
    - h = layer activations
    - θ = threshold (learned or fixed)

    Update rule:
    - Δw ∝ (G_pos - G_neg) * x * h
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        threshold: float = 2.0,
        learning_rate: float = 0.03,
        normalize: bool = True,
    ):
        # Xavier initialization
        self.W = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros(hidden_dim)
        self.threshold = threshold
        self.lr = learning_rate
        self.normalize = normalize

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass with goodness computation.

        Returns:
            (activations, goodness)
        """
        # Normalize input (standard in FF)
        if self.normalize:
            x = x / (np.linalg.norm(x) + 1e-8)

        # Linear + ReLU
        h = np.maximum(0, self.W @ x + self.b)

        # Goodness: sum of squared activations - threshold
        goodness = np.sum(h ** 2) - self.threshold

        return h, goodness

    def update(
        self,
        x_pos: np.ndarray,
        x_neg: np.ndarray,
        learning_rate: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        FF update from positive and negative examples.

        Args:
            x_pos: Real/useful data
            x_neg: Synthetic/useless data

        Returns:
            (goodness_pos, goodness_neg)
        """
        lr = learning_rate or self.lr

        # Normalize inputs
        if self.normalize:
            x_pos = x_pos / (np.linalg.norm(x_pos) + 1e-8)
            x_neg = x_neg / (np.linalg.norm(x_neg) + 1e-8)

        # Positive pass
        h_pos, g_pos = self.forward(x_pos)

        # Negative pass
        h_neg, g_neg = self.forward(x_neg)

        # Update: maximize g_pos - g_neg
        # Gradient: ∂G/∂W = 2 * h * x^T (for positive)
        dW_pos = 2 * np.outer(h_pos, x_pos)
        dW_neg = 2 * np.outer(h_neg, x_neg)

        # Update weights
        self.W += lr * (dW_pos - dW_neg)

        # Update biases
        db_pos = 2 * h_pos
        db_neg = 2 * h_neg
        self.b += lr * (db_pos - db_neg)

        return g_pos, g_neg


class GoodnessMixin:
    """
    Mixin for adding FF goodness to existing learned components.

    Usage:
        class MyLearnedComponent(GoodnessMixin):
            def __init__(self, ...):
                super().__init__(goodness_threshold=2.0)
                # ... rest of init ...

            def compute_features(self, x):
                # Your feature computation
                return features

            def update_from_feedback(self, ...):
                # Your existing delayed learning
                ...

            # NEW: Add FF learning
            def update_from_ff(self, x_pos, x_neg):
                self.ff_update(x_pos, x_neg)
    """

    def __init__(self, goodness_threshold: float = 2.0, ff_lr: float = 0.03):
        self.goodness_threshold = goodness_threshold
        self.ff_lr = ff_lr

    def compute_goodness(self, activations: np.ndarray) -> float:
        """Compute goodness of layer activations."""
        return np.sum(activations ** 2) - self.goodness_threshold

    def ff_gradient(
        self,
        activations: np.ndarray,
        input: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FF gradient for weight update.

        Returns:
            (dW, db)
        """
        dW = 2 * np.outer(activations, input)
        db = 2 * activations
        return dW, db


def generate_negative_sample(
    positive: np.ndarray,
    method: str = "corrupt",
    corruption_rate: float = 0.3,
) -> np.ndarray:
    """
    Generate negative sample from positive.

    Methods:
    - 'corrupt': Randomly zero out entries
    - 'shuffle': Shuffle entries
    - 'noise': Add Gaussian noise
    - 'antithesis': Negate and normalize
    """
    if method == "corrupt":
        mask = np.random.rand(len(positive)) > corruption_rate
        return positive * mask

    elif method == "shuffle":
        return np.random.permutation(positive)

    elif method == "noise":
        noise = np.random.randn(len(positive)) * 0.5
        return positive + noise

    elif method == "antithesis":
        return -positive / (np.linalg.norm(positive) + 1e-8)

    else:
        raise ValueError(f"Unknown method: {method}")
```

**Acceptance Criteria**:
- [ ] `ForwardForwardLayer` class with forward, update methods
- [ ] `GoodnessMixin` for adding FF to existing components
- [ ] `generate_negative_sample()` utility
- [ ] Unit tests: forward pass, update, goodness computation

**Testing Requirements**:
- Unit tests: `/mnt/projects/t4d/t4dm/tests/unit/test_forward_forward.py`
  - Test forward pass (activations, goodness)
  - Test update (weights change in expected direction)
  - Test negative sample generation (all methods)
  - Test GoodnessMixin integration

**Success Metrics**:
- Goodness increases for positive examples
- Goodness decreases for negative examples
- Weights converge after repeated updates

**Effort**: Small (1-2 days implementation + 0.5 day testing)

---

### Task 1.2: Add FF to LearnedMemoryGate

**Files Modified**:
- `/mnt/projects/t4d/t4dm/src/t4dm/learning/learned_gate.py` (if exists)
- Or create new file if gate is inline in episodic.py

**Purpose**: Gate learns during encoding pass, not just from delayed feedback

**Implementation Details**:

```python
from t4dm.learning.forward_forward import GoodnessMixin, generate_negative_sample

class LearnedMemoryGate(GoodnessMixin):
    """
    Learned storage gate with dual learning modes:

    1. IMMEDIATE (new): Forward-Forward learning during encoding
       - Positive: Memories with high neuromod signal (important/novel)
       - Negative: Corrupted embeddings or low-neuromod examples
       - Updates weights DURING the store() call

    2. DELAYED (existing): Bayesian logistic from outcome feedback
       - Positive: Memories that were useful in retrieval
       - Negative: Memories that were never retrieved
       - Updates weights from register_outcome() calls

    Both modes update the SAME weights, but with different signals:
    - FF provides immediate Hebbian learning
    - Bayesian provides long-term credit assignment
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 64,
        learning_rate: float = 0.01,
        ff_learning_rate: float = 0.03,
        goodness_threshold: float = 2.0,
        enable_ff: bool = True,
    ):
        # Initialize goodness mixin
        GoodnessMixin.__init__(
            self,
            goodness_threshold=goodness_threshold,
            ff_lr=ff_learning_rate,
        )

        # ... existing gate initialization ...
        self.enable_ff = enable_ff

    async def should_store(
        self,
        embedding: np.ndarray,
        neuromod_state: NeuromodulatorState,
        context: Optional[EpisodeContext] = None,
    ) -> Tuple[bool, float]:
        """
        Decide whether to store memory.

        NEW: If FF enabled, also perform immediate learning.
        """
        # ... existing decision logic ...
        p_store = self._predict_storage_probability(embedding, neuromod_state)

        # NEW: Forward-Forward immediate learning
        if self.enable_ff:
            # Positive: High neuromod signal indicates important memory
            is_positive = (
                neuromod_state.norepinephrine_gain > 0.6  # Novel
                or neuromod_state.dopamine_rpe > 0.3      # Rewarding
                or neuromod_state.acetylcholine_mode == "encoding"  # Active learning
            )

            if is_positive:
                # This is a positive example - should be stored
                # Generate negative counterpart
                negative_emb = generate_negative_sample(
                    embedding,
                    method="corrupt",
                    corruption_rate=0.3,
                )

                # FF update
                self._ff_update_immediate(
                    positive=embedding,
                    negative=negative_emb,
                    neuromod_state=neuromod_state,
                )

        # ... existing return ...
        return decision, p_store

    def _ff_update_immediate(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        neuromod_state: NeuromodulatorState,
    ):
        """
        Immediate FF update during encoding.

        Updates same weights that Bayesian learning uses, but with
        different signal (immediate Hebbian vs delayed outcome).
        """
        # Compute activations
        h_pos = self._compute_hidden(positive, neuromod_state)
        h_neg = self._compute_hidden(negative, neuromod_state)

        # Compute goodness
        g_pos = self.compute_goodness(h_pos)
        g_neg = self.compute_goodness(h_neg)

        # FF gradients
        dW_pos, db_pos = self.ff_gradient(h_pos, positive)
        dW_neg, db_neg = self.ff_gradient(h_neg, negative)

        # Update weights (maximize g_pos - g_neg)
        self.W += self.ff_lr * (dW_pos - dW_neg)
        self.b += self.ff_lr * (db_pos - db_neg)

        # Log for monitoring
        logger.debug(
            f"FF update: g_pos={g_pos:.3f}, g_neg={g_neg:.3f}, "
            f"delta={g_pos - g_neg:.3f}"
        )
```

**Acceptance Criteria**:
- [ ] LearnedMemoryGate inherits from GoodnessMixin
- [ ] FF update happens during should_store() for positive examples
- [ ] BOTH FF and Bayesian learning update same weights
- [ ] Feature flag enables/disables FF
- [ ] Unit tests: FF updates work correctly
- [ ] Integration test: Dual learning (FF + Bayesian) works together

**Testing Requirements**:
- Unit tests: `/mnt/projects/t4d/t4dm/tests/unit/test_learned_gate_ff.py`
  - Test FF update during should_store()
  - Test positive example detection (high neuromod)
  - Test negative sample generation
  - Test weight updates (FF vs Bayesian)
- Integration test: Full episode storage with FF enabled

**Success Metrics**:
- Gate adapts faster with FF enabled (fewer examples to learn)
- No degradation in final performance vs Bayesian-only
- FF and Bayesian gradients align (not conflicting)

**Effort**: Small-Medium (1-2 days implementation + 1 day testing)

---

### Task 1.3: Add FF to LearnedFusionWeights

**Files Modified**:
- `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` (LearnedFusionWeights class)

**Purpose**: Scorer learns during retrieval pass, not just from outcome feedback

**Implementation Details**:

```python
class LearnedFusionWeights(GoodnessMixin):
    """
    Query-dependent fusion weights with dual learning:

    1. IMMEDIATE (new): Forward-Forward during retrieval
       - Positive: Retrieved memories with high scores
       - Negative: Retrieved memories with low scores (or corrupted)
       - Updates DURING recall() to adapt weights

    2. DELAYED (existing): Outcome-based updates
       - Positive: Memories marked as useful by user/system
       - Negative: Memories marked as not useful
       - Updates from register_outcome() calls
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        ff_learning_rate: float = 0.03,
        goodness_threshold: float = 1.0,  # Lower than gate
        enable_ff: bool = True,
        n_components: int = 4,
    ):
        # Initialize goodness mixin
        GoodnessMixin.__init__(
            self,
            goodness_threshold=goodness_threshold,
            ff_lr=ff_learning_rate,
        )

        # ... existing fusion weights initialization ...
        self.enable_ff = enable_ff

    def compute_weights(
        self,
        query_embedding: np.ndarray,
        neuromod_state: Optional[NeuromodulatorState] = None,
    ) -> np.ndarray:
        """
        Compute fusion weights for this query.

        NEW: If FF enabled and we have retrieval results,
        perform immediate learning.
        """
        # ... existing weight computation ...
        weights = self._forward_pass(query_embedding, neuromod_state)

        return weights

    def update_from_retrieval(
        self,
        query_embedding: np.ndarray,
        retrieved_scores: List[float],
        neuromod_state: Optional[NeuromodulatorState] = None,
    ):
        """
        NEW: Immediate FF update during retrieval.

        Positive: Query when top results have high scores
        Negative: Corrupted query or query when scores are low
        """
        if not self.enable_ff:
            return

        # Compute average retrieval quality
        avg_score = np.mean(retrieved_scores)

        # Is this a good retrieval? (positive example)
        is_positive = avg_score > 0.5

        if is_positive:
            # Good retrieval - reinforce current weights
            positive = query_embedding
            negative = generate_negative_sample(
                query_embedding,
                method="corrupt",
                corruption_rate=0.2,
            )
        else:
            # Poor retrieval - treat query as negative
            # (we want to learn to assign different weights for this query)
            positive = generate_negative_sample(
                query_embedding,
                method="noise",  # Slight perturbation
            )
            negative = query_embedding

        # FF update
        self._ff_update_immediate(positive, negative, neuromod_state)

    def _ff_update_immediate(
        self,
        positive: np.ndarray,
        negative: np.ndarray,
        neuromod_state: Optional[NeuromodulatorState],
    ):
        """FF update for fusion weights."""
        # Compute hidden representations
        h_pos = self._compute_hidden(positive, neuromod_state)
        h_neg = self._compute_hidden(negative, neuromod_state)

        # Goodness
        g_pos = self.compute_goodness(h_pos)
        g_neg = self.compute_goodness(h_neg)

        # FF gradients (for W1, b1, W2, b2)
        dW1_pos, db1_pos = self.ff_gradient(h_pos, positive)
        dW1_neg, db1_neg = self.ff_gradient(h_neg, negative)

        # Update first layer
        self.W1 += self.ff_lr * (dW1_pos - dW1_neg)
        self.b1 += self.ff_lr * (db1_pos - db1_neg)

        # Note: W2, b2 updated separately by outcome-based learning

        logger.debug(
            f"Fusion FF update: g_pos={g_pos:.3f}, g_neg={g_neg:.3f}"
        )
```

**Acceptance Criteria**:
- [ ] LearnedFusionWeights inherits from GoodnessMixin
- [ ] FF update happens during recall() based on retrieval quality
- [ ] BOTH FF and outcome-based learning update same weights
- [ ] Feature flag enables/disables FF
- [ ] Unit tests: FF updates work correctly

**Testing Requirements**:
- Unit tests: `/mnt/projects/t4d/t4dm/tests/unit/test_fusion_weights_ff.py`
  - Test FF update from good retrieval (positive)
  - Test FF update from poor retrieval (negative)
  - Test weight updates align with outcomes
- Integration test: Full recall with FF-enabled fusion weights

**Success Metrics**:
- Fusion weights adapt faster with FF (fewer retrievals to learn)
- Retrieval quality improves faster with FF enabled
- No conflicts between FF and outcome-based gradients

**Effort**: Small-Medium (1-2 days implementation + 1 day testing)

---

### Task 1.4: Ablation Study and Tuning

**File**: `/mnt/projects/t4d/t4dm/experiments/forward_forward_ablation.py`

**Purpose**: Validate FF learning improves over delayed-only

**Experimental Design**:

```python
"""
Forward-Forward Ablation Study.

Compares 4 configurations:
1. Baseline: No learned components (fixed heuristics)
2. Delayed-only: Bayesian updates from outcomes
3. FF-only: Forward-Forward immediate learning
4. Dual: FF + Delayed (both enabled)

Metrics:
- Learning speed: Episodes to reach 80% performance
- Final performance: Retrieval quality after 1000 episodes
- Stability: Variance in performance
- Computational cost: Update time per episode
"""

import asyncio
import numpy as np
from typing import List, Dict
from t4dm.memory.episodic import EpisodicMemory
from t4dm.core.config import get_settings

async def run_ablation(
    n_episodes: int = 1000,
    n_trials: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Run ablation study across configurations.

    Returns:
        {
            "baseline": {"learning_speed": X, "final_perf": Y, ...},
            "delayed_only": {...},
            "ff_only": {...},
            "dual": {...},
        }
    """
    results = {}

    for config_name in ["baseline", "delayed_only", "ff_only", "dual"]:
        # Configure system
        settings = get_settings()
        if config_name == "baseline":
            settings.enable_learned_gate = False
            settings.enable_learned_fusion = False
        elif config_name == "delayed_only":
            settings.enable_learned_gate = True
            settings.enable_learned_fusion = True
            settings.enable_ff_gate = False
            settings.enable_ff_fusion = False
        elif config_name == "ff_only":
            settings.enable_learned_gate = True
            settings.enable_learned_fusion = True
            settings.enable_ff_gate = True
            settings.enable_ff_fusion = True
            settings.enable_delayed_learning = False
        else:  # dual
            settings.enable_learned_gate = True
            settings.enable_learned_fusion = True
            settings.enable_ff_gate = True
            settings.enable_ff_fusion = True
            settings.enable_delayed_learning = True

        # Run trials
        trial_results = []
        for trial in range(n_trials):
            metrics = await run_single_trial(
                config_name=config_name,
                n_episodes=n_episodes,
            )
            trial_results.append(metrics)

        # Aggregate
        results[config_name] = aggregate_metrics(trial_results)

    return results


async def run_single_trial(
    config_name: str,
    n_episodes: int,
) -> Dict[str, float]:
    """Run single trial and return metrics."""
    # ... implementation ...
    pass
```

**Metrics to Track**:
1. **Learning Speed**: Episodes until 80% of final performance
2. **Final Performance**: Retrieval quality at episode 1000
3. **Sample Efficiency**: Performance at episode 100, 500, 1000
4. **Stability**: Standard deviation of performance
5. **Computational Cost**: Average update time per episode

**Expected Results**:
- **FF-only**: Fast initial learning, may plateau early
- **Delayed-only**: Slow but steady, best final performance
- **Dual**: Best of both - fast learning AND high final performance
- **Baseline**: Worst performance, no adaptation

**Acceptance Criteria**:
- [ ] Ablation script runs all 4 configurations
- [ ] Statistical significance testing (t-test)
- [ ] Results visualization (learning curves)
- [ ] Written report with findings

**Effort**: Medium (2-3 days experiment design + 1 day analysis)

---

### Phase 1 Summary

**Total Effort**: 1.5-2 weeks
**Files Created**: 2 new modules + 3 test files + 1 experiment
**Files Modified**: `episodic.py`, learned gate (if separate)
**Risk**: Medium (changes learning, needs tuning)
**Deliverable**: Immediate + delayed dual learning, faster adaptation

**Handoff Checklist**:
- [ ] All 4 tasks complete
- [ ] Ablation study shows dual learning improves over delayed-only
- [ ] No degradation in existing test suite
- [ ] Feature flags for gradual rollout
- [ ] Documentation on FF tuning parameters

---

## Phase 2: Hopfield-Fenchel-Young Memory (Week 5-8)

**Goal**: Add energy-based associative memory with pattern completion

### Priority: SHOULD HAVE (3/5)
**Impact**: Exponential capacity, pattern completion from partial cues
**Complexity**: High
**Risk**: High (complex, may not beat Qdrant in practice)
**Timeline**: 4 weeks
**Decision**: Prototype first, evaluate vs Qdrant

---

### Background: Why HFY Memory?

**Current**: Qdrant vector store
- Capacity: Linear in number of memories
- Retrieval: k-NN similarity search
- Pattern completion: None (requires full query)

**HFY Memory**:
- Capacity: Exponential in dimension (α-entmax sparsity)
- Retrieval: Sparse attention (entmax-α)
- Pattern completion: Yes (partial cue → full pattern)

**Trade-offs**:
| Dimension | Qdrant | HFY Memory |
|-----------|--------|------------|
| Proven | ✓ (production) | ✗ (research) |
| Scalable | ✓ (billions) | ? (unknown) |
| Fast | ✓ (optimized) | ? (may be slow) |
| Interpretable | ✓ (clear results) | ✗ (energy landscape) |
| Pattern completion | ✗ | ✓ |

**Recommendation**: Build HFY as PARALLEL to Qdrant, not replacement. Let learned fusion choose best source.

---

### Task 2.1: Implement HFY Memory Core

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/hopfield_memory.py`

**Purpose**: Energy-based associative memory with sparse attention

**Implementation Details**:

```python
"""
Hopfield-Fenchel-Young (HFY) Memory.

Modern Hopfield network with exponential capacity via α-entmax sparsity.

Energy Function:
    E(ξ) = -lse_Ω(Xξ) + Ψ(ξ)

Where:
    - X: Stored patterns (N x d)
    - ξ: Query pattern (d)
    - lse_Ω: Log-sum-exp with conjugate Ω
    - Ψ: Regularizer (e.g., negative entropy)

Retrieval:
    ξ* = argmin_ξ E(ξ)
       = entmax_α(X^T u)  # Sparse attention

Where:
    - u: Update from energy minimization
    - entmax_α: Sparse softmax (α > 1)

Capacity:
    C ≈ d^(α-1) / log(d)  # Exponential in d for α > 1

Reference: Ramsauer et al. (2020) "Hopfield Networks is All You Need"
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.special import softmax

class HopfieldMemory:
    """
    Modern Hopfield network with sparse retrieval.

    Stores patterns and retrieves via energy minimization.
    """

    def __init__(
        self,
        dim: int = 1024,
        alpha: float = 2.0,  # Sparsity parameter (α=1: softmax, α>1: sparse)
        beta: float = 1.0,   # Inverse temperature
        max_iter: int = 5,   # Update iterations
    ):
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

        # Stored patterns (N x d)
        self.patterns: Optional[np.ndarray] = None
        self.pattern_ids: List[str] = []

    def store(
        self,
        patterns: np.ndarray,
        pattern_ids: List[str],
    ):
        """
        Store patterns in memory.

        Args:
            patterns: (N x d) array of patterns
            pattern_ids: List of pattern identifiers
        """
        if self.patterns is None:
            self.patterns = patterns
            self.pattern_ids = pattern_ids
        else:
            # Append new patterns
            self.patterns = np.vstack([self.patterns, patterns])
            self.pattern_ids.extend(pattern_ids)

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Retrieve patterns similar to query via energy minimization.

        Args:
            query: (d,) query pattern
            k: Number of patterns to retrieve

        Returns:
            (pattern_ids, attention_weights)
        """
        if self.patterns is None:
            return [], np.array([])

        # Normalize query
        query = query / (np.linalg.norm(query) + 1e-8)

        # Iterative update (energy minimization)
        u = query.copy()
        for _ in range(self.max_iter):
            # Compute attention: entmax_α(β * X^T u)
            logits = self.beta * (self.patterns @ u)
            attention = self._entmax_alpha(logits, alpha=self.alpha)

            # Update: u ← X^T attention
            u = self.patterns.T @ attention
            u = u / (np.linalg.norm(u) + 1e-8)

        # Final attention weights
        logits = self.beta * (self.patterns @ u)
        attention = self._entmax_alpha(logits, alpha=self.alpha)

        # Select top-k
        top_k_indices = np.argsort(attention)[-k:][::-1]
        top_k_ids = [self.pattern_ids[i] for i in top_k_indices]
        top_k_weights = attention[top_k_indices]

        return top_k_ids, top_k_weights

    def _entmax_alpha(
        self,
        logits: np.ndarray,
        alpha: float,
        n_iter: int = 50,
    ) -> np.ndarray:
        """
        α-entmax: Sparse generalization of softmax.

        For α=1: softmax
        For α>1: Sparse (many zeros)

        Uses bisection method to solve for threshold τ.
        """
        if alpha == 1.0:
            return softmax(logits)

        # Sort logits
        sorted_logits = np.sort(logits)[::-1]

        # Bisection to find threshold τ
        tau_min, tau_max = sorted_logits[-1] - 1, sorted_logits[0] + 1

        for _ in range(n_iter):
            tau = (tau_min + tau_max) / 2

            # Compute p(τ)
            p = np.maximum(0, sorted_logits - tau) ** (1 / (alpha - 1))

            # Check constraint: Σp = 1
            p_sum = p.sum()

            if p_sum > 1:
                tau_min = tau
            else:
                tau_max = tau

        # Final sparse probabilities
        tau = (tau_min + tau_max) / 2
        p = np.maximum(0, logits - tau) ** (1 / (alpha - 1))

        return p / (p.sum() + 1e-8)

    def pattern_completion(
        self,
        partial_query: np.ndarray,
        mask: np.ndarray,  # 1=known, 0=unknown
    ) -> np.ndarray:
        """
        Complete partial pattern.

        Args:
            partial_query: (d,) partial pattern
            mask: (d,) binary mask of known dimensions

        Returns:
            completed_pattern: (d,) full pattern
        """
        # Initialize with partial query
        completed = partial_query.copy()

        # Iterative refinement
        for _ in range(self.max_iter * 2):  # More iterations for completion
            # Retrieve similar patterns
            _, attention = self.retrieve(completed, k=len(self.pattern_ids))

            # Weighted average of patterns
            reconstruction = self.patterns.T @ attention

            # Keep known parts, update unknown
            completed = mask * partial_query + (1 - mask) * reconstruction

            # Normalize
            completed = completed / (np.linalg.norm(completed) + 1e-8)

        return completed
```

**Acceptance Criteria**:
- [ ] HopfieldMemory class with store, retrieve, pattern_completion
- [ ] α-entmax implementation (sparse attention)
- [ ] Iterative energy minimization update
- [ ] Unit tests: store, retrieve, pattern completion
- [ ] Capacity test: verify exponential scaling

**Effort**: Large (4-5 days implementation + 2 days testing)

---

### Task 2.2: Integrate HFY with EpisodicMemory

**Files Modified**:
- `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py`

**Purpose**: Add HFY as third retrieval source (Qdrant, Neo4j, HFY)

**Implementation Details**:

```python
class EpisodicMemory:
    def __init__(self, ...):
        # ... existing initialization ...

        # NEW: HFY memory (optional)
        self.hopfield_memory: Optional[HopfieldMemory] = None
        if get_settings().enable_hopfield_memory:
            self.hopfield_memory = HopfieldMemory(
                dim=get_settings().embedding_dim,
                alpha=get_settings().hopfield_alpha,
            )

        # Update fusion weights to 5 components if HFY enabled
        if self.hopfield_memory:
            self.fusion_weights = LearnedFusionWeights(n_components=5)
            # Components: semantic(Qdrant), recency, outcome, importance, associative(HFY)

    async def store(
        self,
        content: str,
        session_id: str,
        context: Optional[EpisodeContext] = None,
    ) -> Episode:
        # ... existing storage to Qdrant + Neo4j ...

        # NEW: Store in HFY memory if enabled
        if self.hopfield_memory:
            self.hopfield_memory.store(
                patterns=np.array([embedding]),
                pattern_ids=[str(episode.id)],
            )

        return episode

    async def recall(
        self,
        query: str,
        k: int = 10,
        session_id: Optional[str] = None,
    ) -> List[Episode]:
        # ... existing retrieval from Qdrant, Neo4j ...

        # NEW: Retrieve from HFY if enabled
        hfy_results = []
        if self.hopfield_memory:
            hfy_ids, hfy_weights = self.hopfield_memory.retrieve(
                query=query_emb,
                k=k * 2,
            )
            hfy_results = await self._fetch_episodes_by_ids(hfy_ids, session_id)

        # Combine sources with learned fusion
        if self.hopfield_memory:
            # 5-component fusion
            final_scores = self._fuse_scores_5way(
                qdrant_results,
                neo4j_results,
                hfy_results,
                query_emb,
                neuromod_state,
            )
        else:
            # 4-component fusion (existing)
            final_scores = self._fuse_scores_4way(...)

        # ... existing re-ranking, return ...
```

**Acceptance Criteria**:
- [ ] HFY memory integrated as optional third source
- [ ] Fusion weights extended to 5 components when HFY enabled
- [ ] Storage writes to HFY in addition to Qdrant/Neo4j
- [ ] Retrieval combines all three sources
- [ ] Feature flag controls HFY usage

**Effort**: Medium (2-3 days integration + 1-2 days testing)

---

### Task 2.3: Benchmark HFY vs Qdrant

**File**: `/mnt/projects/t4d/t4dm/benchmarks/hopfield_vs_qdrant_benchmark.py`

**Purpose**: Determine if HFY adds value in practice

**Benchmark Scenarios**:
1. **Capacity Scaling**: 1K, 10K, 100K patterns
   - Measure: Retrieval accuracy (recall@10)
   - Expected: HFY scales better at very large N

2. **Pattern Completion**: Queries with 30%, 50%, 70% of dimensions masked
   - Measure: Completion accuracy (cosine similarity to full pattern)
   - Expected: HFY completes patterns, Qdrant fails

3. **Sparse Queries**: Queries with only 10-20% non-zero dimensions
   - Measure: Retrieval quality
   - Expected: HFY handles sparse queries better

4. **Speed**: Query time for different N
   - Measure: Latency (ms)
   - Expected: Qdrant faster for N < 100K

**Decision Criteria**:
- **Keep HFY if**: Pattern completion useful OR capacity scaling critical
- **Use Qdrant-only if**: Speed more important, no sparse queries

**Effort**: Medium (2-3 days benchmarking + 1 day analysis)

---

### Phase 2 Summary

**Total Effort**: 4 weeks
**Files Created**: 1 new module + tests + benchmark
**Files Modified**: `episodic.py`
**Risk**: High (complex, unclear benefit)
**Deliverable**: HFY memory as optional alternative to Qdrant

**Decision Point**: After benchmarking, decide:
- **Option A**: HFY shows clear wins → promote to production
- **Option B**: HFY marginal → keep as research feature
- **Option C**: HFY worse → deprecate, use Qdrant only

---

## Phase 3: Pattern Separation (Week 9-10)

**Goal**: Add Dentate Gyrus-style orthogonalization to reduce interference

### Priority: SHOULD HAVE (4/5)
**Impact**: Reduced interference between similar memories
**Complexity**: Medium
**Risk**: Medium
**Timeline**: 2 weeks

---

### Task 3.1: Implement Pattern Separation Layer

**File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/pattern_separation.py`

**Purpose**: Sparse random projection for orthogonalization

**Implementation Details**:

```python
"""
Pattern Separation Layer (Dentate Gyrus analog).

Prevents interference between similar memories via sparse coding.

Algorithm:
1. Random sparse projection: x → h (1024 → 4096, 2% connectivity)
2. Winner-take-all sparsity: Keep top 5% activations, zero rest
3. Result: Similar inputs → orthogonal sparse codes

Biological Motivation:
- Dentate Gyrus has 10x more neurons than input (EC)
- Only ~5% active at any time (sparse)
- Random connectivity ensures orthogonalization
"""

import numpy as np
from typing import Optional

class PatternSeparationLayer:
    """
    Sparse random projection for pattern separation.

    Maps dense embeddings to sparse orthogonal codes.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        output_dim: int = 4096,  # Expansion (like DG)
        connectivity: float = 0.02,  # 2% sparse connectivity
        sparsity: float = 0.05,  # 5% active neurons
        seed: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.connectivity = connectivity
        self.sparsity = sparsity

        # Random sparse projection matrix
        rng = np.random.RandomState(seed)
        self.W = self._create_sparse_projection(rng)

    def _create_sparse_projection(self, rng) -> np.ndarray:
        """
        Create sparse random projection matrix.

        Each output neuron connects to ~2% of input neurons.
        Weights are +1 or -1 (binary).
        """
        W = np.zeros((self.output_dim, self.input_dim))

        for i in range(self.output_dim):
            # Select random input connections
            n_connections = int(self.input_dim * self.connectivity)
            connections = rng.choice(
                self.input_dim,
                size=n_connections,
                replace=False,
            )

            # Random +1 or -1 weights
            weights = rng.choice([-1, 1], size=n_connections)

            W[i, connections] = weights

        return W

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Orthogonalize input via sparse coding.

        Args:
            x: (d,) dense input

        Returns:
            h: (D,) sparse output (D > d)
        """
        # Linear projection
        h = self.W @ x

        # Winner-take-all sparsity
        threshold = np.percentile(h, 100 * (1 - self.sparsity))
        h = np.where(h > threshold, h, 0)

        # Normalize
        h = h / (np.linalg.norm(h) + 1e-8)

        return h

    def similarity_reduction(
        self,
        x1: np.ndarray,
        x2: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Measure interference reduction.

        Returns:
            (input_similarity, output_similarity)
        """
        # Input similarity
        input_sim = np.dot(x1, x2) / (
            np.linalg.norm(x1) * np.linalg.norm(x2) + 1e-8
        )

        # Output similarity (after separation)
        h1 = self.forward(x1)
        h2 = self.forward(x2)
        output_sim = np.dot(h1, h2) / (
            np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-8
        )

        return input_sim, output_sim
```

**Acceptance Criteria**:
- [ ] PatternSeparationLayer with forward, similarity_reduction
- [ ] Sparse random projection (2% connectivity)
- [ ] Winner-take-all sparsity (5% active)
- [ ] Unit tests: projection, sparsity, interference reduction
- [ ] Verify: Similar inputs → dissimilar outputs

**Effort**: Small-Medium (2-3 days implementation + 1 day testing)

---

### Task 3.2: Integrate with BufferManager

**Files Modified**:
- `/mnt/projects/t4d/t4dm/src/t4dm/memory/buffer_manager.py` (if exists)
- Or `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py`

**Purpose**: Apply pattern separation before buffer storage

**Implementation**:
- Insert pattern separation layer between encoding and buffer
- Store both dense (original) and sparse (separated) representations
- Use sparse for interference-sensitive operations (consolidation)

**Effort**: Small (1-2 days integration + 1 day testing)

---

### Phase 3 Summary

**Total Effort**: 2 weeks
**Files Created**: 1 module + tests
**Files Modified**: Buffer manager or episodic
**Risk**: Medium
**Deliverable**: Reduced interference for similar memories

---

## Implementation Schedule

### Week 1-2: Phase 0 (Hierarchical Sparse Addressing)
- **Day 1-2**: ClusterIndex implementation
- **Day 3-4**: LearnedSparseIndex implementation
- **Day 5-6**: FeatureAligner implementation
- **Day 7-8**: Integration + benchmarking
- **Day 9-10**: Testing + documentation

**Deliverable**: 10-67x faster retrieval

---

### Week 3-4: Phase 1 (Forward-Forward Learning)
- **Day 1-2**: ForwardForward base module
- **Day 3-4**: FF for LearnedMemoryGate
- **Day 5-6**: FF for LearnedFusionWeights
- **Day 7-10**: Ablation study + tuning

**Deliverable**: Immediate + delayed dual learning

---

### Week 5-8: Phase 2 (HFY Memory) - OPTIONAL
- **Week 1**: HFY core implementation
- **Week 2**: Integration with EpisodicMemory
- **Week 3**: Benchmarking vs Qdrant
- **Week 4**: Decision + refinement

**Deliverable**: HFY memory (if benchmarks positive)

---

### Week 9-10: Phase 3 (Pattern Separation) - OPTIONAL
- **Week 1**: PatternSeparationLayer implementation
- **Week 2**: Integration + testing

**Deliverable**: Reduced interference

---

## Success Metrics

### Phase 0 (Must Have)
- [ ] 10-67x speedup vs flat k-NN
- [ ] 95%+ retrieval accuracy (vs flat baseline)
- [ ] Zero breaking changes
- [ ] All existing tests pass

### Phase 1 (Must Have)
- [ ] Faster learning (50% fewer episodes to 80% performance)
- [ ] No degradation in final performance
- [ ] Stable training (no divergence)
- [ ] FF + delayed learning outperform either alone

### Phase 2 (Should Have - Decision After Benchmarks)
- [ ] Pattern completion works (70%+ accuracy from 50% cue)
- [ ] Competitive with Qdrant on speed (<2x slower)
- [ ] Clear use case identified

### Phase 3 (Should Have - If Interference Observed)
- [ ] 30-50% reduction in similarity for similar inputs
- [ ] No degradation in retrieval quality
- [ ] Improved consolidation quality

---

## Risk Mitigation

### Technical Risks
1. **Hierarchical retrieval slower than expected**
   - Mitigation: Keep flat k-NN as fallback, feature flag

2. **FF learning unstable**
   - Mitigation: Careful learning rate tuning, conservative thresholds

3. **HFY memory underperforms Qdrant**
   - Mitigation: Benchmark early, make go/no-go decision

4. **Pattern separation increases dimensionality too much**
   - Mitigation: Adjustable expansion ratio, monitor memory usage

### Integration Risks
1. **Breaking changes to existing API**
   - Mitigation: All changes additive, feature-flagged

2. **Test regressions**
   - Mitigation: Run full test suite after each task

3. **Performance degradation**
   - Mitigation: Benchmark before/after each phase

### Timeline Risks
1. **Phases take longer than estimated**
   - Mitigation: Phase 0-1 are must-have, Phase 2-3 optional

2. **PhD defense prep conflicts (Jan 28, 2026)**
   - Mitigation: Phase 0-1 by Dec 20, pause for PhD if needed

---

## Configuration

New settings in `/mnt/projects/t4d/t4dm/src/t4dm/core/config.py`:

```python
class Settings(BaseSettings):
    # ... existing settings ...

    # Phase 0: Hierarchical Sparse Addressing
    use_hierarchical_retrieval: bool = False  # Feature flag
    cluster_index_clusters: int = 100
    cluster_index_rebuild_threshold: float = 0.1
    min_memories_for_clustering: int = 500

    # Phase 1: Forward-Forward Learning
    enable_ff_gate: bool = False  # Feature flag
    enable_ff_fusion: bool = False  # Feature flag
    ff_learning_rate: float = 0.03
    ff_goodness_threshold_gate: float = 2.0
    ff_goodness_threshold_fusion: float = 1.0
    enable_delayed_learning: bool = True  # Can disable to test FF-only

    # Phase 2: HFY Memory (Optional)
    enable_hopfield_memory: bool = False  # Feature flag
    hopfield_alpha: float = 2.0  # Sparsity parameter
    hopfield_beta: float = 1.0  # Inverse temperature

    # Phase 3: Pattern Separation (Optional)
    enable_pattern_separation: bool = False  # Feature flag
    pattern_separation_expansion: int = 4096  # Output dim
    pattern_separation_connectivity: float = 0.02
    pattern_separation_sparsity: float = 0.05

    # Feature Alignment (Phase 0.3)
    enable_feature_alignment: bool = False  # Feature flag
    feature_alignment_dim: int = 128
```

---

## Testing Strategy

### Unit Tests
- All new modules have >90% coverage
- Test both happy path and edge cases
- Test numerical stability (no NaN, Inf)

### Integration Tests
- Test end-to-end flows with new components
- Test feature flag on/off states
- Test backward compatibility

### Regression Tests
- All existing 1,259 tests must pass
- No degradation in performance benchmarks

### Ablation Tests
- Compare new vs old approaches
- Statistical significance testing
- Learning curves and convergence

---

## Documentation Updates

Files to update:
- `/mnt/projects/t4d/t4dm/README.md` - Add new features
- `/mnt/projects/t4d/t4dm/ARCHITECTURE.md` - Update architecture diagram
- `/mnt/projects/t4d/t4dm/docs/HIERARCHICAL_RETRIEVAL.md` - New doc
- `/mnt/projects/t4d/t4dm/docs/FORWARD_FORWARD_LEARNING.md` - New doc
- `/mnt/projects/t4d/t4dm/docs/HOPFIELD_MEMORY.md` - New doc (if Phase 2)
- `/mnt/projects/t4d/t4dm/docs/PATTERN_SEPARATION.md` - New doc (if Phase 3)

---

## Handoff and Continuation

### State Checkpoints

After each phase, create checkpoint document:
- `docs/checkpoints/PHASE_0_COMPLETE.md`
- `docs/checkpoints/PHASE_1_COMPLETE.md`
- etc.

Include:
- What was implemented
- What was tested
- What was learned
- Next steps

### Continuation Guide

If interrupted, read:
1. This plan document
2. Latest checkpoint document
3. `docs/SESSION_STATE.md` (existing)
4. Git log for recent commits

---

## References

1. **Gap Analysis**: `/home/aaron/mem/NEURAL_MEMORY_VS_T4DM_COMPARISON.md`
2. **WW Architecture**: `/mnt/projects/t4d/t4dm/ARCHITECTURE.md`
3. **Forward-Forward**: Hinton (2022) "The Forward-Forward Algorithm"
4. **Hopfield Networks**: Ramsauer et al. (2020) "Hopfield Networks is All You Need"
5. **Pattern Separation**: Treves & Rolls (1994) "Computational analysis of the role of the hippocampus"

---

## Conclusion

This plan provides a structured, phased approach to integrating CompBio neural memory components into T4DM. The implementation prioritizes:

1. **Must Have** (Phase 0-1): Hierarchical addressing + FF learning
2. **Should Have** (Phase 2-3): HFY memory + Pattern separation (evaluate first)
3. **Incremental**: Each phase standalone, no breaking changes
4. **Risk-managed**: Feature flags, benchmarks, fallbacks

**Total Timeline**: 4-12 weeks (depending on optional phases)
**Next Action**: Begin Phase 0, Task 0.1 (ClusterIndex)
