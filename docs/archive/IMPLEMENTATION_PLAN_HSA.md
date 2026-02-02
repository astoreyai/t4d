# T4DM HSA-Inspired Implementation Plan

**Created**: 2025-12-06
**Status**: Design Complete - Ready for Implementation
**Based on**: arxiv:2511.23319 (Hierarchical Sparse Attention)

## Executive Summary

This plan addresses critical gaps in WW's memory system identified through HSA paper analysis:

1. **Embedding Pipeline** (P0) - Raw BGE-M3 used without optimization
2. **Hierarchical Retrieval** (Phase 1) - Flat k-NN doesn't scale
3. **Learned Sparse Addressing** (Phase 2) - Fixed 10% sparsity not learned
4. **Joint Optimization** (Phase 3) - Gate and retrieval train independently

## Priority Order

| Priority | Component | Effort | Impact | Status |
|----------|-----------|--------|--------|--------|
| **P0a** | Learned Content Projection | 2 days | High | NOT STARTED |
| **P0b** | Semantic Context Embedding | 1 day | Medium | NOT STARTED |
| **P0c** | Learned Re-ranking | 2 days | High | NOT STARTED |
| **Phase 1** | ClusterIndex | 3 days | High | DESIGNED |
| **Phase 2** | LearnedSparseIndex | 4 days | High | DESIGNED |
| **Phase 3** | Joint Optimization | 3 days | Medium | DESIGNED |
| **Phase 4** | Testing Protocols | 2 days | Required | IN PROGRESS |
| **Phase 5** | Performance Optimization | 2 days | Medium | NOT STARTED |

---

## P0: Embedding Pipeline Fixes

### P0a: Learned Content Projection (1024→128)

**Location**: `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py`

**Current Problem** (line 321-322):
```python
content_features = content_embedding[:self.CONTENT_DIM]  # Raw 1024-dim
```

**Solution**: Add trainable projection layer:
```python
class LearnedMemoryGate:
    def __init__(self, ...):
        # ... existing code ...

        # Learned content projection: 1024 → 128
        self.content_projection = np.random.randn(128, 1024).astype(np.float32) * 0.01
        self.content_projection_bias = np.zeros(128, dtype=np.float32)

    def _project_content(self, embedding: np.ndarray) -> np.ndarray:
        """Project content embedding to task-specific space."""
        projected = self.content_projection @ embedding + self.content_projection_bias
        return np.tanh(projected)  # Bounded activation
```

**Update feature dimensions**:
- CONTENT_DIM: 1024 → 128
- TOTAL_DIM: 1143 → 247 (significant reduction)

### P0b: Semantic Context Embedding

**Location**: `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py` lines 395-421

**Current Problem**: Hash-based random projection (NO SEMANTICS):
```python
np.random.seed(hash(s) % (2**32))
embedding = np.random.randn(dim).astype(np.float32)
```

**Solution**: Use BGE-M3 for context strings:
```python
async def _embed_context_string(self, s: str, dim: int) -> np.ndarray:
    """Semantic embedding for context strings."""
    if s in self._string_embed_cache:
        return self._string_embed_cache[s]

    # Use BGE-M3 for semantic understanding
    full_emb = await self.embedding_provider.embed_query(s)

    # PCA or learned projection to target dim
    projected = self.context_projection @ np.array(full_emb)

    self._string_embed_cache[s] = projected
    return projected
```

### P0c: Learned Re-ranking

**Location**: New file `/mnt/projects/t4d/t4dm/src/t4dm/memory/reranker.py`

**Design**: Lightweight MLP that re-ranks Qdrant results:
```python
class LearnedReranker:
    """Re-rank retrieval results using learned relevance model."""

    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 64):
        self.W1 = np.random.randn(hidden_dim, embedding_dim * 2) * 0.01
        self.W2 = np.random.randn(1, hidden_dim) * 0.01

    def score(self, query_emb: np.ndarray, doc_emb: np.ndarray) -> float:
        """Compute relevance score for query-document pair."""
        combined = np.concatenate([query_emb, doc_emb])
        hidden = np.tanh(self.W1 @ combined)
        return float(sigmoid(self.W2 @ hidden))
```

---

## Phase 1: Hierarchical Episode Retrieval

### New File: `/mnt/projects/t4d/t4dm/src/t4dm/memory/cluster_index.py`

**Core Class**: `ClusterIndex` - CA3-like semantic grouping

**Key Components**:
- `ClusterMeta`: Centroid + statistics per cluster
- `register_cluster()`: Called during sleep consolidation
- `select_clusters()`: NE-modulated cluster selection
- `record_retrieval_outcome()`: Learning signal for cluster priority

**Integration Points**:
1. `SleepConsolidation.rem_phase()` → registers clusters
2. `EpisodicMemory.recall()` → two-stage search
3. `learn_from_outcome()` → updates cluster statistics

**Complexity Improvement**:
- Current: O(n) flat search
- Hierarchical: O(K + k*n/K) where K=clusters, k=selected
- For 100K episodes, K=500, k=5: **67x speedup**

### Modified Recall Flow

```
Query → ClusterIndex.select_clusters(q, ne_gain, ach_mode)
                    ↓
        [cluster_1, cluster_2, ..., cluster_k]
                    ↓
        Parallel Qdrant search with cluster filters
                    ↓
        Merge + Re-rank → Results
```

---

## Phase 2: Learned Sparse Addressing

### New File: `/mnt/projects/t4d/t4dm/src/t4dm/memory/learned_sparse_index.py`

**Core Class**: `LearnedSparseIndex` - Replaces fixed 10% sparsity

**Architecture**:
```
q [d=1024]
    │
    ▼
Shared MLP → h [hidden=256]
    │
    ├─→ Cluster Head → softmax → cluster attention [K]
    ├─→ Feature Head → sigmoid → feature attention [d]
    └─→ Sparsity Gate → sigmoid → sparsity level [1]
```

**Training Signal**:
- Retrieval outcomes (success/failure)
- Per-cluster contribution scores
- Neuromodulator guidance (NE, ACh)

**Key Methods**:
- `forward()`: Compute sparse addressing from query
- `register_pending()`: Track for training
- `update()`: Online gradient descent from outcomes

---

## Phase 3: Joint Gating-Retrieval Optimization

### Joint Loss Function

```
L_joint = L_gate + λ_r*L_retrieval + λ_c*L_consistency + λ_d*L_diversity
```

**Components**:
1. `L_gate`: Existing BCE on utility prediction
2. `L_retrieval`: Existing fusion ranking loss
3. `L_consistency` (NEW): Gate predictions ↔ retrieval scores alignment
4. `L_diversity` (NEW): Entropy of gate decisions (prevents collapse)

### New File: `/mnt/projects/t4d/t4dm/src/t4dm/memory/feature_aligner.py`

Projects gate features → retrieval space for consistency:
```python
class FeatureAligner:
    def project(self, gate_features: np.ndarray) -> np.ndarray:
        """Project 1143-dim gate features to 4-dim retrieval space."""
        return self.W @ gate_features + self.b
```

### Neuromodulator Learning Rate Modulation

```python
def get_gate_learning_rate(base_lr, neuromod_state):
    lr = base_lr
    lr *= 1.0 + abs(neuromod_state.dopamine_rpe)  # Surprise boost
    lr *= neuromod_state.norepinephrine_gain       # Arousal boost
    if neuromod_state.acetylcholine_mode == "encoding":
        lr *= 1.3  # Faster learning in encoding
    return clip(lr, base_lr * 0.3, base_lr * 3.0)
```

---

## Phase 4: Testing Protocols

### Test Categories

1. **Hierarchical Retrieval Tests**
   - Pattern completion accuracy
   - Cluster coherence (semantic grouping)
   - Latency scaling (O(log n) target)

2. **Sparse Addressing Tests**
   - Adaptive sparsity levels
   - Addressing accuracy
   - Interference resistance

3. **Joint Optimization Tests**
   - Gate-retrieval correlation
   - Consistency loss convergence
   - Catastrophic forgetting resistance

4. **Biological Validation**
   - DG pattern separation ratio (~0.5% target)
   - CA3 pattern completion threshold
   - CA1 temporal integration window

### Performance Benchmarks

| Metric | Current | Target |
|--------|---------|--------|
| Retrieval latency (100K eps) | ~500ms | <50ms |
| Gate accuracy | ~65% | >80% |
| Storage efficiency | baseline | +20% |
| Memory usage | 1024-dim | 128-dim |

---

## Implementation Schedule

### Week 1: P0 Embedding Fixes
- Day 1-2: P0a Learned projection
- Day 3: P0b Semantic context
- Day 4-5: P0c Learned re-ranking

### Week 2: Phase 1-2 Core Implementation
- Day 1-2: ClusterIndex
- Day 3-4: LearnedSparseIndex
- Day 5: Integration with recall()

### Week 3: Phase 3 + Testing
- Day 1-2: Joint optimization + FeatureAligner
- Day 3-4: Test suite implementation
- Day 5: Performance benchmarks

---

## Files to Create

1. `/mnt/projects/t4d/t4dm/src/t4dm/memory/cluster_index.py` (~300 lines)
2. `/mnt/projects/t4d/t4dm/src/t4dm/memory/learned_sparse_index.py` (~400 lines)
3. `/mnt/projects/t4d/t4dm/src/t4dm/memory/feature_aligner.py` (~100 lines)
4. `/mnt/projects/t4d/t4dm/src/t4dm/memory/reranker.py` (~150 lines)
5. `/mnt/projects/t4d/t4dm/tests/unit/test_cluster_index.py` (~200 lines)
6. `/mnt/projects/t4d/t4dm/tests/unit/test_sparse_index.py` (~200 lines)
7. `/mnt/projects/t4d/t4dm/tests/unit/test_joint_optimization.py` (~150 lines)

## Files to Modify

1. `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py` - Add projection, modulated LR
2. `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` - Hierarchical recall, joint optimization
3. `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/sleep.py` - Register clusters
4. `/mnt/projects/t4d/t4dm/src/t4dm/core/config.py` - New settings

---

## Risk Mitigation

1. **Circular Dependency** → Use RPE signals, not raw outcomes
2. **Feature Drift** → Continuous FeatureAligner adaptation
3. **Gate Collapse** → Diversity loss + monitoring
4. **Performance Regression** → Fallback to flat search if coverage <30%

---

## Success Criteria

- [ ] All 1,257+ tests passing
- [ ] Retrieval latency <100ms at 100K episodes
- [ ] Gate accuracy >75%
- [ ] Consistency loss <0.15
- [ ] No decision rate >85% (diversity maintained)
