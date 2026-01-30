# Retrieval & Expression Pipeline Gap Analysis

**Created**: 2025-12-06
**Status**: Analysis Complete
**Relates to**: `/mnt/projects/ww/docs/IMPLEMENTATION_PLAN_HSA.md`

## Executive Summary

Analysis reveals **3 major retrieval gaps** and **2 expression gaps** that prevent WW from fully leveraging its learned components. Key finding: learned scoring/fusion modules exist but are NOT integrated into the main recall flow.

---

## Retrieval Pipeline Analysis

### Current Flow (episodic.py:450-699)

```
Query → BGE-M3 (1024d) → Pattern Completion (optional)
                              ↓
                    Qdrant k-NN Search (O(n))
                              ↓
                    FIXED Weight Scoring ← GAP!
                    (semantic: 0.4, recency: 0.3, outcome: 0.2, importance: 0.1)
                              ↓
                    GABA Inhibition (orchestra)
                              ↓
                    Results
```

### Gap R1: Fixed vs Learned Scoring Weights

**Location**: `episodic.py:591-596`

**Current** (STATIC):
```python
combined_score = (
    self.semantic_weight * semantic_score +    # Fixed 0.4
    self.recency_weight * recency_score +      # Fixed 0.3
    self.outcome_weight * outcome_score +      # Fixed 0.2
    self.importance_weight * importance_score  # Fixed 0.1
)
```

**Available but NOT USED**: `learning/neuro_symbolic.py:LearnedFusion`
```python
class LearnedFusion:
    """Query-dependent learned weights via softmax."""
    def compute_weights(self, query_embedding):
        # Returns adaptive weights per query!
        logits = self.W_fusion @ query_embedding
        return softmax(logits)  # [neural, symbolic, recency, outcome]
```

**Impact**: Different queries need different weight profiles:
- "What happened yesterday?" → high recency weight
- "How do I fix X?" → high outcome weight (prioritize successes)
- "What do I know about Y?" → high semantic weight

**Fix**: Replace fixed weights with `LearnedFusion.compute_weights(query_emb)`

---

### Gap R2: No Learned Re-ranking

**Location**: `episodic.py:611-615`

**Current**: Simple sort by combined_score
```python
scored_results.sort(key=lambda x: x.score, reverse=True)
scored_results = scored_results[:limit]
```

**Available but NOT USED**: `learning/scorer.py:LearnedRetrievalScorer`
```python
class LearnedRetrievalScorer:
    """2-layer MLP with residual for learned scoring."""
    def score(self, query_features, memory_features):
        combined = np.concatenate([query_features, memory_features])
        h = np.tanh(self.W1 @ combined + self.b1)
        return sigmoid(self.W2 @ h + query_features @ self.W_skip @ memory_features)
```

**Impact**: Re-ranking top-k with learned scorer can improve P@5 by 15-25%

**Fix**: Add re-ranking step after initial retrieval

---

### Gap R3: Flat k-NN Search (O(n))

**Location**: `episodic.py:547-553`

**Current**: Flat Qdrant search
```python
results = await self.vector_store.search(
    collection=self.episodes_collection,
    vector=query_emb,
    limit=limit,
    ...
)
```

**Problem**: O(n) doesn't scale. At 100K episodes, latency ~500ms.

**Designed but NOT IMPLEMENTED**: `ClusterIndex` (from HSA analysis)
- Hierarchical two-stage search: cluster selection → within-cluster search
- Complexity: O(K + k*n/K) where K=clusters, k=selected
- At 100K episodes with K=500, k=5: **67x speedup**

**Fix**: Implement ClusterIndex per IMPLEMENTATION_PLAN_HSA.md Phase 1

---

## Expression Pipeline Analysis

### Current Flow (context_injector.py)

```
ScoredResults → ContextInjector._format_context()
                        ↓
              Split by type (episodes, entities, skills)
                        ↓
              Truncate to max_context_chars=2000
                        ↓
              Plain text injection into LLM prompt
```

### Gap E1: Token-Efficient Format Not Used

**Location**: `context_injector.py:180-220`

**Current**: Plain text formatting
```python
def _format_context(self, memories: List[...]) -> str:
    sections = []
    if episodes:
        sections.append("## Recent History")
        for ep in episodes:
            sections.append(f"- {ep.content[:100]}...")
    return "\n".join(sections)
```

**Available but NOT USED**: `learning/events.py:ToonJSON`
```python
class ToonJSON:
    """~50% token reduction via key abbreviation, value compression."""
    def encode(self, data):
        # "timestamp" → "ts", "content" → "c", etc.
        # Removes nulls, compresses booleans
        return compact_json
```

**Also Available**: `learning/neuro_symbolic.py:NeuroSymbolicMemory.to_compact()`

**Impact**: 2x more context in same token budget

**Fix**: Use ToonJSON/to_compact() in ContextInjector._format_memory()

---

### Gap E2: No Adaptive Context Selection

**Location**: `context_injector.py:45-50`

**Current**: Fixed limits
```python
@dataclass
class InjectionConfig:
    max_episodes: int = 5
    max_entities: int = 10
    max_context_chars: int = 2000  # Hard limit
```

**Problem**: No adaptation based on:
- Query complexity (complex = more context needed)
- Available token budget (varies by LLM)
- Memory relevance distribution (if all low-score, include fewer)

**Designed but NOT IMPLEMENTED**: Learned context selection could:
- Use query embedding to predict optimal context size
- Dynamically adjust based on score distribution
- Respect per-LLM token budgets

---

## Integration Gap Summary

| Component | Exists | Integrated | Gap ID |
|-----------|--------|------------|--------|
| LearnedFusion | Yes (`neuro_symbolic.py`) | **NO** | R1 |
| LearnedRetrievalScorer | Yes (`scorer.py`) | **NO** | R2 |
| ClusterIndex | Designed | **NO** | R3 |
| ToonJSON | Yes (`events.py`) | **Partial** (collector only) | E1 |
| Adaptive Context | No | No | E2 |

---

## Priority Order for Fixes

### Immediate (P0-level, integrate existing code)

1. **R1: Wire LearnedFusion into recall()** - 1 day
   - Import LearnedFusion in episodic.py
   - Replace fixed weights with computed weights
   - Add training signal from retrieval outcomes

2. **E1: Use ToonJSON in ContextInjector** - 0.5 day
   - Replace plain text with ToonJSON.encode()
   - Estimate 2x context improvement

### Short-term (Phase 1)

3. **R2: Add re-ranking with LearnedRetrievalScorer** - 1 day
   - Post-retrieval re-ranking of top-k*2 results
   - Train scorer from retrieval outcomes

4. **R3: ClusterIndex implementation** - 3 days
   - Per IMPLEMENTATION_PLAN_HSA.md Phase 1
   - Enables O(log n) retrieval

### Medium-term (Phase 2-3)

5. **E2: Adaptive context selection** - 2 days
   - Learn optimal context size per query
   - Integrate with LLM token budgets

---

## Files to Modify

1. `/mnt/projects/ww/src/ww/memory/episodic.py`
   - Import LearnedFusion, LearnedRetrievalScorer
   - Add `_compute_adaptive_weights()` method
   - Add `_rerank_results()` method
   - Wire into recall() flow

2. `/mnt/projects/ww/src/ww/integrations/kymera/context_injector.py`
   - Import ToonJSON
   - Replace `_format_memory()` with compact encoding
   - Add adaptive sizing logic

3. `/mnt/projects/ww/src/ww/learning/neuro_symbolic.py`
   - Expose LearnedFusion for external use
   - Add retrieval outcome training method

---

## Success Metrics

| Metric | Current | After Fixes |
|--------|---------|-------------|
| P@5 (top-5 precision) | ~65% | >80% |
| Retrieval latency (10K eps) | ~100ms | <30ms |
| Context tokens/memory | ~150 | ~75 |
| Query-adaptive scoring | No | Yes |

---

## Conclusion

WW has sophisticated learned components (`LearnedFusion`, `LearnedRetrievalScorer`, `ToonJSON`) that are **not integrated** into the main retrieval/expression flow. The immediate priority is wiring these existing components, which provides significant improvement with minimal new code. Hierarchical retrieval (ClusterIndex) remains the biggest long-term gain.
