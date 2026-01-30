# NEXUS Phase 5-8: Testing, Visualization, Optimization

**NEXUS** = **N**euro-symbolic **E**xperience **X**traction with **U**nified **S**coring

## Current State
- Tests: 1273 passed
- Coverage: 54%
- Target: 80%+

## Phase 5: Test Coverage Sprint (80%+)

### 5.1 Critical Coverage Gaps (0% → 80%+)
| Module | Current | Target | Priority |
|--------|---------|--------|----------|
| `integration/ccapi_*` | 0% | 90% | P0 |
| `api/routes/*` | 0% | 80% | P1 |
| `learning/collector` | 41% | 85% | P0 |
| `learning/events` | 41% | 90% | P0 |
| `hooks/*` | 22-50% | 80% | P1 |
| `core/actions` | 0% | 75% | P2 |
| `core/memory_gate` | 0% | 80% | P2 |

### 5.2 Test Types
1. **Unit Tests** - Isolated component testing
2. **Integration Tests** - Qdrant/Neo4j with real data
3. **E2E Bash Tests** - Full pipeline validation
4. **Wikipedia Corpus Tests** - Ground truth reproducibility

### 5.3 Wikipedia Ground Truth
- Download: Simple English Wikipedia (~200MB)
- Extract: 50k articles for test corpus
- Use cases:
  - Entity extraction accuracy
  - Memory consolidation quality
  - Retrieval relevance scoring
  - Learning system convergence

## Phase 6: Observability & Metrics

### 6.1 Prometheus Integration
- Memory operation latencies
- Retrieval quality scores
- Learning convergence metrics
- Buffer utilization

### 6.2 Memory Trace Visualization
- Network graph of memory relationships
- Temporal decision flow
- Credit assignment paths
- Eligibility trace decay

## Phase 7: Hinton Architecture Review

### 7.1 Analysis Points
- Score fusion optimization (60/40 → learned)
- Hebbian update completeness
- Symbol grounding via embeddings
- Reconsolidation mechanisms

### 7.2 Visual Brain Representation
- 3D neural graph
- Real-time activation display
- Memory trace pathways
- Decision explanation graphs

## Phase 8: Branding

### Acronym Candidates
1. **NEXUS** - Neuro-symbolic Experience Xtraction Unified Scoring
2. **MNEME** - Memory Network for Experience-based Machine Evolution
3. **SYNAPTIC** - Symbolic-Neural Adaptive Procedural Trace Integration Core
4. **ENGRAM** - Experience-based Neural Graph for Retrieval And Memory

## Execution Order

```
Phase 5.1: Unit tests for learning module (2h)
Phase 5.2: Integration tests for ccapi (1h)
Phase 5.3: E2E bash tests (1h)
Phase 5.4: Wikipedia corpus setup (1h)
Phase 5.5: Remaining coverage gaps (2h)
Phase 6.1: Prometheus metrics (1h)
Phase 6.2: Trace visualization (2h)
Phase 7.1: Hinton review agent (1h)
Phase 8.1: Branding finalization (30m)
```

## Success Criteria
- [ ] 80%+ test coverage
- [ ] All E2E tests pass
- [ ] Wikipedia tests reproducible
- [ ] Prometheus dashboard functional
- [ ] Memory trace visualization working
- [ ] Hinton analysis documented
- [ ] NEXUS branding complete
