# T4DM Implementation Plan

**Version**: 1.0
**Date**: 2026-01-24
**Status**: Active Development (forked from WW)

---

## Existing Codebase

T4DM inherits from T4DM:
- 8,905 tests passing
- 81% coverage
- Core memory operations functional

**This plan covers NEW encoding features** transferred from T4DX.

---

## Execution Model

```
PARALLEL TRACKS (can run simultaneously)
├── Track B: Temporal Encoding (NEW)
├── Track M: Memory Consolidation (enhancement)
└── Track L: Learning Systems (enhancement)

LINEAR DEPENDENCIES (must complete in order)
├── Phase 1: Encoding → Phase 2: Integration → Phase 3: Validation
```

---

## Phase 1: Temporal Encoding (Weeks 1-3)

### Track B: Temporal Encoding (NEW)

#### B1: Time2Vec Implementation
| ID | Task | Requirement | Completion Criteria | Parallel? |
|----|------|-------------|---------------------|-----------|
| B1.1 | Time2Vec encoder | Learnable temporal encoding | Forward pass working | ✅ Yes |
| B1.2 | Time2Vec decoder | Timestamp recovery | <1% error on synthetic | ✅ Yes |
| B1.3 | Multi-scale encoding | Hour/day/week scales | 3+ scales functional | ⬇️ After B1.1 |
| B1.4 | Weber-scaled temporal basis | Biological plausibility | Log-compressed time cells | ✅ Yes |
| B1.5 | Gradient verification | Training stability | Gradients flow correctly | ⬇️ After B1.1 |

**Dependencies**: None (start immediately)
**Estimated Time**: 1 week
**Owner**: TBD

#### B2: Memory Consolidation Enhancement
| ID | Task | Requirement | Completion Criteria | Parallel? |
|----|------|-------------|---------------------|-----------|
| B2.1 | Weighted averaging baseline | Simple consolidation | Gist preserved | ✅ Yes |
| B2.2 | Importance weighting | Priority-based compression | High-importance preserved | ⬇️ After B2.1 |
| B2.3 | Rate-distortion loss | Info-theoretic optimal | Loss function implemented | ✅ Yes |
| B2.4 | Interference detection | Prevent catastrophic forgetting | Cosine similarity threshold | ⬇️ After B2.1 |
| B2.5 | Consolidation scheduling | Automated triggers | Time/similarity triggers | ⬇️ After B2.2 |

**Dependencies**: None (start immediately)
**Estimated Time**: 1 week
**Owner**: TBD

#### B3: Associative Decoding
| ID | Task | Requirement | Completion Criteria | Parallel? |
|----|------|-------------|---------------------|-----------|
| B3.1 | Modern Hopfield retrieval | Content-addressable memory | Retrieval accuracy >95% | ✅ Yes |
| B3.2 | Pattern completion | Partial cue reconstruction | Works with 50% masking | ⬇️ After B3.1 |
| B3.3 | Attractor dynamics | Settling to stable state | Convergence in <10 iterations | ⬇️ After B3.1 |
| B3.4 | Capacity testing | Memory limits | Document capacity curve | ⬇️ After B3.1 |

**Dependencies**: B1 complete
**Estimated Time**: 1 week
**Owner**: TBD

---

## Phase 2: Learning Enhancement (Weeks 4-5)

### Track L: Learning Systems

#### L1: STDP Enhancement
| ID | Task | Requirement | Completion Criteria | Parallel? |
|----|------|-------------|---------------------|-----------|
| L1.1 | STDP time scaling | Adapt ms→s timescales | τ configurable | ⬇️ After B1 |
| L1.2 | Co-retrieval tracking | Track access patterns | Access log implemented | ✅ Yes |
| L1.3 | Connection strengthening | Apply STDP rule | Weights update correctly | ⬇️ After L1.1, L1.2 |
| L1.4 | Decay mechanism | Prevent runaway | Bounded weights | ⬇️ After L1.3 |

**Dependencies**: B1 complete
**Estimated Time**: 1 week
**Owner**: TBD

#### L2: Pattern Separation
| ID | Task | Requirement | Completion Criteria | Parallel? |
|----|------|-------------|---------------------|-----------|
| L2.1 | Similarity threshold | Detect interference | Threshold configurable | ✅ Yes |
| L2.2 | Orthogonalization | Push apart similar vectors | Cosine distance increases | ⬇️ After L2.1 |
| L2.3 | Sparse coding option | Alternative separation | L1 regularization works | ⬇️ After L2.1 |
| L2.4 | Separation metrics | Track effectiveness | Metrics logged | ⬇️ After L2.2 |

**Dependencies**: B2 complete
**Estimated Time**: 1 week
**Owner**: TBD

---

## Phase 3: Integration & Validation (Weeks 6-8)

### I1: T4DX Integration
| ID | Task | Requirement | Completion Criteria | Parallel? |
|----|------|-------------|---------------------|-----------|
| I1.1 | Encoding→Index pipeline | T4DM encodes, T4DX indexes | End-to-end working | ⬇️ After Phase 2 |
| I1.2 | Consolidation hooks | T4DX provenance updated | Edges created on consolidation | ⬇️ After I1.1 |
| I1.3 | Bidirectional sync | Consistency maintained | No data loss on sync | ⬇️ After I1.1 |

**Dependencies**: Phase 2 complete, T4DX available
**Estimated Time**: 1 week
**Owner**: TBD

### V1: Validation
| ID | Task | Requirement | Completion Criteria | Parallel? |
|----|------|-------------|---------------------|-----------|
| V1.1 | Encoding accuracy tests | Time2Vec quality | <1% timestamp error | ⬇️ After I1 |
| V1.2 | Consolidation quality | Gist preservation | Human evaluation pass | ⬇️ After I1 |
| V1.3 | Retrieval accuracy | Hopfield performance | >95% accuracy | ⬇️ After I1 |
| V1.4 | Pattern separation tests | Interference reduction | Measured improvement | ⬇️ After I1 |
| V1.5 | Integration tests | Full pipeline | All tests pass | ⬇️ After V1.1-4 |

**Dependencies**: I1 complete
**Estimated Time**: 2 weeks
**Owner**: TBD

---

## Parallelization Matrix

```
Week:  1    2    3    4    5    6    7    8
       ─────────────────────────────────────
B1     ████████                              Time2Vec
B2     ████████                              Consolidation
B3               ████████                    Hopfield
L1                    ████████               STDP
L2                    ████████               Pattern Sep
I1                              ████████     T4DX Integration
V1                                   ████████████ Validation
```

**Maximum Parallelism**: 2 tracks (Weeks 1-2)
**Critical Path**: B1 → B3 → L1 → I1 → V1

---

## Task Summary

| Track | Tasks | Description |
|-------|-------|-------------|
| B | 14 | Temporal Encoding |
| L | 8 | Learning Systems |
| I | 3 | Integration |
| V | 5 | Validation |
| **Total** | **30** | (new tasks only) |

---

## Completion Criteria Summary

### Phase 1 Exit Criteria
- [ ] Time2Vec encoder/decoder working (B1.1-2)
- [ ] Multi-scale encoding functional (B1.3)
- [ ] Consolidation with importance weighting (B2.2)
- [ ] Hopfield retrieval >95% accuracy (B3.1)

### Phase 2 Exit Criteria
- [ ] STDP scales to seconds/minutes (L1.1)
- [ ] Pattern separation reduces interference (L2.2)

### Phase 3 Exit Criteria
- [ ] T4DX integration working (I1.1)
- [ ] All validation tests pass (V1.5)

---

## Resource Requirements

| Resource | Phase 1 | Phase 2 | Phase 3 | Total |
|----------|---------|---------|---------|-------|
| Developer hours | 80 | 60 | 60 | 200 |
| GPU hours | 20 | 10 | 20 | 50 |

---

## Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Time2Vec periodic ambiguity | 30% | Medium | Use linear component for disambiguation |
| Hopfield capacity limits | 20% | Medium | Document limits, use sparse coding |
| T4DX API changes | 20% | Medium | Define interface contract early |
| Consolidation loses detail | 40% | Medium | Importance weighting, validation tests |

---

## File Locations

| Document | Path |
|----------|------|
| This plan | `/mnt/projects/t4dm/IMPLEMENTATION_PLAN.md` |
| Equations map | `/mnt/projects/t4dm/EQUATIONS_MAP.md` |
| Project overview | `/mnt/projects/t4dm/CLAUDE.md` |
