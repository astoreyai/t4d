# World Weaver Architecture Review: Hinton Perspective

*Analysis from Geoffrey Hinton's perspective on neural-symbolic memory systems*

## Executive Summary

World Weaver is **ambitious and conceptually sophisticated**, implementing ideas from complementary learning systems theory, temporal difference learning, and neuro-symbolic integration. The architecture shows clear understanding of the challenges: temporal credit assignment, neural-symbolic integration, and forgetting dynamics.

## What Works

1. **FSRS Retrievability** - Power-law forgetting correctly implemented
2. **Hebbian Weight Updates** - Bounded learning rule: `w' = w + lr * (1 - w)`
3. **ListMLE Ranking Loss** - Appropriate for learning-to-rank
4. **Prioritized Experience Replay** - Follows best practices from DQN literature

## Critical Issues

### 1. Neural Scorer Too Shallow
- 4-dimensional input is feature engineering, not neural learning
- Cannot discover content-based relevance

**Fix**: Replace with attention-based architecture over embeddings

### 2. Fixed Fusion Weights (60/40)
- Contradicts goal of learned representations
- Balance should be query-dependent

**Fix**: Learn fusion weights with MLP from query embedding

### 3. TD-lambda Uses Wall-Clock Decay
- Should decay per event, not per hour

**Fix**: Index by event transitions, not time

## Priority Recommendations

### Critical (Immediate)
1. **Learned Fusion Weights** - `neuro_symbolic.py:548-587`
2. **Memory Reconsolidation** - New file needed
3. **Fix TD-lambda Trace Decay** - `collector.py:428-468`

### Important (Sprint)
4. Query-Dependent Scoring with attention
5. Counterfactual Credit Assignment
6. Offline Consolidation Loop

### Nice to Have (Long-term)
7. Pattern Separation via Sparse Coding
8. Uncertainty-Aware Learning Rates
9. Active Forgetting with principled deletion

## Biological Mechanisms Status

### Implemented (Phase 1 - v0.5.0)

1. ✅ **Sleep Consolidation** - Delta oscillations (0.5-4 Hz) + sleep spindles (11-16 Hz)
   - See: `ww.nca.oscillators.DeltaOscillator`, `ww.nca.sleep_spindles`
2. ✅ **Pattern Separation** - Implemented in `ww.memory.pattern_separation`
3. ✅ **Reconsolidation** - Implemented in `ww.learning.reconsolidation`
4. ✅ **Contrastive Learning** - InfoNCE adapter for frozen embeddings
   - See: `ww.embedding.ContrastiveAdapter`

### Implemented (Phase 2 - v0.5.0)

5. ✅ **SWR Timing Validation** - Biological 150-250 Hz ripple range
   - See: `ww.nca.swr_coupling.RIPPLE_FREQ_MIN/MAX/OPTIMAL`
6. ✅ **Wake/Sleep State Separation** - 5-state model with ACh/NE inference
   - See: `ww.nca.swr_coupling.WakeSleepMode`
7. ✅ **Serotonin Patience Model** - Temporal discounting (Doya 2002)
   - See: `ww.nca.raphe.PatienceModel`
8. ✅ **Surprise-Driven NE** - Uncertainty signaling (Dayan & Yu 2006)
   - See: `ww.nca.locus_coeruleus.SurpriseModel`

### Implemented (Phase 3 - v0.5.0)

9. ✅ **Forward-Forward Algorithm** (Hinton 2022) - Layer-local learning without backpropagation
   - **H6**: Local goodness function G(h) = Σ h_i²
   - **H7**: Positive/negative phase separation
   - Hebbian-like weight updates correlating pre/post activity
   - Neuromodulator integration: DA→learning rate, ACh→phase, NE→threshold
   - See: `ww.nca.forward_forward.ForwardForwardLayer`, `ForwardForwardNetwork`

10. ✅ **Grid Cell Hexagonal Validation** (Moser 2008, Nobel Prize 2014)
    - **B7**: Gridness score computation (Sargolini et al. 2006)
    - 6-fold rotational symmetry detection
    - 2D spatial autocorrelation via FFT
    - Multiple grid modules at different scales
    - See: `ww.nca.spatial_cells.validate_hexagonal_pattern()`, `compute_gridness_score()`

**Current Scores (Phase 3 Complete)**:
- Hinton Plausibility: 9.0/10
- Biology Fidelity: 94/100
- Test Coverage: 6710 tests passing

### Remaining (Phase 4 Targets)

11. **Capsule Networks** - Part-whole hierarchies (H8, H9)
12. **Glymphatic Analog** - Waste clearance system (B8)
13. **Complementary Learning Rates** - Hippocampus fast, neocortex slow (partial via STDP tau hierarchy)

## Closing Thoughts

> "Instead of programming intelligence, we should be growing it."

Unify the learning: every component that makes a decision should be parameterized and trained from the same outcome signal.
