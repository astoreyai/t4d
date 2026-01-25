# T4DM Equations Map

**Purpose**: Map each implementation task to required equations, their status, and verification criteria.

---

## Equation Status Legend

| Status | Meaning |
|--------|---------|
| ✅ EXISTS | Proven, can use directly |
| 🔧 ADAPT | Exists but needs modification |
| 🔬 INVENT | Must be developed and proven |

---

## Track B: Temporal Encoding

### B1.1 Time2Vec Encoder

**Equation**:
```
t2v(τ)[i] = {
    ω_i · τ + φ_i,           if i = 0 (linear)
    sin(ω_i · τ + φ_i),      if 1 ≤ i ≤ k (periodic)
}
```
**Status**: ✅ EXISTS
**Verification**:
- Gradients flow correctly
- Frequencies learnable
- Covers target time range

**Extension** (2025 improvement):
```
t2v_extended(τ)[i] = sin(ω_i · τ + φ_i) + cos(ω_i · τ + φ_i)
```
**Status**: ✅ EXISTS (TT2VFin, 2025)

---

### B1.2 Time2Vec Decoder

**Equation**: Timestamp recovery
```
τ_recovered = (arcsin(t2v(τ)[i]) - φ_i) / ω_i
```
**Status**: 🔧 ADAPT
**Problem**: Periodic ambiguity (arcsin has multiple solutions)
**Solution**: Constrain to valid time window, use linear component

**Verification**:
- < 1% error on synthetic data
- Handles edge cases (t=0, t=max)

---

### B1.3 Multi-Scale Encoding

**Equation**: Hierarchical temporal encoding
```
t_multi(τ) = concat([
    t2v(τ, τ_hour),      # hourly patterns
    t2v(τ, τ_day),       # daily patterns
    t2v(τ, τ_week),      # weekly patterns
    t2v(τ, τ_year)       # yearly patterns
])
```
**Status**: ✅ EXISTS (MultiTEmb, 2025)
**Verification**: Each scale captures appropriate patterns

---

### B1.4 Weber-Scaled Temporal Basis

**Equation**: Log-compressed time cells
```
temporal_basis(t) = log(1 + t/τ_base)
```
**Status**: ✅ EXISTS (Temporal Context Model)
**Verification**: Matches Weber's law for time perception

---

## Track B2: Memory Consolidation

### B2.1 Weighted Averaging Consolidation

**Equation**:
```
v_consolidated = Σᵢ wᵢ · vᵢ / Σᵢ wᵢ
```
**Status**: ✅ EXISTS
**Optimality**: Optimal for L² loss under Gaussian noise
**Verification**: Gist preserved, details lost (as expected)

---

### B2.2 Importance Weighting

**Equation**:
```
wᵢ = importance(mᵢ) × recency(tᵢ) × relevance(mᵢ, context)
importance(m) = access_count(m) × avg_attention(m)
recency(t) = exp(-(t_now - t) / τ_decay)
```
**Status**: 🔧 ADAPT (combine existing components)
**Verification**: High-importance memories have higher weights

---

### B2.3 Rate-Distortion Consolidation

**Equation**:
```
L = ||v - decode(encode(v))||² + λ · H(encode(v))
```
Where H is entropy (compression rate)

**Status**: 🔬 INVENT (novel application)
**Verification**:
- Compression ratio measurable
- Reconstruction error bounded
- Pareto frontier documented

---

### B2.4 Interference Detection

**Equation**:
```
interference(m_new, M_existing) = max_{m ∈ M} cosine_sim(m_new, m)
if interference > θ: trigger_pattern_separation()
```
**Status**: ✅ EXISTS
**Verification**: High-similarity memories flagged

---

## Track B3: Associative Decoding

### B3.1 Modern Hopfield Retrieval

**Equation**:
```
E = -log(Σᵢ exp(xᵢᵀ · query))
update = softmax(β · Xᵀ · query) · X
```
**Status**: ✅ EXISTS (Ramsauer 2020)
**Capacity**: O(d^(n-1)) for n-th order
**Verification**: Retrieval accuracy >95%

---

### B3.2 Pattern Completion

**Equation**:
```
v_complete = hopfield_update(v_partial, M, iterations=k)
```
**Status**: ✅ EXISTS
**Verification**: Works with 50% masking

---

## Track L: Learning Systems

### L1.1 STDP Learning Rule

**Equation**:
```
Δw = {
    A₊ · exp(-Δt/τ₊),  if Δt > 0 (pre before post → LTP)
    -A₋ · exp(Δt/τ₋),  if Δt < 0 (post before pre → LTD)
}
```
**Status**: ✅ EXISTS
**Adaptation**: Scale τ from milliseconds to seconds/minutes
**Verification**: Co-retrieved memories strengthen connections

---

### L2.2 Pattern Separation

**Equation**:
```
v_separated = v + noise · (v - nearest_neighbor(v, M))
```
Or via sparse coding:
```
minimize ||v - D · s||² + λ ||s||₁
```
**Status**: ✅ EXISTS
**Verification**: Separated vectors are more orthogonal

---

## Open Research Questions

### Event Segmentation

**Problem**: Where does one memory end and another begin?
**Current**: Heuristic (fixed time windows, change detection)
**Needed**:
```
boundary_score(t) = f(Δembedding(t), Δattention(t), Δtopic(t))
segment if boundary_score(t) > θ
```
**Status**: 🔬 INVENT
**Research**: Theta-gated windows (~125ms in biology)

---

## Verification Checklist

### Mathematical Proofs Required

| Equation | Proof Type | Status |
|----------|------------|--------|
| Time2Vec decoder | Error bounds | 🔧 Needs bounds |
| Rate-distortion consolidation | Optimality | 🔬 To prove |
| Hopfield capacity | Capacity formula | ✅ Proven |

### Empirical Verification Required

| Equation | Test Type | Dataset |
|----------|-----------|---------|
| Time2Vec | Reconstruction error | Synthetic timestamps |
| Consolidation | Gist preservation | Conversation memory |
| Pattern completion | Reconstruction | Masked retrieval |
| STDP | Connection strength | Co-retrieval logs |

---

## Summary: Equations by Status

| Status | Count | Examples |
|--------|-------|----------|
| ✅ EXISTS | 10 | Time2Vec, Hopfield, STDP, Weber basis |
| 🔧 ADAPT | 2 | Time2Vec decoder, Importance weighting |
| 🔬 INVENT | 2 | Rate-distortion, Event segmentation |

**Bottom Line**: 10 equations exist, 2 need adaptation, 2 need invention.
