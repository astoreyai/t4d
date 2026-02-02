# Biologically-Inspired Memory Integration Architecture
## CompBio System Integration into T4DM

**Author**: T4DM Team
**Date**: 2025-12-06
**Status**: Architecture Specification
**Version**: 1.0.0

---

## Executive Summary

This document defines the integration architecture for incorporating biologically-inspired neural memory components (from the CompBio agent analysis at `~/mem`) into T4DM's existing tripartite memory system. The goal is to create a unified cognitive architecture that combines:

1. **WW Strengths**: Neo4j symbolic reasoning, MCP integration, tripartite memory, NeuromodulatorOrchestra (5 signals), Thompson sampling, FSRS decay, ACT-R activation
2. **CompBio Additions**: Dendritic computation, sparse coding (k-WTA), attractor dynamics, eligibility traces, Forward-Forward learning (future)

---

## 1. Architecture Overview

### 1.1 Layered Integration Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP Gateway Layer                                   │
│  (17 tools, rate limiting, validation, error handling)                       │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────────────┐
│                        Learning Layer (Enhanced)                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │ NeuromodulatorOrch  │  │ EligibilityTrace │  │ LearnedRetrievalScorer  │ │
│  │ (DA,NE,ACh,5-HT,    │  │ (Temporal credit │  │ (Adaptive weights,      │ │
│  │  GABA) + CompBio    │  │  assignment, TD-λ)│  │  ListMLE ranking)       │ │
│  │  gain functions     │  │                  │  │                         │ │
│  └─────────────────────┘  └──────────────────┘  └─────────────────────────┘ │
│  ┌─────────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │ AdaptiveDecay       │  │ MultiStepCredit  │  │ ExperienceReplayBuffer  │ │
│  │ (Per-type decay     │  │ Graph (DAG-based │  │ (Prioritized replay,    │ │
│  │  optimization)      │  │  credit flow)    │  │  importance sampling)   │ │
│  └─────────────────────┘  └──────────────────┘  └─────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────────────┐
│                      Encoding Layer (NEW - CompBio)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌──────────────────┐  ┌─────────────────────────┐ │
│  │ DendriticProcessor  │  │ SparseEncoder    │  │ AttractorNetwork        │ │
│  │ - Two-compartment   │  │ - k-WTA (2%)     │  │ - Hopfield dynamics     │ │
│  │ - Mismatch signal   │  │ - 10x expansion  │  │ - Pattern completion    │ │
│  │ - Context gating    │  │ - Reconstruction │  │ - Energy minimization   │ │
│  └─────────────────────┘  └──────────────────┘  └─────────────────────────┘ │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────────────────┐
│                        Memory Layer (Enhanced)                               │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ FastEpisodic    │   Episodic      │    Semantic     │    Procedural         │
│ Store (NEW)     │   (FSRS decay)  │ (ACT-R, Hebb)   │    (skill matching)   │
│ - 10K capacity  │ - Episodes      │ - Entities      │    - Procedures       │
│ - Pattern sep.  │ - Context       │ - Relationships │    - Success tracking │
│ - CA3 attractor │ - Outcomes      │ - Spreading     │    - Version control  │
└────────┬────────┴────────┬────────┴────────┬────────┴────────┬──────────────┘
         │                 │                 │                 │
┌────────┴─────────────────┴─────────────────┴─────────────────┴──────────────┐
│                         Storage Layer                                        │
├──────────────────────────┬──────────────────────────────────────────────────┤
│   Qdrant (vectors)       │       Neo4j (graph)                              │
│   - Episode embeddings   │       - Entity relationships                     │
│   - Entity embeddings    │       - Procedure steps                          │
│   - Sparse codes (NEW)   │       - Attractor states (NEW)                   │
│   - Dendritic outputs    │       - Temporal links                           │
└──────────────────────────┴──────────────────────────────────────────────────┘
```

### 1.2 Data Flow with CompBio Processing

```
Input Content
     │
     ▼
┌────────────────────┐
│ 1. BGE-M3 Embed    │ → Base 1024-dim embedding
└────────────────────┘
     │
     ▼
┌────────────────────┐
│ 2. DendriticProc   │ → Context-gated representation + mismatch signal
│    (soma/dendrite) │
└────────────────────┘
     │
     ▼
┌────────────────────┐
│ 3. SparseEncoder   │ → 8192-dim sparse code (2% active)
│    (k-WTA)         │
└────────────────────┘
     │
     ├──────────────────────────┐
     ▼                          ▼
┌────────────────────┐    ┌────────────────────┐
│ 4a. FastEpisodic   │    │ 4b. Attractor      │
│     Store (FES)    │    │     Settle         │
│  - Quick storage   │    │  - Pattern complete│
│  - Salience gate   │    │  - Noise tolerance │
└────────────────────┘    └────────────────────┘
     │                          │
     └──────────┬───────────────┘
                ▼
┌────────────────────────────────┐
│ 5. NeuromodulatorOrchestra     │
│    - DA: reward prediction     │
│    - NE: novelty gain          │
│    - ACh: encoding/retrieval   │
│    - 5-HT: patience, credit    │
│    - GABA: sparsity control    │
└────────────────────────────────┘
                │
                ▼
┌────────────────────────────────┐
│ 6. Store Decision (gated)      │
│    - FES: if high salience     │
│    - WW Memory: if standard    │
│    - Both: if consolidation    │
└────────────────────────────────┘
```

---

## 2. Component Integration Mapping

### 2.1 Neuromodulator Enhancement

| CompBio System | WW System | Integration Strategy |
|----------------|-----------|---------------------|
| `NeuromodulatorSystem` (DA/NE/ACh/5-HT) | `NeuromodulatorOrchestra` (DA/NE/ACh/5-HT/GABA) | **Merge**: Extend Orchestra with CompBio gain functions |
| `compute_learning_rate()` | `effective_learning_rate` property | **Replace**: Use CompBio's multiplicative formula |
| `alpha_ne`, `rho_da` params | Config-based params | **Add**: To WW config system |

**Integration Code Location**: `t4dm/learning/neuromodulators.py`

```python
# Enhanced NeuromodulatorOrchestra
class NeuromodulatorOrchestra:
    def compute_learning_rate(self, base_rate: float, store_type: str = 'standard') -> float:
        """
        Compute modulated learning rate using CompBio formula.

        η_eff = η_base × g_DA × g_NE × g_ACh × g_5HT

        Where:
        - g_DA = max(0.1, 1 + ρ_DA × δ)  # TD error scaling
        - g_NE = 1 + ρ_NE × σ(NE)        # Novelty boost
        - g_ACh = 1 + ρ_ACh × ACh        # Encoding mode boost
        - g_5HT = 0.5 + 0.5 × 5HT        # Patience scaling
        """
        # Use CompBio rho parameters based on store_type
        if store_type == 'fast':  # FES
            rho_ach = self.config.rho_ach_fast  # 3.0
        else:  # Standard WW memory
            rho_ach = self.config.rho_ach_slow  # 0.5

        g_da = max(0.1, 1.0 + self.config.rho_da * self._current_state.dopamine_rpe)
        g_ne = 1.0 + self.config.rho_ne * sigmoid(self._current_state.norepinephrine_gain)
        g_ach = 1.0 + rho_ach * self._get_ach_level()
        g_5ht = 0.5 + 0.5 * self._current_state.serotonin_mood

        return base_rate * g_da * g_ne * g_ach * g_5ht
```

### 2.2 Encoding Layer Components

| CompBio Component | WW Integration Point | File Location |
|-------------------|---------------------|---------------|
| `DendriticNeuron` | NEW: `t4dm/encoding/dendritic.py` | Pre-processing layer |
| `SparseEncoder` | NEW: `t4dm/encoding/sparse.py` | Pattern separation |
| `AttractorNetwork` | NEW: `t4dm/encoding/attractor.py` | Pattern completion |
| `FastEpisodicStore` | NEW: `t4dm/memory/fast_episodic.py` | Parallel to episodic |

### 2.3 Memory Store Mapping

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       Memory Store Hierarchy                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────┐                   ┌───────────────────┐          │
│  │ FastEpisodicStore │  ──(consolidate)──►│ EpisodicMemory    │          │
│  │ (10K capacity)    │                   │ (FSRS-managed)    │          │
│  │ τ = 100x faster   │                   │ τ = standard      │          │
│  └───────────────────┘                   └───────────────────┘          │
│           │                                       │                      │
│           │                                       │                      │
│           └──────────────┬────────────────────────┘                      │
│                          │                                               │
│                          ▼                                               │
│                 ┌───────────────────┐                                    │
│                 │ SemanticMemory    │  (entity extraction)               │
│                 │ (ACT-R activation)│                                    │
│                 └───────────────────┘                                    │
│                          │                                               │
│                          ▼                                               │
│                 ┌───────────────────┐                                    │
│                 │ ProceduralMemory  │  (skill distillation)              │
│                 │ (success-tracked) │                                    │
│                 └───────────────────┘                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration Schema Extension

### 3.1 New Configuration Sections

```python
# t4dm/core/config.py - Extended

@dataclass
class DendriticConfig:
    """Configuration for dendritic processing."""
    input_dim: int = 1024  # BGE-M3 output
    hidden_dim: int = 512
    context_dim: int = 512
    activation: str = 'tanh'
    tau_dendrite: float = 10.0  # ms
    tau_soma: float = 15.0  # ms
    coupling_strength: float = 0.5

@dataclass
class SparseEncoderConfig:
    """Configuration for sparse encoding."""
    input_dim: int = 1024
    hidden_dim: int = 8192  # 8x expansion
    sparsity: float = 0.02  # 2% activation
    use_kwta: bool = True
    lateral_inhibition: float = 0.2

@dataclass
class AttractorConfig:
    """Configuration for attractor networks."""
    dim: int = 8192  # Match sparse encoder
    symmetric: bool = True
    noise_std: float = 0.01
    adaptation_tau: float = 200.0
    settling_steps: int = 10
    step_size: float = 0.1
    capacity_ratio: float = 0.138  # Hopfield capacity

@dataclass
class FastEpisodicConfig:
    """Configuration for Fast Episodic Store."""
    capacity: int = 10000
    learning_rate: float = 0.1  # 100x faster than SSS
    eviction_strategy: str = 'lru_salience'
    salience_weights: dict = field(default_factory=lambda: {
        'da': 0.4, 'ne': 0.3, 'ach': 0.3
    })
    consolidation_threshold: float = 0.7

@dataclass
class EligibilityTraceConfig:
    """Configuration for eligibility traces."""
    decay: float = 0.95  # λ parameter
    tau_trace: float = 20.0  # ms
    a_plus: float = 0.005  # LTP amplitude
    a_minus: float = 0.00525  # LTD amplitude

@dataclass
class BioinspiredConfig:
    """Master config for biologically-inspired components."""
    enabled: bool = True
    dendritic: DendriticConfig = field(default_factory=DendriticConfig)
    sparse_encoder: SparseEncoderConfig = field(default_factory=SparseEncoderConfig)
    attractor: AttractorConfig = field(default_factory=AttractorConfig)
    fast_episodic: FastEpisodicConfig = field(default_factory=FastEpisodicConfig)
    eligibility: EligibilityTraceConfig = field(default_factory=EligibilityTraceConfig)

    # Enhanced neuromodulator gains
    rho_da: float = 2.0
    rho_ne: float = 1.5
    rho_ach_fast: float = 3.0
    rho_ach_slow: float = 0.5
    alpha_ne: float = 0.3
```

### 3.2 Full SystemConfig

```python
@dataclass
class SystemConfig:
    """Complete T4DM configuration."""

    # Existing WW configs
    embedding: EmbeddingConfig
    fsrs: FSRSConfig
    actr: ACTRConfig
    hebbian: HebbianConfig
    neuromodulator: NeuromodulatorConfig
    pattern_separation: PatternSeparationConfig
    memory_gate: MemoryGateConfig
    consolidation: ConsolidationConfig

    # NEW: Bioinspired configs
    bioinspired: BioinspiredConfig = field(default_factory=BioinspiredConfig)

    # Storage
    qdrant: QdrantConfig
    neo4j: Neo4jConfig

    # Runtime
    device: str = 'cuda'
    dtype: str = 'float32'
    seed: int = 42
```

---

## 4. API Integration

### 4.1 New MCP Tools

```python
# t4dm/mcp/memory_gateway.py - New tools

@mcp.tool()
async def encode_with_sparse(
    content: str,
    return_reconstruction: bool = False,
    ctx: Context = None,
) -> dict:
    """
    Encode content using sparse coding.

    Returns sparse code with 2% sparsity for pattern separation.
    Useful for debugging encoding or getting interpretable codes.
    """

@mcp.tool()
async def attractor_complete(
    partial_pattern: list[float],
    num_steps: int = 10,
    ctx: Context = None,
) -> dict:
    """
    Complete a partial pattern using attractor dynamics.

    Given a noisy or partial input, settle to nearest stored pattern.
    Returns completed pattern and energy trajectory.
    """

@mcp.tool()
async def get_fes_statistics(ctx: Context = None) -> dict:
    """
    Get Fast Episodic Store statistics.

    Returns utilization, average salience, consolidation candidates.
    """

@mcp.tool()
async def force_consolidation(
    strategy: str = 'salience',  # salience, age, hybrid
    max_items: int = 100,
    ctx: Context = None,
) -> dict:
    """
    Force consolidation from FES to standard episodic memory.

    Useful for maintenance or before shutdown.
    """
```

### 4.2 Enhanced Retrieval API

```python
@mcp.tool()
async def recall_episodes(
    query: str,
    limit: int = 10,
    use_attractor: bool = True,  # NEW: pattern completion
    use_sparse_match: bool = True,  # NEW: sparse code matching
    time_filter: Optional[TimeRange] = None,
    ctx: Context = None,
) -> list[ScoredResult]:
    """
    Enhanced recall with optional CompBio features.
    """
```

---

## 5. Frontend Integration

### 5.1 New ConfigPanel Sections

Add to `frontend/src/components/ConfigPanel.tsx`:

```typescript
// New configuration sections for bioinspired components

interface BioinspiredConfig {
  enabled: boolean;
  dendritic: {
    hidden_dim: number;
    context_dim: number;
    coupling_strength: number;
    tau_dendrite: number;
    tau_soma: number;
  };
  sparse_encoder: {
    hidden_dim: number;
    sparsity: number;
    use_kwta: boolean;
    lateral_inhibition: number;
  };
  attractor: {
    settling_steps: number;
    noise_std: number;
    adaptation_tau: number;
  };
  fast_episodic: {
    capacity: number;
    learning_rate: number;
    consolidation_threshold: number;
  };
  neuromod_gains: {
    rho_da: number;
    rho_ne: number;
    rho_ach_fast: number;
    rho_ach_slow: number;
  };
}
```

### 5.2 Visualization Components

New components for BioDashboard:

1. **SparseCodeVisualizer**: Shows 8192-dim sparse activations
2. **AttractorEnergyPlot**: Energy landscape during settling
3. **DendriticMismatchChart**: Real-time mismatch signals
4. **FESUtilizationGauge**: Capacity and salience distribution
5. **ConsolidationFlowDiagram**: FES → Episodic → Semantic flow

---

## 6. Testing Strategy

### 6.1 Unit Tests

```python
# tests/encoding/test_dendritic.py
class TestDendriticNeuron:
    def test_mismatch_computation(self):
        """Verify mismatch = soma - dendrite."""

    def test_context_gating(self):
        """Verify context modulates output."""

    def test_gradient_flow(self):
        """Verify gradients flow through both compartments."""

# tests/encoding/test_sparse.py
class TestSparseEncoder:
    def test_kwta_sparsity(self):
        """Verify exactly 2% neurons active."""

    def test_reconstruction_error(self):
        """Verify low reconstruction error."""

    def test_pattern_orthogonality(self):
        """Verify sparse codes are decorrelated."""

# tests/encoding/test_attractor.py
class TestAttractorNetwork:
    def test_energy_decrease(self):
        """Verify energy decreases during settling."""

    def test_pattern_storage(self):
        """Verify stored patterns are fixed points."""

    def test_basin_of_attraction(self):
        """Verify noisy inputs converge to stored patterns."""

# tests/memory/test_fast_episodic.py
class TestFastEpisodicStore:
    def test_capacity_eviction(self):
        """Verify LRU-salience eviction at capacity."""

    def test_salience_scoring(self):
        """Verify neuromodulator-weighted salience."""

    def test_retrieval_accuracy(self):
        """Verify high retrieval accuracy for recent items."""
```

### 6.2 Integration Tests

```python
# tests/integration/test_bioinspired_flow.py
class TestBioinspiredFlow:
    async def test_encode_store_retrieve(self):
        """Full flow: embed → dendritic → sparse → FES → retrieve."""

    async def test_consolidation_flow(self):
        """FES → Episodic → Semantic consolidation."""

    async def test_neuromod_learning_rate(self):
        """Verify neuromodulator-modulated learning rates."""

# tests/integration/test_compbio_ww_compatibility.py
class TestCompBioWWCompatibility:
    async def test_mixed_retrieval(self):
        """Retrieve from both FES and standard episodic."""

    async def test_config_migration(self):
        """Verify old configs work with new system."""
```

### 6.3 Biological Validation Tests

```python
# tests/validation/test_biological_targets.py
class TestBiologicalTargets:
    def test_learning_rate_ratio(self):
        """Verify 100x ratio between FES and standard."""
        assert 50 <= fes_lr / standard_lr <= 200

    def test_sparsity_range(self):
        """Verify 1-5% sparsity (cortical firing rates)."""
        assert 0.01 <= actual_sparsity <= 0.05

    def test_settling_dynamics(self):
        """Verify convergence in ~10 steps."""
        assert settling_steps <= 15

    def test_capacity_scaling(self):
        """Verify ~0.14N patterns storable (Hopfield)."""
```

---

## 7. Implementation Phases

### Phase 0: Foundation (Week 1-2)

**Goal**: Core encoding layer components

**Files to Create**:
- `t4dm/encoding/__init__.py`
- `t4dm/encoding/dendritic.py` - DendriticProcessor class
- `t4dm/encoding/sparse.py` - SparseEncoder class
- `t4dm/encoding/attractor.py` - AttractorNetwork class
- `tests/encoding/` - Unit tests

**Success Criteria**:
- [ ] Dendritic mismatch computable
- [ ] 2% sparsity achieved consistently
- [ ] Attractor settling in <15 steps
- [ ] All unit tests passing

### Phase 1: Memory Integration (Week 3-4)

**Goal**: FastEpisodicStore with consolidation

**Files to Create/Modify**:
- `t4dm/memory/fast_episodic.py` - FastEpisodicStore class
- `t4dm/consolidation/fes_consolidator.py` - FES → Episodic transfer
- `t4dm/core/config.py` - Add BioinspiredConfig
- `tests/memory/test_fast_episodic.py`

**Success Criteria**:
- [ ] FES stores up to 10K items
- [ ] Salience-based eviction working
- [ ] Consolidation transfers high-value memories
- [ ] 100x learning rate differential

### Phase 2: Neuromodulator Enhancement (Week 5-6)

**Goal**: Enhanced learning rate modulation

**Files to Modify**:
- `t4dm/learning/neuromodulators.py` - Add CompBio gain functions
- `t4dm/learning/eligibility.py` - Add trace management
- `t4dm/core/config.py` - Add rho parameters

**Success Criteria**:
- [ ] Multiplicative gain formula working
- [ ] Eligibility traces decaying correctly
- [ ] Learning rate varies with neuromod state

### Phase 3: API & Frontend (Week 7-8)

**Goal**: Complete integration with UI

**Files to Create/Modify**:
- `t4dm/mcp/memory_gateway.py` - New tools
- `t4dm/api/routes/bioinspired.py` - REST endpoints
- `frontend/src/components/BioDashboard.tsx` - Enhanced
- `frontend/src/components/ConfigPanel.tsx` - New sections

**Success Criteria**:
- [ ] MCP tools functional
- [ ] REST API documented
- [ ] ConfigPanel has all new params
- [ ] BioDashboard visualizes new components

---

## 8. Metrics & Monitoring

### 8.1 Key Metrics

```python
@dataclass
class BioinspiredMetrics:
    # Encoding metrics
    sparse_code_sparsity: float  # Target: 0.02
    reconstruction_error: float  # Target: < 0.1
    attractor_convergence_steps: int  # Target: < 15
    dendritic_mismatch_mean: float  # Monitor for anomalies

    # FES metrics
    fes_utilization: float  # 0.0 - 1.0
    fes_avg_salience: float
    consolidation_rate: float  # Items/hour
    eviction_rate: float  # Items/hour

    # Learning metrics
    effective_learning_rate: float
    da_level: float
    ne_level: float
    ach_mode: str

    # Biological validation
    learning_rate_ratio: float  # Target: ~100
    pattern_separation_ratio: float  # Target: ~16x
```

### 8.2 Dashboard Panels

1. **Encoding Health**: Sparsity, reconstruction, convergence
2. **FES Status**: Capacity, salience distribution, candidates
3. **Neuromodulator State**: All 5 systems real-time
4. **Consolidation Flow**: Sankey diagram of memory flow
5. **Learning Metrics**: Effective LR, traces, credit attribution

---

## 9. Backward Compatibility

### 9.1 Feature Flags

```python
# All CompBio features are opt-in via config

bioinspired:
  enabled: true  # Master switch

  # Individual component flags
  use_dendritic: true
  use_sparse_encoder: true
  use_attractor: true
  use_fast_episodic: true
  use_enhanced_neuromod: true
```

### 9.2 Graceful Degradation

If CompBio components fail:
1. Fall back to standard BGE-M3 embedding
2. Use existing episodic memory directly
3. Use existing NeuromodulatorOrchestra
4. Log warning but continue operation

---

## 10. Future Extensions

### 10.1 Forward-Forward Learning

When ready to implement FF learning:
- Replace backprop in dendritic/sparse layers
- Use "goodness" functions per layer
- Local learning rules only

### 10.2 Hopfield-Fenchel-Young Memory

Enhanced attractor dynamics:
- Continuous Hopfield with Fenchel-Young loss
- Differentiable attention-like retrieval
- End-to-end learning through memory

### 10.3 Complementary Learning Systems (CLS)

Full CLS implementation:
- Sharp-wave ripple (SWR) replay scheduling
- Interleaved training during consolidation
- Schema-accelerated learning

---

## References

1. CompBio Architecture: `~/mem/bio_memory_architecture.md`
2. Implementation Starter: `~/mem/implementation_starter.py`
3. Config System: `~/mem/config.py`
4. WW Learning Architecture: `docs/LEARNING_ARCHITECTURE.md`
5. WW Architecture: `docs/architecture.md`
6. Häusser & Mel, 2003 - Dendritic computation
7. McClelland et al., 1995 - CLS theory
8. Hopfield, 1982 - Attractor networks
9. Olshausen & Field, 1996 - Sparse coding
