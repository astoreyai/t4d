# World Weaver NCA: Multi-Layered Telemetry Architecture Plan

**Authors**: Hinton Agent + CompBio Agent (Synthesized)
**Date**: 2026-01-01
**Status**: DESIGN COMPLETE - Ready for Implementation

---

## Executive Summary

Two expert evaluations (Hinton perspective on learning dynamics, CompBio perspective on biological fidelity) converge on a unified **5-layer telemetry architecture** that provides complete observability into:

1. **What the system has learned** (weights, representations)
2. **How it learns** (three-factor signals, eligibility traces)
3. **Why memories are retrieved** (attribution, gate decisions)
4. **Where it might fail** (anomalies, pathologies)
5. **Biological validity** (SWRs, PAC, neural observables)

**Current Score**: 78/100 (Good)
**Projected Score**: 94/100 (Excellent)

---

## Gap Analysis Summary

### Hinton Critique: "Watching the car, not the driver"

| Gap | Impact | Priority |
|-----|--------|----------|
| LearnedMemoryGate weights invisible | Cannot see what's learned | P0 |
| Three-factor signal not visualized | Cannot see credit assignment | P0 |
| Prediction error (dendritic mismatch) hidden | Cannot see where model is wrong | P1 |
| Content projection rank not tracked | Cannot detect representation collapse | P1 |
| Retrieval attribution missing | Cannot explain decisions | P0 |

### CompBio Critique: "Abstract, not experimentally grounded"

| Gap | Impact | Priority |
|-----|--------|----------|
| SWRs implemented but not visualized | Cannot validate consolidation | P0 |
| Theta-gamma PAC missing | Cannot measure WM capacity | P0 |
| DA temporal structure collapsed | Cannot distinguish ramps vs bursts | P1 |
| No multi-scale integration | All metrics at same rate | P1 |
| No validation framework | Cannot compare to real data | P1 |

---

## Unified 5-Layer Telemetry Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WORLD WEAVER TELEMETRY STACK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LAYER 5: HOMEOSTATIC (hourly)                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Adenosine/sleep pressure, receptor adaptation, synaptic scaling,     │  │
│  │ consolidation progress, long-term weight drift, metaplasticity       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  LAYER 4: LEARNING TRAJECTORY (minutes)                                     │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Gate weight evolution (μ trajectory), content projection spectrum,   │  │
│  │ representational similarity matrices, feature importance ranking,    │  │
│  │ uncertainty reduction rate, catastrophic forgetting indicators       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  LAYER 3: BEHAVIORAL (seconds)                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Attractor state, basin stability, energy landscape, memory           │  │
│  │ performance (encoding/retrieval), pattern separation/completion,     │  │
│  │ arousal index, working memory load, interference detection           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  LAYER 2: CIRCUIT (100ms)                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Oscillations (θ/α/β/γ), PAC modulation index, SWR events,            │  │
│  │ DA phasic/tonic/ramp, three-factor signal decomposition,             │  │
│  │ eligibility trace decay, prediction error (mismatch)                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  LAYER 1: SYNAPTIC (1ms)                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Glu/GABA dynamics, LFP proxy, ripple power (150-250Hz),              │  │
│  │ NMDA/AMPA activation, excitotoxicity detection, E/I ratio            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                    CROSS-CUTTING CONCERNS                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │ Anomaly Detector│  │ Causal Inference│  │ Validation Framework        │ │
│  │ - Excitotoxicity│  │ - Granger       │  │ - Buzsáki HC-3 (SWR)        │ │
│  │ - Seizure-like  │  │ - CCM           │  │ - Axmacher MEG (PAC)        │ │
│  │ - Forgetting    │  │ - Transfer      │  │ - Schultz VTA (DA RPE)      │ │
│  │ - Depletion     │  │   entropy       │  │ - KL/DTW/PSD metrics        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 0: Critical (P0) - 2-3 weeks

These are **MUST HAVE** for research validity.

#### P0-1: Learning Weights Visualizer (Hinton Priority)
**File**: `src/t4dm/visualization/learning_weights.py`

```python
@dataclass
class WeightSnapshot:
    timestamp: datetime
    mu: np.ndarray              # Gate weights [247]
    sigma_diag: np.ndarray      # Weight uncertainties [247]
    W_content: np.ndarray       # Content projection [128, 1024]
    n_observations: int
    effective_dimensionality: float  # Entropy of |mu|

class LearningWeightsVisualizer:
    """Tracks evolution of learned parameters."""

    def record_weights(self, gate: LearnedMemoryGate) -> WeightSnapshot
    def compute_weight_pca(self) -> tuple[np.ndarray, float]
    def get_feature_importance(self) -> dict[str, float]
    def get_content_projection_spectrum(self) -> np.ndarray  # SVD
    def detect_representation_collapse(self) -> bool
```

**Metrics**:
- Weight norm trajectory
- Effective feature dimensionality
- Content projection condition number
- Uncertainty reduction rate

#### P0-2: Three-Factor Signal Dashboard (Hinton Priority)
**File**: `src/t4dm/visualization/three_factor_dashboard.py`

```python
@dataclass
class ThreeFactorFrame:
    timestamp: datetime
    memory_id: str
    eligibility: float          # e(t)
    neuromod_gate: float        # ACh mode effect
    dopamine_surprise: float    # RPE
    effective_lr_multiplier: float  # e × neuromod × DA
    outcome: float

class ThreeFactorVisualizer:
    """Visualizes credit assignment decomposition."""

    def plot_factor_decomposition(self, ax) -> None  # Stacked area
    def plot_eligibility_decay(self, ax) -> None
    def plot_lr_modulation_histogram(self, ax) -> None
    def get_learning_hotspots(self) -> list[tuple[str, float]]
```

**Key Insight**: This surfaces the biologically-plausible credit assignment that makes WW unique.

#### P0-3: Retrieval Attribution (Hinton Priority)
**File**: `src/t4dm/visualization/retrieval_attribution.py`

```python
@dataclass
class RetrievalAttribution:
    query_embedding: np.ndarray
    memory_embedding: np.ndarray
    similarity: float
    dimension_contributions: np.ndarray  # [128]
    gate_feature_contributions: dict[str, float]
    nt_state_at_retrieval: np.ndarray

class RetrievalAttributionVisualizer:
    """Explains WHY a memory was retrieved."""

    def attribute_retrieval(self, query, memory, gate, context) -> RetrievalAttribution
    def plot_dimension_importance(self, attribution) -> None
    def plot_gate_feature_importance(self, attribution) -> None
```

#### P0-4: SWR Telemetry (CompBio Priority)
**File**: `src/t4dm/visualization/swr_telemetry.py` (ALREADY WRITTEN)

**Metrics**:
- Ripple frequency (150-250 Hz)
- Duration (50-150 ms)
- Compression factor (10x)
- Reactivated patterns
- Inter-event intervals

#### P0-5: PAC Telemetry (CompBio Priority)
**File**: `src/t4dm/visualization/pac_telemetry.py` (ALREADY WRITTEN)

**Metrics**:
- Modulation index (Tort et al. 2010)
- Preferred phase
- Working memory capacity estimate (gamma/theta ratio)

#### P0-6: Enhanced Anomaly Detection (Both)
**Update**: `src/t4dm/visualization/stability_monitor.py`

**New Methods**:
- `detect_excitotoxicity()`: Glu > 0.85, E/I > 3.0, GABA < 0.2
- `detect_forgetting_risk()`: CA3 capacity > 90%, pattern overlap > 0.6
- `detect_representation_collapse()`: Content projection rank declining

---

### Phase 1: High Priority (P1) - 5-6 weeks

#### P1-1: Multi-Scale Telemetry Hub
**File**: `src/t4dm/visualization/telemetry_hub.py`

```python
class TelemetryHub:
    """Orchestrates multi-scale telemetry."""

    def __init__(self):
        self.synaptic = SynapticTelemetry()      # 1 kHz
        self.circuit = CircuitTelemetry()        # 10 Hz
        self.behavioral = BehavioralTelemetry()  # 0.1 Hz
        self.learning = LearningTelemetry()      # 1 Hz
        self.homeostatic = HomeostaticTelemetry()  # hourly

    def record(self, state, timestamp):
        """Route to appropriate layer based on elapsed time."""

    def get_cross_scale_view(self, start, end) -> dict:
        """Align all scales for a time window."""
```

#### P1-2: Prediction Error Visualizer (Hinton)
**File**: `src/t4dm/visualization/prediction_error.py`

```python
class PredictionErrorVisualizer:
    """Tracks dendritic mismatch signals."""

    def record_mismatch(self, basal, apical, layer_idx) -> float
    def plot_mismatch_by_layer(self) -> None
    def plot_mismatch_correlation_with_learning(self) -> None
    def identify_prediction_failures(self) -> list[dict]
```

#### P1-3: DA Temporal Structure (CompBio)
**File**: `src/t4dm/visualization/da_telemetry.py`

```python
class DATelemetry:
    """Distinguishes DA ramps vs phasic bursts."""

    def classify_da_signal(self, da_trace) -> dict:
        return {
            "signal_type": ["tonic", "phasic", "ramp"],
            "ramp_slope": float,       # Howe et al. 2013
            "phasic_peak_latency": float,
            "phasic_amplitude": float,
            "pause_duration": float,   # Negative RPE
        }
```

#### P1-4: Causal Inference Engine (CompBio)
**File**: `src/t4dm/visualization/causal_inference.py`

```python
class CausalInferenceEngine:
    """Cross-scale causal analysis."""

    def test_granger_causality(self, X, Y, max_lag=10) -> dict
    def test_ccm(self, X, Y) -> float  # Convergent cross-mapping
    def compute_transfer_entropy(self, X, Y) -> float
    def infer_causality_graph(self) -> nx.DiGraph
```

#### P1-5: Validation Framework (CompBio)
**Directory**: `src/t4dm/validation/`

```python
class NeuroscienceBenchmarks:
    """Compare WW telemetry to real neuroscience data."""

    def load_dataset(self, name: str) -> Dataset
        # "buzsaki_hc3_lfp", "axmacher_meg", "schultz_vta"

    def compare_swr(self, ww_data) -> dict  # KL, DTW
    def compare_pac(self, ww_data) -> dict
    def compare_da_rpe(self, ww_data) -> dict
```

---

### Phase 2: Medium Priority (P2) - Future

1. **Representation Health Monitor**: Embedding diversity, calibration curves
2. **Forward-Forward Metrics**: Goodness function visualization
3. **Sleep Replay Visualization**: Consolidation sequence tracking
4. **Spatial Heterogeneity**: Region-specific NT dynamics
5. **Traveling Wave Detection**: Cortical wave propagation

---

## Data Flow Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         MEMORY OPERATIONS                                 │
│  (create_episode, recall, learn, consolidate)                             │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     TELEMETRY BUS (Hook-based)                           │
│  - Captures every gate.predict() with full features                      │
│  - Captures every gate.update() with utility + neuromod                  │
│  - Captures every retrieval with query/result embeddings                 │
│  - Captures every three_factor.compute() signal                          │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────────────┐
         │                   │                           │
         ▼                   ▼                           ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐
│ EXISTING VIZ    │  │ NEW LEARNING VIZ│  │ NEW BIO VIZ                 │
│                 │  │                 │  │                             │
│ - Energy        │  │ - Weights       │  │ - SWR Telemetry             │
│ - Coupling      │  │ - Three-Factor  │  │ - PAC Telemetry             │
│ - NT State      │  │ - Attribution   │  │ - DA Temporal               │
│ - Stability     │  │ - Pred Error    │  │ - Multi-Scale Hub           │
└─────────────────┘  └─────────────────┘  └─────────────────────────────┘
         │                   │                           │
         └───────────────────┴───────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     UNIFIED DASHBOARD                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Synaptic │  │ Circuit  │  │Behavioral│  │ Learning │  │Homeostatic   │
│  │  1 kHz   │  │  10 Hz   │  │  0.1 Hz  │  │  1 Hz    │  │  hourly  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │               ANOMALY ALERTS (real-time)                         │    │
│  │  [EXCITOTOXICITY] [SEIZURE-LIKE] [FORGETTING] [COLLAPSE]        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
src/t4dm/visualization/
├── __init__.py                    # Updated exports
│
├── # EXISTING (Phases 1-3)
├── energy_landscape.py            # Hopfield energy, basins
├── coupling_dynamics.py           # 6x6 coupling, eligibility
├── nt_state_dashboard.py          # NT channels, homeostasis
├── stability_monitor.py           # Eigenvalues, Lyapunov, bifurcation
│
├── # NEW P0 (Learning Dynamics)
├── learning_weights.py            # Gate μ/Σ evolution, W_content
├── three_factor_dashboard.py      # e × neuromod × DA decomposition
├── retrieval_attribution.py       # Explain WHY memory retrieved
│
├── # NEW P0 (Biological Observables)
├── swr_telemetry.py               # Sharp-wave ripples
├── pac_telemetry.py               # Theta-gamma PAC
│
├── # NEW P1 (Infrastructure)
├── telemetry_hub.py               # Multi-scale orchestration
├── prediction_error.py            # Dendritic mismatch
├── da_telemetry.py                # DA ramps vs phasic
├── causal_inference.py            # Granger, CCM, TE
│
└── # NEW P1 (Validation)
    validation/
    ├── __init__.py
    ├── benchmarks.py              # Load neuroscience datasets
    ├── comparators.py             # KL, DTW, PSD metrics
    └── datasets/                  # Cached benchmark data
```

---

## Metrics Summary

### Layer 1: Synaptic (1 kHz)
| Metric | Range | Alert Threshold |
|--------|-------|-----------------|
| Glutamate | 0-1 | > 0.85 (excitotoxicity) |
| GABA | 0-1 | < 0.2 (depletion) |
| E/I ratio | 0-∞ | > 3.0 (seizure-like) |
| Ripple power | 0-∞ | > 2σ (SWR detected) |

### Layer 2: Circuit (10 Hz)
| Metric | Range | Target |
|--------|-------|--------|
| PAC modulation index | 0-1 | 0.3-0.7 (strong coupling) |
| Theta power | 0-∞ | Stable during encoding |
| DA phasic | 0-1 | Burst on surprise |
| Eligibility trace | 0-1 | Decay τ ~ 1-10s |

### Layer 3: Behavioral (0.1 Hz)
| Metric | Range | Target |
|--------|-------|--------|
| Basin stability margin | -∞ to +∞ | > 0.1 (stable) |
| Encoding success rate | 0-1 | > 0.7 |
| Pattern separation | 0-1 | > 0.5 (DG working) |
| WM capacity | 3-9 | 4-7 items |

### Layer 4: Learning (minutes)
| Metric | Range | Alert Threshold |
|--------|-------|-----------------|
| Weight norm | 0-∞ | Stable growth |
| Feature dimensionality | 0-247 | < 10 (collapse) |
| Content projection rank | 0-128 | < 20 (collapse) |
| Uncertainty trend | decreasing/stable | increasing (problem) |

### Layer 5: Homeostatic (hourly)
| Metric | Range | Alert Threshold |
|--------|-------|-----------------|
| Adenosine | 0-1 | > 0.8 (sleep pressure) |
| Synaptic scaling | 0.5-2.0 | Outside range |
| Consolidation score | 0-1 | < 0.3 (not consolidating) |

---

## Validation Targets

### SWR Validation (vs Buzsáki HC-3)
- [ ] Frequency KL divergence < 0.3
- [ ] Duration distribution match (50-150ms)
- [ ] Event rate correlation > 0.6

### PAC Validation (vs Axmacher MEG)
- [ ] MI in 0.3-0.7 range
- [ ] Preferred phase stable (~π)
- [ ] WM capacity 4-7

### DA Validation (vs Schultz VTA)
- [ ] RPE timing < 200ms post-reward
- [ ] Burst firing 15-30 Hz
- [ ] Ramp slope matches motivation

---

## Implementation Timeline

| Week | Deliverable | Owner | Score Impact |
|------|-------------|-------|--------------|
| 1 | Learning Weights Visualizer | P0 | +4 |
| 1 | Three-Factor Dashboard | P0 | +4 |
| 2 | Retrieval Attribution | P0 | +3 |
| 2 | SWR Telemetry (test) | P0 | +3 |
| 3 | PAC Telemetry (test) | P0 | +3 |
| 3 | Enhanced Anomaly Detection | P0 | +2 |
| 4-5 | Multi-Scale Hub | P1 | +3 |
| 5-6 | Prediction Error | P1 | +2 |
| 6-7 | DA Temporal | P1 | +2 |
| 7-8 | Causal Inference | P1 | +2 |
| 8-10 | Validation Framework | P1 | +4 |

**Total Score Improvement**: 78 → 94 (+16 points)

---

## Approval

**Hinton Agent Assessment**:
> "The Learning Telemetry layer will transform this from a system that works to a system researchers can understand. The three-factor signal decomposition is critical for validating biologically-plausible credit assignment."

**CompBio Agent Assessment**:
> "Adding SWR and PAC telemetry with validation against Buzsáki/Axmacher data will make this publication-quality. The multi-scale architecture properly respects the temporal hierarchy of neural computation."

**Combined Verdict**: ✅ **APPROVED** for implementation

---

## Next Steps

1. **Immediate**: Create P0 files (learning_weights.py, three_factor_dashboard.py, retrieval_attribution.py)
2. **This Week**: Write tests for SWR/PAC telemetry (already have implementations)
3. **Next Sprint**: Implement Multi-Scale Hub
4. **Validation Sprint**: Build benchmark comparison framework

---

END OF ARCHITECTURE PLAN
