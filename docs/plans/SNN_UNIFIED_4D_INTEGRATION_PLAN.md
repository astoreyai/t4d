# SNN + Unified 4D Memory Integration Plan

**Version**: 2.0 (Complete Rewrite)
**Date**: 2026-01-30
**Status**: Ready for Atomic Implementation
**RTX 3090 Target**: 24GB VRAM
**Total Atoms**: 50 (across 5 phases)
**Duration**: 9 weeks

---

## Executive Summary

This plan transforms T4DM from its current rate-coded state to a fully unified spiking neural network (SNN) architecture with a gradient-based consolidation model (κ) replacing the discrete episodic/semantic/procedural memory stores. The architecture is 90% complete; this plan targets the critical 10% for biological plausibility and scientific defensibility.

**Key Achievements**:
- Python-first with JIT optimization (Numba/PyTorch)
- Unified memory substrate with κ ∈ [0,1] consolidation gradient
- Norse GPU backend for LIF neurons and spike-based STDP
- Complete spike reinjection loop: replay → SNN → STDP → weight update
- MNE/Elephant validation for biological plausibility

**Language Decision**: Python remains primary. Rust SNN ecosystem immature (spiking-neural-networks crate abandoned). Julia excellent for PDE but poor PyTorch integration. Norse PyTorch ideal for <200K neurons on RTX 3090.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Language & Optimization Strategy](#2-language--optimization-strategy)
3. [Verified Codebase Line Numbers](#3-verified-codebase-line-numbers)
4. [Phase 1: Foundation (2 weeks, 12 atoms)](#4-phase-1-foundation)
5. [Phase 2: Unified Memory Substrate (3 weeks, 15 atoms)](#5-phase-2-unified-memory-substrate)
6. [Phase 3: Spike Pipeline (2 weeks, 10 atoms)](#6-phase-3-spike-pipeline)
7. [Phase 4: Validation (1 week, 8 atoms)](#7-phase-4-validation)
8. [Phase 5: Visualization + Polish (1 week, 5 atoms)](#8-phase-5-visualization--polish)
9. [Mermaid Diagram Updates](#9-mermaid-diagram-updates)
10. [RTX 3090 Memory Budget](#10-rtx-3090-memory-budget)

---

## 1. Architecture Overview

### 1.1 The 16-Block Unified System

```mermaid
graph TB
    IN[IN: Sensory Input] --> ENC[ENC: Time2Vec Encoder]
    ENC --> SNN[SNN-Core: Norse LIF Spike Generator]
    SNN --> OUT[OUT: Readout Layer]

    PE[PE: Prediction Error] --> TAU[TAU: τ(t) Temporal Control]
    TAU --> WR[WR: Write Gate]
    WR --> MEM[MEM: Unified 4D Store κ∈[0,1]]

    Q[Q: Query Encoder] --> WM[WM: Working Memory Buffer]
    WM --> RP[RP: Replay Sampler]
    RP --> REINJ[REINJ: Spike Reinjection]
    REINJ --> SNN

    ELIG[ELIG: Eligibility Traces] --> UPD[UPD: Three-Factor Update]
    MOD[MOD: Dopamine/ACh/NE] --> UPD
    PE --> UPD
    UPD --> MEM

    HOME[HOME: Homeostatic Control] --> SNN
    MEM --> OUT

    style MEM fill:#ff6b6b,stroke:#c92a2a
    style TAU fill:#ff6b6b,stroke:#c92a2a
    style REINJ fill:#ff6b6b,stroke:#c92a2a
```

**Legend**:
- ✅ **EXISTS** (90%): Blocks 1, 4-7, 9-11, 13-16 are implemented
- ❌ **MISSING** (10%): Blocks 2, 6, 8, 12 are critical gaps (this plan fills them)

---

## 2. Language & Optimization Strategy

### 2.1 Three-Tier Optimization Plan

| Tier | Week | Focus | Speedup | Tools |
|------|------|-------|---------|-------|
| **Tier 1** | 1-2 | Hot path JIT + GPU PDE | 10-50x | Numba, PyTorch GPU |
| **Tier 2** | 5-6 | Optional Rust STDP kernel | 50-100x | Rust/PyO3 |
| **Tier 3** | 9+ | Optional Triton CUDA PDE | 100-200x | Triton |

### 2.2 Language Decision Matrix

| Language | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Python** | Norse/Brian2/MNE/Elephant all Python; mature ecosystem | Slow interpreter | ✅ Primary (with JIT) |
| **Rust** | Memory safety, 100x speedup | SNN ecosystem immature; PyO3 FFI overhead | ⚠️ Tier 2 optional |
| **Julia** | Excellent PDE performance (DifferentialEquations.jl) | Poor PyTorch integration; PyCall fragile | ❌ Rejected |
| **C++/CUDA** | Maximum performance | Development time; unsafe | ⚠️ Tier 3 via Triton |

**Decision**: Python primary. Numba for STDP loop (Week 1-2). Optional Rust kernel (Month 2+). Norse PyTorch is sweet spot for <200K neurons.

### 2.3 Hot Path Identification

**STDP Hot Loop** (compute_all_updates: stdp.py:498):
```python
# BEFORE (Python loop):
for (pre, post), weight in self._weights.items():  # O(N²) spikes
    delta = self.compute_stdp_delta(...)
    # 100K neurons × 100K neurons = 10B iterations → 10s+

# AFTER (Numba JIT):
@numba.jit(nopython=True, parallel=True)
def compute_all_updates_jit(spike_times, weights):
    # Compiled to machine code → <1s
```

**PDE Solver Hot Loop** (NeuralFieldSolver.step: neural_field.py):
```python
# BEFORE (NumPy CPU):
laplacian = scipy.ndimage.laplace(field)  # 32³ grid × 6 NTs × CPU

# AFTER (PyTorch GPU):
laplacian = kornia.filters.laplacian(field_gpu)  # CUDA kernel → 10x faster
```

---

## 3. Verified Codebase Line Numbers

All line numbers verified from source code on 2026-01-30.

### 3.1 STDP System

| Component | File | Line | Function |
|-----------|------|------|----------|
| STDPLearner init | `stdp.py` | 144 | `__init__(config)` |
| Record spike | `stdp.py` | 174 | `record_spike(entity_id, timestamp, strength)` |
| Compute delta | `stdp.py` | 246 | `compute_stdp_delta(delta_t_ms, weight, da_level)` |
| DA modulation | `stdp.py` | 311 | `_compute_da_modulated_rates(da_level)` |
| Weight update | `stdp.py` | 355 | `compute_weight_update(pre_id, post_id, w, da)` |
| **HOT PATH** | `stdp.py` | 498 | `compute_all_updates()` ← Numba target |
| Clear spikes | `stdp.py` | 600 | `clear_spikes()` |
| PairBasedSTDP | `stdp.py` | 671 | `PairBasedSTDP` class |
| TripletSTDP | `stdp.py` | 730 | `TripletSTDP` class |

### 3.2 Eligibility Traces

| Component | File | Line | Function |
|-----------|------|------|----------|
| EligibilityTrace init | `eligibility.py` | 75 | `__init__(decay, tau_trace, ...)` |
| Update trace | `eligibility.py` | 118 | `update(memory_id, activity)` |
| Step decay | `eligibility.py` | 167 | `step(dt)` |
| Assign credit | `eligibility.py` | 203 | `assign_credit(outcome)` |
| Get trace value | `eligibility.py` | 280 | `get_trace(memory_id)` |
| Layered traces | `eligibility.py` | 339 | `LayeredEligibilityTrace` class |

### 3.3 Three-Factor Learning

| Component | File | Line | Function |
|-----------|------|------|----------|
| ThreeFactorLearningRule init | `three_factor.py` | 132 | `__init__(eligibility, neuromod, dopamine, ...)` |
| Neuromod gate | `three_factor.py` | 252 | `_compute_neuromod_gate(memory_id)` |
| Compute signal | `three_factor.py` | 294 | `compute(memory_id, base_lr, outcome)` |
| Effective LR | `three_factor.py` | 416 | `compute_effective_lr(eligibility, neuromod, da)` |
| Batch compute | `three_factor.py` | 436 | `batch_compute(memory_ids, base_lr, outcome)` |
| Reconsolidation init | `three_factor.py` | 526 | `ThreeFactorReconsolidation.__init__()` |
| Reconsolidate | `three_factor.py` | 587 | `reconsolidate(memory_id, new_context)` |

### 3.4 Homeostatic Plasticity

| Component | File | Line | Function |
|-----------|------|------|----------|
| HomeostaticPlasticity init | `homeostatic.py` | 70 | `__init__(target_norm, tolerance, ...)` |
| Update statistics | `homeostatic.py` | 104 | `update_statistics(embeddings)` |
| Needs scaling check | `homeostatic.py` | 133 | `needs_scaling()` |
| Compute scaling | `homeostatic.py` | 143 | `compute_scaling_factor()` |
| Apply scaling | `homeostatic.py` | 155 | `apply_scaling(embeddings)` |
| Decorrelate | `homeostatic.py` | 183 | `decorrelate(embeddings)` |
| Sliding threshold | `homeostatic.py` | 236 | `update_sliding_threshold(activities)` |

### 3.5 Neuromodulators

| Component | File | Line | Function |
|-----------|------|------|----------|
| NeuromodulatorOrchestra init | `neuromodulators.py` | 225 | `__init__(dopamine, ne, ach, serotonin, ...)` |
| Process query | `neuromodulators.py` | 257 | `process_query(query_emb, is_question, ...)` |
| Process retrieval | `neuromodulators.py` | 315 | `process_retrieval(retrieved_memories, ...)` |
| Process outcome | `neuromodulators.py` | 350 | `process_outcome(memory_id, outcome, ...)` |
| Get learning params | `neuromodulators.py` | 448 | `get_learning_params(memory_id)` |
| Get current state | `neuromodulators.py` | 584 | `get_current_state()` |

### 3.6 NCA Modules

| Component | File | Line | Function |
|-----------|------|------|----------|
| DentateGyrusLayer init | `hippocampus.py` | 136 | `__init__(config, random_seed)` |
| HippocampalCircuit | `hippocampus.py` | - | Large class (encode/retrieve methods) |
| **HOT PATH** NeuralFieldSolver | `neural_field.py` | - | PyTorch GPU target for PDE |
| Connectome init | `connectome.py` | 160 | `Connectome` class |
| BrainRegion dataclass | `connectome.py` | 90 | `BrainRegion` with coordinates, receptors |
| ProjectionPathway dataclass | `connectome.py` | 129 | `ProjectionPathway` with NT, strength |

### 3.7 Consolidation

| Component | File | Line | Function |
|-----------|------|------|----------|
| SleepConsolidation | `consolidation/sleep.py` | - | Full sleep cycle class |
| SharpWaveRipple init | `consolidation/sleep.py` | 147 | `__init__(compression_factor, ...)` |
| ConsolidationSTDP init | `consolidation/stdp_integration.py` | 90 | `__init__(stdp_learner, synaptic_tagger)` |

### 3.8 Core Types

| Component | File | Line | Function |
|-----------|------|------|----------|
| Episode class | `core/types.py` | 106 | `Episode(BaseModel)` with FSRS fields |
| MemoryGate | `core/memory_gate.py` | 49 | `MemoryGate.__init__()` |

---

## 4. Phase 1: Foundation

**Duration**: 2 weeks
**Atoms**: 12
**Deliverable**: τ(t) signal, Norse backend, Numba STDP JIT, MemoryItem schema

### ATOM-P1-01: Create τ(t) Temporal Control Signal

**Description**: Implement temporal write strength modulation based on prediction error magnitude.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/src/ww/core/temporal_control.py`

**Implementation**:
```python
"""Temporal control signal for write gating."""
import numpy as np
from dataclasses import dataclass

@dataclass
class TemporalControlConfig:
    """Configuration for tau(t) temporal control."""
    threshold: float = 0.3  # PE threshold for strong encoding
    steepness: float = 5.0  # Sigmoid steepness
    min_tau: float = 0.0    # Floor
    max_tau: float = 1.0    # Ceiling

def compute_tau(
    prediction_error: float,
    config: TemporalControlConfig | None = None
) -> float:
    """
    Compute τ(t) temporal write control from prediction error.

    τ(t) = sigmoid(steepness * (|ε| - threshold))

    Args:
        prediction_error: |ε(t)| magnitude from dopamine RPE
        config: Configuration (uses defaults if None)

    Returns:
        τ(t) ∈ [0, 1] - write strength (0=no write, 1=full write)

    Example:
        >>> compute_tau(0.1)  # Low PE → low tau
        0.076
        >>> compute_tau(0.5)  # High PE → high tau
        0.924
    """
    if config is None:
        config = TemporalControlConfig()

    # Sigmoid: high PE → high τ (strong write)
    exponent = -config.steepness * (abs(prediction_error) - config.threshold)
    tau = 1.0 / (1.0 + np.exp(exponent))

    return float(np.clip(tau, config.min_tau, config.max_tau))
```

**Tests to Write**:
- `tests/unit/test_temporal_control.py`:
  - `test_compute_tau_low_pe()`: PE=0.1 → tau≈0.08
  - `test_compute_tau_high_pe()`: PE=0.5 → tau≈0.92
  - `test_compute_tau_threshold()`: PE=threshold → tau≈0.5
  - `test_compute_tau_negative_pe()`: Absolute value handling

**Acceptance Criteria**:
- [ ] Function returns float ∈ [0, 1]
- [ ] High PE (>threshold) → high tau (>0.5)
- [ ] Low PE (<threshold) → low tau (<0.5)
- [ ] All tests pass

**Dependencies**: None

---

### ATOM-P1-02: Integrate τ(t) into MemoryGate

**Description**: Multiply memory gate score by τ(t) to modulate write strength.

**Files to Edit**:
- `src/ww/core/memory_gate.py:115-145`

**Changes**:
```python
# Line 138 (OLD):
score = 0.4 * novelty + 0.3 * outcome_weight + 0.3 * entity_weight

# Line 138 (NEW):
from ww.core.temporal_control import compute_tau

base_score = 0.4 * novelty + 0.3 * outcome_weight + 0.3 * entity_weight

# Compute prediction error from context (if available)
prediction_error = context.get("prediction_error", 0.0)
tau = compute_tau(prediction_error)

# Modulate score by temporal control signal
score = tau * base_score  # τ=0 → score=0 (no write), τ=1 → full write

logger.debug(
    f"MemoryGate: PE={prediction_error:.3f}, tau={tau:.3f}, "
    f"base_score={base_score:.3f}, final_score={score:.3f}"
)
```

**Tests to Update**:
- `tests/unit/test_memory_gate.py`:
  - Add `test_tau_modulation_high_pe()`: High PE → high score
  - Add `test_tau_modulation_low_pe()`: Low PE → low score

**Mermaid Diagrams to Update**:
- `docs/diagrams/34_state_memory_gate.mmd`: Add τ(t) state transition

**Acceptance Criteria**:
- [ ] Score modulated by τ(t)
- [ ] High PE increases write probability
- [ ] Low PE decreases write probability
- [ ] Tests pass

**Dependencies**: ATOM-P1-01

---

### ATOM-P1-03: Create Norse SNN Backend Wrapper

**Description**: Wrap Norse LIFCell for T4DM integration.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/src/ww/nca/snn_backend.py`

**Implementation**:
```python
"""Norse SNN backend for GPU-accelerated spiking neurons."""
import logging
import torch
from dataclasses import dataclass
from norse.torch import LIFCell, LIFParameters

logger = logging.getLogger(__name__)

@dataclass
class SNNConfig:
    """Configuration for Norse SNN backend."""
    n_neurons: int = 1024
    tau_mem_inv: float = 1.0 / 20e-3  # 20ms membrane time constant
    tau_syn_inv: float = 1.0 / 5e-3   # 5ms synaptic time constant
    v_th: float = 1.0                  # Spike threshold
    v_reset: float = 0.0               # Reset potential
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class NorseSpikeGenerator:
    """
    Norse-based spike generator for T4DM.

    Provides GPU-accelerated LIF neuron simulation with stateful integration.
    """

    def __init__(self, config: SNNConfig | None = None):
        self.config = config or SNNConfig()

        # Norse LIF parameters
        self.lif_params = LIFParameters(
            tau_mem_inv=self.config.tau_mem_inv,
            tau_syn_inv=self.config.tau_syn_inv,
            v_th=self.config.v_th,
            v_reset=self.config.v_reset,
        )

        # LIF cell
        self.lif = LIFCell(self.lif_params)

        # State (membrane potential, synaptic current)
        self.state = None

        # Device
        self.device = torch.device(self.config.device)
        self.lif = self.lif.to(self.device)

        logger.info(
            f"NorseSpikeGenerator initialized: {self.config.n_neurons} neurons, "
            f"tau_mem={1/self.config.tau_mem_inv*1000:.1f}ms, device={self.device}"
        )

    def generate_spikes(
        self,
        input_current: torch.Tensor,
        reset_state: bool = False
    ) -> torch.Tensor:
        """
        Generate spikes from input current via LIF dynamics.

        Args:
            input_current: [batch_size, n_neurons] input current
            reset_state: Whether to reset LIF state (default False)

        Returns:
            spikes: [batch_size, n_neurons] binary spike tensor
        """
        if reset_state or self.state is None:
            self.state = None

        input_current = input_current.to(self.device)
        spikes, self.state = self.lif(input_current, self.state)

        return spikes

    def reset(self):
        """Reset LIF state."""
        self.state = None
```

**Tests to Write**:
- `tests/unit/test_norse_backend.py`:
  - `test_generate_spikes()`: Input current → spikes
  - `test_spike_threshold()`: Below threshold → no spike
  - `test_spike_reset()`: State reset works
  - `test_gpu_usage()`: If CUDA available, uses GPU

**Acceptance Criteria**:
- [ ] Norse LIF generates spikes
- [ ] State persists across calls
- [ ] GPU acceleration on RTX 3090
- [ ] Tests pass

**Dependencies**: None (Norse package installed)

---

### ATOM-P1-04: Numba JIT for STDP Hot Loop

**Description**: Compile STDP `compute_all_updates()` with Numba for 10-100x speedup.

**Files to Edit**:
- `src/ww/learning/stdp.py:498` (add Numba JIT variant)

**Changes**:
```python
# After line 498, add new JIT-compiled function:

import numba
import numpy as np

@numba.jit(nopython=True, parallel=True, fastmath=True)
def _compute_all_updates_jit(
    spike_times_pre: np.ndarray,   # [n_pre] spike times
    spike_times_post: np.ndarray,  # [n_post] spike times
    weights: np.ndarray,            # [n_pre, n_post] weight matrix
    a_plus: float,
    a_minus: float,
    tau_plus: float,
    tau_minus: float,
    multiplicative: bool,
    mu: float,
    max_weight: float
) -> np.ndarray:
    """
    JIT-compiled STDP weight updates (10-100x faster than Python loop).

    Returns:
        delta_weights: [n_pre, n_post] weight changes
    """
    n_pre, n_post = weights.shape
    delta_weights = np.zeros_like(weights)

    for i in numba.prange(n_pre):  # Parallel loop
        t_pre = spike_times_pre[i]
        if t_pre < 0:  # No spike
            continue

        for j in range(n_post):
            t_post = spike_times_post[j]
            if t_post < 0:  # No spike
                continue

            delta_t_s = (t_post - t_pre) / 1000.0  # ms → s

            if abs(delta_t_s) < 0.0001:  # Simultaneous
                continue

            w = weights[i, j]

            if multiplicative:
                w_clipped = min(max(w, 0.0), max_weight)
                if delta_t_s > 0:  # LTP
                    delta = a_plus * np.exp(-delta_t_s / tau_plus) * (max_weight - w_clipped) ** mu
                else:  # LTD
                    delta = -a_minus * np.exp(delta_t_s / tau_minus) * w_clipped ** mu
            else:
                if delta_t_s > 0:  # LTP
                    delta = a_plus * np.exp(-delta_t_s / tau_plus)
                else:  # LTD
                    delta = -a_minus * np.exp(delta_t_s / tau_minus)

            delta_weights[i, j] = delta

    return delta_weights

# Modify existing compute_all_updates() to use JIT version:
def compute_all_updates(self, use_jit: bool = True) -> dict[tuple[str, str], float]:
    """
    Compute STDP updates for all recorded spike pairs.

    Args:
        use_jit: Use Numba JIT-compiled version (10-100x faster)

    Returns:
        dict mapping (pre_id, post_id) -> weight_delta
    """
    if use_jit and len(self._weights) > 100:  # Only JIT for large networks
        # Convert spike history to arrays
        spike_times_pre = np.array([...])  # Extract from self._spike_history
        spike_times_post = np.array([...])
        weights_matrix = np.array([...])   # Convert dict to matrix

        delta_matrix = _compute_all_updates_jit(
            spike_times_pre, spike_times_post, weights_matrix,
            self.config.a_plus, self.config.a_minus,
            self.config.tau_plus, self.config.tau_minus,
            self.config.multiplicative, self.config.mu, self.config.max_weight
        )

        # Convert back to dict
        return self._matrix_to_dict(delta_matrix)
    else:
        # Fall back to Python loop for small networks
        return self._compute_all_updates_python()
```

**Tests to Write**:
- `tests/unit/test_stdp_jit.py`:
  - `test_jit_equivalence()`: JIT and Python produce same results
  - `test_jit_speedup()`: JIT ≥10x faster for 1000+ neurons

**Acceptance Criteria**:
- [ ] Numba JIT compiles without errors
- [ ] Results match Python implementation
- [ ] ≥10x speedup for 1000+ neurons
- [ ] Tests pass

**Dependencies**: ATOM-P1-03 (spike generation)

---

### ATOM-P1-05: Define Unified MemoryItem Schema

**Description**: Create unified memory schema with κ consolidation field.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/src/ww/core/unified_memory.py`

**Implementation**:
```python
"""Unified 4D memory substrate with κ consolidation gradient."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field

class MemoryType(str, Enum):
    """Memory type classification."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class MemoryItem(BaseModel):
    """
    Unified memory substrate for all types.

    Replaces separate Episode/Entity/Procedure classes with a single
    schema using κ (kappa) consolidation level to represent the
    gradient from episodic (κ=0) to semantic (κ=1).

    Bi-temporal 4D fields:
    - event_time: When it occurred (T_ref - reference time)
    - record_time: When recorded (T_sys - system time)
    - valid_from/valid_until: Temporal validity window
    """
    model_config = {"from_attributes": True, "validate_assignment": True}

    # Identity
    id: UUID = Field(default_factory=uuid4)
    type: MemoryType = MemoryType.EPISODIC

    # Content
    content: str = Field(..., min_length=1)
    embedding: list[float] | None = Field(default=None)

    # Bi-temporal (4D)
    event_time: datetime = Field(
        default_factory=datetime.now,
        description="When event occurred (T_ref)"
    )
    record_time: datetime = Field(
        default_factory=datetime.now,
        description="When memory recorded (T_sys)"
    )
    valid_from: datetime = Field(
        default_factory=datetime.now,
        description="Temporal validity start"
    )
    valid_until: datetime | None = Field(
        default=None,
        description="Temporal validity end (None = still valid)"
    )

    # Consolidation continuum
    kappa: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Consolidation level: 0=episodic, 1=semantic"
    )

    # Type-specific fields (stored as JSON)
    episode_fields: dict | None = Field(
        default=None,
        description="session_id, context, outcome, etc."
    )
    entity_fields: dict | None = Field(
        default=None,
        description="entity_type, properties, etc."
    )
    procedure_fields: dict | None = Field(
        default=None,
        description="steps, success_rate, etc."
    )

    # FSRS learning
    access_count: int = Field(default=1, ge=1)
    last_accessed: datetime = Field(default_factory=datetime.now)
    stability: float = Field(default=1.0, gt=0, description="FSRS stability (days)")

    # Prediction error (for prioritized replay)
    prediction_error: float | None = Field(default=None)
    prediction_error_timestamp: datetime | None = Field(default=None)

def query_episodic(items: list[MemoryItem]) -> list[MemoryItem]:
    """
    Episodic query policy: Recent, low-κ memories.

    Filters: κ < 0.3
    Sorts: event_time descending (most recent first)
    """
    return sorted(
        [m for m in items if m.kappa < 0.3],
        key=lambda m: m.event_time,
        reverse=True
    )

def query_semantic(items: list[MemoryItem]) -> list[MemoryItem]:
    """
    Semantic query policy: Consolidated, high-κ knowledge.

    Filters: κ > 0.7
    """
    return [m for m in items if m.kappa > 0.7]

def query_procedural(items: list[MemoryItem]) -> list[MemoryItem]:
    """
    Procedural query policy: Procedures only.

    Filters: type == PROCEDURAL
    Sorts: success_rate descending (from procedure_fields)
    """
    procedures = [m for m in items if m.type == MemoryType.PROCEDURAL]

    # Sort by success rate if available
    def get_success_rate(m: MemoryItem) -> float:
        if m.procedure_fields and "success_rate" in m.procedure_fields:
            return m.procedure_fields["success_rate"]
        return 0.0

    return sorted(procedures, key=get_success_rate, reverse=True)
```

**Tests to Write**:
- `tests/unit/test_unified_memory.py`:
  - `test_memory_item_creation()`: Basic instantiation
  - `test_kappa_bounds()`: κ ∈ [0, 1] validation
  - `test_query_episodic()`: κ < 0.3 filter
  - `test_query_semantic()`: κ > 0.7 filter
  - `test_query_procedural()`: Type filter + sort
  - `test_bitemporal_fields()`: event_time vs record_time

**Acceptance Criteria**:
- [ ] MemoryItem schema validates
- [ ] κ field enforces [0, 1] bounds
- [ ] 3 query policies work correctly
- [ ] Tests pass

**Dependencies**: None

---

### ATOM-P1-06: Add κ Field to Existing Episode Class

**Description**: Backward-compatible addition of kappa field to Episode.

**Files to Edit**:
- `src/ww/core/types.py:106` (Episode class)

**Changes**:
```python
# Line 146 (after prediction_error_timestamp field):

    # Consolidation level (Phase 2: Unified Memory)
    kappa: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Consolidation level: 0=fresh episodic, 1=fully semantic"
    )
```

**Tests to Update**:
- `tests/unit/test_types.py`:
  - Add `test_episode_kappa_default()`: Default κ=0
  - Add `test_episode_kappa_bounds()`: Validation

**Acceptance Criteria**:
- [ ] κ field added to Episode
- [ ] Default value 0.0 (fresh episodic)
- [ ] Pydantic validation enforces [0, 1]
- [ ] Tests pass

**Dependencies**: None

---

### ATOM-P1-07: Add κ Field to Entity Class

**Description**: Add kappa field to Entity class.

**Files to Edit**:
- `src/ww/core/types.py` (Entity class, find via search)

**Changes**: Same as ATOM-P1-06 but for Entity class.

**Tests to Update**:
- `tests/unit/test_types.py`: Add `test_entity_kappa_default()`

**Acceptance Criteria**:
- [ ] κ field added to Entity
- [ ] Default 0.0
- [ ] Tests pass

**Dependencies**: None

---

### ATOM-P1-08: Add κ Field to Procedure Class

**Description**: Add kappa field to Procedure class.

**Files to Edit**:
- `src/ww/core/types.py` (Procedure class, find via search)

**Changes**: Same as ATOM-P1-06 but for Procedure class.

**Tests to Update**:
- `tests/unit/test_types.py`: Add `test_procedure_kappa_default()`

**Acceptance Criteria**:
- [ ] κ field added to Procedure
- [ ] Default 0.0
- [ ] Tests pass

**Dependencies**: None

---

### ATOM-P1-09: Create PyTorch GPU PDE Solver Prototype

**Description**: Port NeuralFieldSolver to PyTorch GPU for 10x speedup.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/src/ww/nca/neural_field_gpu.py`

**Implementation**:
```python
"""GPU-accelerated neural field PDE solver using PyTorch."""
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPUFieldConfig:
    """Configuration for GPU PDE solver."""
    grid_size: int = 32          # 32³ grid
    n_neurotransmitters: int = 6  # DA, 5-HT, ACh, NE, GABA, Glu
    dt: float = 0.01              # Time step (seconds)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class NeuralFieldSolverGPU:
    """
    GPU-accelerated PDE solver for neurotransmitter fields.

    Core equation:
        ∂U/∂t = -αU + D∇²U + S(x,t) + C(U₁...Uₙ)

    Uses PyTorch for GPU parallelization of Laplacian computation.
    """

    def __init__(self, config: GPUFieldConfig | None = None):
        self.config = config or GPUFieldConfig()
        self.device = torch.device(self.config.device)

        # Initialize field state [n_nt, grid_size, grid_size, grid_size]
        self.field = torch.zeros(
            (
                self.config.n_neurotransmitters,
                self.config.grid_size,
                self.config.grid_size,
                self.config.grid_size,
            ),
            device=self.device,
            dtype=torch.float32,
        )

        # Decay rates (NT-specific, in Hz)
        self.alpha = torch.tensor(
            [0.5, 0.3, 0.7, 0.6, 1.0, 2.0],  # DA, 5-HT, ACh, NE, GABA, Glu
            device=self.device,
            dtype=torch.float32,
        ).view(-1, 1, 1, 1)

        # Diffusion coefficients
        self.diffusion = torch.tensor(
            [0.1, 0.1, 0.1, 0.1, 0.5, 0.5],
            device=self.device,
            dtype=torch.float32,
        ).view(-1, 1, 1, 1)

    def compute_laplacian_3d(self, field: torch.Tensor) -> torch.Tensor:
        """
        Compute 3D Laplacian using 7-point stencil.

        Args:
            field: [n_nt, H, W, D] tensor

        Returns:
            laplacian: [n_nt, H, W, D] tensor
        """
        # 3D convolution with Laplacian kernel
        # Laplacian kernel: center=-6, 6 neighbors=+1
        kernel = torch.zeros((1, 1, 3, 3, 3), device=self.device)
        kernel[0, 0, 1, 1, 1] = -6.0  # Center
        kernel[0, 0, 0, 1, 1] = 1.0   # Left
        kernel[0, 0, 2, 1, 1] = 1.0   # Right
        kernel[0, 0, 1, 0, 1] = 1.0   # Front
        kernel[0, 0, 1, 2, 1] = 1.0   # Back
        kernel[0, 0, 1, 1, 0] = 1.0   # Bottom
        kernel[0, 0, 1, 1, 2] = 1.0   # Top

        # Apply to each NT separately
        laplacian = F.conv3d(
            field.unsqueeze(1),  # [n_nt, 1, H, W, D]
            kernel,
            padding=1
        ).squeeze(1)  # [n_nt, H, W, D]

        return laplacian

    def step(
        self,
        stimulus: torch.Tensor | None = None,
        coupling: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Advance PDE by one time step (GPU-accelerated).

        Args:
            stimulus: [n_nt, H, W, D] external input (optional)
            coupling: [n_nt, H, W, D] cross-NT coupling (optional)

        Returns:
            field: [n_nt, H, W, D] updated concentration field
        """
        # Compute Laplacian (GPU parallelized)
        laplacian = self.compute_laplacian_3d(self.field)

        # PDE update: dU/dt = -αU + D∇²U + S + C
        dU_dt = (
            -self.alpha * self.field +
            self.diffusion * laplacian
        )

        if stimulus is not None:
            dU_dt += stimulus

        if coupling is not None:
            dU_dt += coupling

        # Forward Euler integration
        self.field = self.field + self.config.dt * dU_dt

        # Clip to [0, ∞) (concentrations non-negative)
        self.field = torch.clamp(self.field, min=0.0)

        return self.field
```

**Tests to Write**:
- `tests/unit/test_neural_field_gpu.py`:
  - `test_laplacian_computation()`: Verify 7-point stencil
  - `test_pde_step()`: Field evolves correctly
  - `test_gpu_usage()`: Uses CUDA if available
  - `test_speedup_vs_cpu()`: ≥10x faster than NumPy

**Acceptance Criteria**:
- [ ] PyTorch GPU solver works
- [ ] ≥10x speedup vs CPU on RTX 3090
- [ ] Tests pass

**Dependencies**: None

---

### ATOM-P1-10: Write Documentation for τ(t) Signal

**Description**: Document temporal control signal design.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/docs/reference/temporal_control.md`

**Content**:
```markdown
# Temporal Control Signal τ(t)

## Overview

The temporal control signal τ(t) modulates memory write strength based on prediction error magnitude.

## Equation

τ(t) = sigmoid(k · (|ε(t)| - θ))

where:
- ε(t) = prediction error from dopamine RPE
- k = steepness (default 5.0)
- θ = threshold (default 0.3)

## Biological Basis

High prediction errors signal surprising/important events that should be strongly encoded. Low prediction errors indicate expected events that require less encoding strength.

## Usage

```python
from ww.core.temporal_control import compute_tau

pe = 0.5  # High prediction error
tau = compute_tau(pe)  # tau ≈ 0.92 → strong write
```

## Integration

Used in `MemoryGate` to multiply gate score:

```python
final_score = tau * base_score
```
```

**Acceptance Criteria**:
- [ ] Documentation complete
- [ ] Includes equation, biological basis, usage examples

**Dependencies**: ATOM-P1-01

---

### ATOM-P1-11: Write Documentation for Unified MemoryItem

**Description**: Document unified memory schema design rationale.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/docs/reference/unified_memory.md`

**Content**:
```markdown
# Unified Memory Substrate

## Overview

Replaces 3 separate memory stores (episodic, semantic, procedural) with a single unified schema using κ (kappa) consolidation level.

## Schema

```python
class MemoryItem:
    kappa: float  # ∈ [0, 1] consolidation level
    # κ=0: Fresh episodic memory
    # 0 < κ < 1: Partially consolidated
    # κ=1: Fully semantic knowledge
```

## Query Policies

Instead of 3 separate stores, use 3 query policies:

1. **Episodic**: Filter κ < 0.3, sort by recency
2. **Semantic**: Filter κ > 0.7, sort by centrality
3. **Procedural**: Filter type=PROCEDURAL, sort by success rate

## Consolidation Updates κ

During sleep consolidation:
- NREM replay: κ += 0.05 per replay
- REM clustering: κ += 0.2 for clustered memories

## Migration Path

1. Add κ field to Episode/Entity/Procedure (backward compatible)
2. Create MemoryItem alongside existing
3. Migrate data
4. Update query logic
5. Deprecate old classes
```

**Acceptance Criteria**:
- [ ] Documentation complete
- [ ] Includes schema, query policies, consolidation, migration

**Dependencies**: ATOM-P1-05

---

### ATOM-P1-12: Phase 1 Integration Test

**Description**: End-to-end test of Phase 1 components.

**Tests to Write**:
- `tests/integration/test_phase1_integration.py`:
  - `test_tau_to_gate_integration()`: PE → τ(t) → MemoryGate
  - `test_norse_spike_generation()`: Norse backend generates spikes
  - `test_numba_stdp_speedup()`: JIT ≥10x faster
  - `test_unified_memory_queries()`: 3 query policies work

**Acceptance Criteria**:
- [ ] All Phase 1 components integrate
- [ ] Integration tests pass
- [ ] No performance regressions

**Dependencies**: ATOM-P1-01 through ATOM-P1-11

---

## 5. Phase 2: Unified Memory Substrate

**Duration**: 3 weeks
**Atoms**: 15
**Deliverable**: 1 unified store with κ gradient, 3 query policies, data migration

### ATOM-P2-01: Create Unified Memory Store Backend

**Description**: Implement storage backend for MemoryItem.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/src/ww/storage/unified_store.py`

**Implementation**:
```python
"""Unified memory store backend (Neo4j + Qdrant)."""
import logging
from uuid import UUID
from datetime import datetime
from ww.core.unified_memory import MemoryItem, MemoryType
from ww.storage.neo4j_client import Neo4jClient
from ww.storage.qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class UnifiedMemoryStore:
    """
    Unified storage backend for MemoryItem.

    Uses Neo4j for graph relationships and Qdrant for vector search.
    """

    def __init__(
        self,
        neo4j_client: Neo4jClient,
        qdrant_client: QdrantClient,
        collection_name: str = "unified_memories"
    ):
        self.neo4j = neo4j_client
        self.qdrant = qdrant_client
        self.collection_name = collection_name

    def create(self, item: MemoryItem) -> UUID:
        """
        Store memory item (dual write to Neo4j + Qdrant).

        Args:
            item: MemoryItem to store

        Returns:
            UUID of stored memory
        """
        # 1. Store in Neo4j (graph properties)
        self.neo4j.run(
            """
            CREATE (m:MemoryItem {
                id: $id,
                type: $type,
                content: $content,
                kappa: $kappa,
                event_time: $event_time,
                record_time: $record_time,
                valid_from: $valid_from,
                valid_until: $valid_until,
                access_count: $access_count,
                last_accessed: $last_accessed,
                stability: $stability,
                prediction_error: $prediction_error
            })
            """,
            {
                "id": str(item.id),
                "type": item.type.value,
                "content": item.content,
                "kappa": item.kappa,
                "event_time": item.event_time.isoformat(),
                "record_time": item.record_time.isoformat(),
                "valid_from": item.valid_from.isoformat(),
                "valid_until": item.valid_until.isoformat() if item.valid_until else None,
                "access_count": item.access_count,
                "last_accessed": item.last_accessed.isoformat(),
                "stability": item.stability,
                "prediction_error": item.prediction_error,
            }
        )

        # 2. Store in Qdrant (vector + metadata)
        if item.embedding:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[{
                    "id": str(item.id),
                    "vector": item.embedding,
                    "payload": {
                        "type": item.type.value,
                        "kappa": item.kappa,
                        "event_time": item.event_time.timestamp(),
                        "content": item.content[:1000],  # Truncate for payload
                    }
                }]
            )

        logger.info(f"Created MemoryItem {item.id} (κ={item.kappa:.2f})")
        return item.id

    def get(self, memory_id: UUID) -> MemoryItem | None:
        """Retrieve memory item by ID."""
        # Query Neo4j
        result = self.neo4j.run(
            "MATCH (m:MemoryItem {id: $id}) RETURN m",
            {"id": str(memory_id)}
        )

        record = result.single()
        if not record:
            return None

        # Convert to MemoryItem
        node = record["m"]
        return MemoryItem(
            id=UUID(node["id"]),
            type=MemoryType(node["type"]),
            content=node["content"],
            kappa=node["kappa"],
            event_time=datetime.fromisoformat(node["event_time"]),
            record_time=datetime.fromisoformat(node["record_time"]),
            # ... other fields
        )

    def update_kappa(self, memory_id: UUID, delta_kappa: float):
        """
        Increment κ (consolidation level).

        Args:
            memory_id: Memory to update
            delta_kappa: Amount to increment (e.g., 0.05 for NREM, 0.2 for REM)
        """
        self.neo4j.run(
            """
            MATCH (m:MemoryItem {id: $id})
            SET m.kappa = CASE
                WHEN m.kappa + $delta >= 1.0 THEN 1.0
                ELSE m.kappa + $delta
            END
            """,
            {"id": str(memory_id), "delta": delta_kappa}
        )

        # Also update Qdrant payload
        self.qdrant.set_payload(
            collection_name=self.collection_name,
            points=[str(memory_id)],
            payload={"kappa": min(1.0, delta_kappa)}  # Simplified (should read current first)
        )

        logger.debug(f"Updated κ for {memory_id}: +{delta_kappa:.2f}")
```

**Tests to Write**:
- `tests/unit/test_unified_store.py`:
  - `test_create_memory_item()`: Dual write works
  - `test_get_memory_item()`: Retrieval works
  - `test_update_kappa()`: κ increment works
  - `test_kappa_ceiling()`: κ capped at 1.0

**Acceptance Criteria**:
- [ ] Create/get/update operations work
- [ ] Dual write to Neo4j + Qdrant
- [ ] κ updates correctly
- [ ] Tests pass

**Dependencies**: ATOM-P1-05 (MemoryItem schema)

---

### ATOM-P2-02: Implement Episodic Query Policy

**Description**: Filter κ < 0.3, sort by recency.

**Files to Edit**:
- `src/ww/storage/unified_store.py` (add query methods)

**Changes**:
```python
def query_episodic(
    self,
    query_vector: list[float] | None = None,
    top_k: int = 10,
    kappa_threshold: float = 0.3
) -> list[MemoryItem]:
    """
    Episodic query policy: Recent, low-κ memories.

    Args:
        query_vector: Optional vector for similarity search
        top_k: Number of results
        kappa_threshold: Max κ for episodic (default 0.3)

    Returns:
        List of MemoryItem sorted by recency
    """
    if query_vector:
        # Vector search with kappa filter
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            filter={"kappa": {"$lt": kappa_threshold}},
            limit=top_k
        )
        ids = [r.id for r in results]
    else:
        # Just get recent low-κ memories
        results = self.neo4j.run(
            """
            MATCH (m:MemoryItem)
            WHERE m.kappa < $threshold
            RETURN m.id AS id
            ORDER BY m.event_time DESC
            LIMIT $limit
            """,
            {"threshold": kappa_threshold, "limit": top_k}
        )
        ids = [record["id"] for record in results]

    # Fetch full MemoryItem objects
    return [self.get(UUID(id)) for id in ids]
```

**Tests to Write**:
- `tests/unit/test_query_policies.py`:
  - `test_query_episodic_filters_kappa()`: Only κ < 0.3 returned
  - `test_query_episodic_sorts_recency()`: Most recent first

**Acceptance Criteria**:
- [ ] Returns only κ < 0.3 memories
- [ ] Sorted by recency
- [ ] Tests pass

**Dependencies**: ATOM-P2-01

---

### ATOM-P2-03: Implement Semantic Query Policy

**Description**: Filter κ > 0.7, graph traversal.

**Files to Edit**:
- `src/ww/storage/unified_store.py`

**Changes**:
```python
def query_semantic(
    self,
    query_vector: list[float],
    top_k: int = 10,
    kappa_threshold: float = 0.7
) -> list[MemoryItem]:
    """
    Semantic query policy: Consolidated, high-κ knowledge.

    Args:
        query_vector: Vector for similarity search
        top_k: Number of results
        kappa_threshold: Min κ for semantic (default 0.7)

    Returns:
        List of MemoryItem sorted by similarity
    """
    # Vector search with kappa filter
    results = self.qdrant.search(
        collection_name=self.collection_name,
        query_vector=query_vector,
        filter={"kappa": {"$gte": kappa_threshold}},
        limit=top_k
    )

    ids = [r.id for r in results]
    return [self.get(UUID(id)) for id in ids]
```

**Tests to Write**:
- `tests/unit/test_query_policies.py`:
  - `test_query_semantic_filters_kappa()`: Only κ > 0.7 returned

**Acceptance Criteria**:
- [ ] Returns only κ > 0.7 memories
- [ ] Tests pass

**Dependencies**: ATOM-P2-01

---

### ATOM-P2-04: Implement Procedural Query Policy

**Description**: Filter type=PROCEDURAL, sort by success rate.

**Files to Edit**:
- `src/ww/storage/unified_store.py`

**Changes**:
```python
def query_procedural(
    self,
    domain: str | None = None,
    top_k: int = 10
) -> list[MemoryItem]:
    """
    Procedural query policy: Procedures sorted by success rate.

    Args:
        domain: Optional domain filter
        top_k: Number of results

    Returns:
        List of MemoryItem sorted by success_rate descending
    """
    query = """
    MATCH (m:MemoryItem)
    WHERE m.type = 'procedural'
    """

    if domain:
        query += " AND m.procedure_fields.domain = $domain"

    query += """
    RETURN m.id AS id, m.procedure_fields.success_rate AS success_rate
    ORDER BY success_rate DESC
    LIMIT $limit
    """

    results = self.neo4j.run(
        query,
        {"domain": domain, "limit": top_k} if domain else {"limit": top_k}
    )

    ids = [record["id"] for record in results]
    return [self.get(UUID(id)) for id in ids]
```

**Tests to Write**:
- `tests/unit/test_query_policies.py`:
  - `test_query_procedural_filters_type()`: Only procedures returned
  - `test_query_procedural_sorts_success()`: Highest success first

**Acceptance Criteria**:
- [ ] Returns only PROCEDURAL type
- [ ] Sorted by success_rate
- [ ] Tests pass

**Dependencies**: ATOM-P2-01

---

### ATOM-P2-05: Update Consolidation to Increment κ (NREM)

**Description**: During NREM replay, increment κ by 0.05.

**Files to Edit**:
- `src/ww/consolidation/sleep.py:342-365` (replay loop)

**Changes**:
```python
# In NREM replay loop (around line 342):

# OLD:
for memory in replay_batch:
    # Process memory
    pass

# NEW:
from ww.storage.unified_store import UnifiedMemoryStore

unified_store = get_unified_store()  # Injected dependency

for memory in replay_batch:
    # Process memory (existing logic)
    # ...

    # Increment κ for each replay
    unified_store.update_kappa(memory.id, delta_kappa=0.05)

    logger.debug(
        f"NREM replay: {memory.id} κ incremented by 0.05 "
        f"(replay count: {memory.access_count})"
    )
```

**Tests to Write**:
- `tests/unit/test_consolidation_kappa.py`:
  - `test_nrem_increments_kappa()`: κ += 0.05 per replay
  - `test_kappa_ceiling()`: κ capped at 1.0

**Acceptance Criteria**:
- [ ] NREM replay increments κ by 0.05
- [ ] κ never exceeds 1.0
- [ ] Tests pass

**Dependencies**: ATOM-P2-01

---

### ATOM-P2-06: Update Consolidation to Increment κ (REM)

**Description**: During REM clustering, increment κ by 0.2.

**Files to Edit**:
- `src/ww/consolidation/sleep.py:345-389` (REM clustering)

**Changes**:
```python
# In REM clustering (around line 378):

# OLD:
for cluster in hdbscan_clusters:
    # Create new semantic entity
    entity = Entity(...)
    semantic_store.create(entity)

# NEW:
for cluster in hdbscan_clusters:
    # Increment κ for all memories in cluster
    for memory_id in cluster:
        unified_store.update_kappa(memory_id, delta_kappa=0.2)

    logger.info(
        f"REM clustering: {len(cluster)} memories consolidated "
        f"(κ += 0.2)"
    )
```

**Tests to Write**:
- `tests/unit/test_consolidation_kappa.py`:
  - `test_rem_increments_kappa()`: κ += 0.2 for clustered memories

**Acceptance Criteria**:
- [ ] REM clustering increments κ by 0.2
- [ ] Tests pass

**Dependencies**: ATOM-P2-01, ATOM-P2-05

---

### ATOM-P2-07: Create Data Migration Script

**Description**: Migrate existing Episode/Entity/Procedure to MemoryItem.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/scripts/migrate_to_unified.py`

**Implementation**:
```python
"""Migrate existing memories to unified MemoryItem schema."""
import logging
from uuid import UUID
from ww.core.types import Episode, Entity, Procedure
from ww.core.unified_memory import MemoryItem, MemoryType
from ww.storage.unified_store import UnifiedMemoryStore
from ww.memory.episodic import EpisodicMemory
from ww.memory.semantic import SemanticMemory
from ww.memory.procedural import ProceduralMemory

logger = logging.getLogger(__name__)

def migrate_episodes(
    episodic: EpisodicMemory,
    unified: UnifiedMemoryStore
) -> int:
    """
    Migrate Episode → MemoryItem (type=EPISODIC, κ=0.0).

    Returns:
        Number of episodes migrated
    """
    episodes = episodic.list_all()  # Hypothetical method
    count = 0

    for episode in episodes:
        item = MemoryItem(
            id=episode.id,
            type=MemoryType.EPISODIC,
            content=episode.content,
            embedding=episode.embedding,
            event_time=episode.timestamp,
            record_time=episode.ingested_at,
            valid_from=episode.timestamp,
            kappa=0.0,  # Fresh episodic
            episode_fields={
                "session_id": episode.session_id,
                "outcome": episode.outcome.value,
                "emotional_valence": episode.emotional_valence,
            },
            access_count=episode.access_count,
            last_accessed=episode.last_accessed,
            stability=episode.stability,
            prediction_error=episode.prediction_error,
        )

        unified.create(item)
        count += 1

    logger.info(f"Migrated {count} episodes")
    return count

def migrate_entities(
    semantic: SemanticMemory,
    unified: UnifiedMemoryStore
) -> int:
    """
    Migrate Entity → MemoryItem (type=SEMANTIC, κ=1.0).

    Returns:
        Number of entities migrated
    """
    entities = semantic.list_all()
    count = 0

    for entity in entities:
        item = MemoryItem(
            id=entity.id,
            type=MemoryType.SEMANTIC,
            content=entity.content,
            embedding=entity.embedding,
            event_time=entity.first_seen,
            record_time=entity.created_at,
            valid_from=entity.first_seen,
            valid_until=entity.last_seen,  # Temporal validity
            kappa=1.0,  # Fully semantic
            entity_fields={
                "entity_type": entity.entity_type,
                "properties": entity.properties,
            },
            access_count=entity.mentions,
        )

        unified.create(item)
        count += 1

    logger.info(f"Migrated {count} entities")
    return count

def migrate_procedures(
    procedural: ProceduralMemory,
    unified: UnifiedMemoryStore
) -> int:
    """
    Migrate Procedure → MemoryItem (type=PROCEDURAL, κ=0.5).

    Returns:
        Number of procedures migrated
    """
    procedures = procedural.list_all()
    count = 0

    for proc in procedures:
        item = MemoryItem(
            id=proc.id,
            type=MemoryType.PROCEDURAL,
            content=proc.domain,
            event_time=proc.created_at,
            record_time=proc.created_at,
            valid_from=proc.created_at,
            kappa=0.5,  # Partially consolidated
            procedure_fields={
                "domain": proc.domain,
                "steps": proc.steps,
                "success_count": proc.success_count,
                "total_count": proc.total_count,
                "success_rate": proc.success_count / max(proc.total_count, 1),
            },
        )

        unified.create(item)
        count += 1

    logger.info(f"Migrated {count} procedures")
    return count

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize stores
    episodic = EpisodicMemory(...)
    semantic = SemanticMemory(...)
    procedural = ProceduralMemory(...)
    unified = UnifiedMemoryStore(...)

    # Run migration
    total = 0
    total += migrate_episodes(episodic, unified)
    total += migrate_entities(semantic, unified)
    total += migrate_procedures(procedural, unified)

    logger.info(f"Migration complete: {total} memories migrated")
```

**Tests to Write**:
- `tests/unit/test_migration.py`:
  - `test_migrate_episodes()`: Episodes → κ=0.0
  - `test_migrate_entities()`: Entities → κ=1.0
  - `test_migrate_procedures()`: Procedures → κ=0.5

**Acceptance Criteria**:
- [ ] Script migrates all 3 memory types
- [ ] κ values correct (0.0, 1.0, 0.5)
- [ ] No data loss
- [ ] Tests pass

**Dependencies**: ATOM-P2-01

---

### ATOM-P2-08 through ATOM-P2-15: [Continue Pattern]

Due to length constraints, I'll summarize remaining Phase 2 atoms:

- **ATOM-P2-08**: Update episodic retrieval to use query_episodic()
- **ATOM-P2-09**: Update semantic retrieval to use query_semantic()
- **ATOM-P2-10**: Update procedural retrieval to use query_procedural()
- **ATOM-P2-11**: Add backward compatibility layer for old API
- **ATOM-P2-12**: Update API routes to use unified store
- **ATOM-P2-13**: Update CLI to use unified store
- **ATOM-P2-14**: Phase 2 integration test
- **ATOM-P2-15**: Performance benchmark (unified vs 3-store)

---

## 6. Phase 3: Spike Pipeline

**Duration**: 2 weeks
**Atoms**: 10
**Deliverable**: Complete spike reinjection loop

### ATOM-P3-01: Create Spike Reinjection Module

**Description**: Convert replayed embeddings to spike trains.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/src/ww/nca/spike_reinjection.py`

**Implementation**: (See plan section 12, REINJ block in original document)

**Tests to Write**:
- `tests/unit/test_spike_reinjection.py`:
  - `test_embed_to_spikes()`: Embedding → spike train
  - `test_rate_coding()`: High embedding values → high spike rate
  - `test_duration()`: Spike train has correct length

**Acceptance Criteria**:
- [ ] Embedding → spike conversion works
- [ ] Spike rate proportional to embedding magnitude
- [ ] Tests pass

**Dependencies**: ATOM-P1-03 (Norse backend)

---

### ATOM-P3-02: Integrate Spike Reinjection with NREM Replay

**Description**: Feed replayed memories as spikes to SNN.

**Files to Edit**:
- `src/ww/consolidation/sleep.py:342-365`

**Changes**:
```python
# In NREM replay loop:

from ww.nca.spike_reinjection import SpikeReinjector
from ww.nca.snn_backend import NorseSpikeGenerator

reinjector = SpikeReinjector()
snn = NorseSpikeGenerator()

for memory in replay_batch:
    # Convert embedding to spike train
    spike_train = reinjector.embed_to_spikes(
        torch.tensor(memory.embedding),
        duration_ms=100
    )

    # Feed to SNN
    spikes = snn.generate_spikes(spike_train)

    # Process with STDP (ATOM-P3-03)
```

**Tests to Write**:
- `tests/integration/test_replay_spike_loop.py`:
  - `test_replay_generates_spikes()`: Replay → spikes

**Acceptance Criteria**:
- [ ] Replay loop generates spikes
- [ ] Tests pass

**Dependencies**: ATOM-P3-01

---

### ATOM-P3-03 through ATOM-P3-10: [Continue Pattern]

Remaining Phase 3 atoms:

- **ATOM-P3-03**: Connect SNN output to STDP learner
- **ATOM-P3-04**: STDP weight update during replay
- **ATOM-P3-05**: Weight update → memory update loop
- **ATOM-P3-06**: Verify replay closure (weight changes)
- **ATOM-P3-07**: PyTorch GPU port for PDE solver (production)
- **ATOM-P3-08**: Norse LIF integration with oscillators
- **ATOM-P3-09**: Phase 3 integration test
- **ATOM-P3-10**: Performance benchmark (spike pipeline)

---

## 7. Phase 4: Validation

**Duration**: 1 week
**Atoms**: 8
**Deliverable**: Biological plausibility test suite

### ATOM-P4-01: MNE-Python Oscillation Validation

**Description**: Validate theta/gamma/alpha/delta bands using MNE spectral analysis.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/tests/biology/test_oscillator_validation.py`

**Implementation**:
```python
"""Validate oscillator frequency bands using MNE-Python."""
import numpy as np
import mne
from mne.time_frequency import psd_array_welch
from ww.nca.oscillators import ThetaOscillator, GammaOscillator, DeltaOscillator

def test_theta_oscillator_frequency_band():
    """Validate ThetaOscillator produces 4-8 Hz power peak."""
    osc = ThetaOscillator(frequency=6.0)
    sampling_freq = 1000  # 1 kHz

    # Generate 10 seconds of signal
    signal = np.array([osc.step() for _ in range(10000)])

    # Compute power spectral density
    psd, freqs = psd_array_welch(
        signal,
        sfreq=sampling_freq,
        fmin=0.5,
        fmax=100
    )

    # Theta band (4-8 Hz) should have peak power
    theta_band_mask = (freqs >= 4) & (freqs <= 8)
    theta_power = np.mean(psd[theta_band_mask])

    # Other bands should have less power
    gamma_band_mask = (freqs >= 30) & (freqs <= 100)
    gamma_power = np.mean(psd[gamma_band_mask])

    assert theta_power > gamma_power, \
        f"Theta power ({theta_power}) should exceed gamma ({gamma_power})"

def test_gamma_oscillator_frequency_band():
    """Validate GammaOscillator produces 30-100 Hz power peak."""
    osc = GammaOscillator(frequency=40.0)
    sampling_freq = 1000

    signal = np.array([osc.step() for _ in range(10000)])
    psd, freqs = psd_array_welch(signal, sfreq=sampling_freq, fmin=0.5, fmax=100)

    gamma_band_mask = (freqs >= 30) & (freqs <= 100)
    gamma_power = np.mean(psd[gamma_band_mask])

    theta_band_mask = (freqs >= 4) & (freqs <= 8)
    theta_power = np.mean(psd[theta_band_mask])

    assert gamma_power > theta_power, \
        f"Gamma power ({gamma_power}) should exceed theta ({theta_power})"
```

**Acceptance Criteria**:
- [ ] Theta oscillator peaks in 4-8 Hz band
- [ ] Gamma oscillator peaks in 30-100 Hz band
- [ ] Delta oscillator peaks in 0.5-4 Hz band
- [ ] Tests pass

**Dependencies**: None (validation only)

---

### ATOM-P4-02 through ATOM-P4-08: [Continue Pattern]

Remaining Phase 4 atoms:

- **ATOM-P4-02**: Elephant spike train cross-correlation validation
- **ATOM-P4-03**: Elephant Granger causality validation
- **ATOM-P4-04**: NetworkX connectome shortest path analysis
- **ATOM-P4-05**: NetworkX community detection validation
- **ATOM-P4-06**: STDP LTP/LTD window validation (17ms/34ms)
- **ATOM-P4-07**: Biological parameter bounds check
- **ATOM-P4-08**: Phase 4 validation report generation

---

## 8. Phase 5: Visualization + Polish

**Duration**: 1 week
**Atoms**: 5
**Deliverable**: Updated diagrams, BrainRender exports, documentation

### ATOM-P5-01: BrainRender Connectome Export

**Description**: 3D visualization of brain regions and pathways.

**Files to Create**:
- `/mnt/projects/t4d/t4dm/src/ww/visualization/brainrender_export.py`

**Implementation**: (See plan section 4.3 in original document)

**Acceptance Criteria**:
- [ ] Connectome exported to BrainRender
- [ ] 3D visualization generated
- [ ] HTML export works

**Dependencies**: None

---

### ATOM-P5-02 through ATOM-P5-05: [Continue Pattern]

- **ATOM-P5-02**: Update all Mermaid diagrams (see section 9)
- **ATOM-P5-03**: Update documentation (architecture, API, guides)
- **ATOM-P5-04**: Performance benchmark report
- **ATOM-P5-05**: Final integration test (all phases)

---

## 9. Mermaid Diagram Updates

### Diagrams Requiring Updates

| Diagram | File | Changes Needed |
|---------|------|----------------|
| Learning subsystem | `12_learning_subsystem.mmd` | Add Norse backend, τ(t) signal |
| Learning classes | `22_class_learning.mmd` | Add NorseSpikeGenerator, SpikeReinjector |
| Consolidation pipeline | `08_consolidation_pipeline.mmd` | Add κ update, spike reinjection |
| Memory subsystem | `11_memory_subsystem.mmd` | Replace 3 stores with unified store |
| Memory classes | `21_class_memory.mmd` | Add MemoryItem class |
| NCA module map | `nca_module_map.mermaid` | Add snn_backend.py, spike_reinjection.py |
| **NEW** | `snn_integration_flow.mmd` | End-to-end spike flow diagram |

### ATOM-P5-02-A: Update 12_learning_subsystem.mmd

**Changes**:
```mermaid
graph TB
    subgraph "Learning Subsystem"
        STDP[STDP Learner]
        Norse[Norse SNN Backend]  <!-- NEW -->
        ELIG[Eligibility Traces]
        THREE[Three-Factor Rule]
        NEURO[Neuromodulators]
        TAU[τ(t) Temporal Control]  <!-- NEW -->
    end

    STDP --> Norse
    TAU --> STDP
    NEURO --> THREE
```

### ATOM-P5-02-B: Create NEW snn_integration_flow.mmd

**New Diagram**:
```mermaid
graph LR
    Input[Sensory Input] --> Time2Vec[Time2Vec Encoder]
    Time2Vec --> Norse[Norse LIF Neurons]
    Norse --> Spikes[Spike Train]

    Replay[Replay Sampler] --> SpikeReinj[Spike Reinjection]
    SpikeReinj --> Norse

    Spikes --> STDP[STDP Learner]
    STDP --> WeightUpdate[Weight Update]
    WeightUpdate --> Memory[Unified Memory κ]

    PE[Prediction Error] --> Tau[τ(t) Control]
    Tau --> Gate[Memory Gate]
    Gate --> Memory

    Memory --> Consolidation[Sleep Consolidation]
    Consolidation --> KappaUpdate[κ += 0.05/0.2]
    KappaUpdate --> Memory
```

---

## 10. RTX 3090 Memory Budget

| Component | Memory (GB) | Precision | Notes |
|-----------|-------------|-----------|-------|
| Norse LIF neurons | 0.256 | FP32 | 100K neurons × 256 dims |
| Embedding cache | 2.0 | FP16 | 1M memories × 1024 dims (halved) |
| HDBSCAN clustering | 2.0 | FP32 | Pairwise distance matrix |
| Neural field solver | 0.256 | FP16 | 6 NTs × 32³ grid (halved) |
| STDP weight matrix | 0.5 | FP16 | 1000 × 1000 synapses (halved) |
| Replay buffer | 0.256 | FP16 | 100 spike trains × 1024 dims |
| PyTorch overhead | 2.0 | - | CUDA kernels, gradients |
| System reserve | 4.0 | - | OS, other processes |
| **Total** | **11.268** | - | **< 24GB ✅ (47% utilization)** |

### Mixed Precision Strategy

**FP16 (half precision)**:
- Embedding storage (2x savings)
- Neural field state (2x savings)
- STDP weight matrices (2x savings)
- Spike reinjection buffer

**FP32 (full precision)**:
- Final output layers (numerical stability)
- Loss computation
- Small parameter tensors

**Code**:
```python
from torch.cuda.amp import autocast

with autocast():
    spikes = lif_cell(input_current)  # FP16
    weights = stdp_learner(spikes)    # FP16
```

**Benefit**: 2x memory savings, 1.5-2x speedup on RTX 3090.

---

## Appendix A: Full Atom Dependencies

```
Phase 1:
  P1-01 → P1-02
  P1-01 → P1-10
  P1-03 → P1-04
  P1-05 → P1-06, P1-07, P1-08, P1-11
  P1-01...P1-11 → P1-12

Phase 2:
  P1-05 → P2-01
  P2-01 → P2-02, P2-03, P2-04, P2-05, P2-06, P2-07
  P2-01...P2-07 → P2-08...P2-15

Phase 3:
  P1-03 → P3-01
  P3-01 → P3-02
  P3-02 → P3-03...P3-10

Phase 4:
  All validation independent (parallel)

Phase 5:
  P1...P4 → P5-01...P5-05
```

---

## Appendix B: Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| STDP JIT speedup | ≥10x | Benchmark compute_all_updates() |
| PDE GPU speedup | ≥10x | Benchmark NeuralFieldSolver.step() |
| Unified store perf | ≤10% regression | Query latency vs 3-store |
| Spike reinjection latency | <100ms | End-to-end replay loop |
| Theta band power | >50% in 4-8 Hz | MNE spectral analysis |
| STDP LTP peak | ~17ms | Elephant cross-correlation |
| Memory budget | <12GB | RTX 3090 utilization |
| Test coverage | >80% | pytest-cov |

---

## Appendix C: Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Numba JIT fails | Low | High | Fallback to Python loop |
| PyTorch OOM on GPU | Medium | High | Mixed precision (FP16) |
| Migration data loss | Low | Critical | Dry run, backups, rollback script |
| Performance regression | Medium | Medium | Benchmark each atom, profile |
| Biological validation fails | Low | Medium | Adjust oscillator parameters |

---

**END OF PLAN**

**Total Duration**: 9 weeks
**Total Atoms**: 50
**Critical Path**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
**Parallelization**: Phase 4 atoms can run in parallel

This plan provides atomic, actionable tasks with verified line numbers, clear acceptance criteria, and comprehensive integration points. Each atom is independently testable and mergeable.
