# Cross-Region Integration (H10)

Consistency mechanisms for integrating neural subsystems in World Weaver.

**Implementation**: H10 (cross-region consistency validation)

## Overview

H10 ensures that the various neural subsystems (NCA, hippocampus, neuromodulators, capsules, etc.) operate coherently. This involves:

1. **Bidirectional coupling** between components
2. **State consistency** across regions
3. **Neuromodulator orchestration**
4. **Credit flow propagation**

## Coupling Bridges

Three primary coupling bridges connect major subsystems:

### 1. FF-NCA Coupling

Aligns Forward-Forward local learning with global energy landscape:

```
FF Goodness ←→ Energy Landscape
─────────────────────────────────
Positive phase  →  Descend to basin
Negative phase  →  Ascend barrier
Threshold θ     ←→  Energy barrier
```

```python
from ww.nca import FFNCACoupling, FFNCACouplingConfig

coupling = FFNCACoupling(FFNCACouplingConfig(
    goodness_to_energy_scale=1.0,   # G → -E
    energy_to_goodness_scale=1.0,   # E → -G
    temperature_coupling=0.5         # NT modulation
))

# Bidirectional update
ff_state = coupling.ff_to_nca(goodness_values, layer_activations)
nca_feedback = coupling.nca_to_ff(energy_landscape, attractors)
```

### 2. Capsule-NCA Coupling

Integrates capsule routing with neural field dynamics:

| NT | Effect on Capsules |
|----|-------------------|
| DA | Routing temperature (sharpness) |
| NE | Squashing threshold (arousal-gated) |
| ACh | Encoding vs retrieval mode |
| 5-HT | Routing convergence (patience) |

```python
from ww.nca import CapsuleNCACoupling, CapsuleNCACouplingConfig

coupling = CapsuleNCACoupling(CapsuleNCACouplingConfig(
    da_routing_gain=0.5,
    ne_threshold_gain=0.4,
    ach_mode_threshold=0.6,
    serotonin_stability_gain=0.3
))

# NT state modulates capsule behavior
nt_state = {'da': 0.7, 'ne': 0.5, 'ach': 0.8, 'sht': 0.6}
modulated_params = coupling.modulate_routing(capsule_state, nt_state)

# Capsule agreement feeds back to field stability
stability_signal = coupling.compute_stability_feedback(
    routing_agreement=0.85,
    pose_coherence=0.9
)
```

### 3. Glymphatic-Consolidation Bridge

Coordinates waste clearance with memory consolidation during sleep:

```
Sleep State → SWR Events → Replay Selection
                               │
                     ┌─────────┴─────────┐
                     ▼                   ▼
              Consolidate           Glymphatic
              (strengthen)          (clear waste)
```

```python
from ww.nca import GlymphaticConsolidationBridge

bridge = GlymphaticConsolidationBridge(
    glymphatic=glymphatic_system,
    swr_coupling=swr_coupling,
    consolidation_engine=consolidation_engine
)

# During NREM sleep cycle
result = bridge.sleep_cycle_step(
    sleep_state=SleepWakeState.NREM_DEEP,
    delta_phase=0.4  # Up-state
)
```

## Neuromodulator Orchestration

The six-NT system coordinates all regions:

```
           VTA                Raphe              LC              NBM
           (DA)               (5-HT)            (NE)            (ACh)
            │                   │                │                │
            ▼                   ▼                ▼                ▼
    ┌───────────────────────────────────────────────────────────────┐
    │                    NT Orchestration Layer                     │
    │                                                               │
    │  DA: Learning rate, salience                                  │
    │  5-HT: Patience, temporal discounting                         │
    │  NE: Arousal, threshold modulation                            │
    │  ACh: Encoding/retrieval gate                                 │
    │  GABA: Inhibitory control                                     │
    │  Glu: Excitatory drive                                        │
    └───────────────────────────────────────────────────────────────┘
            │                   │                │                │
            ▼                   ▼                ▼                ▼
       Striatum             Energy         Hippocampus        Capsules
       (action)            (attractors)      (memory)         (binding)
```

## State Consistency Checks

H10 validates consistency across regions:

### 1. Energy-Goodness Alignment

```python
def check_energy_goodness_alignment(ff_state, energy_state):
    """Verify FF goodness matches energy landscape."""
    expected_energy = -ff_state.total_goodness
    actual_energy = energy_state.current_energy
    return abs(expected_energy - actual_energy) < tolerance
```

### 2. NT State Coherence

```python
def check_nt_coherence(nt_states: Dict[str, float]):
    """Verify NT levels follow biological constraints."""
    # DA and 5-HT are opponent processes
    assert not (nt_states['da'] > 0.8 and nt_states['sht'] > 0.8)

    # High NE implies arousal
    assert not (nt_states['ne'] > 0.7 and nt_states['ach'] < 0.3)

    return True
```

### 3. Memory-Field Consistency

```python
def check_memory_field_consistency(hpc_state, nca_state):
    """Verify hippocampal output affects field attractors."""
    if hpc_state.novelty_score > 0.5:
        # Novel input should perturb field
        assert nca_state.stability < 0.9
    return True
```

## Credit Flow Integration

Credit assignment propagates across regions:

```
Outcome Signal (reward/error)
         │
         ▼
    ┌─────────┐
    │   VTA   │ → RPE (reward prediction error)
    └────┬────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │         Eligibility Traces          │
    │                                     │
    │  Synapse → Tagged → Waiting → Apply │
    └─────────────────────────────────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │ FF Layer│   │ Capsule │   │ HPC     │
    │ Weights │   │ Routing │   │ Weights │
    └─────────┘   └─────────┘   └─────────┘
```

## Testing

The H10 test suite validates cross-region consistency:

```bash
# Run all H10 tests
pytest tests/integration/test_h10_cross_region_consistency.py -v

# Specific coupling tests
pytest tests/integration/test_h10_cross_region_consistency.py::TestFFNCACoupling -v
pytest tests/integration/test_h10_cross_region_consistency.py::TestCapsuleNCACoupling -v
pytest tests/integration/test_h10_cross_region_consistency.py::TestGlymphaticBridge -v
```

### Test Categories

| Category | Tests | Validates |
|----------|-------|-----------|
| Coupling | 12 | Bidirectional information flow |
| Consistency | 8 | State coherence across regions |
| Credit | 6 | Credit propagation pathways |
| Integration | 10 | Full system behavior |
| Convergence | 5 | Cross-region stability |

## Implementation Files

| File | Purpose |
|------|---------|
| `forward_forward_nca_coupling.py` | FF ↔ Energy coupling |
| `capsule_nca_coupling.py` | Capsule ↔ NCA coupling |
| `glymphatic_consolidation_bridge.py` | Sleep ↔ Memory coupling |
| `dopamine_integration.py` | DA pathway integration |
| `swr_coupling.py` | SWR ↔ Replay coupling |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        World Weaver Architecture                    │
│                                                                     │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐                 │
│  │Forward-   │◄───►│  Energy   │◄───►│ Capsule   │                 │
│  │Forward    │     │ Landscape │     │ Network   │                 │
│  └─────┬─────┘     └─────┬─────┘     └─────┬─────┘                 │
│        │                 │                 │                        │
│        └────────────┬────┴─────┬───────────┘                       │
│                     │          │                                    │
│              ┌──────▼──────────▼──────┐                            │
│              │   NCA Neural Field     │                            │
│              │   (3D PDE Solver)      │                            │
│              └──────────┬─────────────┘                            │
│                         │                                          │
│        ┌────────────────┼────────────────┐                         │
│        ▼                ▼                ▼                         │
│  ┌───────────┐   ┌───────────┐   ┌───────────┐                    │
│  │Hippocampus│   │ Striatum  │   │ Sleep/Wake│                    │
│  │DG→CA3→CA1 │   │ D1/D2 MSN │   │Glymphatic │                    │
│  └───────────┘   └───────────┘   └───────────┘                    │
│        │                │                │                         │
│        └────────────────┼────────────────┘                         │
│                         │                                          │
│              ┌──────────▼───────────┐                              │
│              │   Memory Systems     │                              │
│              │ Episodic | Semantic  │                              │
│              └──────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

## References

1. Hinton (2022). "The Forward-Forward Algorithm"
2. Sabour et al. (2017). "Dynamic Routing Between Capsules"
3. Hasselmo (2006). "The role of acetylcholine in learning and memory"
4. Xie et al. (2013). "Sleep Drives Metabolite Clearance"
5. Ramsauer et al. (2020). "Hopfield Networks is All You Need"
