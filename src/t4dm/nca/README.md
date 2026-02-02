# Neural Cognitive Architecture (NCA) Module

**30 files | ~12,000 lines | Centrality: 5**

The NCA module implements a biologically-grounded neural dynamics system with learnable neurotransmitter coupling, Hinton-inspired learning algorithms, and realistic biological subsystems.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        NEURAL COGNITIVE ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 1: Core Dynamics                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Neural Field │  │   Coupling   │  │  Attractors  │  │    Energy    │ │
│  │  PDE Solver  │  │   K[6x6]     │  │   5 States   │  │  Landscape   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 2: Neuromodulators                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────────┐ │
│  │ VTA (DA) │  │Raphe(5HT)│  │  LC (NE) │  │  Dopamine Integration    │ │
│  │  RPE/TD  │  │ Patience │  │ Arousal  │  │  VTA + Field + Memory    │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 3: Biological Subsystems                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │ Hippocampus│  │ Oscillators│  │ Glymphatic │  │ Sleep Spindles     │ │
│  │ DG→CA3→CA1 │  │ θ/γ/α/β/δ  │  │  Clearance │  │ 11-16 Hz bursts    │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────┘ │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │  Adenosine │  │ Astrocytes │  │  Striatum  │  │ Spatial Cells      │ │
│  │  Process S │  │ Glu/GABA   │  │  D1/D2 MSN │  │ Place/Grid         │ │
│  └────────────┘  └────────────┘  └────────────┘  └────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────┤
│  LAYER 4: Advanced Learning (Hinton)                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐ │
│  │ Forward-Forward│  │Capsule Networks│  │  NCA Integration Bridges   │ │
│  │  Layer-local   │  │ Routing-by-    │  │  FF↔Energy, Caps↔Field     │ │
│  │  No backprop   │  │ Agreement      │  │  Glymphatic↔Consolidation  │ │
│  └────────────────┘  └────────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

### Core Dynamics (4 files)

| File | Lines | Purpose |
|------|-------|---------|
| `neural_field.py` | ~800 | 6-NT PDE solver with semi-implicit Euler |
| `coupling.py` | ~600 | Learnable 6x6 coupling matrix K |
| `attractors.py` | ~400 | 5 cognitive state basins (ALERT, FOCUS, REST, EXPLORE, CONSOLIDATE) |
| `energy.py` | ~700 | Hopfield energy + contrastive divergence |

### Neuromodulators (4 files)

| File | Lines | Purpose |
|------|-------|---------|
| `vta.py` | ~500 | VTA dopamine circuit with TD learning |
| `raphe.py` | ~400 | Raphe serotonin with autoreceptor feedback |
| `locus_coeruleus.py` | ~450 | LC norepinephrine with Yerkes-Dodson |
| `dopamine_integration.py` | ~350 | Cross-system DA unification |

### Biological Subsystems (11 files)

| File | Lines | Purpose |
|------|-------|---------|
| `hippocampus.py` | ~600 | DG pattern separation, CA3 Hopfield, CA1 novelty |
| `oscillators.py` | ~500 | Multi-band frequency generator with PAC |
| `adenosine.py` | ~400 | Sleep pressure (Borbely Process S) |
| `glymphatic.py` | ~450 | Brain waste clearance (Xie 2013) |
| `sleep_spindles.py` | ~350 | Thalamocortical 11-16 Hz bursts |
| `swr_coupling.py` | ~400 | Sharp-wave ripple to neural field |
| `striatal_msn.py` | ~450 | D1/D2 MSN action selection |
| `astrocyte.py` | ~400 | Tripartite synapse, gliotransmission |
| `glutamate_signaling.py` | ~350 | Synaptic vs extrasynaptic NMDA |
| `theta_gamma_integration.py` | ~400 | Working memory binding |
| `spatial_cells.py` | ~350 | Place cells and grid cells |

### Advanced Learning (4 files)

| File | Lines | Purpose |
|------|-------|---------|
| `forward_forward.py` | ~600 | Hinton 2022 FF algorithm |
| `capsules.py` | ~700 | Capsule networks with routing |
| `pose.py` | ~300 | 4x4 semantic pose matrices |
| `connectome.py` | ~400 | Anatomical brain connectivity |

### Integration Bridges (4 files)

| File | Lines | Purpose |
|------|-------|---------|
| `forward_forward_nca_coupling.py` | ~350 | FF goodness ↔ energy landscape |
| `capsule_nca_coupling.py` | ~350 | Capsule routing ↔ NT modulation |
| `glymphatic_consolidation_bridge.py` | ~400 | Clearance ↔ consolidation timing |
| `striatal_coupling.py` | ~300 | DA-ACh traveling waves |

### Support (3 files)

| File | Lines | Purpose |
|------|-------|---------|
| `delays.py` | ~350 | Axonal + synaptic transmission delays |
| `stability.py` | ~300 | Hessian/Jacobian stability analysis |
| `__init__.py` | ~100 | Public API exports |

## Core Equation

The neural field evolves according to:

```
∂U/∂t = -αU + D∇²U + S(x,t) + C(U₁...Uₙ) + Attractor + Astrocyte
```

Where:
- **U**: 6-dimensional NT state vector [DA, 5-HT, ACh, NE, GABA, Glu]
- **α**: Decay rates (biologically calibrated per NT)
- **D**: Diffusion coefficients
- **S**: External stimuli
- **C**: Coupling function via learnable K matrix
- **Attractor**: Pull toward cognitive state basins
- **Astrocyte**: Gliotransmitter contributions

## Cognitive States

| State | NT Signature | Function |
|-------|-------------|----------|
| ALERT | High DA/NE, low GABA | Vigilance |
| FOCUS | High ACh/Glu, moderate DA | Sustained attention |
| REST | High 5-HT/GABA, low NE | Default mode |
| EXPLORE | High DA/NE/ACh | Novelty seeking |
| CONSOLIDATE | High GABA/5-HT, low Glu | Memory consolidation |

## Hinton Algorithm Implementations

### Forward-Forward (Hinton 2022)

Layer-local learning without backpropagation:

```python
# Goodness function per layer
G(h) = Σh_i² / layer_size

# Learning rule
if G > θ:  # Positive phase (real data)
    W += α(h_in ⊗ h_out)
else:       # Negative phase (corrupted)
    W -= α(h_in ⊗ h_out)
```

### Capsule Networks

Routing-by-agreement with semantic pose matrices:

```python
# 4 semantic dimensions
TEMPORAL, CAUSAL, SEMANTIC_ROLE, CERTAINTY

# Routing coefficient update
c_ij = softmax(agreement(pose_i, pose_j))
```

### Energy-Based Learning

Modern Hopfield with contrastive divergence:

```python
E_total = E_hopfield + E_boundary + E_attractor
# E_hopfield = -0.5 * U^T K U
# E_boundary = λ[log(U) + log(1-U)]
# E_attractor = -Σ w_s exp(-||U - c_s||²/2σ_s²)
```

## Biological Parameters

| System | Parameter | Value | Citation |
|--------|-----------|-------|----------|
| VTA | Tonic firing | 4-5 Hz | Schultz 1998 |
| VTA | Phasic burst | 20-40 Hz | Schultz 1998 |
| Raphe | Baseline 5-HT | 2.5 Hz | Doya 2002 |
| LC | Optimal arousal | 3 Hz | Aston-Jones 2005 |
| Theta | Frequency | 4-8 Hz | Buzsaki 2006 |
| Gamma | Frequency | 30-100 Hz | Lisman 2013 |
| SWR | Ripple frequency | 150-250 Hz | Buzsaki 2015 |
| Spindles | Frequency | 11-16 Hz | Steriade 1993 |
| DG | Sparsity | 4% | Treves 1994 |
| Glymphatic | NREM clearance | 70% | Xie 2013 |

## Usage

### Basic Neural Field

```python
from t4dm.nca import NeuralFieldConfig, NeuralFieldSolver, NeurotransmitterState

config = NeuralFieldConfig()
solver = NeuralFieldSolver(config)

# Set initial state
state = NeurotransmitterState(
    dopamine=0.5,
    serotonin=0.4,
    acetylcholine=0.6,
    norepinephrine=0.3,
    gaba=0.4,
    glutamate=0.5
)

# Step dynamics
new_state = solver.step(state, dt=0.01)
```

### Neuromodulator Integration

```python
from t4dm.nca import VTACircuit, RapheNucleus, LocusCoeruleus, DopamineIntegration

vta = VTACircuit()
raphe = RapheNucleus()
lc = LocusCoeruleus()

# Compute RPE
rpe = vta.compute_rpe(actual_reward=1.0, expected_reward=0.3)

# Integrate DA signals
integration = DopamineIntegration()
integrated_state = integration.integrate(vta_state, field_state, memory_rpe)
```

### Forward-Forward Learning

```python
from t4dm.nca import ForwardForwardNetwork, FFPhase

ff_net = ForwardForwardNetwork(layer_dims=[1024, 512, 256])

# Positive phase (real data)
pos_goodness = ff_net.forward(real_data, phase=FFPhase.POSITIVE)

# Negative phase (corrupted data)
neg_goodness = ff_net.forward(corrupted_data, phase=FFPhase.NEGATIVE)

# Update weights
ff_net.update_weights()
```

### Hippocampal Processing

```python
from t4dm.nca import HippocampalCircuit

hc = HippocampalCircuit()

# Pattern separation via DG
sparse_pattern = hc.dentate_gyrus(input_pattern)

# Pattern completion via CA3
completed = hc.ca3_completion(partial_cue)

# Novelty detection via CA1
novelty = hc.ca1_novelty(ec_input, ca3_output)
```

## Integration Points

### With Learning Module
```
VTA.rpe → Three-Factor Learning.surprise
Raphe.5-HT → Eligibility.temporal_credit
LC.NE → Neuromod Orchestra.arousal
```

### With Memory Module
```
Hippocampus.novelty → EpisodicMemory.encoding_gate
Oscillators.theta → WorkingMemory.binding
SWR.replay → Consolidation.NREM
```

### With Consolidation
```
Glymphatic → Sleep.waste_clearance
Spindles → Sleep.memory_gate
Adenosine → Sleep.pressure
```

## Testing

```bash
# Run NCA tests
pytest tests/nca/ -v

# Run with biology benchmarks
pytest tests/nca/ -v -m biology

# Coverage report
pytest tests/nca/ --cov=t4dm.nca --cov-report=term-missing
```

**Test Coverage**: 664 tests, 79% coverage

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Field step | O(n²) | n = grid points |
| Hopfield retrieval | O(pattern_count) | Modern: exponential capacity |
| FF forward | O(layer_dims) | Layer-local, parallelizable |
| Capsule routing | O(iterations × capsules²) | 3 iterations typical |
| Stability analysis | O(dim³) | Eigenvalue computation |

## References

- Hinton (2022) "The Forward-Forward Algorithm"
- Sabour et al. (2017) "Dynamic Routing Between Capsules"
- Schultz (1998) "Dopamine Reward Prediction Error"
- Buzsaki (2006) "Rhythms of the Brain"
- Xie et al. (2013) "Sleep Drives Metabolite Clearance"
- Doya (2002) "Metalearning and Neuromodulation"
