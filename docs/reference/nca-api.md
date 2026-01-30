# NCA Module API Reference

Complete API reference for the Neural Cellular Automata (NCA) module.

## Module Structure

```
ww/nca/
├── Core
│   ├── neural_field.py      # 3D PDE solver
│   ├── coupling.py          # Learnable coupling matrix
│   ├── energy.py            # Hopfield integration
│   └── attractors.py        # State transitions
├── Neuromodulatory
│   ├── vta.py               # DA production
│   ├── raphe.py             # 5-HT + Patience
│   ├── locus_coeruleus.py   # NE + Surprise
│   ├── dopamine_integration.py
│   └── striatal_msn.py      # D1/D2 populations
├── Hippocampal
│   ├── hippocampus.py       # DG→CA3→CA1
│   ├── spatial_cells.py     # Place/Grid cells
│   └── theta_gamma_integration.py
├── Learning
│   ├── forward_forward.py   # FF algorithm
│   ├── capsules.py          # Dynamic routing
│   └── pose.py              # Pose matrices
├── Sleep/Wake
│   ├── oscillators.py       # Frequency bands
│   ├── sleep_spindles.py    # Spindle-delta
│   ├── adenosine.py         # Sleep pressure
│   ├── swr_coupling.py      # Sharp-wave ripples
│   └── glymphatic.py        # Waste clearance
├── Glial
│   ├── astrocyte.py         # Tripartite synapse
│   └── glutamate_signaling.py
└── Coupling Bridges
    ├── forward_forward_nca_coupling.py
    ├── capsule_nca_coupling.py
    ├── glymphatic_consolidation_bridge.py
    └── striatal_coupling.py
```

## Core Components

### NeuralFieldSolver

3D spatiotemporal PDE solver for neurotransmitter dynamics.

```python
from ww.nca import NeuralFieldSolver, NeuralFieldConfig, NeurotransmitterState

config = NeuralFieldConfig(
    grid_size=(32, 32, 32),
    dt=0.001,
    diffusion_coeff=0.1,
    decay_rate=0.05
)

solver = NeuralFieldSolver(config)

# Initialize NT state
nt_state = NeurotransmitterState(
    da=0.5, sht=0.5, ach=0.5, ne=0.5, gaba=0.5, glu=0.5
)

# Step simulation
new_state = solver.step(nt_state, external_input=stimulus)
```

### EnergyLandscape

Hopfield-inspired energy function with contrastive learning.

```python
from ww.nca import EnergyLandscape, EnergyConfig, HopfieldIntegration

config = EnergyConfig(
    beta=8.0,           # Inverse temperature
    max_patterns=1000,  # Memory capacity
    cd_k=1              # Contrastive divergence steps
)

energy = EnergyLandscape(config)

# Compute energy for state
E = energy.compute_energy(state_vector)

# Store pattern
energy.store_pattern(memory_embedding)

# Retrieve via attractor
retrieved = energy.relax_to_equilibrium(query, max_steps=10)
```

### AttractorBasin

Cognitive state attractor management.

```python
from ww.nca import AttractorBasin, CognitiveState, StateTransitionManager

# Define cognitive states
states = [
    CognitiveState.EXPLORATION,
    CognitiveState.EXPLOITATION,
    CognitiveState.CONSOLIDATION
]

manager = StateTransitionManager(states)

# Transition based on signals
manager.update(da_level=0.8, uncertainty=0.3)
current = manager.get_current_state()
```

## Neuromodulatory Systems

### VTACircuit

Dopamine production from reward prediction error.

```python
from ww.nca import VTACircuit, VTAConfig, create_vta_circuit

config = VTAConfig(
    tonic_rate=4.5,       # Hz (baseline)
    burst_peak_rate=30.0,  # Hz (max burst)
    da_decay_tau=0.3       # seconds
)

vta = create_vta_circuit(config)

# Compute RPE-driven DA
da_output = vta.compute_da_release(
    predicted_reward=0.5,
    actual_reward=0.8  # Positive surprise
)
```

### RapheNucleus

Serotonin production with patience model.

```python
from ww.nca import RapheNucleus, PatienceModel, create_raphe_nucleus

raphe = create_raphe_nucleus(RapheConfig(
    baseline_rate=2.0,  # Hz
    sht_release=10.0    # nM
))

# Patience model for temporal discounting
patience = PatienceModel(PatienceConfig(
    baseline_patience=0.7,
    sht_sensitivity=0.5
))

discount_factor = patience.compute_discount(delay=5.0, sht_level=0.6)
```

### LocusCoeruleus

Norepinephrine with surprise model.

```python
from ww.nca import LocusCoeruleus, SurpriseModel, create_locus_coeruleus

lc = create_locus_coeruleus(LCConfig(
    tonic_rate=2.0,    # Hz
    phasic_rate=15.0   # Hz (burst)
))

# Surprise model for uncertainty signaling
surprise = SurpriseModel(SurpriseConfig(
    entropy_sensitivity=0.5
))

ne_level = surprise.compute_surprise_signal(
    prediction_entropy=1.5,
    actual_entropy=2.5
)
```

## Hippocampal System

### HippocampalCircuit

DG→CA3→CA1 circuit for memory encoding/retrieval.

```python
from ww.nca import (
    HippocampalCircuit, HippocampalConfig, HippocampalMode,
    create_hippocampal_circuit
)

config = HippocampalConfig(
    ec_dim=1024,
    dg_dim=4096,
    ca3_dim=1024,
    ca1_dim=1024,
    dg_sparsity=0.04,
    ca3_beta=8.0
)

hpc = create_hippocampal_circuit(config)

# Process input
state = hpc.process(
    ec_input=embedding,
    mode=HippocampalMode.AUTOMATIC
)

print(f"Novelty: {state.novelty_score}")
print(f"Mode: {'encoding' if state.novelty_score > 0.5 else 'retrieval'}")
```

### SpatialCellSystem

Place cells and grid modules.

```python
from ww.nca import SpatialCellSystem, SpatialConfig, Position2D

config = SpatialConfig(
    n_place_cells=256,
    n_grid_modules=3,
    place_field_width=0.1
)

spatial = SpatialCellSystem(config)

# Encode position
position = Position2D(x=0.5, y=0.3)
spatial_code = spatial.encode(position)
```

## Learning Systems

### ForwardForwardNetwork

Hinton (2022) Forward-Forward algorithm.

```python
from ww.nca import (
    ForwardForwardNetwork, ForwardForwardConfig,
    create_ff_network, FFPhase
)

config = ForwardForwardConfig(
    layer_sizes=[1024, 512, 256],
    threshold_theta=2.0,
    learning_rate=0.01,
    negative_method="hybrid"
)

ff = create_ff_network(config)

# Training
ff.train_step(
    positive_data=real_sample,
    negative_data=synthetic_sample
)

# Inference
goodness = ff.forward(test_input, phase=FFPhase.INFERENCE)
```

### CapsuleNetwork

Dynamic routing for part-whole relationships.

```python
from ww.nca import (
    CapsuleNetwork, CapsuleConfig, RoutingType,
    create_capsule_network
)

config = CapsuleConfig(
    input_dim=1024,
    n_primary_caps=32,
    primary_cap_dim=8,
    n_output_caps=16,
    output_cap_dim=16,
    routing_iterations=3,
    routing_type=RoutingType.DYNAMIC
)

caps = create_capsule_network(config)

# Encode with routing
capsules = caps.encode(input_vector)

# Get entity probabilities (lengths)
probs = caps.get_entity_probabilities()

# Get poses (directions)
poses = caps.get_poses()
```

## Sleep/Wake Systems

### AdenosineDynamics

Sleep pressure accumulation and clearance.

```python
from ww.nca import AdenosineDynamics, AdenosineConfig, SleepWakeState

adenosine = AdenosineDynamics(AdenosineConfig(
    accumulation_rate=0.1,
    decay_rate_wake=0.01,
    decay_rate_sleep=0.1,
    sleep_threshold=0.8
))

# Simulate wake period
for _ in range(1000):
    adenosine.step(is_awake=True)

state = adenosine.get_state()
print(f"Sleep pressure: {state.adenosine_level}")
print(f"Recommended: {state.sleep_recommendation}")
```

### SWRNeuralFieldCoupling

Sharp-wave ripple generation and replay.

```python
from ww.nca import SWRNeuralFieldCoupling, SWRConfig, create_swr_coupling

swr = create_swr_coupling(SWRConfig(
    ripple_freq_min=80.0,   # Hz
    ripple_freq_max=120.0,  # Hz
    replay_compression=20.0  # 20x speedup
))

# During sleep
events = swr.generate_events(
    memory_buffer=recent_memories,
    field_state=nca_state
)

for event in events:
    print(f"Replay: {event.memory_id} at {event.ripple_frequency}Hz")
```

### GlymphaticSystem

Sleep-gated waste clearance.

```python
from ww.nca import GlymphaticSystem, GlymphaticConfig, create_glymphatic_system

glymphatic = create_glymphatic_system(GlymphaticConfig(
    clearance_nrem_deep=0.7,  # Xie et al. 2013
    unused_embedding_days=30,
    max_clearance_fraction=0.1
))

# During NREM
waste = glymphatic.scan_for_waste(memory_store)
cleared = glymphatic.process_clearance_batch(
    sleep_state=WakeSleepMode.NREM_DEEP
)
```

## Coupling Bridges

### FFNCACoupling

Forward-Forward ↔ Energy landscape coupling.

```python
from ww.nca import FFNCACoupling, FFNCACouplingConfig

coupling = FFNCACoupling(FFNCACouplingConfig(
    goodness_to_energy_scale=1.0,
    energy_to_goodness_scale=1.0
))

# Bidirectional update
energy_update = coupling.ff_to_nca(ff_state)
ff_feedback = coupling.nca_to_ff(energy_state)
```

### CapsuleNCACoupling

Capsule ↔ NCA coupling with NT modulation.

```python
from ww.nca import CapsuleNCACoupling, CapsuleNCACouplingConfig

coupling = CapsuleNCACoupling(CapsuleNCACouplingConfig(
    da_routing_gain=0.5,
    ne_threshold_gain=0.4,
    ach_mode_threshold=0.6
))

# Modulate capsules with NT state
params = coupling.modulate_routing(capsules, nt_state)
```

## Glial Systems

### AstrocyteLayer

Tripartite synapse modeling.

```python
from ww.nca import AstrocyteLayer, AstrocyteConfig, compute_tripartite_synapse

astrocyte = AstrocyteLayer(AstrocyteConfig(
    n_astrocytes=100,
    glutamate_uptake_tau=0.01,
    calcium_dynamics_tau=0.5
))

# Modulate synapse
synaptic_output = compute_tripartite_synapse(
    presynaptic=pre_activity,
    postsynaptic=post_activity,
    astrocyte_state=astrocyte.get_state()
)
```

## Factory Functions

All major components have `create_*` factory functions:

```python
from ww.nca import (
    create_vta_circuit,
    create_raphe_nucleus,
    create_locus_coeruleus,
    create_striatal_msn,
    create_hippocampal_circuit,
    create_swr_coupling,
    create_glymphatic_system,
    create_ff_layer,
    create_ff_network,
    create_capsule_layer,
    create_capsule_network,
    create_glutamate_signaling,
    create_dopamine_integration,
    create_default_connectome,
    create_minimal_connectome,
)
```

## Type Exports

All public types are exported from `ww.nca`:

```python
from ww.nca import (
    # Configs
    NeuralFieldConfig, CouplingConfig, EnergyConfig,
    VTAConfig, RapheConfig, LCConfig, MSNConfig,
    HippocampalConfig, SpatialConfig,
    ForwardForwardConfig, CapsuleConfig, PoseConfig,
    AdenosineConfig, SWRConfig, SpindleConfig, GlymphaticConfig,

    # States
    NeurotransmitterState, VTAState, RapheState, LCState,
    HippocampalState, SpatialState, ForwardForwardState,
    CapsuleState, PoseState, AdenosineState, SWREvent,

    # Enums
    CognitiveState, VTAFiringMode, LCFiringMode, HippocampalMode,
    FFPhase, RoutingType, SquashType, WakeSleepMode, SWRPhase,
)
```

## See Also

- [NCA Concepts](../concepts/nca.md) - Conceptual overview
- [Forward-Forward](../concepts/forward-forward.md) - FF algorithm details
- [Capsules](../concepts/capsules.md) - Capsule network concepts
- [Glymphatic](../concepts/glymphatic.md) - Sleep clearance
- [Cross-Region](../concepts/cross-region-integration.md) - H10 integration
