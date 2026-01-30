# Forward-Forward Algorithm API Reference

**Module**: `ww.nca.forward_forward`, `ww.nca.forward_forward_nca_coupling`
**Version**: Phase 4
**Hinton Score**: 0.8/1.0

---

## Overview

The Forward-Forward (FF) algorithm implements layer-local learning without backpropagation, based on Hinton (2022). Each layer learns independently by maximizing "goodness" for positive data and minimizing it for negative data.

### Key Concepts

- **Goodness**: G(h) = Σh² - measures how well a layer represents its input
- **Positive Phase**: Real data → maximize goodness above threshold θ
- **Negative Phase**: Generated/corrupted data → minimize goodness below θ
- **Energy Alignment**: FF goodness maps inversely to NCA energy landscape

---

## Core Classes

### ForwardForwardLayer

```python
from ww.nca.forward_forward import ForwardForwardLayer

layer = ForwardForwardLayer(
    input_dim=256,
    output_dim=128,
    threshold=0.5,
    learning_rate=0.01,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_dim` | int | required | Input dimension |
| `output_dim` | int | required | Output dimension |
| `threshold` | float | 0.5 | Goodness threshold θ |
| `learning_rate` | float | 0.01 | Base learning rate |
| `activation` | str | "relu" | Activation function |

#### Methods

##### `forward(x: Tensor) -> Tensor`
Forward pass through layer.

```python
h = layer.forward(x)  # Returns activations
```

##### `compute_goodness(h: Tensor) -> float`
Compute layer goodness G(h) = Σh².

```python
goodness = layer.compute_goodness(h)
# Returns: float, sum of squared activations
```

##### `train_step(x_pos: Tensor, x_neg: Tensor) -> dict`
Single training step with positive and negative examples.

```python
metrics = layer.train_step(x_pos, x_neg)
# Returns: {"loss": float, "goodness_pos": float, "goodness_neg": float}
```

##### `update_threshold(delta: float)`
Update goodness threshold θ.

```python
layer.update_threshold(0.1)  # Increase threshold
```

---

### ForwardForwardNetwork

```python
from ww.nca.forward_forward import ForwardForwardNetwork

network = ForwardForwardNetwork(
    layer_dims=[256, 128, 64],
    threshold=0.5,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer_dims` | List[int] | required | Dimensions for each layer |
| `threshold` | float | 0.5 | Global goodness threshold |
| `learning_rate` | float | 0.01 | Learning rate |

#### Methods

##### `forward(x: Tensor) -> List[Tensor]`
Forward pass returning all layer activations.

##### `train_positive(x: Tensor) -> float`
Train on positive (real) data.

##### `train_negative(x: Tensor) -> float`
Train on negative (generated) data.

##### `get_layer_goodness() -> List[float]`
Get goodness values for all layers.

---

### FFNCACoupling

Couples Forward-Forward learning with NCA neural field dynamics.

```python
from ww.nca.forward_forward_nca_coupling import (
    FFNCACoupling,
    FFNCACouplingConfig,
    FFPhase,
)

config = FFNCACouplingConfig(
    da_learning_rate_gain=0.5,
    ach_phase_threshold=0.6,
    ne_threshold_gain=0.4,
)
coupling = FFNCACoupling(config=config)
```

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `da_learning_rate_gain` | float | 0.5 | DA → learning rate multiplier |
| `ach_phase_threshold` | float | 0.6 | ACh level for positive phase |
| `ne_threshold_gain` | float | 0.4 | NE → goodness threshold |
| `serotonin_credit_window` | float | 0.3 | 5-HT → credit decay |
| `goodness_energy_scale` | float | 1.0 | Goodness to energy conversion |
| `enable_energy_alignment` | bool | True | Enable energy landscape alignment |
| `enable_neuromod_gating` | bool | True | Enable NT modulation |

#### Methods

##### `determine_phase(nt_state) -> FFPhase`
Determine FF phase from ACh level.

```python
phase = coupling.determine_phase(nt_state)
# Returns: FFPhase.POSITIVE, FFPhase.NEGATIVE, or FFPhase.INFERENCE
```

##### `compute_learning_rate_multiplier(nt_state) -> float`
Compute DA-modulated learning rate.

```python
lr_mult = coupling.compute_learning_rate_multiplier(nt_state)
# High DA (surprise) → higher multiplier (up to 3x)
# Low DA → lower multiplier (down to 0.1x)
```

##### `compute_goodness_threshold(nt_state) -> float`
Compute NE-modulated goodness threshold.

```python
threshold = coupling.compute_goodness_threshold(nt_state)
# High NE (arousal) → higher threshold (more selective)
```

##### `align_with_energy(goodness: float, energy: float) -> EnergyAlignment`
Determine alignment between FF goodness and NCA energy.

```python
alignment = coupling.align_with_energy(goodness, energy)
# Returns: EnergyAlignment.BASIN, BARRIER, SADDLE, or TRANSITION
```

---

## Enumerations

### FFPhase

```python
from ww.nca.forward_forward_nca_coupling import FFPhase

FFPhase.POSITIVE   # Real data: maximize goodness
FFPhase.NEGATIVE   # Fake data: minimize goodness
FFPhase.INFERENCE  # No learning, just forward pass
```

### EnergyAlignment

```python
from ww.nca.forward_forward_nca_coupling import EnergyAlignment

EnergyAlignment.BASIN      # In attractor basin (high goodness, low energy)
EnergyAlignment.BARRIER    # At energy barrier (threshold goodness)
EnergyAlignment.SADDLE     # Saddle point (unstable)
EnergyAlignment.TRANSITION # Transitioning between attractors
```

---

## Usage Examples

### Basic Training

```python
from ww.nca.forward_forward import ForwardForwardNetwork

# Create network
net = ForwardForwardNetwork(layer_dims=[784, 500, 500])

# Training loop
for x_real in dataloader:
    # Generate negative examples (e.g., random noise or corrupted)
    x_neg = generate_negative(x_real)

    # Train both phases
    loss_pos = net.train_positive(x_real)
    loss_neg = net.train_negative(x_neg)
```

### NT-Modulated Learning

```python
from ww.nca.forward_forward_nca_coupling import FFNCACoupling
from ww.nca.neural_field import NeurotransmitterState

coupling = FFNCACoupling()
nt_state = NeurotransmitterState(
    dopamine=0.8,      # High DA → faster learning
    acetylcholine=0.7, # High ACh → positive phase
    norepinephrine=0.6 # Moderate NE → moderate threshold
)

# Get modulated parameters
phase = coupling.determine_phase(nt_state)
lr_mult = coupling.compute_learning_rate_multiplier(nt_state)
threshold = coupling.compute_goodness_threshold(nt_state)

# Apply to FF layer
layer.learning_rate = base_lr * lr_mult
layer.threshold = threshold
```

### Sleep-Phase Negative Generation

```python
# During NREM sleep, generate negative examples for memory pruning
if sleep_stage == SleepStage.NREM_DEEP:
    # Replay memories as positive
    net.train_positive(replay_memory)

    # Generate confabulated versions as negative
    x_neg = generate_sleep_negative(replay_memory)
    net.train_negative(x_neg)
```

---

## Integration Points

### With NCA Energy Landscape

```python
from ww.nca.energy import EnergyLandscape

energy = EnergyLandscape()
ff_coupling = FFNCACoupling()

# Get goodness from FF layer
goodness = layer.compute_goodness(h)

# Get energy from NCA field
e = energy.compute_energy(nt_state)

# Check alignment
alignment = ff_coupling.align_with_energy(goodness, e)
if alignment == EnergyAlignment.BASIN:
    # Stable memory state
    pass
```

### With Telemetry Hub

```python
from ww.visualization.telemetry_hub import TelemetryHub
from ww.visualization.ff_visualizer import ForwardForwardVisualizer

hub = TelemetryHub()
ff_viz = ForwardForwardVisualizer()
hub.register_ff(ff_viz)

# Record FF state during training
hub.record_state(ff_state=layer.get_state())
```

---

## Biological Rationale

### Hinton (2022) Alignment

| Concept | Implementation | Fidelity |
|---------|---------------|----------|
| Layer-local learning | Each layer trains independently | High |
| Goodness function | G(h) = Σh² | Exact |
| Threshold learning | θ adapts to maintain separation | High |
| Sleep negative | NREM generates negative examples | Moderate |

### Neuromodulator Mapping

| NT | FF Effect | Biological Basis |
|----|-----------|------------------|
| DA | Learning rate | Schultz (1998) RPE |
| ACh | Phase selection | Hasselmo encoding/retrieval |
| NE | Threshold | Aston-Jones arousal |
| 5-HT | Credit window | Daw temporal discounting |

---

## Phase 4 Enhancements

### Adaptive Threshold (Future)

The ww-hinton agent (H1 analysis, 0.85/1.0) identified that adaptive threshold θ is a key gap:

```python
# Current: Static threshold
layer.threshold = 0.5

# Future: Adaptive threshold (Hinton recommendation)
layer.enable_adaptive_threshold(
    margin=0.1,           # Min separation margin
    update_rate=0.01,     # Threshold learning rate
    goodness_target=1.0,  # Target goodness for positive
)
```

### Sleep Integration

During NREM, the FF system integrates with consolidation:

1. **Positive phase**: Replay of recent memories
2. **Negative phase**: Confabulated/corrupted memories
3. **Clearance**: Low-goodness patterns pruned via glymphatic

```python
from ww.nca.forward_forward_nca_coupling import FFNCACoupling
from ww.nca.glymphatic_consolidation_bridge import GlymphaticConsolidationBridge

ff_coupling = FFNCACoupling()
glymphatic = GlymphaticConsolidationBridge()

# During NREM
for memory in replay_buffer:
    # Train FF on replay (positive)
    goodness = ff_layer.train_positive(memory.embedding)

    # Generate negative (corrupted version)
    neg = corrupt_memory(memory.embedding)
    ff_layer.train_negative(neg)

    # Low goodness → candidate for glymphatic clearance
    if goodness < ff_coupling.config.min_goodness_threshold:
        glymphatic.mark_for_clearance(memory.id)
```

### Cross-Scale Telemetry Events

```python
from ww.visualization.telemetry_hub import TelemetryHub

hub = TelemetryHub()

# FF-specific events detected automatically
# - "sleep_negative_generation": NREM negative phase
# - "ff_goodness_drop": Sudden goodness decrease
# - "ff_threshold_adaptation": θ updated
```

---

## See Also

- [Capsule API](capsule-api.md) - Capsule network integration
- [Glymphatic API](glymphatic-api.md) - Waste clearance
- [NCA API](nca-api.md) - Neural field dynamics
- [Learning Theory](../science/learning-theory.md) - Theoretical foundations
- [Forward-Forward Concept](../concepts/forward-forward.md) - Conceptual overview
