# Capsule Networks API Reference

**Module**: `ww.nca.capsules`, `ww.nca.capsule_nca_coupling`, `ww.nca.pose`
**Version**: Phase 4
**Hinton Score**: 0.7/1.0

---

## Overview

Capsule networks implement hierarchical part-whole relationships through pose vectors and dynamic routing. Based on Sabour et al. (2017) and Hinton et al. (2018), with NT-modulated routing via ACh/DA coupling.

### Key Concepts

- **Capsule**: Group of neurons representing an entity with pose and probability
- **Pose Vector**: Encodes entity properties (position, orientation, scale, etc.)
- **Dynamic Routing**: Iterative agreement-based routing between layers
- **Routing Coefficients**: Learned attention weights between parent-child capsules

---

## Core Classes

### Capsule

```python
from ww.nca.capsules import Capsule

capsule = Capsule(
    pose_dim=16,
    num_capsules=32,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pose_dim` | int | 16 | Dimension of pose vector |
| `num_capsules` | int | 32 | Number of capsules in layer |
| `squash_eps` | float | 1e-7 | Epsilon for squashing |

#### Methods

##### `squash(v: Tensor) -> Tensor`
Squashing function to normalize pose vectors.

```python
squashed = capsule.squash(v)
# ||squashed|| ∈ [0, 1), represents activation probability
```

##### `get_activation_probability() -> Tensor`
Get probability that entity exists.

```python
prob = capsule.get_activation_probability()
# Returns: Tensor of shape (batch, num_capsules)
```

##### `get_pose_vector() -> Tensor`
Get pose vectors for all capsules.

```python
pose = capsule.get_pose_vector()
# Returns: Tensor of shape (batch, num_capsules, pose_dim)
```

---

### CapsuleLayer

```python
from ww.nca.capsules import CapsuleLayer

layer = CapsuleLayer(
    in_capsules=32,
    out_capsules=16,
    in_pose_dim=16,
    out_pose_dim=16,
    routing_iterations=3,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `in_capsules` | int | required | Number of input capsules |
| `out_capsules` | int | required | Number of output capsules |
| `in_pose_dim` | int | 16 | Input pose dimension |
| `out_pose_dim` | int | 16 | Output pose dimension |
| `routing_iterations` | int | 3 | Dynamic routing iterations |
| `routing_temperature` | float | 1.0 | Softmax temperature |

#### Methods

##### `forward(u: Tensor) -> Tensor`
Forward pass with dynamic routing.

```python
v = layer.forward(u)
# u: (batch, in_capsules, in_pose_dim)
# v: (batch, out_capsules, out_pose_dim)
```

##### `route(u: Tensor, iterations: int) -> Tuple[Tensor, Tensor]`
Perform dynamic routing.

```python
v, c = layer.route(u, iterations=3)
# v: output capsules
# c: routing coefficients (batch, in_caps, out_caps)
```

##### `get_routing_agreement() -> float`
Get mean routing agreement (convergence metric).

```python
agreement = layer.get_routing_agreement()
# High agreement → stable routing → converged
```

---

### PoseTransform

```python
from ww.nca.pose import PoseTransform

transform = PoseTransform(
    pose_dim=16,
    num_transforms=8,
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pose_dim` | int | 16 | Pose vector dimension |
| `num_transforms` | int | 8 | Number of transformation matrices |
| `learnable` | bool | True | Whether transforms are learnable |

#### Methods

##### `apply(pose: Tensor, transform_id: int) -> Tensor`
Apply geometric transformation to pose.

```python
new_pose = transform.apply(pose, transform_id=0)
```

##### `compose(transforms: List[int]) -> Tensor`
Compose multiple transformations.

```python
composed = transform.compose([0, 2, 5])
```

---

### CapsuleNCACoupling

Couples capsule networks with NCA neural field dynamics.

```python
from ww.nca.capsule_nca_coupling import (
    CapsuleNCACoupling,
    CapsuleNCACouplingConfig,
    CapsuleMode,
)

config = CapsuleNCACouplingConfig(
    da_routing_gain=0.5,
    ach_mode_threshold=0.6,
    ne_threshold_gain=0.4,
)
coupling = CapsuleNCACoupling(config=config)
```

#### Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `da_routing_gain` | float | 0.5 | DA → routing temperature |
| `ne_threshold_gain` | float | 0.4 | NE → squashing threshold |
| `ach_mode_threshold` | float | 0.6 | ACh level for encoding mode |
| `serotonin_stability_gain` | float | 0.3 | 5-HT → routing stability |
| `agreement_to_stability` | float | 0.4 | Routing agreement → field stability |
| `coupling_strength` | CouplingStrength | MODERATE | Overall coupling strength |
| `bidirectional` | bool | True | Enable both directions |

#### Methods

##### `determine_mode(nt_state) -> CapsuleMode`
Determine capsule operating mode from ACh level.

```python
mode = coupling.determine_mode(nt_state)
# ENCODING: High ACh → sharp routing, store patterns
# RETRIEVAL: Low ACh → soft routing, pattern completion
# NEUTRAL: Mid ACh → balanced
```

##### `compute_routing_temperature(nt_state) -> float`
Compute DA/mode-modulated routing temperature.

```python
temp = coupling.compute_routing_temperature(nt_state)
# Low temp → sharp routing (high DA, encoding)
# High temp → soft routing (low DA, retrieval)
```

##### `compute_squashing_threshold(nt_state) -> float`
Compute NE-modulated squashing threshold.

```python
threshold = coupling.compute_squashing_threshold(nt_state)
# High NE → higher threshold → more selective
```

##### `compute_field_feedback(layer: CapsuleLayer) -> dict`
Compute capsule → NCA feedback signals.

```python
feedback = coupling.compute_field_feedback(layer)
# Returns: {"stability": float, "pose_coherence": float, "activation": float}
```

---

## Enumerations

### CapsuleMode

```python
from ww.nca.capsule_nca_coupling import CapsuleMode

CapsuleMode.ENCODING   # High ACh: sharp routing, store patterns
CapsuleMode.RETRIEVAL  # Low ACh: soft routing, pattern completion
CapsuleMode.NEUTRAL    # Mid ACh: balanced operation
```

### CouplingStrength

```python
from ww.nca.capsule_nca_coupling import CouplingStrength

CouplingStrength.WEAK     # 0.3
CouplingStrength.MODERATE # 0.5
CouplingStrength.STRONG   # 0.7
CouplingStrength.FULL     # 1.0
```

---

## Usage Examples

### Basic Capsule Network

```python
from ww.nca.capsules import CapsuleLayer, PrimaryCapsules

# Primary capsules from conv features
primary = PrimaryCapsules(
    in_channels=256,
    out_capsules=32,
    pose_dim=8,
)

# Capsule layer with routing
digit_caps = CapsuleLayer(
    in_capsules=32 * 6 * 6,
    out_capsules=10,
    in_pose_dim=8,
    out_pose_dim=16,
    routing_iterations=3,
)

# Forward pass
u = primary(conv_features)
v = digit_caps(u)
```

### NT-Modulated Routing

```python
from ww.nca.capsule_nca_coupling import CapsuleNCACoupling
from ww.nca.neural_field import NeurotransmitterState

coupling = CapsuleNCACoupling()

# High ACh encoding scenario
nt_encoding = NeurotransmitterState(
    acetylcholine=0.8,  # High ACh → ENCODING mode
    dopamine=0.7,       # High DA → sharp routing
)

mode = coupling.determine_mode(nt_encoding)
temp = coupling.compute_routing_temperature(nt_encoding)

# Apply to capsule layer
digit_caps.routing_temperature = temp
```

### Part-Whole Hierarchy

```python
from ww.nca.pose import PoseTransform

# Define part-whole transformations
transform = PoseTransform(pose_dim=16, num_transforms=4)

# Get pose of whole from parts
part_poses = [capsule.get_pose_vector() for capsule in parts]
whole_pose = aggregate_poses(part_poses, transform)
```

---

## Integration Points

### With Forward-Forward

```python
from ww.nca.forward_forward_nca_coupling import FFNCACoupling

# Capsule activations can provide goodness signal
capsule_goodness = capsule.get_activation_probability().sum()

# FF can validate capsule representations
ff_coupling.align_with_energy(capsule_goodness, energy)
```

### With NCA Neural Field

```python
from ww.nca.neural_field import NeuralField

field = NeuralField()
coupling = CapsuleNCACoupling()

# Capsule routing agreement affects field stability
feedback = coupling.compute_field_feedback(capsule_layer)
field.apply_stability_signal(feedback["stability"])
```

### With Telemetry Hub

```python
from ww.visualization.telemetry_hub import TelemetryHub
from ww.visualization.capsule_visualizer import CapsuleVisualizer

hub = TelemetryHub()
capsule_viz = CapsuleVisualizer()
hub.register_capsule(capsule_viz)

# Record capsule state
hub.record_state(capsule_state=layer.get_state())
```

---

## Biological Rationale

### Cortical Microcolumn Analogy

| Capsule Concept | Cortical Analog | Reference |
|-----------------|-----------------|-----------|
| Capsule | Cortical microcolumn | Mountcastle 1997 |
| Pose vector | Columnar representation | Douglas & Martin 2004 |
| Dynamic routing | Lateral connections | Felleman & Van Essen 1991 |
| Activation probability | Column activity | |

### Neuromodulator Mapping

| NT | Capsule Effect | Biological Basis |
|----|----------------|------------------|
| ACh | Mode (enc/ret) | Hasselmo (2006) |
| DA | Routing sharpness | Salience gating |
| NE | Squashing threshold | Arousal modulation |
| 5-HT | Routing stability | Impulse control |

---

## Performance Considerations

- **Routing iterations**: Default 3, can reduce to 1-2 for inference
- **Temperature clamping**: Always clamp to [0.1, 2.0] for stability
- **Mode hysteresis**: 0.1 prevents rapid mode switching

---

## See Also

- [Forward-Forward API](ff-api.md) - FF algorithm integration
- [NCA API](nca-api.md) - Neural field dynamics
- [Capsules Concept](../concepts/capsules.md) - Conceptual overview
- [Brain Mapping](../concepts/brain-mapping.md) - Biological mappings
