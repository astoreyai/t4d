# Forward-Forward Algorithm

Implementation of Hinton's Forward-Forward Algorithm (2022) for layer-local learning without backpropagation.

**Phase 3 Implementation**: Hinton H6 (local learning), H7 (contrastive phases)

## Overview

The Forward-Forward (FF) algorithm replaces backpropagation with a local learning objective at each layer. Each layer learns to distinguish "positive" (real) data from "negative" (corrupted) data using a **goodness function**.

```
Goodness G(h) = sum(h_i^2)   # Sum of squared activations
```

- **Positive data**: Maximize goodness (G > threshold)
- **Negative data**: Minimize goodness (G < threshold)

## Key Concepts

### 1. Goodness Function

The goodness of a layer is simply the sum of squared activations:

```python
def compute_goodness(h: np.ndarray) -> float:
    return float(np.sum(h ** 2))
```

High goodness indicates the layer "recognizes" the input as positive (real) data.

### 2. Threshold Classification

A threshold θ separates positive from negative:

```
if G(h) > θ: classify as positive
if G(h) < θ: classify as negative
```

### 3. Local Learning Rule

Each layer learns independently:

```python
# Positive phase: increase goodness
Δw = lr × (1 - p) × h × x^T

# Negative phase: decrease goodness
Δw = -lr × p × h × x^T
```

Where `p = sigmoid(goodness - threshold)` is the classification probability.

## Usage

### Basic Layer

```python
from ww.nca import create_ff_layer, ForwardForwardConfig

# Create a layer
layer = create_ff_layer(
    input_dim=1024,
    hidden_dim=512,
    learning_rate=0.03,
    threshold=2.0
)

# Forward pass
x = np.random.randn(1024)
h = layer.forward(x)

# Check classification
print(f"Goodness: {layer.state.goodness}")
print(f"Is positive: {layer.state.is_positive}")
print(f"Confidence: {layer.state.confidence}")

# Learn from positive sample
layer.learn_positive(x, h)

# Learn from negative sample
layer.learn_negative(x, h)
```

### Multi-Layer Network

```python
from ww.nca import create_ff_network

# Create a 3-layer network
network = create_ff_network(
    layer_dims=[1024, 512, 256],
    learning_rate=0.03,
    threshold=2.0
)

# Forward pass through all layers
activations = network.forward(x)

# Generate negative samples
negative, _ = network.generate_negative(positive_data, method="hybrid")

# Training step (positive and negative phases)
stats = network.train_step(positive_data, negative_data)

# Inference
is_positive, confidence = network.infer(x)
```

### Negative Sample Generation

Multiple methods available:

```python
# Gaussian noise
neg, _ = network.generate_negative(pos, method="noise")

# Feature shuffling
neg, _ = network.generate_negative(pos, method="shuffle")

# Adversarial (gradient ascent on goodness)
neg, _ = network.generate_negative(pos, method="adversarial")

# Hybrid (mix of above)
neg, _ = network.generate_negative(pos, method="hybrid")

# Wrong label (for classification)
neg, wrong_labels = network.generate_negative(pos, method="wrong_label", labels=true_labels)
```

## Architecture Diagram

```
Input Layer              Hidden Layer 1           Hidden Layer 2
    x                         h1                       h2
    |                         |                        |
    v                         v                        v
[Linear] -----> [ReLU] ---> [Linear] --> [ReLU] --> [Linear] --> [ReLU]
    |                         |                        |
    v                         v                        v
G1 = Σ(h1²)              G2 = Σ(h2²)              G3 = Σ(h3²)
    |                         |                        |
    v                         v                        v
Local Update             Local Update             Local Update
(no backprop)            (no backprop)            (no backprop)
```

Each layer computes its own goodness and updates independently.

## Biological Plausibility

The FF algorithm is more biologically plausible than backpropagation:

1. **Local Learning**: Each synapse updates based on local activity only
2. **No Backward Pass**: No need to propagate errors backward through layers
3. **Hebbian-Like**: Weight updates correlate with pre/post-synaptic activity
4. **Phase-Dependent**: Positive/negative phases map to wake/sleep or exploration/consolidation

### Mapping to Neuromodulators

| Neuromodulator | FF Role |
|----------------|---------|
| Dopamine | Modulates learning rate (surprise → higher LR) |
| Acetylcholine | Gates encoding (positive phase) vs retrieval |
| Norepinephrine | Modulates threshold (arousal → higher threshold) |

## Configuration

```python
from ww.nca import ForwardForwardConfig

config = ForwardForwardConfig(
    # Layer dimensions
    input_dim=1024,
    hidden_dim=512,

    # Goodness parameters
    threshold_theta=2.0,      # Classification threshold
    normalize_goodness=True,  # Divide by layer size

    # Learning parameters
    learning_rate=0.03,       # Higher than backprop
    momentum=0.9,
    weight_decay=1e-4,

    # Activation
    activation="relu",        # relu, leaky_relu, gelu

    # Negative generation
    negative_method="hybrid",
    noise_scale=0.3,
    adversarial_steps=5,

    # Neuromodulation
    use_neuromod_gating=True,
    da_modulates_lr=True,
    ach_modulates_phase=True,
    ne_modulates_threshold=True,

    # Weight bounds
    max_weight=2.0,
    min_weight=-2.0,
    use_layer_norm=True,
)
```

## Integration with World Weaver

### With NCA Energy System

The FF algorithm integrates with the existing energy-based learning in `energy.py`:

```python
from ww.nca import EnergyLandscape, ForwardForwardNetwork

# FF can be used for learned gates and rerankers
ff_network = create_ff_network([1024, 256, 64])

# Train on positive (real memories) and negative (noise)
for episode in episodes:
    embedding = episode.embedding
    negative = generate_noise(embedding)
    ff_network.train_step(embedding, negative)
```

### With Hippocampal Circuit

FF layers can work alongside the DG→CA3→CA1 pipeline:

```python
# Each hippocampal layer could use FF-style learning
# DG: Pattern separation with local goodness
# CA3: Pattern completion with Hopfield dynamics
# CA1: Novelty detection with FF threshold
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Forward pass | O(n × m) per layer |
| Learning | O(n × m) per layer (no backprop) |
| Memory | O(n × m) weights only |
| Parallelization | Fully parallel across layers |

## FF-NCA Coupling (H10)

The FF algorithm integrates with the NCA energy landscape via `forward_forward_nca_coupling.py`:

```
FF Goodness ←→ NCA Energy
─────────────────────────
Goodness = -Energy
Positive phase = Descend to basin
Negative phase = Ascend barrier
Threshold θ = Energy barrier height
```

### Bidirectional Information Flow

```python
from ww.nca import FFNCACoupling, FFNCACouplingConfig

coupling = FFNCACoupling(FFNCACouplingConfig(
    goodness_to_energy_scale=1.0,
    energy_to_goodness_scale=1.0,
    temperature_coupling=0.5
))

# FF → NCA: Layer goodness influences field energy
energy_update = coupling.ff_to_nca(ff_state)

# NCA → FF: Attractor basins guide layer activation
ff_feedback = coupling.nca_to_ff(energy_state)
```

### Wake/Sleep Analogy

| FF Phase | Brain State | Function |
|----------|-------------|----------|
| Positive | Wake | Learn from real experiences |
| Negative | Sleep/Dreams | Learn from generated data |
| Inference | Both | Pattern recognition |

## P6.4: FF Retrieval Scoring

The Forward-Forward algorithm integrates with retrieval through the `FFRetrievalScorer` bridge:

```python
from ww.bridges import FFRetrievalScorer, FFRetrievalConfig

# Create scorer
scorer = FFRetrievalScorer(
    ff_layer=ff_layer,
    config=FFRetrievalConfig(
        max_boost=0.3,
        learn_from_outcomes=True,
    )
)

# Score retrieval candidates
scores = scorer.score_candidates(
    candidate_embeddings=embeddings,
    candidate_ids=memory_ids,
)

# High goodness = confident pattern match
for score in scores:
    if score.is_confident:
        base_score[score.memory_id] += score.boost
```

### Retrieval Confidence Scoring

FF goodness indicates pattern match confidence:

| Goodness | Interpretation | Boost |
|----------|----------------|-------|
| High (>θ+1) | Confident match (familiar pattern) | +0.15 to +0.30 |
| Medium (θ±1) | Uncertain | 0.0 |
| Low (<θ-1) | Poor match (novel/corrupted) | 0.0 |

### Learning from Outcomes

The FF layer trains from retrieval outcomes:

```python
# After retrieval, when outcome is known
scorer.learn_from_outcome(
    embeddings=[query_emb] + retrieved_embs,
    memory_ids=["query"] + retrieved_ids,
    outcome_score=0.8,  # 1.0 = success
)
# Positive outcome → increase goodness for these patterns
# Negative outcome → decrease goodness
```

This closes the learning loop: retrieval → outcome → FF weight update.

## Hinton Validation Score

**Architecture Fidelity: 8.5/10** (ww-hinton agent analysis)

### Strengths
- Correct goodness function: `G(h) = sum(h_i^2)`
- Layer-local learning without backward pass
- Multiple negative sample methods (exceeds original paper)
- Neuromodulator integration (biological extension)
- Label embedding in first layer

### Gaps Identified
- No sleep/offline replay phase for negative generation
- Global threshold (could be per-layer adaptive)
- No temporal integration over 10-20ms windows
- Strictly feedforward (no backward activity flow)

### Recommended Improvements
1. Add sleep consolidation with FF negative phase
2. Implement per-layer adaptive thresholds
3. Add temporal goodness smoothing

## References

- Hinton, G. (2022). The Forward-Forward Algorithm: Some Preliminary Investigations
- Lillicrap, T. et al. (2020). Backpropagation and the brain
- Bi, G. & Poo, M. (1998). Synaptic modifications in cultured hippocampal neurons
- Friston, K. (2010). The free-energy principle
- Whittington & Bogacz (2017). Approximation of the error backpropagation

## API Reference

### Classes

- `ForwardForwardConfig`: Configuration dataclass
- `ForwardForwardState`: State tracking dataclass
- `ForwardForwardLayer`: Single FF layer
- `ForwardForwardNetwork`: Multi-layer FF network
- `FFPhase`: Enum (POSITIVE, NEGATIVE, INFERENCE)

### Factory Functions

- `create_ff_layer(input_dim, hidden_dim, learning_rate, threshold, random_seed)`
- `create_ff_network(layer_dims, learning_rate, threshold, random_seed)`
