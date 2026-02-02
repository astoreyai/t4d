# Encoding Module

**5 files | ~1,200 lines | Centrality: 1**

The encoding module implements biologically-inspired neural encoding mechanisms including sparse coding, dendritic computation, and associative memory networks.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      NEURAL ENCODING SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    SPARSE ENCODER (Dentate Gyrus)                   ││
│  │  [Input: 1024] → 8x Expansion → k-WTA → [Sparse: 8192, 2% active]   ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    DENDRITIC NEURON (L5 Pyramidal)                  ││
│  │  [Basal: feedforward] + [Apical: top-down] → Mismatch signal       ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    ATTRACTOR NETWORK (CA3 Hopfield)                 ││
│  │  Content-addressable memory → Pattern completion → Retrieval        ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `sparse.py` | ~275 | k-WTA sparse encoding (DG-inspired) |
| `dendritic.py` | ~240 | Two-compartment dendritic neurons |
| `attractor.py` | ~485 | Hopfield associative memory network |
| `utils.py` | ~175 | Utility functions |
| `__init__.py` | ~20 | Public API exports |

## Sparse Coding (Pattern Separation)

### k-Winner-Take-All

```python
from t4dm.encoding import kwta

# Keep exactly top-k activations
sparse = kwta(activations, k=164)  # 2% of 8192
# Gradient: Straight-through estimator
```

### SparseEncoder

Hippocampal dentate gyrus-inspired expansion:

```python
from t4dm.encoding import SparseEncoder, SparseEncoderConfig

config = SparseEncoderConfig(
    input_dim=1024,          # BGE-M3 embedding
    hidden_dim=8192,         # 8x expansion
    sparsity=0.02,           # 2% target (biological)
    use_kwta=True,           # vs soft thresholding
    lateral_inhibition=0.2   # Competition
)

encoder = SparseEncoder(config)

# Encode to sparse representation
sparse_code = encoder(embedding)
# sparse_code.shape: [8192], ~164 non-zero values

# Get active indices
indices = encoder.get_active_indices(sparse_code)

# Pattern overlap (Jaccard similarity)
overlap = encoder.compute_pattern_overlap(code1, code2)

# Verify sparsity
actual = encoder.actual_sparsity  # k/hidden_dim
```

**Biological Basis**:
- Dentate gyrus granule cells: ~2% active
- 8x expansion decorrelates similar inputs
- Lateral inhibition enforces competition

### AdaptiveSparseEncoder

Dynamic sparsity adjustment:

```python
from t4dm.encoding import AdaptiveSparseEncoder

encoder = AdaptiveSparseEncoder(
    min_sparsity=0.01,    # 1% minimum
    max_sparsity=0.05,    # 5% maximum
    adaptation_rate=0.1
)

# Sparsity adapts to input statistics
sparse_code = encoder(embedding)
```

## Dendritic Computation

### DendriticNeuron

Two-compartment pyramidal neuron model:

```python
from t4dm.encoding import DendriticNeuron, DendriticConfig

config = DendriticConfig(
    input_dim=1024,
    hidden_dim=512,
    context_dim=512,
    coupling_strength=0.5,    # Basal-apical modulation
    tau_dendrite=10.0,        # ms (faster)
    tau_soma=15.0,            # ms (slower)
    activation="tanh"
)

neuron = DendriticNeuron(config)

# Forward pass
output, mismatch = neuron(feedforward_input, context_input)
# mismatch: ||h_basal - h_apical|| (prediction error)

# Measure context influence
influence = neuron.compute_context_influence(feedforward, context)
```

**Architecture**:
```
Basal (feedforward) ──┐
                      ├─→ Gate (sigmoid) ──┐
Apical (top-down)  ───┘                    ├─→ Coupling ──→ Soma ──→ Output
                                           │
                                    Gating mechanism
```

### DendriticProcessor

Stacked dendritic layers:

```python
from t4dm.encoding import DendriticProcessor

processor = DendriticProcessor(
    input_dim=1024,
    hidden_dims=[512, 256],
    context_dim=512,
    dropout=0.1
)

# Hierarchical processing
output, mismatches = processor(input, context)
# mismatches: List of mismatch signals per layer
```

**Biological Basis**: Layer 5 pyramidal neurons (Häusser & Mel, 2003)
- Basal dendrites: bottom-up sensory input
- Apical dendrites: top-down predictions
- Mismatch: prediction error signal

## Associative Memory (Pattern Completion)

### AttractorNetwork

Classical Hopfield-style associative memory:

```python
from t4dm.encoding import AttractorNetwork, AttractorConfig

config = AttractorConfig(
    dim=8192,                # Pattern dimension
    symmetric=True,          # Enforce weight symmetry
    noise_std=0.01,          # Exploration noise
    settling_steps=10,       # Default iterations
    step_size=0.1,           # Update magnitude
    capacity_ratio=0.138     # ~0.14N theoretical limit
)

network = AttractorNetwork(config)

# Store pattern (Hebbian)
result = network.store(pattern, pattern_id="memory_1")
# result.capacity_used, result.overlap_with_existing

# Retrieve from partial/noisy cue
result = network.retrieve(noisy_cue, max_steps=20)
# result.pattern: Final state
# result.pattern_id: Best matching stored pattern
# result.confidence: Cosine similarity
# result.steps: Iterations to converge
# result.energy: Final energy value

# Remove pattern (unlearn)
network.remove(pattern_id="memory_1")

# Analyze network
stats = network.analyze()
# capacity_used, weight_stats, overlaps

# Estimate basin of attraction
basin = network.get_basin_estimate(pattern, noise_levels=[0.1, 0.2, 0.3])
```

**Energy Function**:
```
E = -0.5 × s^T W s
```
Lower energy = more stable state

**Capacity**: ~0.14N patterns (Hopfield theoretical limit)

### ModernHopfieldNetwork

Enhanced with exponential energy (Ramsauer et al. 2020):

```python
from t4dm.encoding import ModernHopfieldNetwork

network = ModernHopfieldNetwork(
    dim=8192,
    beta=1.0,               # Sharpness (softmax temperature)
    capacity_ratio=0.5      # Much higher capacity
)

# Arousal-modulated retrieval (NE coupling)
network.set_beta_from_arousal(ne_level=0.8)
# Higher NE → higher beta → sharper retrieval

# Retrieve with attention-like update
result = network.retrieve(cue)
```

**Key Improvement**: O(d^α) capacity where α > 1

## Utility Functions

```python
from t4dm.encoding.utils import (
    compute_sparsity,
    validate_sparsity,
    cosine_similarity_matrix,
    compute_pattern_orthogonality,
    straight_through_estimator,
    exponential_decay,
    normalize_to_range,
    add_noise
)

# Sparsity
sparsity = compute_sparsity(sparse_code)  # Fraction non-zero
is_valid = validate_sparsity(code, min=0.01, max=0.05)

# Similarity
sim_matrix = cosine_similarity_matrix(patterns)  # (N,D) → (N,N)
orthogonality = compute_pattern_orthogonality(patterns)

# Gradient tricks
output = straight_through_estimator(x, threshold=0.5)

# Dynamics
decayed = exponential_decay(initial=1.0, tau=10.0, dt=1.0)

# Preprocessing
normalized = normalize_to_range(x, min_val=0.0, max_val=1.0)
noisy = add_noise(x, noise_std=0.1, noise_type="gaussian")
```

## Biological Grounding

| Component | Brain Region | Function | Reference |
|-----------|--------------|----------|-----------|
| Sparse Encoder | Dentate Gyrus | Pattern separation | Treves & Rolls 1994 |
| k-WTA | Lateral inhibition | Competition | |
| Dendritic Neuron | L5 Pyramidal | Prediction error | Häusser & Mel 2003 |
| Attractor Network | CA3 | Pattern completion | Hopfield 1982 |
| Modern Hopfield | CA3/neocortex | High-capacity retrieval | Ramsauer 2020 |

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| k-WTA | O(n log k) | torch.topk |
| Sparse encode | O(nd) | Linear projection |
| Dendritic forward | O(d²) | Matrix multiply |
| Hopfield store | O(n²) | Outer product |
| Hopfield retrieve | O(kn²) | k settling steps |

## Security Limits

```python
# Input validation
MAX_INPUT_DIM = 16384
MAX_HIDDEN_DIM = 131072
MAX_DIM = 65536
MAX_SETTLING_STEPS = 1000
MAX_BASIN_SAMPLES = 1000
```

## Testing

```bash
# Run encoding tests
pytest tests/encoding/ -v

# With coverage
pytest tests/encoding/ --cov=t4dm.encoding

# Test categories
pytest tests/encoding/test_sparse.py -v      # 246 lines
pytest tests/encoding/test_dendritic.py -v   # 185 lines
pytest tests/encoding/test_attractor.py -v   # 267 lines
```

**Test Coverage**: 700+ lines of comprehensive tests

## Installation

```bash
# Encoding included in core
pip install -e "."

# Optional GPU support
pip install torch  # For CUDA acceleration
```

## Public API

```python
# Sparse coding
SparseEncoder, SparseEncoderConfig, AdaptiveSparseEncoder
kwta

# Dendritic computation
DendriticNeuron, DendriticConfig, DendriticProcessor

# Associative memory
AttractorNetwork, AttractorConfig, RetrievalResult
ModernHopfieldNetwork

# Utilities
compute_sparsity, validate_sparsity
cosine_similarity_matrix, compute_pattern_orthogonality
straight_through_estimator, exponential_decay
normalize_to_range, add_noise
```

## References

- Hopfield (1982) "Neural networks and physical systems"
- Treves & Rolls (1994) "Hippocampal pattern separation"
- Häusser & Mel (2003) "Dendrites: bug or feature?"
- Ramsauer et al. (2020) "Hopfield Networks is All You Need"
