# Capsule Networks

Implementation of Hinton's Capsule Networks (2017) for part-whole hierarchical representations.

**Phase 4 Implementation**: Hinton H8 (part-whole), H9 (spatial reasoning)
**Phase 6 Update**: Pose learning from routing agreement (poses emerge from data, not hand-set)

## Overview

Capsule Networks encode entities as vectors (capsules) that represent both the presence and pose (position, orientation, scale) of visual/conceptual parts. Unlike traditional CNNs that lose spatial relationships through pooling, capsules preserve geometric information.

```
Capsule Output:
- Length = probability of entity existing
- Direction = instantiation parameters (pose)
```

## Key Concepts

### 1. Capsule Representation

A capsule is a vector where:

```python
class Capsule:
    """
    Vector representation of an entity.

    Properties:
    - Length (0-1): Probability of entity existence
    - Direction: Pose parameters (position, orientation, scale, etc.)
    """

    def squash(v: np.ndarray) -> np.ndarray:
        """Squashing non-linearity (length → probability)."""
        norm = np.linalg.norm(v)
        return (norm ** 2 / (1 + norm ** 2)) * (v / (norm + 1e-8))
```

### 2. Dynamic Routing by Agreement

Capsules communicate through routing by agreement:

```python
def routing_by_agreement(lower_caps, upper_caps, iterations=3):
    """
    Route lower-level capsules to higher-level capsules.

    1. Initialize coupling coefficients uniformly
    2. For each iteration:
       - Compute weighted sum of predictions
       - Update coupling based on agreement
    """
    b_ij = np.zeros((n_lower, n_upper))  # Logits

    for _ in range(iterations):
        c_ij = softmax(b_ij, axis=1)  # Coupling coefficients
        s_j = sum(c_ij * u_hat_ji)     # Weighted predictions
        v_j = squash(s_j)              # Output capsules
        b_ij += u_hat_ji @ v_j         # Agreement update
```

### 3. Pose Transformation

Each capsule predicts higher-level capsule poses via learned transformation matrices:

```python
# Prediction: u_hat = W @ u_i
prediction = pose_matrix @ lower_capsule

# Agreement: dot(prediction, actual)
agreement = np.dot(prediction, upper_capsule)
```

## Integration with T4DM

### Neuromodulator Coupling

Capsule routing is modulated by the NT state:

| Neuromodulator | Effect on Capsules |
|----------------|-------------------|
| DA | Routing temperature (sharpness) |
| NE | Squashing threshold (arousal-gated) |
| ACh | Encoding vs retrieval mode |
| 5-HT | Routing convergence (patience) |

```python
from t4dm.nca import CapsuleNCACoupling

coupling = CapsuleNCACoupling()

# Modulate routing based on NT state
nt_state = {'da': 0.7, 'ne': 0.5, 'ach': 0.3, 'sht': 0.6}
modulated = coupling.modulate_routing(
    capsule_state=capsule_output,
    nt_state=nt_state
)

# High DA → low temperature → sharper routing
# Low DA → high temperature → exploratory routing
```

### NCA Field Coupling

Capsules provide feedback to the NCA neural field:

```python
# Capsule agreement → Field stability
stability_signal = coupling.compute_stability_feedback(
    routing_agreement=capsule_agreement,
    pose_coherence=pose_variance
)

# Inject into NCA
nca_field.apply_external_input(stability_signal)
```

## Usage

### Basic Capsule Layer

```python
from t4dm.nca import CapsuleNetwork, CapsuleConfig

config = CapsuleConfig(
    input_dim=1024,
    n_primary_caps=32,
    primary_cap_dim=8,
    n_output_caps=16,
    output_cap_dim=16,
    routing_iterations=3
)

caps = CapsuleNetwork(config)

# Encode input
input_vector = np.random.randn(1024)
capsules = caps.encode(input_vector)

# Get entity probabilities (capsule lengths)
probs = caps.get_entity_probabilities()

# Get pose parameters (capsule directions)
poses = caps.get_poses()
```

### With NT Modulation

```python
from t4dm.nca import CapsuleNCACoupling

coupling = CapsuleNCACoupling()

# NT-modulated encoding
output = coupling.forward(
    input_vector=x,
    nt_state={'da': 0.8, 'ne': 0.4, 'ach': 0.6, 'sht': 0.5}
)

# Access modulated parameters
print(f"Routing temperature: {output.routing_temperature}")
print(f"Squash threshold: {output.squash_threshold}")
print(f"Mode: {output.mode}")  # 'encoding' or 'retrieval'
```

## Architecture Diagram

```
Input (1024-dim)
       │
       ▼
┌──────────────────┐
│ Primary Capsules │ ← 32 capsules × 8-dim
│   (conv + reshape)│
└────────┬─────────┘
         │
         │ Routing by Agreement (3 iterations)
         │ + NT Modulation (DA→temp, NE→thresh)
         ▼
┌──────────────────┐
│ Output Capsules  │ ← 16 capsules × 16-dim
│   (digit/entity) │
└────────┬─────────┘
         │
         ▼
    ┌─────────┐
    │ Length  │ → Entity probability
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │Direction│ → Pose parameters
    └─────────┘
```

## Biological Plausibility

### Cortical Column Analogy

Capsules map to cortical minicolumns:

| Capsule Concept | Cortical Analog |
|-----------------|-----------------|
| Capsule | Minicolumn (~100 neurons) |
| Capsule length | Column activation probability |
| Pose vector | Population code for features |
| Routing | Inter-columnar coordination |

### Part-Whole Hierarchy

The hierarchical routing reflects visual cortex organization:

```
V1 (edges) → V2 (textures) → V4 (shapes) → IT (objects)
     │              │              │            │
Primary Caps  →  Mid Caps   →  High Caps  →  Entity Caps
```

## P6.2: Capsule Retrieval Scoring

Capsules integrate with memory retrieval through the `CapsuleRetrievalBridge`:

```python
from t4dm.bridges import CapsuleRetrievalBridge, create_capsule_bridge

# Create bridge
bridge = create_capsule_bridge(
    capsule_layer=capsule_layer,
    activation_weight=0.4,
    pose_weight=0.6,
    max_boost=0.3,
)

# Score candidates based on part-whole agreement
boosts = bridge.compute_boosts(query_embedding, candidate_embeddings)

# High agreement = coherent hierarchical composition
for memory_id, boost in boosts.items():
    base_score[memory_id] += boost
```

### Retrieval Boost Mechanism

Capsule routing agreement indicates hierarchical coherence:

| Agreement | Interpretation | Boost |
|-----------|----------------|-------|
| High (>0.8) | Strong part-whole match | +0.2 to +0.3 |
| Medium (0.5-0.8) | Partial match | +0.1 to +0.2 |
| Low (<0.5) | Poor hierarchical fit | 0.0 |

### Scoring Components

```python
# Total boost combines activation and pose agreement
boost = (
    activation_weight * activation_similarity +
    pose_weight * pose_agreement
)

# activation_similarity: How active are similar capsules?
# pose_agreement: Do pose vectors align (same viewpoint/orientation)?
```

### Integration with Episodic Recall

In the recall pipeline, capsule scoring adds hierarchical coherence:

```
Query Embedding
    │
    ▼
[Semantic Search] → Base similarity scores
    │
    ▼
[FF Scoring] → Pattern confidence boost (P6.4)
    │
    ▼
[Capsule Scoring] → Hierarchical coherence boost (P6.2)
    │
    ▼
[Reranking] → Final ordering
```

## Phase 6: Pose Learning from Routing

Phase 6 addresses a critical gap identified by the Hinton agent: "Capsule poses are hand-set, not learned."

### The Problem (Before Phase 6)

```python
# Poses were manually set (BAD)
pose.set_temporal(offset=0.5)
pose.set_causal(cause_strength=0.8)
```

Poses should **emerge from data** through routing-by-agreement, not be hand-coded.

### The Solution: Pose Learning from Agreement

```python
# Now poses emerge from routing (GOOD)
activations, poses = capsule_layer.forward(embedding)

# Higher-level capsule poses emerge from routing
higher_activations, higher_poses, agreement = capsule_layer.route(
    lower_activations=activations,
    lower_poses=poses,
    num_output_capsules=16,
    learn_poses=True,  # Phase 6: Enable pose learning
    learning_rate=0.01,
)
```

### Learning Mechanism

Pose weights update based on routing agreement:

```python
def learn_pose_from_routing(self, agreement_scores):
    """Update pose weights when capsules agree."""
    for i in range(num_lower):
        for j in range(num_upper):
            # Agreement measures prediction accuracy
            agreement = agreement_scores[i, j]

            if agreement > threshold:
                # Strengthen accurate pose predictions
                pose_update = lr * agreement * prediction_gradient
                self._transform_matrices[i, j] += pose_update
```

### Integration with Episodic Memory

Capsule representations are now stored and used in retrieval using `forward_with_routing()`:

```python
# In EpisodicMemory.create() - uses forward_with_routing for pose learning
activations, poses, routing_stats = self._capsule_layer.forward_with_routing(
    embedding,
    learn_poses=True,  # Enable pose weight learning during storage
)
episode.capsule_activations = activations
episode.capsule_poses = poses
# routing_stats contains: mean_agreement, pose_change, etc.

# In EpisodicMemory.recall() - uses forward_with_routing WITHOUT learning
query_activations, query_poses, _ = self._capsule_layer.forward_with_routing(
    query_embedding,
    learn_poses=False,  # Don't modify weights during retrieval
)
# Use pose agreement to boost retrieval ranking
```

### forward_with_routing() Method

The key Phase 6 method combines forward pass with routing-by-agreement:

```python
def forward_with_routing(
    self,
    x: np.ndarray,
    routing_iterations: int | None = None,
    learn_poses: bool = True,
    learning_rate: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Forward pass with self-routing for pose refinement.

    1. Initial forward pass to get activations/poses
    2. Self-routing to refine poses through agreement
    3. Optional pose weight learning from routing
    """
    # Step 1: Initial forward
    initial_activations, initial_poses = self.forward(x)

    # Step 2: Self-routing refines poses via agreement
    refined_activations, refined_poses, routing_coeff = self.route(
        lower_activations=initial_activations,
        lower_poses=initial_poses,
        num_output_capsules=self.config.num_capsules,
        iterations=routing_iterations,
        learn_poses=learn_poses,
        learning_rate=learning_rate,
    )

    return refined_activations, refined_poses, routing_stats
```

### Pose Agreement in Retrieval

Pose agreement provides hierarchical coherence scoring:

```python
# Compare query poses to stored poses
pose_diff = query_poses - stored_poses
distances = np.linalg.norm(pose_diff, axis=-1)
pose_agreement = 1.0 - distances / max_distance

# High agreement = similar viewpoint/configuration
# Boost score for memories with aligned poses
```

### Three-Factor Integration

Pose learning integrates with three-factor learning:

```python
# In EpisodicMemory.learn_from_outcome()
if outcome_score > 0.5:
    # Positive outcome: strengthen current pose representation
    capsule_layer.learn_positive(
        x=query_embedding,
        activations=capsule_activations,
        poses=capsule_poses,  # Phase 6: Update pose weights
        learning_rate=three_factor_signal.effective_lr_multiplier,
    )
```

## Testing

```bash
# Run all capsule tests (50 tests)
pytest tests/nca/test_capsules.py -v

# Run Phase 6 pose learning tests specifically
pytest tests/nca/test_capsules.py::TestPhase6PoseLearning -v
pytest tests/nca/test_capsules.py::TestPhase6CapsuleNetworkTraining -v
pytest tests/nca/test_capsules.py::TestPhase6LastInputTracking -v

# Run forward_with_routing tests (THE KEY Phase 6 tests)
pytest tests/nca/test_capsules.py::TestPhase6ForwardWithRouting -v

# Run capsule bridge tests
pytest tests/bridges/test_capsule_bridge.py -v

# Run coupling tests
pytest tests/integration/test_h10_cross_region_consistency.py::TestCapsuleNCACoupling -v
```

## Hinton Validation Score

**Architecture Fidelity: 8.0/10** (updated after Phase 6)

### Strengths
- Correct squashing function: `scale = norm^2 / (1 + norm^2)`
- Proper dynamic routing by agreement
- Novel semantic pose dimensions (TEMPORAL, CAUSAL, SEMANTIC_ROLE, CERTAINTY)
- Transformation matrices for part-whole predictions
- FF learning integration via `learn_positive/learn_negative`
- **Phase 6**: Pose learning from routing agreement (poses emerge from data)
- **Phase 6**: Three-factor modulated pose weight updates
- **Phase 6**: Capsule encoding integrated with episodic memory

### Gaps Addressed (Phase 6)
- ~~Poses hand-set not learned~~ → Poses now emerge from routing agreement
- ~~No pose weight updates~~ → `learn_pose_from_routing()` updates transform matrices
- ~~Capsules disconnected from memory~~ → Integrated into EpisodicMemory pipeline

### Remaining Gaps
- Only dynamic routing (no EM routing from Hinton 2018)
- No LayerNorm between capsule layers
- Non-local einsum operations (not neurally plausible)
- Missing k-winner-take-all for sparse activation

### Recommended Improvements
1. Implement EM routing for semantic capsules
2. Add coordinate addition for semantic poses
3. Reformulate routing using only local operations
4. Add sparse capsule activation via k-WTA

## Capsule-NCA Coupling

See [Cross-Region Integration](cross-region-integration.md) for full details on how capsules couple with the NCA neural field through neuromodulator gating.

### Phase 6: NT-Modulated Capsule Forward

The coupling now provides `forward_with_nt_modulation()` for integrated NT-modulated capsule encoding:

```python
from t4dm.nca import CapsuleNCACoupling, CapsuleLayer

# Create coupling and layer
coupling = CapsuleNCACoupling()
layer = CapsuleLayer(config)

# NT-modulated forward pass
activations, poses, stats = coupling.forward_with_nt_modulation(
    capsule_layer=layer,
    embedding=query_embedding,
    nt_state=current_nt_state,
)

# NT modulation determines:
# - learn_poses: True if ACh high (encoding mode), False if low (retrieval)
# - learning_rate: DA-modulated (higher DA = stronger learning)
# - routing_iterations: 5-HT-modulated (higher = more patient routing)

print(f"Mode: {stats['nt_mode']}")  # 'ENCODING' or 'RETRIEVAL'
print(f"Learn poses: {stats['nt_learn_poses']}")
print(f"Agreement: {stats['mean_agreement']:.3f}")
```

### NT Modulation Effects

| Neuromodulator | Capsule Effect |
|----------------|----------------|
| DA (dopamine) | Routing temperature (sharpness) + learning rate |
| NE (norepinephrine) | Squashing threshold (arousal-gated selectivity) |
| ACh (acetylcholine) | Encoding vs retrieval mode switch |
| 5-HT (serotonin) | Routing convergence patience (iterations) |

## References

1. Sabour, Frosst, Hinton (2017). "Dynamic Routing Between Capsules"
2. Hinton, Krizhevsky, Wang (2011). "Transforming Auto-encoders"
3. Kosiorek et al. (2019). "Stacked Capsule Autoencoders"
4. Hinton et al. (2018). "Matrix Capsules with EM Routing"
5. Hasselmo (2006). "The role of acetylcholine in learning and memory"
