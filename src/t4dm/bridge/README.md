# Bridge Module

**2 files | ~450 lines | Centrality: 2**

The bridge module connects T4DM's memory operations to the Neural Cognitive Architecture (NCA), enabling state-dependent encoding and retrieval, neuromodulated learning, and consolidation triggering.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MEMORY-NCA BRIDGE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    STATE-DEPENDENT ENCODING                         ││
│  │  [Embedding] + [NT State: 6-dim] → [Augmented Embedding]            ││
│  │  Cognitive state modulation: FOCUS (+1.3x), EXPLORE (+noise)        ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    STATE-MODULATED RETRIEVAL                        ││
│  │  Rank candidates by NT similarity + cognitive state bonus           ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    LEARNING SIGNAL INTEGRATION                      ││
│  │  Outcome → RPE → Coupling update → Effective LR modulation          ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    CONSOLIDATION TRIGGERING                         ││
│  │  Select memories by recency × energy for replay                     ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `memory_nca.py` | ~420 | Core bridge implementation |
| `__init__.py` | ~30 | Public API exports |

## Core Components

### BridgeConfig

Configuration for bridge behavior:

```python
from t4dm.bridge import BridgeConfig

config = BridgeConfig(
    # Encoding
    encoding_nt_weight=0.3,       # NT state influence
    state_context_dim=32,         # State projection dimension

    # Retrieval
    retrieval_state_matching=True,
    state_similarity_weight=0.2,

    # Learning
    use_nca_gradients=True,
    coupling_lr_scale=0.5,

    # Cognitive state modifiers
    focus_boost=1.3,              # Encoding boost in FOCUS
    explore_diversity=1.5,         # Retrieval diversity in EXPLORE
    consolidate_replay=10          # Memories per CONSOLIDATE trigger
)
```

### MemoryNCABridge

Main bridge class:

```python
from t4dm.bridge import MemoryNCABridge, BridgeConfig

bridge = MemoryNCABridge(
    config=BridgeConfig(),
    neural_field=neural_field_solver,
    coupling=learnable_coupling,
    state_manager=state_transition_manager,
    energy_landscape=energy_landscape,
    dopamine=dopamine_system
)

# Get current NCA state
nt_state = bridge.get_current_nt_state()
# [DA, 5-HT, ACh, NE, GABA, Glu] (6-dim)

cognitive_state = bridge.get_current_cognitive_state()
# CognitiveState.FOCUS, ALERT, REST, EXPLORE, CONSOLIDATE
```

## State-Dependent Encoding

Augment embeddings with NT state:

```python
augmented, context = bridge.augment_encoding(
    embedding=episode_embedding,
    memory_id=episode.id
)

# context.memory_id: UUID
# context.embedding: Original embedding
# context.nt_state: NT state at encoding (6-dim)
# context.cognitive_state: CognitiveState
# context.timestamp: datetime
# context.energy: System energy
```

**Algorithm**:
1. Get current NT state (6-dim)
2. Project NT → context vector: `nt_state @ projection`
3. Apply cognitive state modulation:
   - FOCUS: Multiply context by 1.3x
   - EXPLORE: Add 0.1x Gaussian noise
4. Blend: `concat(embedding×(1-w), context×w)`
5. Create EncodingContext

## State-Modulated Retrieval

Bias retrieval toward memories encoded in similar states:

```python
ranked_ids, context = bridge.modulate_retrieval(
    query_embedding=query,
    candidate_contexts=encoding_contexts,
    top_k=10
)

# context.query_embedding: Query vector
# context.query_nt_state: Current NT state
# context.query_cognitive_state: Current cognitive state
# context.retrieved_ids: Ranked memory IDs
# context.state_similarities: Similarity scores
```

**Ranking**:
1. Compute NT similarity: `1 - ||current_nt - candidate_nt|| / sqrt(6)`
2. Cognitive bonus: +0.2 if cognitive state matches
3. Weight by state_similarity_weight (0.2)
4. Rank and return top-k

## Learning Signal Integration

Route learning signals between memory and NCA:

```python
signal = bridge.compute_learning_signal(
    memory_id=episode.id,
    outcome=0.8,  # [0, 1]
    encoding_context=context
)

# signal["memory_id"]: UUID
# signal["outcome"]: float
# signal["rpe"]: Reward prediction error
# signal["coupling_gradient"]: NCA coupling gradient
# signal["effective_lr"]: Modulated learning rate
```

**Learning Rate Modulation**:
- FOCUS: 1.5x boost
- REST: 0.5x reduction
- CONSOLIDATE: 2.0x strong boost

**Coupling Update**:
```python
coupling.update_from_rpe(nt_state, rpe)
```

## Consolidation Triggering

Select memories for replay:

```python
replay_ids = bridge.trigger_consolidation()
# Returns list[UUID] sorted by score
```

**Scoring** (from last 1000 encoded):
- Recency: `exp(-age_hours)`
- Energy: `min(energy/10, 1.0)` (prefer unstable)
- Score: recency × energy
- Return top-k (default 10)

## Simulation Step

Synchronize bridge with simulation time:

```python
bridge.step(dt=0.01)
# 1. Step neural field
# 2. Get current NT state
# 3. Update state manager
# 4. Trigger consolidation if in CONSOLIDATE state
```

## Cognitive States

| State | Effect on Encoding | Effect on Retrieval | Learning Rate |
|-------|-------------------|---------------------|---------------|
| ALERT | Normal | Normal | 1.0x |
| FOCUS | +1.3x context | Exact matching | 1.5x |
| REST | Normal | Relaxed | 0.5x |
| EXPLORE | +Noise | Diverse results | 1.0x |
| CONSOLIDATE | N/A | N/A | 2.0x |

## Integration Points

### With NCA Module

```python
from t4dm.nca import (
    NeuralFieldSolver,
    LearnableCoupling,
    StateTransitionManager,
    EnergyLandscape,
    CognitiveState
)

# Neural field provides NT state
nt_state = neural_field.get_current_state()

# Coupling updated via RPE
coupling.update_from_rpe(nt_state, rpe)

# Energy used for consolidation scoring
energy = energy_landscape.compute_energy(nt_state)
```

### With Learning Module

```python
from t4dm.learning import DopamineSystem

# Dopamine provides RPE
rpe = dopamine.compute_rpe(actual_reward, expected_reward)

# Three-factor learning compatibility
effective_lr = eligibility × neuromod × |rpe|
```

### With Memory Module

```python
# Before storage
augmented, context = bridge.augment_encoding(embedding, memory_id)
episode.embedding = augmented
episode.encoding_context = context

# Before retrieval
ranked_ids, _ = bridge.modulate_retrieval(query, candidates)
```

## Data Structures

### EncodingContext

```python
@dataclass
class EncodingContext:
    memory_id: UUID
    embedding: np.ndarray        # Original embedding
    nt_state: np.ndarray         # 6-dim NT state at encoding
    cognitive_state: CognitiveState
    timestamp: datetime
    energy: float                # System energy at encoding
```

### RetrievalContext

```python
@dataclass
class RetrievalContext:
    query_embedding: np.ndarray
    query_nt_state: np.ndarray
    query_cognitive_state: CognitiveState
    retrieved_ids: list[UUID]
    state_similarities: list[float]
    timestamp: datetime
```

## Statistics

```python
stats = bridge.get_stats()
# {
#     "encoding_history_size": 500,
#     "retrieval_history_size": 200,
#     "current_nt_state": [0.5, 0.4, 0.6, ...],
#     "current_cognitive_state": "FOCUS",
#     "config": {...}
# }
```

## Testing

```bash
# Run bridge tests
pytest tests/bridge/ -v

# With coverage
pytest tests/bridge/ --cov=t4dm.bridge
```

**Test Coverage**: 21 tests (315 lines)

## Installation

```bash
# Bridge included in core
pip install -e "."
```

## Public API

```python
# Main class
MemoryNCABridge

# Configuration
BridgeConfig

# Context dataclasses
EncodingContext
RetrievalContext
```

## Design Patterns

| Pattern | Usage |
|---------|-------|
| Lazy Initialization | All NCA components optional |
| Context Recording | Store encoding/retrieval history |
| State-Dependent | Same input → different output based on state |
| Multi-Factor Signal | RPE + coupling + state → learning |

## Current Status

**Integration Level**: STUB (90% ready)

**Missing Integrations**:
- Direct memory encoding/retrieval hookup
- Full eligibility trace integration
- Sleep/consolidation cycle triggering

## Biological Grounding

| Component | Brain Mechanism |
|-----------|-----------------|
| NT projection | Modulatory influence on encoding |
| Cognitive state boost | Attention-gated plasticity |
| State-dependent retrieval | Context reinstatement |
| RPE → coupling | Dopaminergic learning signal |
| Energy-based selection | Consolidation priority |
