# Prediction Module

**6 files | ~2,000 lines | Centrality: 1**

The prediction module implements JEPA-style (Joint Embedding Predictive Architecture) latent prediction for anticipatory memory retrieval and dreaming.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LATENT PREDICTION SYSTEM                           │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                      P2-1: CONTEXT ENCODER                          ││
│  │  [Recent Episodes: n×1024] → Position + Attention → [Context: 1024] ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                      P2-2: LATENT PREDICTOR                         ││
│  │  [Context: 1024] → 2-layer MLP + Residual → [Predicted: 1024]       ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                      P2-3: PREDICTION TRACKER                       ││
│  │  Error computation → Z-score normalization → Priority queue         ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                      P2-4: INTEGRATION                              ││
│  │  Memory lifecycle hooks → Consolidation priority → Dream seeds      ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `context_encoder.py` | ~350 | Context representation with attention aggregation |
| `latent_predictor.py` | ~500 | MLP predicting next latent state |
| `prediction_tracker.py` | ~400 | Error tracking and priority scoring |
| `prediction_integration.py` | ~400 | Memory lifecycle bridge |
| `hierarchical_predictor.py` | ~315 | Multi-timescale prediction (fast/medium/slow) |
| `__init__.py` | ~70 | Public API exports |

## Core Principle: JEPA

**Key Insight**: Predict in embedding space, NOT raw content space.

- Avoids "snowball error" of pixel-level prediction drift
- Prediction error is the primary learning signal
- Biologically grounded: PFC predicts future states for planning

## Components

### ContextEncoder (P2-1)

Compresses recent episodes into a single context vector:

```python
from ww.prediction import ContextEncoder, ContextEncoderConfig

config = ContextEncoderConfig(
    embedding_dim=1024,       # BGE-M3 compatibility
    hidden_dim=512,           # Compression
    context_dim=1024,         # Output dimension
    max_context_length=8,     # Hippocampal CA3 limit
    aggregation="attention"   # "attention" | "mean" | "last"
)

encoder = ContextEncoder(config)
context = encoder.encode([ep1_embedding, ep2_embedding, ep3_embedding])
# context.vector: np.ndarray[1024]
# context.attention_weights: list[float]
```

**Aggregation Methods**:
- `attention`: Global learnable query attends to all episodes (default)
- `mean`: Uniform weighting
- `last`: Most recent only

### LatentPredictor (P2-2)

MLP predicting next latent state:

```python
from ww.prediction import LatentPredictor, LatentPredictorConfig

config = LatentPredictorConfig(
    context_dim=1024,
    hidden_dim=512,
    output_dim=1024,
    num_hidden_layers=2,
    activation="relu",       # "relu" | "gelu" | "tanh"
    learning_rate=0.001,
    residual=True,
    layer_norm=True
)

predictor = LatentPredictor(config)

# Predict next state
prediction = predictor.predict(context_vector)
# prediction.embedding: np.ndarray[1024]
# prediction.confidence: float (0-1)

# Compute error when outcome known
error = predictor.compute_error(prediction, actual_embedding, episode_id)
# error.l2_error: float
# error.cosine_error: float
# error.combined_error: float

# Train on error signal
loss = predictor.train_step(context_vector, target_embedding, lr=0.001)
```

**Loss Function**: `L = 0.5 × L2_loss + 0.5 × cosine_loss`

### PredictionTracker (P2-3)

Tracks pending predictions and prioritizes high-error episodes:

```python
from ww.prediction import PredictionTracker, TrackerConfig

config = TrackerConfig(
    max_pending_predictions=1000,
    max_error_history=10000,
    prediction_timeout=timedelta(hours=24),
    normalize_errors=True,
    min_error_for_priority=0.1,
    priority_decay=0.95      # Per-hour decay
)

tracker = PredictionTracker(config)

# Make prediction before episode stored
tracked = tracker.make_prediction(context_episodes, predictor, encoder)

# Resolve when actual embedding known
error = tracker.resolve_prediction(before_id, actual_embedding, actual_id)

# Get high-error episodes for consolidation
priorities = tracker.get_high_error_episodes(k=10)
# [(episode_id, priority_score), ...]
```

**Priority Scoring**:
1. Compute combined error (L2 + cosine)
2. Update running statistics (Welford algorithm)
3. Z-score normalization: `z = (error - mean) / std`
4. Time decay: `priority_t = priority × 0.95^(hours)`

### PredictionIntegration (P2-4)

Bridges prediction to memory lifecycle:

```python
from ww.prediction import PredictionIntegration, create_prediction_integration

integration = create_prediction_integration(
    context_size=5,
    context_hours=4,
    train_on_resolve=True,
    train_during_replay=True,
    high_error_threshold=0.5
)

# Hook into episode creation
integration.on_episode_created(episode, recent_context_embeddings)

# Hook into episode storage (resolves prediction)
error = integration.on_episode_stored(episode)

# Hook into consolidation replay
integration.on_replay(episode, context_embeddings)

# Get priority episodes for replay
priorities = integration.get_priority_episodes(k=10)

# Anticipatory retrieval hint
predicted_next = integration.predict_next(recent_embeddings)
```

### HierarchicalPredictor (P4-1)

Multi-timescale prediction:

```python
from ww.prediction import HierarchicalPredictor, HierarchicalConfig

config = HierarchicalConfig(
    # Horizons
    fast_horizon=1,     # Next step
    medium_horizon=5,   # ~5 steps
    slow_horizon=15,    # Long horizon

    # Learning rates (gradient decay)
    fast_lr=0.01,       # Fast plasticity
    medium_lr=0.001,
    slow_lr=0.0001,     # Long-term memory

    # Confidence weights
    fast_weight=0.5,
    medium_weight=0.3,
    slow_weight=0.2
)

hierarchical = HierarchicalPredictor(config)

# Generate 3 simultaneous predictions
prediction = hierarchical.predict()
# prediction.fast: Prediction
# prediction.medium: Prediction
# prediction.slow: Prediction
# prediction.combined_confidence: float

# Add new episode (resolves matured predictions)
errors = hierarchical.add_episode(episode_id, embedding)
# errors.fast_error: PredictionError | None
# errors.medium_error: PredictionError | None
# errors.slow_error: PredictionError | None
```

## Data Flow

```
Episode Created (embedding unknown)
    │
    ├─ on_episode_created()
    │   ├─ encoder.encode(context) → context_vector
    │   ├─ predictor.predict(context) → Prediction
    │   └─ tracker.make_prediction() → pending
    │
Episode Stored (embedding known)
    │
    ├─ on_episode_stored()
    │   ├─ tracker.resolve_prediction()
    │   ├─ predictor.compute_error() → PredictionError
    │   ├─ predictor.train_step() [if enabled]
    │   ├─ z-score → priority_scores
    │   └─ tag episode with prediction_error
    │
Consolidation (replay)
    │
    ├─ get_high_error_episodes(k=10)
    │   └─ Seeds for REM dreams
    │
    ├─ on_replay()
    │   └─ predictor.train_step()
    │
Dreaming
    │
    └─ High-error episodes → dream trajectories
```

## Integration Points

### With Consolidation

```python
# consolidation/sleep.py
from ww.prediction import PredictionTracker

# REM phase seeds from high-error episodes
high_error = tracker.get_high_error_episodes(k=10)
```

### With Dreaming

```python
# dreaming/trajectory.py
from ww.prediction import ContextEncoder, LatentPredictor

# Dream generation uses prediction for trajectories
dreamer = DreamingSystem(encoder, predictor)
dream = dreamer.dream(seed_embedding)
```

### With Learning

```python
# learning/dopamine.py
# Prediction error → RPE signal
rpe = predictor.compute_error(prediction, actual).combined_error

# Three-factor learning gate
effective_lr = eligibility × neuromod × |rpe|
```

## Biological Grounding

| Component | Brain Region | Function |
|-----------|--------------|----------|
| Context Encoder | Hippocampal CA3 | Compressed representation |
| Attention Aggregation | Cholinergic modulation | Context weighting |
| Prediction Error | PFC | Model-based prediction |
| Priority Queue | Hippocampus | REM dream selection |
| Hierarchical Timescales | Cerebellum/PFC | Multi-scale planning |

## Configuration Reference

### ContextEncoderConfig

```python
embedding_dim: int = 1024         # Input dimension
hidden_dim: int = 512             # Projection dimension
context_dim: int = 1024           # Output dimension
max_context_length: int = 8       # Window size
use_position_encoding: bool = True
aggregation: str = "attention"
attention_heads: int = 4
dropout: float = 0.1
layer_norm: bool = True
```

### LatentPredictorConfig

```python
context_dim: int = 1024
hidden_dim: int = 512
output_dim: int = 1024
num_hidden_layers: int = 2
activation: str = "relu"
learning_rate: float = 0.001
weight_decay: float = 1e-4
momentum: float = 0.9
prediction_horizon: int = 1
layer_norm: bool = True
residual: bool = True
dropout: float = 0.1
```

## Performance

| Operation | Speed | Complexity |
|-----------|-------|------------|
| Prediction | >100/sec | O(1) forward pass |
| Training | >50 steps/sec | O(1) backprop |
| Context Encoding | >100/sec | O(n), n ≤ 8 |
| Error computation | O(1) | Cosine + L2 |

## Testing

```bash
# Run prediction tests
pytest tests/prediction/ -v

# With coverage
pytest tests/prediction/ --cov=ww.prediction

# Benchmarks
pytest tests/prediction/ -v -m benchmark
```

**Test Coverage**: 49 unit tests + 10 benchmarks

## Public API

```python
# Core classes
ContextEncoder, ContextEncoderConfig, EncodedContext
LatentPredictor, LatentPredictorConfig, Prediction, PredictionError
PredictionTracker, TrackerConfig
PredictionIntegration, PredictionIntegrationConfig
HierarchicalPredictor, HierarchicalConfig, HierarchicalPrediction

# Factory
create_prediction_integration
```

## References

- LeCun (2022) "A Path Towards Autonomous Machine Intelligence" (JEPA)
- Schuck & Niv (2020) "Sequential replay and generalization"
- Buzsáki (2015) "Hippocampal sharp-wave ripple"
