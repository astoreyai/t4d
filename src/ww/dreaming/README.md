# Dreaming Module

**4 files | ~1,260 lines | Centrality: 0**

The dreaming module implements REM-sleep-inspired imagination for memory consolidation, generating dream trajectories from high-error episodes and training predictors on imagined experience.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DREAMING SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    P3-1: DREAM TRAJECTORY                           ││
│  │  [Seed Episode] → ContextEncoder → LatentPredictor → [Trajectory]   ││
│  │  + Noise (variability) + Coherence check (manifold constraint)      ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    P3-2: QUALITY EVALUATION                         ││
│  │  Coherence + Smoothness + Novelty + Informativeness → Score         ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    P3-3/P3-4: CONSOLIDATION                         ││
│  │  High-quality dreams → Train predictor → Boost seed priority        ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `trajectory.py` | ~450 | Dream trajectory generation |
| `quality.py` | ~345 | Multi-dimensional quality evaluation |
| `consolidation.py` | ~425 | Sleep consolidation integration |
| `__init__.py` | ~47 | Public API exports |

## Dream Trajectory Generation

### DreamingSystem

Generate imagined trajectories from seed episodes:

```python
from ww.dreaming import DreamingSystem, DreamingConfig

config = DreamingConfig(
    max_dream_length=15,           # DreamerV3 benchmark
    confidence_threshold=0.3,      # Prediction certainty floor
    coherence_threshold=0.5,       # Manifold alignment floor
    noise_scale=0.05               # Dream variability
)

dreamer = DreamingSystem(
    encoder=context_encoder,
    predictor=latent_predictor,
    config=config
)

# Generate dream from seed
dream = dreamer.dream(
    seed_embedding=episode.embedding,
    context_embeddings=recent_embeddings,
    seed_episode_id=episode.id
)

# dream.id: UUID
# dream.seed_episode_id: UUID
# dream.steps: list[DreamStep]
# dream.termination_reason: str
# dream.length: int
# dream.mean_confidence: float
# dream.mean_coherence: float
# dream.duration_ms: float

# Batch dreaming
dreams = dreamer.dream_batch(seed_embeddings, max_concurrent=5)

# Add reference embeddings for coherence
dreamer.add_reference_embeddings(real_episode_embeddings)

# Query dreams
recent = dreamer.get_recent_dreams(limit=10)
high_quality = dreamer.get_high_quality_dreams(
    min_confidence=0.5,
    min_coherence=0.6,
    min_length=5
)

# Statistics
stats = dreamer.get_statistics()
# {total_dreams, termination_counts, avg_length, avg_confidence}
```

### DreamTrajectory

Represents a complete imagined sequence:

```python
from ww.dreaming import DreamTrajectory, DreamStep

# Single step in dream
step = DreamStep(
    embedding=predicted_embedding,
    confidence=0.7,
    coherence=0.8
)

# Full trajectory properties
trajectory.embeddings      # [seed, step1, step2, ...]
trajectory.length          # Number of steps
trajectory.mean_confidence # Average prediction confidence
trajectory.mean_coherence  # Average manifold coherence
trajectory.duration_ms     # Generation time

# Termination reasons
# "low_confidence" - Prediction became uncertain
# "low_coherence" - Drifted from learned manifold
# "max_length" - Reached limit
# "incomplete" - Still generating
```

### Generation Algorithm

```
Input: seed_embedding, context_embeddings

1. Initialize trajectory with seed
2. Build context window (5 recent + seed)
3. FOR each step (max 15):
   a) Encode context → context_vector
   b) Predict next → predicted_embedding
   c) Add noise (0.05 scale) for variability
   d) Compute coherence with manifold
   e) Create DreamStep
   f) Check termination:
      - confidence < 0.3 → stop
      - coherence < 0.5 → stop
   g) Update context
4. Return DreamTrajectory
```

## Quality Evaluation

### DreamQualityEvaluator

Multi-dimensional assessment:

```python
from ww.dreaming import DreamQualityEvaluator, DreamQualityConfig

config = DreamQualityConfig(
    coherence_weight=0.3,
    smoothness_weight=0.2,
    novelty_weight=0.25,
    informativeness_weight=0.25,
    high_quality_threshold=0.7,
    min_quality_threshold=0.4
)

evaluator = DreamQualityEvaluator(config, reference_embeddings)

# Evaluate single dream
quality = evaluator.evaluate(dream)
# quality.coherence_score: float [0-1]
# quality.smoothness_score: float [0-1]
# quality.novelty_score: float [0-1]
# quality.informativeness_score: float [0-1]
# quality.overall_score: float (weighted)
# quality.is_high_quality: bool

# Batch evaluation
qualities = evaluator.evaluate_batch(dreams)

# Filter high-quality
high_quality = evaluator.filter_high_quality(dreams)
```

### Quality Metrics

**Coherence** (0.3 weight):
- Mean of dream's confidence + coherence
- Formula: `(confidence + coherence) / 2`

**Smoothness** (0.2 weight):
- Step-to-step similarity (expect high)
- Low variance in transitions
- Formula: `(mean_similarity + stability) / 2`

**Novelty** (0.25 weight):
- Distance from known patterns
- Uses 10 nearest reference embeddings
- Explores underrepresented regions

**Informativeness** (0.25 weight):
- Optimal confidence around 0.6 (challenging but achievable)
- Bell curve: `1 - |mean_conf - 0.6| × 2`
- Bonus for longer trajectories

## Sleep Consolidation Integration

### DreamConsolidation

Orchestrates dream-based learning during REM:

```python
from ww.dreaming import DreamConsolidation, DreamConsolidationConfig

config = DreamConsolidationConfig(
    dreams_per_cycle=5,
    seed_from_high_error=True,
    min_error_for_seed=0.3,
    min_quality_for_replay=0.5,
    priority_boost_factor=0.2,
    train_on_dreams=True,
    dream_learning_rate=0.0005      # Lower than normal
)

consolidation = DreamConsolidation(
    dreamer=dreaming_system,
    evaluator=quality_evaluator,
    tracker=prediction_tracker,
    encoder=context_encoder,
    predictor=latent_predictor,
    config=config
)

# Run dream cycle (during REM)
result = await consolidation.run_dream_cycle(
    reference_embeddings=recent_embeddings,
    high_error_episodes=nrem_high_error
)

# result.dreams_generated: int
# result.high_quality_count: int
# result.mean_quality: float
# result.training_steps: int
# result.priority_updates: int
# result.replay_events: list[DreamReplayEvent]

# Get recent dream cycles
recent = consolidation.get_recent_cycles(limit=5)
```

### Dream Cycle Workflow

```
REM Sleep Phase:

1. Set reference embeddings for coherence/novelty

2. Select dream seeds:
   - Explicit high-error episodes (from NREM)
   - High-error from PredictionTracker
   - Fallback: most recent episodes

3. FOR each seed (up to 5 per cycle):
   a) Generate dream trajectory
   b) Evaluate quality
   c) IF high-quality (>0.5):
      - Train predictor on dream steps
      - Record DreamReplayEvent
      - Boost priority of seed episode

4. Return DreamCycleResult with statistics
```

### Training on Dreams

```python
# Per high-quality dream:
for step_idx in range(1, len(dream.steps)):
    # Context: previous embeddings (window=5)
    context = dream.embeddings[max(0, step_idx-5):step_idx]

    # Target: current embedding
    target = dream.steps[step_idx].embedding

    # Train with lower learning rate (0.0005)
    encoded = encoder.encode(context)
    loss = predictor.train_step(encoded, target, lr=0.0005)
```

## Data Flow

```
Episodic Memory (high-error episodes)
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
NREM Phase              PredictionTracker
(selects candidates)    (priority queue)
    │                         │
    └────────┬────────────────┘
             │
             ▼
    DreamingSystem (generate)
             │
             ▼
    DreamQualityEvaluator (filter)
             │
             ▼
    DreamConsolidation (train)
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
Predictor         Priority
trained           boosts
             (next NREM)
```

## Biological Grounding

| Component | Brain Mechanism | Function |
|-----------|-----------------|----------|
| Dream seeds | High-error episodes | Prioritized replay |
| Trajectory generation | REM PGO waves | Imagined experience |
| Noise injection | Neural variability | Dream stochasticity |
| Coherence constraint | Hippocampal attractor | Stay on manifold |
| Quality filtering | Replay selection | Useful dreams only |
| Predictor training | Synaptic consolidation | World model improvement |
| Priority boost | Tag-and-capture | Consolidation priority |

**DreamerV3 Inspiration**:
- Learn from imagined experience
- 15-step trajectories
- Confidence-based termination
- Coherence constraints

## Configuration Reference

### DreamingConfig

```python
max_dream_length: int = 15
confidence_threshold: float = 0.3
coherence_threshold: float = 0.5
noise_scale: float = 0.05
```

### DreamQualityConfig

```python
coherence_weight: float = 0.3
smoothness_weight: float = 0.2
novelty_weight: float = 0.25
informativeness_weight: float = 0.25
high_quality_threshold: float = 0.7
min_quality_threshold: float = 0.4
```

### DreamConsolidationConfig

```python
dreams_per_cycle: int = 5
seed_from_high_error: bool = True
min_error_for_seed: float = 0.3
min_quality_for_replay: float = 0.5
priority_boost_factor: float = 0.2
train_on_dreams: bool = True
dream_learning_rate: float = 0.0005
```

## Testing

```bash
# Run dreaming tests
pytest tests/dreaming/ -v

# With coverage
pytest tests/dreaming/ --cov=ww.dreaming
```

**Test Coverage**: 495 lines, comprehensive tests for all components

## Public API

```python
# Trajectory generation
DreamingSystem, DreamingConfig
DreamTrajectory, DreamStep

# Quality evaluation
DreamQualityEvaluator, DreamQualityConfig
DreamQuality

# Consolidation
DreamConsolidation, DreamConsolidationConfig
DreamCycleResult, DreamReplayEvent
```

## References

- Stickgold (2005) "Sleep-dependent memory consolidation"
- Wilson & McNaughton (1994) "Reactivation of hippocampal memories during sleep"
- Hafner et al. (2023) "DreamerV3: World Models in Reinforcement Learning"
