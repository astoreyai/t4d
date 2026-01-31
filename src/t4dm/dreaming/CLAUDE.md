# Dreaming Module
**Path**: `/mnt/projects/t4d/t4dm/src/ww/dreaming/`

## What
REM-sleep-inspired imagination system for memory consolidation. Generates dream trajectories from high-error episodes, evaluates dream quality across multiple dimensions, and trains the world model predictor on imagined experience. Inspired by DreamerV3 and biological REM sleep.

## How
- **DreamingSystem** (P3-1): Generates trajectories from seed episodes using ContextEncoder + LatentPredictor + noise injection. Terminates when confidence < 0.3 or coherence < 0.5 or max 15 steps reached.
- **DreamQualityEvaluator** (P3-2): Scores dreams on 4 weighted dimensions:
  - Coherence (0.3): prediction confidence + manifold alignment
  - Smoothness (0.2): step-to-step similarity stability
  - Novelty (0.25): distance from known reference embeddings
  - Informativeness (0.25): optimal challenge (peak at confidence ~0.6)
- **DreamConsolidation** (P3-3): Orchestrates dream cycles during REM phase -- seeds from high-error episodes, generates dreams, filters by quality (>0.5), trains predictor on high-quality dream steps at reduced LR (0.0005), boosts seed episode priority.

## Why
Enables learning from imagined experience without real-world interaction. High-error episodes get extra training through dream replay, improving the world model predictor and boosting consolidation priority for subsequent NREM cycles.

## Key Files
| File | Purpose |
|------|---------|
| `trajectory.py` | DreamingSystem, DreamTrajectory, DreamStep (~450 lines) |
| `quality.py` | DreamQualityEvaluator, multi-dimensional scoring (~345 lines) |
| `consolidation.py` | DreamConsolidation, REM integration (~425 lines) |

## Data Flow
```
High-error episodes (from NREM/PredictionTracker)
    -> DreamingSystem.dream() -> DreamTrajectory (up to 15 steps)
    -> DreamQualityEvaluator.evaluate() -> quality score
    -> DreamConsolidation (if quality > 0.5):
        -> Train predictor on dream steps (LR=0.0005)
        -> Boost seed episode priority for next NREM cycle
```

## Integration Points
- **consolidation**: Called during REM phase of sleep cycle
- **prediction**: Uses ContextEncoder + LatentPredictor for trajectory generation
- **memory**: Seeds from episodic memory high-error episodes

## Learning Modalities
- **Imagination-based**: Learn from generated (not real) experience
- **Reduced learning rate**: Dream training uses 0.0005 vs normal rates to prevent hallucination overfitting
- **Priority feedback loop**: Good dreams boost seed priority, creating a virtuous cycle with NREM replay
