# Prediction
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/prediction/`

## What
JEPA-style latent prediction, hierarchical multi-timescale prediction, active inference, and predictive coding hierarchy inspired by DreamerV3, Friston's Free Energy Principle, and Rao & Ballard.

## How
- **Context Encoder** (`context_encoder.py`): Encodes recent episode embeddings into `EncodedContext` for prediction input.
- **Latent Predictor** (`latent_predictor.py`): MLP predicting next latent state in embedding space (not raw content). Produces `Prediction` with confidence and `PredictionError`.
- **Hierarchical Predictor** (`hierarchical_predictor.py`): Multi-timescale prediction at fast (1-step), medium (5-step), and slow (15-step) horizons.
- **Predictive Coding** (`predictive_coding.py`): Rao & Ballard / Friston hierarchical prediction error minimization with top-down predictions and bottom-up errors across `PredictiveLevel` layers.
- **Active Inference** (`active_inference.py`): Free energy minimization with precision-weighted prediction errors, belief updating (dmu/dt), and action selection via expected free energy.
- **Tracker** (`prediction_tracker.py`): Tracks prediction errors per episode for consolidation prioritization.
- **Integration** (`prediction_integration.py`): Connects prediction to memory lifecycle -- high-error episodes get priority replay.

## Why
Predicting in latent space avoids snowball error from raw content prediction. Prediction errors drive consolidation priority (surprising memories are replayed more), mirroring biological memory consolidation.

## Key Files
| File | Purpose |
|------|---------|
| `latent_predictor.py` | Core JEPA-style next-state prediction |
| `active_inference.py` | Free energy minimization and action selection |
| `predictive_coding.py` | Hierarchical prediction error minimization |
| `hierarchical_predictor.py` | Multi-timescale (fast/medium/slow) prediction |
| `prediction_integration.py` | Connects prediction errors to consolidation |

## Data Flow
```
Episode embeddings --> ContextEncoder --> EncodedContext
EncodedContext --> LatentPredictor --> Prediction
Actual next state vs Prediction --> PredictionError
PredictionError --> PredictionTracker --> consolidation priority
```

## Learning Modalities
- **Predictive coding**: Top-down generative model + bottom-up error signals
- **Active inference**: Precision-weighted free energy minimization
- **Multi-timescale**: Fast/medium/slow prediction horizons for different temporal patterns
