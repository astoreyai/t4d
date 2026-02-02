"""
T4DM Latent Prediction Module.

P2: JEPA-style prediction in latent/embedding space.
P4-1: Hierarchical multi-timescale prediction.
P4-5: Hierarchical predictive coding (Rao & Ballard / Friston).

Inspired by DreamerV3 and JEPA (Joint Embedding Predictive Architecture):
- Predict in embedding space, not raw content space (avoids snowball error)
- Learn from prediction errors without explicit supervision
- Integrate with consolidation for prioritized replay
- Multi-horizon prediction: fast (1-step), medium (5-step), slow (15-step)

Modules:
- context_encoder: Encode context from recent episode embeddings
- latent_predictor: MLP that predicts next latent state
- prediction_tracker: Track prediction errors for episodes
- prediction_integration: Connect prediction to memory lifecycle
- hierarchical_predictor: P4-1 multi-timescale prediction
- predictive_coding: P4-5 hierarchical prediction error minimization
"""

from t4dm.prediction.context_encoder import (
    ContextEncoder,
    ContextEncoderConfig,
    EncodedContext,
)
from t4dm.prediction.hierarchical_predictor import (
    HierarchicalConfig,
    HierarchicalError,
    HierarchicalPrediction,
    HierarchicalPredictor,
)
from t4dm.prediction.latent_predictor import (
    LatentPredictor,
    LatentPredictorConfig,
    Prediction,
    PredictionError,
)
from t4dm.prediction.prediction_integration import (
    PredictionIntegration,
    PredictionIntegrationConfig,
    create_prediction_integration,
)
from t4dm.prediction.prediction_tracker import (
    PredictionTracker,
    TrackerConfig,
)
from t4dm.prediction.predictive_coding import (
    HierarchyState,
    LevelState,
    PredictionDirection,
    PredictiveCodingConfig,
    PredictiveCodingHierarchy,
    PredictiveLevel,
    create_predictive_hierarchy,
)

__all__ = [
    # Context Encoder
    "ContextEncoder",
    "ContextEncoderConfig",
    "EncodedContext",
    # Latent Predictor
    "LatentPredictor",
    "LatentPredictorConfig",
    "Prediction",
    "PredictionError",
    # Tracker
    "PredictionTracker",
    "TrackerConfig",
    # Integration (P2-4)
    "PredictionIntegration",
    "PredictionIntegrationConfig",
    "create_prediction_integration",
    # Hierarchical Prediction (P4-1)
    "HierarchicalPredictor",
    "HierarchicalConfig",
    "HierarchicalPrediction",
    "HierarchicalError",
    # Predictive Coding (Rao & Ballard / Friston)
    "PredictionDirection",
    "PredictiveCodingConfig",
    "LevelState",
    "HierarchyState",
    "PredictiveLevel",
    "PredictiveCodingHierarchy",
    "create_predictive_hierarchy",
]
