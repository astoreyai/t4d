"""
T4DM Sleep Replay System (internally called "Dreaming").

P3: Imagination-based learning during consolidation.

IMPORTANT TERMINOLOGY CLARIFICATION:
    "Dreaming" here refers to *generative replay during simulated REM sleep*,
    inspired by DreamerV3 (Hafner et al., 2023) world model training.
    This is a computational process for memory consolidation, NOT a claim
    about subjective experience or phenomenal consciousness.

Inspired by DreamerV3 and biological REM sleep patterns:
- Generate imagined trajectories from high-error memories
- Evaluate trajectory quality for consolidation priority
- Train world model on generated outcomes
- Create abstract concepts from trajectory patterns

Modules:
- trajectory: Generative trajectory generation
- quality: Trajectory quality evaluation metrics
- consolidation: Integration with sleep consolidation
"""

from t4dm.dreaming.consolidation import (
    DreamConsolidation,
    DreamConsolidationConfig,
    DreamReplayEvent,
)
from t4dm.dreaming.quality import (
    DreamQuality,
    DreamQualityEvaluator,
    QualityConfig,
)
from t4dm.dreaming.trajectory import (
    DreamingConfig,
    DreamingSystem,
    DreamTrajectory,
)

__all__ = [
    # Trajectory Generation (P3-1)
    "DreamTrajectory",
    "DreamingSystem",
    "DreamingConfig",
    # Quality Evaluation (P3-2)
    "DreamQuality",
    "DreamQualityEvaluator",
    "QualityConfig",
    # Consolidation Integration (P3-3)
    "DreamConsolidation",
    "DreamConsolidationConfig",
    "DreamReplayEvent",
]
