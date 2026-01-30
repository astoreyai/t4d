"""
World Weaver Dreaming System.

P3: Imagination-based learning during consolidation.

Inspired by DreamerV3 and biological REM sleep:
- Generate imagined trajectories from high-error memories
- Evaluate dream quality for consolidation priority
- Train world model on dream outcomes
- Create abstract concepts from dream patterns

Modules:
- trajectory: Dream trajectory generation
- quality: Dream quality evaluation metrics
- consolidation: Integration with sleep consolidation
"""

from ww.dreaming.consolidation import (
    DreamConsolidation,
    DreamConsolidationConfig,
    DreamReplayEvent,
)
from ww.dreaming.quality import (
    DreamQuality,
    DreamQualityEvaluator,
    QualityConfig,
)
from ww.dreaming.trajectory import (
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
