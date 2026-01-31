"""P4-08: Homeostatic scaling integration with T4DX batch_scale_weights."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from t4dm.storage.t4dx.engine import T4DXEngine

logger = logging.getLogger(__name__)


@dataclass
class HomeostaticV2Config:
    """Homeostatic scaling configuration."""

    target_firing_rate: float = 0.05  # 5% target
    gamma: float = 0.1  # scaling exponent
    bcm_threshold: float = 0.5  # BCM sliding threshold
    min_weight: float = 0.01
    max_weight: float = 0.99
    sample_size: int = 100


@dataclass
class HomeostaticV2Result:
    """Results from homeostatic scaling."""

    scale_factor: float = 1.0
    estimated_firing_rate: float = 0.0
    applied: bool = False


class HomeostaticScalingV2:
    """Homeostatic synaptic scaling via T4DX batch_scale_weights.

    w ← w · (r* / r̂)^γ
    where r* = target firing rate, r̂ = estimated rate, γ = scaling exponent.

    Uses BCM sliding threshold to determine LTP/LTD transition.
    """

    def __init__(
        self,
        engine: T4DXEngine,
        cfg: HomeostaticV2Config | None = None,
    ) -> None:
        self.engine = engine
        self.cfg = cfg or HomeostaticV2Config()

    def estimate_firing_rate(self) -> float:
        """Estimate network firing rate from edge weights."""
        # Sample edges from recent items
        items = self.engine.scan()
        if not items:
            return 0.0

        sample = items[:self.cfg.sample_size]
        weights = []
        for rec in sample:
            edges = self.engine.traverse(rec.id, direction="out")
            for e in edges:
                weights.append(e.weight)

        if not weights:
            return 0.0

        # Firing rate proxy: fraction of edges above BCM threshold
        arr = np.array(weights)
        return float(np.mean(arr > self.cfg.bcm_threshold))

    def run(self) -> HomeostaticV2Result:
        """Apply homeostatic scaling."""
        result = HomeostaticV2Result()

        rate = self.estimate_firing_rate()
        result.estimated_firing_rate = rate

        if rate == 0.0:
            logger.info("Homeostatic: no edges to scale")
            return result

        # Compute scale factor: (target / actual)^gamma
        ratio = self.cfg.target_firing_rate / max(rate, 1e-6)
        factor = ratio ** self.cfg.gamma
        factor = max(0.5, min(2.0, factor))  # clamp to prevent extreme scaling

        result.scale_factor = factor

        if abs(factor - 1.0) < 0.01:
            logger.info("Homeostatic: rate=%.3f, no scaling needed", rate)
            return result

        self.engine.batch_scale_weights(factor)
        result.applied = True

        logger.info(
            "Homeostatic: rate=%.3f, target=%.3f, scale=%.3f",
            rate, self.cfg.target_firing_rate, factor,
        )
        return result
