"""P4-04: Prune phase — GC tombstoned + low-κ items via T4DX compaction."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ww.storage.t4dx.engine import T4DXEngine

logger = logging.getLogger(__name__)


@dataclass
class PruneConfig:
    """Prune phase configuration."""

    kappa_threshold: float = 0.05
    importance_threshold: float = 0.1
    max_age_days: float | None = None  # None = no age limit


@dataclass
class PruneResult:
    """Results from prune phase."""

    deleted: int = 0
    tombstoned: int = 0


class PrunePhase:
    """Prune consolidation: remove low-value items.

    1. DELETE items where κ < threshold AND importance < threshold
    2. Trigger Compactor.prune() to rewrite segments without tombstoned items
    """

    def __init__(
        self,
        engine: T4DXEngine,
        cfg: PruneConfig | None = None,
    ) -> None:
        self.engine = engine
        self.cfg = cfg or PruneConfig()

    def run(self) -> PruneResult:
        """Execute prune phase."""
        result = PruneResult()

        # Scan for low-value items
        candidates = self.engine.scan(
            kappa_max=self.cfg.kappa_threshold,
        )

        for rec in candidates:
            if rec.importance < self.cfg.importance_threshold:
                self.engine.delete(rec.id)
                result.tombstoned += 1

        # Trigger compaction to physically remove
        max_age_sec = (
            self.cfg.max_age_days * 86400.0
            if self.cfg.max_age_days is not None
            else None
        )
        removed = self.engine.prune(
            max_age_seconds=max_age_sec,
            min_kappa=0.0,
        )
        result.deleted = removed

        logger.info(
            "Prune complete: tombstoned=%d, deleted=%d",
            result.tombstoned, result.deleted,
        )
        return result
