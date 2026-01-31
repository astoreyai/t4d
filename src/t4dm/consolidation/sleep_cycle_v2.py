"""P4-06: Full sleep cycle v2 — orchestrates NREM → REM → PRUNE with T4DX."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from t4dm.consolidation.nrem_phase import NREMConfig, NREMPhase, NREMResult
from t4dm.consolidation.prune_phase import PruneConfig, PrunePhase, PruneResult
from t4dm.consolidation.rem_phase import REMConfig, REMPhase, REMResult
from t4dm.consolidation.spike_reinjection import SpikeReinjection, ReinjectionConfig
from t4dm.storage.t4dx.engine import T4DXEngine

logger = logging.getLogger(__name__)


@dataclass
class SleepCycleV2Config:
    """Full sleep cycle configuration."""

    num_cycles: int = 4  # biological: 4-5 NREM-REM cycles per night
    nrem: NREMConfig = field(default_factory=NREMConfig)
    rem: REMConfig = field(default_factory=REMConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    reinjection: ReinjectionConfig = field(default_factory=ReinjectionConfig)
    enable_reinjection: bool = True


@dataclass
class SleepCycleV2Result:
    """Aggregate results from full sleep cycle."""

    nrem_results: list[NREMResult] = field(default_factory=list)
    rem_results: list[REMResult] = field(default_factory=list)
    prune_result: PruneResult | None = None
    total_replayed: int = 0
    total_prototypes: int = 0
    total_pruned: int = 0
    duration_seconds: float = 0.0


class SleepCycleV2:
    """Orchestrates NREM → REM → PRUNE cycles with T4DX compactor.

    Mirrors biological sleep: multiple NREM-REM cycles followed by
    a final prune phase. Each NREM includes spiking replay via
    SpikeReinjection if enabled.
    """

    def __init__(
        self,
        engine: T4DXEngine,
        spiking_stack: Any = None,
        cfg: SleepCycleV2Config | None = None,
    ) -> None:
        self.engine = engine
        self.spiking = spiking_stack
        self.cfg = cfg or SleepCycleV2Config()

        self._nrem = NREMPhase(engine, spiking_stack, self.cfg.nrem)
        self._rem = REMPhase(engine, self.cfg.rem)
        self._prune = PrunePhase(engine, self.cfg.prune)
        self._reinjection = (
            SpikeReinjection(engine, spiking_stack, self.cfg.reinjection)
            if spiking_stack and self.cfg.enable_reinjection
            else None
        )

    def run(self) -> SleepCycleV2Result:
        """Execute full sleep cycle: N × (NREM + REM) + PRUNE."""
        result = SleepCycleV2Result()
        start = time.time()

        for cycle in range(self.cfg.num_cycles):
            logger.info("Sleep cycle %d/%d", cycle + 1, self.cfg.num_cycles)

            # NREM phase
            nrem_result = self._nrem.run()
            result.nrem_results.append(nrem_result)
            result.total_replayed += nrem_result.replayed

            # Optional spike reinjection during NREM
            if self._reinjection:
                ids = self._reinjection.select_for_replay(max_items=20)
                if ids:
                    self._reinjection.replay(ids)

            # REM phase
            rem_result = self._rem.run()
            result.rem_results.append(rem_result)
            result.total_prototypes += rem_result.prototypes_created

        # Final PRUNE
        prune_result = self._prune.run()
        result.prune_result = prune_result
        result.total_pruned = prune_result.deleted

        result.duration_seconds = time.time() - start

        logger.info(
            "Sleep cycle v2 complete: replayed=%d, prototypes=%d, pruned=%d, %.1fs",
            result.total_replayed, result.total_prototypes, result.total_pruned,
            result.duration_seconds,
        )
        return result
