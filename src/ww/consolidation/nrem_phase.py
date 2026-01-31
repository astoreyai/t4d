"""P4-02: NREM phase — spiking replay + STDP + κ boost + T4DX compaction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ww.storage.t4dx.engine import T4DXEngine
from ww.storage.t4dx.types import ItemRecord

logger = logging.getLogger(__name__)


@dataclass
class NREMConfig:
    """NREM consolidation configuration."""

    max_replay_items: int = 100
    kappa_boost: float = 0.05
    min_importance: float = 0.3
    kappa_ceiling: float = 0.3  # only replay items with κ < this
    replay_passes: int = 3  # number of replay sweeps
    stdp_lr: float = 0.01


@dataclass
class NREMResult:
    """Results from NREM consolidation."""

    replayed: int = 0
    kappa_updated: int = 0
    edges_strengthened: int = 0
    segment_id: int | None = None


class NREMPhase:
    """NREM consolidation: replay high-PE items through spiking blocks.

    1. SCAN(κ < 0.3, importance > 0.3) from T4DX
    2. Replay through spiking cortical blocks
    3. STDP strengthening via UPDATE_EDGE_WEIGHT
    4. κ += 0.05 via UPDATE_FIELDS
    5. Trigger Compactor.nrem_compact()
    """

    def __init__(
        self,
        engine: T4DXEngine,
        spiking_stack: Any = None,
        cfg: NREMConfig | None = None,
    ) -> None:
        self.engine = engine
        self.spiking = spiking_stack
        self.cfg = cfg or NREMConfig()

    def run(self) -> NREMResult:
        """Execute NREM consolidation phase."""
        result = NREMResult()

        # 1. Select items for replay
        candidates = self.engine.scan(
            kappa_max=self.cfg.kappa_ceiling,
        )

        # Sort by importance (high PE items first)
        candidates.sort(key=lambda r: r.importance, reverse=True)
        candidates = [
            c for c in candidates if c.importance >= self.cfg.min_importance
        ][:self.cfg.max_replay_items]

        if not candidates:
            logger.info("NREM: no candidates for replay")
            return result

        # 2. Replay through spiking blocks (if available)
        for _pass in range(self.cfg.replay_passes):
            for rec in candidates:
                self._replay_item(rec, result)

        # 3. Boost κ for all replayed items
        for rec in candidates:
            new_kappa = min(1.0, rec.kappa + self.cfg.kappa_boost)
            self.engine.update_fields(rec.id, {"kappa": new_kappa})
            result.kappa_updated += 1

        result.replayed = len(candidates)

        # 4. Trigger NREM compaction
        new_sid = self.engine.nrem_compact(kappa_boost=self.cfg.kappa_boost)
        result.segment_id = new_sid

        logger.info(
            "NREM complete: replayed=%d, κ_updated=%d, edges=%d, segment=%s",
            result.replayed, result.kappa_updated, result.edges_strengthened,
            result.segment_id,
        )
        return result

    def _replay_item(self, rec: ItemRecord, result: NREMResult) -> None:
        """Replay a single item through spiking blocks."""
        if self.spiking is None or not rec.vector:
            return

        vec = torch.tensor([rec.vector], dtype=torch.float32)
        # Reshape for spiking: (batch=1, seq=1, dim)
        x = vec.unsqueeze(1)

        try:
            with torch.no_grad():
                out, states, metrics = self.spiking(x)
        except Exception as e:
            logger.debug("Spiking replay failed for %s: %s", rec.id.hex()[:8], e)
            return

        # STDP: strengthen edges between this item and neighbors
        edges = self.engine.traverse(rec.id, direction="out")
        for edge in edges:
            self.engine.update_edge_weight(
                edge.source_id, edge.target_id, edge.edge_type,
                self.cfg.stdp_lr,
            )
            result.edges_strengthened += 1
