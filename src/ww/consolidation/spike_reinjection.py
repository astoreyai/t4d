"""P4-05: Spike reinjection loop — T4DX read → spike → learn → T4DX write."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ww.storage.t4dx.engine import T4DXEngine
from ww.storage.t4dx.types import ItemRecord

logger = logging.getLogger(__name__)


@dataclass
class ReinjectionConfig:
    """Spike reinjection configuration."""

    alpha: float = 0.7  # blend: α·live + (1-α)·replay
    stdp_lr: float = 0.01
    kappa_increment: float = 0.03
    batch_size: int = 16


@dataclass
class ReinjectionResult:
    """Results from spike reinjection."""

    items_replayed: int = 0
    edges_updated: int = 0
    kappa_updated: int = 0


class SpikeReinjection:
    """Full loop: T4DX read → spiking blocks → STDP → T4DX write back.

    Supports interleaved replay: α·z_live + (1-α)·z_replay
    where live input comes from current context and replay from T4DX.
    """

    def __init__(
        self,
        engine: T4DXEngine,
        spiking_stack: Any,
        cfg: ReinjectionConfig | None = None,
    ) -> None:
        self.engine = engine
        self.spiking = spiking_stack
        self.cfg = cfg or ReinjectionConfig()

    def replay(
        self,
        item_ids: list[bytes],
        live_input: Tensor | None = None,
    ) -> ReinjectionResult:
        """Replay items through spiking blocks with optional live interleaving."""
        result = ReinjectionResult()

        for item_id in item_ids:
            rec = self.engine.get(item_id)
            if rec is None or not rec.vector:
                continue

            replay_vec = torch.tensor([rec.vector], dtype=torch.float32).unsqueeze(1)

            # Interleave with live input if available
            if live_input is not None:
                x = self.cfg.alpha * live_input + (1 - self.cfg.alpha) * replay_vec
            else:
                x = replay_vec

            try:
                with torch.no_grad():
                    out, states, metrics = self.spiking(x)
            except Exception as e:
                logger.debug("Reinjection failed for %s: %s", item_id.hex()[:8], e)
                continue

            result.items_replayed += 1

            # STDP: update edge weights
            edges = self.engine.traverse(item_id, direction="out")
            for edge in edges:
                self.engine.update_edge_weight(
                    edge.source_id, edge.target_id, edge.edge_type,
                    self.cfg.stdp_lr,
                )
                result.edges_updated += 1

            # κ boost
            new_kappa = min(1.0, rec.kappa + self.cfg.kappa_increment)
            self.engine.update_fields(item_id, {"kappa": new_kappa})
            result.kappa_updated += 1

        logger.info(
            "Reinjection: replayed=%d, edges=%d, κ_updated=%d",
            result.items_replayed, result.edges_updated, result.kappa_updated,
        )
        return result

    def select_for_replay(
        self,
        max_items: int = 50,
        kappa_max: float = 0.3,
    ) -> list[bytes]:
        """Select high-importance, low-κ items for replay."""
        candidates = self.engine.scan(kappa_max=kappa_max)
        candidates.sort(key=lambda r: r.importance, reverse=True)
        return [r.id for r in candidates[:max_items]]
