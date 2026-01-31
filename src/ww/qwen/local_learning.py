"""P3-07: Phase 2 local learning — three-factor rule for spiking params."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class LocalLearningConfig:
    """Phase 2 local learning configuration."""

    base_lr: float = 1e-3
    da_modulation: float = 1.0  # dopamine scales LR
    ach_modulation: float = 1.0  # acetylcholine gates plasticity
    freeze_qlora: bool = True
    eligibility_decay: float = 0.95


class LocalLearner:
    """Three-factor local learning for spiking parameters.

    Δw = η · DA · (pre_spike × post_spike × eligibility)

    Phase 2 optionally freezes QLoRA adapters and uses only local
    learning rules for spiking block weights.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: LocalLearningConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or LocalLearningConfig()
        self._eligibility_traces: dict[str, Tensor] = {}

        if self.cfg.freeze_qlora:
            self._freeze_qlora()

    def _freeze_qlora(self) -> None:
        """Freeze QLoRA adapter params."""
        frozen = 0
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = False
                frozen += param.numel()
        logger.info("Froze %d QLoRA parameters for Phase 2", frozen)

    def _unfreeze_qlora(self) -> None:
        """Unfreeze QLoRA adapter params."""
        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

    def update(
        self,
        spiking_metrics: list[dict],
        da_level: float = 1.0,
        ach_level: float = 1.0,
    ) -> dict[str, float]:
        """Apply three-factor local learning update.

        For each spiking block, compute:
          Δw = η · DA · eligibility_trace

        where eligibility accumulates pre×post spike coincidences.
        """
        total_delta = 0.0
        num_updates = 0
        lr = self.cfg.base_lr * da_level * self.cfg.da_modulation

        # Gate by ACh — low ACh suppresses plasticity
        if ach_level < 0.2:
            return {"total_delta": 0.0, "num_updates": 0, "gated": True}

        for block_idx, metrics in enumerate(spiking_metrics):
            attn_w = metrics.get("attn")
            if attn_w is None:
                continue

            key = f"block_{block_idx}"
            # Update eligibility trace with exponential decay
            if key in self._eligibility_traces:
                self._eligibility_traces[key] = (
                    self.cfg.eligibility_decay * self._eligibility_traces[key]
                    + attn_w.detach()
                )
            else:
                self._eligibility_traces[key] = attn_w.detach()

            trace = self._eligibility_traces[key]
            delta = lr * trace

            # Apply to spiking attention weights if accessible
            spiking_blocks = self._get_spiking_blocks()
            if block_idx < len(spiking_blocks):
                block = spiking_blocks[block_idx]
                if hasattr(block, "spike_attention"):
                    for p in block.spike_attention.parameters():
                        if p.requires_grad and p.shape == delta.shape:
                            with torch.no_grad():
                                p.add_(delta)
                                total_delta += delta.abs().sum().item()
                                num_updates += 1

        return {
            "total_delta": total_delta,
            "num_updates": num_updates,
            "gated": False,
        }

    def _get_spiking_blocks(self) -> nn.ModuleList:
        """Navigate to spiking blocks."""
        if hasattr(self.model, "spiking") and hasattr(self.model.spiking, "blocks"):
            return self.model.spiking.blocks
        return nn.ModuleList()

    def reset_eligibility(self) -> None:
        """Clear eligibility traces (e.g., at episode boundary)."""
        self._eligibility_traces.clear()
