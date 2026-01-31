"""P3-06: Phase 1 training loop — surrogate gradient + QLoRA backprop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Phase 1 training configuration."""

    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 10000
    gradient_accumulation: int = 4
    max_grad_norm: float = 1.0
    ff_goodness_weight: float = 0.1  # Forward-Forward goodness loss weight
    reconstruction_weight: float = 0.05  # Memory projection reconstruction loss


class Phase1Trainer:
    """Phase 1: joint backprop through QLoRA + spiking via STE surrogate gradient.

    Loss = CE(logits, targets) + λ_ff * FF_goodness + λ_recon * reconstruction
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg or TrainingConfig()
        self._step = 0

        # Collect trainable params
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    def train_step(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Tensor | None = None,
    ) -> dict[str, float]:
        """Single training step. Returns loss components."""
        self.model.train()
        device = input_ids.device
        use_amp = device.type == "cuda"

        with torch.amp.autocast(device.type, enabled=use_amp):
            output = self.model(input_ids, attention_mask=attention_mask)
            logits = output["logits"]

            # Cross-entropy loss (shift by 1 for next-token prediction)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            ce_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            # FF goodness loss (from spiking metrics)
            ff_loss = torch.tensor(0.0, device=device)
            for metrics in output.get("spiking_metrics", []):
                if "goodness" in metrics and metrics["goodness"] is not None:
                    ff_loss = ff_loss + metrics["goodness"].mean()

            # Reconstruction loss (encode → decode should approximate identity)
            recon_loss = torch.tensor(0.0, device=device)
            if "hidden_mid" in output and "encoded_memory" in output:
                decoded = self.model.projection.decode(output["encoded_memory"])
                recon_loss = nn.functional.mse_loss(decoded, output["hidden_mid"])

            total_loss = (
                ce_loss
                + self.cfg.ff_goodness_weight * ff_loss
                + self.cfg.reconstruction_weight * recon_loss
            )
            total_loss = total_loss / self.cfg.gradient_accumulation

        self.scaler.scale(total_loss).backward()

        self._step += 1
        if self._step % self.cfg.gradient_accumulation == 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {
            "loss": total_loss.item() * self.cfg.gradient_accumulation,
            "ce_loss": ce_loss.item(),
            "ff_loss": ff_loss.item(),
            "recon_loss": recon_loss.item(),
            "step": self._step,
        }

    @property
    def step(self) -> int:
        return self._step
