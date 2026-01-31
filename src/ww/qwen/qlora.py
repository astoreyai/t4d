"""P3-02: QLoRA adapter setup via PEFT."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from peft import LoraConfig, TaskType, get_peft_model
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class QLoRAConfig:
    """QLoRA configuration."""

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]


def apply_qlora(
    model: PreTrainedModel,
    cfg: QLoRAConfig | None = None,
) -> PreTrainedModel:
    """Apply LoRA adapters to a frozen model.

    Returns the PEFT-wrapped model with only LoRA params trainable.
    """
    cfg = cfg or QLoRAConfig()

    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        bias=cfg.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "QLoRA applied: r=%d, alpha=%d, targets=%s, trainable=%d (%.2f%%)",
        cfg.r, cfg.lora_alpha, cfg.target_modules, trainable,
        100.0 * trainable / total if total > 0 else 0,
    )

    return peft_model


def get_trainable_params(model: PreTrainedModel) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
