"""P3-08: LoRA merge/export — merge adapters into base for inference."""

from __future__ import annotations

import logging
from pathlib import Path

from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def merge_lora(model: PreTrainedModel) -> PreTrainedModel:
    """Merge LoRA weights into the base model (in-place).

    After merging, the adapter overhead is eliminated and the model
    runs as a standard transformer with the adapted weights baked in.
    """
    if not hasattr(model, "merge_and_unload"):
        logger.warning("Model does not have merge_and_unload — not a PEFT model?")
        return model

    merged = model.merge_and_unload()
    trainable = sum(p.numel() for p in merged.parameters() if p.requires_grad)
    logger.info("LoRA merged: %d trainable params remaining (should be 0)", trainable)
    return merged


def save_merged(model: PreTrainedModel, output_dir: str | Path) -> None:
    """Save the merged model to disk in safetensors format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)
    logger.info("Merged model saved to %s", output_dir)


def save_lora_only(model: PreTrainedModel, output_dir: str | Path) -> None:
    """Save only the LoRA adapter weights (much smaller)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(output_dir))
        logger.info("LoRA adapters saved to %s", output_dir)
    else:
        logger.warning("Cannot save — model has no save_pretrained")
