"""P3-01: Qwen 2.5-3B model loader with 4-bit NF4 quantization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B"


@dataclass
class QwenConfig:
    """Configuration for Qwen model loading."""

    model_id: str = _DEFAULT_MODEL_ID
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str = "sdpa"


def _get_bnb_config(cfg: QwenConfig) -> Any:
    """Build BitsAndBytes config if available, else None."""
    if not cfg.use_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        )
    except ImportError:
        logger.warning("bitsandbytes not installed, loading without quantization")
        return None


def load_qwen(
    cfg: QwenConfig | None = None,
) -> tuple[PreTrainedModel, AutoTokenizer]:
    """Load frozen Qwen 2.5-3B (optionally 4-bit quantized).

    Returns (model, tokenizer). All parameters are frozen.
    """
    cfg = cfg or QwenConfig()
    bnb_config = _get_bnb_config(cfg)

    load_kwargs: dict[str, Any] = {
        "device_map": cfg.device_map,
        "torch_dtype": getattr(torch, cfg.torch_dtype),
        "trust_remote_code": cfg.trust_remote_code,
        "attn_implementation": cfg.attn_implementation,
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id, trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Loaded %s: %d total params, %d trainable (frozen), dtype=%s",
        cfg.model_id, total, trainable, cfg.torch_dtype,
    )

    return model, tokenizer


def get_hidden_dim(model: PreTrainedModel) -> int:
    """Return the model's hidden dimension."""
    return model.config.hidden_size


def get_num_layers(model: PreTrainedModel) -> int:
    """Return the number of transformer layers."""
    return model.config.num_hidden_layers
