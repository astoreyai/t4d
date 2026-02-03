"""P3-05: Unified model — Qwen(0-17) → Spiking Adapter → Qwen(18-35) → LM head."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from t4dm.qwen.projections import MemoryProjection
from t4dm.spiking.cortical_stack import CorticalStack

logger = logging.getLogger(__name__)


class UnifiedModel(nn.Module):
    """Qwen + QLoRA + Spiking Cortical Adapter.

    Architecture:
        Qwen layers [0, split_layer)  →  encode to mem_dim
        → CorticalStack (spiking adapter)
        → decode back to qwen_dim
        → Qwen layers [split_layer, N) → LM head

    The Qwen model (with QLoRA) is held externally; this module
    owns the spiking adapter and projections.
    """

    def __init__(
        self,
        qwen_model: nn.Module,
        qwen_dim: int = 2048,
        mem_dim: int = 1024,
        num_spiking_blocks: int = 6,
        num_heads: int = 8,
        split_layer: int = 18,
    ) -> None:
        super().__init__()
        self.qwen = qwen_model
        self.split_layer = split_layer
        self.qwen_dim = qwen_dim
        self.mem_dim = mem_dim

        self.projection = MemoryProjection(qwen_dim, mem_dim)
        self.spiking = CorticalStack(
            dim=mem_dim,
            num_blocks=num_spiking_blocks,
            num_heads=num_heads,
        )
        # Gate to blend spiking output with skip connection
        self.gate = nn.Sequential(
            nn.Linear(qwen_dim, 1),
            nn.Sigmoid(),
        )

    def _get_layers(self) -> nn.ModuleList:
        if hasattr(self.qwen, "model") and hasattr(self.qwen.model, "layers"):
            return self.qwen.model.layers
        if hasattr(self.qwen, "base_model"):
            base = self.qwen.base_model
            if hasattr(base, "model") and hasattr(base.model, "model"):
                return base.model.model.layers
        raise AttributeError("Cannot locate transformer layers")

    def _get_embed(self) -> nn.Module:
        if hasattr(self.qwen, "model") and hasattr(self.qwen.model, "embed_tokens"):
            return self.qwen.model.embed_tokens
        if hasattr(self.qwen, "base_model"):
            base = self.qwen.base_model
            if hasattr(base, "model") and hasattr(base.model, "model"):
                return base.model.model.embed_tokens
        raise AttributeError("Cannot locate embedding layer")

    def _get_norm(self) -> nn.Module:
        if hasattr(self.qwen, "model") and hasattr(self.qwen.model, "norm"):
            return self.qwen.model.norm
        if hasattr(self.qwen, "base_model"):
            base = self.qwen.base_model
            if hasattr(base, "model") and hasattr(base.model, "model"):
                return base.model.model.norm
        raise AttributeError("Cannot locate final norm")

    def _get_rotary_emb(self, hidden: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor] | None:
        """Get (cos, sin) position embeddings from the Qwen rotary embedding.

        Returns None if not available (e.g. mock/test models).
        """
        if hasattr(self.qwen, "model") and hasattr(self.qwen.model, "rotary_emb"):
            return self.qwen.model.rotary_emb(hidden, position_ids)
        if hasattr(self.qwen, "base_model"):
            base = self.qwen.base_model
            if hasattr(base, "model") and hasattr(base.model, "model"):
                if hasattr(base.model.model, "rotary_emb"):
                    return base.model.model.rotary_emb(hidden, position_ids)
        return None

    def _get_lm_head(self) -> nn.Module:
        if hasattr(self.qwen, "lm_head"):
            return self.qwen.lm_head
        if hasattr(self.qwen, "base_model"):
            base = self.qwen.base_model
            if hasattr(base, "model") and hasattr(base.model, "lm_head"):
                return base.model.lm_head
        raise AttributeError("Cannot locate lm_head")

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        spiking_states: list[dict] | None = None,
        neuromod_state: Any = None,
    ) -> dict[str, Any]:
        """Full forward pass.

        Returns dict with keys: logits, spiking_states, spiking_metrics,
        encoded_memory, hidden_mid.
        """
        layers = self._get_layers()
        embed = self._get_embed()
        norm = self._get_norm()
        lm_head = self._get_lm_head()

        # Embedding
        hidden = embed(input_ids)

        # Compute rotary position embeddings once for all layers
        seq_len = hidden.shape[1]
        position_ids = torch.arange(
            seq_len, device=hidden.device, dtype=torch.long,
        ).unsqueeze(0).expand(hidden.shape[0], -1)
        position_embeddings = self._get_rotary_emb(hidden, position_ids)

        # Build causal attention mask compatible with SDPA
        # None works for full-sequence forward (SDPA uses causal by default)
        causal_mask = None
        if attention_mask is not None and attention_mask.dim() == 2:
            # Convert [B, S] padding mask to [B, 1, S, S] float mask
            bsz = attention_mask.shape[0]
            causal_mask = torch.full(
                (bsz, 1, seq_len, seq_len), float("-inf"),
                device=hidden.device, dtype=hidden.dtype,
            )
            causal_mask = causal_mask.triu(diagonal=1)  # causal
            # Apply padding: mask out padded positions
            pad_mask = (1 - attention_mask[:, None, None, :].to(hidden.dtype)) * float("-inf")
            causal_mask = causal_mask + pad_mask

        # Build layer kwargs (only pass position_embeddings if available)
        layer_kw: dict[str, Any] = {"attention_mask": causal_mask}
        if position_embeddings is not None:
            layer_kw["position_embeddings"] = position_embeddings

        # Qwen layers [0, split_layer) — lower half
        for i in range(self.split_layer):
            layer_out = layers[i](hidden, **layer_kw)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_mid = hidden  # save for skip connection

        # Project to memory dim → spiking adapter → project back
        # Spiking blocks use float32 internally (LIF neurons need precision)
        compute_dtype = hidden.dtype
        mem_encoded = self.projection.encode(hidden.float()).float()
        spiking_out, new_states, spiking_metrics = self.spiking(
            mem_encoded, context=None,
            neuromod_state=neuromod_state,
            states=spiking_states,
        )
        mem_decoded = self.projection.decode(spiking_out).to(compute_dtype)

        # Clamp spiking output to prevent NaN propagation from untrained adapter
        mem_decoded = torch.clamp(mem_decoded, -100.0, 100.0)

        # Gated residual: α * spiking_decoded + (1-α) * skip
        alpha = self.gate(hidden_mid.float()).to(compute_dtype)
        hidden = alpha * mem_decoded + (1 - alpha) * hidden_mid

        # Qwen layers [split_layer, N) — upper half
        for i in range(self.split_layer, len(layers)):
            layer_out = layers[i](hidden, **layer_kw)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden = norm(hidden)
        logits = lm_head(hidden)

        return {
            "logits": logits,
            "spiking_states": new_states,
            "spiking_metrics": spiking_metrics,
            "encoded_memory": mem_encoded.detach(),
            "hidden_mid": hidden_mid.detach(),
        }

    def trainable_param_count(self) -> dict[str, int]:
        """Count trainable params by component."""
        spiking_params = sum(
            p.numel() for p in self.spiking.parameters() if p.requires_grad
        )
        proj_params = sum(
            p.numel() for p in self.projection.parameters() if p.requires_grad
        )
        gate_params = sum(
            p.numel() for p in self.gate.parameters() if p.requires_grad
        )
        # QLoRA params are in the qwen model
        qlora_params = sum(
            p.numel() for n, p in self.qwen.named_parameters()
            if p.requires_grad and "lora" in n.lower()
        )
        return {
            "spiking": spiking_params,
            "projection": proj_params,
            "gate": gate_params,
            "qlora": qlora_params,
            "total": spiking_params + proj_params + gate_params + qlora_params,
        }
