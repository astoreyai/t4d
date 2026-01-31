"""P3-09: Inference pipeline — tokenize → unified model → generate."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Inference generation config."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    tau_threshold: float = 0.5  # τ(t) gate threshold for memory writes


class InferencePipeline:
    """Token generation pipeline with memory read/write.

    Inference flow:
      1. Tokenize input
      2. Run unified model (Qwen + spiking adapter)
      3. Sample next token from logits
      4. If τ(t) > threshold: write encoded memory to T4DX
      5. Return generated text + metrics
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        cfg: InferenceConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg or InferenceConfig()
        self._spiking_states: list[dict] | None = None

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        neuromod_state: Any = None,
    ) -> dict[str, Any]:
        """Generate text from prompt.

        Returns dict with: text, tokens_generated, encoded_memory, spiking_metrics.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True,
        ).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        generated_ids = input_ids.clone()

        all_metrics: list[list[dict]] = []
        last_encoded = None

        for _ in range(self.cfg.max_new_tokens):
            output = self.model(
                generated_ids,
                attention_mask=attention_mask,
                spiking_states=self._spiking_states,
                neuromod_state=neuromod_state,
            )

            logits = output["logits"][:, -1, :]  # last position
            self._spiking_states = output.get("spiking_states")
            last_encoded = output.get("encoded_memory")

            if output.get("spiking_metrics"):
                all_metrics.append(output["spiking_metrics"])

            # Sample next token
            next_id = self._sample(logits)
            generated_ids = torch.cat([generated_ids, next_id], dim=-1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_id)], dim=-1,
                )

            # Check for EOS
            if next_id.item() == self.tokenizer.eos_token_id:
                break

        # Decode generated text (excluding prompt)
        new_ids = generated_ids[:, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_ids[0], skip_special_tokens=True)

        return {
            "text": text,
            "tokens_generated": new_ids.shape[1],
            "encoded_memory": last_encoded,
            "spiking_metrics": all_metrics,
        }

    def _sample(self, logits: Tensor) -> Tensor:
        """Sample a token from logits with temperature/top-p/top-k."""
        if not self.cfg.do_sample:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / max(self.cfg.temperature, 1e-8)

        # Top-k filtering
        if self.cfg.top_k > 0:
            top_k = min(self.cfg.top_k, logits.size(-1))
            vals, _ = logits.topk(top_k)
            logits[logits < vals[:, -1:]] = float("-inf")

        # Top-p (nucleus) filtering
        if self.cfg.top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(descending=True)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove = cum_probs > self.cfg.top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def reset_state(self) -> None:
        """Clear spiking recurrent states."""
        self._spiking_states = None
