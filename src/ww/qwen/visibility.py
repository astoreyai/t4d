"""P3-10: Activation visibility hooks — glass-box for all layers."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class ActivationCollector:
    """Collects activations from Qwen layers and spiking blocks.

    Provides full glass-box visibility: every hidden state, attention weight,
    spike pattern, membrane potential, and LoRA delta is observable.
    """

    def __init__(self) -> None:
        self._hooks: list[Any] = []
        self._activations: dict[str, Tensor] = {}

    def attach_qwen(self, model: nn.Module, layers: list[int] | None = None) -> None:
        """Attach hooks to Qwen transformer layers."""
        qwen_layers = self._get_layers(model)
        if layers is None:
            layers = list(range(len(qwen_layers)))

        for i in layers:
            name = f"qwen_layer_{i}"
            hook = qwen_layers[i].register_forward_hook(
                self._make_hook(name)
            )
            self._hooks.append(hook)

    def attach_spiking(self, spiking_stack: nn.Module) -> None:
        """Attach hooks to spiking cortical blocks."""
        if not hasattr(spiking_stack, "blocks"):
            return
        for i, block in enumerate(spiking_stack.blocks):
            # Hook each stage
            for stage_name in ["thalamic", "lif_integration", "spike_attention",
                               "apical", "rwkv", "lif_output"]:
                if hasattr(block, stage_name):
                    name = f"spiking_block_{i}_{stage_name}"
                    hook = getattr(block, stage_name).register_forward_hook(
                        self._make_hook(name)
                    )
                    self._hooks.append(hook)

    def attach_lora_deltas(self, model: nn.Module) -> None:
        """Attach hooks to capture LoRA adapter contributions."""
        for name, module in model.named_modules():
            if "lora" in name.lower() and hasattr(module, "weight"):
                hook = module.register_forward_hook(
                    self._make_hook(f"lora_{name}")
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(module: nn.Module, inputs: Any, output: Any) -> None:
            if isinstance(output, tuple):
                self._activations[name] = output[0].detach()
            elif isinstance(output, Tensor):
                self._activations[name] = output.detach()
        return hook_fn

    def _get_layers(self, model: nn.Module) -> nn.ModuleList:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "base_model"):
            base = model.base_model
            if hasattr(base, "model") and hasattr(base.model, "model"):
                return base.model.model.layers
        raise AttributeError("Cannot locate transformer layers")

    def detach_all(self) -> None:
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @property
    def activations(self) -> dict[str, Tensor]:
        """All captured activations from the last forward pass."""
        return dict(self._activations)

    def clear(self) -> None:
        """Clear captured activations."""
        self._activations.clear()

    def summary(self) -> dict[str, tuple[int, ...]]:
        """Return shapes of all captured activations."""
        return {k: tuple(v.shape) for k, v in self._activations.items()}
