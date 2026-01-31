"""P3-03: Hidden state extractor — hooks a specific Qwen layer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class HiddenStateExtractor:
    """Hooks a transformer layer to capture hidden states.

    Default tap layer is 17 (0-indexed) = layer 18 of 36,
    the midpoint where states are routed to the spiking adapter.
    """

    def __init__(self, model: nn.Module, tap_layer: int = 17) -> None:
        self._model = model
        self._tap_layer = tap_layer
        self._hidden: Tensor | None = None
        self._hook = None

    def _get_layers(self) -> nn.ModuleList:
        """Navigate to the transformer layers list."""
        # Qwen2 structure: model.model.layers
        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            return self._model.model.layers
        # PEFT-wrapped
        if hasattr(self._model, "base_model"):
            base = self._model.base_model
            if hasattr(base, "model") and hasattr(base.model, "model"):
                return base.model.model.layers
        raise AttributeError("Cannot locate transformer layers in model")

    def attach(self) -> None:
        """Register forward hook on the tap layer."""
        layers = self._get_layers()
        target = layers[self._tap_layer]

        def hook_fn(module: nn.Module, inputs: Any, output: Any) -> None:
            # Qwen2 layer output is (hidden_states, ...) tuple
            if isinstance(output, tuple):
                self._hidden = output[0].detach()
            else:
                self._hidden = output.detach()

        self._hook = target.register_forward_hook(hook_fn)

    def detach(self) -> None:
        """Remove the hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    @property
    def hidden_states(self) -> Tensor | None:
        """Last captured hidden states [B, S, D]."""
        return self._hidden

    @property
    def tap_layer(self) -> int:
        return self._tap_layer
