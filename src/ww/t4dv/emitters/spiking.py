"""Spiking emitter — attaches PyTorch forward hooks to CorticalStack."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from ww.t4dv.bus import ObservationBus
from ww.t4dv.events import SpikeEvent

if TYPE_CHECKING:
    from ww.spiking.cortical_block import CorticalBlock
    from ww.spiking.cortical_stack import CorticalStack


def attach_spiking_hooks(
    stack: CorticalStack,
    bus: ObservationBus,
) -> list[torch.utils.hooks.RemovableHook]:
    """Register forward hooks on each CorticalBlock in *stack*.

    Each hook emits a ``SpikeEvent`` with pe, goodness, and firing_rate
    extracted from the block's metrics dict.

    Returns the hook handles so callers can remove them later.
    """
    handles: list[torch.utils.hooks.RemovableHook] = []

    for idx, block in enumerate(stack.blocks):
        handle = block.register_forward_hook(_make_hook(idx, bus))
        handles.append(handle)

    return handles


def _make_hook(block_index: int, bus: ObservationBus):
    def hook(
        module: CorticalBlock,
        input: Any,
        output: tuple,
    ) -> None:
        # output = (tensor, new_state, metrics)
        _out_tensor, _state, metrics = output

        pe_val = metrics.get("pe")
        goodness_val = metrics.get("goodness")

        pe_float = _to_float(pe_val)
        goodness_float = _to_float(goodness_val)

        # firing rate from output spikes
        firing_rate = 0.0
        if isinstance(_out_tensor, torch.Tensor):
            firing_rate = float(_out_tensor.mean().item())

        event = SpikeEvent(
            block_index=block_index,
            firing_rate=firing_rate,
            prediction_error=pe_float,
            goodness=goodness_float,
            source=f"cortical_block_{block_index}",
        )
        bus.emit_sync(event)

    return hook


def _to_float(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, torch.Tensor):
        return float(val.mean().item())
    return float(val)
