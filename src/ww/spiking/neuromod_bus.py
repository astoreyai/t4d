"""
Neuromodulator Bus — routes NT signals to spiking block layers.

Mapping (follows cortical laminar organization):
  DA  → L2/3 + L5 (stages 3, 4)
  NE  → L5 (stages 2, 6)
  ACh → L1/L4 (stage 1)
  5-HT → L5/6 (stage 5)
"""

from __future__ import annotations

from ww.learning.neuromodulators import NeuromodulatorOrchestra, NeuromodulatorState


class NeuromodBus:
    """Routes neuromodulator state to per-layer modulation parameters."""

    def __init__(self, orchestra: NeuromodulatorOrchestra | None = None):
        self.orchestra = orchestra

    def get_layer_modulation(self, state: NeuromodulatorState) -> dict:
        """
        Convert global neuromodulator state to per-stage parameters.

        Args:
            state: Current neuromodulator state.

        Returns:
            Dict mapping stage parameter names to modulation values.
        """
        return {
            "thalamic_ach": state.acetylcholine_mode,
            "lif_ne_gain": state.norepinephrine_gain,
            "attention_da_lr": state.dopamine_rpe,
            "apical_da": state.dopamine_rpe,
            "rwkv_5ht_patience": state.serotonin_mood,
            "output_ne_gain": state.norepinephrine_gain,
        }

    def ach_level(self, state: NeuromodulatorState) -> float:
        """Extract numeric ACh level for thalamic gate."""
        mode = state.acetylcholine_mode
        if mode == "encoding":
            return 0.8
        elif mode == "retrieval":
            return 0.3
        return 0.5
