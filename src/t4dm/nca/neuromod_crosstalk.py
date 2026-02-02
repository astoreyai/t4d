"""
C3: Cross-Neuromodulator Crosstalk for T4DM.

Implements bidirectional modulation between neurotransmitter systems:
1. DA inhibits 5-HT (dopamine-serotonin antagonism)
2. NE excites ACh (norepinephrine-acetylcholine facilitation)
3. ACh gates DA (cholinergic control of dopamine release)

Biological Basis:
- DA-5HT: VTA dopamine neurons inhibit raphe serotonin (Daw et al. 2002)
- NE-ACh: LC norepinephrine enhances basal forebrain ACh (Sarter & Bruno 2000)
- ACh-DA: ACh interneurons gate striatal DA release (Threlfell et al. 2012)

References:
- Daw et al. (2002): Opponent interactions between serotonin and dopamine
- Sarter & Bruno (2000): Cortical cholinergic inputs mediating arousal
- Threlfell et al. (2012): Striatal dopamine release is triggered by ACh interneurons
- Aston-Jones & Cohen (2005): LC-NE adaptive gain theory
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NeuromodCrosstalkConfig:
    """Configuration for cross-neuromodulator interactions."""

    # DA → 5-HT inhibition
    da_inhibits_5ht_strength: float = 0.3  # How much DA suppresses 5-HT

    # NE → ACh excitation
    ne_excites_ach_strength: float = 0.4  # How much NE boosts ACh

    # ACh → DA gating
    ach_gates_da_strength: float = 1.0  # ACh gating of DA release
    ach_gate_midpoint: float = 0.5  # ACh level for 50% DA gating

    # Safety bounds
    enable_crosstalk: bool = True
    min_nt_level: float = 0.05  # ATOM-P2-7: Biological baseline (never absolute zero)
    max_nt_level: float = 1.0


class NeuromodCrosstalk:
    """
    C3: Cross-modulation between neurotransmitter systems.

    Implements three key crosstalk pathways:
    1. DA inhibits 5-HT: High dopamine suppresses serotonin
    2. NE excites ACh: High norepinephrine boosts acetylcholine
    3. ACh gates DA: Acetylcholine controls dopamine release

    Usage:
        crosstalk = NeuromodCrosstalk()
        nt_levels = {"da": 0.8, "5ht": 0.5, "ne": 0.6, "ach": 0.4}
        modulated = crosstalk.modulate(nt_levels)
        # modulated["5ht"] is now reduced by DA
        # modulated["ach"] is now increased by NE
        # modulated["da"] is now gated by ACh
    """

    def __init__(self, config: NeuromodCrosstalkConfig | None = None):
        """
        Initialize crosstalk system.

        Args:
            config: Crosstalk configuration
        """
        self._config = config or NeuromodCrosstalkConfig()
        # ATOM-P3-14: Make enable_crosstalk read-only after construction
        self._enable_crosstalk = self._config.enable_crosstalk
        if not self._enable_crosstalk:
            logger.warning("Crosstalk is disabled — all cross-system regulation off")

        logger.info(
            f"NeuromodCrosstalk initialized: "
            f"DA→5HT={self._config.da_inhibits_5ht_strength}, "
            f"NE→ACh={self._config.ne_excites_ach_strength}, "
            f"ACh→DA gating enabled"
        )

    @property
    def config(self) -> NeuromodCrosstalkConfig:
        """Get configuration (read-only)."""
        return self._config

    @property
    def enable_crosstalk(self) -> bool:
        """Get crosstalk enable status (read-only after construction)."""
        return self._enable_crosstalk

    def da_inhibits_5ht(self, da_level: float, serotonin_level: float) -> float:
        """
        C3.1: Dopamine inhibits serotonin.

        Biological basis: VTA dopamine neurons inhibit raphe serotonin neurons.
        High reward (DA) suppresses patience/temporal discounting (5-HT).

        Args:
            da_level: Dopamine level [0, 1]
            serotonin_level: Current serotonin level [0, 1]

        Returns:
            Modulated serotonin level
        """
        if not self._enable_crosstalk:
            return serotonin_level

        # Linear suppression: high DA → low 5-HT
        suppression = self.config.da_inhibits_5ht_strength * da_level
        modulated = serotonin_level * (1.0 - suppression)

        return float(np.clip(modulated, self.config.min_nt_level, self.config.max_nt_level))

    def ne_excites_ach(self, ne_level: float, ach_level: float) -> float:
        """
        C3.2: Norepinephrine excites acetylcholine.

        Biological basis: LC-NE enhances basal forebrain ACh release.
        Arousal (NE) boosts attention/encoding (ACh).

        Args:
            ne_level: Norepinephrine level [0, 1]
            ach_level: Current acetylcholine level [0, 1]

        Returns:
            Modulated acetylcholine level
        """
        if not self._enable_crosstalk:
            return ach_level

        # Additive boost: high NE → high ACh
        boost = self.config.ne_excites_ach_strength * ne_level
        modulated = ach_level + boost

        return float(np.clip(modulated, self.config.min_nt_level, self.config.max_nt_level))

    def ach_gates_da(self, ach_level: float, da_signal: float) -> float:
        """
        C3.3: Acetylcholine gates dopamine release.

        Biological basis: Striatal ACh interneurons PAUSE to enable DA release.
        Low ACh (interneuron pause) enables DA release. High ACh suppresses DA.

        Args:
            ach_level: Acetylcholine level [0, 1]
            da_signal: Dopamine signal to gate [0, 1]

        Returns:
            Gated dopamine signal
        """
        if not self._enable_crosstalk:
            return da_signal

        # Sigmoid gating: ACh controls DA release gain
        # Biological: ACh pause → DA burst (Threlfell 2012)
        # HIGH ACh = suppression (gate~0), LOW ACh = release (gate~1)
        gate = 1.0 / (1.0 + np.exp(10 * (ach_level - self.config.ach_gate_midpoint)))
        gate *= self.config.ach_gates_da_strength

        gated_da = da_signal * gate

        return float(np.clip(gated_da, self.config.min_nt_level, self.config.max_nt_level))

    def modulate(self, nt_levels: dict[str, float]) -> dict[str, float]:
        """
        Apply all cross-modulation pathways to NT levels.

        Args:
            nt_levels: Dictionary with keys: "da", "5ht", "ne", "ach"
                      (also accepts "serotonin" as alias for "5ht")

        Returns:
            Modulated NT levels dictionary
        """
        if not self._enable_crosstalk:
            return nt_levels.copy()

        # Extract levels (with defaults)
        da = nt_levels.get("da", 0.5)
        serotonin = nt_levels.get("5ht", nt_levels.get("serotonin", 0.5))
        ne = nt_levels.get("ne", 0.5)
        ach = nt_levels.get("ach", 0.5)

        # Apply crosstalk pathways
        modulated_5ht = self.da_inhibits_5ht(da, serotonin)
        modulated_ach = self.ne_excites_ach(ne, ach)
        modulated_da = self.ach_gates_da(ach, da)

        # Build output dictionary
        result = nt_levels.copy()
        result["da"] = modulated_da
        result["5ht"] = modulated_5ht
        if "serotonin" in result:
            result["serotonin"] = modulated_5ht
        result["ach"] = modulated_ach

        return result

    def get_crosstalk_effects(self, nt_levels: dict[str, float]) -> dict[str, dict[str, float]]:
        """
        Get detailed crosstalk effect breakdown.

        Args:
            nt_levels: NT levels dictionary

        Returns:
            Dictionary showing crosstalk effects per pathway
        """
        da = nt_levels.get("da", 0.5)
        serotonin = nt_levels.get("5ht", nt_levels.get("serotonin", 0.5))
        ne = nt_levels.get("ne", 0.5)
        ach = nt_levels.get("ach", 0.5)

        # Calculate deltas
        original_5ht = serotonin
        modulated_5ht = self.da_inhibits_5ht(da, serotonin)
        delta_5ht = modulated_5ht - original_5ht

        original_ach = ach
        modulated_ach = self.ne_excites_ach(ne, ach)
        delta_ach = modulated_ach - original_ach

        original_da = da
        modulated_da = self.ach_gates_da(ach, da)
        delta_da = modulated_da - original_da

        return {
            "da_inhibits_5ht": {
                "original": original_5ht,
                "modulated": modulated_5ht,
                "delta": delta_5ht,
                "source_level": da,
            },
            "ne_excites_ach": {
                "original": original_ach,
                "modulated": modulated_ach,
                "delta": delta_ach,
                "source_level": ne,
            },
            "ach_gates_da": {
                "original": original_da,
                "modulated": modulated_da,
                "delta": delta_da,
                "source_level": ach,
            },
        }


def create_neuromod_crosstalk(
    da_inhibits_5ht: float = 0.3,
    ne_excites_ach: float = 0.4,
    ach_gates_da: float = 1.0
) -> NeuromodCrosstalk:
    """
    Factory function for creating cross-NT modulation.

    Args:
        da_inhibits_5ht: Strength of DA→5HT inhibition
        ne_excites_ach: Strength of NE→ACh excitation
        ach_gates_da: Strength of ACh→DA gating

    Returns:
        Configured NeuromodCrosstalk instance
    """
    config = NeuromodCrosstalkConfig(
        da_inhibits_5ht_strength=da_inhibits_5ht,
        ne_excites_ach_strength=ne_excites_ach,
        ach_gates_da_strength=ach_gates_da,
    )
    return NeuromodCrosstalk(config)


__all__ = [
    "NeuromodCrosstalk",
    "NeuromodCrosstalkConfig",
    "create_neuromod_crosstalk",
]
