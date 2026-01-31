"""
Phase 1B: Bridge connecting VTA dopamine to STDP modulation.

Biological Basis:
- Dopamine modulates synaptic plasticity by gating STDP
- High DA (reward/positive RPE) enhances LTP, reduces LTD
- Low DA (punishment/negative RPE) reduces LTP, enhances LTD
- DA acts as a "learning gate" - only update when DA signals importance

Architecture:
- STDPVTABridge connects VTACircuit DA level to STDP learning rates
- Modulates A+ (LTP amplitude) and A- (LTD amplitude) based on DA
- Provides DA-modulated learning rates to STDPLearner

Integration:
- VTACircuit: provides current DA level via get_da_for_neural_field()
- STDPLearner: compute_stdp_delta() accepts optional da_level parameter
- Bridge: get_da_modulated_rates() computes modulated A+/A-

References:
- Izhikevich (2007): "Solving the distal reward problem through linkage of STDP and DA"
- Schultz (1998): "Dopamine reward prediction error theory"
- Frémaux & Gerstner (2016): "Neuromodulated STDP in computational neuroscience"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from t4dm.learning.stdp import STDPLearner
    from t4dm.nca.vta import VTACircuit

logger = logging.getLogger(__name__)


@dataclass
class STDPVTAConfig:
    """Configuration for STDP-VTA dopamine modulation."""
    # Modulation strengths
    ltp_da_gain: float = 0.5    # How much DA modulates LTP (0-1)
    ltd_da_gain: float = 0.3    # How much DA modulates LTD (0-1)

    # Baseline DA level (no modulation)
    baseline_da: float = 0.5    # DA level for no modulation

    # Modulation curve shape
    da_threshold: float = 0.1   # Minimum DA change for modulation
    saturation_da: float = 0.95 # DA level for maximum modulation

    # Gating
    enable_gating: bool = True  # Gate learning by DA level
    min_da_for_learning: float = 0.1  # Minimum DA to allow learning


class STDPVTABridge:
    """
    Bridge connecting VTA dopamine to STDP learning rate modulation.

    Implements dopamine-modulated STDP:
    - High DA (>baseline): Increases LTP, decreases LTD (reward learning)
    - Low DA (<baseline): Decreases LTP, increases LTD (punishment learning)
    - DA near baseline: Minimal modulation (exploration)

    Biological Basis:
    - D1 receptors enhance LTP via PKA/DARPP-32 pathway
    - D2 receptors suppress LTD via reduced Ca2+ influx
    - DA acts as a "go" signal for plasticity gating

    Usage:
        bridge = STDPVTABridge(stdp_learner, vta_circuit)

        # Get DA-modulated learning rates
        a_plus_mod, a_minus_mod = bridge.get_da_modulated_rates()

        # Or get current DA level for STDP
        da_level = bridge.get_current_da()
        delta_w = stdp.compute_stdp_delta(delta_t, weight, da_level=da_level)
    """

    def __init__(
        self,
        stdp_learner: STDPLearner,
        vta_circuit: VTACircuit | None = None,
        config: STDPVTAConfig | None = None
    ):
        """
        Initialize STDP-VTA bridge.

        Args:
            stdp_learner: STDPLearner instance to modulate
            vta_circuit: VTACircuit instance (optional, can be set later)
            config: Modulation configuration
        """
        self.stdp = stdp_learner
        self._vta = vta_circuit
        self.config = config or STDPVTAConfig()

        # Cache last DA level for efficiency
        self._last_da_level: float = self.config.baseline_da
        self._last_modulation: tuple[float, float] | None = None

        logger.info(
            f"STDPVTABridge initialized: "
            f"LTP_gain={self.config.ltp_da_gain}, "
            f"LTD_gain={self.config.ltd_da_gain}, "
            f"baseline_DA={self.config.baseline_da}, "
            f"gating={self.config.enable_gating}"
        )

    def set_vta_circuit(self, vta: VTACircuit) -> None:
        """
        Set VTA circuit for dopamine source.

        Args:
            vta: VTACircuit instance
        """
        self._vta = vta
        logger.debug("VTA circuit connected to STDP bridge")

    def _get_da_level(self) -> float:
        """
        Get current dopamine level from VTA.

        Returns:
            DA level [0, 1], or baseline if VTA not connected
        """
        if self._vta is None:
            return self.config.baseline_da

        da = self._vta.get_da_for_neural_field()
        self._last_da_level = da
        return da

    def get_current_da(self) -> float:
        """
        Get current DA level (public API).

        Returns:
            DA level [0, 1]
        """
        return self._get_da_level()

    def compute_da_modulation(self, da_level: float | None = None) -> tuple[float, float]:
        """
        Compute DA modulation factors for A+ and A-.

        High DA (>0.5):
        - Increases LTP (A+ * (1 + gain * da_mod))
        - Decreases LTD (A- * (1 - gain * da_mod))

        Low DA (<0.5):
        - Decreases LTP (A+ * (1 + gain * da_mod)) [negative da_mod]
        - Increases LTD (A- * (1 - gain * da_mod)) [negative da_mod]

        Args:
            da_level: DA level to use (gets from VTA if None)

        Returns:
            (ltp_modulation, ltd_modulation) multipliers for A+/A-
        """
        if da_level is None:
            da_level = self._get_da_level()

        # Normalize around baseline: range [-1, 1]
        # da_mod = +1 when DA = 1.0 (max reward)
        # da_mod = 0 when DA = baseline (neutral)
        # da_mod = -1 when DA = 0.0 (max punishment)
        da_mod = (da_level - self.config.baseline_da) / self.config.baseline_da
        da_mod = np.clip(da_mod, -1.0, 1.0)

        # Apply threshold - no modulation for small DA changes
        if abs(da_mod) < self.config.da_threshold:
            da_mod = 0.0

        # LTP modulation: High DA increases LTP
        # Range: [1 - ltp_gain, 1 + ltp_gain]
        ltp_mod = 1.0 + self.config.ltp_da_gain * da_mod
        ltp_mod = max(0.1, ltp_mod)  # Don't completely suppress

        # LTD modulation: High DA decreases LTD (inverse relationship)
        # Range: [1 - ltd_gain, 1 + ltd_gain]
        ltd_mod = 1.0 - self.config.ltd_da_gain * da_mod
        ltd_mod = max(0.1, ltd_mod)  # Don't completely suppress

        return ltp_mod, ltd_mod

    def get_da_modulated_rates(self, da_level: float | None = None) -> tuple[float, float]:
        """
        Get A+/A- modulated by current DA level.

        This is the main API for getting modulated learning rates.

        Args:
            da_level: DA level to use (gets from VTA if None)

        Returns:
            (a_plus_modulated, a_minus_modulated) learning rates
        """
        # Get modulation factors
        ltp_mod, ltd_mod = self.compute_da_modulation(da_level)

        # Apply to base STDP rates
        a_plus_mod = self.stdp.config.a_plus * ltp_mod
        a_minus_mod = self.stdp.config.a_minus * ltd_mod

        # Cache for efficiency
        self._last_modulation = (a_plus_mod, a_minus_mod)

        return a_plus_mod, a_minus_mod

    def should_gate_learning(self, da_level: float | None = None) -> bool:
        """
        Check if learning should be gated by DA level.

        Implements DA-based learning gate: only allow plasticity
        when DA is sufficiently high (signals importance).

        Args:
            da_level: DA level to check (gets from VTA if None)

        Returns:
            True if learning should proceed, False if gated
        """
        if not self.config.enable_gating:
            return True

        if da_level is None:
            da_level = self._get_da_level()

        return da_level >= self.config.min_da_for_learning

    def compute_modulated_stdp_delta(
        self,
        delta_t_ms: float,
        current_weight: float | None = None,
        da_level: float | None = None
    ) -> float:
        """
        Compute STDP weight change with DA modulation.

        This is a convenience method that combines:
        1. Getting DA level
        2. Computing modulated A+/A-
        3. Computing STDP delta with modulated rates

        Args:
            delta_t_ms: Spike timing difference (ms)
            current_weight: Current synaptic weight
            da_level: DA level (gets from VTA if None)

        Returns:
            Weight change Δw with DA modulation
        """
        # Check gating
        if not self.should_gate_learning(da_level):
            return 0.0

        # Get DA level
        if da_level is None:
            da_level = self._get_da_level()

        # Get modulated rates
        a_plus_mod, a_minus_mod = self.get_da_modulated_rates(da_level)

        # Temporarily override STDP config
        original_a_plus = self.stdp.config.a_plus
        original_a_minus = self.stdp.config.a_minus

        try:
            self.stdp.config.a_plus = a_plus_mod
            self.stdp.config.a_minus = a_minus_mod

            # Compute STDP delta with modulated rates
            delta_w = self.stdp.compute_stdp_delta(delta_t_ms, current_weight)

            return delta_w
        finally:
            # Restore original rates
            self.stdp.config.a_plus = original_a_plus
            self.stdp.config.a_minus = original_a_minus

    def get_stats(self) -> dict:
        """Get bridge statistics."""
        da_level = self._get_da_level()
        ltp_mod, ltd_mod = self.compute_da_modulation(da_level)
        a_plus_mod, a_minus_mod = self.get_da_modulated_rates(da_level)

        return {
            "da_level": da_level,
            "ltp_modulation": ltp_mod,
            "ltd_modulation": ltd_mod,
            "a_plus_modulated": a_plus_mod,
            "a_minus_modulated": a_minus_mod,
            "a_plus_base": self.stdp.config.a_plus,
            "a_minus_base": self.stdp.config.a_minus,
            "learning_gated": not self.should_gate_learning(da_level),
            "vta_connected": self._vta is not None,
        }


# Singleton instance for global use
_stdp_vta_bridge: STDPVTABridge | None = None


def get_stdp_vta_bridge(
    stdp_learner: STDPLearner | None = None,
    vta_circuit: VTACircuit | None = None
) -> STDPVTABridge:
    """
    Get or create global STDPVTABridge instance.

    Args:
        stdp_learner: STDP learner (uses global if None)
        vta_circuit: VTA circuit (optional)

    Returns:
        Global STDPVTABridge instance
    """
    global _stdp_vta_bridge

    if _stdp_vta_bridge is None:
        if stdp_learner is None:
            from t4dm.learning.stdp import get_stdp_learner
            stdp_learner = get_stdp_learner()

        _stdp_vta_bridge = STDPVTABridge(stdp_learner, vta_circuit)

    return _stdp_vta_bridge


def reset_stdp_vta_bridge() -> None:
    """Reset global STDP-VTA bridge (for testing)."""
    global _stdp_vta_bridge
    _stdp_vta_bridge = None


__all__ = [
    "STDPVTABridge",
    "STDPVTAConfig",
    "get_stdp_vta_bridge",
    "reset_stdp_vta_bridge",
]
