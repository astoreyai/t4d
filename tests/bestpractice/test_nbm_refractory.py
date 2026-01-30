"""
ATOM-P2-6: Tests for NBM phasic refractory period.

Ensures that NBM does not trigger multiple phasic bursts within
the refractory period (100ms default).
"""

import time

import numpy as np
import pytest


class TestNBMRefractory:
    """Test NBM phasic refractory period."""

    def test_nbm_respects_refractory_period(self):
        """P2-6: NBM skips phasic bursts during refractory period."""
        from ww.nca.nucleus_basalis import NucleusBasalisCircuit, NBMConfig

        config = NBMConfig(phasic_refractory_ms=100.0)
        nbm = NucleusBasalisCircuit(config)

        # First burst should trigger
        nbm._trigger_phasic_burst(salience=0.8)
        assert nbm.state.in_phasic_burst is True
        first_phasic_time = nbm._last_phasic_time

        # Immediately try another burst (should be skipped)
        nbm.state.in_phasic_burst = False  # Reset state to test
        nbm._trigger_phasic_burst(salience=0.9)

        # Should not have triggered (still in refractory)
        assert nbm.state.in_phasic_burst is False
        assert nbm._last_phasic_time == first_phasic_time  # Time unchanged

    def test_nbm_allows_burst_after_refractory(self):
        """P2-6: NBM allows phasic bursts after refractory period."""
        from ww.nca.nucleus_basalis import NucleusBasalisCircuit, NBMConfig

        config = NBMConfig(phasic_refractory_ms=50.0)  # Short refractory for test
        nbm = NucleusBasalisCircuit(config)

        # First burst
        nbm._trigger_phasic_burst(salience=0.8)
        assert nbm.state.in_phasic_burst is True
        first_time = nbm._last_phasic_time

        # Wait for refractory to pass
        time.sleep(0.06)  # 60ms > 50ms refractory

        # Reset burst state
        nbm.state.in_phasic_burst = False

        # Second burst should trigger
        nbm._trigger_phasic_burst(salience=0.9)
        assert nbm.state.in_phasic_burst is True
        assert nbm._last_phasic_time > first_time  # Time updated

    def test_nbm_first_burst_always_allowed(self):
        """P2-6: First phasic burst is always allowed (no prior timing)."""
        from ww.nca.nucleus_basalis import NucleusBasalisCircuit, NBMConfig

        nbm = NucleusBasalisCircuit(NBMConfig())

        # No previous burst, so should trigger immediately
        assert nbm._last_phasic_time is None
        nbm._trigger_phasic_burst(salience=0.8)
        assert nbm.state.in_phasic_burst is True
        assert nbm._last_phasic_time is not None
