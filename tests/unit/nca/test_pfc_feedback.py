"""
Test PFC feedback modulation to neuromodulatory nuclei.

Tests top-down prefrontal cortex control of VTA, LC, and Raphe nuclei.
"""

import pytest

from ww.nca.connectome import Connectome, NTSystem
from ww.nca.locus_coeruleus import LocusCoeruleus
from ww.nca.raphe import RapheNucleus
from ww.nca.vta import VTACircuit


class TestVTAPFCModulation:
    """Test VTA PFC feedback modulation."""

    def test_vta_pfc_goal_context(self):
        """Test dlPFC goal-directed modulation of VTA."""
        vta = VTACircuit()
        initial_tonic = vta.config.tonic_da_level

        # High PFC signal (goal-directed)
        vta.receive_pfc_modulation(pfc_signal=0.8, context="goal")

        # Should increase tonic DA level
        assert vta.config.tonic_da_level > initial_tonic
        assert vta.config.tonic_da_level <= 0.5  # Within bounds

    def test_vta_pfc_value_context(self):
        """Test vmPFC value-based modulation of VTA."""
        vta = VTACircuit()
        initial_value = vta.state.value_estimate

        # High PFC signal (value-based)
        vta.receive_pfc_modulation(pfc_signal=0.9, context="value")

        # Should bias value estimate
        assert vta.state.value_estimate != initial_value
        assert 0.0 <= vta.state.value_estimate <= 1.0

    def test_vta_pfc_low_signal(self):
        """Test low PFC signal reduces modulation."""
        vta = VTACircuit()
        initial_tonic = vta.config.tonic_da_level

        # Low PFC signal
        vta.receive_pfc_modulation(pfc_signal=0.1, context="goal")

        # Should decrease or maintain tonic DA
        assert vta.config.tonic_da_level <= initial_tonic * 1.1

    def test_vta_pfc_invalid_context(self):
        """Test invalid context is handled gracefully."""
        vta = VTACircuit()

        # Should not crash with invalid context
        vta.receive_pfc_modulation(pfc_signal=0.5, context="invalid")


class TestLCPFCModulation:
    """Test LC PFC feedback modulation."""

    def test_lc_pfc_increases_arousal(self):
        """Test high PFC increases arousal (tonic mode)."""
        lc = LocusCoeruleus()
        initial_arousal = lc.state.arousal_drive

        # High PFC signal → focused attention (tonic)
        lc.receive_pfc_modulation(pfc_signal=0.8)

        # Should increase arousal drive
        assert lc.state.arousal_drive > initial_arousal
        assert lc.state.arousal_drive <= 1.0

    def test_lc_pfc_decreases_arousal(self):
        """Test low PFC decreases arousal (exploratory)."""
        lc = LocusCoeruleus()
        lc.state.arousal_drive = 0.6  # Start elevated

        # Low PFC signal → exploratory attention (phasic)
        lc.receive_pfc_modulation(pfc_signal=0.2)

        # Should decrease arousal drive
        assert lc.state.arousal_drive < 0.6
        assert lc.state.arousal_drive >= 0.0

    def test_lc_pfc_modulates_phasic_gain(self):
        """Test PFC modulates phasic responsiveness."""
        lc = LocusCoeruleus()

        # High PFC → reduced phasic gain (less distractible)
        lc.receive_pfc_modulation(pfc_signal=0.9)
        high_pfc_gain = lc.state.pfc_gain

        # Low PFC → increased phasic gain (more distractible)
        lc.receive_pfc_modulation(pfc_signal=0.1)
        low_pfc_gain = lc.state.pfc_gain

        assert low_pfc_gain > high_pfc_gain
        assert 0.5 <= high_pfc_gain <= 1.0
        assert 0.5 <= low_pfc_gain <= 1.0

    def test_lc_pfc_mode_transition(self):
        """Test PFC drives LC mode transitions."""
        lc = LocusCoeruleus()
        lc.state.arousal_drive = 0.3  # Start low

        # High PFC should push toward tonic optimal
        lc.receive_pfc_modulation(pfc_signal=0.8)
        lc.step(dt=0.1)

        # Check mode moved toward tonic optimal
        assert lc.state.arousal_drive > 0.3


class TestRaphePFCModulation:
    """Test Raphe PFC feedback modulation."""

    def test_raphe_pfc_dampens_stress(self):
        """Test PFC dampens stress-induced 5-HT response."""
        raphe = RapheNucleus()
        raphe.state.stress_input = 0.8  # High stress

        initial_stress = raphe.state.stress_input

        # High PFC → stress reduction (emotional regulation)
        raphe.receive_pfc_modulation(pfc_signal=0.7)

        # Should reduce stress
        assert raphe.state.stress_input < initial_stress
        assert raphe.state.stress_input >= 0.0

    def test_raphe_pfc_mood_boost(self):
        """Test PFC provides mood stabilization."""
        raphe = RapheNucleus()
        initial_5ht = raphe.state.extracellular_5ht

        # High PFC → mood boost
        raphe.receive_pfc_modulation(pfc_signal=0.8)

        # Should increase 5-HT slightly
        assert raphe.state.extracellular_5ht > initial_5ht
        assert raphe.state.extracellular_5ht <= 1.0

    def test_raphe_pfc_low_signal(self):
        """Test low PFC has minimal effect."""
        raphe = RapheNucleus()
        raphe.state.stress_input = 0.5
        initial_stress = raphe.state.stress_input

        # Low PFC signal
        raphe.receive_pfc_modulation(pfc_signal=0.1)

        # Should have minimal stress damping
        stress_reduction = initial_stress - raphe.state.stress_input
        assert stress_reduction < 0.1  # Less than 10% reduction

    def test_raphe_pfc_combined_effects(self):
        """Test combined stress damping and mood boost."""
        raphe = RapheNucleus()
        raphe.state.stress_input = 0.9
        initial_5ht = raphe.state.extracellular_5ht

        # Strong PFC signal
        raphe.receive_pfc_modulation(pfc_signal=1.0)

        # Should reduce stress and boost 5-HT
        assert raphe.state.stress_input < 0.9
        assert raphe.state.extracellular_5ht > initial_5ht


class TestConnectomePFCBackprojections:
    """Test PFC backprojections in connectome."""

    def test_connectome_has_pfc_vta(self):
        """Test PFC → VTA pathway exists."""
        connectome = Connectome()

        # Find PFC → VTA pathway
        pfc_vta = [
            p for p in connectome.pathways
            if p.source == "PFC" and p.target == "VTA"
        ]

        assert len(pfc_vta) > 0
        pathway = pfc_vta[0]
        assert pathway.nt_system == NTSystem.GLUTAMATE
        assert pathway.strength > 0
        assert not pathway.is_inhibitory

    def test_connectome_has_pfc_lc(self):
        """Test PFC → LC pathway exists."""
        connectome = Connectome()

        # Find PFC → LC pathway
        pfc_lc = [
            p for p in connectome.pathways
            if p.source == "PFC" and p.target == "LC"
        ]

        assert len(pfc_lc) > 0
        pathway = pfc_lc[0]
        assert pathway.nt_system == NTSystem.GLUTAMATE
        assert pathway.strength > 0

    def test_connectome_has_pfc_raphe(self):
        """Test PFC → Raphe (DRN) pathway exists."""
        connectome = Connectome()

        # Find PFC → Raphe pathway
        pfc_raphe = [
            p for p in connectome.pathways
            if p.source == "PFC" and p.target == "Raphe"
        ]

        assert len(pfc_raphe) > 0
        pathway = pfc_raphe[0]
        assert pathway.nt_system == NTSystem.GLUTAMATE
        assert pathway.strength > 0

    def test_connectome_pfc_backprojections_count(self):
        """Test all PFC backprojections are present."""
        connectome = Connectome()

        # Count PFC outgoing pathways to neuromodulatory nuclei
        pfc_backproj = [
            p for p in connectome.pathways
            if p.source == "PFC" and p.target in ["VTA", "LC", "Raphe"]
        ]

        # Should have 3 backprojections
        assert len(pfc_backproj) == 3

    def test_connectome_pfc_backproj_properties(self):
        """Test PFC backprojections have correct properties."""
        connectome = Connectome()

        pfc_backproj = [
            p for p in connectome.pathways
            if p.source == "PFC" and p.target in ["VTA", "LC", "Raphe"]
        ]

        for pathway in pfc_backproj:
            # All should be glutamatergic
            assert pathway.nt_system == NTSystem.GLUTAMATE
            # All should be excitatory
            assert not pathway.is_inhibitory
            # Should have moderate strength
            assert 0.2 <= pathway.strength <= 0.5
            # Should have high probability
            assert pathway.probability >= 0.7


class TestIntegratedPFCModulation:
    """Test integrated PFC modulation scenarios."""

    def test_pfc_coordinated_modulation(self):
        """Test coordinated PFC modulation of all nuclei."""
        vta = VTACircuit()
        lc = LocusCoeruleus()
        raphe = RapheNucleus()

        # Simulate high PFC state (focused, goal-directed)
        pfc_signal = 0.8

        vta.receive_pfc_modulation(pfc_signal, context="goal")
        lc.receive_pfc_modulation(pfc_signal)
        raphe.receive_pfc_modulation(pfc_signal)

        # VTA should increase tonic DA
        assert vta.config.tonic_da_level > 0.3

        # LC should increase arousal
        assert lc.state.arousal_drive > 0.5

        # Raphe should have reduced stress (if any)
        assert raphe.state.stress_input < 0.5

    def test_pfc_low_exploration_mode(self):
        """Test low PFC enables exploration mode."""
        vta = VTACircuit()
        lc = LocusCoeruleus()
        lc.state.arousal_drive = 0.7  # Start elevated

        # Low PFC signal (exploratory state)
        pfc_signal = 0.2

        vta.receive_pfc_modulation(pfc_signal, context="goal")
        lc.receive_pfc_modulation(pfc_signal)

        # VTA should reduce tonic modulation
        # LC should reduce arousal (exploratory)
        assert lc.state.arousal_drive < 0.7

        # LC should have high phasic gain (more distractible)
        assert lc.state.pfc_gain > 0.7

    def test_pfc_bounds_checking(self):
        """Test PFC modulation respects bounds."""
        vta = VTACircuit()
        lc = LocusCoeruleus()
        raphe = RapheNucleus()

        # Extreme PFC signals
        vta.receive_pfc_modulation(pfc_signal=1.5, context="goal")  # Over max
        lc.receive_pfc_modulation(pfc_signal=-0.5)  # Below min
        raphe.receive_pfc_modulation(pfc_signal=2.0)  # Over max

        # All should be within bounds
        assert 0.1 <= vta.config.tonic_da_level <= 0.5
        assert 0.0 <= lc.state.arousal_drive <= 1.0
        assert 0.0 <= raphe.state.extracellular_5ht <= 1.0
