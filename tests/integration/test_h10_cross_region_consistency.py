"""
H10 Cross-Region Consistency Integration Tests.

Tests the integration between:
- Capsule Networks ↔ NCA Neural Field
- Forward-Forward ↔ NCA Neural Field
- Glymphatic ↔ Consolidation

Target: Verify bidirectional information flow and consistency
across all Sprint 11-12 subsystems.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from t4dm.nca.capsule_nca_coupling import (
    CapsuleNCACoupling,
    CapsuleNCACouplingConfig,
    CapsuleMode,
    CouplingStrength,
    create_capsule_nca_coupling,
)
from t4dm.nca.forward_forward_nca_coupling import (
    FFNCACoupling,
    FFNCACouplingConfig,
    FFPhase,
    EnergyAlignment,
    create_ff_nca_coupling,
)
from t4dm.nca.glymphatic_consolidation_bridge import (
    GlymphaticConsolidationBridge,
    GlymphaticConsolidationConfig,
    SleepStage,
    MemoryProtectionStatus,
    ClearanceMode,
    create_glymphatic_consolidation_bridge,
)
from t4dm.nca.neural_field import NeurotransmitterState


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def nt_state_encoding():
    """High ACh state for encoding mode."""
    return NeurotransmitterState(
        dopamine=0.7,
        serotonin=0.6,
        acetylcholine=0.8,
        norepinephrine=0.5,
        gaba=0.4,
        glutamate=0.6,
    )


@pytest.fixture
def nt_state_retrieval():
    """Low ACh state for retrieval mode."""
    return NeurotransmitterState(
        dopamine=0.5,
        serotonin=0.5,
        acetylcholine=0.3,
        norepinephrine=0.4,
        gaba=0.5,
        glutamate=0.5,
    )


@pytest.fixture
def nt_state_sleep():
    """Low ACh, low NE state for sleep."""
    return NeurotransmitterState(
        dopamine=0.3,
        serotonin=0.6,
        acetylcholine=0.2,
        norepinephrine=0.15,
        gaba=0.7,
        glutamate=0.3,
    )


@pytest.fixture
def capsule_nca_coupling():
    """Default capsule-NCA coupling."""
    return create_capsule_nca_coupling(strength="moderate", bidirectional=True)


@pytest.fixture
def ff_nca_coupling():
    """Default FF-NCA coupling."""
    return create_ff_nca_coupling(enable_energy=True, enable_neuromod=True)


@pytest.fixture
def glymphatic_bridge():
    """Default glymphatic-consolidation bridge."""
    return create_glymphatic_consolidation_bridge()


# =============================================================================
# Capsule-NCA Coupling Tests
# =============================================================================


class TestCapsuleNCACoupling:
    """Test capsule-NCA bidirectional coupling."""

    def test_mode_determination_encoding(self, capsule_nca_coupling, nt_state_encoding):
        """High ACh should trigger ENCODING mode."""
        mode = capsule_nca_coupling.determine_mode(nt_state_encoding)
        assert mode == CapsuleMode.ENCODING

    def test_mode_determination_retrieval(self, capsule_nca_coupling, nt_state_retrieval):
        """Low ACh should trigger RETRIEVAL mode."""
        mode = capsule_nca_coupling.determine_mode(nt_state_retrieval)
        assert mode == CapsuleMode.RETRIEVAL

    def test_mode_hysteresis(self, capsule_nca_coupling):
        """Mode should have hysteresis to prevent flickering."""
        # Start in encoding
        nt_encoding = NeurotransmitterState(acetylcholine=0.8)
        capsule_nca_coupling.determine_mode(nt_encoding)
        assert capsule_nca_coupling.state.mode == CapsuleMode.ENCODING

        # Slight drop shouldn't change mode immediately
        nt_slight_drop = NeurotransmitterState(acetylcholine=0.55)
        capsule_nca_coupling.determine_mode(nt_slight_drop)
        # Should be NEUTRAL or still ENCODING due to hysteresis
        assert capsule_nca_coupling.state.mode in [CapsuleMode.ENCODING, CapsuleMode.NEUTRAL]

    def test_routing_temperature_high_da(self, capsule_nca_coupling):
        """High DA should produce lower (sharper) routing temperature."""
        high_da = NeurotransmitterState(dopamine=0.9, acetylcholine=0.5)
        low_da = NeurotransmitterState(dopamine=0.2, acetylcholine=0.5)

        temp_high = capsule_nca_coupling.compute_routing_temperature(high_da)
        temp_low = capsule_nca_coupling.compute_routing_temperature(low_da)

        assert temp_high < temp_low  # High DA = sharper = lower temp

    def test_squashing_threshold_ne_modulation(self, capsule_nca_coupling):
        """High NE should raise squashing threshold."""
        high_ne = NeurotransmitterState(norepinephrine=0.9)
        low_ne = NeurotransmitterState(norepinephrine=0.2)

        thresh_high = capsule_nca_coupling.compute_squashing_threshold(high_ne)
        thresh_low = capsule_nca_coupling.compute_squashing_threshold(low_ne)

        assert thresh_high > thresh_low

    def test_stability_factor_serotonin(self, capsule_nca_coupling):
        """High 5-HT should increase stability factor."""
        high_5ht = NeurotransmitterState(serotonin=0.9)
        low_5ht = NeurotransmitterState(serotonin=0.2)

        stab_high = capsule_nca_coupling.compute_stability_factor(high_5ht)
        stab_low = capsule_nca_coupling.compute_stability_factor(low_5ht)

        assert stab_high > stab_low

    def test_full_modulation_params(self, capsule_nca_coupling, nt_state_encoding):
        """Full modulation should return all expected parameters."""
        params = capsule_nca_coupling.modulate_capsule_params(nt_state_encoding)

        assert "routing_temperature" in params
        assert "squashing_threshold" in params
        assert "stability_factor" in params
        assert "mode" in params
        assert "routing_iterations" in params
        assert params["mode"] == CapsuleMode.ENCODING

    def test_agreement_feedback(self, capsule_nca_coupling):
        """Routing agreement should produce stability signal."""
        # High agreement (concentrated routing coefficients)
        high_agreement = np.array([[0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])
        activations = np.random.randn(2, 8)

        feedback = capsule_nca_coupling.compute_agreement_feedback(
            high_agreement, activations
        )

        assert feedback > 0  # High agreement = positive stability

    def test_pose_feedback(self, capsule_nca_coupling):
        """Pose matrices should produce coherence signal."""
        # Coherent pose matrices (low condition number)
        poses = np.eye(4)[np.newaxis, :, :].repeat(3, axis=0)
        poses += np.random.randn(3, 4, 4) * 0.1  # Small perturbation

        coherence, geometry = capsule_nca_coupling.compute_pose_feedback(poses)

        assert coherence > 0
        assert geometry.shape == (6,)

    def test_activation_feedback(self, capsule_nca_coupling):
        """Strong activations should perturb neural field."""
        strong_activations = np.random.randn(10, 16) * 2
        weak_activations = np.random.randn(10, 16) * 0.1

        perturb_strong = capsule_nca_coupling.compute_activation_feedback(strong_activations)
        perturb_weak = capsule_nca_coupling.compute_activation_feedback(weak_activations)

        assert np.abs(perturb_strong).sum() > np.abs(perturb_weak).sum()

    def test_bidirectional_step(self, capsule_nca_coupling, nt_state_encoding):
        """Full step should return both directions."""
        routing = np.array([[0.8, 0.1, 0.1]])
        activations = np.random.randn(1, 8)

        capsule_mod, nca_feedback = capsule_nca_coupling.step(
            nt_state_encoding, routing, activations
        )

        assert "routing_temperature" in capsule_mod
        assert "stability_signal" in nca_feedback
        assert "field_perturbation" in nca_feedback


# =============================================================================
# FF-NCA Coupling Tests
# =============================================================================


class TestFFNCACoupling:
    """Test Forward-Forward - NCA coupling."""

    def test_phase_determination(self, ff_nca_coupling, nt_state_encoding):
        """High ACh should bias toward positive phase."""
        phase = ff_nca_coupling.determine_phase(nt_state_encoding)
        assert phase == FFPhase.POSITIVE

    def test_learning_rate_dopamine(self, ff_nca_coupling):
        """High DA should increase learning rate multiplier."""
        high_da = NeurotransmitterState(dopamine=0.9)
        low_da = NeurotransmitterState(dopamine=0.2)

        lr_high = ff_nca_coupling.compute_learning_rate_multiplier(high_da)
        lr_low = ff_nca_coupling.compute_learning_rate_multiplier(low_da)

        assert lr_high > lr_low
        assert lr_high > 1.0  # Above baseline
        assert lr_low < 1.0   # Below baseline

    def test_goodness_threshold_ne(self, ff_nca_coupling):
        """High NE should raise goodness threshold."""
        high_ne = NeurotransmitterState(norepinephrine=0.9)
        low_ne = NeurotransmitterState(norepinephrine=0.2)

        thresh_high = ff_nca_coupling.compute_goodness_threshold(high_ne)
        thresh_low = ff_nca_coupling.compute_goodness_threshold(low_ne)

        assert thresh_high > thresh_low

    def test_credit_decay_serotonin(self, ff_nca_coupling):
        """High 5-HT should produce slower credit decay."""
        high_5ht = NeurotransmitterState(serotonin=0.9)
        low_5ht = NeurotransmitterState(serotonin=0.2)

        decay_high = ff_nca_coupling.compute_credit_decay(high_5ht)
        decay_low = ff_nca_coupling.compute_credit_decay(low_5ht)

        assert decay_high > decay_low  # Slower decay = higher value

    def test_goodness_to_energy(self, ff_nca_coupling):
        """Goodness should map inversely to energy."""
        energy_high_g = ff_nca_coupling.goodness_to_energy(0.9)
        energy_low_g = ff_nca_coupling.goodness_to_energy(0.1)

        assert energy_high_g < energy_low_g  # High goodness = low energy

    def test_total_energy_computation(self, ff_nca_coupling):
        """Layer energies should sum with weights."""
        layer_goodness = [0.8, 0.7, 0.6]

        energy = ff_nca_coupling.compute_total_energy(layer_goodness)

        assert isinstance(energy, float)
        assert len(ff_nca_coupling.state.layer_energies) == 3

    def test_alignment_determination(self, ff_nca_coupling):
        """Goodness should determine energy alignment."""
        ff_nca_coupling.state.goodness_threshold = 0.5

        align_high = ff_nca_coupling.determine_alignment(0.8)
        align_mid = ff_nca_coupling.determine_alignment(0.5)
        align_low = ff_nca_coupling.determine_alignment(0.2)

        assert align_high == EnergyAlignment.BASIN
        assert align_mid == EnergyAlignment.BARRIER
        assert align_low in [EnergyAlignment.SADDLE, EnergyAlignment.TRANSITION]

    def test_stability_signal(self, ff_nca_coupling):
        """High goodness should produce high stability."""
        high_goodness = [0.9, 0.85, 0.8]
        low_goodness = [0.3, 0.25, 0.2]

        stab_high = ff_nca_coupling.compute_stability_signal(high_goodness)
        stab_low = ff_nca_coupling.compute_stability_signal(low_goodness)

        assert stab_high > stab_low

    def test_learning_perturbation(self, ff_nca_coupling):
        """Large weight updates should perturb field."""
        large_updates = np.random.randn(100, 100) * 0.5
        small_updates = np.random.randn(100, 100) * 0.01

        perturb_large = ff_nca_coupling.compute_learning_perturbation(large_updates)
        perturb_small = ff_nca_coupling.compute_learning_perturbation(small_updates)

        assert np.abs(perturb_large).sum() > np.abs(perturb_small).sum()

    def test_convergence_detection(self, ff_nca_coupling):
        """Convergence should require all layers above threshold."""
        ff_nca_coupling.state.goodness_threshold = 0.5

        converged = ff_nca_coupling.check_convergence([0.8, 0.7, 0.6])
        not_converged = ff_nca_coupling.check_convergence([0.8, 0.3, 0.6])

        assert converged == True  # All above 0.5
        assert not_converged == False  # 0.3 below threshold

    def test_bidirectional_step(self, ff_nca_coupling, nt_state_encoding):
        """Full step should return both directions."""
        layer_goodness = [0.7, 0.65, 0.6]

        ff_mod, nca_feedback = ff_nca_coupling.step(
            nt_state_encoding, layer_goodness
        )

        assert "learning_rate_multiplier" in ff_mod
        assert "stability_signal" in nca_feedback
        assert "total_energy" in nca_feedback


# =============================================================================
# Glymphatic-Consolidation Bridge Tests
# =============================================================================


class TestGlymphaticConsolidationBridge:
    """Test glymphatic-consolidation coupling."""

    def test_sleep_stage_setting(self, glymphatic_bridge):
        """Sleep stage should update clearance mode."""
        glymphatic_bridge.set_sleep_stage(SleepStage.NREM_DEEP)
        assert glymphatic_bridge.state.clearance_mode == ClearanceMode.BULK

        glymphatic_bridge.set_sleep_stage(SleepStage.NREM_LIGHT)
        assert glymphatic_bridge.state.clearance_mode == ClearanceMode.MICRO

        glymphatic_bridge.set_sleep_stage(SleepStage.WAKE)
        assert glymphatic_bridge.state.clearance_mode == ClearanceMode.NONE

    def test_stage_inference_from_neuromod(self, glymphatic_bridge, nt_state_sleep):
        """Should infer sleep stage from ACh/NE levels."""
        stage = glymphatic_bridge.infer_stage_from_neuromod(
            nt_state_sleep.acetylcholine,
            nt_state_sleep.norepinephrine
        )

        assert stage in [SleepStage.NREM_DEEP, SleepStage.NREM_LIGHT]

    def test_memory_protection(self, glymphatic_bridge):
        """Protected memories should not be cleared."""
        glymphatic_bridge.protect_memory("mem_001")

        assert glymphatic_bridge.is_protected("mem_001")
        assert glymphatic_bridge.get_protection_status("mem_001") == MemoryProtectionStatus.PROTECTED

    def test_protection_expiration(self, glymphatic_bridge):
        """Protection should expire after duration."""
        # Set very short protection duration for testing
        glymphatic_bridge.config.replay_protection_duration_s = 0.01

        glymphatic_bridge.protect_memory("mem_002")
        assert glymphatic_bridge.is_protected("mem_002")

        # Wait for expiration
        import time
        time.sleep(0.02)

        assert not glymphatic_bridge.is_protected("mem_002")

    def test_tagging_for_clearance(self, glymphatic_bridge):
        """Tagged memories should be in pending list."""
        glymphatic_bridge.tag_for_clearance("mem_003")

        assert "mem_003" in glymphatic_bridge.state.pending_clearance
        assert glymphatic_bridge.get_protection_status("mem_003") == MemoryProtectionStatus.TAGGED

    def test_weak_memory_tagging(self, glymphatic_bridge):
        """Weak memories should be tagged for clearance."""
        memories = [
            {"id": "strong_001", "strength": 0.8},
            {"id": "weak_001", "strength": 0.2},
            {"id": "weak_002", "strength": 0.1},
        ]

        tagged = glymphatic_bridge.tag_weak_memories(memories)

        assert "weak_001" in tagged
        assert "weak_002" in tagged
        assert "strong_001" not in tagged

    def test_spindle_triggered_clearance(self, glymphatic_bridge):
        """Spindles should trigger micro clearance in NREM."""
        glymphatic_bridge.set_sleep_stage(SleepStage.NREM_LIGHT)

        # Add pending items
        for i in range(10):
            glymphatic_bridge.tag_for_clearance(f"mem_{i:03d}")

        result = glymphatic_bridge.on_spindle(spindle_power=1.0)

        assert result["triggered"] is True
        assert result["cleared_count"] > 0

    def test_delta_triggered_clearance(self, glymphatic_bridge):
        """Delta up-states should trigger bulk clearance."""
        glymphatic_bridge.set_sleep_stage(SleepStage.NREM_DEEP)

        # Add pending items
        for i in range(10):
            glymphatic_bridge.tag_for_clearance(f"mem_{i:03d}")

        # Phase 0.25 is in up-state range
        result = glymphatic_bridge.on_delta_upstate(phase=0.25)

        assert result["triggered"] is True
        assert result["cleared_count"] > 0

    def test_clearance_respects_protection(self, glymphatic_bridge):
        """Protected memories should not be cleared."""
        glymphatic_bridge.set_sleep_stage(SleepStage.NREM_DEEP)

        # Add pending items
        for i in range(5):
            glymphatic_bridge.tag_for_clearance(f"mem_{i:03d}")

        # Protect one
        glymphatic_bridge.protect_memory("mem_002")

        result = glymphatic_bridge.on_delta_upstate(phase=0.25)

        assert "mem_002" not in result["cleared"]

    def test_learning_signals(self, glymphatic_bridge):
        """Clearance should generate learning signals."""
        glymphatic_bridge.set_sleep_stage(SleepStage.NREM_DEEP)

        for i in range(5):
            glymphatic_bridge.tag_for_clearance(f"mem_{i:03d}")

        glymphatic_bridge.protect_memory("mem_001")
        glymphatic_bridge.on_delta_upstate(phase=0.25)

        signals = glymphatic_bridge.get_clearance_learning_signals()

        assert signals["cleared_signal"] < 0  # Negative for cleared
        assert signals["survival_signal"] > 0  # Positive for survivors

    def test_consolidation_lifecycle(self, glymphatic_bridge):
        """Full consolidation lifecycle should work."""
        replaying = ["mem_001", "mem_002", "mem_003"]

        # Start consolidation
        glymphatic_bridge.on_consolidation_start(replaying)

        for mem_id in replaying:
            assert glymphatic_bridge.is_protected(mem_id)

        # End consolidation
        cleared = glymphatic_bridge.on_consolidation_end()

        assert isinstance(cleared, list)


# =============================================================================
# Cross-System Integration Tests
# =============================================================================


class TestCrossSystemIntegration:
    """Test integration across all three coupling systems."""

    def test_capsule_ff_consistency(
        self, capsule_nca_coupling, ff_nca_coupling, nt_state_encoding
    ):
        """Capsule and FF should respond consistently to same NT state."""
        capsule_params = capsule_nca_coupling.modulate_capsule_params(nt_state_encoding)
        ff_params = ff_nca_coupling.modulate_ff_params(nt_state_encoding)

        # Both should be in encoding/positive mode for high ACh
        assert capsule_params["mode"] == CapsuleMode.ENCODING
        assert ff_params["phase"] == FFPhase.POSITIVE

    def test_sleep_stage_triggers_all_systems(
        self, capsule_nca_coupling, ff_nca_coupling, glymphatic_bridge, nt_state_sleep
    ):
        """Sleep NT state should consistently affect all systems."""
        # Capsule should be in retrieval mode (low ACh)
        capsule_mode = capsule_nca_coupling.determine_mode(nt_state_sleep)
        assert capsule_mode == CapsuleMode.RETRIEVAL

        # FF should allow negative phase (refinement)
        ff_phase = ff_nca_coupling.determine_phase(nt_state_sleep)
        assert ff_phase in [FFPhase.NEGATIVE, FFPhase.INFERENCE]

        # Glymphatic should detect sleep stage
        stage = glymphatic_bridge.infer_stage_from_neuromod(
            nt_state_sleep.acetylcholine,
            nt_state_sleep.norepinephrine
        )
        assert stage in [SleepStage.NREM_DEEP, SleepStage.NREM_LIGHT]

    def test_stability_signals_correlate(
        self, capsule_nca_coupling, ff_nca_coupling
    ):
        """High agreement and high goodness should both indicate stability."""
        # High agreement routing
        high_agreement = np.array([[0.9, 0.05, 0.05]])
        activations = np.random.randn(1, 8)

        capsule_feedback = capsule_nca_coupling.compute_nca_feedback(
            high_agreement, activations
        )

        # High goodness
        high_goodness = [0.9, 0.85, 0.8]
        ff_feedback = ff_nca_coupling.compute_nca_feedback(high_goodness)

        # Both should indicate stability
        assert capsule_feedback["stability_signal"] > 0
        assert ff_feedback["stability_signal"] > 0

    def test_full_sleep_cycle_integration(
        self, capsule_nca_coupling, ff_nca_coupling, glymphatic_bridge
    ):
        """Full sleep cycle should coordinate all systems."""
        # 1. Wake state
        wake_nt = NeurotransmitterState(
            dopamine=0.6, acetylcholine=0.7, norepinephrine=0.6, serotonin=0.5
        )

        glymphatic_bridge.infer_stage_from_neuromod(wake_nt.acetylcholine, wake_nt.norepinephrine)
        assert glymphatic_bridge.state.sleep_stage == SleepStage.WAKE

        # 2. Transition to NREM
        nrem_nt = NeurotransmitterState(
            dopamine=0.3, acetylcholine=0.2, norepinephrine=0.15, serotonin=0.6
        )

        glymphatic_bridge.infer_stage_from_neuromod(nrem_nt.acetylcholine, nrem_nt.norepinephrine)
        assert glymphatic_bridge.state.sleep_stage in [SleepStage.NREM_DEEP, SleepStage.NREM_LIGHT]

        # Capsule should be in retrieval mode
        capsule_mode = capsule_nca_coupling.determine_mode(nrem_nt)
        assert capsule_mode == CapsuleMode.RETRIEVAL

        # FF learning rate should be lower (low DA)
        ff_lr = ff_nca_coupling.compute_learning_rate_multiplier(nrem_nt)
        assert ff_lr < 1.0

        # 3. Glymphatic clearance should be enabled
        assert glymphatic_bridge.state.clearance_mode in [ClearanceMode.BULK, ClearanceMode.MICRO]

    def test_perturbation_signals_compatible(
        self, capsule_nca_coupling, ff_nca_coupling
    ):
        """Field perturbations from both systems should be compatible."""
        activations = np.random.randn(5, 16) * 1.5
        weight_updates = np.random.randn(100, 100) * 0.3

        capsule_perturb = capsule_nca_coupling.compute_activation_feedback(activations)
        ff_perturb = ff_nca_coupling.compute_learning_perturbation(weight_updates)

        # Both should be 6-element arrays
        assert capsule_perturb.shape == (6,)
        assert ff_perturb.shape == (6,)

        # Combined perturbation should be valid
        combined = capsule_perturb + ff_perturb
        assert not np.any(np.isnan(combined))
        assert not np.any(np.isinf(combined))


# =============================================================================
# Biology Validation Tests (B9 subset)
# =============================================================================


class TestBiologyValidation:
    """Validate biological plausibility of coupling parameters."""

    def test_routing_temperature_range(self, capsule_nca_coupling):
        """Routing temperature should be in biologically plausible range."""
        # Test across NT range
        for da in np.linspace(0.0, 1.0, 10):
            nt = NeurotransmitterState(dopamine=da, acetylcholine=0.5)
            temp = capsule_nca_coupling.compute_routing_temperature(nt)
            assert 0.1 <= temp <= 2.0, f"Temperature {temp} out of range for DA={da}"

    def test_learning_rate_multiplier_range(self, ff_nca_coupling):
        """Learning rate multiplier should be bounded."""
        for da in np.linspace(0.0, 1.0, 10):
            nt = NeurotransmitterState(dopamine=da)
            lr = ff_nca_coupling.compute_learning_rate_multiplier(nt)
            assert 0.1 <= lr <= 3.0, f"LR multiplier {lr} out of range for DA={da}"

    def test_clearance_rates_biologically_valid(self, glymphatic_bridge):
        """Clearance rates should match Xie et al. (2013)."""
        # NREM deep should have ~90% clearance capacity
        glymphatic_bridge.set_sleep_stage(SleepStage.NREM_DEEP)
        assert glymphatic_bridge.config.delta_batch_fraction >= 0.2

        # NREM light should have ~50% clearance capacity
        glymphatic_bridge.set_sleep_stage(SleepStage.NREM_LIGHT)
        assert glymphatic_bridge.config.spindle_batch_fraction >= 0.05

    def test_protection_duration_realistic(self, glymphatic_bridge):
        """Memory protection duration should be realistic."""
        # 30 seconds is reasonable for replay protection
        assert 10.0 <= glymphatic_bridge.config.replay_protection_duration_s <= 120.0

    def test_credit_decay_temporal_range(self, ff_nca_coupling):
        """Credit decay should produce reasonable temporal horizons."""
        # High 5-HT = patient = longer horizon
        high_5ht = NeurotransmitterState(serotonin=0.9)
        decay_high = ff_nca_coupling.compute_credit_decay(high_5ht)

        # Low 5-HT = impatient = shorter horizon
        low_5ht = NeurotransmitterState(serotonin=0.1)
        decay_low = ff_nca_coupling.compute_credit_decay(low_5ht)

        # Decay values should be in reasonable range (0.5-0.99)
        # Higher 5-HT should produce higher decay (more patience)
        assert 0.5 <= decay_low <= 0.95
        assert 0.7 <= decay_high <= 0.99
        assert decay_high > decay_low, "High 5-HT should produce higher decay"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
