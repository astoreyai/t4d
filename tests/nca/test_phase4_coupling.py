"""
Tests for Phase 4 Coupling Modules.

Tests the three Phase 4 cross-region coupling bridges:
1. FFNCACoupling - Forward-Forward ↔ NCA neural field
2. CapsuleNCACoupling - Capsule networks ↔ NCA neural field
3. GlymphaticConsolidationBridge - Glymphatic ↔ memory consolidation

These implement H10 (cross-region consistency) and B8 (waste clearance).
"""

from __future__ import annotations

import pytest
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from ww.nca.forward_forward_nca_coupling import (
    FFNCACoupling,
    FFNCACouplingConfig,
    FFNCACouplingState,
    FFPhase,
    EnergyAlignment,
)
from ww.nca.capsule_nca_coupling import (
    CapsuleNCACoupling,
    CapsuleNCACouplingConfig,
    CapsuleNCACouplingState,
    CapsuleMode,
    CouplingStrength,
)
from ww.nca.glymphatic_consolidation_bridge import (
    GlymphaticConsolidationBridge,
    GlymphaticConsolidationConfig,
    ConsolidationBridgeState,
    SleepStage,
    MemoryProtectionStatus,
    ClearanceMode,
)


# =============================================================================
# Mock NT State for testing
# =============================================================================


@dataclass
class MockNTState:
    """Mock neurotransmitter state for testing."""

    dopamine: float = 0.5
    acetylcholine: float = 0.5
    norepinephrine: float = 0.5
    serotonin: float = 0.5
    gaba: float = 0.5
    glutamate: float = 0.5


# =============================================================================
# FFNCACoupling Tests
# =============================================================================


class TestFFNCACoupling:
    """Tests for Forward-Forward ↔ NCA coupling."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        coupling = FFNCACoupling()

        assert coupling.config.da_learning_rate_gain == 0.5
        assert coupling.config.ach_phase_threshold == 0.6
        assert coupling.config.enable_energy_alignment is True
        assert coupling.config.enable_neuromod_gating is True
        assert coupling.state.phase == FFPhase.INFERENCE

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = FFNCACouplingConfig(
            da_learning_rate_gain=0.8,
            ach_phase_threshold=0.7,
            ne_threshold_gain=0.5,
        )
        coupling = FFNCACoupling(config=config)

        assert coupling.config.da_learning_rate_gain == 0.8
        assert coupling.config.ach_phase_threshold == 0.7

    def test_determine_phase_high_ach(self):
        """Test phase determination with high ACh (encoding mode)."""
        coupling = FFNCACoupling()
        nt_state = MockNTState(acetylcholine=0.8)

        phase = coupling.determine_phase(nt_state)

        assert phase == FFPhase.POSITIVE
        assert coupling.state.phase == FFPhase.POSITIVE

    def test_determine_phase_low_ach(self):
        """Test phase determination with low ACh (refinement mode)."""
        coupling = FFNCACoupling()
        nt_state = MockNTState(acetylcholine=0.3)

        phase = coupling.determine_phase(nt_state)

        assert phase == FFPhase.NEGATIVE
        assert coupling.state.phase == FFPhase.NEGATIVE

    def test_determine_phase_mid_ach(self):
        """Test phase determination with mid ACh (inference mode)."""
        coupling = FFNCACoupling()
        nt_state = MockNTState(acetylcholine=0.5)

        phase = coupling.determine_phase(nt_state)

        assert phase == FFPhase.INFERENCE

    def test_determine_phase_gating_disabled(self):
        """Test phase is INFERENCE when neuromod gating disabled."""
        config = FFNCACouplingConfig(enable_neuromod_gating=False)
        coupling = FFNCACoupling(config=config)
        nt_state = MockNTState(acetylcholine=0.9)

        phase = coupling.determine_phase(nt_state)

        assert phase == FFPhase.INFERENCE

    def test_learning_rate_multiplier_high_da(self):
        """Test high DA increases learning rate."""
        coupling = FFNCACoupling()
        nt_state = MockNTState(dopamine=0.9)

        multiplier = coupling.compute_learning_rate_multiplier(nt_state)

        assert multiplier > 1.0
        assert multiplier <= 3.0  # Max clamp

    def test_learning_rate_multiplier_low_da(self):
        """Test low DA decreases learning rate."""
        coupling = FFNCACoupling()
        nt_state = MockNTState(dopamine=0.1)

        multiplier = coupling.compute_learning_rate_multiplier(nt_state)

        assert multiplier < 1.0
        assert multiplier >= 0.1  # Min clamp

    def test_learning_rate_multiplier_baseline_da(self):
        """Test baseline DA gives multiplier ~1.0."""
        coupling = FFNCACoupling()
        nt_state = MockNTState(dopamine=0.5)

        multiplier = coupling.compute_learning_rate_multiplier(nt_state)

        assert 0.95 <= multiplier <= 1.05

    def test_learning_rate_multiplier_gating_disabled(self):
        """Test multiplier is 1.0 when gating disabled."""
        config = FFNCACouplingConfig(enable_neuromod_gating=False)
        coupling = FFNCACoupling(config=config)
        nt_state = MockNTState(dopamine=0.9)

        multiplier = coupling.compute_learning_rate_multiplier(nt_state)

        assert multiplier == 1.0

    def test_goodness_threshold_high_ne(self):
        """Test high NE increases goodness threshold (more selective)."""
        coupling = FFNCACoupling()
        nt_state = MockNTState(norepinephrine=0.9)

        threshold = coupling.compute_goodness_threshold(nt_state)

        assert threshold > coupling.config.goodness_threshold_base

    def test_goodness_threshold_low_ne(self):
        """Test low NE decreases goodness threshold (less selective)."""
        coupling = FFNCACoupling()
        nt_state = MockNTState(norepinephrine=0.1)

        threshold = coupling.compute_goodness_threshold(nt_state)

        assert threshold < coupling.config.goodness_threshold_base

    def test_state_history_update(self):
        """Test state history is properly bounded."""
        coupling = FFNCACoupling()

        # Add more than max_len entries
        for i in range(150):
            coupling.state.goodness_history.append(float(i))
            coupling.state.energy_history.append(float(i))

        coupling.state.update_history(max_len=100)

        assert len(coupling.state.goodness_history) == 100
        assert len(coupling.state.energy_history) == 100
        assert coupling.state.goodness_history[0] == 50.0  # Oldest kept


# =============================================================================
# CapsuleNCACoupling Tests
# =============================================================================


class TestCapsuleNCACoupling:
    """Tests for Capsule ↔ NCA coupling."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        coupling = CapsuleNCACoupling()

        assert coupling.config.da_routing_gain == 0.5
        assert coupling.config.ach_mode_threshold == 0.6
        assert coupling.config.coupling_strength == CouplingStrength.MODERATE
        assert coupling.state.mode == CapsuleMode.NEUTRAL

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = CapsuleNCACouplingConfig(
            da_routing_gain=0.8,
            coupling_strength=CouplingStrength.STRONG,
        )
        coupling = CapsuleNCACoupling(config=config)

        assert coupling.config.da_routing_gain == 0.8
        assert coupling.config.coupling_strength == CouplingStrength.STRONG

    def test_determine_mode_high_ach(self):
        """Test mode determination with high ACh (encoding)."""
        coupling = CapsuleNCACoupling()
        nt_state = MockNTState(acetylcholine=0.8)

        mode = coupling.determine_mode(nt_state)

        assert mode == CapsuleMode.ENCODING

    def test_determine_mode_low_ach(self):
        """Test mode determination with low ACh (retrieval)."""
        coupling = CapsuleNCACoupling()
        nt_state = MockNTState(acetylcholine=0.3)

        mode = coupling.determine_mode(nt_state)

        assert mode == CapsuleMode.RETRIEVAL

    def test_determine_mode_mid_ach(self):
        """Test mode determination with mid ACh (neutral)."""
        coupling = CapsuleNCACoupling()
        nt_state = MockNTState(acetylcholine=0.55)

        mode = coupling.determine_mode(nt_state)

        assert mode == CapsuleMode.NEUTRAL

    def test_mode_hysteresis_prevents_flickering(self):
        """Test hysteresis prevents rapid mode switching."""
        config = CapsuleNCACouplingConfig(
            ach_mode_threshold=0.6,
            mode_hysteresis=0.1,
        )
        coupling = CapsuleNCACoupling(config=config)

        # Start in ENCODING mode
        coupling.determine_mode(MockNTState(acetylcholine=0.8))
        assert coupling.state.mode == CapsuleMode.ENCODING

        # Small drop to 0.55 shouldn't change mode (within hysteresis)
        coupling.determine_mode(MockNTState(acetylcholine=0.55))
        assert coupling.state.mode == CapsuleMode.NEUTRAL

        # Larger drop to 0.45 should change to RETRIEVAL
        coupling.determine_mode(MockNTState(acetylcholine=0.45))
        assert coupling.state.mode == CapsuleMode.RETRIEVAL

    def test_routing_temperature_high_da(self):
        """Test high DA gives sharper routing (lower temperature)."""
        coupling = CapsuleNCACoupling()
        nt_state = MockNTState(dopamine=0.9, acetylcholine=0.5)

        temp = coupling.compute_routing_temperature(nt_state)

        # Higher DA should give lower temperature
        assert temp < 1.0

    def test_routing_temperature_low_da(self):
        """Test low DA gives softer routing (higher temperature)."""
        coupling = CapsuleNCACoupling()
        nt_state = MockNTState(dopamine=0.1, acetylcholine=0.5)

        temp = coupling.compute_routing_temperature(nt_state)

        # Lower DA should give higher temperature
        assert temp > 1.0

    def test_routing_temperature_encoding_mode(self):
        """Test encoding mode sharpens routing."""
        coupling = CapsuleNCACoupling()
        nt_neutral = MockNTState(dopamine=0.5, acetylcholine=0.5)
        nt_encoding = MockNTState(dopamine=0.5, acetylcholine=0.8)

        temp_neutral = coupling.compute_routing_temperature(nt_neutral)
        temp_encoding = coupling.compute_routing_temperature(nt_encoding)

        # Encoding should have sharper (lower) temperature
        assert temp_encoding < temp_neutral

    def test_routing_temperature_retrieval_mode(self):
        """Test retrieval mode softens routing for pattern completion."""
        coupling = CapsuleNCACoupling()
        nt_neutral = MockNTState(dopamine=0.5, acetylcholine=0.5)
        nt_retrieval = MockNTState(dopamine=0.5, acetylcholine=0.3)

        temp_neutral = coupling.compute_routing_temperature(nt_neutral)
        temp_retrieval = coupling.compute_routing_temperature(nt_retrieval)

        # Retrieval should have softer (higher) temperature
        assert temp_retrieval > temp_neutral

    def test_routing_temperature_clamped(self):
        """Test temperature is clamped to valid range."""
        coupling = CapsuleNCACoupling()

        # Extreme DA levels
        temp_high = coupling.compute_routing_temperature(
            MockNTState(dopamine=1.0, acetylcholine=0.8)
        )
        temp_low = coupling.compute_routing_temperature(
            MockNTState(dopamine=0.0, acetylcholine=0.3)
        )

        assert temp_high >= coupling.config.min_routing_temperature
        assert temp_low <= coupling.config.max_routing_temperature

    def test_mode_history_tracked(self):
        """Test mode history is tracked correctly."""
        coupling = CapsuleNCACoupling()

        # Transition through modes
        coupling.determine_mode(MockNTState(acetylcholine=0.8))
        coupling.determine_mode(MockNTState(acetylcholine=0.3))
        coupling.determine_mode(MockNTState(acetylcholine=0.5))

        assert len(coupling.state.mode_history) >= 3

    def test_coupling_strength_values(self):
        """Test coupling strength enum values."""
        assert CouplingStrength.WEAK.value == 0.3
        assert CouplingStrength.MODERATE.value == 0.5
        assert CouplingStrength.STRONG.value == 0.7
        assert CouplingStrength.FULL.value == 1.0


# =============================================================================
# GlymphaticConsolidationBridge Tests
# =============================================================================


class TestGlymphaticConsolidationBridge:
    """Tests for Glymphatic ↔ Consolidation bridge."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        bridge = GlymphaticConsolidationBridge()

        assert bridge.config.spindle_batch_fraction == 0.1
        assert bridge.config.delta_batch_fraction == 0.3
        assert bridge.state.sleep_stage == SleepStage.WAKE
        assert bridge.state.clearance_mode == ClearanceMode.NONE

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = GlymphaticConsolidationConfig(
            spindle_batch_fraction=0.2,
            weakness_threshold=0.4,
        )
        bridge = GlymphaticConsolidationBridge(config=config)

        assert bridge.config.spindle_batch_fraction == 0.2
        assert bridge.config.weakness_threshold == 0.4

    def test_set_sleep_stage_wake(self):
        """Test setting WAKE stage disables clearance."""
        bridge = GlymphaticConsolidationBridge()
        bridge.set_sleep_stage(SleepStage.WAKE)

        assert bridge.state.sleep_stage == SleepStage.WAKE
        assert bridge.state.clearance_mode == ClearanceMode.NONE

    def test_set_sleep_stage_nrem_light(self):
        """Test NREM light enables micro clearance."""
        bridge = GlymphaticConsolidationBridge()
        bridge.set_sleep_stage(SleepStage.NREM_LIGHT)

        assert bridge.state.sleep_stage == SleepStage.NREM_LIGHT
        assert bridge.state.clearance_mode == ClearanceMode.MICRO

    def test_set_sleep_stage_nrem_deep(self):
        """Test NREM deep enables bulk clearance."""
        bridge = GlymphaticConsolidationBridge()
        bridge.set_sleep_stage(SleepStage.NREM_DEEP)

        assert bridge.state.sleep_stage == SleepStage.NREM_DEEP
        assert bridge.state.clearance_mode == ClearanceMode.BULK

    def test_set_sleep_stage_rem(self):
        """Test REM disables clearance."""
        bridge = GlymphaticConsolidationBridge()
        bridge.set_sleep_stage(SleepStage.REM)

        assert bridge.state.sleep_stage == SleepStage.REM
        assert bridge.state.clearance_mode == ClearanceMode.NONE

    def test_infer_stage_wake(self):
        """Test stage inference: high ACh + high NE = WAKE."""
        bridge = GlymphaticConsolidationBridge()

        stage = bridge.infer_stage_from_neuromod(ach_level=0.7, ne_level=0.6)

        assert stage == SleepStage.WAKE

    def test_infer_stage_nrem_deep(self):
        """Test stage inference: low ACh + low NE = NREM_DEEP."""
        bridge = GlymphaticConsolidationBridge()

        stage = bridge.infer_stage_from_neuromod(ach_level=0.2, ne_level=0.1)

        assert stage == SleepStage.NREM_DEEP

    def test_infer_stage_nrem_light(self):
        """Test stage inference: low ACh + moderate NE = NREM_LIGHT."""
        bridge = GlymphaticConsolidationBridge()

        stage = bridge.infer_stage_from_neuromod(ach_level=0.35, ne_level=0.35)

        assert stage == SleepStage.NREM_LIGHT

    def test_infer_stage_rem(self):
        """Test stage inference: high ACh + low NE = REM."""
        bridge = GlymphaticConsolidationBridge()

        stage = bridge.infer_stage_from_neuromod(ach_level=0.6, ne_level=0.2)

        assert stage == SleepStage.REM

    def test_protect_memory(self):
        """Test memory protection adds to protected set."""
        bridge = GlymphaticConsolidationBridge()

        bridge.protect_memory("episode_123")

        assert "episode_123" in bridge.state.protected_memories
        assert "episode_123" in bridge.state.protection_timestamps

    def test_protect_memory_updates_timestamp(self):
        """Test re-protecting memory updates timestamp."""
        bridge = GlymphaticConsolidationBridge()

        bridge.protect_memory("episode_123")
        old_time = bridge.state.protection_timestamps["episode_123"]

        # Small delay
        bridge.protect_memory("episode_123")
        new_time = bridge.state.protection_timestamps["episode_123"]

        assert new_time >= old_time

    def test_protected_memory_count_capped(self):
        """Test protected memories are capped at max_protected_memories."""
        config = GlymphaticConsolidationConfig(max_protected_memories=5)
        bridge = GlymphaticConsolidationBridge(config=config)

        # Add more than max
        for i in range(10):
            bridge.protect_memory(f"episode_{i}")

        assert len(bridge.state.protected_memories) <= 5

    def test_sleep_stage_enum_values(self):
        """Test sleep stage enum has expected values."""
        assert SleepStage.WAKE.name == "WAKE"
        assert SleepStage.NREM_LIGHT.name == "NREM_LIGHT"
        assert SleepStage.NREM_DEEP.name == "NREM_DEEP"
        assert SleepStage.REM.name == "REM"

    def test_clearance_mode_enum_values(self):
        """Test clearance mode enum has expected values."""
        assert ClearanceMode.NONE.name == "NONE"
        assert ClearanceMode.MICRO.name == "MICRO"
        assert ClearanceMode.BULK.name == "BULK"

    def test_memory_protection_status_enum(self):
        """Test memory protection status enum values."""
        assert MemoryProtectionStatus.PROTECTED.name == "PROTECTED"
        assert MemoryProtectionStatus.VULNERABLE.name == "VULNERABLE"
        assert MemoryProtectionStatus.TAGGED.name == "TAGGED"
        assert MemoryProtectionStatus.CLEARED.name == "CLEARED"

    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = GlymphaticConsolidationConfig(
            spindle_batch_fraction=0.5,
            weakness_threshold=0.5,
        )
        assert config.spindle_batch_fraction == 0.5

        # Invalid spindle_batch_fraction should raise
        with pytest.raises(AssertionError):
            GlymphaticConsolidationConfig(spindle_batch_fraction=1.5)

        # Invalid weakness_threshold should raise
        with pytest.raises(AssertionError):
            GlymphaticConsolidationConfig(weakness_threshold=-0.1)


# =============================================================================
# Cross-Module Integration Tests
# =============================================================================


class TestPhase4CouplingIntegration:
    """Integration tests for Phase 4 coupling modules working together."""

    def test_ff_capsule_mode_consistency(self):
        """Test FF phase and Capsule mode are consistent for same ACh level."""
        ff_coupling = FFNCACoupling()
        capsule_coupling = CapsuleNCACoupling()

        # High ACh should give encoding modes in both
        high_ach = MockNTState(acetylcholine=0.8)
        ff_phase = ff_coupling.determine_phase(high_ach)
        capsule_mode = capsule_coupling.determine_mode(high_ach)

        assert ff_phase == FFPhase.POSITIVE
        assert capsule_mode == CapsuleMode.ENCODING

        # Low ACh should give refinement modes in both
        low_ach = MockNTState(acetylcholine=0.3)
        ff_phase = ff_coupling.determine_phase(low_ach)
        capsule_mode = capsule_coupling.determine_mode(low_ach)

        assert ff_phase == FFPhase.NEGATIVE
        assert capsule_mode == CapsuleMode.RETRIEVAL

    def test_glymphatic_stage_affects_clearance(self):
        """Test sleep stage transitions affect clearance appropriately."""
        bridge = GlymphaticConsolidationBridge()

        # Wake → no clearance
        bridge.set_sleep_stage(SleepStage.WAKE)
        assert bridge.state.clearance_mode == ClearanceMode.NONE

        # NREM Deep → bulk clearance
        bridge.set_sleep_stage(SleepStage.NREM_DEEP)
        assert bridge.state.clearance_mode == ClearanceMode.BULK

        # REM → no clearance (consolidation active)
        bridge.set_sleep_stage(SleepStage.REM)
        assert bridge.state.clearance_mode == ClearanceMode.NONE

    def test_da_modulates_both_ff_and_capsule(self):
        """Test DA affects both FF learning rate and capsule routing."""
        ff_coupling = FFNCACoupling()
        capsule_coupling = CapsuleNCACoupling()

        high_da = MockNTState(dopamine=0.9, acetylcholine=0.5)
        low_da = MockNTState(dopamine=0.1, acetylcholine=0.5)

        # High DA should increase FF learning and sharpen capsule routing
        ff_lr_high = ff_coupling.compute_learning_rate_multiplier(high_da)
        capsule_temp_high = capsule_coupling.compute_routing_temperature(high_da)

        ff_lr_low = ff_coupling.compute_learning_rate_multiplier(low_da)
        capsule_temp_low = capsule_coupling.compute_routing_temperature(low_da)

        # High DA: higher learning rate, lower routing temperature
        assert ff_lr_high > ff_lr_low
        assert capsule_temp_high < capsule_temp_low

    def test_neuromod_sleep_stage_inference(self):
        """Test neuromodulator levels correctly infer sleep stages."""
        bridge = GlymphaticConsolidationBridge()

        # Simulate circadian cycle
        stages = []

        # Morning wake-up (high ACh, high NE)
        stages.append(bridge.infer_stage_from_neuromod(0.7, 0.6))

        # Evening drowsy (moderate ACh, moderate NE)
        stages.append(bridge.infer_stage_from_neuromod(0.5, 0.5))

        # Deep sleep (low ACh, low NE)
        stages.append(bridge.infer_stage_from_neuromod(0.2, 0.1))

        # REM (high ACh, low NE)
        stages.append(bridge.infer_stage_from_neuromod(0.6, 0.2))

        assert stages[0] == SleepStage.WAKE
        assert stages[2] == SleepStage.NREM_DEEP
        assert stages[3] == SleepStage.REM


# =============================================================================
# Biological Plausibility Tests
# =============================================================================


class TestBiologicalPlausibility:
    """Tests for biological plausibility of Phase 4 coupling."""

    def test_ff_learning_rate_schultz_da_model(self):
        """Test DA modulation follows Schultz (1998) reward prediction."""
        coupling = FFNCACoupling()

        # DA above baseline (positive surprise) → enhanced learning
        positive_surprise = MockNTState(dopamine=0.8)
        lr_positive = coupling.compute_learning_rate_multiplier(positive_surprise)
        assert lr_positive > 1.0

        # DA at baseline (expected) → normal learning
        expected = MockNTState(dopamine=0.5)
        lr_baseline = coupling.compute_learning_rate_multiplier(expected)
        assert 0.9 < lr_baseline < 1.1

        # DA below baseline (negative surprise) → reduced learning
        negative_surprise = MockNTState(dopamine=0.2)
        lr_negative = coupling.compute_learning_rate_multiplier(negative_surprise)
        assert lr_negative < 1.0

    def test_capsule_ach_hasselmo_model(self):
        """Test ACh gating follows Hasselmo (2006) encoding/retrieval."""
        coupling = CapsuleNCACoupling()

        # High ACh should bias toward encoding (new pattern storage)
        encoding_state = MockNTState(acetylcholine=0.8)
        mode = coupling.determine_mode(encoding_state)
        assert mode == CapsuleMode.ENCODING

        # Low ACh should bias toward retrieval (pattern completion)
        retrieval_state = MockNTState(acetylcholine=0.3)
        mode = coupling.determine_mode(retrieval_state)
        assert mode == CapsuleMode.RETRIEVAL

    def test_glymphatic_xie_sleep_clearance(self):
        """Test clearance follows Xie et al. (2013) sleep-dependent increase."""
        bridge = GlymphaticConsolidationBridge()

        # No clearance during wake (Xie: 60% reduction vs sleep)
        bridge.set_sleep_stage(SleepStage.WAKE)
        assert bridge.state.clearance_mode == ClearanceMode.NONE

        # Maximum clearance during NREM deep
        bridge.set_sleep_stage(SleepStage.NREM_DEEP)
        assert bridge.state.clearance_mode == ClearanceMode.BULK

    def test_ff_ne_arousal_threshold(self):
        """Test NE modulation follows Aston-Jones arousal model."""
        coupling = FFNCACoupling()

        # High NE (arousal) → higher threshold → more selective
        high_arousal = MockNTState(norepinephrine=0.9)
        threshold_high = coupling.compute_goodness_threshold(high_arousal)

        # Low NE (relaxed) → lower threshold → less selective
        low_arousal = MockNTState(norepinephrine=0.1)
        threshold_low = coupling.compute_goodness_threshold(low_arousal)

        assert threshold_high > threshold_low


# =============================================================================
# Performance and Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for Phase 4 coupling."""

    def test_ff_extreme_da_clamped(self):
        """Test extreme DA values are clamped."""
        coupling = FFNCACoupling()

        # Very high DA
        extreme_high = MockNTState(dopamine=1.0)
        lr = coupling.compute_learning_rate_multiplier(extreme_high)
        assert lr <= 3.0

        # Very low DA
        extreme_low = MockNTState(dopamine=0.0)
        lr = coupling.compute_learning_rate_multiplier(extreme_low)
        assert lr >= 0.1

    def test_capsule_temperature_bounds(self):
        """Test routing temperature stays in valid range."""
        coupling = CapsuleNCACoupling()

        # Test many random NT states
        for _ in range(100):
            nt = MockNTState(
                dopamine=np.random.random(),
                acetylcholine=np.random.random(),
            )
            temp = coupling.compute_routing_temperature(nt)
            assert coupling.config.min_routing_temperature <= temp
            assert temp <= coupling.config.max_routing_temperature

    def test_glymphatic_empty_protection_set(self):
        """Test bridge handles empty protection set."""
        bridge = GlymphaticConsolidationBridge()

        # Should not raise with empty set
        assert len(bridge.state.protected_memories) == 0

    def test_config_parameter_validation(self):
        """Test config validates parameter ranges."""
        # FF config validation
        with pytest.raises(AssertionError):
            FFNCACouplingConfig(da_learning_rate_gain=1.5)

        with pytest.raises(AssertionError):
            FFNCACouplingConfig(ach_phase_threshold=2.0)

        # Capsule config validation
        with pytest.raises(AssertionError):
            CapsuleNCACouplingConfig(da_routing_gain=-0.1)

        with pytest.raises(AssertionError):
            CapsuleNCACouplingConfig(max_routing_temperature=10.0)

    def test_state_history_bounded(self):
        """Test all state histories are properly bounded."""
        ff = FFNCACoupling()
        capsule = CapsuleNCACoupling()

        # Add many entries
        for i in range(200):
            ff.state.goodness_history.append(float(i))
            ff.state.energy_history.append(float(i))
            capsule.state.mode_history.append(CapsuleMode.NEUTRAL)
            capsule.state.agreement_history.append(0.5)

        # Update histories
        ff.state.update_history(max_len=100)
        capsule.state.update_history(max_len=100)

        assert len(ff.state.goodness_history) == 100
        assert len(capsule.state.mode_history) == 100
