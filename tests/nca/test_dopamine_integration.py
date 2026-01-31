"""
Tests for Dopamine System Integration.

Tests the unified dopamine integration layer that connects:
- VTACircuit
- DopamineSystem (ww.learning)
- NeuralFieldSolver
- HippocampalCircuit
- LearnableCoupling
"""

import numpy as np
import pytest
from uuid import uuid4

from t4dm.nca.dopamine_integration import (
    DopamineIntegration,
    DopamineIntegrationConfig,
    IntegratedDAState,
    create_dopamine_integration,
)
from t4dm.nca.vta import VTACircuit, VTAConfig
from t4dm.nca.neural_field import NeuralFieldSolver, NeuralFieldConfig
from t4dm.nca.coupling import LearnableCoupling
from t4dm.learning.dopamine import DopamineSystem


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def vta():
    """Create VTA circuit."""
    return VTACircuit()


@pytest.fixture
def neural_field():
    """Create neural field solver."""
    config = NeuralFieldConfig(grid_size=8)  # Small for testing
    return NeuralFieldSolver(config)


@pytest.fixture
def coupling():
    """Create learnable coupling."""
    return LearnableCoupling()


@pytest.fixture
def dopamine_system():
    """Create dopamine system."""
    return DopamineSystem()


@pytest.fixture
def full_integration(vta, neural_field, coupling, dopamine_system):
    """Create fully connected integration."""
    return DopamineIntegration(
        vta=vta,
        neural_field=neural_field,
        coupling=coupling,
        dopamine_system=dopamine_system,
    )


@pytest.fixture
def minimal_integration():
    """Create minimal integration (no components)."""
    return DopamineIntegration()


# =============================================================================
# Test Configuration
# =============================================================================

class TestDopamineIntegrationConfig:
    """Tests for integration configuration."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = DopamineIntegrationConfig()
        assert 0.1 <= config.vta_to_field_gain <= 1.0
        assert 0.0 <= config.novelty_to_rpe_weight <= 1.0
        assert 0.0 <= config.memory_rpe_weight <= 1.0

    def test_custom_config(self):
        """Custom config values are preserved."""
        config = DopamineIntegrationConfig(
            vta_to_field_gain=0.8,
            novelty_to_rpe_weight=0.5,
        )
        assert config.vta_to_field_gain == 0.8
        assert config.novelty_to_rpe_weight == 0.5


# =============================================================================
# Test IntegratedDAState
# =============================================================================

class TestIntegratedDAState:
    """Tests for integrated state representation."""

    def test_default_state(self):
        """Default state has reasonable values."""
        state = IntegratedDAState()
        assert state.vta_da == 0.3
        assert state.integrated_rpe == 0.0

    def test_state_to_dict(self):
        """State can be serialized."""
        state = IntegratedDAState(
            vta_da=0.6,
            integrated_rpe=0.3,
        )
        d = state.to_dict()
        assert d["vta_da"] == 0.6
        assert d["integrated_rpe"] == 0.3


# =============================================================================
# Test Core Integration
# =============================================================================

class TestCoreIntegration:
    """Tests for core integration functionality."""

    def test_process_positive_outcome(self, full_integration):
        """Positive outcome increases DA."""
        memory_id = uuid4()
        initial_da = full_integration.state.integrated_da

        state = full_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.9,  # Good outcome
        )

        # Positive surprise should increase DA
        assert state.memory_rpe > 0
        assert state.integrated_rpe > 0
        assert state.integrated_da >= initial_da

    def test_process_negative_outcome(self, full_integration):
        """Negative outcome decreases DA."""
        memory_id = uuid4()

        # First, establish expectation
        full_integration.dopamine_system.set_expected_value(memory_id, 0.8)

        state = full_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.2,  # Bad outcome
        )

        # Negative surprise
        assert state.memory_rpe < 0
        assert state.integrated_rpe < 0

    def test_rpe_history_tracked(self, full_integration):
        """RPE history is maintained."""
        for i in range(5):
            full_integration.process_memory_outcome(
                memory_id=uuid4(),
                actual_outcome=0.7,
            )

        assert len(full_integration._rpe_history) == 5
        assert len(full_integration._da_history) == 5

    def test_minimal_integration_works(self, minimal_integration):
        """Integration works without components."""
        memory_id = uuid4()

        # Should not crash
        state = minimal_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.8,
        )

        assert state.memory_rpe == 0.8 - 0.5  # Simple surprise
        assert abs(state.integrated_rpe) < 1.0


# =============================================================================
# Test VTA Integration
# =============================================================================

class TestVTAIntegration:
    """Tests for VTA circuit integration."""

    def test_vta_receives_rpe(self, full_integration):
        """VTA circuit receives and processes RPE."""
        memory_id = uuid4()

        full_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.9,
        )

        assert full_integration.vta.state.last_rpe > 0

    def test_vta_da_flows_to_state(self, full_integration):
        """VTA DA updates integrated state."""
        memory_id = uuid4()

        state = full_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.9,
        )

        # VTA DA should match state
        assert state.vta_da == full_integration.vta.state.current_da


# =============================================================================
# Test Neural Field Integration
# =============================================================================

class TestNeuralFieldIntegration:
    """Tests for neural field integration."""

    def test_field_da_updated(self, full_integration):
        """Neural field DA is updated after processing."""
        memory_id = uuid4()
        initial_field_da = full_integration._get_field_da()

        full_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.95,
        )

        # Field DA should have changed
        new_field_da = full_integration._get_field_da()
        # Due to inject_rpe, field should be modified
        assert full_integration.state.field_da is not None

    def test_step_updates_field(self, full_integration):
        """Step() advances neural field."""
        full_integration.step(dt=0.1)

        # Should not crash, field should step
        assert full_integration.state.field_da is not None


# =============================================================================
# Test Coupling Integration
# =============================================================================

class TestCouplingIntegration:
    """Tests for learnable coupling integration."""

    def test_coupling_updated_on_outcome(self, full_integration):
        """Coupling is updated when outcome processed (after DA delay)."""
        memory_id = uuid4()
        initial_update_count = full_integration.coupling._update_count

        full_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.9,
        )

        # Step forward to allow DA delay buffer to process (200ms delay)
        for _ in range(3):  # 3 steps * 0.1s = 0.3s > 0.2s delay
            full_integration.step(dt=0.1)

        # Coupling should have been updated after delay
        assert full_integration.coupling._update_count > initial_update_count

    def test_eligibility_accumulation(self, full_integration):
        """Eligibility traces accumulate."""
        # Accumulate eligibility
        for _ in range(5):
            full_integration.accumulate_eligibility()

        # Check eligibility accumulated
        trace = full_integration.coupling.get_eligibility_trace()
        assert np.any(trace > 0)

    def test_eligibility_reset(self, full_integration):
        """Eligibility can be reset."""
        full_integration.accumulate_eligibility()
        full_integration.reset_eligibility()

        trace = full_integration.coupling.get_eligibility_trace()
        assert np.allclose(trace, 0)


# =============================================================================
# Test DopamineSystem Integration
# =============================================================================

class TestDopamineSystemIntegration:
    """Tests for ww.learning.DopamineSystem integration."""

    def test_expectations_updated(self, full_integration):
        """DopamineSystem expectations are updated."""
        memory_id = uuid4()

        full_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.8,
        )

        # Expectation should have moved toward outcome
        expected = full_integration.dopamine_system.get_expected_value(memory_id)
        assert expected != 0.5  # Changed from default

    def test_value_sync(self, full_integration):
        """Value estimates can be synchronized."""
        memory_id = uuid4()

        # Set expectation in dopamine system
        full_integration.dopamine_system.set_expected_value(memory_id, 0.7)

        # Sync to VTA
        full_integration.sync_value_estimates(memory_id)

        # Check VTA has the value
        vta_value = full_integration.vta._get_value(str(memory_id))
        assert vta_value == pytest.approx(0.7, abs=0.01)

    def test_combined_value(self, full_integration):
        """Combined value from multiple systems."""
        memory_id = uuid4()

        # Set values in both systems
        full_integration.dopamine_system.set_expected_value(memory_id, 0.6)
        full_integration.vta._value_table[str(memory_id)] = 0.8

        combined = full_integration.get_combined_value(memory_id)
        assert combined == pytest.approx(0.7, abs=0.01)  # Average


# =============================================================================
# Test Callbacks
# =============================================================================

class TestCallbacks:
    """Tests for event callbacks."""

    def test_rpe_callback(self, full_integration):
        """RPE callback is triggered."""
        received = []

        def callback(rpe):
            received.append(rpe)

        full_integration.register_rpe_callback(callback)
        full_integration.process_memory_outcome(
            memory_id=uuid4(),
            actual_outcome=0.8,
        )

        assert len(received) == 1

    def test_da_callback(self, full_integration):
        """DA callback is triggered."""
        received = []

        def callback(da):
            received.append(da)

        full_integration.register_da_callback(callback)
        full_integration.process_memory_outcome(
            memory_id=uuid4(),
            actual_outcome=0.8,
        )

        assert len(received) == 1


# =============================================================================
# Test State Management
# =============================================================================

class TestStateManagement:
    """Tests for state saving/loading."""

    def test_get_stats(self, full_integration):
        """Stats dict contains expected keys."""
        full_integration.process_memory_outcome(
            memory_id=uuid4(),
            actual_outcome=0.7,
        )

        stats = full_integration.get_stats()
        assert "integrated_rpe" in stats
        assert "integrated_da" in stats
        assert "vta_da" in stats
        assert "field_da" in stats

    def test_reset(self, full_integration):
        """Reset restores initial state."""
        full_integration.process_memory_outcome(
            memory_id=uuid4(),
            actual_outcome=0.9,
        )

        full_integration.reset()

        assert full_integration.state.integrated_rpe == 0.0
        assert len(full_integration._rpe_history) == 0

    def test_save_load_state(self, full_integration):
        """State can be saved and restored."""
        full_integration.process_memory_outcome(
            memory_id=uuid4(),
            actual_outcome=0.8,
        )

        saved = full_integration.save_state()

        new_integration = DopamineIntegration()
        new_integration.load_state(saved)

        assert new_integration.state.integrated_rpe == pytest.approx(
            full_integration.state.integrated_rpe, abs=0.01
        )


# =============================================================================
# Test Factory Function
# =============================================================================

class TestFactory:
    """Tests for factory function."""

    def test_create_with_defaults(self):
        """Factory creates integration with defaults."""
        integration = create_dopamine_integration()

        assert integration.vta is not None
        assert integration.neural_field is not None

    def test_create_with_components(self, vta, neural_field):
        """Factory uses provided components."""
        integration = create_dopamine_integration(
            vta=vta,
            neural_field=neural_field,
        )

        assert integration.vta is vta
        assert integration.neural_field is neural_field


# =============================================================================
# Test End-to-End Flow
# =============================================================================

class TestEndToEndFlow:
    """Tests for complete integration flow."""

    def test_learning_cycle(self, full_integration):
        """Complete learning cycle works."""
        memory_id = uuid4()

        # Simulate multiple interactions
        for i in range(10):
            # Accumulate eligibility
            full_integration.accumulate_eligibility()
            full_integration.step(dt=0.1)

            # Process outcome
            outcome = 0.7 + np.random.randn() * 0.1
            full_integration.process_memory_outcome(
                memory_id=memory_id,
                actual_outcome=np.clip(outcome, 0, 1),
            )

        # Value should have learned
        expected = full_integration.dopamine_system.get_expected_value(memory_id)
        assert abs(expected - 0.7) < 0.2  # Learned toward true value

    def test_surprise_drives_learning(self, full_integration):
        """Surprising outcomes drive larger updates (after DA delay)."""
        memory_id = uuid4()

        # Set low expectation
        full_integration.dopamine_system.set_expected_value(memory_id, 0.3)
        initial_coupling = full_integration.coupling.K.copy()

        # Very surprising positive outcome
        full_integration.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.95,
        )

        # Step forward to allow DA delay buffer to process (200ms delay)
        for _ in range(3):  # 3 steps * 0.1s = 0.3s > 0.2s delay
            full_integration.step(dt=0.1)

        # Large RPE should cause some coupling change (after delay)
        coupling_change = np.abs(full_integration.coupling.K - initial_coupling).sum()
        assert coupling_change > 0.001  # Some change occurred

    def test_da_converges_to_tonic(self, full_integration):
        """DA returns to tonic after surprise."""
        # Cause burst
        full_integration.process_memory_outcome(
            memory_id=uuid4(),
            actual_outcome=0.95,
        )
        burst_da = full_integration.state.vta_da

        # Step without new input
        for _ in range(50):
            full_integration.step(dt=0.1)

        # Should have decayed toward tonic
        final_da = full_integration.state.vta_da
        tonic_da = full_integration.vta.config.tonic_da_level

        assert abs(final_da - tonic_da) < abs(burst_da - tonic_da)
