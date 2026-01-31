"""Tests for neuromodulator manipulation prevention."""
import time
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from t4dm.core.validation import ValidationError
from t4dm.nca.vta import VTAConfig, VTAState, VTACircuit


class TestVTAConfigImmutable:
    """Test that VTAConfig is frozen and cannot be mutated."""

    def test_config_frozen(self):
        """VTAConfig should be frozen and reject attribute assignment."""
        config = VTAConfig()
        with pytest.raises(AttributeError):
            config.tonic_da_level = 999.0

    def test_pfc_modulation_preserves_config(self):
        """Call receive_pfc_modulation 1000x, config must not change."""
        circuit = VTACircuit()
        original = circuit.config.tonic_da_level
        for _ in range(1000):
            circuit.receive_pfc_modulation(1.0, context="goal")
        assert circuit.config.tonic_da_level == original

    def test_reset_clears_modulation(self):
        """Reset should clear tonic_modulation in state."""
        circuit = VTACircuit()
        circuit.receive_pfc_modulation(1.0, context="goal")
        assert circuit.state.tonic_modulation > 0
        circuit.reset()
        assert circuit.state.tonic_modulation == 0.0

    def test_tonic_modulation_capped(self):
        """Tonic modulation should be capped at 0.2."""
        circuit = VTACircuit()
        for _ in range(10000):
            circuit.receive_pfc_modulation(1.0, context="goal")
        assert circuit.state.tonic_modulation <= 0.2


class TestRewardValidation:
    """ATOM-P0-4: Validate reward signals in dopamine integration."""

    def test_reward_clamped_positive(self):
        """P0-4: Outcome=100.0 is clamped to 1.0."""
        from t4dm.nca.dopamine_integration import DopamineIntegration

        di = DopamineIntegration()
        memory_id = uuid4()

        # Process extreme positive value
        state = di.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=100.0
        )

        # Verify RPE is based on clamped value (1.0), not raw 100.0
        # Without clamping, memory_rpe would be catastrophically large
        assert abs(state.memory_rpe) <= 1.0, "RPE should be bounded by clamped outcome"

    def test_reward_clamped_negative(self):
        """P0-4: Outcome=-100.0 is clamped to -1.0."""
        from t4dm.nca.dopamine_integration import DopamineIntegration

        di = DopamineIntegration()
        memory_id = uuid4()

        # Process extreme negative value
        state = di.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=-100.0
        )

        # Verify RPE is based on clamped value (-1.0)
        assert abs(state.memory_rpe) <= 1.5, "RPE should be bounded by clamped outcome"

    def test_reward_rate_limited(self):
        """P0-4: 11th call within 60s raises ValueError."""
        from t4dm.nca.dopamine_integration import DopamineIntegration

        di = DopamineIntegration()
        memory_id = uuid4()

        # Submit 10 outcomes (should succeed)
        for i in range(10):
            di.process_memory_outcome(
                memory_id=memory_id,
                actual_outcome=0.5
            )

        # 11th outcome within same minute should fail
        with pytest.raises(ValueError, match="Rate limit exceeded"):
            di.process_memory_outcome(
                memory_id=memory_id,
                actual_outcome=0.5
            )

    def test_reward_rate_limit_per_memory(self):
        """P0-4: Rate limit is per memory_id, not global."""
        from t4dm.nca.dopamine_integration import DopamineIntegration

        di = DopamineIntegration()
        memory_id_a = uuid4()
        memory_id_b = uuid4()

        # Submit 10 outcomes for memory A
        for i in range(10):
            di.process_memory_outcome(
                memory_id=memory_id_a,
                actual_outcome=0.5
            )

        # Should still be able to submit for memory B
        state = di.process_memory_outcome(
            memory_id=memory_id_b,
            actual_outcome=0.6
        )
        assert state is not None

    def test_reward_rate_limit_window_expiry(self):
        """P0-4: Rate limit window expires after 60 seconds."""
        from t4dm.nca.dopamine_integration import DopamineIntegration

        di = DopamineIntegration()
        memory_id = uuid4()

        # Submit 10 outcomes
        for i in range(10):
            di.process_memory_outcome(
                memory_id=memory_id,
                actual_outcome=0.5
            )

        # Manually expire timestamps by manipulating the deque
        mid = str(memory_id)
        if mid in di._outcome_timestamps:
            # Set all timestamps to 61 seconds ago
            old_time = time.monotonic() - 61.0
            di._outcome_timestamps[mid].clear()
            di._outcome_timestamps[mid].append(old_time)

        # Should now be able to submit again
        state = di.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.7
        )
        assert state is not None

    def test_reward_capability_check(self):
        """P0-4: Token without 'submit_reward' capability is rejected."""
        from t4dm.core.access_control import CallerToken, AccessDenied
        from t4dm.nca.dopamine_integration import DopamineIntegration

        di = DopamineIntegration()
        memory_id = uuid4()

        # Token without submit_reward capability
        invalid_token = CallerToken(
            module="test",
            trust_level="external",
            capabilities=frozenset({"read"})
        )

        with pytest.raises(AccessDenied, match="submit_reward"):
            di.process_memory_outcome(
                memory_id=memory_id,
                actual_outcome=0.5,
                token=invalid_token
            )

    def test_reward_valid_capability(self):
        """P0-4: Token with 'submit_reward' capability succeeds."""
        from t4dm.core.access_control import VTA_TOKEN
        from t4dm.nca.dopamine_integration import DopamineIntegration

        di = DopamineIntegration()
        memory_id = uuid4()

        # VTA_TOKEN has submit_reward capability
        state = di.process_memory_outcome(
            memory_id=memory_id,
            actual_outcome=0.5,
            token=VTA_TOKEN
        )
        assert state is not None


class TestSTDPValidation:
    """ATOM-P0-7: Validate STDP spike timestamps."""

    def test_spike_timestamp_not_future(self):
        """P0-7: Reject timestamps in the future."""
        from t4dm.learning.stdp import STDPLearner

        stdp = STDPLearner()
        future_time = datetime.now() + timedelta(seconds=10)

        with pytest.raises(ValidationError, match="in the future"):
            stdp.record_spike("neuron_a", timestamp=future_time)

    def test_spike_timestamp_not_too_old(self):
        """P0-7: Reject timestamps more than 1 year in the past."""
        from t4dm.learning.stdp import STDPLearner

        stdp = STDPLearner()
        old_time = datetime.now() - timedelta(days=400)

        with pytest.raises(ValidationError, match="in the past"):
            stdp.record_spike("neuron_a", timestamp=old_time)

    def test_spike_interval_validation(self):
        """P0-7: Reject inter-spike intervals < 1ms (physiologically implausible)."""
        from t4dm.learning.stdp import STDPLearner

        stdp = STDPLearner()

        # First spike
        t1 = datetime.now()
        stdp.record_spike("neuron_a", timestamp=t1)

        # Second spike 0.5ms later (too soon)
        t2 = t1 + timedelta(microseconds=500)
        with pytest.raises(ValidationError, match="Inter-spike interval"):
            stdp.record_spike("neuron_a", timestamp=t2)

    def test_spike_interval_valid(self):
        """P0-7: Accept inter-spike intervals >= 1ms."""
        from t4dm.learning.stdp import STDPLearner

        stdp = STDPLearner()

        # First spike
        t1 = datetime.now()
        stdp.record_spike("neuron_a", timestamp=t1)

        # Second spike 2ms later (valid)
        t2 = t1 + timedelta(microseconds=2000)
        stdp.record_spike("neuron_a", timestamp=t2)

        # Verify both spikes recorded
        history = stdp._spike_history.get("neuron_a", [])
        assert len(history) == 2

    def test_spike_interval_per_entity(self):
        """P0-7: Inter-spike validation is per entity, not global."""
        from t4dm.learning.stdp import STDPLearner

        stdp = STDPLearner()

        # Spike for neuron A
        t1 = datetime.now()
        stdp.record_spike("neuron_a", timestamp=t1)

        # Simultaneous spike for neuron B should succeed (different entity)
        stdp.record_spike("neuron_b", timestamp=t1)

        assert len(stdp._spike_history["neuron_a"]) == 1
        assert len(stdp._spike_history["neuron_b"]) == 1

    def test_spike_default_timestamp_valid(self):
        """P0-7: Default timestamp (now) passes validation."""
        from t4dm.learning.stdp import STDPLearner

        stdp = STDPLearner()

        # Should not raise
        stdp.record_spike("neuron_a")
        stdp.record_spike("neuron_b")

        assert len(stdp._spike_history) == 2
