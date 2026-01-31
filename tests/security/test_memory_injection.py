"""Tests for memory injection prevention."""
import numpy as np
import pytest
from t4dm.core.access_control import API_TOKEN, HIPPOCAMPUS_TOKEN, CONSOLIDATION_TOKEN, AccessDenied


class TestHippocampalAccessControl:
    """P0-3: Test access control on hippocampal methods."""

    def test_ca3_process_with_token(self):
        """P0-3: Internal token allows hippocampal operations."""
        from t4dm.nca.hippocampus import HippocampalCircuit, HippocampalConfig
        hpc = HippocampalCircuit(HippocampalConfig(ec_dim=128))
        # Internal token should work
        hpc.receive_ach(0.7, token=HIPPOCAMPUS_TOKEN)
        assert hpc._ach_level == 0.7

    def test_neuromod_rejects_external(self):
        """P0-3: External token cannot set ACh/NE."""
        from t4dm.nca.hippocampus import HippocampalCircuit, HippocampalConfig
        hpc = HippocampalCircuit(HippocampalConfig(ec_dim=128))
        with pytest.raises(AccessDenied):
            hpc.receive_ach(0.7, token=API_TOKEN)

    def test_backward_compat_no_token(self):
        """P0-3: No token = backward compatible (no check)."""
        from t4dm.nca.hippocampus import HippocampalCircuit, HippocampalConfig
        hpc = HippocampalCircuit(HippocampalConfig(ec_dim=128))
        hpc.receive_ach(0.7)  # Should not raise
        assert hpc._ach_level == 0.7

    def test_dg_rejects_wrong_dimension(self):
        """P0-3: Mismatched input dimension raises ValidationError."""
        from t4dm.nca.hippocampus import HippocampalCircuit, HippocampalConfig
        from t4dm.core.validation import ValidationError
        hpc = HippocampalCircuit(HippocampalConfig(ec_dim=128))
        wrong_dim = np.random.default_rng(42).random(64).astype(np.float32)
        with pytest.raises(ValidationError):
            hpc.process(wrong_dim)

    def test_ne_access_control(self):
        """P0-3: Test NE access control."""
        from t4dm.nca.hippocampus import HippocampalCircuit, HippocampalConfig
        hpc = HippocampalCircuit(HippocampalConfig(ec_dim=128))
        
        # Internal token should work
        hpc.receive_ne(0.5, token=HIPPOCAMPUS_TOKEN)
        assert hpc._ne_level == 0.5
        
        # External token should fail
        with pytest.raises(AccessDenied):
            hpc.receive_ne(0.8, token=API_TOKEN)
        
        # No token should work (backward compat)
        hpc.receive_ne(0.3)
        assert hpc._ne_level == 0.3


class TestSWRAccessControl:
    """P0-5: Test access control on SWR methods."""

    def test_trigger_replay_validates_pattern(self):
        """P0-5: NaN pattern rejected."""
        from t4dm.nca.swr_coupling import SWRNeuralFieldCoupling
        swr = SWRNeuralFieldCoupling()
        nan_pattern = np.full(128, float('nan'))
        with pytest.raises(ValueError, match="NaN or Inf"):
            swr.trigger_replay(nan_pattern, token=CONSOLIDATION_TOKEN)

    def test_trigger_replay_access_control(self):
        """P0-5: Test trigger_replay access control."""
        from t4dm.nca.swr_coupling import SWRNeuralFieldCoupling
        swr = SWRNeuralFieldCoupling()
        
        # Force SWR state for testing
        swr._initiate_swr()
        
        valid_pattern = np.random.default_rng(42).random(128).astype(np.float32)
        
        # Internal token should work
        result = swr.trigger_replay(valid_pattern, token=CONSOLIDATION_TOKEN)
        # May be False if hippocampus not connected, but shouldn't raise
        
        # External token should fail
        with pytest.raises(AccessDenied):
            swr.trigger_replay(valid_pattern, token=API_TOKEN)
        
        # No token should work (backward compat)
        result = swr.trigger_replay(valid_pattern)

    def test_force_swr_access_control(self):
        """P0-5: Test force_swr access control."""
        from t4dm.nca.swr_coupling import SWRNeuralFieldCoupling
        swr = SWRNeuralFieldCoupling()
        
        # Internal token should work
        result = swr.force_swr(token=CONSOLIDATION_TOKEN)
        assert result is True
        
        # Reset state
        swr.reset()
        
        # External token should fail
        with pytest.raises(AccessDenied):
            swr.force_swr(token=API_TOKEN)
        
        # No token should work (backward compat)
        result = swr.force_swr()
        assert result is True

    def test_inf_pattern_rejected(self):
        """P0-5: Inf pattern rejected."""
        from t4dm.nca.swr_coupling import SWRNeuralFieldCoupling
        swr = SWRNeuralFieldCoupling()
        swr._initiate_swr()

        inf_pattern = np.full(128, float('inf'))
        with pytest.raises(ValueError, match="NaN or Inf"):
            swr.trigger_replay(inf_pattern, token=CONSOLIDATION_TOKEN)


class TestConsolidationSecurity:
    """ATOM-P0-14: Consolidation trigger authorization and validation."""

    def test_consolidation_type_validated(self):
        """P0-14: Invalid consolidation type rejected."""
        from t4dm.consolidation.service import ConsolidationService
        import asyncio

        service = ConsolidationService()

        # Invalid type should return error result
        result = asyncio.run(service.consolidate(consolidation_type="invalid_type"))
        assert result["status"] == "error"
        assert "Unknown consolidation type" in result["error"]

    def test_consolidation_accepts_valid_types(self):
        """P0-14: Valid consolidation types accepted."""
        from t4dm.consolidation.service import ConsolidationService
        import asyncio

        service = ConsolidationService()
        valid_types = ["light", "deep", "skill", "all"]

        # All valid types should be accepted (may fail for other reasons, but not validation)
        for ctype in valid_types:
            try:
                asyncio.run(service.consolidate(consolidation_type=ctype))
            except ValueError as e:
                # Should not be a validation error about type
                assert "Invalid consolidation_type" not in str(e)
            except Exception:
                # Other exceptions (like missing services) are OK for this test
                pass

    def test_consolidation_requires_token_capability(self):
        """P0-14: Consolidation requires trigger_consolidation capability."""
        from t4dm.consolidation.service import ConsolidationService
        import asyncio

        service = ConsolidationService()

        # API token lacks trigger_consolidation capability
        with pytest.raises(AccessDenied, match="trigger_consolidation"):
            asyncio.run(service.consolidate(consolidation_type="light", token=API_TOKEN))


class TestSWRPhaseSetters:
    """ATOM-P0-12: SWR phase setter access control."""

    def test_swr_phase_setters_accept_internal(self):
        """P0-12: Consolidation token can set sleep state."""
        from t4dm.nca.swr_coupling import SWRNeuralFieldCoupling, WakeSleepMode

        swr = SWRNeuralFieldCoupling()

        # Should not raise with consolidation token
        swr.set_wake_sleep_mode(WakeSleepMode.NREM_DEEP, token=CONSOLIDATION_TOKEN)
        swr.set_ach_level(0.2, token=CONSOLIDATION_TOKEN)
        swr.set_ne_level(0.1, token=CONSOLIDATION_TOKEN)

        assert swr.state.wake_sleep_mode == WakeSleepMode.NREM_DEEP
        assert abs(swr.state.ach_level - 0.2) < 0.01
        assert abs(swr.state.ne_level - 0.1) < 0.01

    def test_swr_phase_setters_reject_external(self):
        """P0-12: API token cannot set neuromodulator levels."""
        from t4dm.nca.swr_coupling import SWRNeuralFieldCoupling, WakeSleepMode

        swr = SWRNeuralFieldCoupling()

        # API token lacks set_neuromod capability
        with pytest.raises(AccessDenied, match="set_neuromod"):
            swr.set_ach_level(0.5, token=API_TOKEN)

        with pytest.raises(AccessDenied, match="set_neuromod"):
            swr.set_ne_level(0.5, token=API_TOKEN)

        # API token lacks set_sleep_state capability
        with pytest.raises(AccessDenied, match="set_sleep_state"):
            swr.set_wake_sleep_mode(WakeSleepMode.REM, token=API_TOKEN)

    def test_swr_phase_setters_optional_token(self):
        """P0-12: Phase setters work without token (backward compat)."""
        from t4dm.nca.swr_coupling import SWRNeuralFieldCoupling, WakeSleepMode

        swr = SWRNeuralFieldCoupling()

        # Should work without token (for internal testing)
        swr.set_wake_sleep_mode(WakeSleepMode.QUIET_WAKE)
        swr.set_ach_level(0.3)
        swr.set_ne_level(0.3)

        assert swr.state.wake_sleep_mode == WakeSleepMode.QUIET_WAKE


class TestNucleusBasalisSecurity:
    """ATOM-P0-13: Nucleus basalis salience protection."""

    def test_salience_clamped_to_valid_range(self):
        """P0-13: Salience values clamped to [0, 1]."""
        from t4dm.nca.nucleus_basalis import NucleusBasalisCircuit

        nbm = NucleusBasalisCircuit()

        # Out-of-range values should be clamped
        nbm.process_salience(1.5)  # > 1.0
        assert nbm.state.salience_signal <= 1.0

        nbm.process_salience(-0.5)  # < 0.0
        assert nbm.state.salience_signal >= 0.0

    def test_salience_rate_limited(self):
        """P0-13: Phasic bursts rate-limited to 10/minute."""
        from unittest.mock import patch
        from t4dm.nca.nucleus_basalis import NucleusBasalisCircuit

        nbm = NucleusBasalisCircuit()

        # Mock time.monotonic to space bursts past refractory period
        fake_time = [0.0]
        def mock_monotonic():
            return fake_time[0]

        with patch("time.monotonic", mock_monotonic):
            for i in range(10):
                fake_time[0] = i * 0.5  # 500ms apart, past refractory + burst duration
                nbm.process_salience(0.8)
                # Step enough to clear phasic burst (duration=0.3s)
                for _ in range(4):
                    nbm.step(dt=0.1)

        # All 10 should have fired
        assert len(nbm._phasic_burst_times) == 10

        # 11th burst within 60s should be rate-limited
        initial_burst_count = len(nbm._phasic_burst_times)
        nbm.process_salience(0.9)
        assert len(nbm._phasic_burst_times) <= initial_burst_count

    def test_salience_requires_token_capability(self):
        """P0-13: Salience submission requires submit_salience capability."""
        from t4dm.nca.nucleus_basalis import NucleusBasalisCircuit

        nbm = NucleusBasalisCircuit()

        # API token lacks submit_salience capability - should be rejected
        with pytest.raises(AccessDenied, match="submit_salience"):
            nbm.process_salience(0.5, token=API_TOKEN)

        # Hippocampus token has capability - should work
        nbm.process_salience(0.5, token=HIPPOCAMPUS_TOKEN)
        assert abs(nbm.state.salience_signal - 0.5) < 0.01

    def test_salience_optional_token(self):
        """P0-13: Salience processing works without token (backward compat)."""
        from t4dm.nca.nucleus_basalis import NucleusBasalisCircuit

        nbm = NucleusBasalisCircuit()

        # Should work without token
        nbm.process_salience(0.6)
        assert abs(nbm.state.salience_signal - 0.6) < 0.01
