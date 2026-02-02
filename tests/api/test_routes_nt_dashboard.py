"""
Tests for neurotransmitter dashboard routes.

Tests cover:
- NT state management (injection, decay, reset)
- Receptor saturation via Michaelis-Menten kinetics
- Cognitive mode classification from NT levels
- Dynamic updates over time
- Endpoint request/response models
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from time import sleep

import numpy as np

from t4dm.api.routes.nt_dashboard import (
    NeurotransmitterLevels,
    ReceptorSaturation,
    CognitiveMode,
    NTDashboardState,
    NTTraceEntry,
    InjectNTRequest,
    InjectNTResponse,
    NTStateManager,
    nt_router,
)


# =============================================================================
# Test Models
# =============================================================================


class TestNeurotransmitterLevels:
    """Tests for NeurotransmitterLevels model."""

    def test_default_levels(self):
        """Default levels are initialized correctly."""
        levels = NeurotransmitterLevels()
        assert levels.dopamine == 0.5
        assert levels.serotonin == 0.5
        assert levels.acetylcholine == 0.5
        assert levels.norepinephrine == 0.3
        assert levels.gaba == 0.4
        assert levels.glutamate == 0.5

    def test_custom_levels(self):
        """Custom levels can be set."""
        levels = NeurotransmitterLevels(
            dopamine=0.8,
            serotonin=0.6,
            acetylcholine=0.7,
            norepinephrine=0.4,
            gaba=0.5,
            glutamate=0.6,
        )
        assert levels.dopamine == 0.8
        assert levels.serotonin == 0.6

    def test_bounds_enforced(self):
        """NT levels must be in [0, 1]."""
        # Valid
        NeurotransmitterLevels(dopamine=0.0)
        NeurotransmitterLevels(dopamine=1.0)

        # Invalid (will fail validation)
        with pytest.raises(ValueError):
            NeurotransmitterLevels(dopamine=-0.1)
        with pytest.raises(ValueError):
            NeurotransmitterLevels(dopamine=1.1)


class TestReceptorSaturation:
    """Tests for ReceptorSaturation model."""

    def test_default_receptors(self):
        """Default receptor saturation is initialized."""
        receptors = ReceptorSaturation()
        assert receptors.d1_saturation == 0.5
        assert receptors.d2_saturation == 0.5
        assert receptors.alpha1_saturation == 0.3
        assert receptors.beta_saturation == 0.3
        assert receptors.m1_saturation == 0.5
        assert receptors.nmda_saturation == 0.5
        assert receptors.gaba_a_saturation == 0.4

    def test_custom_saturation(self):
        """Custom saturation values can be set."""
        receptors = ReceptorSaturation(d1_saturation=0.8, m1_saturation=0.6)
        assert receptors.d1_saturation == 0.8
        assert receptors.m1_saturation == 0.6


class TestCognitiveMode:
    """Tests for CognitiveMode model."""

    def test_default_mode(self):
        """Default cognitive mode is 'balanced'."""
        mode = CognitiveMode()
        assert mode.mode == "balanced"
        assert mode.confidence == 0.5
        assert mode.exploration_drive == 0.5
        assert mode.exploitation_drive == 0.5
        assert mode.learning_rate_mod == 1.0
        assert mode.attention_gain == 1.0

    def test_mode_types(self):
        """Mode can be set to valid types."""
        for mode_type in ["explore", "exploit", "encode", "retrieve", "rest", "balanced"]:
            mode = CognitiveMode(mode=mode_type)
            assert mode.mode == mode_type


class TestNTDashboardState:
    """Tests for NTDashboardState model."""

    def test_default_state(self):
        """Default state is initialized correctly."""
        state = NTDashboardState()
        assert state.levels is not None
        assert state.receptors is not None
        assert state.cognitive_mode is not None
        assert state.da_rpe == 0.0
        assert state.ach_uncertainty == 0.0
        assert state.ne_surprise == 0.0
        assert state.da_history == []
        assert state.ach_history == []
        assert isinstance(state.timestamp, datetime)

    def test_state_with_custom_levels(self):
        """State can be created with custom levels."""
        levels = NeurotransmitterLevels(dopamine=0.8)
        state = NTDashboardState(levels=levels)
        assert state.levels.dopamine == 0.8


class TestInjectNTRequest:
    """Tests for InjectNTRequest model."""

    def test_valid_da_injection(self):
        """Valid DA injection request."""
        req = InjectNTRequest(nt_type="da", amount=0.2)
        assert req.nt_type == "da"
        assert req.amount == 0.2
        assert req.event_type is None

    def test_valid_all_nt_types(self):
        """All NT types are valid."""
        for nt_type in ["da", "5ht", "ach", "ne", "gaba", "glu"]:
            req = InjectNTRequest(nt_type=nt_type, amount=0.1)
            assert req.nt_type == nt_type

    def test_injection_amount_bounds(self):
        """Injection amount must be in [-1, 1]."""
        # Valid
        InjectNTRequest(nt_type="da", amount=-1.0)
        InjectNTRequest(nt_type="da", amount=0.0)
        InjectNTRequest(nt_type="da", amount=1.0)

        # Invalid
        with pytest.raises(ValueError):
            InjectNTRequest(nt_type="da", amount=-1.1)
        with pytest.raises(ValueError):
            InjectNTRequest(nt_type="da", amount=1.1)

    def test_with_event_type(self):
        """Event type can be specified."""
        req = InjectNTRequest(nt_type="da", amount=0.2, event_type="reward")
        assert req.event_type == "reward"


class TestInjectNTResponse:
    """Tests for InjectNTResponse model."""

    def test_response_structure(self):
        """Response has required fields."""
        resp = InjectNTResponse(
            success=True,
            new_level=0.7,
            cognitive_mode="exploit",
            receptor_effects={"d1": 0.7, "d2": 0.6},
        )
        assert resp.success is True
        assert resp.new_level == 0.7
        assert resp.cognitive_mode == "exploit"
        assert resp.receptor_effects["d1"] == 0.7


# =============================================================================
# Test NTStateManager - Core Functionality
# =============================================================================


class TestNTStateManagerInit:
    """Tests for NTStateManager initialization."""

    def test_init_creates_state(self):
        """Manager initializes with default state."""
        manager = NTStateManager()
        state = manager.get_state()
        assert isinstance(state, NTDashboardState)

    def test_init_empty_traces(self):
        """Manager initializes with empty traces."""
        manager = NTStateManager()
        traces = manager.get_traces()
        assert traces == []


class TestNTStateManagerInject:
    """Tests for NT injection functionality."""

    def test_inject_dopamine(self):
        """Inject dopamine increases level."""
        manager = NTStateManager()
        initial = manager._state.levels.dopamine
        result = manager.inject_nt("da", 0.2)

        assert result.success is True
        assert result.new_level == initial + 0.2
        assert manager._state.levels.dopamine == initial + 0.2

    def test_inject_all_nt_types(self):
        """All NT types can be injected."""
        manager = NTStateManager()
        nt_types = ["da", "5ht", "ach", "ne", "gaba", "glu"]
        field_map = {
            "da": "dopamine",
            "5ht": "serotonin",
            "ach": "acetylcholine",
            "ne": "norepinephrine",
            "gaba": "gaba",
            "glu": "glutamate",
        }
        baseline_map = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "acetylcholine": 0.5,
            "norepinephrine": 0.3,
            "gaba": 0.4,
            "glutamate": 0.5,
        }

        for nt_type in nt_types:
            manager = NTStateManager()
            result = manager.inject_nt(nt_type, 0.1)
            assert result.success is True
            field = field_map[nt_type]
            baseline = baseline_map[field]
            expected = baseline + 0.1
            assert getattr(manager._state.levels, field) == expected

    def test_inject_invalid_nt_type(self):
        """Invalid NT type raises exception."""
        manager = NTStateManager()
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc:
            manager.inject_nt("invalid", 0.1)
        assert exc.value.status_code == 400
        assert "Unknown NT type" in exc.value.detail

    def test_inject_clamps_upper_bound(self):
        """Injection clamps to [0, 1]."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.9
        result = manager.inject_nt("da", 0.5)

        assert result.new_level == 1.0
        assert manager._state.levels.dopamine == 1.0

    def test_inject_clamps_lower_bound(self):
        """Negative injection clamps to [0, 1]."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.1
        result = manager.inject_nt("da", -0.3)

        assert result.new_level == 0.0
        assert manager._state.levels.dopamine == 0.0

    def test_inject_records_trace(self):
        """Injection records trace entry."""
        manager = NTStateManager()
        manager.inject_nt("da", 0.2)
        manager.inject_nt("5ht", 0.1)

        traces = manager.get_traces()
        assert len(traces) == 2
        assert traces[0].nt_type == "da"
        assert traces[1].nt_type == "5ht"

    def test_inject_with_event_type(self):
        """Injection can record event type."""
        manager = NTStateManager()
        manager.inject_nt("da", 0.2, event_type="reward")

        traces = manager.get_traces()
        assert traces[0].event == "reward"

    def test_inject_trace_pruning(self):
        """Traces are pruned to 500 entries."""
        manager = NTStateManager()
        for i in range(600):
            manager.inject_nt("da", 0.01)

        assert len(manager._traces) <= 500

    def test_inject_updates_receptors(self):
        """Injection updates receptor saturation."""
        manager = NTStateManager()
        initial_d1 = manager._state.receptors.d1_saturation
        manager.inject_nt("da", 0.3)

        # D1 should increase with dopamine
        assert manager._state.receptors.d1_saturation > initial_d1

    def test_inject_updates_cognitive_mode(self):
        """Injection updates cognitive mode."""
        manager = NTStateManager()
        manager.inject_nt("ne", 0.5)
        # High NE should bias toward explore

        mode = manager._state.cognitive_mode
        assert mode.exploration_drive > mode.exploitation_drive

    def test_inject_returns_correct_response(self):
        """Injection returns correct response structure."""
        manager = NTStateManager()
        result = manager.inject_nt("da", 0.2, event_type="test")

        assert isinstance(result, InjectNTResponse)
        assert result.success is True
        assert 0 <= result.new_level <= 1
        assert result.cognitive_mode in ["explore", "exploit", "encode", "retrieve", "rest", "balanced"]
        assert isinstance(result.receptor_effects, dict)
        assert "d1" in result.receptor_effects


class TestNTStateManagerDynamics:
    """Tests for NT dynamics (decay over time)."""

    def test_dynamics_decay_dopamine(self):
        """Dopamine decays toward baseline (0.5)."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.9
        manager._last_update = datetime.now() - timedelta(seconds=1)

        manager._update_dynamics()

        # Should decay toward 0.5
        assert manager._state.levels.dopamine < 0.9
        assert manager._state.levels.dopamine > 0.5

    def test_dynamics_decay_norepinephrine(self):
        """Norepinephrine decays toward baseline (0.3)."""
        manager = NTStateManager()
        manager._state.levels.norepinephrine = 0.8
        manager._last_update = datetime.now() - timedelta(seconds=1)

        manager._update_dynamics()

        # Should decay toward 0.3
        assert manager._state.levels.norepinephrine < 0.8
        assert manager._state.levels.norepinephrine > 0.3

    def test_dynamics_decay_gaba(self):
        """GABA decays toward baseline (0.4)."""
        manager = NTStateManager()
        manager._state.levels.gaba = 0.1
        manager._last_update = datetime.now() - timedelta(seconds=1)

        manager._update_dynamics()

        # Should decay toward 0.4
        assert manager._state.levels.gaba > 0.1
        assert manager._state.levels.gaba < 0.4

    def test_dynamics_adds_noise(self):
        """Dynamics add small noise for realism."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.5
        manager._last_update = datetime.now() - timedelta(seconds=1)

        # Run multiple times to check variance
        values = []
        for _ in range(15):
            manager = NTStateManager()
            manager._state.levels.dopamine = 0.5
            manager._last_update = datetime.now() - timedelta(seconds=1)
            manager._update_dynamics()
            values.append(manager._state.levels.dopamine)

        # Values should vary slightly due to noise
        assert len(set(values)) > 1

    def test_dynamics_skips_if_dt_small(self):
        """Dynamics skips if time delta < 0.1s."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.9
        manager._last_update = datetime.now()  # Just now

        manager._update_dynamics()

        # Should not decay because dt < 0.1
        assert manager._state.levels.dopamine == 0.9

    def test_dynamics_updates_history(self):
        """Dynamics records DA and ACh history."""
        manager = NTStateManager()
        manager._last_update = datetime.now() - timedelta(seconds=1)

        manager._update_dynamics()

        assert len(manager._state.da_history) > 0
        assert len(manager._state.ach_history) > 0

    def test_dynamics_history_pruned_to_50(self):
        """History is pruned to 50 entries."""
        manager = NTStateManager()
        for _ in range(100):
            manager._last_update = datetime.now() - timedelta(seconds=0.2)
            manager._update_dynamics()

        assert len(manager._state.da_history) <= 50
        assert len(manager._state.ach_history) <= 50

    def test_dynamics_updates_timestamp(self):
        """Dynamics updates state timestamp."""
        manager = NTStateManager()
        initial_time = manager._state.timestamp
        manager._last_update = datetime.now() - timedelta(seconds=0.2)

        manager._update_dynamics()

        assert manager._state.timestamp > initial_time


class TestNTStateManagerMichaelisMenten:
    """Tests for Michaelis-Menten receptor kinetics."""

    def test_michaelis_menten_at_zero(self):
        """MM at zero concentration."""
        manager = NTStateManager()
        saturation = manager._michaelis_menten(0.0, 0.5)
        assert saturation == 0.0

    def test_michaelis_menten_at_km(self):
        """MM at Km gives 0.5 saturation."""
        manager = NTStateManager()
        saturation = manager._michaelis_menten(0.5, 0.5)
        assert saturation == pytest.approx(0.5)

    def test_michaelis_menten_high_concentration(self):
        """MM at high concentration approaches 1."""
        manager = NTStateManager()
        saturation = manager._michaelis_menten(10.0, 0.5)
        assert saturation > 0.95

    def test_michaelis_menten_monotonic(self):
        """MM is monotonically increasing."""
        manager = NTStateManager()
        values = [
            manager._michaelis_menten(c, 0.5)
            for c in [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        ]
        assert values == sorted(values)

    def test_michaelis_menten_bounds(self):
        """MM always returns [0, 1)."""
        manager = NTStateManager()
        for conc in [0.0, 0.1, 0.5, 1.0, 10.0]:
            sat = manager._michaelis_menten(conc, 0.4)
            assert 0 <= sat < 1


class TestNTStateManagerReceptors:
    """Tests for receptor saturation updates."""

    def test_update_receptors_d1_increases_with_da(self):
        """D1 saturation increases with dopamine."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.2
        manager._update_receptors()
        d1_low = manager._state.receptors.d1_saturation

        manager._state.levels.dopamine = 0.8
        manager._update_receptors()
        d1_high = manager._state.receptors.d1_saturation

        assert d1_high > d1_low

    def test_update_receptors_d2_increases_with_da(self):
        """D2 saturation increases with dopamine."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.2
        manager._update_receptors()
        d2_low = manager._state.receptors.d2_saturation

        manager._state.levels.dopamine = 0.8
        manager._update_receptors()
        d2_high = manager._state.receptors.d2_saturation

        assert d2_high > d2_low

    def test_update_receptors_alpha1_increases_with_ne(self):
        """Alpha-1 saturation increases with norepinephrine."""
        manager = NTStateManager()
        manager._state.levels.norepinephrine = 0.1
        manager._update_receptors()
        alpha1_low = manager._state.receptors.alpha1_saturation

        manager._state.levels.norepinephrine = 0.8
        manager._update_receptors()
        alpha1_high = manager._state.receptors.alpha1_saturation

        assert alpha1_high > alpha1_low

    def test_update_receptors_m1_increases_with_ach(self):
        """M1 saturation increases with acetylcholine."""
        manager = NTStateManager()
        manager._state.levels.acetylcholine = 0.2
        manager._update_receptors()
        m1_low = manager._state.receptors.m1_saturation

        manager._state.levels.acetylcholine = 0.8
        manager._update_receptors()
        m1_high = manager._state.receptors.m1_saturation

        assert m1_high > m1_low

    def test_update_receptors_nmda_increases_with_glu(self):
        """NMDA saturation increases with glutamate."""
        manager = NTStateManager()
        manager._state.levels.glutamate = 0.2
        manager._update_receptors()
        nmda_low = manager._state.receptors.nmda_saturation

        manager._state.levels.glutamate = 0.8
        manager._update_receptors()
        nmda_high = manager._state.receptors.nmda_saturation

        assert nmda_high > nmda_low

    def test_update_receptors_gaba_a_increases_with_gaba(self):
        """GABA-A saturation increases with GABA."""
        manager = NTStateManager()
        manager._state.levels.gaba = 0.2
        manager._update_receptors()
        gaba_a_low = manager._state.receptors.gaba_a_saturation

        manager._state.levels.gaba = 0.8
        manager._update_receptors()
        gaba_a_high = manager._state.receptors.gaba_a_saturation

        assert gaba_a_high > gaba_a_low


class TestNTStateManagerCognitiveMode:
    """Tests for cognitive mode classification."""

    def test_mode_explore_high_ne(self):
        """High NE drives explore mode."""
        manager = NTStateManager()
        manager._state.levels.norepinephrine = 0.8
        manager._state.levels.dopamine = 0.3
        manager._state.levels.acetylcholine = 0.3
        manager._update_cognitive_mode()

        assert manager._state.cognitive_mode.mode == "explore"

    def test_mode_exploit_high_da_high_serotonin(self):
        """High DA + high 5-HT drive exploit mode."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.75
        manager._state.levels.serotonin = 0.6
        manager._state.levels.norepinephrine = 0.2
        manager._state.levels.acetylcholine = 0.4
        manager._state.levels.gaba = 0.3
        manager._update_cognitive_mode()

        assert manager._state.cognitive_mode.mode == "exploit"

    def test_mode_encode_high_ach(self):
        """High ACh drives encode mode."""
        manager = NTStateManager()
        manager._state.levels.acetylcholine = 0.75
        manager._state.levels.dopamine = 0.3
        manager._state.levels.norepinephrine = 0.2
        manager._update_cognitive_mode()

        assert manager._state.cognitive_mode.mode == "encode"

    def test_mode_retrieve_da_ach(self):
        """Moderate-high DA + ACh drive retrieve mode."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.65
        manager._state.levels.acetylcholine = 0.55
        manager._state.levels.norepinephrine = 0.3
        manager._state.levels.serotonin = 0.4
        manager._state.levels.gaba = 0.3
        manager._update_cognitive_mode()

        assert manager._state.cognitive_mode.mode == "retrieve"

    def test_mode_rest_high_gaba_low_ne(self):
        """High GABA + low NE drive rest mode."""
        manager = NTStateManager()
        manager._state.levels.gaba = 0.7
        manager._state.levels.norepinephrine = 0.2
        manager._state.levels.dopamine = 0.3
        manager._state.levels.acetylcholine = 0.3
        manager._update_cognitive_mode()

        assert manager._state.cognitive_mode.mode == "rest"

    def test_mode_balanced_default(self):
        """Balanced mode is default."""
        manager = NTStateManager()
        manager._state.levels = NeurotransmitterLevels()
        manager._update_cognitive_mode()

        assert manager._state.cognitive_mode.mode == "balanced"

    def test_mode_confidence_high(self):
        """Confidence is high for strong mode signals."""
        manager = NTStateManager()
        manager._state.levels.norepinephrine = 0.9
        manager._update_cognitive_mode()

        assert manager._state.cognitive_mode.confidence > 0.7

    def test_mode_exploration_drive_increases_with_ne(self):
        """Exploration drive increases with NE."""
        manager = NTStateManager()

        manager._state.levels.norepinephrine = 0.2
        manager._update_cognitive_mode()
        exploration_low = manager._state.cognitive_mode.exploration_drive

        manager._state.levels.norepinephrine = 0.8
        manager._update_cognitive_mode()
        exploration_high = manager._state.cognitive_mode.exploration_drive

        assert exploration_high > exploration_low

    def test_mode_exploitation_drive_increases_with_da(self):
        """Exploitation drive increases with DA."""
        manager = NTStateManager()

        manager._state.levels.dopamine = 0.2
        manager._update_cognitive_mode()
        exploitation_low = manager._state.cognitive_mode.exploitation_drive

        manager._state.levels.dopamine = 0.9
        manager._update_cognitive_mode()
        exploitation_high = manager._state.cognitive_mode.exploitation_drive

        assert exploitation_high > exploitation_low

    def test_mode_learning_rate_increases_with_ach(self):
        """Learning rate modifier increases with ACh."""
        manager = NTStateManager()

        manager._state.levels.acetylcholine = 0.1
        manager._update_cognitive_mode()
        lr_low = manager._state.cognitive_mode.learning_rate_mod

        manager._state.levels.acetylcholine = 0.9
        manager._update_cognitive_mode()
        lr_high = manager._state.cognitive_mode.learning_rate_mod

        assert lr_high > lr_low

    def test_mode_attention_gain_increases_with_ach_ne(self):
        """Attention gain increases with ACh and NE."""
        manager = NTStateManager()

        manager._state.levels.acetylcholine = 0.1
        manager._state.levels.norepinephrine = 0.1
        manager._update_cognitive_mode()
        attn_low = manager._state.cognitive_mode.attention_gain

        manager._state.levels.acetylcholine = 0.9
        manager._state.levels.norepinephrine = 0.9
        manager._update_cognitive_mode()
        attn_high = manager._state.cognitive_mode.attention_gain

        assert attn_high > attn_low


class TestNTStateManagerSignals:
    """Tests for derived signals (RPE, uncertainty)."""

    def test_update_signals_rpe_from_da(self):
        """RPE is derived from DA deviation."""
        manager = NTStateManager()
        manager._state.levels.dopamine = 0.5
        manager._update_signals()
        assert manager._state.da_rpe == 0.0

        manager._state.levels.dopamine = 0.8
        manager._update_signals()
        assert manager._state.da_rpe > 0

        manager._state.levels.dopamine = 0.2
        manager._update_signals()
        assert manager._state.da_rpe < 0

    def test_update_signals_ach_uncertainty(self):
        """Expected uncertainty derived from ACh."""
        manager = NTStateManager()
        manager._state.levels.acetylcholine = 0.5
        manager._update_signals()
        assert manager._state.ach_uncertainty == pytest.approx(0.4)

        manager._state.levels.acetylcholine = 1.0
        manager._update_signals()
        assert manager._state.ach_uncertainty == pytest.approx(0.8)

    def test_update_signals_ne_surprise(self):
        """Unexpected uncertainty derived from NE."""
        manager = NTStateManager()
        manager._state.levels.norepinephrine = 0.5
        manager._update_signals()
        assert manager._state.ne_surprise == pytest.approx(0.45)

        manager._state.levels.norepinephrine = 1.0
        manager._update_signals()
        assert manager._state.ne_surprise == pytest.approx(0.9)


class TestNTStateManagerReset:
    """Tests for reset functionality."""

    def test_reset_clears_levels(self):
        """Reset returns levels to baseline."""
        manager = NTStateManager()
        manager.inject_nt("da", 0.4)
        manager.inject_nt("ne", 0.5)

        manager.reset()

        assert manager._state.levels.dopamine == 0.5
        assert manager._state.levels.serotonin == 0.5
        assert manager._state.levels.acetylcholine == 0.5
        assert manager._state.levels.norepinephrine == 0.3
        assert manager._state.levels.gaba == 0.4
        assert manager._state.levels.glutamate == 0.5

    def test_reset_clears_traces(self):
        """Reset clears trace history."""
        manager = NTStateManager()
        manager.inject_nt("da", 0.2)
        manager.inject_nt("5ht", 0.1)

        manager.reset()

        assert manager.get_traces() == []

    def test_reset_clears_history(self):
        """Reset clears DA/ACh history."""
        manager = NTStateManager()
        manager._state.da_history = [0.5, 0.6, 0.7]
        manager._state.ach_history = [0.5, 0.4]

        manager.reset()

        assert manager._state.da_history == []
        assert manager._state.ach_history == []

    def test_reset_clears_signals(self):
        """Reset clears RPE and uncertainty signals."""
        manager = NTStateManager()
        manager._state.da_rpe = 0.5
        manager._state.ach_uncertainty = 0.3
        manager._state.ne_surprise = 0.4

        manager.reset()

        assert manager._state.da_rpe == 0.0
        assert manager._state.ach_uncertainty == 0.0
        assert manager._state.ne_surprise == 0.0

    def test_reset_resets_timestamp(self):
        """Reset updates timestamp."""
        manager = NTStateManager()
        old_time = manager._state.timestamp
        sleep(0.01)

        manager.reset()

        assert manager._state.timestamp > old_time


class TestNTStateManagerGetters:
    """Tests for getter methods."""

    def test_get_state_returns_current_state(self):
        """get_state returns updated state."""
        manager = NTStateManager()
        manager.inject_nt("da", 0.2)

        state = manager.get_state()

        assert state.levels.dopamine == 0.7

    def test_get_traces_limit(self):
        """get_traces respects limit parameter."""
        manager = NTStateManager()
        for i in range(50):
            manager.inject_nt("da", 0.01)

        traces_10 = manager.get_traces(limit=10)
        traces_30 = manager.get_traces(limit=30)
        traces_all = manager.get_traces(limit=1000)

        assert len(traces_10) == 10
        assert len(traces_30) == 30
        assert len(traces_all) == 50

    def test_get_traces_default_limit(self):
        """get_traces defaults to limit=100."""
        manager = NTStateManager()
        for i in range(150):
            manager.inject_nt("da", 0.01)

        traces = manager.get_traces()

        assert len(traces) == 100

    def test_get_traces_returns_recent(self):
        """get_traces returns most recent entries."""
        manager = NTStateManager()
        manager.inject_nt("da", 0.1)
        manager.inject_nt("5ht", 0.2)
        manager.inject_nt("ach", 0.3)

        traces = manager.get_traces()

        # Last trace should be ach
        assert traces[-1].nt_type == "ach"
        assert traces[-2].nt_type == "5ht"
        assert traces[-3].nt_type == "da"

    def test_get_receptor_dict_format(self):
        """_get_receptor_dict returns correct format."""
        manager = NTStateManager()
        receptor_dict = manager._get_receptor_dict()

        required_keys = ["d1", "d2", "alpha1", "beta", "m1", "nmda", "gaba_a"]
        for key in required_keys:
            assert key in receptor_dict
            assert isinstance(receptor_dict[key], float)
            assert 0 <= receptor_dict[key] <= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestNTStateManagerIntegration:
    """Integration tests for complex scenarios."""

    def test_da_injection_exploration_recovery(self):
        """DA injection followed by recovery to baseline."""
        manager = NTStateManager()
        manager.inject_nt("da", 0.3)
        initial_da = manager._state.levels.dopamine

        # Simulate decay over time
        for _ in range(15):
            manager._last_update = datetime.now() - timedelta(seconds=0.3)
            manager._update_dynamics()

        final_da = manager._state.levels.dopamine

        # Should decay back toward baseline (0.5)
        assert final_da < initial_da
        assert 0.5 < final_da < initial_da

    def test_multiple_nt_injection_combined_mode(self):
        """Multiple NT injections create combined cognitive effect."""
        manager = NTStateManager()
        # High DA + high ACh should drive toward encode or retrieve
        manager.inject_nt("da", 0.2)
        manager.inject_nt("ach", 0.25)

        # Either encode or retrieve is acceptable with ACh high
        assert manager._state.cognitive_mode.mode in ["encode", "retrieve"]

    def test_receptor_cascade_from_injection(self):
        """Single injection affects multiple receptor types."""
        manager = NTStateManager()
        manager.inject_nt("ne", 0.4)

        # NE affects alpha1 and beta
        assert manager._state.receptors.alpha1_saturation > 0.3
        assert manager._state.receptors.beta_saturation > 0.3

    def test_trace_and_history_consistency(self):
        """Traces and history remain consistent."""
        manager = NTStateManager()
        manager.inject_nt("da", 0.2)
        manager._last_update = datetime.now() - timedelta(seconds=0.2)
        manager._update_dynamics()
        manager.inject_nt("da", 0.1)

        traces = manager.get_traces()
        history = manager._state.da_history

        assert len(traces) >= 2
        assert len(history) >= 1

    def test_sequential_injections_within_bounds(self):
        """Sequential injections accumulate within bounds."""
        manager = NTStateManager()
        initial = manager._state.levels.dopamine

        manager.inject_nt("da", 0.15)
        first = manager._state.levels.dopamine
        assert first == initial + 0.15

        manager.inject_nt("da", 0.15)
        second = manager._state.levels.dopamine
        assert second == first + 0.15

        manager.inject_nt("da", 0.21)
        third = manager._state.levels.dopamine
        # Should clamp at 1.0 but may include noise from dynamics
        assert third <= 1.0
        assert third > 0.9

    def test_full_workflow_inject_decay_reset(self):
        """Full workflow: inject -> decay -> reset."""
        manager = NTStateManager()

        # Inject
        manager.inject_nt("da", 0.3)
        assert manager._state.levels.dopamine == 0.8

        # Decay (use longer period for reliable decay)
        manager._last_update = datetime.now() - timedelta(seconds=2.0)
        manager._update_dynamics()
        decayed = manager._state.levels.dopamine
        # Should decay towards baseline (0.5) but not reach it yet
        assert 0.5 < decayed < 0.8

        # Check traces exist
        assert len(manager.get_traces()) > 0

        # Reset
        manager.reset()
        assert manager._state.levels.dopamine == 0.5
        assert len(manager.get_traces()) == 0


# =============================================================================
# Router Tests (if routes are accessible)
# =============================================================================


class TestNTRouterEndpoints:
    """Tests for NT router endpoints."""

    def test_router_has_endpoints(self):
        """Router includes all required endpoints."""
        routes = [route.path for route in nt_router.routes]
        assert any("/state" in route for route in routes)
        assert any("/traces" in route for route in routes)
        assert any("/inject" in route for route in routes)
        assert any("/reset" in route for route in routes)
        assert any("/receptors" in route for route in routes)
        assert any("/mode" in route for route in routes)


__all__ = [
    "TestNeurotransmitterLevels",
    "TestReceptorSaturation",
    "TestCognitiveMode",
    "TestNTDashboardState",
    "TestInjectNTRequest",
    "TestInjectNTResponse",
    "TestNTStateManagerInit",
    "TestNTStateManagerInject",
    "TestNTStateManagerDynamics",
    "TestNTStateManagerMichaelisMenten",
    "TestNTStateManagerReceptors",
    "TestNTStateManagerCognitiveMode",
    "TestNTStateManagerSignals",
    "TestNTStateManagerReset",
    "TestNTStateManagerGetters",
    "TestNTStateManagerIntegration",
    "TestNTRouterEndpoints",
]
