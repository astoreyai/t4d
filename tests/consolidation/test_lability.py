"""
Tests for Lability Window (Phase 7).

Tests the protein synthesis gate for memory reconsolidation.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from t4dm.consolidation.lability import (
    DEFAULT_LABILITY_WINDOW_HOURS,
    LabilityConfig,
    LabilityManager,
    LabilityPhase,
    LabilityState,
    compute_reconsolidation_strength,
    get_lability_manager,
    get_reconsolidation_learning_rate,
    is_reconsolidation_eligible,
    reset_lability_manager,
)


class TestLabilityConfig:
    """Tests for LabilityConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LabilityConfig()
        assert config.window_hours == DEFAULT_LABILITY_WINDOW_HOURS
        assert config.min_retrieval_strength == 0.3
        assert config.emotional_modulation is True
        assert config.require_prediction_error is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LabilityConfig(
            window_hours=5.0,
            min_retrieval_strength=0.5,
            emotional_modulation=False,
        )
        assert config.window_hours == 5.0
        assert config.min_retrieval_strength == 0.5
        assert config.emotional_modulation is False

    def test_window_clamping(self):
        """Test window hours clamped to biological range."""
        # Too short - should clamp to minimum
        config = LabilityConfig(window_hours=1.0)
        assert config.window_hours == 4.0

        # Too long - should clamp to maximum
        config = LabilityConfig(window_hours=20.0)
        assert config.window_hours == 8.0


class TestLabilityState:
    """Tests for LabilityState."""

    def test_initial_state(self):
        """Test initial state values."""
        memory_id = uuid4()
        state = LabilityState(memory_id=memory_id)

        assert state.memory_id == memory_id
        assert state.last_retrieval is None
        assert state.phase == LabilityPhase.STABLE
        assert state.retrieval_strength == 0.0
        assert state.reconsolidation_count == 0

    def test_to_dict(self):
        """Test state serialization."""
        memory_id = uuid4()
        state = LabilityState(
            memory_id=memory_id,
            last_retrieval=datetime(2025, 1, 1, 12, 0, 0),
            phase=LabilityPhase.LABILE,
            retrieval_strength=0.8,
            prediction_error=0.2,
        )

        data = state.to_dict()
        assert data["memory_id"] == str(memory_id)
        assert data["phase"] == "labile"
        assert data["retrieval_strength"] == 0.8
        assert data["prediction_error"] == 0.2


class TestLabilityManager:
    """Tests for LabilityManager."""

    @pytest.fixture
    def manager(self):
        """Create fresh lability manager."""
        reset_lability_manager()
        return LabilityManager()

    @pytest.fixture
    def memory_id(self):
        """Create a test memory ID."""
        return uuid4()

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.config is not None
        assert len(manager._states) == 0

    def test_on_retrieval_basic(self, manager, memory_id):
        """Test basic retrieval recording."""
        state = manager.on_retrieval(
            memory_id=memory_id,
            strength=0.8,
            emotional_valence=0.6,
            prediction_error=0.3,
        )

        assert state.memory_id == memory_id
        assert state.phase == LabilityPhase.LABILE
        assert state.retrieval_strength == 0.8
        assert state.last_retrieval is not None

    def test_on_retrieval_weak_strength(self, manager, memory_id):
        """Test weak retrieval doesn't trigger lability."""
        state = manager.on_retrieval(
            memory_id=memory_id,
            strength=0.1,  # Below threshold
            prediction_error=0.5,
        )

        assert state.phase == LabilityPhase.STABLE

    def test_on_retrieval_low_prediction_error(self, manager, memory_id):
        """Test low prediction error doesn't trigger lability."""
        manager.config.require_prediction_error = True

        state = manager.on_retrieval(
            memory_id=memory_id,
            strength=0.8,
            prediction_error=0.01,  # Below threshold
        )

        # Memory was retrieved but not destabilized
        assert state.phase == LabilityPhase.STABLE
        assert state.last_retrieval is not None

    def test_is_labile_within_window(self, manager, memory_id):
        """Test memory is labile within window."""
        manager.on_retrieval(memory_id, strength=0.8, prediction_error=0.5)

        # Immediately after retrieval
        assert manager.is_labile(memory_id) is True

    def test_is_labile_outside_window(self, manager, memory_id):
        """Test memory not labile outside window."""
        manager.on_retrieval(memory_id, strength=0.8, prediction_error=0.5)

        # Check 10 hours later
        future = datetime.now() + timedelta(hours=10)
        assert manager.is_labile(memory_id, now=future) is False

    def test_is_labile_unknown_memory(self, manager):
        """Test unknown memory is not labile."""
        assert manager.is_labile(uuid4()) is False

    def test_emotional_modulation_extends_window(self, manager, memory_id):
        """Test high emotion extends lability window."""
        manager.config.emotional_modulation = True

        # High emotion (0.9) should extend window by factor of 1.4
        manager.on_retrieval(
            memory_id,
            strength=0.8,
            emotional_valence=0.9,
            prediction_error=0.5,
        )

        # At 7 hours (beyond base 6h window, but within extended)
        future = datetime.now() + timedelta(hours=7)
        assert manager.is_labile(memory_id, now=future) is True

    def test_get_window_remaining(self, manager, memory_id):
        """Test remaining window calculation."""
        manager.on_retrieval(memory_id, strength=0.8, prediction_error=0.5)

        # Immediately after: should be close to full window
        remaining = manager.get_window_remaining(memory_id)
        assert remaining > 5.5

        # 2 hours later
        future = datetime.now() + timedelta(hours=2)
        remaining = manager.get_window_remaining(memory_id, now=future)
        assert 3.5 < remaining < 4.5

    def test_on_reconsolidation_success(self, manager, memory_id):
        """Test successful reconsolidation."""
        manager.on_retrieval(memory_id, strength=0.8, prediction_error=0.5)
        state = manager.on_reconsolidation(memory_id, success=True)

        assert state.phase == LabilityPhase.STABLE
        assert state.reconsolidation_count == 1

    def test_on_reconsolidation_failure(self, manager, memory_id):
        """Test failed reconsolidation."""
        manager.on_retrieval(memory_id, strength=0.8, prediction_error=0.5)
        state = manager.on_reconsolidation(memory_id, success=False)

        assert state.phase == LabilityPhase.DESTABILIZED
        assert state.reconsolidation_count == 0

    def test_on_reconsolidation_unknown(self, manager):
        """Test reconsolidation of unknown memory."""
        result = manager.on_reconsolidation(uuid4())
        assert result is None

    def test_get_labile_memories(self, manager):
        """Test getting all labile memories."""
        ids = [uuid4() for _ in range(5)]

        for mid in ids:
            manager.on_retrieval(mid, strength=0.8, prediction_error=0.5)

        labile = manager.get_labile_memories()
        assert len(labile) == 5
        assert set(labile) == set(ids)

    def test_get_stats(self, manager, memory_id):
        """Test statistics retrieval."""
        manager.on_retrieval(memory_id, strength=0.8, prediction_error=0.5)
        manager.on_reconsolidation(memory_id, success=True)

        stats = manager.get_stats()

        assert stats["tracked_memories"] == 1
        assert stats["total_retrievals"] == 1
        assert stats["total_reconsolidations"] == 1
        assert "phase_distribution" in stats

    def test_clear(self, manager, memory_id):
        """Test clearing all states."""
        manager.on_retrieval(memory_id, strength=0.8, prediction_error=0.5)
        manager.clear()

        assert len(manager._states) == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_is_reconsolidation_eligible_within_window(self):
        """Test eligibility within window."""
        recent = datetime.now() - timedelta(hours=2)
        assert is_reconsolidation_eligible(recent) is True

    def test_is_reconsolidation_eligible_outside_window(self):
        """Test eligibility outside window."""
        old = datetime.now() - timedelta(hours=10)
        assert is_reconsolidation_eligible(old) is False

    def test_is_reconsolidation_eligible_custom_window(self):
        """Test eligibility with custom window."""
        retrieval = datetime.now() - timedelta(hours=5)

        # Should be outside 4h window
        assert is_reconsolidation_eligible(retrieval, window_hours=4.0) is False

        # Should be inside 8h window
        assert is_reconsolidation_eligible(retrieval, window_hours=8.0) is True

    def test_compute_reconsolidation_strength_basic(self):
        """Test reconsolidation strength computation."""
        strength = compute_reconsolidation_strength(
            retrieval_strength=0.8,
            emotional_valence=0.5,
            prediction_error=0.2,
            hours_elapsed=1.0,
        )

        assert 0.0 <= strength <= 1.0
        assert strength > 0.5  # Should be reasonably strong

    def test_compute_reconsolidation_strength_decays_with_time(self):
        """Test strength decays over time."""
        base_args = {
            "retrieval_strength": 0.8,
            "emotional_valence": 0.5,
            "prediction_error": 0.2,
        }

        early = compute_reconsolidation_strength(**base_args, hours_elapsed=0.5)
        late = compute_reconsolidation_strength(**base_args, hours_elapsed=4.0)

        assert early > late

    def test_compute_reconsolidation_strength_emotion_boost(self):
        """Test high emotion boosts strength."""
        base_args = {
            "retrieval_strength": 0.8,
            "prediction_error": 0.2,
            "hours_elapsed": 1.0,
        }

        low_emotion = compute_reconsolidation_strength(**base_args, emotional_valence=0.2)
        high_emotion = compute_reconsolidation_strength(**base_args, emotional_valence=0.9)

        assert high_emotion > low_emotion

    def test_compute_reconsolidation_strength_pe_boost(self):
        """Test high prediction error boosts strength."""
        base_args = {
            "retrieval_strength": 0.8,
            "emotional_valence": 0.5,
            "hours_elapsed": 1.0,
        }

        low_pe = compute_reconsolidation_strength(**base_args, prediction_error=0.1)
        high_pe = compute_reconsolidation_strength(**base_args, prediction_error=0.8)

        assert high_pe > low_pe

    def test_get_reconsolidation_learning_rate_basic(self):
        """Test learning rate computation."""
        lr = get_reconsolidation_learning_rate(
            base_lr=0.01,
            reconsolidation_strength=0.8,
            reconsolidation_count=0,
        )

        assert lr > 0
        assert lr <= 0.01

    def test_get_reconsolidation_learning_rate_decays_with_count(self):
        """Test learning rate decays with repeated reconsolidations."""
        base_args = {
            "base_lr": 0.01,
            "reconsolidation_strength": 0.8,
        }

        first_time = get_reconsolidation_learning_rate(**base_args, reconsolidation_count=0)
        fifth_time = get_reconsolidation_learning_rate(**base_args, reconsolidation_count=5)

        assert first_time > fifth_time


class TestSingleton:
    """Tests for singleton pattern."""

    def test_get_lability_manager(self):
        """Test singleton creation."""
        reset_lability_manager()
        manager1 = get_lability_manager()
        manager2 = get_lability_manager()

        assert manager1 is manager2

    def test_reset_lability_manager(self):
        """Test singleton reset."""
        manager1 = get_lability_manager()
        reset_lability_manager()
        manager2 = get_lability_manager()

        assert manager1 is not manager2


class TestBiologicalConstraints:
    """Tests for biological plausibility."""

    def test_window_duration_range(self):
        """Test lability window is in biological range (4-8 hours)."""
        config = LabilityConfig()
        assert 4.0 <= config.window_hours <= 8.0

    def test_default_window_is_six_hours(self):
        """Test default window matches literature (Nader et al.)."""
        assert DEFAULT_LABILITY_WINDOW_HOURS == 6.0

    def test_prediction_error_gating(self):
        """Test prediction error gates reconsolidation (Sevenster et al. 2012)."""
        manager = LabilityManager(LabilityConfig(require_prediction_error=True))
        memory_id = uuid4()

        # Retrieval without surprise should not destabilize
        state = manager.on_retrieval(
            memory_id,
            strength=0.9,
            prediction_error=0.01,  # No surprise
        )

        assert state.phase == LabilityPhase.STABLE

    def test_emotional_enhancement(self):
        """Test emotional memories have extended windows (McGaugh 2004)."""
        manager = LabilityManager(LabilityConfig(emotional_modulation=True))
        memory_id = uuid4()

        manager.on_retrieval(
            memory_id,
            strength=0.8,
            emotional_valence=1.0,  # Maximum emotion
            prediction_error=0.5,
        )

        # Window should be extended to ~9h (6h * 1.5 = 9h)
        # Check at 8 hours - should still be labile
        future = datetime.now() + timedelta(hours=8)
        assert manager.is_labile(memory_id, now=future) is True
