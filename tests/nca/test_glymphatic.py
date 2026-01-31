"""
Unit tests for Glymphatic System (B8).

Tests waste clearance rates, sleep-state modulation,
delta coupling, and NE/ACh modulation.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta

from t4dm.nca.glymphatic import (
    GlymphaticConfig,
    GlymphaticSystem,
    WasteState,
    WasteTracker,
    WasteCategory,
    ClearanceEvent,
    WakeSleepMode,
    create_glymphatic_system,
)


class TestGlymphaticConfig:
    """Test GlymphaticConfig dataclass."""

    def test_default_config(self):
        """Default config has biological values."""
        config = GlymphaticConfig()

        # Clearance rates from Xie et al. 2013 (60% higher than wake, not 90%)
        assert config.clearance_nrem_deep == 0.7
        assert config.clearance_nrem_light == 0.5
        assert config.clearance_quiet_wake == 0.3
        assert config.clearance_rem == 0.05  # Low due to ACh

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = GlymphaticConfig(
            clearance_nrem_deep=0.95,
            unused_embedding_days=60,
            ne_modulation=0.8
        )

        assert config.clearance_nrem_deep == 0.95
        assert config.unused_embedding_days == 60
        assert config.ne_modulation == 0.8

    def test_safety_limits(self):
        """Config includes safety limits."""
        config = GlymphaticConfig()

        assert config.max_clearance_fraction == 0.1  # Max 10% per cycle
        assert config.preserve_recent_hours == 24  # Keep items < 24h


class TestWasteState:
    """Test WasteState dataclass."""

    def test_empty_state(self):
        """Empty state has zero counts."""
        state = WasteState()

        assert state.unused_embeddings == 0
        assert state.weak_connections == 0
        assert state.stale_memories == 0
        assert state.total_waste == 0
        assert state.total_cleared == 0

    def test_total_waste_property(self):
        """total_waste sums all categories."""
        state = WasteState(
            unused_embeddings=10,
            weak_connections=20,
            stale_memories=5,
            orphan_entities=3,
            expired_episodes=2
        )

        assert state.total_waste == 40

    def test_cleared_by_category(self):
        """Breakdown of cleared items by category."""
        state = WasteState(
            cleared_embeddings=5,
            cleared_connections=10,
            cleared_memories=3,
            total_cleared=18
        )

        breakdown = state.total_cleared_by_category
        assert breakdown["embeddings"] == 5
        assert breakdown["connections"] == 10
        assert breakdown["memories"] == 3


class TestWasteTracker:
    """Test WasteTracker class."""

    @pytest.fixture
    def tracker(self):
        """Create test waste tracker."""
        config = GlymphaticConfig()
        return WasteTracker(config)

    def test_initial_state(self, tracker):
        """Tracker starts empty."""
        candidates = tracker.get_clearance_candidates(WasteCategory.UNUSED_EMBEDDING)
        assert len(candidates) == 0

    def test_scan_without_memory_system(self, tracker):
        """Scan without memory system returns empty state."""
        state = tracker.scan_for_waste()
        assert state.total_waste == 0

    def test_mark_cleared(self, tracker):
        """Marking items as cleared removes them."""
        # Manually add waste item
        tracker._waste_items[WasteCategory.WEAK_CONNECTION].append({
            "id": "conn_1",
            "source": "a",
            "target": "b",
            "weight": 0.05
        })

        assert len(tracker.get_clearance_candidates(WasteCategory.WEAK_CONNECTION)) == 1

        tracker.mark_cleared(
            WasteCategory.WEAK_CONNECTION,
            "conn_1",
            WakeSleepMode.NREM_DEEP,
            0.9
        )

        assert len(tracker.get_clearance_candidates(WasteCategory.WEAK_CONNECTION)) == 0

    def test_clearance_history(self, tracker):
        """Clearance events are recorded."""
        tracker.mark_cleared(
            WasteCategory.STALE_MEMORY,
            "mem_1",
            WakeSleepMode.NREM_DEEP,
            0.8
        )
        tracker.mark_cleared(
            WasteCategory.UNUSED_EMBEDDING,
            "emb_1",
            WakeSleepMode.NREM_LIGHT,
            0.5
        )

        history = tracker.get_clearance_history()
        assert len(history) == 2

        # Filter by category
        mem_history = tracker.get_clearance_history(category=WasteCategory.STALE_MEMORY)
        assert len(mem_history) == 1
        assert mem_history[0].item_id == "mem_1"

    def test_reset(self, tracker):
        """Reset clears all tracked waste."""
        tracker._waste_items[WasteCategory.UNUSED_EMBEDDING].append({"id": "x"})
        tracker.mark_cleared(
            WasteCategory.UNUSED_EMBEDDING,
            "y",
            WakeSleepMode.QUIET_WAKE,
            0.3
        )

        tracker.reset()

        assert len(tracker.get_clearance_candidates(WasteCategory.UNUSED_EMBEDDING)) == 0
        assert len(tracker.get_clearance_history()) == 0


class TestGlymphaticSystem:
    """Test GlymphaticSystem class."""

    @pytest.fixture
    def system(self):
        """Create test glymphatic system."""
        return GlymphaticSystem()

    def test_initial_state(self, system):
        """System starts with zero state."""
        assert system.state.total_cleared == 0
        assert system.state.clearance_rate == 0.0
        assert system._total_steps == 0

    def test_get_state_clearance_rate(self, system):
        """Correct clearance rates per sleep state."""
        # NREM deep has highest clearance (Xie et al. 2013: ~2x wake)
        rate_deep = system.get_state_clearance_rate(WakeSleepMode.NREM_DEEP)
        assert rate_deep == 0.7

        # REM has lowest (ACh blocks AQP4)
        rate_rem = system.get_state_clearance_rate(WakeSleepMode.REM)
        assert rate_rem == 0.05

        # Active wake is low
        rate_active = system.get_state_clearance_rate(WakeSleepMode.ACTIVE_WAKE)
        assert rate_active == 0.1


class TestClearanceRateComputation:
    """Test effective clearance rate computation."""

    @pytest.fixture
    def system(self):
        """Create test glymphatic system."""
        return GlymphaticSystem()

    def test_base_rate_by_state(self, system):
        """Base rate varies by sleep state."""
        # Full clearance conditions: delta up-state, low NE
        rate_deep = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.0
        )
        rate_wake = system.compute_effective_rate(
            WakeSleepMode.ACTIVE_WAKE,
            delta_up_state=True,
            ne_level=0.0
        )

        assert rate_deep > rate_wake
        assert rate_deep == 0.7  # Xie et al. 2013: ~2x wake clearance
        assert rate_wake == 0.1

    def test_delta_upstate_gating(self, system):
        """Clearance reduced outside delta up-states."""
        rate_up = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.0
        )
        rate_down = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=False,
            ne_level=0.0
        )

        # Down-state should have 10% of up-state rate
        assert rate_down == rate_up * 0.1

    def test_ne_modulation(self, system):
        """High NE reduces clearance."""
        rate_low_ne = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.0
        )
        rate_high_ne = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=1.0
        )

        assert rate_high_ne < rate_low_ne

        # NE modulation factor at max NE
        # ne_factor = 1 - 1.0 * 0.6 = 0.4
        expected_ratio = 0.4
        assert np.isclose(rate_high_ne / rate_low_ne, expected_ratio, rtol=0.01)

    def test_ach_modulation(self, system):
        """High ACh reduces clearance (REM-like)."""
        rate_low_ach = system.compute_effective_rate(
            WakeSleepMode.NREM_LIGHT,
            delta_up_state=True,
            ne_level=0.0,
            ach_level=0.0
        )
        rate_high_ach = system.compute_effective_rate(
            WakeSleepMode.NREM_LIGHT,
            delta_up_state=True,
            ne_level=0.0,
            ach_level=1.0
        )

        assert rate_high_ach < rate_low_ach

    def test_combined_modulation(self, system):
        """NE and ACh modulate together."""
        rate_optimal = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.0,
            ach_level=0.0
        )
        rate_suboptimal = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.5,
            ach_level=0.5
        )

        assert rate_suboptimal < rate_optimal


class TestGlymphaticStep:
    """Test glymphatic step execution."""

    @pytest.fixture
    def system(self):
        """Create test glymphatic system."""
        return GlymphaticSystem()

    def test_step_updates_state(self, system):
        """Step updates clearance rate in state."""
        state = system.step(
            wake_sleep_mode=WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.1,
            dt=1.0
        )

        assert state.clearance_rate > 0
        assert system._total_steps == 1

    def test_step_accumulates_clearance(self, system):
        """Clearance accumulates over steps."""
        for _ in range(5):
            system.step(
                wake_sleep_mode=WakeSleepMode.NREM_DEEP,
                delta_up_state=True,
                ne_level=0.1,
                dt=0.1
            )

        assert system._accumulated_clearance > 0
        assert system._total_steps == 5

    def test_step_with_time(self, system):
        """Step tracks current time."""
        test_time = datetime(2026, 1, 4, 3, 0, 0)

        system.step(
            wake_sleep_mode=WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.2,
            dt=1.0,
            current_time=test_time
        )

        assert system._last_step_time == test_time


class TestBiologicalConstraints:
    """Test biological accuracy of clearance (B8)."""

    @pytest.fixture
    def system(self):
        """Create glymphatic system with default biological parameters."""
        return GlymphaticSystem()

    def test_nrem_higher_than_wake(self, system):
        """NREM clearance is ~2x higher than wake (Xie 2013)."""
        rate_nrem = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.1  # Low NE during sleep
        )
        rate_wake = system.compute_effective_rate(
            WakeSleepMode.QUIET_WAKE,
            delta_up_state=True,
            ne_level=0.5  # Moderate NE when awake
        )

        # NREM should be significantly higher
        assert rate_nrem > rate_wake * 1.5

    def test_rem_minimal_clearance(self, system):
        """REM has minimal clearance due to high ACh."""
        rate_rem = system.get_state_clearance_rate(WakeSleepMode.REM)
        rate_nrem = system.get_state_clearance_rate(WakeSleepMode.NREM_DEEP)

        # REM should be <10% of NREM
        assert rate_rem < rate_nrem * 0.1

    def test_low_ne_required_for_clearance(self, system):
        """Low NE (sleep) is required for high clearance."""
        # Biological: NE contracts astrocytes, blocking interstitial flow
        rate_sleep_ne = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.1  # Low NE during NREM
        )
        rate_wake_ne = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,  # Same state
            delta_up_state=True,
            ne_level=0.8  # High NE (aroused)
        )

        assert rate_sleep_ne > rate_wake_ne


class TestGlymphaticIntegration:
    """Test integration with other components."""

    def test_scan_with_mock_memory(self):
        """Scan works with mock memory system."""
        class MockMemory:
            def get_unused_embeddings(self, days):
                return ["emb_1", "emb_2", "emb_3"]

            def get_weak_connections(self, threshold):
                return [("a", "b", 0.05), ("c", "d", 0.08)]

            def get_stale_memories(self, stability):
                return ["mem_1"]

            def get_orphan_entities(self):
                return []

        system = GlymphaticSystem(memory_system=MockMemory())
        state = system.scan()

        assert state.unused_embeddings == 3
        assert state.weak_connections == 2
        assert state.stale_memories == 1

    def test_statistics(self):
        """Get system statistics."""
        system = GlymphaticSystem()

        # Run some steps
        for i in range(10):
            system.step(
                wake_sleep_mode=WakeSleepMode.NREM_DEEP,
                delta_up_state=True,
                ne_level=0.1,
                dt=0.5
            )

        stats = system.get_statistics()

        assert stats["total_steps"] == 10
        assert "current_clearance_rate" in stats
        assert "total_cleared" in stats
        assert "waste_breakdown" in stats

    def test_reset(self):
        """Reset clears all state."""
        system = GlymphaticSystem()

        system.step(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.1,
            dt=1.0
        )

        system.reset()

        assert system._total_steps == 0
        assert system._accumulated_clearance == 0.0
        assert system.state.total_cleared == 0


class TestFactoryFunction:
    """Test factory function."""

    def test_create_glymphatic_system(self):
        """Factory creates configured system."""
        system = create_glymphatic_system(
            clearance_nrem_deep=0.95,
            clearance_wake=0.25,
            ne_modulation=0.7
        )

        assert system.config.clearance_nrem_deep == 0.95
        assert system.config.clearance_quiet_wake == 0.25
        assert system.config.ne_modulation == 0.7


class TestClearanceEvent:
    """Test ClearanceEvent dataclass."""

    def test_event_creation(self):
        """Events store all required information."""
        event = ClearanceEvent(
            timestamp=datetime.now(),
            category=WasteCategory.WEAK_CONNECTION,
            item_id="conn_123",
            reason="Weight below threshold",
            wake_sleep_mode=WakeSleepMode.NREM_DEEP,
            clearance_rate=0.85
        )

        assert event.category == WasteCategory.WEAK_CONNECTION
        assert event.item_id == "conn_123"
        assert event.clearance_rate == 0.85


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_dt(self):
        """Zero dt step is valid (no accumulation)."""
        system = GlymphaticSystem()

        state = system.step(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=0.1,
            dt=0.0
        )

        assert state is not None
        assert system._accumulated_clearance == 0.0

    def test_extreme_ne_levels(self):
        """Extreme NE values are clamped."""
        system = GlymphaticSystem()

        # NE > 1 should be handled
        rate_high = system.compute_effective_rate(
            WakeSleepMode.NREM_DEEP,
            delta_up_state=True,
            ne_level=10.0  # Way above 1
        )

        # Should still be valid (clamped factor >= 0)
        assert rate_high >= 0

    def test_no_memory_system(self):
        """System works without memory system (no actual deletion)."""
        system = GlymphaticSystem()

        # Multiple steps without memory system
        for _ in range(100):
            system.step(
                WakeSleepMode.NREM_DEEP,
                delta_up_state=True,
                ne_level=0.1,
                dt=1.0
            )

        # Should not crash, but no actual clearing
        assert system.state.total_cleared == 0
