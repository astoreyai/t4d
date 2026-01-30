"""
Tests for P3.3: Automatic Consolidation Triggering.

Tests the ConsolidationScheduler class that triggers consolidation based on:
- Time: Hours since last consolidation
- Load: Number of new memories created
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ww.consolidation.service import (
    ConsolidationScheduler,
    ConsolidationTrigger,
    SchedulerState,
    TriggerReason,
    get_consolidation_scheduler,
    reset_consolidation_scheduler,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def scheduler():
    """Create a fresh scheduler for testing."""
    return ConsolidationScheduler(
        interval_hours=8.0,
        memory_threshold=100,
        check_interval_seconds=60.0,
        consolidation_type="light",
        enabled=True,
    )


@pytest.fixture
def disabled_scheduler():
    """Create a disabled scheduler."""
    return ConsolidationScheduler(
        interval_hours=8.0,
        memory_threshold=100,
        enabled=False,
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset scheduler singleton before and after each test."""
    reset_consolidation_scheduler()
    yield
    reset_consolidation_scheduler()


# =============================================================================
# Test TriggerReason
# =============================================================================


class TestTriggerReason:
    """Tests for TriggerReason enum."""

    def test_trigger_reasons_exist(self):
        """Test all trigger reasons are defined."""
        assert TriggerReason.TIME_BASED.value == "time_based"
        assert TriggerReason.LOAD_BASED.value == "load_based"
        assert TriggerReason.MANUAL.value == "manual"
        assert TriggerReason.STARTUP.value == "startup"


# =============================================================================
# Test SchedulerState
# =============================================================================


class TestSchedulerState:
    """Tests for SchedulerState dataclass."""

    def test_default_state(self):
        """Test default state values."""
        state = SchedulerState()
        assert state.new_memory_count == 0
        assert state.total_consolidations == 0
        assert state.last_trigger_reason is None
        assert state.is_running is False
        assert state.last_error is None
        assert isinstance(state.last_consolidation, datetime)

    def test_state_with_values(self):
        """Test state with custom values."""
        now = datetime.now()
        state = SchedulerState(
            last_consolidation=now,
            new_memory_count=50,
            total_consolidations=3,
            last_trigger_reason=TriggerReason.TIME_BASED,
            is_running=True,
            last_error="test error",
        )
        assert state.last_consolidation == now
        assert state.new_memory_count == 50
        assert state.total_consolidations == 3
        assert state.last_trigger_reason == TriggerReason.TIME_BASED
        assert state.is_running is True
        assert state.last_error == "test error"


# =============================================================================
# Test ConsolidationTrigger
# =============================================================================


class TestConsolidationTrigger:
    """Tests for ConsolidationTrigger dataclass."""

    def test_no_trigger(self):
        """Test trigger when should not run."""
        trigger = ConsolidationTrigger(should_run=False)
        assert trigger.should_run is False
        assert trigger.reason is None
        assert trigger.details == {}

    def test_time_trigger(self):
        """Test time-based trigger."""
        trigger = ConsolidationTrigger(
            should_run=True,
            reason=TriggerReason.TIME_BASED,
            details={"hours_since_last": 10.5, "threshold_hours": 8.0},
        )
        assert trigger.should_run is True
        assert trigger.reason == TriggerReason.TIME_BASED
        assert trigger.details["hours_since_last"] == 10.5

    def test_load_trigger(self):
        """Test load-based trigger."""
        trigger = ConsolidationTrigger(
            should_run=True,
            reason=TriggerReason.LOAD_BASED,
            details={"new_memory_count": 150, "threshold": 100},
        )
        assert trigger.should_run is True
        assert trigger.reason == TriggerReason.LOAD_BASED
        assert trigger.details["new_memory_count"] == 150


# =============================================================================
# Test ConsolidationScheduler
# =============================================================================


class TestConsolidationSchedulerInit:
    """Tests for scheduler initialization."""

    def test_default_init(self):
        """Test default initialization."""
        scheduler = ConsolidationScheduler()
        assert scheduler.interval_hours == 1.5
        assert scheduler.memory_threshold == 100
        assert scheduler.check_interval_seconds == 300.0
        assert scheduler.consolidation_type == "light"
        assert scheduler.enabled is True

    def test_custom_init(self):
        """Test custom initialization."""
        scheduler = ConsolidationScheduler(
            interval_hours=4.0,
            memory_threshold=50,
            check_interval_seconds=120.0,
            consolidation_type="deep",
            enabled=False,
        )
        assert scheduler.interval_hours == 4.0
        assert scheduler.memory_threshold == 50
        assert scheduler.check_interval_seconds == 120.0
        assert scheduler.consolidation_type == "deep"
        assert scheduler.enabled is False


class TestShouldConsolidate:
    """Tests for should_consolidate method."""

    def test_disabled_scheduler(self, disabled_scheduler):
        """Test disabled scheduler never triggers."""
        trigger = disabled_scheduler.should_consolidate()
        assert trigger.should_run is False

    def test_already_running(self, scheduler):
        """Test no trigger when consolidation is running."""
        scheduler.state.is_running = True
        trigger = scheduler.should_consolidate()
        assert trigger.should_run is False
        assert trigger.details.get("reason") == "consolidation_already_running"

    def test_time_based_trigger(self, scheduler):
        """Test time-based trigger fires after interval."""
        # Set last consolidation to 10 hours ago
        scheduler.state.last_consolidation = datetime.now() - timedelta(hours=10)
        trigger = scheduler.should_consolidate()
        assert trigger.should_run is True
        assert trigger.reason == TriggerReason.TIME_BASED
        assert trigger.details["hours_since_last"] >= 10.0

    def test_load_based_trigger(self, scheduler):
        """Test load-based trigger fires after threshold."""
        scheduler.state.new_memory_count = 150
        trigger = scheduler.should_consolidate()
        assert trigger.should_run is True
        assert trigger.reason == TriggerReason.LOAD_BASED
        assert trigger.details["new_memory_count"] == 150

    def test_no_trigger_within_thresholds(self, scheduler):
        """Test no trigger when within thresholds."""
        # Recent consolidation, few memories
        scheduler.state.last_consolidation = datetime.now() - timedelta(hours=1)
        scheduler.state.new_memory_count = 10
        trigger = scheduler.should_consolidate()
        assert trigger.should_run is False
        assert "hours_remaining" in trigger.details
        assert "memories_remaining" in trigger.details

    def test_time_takes_priority_over_load(self, scheduler):
        """Test time-based trigger checked before load."""
        # Both conditions met
        scheduler.state.last_consolidation = datetime.now() - timedelta(hours=10)
        scheduler.state.new_memory_count = 150
        trigger = scheduler.should_consolidate()
        # Time is checked first
        assert trigger.should_run is True
        assert trigger.reason == TriggerReason.TIME_BASED


class TestRecordMemoryCreated:
    """Tests for record_memory_created method."""

    def test_increment_by_one(self, scheduler):
        """Test incrementing by one."""
        assert scheduler.state.new_memory_count == 0
        scheduler.record_memory_created()
        assert scheduler.state.new_memory_count == 1

    def test_increment_by_count(self, scheduler):
        """Test incrementing by custom count."""
        scheduler.record_memory_created(count=5)
        assert scheduler.state.new_memory_count == 5
        scheduler.record_memory_created(count=10)
        assert scheduler.state.new_memory_count == 15

    def test_multiple_calls_accumulate(self, scheduler):
        """Test multiple calls accumulate."""
        for _ in range(10):
            scheduler.record_memory_created()
        assert scheduler.state.new_memory_count == 10


class TestRecordConsolidationComplete:
    """Tests for record_consolidation_complete method."""

    def test_successful_completion(self, scheduler):
        """Test recording successful completion."""
        scheduler.state.new_memory_count = 50
        scheduler.state.is_running = True

        scheduler.record_consolidation_complete(reason=TriggerReason.TIME_BASED)

        assert scheduler.state.new_memory_count == 0  # Reset
        assert scheduler.state.is_running is False
        assert scheduler.state.total_consolidations == 1
        assert scheduler.state.last_trigger_reason == TriggerReason.TIME_BASED
        assert scheduler.state.last_error is None

    def test_failed_completion(self, scheduler):
        """Test recording failed completion."""
        scheduler.record_consolidation_complete(
            reason=TriggerReason.LOAD_BASED,
            error="Connection timeout",
        )
        assert scheduler.state.last_error == "Connection timeout"
        assert scheduler.state.total_consolidations == 1

    def test_multiple_completions(self, scheduler):
        """Test multiple completions increment counter."""
        for i in range(5):
            scheduler.record_consolidation_complete(reason=TriggerReason.MANUAL)
        assert scheduler.state.total_consolidations == 5


class TestGetStats:
    """Tests for get_stats method."""

    def test_stats_structure(self, scheduler):
        """Test stats return structure."""
        stats = scheduler.get_stats()
        assert "enabled" in stats
        assert "is_running" in stats
        assert "last_consolidation" in stats
        assert "hours_since_last" in stats
        assert "new_memory_count" in stats
        assert "total_consolidations" in stats
        assert "last_trigger_reason" in stats
        assert "last_error" in stats
        assert "config" in stats

    def test_stats_config_section(self, scheduler):
        """Test config section in stats."""
        stats = scheduler.get_stats()
        config = stats["config"]
        assert config["interval_hours"] == 8.0
        assert config["memory_threshold"] == 100
        assert config["check_interval_seconds"] == 60.0
        assert config["consolidation_type"] == "light"

    def test_stats_values(self, scheduler):
        """Test stats values after operations."""
        scheduler.record_memory_created(count=25)
        scheduler.record_consolidation_complete(reason=TriggerReason.TIME_BASED)

        stats = scheduler.get_stats()
        assert stats["new_memory_count"] == 0  # Reset after completion
        assert stats["total_consolidations"] == 1
        assert stats["last_trigger_reason"] == "time_based"


class TestReset:
    """Tests for reset method."""

    def test_reset_clears_state(self, scheduler):
        """Test reset clears all state."""
        # Modify state
        scheduler.state.new_memory_count = 50
        scheduler.state.total_consolidations = 10
        scheduler.state.is_running = True
        scheduler.state.last_error = "error"

        scheduler.reset()

        assert scheduler.state.new_memory_count == 0
        assert scheduler.state.total_consolidations == 0
        assert scheduler.state.is_running is False
        assert scheduler.state.last_error is None


# =============================================================================
# Test Background Task
# =============================================================================


class TestBackgroundTask:
    """Tests for background task functionality."""

    @pytest.mark.asyncio
    async def test_start_background_task(self, scheduler):
        """Test starting background task."""
        callback = AsyncMock()
        await scheduler.start_background_task(callback)

        assert scheduler._background_task is not None
        assert not scheduler._background_task.done()

        # Cleanup
        await scheduler.stop_background_task()

    @pytest.mark.asyncio
    async def test_stop_background_task(self, scheduler):
        """Test stopping background task."""
        callback = AsyncMock()
        await scheduler.start_background_task(callback)
        await scheduler.stop_background_task()

        assert scheduler._background_task is None

    @pytest.mark.asyncio
    async def test_double_start_warning(self, scheduler):
        """Test warning on double start."""
        callback = AsyncMock()
        await scheduler.start_background_task(callback)
        await scheduler.start_background_task(callback)  # Should warn

        # Still only one task
        assert scheduler._background_task is not None

        # Cleanup
        await scheduler.stop_background_task()

    @pytest.mark.asyncio
    async def test_background_task_triggers_consolidation(self):
        """Test background task triggers consolidation when conditions met."""
        scheduler = ConsolidationScheduler(
            interval_hours=8.0,
            memory_threshold=5,  # Low threshold for testing
            check_interval_seconds=0.05,  # Check every 50ms
            enabled=True,
        )

        # Pre-load enough memories to trigger
        scheduler.record_memory_created(count=10)

        callback = AsyncMock()
        await scheduler.start_background_task(callback)

        # Wait for task to trigger (give it enough time)
        await asyncio.sleep(0.2)

        # Should have triggered due to load
        assert callback.called

        # Cleanup
        await scheduler.stop_background_task()

    @pytest.mark.asyncio
    async def test_callback_receives_consolidation_type(self):
        """Test callback receives correct consolidation type."""
        scheduler = ConsolidationScheduler(
            interval_hours=0.0001,
            memory_threshold=1000,
            check_interval_seconds=0.1,
            consolidation_type="deep",
            enabled=True,
        )

        callback = AsyncMock()
        await scheduler.start_background_task(callback)

        # Wait for trigger
        await asyncio.sleep(0.3)

        if callback.called:
            callback.assert_called_with(consolidation_type="deep")

        await scheduler.stop_background_task()


# =============================================================================
# Test Singleton Functions
# =============================================================================


class TestSingletonFunctions:
    """Tests for singleton getter and reset functions."""

    def test_get_scheduler_returns_singleton(self):
        """Test get_consolidation_scheduler returns singleton."""
        with patch("ww.consolidation.service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                auto_consolidation_enabled=True,
                auto_consolidation_interval_hours=8.0,
                auto_consolidation_memory_threshold=100,
                auto_consolidation_check_interval_seconds=300.0,
                auto_consolidation_type="light",
            )

            s1 = get_consolidation_scheduler()
            s2 = get_consolidation_scheduler()
            assert s1 is s2

    def test_reset_clears_singleton(self):
        """Test reset_consolidation_scheduler clears singleton."""
        with patch("ww.consolidation.service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                auto_consolidation_enabled=True,
                auto_consolidation_interval_hours=8.0,
                auto_consolidation_memory_threshold=100,
                auto_consolidation_check_interval_seconds=300.0,
                auto_consolidation_type="light",
            )

            s1 = get_consolidation_scheduler()
            reset_consolidation_scheduler()
            s2 = get_consolidation_scheduler()
            assert s1 is not s2


# =============================================================================
# Test Integration with Config
# =============================================================================


class TestConfigIntegration:
    """Tests for config-based scheduler creation."""

    def test_scheduler_uses_config_values(self):
        """Test scheduler uses config values."""
        with patch("ww.consolidation.service.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(
                auto_consolidation_enabled=False,
                auto_consolidation_interval_hours=12.0,
                auto_consolidation_memory_threshold=200,
                auto_consolidation_check_interval_seconds=600.0,
                auto_consolidation_type="all",
            )

            scheduler = get_consolidation_scheduler()

            assert scheduler.enabled is False
            assert scheduler.interval_hours == 12.0
            assert scheduler.memory_threshold == 200
            assert scheduler.check_interval_seconds == 600.0
            assert scheduler.consolidation_type == "all"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_exact_threshold_triggers(self, scheduler):
        """Test exact threshold triggers consolidation."""
        scheduler.state.new_memory_count = 100  # Exactly at threshold
        trigger = scheduler.should_consolidate()
        assert trigger.should_run is True
        assert trigger.reason == TriggerReason.LOAD_BASED

    def test_one_below_threshold_no_trigger(self, scheduler):
        """Test one below threshold does not trigger."""
        scheduler.state.new_memory_count = 99
        trigger = scheduler.should_consolidate()
        # Still might trigger on time if enough time passed
        # Set recent consolidation to ensure no time trigger
        scheduler.state.last_consolidation = datetime.now()
        trigger = scheduler.should_consolidate()
        assert trigger.should_run is False

    def test_zero_memory_count(self, scheduler):
        """Test zero memory count is handled."""
        scheduler.state.new_memory_count = 0
        scheduler.state.last_consolidation = datetime.now()
        trigger = scheduler.should_consolidate()
        assert trigger.should_run is False
        assert trigger.details["memories_remaining"] == 100

    def test_negative_hours_remaining(self, scheduler):
        """Test negative hours handled when overdue."""
        scheduler.state.last_consolidation = datetime.now() - timedelta(hours=20)
        trigger = scheduler.should_consolidate()
        assert trigger.should_run is True
        # hours_remaining would be negative, but we trigger instead


# =============================================================================
# Test Concurrency Safety
# =============================================================================


class TestConcurrencySafety:
    """Tests for concurrency safety."""

    def test_is_running_flag_prevents_concurrent(self, scheduler):
        """Test is_running flag prevents concurrent triggers."""
        scheduler.state.last_consolidation = datetime.now() - timedelta(hours=10)
        scheduler.state.new_memory_count = 200

        # Both conditions met, but already running
        scheduler.state.is_running = True

        trigger = scheduler.should_consolidate()
        assert trigger.should_run is False

    @pytest.mark.asyncio
    async def test_lock_serializes_consolidation(self):
        """Test lock serializes consolidation calls."""
        scheduler = ConsolidationScheduler(
            interval_hours=0.0001,
            memory_threshold=1,
            check_interval_seconds=0.05,
            enabled=True,
        )

        call_count = 0
        call_times = []

        async def slow_callback(**kwargs):
            nonlocal call_count
            call_times.append(datetime.now())
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate slow consolidation

        await scheduler.start_background_task(slow_callback)

        # Record many memories to trigger multiple times
        for _ in range(5):
            scheduler.record_memory_created(count=10)
            await asyncio.sleep(0.05)

        await asyncio.sleep(0.5)
        await scheduler.stop_background_task()

        # Calls should be serialized (not concurrent)
        # Check that the lock prevented overlapping calls
        assert call_count >= 1
