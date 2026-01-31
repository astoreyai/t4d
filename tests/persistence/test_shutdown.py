"""
Tests for persistence/shutdown.py module.

Tests the ShutdownManager, ShutdownConfig, and related utilities
for graceful shutdown handling.
"""

import asyncio
import signal
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from t4dm.persistence.shutdown import (
    ShutdownPhase,
    ShutdownConfig,
    ShutdownManager,
    OperationContext,
    track_operation,
    register_shutdown_handlers,
)


# =============================================================================
# Test ShutdownPhase Enum
# =============================================================================


class TestShutdownPhase:
    """Tests for ShutdownPhase enum."""

    def test_running_phase(self):
        """Test RUNNING phase exists."""
        assert ShutdownPhase.RUNNING is not None
        assert ShutdownPhase.RUNNING.name == "RUNNING"

    def test_draining_phase(self):
        """Test DRAINING phase exists."""
        assert ShutdownPhase.DRAINING is not None
        assert ShutdownPhase.DRAINING.name == "DRAINING"

    def test_checkpointing_phase(self):
        """Test CHECKPOINTING phase exists."""
        assert ShutdownPhase.CHECKPOINTING is not None
        assert ShutdownPhase.CHECKPOINTING.name == "CHECKPOINTING"

    def test_cleaning_phase(self):
        """Test CLEANING phase exists."""
        assert ShutdownPhase.CLEANING is not None
        assert ShutdownPhase.CLEANING.name == "CLEANING"

    def test_closed_phase(self):
        """Test CLOSED phase exists."""
        assert ShutdownPhase.CLOSED is not None
        assert ShutdownPhase.CLOSED.name == "CLOSED"


# =============================================================================
# Test ShutdownConfig
# =============================================================================


class TestShutdownConfig:
    """Tests for ShutdownConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = ShutdownConfig()
        assert config.drain_timeout_seconds == 30.0
        assert config.checkpoint_timeout_seconds == 60.0
        assert config.cleanup_timeout_seconds == 30.0
        assert config.force_exit_on_second_signal is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = ShutdownConfig(
            drain_timeout_seconds=10.0,
            checkpoint_timeout_seconds=30.0,
            cleanup_timeout_seconds=15.0,
            force_exit_on_second_signal=False,
        )
        assert config.drain_timeout_seconds == 10.0
        assert config.checkpoint_timeout_seconds == 30.0
        assert config.cleanup_timeout_seconds == 15.0
        assert config.force_exit_on_second_signal is False


# =============================================================================
# Test ShutdownManager Initialization
# =============================================================================


class TestShutdownManagerInit:
    """Tests for ShutdownManager initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        manager = ShutdownManager()
        assert manager.config is not None
        assert manager.phase == ShutdownPhase.RUNNING
        assert manager.should_shutdown is False

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = ShutdownConfig(drain_timeout_seconds=5.0)
        manager = ShutdownManager(config)
        assert manager.config.drain_timeout_seconds == 5.0


# =============================================================================
# Test ShutdownManager Properties
# =============================================================================


class TestShutdownManagerProperties:
    """Tests for ShutdownManager properties."""

    def test_should_shutdown_initially_false(self):
        """Test should_shutdown is False initially."""
        manager = ShutdownManager()
        assert manager.should_shutdown is False

    def test_should_shutdown_after_request(self):
        """Test should_shutdown is True after request."""
        manager = ShutdownManager()
        manager._shutdown_requested.set()
        assert manager.should_shutdown is True

    def test_phase_property(self):
        """Test phase property is thread-safe."""
        manager = ShutdownManager()
        assert manager.phase == ShutdownPhase.RUNNING


# =============================================================================
# Test In-Flight Operation Tracking
# =============================================================================


class TestInFlightOperations:
    """Tests for in-flight operation tracking."""

    def test_track_operation_start(self):
        """Test tracking operation start."""
        manager = ShutdownManager()
        manager.track_operation_start()
        assert manager._in_flight_count == 1
        assert not manager._in_flight_zero.is_set()

    def test_track_operation_end(self):
        """Test tracking operation end."""
        manager = ShutdownManager()
        manager.track_operation_start()
        manager.track_operation_end()
        assert manager._in_flight_count == 0
        assert manager._in_flight_zero.is_set()

    def test_multiple_operations(self):
        """Test tracking multiple operations."""
        manager = ShutdownManager()
        manager.track_operation_start()
        manager.track_operation_start()
        manager.track_operation_start()
        assert manager._in_flight_count == 3

        manager.track_operation_end()
        assert manager._in_flight_count == 2

        manager.track_operation_end()
        manager.track_operation_end()
        assert manager._in_flight_count == 0
        assert manager._in_flight_zero.is_set()

    def test_operation_end_clamps_to_zero(self):
        """Test that end doesn't go negative."""
        manager = ShutdownManager()
        manager.track_operation_end()
        manager.track_operation_end()
        assert manager._in_flight_count == 0


# =============================================================================
# Test Cleanup Callback Registration
# =============================================================================


class TestCleanupCallbacks:
    """Tests for cleanup callback registration."""

    def test_register_cleanup(self):
        """Test registering cleanup callback."""
        manager = ShutdownManager()
        callback = MagicMock()
        manager.register_cleanup(callback)
        assert len(manager._cleanup_callbacks) == 1

    def test_register_cleanup_with_priority(self):
        """Test registering with priority."""
        manager = ShutdownManager()
        low_priority = MagicMock()
        high_priority = MagicMock()

        manager.register_cleanup(low_priority, priority=10)
        manager.register_cleanup(high_priority, priority=90)

        # Should be sorted by priority descending
        assert manager._cleanup_callbacks[0][0] == 90
        assert manager._cleanup_callbacks[1][0] == 10

    def test_register_async_cleanup(self):
        """Test registering async cleanup callback."""
        manager = ShutdownManager()
        async_callback = AsyncMock()
        manager.register_cleanup(async_callback, is_async=True)

        assert len(manager._cleanup_callbacks) == 1
        assert manager._cleanup_callbacks[0][2] is True


# =============================================================================
# Test Checkpoint and WAL Functions
# =============================================================================


class TestFunctionRegistration:
    """Tests for checkpoint and WAL function registration."""

    def test_set_checkpoint_function(self):
        """Test setting checkpoint function."""
        manager = ShutdownManager()
        fn = MagicMock()
        manager.set_checkpoint_function(fn)
        assert manager._checkpoint_fn == fn

    def test_set_wal_close_function(self):
        """Test setting WAL close function."""
        manager = ShutdownManager()
        fn = MagicMock()
        manager.set_wal_close_function(fn)
        assert manager._wal_close_fn == fn


# =============================================================================
# Test Execute Shutdown
# =============================================================================


class TestExecuteShutdown:
    """Tests for execute_shutdown method."""

    @pytest.mark.asyncio
    async def test_execute_shutdown_success(self):
        """Test successful shutdown execution."""
        manager = ShutdownManager()
        success = await manager.execute_shutdown()

        assert success is True
        assert manager.phase == ShutdownPhase.CLOSED

    @pytest.mark.asyncio
    async def test_execute_shutdown_with_checkpoint(self):
        """Test shutdown with checkpoint function."""
        manager = ShutdownManager()
        checkpoint_fn = MagicMock()
        manager.set_checkpoint_function(checkpoint_fn)

        await manager.execute_shutdown()

        checkpoint_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_shutdown_with_async_checkpoint(self):
        """Test shutdown with async checkpoint function."""
        manager = ShutdownManager()
        checkpoint_fn = AsyncMock()
        manager.set_checkpoint_function(checkpoint_fn)

        await manager.execute_shutdown()

        checkpoint_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_shutdown_with_wal_close(self):
        """Test shutdown with WAL close function."""
        manager = ShutdownManager()
        wal_close = MagicMock()
        manager.set_wal_close_function(wal_close)

        await manager.execute_shutdown()

        wal_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_shutdown_runs_callbacks(self):
        """Test shutdown runs cleanup callbacks."""
        manager = ShutdownManager()
        callback1 = MagicMock()
        callback2 = MagicMock()

        manager.register_cleanup(callback1, priority=50)
        manager.register_cleanup(callback2, priority=60)

        await manager.execute_shutdown()

        callback1.assert_called_once()
        callback2.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_shutdown_handles_callback_error(self):
        """Test shutdown handles callback errors."""
        manager = ShutdownManager()
        failing_callback = MagicMock(side_effect=RuntimeError("Callback failed"))
        manager.register_cleanup(failing_callback)

        success = await manager.execute_shutdown()

        # Should complete but report failure
        assert success is False

    @pytest.mark.asyncio
    async def test_execute_shutdown_waits_for_in_flight(self):
        """Test shutdown waits for in-flight operations."""
        manager = ShutdownManager(ShutdownConfig(drain_timeout_seconds=2.0))
        manager.track_operation_start()

        # End operation after a short delay
        async def end_operation():
            await asyncio.sleep(0.1)
            manager.track_operation_end()

        asyncio.create_task(end_operation())

        success = await manager.execute_shutdown()

        assert success is True
        assert manager._in_flight_count == 0

    @pytest.mark.asyncio
    async def test_execute_shutdown_drain_timeout(self):
        """Test shutdown timeout on drain."""
        manager = ShutdownManager(ShutdownConfig(drain_timeout_seconds=0.1))
        manager.track_operation_start()  # Never ends

        success = await manager.execute_shutdown()

        # Should complete with failure
        assert success is False

    @pytest.mark.asyncio
    async def test_shutdown_phases_progress(self):
        """Test shutdown phases progress correctly."""
        manager = ShutdownManager()
        phases_seen = []

        original_phase_lock = manager._phase_lock

        # Track phases during shutdown
        async def track_phases():
            for _ in range(10):
                phases_seen.append(manager.phase)
                await asyncio.sleep(0.01)

        asyncio.create_task(track_phases())
        await manager.execute_shutdown()

        # Final phase should be CLOSED
        assert manager.phase == ShutdownPhase.CLOSED


# =============================================================================
# Test Wait for Shutdown
# =============================================================================


class TestWaitForShutdown:
    """Tests for wait_for_shutdown method."""

    @pytest.mark.asyncio
    async def test_wait_for_shutdown_returns_on_signal(self):
        """Test wait_for_shutdown returns when signal received."""
        manager = ShutdownManager()

        async def trigger_shutdown():
            await asyncio.sleep(0.1)
            manager._shutdown_requested.set()

        asyncio.create_task(trigger_shutdown())

        # Should return without hanging
        await asyncio.wait_for(
            manager.wait_for_shutdown(),
            timeout=1.0,
        )


# =============================================================================
# Test OperationContext
# =============================================================================


class TestOperationContext:
    """Tests for OperationContext context manager."""

    @pytest.mark.asyncio
    async def test_context_tracks_operation(self):
        """Test context manager tracks operation."""
        manager = ShutdownManager()

        async with OperationContext(manager):
            assert manager._in_flight_count == 1

        assert manager._in_flight_count == 0

    @pytest.mark.asyncio
    async def test_context_tracks_on_exception(self):
        """Test context manager tracks even on exception."""
        manager = ShutdownManager()

        with pytest.raises(ValueError):
            async with OperationContext(manager):
                assert manager._in_flight_count == 1
                raise ValueError("Test error")

        assert manager._in_flight_count == 0

    @pytest.mark.asyncio
    async def test_context_rejects_during_shutdown(self):
        """Test context manager rejects during shutdown."""
        manager = ShutdownManager()
        manager._shutdown_requested.set()

        with pytest.raises(RuntimeError, match="shutting down"):
            async with OperationContext(manager):
                pass


# =============================================================================
# Test track_operation Decorator
# =============================================================================


class TestTrackOperationDecorator:
    """Tests for track_operation decorator."""

    @pytest.mark.asyncio
    async def test_decorator_tracks_operation(self):
        """Test decorator tracks operation."""
        manager = ShutdownManager()

        @track_operation(manager)
        async def do_work():
            assert manager._in_flight_count == 1
            return "done"

        result = await do_work()

        assert result == "done"
        assert manager._in_flight_count == 0

    @pytest.mark.asyncio
    async def test_decorator_tracks_on_exception(self):
        """Test decorator tracks even on exception."""
        manager = ShutdownManager()

        @track_operation(manager)
        async def failing_work():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await failing_work()

        assert manager._in_flight_count == 0

    @pytest.mark.asyncio
    async def test_decorator_rejects_during_shutdown(self):
        """Test decorator rejects during shutdown."""
        manager = ShutdownManager()
        manager._shutdown_requested.set()

        @track_operation(manager)
        async def do_work():
            return "done"

        with pytest.raises(RuntimeError, match="shutting down"):
            await do_work()


# =============================================================================
# Test register_shutdown_handlers Helper
# =============================================================================


class TestRegisterShutdownHandlers:
    """Tests for register_shutdown_handlers function."""

    def test_returns_manager(self):
        """Test function returns the manager."""
        manager = ShutdownManager()
        with patch.object(manager, 'install_handlers'):
            result = register_shutdown_handlers(manager)
        assert result is manager

    def test_installs_handlers(self):
        """Test function installs handlers."""
        manager = ShutdownManager()
        with patch.object(manager, 'install_handlers') as mock_install:
            register_shutdown_handlers(manager)
        mock_install.assert_called_once()


# =============================================================================
# Test Signal Handler (without actually sending signals)
# =============================================================================


class TestSignalHandler:
    """Tests for signal handler behavior."""

    def test_first_signal_sets_flag(self):
        """Test first signal sets shutdown flag."""
        manager = ShutdownManager()

        # Simulate signal handler call
        with patch('os.write'):
            manager._signal_handler(signal.SIGTERM, None)

        assert manager.should_shutdown is True

    def test_second_signal_sets_second_flag(self):
        """Test second signal sets second signal flag."""
        manager = ShutdownManager(ShutdownConfig(force_exit_on_second_signal=False))
        manager._shutdown_requested.set()

        with patch('os.write'):
            manager._signal_handler(signal.SIGTERM, None)

        assert manager._second_signal_received.is_set()

    def test_second_signal_force_exit(self):
        """Test second signal with force exit enabled."""
        manager = ShutdownManager(ShutdownConfig(force_exit_on_second_signal=True))
        manager._shutdown_requested.set()

        with patch('os.write'), patch('os._exit') as mock_exit:
            manager._signal_handler(signal.SIGTERM, None)

        mock_exit.assert_called_once_with(1)
