"""
Tests for persistence/manager.py module.

Tests the PersistenceManager orchestrator and related utilities.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from t4dm.persistence.manager import (
    PersistenceConfig,
    PersistenceManager,
    create_persistence_manager,
    PersistentBuffer,
)
from t4dm.persistence.recovery import RecoveryMode, RecoveryResult
from t4dm.persistence.shutdown import ShutdownPhase
from t4dm.persistence.wal import WALOperation


# =============================================================================
# Test PersistenceConfig
# =============================================================================


class TestPersistenceConfig:
    """Tests for PersistenceConfig dataclass."""

    def test_default_values(self, tmp_path):
        """Test default configuration."""
        config = PersistenceConfig(data_directory=tmp_path)
        assert config.wal_sync_mode == "fsync"
        assert config.wal_segment_max_size == 64 * 1024 * 1024
        assert config.wal_max_segments == 100
        assert config.checkpoint_interval_seconds == 300.0
        assert config.checkpoint_operation_threshold == 1000
        assert config.checkpoint_max_count == 5
        assert config.checkpoint_compression is True
        assert config.drain_timeout_seconds == 30.0
        assert config.checkpoint_timeout_seconds == 60.0
        assert config.verify_consistency is True

    def test_custom_values(self, tmp_path):
        """Test custom configuration."""
        config = PersistenceConfig(
            data_directory=tmp_path,
            wal_sync_mode="none",
            checkpoint_interval_seconds=60.0,
            drain_timeout_seconds=10.0,
        )
        assert config.wal_sync_mode == "none"
        assert config.checkpoint_interval_seconds == 60.0
        assert config.drain_timeout_seconds == 10.0


# =============================================================================
# Test PersistenceManager Initialization
# =============================================================================


class TestPersistenceManagerInit:
    """Tests for PersistenceManager initialization."""

    def test_initialization(self, tmp_path):
        """Test basic initialization."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        assert manager.config == config
        assert manager._started is False
        assert manager._wal is not None
        assert manager._checkpoint is not None
        assert manager._recovery is not None
        assert manager._shutdown is not None

    def test_is_started_property(self, tmp_path):
        """Test is_started property."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        assert manager.is_started is False


# =============================================================================
# Test Component Registration
# =============================================================================


class TestComponentRegistration:
    """Tests for component registration."""

    def test_register_component(self, tmp_path):
        """Test registering a component."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        component = MagicMock()
        component.get_checkpoint_state = MagicMock(return_value={})
        component.restore_from_checkpoint = MagicMock()

        manager.register_component("test_component", component)
        # Should not raise

    def test_register_replay_handler(self, tmp_path):
        """Test registering a replay handler."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        handler = MagicMock()
        manager.register_replay_handler(WALOperation.BUFFER_ADD, handler)

        assert WALOperation.BUFFER_ADD in manager._replay_handlers

    def test_register_cold_start_initializer(self, tmp_path):
        """Test registering cold start initializer."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        initializer = MagicMock()
        manager.register_cold_start_initializer(initializer)
        # Should not raise

    def test_register_cleanup(self, tmp_path):
        """Test registering cleanup callback."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        cleanup = MagicMock()
        manager.register_cleanup(cleanup, priority=80)
        # Should not raise


# =============================================================================
# Test Lifecycle - Start
# =============================================================================


class TestPersistenceManagerStart:
    """Tests for PersistenceManager start."""

    @pytest.mark.asyncio
    async def test_start_creates_directories(self, tmp_path):
        """Test start creates necessary directories."""
        config = PersistenceConfig(data_directory=tmp_path / "test_data")
        manager = PersistenceManager(config)

        # Mock recovery
        with patch.object(manager._recovery, 'recover', new_callable=AsyncMock) as mock_recover:
            mock_recover.return_value = RecoveryResult(
                mode=RecoveryMode.COLD_START,
                success=True,
            )
            with patch.object(manager._checkpoint, 'start', new_callable=AsyncMock):
                with patch.object(manager._shutdown, 'install_handlers'):
                    result = await manager.start()

        assert (tmp_path / "test_data").exists()
        assert (tmp_path / "test_data" / "wal").exists()
        assert (tmp_path / "test_data" / "checkpoints").exists()

    @pytest.mark.asyncio
    async def test_start_returns_recovery_result(self, tmp_path):
        """Test start returns recovery result."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        expected_result = RecoveryResult(
            mode=RecoveryMode.COLD_START,
            success=True,
        )

        with patch.object(manager._recovery, 'recover', new_callable=AsyncMock) as mock_recover:
            mock_recover.return_value = expected_result
            with patch.object(manager._checkpoint, 'start', new_callable=AsyncMock):
                with patch.object(manager._shutdown, 'install_handlers'):
                    result = await manager.start()

        assert result.success is True
        assert result.mode == RecoveryMode.COLD_START

    @pytest.mark.asyncio
    async def test_start_marks_started(self, tmp_path):
        """Test start sets started flag."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        with patch.object(manager._recovery, 'recover', new_callable=AsyncMock) as mock_recover:
            mock_recover.return_value = RecoveryResult(
                mode=RecoveryMode.COLD_START,
                success=True,
            )
            with patch.object(manager._checkpoint, 'start', new_callable=AsyncMock):
                with patch.object(manager._shutdown, 'install_handlers'):
                    await manager.start()

        assert manager._started is True

    @pytest.mark.asyncio
    async def test_start_fails_if_already_started(self, tmp_path):
        """Test start raises if already started."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)
        manager._started = True

        with pytest.raises(RuntimeError, match="already started"):
            await manager.start()

    @pytest.mark.asyncio
    async def test_start_force_cold_start(self, tmp_path):
        """Test start with force cold start."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        with patch.object(manager._recovery, 'recover', new_callable=AsyncMock) as mock_recover:
            mock_recover.return_value = RecoveryResult(
                mode=RecoveryMode.FORCED_COLD,
                success=True,
            )
            with patch.object(manager._checkpoint, 'start', new_callable=AsyncMock):
                with patch.object(manager._shutdown, 'install_handlers'):
                    result = await manager.start(force_cold_start=True)

        mock_recover.assert_called_with(True)

    @pytest.mark.asyncio
    async def test_start_returns_failed_recovery(self, tmp_path):
        """Test start returns failure if recovery fails."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        with patch.object(manager._recovery, 'recover', new_callable=AsyncMock) as mock_recover:
            mock_recover.return_value = RecoveryResult(
                mode=RecoveryMode.WARM_START,
                success=False,
                errors=["Recovery failed"],
            )
            result = await manager.start()

        assert result.success is False
        assert manager._started is False


# =============================================================================
# Test Lifecycle - Shutdown
# =============================================================================


class TestPersistenceManagerShutdown:
    """Tests for PersistenceManager shutdown."""

    @pytest.mark.asyncio
    async def test_shutdown_not_started(self, tmp_path):
        """Test shutdown when not started returns True."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        result = await manager.shutdown()
        assert result is True

    @pytest.mark.asyncio
    async def test_shutdown_executes(self, tmp_path):
        """Test shutdown executes shutdown sequence."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)
        manager._started = True

        with patch.object(manager._shutdown, 'execute_shutdown', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = True
            with patch.object(manager._checkpoint, 'stop', new_callable=AsyncMock):
                result = await manager.shutdown()

        assert result is True
        assert manager._started is False

    @pytest.mark.asyncio
    async def test_shutdown_stops_checkpoint(self, tmp_path):
        """Test shutdown stops checkpoint manager."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)
        manager._started = True

        with patch.object(manager._shutdown, 'execute_shutdown', new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = True
            with patch.object(manager._checkpoint, 'stop', new_callable=AsyncMock) as mock_stop:
                await manager.shutdown()

        mock_stop.assert_called_once()


# =============================================================================
# Test WAL Operations
# =============================================================================


class TestWALOperations:
    """Tests for WAL operations."""

    @pytest.mark.asyncio
    async def test_log_requires_started(self, tmp_path):
        """Test log raises if not started."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        with pytest.raises(RuntimeError, match="not started"):
            await manager.log(WALOperation.BUFFER_ADD, {})

    @pytest.mark.asyncio
    async def test_log_rejects_during_shutdown(self, tmp_path):
        """Test log raises during shutdown."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)
        manager._started = True
        manager._shutdown._shutdown_requested.set()

        with pytest.raises(RuntimeError, match="shutting down"):
            await manager.log(WALOperation.BUFFER_ADD, {})

    @pytest.mark.asyncio
    async def test_log_appends_to_wal(self, tmp_path):
        """Test log appends to WAL."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)
        manager._started = True

        with patch.object(manager._wal, 'append', new_callable=AsyncMock) as mock_append:
            mock_append.return_value = 42
            with patch.object(manager._checkpoint, 'record_operation'):
                lsn = await manager.log(WALOperation.BUFFER_ADD, {"key": "value"})

        assert lsn == 42
        mock_append.assert_called_once_with(WALOperation.BUFFER_ADD, {"key": "value"})

    @pytest.mark.asyncio
    async def test_log_batch(self, tmp_path):
        """Test log_batch logs multiple operations."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)
        manager._started = True

        call_count = [0]

        async def mock_append(op, payload):
            call_count[0] += 1
            return call_count[0]

        with patch.object(manager._wal, 'append', side_effect=mock_append):
            with patch.object(manager._checkpoint, 'record_operation'):
                lsns = await manager.log_batch([
                    (WALOperation.BUFFER_ADD, {"id": "1"}),
                    (WALOperation.BUFFER_ADD, {"id": "2"}),
                    (WALOperation.BUFFER_REMOVE, {"id": "3"}),
                ])

        assert lsns == [1, 2, 3]


# =============================================================================
# Test Checkpoint Operations
# =============================================================================


class TestCheckpointOperations:
    """Tests for checkpoint operations."""

    @pytest.mark.asyncio
    async def test_create_checkpoint(self, tmp_path):
        """Test create_checkpoint calls checkpoint manager."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        mock_checkpoint = MagicMock()
        with patch.object(manager._checkpoint, 'create_checkpoint', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_checkpoint
            manager._wal._current_lsn = 100  # Set internal attribute
            result = await manager.create_checkpoint()

        mock_create.assert_called_once()


# =============================================================================
# Test Status Properties
# =============================================================================


class TestStatusProperties:
    """Tests for status properties."""

    def test_current_lsn(self, tmp_path):
        """Test current_lsn property."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        manager._wal._current_lsn = 500  # Set internal attribute
        assert manager.current_lsn == 500

    def test_last_checkpoint_lsn(self, tmp_path):
        """Test last_checkpoint_lsn property."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        manager._checkpoint._last_checkpoint_lsn = 400  # Set internal attribute
        assert manager.last_checkpoint_lsn == 400

    def test_should_shutdown(self, tmp_path):
        """Test should_shutdown property."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        assert manager.should_shutdown is False
        manager._shutdown._shutdown_requested.set()
        assert manager.should_shutdown is True

    def test_operation_context(self, tmp_path):
        """Test operation_context returns context."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        ctx = manager.operation_context()
        assert ctx is not None

    def test_get_status(self, tmp_path):
        """Test get_status returns status dict."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        manager._wal._current_lsn = 100  # Set internal attribute
        status = manager.get_status()

        assert "started" in status
        assert "shutdown_requested" in status
        assert "shutdown_phase" in status
        assert "wal_directory" in status
        assert "checkpoint_directory" in status


# =============================================================================
# Test Utilities
# =============================================================================


class TestUtilities:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_truncate_wal_no_checkpoint(self, tmp_path):
        """Test truncate_wal with no checkpoint returns 0."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        manager._checkpoint._last_checkpoint_lsn = 0  # Set internal attribute
        count = await manager.truncate_wal()

        assert count == 0

    @pytest.mark.asyncio
    async def test_truncate_wal_with_checkpoint(self, tmp_path):
        """Test truncate_wal with checkpoint."""
        config = PersistenceConfig(data_directory=tmp_path)
        manager = PersistenceManager(config)

        manager._checkpoint._last_checkpoint_lsn = 500  # Set internal attribute
        with patch.object(manager._wal, 'truncate_before', new_callable=AsyncMock) as mock_truncate:
            mock_truncate.return_value = 3
            count = await manager.truncate_wal()

        assert count == 3
        mock_truncate.assert_called_once_with(500)


# =============================================================================
# Test create_persistence_manager Factory
# =============================================================================


class TestCreatePersistenceManager:
    """Tests for create_persistence_manager factory."""

    def test_create_with_path_string(self, tmp_path):
        """Test factory with string path."""
        manager = create_persistence_manager(str(tmp_path / "data"))
        assert manager.config.data_directory == tmp_path / "data"

    def test_create_with_path_object(self, tmp_path):
        """Test factory with Path object."""
        manager = create_persistence_manager(tmp_path / "data")
        assert manager.config.data_directory == tmp_path / "data"

    def test_create_with_kwargs(self, tmp_path):
        """Test factory with additional kwargs."""
        manager = create_persistence_manager(
            tmp_path,
            checkpoint_interval_seconds=60.0,
            drain_timeout_seconds=5.0,
        )
        assert manager.config.checkpoint_interval_seconds == 60.0
        assert manager.config.drain_timeout_seconds == 5.0


# =============================================================================
# Test PersistentBuffer Integration Helper
# =============================================================================


class TestPersistentBuffer:
    """Tests for PersistentBuffer integration helper."""

    def test_initialization(self, tmp_path):
        """Test PersistentBuffer initialization."""
        config = PersistenceConfig(data_directory=tmp_path)
        persistence = PersistenceManager(config)

        buffer_manager = MagicMock()
        buffer_manager.get_checkpoint_state = MagicMock(return_value={})
        buffer_manager.restore_from_checkpoint = MagicMock()

        pb = PersistentBuffer(buffer_manager, persistence)

        assert pb._buffer == buffer_manager
        assert pb._persistence == persistence

    def test_registers_handlers(self, tmp_path):
        """Test PersistentBuffer registers replay handlers."""
        config = PersistenceConfig(data_directory=tmp_path)
        persistence = PersistenceManager(config)

        buffer_manager = MagicMock()
        buffer_manager.get_checkpoint_state = MagicMock(return_value={})
        buffer_manager.restore_from_checkpoint = MagicMock()

        PersistentBuffer(buffer_manager, persistence)

        assert WALOperation.BUFFER_ADD in persistence._replay_handlers
        assert WALOperation.BUFFER_REMOVE in persistence._replay_handlers
        assert WALOperation.BUFFER_PROMOTE in persistence._replay_handlers

    @pytest.mark.asyncio
    async def test_add_logs_first(self, tmp_path):
        """Test add logs to WAL before applying."""
        config = PersistenceConfig(data_directory=tmp_path)
        persistence = PersistenceManager(config)
        persistence._started = True

        buffer_manager = MagicMock()
        buffer_manager.get_checkpoint_state = MagicMock(return_value={})
        buffer_manager.restore_from_checkpoint = MagicMock()
        buffer_manager.add = MagicMock()

        pb = PersistentBuffer(buffer_manager, persistence)

        memory = MagicMock()
        memory.id = "mem-1"
        memory.content = "test content"
        memory.embedding = [0.1, 0.2, 0.3]
        memory.timestamp = 12345
        memory.metadata = {}

        with patch.object(persistence, 'log', new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 10
            lsn = await pb.add(memory)

        assert lsn == 10
        mock_log.assert_called_once()
        buffer_manager.add.assert_called_once_with(memory)

    @pytest.mark.asyncio
    async def test_remove_logs_first(self, tmp_path):
        """Test remove logs to WAL before applying."""
        config = PersistenceConfig(data_directory=tmp_path)
        persistence = PersistenceManager(config)
        persistence._started = True

        buffer_manager = MagicMock()
        buffer_manager.get_checkpoint_state = MagicMock(return_value={})
        buffer_manager.restore_from_checkpoint = MagicMock()
        buffer_manager.remove = MagicMock()

        pb = PersistentBuffer(buffer_manager, persistence)

        with patch.object(persistence, 'log', new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 11
            lsn = await pb.remove("mem-1")

        assert lsn == 11
        buffer_manager.remove.assert_called_once_with("mem-1")

    @pytest.mark.asyncio
    async def test_promote_logs_first(self, tmp_path):
        """Test promote logs to WAL before applying."""
        config = PersistenceConfig(data_directory=tmp_path)
        persistence = PersistenceManager(config)
        persistence._started = True

        buffer_manager = MagicMock()
        buffer_manager.get_checkpoint_state = MagicMock(return_value={})
        buffer_manager.restore_from_checkpoint = MagicMock()
        buffer_manager.promote = MagicMock()

        pb = PersistentBuffer(buffer_manager, persistence)

        with patch.object(persistence, 'log', new_callable=AsyncMock) as mock_log:
            mock_log.return_value = 12
            lsn = await pb.promote("mem-1", "semantic")

        assert lsn == 12
        buffer_manager.promote.assert_called_once_with("mem-1", "semantic")
