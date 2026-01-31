"""
Tests for persistence/recovery.py module.

Tests the RecoveryManager and ColdStartHelper classes that handle
startup recovery with cold and warm start modes.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from t4dm.persistence.recovery import (
    RecoveryMode,
    RecoveryResult,
    RecoveryManager,
    ColdStartConfig,
    ColdStartHelper,
)
from t4dm.persistence.checkpoint import Checkpoint
from time import time
from t4dm.persistence.wal import WALEntry, WALOperation


# =============================================================================
# Test RecoveryMode Enum
# =============================================================================


class TestRecoveryMode:
    """Tests for RecoveryMode enum."""

    def test_cold_start_mode(self):
        """Test COLD_START mode exists."""
        assert RecoveryMode.COLD_START is not None
        assert RecoveryMode.COLD_START.name == "COLD_START"

    def test_warm_start_mode(self):
        """Test WARM_START mode exists."""
        assert RecoveryMode.WARM_START is not None
        assert RecoveryMode.WARM_START.name == "WARM_START"

    def test_forced_cold_mode(self):
        """Test FORCED_COLD mode exists."""
        assert RecoveryMode.FORCED_COLD is not None
        assert RecoveryMode.FORCED_COLD.name == "FORCED_COLD"


# =============================================================================
# Test RecoveryResult Dataclass
# =============================================================================


class TestRecoveryResult:
    """Tests for RecoveryResult dataclass."""

    def test_default_values(self):
        """Test default initialization."""
        result = RecoveryResult(
            mode=RecoveryMode.COLD_START,
            success=True,
        )
        assert result.checkpoint_lsn == 0
        assert result.wal_entries_replayed == 0
        assert result.components_restored == {}
        assert result.errors == []
        assert result.duration_seconds == 0.0

    def test_full_initialization(self):
        """Test full initialization with all fields."""
        result = RecoveryResult(
            mode=RecoveryMode.WARM_START,
            success=True,
            checkpoint_lsn=1000,
            wal_entries_replayed=50,
            components_restored={"gate": True, "scorer": True},
            errors=[],
            duration_seconds=2.5,
        )
        assert result.mode == RecoveryMode.WARM_START
        assert result.checkpoint_lsn == 1000
        assert result.wal_entries_replayed == 50
        assert len(result.components_restored) == 2

    def test_str_success(self):
        """Test string representation for success."""
        result = RecoveryResult(
            mode=RecoveryMode.COLD_START,
            success=True,
            duration_seconds=1.5,
        )
        s = str(result)
        assert "SUCCESS" in s
        assert "COLD_START" in s

    def test_str_failure(self):
        """Test string representation for failure."""
        result = RecoveryResult(
            mode=RecoveryMode.WARM_START,
            success=False,
            errors=["error1"],
        )
        s = str(result)
        assert "FAILED" in s


# =============================================================================
# Test RecoveryManager
# =============================================================================


@pytest.fixture
def mock_wal():
    """Create a mock WAL."""
    wal = MagicMock()
    wal.open = AsyncMock()
    wal.close = AsyncMock()
    wal.iter_uncommitted = MagicMock(return_value=AsyncIteratorMock([]))
    return wal


@pytest.fixture
def mock_checkpoint_manager():
    """Create a mock checkpoint manager."""
    manager = MagicMock()
    manager.load_latest_checkpoint = AsyncMock(return_value=None)
    manager.restore_all = MagicMock(return_value={})
    return manager


@pytest.fixture
def recovery_manager(mock_wal, mock_checkpoint_manager):
    """Create a RecoveryManager with mocks."""
    return RecoveryManager(
        wal=mock_wal,
        checkpoint_manager=mock_checkpoint_manager,
    )


class AsyncIteratorMock:
    """Mock async iterator for WAL entries."""
    def __init__(self, items):
        self.items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.items)
        except StopIteration:
            raise StopAsyncIteration


class TestRecoveryManager:
    """Tests for RecoveryManager class."""

    @pytest.mark.asyncio
    async def test_cold_start_no_checkpoint(self, recovery_manager, mock_checkpoint_manager):
        """Test cold start when no checkpoint exists."""
        mock_checkpoint_manager.load_latest_checkpoint = AsyncMock(return_value=None)

        result = await recovery_manager.recover()

        assert result.success is True
        assert result.mode == RecoveryMode.COLD_START
        assert result.checkpoint_lsn == 0
        assert result.wal_entries_replayed == 0

    @pytest.mark.asyncio
    async def test_forced_cold_start(self, recovery_manager):
        """Test forced cold start ignores checkpoint."""
        result = await recovery_manager.recover(force_cold_start=True)

        assert result.success is True
        assert result.mode == RecoveryMode.FORCED_COLD

    @pytest.mark.asyncio
    async def test_warm_start_with_checkpoint(self, recovery_manager, mock_checkpoint_manager, mock_wal):
        """Test warm start when checkpoint exists."""
        checkpoint = Checkpoint(
            lsn=1000,
            timestamp=time(),
        )
        mock_checkpoint_manager.load_latest_checkpoint = AsyncMock(return_value=checkpoint)
        mock_checkpoint_manager.restore_all = MagicMock(return_value={"gate": True})
        mock_wal.iter_uncommitted = MagicMock(return_value=AsyncIteratorMock([]))

        result = await recovery_manager.recover()

        assert result.success is True
        assert result.mode == RecoveryMode.WARM_START
        assert result.checkpoint_lsn == 1000

    @pytest.mark.asyncio
    async def test_register_handler(self, recovery_manager):
        """Test registering a replay handler."""
        handler = MagicMock()
        recovery_manager.register_handler(WALOperation.BUFFER_ADD, handler)

        assert WALOperation.BUFFER_ADD in recovery_manager._handlers

    @pytest.mark.asyncio
    async def test_register_cold_start_initializer(self, recovery_manager):
        """Test registering cold start initializer."""
        initializer = MagicMock()
        recovery_manager.register_cold_start_initializer(initializer)

        result = await recovery_manager.recover()

        initializer.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_cold_start_initializer(self, recovery_manager):
        """Test async cold start initializer."""
        async_init = AsyncMock()
        recovery_manager.register_cold_start_initializer(async_init)

        result = await recovery_manager.recover()

        async_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_initializer_error_captured(self, recovery_manager):
        """Test that initializer errors are captured."""
        def failing_init():
            raise ValueError("Init failed")

        recovery_manager.register_cold_start_initializer(failing_init)

        result = await recovery_manager.recover()

        assert len(result.errors) > 0
        assert "Init failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_register_consistency_validator(self, recovery_manager):
        """Test registering consistency validator."""
        validator = MagicMock(return_value=True)
        recovery_manager.register_consistency_validator(validator)

        result = await recovery_manager.recover()

        validator.assert_called_once()
        assert result.success is True

    @pytest.mark.asyncio
    async def test_consistency_validator_failure(self, recovery_manager):
        """Test consistency validator failure."""
        validator = MagicMock(return_value=False)
        recovery_manager.register_consistency_validator(validator)

        result = await recovery_manager.recover()

        assert result.success is False
        assert any("validator" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_async_consistency_validator(self, recovery_manager):
        """Test async consistency validator."""
        async_validator = AsyncMock(return_value=True)
        recovery_manager.register_consistency_validator(async_validator)

        result = await recovery_manager.recover()

        async_validator.assert_called_once()

    @pytest.mark.asyncio
    async def test_wal_replay(self, recovery_manager, mock_checkpoint_manager, mock_wal):
        """Test WAL entry replay during warm start."""
        checkpoint = Checkpoint(lsn=100, timestamp=time())
        mock_checkpoint_manager.load_latest_checkpoint = AsyncMock(return_value=checkpoint)
        mock_checkpoint_manager.restore_all = MagicMock(return_value={})

        # Create WAL entries
        entries = [
            WALEntry(lsn=101, timestamp_ns=int(time() * 1e9), operation=WALOperation.BUFFER_ADD, payload={}),
            WALEntry(lsn=102, timestamp_ns=int(time() * 1e9), operation=WALOperation.BUFFER_ADD, payload={}),
        ]
        mock_wal.iter_uncommitted = MagicMock(return_value=AsyncIteratorMock(entries))

        # Register handler
        handler = MagicMock()
        recovery_manager.register_handler(WALOperation.BUFFER_ADD, handler)

        result = await recovery_manager.recover()

        assert result.wal_entries_replayed == 2
        assert handler.call_count == 2

    @pytest.mark.asyncio
    async def test_wal_replay_async_handler(self, recovery_manager, mock_checkpoint_manager, mock_wal):
        """Test WAL replay with async handler."""
        checkpoint = Checkpoint(lsn=100, timestamp=time())
        mock_checkpoint_manager.load_latest_checkpoint = AsyncMock(return_value=checkpoint)
        mock_checkpoint_manager.restore_all = MagicMock(return_value={})

        entries = [
            WALEntry(lsn=101, timestamp_ns=int(time() * 1e9), operation=WALOperation.BUFFER_ADD, payload={}),
        ]
        mock_wal.iter_uncommitted = MagicMock(return_value=AsyncIteratorMock(entries))

        async_handler = AsyncMock()
        recovery_manager.register_handler(WALOperation.BUFFER_ADD, async_handler)

        result = await recovery_manager.recover()

        async_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_wal_replay_error_continues(self, recovery_manager, mock_checkpoint_manager, mock_wal):
        """Test WAL replay continues after error."""
        checkpoint = Checkpoint(lsn=100, timestamp=time())
        mock_checkpoint_manager.load_latest_checkpoint = AsyncMock(return_value=checkpoint)
        mock_checkpoint_manager.restore_all = MagicMock(return_value={})

        entries = [
            WALEntry(lsn=101, timestamp_ns=int(time() * 1e9), operation=WALOperation.BUFFER_ADD, payload={}),
            WALEntry(lsn=102, timestamp_ns=int(time() * 1e9), operation=WALOperation.BUFFER_ADD, payload={}),
        ]
        mock_wal.iter_uncommitted = MagicMock(return_value=AsyncIteratorMock(entries))

        # Handler that fails on first call
        call_count = [0]
        def failing_handler(entry):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Replay error")

        recovery_manager.register_handler(WALOperation.BUFFER_ADD, failing_handler)

        result = await recovery_manager.recover()

        # Both entries attempted
        assert call_count[0] == 2
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_recovery_duration_tracked(self, recovery_manager):
        """Test that recovery duration is tracked."""
        result = await recovery_manager.recover()

        assert result.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_recovery_exception_captured(self, recovery_manager, mock_wal):
        """Test recovery exception is captured."""
        mock_wal.open = AsyncMock(side_effect=RuntimeError("WAL open failed"))

        result = await recovery_manager.recover()

        assert result.success is False
        assert any("WAL open failed" in e for e in result.errors)


# =============================================================================
# Test ColdStartConfig
# =============================================================================


class TestColdStartConfig:
    """Tests for ColdStartConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = ColdStartConfig()
        assert config.rebuild_indexes is True
        assert config.load_from_storage is True
        assert config.verify_storage is True
        assert config.default_gate_weights is None
        assert config.default_scorer_weights is None

    def test_custom_values(self):
        """Test custom configuration."""
        config = ColdStartConfig(
            rebuild_indexes=False,
            load_from_storage=False,
            verify_storage=False,
            default_gate_weights={"w1": 0.5},
        )
        assert config.rebuild_indexes is False
        assert config.default_gate_weights == {"w1": 0.5}


# =============================================================================
# Test ColdStartHelper
# =============================================================================


class TestColdStartHelper:
    """Tests for ColdStartHelper class."""

    @pytest.fixture
    def helper(self):
        """Create a ColdStartHelper with default config."""
        return ColdStartHelper(ColdStartConfig())

    @pytest.mark.asyncio
    async def test_initialize_empty(self, helper):
        """Test initialization with no verifiers or builders."""
        result = await helper.initialize()
        assert result is True

    @pytest.mark.asyncio
    async def test_storage_verifier_success(self, helper):
        """Test successful storage verification."""
        verifier = MagicMock(return_value=True)
        helper.add_storage_verifier(verifier)

        result = await helper.initialize()

        verifier.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_storage_verifier_failure(self, helper):
        """Test storage verification failure."""
        verifier = MagicMock(return_value=False)
        helper.add_storage_verifier(verifier)

        result = await helper.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_storage_verifier_exception(self, helper):
        """Test storage verification exception."""
        verifier = MagicMock(side_effect=RuntimeError("Connection failed"))
        helper.add_storage_verifier(verifier)

        result = await helper.initialize()

        assert result is False

    @pytest.mark.asyncio
    async def test_async_storage_verifier(self, helper):
        """Test async storage verifier."""
        async_verifier = AsyncMock(return_value=True)
        helper.add_storage_verifier(async_verifier)

        result = await helper.initialize()

        async_verifier.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_index_builder(self, helper):
        """Test index builder is called."""
        builder = MagicMock()
        helper.add_index_builder(builder)

        result = await helper.initialize()

        builder.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_storage_verification(self):
        """Test skipping storage verification."""
        config = ColdStartConfig(verify_storage=False)
        helper = ColdStartHelper(config)

        # Even a failing verifier shouldn't be called
        verifier = MagicMock(return_value=False)
        helper.add_storage_verifier(verifier)

        result = await helper.initialize()

        verifier.assert_not_called()
        assert result is True

    @pytest.mark.asyncio
    async def test_skip_index_rebuild(self):
        """Test skipping index rebuild."""
        config = ColdStartConfig(rebuild_indexes=False)
        helper = ColdStartHelper(config)

        builder = MagicMock()
        helper.add_index_builder(builder)

        result = await helper.initialize()

        builder.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_verifiers(self, helper):
        """Test multiple storage verifiers."""
        v1 = MagicMock(return_value=True)
        v2 = MagicMock(return_value=True)
        helper.add_storage_verifier(v1)
        helper.add_storage_verifier(v2)

        result = await helper.initialize()

        v1.assert_called_once()
        v2.assert_called_once()
        assert result is True

    @pytest.mark.asyncio
    async def test_verifier_order(self, helper):
        """Test verifiers run in order."""
        order = []
        def v1():
            order.append(1)
            return True
        def v2():
            order.append(2)
            return True

        helper.add_storage_verifier(v1)
        helper.add_storage_verifier(v2)

        await helper.initialize()

        assert order == [1, 2]
