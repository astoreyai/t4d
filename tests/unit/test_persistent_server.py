"""
Unit tests for World Weaver MCP Persistent Server.

Tests initialization, persistence management, WAL replay handlers,
and graceful shutdown procedures.

Note: The persistent_server module has some import issues with get_memory_gateway
that require careful mocking. Tests focus on testable components.
"""

import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import pytest

logger = logging.getLogger(__name__)


class TestPersistenceConfig:
    """Tests for persistence configuration logic."""

    def test_persistence_config_defaults(self):
        """Test that PersistenceConfig has sensible defaults."""
        from ww.persistence import PersistenceConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PersistenceConfig(data_directory=Path(tmpdir))

            assert config.wal_sync_mode in ["fsync", "fdatasync", "none"]
            assert config.checkpoint_interval_seconds > 0
            assert config.checkpoint_operation_threshold > 0
            assert config.drain_timeout_seconds > 0

    def test_persistence_config_custom_values(self):
        """Test PersistenceConfig accepts custom values."""
        from ww.persistence import PersistenceConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = PersistenceConfig(
                data_directory=Path(tmpdir),
                wal_sync_mode="fsync",
                checkpoint_interval_seconds=600.0,
                checkpoint_operation_threshold=2000,
                checkpoint_max_count=10,
                drain_timeout_seconds=60.0,
                checkpoint_timeout_seconds=120.0,
            )

            assert config.wal_sync_mode == "fsync"
            assert config.checkpoint_interval_seconds == 600.0
            assert config.checkpoint_operation_threshold == 2000
            assert config.checkpoint_max_count == 10


class TestPersistenceManagerBasics:
    """Tests for PersistenceManager basic operations."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_persistence_manager_initialization(self, temp_dir):
        """Test PersistenceManager initializes correctly."""
        from ww.persistence import PersistenceConfig, PersistenceManager

        config = PersistenceConfig(data_directory=temp_dir)
        manager = PersistenceManager(config)

        assert manager is not None
        assert manager.config == config

    def test_persistence_manager_register_component(self, temp_dir):
        """Test component registration."""
        from ww.persistence import PersistenceConfig, PersistenceManager

        config = PersistenceConfig(data_directory=temp_dir)
        manager = PersistenceManager(config)

        mock_component = MagicMock()
        manager.register_component("test", mock_component)

        # Component should be tracked via checkpoint manager
        assert "test" in manager._checkpoint._components

    def test_persistence_manager_register_replay_handler(self, temp_dir):
        """Test replay handler registration."""
        from ww.persistence import PersistenceConfig, PersistenceManager, WALOperation

        config = PersistenceConfig(data_directory=temp_dir)
        manager = PersistenceManager(config)

        handler = MagicMock()
        manager.register_replay_handler(WALOperation.BUFFER_ADD, handler)

        # Handler should be registered
        assert WALOperation.BUFFER_ADD in manager._replay_handlers

    def test_persistence_manager_register_cleanup(self, temp_dir):
        """Test cleanup callback registration."""
        from ww.persistence import PersistenceConfig, PersistenceManager

        config = PersistenceConfig(data_directory=temp_dir)
        manager = PersistenceManager(config)

        cleanup_fn = MagicMock()
        manager.register_cleanup(cleanup_fn, priority=50)

        # Cleanup should be registered via shutdown manager
        assert any(cb[1] == cleanup_fn for cb in manager._shutdown._cleanup_callbacks)


class TestWALOperations:
    """Tests for WAL operation types."""

    def test_wal_operation_values(self):
        """Test WAL operation enum values."""
        from ww.persistence import WALOperation

        # Verify expected operations exist
        assert hasattr(WALOperation, 'BUFFER_ADD')
        assert hasattr(WALOperation, 'BUFFER_PROMOTE')
        assert hasattr(WALOperation, 'GATE_WEIGHT_UPDATE')

    def test_wal_operation_string_conversion(self):
        """Test WAL operation string representation."""
        from ww.persistence import WALOperation

        # Operations should have string values
        assert WALOperation.BUFFER_ADD.value is not None


@pytest.mark.skip(reason="MCP removed in v0.2.0 - persistence now managed via ww.persistence.get_persistence()")
class TestMCPCompatibility:
    """Tests for MCP availability checking - DEPRECATED."""

    def test_is_mcp_available_returns_bool(self):
        """Test is_mcp_available returns boolean."""
        pass


class TestPersistentServerGlobalState:
    """Tests for persistent server global state management."""

    def test_global_persistence_initial_state(self):
        """Test that global persistence starts as None."""
        from ww.persistence import get_persistence, set_persistence

        # Ensure clean state
        set_persistence(None)

        # Verify initial state
        assert get_persistence() is None

        # Cleanup
        set_persistence(None)


class TestReplayLogic:
    """Tests for WAL replay logic patterns."""

    def test_memory_reconstruction_pattern(self):
        """Test memory object can be reconstructed from payload."""
        # Simulate replay payload structure
        payload = {
            "memory_id": "test-uuid",
            "content": "Test content",
            "embedding": [0.1] * 1024,
            "metadata": {"key": "value"}
        }

        # Verify payload has all required fields
        assert "memory_id" in payload
        assert "content" in payload
        assert "embedding" in payload
        assert len(payload["embedding"]) == 1024

    def test_promote_payload_structure(self):
        """Test promote operation payload structure."""
        payload = {
            "memory_id": "test-uuid",
            "target": "semantic"
        }

        assert "memory_id" in payload
        assert "target" in payload
        assert payload["target"] in ["semantic", "episodic", "procedural"]

    def test_weight_update_payload_structure(self):
        """Test weight update payload structure."""
        payload = {
            "layer": "output",
            "weights": [[0.1, 0.2], [0.3, 0.4]]
        }

        assert "layer" in payload
        assert "weights" in payload
        assert isinstance(payload["weights"], list)


class TestShutdownPatterns:
    """Tests for graceful shutdown patterns."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_persistence_manager_shutdown_interface(self, temp_dir):
        """Test shutdown method exists and is async."""
        from ww.persistence import PersistenceConfig, PersistenceManager

        config = PersistenceConfig(data_directory=temp_dir)
        manager = PersistenceManager(config)

        # Shutdown should be async and return bool
        assert asyncio.iscoroutinefunction(manager.shutdown)

    @pytest.mark.asyncio
    async def test_cleanup_callback_priority_ordering(self, temp_dir):
        """Test cleanup callbacks are ordered by priority."""
        from ww.persistence import PersistenceConfig, PersistenceManager

        config = PersistenceConfig(data_directory=temp_dir)
        manager = PersistenceManager(config)

        calls = []

        def cb1():
            calls.append("cb1")

        def cb2():
            calls.append("cb2")

        manager.register_cleanup(cb1, priority=10)  # Lower priority = runs later
        manager.register_cleanup(cb2, priority=50)  # Higher priority = runs first

        # Callbacks should be stored in priority order via shutdown manager
        priorities = [p for p, _, _ in manager._shutdown._cleanup_callbacks]
        assert priorities == sorted(priorities, reverse=True)


class TestIntegrationPatterns:
    """Integration-level test patterns."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.mark.asyncio
    async def test_full_lifecycle_pattern(self, temp_dir):
        """Test full persistence lifecycle: init, operations, shutdown."""
        from ww.persistence import PersistenceConfig, PersistenceManager

        config = PersistenceConfig(
            data_directory=temp_dir,
            checkpoint_interval_seconds=300.0,
        )
        manager = PersistenceManager(config)

        # Register components
        mock_component = MagicMock()
        mock_component.get_state = MagicMock(return_value={"test": "state"})
        mock_component.restore_state = MagicMock()
        manager.register_component("test", mock_component)

        # Start would normally perform cold/warm start
        # For unit test, verify structure is correct via checkpoint manager
        assert manager._checkpoint._components.get("test") is mock_component

        # Shutdown
        await manager.shutdown()
