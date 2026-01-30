"""
Tests for PO-2: Cold Storage Archive System.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ww.storage.archive import (
    ArchiveConfig,
    ArchiveMetadata,
    ColdStorageManager,
    FilesystemArchive,
    get_cold_storage,
    reset_cold_storage,
)


class TestArchiveConfig:
    """Test ArchiveConfig defaults."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = ArchiveConfig()
        assert config.backend == "filesystem"
        assert config.compress is True
        assert config.archive_retention_days == 365 * 5


class TestFilesystemArchive:
    """Test filesystem archive backend."""

    @pytest.fixture
    def temp_archive(self, tmp_path):
        """Create a temporary archive."""
        config = ArchiveConfig(base_path=str(tmp_path / "archive"))
        return FilesystemArchive(config)

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, temp_archive):
        """Store and retrieve a memory."""
        memory_id = "test-memory-123"
        data = {"content": "Test content", "importance": 0.8}
        metadata = ArchiveMetadata(
            memory_id=memory_id,
            memory_type="episodes",
            archived_at=datetime.now(),
            original_created_at=datetime.now() - timedelta(days=30),
            size_bytes=0,
            archive_path="",
        )

        # Store
        success = await temp_archive.store(memory_id, data, metadata)
        assert success is True
        assert metadata.size_bytes > 0

        # Retrieve
        retrieved = await temp_archive.retrieve(memory_id)
        assert retrieved is not None
        assert retrieved["content"] == "Test content"
        assert retrieved["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_retrieve_nonexistent(self, temp_archive):
        """Retrieve returns None for nonexistent memory."""
        result = await temp_archive.retrieve("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, temp_archive):
        """Delete removes archived memory."""
        memory_id = "delete-test-123"
        data = {"content": "To be deleted"}
        metadata = ArchiveMetadata(
            memory_id=memory_id,
            memory_type="episodes",
            archived_at=datetime.now(),
            original_created_at=None,
            size_bytes=0,
            archive_path="",
        )

        await temp_archive.store(memory_id, data, metadata)
        assert await temp_archive.retrieve(memory_id) is not None

        success = await temp_archive.delete(memory_id)
        assert success is True
        assert await temp_archive.retrieve(memory_id) is None

    @pytest.mark.asyncio
    async def test_list_archived(self, temp_archive):
        """List archived memories."""
        # Store a few memories
        for i in range(3):
            metadata = ArchiveMetadata(
                memory_id=f"list-test-{i}",
                memory_type="episodes",
                archived_at=datetime.now(),
                original_created_at=None,
                size_bytes=0,
                archive_path="",
            )
            await temp_archive.store(f"list-test-{i}", {"index": i}, metadata)

        # List all
        archived = await temp_archive.list_archived()
        assert len(archived) >= 3

        # List with limit
        limited = await temp_archive.list_archived(limit=2)
        assert len(limited) == 2


class TestColdStorageManager:
    """Test ColdStorageManager."""

    @pytest.fixture
    def temp_manager(self, tmp_path):
        """Create a temporary manager."""
        reset_cold_storage()
        config = ArchiveConfig(base_path=str(tmp_path / "archive"))
        return ColdStorageManager(config)

    @pytest.mark.asyncio
    async def test_archive_memory(self, temp_manager):
        """Archive a memory."""
        metadata = await temp_manager.archive_memory(
            memory_id="mgr-test-1",
            memory_type="episodes",
            data={"content": "Manager test"},
            created_at=datetime.now() - timedelta(days=10),
        )

        assert metadata is not None
        assert metadata.memory_id == "mgr-test-1"
        assert metadata.size_bytes > 0

    @pytest.mark.asyncio
    async def test_retrieve_memory(self, temp_manager):
        """Retrieve an archived memory."""
        await temp_manager.archive_memory(
            memory_id="retrieve-test",
            memory_type="entities",
            data={"name": "Test Entity"},
        )

        data = await temp_manager.retrieve_memory("retrieve-test")
        assert data is not None
        assert data["name"] == "Test Entity"

    @pytest.mark.asyncio
    async def test_stats(self, temp_manager):
        """Stats are tracked correctly."""
        await temp_manager.archive_memory(
            memory_id="stats-test",
            memory_type="episodes",
            data={"test": True},
        )

        stats = temp_manager.get_stats()
        assert stats["total_archived"] == 1
        assert stats["bytes_archived"] > 0


class TestSingleton:
    """Test singleton pattern."""

    def test_get_cold_storage_singleton(self, tmp_path):
        """get_cold_storage returns same instance."""
        reset_cold_storage()
        config = ArchiveConfig(base_path=str(tmp_path / "archive"))
        s1 = get_cold_storage(config)
        s2 = get_cold_storage()
        assert s1 is s2

    def test_reset_cold_storage(self, tmp_path):
        """reset_cold_storage creates new instance."""
        config = ArchiveConfig(base_path=str(tmp_path / "archive"))
        s1 = get_cold_storage(config)
        reset_cold_storage()
        s2 = get_cold_storage(config)
        assert s1 is not s2
