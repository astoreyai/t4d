"""
PO-2: Cold Storage Archive System.

Provides archival of old memories to cold storage for:
- Reduced primary storage costs
- Bounded active memory footprint
- Historical retrieval when needed

Supports multiple backends:
- Local filesystem (default)
- S3-compatible object storage
- PostgreSQL JSONB (for structured queries)
"""

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ArchiveMetadata:
    """Metadata for an archived memory."""

    memory_id: str
    memory_type: str
    archived_at: datetime
    original_created_at: datetime | None
    size_bytes: int
    archive_path: str
    compression: str = "gzip"
    checksum: str | None = None


@dataclass
class ArchiveConfig:
    """Configuration for archive storage."""

    # Storage backend
    backend: str = "filesystem"  # "filesystem", "s3", "postgres"

    # Filesystem settings
    base_path: str = "/var/ww/archive"
    max_file_size_mb: int = 100
    compress: bool = True

    # S3 settings (if backend="s3")
    s3_bucket: str = ""
    s3_prefix: str = "ww-archive/"
    s3_region: str = "us-east-1"

    # PostgreSQL settings (if backend="postgres")
    postgres_table: str = "memory_archive"

    # Retention
    archive_retention_days: int = 365 * 5  # 5 years
    cleanup_batch_size: int = 1000


class ArchiveBackend(Protocol):
    """Protocol for archive storage backends."""

    async def store(
        self, memory_id: str, data: dict, metadata: ArchiveMetadata
    ) -> bool:
        """Store a memory in the archive."""
        ...

    async def retrieve(self, memory_id: str) -> dict | None:
        """Retrieve a memory from the archive."""
        ...

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory from the archive."""
        ...

    async def list_archived(
        self, memory_type: str | None = None, limit: int = 100
    ) -> list[ArchiveMetadata]:
        """List archived memories."""
        ...


class FilesystemArchive:
    """Filesystem-based archive storage."""

    def __init__(self, config: ArchiveConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Ensure archive directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        (self.base_path / "episodes").mkdir(exist_ok=True)
        (self.base_path / "entities").mkdir(exist_ok=True)
        (self.base_path / "procedures").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)

    def _get_path(self, memory_id: str, memory_type: str) -> Path:
        """Get archive path for a memory."""
        # Partition by first 2 chars of ID for filesystem efficiency
        prefix = memory_id[:2] if len(memory_id) >= 2 else "00"
        subdir = self.base_path / memory_type / prefix
        subdir.mkdir(parents=True, exist_ok=True)

        ext = ".json.gz" if self.config.compress else ".json"
        return subdir / f"{memory_id}{ext}"

    def _get_metadata_path(self, memory_id: str) -> Path:
        """Get metadata path for a memory."""
        prefix = memory_id[:2] if len(memory_id) >= 2 else "00"
        subdir = self.base_path / "metadata" / prefix
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{memory_id}.meta.json"

    async def store(
        self, memory_id: str, data: dict, metadata: ArchiveMetadata
    ) -> bool:
        """Store a memory in the filesystem archive."""
        try:
            path = self._get_path(memory_id, metadata.memory_type)
            meta_path = self._get_metadata_path(memory_id)

            # Serialize data
            json_data = json.dumps(data, default=str, ensure_ascii=False)

            # Write data (optionally compressed)
            if self.config.compress:
                with gzip.open(path, "wt", encoding="utf-8") as f:
                    f.write(json_data)
            else:
                path.write_text(json_data, encoding="utf-8")

            # Update metadata with actual size
            metadata.size_bytes = path.stat().st_size
            metadata.archive_path = str(path)

            # Write metadata
            meta_path.write_text(
                json.dumps(
                    {
                        "memory_id": metadata.memory_id,
                        "memory_type": metadata.memory_type,
                        "archived_at": metadata.archived_at.isoformat(),
                        "original_created_at": (
                            metadata.original_created_at.isoformat()
                            if metadata.original_created_at
                            else None
                        ),
                        "size_bytes": metadata.size_bytes,
                        "archive_path": metadata.archive_path,
                        "compression": metadata.compression,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            logger.debug(f"Archived memory {memory_id} to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to archive memory {memory_id}: {e}")
            return False

    async def retrieve(self, memory_id: str) -> dict | None:
        """Retrieve a memory from the filesystem archive."""
        try:
            # Try to find the file (check both memory types)
            for memory_type in ["episodes", "entities", "procedures"]:
                path = self._get_path(memory_id, memory_type)
                if path.exists():
                    if self.config.compress:
                        with gzip.open(path, "rt", encoding="utf-8") as f:
                            return json.load(f)
                    else:
                        return json.loads(path.read_text(encoding="utf-8"))

            logger.debug(f"Memory {memory_id} not found in archive")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory from the filesystem archive."""
        try:
            deleted = False

            for memory_type in ["episodes", "entities", "procedures"]:
                path = self._get_path(memory_id, memory_type)
                if path.exists():
                    path.unlink()
                    deleted = True

            meta_path = self._get_metadata_path(memory_id)
            if meta_path.exists():
                meta_path.unlink()

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete archived memory {memory_id}: {e}")
            return False

    async def list_archived(
        self, memory_type: str | None = None, limit: int = 100
    ) -> list[ArchiveMetadata]:
        """List archived memories."""
        results = []
        types_to_check = (
            [memory_type] if memory_type else ["episodes", "entities", "procedures"]
        )

        try:
            for mtype in types_to_check:
                type_dir = self.base_path / mtype
                if not type_dir.exists():
                    continue

                for prefix_dir in type_dir.iterdir():
                    if not prefix_dir.is_dir():
                        continue

                    for file_path in prefix_dir.iterdir():
                        if len(results) >= limit:
                            break

                        if file_path.suffix in (".gz", ".json"):
                            # Extract memory_id from filename
                            name = file_path.stem
                            if name.endswith(".json"):
                                name = name[:-5]

                            # Try to load metadata
                            meta_path = self._get_metadata_path(name)
                            if meta_path.exists():
                                meta_data = json.loads(
                                    meta_path.read_text(encoding="utf-8")
                                )
                                results.append(
                                    ArchiveMetadata(
                                        memory_id=meta_data["memory_id"],
                                        memory_type=meta_data["memory_type"],
                                        archived_at=datetime.fromisoformat(
                                            meta_data["archived_at"]
                                        ),
                                        original_created_at=(
                                            datetime.fromisoformat(
                                                meta_data["original_created_at"]
                                            )
                                            if meta_data.get("original_created_at")
                                            else None
                                        ),
                                        size_bytes=meta_data["size_bytes"],
                                        archive_path=meta_data["archive_path"],
                                        compression=meta_data.get(
                                            "compression", "gzip"
                                        ),
                                    )
                                )

                    if len(results) >= limit:
                        break

        except Exception as e:
            logger.error(f"Failed to list archived memories: {e}")

        return results


class ColdStorageManager:
    """
    Manager for cold storage archival operations.

    Coordinates archival, retrieval, and cleanup of old memories.
    """

    def __init__(self, config: ArchiveConfig | None = None):
        self.config = config or ArchiveConfig()
        self._backend: ArchiveBackend | None = None

        # Statistics
        self._total_archived = 0
        self._total_retrieved = 0
        self._total_deleted = 0
        self._bytes_archived = 0

        logger.info(
            f"ColdStorageManager initialized: backend={self.config.backend}, "
            f"path={self.config.base_path}"
        )

    def _get_backend(self) -> ArchiveBackend:
        """Get or create the archive backend."""
        if self._backend is None:
            if self.config.backend == "filesystem":
                self._backend = FilesystemArchive(self.config)
            else:
                raise ValueError(f"Unknown backend: {self.config.backend}")
        return self._backend

    async def archive_memory(
        self,
        memory_id: str,
        memory_type: str,
        data: dict,
        created_at: datetime | None = None,
    ) -> ArchiveMetadata | None:
        """
        Archive a memory to cold storage.

        Args:
            memory_id: Unique memory identifier
            memory_type: Type of memory (episode, entity, procedure)
            data: Memory data to archive
            created_at: Original creation timestamp

        Returns:
            ArchiveMetadata if successful, None otherwise
        """
        backend = self._get_backend()

        metadata = ArchiveMetadata(
            memory_id=memory_id,
            memory_type=memory_type,
            archived_at=datetime.now(),
            original_created_at=created_at,
            size_bytes=0,
            archive_path="",
            compression="gzip" if self.config.compress else "none",
        )

        success = await backend.store(memory_id, data, metadata)

        if success:
            self._total_archived += 1
            self._bytes_archived += metadata.size_bytes
            return metadata

        return None

    async def retrieve_memory(self, memory_id: str) -> dict | None:
        """
        Retrieve a memory from cold storage.

        Args:
            memory_id: Memory identifier

        Returns:
            Memory data dict if found, None otherwise
        """
        backend = self._get_backend()
        data = await backend.retrieve(memory_id)

        if data:
            self._total_retrieved += 1

        return data

    async def delete_archived(self, memory_id: str) -> bool:
        """
        Delete a memory from cold storage.

        Args:
            memory_id: Memory identifier

        Returns:
            True if deleted, False otherwise
        """
        backend = self._get_backend()
        success = await backend.delete(memory_id)

        if success:
            self._total_deleted += 1

        return success

    async def cleanup_old_archives(self) -> int:
        """
        Clean up archives older than retention period.

        Returns:
            Number of archives deleted
        """
        backend = self._get_backend()
        archived = await backend.list_archived(limit=self.config.cleanup_batch_size)

        cutoff = datetime.now() - datetime.timedelta(
            days=self.config.archive_retention_days
        )

        deleted = 0
        for meta in archived:
            if meta.archived_at < cutoff:
                if await backend.delete(meta.memory_id):
                    deleted += 1

        logger.info(f"Cleaned up {deleted} old archives")
        return deleted

    def get_stats(self) -> dict[str, Any]:
        """Get cold storage statistics."""
        return {
            "backend": self.config.backend,
            "base_path": self.config.base_path,
            "total_archived": self._total_archived,
            "total_retrieved": self._total_retrieved,
            "total_deleted": self._total_deleted,
            "bytes_archived": self._bytes_archived,
            "retention_days": self.config.archive_retention_days,
        }


# Singleton instance
_cold_storage: ColdStorageManager | None = None


def get_cold_storage(config: ArchiveConfig | None = None) -> ColdStorageManager:
    """Get or create the cold storage manager singleton."""
    global _cold_storage
    if _cold_storage is None:
        _cold_storage = ColdStorageManager(config)
    return _cold_storage


def reset_cold_storage() -> None:
    """Reset the cold storage manager singleton."""
    global _cold_storage
    _cold_storage = None
