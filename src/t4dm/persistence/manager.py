"""
Persistence Manager - Main Orchestrator for T4DM Durability

Coordinates WAL, checkpointing, recovery, and shutdown into a unified system.

This is the primary interface for the persistence layer.

Usage:
======

    from t4dm.persistence import PersistenceManager, PersistenceConfig

    # Create manager
    config = PersistenceConfig(
        data_directory=Path("/var/lib/ww/data"),
        checkpoint_interval=300,  # 5 minutes
    )
    persistence = PersistenceManager(config)

    # Register components to persist
    persistence.register_component("buffer", buffer_manager)
    persistence.register_component("gate", learned_gate)
    persistence.register_component("scorer", relevance_scorer)

    # Start system (handles cold/warm start)
    result = await persistence.start()
    if not result.success:
        raise RuntimeError(f"Recovery failed: {result.errors}")

    # In your operation handlers, log to WAL
    async def add_memory(memory):
        # Log BEFORE applying
        lsn = await persistence.log(WALOperation.BUFFER_ADD, {
            "memory_id": memory.id,
            "content": memory.content,
            "embedding": memory.embedding.tolist(),
        })

        # Then apply to in-memory state
        buffer_manager.add(memory)

        return lsn

    # Shutdown properly
    await persistence.shutdown()

Architecture:
============

    ┌─────────────────────────────────────────────────────────────────┐
    │                     PersistenceManager                          │
    │                                                                 │
    │  ┌──────────┐  ┌────────────────┐  ┌──────────────────────┐   │
    │  │   WAL    │  │  Checkpoint    │  │     Recovery         │   │
    │  │          │  │    Manager     │  │     Manager          │   │
    │  └────┬─────┘  └───────┬────────┘  └──────────┬───────────┘   │
    │       │                │                      │               │
    │       └────────────────┼──────────────────────┘               │
    │                        │                                       │
    │              ┌─────────┴─────────┐                             │
    │              │  Shutdown Manager │                             │
    │              └───────────────────┘                             │
    │                                                                 │
    │  Components:                                                    │
    │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │
    │  │ Buffer │ │  Gate  │ │ Scorer │ │ Traces │ │Neuromod│      │
    │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘      │
    └─────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint import Checkpoint, CheckpointableComponent, CheckpointConfig, CheckpointManager
from .recovery import RecoveryManager, RecoveryResult
from .shutdown import OperationContext, ShutdownConfig, ShutdownManager
from .wal import WALConfig, WALEntry, WALOperation, WriteAheadLog

logger = logging.getLogger(__name__)


@dataclass
class PersistenceConfig:
    """Configuration for the persistence layer."""
    # Base directory for all persistence data
    data_directory: Path

    # WAL settings
    wal_sync_mode: str = "fsync"  # "fsync", "fdatasync", "none"
    wal_segment_max_size: int = 64 * 1024 * 1024  # 64MB
    wal_max_segments: int = 100

    # Checkpoint settings
    checkpoint_interval_seconds: float = 300.0  # 5 minutes
    checkpoint_operation_threshold: int = 1000
    checkpoint_max_count: int = 5
    checkpoint_compression: bool = True

    # Shutdown settings
    drain_timeout_seconds: float = 30.0
    checkpoint_timeout_seconds: float = 60.0

    # Recovery settings
    verify_consistency: bool = True


class PersistenceManager:
    """
    Main orchestrator for T4DM persistence.

    Provides unified interface for:
    - Write-ahead logging
    - Checkpointing
    - Recovery (cold/warm start)
    - Graceful shutdown
    """

    def __init__(self, config: PersistenceConfig):
        self.config = config
        self._started = False

        # Create sub-managers
        self._wal = WriteAheadLog(WALConfig(
            directory=config.data_directory / "wal",
            segment_max_size=config.wal_segment_max_size,
            sync_mode=config.wal_sync_mode,
            max_segments=config.wal_max_segments,
        ))

        self._checkpoint = CheckpointManager(CheckpointConfig(
            directory=config.data_directory / "checkpoints",
            interval_seconds=config.checkpoint_interval_seconds,
            operation_threshold=config.checkpoint_operation_threshold,
            max_checkpoints=config.checkpoint_max_count,
            compression=config.checkpoint_compression,
        ))

        self._recovery = RecoveryManager(self._wal, self._checkpoint)

        self._shutdown = ShutdownManager(ShutdownConfig(
            drain_timeout_seconds=config.drain_timeout_seconds,
            checkpoint_timeout_seconds=config.checkpoint_timeout_seconds,
        ))

        # Wire up components
        self._checkpoint.set_lsn_provider(lambda: self._wal.current_lsn)
        self._shutdown.set_checkpoint_function(self._create_final_checkpoint)
        self._shutdown.set_wal_close_function(self._wal.close)

        # Replay handlers
        self._replay_handlers: dict[WALOperation, Callable[[WALEntry], Any]] = {}

    # ==================== Component Registration ====================

    def register_component(
        self,
        name: str,
        component: CheckpointableComponent,
    ) -> None:
        """
        Register a component for checkpointing.

        Component must implement:
        - get_checkpoint_state() -> dict
        - restore_from_checkpoint(state: dict) -> None
        """
        self._checkpoint.register_component(name, component)
        logger.info(f"Registered persistence component: {name}")

    def register_replay_handler(
        self,
        operation: WALOperation,
        handler: Callable[[WALEntry], Any],
    ) -> None:
        """
        Register a handler for replaying WAL entries during recovery.

        Handler receives WALEntry and should replay the operation.
        """
        self._replay_handlers[operation] = handler
        self._recovery.register_handler(operation, handler)
        logger.debug(f"Registered replay handler: {operation.name}")

    def register_cold_start_initializer(
        self,
        initializer: Callable[[], Any],
    ) -> None:
        """
        Register a function to call during cold start.

        Called when no checkpoint exists (first run or data loss).
        """
        self._recovery.register_cold_start_initializer(initializer)

    def register_cleanup(
        self,
        callback: Callable[[], Any],
        priority: int = 50,
    ) -> None:
        """
        Register a cleanup callback for shutdown.

        Higher priority callbacks run first.
        """
        self._shutdown.register_cleanup(callback, priority)

    # ==================== Lifecycle ====================

    async def start(
        self,
        force_cold_start: bool = False,
    ) -> RecoveryResult:
        """
        Start the persistence system.

        Performs cold or warm start as appropriate.
        Returns recovery result with success status.
        """
        if self._started:
            raise RuntimeError("PersistenceManager already started")

        logger.info("Starting persistence manager")

        # Ensure directories exist
        self.config.data_directory.mkdir(parents=True, exist_ok=True)
        (self.config.data_directory / "wal").mkdir(exist_ok=True)
        (self.config.data_directory / "checkpoints").mkdir(exist_ok=True)

        # Perform recovery
        result = await self._recovery.recover(force_cold_start)

        if not result.success:
            logger.error(f"Recovery failed: {result.errors}")
            return result

        # Start checkpoint manager
        await self._checkpoint.start()

        # Install shutdown handlers
        self._shutdown.install_handlers()

        self._started = True
        logger.info(
            f"Persistence manager started ({result.mode.name}), "
            f"current LSN: {self._wal.current_lsn}"
        )

        return result

    async def shutdown(self) -> bool:
        """
        Gracefully shut down the persistence system.

        Creates final checkpoint and closes WAL.
        Returns True if shutdown completed cleanly.
        """
        if not self._started:
            return True

        logger.info("Shutting down persistence manager")

        # Request shutdown
        self._shutdown._shutdown_requested.set()

        # Execute shutdown sequence
        success = await self._shutdown.execute_shutdown()

        # Stop checkpoint manager
        await self._checkpoint.stop()

        self._started = False

        return success

    # ==================== WAL Operations ====================

    async def log(
        self,
        operation: WALOperation,
        payload: dict[str, Any],
    ) -> int:
        """
        Log an operation to WAL.

        Returns the assigned LSN.
        This MUST be called BEFORE applying the operation to in-memory state.
        """
        if not self._started:
            raise RuntimeError("PersistenceManager not started")

        if self._shutdown.should_shutdown:
            raise RuntimeError("System is shutting down")

        lsn = await self._wal.append(operation, payload)

        # Track for checkpoint threshold
        self._checkpoint.record_operation()

        return lsn

    async def log_batch(
        self,
        operations: list[tuple[WALOperation, dict[str, Any]]],
    ) -> list[int]:
        """
        Log multiple operations to WAL.

        Returns list of assigned LSNs.
        """
        lsns = []
        for operation, payload in operations:
            lsn = await self.log(operation, payload)
            lsns.append(lsn)
        return lsns

    # ==================== Checkpoint Operations ====================

    async def create_checkpoint(self) -> Checkpoint:
        """
        Create a checkpoint immediately.

        Called automatically based on config, but can be triggered manually.
        """
        return await self._checkpoint.create_checkpoint(self._wal.current_lsn)

    async def _create_final_checkpoint(self) -> Checkpoint:
        """Create final checkpoint during shutdown."""
        return await self._checkpoint.create_checkpoint(self._wal.current_lsn)

    # ==================== Status ====================

    @property
    def current_lsn(self) -> int:
        """Current WAL LSN."""
        return self._wal.current_lsn

    @property
    def last_checkpoint_lsn(self) -> int:
        """LSN of last checkpoint."""
        return self._checkpoint.last_checkpoint_lsn

    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown.should_shutdown

    @property
    def is_started(self) -> bool:
        """Check if persistence manager is running."""
        return self._started

    def operation_context(self) -> OperationContext:
        """
        Get context manager for tracking in-flight operations.

        Usage:
            async with persistence.operation_context():
                await process_request()
        """
        return OperationContext(self._shutdown)

    # ==================== Utilities ====================

    async def truncate_wal(self) -> int:
        """
        Truncate WAL entries before last checkpoint.

        Returns number of segments removed.
        """
        checkpoint_lsn = self._checkpoint.last_checkpoint_lsn
        if checkpoint_lsn == 0:
            return 0
        return await self._wal.truncate_before(checkpoint_lsn)

    def get_status(self) -> dict[str, Any]:
        """Get persistence system status."""
        return {
            "started": self._started,
            "current_lsn": self._wal.current_lsn if self._started else 0,
            "last_checkpoint_lsn": self._checkpoint.last_checkpoint_lsn,
            "shutdown_requested": self._shutdown.should_shutdown,
            "shutdown_phase": self._shutdown.phase.name,
            "wal_directory": str(self.config.data_directory / "wal"),
            "checkpoint_directory": str(self.config.data_directory / "checkpoints"),
        }


# ==================== Convenience Functions ====================

def create_persistence_manager(
    data_directory: Path | str,
    **kwargs,
) -> PersistenceManager:
    """
    Create a persistence manager with sensible defaults.

    Usage:
        persistence = create_persistence_manager("/var/lib/ww/data")
        await persistence.start()
    """
    config = PersistenceConfig(
        data_directory=Path(data_directory),
        **kwargs,
    )
    return PersistenceManager(config)


# ==================== Integration Helpers ====================

class PersistentBuffer:
    """
    Example wrapper showing how to integrate persistence with BufferManager.

    This pattern should be applied to all components that need durability.
    """

    def __init__(
        self,
        buffer_manager: Any,  # BufferManager
        persistence: PersistenceManager,
    ):
        self._buffer = buffer_manager
        self._persistence = persistence

        # Register for checkpointing
        persistence.register_component("buffer", buffer_manager)

        # Register replay handlers
        persistence.register_replay_handler(
            WALOperation.BUFFER_ADD,
            self._replay_add,
        )
        persistence.register_replay_handler(
            WALOperation.BUFFER_REMOVE,
            self._replay_remove,
        )
        persistence.register_replay_handler(
            WALOperation.BUFFER_PROMOTE,
            self._replay_promote,
        )

    async def add(self, memory: Any) -> int:
        """Add memory with WAL logging."""
        # Log FIRST
        lsn = await self._persistence.log(
            WALOperation.BUFFER_ADD,
            {
                "memory_id": memory.id,
                "content": memory.content,
                "embedding": memory.embedding.tolist() if hasattr(memory.embedding, "tolist") else memory.embedding,
                "timestamp": memory.timestamp,
                "metadata": memory.metadata,
            }
        )

        # Then apply
        self._buffer.add(memory)

        return lsn

    async def remove(self, memory_id: str) -> int:
        """Remove memory with WAL logging."""
        lsn = await self._persistence.log(
            WALOperation.BUFFER_REMOVE,
            {"memory_id": memory_id}
        )

        self._buffer.remove(memory_id)

        return lsn

    async def promote(self, memory_id: str, target: str) -> int:
        """Promote memory with WAL logging."""
        lsn = await self._persistence.log(
            WALOperation.BUFFER_PROMOTE,
            {"memory_id": memory_id, "target": target}
        )

        self._buffer.promote(memory_id, target)

        return lsn

    def _replay_add(self, entry: WALEntry) -> None:
        """Replay add operation during recovery."""
        # Reconstruct memory from payload
        from t4dm.memory import Memory  # Import here to avoid circular deps

        memory = Memory(
            id=entry.payload["memory_id"],
            content=entry.payload["content"],
            embedding=entry.payload["embedding"],
            timestamp=entry.payload.get("timestamp"),
            metadata=entry.payload.get("metadata", {}),
        )
        self._buffer.add(memory)

    def _replay_remove(self, entry: WALEntry) -> None:
        """Replay remove operation during recovery."""
        self._buffer.remove(entry.payload["memory_id"])

    def _replay_promote(self, entry: WALEntry) -> None:
        """Replay promote operation during recovery."""
        self._buffer.promote(
            entry.payload["memory_id"],
            entry.payload["target"],
        )


# =============================================================================
# Global Persistence Instance Management
# =============================================================================

_global_persistence: PersistenceManager | None = None


def get_persistence() -> PersistenceManager | None:
    """
    Get the global persistence manager instance.

    Returns:
        The global PersistenceManager, or None if not initialized
    """
    return _global_persistence


def set_persistence(manager: PersistenceManager | None) -> None:
    """
    Set the global persistence manager instance.

    Args:
        manager: The PersistenceManager instance to use globally, or None to clear
    """
    global _global_persistence
    _global_persistence = manager
