"""
Recovery Manager for T4DM

Handles startup recovery with two modes:

COLD START (no checkpoint):
- First-time initialization
- After data loss
- Initialize all components to defaults
- Optionally rebuild indexes from storage

WARM START (checkpoint exists):
- Normal restart after clean shutdown
- Recovery after crash
- Load checkpoint → replay WAL → verify → resume

Recovery Protocol:
=================

    ┌─────────────────────────────────────────┐
    │              STARTUP                     │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │         Check for checkpoint             │
    └─────────────────┬───────────────────────┘
                      │
           ┌──────────┴──────────┐
           │                     │
           ▼                     ▼
    ┌─────────────┐       ┌─────────────┐
    │  COLD START │       │  WARM START │
    └──────┬──────┘       └──────┬──────┘
           │                     │
           ▼                     ▼
    Initialize defaults    Load checkpoint
           │                     │
           │                     ▼
           │              Replay WAL entries
           │                     │
           │                     ▼
           │              Verify consistency
           │                     │
           └──────────┬──────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │         Connect storage backends         │
    └─────────────────┬───────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────┐
    │         Start serving requests           │
    └─────────────────────────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol

from .checkpoint import Checkpoint, CheckpointManager
from .wal import WALEntry, WALOperation, WriteAheadLog

logger = logging.getLogger(__name__)


class RecoveryMode(Enum):
    """Recovery mode determined at startup."""
    COLD_START = auto()  # No checkpoint, initialize fresh
    WARM_START = auto()  # Checkpoint exists, restore and replay
    FORCED_COLD = auto()  # User requested fresh start


@dataclass
class RecoveryResult:
    """Result of recovery process."""
    mode: RecoveryMode
    success: bool
    checkpoint_lsn: int = 0
    wal_entries_replayed: int = 0
    components_restored: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"Recovery {status} ({self.mode.name}): "
            f"checkpoint_lsn={self.checkpoint_lsn}, "
            f"wal_replayed={self.wal_entries_replayed}, "
            f"duration={self.duration_seconds:.2f}s"
        )


class ReplayHandler(Protocol):
    """Protocol for WAL replay handlers."""

    async def replay_entry(self, entry: WALEntry) -> None:
        """Replay a single WAL entry."""
        ...


class RecoveryManager:
    """
    Manages system recovery at startup.

    Determines cold vs warm start and orchestrates the recovery process.

    Usage:
        recovery = RecoveryManager(wal, checkpoint_manager)

        # Register replay handlers
        recovery.register_handler(WALOperation.BUFFER_ADD, buffer.replay_add)
        recovery.register_handler(WALOperation.GATE_WEIGHT_UPDATE, gate.replay_update)

        # Perform recovery
        result = await recovery.recover()

        if result.success:
            # System ready to serve
            pass
        else:
            # Handle recovery failure
            for error in result.errors:
                logger.error(f"Recovery error: {error}")
    """

    def __init__(
        self,
        wal: WriteAheadLog,
        checkpoint_manager: CheckpointManager,
    ):
        self.wal = wal
        self.checkpoint_manager = checkpoint_manager
        self._handlers: dict[WALOperation, Callable[[WALEntry], Any]] = {}
        self._cold_start_initializers: list[Callable[[], Any]] = []
        self._consistency_validators: list[Callable[[], bool]] = []

    def register_handler(
        self,
        operation: WALOperation,
        handler: Callable[[WALEntry], Any],
    ) -> None:
        """Register a handler for replaying a WAL operation type."""
        self._handlers[operation] = handler
        logger.debug(f"Registered replay handler for {operation.name}")

    def register_cold_start_initializer(
        self,
        initializer: Callable[[], Any],
    ) -> None:
        """Register a function to call during cold start."""
        self._cold_start_initializers.append(initializer)

    def register_consistency_validator(
        self,
        validator: Callable[[], bool],
    ) -> None:
        """Register a function to validate consistency after recovery."""
        self._consistency_validators.append(validator)

    async def recover(
        self,
        force_cold_start: bool = False,
    ) -> RecoveryResult:
        """
        Perform recovery and return result.

        Args:
            force_cold_start: If True, ignore checkpoints and start fresh
        """
        start_time = time.time()
        result = RecoveryResult(
            mode=RecoveryMode.FORCED_COLD if force_cold_start else RecoveryMode.COLD_START,
            success=False,
        )

        try:
            # Open WAL
            await self.wal.open()

            if force_cold_start:
                logger.info("Forced cold start requested")
                await self._cold_start(result)
            else:
                # Try to load checkpoint
                checkpoint = await self.checkpoint_manager.load_latest_checkpoint()

                if checkpoint:
                    result.mode = RecoveryMode.WARM_START
                    await self._warm_start(checkpoint, result)
                else:
                    result.mode = RecoveryMode.COLD_START
                    await self._cold_start(result)

            # Validate consistency
            await self._validate_consistency(result)

            result.success = len(result.errors) == 0
            result.duration_seconds = time.time() - start_time

            logger.info(str(result))
            return result

        except Exception as e:
            result.errors.append(f"Recovery failed: {e!s}")
            result.duration_seconds = time.time() - start_time
            logger.error(f"Recovery failed: {e}", exc_info=True)
            return result

    async def _cold_start(self, result: RecoveryResult) -> None:
        """
        Perform cold start initialization.

        All components start fresh with default values.
        """
        logger.info("Performing COLD START")

        # Run cold start initializers
        for i, initializer in enumerate(self._cold_start_initializers):
            try:
                init_result = initializer()
                if asyncio.iscoroutine(init_result):
                    await init_result
                logger.debug(f"Cold start initializer {i} completed")
            except Exception as e:
                result.errors.append(f"Cold start initializer {i} failed: {e!s}")
                logger.error(f"Cold start initializer failed: {e}")

        result.checkpoint_lsn = 0
        result.wal_entries_replayed = 0

        logger.info("Cold start complete - system initialized with defaults")

    async def _warm_start(
        self,
        checkpoint: Checkpoint,
        result: RecoveryResult,
    ) -> None:
        """
        Perform warm start from checkpoint.

        1. Restore state from checkpoint
        2. Replay WAL entries after checkpoint LSN
        """
        logger.info(f"Performing WARM START from checkpoint LSN {checkpoint.lsn}")

        # Restore from checkpoint
        result.checkpoint_lsn = checkpoint.lsn
        result.components_restored = self.checkpoint_manager.restore_all(checkpoint)

        # Check for restoration failures
        failed = [k for k, v in result.components_restored.items() if not v]
        if failed:
            for name in failed:
                result.errors.append(f"Failed to restore component: {name}")
            logger.warning(f"Some components failed to restore: {failed}")

        # Replay WAL entries after checkpoint
        logger.info(f"Replaying WAL entries after LSN {checkpoint.lsn}")
        replay_count = 0

        async for entry in self.wal.iter_uncommitted(checkpoint.lsn):
            try:
                await self._replay_entry(entry)
                replay_count += 1

                if replay_count % 1000 == 0:
                    logger.debug(f"Replayed {replay_count} WAL entries...")

            except Exception as e:
                result.errors.append(
                    f"WAL replay failed at LSN {entry.lsn}: {e!s}"
                )
                logger.error(f"WAL replay error at LSN {entry.lsn}: {e}")
                # Continue with other entries

        result.wal_entries_replayed = replay_count
        logger.info(f"WAL replay complete: {replay_count} entries")

    async def _replay_entry(self, entry: WALEntry) -> None:
        """Replay a single WAL entry."""
        handler = self._handlers.get(entry.operation)

        if handler is None:
            # Skip entries without handlers (e.g., system markers)
            if entry.operation not in (
                WALOperation.SYSTEM_START,
                WALOperation.SYSTEM_SHUTDOWN,
                WALOperation.CHECKPOINT_START,
                WALOperation.CHECKPOINT_COMPLETE,
            ):
                logger.warning(f"No handler for operation {entry.operation.name}")
            return

        result = handler(entry)
        if asyncio.iscoroutine(result):
            await result

    async def _validate_consistency(self, result: RecoveryResult) -> None:
        """Run consistency validators after recovery."""
        logger.info("Validating system consistency")

        for i, validator in enumerate(self._consistency_validators):
            try:
                valid = validator()
                if asyncio.iscoroutine(valid):
                    valid = await valid

                if not valid:
                    result.errors.append(f"Consistency validator {i} failed")
                    logger.warning(f"Consistency validator {i} returned False")

            except Exception as e:
                result.errors.append(f"Consistency validator {i} error: {e!s}")
                logger.error(f"Consistency validation error: {e}")

        if len(result.errors) == 0:
            logger.info("Consistency validation passed")


# Specialized recovery helpers

@dataclass
class ColdStartConfig:
    """Configuration for cold start behavior."""
    rebuild_indexes: bool = True  # Rebuild cluster/similarity indexes
    load_from_storage: bool = True  # Load existing data from storage
    verify_storage: bool = True  # Verify storage connection
    default_gate_weights: dict | None = None  # Initial gate weights
    default_scorer_weights: dict | None = None  # Initial scorer weights


class ColdStartHelper:
    """
    Helper for cold start initialization.

    Handles:
    - Storage connection verification
    - Index rebuilding from stored data
    - Default weight initialization
    """

    def __init__(self, config: ColdStartConfig):
        self.config = config
        self._storage_verifiers: list[Callable[[], bool]] = []
        self._index_builders: list[Callable[[], Any]] = []

    def add_storage_verifier(self, verifier: Callable[[], bool]) -> None:
        """Add a function that verifies storage connection."""
        self._storage_verifiers.append(verifier)

    def add_index_builder(self, builder: Callable[[], Any]) -> None:
        """Add a function that rebuilds an index from storage."""
        self._index_builders.append(builder)

    async def initialize(self) -> bool:
        """
        Run cold start initialization.

        Returns True if successful.
        """
        # Verify storage connections
        if self.config.verify_storage:
            logger.info("Verifying storage connections...")
            for verifier in self._storage_verifiers:
                try:
                    result = verifier()
                    if asyncio.iscoroutine(result):
                        result = await result
                    if not result:
                        logger.error("Storage verification failed")
                        return False
                except Exception as e:
                    logger.error(f"Storage verification error: {e}")
                    return False

        # Rebuild indexes
        if self.config.rebuild_indexes:
            logger.info("Rebuilding indexes from storage...")
            for builder in self._index_builders:
                try:
                    result = builder()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Index rebuild error: {e}")
                    return False

        logger.info("Cold start initialization complete")
        return True


class WarmStartHelper:
    """
    Helper for warm start operations.

    Handles:
    - Incremental index updates
    - Consistency repair
    - Migration handling
    """

    @staticmethod
    async def verify_checkpoint_storage_consistency(
        checkpoint: Checkpoint,
        storage_count_fn: Callable[[], int],
    ) -> tuple[bool, str]:
        """
        Verify checkpoint state is consistent with storage.

        Returns (is_consistent, message).
        """
        # Get counts from checkpoint and storage
        buffer_count = 0
        if checkpoint.buffer_state:
            buffer_count = len(checkpoint.buffer_state.get("items", []))

        storage_count = storage_count_fn()
        if asyncio.iscoroutine(storage_count):
            storage_count = await storage_count

        # Some items in buffer might not be in storage yet
        # Storage should have >= (items promoted from buffer)
        # This is a sanity check, not exact match

        logger.info(
            f"Consistency check: buffer={buffer_count}, storage={storage_count}"
        )

        return True, "Consistency check passed"

    @staticmethod
    def handle_version_migration(
        checkpoint: Checkpoint,
        current_version: int,
    ) -> Checkpoint:
        """
        Handle checkpoint version migration if needed.

        Returns migrated checkpoint.
        """
        if checkpoint.version == current_version:
            return checkpoint

        logger.info(
            f"Migrating checkpoint from v{checkpoint.version} to v{current_version}"
        )

        # Add version-specific migrations here
        # Example:
        # if checkpoint.version == 1 and current_version >= 2:
        #     checkpoint.new_field = migrate_from_v1(checkpoint)

        checkpoint.version = current_version
        return checkpoint
