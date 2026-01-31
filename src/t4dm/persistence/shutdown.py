"""
Graceful Shutdown Handler for World Weaver

Fixes the critical signal handling issues (MCP-CRITICAL-001/002/003):
- No async operations in signal handlers
- Thread-safe shutdown flag
- No logging in signal handlers
- Proper cleanup ordering

Shutdown Protocol:
=================

    SIGTERM/SIGINT received
           │
           ▼
    ┌─────────────────────────────┐
    │  Set shutdown flag          │  ← Signal handler (sync, minimal)
    │  (atomic, no logging)       │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Main thread detects flag   │  ← Event loop checks periodically
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Stop accepting requests    │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Wait for in-flight ops     │  ← With timeout (default 30s)
    │  (drain period)             │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Force final checkpoint     │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Flush and close WAL        │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Run cleanup callbacks      │  ← In reverse registration order
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Close storage connections  │
    └─────────────┬───────────────┘
                  │
                  ▼
    ┌─────────────────────────────┐
    │  Exit(0)                    │
    └─────────────────────────────┘

Key Design Decisions:
====================
1. Signal handler ONLY sets a flag (atomic int)
2. NO logging in signal handler (deadlock risk)
3. NO async in signal handler (crash risk)
4. Cleanup happens in main thread via event loop
5. Timeout prevents infinite drain
6. Callbacks run in reverse order (LIFO)
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of shutdown process."""
    RUNNING = auto()  # Normal operation
    DRAINING = auto()  # Stopping new requests, waiting for in-flight
    CHECKPOINTING = auto()  # Creating final checkpoint
    CLEANING = auto()  # Running cleanup callbacks
    CLOSED = auto()  # Shutdown complete


@dataclass
class ShutdownConfig:
    """Shutdown configuration."""
    drain_timeout_seconds: float = 30.0  # Max time to wait for in-flight ops
    checkpoint_timeout_seconds: float = 60.0  # Max time for final checkpoint
    cleanup_timeout_seconds: float = 30.0  # Max time for cleanup callbacks
    force_exit_on_second_signal: bool = True  # SIGINT twice = immediate exit


class ShutdownManager:
    """
    Manages graceful shutdown.

    Thread-safe and signal-safe.

    Usage:
        shutdown = ShutdownManager(config)

        # Register cleanup callbacks
        shutdown.register_cleanup(wal.close, priority=100)
        shutdown.register_cleanup(checkpoint.create_final, priority=90)
        shutdown.register_cleanup(storage.close, priority=50)

        # Start listening for signals
        shutdown.install_handlers()

        # In your main loop, check for shutdown
        while not shutdown.should_shutdown:
            await process_requests()

        # When shutdown triggered, run cleanup
        await shutdown.execute_shutdown()
    """

    def __init__(self, config: ShutdownConfig | None = None):
        self.config = config or ShutdownConfig()

        # Atomic shutdown flag - signal handler only touches this
        self._shutdown_requested = threading.Event()
        self._second_signal_received = threading.Event()

        # State tracking
        self._phase = ShutdownPhase.RUNNING
        self._phase_lock = threading.Lock()

        # Cleanup callbacks: (priority, callback, is_async)
        self._cleanup_callbacks: list[tuple[int, Callable, bool]] = []
        self._callbacks_lock = threading.Lock()

        # In-flight operation tracking
        self._in_flight_count = 0
        self._in_flight_lock = threading.Lock()
        self._in_flight_zero = threading.Event()
        self._in_flight_zero.set()  # Initially zero

        # References to critical components
        self._checkpoint_fn: Callable[[], Any] | None = None
        self._wal_close_fn: Callable[[], Any] | None = None

        # Event loop reference for async cleanup
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def should_shutdown(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested.is_set()

    @property
    def phase(self) -> ShutdownPhase:
        """Current shutdown phase."""
        with self._phase_lock:
            return self._phase

    def set_checkpoint_function(self, fn: Callable[[], Any]) -> None:
        """Set the function to call for final checkpoint."""
        self._checkpoint_fn = fn

    def set_wal_close_function(self, fn: Callable[[], Any]) -> None:
        """Set the function to close WAL."""
        self._wal_close_fn = fn

    def register_cleanup(
        self,
        callback: Callable[[], Any],
        priority: int = 50,
        is_async: bool = False,
    ) -> None:
        """
        Register a cleanup callback.

        Higher priority callbacks run first.
        Set is_async=True if callback is a coroutine function.
        """
        with self._callbacks_lock:
            self._cleanup_callbacks.append((priority, callback, is_async))
            # Keep sorted by priority (descending)
            self._cleanup_callbacks.sort(key=lambda x: -x[0])

    def track_operation_start(self) -> None:
        """Call when starting an in-flight operation."""
        with self._in_flight_lock:
            self._in_flight_count += 1
            self._in_flight_zero.clear()

    def track_operation_end(self) -> None:
        """Call when completing an in-flight operation."""
        with self._in_flight_lock:
            self._in_flight_count -= 1
            if self._in_flight_count <= 0:
                self._in_flight_count = 0
                self._in_flight_zero.set()

    def install_handlers(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """
        Install signal handlers.

        Should be called from main thread.
        """
        self._loop = loop or asyncio.get_event_loop()

        # Install signal handlers
        # NOTE: These handlers are MINIMAL - they only set flags
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # On Unix, also handle SIGHUP (terminal hangup)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._signal_handler)

        logger.info("Shutdown signal handlers installed")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """
        Signal handler - MUST be minimal and synchronous.

        NO LOGGING HERE - logging uses locks, can deadlock!
        NO ASYNC HERE - will crash!
        """
        if self._shutdown_requested.is_set():
            # Second signal
            self._second_signal_received.set()
            if self.config.force_exit_on_second_signal:
                # Write directly to stderr (no logging)
                os.write(2, b"\nForced exit on second signal\n")
                os._exit(1)
        else:
            # First signal
            self._shutdown_requested.set()
            # Write directly to stderr (no logging)
            os.write(2, b"\nShutdown requested, cleaning up...\n")

    async def execute_shutdown(self) -> bool:
        """
        Execute the full shutdown sequence.

        Returns True if shutdown completed cleanly.
        """
        success = True

        # Phase 1: Draining
        logger.info("Shutdown Phase 1: Draining in-flight operations")
        with self._phase_lock:
            self._phase = ShutdownPhase.DRAINING

        # Wait for in-flight operations with timeout
        drain_start = time.time()
        while not self._in_flight_zero.is_set():
            elapsed = time.time() - drain_start
            if elapsed >= self.config.drain_timeout_seconds:
                logger.warning(
                    f"Drain timeout after {elapsed:.1f}s, "
                    f"{self._in_flight_count} operations still in flight"
                )
                success = False
                break

            await asyncio.sleep(0.1)

            # Check for force exit
            if self._second_signal_received.is_set():
                logger.warning("Force exit requested during drain")
                return False

        logger.info(f"Drain complete: {self._in_flight_count} remaining")

        # Phase 2: Checkpointing
        logger.info("Shutdown Phase 2: Creating final checkpoint")
        with self._phase_lock:
            self._phase = ShutdownPhase.CHECKPOINTING

        if self._checkpoint_fn:
            try:
                checkpoint_result = self._checkpoint_fn()
                if asyncio.iscoroutine(checkpoint_result):
                    await asyncio.wait_for(
                        checkpoint_result,
                        timeout=self.config.checkpoint_timeout_seconds,
                    )
                logger.info("Final checkpoint created")
            except TimeoutError:
                logger.error("Checkpoint timeout")
                success = False
            except Exception as e:
                logger.error(f"Checkpoint failed: {e}")
                success = False

        # Phase 3: WAL flush and close
        if self._wal_close_fn:
            try:
                wal_result = self._wal_close_fn()
                if asyncio.iscoroutine(wal_result):
                    await wal_result
                logger.info("WAL closed")
            except Exception as e:
                logger.error(f"WAL close failed: {e}")
                success = False

        # Phase 4: Cleanup callbacks
        logger.info("Shutdown Phase 3: Running cleanup callbacks")
        with self._phase_lock:
            self._phase = ShutdownPhase.CLEANING

        with self._callbacks_lock:
            callbacks = list(self._cleanup_callbacks)

        for priority, callback, is_async in callbacks:
            try:
                if is_async:
                    await asyncio.wait_for(
                        callback(),
                        timeout=self.config.cleanup_timeout_seconds / len(callbacks),
                    )
                else:
                    result = callback()
                    if asyncio.iscoroutine(result):
                        await asyncio.wait_for(
                            result,
                            timeout=self.config.cleanup_timeout_seconds / len(callbacks),
                        )
            except TimeoutError:
                logger.warning(f"Cleanup callback timeout (priority={priority})")
                success = False
            except Exception as e:
                logger.error(f"Cleanup callback failed: {e}")
                success = False

        # Phase 5: Done
        with self._phase_lock:
            self._phase = ShutdownPhase.CLOSED

        logger.info(f"Shutdown complete (success={success})")
        return success

    async def wait_for_shutdown(self) -> None:
        """
        Async wait for shutdown signal.

        Use this in your main loop:
            await shutdown.wait_for_shutdown()
            await shutdown.execute_shutdown()
        """
        while not self.should_shutdown:
            await asyncio.sleep(0.5)


def register_shutdown_handlers(
    shutdown_manager: ShutdownManager,
    loop: asyncio.AbstractEventLoop | None = None,
) -> ShutdownManager:
    """
    Convenience function to set up shutdown handling.

    Usage:
        shutdown = register_shutdown_handlers(ShutdownManager(config))
    """
    shutdown_manager.install_handlers(loop)
    return shutdown_manager


# Context manager for tracking in-flight operations
class OperationContext:
    """
    Context manager for tracking in-flight operations.

    Usage:
        async with OperationContext(shutdown_manager):
            await process_request()
    """

    def __init__(self, shutdown_manager: ShutdownManager):
        self._manager = shutdown_manager

    async def __aenter__(self) -> OperationContext:
        if self._manager.should_shutdown:
            raise RuntimeError("System is shutting down")
        self._manager.track_operation_start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._manager.track_operation_end()


# Decorator for tracking operations
def track_operation(shutdown_manager: ShutdownManager):
    """
    Decorator to track async function as in-flight operation.

    Usage:
        @track_operation(shutdown_manager)
        async def handle_request(request):
            ...
    """
    def decorator(fn):
        async def wrapper(*args, **kwargs):
            if shutdown_manager.should_shutdown:
                raise RuntimeError("System is shutting down")

            shutdown_manager.track_operation_start()
            try:
                return await fn(*args, **kwargs)
            finally:
                shutdown_manager.track_operation_end()
        return wrapper
    return decorator
