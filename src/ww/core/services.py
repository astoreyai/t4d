"""
World Weaver Memory Services Management.

Thread-safe initialization and cleanup of memory services.
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict

from ww.core.config import get_settings
from ww.memory.episodic import get_episodic_memory
from ww.memory.procedural import get_procedural_memory
from ww.memory.semantic import get_semantic_memory
from ww.storage.neo4j_store import close_neo4j_store
from ww.storage.qdrant_store import close_qdrant_store

logger = logging.getLogger(__name__)


# =============================================================================
# Rate Limiting
# =============================================================================


class RateLimiter:
    """
    Rate limiter to prevent DoS attacks.

    Thread-safe rate limiting per session ID with sliding window.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max = max_requests
        self.window = window_seconds
        self.requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def allow(self, session_id: str) -> bool:
        """
        Check if request is allowed for session.

        Args:
            session_id: Session identifier

        Returns:
            True if request is allowed, False if rate limited
        """
        with self._lock:
            now = time.time()
            # Remove expired timestamps
            self.requests[session_id] = [
                t for t in self.requests[session_id]
                if now - t < self.window
            ]
            # Check limit
            if len(self.requests[session_id]) >= self.max:
                return False
            # Add timestamp
            self.requests[session_id].append(now)
            return True

    def reset(self, session_id: str = None):
        """
        Reset rate limit for session(s).

        Args:
            session_id: Specific session to reset, or None for all
        """
        with self._lock:
            if session_id:
                self.requests.pop(session_id, None)
            else:
                self.requests.clear()

    def time_until_allowed(self, session_id: str) -> float:
        """
        Calculate seconds until next request allowed.

        Args:
            session_id: Session identifier

        Returns:
            Seconds until next request allowed
        """
        with self._lock:
            now = time.time()
            # Clean expired
            self.requests[session_id] = [
                t for t in self.requests[session_id]
                if now - t < self.window
            ]
            if len(self.requests[session_id]) < self.max:
                return 0.0
            # Oldest timestamp will expire first
            oldest = min(self.requests[session_id])
            return max(0.0, self.window - (now - oldest))


# =============================================================================
# Service Management
# =============================================================================

# Thread-safe service management
_service_instances: dict[str, dict] = {}
_init_lock: asyncio.Lock | None = None
_init_lock_guard = threading.Lock()  # Guards creation of async lock
_initialized_sessions: set[str] = set()


def _get_init_lock() -> asyncio.Lock:
    """
    Get or create initialization lock (thread-safe).

    Uses a threading lock to ensure the asyncio lock is created exactly once,
    even when multiple coroutines check simultaneously.
    """
    global _init_lock
    if _init_lock is None:
        with _init_lock_guard:
            # Double-check after acquiring thread lock
            if _init_lock is None:
                _init_lock = asyncio.Lock()
    return _init_lock


async def get_services(session_id: str | None = None):
    """
    Get or initialize memory services (thread-safe).

    Args:
        session_id: Session identifier, defaults to settings.session_id

    Returns:
        Tuple of (episodic, semantic, procedural) memory services

    Raises:
        SessionValidationError: If session_id is invalid
    """
    # Import here to avoid circular imports
    from ww.core.validation import validate_session_id

    # Validate session_id at gateway entry point
    validated_session = validate_session_id(
        session_id,
        allow_none=True,
        allow_reserved=True,  # Allow "default" session
    )

    if validated_session is None:
        validated_session = get_settings().session_id

    # Use validated session for all operations
    session_id = validated_session

    # TOCTOU-FIX: Keep lock held during read to prevent race with cleanup_services
    async with _get_init_lock():
        if session_id not in _initialized_sessions:
            logger.info(f"Initializing memory services for session: {session_id}")

            episodic = get_episodic_memory(session_id)
            semantic = get_semantic_memory(session_id)
            procedural = get_procedural_memory(session_id)

            await episodic.initialize()
            await semantic.initialize()
            await procedural.initialize()

            _service_instances[session_id] = {
                "episodic": episodic,
                "semantic": semantic,
                "procedural": procedural,
            }
            _initialized_sessions.add(session_id)
            logger.info(f"Memory services initialized for session: {session_id}")

        # Access services while still holding lock (TOCTOU fix)
        services = _service_instances[session_id]
        return services["episodic"], services["semantic"], services["procedural"]


async def cleanup_services(session_id: str | None = None) -> None:
    """
    Clean up memory service connections.

    Args:
        session_id: Specific session to cleanup, or None for all
    """
    global _initialized_sessions

    async with _get_init_lock():
        if session_id:
            if session_id in _service_instances:
                logger.info(f"Cleaning up services for session: {session_id}")
                del _service_instances[session_id]
                _initialized_sessions.discard(session_id)

                # P7.1 Phase 2B: Clean up bridge container for session
                from ww.core.bridge_container import clear_bridge_containers
                from ww.core.bridge_container import _containers
                if session_id in _containers:
                    _containers.pop(session_id, None)
                    logger.info(f"P7.1 Phase 2B: Bridge container cleaned up for session: {session_id}")
        else:
            logger.info("Cleaning up all memory services...")
            _service_instances.clear()
            _initialized_sessions.clear()

            # P7.1 Phase 2B: Clean up all bridge containers
            from ww.core.bridge_container import clear_bridge_containers
            clear_bridge_containers()
            logger.info("P7.1 Phase 2B: All bridge containers cleaned up")

        # Close storage connections
        await close_qdrant_store(session_id)
        await close_neo4j_store(session_id)

        logger.info("Memory services cleanup complete")


def reset_services() -> None:
    """Reset service management state (for testing)."""
    global _service_instances, _initialized_sessions, _init_lock
    _service_instances.clear()
    _initialized_sessions.clear()
    _init_lock = None
