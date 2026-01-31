"""
World Weaver REST API Dependencies.

Shared dependencies for FastAPI routes including session management
and memory service initialization.
"""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import Depends, Header, HTTPException, Path, status

from t4dm.core.config import get_settings
from t4dm.core.services import RateLimiter, get_services
from t4dm.core.validation import SessionValidationError, validate_session_id
from t4dm.observability.logging import get_audit_logger

logger = logging.getLogger(__name__)

# Global rate limiter instance
_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


async def get_session_id(
    x_session_id: Annotated[str | None, Header()] = None,
) -> str:
    """
    Extract and validate session ID from request header.

    Args:
        x_session_id: Session ID from X-Session-ID header

    Returns:
        Validated session ID

    Raises:
        HTTPException: If session ID is invalid
    """
    try:
        # P2-SEC-M2: Disallow reserved session IDs in API for security
        validated = validate_session_id(
            x_session_id,
            allow_none=True,
            allow_reserved=False,  # Reject "admin", "system", "root", etc.
        )
        return validated or get_settings().session_id
    except SessionValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


async def check_rate_limit(
    session_id: Annotated[str, Depends(get_session_id)],
) -> str:
    """
    Check rate limit for session.

    Args:
        session_id: Validated session ID

    Returns:
        Session ID if allowed

    Raises:
        HTTPException: If rate limited
    """
    if not _rate_limiter.allow(session_id):
        wait_time = _rate_limiter.time_until_allowed(session_id)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {wait_time:.1f} seconds.",
            headers={"Retry-After": str(int(wait_time) + 1)},
        )
    return session_id


async def get_memory_services(
    session_id: Annotated[str, Depends(check_rate_limit)],
):
    """
    Get initialized memory services for session.

    Args:
        session_id: Validated and rate-limited session ID

    Returns:
        Tuple of (episodic, semantic, procedural) services

    Raises:
        HTTPException: If services cannot be initialized
    """
    try:
        episodic, semantic, procedural = await get_services(session_id)
        return {
            "session_id": session_id,
            "episodic": episodic,
            "semantic": semantic,
            "procedural": procedural,
        }
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Memory services unavailable: {e!s}",
        )


def parse_uuid(value: str, field: str = "id") -> UUID:
    """
    Parse and validate a UUID string.

    Use this helper in API routes to validate UUID path/query parameters
    and return proper 400 errors for invalid input.

    Args:
        value: String to parse as UUID
        field: Field name for error messages

    Returns:
        Parsed UUID object

    Raises:
        HTTPException: 400 Bad Request if UUID is invalid
    """
    if not value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required {field}",
        )

    try:
        return UUID(value)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID format for {field}: {value}",
        )


def validate_uuid_path(memory_id: str = Path(..., description="Memory UUID")) -> UUID:
    """
    FastAPI dependency to validate UUID path parameters.

    Usage:
        @router.get("/memory/{memory_id}")
        async def get_memory(memory_id: UUID = Depends(validate_uuid_path)):
            ...
    """
    return parse_uuid(memory_id, "memory_id")


# P2-SEC-M4: API key authentication for all endpoints
async def require_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> bool:
    """
    Require API key for all endpoints (if configured).

    In production mode with api_key set, all requests must include
    X-API-Key header. Disabled in development unless explicitly enabled.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        True if authenticated (or not required)

    Raises:
        HTTPException: If API key is required but missing or invalid
    """
    import secrets
    settings = get_settings()

    # Check if API key is required
    api_key = getattr(settings, "api_key", None)
    api_key_required = getattr(settings, "api_key_required", False)
    env = getattr(settings, "environment", "development")

    # Auto-enable in production if key is configured
    if env == "production" and api_key:
        api_key_required = True

    if not api_key_required or not api_key:
        return True  # API key not required

    if not x_api_key:
        # P3-SEC-L5: Audit log auth failure
        get_audit_logger().log_auth_failure(reason="missing_api_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(x_api_key, api_key):
        logger.warning("Invalid API key attempt")
        # P3-SEC-L5: Audit log auth failure
        get_audit_logger().log_auth_failure(reason="invalid_api_key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return True


# API-CRITICAL-001/002/003 FIX: Add admin authentication for sensitive endpoints
async def require_admin_auth(
    x_admin_key: Annotated[str | None, Header()] = None,
) -> bool:
    """
    Require admin API key for sensitive operations.

    Args:
        x_admin_key: Admin API key from X-Admin-Key header

    Returns:
        True if authenticated

    Raises:
        HTTPException: If admin key is missing or invalid
    """
    settings = get_settings()

    # Get admin key from config (default empty disables admin endpoints)
    admin_key = getattr(settings, "admin_api_key", None)

    if not admin_key:
        # No admin key configured - reject all admin requests
        logger.warning("Admin endpoint accessed but no admin_api_key configured")
        # P3-SEC-L5: Audit log auth failure
        get_audit_logger().log_auth_failure(reason="admin_access_disabled")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access disabled (no admin_api_key configured)",
        )

    if not x_admin_key:
        # P3-SEC-L5: Audit log auth failure
        get_audit_logger().log_auth_failure(reason="missing_admin_key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Admin-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to prevent timing attacks
    import secrets
    if not secrets.compare_digest(x_admin_key, admin_key):
        logger.warning("Invalid admin key attempt")
        # P3-SEC-L5: Audit log auth failure
        get_audit_logger().log_auth_failure(reason="invalid_admin_key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin key",
        )

    # P3-SEC-L5: Audit log successful admin auth
    get_audit_logger().log_auth_success(session_id="admin", method="admin_api_key")
    return True


async def get_bridge_container_dep(
    session_id: Annotated[str, Depends(check_rate_limit)],
):
    """
    Get bridge container for session.

    P7.1 Phase 2B: Provides access to NCA bridges for API endpoints.

    Args:
        session_id: Validated and rate-limited session ID

    Returns:
        BridgeContainer for the session

    Raises:
        HTTPException: If bridge container is not initialized
    """
    from t4dm.core.bridge_container import get_bridge_container

    try:
        container = get_bridge_container(session_id)
        if not container.state.initialized:
            # Container exists but bridges not initialized
            # This should not happen if EpisodicMemory.initialize() was called
            logger.warning(
                f"P7.1 Phase 2B: Bridge container for session {session_id} "
                f"not initialized. This indicates services were not properly initialized."
            )
        return container
    except Exception as e:
        logger.error(f"Failed to get bridge container: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Bridge container unavailable: {e!s}",
        )


# Type aliases for dependency injection
SessionId = Annotated[str, Depends(get_session_id)]
RateLimitedSession = Annotated[str, Depends(check_rate_limit)]
MemoryServices = Annotated[dict, Depends(get_memory_services)]
BridgeContainerDep = Annotated["BridgeContainer", Depends(get_bridge_container_dep)]  # P7.1 Phase 2B
ValidatedUUID = Annotated[UUID, Depends(validate_uuid_path)]
ApiKeyAuth = Annotated[bool, Depends(require_api_key)]  # P2-SEC-M4
AdminAuth = Annotated[bool, Depends(require_admin_auth)]
