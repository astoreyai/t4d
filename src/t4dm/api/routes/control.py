"""
World Weaver REST API Control Plane Routes.

Phase 9: Production infrastructure control endpoints.

Provides:
- Feature flag management
- Emergency controls (panic mode, shutdown)
- Circuit breaker status
- Secrets status (not values)

All endpoints require admin authentication.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from t4dm.api.deps import AdminAuth
from t4dm.core.emergency import (
    CircuitBreakerConfig,
    PanicLevel,
    get_emergency_manager,
)
from t4dm.core.feature_flags import FeatureFlag, get_feature_flags
from t4dm.core.secrets import get_secrets_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/control", tags=["control"])


# ============================================================================
# Models
# ============================================================================


class FeatureFlagResponse(BaseModel):
    """Feature flag state."""

    name: str
    enabled: bool
    rollout_percentage: float
    description: str
    owner: str
    usage_count: int


class FeatureFlagsListResponse(BaseModel):
    """List of all feature flags."""

    flags: dict[str, FeatureFlagResponse]
    total: int
    enabled_count: int


class FeatureFlagUpdateRequest(BaseModel):
    """Request to update a feature flag."""

    enabled: bool | None = None
    rollout_percentage: float | None = Field(None, ge=0.0, le=100.0)


class EmergencyStatusResponse(BaseModel):
    """Emergency manager status."""

    panic_level: str
    is_shutting_down: bool
    in_flight_count: int
    panic_events_count: int
    circuit_breakers: dict[str, dict]


class PanicRequest(BaseModel):
    """Request to trigger panic mode."""

    level: str = Field(..., description="Panic level: DEGRADED, LIMITED, CRITICAL, TOTAL")
    reason: str = Field(..., min_length=5)
    source: str = ""


class RecoveryRequest(BaseModel):
    """Request to recover from panic."""

    level: str = Field(
        "NONE",
        description="Target level: NONE, DEGRADED, LIMITED"
    )


class CircuitBreakerUpdateRequest(BaseModel):
    """Request to update circuit breaker."""

    action: str = Field(..., description="Action: reset, open, close")


class SecretsStatusResponse(BaseModel):
    """Secrets manager status (no secret values)."""

    backend: str
    available_keys_count: int
    audit_enabled: bool
    access_count: int


# ============================================================================
# Feature Flags Endpoints
# ============================================================================


@router.get("/flags", response_model=FeatureFlagsListResponse)
async def list_feature_flags(admin: AdminAuth):
    """
    List all feature flags and their states.

    Requires admin authentication.
    """
    flags = get_feature_flags()
    all_flags = flags.get_all_flags()

    flag_responses = {
        name: FeatureFlagResponse(
            name=name,
            enabled=data["enabled"],
            rollout_percentage=data["rollout_percentage"],
            description=data.get("description", ""),
            owner=data.get("owner", ""),
            usage_count=data.get("usage_count", 0),
        )
        for name, data in all_flags.items()
    }

    stats = flags.get_stats()

    return FeatureFlagsListResponse(
        flags=flag_responses,
        total=stats["total_flags"],
        enabled_count=stats["enabled_flags"],
    )


@router.get("/flags/{flag_name}", response_model=FeatureFlagResponse)
async def get_feature_flag(flag_name: str, admin: AdminAuth):
    """
    Get a specific feature flag.

    Requires admin authentication.
    """
    flags = get_feature_flags()

    # Find the flag
    try:
        flag = FeatureFlag(flag_name)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature flag '{flag_name}' not found",
        )

    config = flags.get_config(flag)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature flag '{flag_name}' not found",
        )

    all_flags = flags.get_all_flags()
    data = all_flags.get(flag_name, {})

    return FeatureFlagResponse(
        name=flag_name,
        enabled=config.enabled,
        rollout_percentage=config.rollout_percentage,
        description=config.description,
        owner=config.owner,
        usage_count=data.get("usage_count", 0),
    )


@router.patch("/flags/{flag_name}", response_model=FeatureFlagResponse)
async def update_feature_flag(
    flag_name: str,
    update: FeatureFlagUpdateRequest,
    admin: AdminAuth,
):
    """
    Update a feature flag.

    Requires admin authentication.
    """
    flags = get_feature_flags()

    # Find the flag
    try:
        flag = FeatureFlag(flag_name)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Feature flag '{flag_name}' not found",
        )

    # Apply updates
    if update.enabled is not None:
        flags.set_enabled(flag, update.enabled)
        logger.warning(
            f"Admin changed flag {flag_name} enabled={update.enabled}"
        )

    if update.rollout_percentage is not None:
        flags.set_rollout_percentage(flag, update.rollout_percentage)
        logger.warning(
            f"Admin changed flag {flag_name} rollout={update.rollout_percentage}%"
        )

    # Return updated state
    config = flags.get_config(flag)
    all_flags = flags.get_all_flags()
    data = all_flags.get(flag_name, {})

    return FeatureFlagResponse(
        name=flag_name,
        enabled=config.enabled if config else False,
        rollout_percentage=config.rollout_percentage if config else 0.0,
        description=config.description if config else "",
        owner=config.owner if config else "",
        usage_count=data.get("usage_count", 0),
    )


# ============================================================================
# Emergency Endpoints
# ============================================================================


@router.get("/emergency", response_model=EmergencyStatusResponse)
async def get_emergency_status(admin: AdminAuth):
    """
    Get emergency manager status.

    Requires admin authentication.
    """
    em = get_emergency_manager()
    stats = em.get_stats()

    return EmergencyStatusResponse(
        panic_level=stats["panic_level"],
        is_shutting_down=stats["is_shutting_down"],
        in_flight_count=stats["in_flight_count"],
        panic_events_count=stats["panic_events"],
        circuit_breakers=stats["circuit_breakers"],
    )


@router.post("/emergency/panic", response_model=EmergencyStatusResponse)
async def trigger_panic(request: PanicRequest, admin: AdminAuth):
    """
    Trigger panic mode.

    Requires admin authentication.

    Panic levels:
    - DEGRADED: Some features disabled, core functional
    - LIMITED: Read-only mode, no writes
    - CRITICAL: Block all requests except health
    - TOTAL: Complete shutdown
    """
    em = get_emergency_manager()

    # Parse level
    try:
        level = PanicLevel[request.level.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid panic level: {request.level}. "
                   f"Valid: DEGRADED, LIMITED, CRITICAL, TOTAL",
        )

    em.panic(
        level=level,
        reason=request.reason,
        source=request.source or "admin_api",
    )

    logger.error(
        f"Admin triggered PANIC: level={level.name}, reason={request.reason}"
    )

    stats = em.get_stats()
    return EmergencyStatusResponse(
        panic_level=stats["panic_level"],
        is_shutting_down=stats["is_shutting_down"],
        in_flight_count=stats["in_flight_count"],
        panic_events_count=stats["panic_events"],
        circuit_breakers=stats["circuit_breakers"],
    )


@router.post("/emergency/recover", response_model=EmergencyStatusResponse)
async def trigger_recovery(request: RecoveryRequest, admin: AdminAuth):
    """
    Recover from panic mode.

    Requires admin authentication.
    """
    em = get_emergency_manager()

    # Parse level
    try:
        level = PanicLevel[request.level.upper()]
    except KeyError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid level: {request.level}. Valid: NONE, DEGRADED, LIMITED",
        )

    current = em.panic_level
    if level >= current:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot recover to {level.name} from {current.name}. "
                   f"Target must be lower than current.",
        )

    em.recover(level)

    logger.warning(
        f"Admin triggered recovery: {current.name} -> {level.name}"
    )

    stats = em.get_stats()
    return EmergencyStatusResponse(
        panic_level=stats["panic_level"],
        is_shutting_down=stats["is_shutting_down"],
        in_flight_count=stats["in_flight_count"],
        panic_events_count=stats["panic_events"],
        circuit_breakers=stats["circuit_breakers"],
    )


@router.get("/emergency/events")
async def get_panic_events(admin: AdminAuth, limit: int = 100):
    """
    Get recent panic events.

    Requires admin authentication.
    """
    em = get_emergency_manager()
    events = em.get_panic_events(limit=limit)
    return {"events": events, "count": len(events)}


# ============================================================================
# Circuit Breaker Endpoints
# ============================================================================


@router.get("/circuits")
async def list_circuit_breakers(admin: AdminAuth):
    """
    List all circuit breakers.

    Requires admin authentication.
    """
    em = get_emergency_manager()
    stats = em.get_stats()
    return {
        "circuits": stats["circuit_breakers"],
        "count": len(stats["circuit_breakers"]),
    }


@router.get("/circuits/{name}")
async def get_circuit_breaker(name: str, admin: AdminAuth):
    """
    Get a specific circuit breaker.

    Requires admin authentication.
    """
    em = get_emergency_manager()
    stats = em.get_stats()

    if name not in stats["circuit_breakers"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circuit breaker '{name}' not found",
        )

    return stats["circuit_breakers"][name]


@router.post("/circuits/{name}")
async def update_circuit_breaker(
    name: str,
    update: CircuitBreakerUpdateRequest,
    admin: AdminAuth,
):
    """
    Update a circuit breaker.

    Requires admin authentication.

    Actions:
    - reset: Reset to closed state
    - open: Force open the circuit
    - close: Force close the circuit
    """
    em = get_emergency_manager()

    try:
        cb = em.get_circuit_breaker(name)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Circuit breaker '{name}' not found",
        )

    if update.action == "reset":
        cb.reset()
        logger.warning(f"Admin reset circuit breaker '{name}'")
    elif update.action == "open":
        # Force open by recording failures
        for _ in range(cb.config.failure_threshold):
            cb.record_failure()
        logger.warning(f"Admin force-opened circuit breaker '{name}'")
    elif update.action == "close":
        cb.reset()
        logger.warning(f"Admin force-closed circuit breaker '{name}'")
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action: {update.action}. Valid: reset, open, close",
        )

    return cb.get_stats()


# ============================================================================
# Secrets Endpoints (status only, no values)
# ============================================================================


@router.get("/secrets", response_model=SecretsStatusResponse)
async def get_secrets_status(admin: AdminAuth):
    """
    Get secrets manager status.

    Returns status only, never secret values.
    Requires admin authentication.
    """
    sm = get_secrets_manager()
    stats = sm.get_stats()

    return SecretsStatusResponse(
        backend=stats["backend"],
        available_keys_count=stats["available_keys"],
        audit_enabled=stats["audit_enabled"],
        access_count=stats["access_count"],
    )


@router.get("/secrets/keys")
async def list_secret_keys(admin: AdminAuth):
    """
    List available secret keys (not values).

    Requires admin authentication.
    """
    sm = get_secrets_manager()
    keys = sm.list_keys()
    return {"keys": keys, "count": len(keys)}


@router.get("/secrets/audit")
async def get_secrets_audit_log(admin: AdminAuth, limit: int = 100):
    """
    Get secrets access audit log.

    Requires admin authentication.
    """
    sm = get_secrets_manager()
    log = sm.get_access_log()
    return {"entries": log[-limit:], "count": len(log)}


__all__ = ["router"]
