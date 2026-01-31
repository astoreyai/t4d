"""
Persistence API Routes.

Provides REST endpoints for:
- System status (cold/warm start, LSN, checkpoint info)
- Manual checkpoint triggering
- WAL management
- Recovery status
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/persistence", tags=["persistence"])


# ==================== Models ====================

class SystemStatus(BaseModel):
    """Overall system status."""
    started: bool = Field(description="Whether persistence is running")
    mode: str = Field(description="Startup mode (cold_start, warm_start)")
    current_lsn: int = Field(description="Current WAL Log Sequence Number")
    checkpoint_lsn: int = Field(description="LSN of last checkpoint")
    operations_since_checkpoint: int = Field(description="Operations since last checkpoint")
    uptime_seconds: float = Field(description="Time since startup")
    shutdown_requested: bool = Field(description="Whether shutdown is pending")


class CheckpointInfo(BaseModel):
    """Checkpoint information."""
    lsn: int = Field(description="Checkpoint LSN")
    timestamp: datetime = Field(description="When checkpoint was created")
    size_bytes: int = Field(description="Checkpoint file size")
    components: list[str] = Field(description="Components included")


class WALStatus(BaseModel):
    """WAL status information."""
    current_lsn: int = Field(description="Current LSN")
    checkpoint_lsn: int = Field(description="Last checkpoint LSN")
    segment_count: int = Field(description="Number of WAL segments")
    total_size_bytes: int = Field(description="Total WAL size")
    oldest_segment: int = Field(description="Oldest segment number")
    current_segment: int = Field(description="Current segment number")


class RecoveryInfo(BaseModel):
    """Recovery information from startup."""
    mode: str = Field(description="Recovery mode (cold_start, warm_start, forced_cold)")
    success: bool = Field(description="Whether recovery succeeded")
    checkpoint_lsn: int = Field(description="Checkpoint used for recovery")
    wal_entries_replayed: int = Field(description="WAL entries replayed")
    components_restored: dict[str, bool] = Field(description="Component restoration status")
    errors: list[str] = Field(description="Any errors during recovery")
    duration_seconds: float = Field(description="Recovery duration")


class CheckpointRequest(BaseModel):
    """Request to create checkpoint."""
    force: bool = Field(default=False, description="Force checkpoint even if recent one exists")


class CheckpointResponse(BaseModel):
    """Response from checkpoint creation."""
    success: bool
    lsn: int
    duration_seconds: float
    message: str


# ==================== Dependency ====================

def get_persistence():
    """Get persistence manager from application state."""
    # Import here to avoid circular imports
    try:
        from t4dm.persistence import get_persistence as _get_persistence
        persistence = _get_persistence()
        if persistence is None:
            raise HTTPException(
                status_code=503,
                detail="Persistence layer not initialized"
            )
        return persistence
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Persistence module not available"
        )


# ==================== Routes ====================

@router.get("/status", response_model=SystemStatus)
async def get_system_status(persistence=Depends(get_persistence)):
    """
    Get overall persistence system status.

    Returns current state including LSN, checkpoint info, and health.
    """
    status = persistence.get_status()

    return SystemStatus(
        started=status.get("started", False),
        mode=status.get("recovery_mode", "unknown"),
        current_lsn=status.get("current_lsn", 0),
        checkpoint_lsn=status.get("last_checkpoint_lsn", 0),
        operations_since_checkpoint=status.get("current_lsn", 0) - status.get("last_checkpoint_lsn", 0),
        uptime_seconds=status.get("uptime_seconds", 0),
        shutdown_requested=status.get("shutdown_requested", False),
    )


@router.get("/checkpoints", response_model=list[CheckpointInfo])
async def list_checkpoints(persistence=Depends(get_persistence)):
    """
    List available checkpoints.

    Returns metadata for all checkpoint files.
    """
    from pathlib import Path

    checkpoint_dir = Path(persistence.config.data_directory) / "checkpoints"
    checkpoints = []

    for path in sorted(checkpoint_dir.glob("checkpoint_*.bin*")):
        try:
            # Extract LSN from filename
            stem = path.stem
            stem = stem.removesuffix(".bin")
            lsn = int(stem.split("_")[1])

            checkpoints.append(CheckpointInfo(
                lsn=lsn,
                timestamp=datetime.fromtimestamp(path.stat().st_mtime),
                size_bytes=path.stat().st_size,
                components=["buffer", "gate", "scorer", "neuromod"],  # Standard components
            ))
        except (ValueError, IndexError):
            continue

    return checkpoints


@router.post("/checkpoint", response_model=CheckpointResponse)
async def create_checkpoint(
    request: CheckpointRequest = CheckpointRequest(),
    persistence=Depends(get_persistence),
):
    """
    Create a checkpoint manually.

    Useful before deployments or maintenance.
    """
    import time

    start_time = time.time()

    try:
        checkpoint = await persistence.create_checkpoint()
        duration = time.time() - start_time

        return CheckpointResponse(
            success=True,
            lsn=checkpoint.lsn,
            duration_seconds=duration,
            message=f"Checkpoint created at LSN {checkpoint.lsn}",
        )
    except Exception as e:
        return CheckpointResponse(
            success=False,
            lsn=0,
            duration_seconds=time.time() - start_time,
            message=f"Checkpoint failed: {e!s}",
        )


@router.get("/wal", response_model=WALStatus)
async def get_wal_status(persistence=Depends(get_persistence)):
    """
    Get WAL status information.

    Shows segment count, sizes, and LSN information.
    """
    from pathlib import Path

    wal_dir = Path(persistence.config.data_directory) / "wal"
    segments = list(wal_dir.glob("segment_*.wal"))

    total_size = sum(s.stat().st_size for s in segments)
    segment_nums = []
    for s in segments:
        try:
            num = int(s.stem.split("_")[1])
            segment_nums.append(num)
        except (ValueError, IndexError):
            continue

    return WALStatus(
        current_lsn=persistence.current_lsn,
        checkpoint_lsn=persistence.last_checkpoint_lsn,
        segment_count=len(segments),
        total_size_bytes=total_size,
        oldest_segment=min(segment_nums) if segment_nums else 0,
        current_segment=max(segment_nums) if segment_nums else 0,
    )


@router.post("/wal/truncate")
async def truncate_wal(persistence=Depends(get_persistence)):
    """
    Truncate WAL segments before last checkpoint.

    Removes old segments to reclaim disk space.
    """
    removed = await persistence.truncate_wal()

    return {
        "success": True,
        "segments_removed": removed,
        "message": f"Removed {removed} WAL segments",
    }


@router.get("/recovery", response_model=RecoveryInfo)
async def get_recovery_info(persistence=Depends(get_persistence)):
    """
    Get information about the last recovery/startup.

    Shows whether system performed cold or warm start.
    """
    # This would typically be stored in the persistence manager
    # For now, return current state
    status = persistence.get_status()

    return RecoveryInfo(
        mode=status.get("recovery_mode", "unknown"),
        success=True,
        checkpoint_lsn=status.get("last_checkpoint_lsn", 0),
        wal_entries_replayed=status.get("wal_entries_replayed", 0),
        components_restored=status.get("components_restored", {}),
        errors=[],
        duration_seconds=status.get("recovery_duration", 0),
    )


@router.get("/health")
async def health_check(persistence=Depends(get_persistence)):
    """
    Health check for persistence layer.

    Used by load balancers and monitoring.
    """
    if not persistence.is_started:
        raise HTTPException(status_code=503, detail="Persistence not started")

    if persistence.should_shutdown:
        raise HTTPException(status_code=503, detail="Shutdown in progress")

    # Check for stale checkpoint
    import time
    time_since_checkpoint = time.time() - persistence._checkpoint._last_checkpoint_time

    status = "healthy"
    warnings = []

    if time_since_checkpoint > 600:  # 10 minutes
        warnings.append(f"Checkpoint stale ({time_since_checkpoint:.0f}s old)")
        status = "degraded"

    ops_since_checkpoint = persistence.current_lsn - persistence.last_checkpoint_lsn
    if ops_since_checkpoint > 10000:
        warnings.append(f"Many uncommitted operations ({ops_since_checkpoint})")
        status = "degraded"

    return {
        "status": status,
        "current_lsn": persistence.current_lsn,
        "checkpoint_lsn": persistence.last_checkpoint_lsn,
        "warnings": warnings,
    }
