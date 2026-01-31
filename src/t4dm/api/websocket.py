"""
WebSocket Support for World Weaver.

Provides real-time updates for:
- System state changes (LSN, checkpoint, shutdown)
- Memory operations (add, promote, consolidate)
- Learning updates (gate, scorer, neuromodulator)
- Health metrics

Usage:
    ws://localhost:8080/ws/events     - All events
    ws://localhost:8080/ws/memory     - Memory events only
    ws://localhost:8080/ws/learning   - Learning events only
    ws://localhost:8080/ws/health     - Health metrics only
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class EventType(str, Enum):
    """Types of WebSocket events."""
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_SHUTDOWN = "system.shutdown"
    CHECKPOINT_CREATED = "system.checkpoint"
    WAL_ROTATED = "system.wal_rotated"

    # Memory events
    MEMORY_ADDED = "memory.added"
    MEMORY_PROMOTED = "memory.promoted"
    MEMORY_REMOVED = "memory.removed"
    MEMORY_CONSOLIDATED = "memory.consolidated"

    # Learning events
    GATE_UPDATED = "learning.gate_updated"
    SCORER_UPDATED = "learning.scorer_updated"
    TRACE_CREATED = "learning.trace_created"
    TRACE_DECAYED = "learning.trace_decayed"

    # Neuromodulator events
    DOPAMINE_RPE = "neuromod.dopamine_rpe"
    SEROTONIN_MOOD = "neuromod.serotonin_mood"
    NOREPINEPHRINE_AROUSAL = "neuromod.norepinephrine_arousal"

    # Health events
    HEALTH_UPDATE = "health.update"
    HEALTH_WARNING = "health.warning"
    HEALTH_ERROR = "health.error"


@dataclass
class WebSocketEvent:
    """A WebSocket event message."""
    type: EventType
    timestamp: float
    data: dict[str, Any]

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "data": self.data,
        })


class ConnectionManager:
    """Manages WebSocket connections and event broadcasting."""

    def __init__(self):
        # Active connections by channel
        self._connections: dict[str, set[WebSocket]] = {
            "events": set(),    # All events
            "memory": set(),    # Memory events
            "learning": set(),  # Learning events
            "health": set(),    # Health events
        }
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, channel: str = "events"):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            if channel not in self._connections:
                channel = "events"
            self._connections[channel].add(websocket)
        logger.info(f"WebSocket connected to channel: {channel}")

    async def disconnect(self, websocket: WebSocket, channel: str = "events"):
        """Remove a WebSocket connection."""
        async with self._lock:
            if channel in self._connections:
                self._connections[channel].discard(websocket)
        logger.debug(f"WebSocket disconnected from channel: {channel}")

    async def broadcast(self, event: WebSocketEvent, channels: list[str] | None = None):
        """
        Broadcast event to appropriate channels.

        Args:
            event: The event to broadcast
            channels: Specific channels (None = auto-detect from event type)
        """
        if channels is None:
            channels = self._get_channels_for_event(event.type)

        message = event.to_json()
        dead_connections = []

        async with self._lock:
            for channel in channels:
                if channel not in self._connections:
                    continue

                for websocket in self._connections[channel]:
                    try:
                        await websocket.send_text(message)
                    except Exception:
                        dead_connections.append((channel, websocket))

            # Clean up dead connections
            for channel, ws in dead_connections:
                self._connections[channel].discard(ws)

    def _get_channels_for_event(self, event_type: EventType) -> list[str]:
        """Determine which channels should receive an event."""
        channels = ["events"]  # Always include 'events'

        if event_type.value.startswith("memory."):
            channels.append("memory")
        elif event_type.value.startswith("learning.") or event_type.value.startswith("neuromod."):
            channels.append("learning")
        elif event_type.value.startswith("health."):
            channels.append("health")

        return channels

    @property
    def connection_count(self) -> dict[str, int]:
        """Get count of connections per channel."""
        return {ch: len(conns) for ch, conns in self._connections.items()}


# Global connection manager
manager = ConnectionManager()


# ==================== Event Emission Functions ====================

async def emit_event(event_type: EventType, data: dict[str, Any]):
    """Emit an event to all appropriate channels."""
    event = WebSocketEvent(
        type=event_type,
        timestamp=time.time(),
        data=data,
    )
    await manager.broadcast(event)


async def emit_memory_added(memory_id: str, content_preview: str):
    """Emit memory added event."""
    await emit_event(EventType.MEMORY_ADDED, {
        "memory_id": memory_id,
        "content_preview": content_preview[:100],
    })


async def emit_memory_promoted(memory_id: str, target: str):
    """Emit memory promoted event."""
    await emit_event(EventType.MEMORY_PROMOTED, {
        "memory_id": memory_id,
        "target": target,
    })


async def emit_checkpoint_created(lsn: int, duration_seconds: float):
    """Emit checkpoint created event."""
    await emit_event(EventType.CHECKPOINT_CREATED, {
        "lsn": lsn,
        "duration_seconds": duration_seconds,
    })


async def emit_gate_updated(layer: str, magnitude: float):
    """Emit gate weight update event."""
    await emit_event(EventType.GATE_UPDATED, {
        "layer": layer,
        "update_magnitude": magnitude,
    })


async def emit_dopamine_rpe(context: str, rpe: float, expected: float, actual: float):
    """Emit dopamine RPE event."""
    await emit_event(EventType.DOPAMINE_RPE, {
        "context": context,
        "rpe": rpe,
        "expected": expected,
        "actual": actual,
    })


async def emit_health_update(metrics: dict[str, Any]):
    """Emit health metrics update."""
    await emit_event(EventType.HEALTH_UPDATE, metrics)


async def emit_health_warning(warning: str, details: dict[str, Any] = None):
    """Emit health warning event."""
    await emit_event(EventType.HEALTH_WARNING, {
        "warning": warning,
        "details": details or {},
    })


# ==================== WebSocket Routes ====================

@router.websocket("/ws/events")
async def websocket_all_events(websocket: WebSocket):
    """WebSocket endpoint for all events."""
    await manager.connect(websocket, "events")
    try:
        while True:
            # Keep connection alive, handle any client messages
            data = await websocket.receive_text()
            # Echo back or handle commands
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
    except WebSocketDisconnect:
        await manager.disconnect(websocket, "events")


@router.websocket("/ws/memory")
async def websocket_memory_events(websocket: WebSocket):
    """WebSocket endpoint for memory events."""
    await manager.connect(websocket, "memory")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
    except WebSocketDisconnect:
        await manager.disconnect(websocket, "memory")


@router.websocket("/ws/learning")
async def websocket_learning_events(websocket: WebSocket):
    """WebSocket endpoint for learning events."""
    await manager.connect(websocket, "learning")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
    except WebSocketDisconnect:
        await manager.disconnect(websocket, "learning")


@router.websocket("/ws/health")
async def websocket_health_events(websocket: WebSocket):
    """WebSocket endpoint for health events."""
    await manager.connect(websocket, "health")
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": time.time()}))
    except WebSocketDisconnect:
        await manager.disconnect(websocket, "health")


@router.get("/ws/status")
async def websocket_status():
    """Get WebSocket connection status."""
    return {
        "connections": manager.connection_count,
        "channels": list(manager._connections.keys()),
    }


# ==================== Health Metrics Background Task ====================

async def health_metrics_broadcaster(interval: float = 5.0):
    """
    Background task that broadcasts health metrics periodically.

    Start this in your application startup.
    """
    while True:
        try:
            # Collect metrics
            metrics = await collect_health_metrics()

            # Broadcast to health channel
            await emit_health_update(metrics)

            # Check for warnings
            if metrics.get("checkpoint_age_seconds", 0) > 600:
                await emit_health_warning(
                    "Checkpoint stale",
                    {"age_seconds": metrics["checkpoint_age_seconds"]}
                )

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Health metrics error: {e}")
            await asyncio.sleep(interval)


async def collect_health_metrics() -> dict[str, Any]:
    """Collect current health metrics."""
    import psutil

    metrics = {
        "timestamp": time.time(),
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
    }

    # Try to get persistence metrics
    try:
        from t4dm.persistence import get_persistence
        persistence = get_persistence()
        if persistence:
            metrics.update({
                "current_lsn": persistence.current_lsn,
                "checkpoint_lsn": persistence.last_checkpoint_lsn,
                "operations_since_checkpoint": persistence.current_lsn - persistence.last_checkpoint_lsn,
            })
    except Exception:
        pass

    return metrics
