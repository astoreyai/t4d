"""
WebSocket endpoint for real-time visualization streaming.

Streams telemetry snapshots as JSON to connected clients.
Clients can filter which modules they receive via subscribe messages.
"""

import asyncio
import json
import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from t4dm.api.websocket import manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Optional global streamer reference; set via set_visualization_streamer()
_streamer = None


def set_visualization_streamer(streamer) -> None:
    """Set the global VisualizationStreamer instance."""
    global _streamer
    _streamer = streamer


def get_visualization_streamer():
    """Get the global VisualizationStreamer instance (may be None)."""
    return _streamer


@router.websocket("/ws/visualization")
async def websocket_visualization(websocket: WebSocket):
    """
    WebSocket endpoint for visualization streaming.

    On connect, registers with ConnectionManager on the "visualization" channel.
    Periodically (every 1s) collects a snapshot from the VisualizationStreamer
    and sends it to the client.

    Clients can send filter messages to limit what modules are streamed:
        {"subscribe": ["spiking", "kappa", "neuromod"]}

    Each outgoing message has the format:
        {"type": "snapshot", "module": "<name>", "data": {...}, "timestamp": ...}
    """
    # Ensure the "visualization" channel exists
    if "visualization" not in manager._connections:
        manager._connections["visualization"] = set()

    await manager.connect(websocket, "visualization")
    subscribed_modules: list[str] | None = None

    async def send_snapshots():
        """Background task that sends periodic snapshots."""
        while True:
            await asyncio.sleep(1.0)
            streamer = get_visualization_streamer()
            if streamer is None:
                continue
            try:
                snapshot = streamer.get_snapshot(modules=subscribed_modules)
                ts = snapshot.get("timestamp", time.time())
                for module_name, module_data in snapshot.get("modules", {}).items():
                    message = json.dumps({
                        "type": "snapshot",
                        "module": module_name,
                        "data": module_data,
                        "timestamp": ts,
                    })
                    await websocket.send_text(message)
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"Snapshot send error: {e}")
                break

    sender_task = asyncio.create_task(send_snapshots())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
                if "subscribe" in msg and isinstance(msg["subscribe"], list):
                    subscribed_modules = msg["subscribe"]
                    logger.debug(f"Client subscribed to: {subscribed_modules}")
                elif raw == "ping":
                    await websocket.send_text(
                        json.dumps({"type": "pong", "timestamp": time.time()})
                    )
            except json.JSONDecodeError:
                if raw == "ping":
                    await websocket.send_text(
                        json.dumps({"type": "pong", "timestamp": time.time()})
                    )
    except WebSocketDisconnect:
        pass
    finally:
        sender_task.cancel()
        try:
            await sender_task
        except asyncio.CancelledError:
            pass
        await manager.disconnect(websocket, "visualization")
