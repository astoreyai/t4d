"""FastAPI WebSocket server for observation streaming."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from t4dm.t4dv.aggregator import SnapshotAggregator
from t4dm.t4dv.bus import ObservationBus, get_bus
from t4dm.t4dv.renderers.protocol import get_registry

logger = logging.getLogger(__name__)

# Max frames a slow consumer can miss before we skip
_MAX_PENDING = 50
_STREAM_INTERVAL_S = 0.1  # 100ms


def create_app(
    bus: ObservationBus | None = None,
    aggregator: SnapshotAggregator | None = None,
) -> FastAPI:
    """Create the T4DV FastAPI app."""
    bus = bus or get_bus()
    agg = aggregator or SnapshotAggregator()

    app = FastAPI(title="T4DV Observation Server", version="0.1.0")

    # Wire aggregator as bus subscriber
    bus.subscribe("*", agg.ingest)

    # --- WebSocket ---

    @app.websocket("/ws/observe")
    async def ws_observe(websocket: WebSocket) -> None:
        await websocket.accept()
        pending = 0
        last_state: dict[str, Any] = {}
        try:
            while True:
                state = agg.get_dashboard_state()
                if state != last_state:
                    if pending < _MAX_PENDING:
                        try:
                            await asyncio.wait_for(
                                websocket.send_json(state), timeout=1.0,
                            )
                            pending = 0
                        except asyncio.TimeoutError:
                            pending += 1
                            logger.debug("Slow consumer, pending=%d", pending)
                    else:
                        # Skip frames for slow consumers
                        pending = 0
                    last_state = state
                await asyncio.sleep(_STREAM_INTERVAL_S)
        except WebSocketDisconnect:
            pass

    # --- REST ---

    @app.get("/api/v1/viz/snapshot/{topic}")
    async def snapshot(topic: str) -> JSONResponse:
        events = bus.snapshot(topic)
        return JSONResponse([e.model_dump(mode="json") for e in events])

    @app.get("/api/v1/viz/dashboard")
    async def dashboard() -> JSONResponse:
        return JSONResponse(agg.get_dashboard_state())

    @app.get("/api/v1/viz/topics")
    async def topics() -> JSONResponse:
        return JSONResponse(bus.topics())

    @app.get("/api/v1/viz/views")
    async def views() -> JSONResponse:
        return JSONResponse(get_registry().list_views())

    @app.on_event("startup")
    async def on_startup() -> None:
        bus.set_loop(asyncio.get_running_loop())

    return app


def run_server(host: str = "0.0.0.0", port: int = 8420) -> None:
    """Standalone entry point: ``t4dm viz-server``."""
    import uvicorn
    app = create_app()
    uvicorn.run(app, host=host, port=port)
