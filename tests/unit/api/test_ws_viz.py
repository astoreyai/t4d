"""Tests for the /ws/visualization WebSocket endpoint."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from t4dm.api.routes.ws_viz import router, set_visualization_streamer
from t4dm.visualization.stream import VisualizationStreamer


@pytest.fixture
def app():
    """Create a minimal FastAPI app with the ws_viz router."""
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def streamer():
    """Create a VisualizationStreamer with a fake visualizer."""
    s = VisualizationStreamer()
    viz = MagicMock()
    viz.export_data.return_value = {"metric": 99}
    s.register_visualizer("test_mod", viz)
    return s


class TestWsVizEndpoint:
    def test_connect_and_receive_ping(self, client):
        """Test basic WebSocket connection and ping/pong."""
        with client.websocket_connect("/ws/visualization") as ws:
            ws.send_text("ping")
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "pong"
            assert "timestamp" in resp

    def test_subscribe_message(self, client, streamer):
        """Test that subscribe messages are accepted without error."""
        set_visualization_streamer(streamer)
        try:
            with client.websocket_connect("/ws/visualization") as ws:
                ws.send_text(json.dumps({"subscribe": ["test_mod"]}))
                # Send ping to verify connection still alive
                ws.send_text("ping")
                resp = json.loads(ws.receive_text())
                assert resp["type"] in ("pong", "snapshot")
        finally:
            set_visualization_streamer(None)

    def test_no_streamer_no_crash(self, client):
        """Without a streamer set, connection works but no snapshots."""
        set_visualization_streamer(None)
        with client.websocket_connect("/ws/visualization") as ws:
            ws.send_text("ping")
            resp = json.loads(ws.receive_text())
            assert resp["type"] == "pong"
