"""
Tests for WebSocket module.

Tests event types, connection management, and broadcasting.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime
import json

from t4dm.api.websocket import (
    EventType,
    WebSocketEvent,
    ConnectionManager,
    router,
)


# =============================================================================
# Test EventType Enum
# =============================================================================


class TestEventType:
    """Tests for EventType enum."""

    def test_system_events_exist(self):
        """Test system event types exist."""
        assert EventType.SYSTEM_START is not None
        assert EventType.SYSTEM_SHUTDOWN is not None
        assert EventType.CHECKPOINT_CREATED is not None
        assert EventType.WAL_ROTATED is not None

    def test_memory_events_exist(self):
        """Test memory event types exist."""
        assert EventType.MEMORY_ADDED is not None
        assert EventType.MEMORY_PROMOTED is not None
        assert EventType.MEMORY_REMOVED is not None
        assert EventType.MEMORY_CONSOLIDATED is not None

    def test_learning_events_exist(self):
        """Test learning event types exist."""
        assert EventType.GATE_UPDATED is not None
        assert EventType.SCORER_UPDATED is not None
        assert EventType.TRACE_CREATED is not None
        assert EventType.TRACE_DECAYED is not None

    def test_neuromod_events_exist(self):
        """Test neuromodulator event types exist."""
        assert EventType.DOPAMINE_RPE is not None
        assert EventType.SEROTONIN_MOOD is not None
        assert EventType.NOREPINEPHRINE_AROUSAL is not None

    def test_health_events_exist(self):
        """Test health event types exist."""
        assert EventType.HEALTH_UPDATE is not None
        assert EventType.HEALTH_WARNING is not None
        assert EventType.HEALTH_ERROR is not None

    def test_event_values(self):
        """Test event type values are correct."""
        assert EventType.SYSTEM_START.value == "system.start"
        assert EventType.MEMORY_ADDED.value == "memory.added"
        assert EventType.GATE_UPDATED.value == "learning.gate_updated"


# =============================================================================
# Test WebSocketEvent
# =============================================================================


class TestWebSocketEvent:
    """Tests for WebSocketEvent dataclass."""

    def test_creation(self):
        """Test event creation."""
        event = WebSocketEvent(
            type=EventType.MEMORY_ADDED,
            timestamp=1234567890.0,
            data={"memory_id": "test-123", "type": "episodic"},
        )
        assert event.type == EventType.MEMORY_ADDED
        assert event.timestamp == 1234567890.0
        assert event.data["memory_id"] == "test-123"

    def test_to_json(self):
        """Test JSON serialization."""
        event = WebSocketEvent(
            type=EventType.CHECKPOINT_CREATED,
            timestamp=1234567890.0,
            data={"lsn": 1000},
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "system.checkpoint"
        assert parsed["timestamp"] == 1234567890.0
        assert "datetime" in parsed
        assert parsed["data"]["lsn"] == 1000

    def test_json_includes_datetime(self):
        """Test JSON includes ISO datetime."""
        import time
        now = time.time()
        event = WebSocketEvent(
            type=EventType.HEALTH_UPDATE,
            timestamp=now,
            data={"status": "healthy"},
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)

        # Check datetime is ISO format
        assert "T" in parsed["datetime"]


# =============================================================================
# Test ConnectionManager
# =============================================================================


class TestConnectionManager:
    """Tests for ConnectionManager class."""

    @pytest.fixture
    def manager(self):
        """Create a connection manager."""
        return ConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create a mock WebSocket."""
        ws = MagicMock()
        ws.accept = AsyncMock()
        ws.send_text = AsyncMock()
        ws.close = AsyncMock()
        return ws

    def test_initialization(self, manager):
        """Test manager initializes with empty channels."""
        assert len(manager._connections) == 4
        assert "events" in manager._connections
        assert "memory" in manager._connections
        assert "learning" in manager._connections
        assert "health" in manager._connections

    @pytest.mark.asyncio
    async def test_connect_default_channel(self, manager, mock_websocket):
        """Test connecting to default channel."""
        await manager.connect(mock_websocket)
        mock_websocket.accept.assert_called_once()
        assert mock_websocket in manager._connections["events"]

    @pytest.mark.asyncio
    async def test_connect_specific_channel(self, manager, mock_websocket):
        """Test connecting to specific channel."""
        await manager.connect(mock_websocket, channel="memory")
        assert mock_websocket in manager._connections["memory"]
        assert mock_websocket not in manager._connections["events"]

    @pytest.mark.asyncio
    async def test_connect_invalid_channel_defaults(self, manager, mock_websocket):
        """Test invalid channel defaults to events."""
        await manager.connect(mock_websocket, channel="invalid")
        assert mock_websocket in manager._connections["events"]

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        """Test disconnecting from channel."""
        await manager.connect(mock_websocket, channel="health")
        await manager.disconnect(mock_websocket, channel="health")
        assert mock_websocket not in manager._connections["health"]

    @pytest.mark.asyncio
    async def test_disconnect_not_connected(self, manager, mock_websocket):
        """Test disconnecting when not connected doesn't raise."""
        # Should not raise
        await manager.disconnect(mock_websocket, channel="memory")

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self, manager, mock_websocket):
        """Test broadcasting to channel."""
        await manager.connect(mock_websocket, channel="memory")

        event = WebSocketEvent(
            type=EventType.MEMORY_ADDED,
            timestamp=1234567890.0,
            data={"id": "test"},
        )
        await manager.broadcast(event, channels=["memory"])

        mock_websocket.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_connections(self, manager):
        """Test broadcasting to multiple connections."""
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()

        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()

        await manager.connect(ws1, channel="learning")
        await manager.connect(ws2, channel="learning")

        event = WebSocketEvent(
            type=EventType.GATE_UPDATED,
            timestamp=1234567890.0,
            data={"weights": [0.5, 0.5]},
        )
        await manager.broadcast(event, channels=["learning"])

        ws1.send_text.assert_called_once()
        ws2.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_handles_disconnect(self, manager, mock_websocket):
        """Test broadcasting handles disconnected clients."""
        mock_websocket.send_text = AsyncMock(side_effect=Exception("Disconnected"))
        await manager.connect(mock_websocket, channel="health")

        event = WebSocketEvent(
            type=EventType.HEALTH_UPDATE,
            timestamp=1234567890.0,
            data={"status": "healthy"},
        )

        # Should not raise
        await manager.broadcast(event, channels=["health"])

    @pytest.mark.asyncio
    async def test_broadcast_auto_detect_channels(self, manager):
        """Test broadcasting with auto-detected channels."""
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()

        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()

        await manager.connect(ws1, channel="events")
        await manager.connect(ws2, channel="memory")

        event = WebSocketEvent(
            type=EventType.MEMORY_ADDED,
            timestamp=1234567890.0,
            data={"mode": "warm_start"},
        )
        # Auto-detect should broadcast to events and memory
        await manager.broadcast(event)

        ws1.send_text.assert_called_once()
        ws2.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_count(self, manager):
        """Test counting connections per channel."""
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws2 = MagicMock()
        ws2.accept = AsyncMock()

        await manager.connect(ws1, channel="events")
        await manager.connect(ws2, channel="memory")

        counts = manager.connection_count
        assert counts["events"] == 1
        assert counts["memory"] == 1

    @pytest.mark.asyncio
    async def test_connections_stored_correctly(self, manager, mock_websocket):
        """Test connections are stored in correct channel."""
        await manager.connect(mock_websocket, channel="health")

        assert mock_websocket in manager._connections["health"]
        assert mock_websocket not in manager._connections["events"]


# =============================================================================
# Test Event Helpers
# =============================================================================


class TestEventHelpers:
    """Tests for event creation helper functions."""

    def test_create_memory_event(self):
        """Test creating a memory event."""
        import time
        event = WebSocketEvent(
            type=EventType.MEMORY_ADDED,
            timestamp=time.time(),
            data={
                "memory_type": "episodic",
                "session_id": "sess-123",
                "content_preview": "User asked about...",
            },
        )
        assert event.type == EventType.MEMORY_ADDED
        assert event.data["memory_type"] == "episodic"

    def test_create_learning_event(self):
        """Test creating a learning event."""
        import time
        event = WebSocketEvent(
            type=EventType.GATE_UPDATED,
            timestamp=time.time(),
            data={
                "old_weights": [0.3, 0.3, 0.4],
                "new_weights": [0.35, 0.3, 0.35],
                "delta": 0.05,
            },
        )
        assert event.type == EventType.GATE_UPDATED
        assert "delta" in event.data

    def test_create_health_event(self):
        """Test creating a health event."""
        import time
        event = WebSocketEvent(
            type=EventType.HEALTH_WARNING,
            timestamp=time.time(),
            data={
                "component": "qdrant",
                "message": "High latency detected",
                "latency_ms": 500,
            },
        )
        assert event.type == EventType.HEALTH_WARNING
        assert event.data["latency_ms"] == 500
