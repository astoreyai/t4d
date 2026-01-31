"""
Tests for persistence API routes.

Tests endpoints for:
- System status
- Checkpoint management
- WAL status
- Recovery info
- Health check
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from t4dm.api.routes.persistence import (
    router,
    SystemStatus,
    CheckpointInfo,
    WALStatus,
    RecoveryInfo,
    CheckpointRequest,
    CheckpointResponse,
)


# =============================================================================
# Test Models
# =============================================================================


class TestSystemStatus:
    """Tests for SystemStatus model."""

    def test_default_creation(self):
        """Test SystemStatus creation."""
        status = SystemStatus(
            started=True,
            mode="warm_start",
            current_lsn=1000,
            checkpoint_lsn=900,
            operations_since_checkpoint=100,
            uptime_seconds=3600.0,
            shutdown_requested=False,
        )
        assert status.started is True
        assert status.mode == "warm_start"
        assert status.current_lsn == 1000
        assert status.checkpoint_lsn == 900
        assert status.operations_since_checkpoint == 100

    def test_shutdown_status(self):
        """Test with shutdown requested."""
        status = SystemStatus(
            started=True,
            mode="warm_start",
            current_lsn=500,
            checkpoint_lsn=500,
            operations_since_checkpoint=0,
            uptime_seconds=100.0,
            shutdown_requested=True,
        )
        assert status.shutdown_requested is True


class TestCheckpointInfo:
    """Tests for CheckpointInfo model."""

    def test_creation(self):
        """Test CheckpointInfo creation."""
        info = CheckpointInfo(
            lsn=1000,
            timestamp=datetime.utcnow(),
            size_bytes=1024 * 1024,
            components=["buffer", "gate"],
        )
        assert info.lsn == 1000
        assert info.size_bytes == 1024 * 1024
        assert len(info.components) == 2


class TestWALStatus:
    """Tests for WALStatus model."""

    def test_creation(self):
        """Test WALStatus creation."""
        status = WALStatus(
            current_lsn=5000,
            checkpoint_lsn=4500,
            segment_count=5,
            total_size_bytes=1024 * 1024 * 10,
            oldest_segment=1,
            current_segment=5,
        )
        assert status.current_lsn == 5000
        assert status.segment_count == 5
        assert status.oldest_segment == 1


class TestRecoveryInfo:
    """Tests for RecoveryInfo model."""

    def test_success_creation(self):
        """Test successful recovery info."""
        info = RecoveryInfo(
            mode="warm_start",
            success=True,
            checkpoint_lsn=1000,
            wal_entries_replayed=50,
            components_restored={"buffer": True, "gate": True},
            errors=[],
            duration_seconds=2.5,
        )
        assert info.success is True
        assert info.mode == "warm_start"
        assert info.wal_entries_replayed == 50

    def test_failure_creation(self):
        """Test failed recovery info."""
        info = RecoveryInfo(
            mode="cold_start",
            success=False,
            checkpoint_lsn=0,
            wal_entries_replayed=0,
            components_restored={},
            errors=["Checkpoint corrupted", "WAL missing entries"],
            duration_seconds=0.5,
        )
        assert info.success is False
        assert len(info.errors) == 2


class TestCheckpointModels:
    """Tests for checkpoint request/response models."""

    def test_request_default(self):
        """Test checkpoint request defaults."""
        request = CheckpointRequest()
        assert request.force is False

    def test_request_force(self):
        """Test checkpoint request with force."""
        request = CheckpointRequest(force=True)
        assert request.force is True

    def test_response_success(self):
        """Test checkpoint response for success."""
        response = CheckpointResponse(
            success=True,
            lsn=1500,
            duration_seconds=1.2,
            message="Checkpoint created at LSN 1500",
        )
        assert response.success is True
        assert response.lsn == 1500

    def test_response_failure(self):
        """Test checkpoint response for failure."""
        response = CheckpointResponse(
            success=False,
            lsn=0,
            duration_seconds=0.1,
            message="Checkpoint failed: disk full",
        )
        assert response.success is False
        assert "failed" in response.message


# =============================================================================
# Test get_persistence dependency
# =============================================================================


class TestGetPersistence:
    """Tests for the get_persistence dependency."""

    def test_import_error_raises_503(self):
        """Test that import error raises 503."""
        from t4dm.api.routes.persistence import get_persistence
        from fastapi import HTTPException

        with patch.dict("sys.modules", {"t4dm.mcp.persistent_server": None}):
            with patch("t4dm.api.routes.persistence.get_persistence") as mock_get:
                mock_get.side_effect = HTTPException(
                    status_code=503,
                    detail="Persistence module not available"
                )
                with pytest.raises(HTTPException) as exc_info:
                    mock_get()
                assert exc_info.value.status_code == 503

    def test_none_persistence_raises_503(self):
        """Test that None persistence raises 503."""
        from t4dm.api.routes.persistence import get_persistence
        from fastapi import HTTPException

        with patch("t4dm.api.routes.persistence.get_persistence") as mock_get:
            mock_get.side_effect = HTTPException(
                status_code=503,
                detail="Persistence layer not initialized"
            )
            with pytest.raises(HTTPException) as exc_info:
                mock_get()
            assert exc_info.value.status_code == 503


# =============================================================================
# Test Routes with Mocked Persistence
# =============================================================================


@pytest.fixture
def mock_persistence():
    """Create mock persistence manager."""
    persistence = MagicMock()
    persistence.get_status.return_value = {
        "started": True,
        "recovery_mode": "warm_start",
        "current_lsn": 1000,
        "last_checkpoint_lsn": 900,
        "uptime_seconds": 3600,
        "shutdown_requested": False,
        "wal_entries_replayed": 100,
        "components_restored": {"buffer": True},
        "recovery_duration": 2.5,
    }
    persistence.current_lsn = 1000
    persistence.last_checkpoint_lsn = 900
    persistence.is_started = True
    persistence.should_shutdown = False
    persistence.config.data_directory = "/tmp/ww_test"

    # Mock checkpoint methods
    checkpoint = MagicMock()
    checkpoint.lsn = 1000
    checkpoint._last_checkpoint_time = 0  # Stale for health check
    persistence._checkpoint = checkpoint
    persistence.create_checkpoint = AsyncMock(return_value=checkpoint)
    persistence.truncate_wal = AsyncMock(return_value=5)

    return persistence


@pytest.fixture
def app_with_mock(mock_persistence):
    """Create FastAPI app with mocked persistence."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)

    # Override dependency
    def override_get_persistence():
        return mock_persistence

    from t4dm.api.routes.persistence import get_persistence
    app.dependency_overrides[get_persistence] = override_get_persistence

    return app


@pytest.fixture
def client(app_with_mock):
    """Create test client."""
    return TestClient(app_with_mock)


class TestSystemStatusRoute:
    """Tests for GET /api/v1/persistence/status."""

    def test_get_status_success(self, client, mock_persistence):
        """Test successful status retrieval."""
        response = client.get("/api/v1/persistence/status")
        assert response.status_code == 200

        data = response.json()
        assert data["started"] is True
        assert data["mode"] == "warm_start"
        assert data["current_lsn"] == 1000
        assert data["checkpoint_lsn"] == 900
        assert data["operations_since_checkpoint"] == 100

    def test_get_status_shutdown_requested(self, client, mock_persistence):
        """Test status when shutdown is requested."""
        mock_persistence.get_status.return_value = {
            "started": True,
            "recovery_mode": "warm_start",
            "current_lsn": 1000,
            "last_checkpoint_lsn": 1000,
            "uptime_seconds": 7200,
            "shutdown_requested": True,
        }
        response = client.get("/api/v1/persistence/status")
        assert response.status_code == 200
        assert response.json()["shutdown_requested"] is True


class TestCheckpointRoutes:
    """Tests for checkpoint routes."""

    def test_list_checkpoints_empty(self, client, mock_persistence):
        """Test listing checkpoints when none exist."""
        with patch("pathlib.Path.glob", return_value=[]):
            response = client.get("/api/v1/persistence/checkpoints")
            assert response.status_code == 200
            assert response.json() == []

    def test_create_checkpoint_success(self, client, mock_persistence):
        """Test creating checkpoint."""
        response = client.post(
            "/api/v1/persistence/checkpoint",
            json={"force": False}
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["lsn"] == 1000

    def test_create_checkpoint_with_force(self, client, mock_persistence):
        """Test creating checkpoint with force flag."""
        response = client.post(
            "/api/v1/persistence/checkpoint",
            json={"force": True}
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

    def test_create_checkpoint_failure(self, client, mock_persistence):
        """Test checkpoint creation failure."""
        mock_persistence.create_checkpoint = AsyncMock(
            side_effect=RuntimeError("Disk full")
        )
        response = client.post("/api/v1/persistence/checkpoint")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False
        assert "Disk full" in data["message"]


class TestWALRoutes:
    """Tests for WAL routes."""

    def test_get_wal_status_empty(self, client, mock_persistence):
        """Test WAL status when no segments exist."""
        with patch("pathlib.Path.glob", return_value=[]):
            response = client.get("/api/v1/persistence/wal")
            assert response.status_code == 200

            data = response.json()
            assert data["segment_count"] == 0
            assert data["total_size_bytes"] == 0

    def test_truncate_wal(self, client, mock_persistence):
        """Test WAL truncation."""
        response = client.post("/api/v1/persistence/wal/truncate")
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["segments_removed"] == 5


class TestRecoveryRoute:
    """Tests for GET /api/v1/persistence/recovery."""

    def test_get_recovery_info(self, client, mock_persistence):
        """Test getting recovery info."""
        response = client.get("/api/v1/persistence/recovery")
        assert response.status_code == 200

        data = response.json()
        assert data["mode"] == "warm_start"
        assert data["success"] is True
        assert data["wal_entries_replayed"] == 100


class TestHealthCheckRoute:
    """Tests for GET /api/v1/persistence/health."""

    def test_health_check_healthy(self, client, mock_persistence):
        """Test healthy system."""
        import time
        mock_persistence._checkpoint._last_checkpoint_time = time.time()
        mock_persistence.current_lsn = 1000
        mock_persistence.last_checkpoint_lsn = 999

        response = client.get("/api/v1/persistence/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["warnings"] == []

    def test_health_check_degraded_stale_checkpoint(self, client, mock_persistence):
        """Test degraded health due to stale checkpoint."""
        import time
        # Set checkpoint time to 15 minutes ago
        mock_persistence._checkpoint._last_checkpoint_time = time.time() - 900

        response = client.get("/api/v1/persistence/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "degraded"
        assert any("stale" in w.lower() for w in data["warnings"])

    def test_health_check_degraded_many_ops(self, client, mock_persistence):
        """Test degraded health due to many uncommitted operations."""
        import time
        mock_persistence._checkpoint._last_checkpoint_time = time.time()
        mock_persistence.current_lsn = 20000
        mock_persistence.last_checkpoint_lsn = 5000

        response = client.get("/api/v1/persistence/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "degraded"
        assert any("uncommitted" in w.lower() for w in data["warnings"])

    def test_health_check_not_started(self, client, mock_persistence):
        """Test health check when not started."""
        mock_persistence.is_started = False

        response = client.get("/api/v1/persistence/health")
        assert response.status_code == 503

    def test_health_check_shutdown_in_progress(self, client, mock_persistence):
        """Test health check during shutdown."""
        mock_persistence.should_shutdown = True

        response = client.get("/api/v1/persistence/health")
        assert response.status_code == 503
