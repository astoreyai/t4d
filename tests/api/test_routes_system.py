"""Tests for system routes."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from t4dm.api.routes.system import (
    HealthResponse,
    StatsResponse,
    ConsolidationRequest,
    ConsolidationResponse,
    router,
)


class TestHealthResponse:
    """Tests for HealthResponse model."""

    def test_valid_response(self):
        """Valid health response is created."""
        resp = HealthResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00",
            version="0.5.0",
            session_id="test",
        )
        assert resp.status == "healthy"
        assert resp.version == "0.5.0"

    def test_optional_session(self):
        """Session ID is optional."""
        resp = HealthResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00",
            version="0.5.0",
        )
        assert resp.session_id is None


class TestStatsResponse:
    """Tests for StatsResponse model."""

    def test_valid_response(self):
        """Valid stats response is created."""
        resp = StatsResponse(
            session_id="test",
            episodic={"total_episodes": 10},
            semantic={"total_entities": 5},
            procedural={"total_skills": 3},
        )
        assert resp.session_id == "test"
        assert resp.episodic["total_episodes"] == 10


class TestConsolidationModels:
    """Tests for consolidation models."""

    def test_request_defaults(self):
        """Request has correct defaults."""
        req = ConsolidationRequest()
        assert req.deep is False

    def test_request_deep(self):
        """Deep flag can be set."""
        req = ConsolidationRequest(deep=True)
        assert req.deep is True

    def test_response_structure(self):
        """Response has required fields."""
        resp = ConsolidationResponse(
            success=True,
            type="light",
            results={"duplicates": 0},
        )
        assert resp.success is True
        assert resp.type == "light"


class TestHealthCheckEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_healthy(self):
        """Health check returns healthy status."""
        from t4dm.api.routes.system import health_check

        result = await health_check(session_id="test")

        assert result.status == "healthy"
        assert result.version == "0.5.0"
        assert result.session_id == "test"

    @pytest.mark.asyncio
    async def test_health_includes_timestamp(self):
        """Health check includes timestamp."""
        from t4dm.api.routes.system import health_check

        result = await health_check()

        assert result.timestamp is not None
        assert len(result.timestamp) > 0


class TestStatsEndpoint:
    """Tests for stats endpoint."""

    @pytest.mark.asyncio
    async def test_stats_returns_counts(self):
        """Stats returns memory counts."""
        from t4dm.api.routes.system import get_stats

        mock_episodic = MagicMock()
        mock_episodic.count = AsyncMock(return_value=10)

        mock_semantic = MagicMock()
        mock_semantic.count_entities = AsyncMock(return_value=5)
        mock_semantic.count_relations = AsyncMock(return_value=3)

        mock_procedural = MagicMock()
        mock_procedural.count = AsyncMock(return_value=2)

        services = {
            "session_id": "test",
            "episodic": mock_episodic,
            "semantic": mock_semantic,
            "procedural": mock_procedural,
        }

        result = await get_stats(services)

        assert result.session_id == "test"
        assert result.episodic["total_episodes"] == 10
        assert result.semantic["total_entities"] == 5
        assert result.procedural["total_skills"] == 2

    @pytest.mark.asyncio
    async def test_stats_handles_missing_methods(self):
        """Stats handles services without count methods."""
        from t4dm.api.routes.system import get_stats

        # Services without count methods
        mock_episodic = MagicMock(spec=[])
        mock_semantic = MagicMock(spec=[])
        mock_procedural = MagicMock(spec=[])

        services = {
            "session_id": "test",
            "episodic": mock_episodic,
            "semantic": mock_semantic,
            "procedural": mock_procedural,
        }

        result = await get_stats(services)

        assert result.episodic["total_episodes"] == 0
        assert result.semantic["total_entities"] == 0
        assert result.procedural["total_skills"] == 0


class TestConsolidateEndpoint:
    """Tests for consolidate endpoint."""

    @pytest.mark.asyncio
    async def test_light_consolidation(self):
        """Light consolidation is triggered."""
        from t4dm.api.routes.system import consolidate_memory

        mock_service = MagicMock()
        mock_service.light_consolidate = AsyncMock(
            return_value={"duplicates_found": 0}
        )

        with patch(
            "t4dm.api.routes.system.get_consolidation_service",
            return_value=mock_service,
        ):
            request = ConsolidationRequest(deep=False)
            services = {"session_id": "test"}

            # Pass admin_auth=True (simulating authenticated request)
            result = await consolidate_memory(request, services, _=True)

            assert result.success is True
            assert result.type == "light"
            mock_service.light_consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_deep_consolidation(self):
        """Deep consolidation is triggered."""
        from t4dm.api.routes.system import consolidate_memory

        mock_service = MagicMock()
        mock_service.deep_consolidate = AsyncMock(
            return_value={"clusters_formed": 5}
        )

        with patch(
            "t4dm.api.routes.system.get_consolidation_service",
            return_value=mock_service,
        ):
            request = ConsolidationRequest(deep=True)
            services = {"session_id": "test"}

            # Pass admin_auth=True (simulating authenticated request)
            result = await consolidate_memory(request, services, _=True)

            assert result.success is True
            assert result.type == "deep"
            mock_service.deep_consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidation_error(self):
        """Consolidation error is handled."""
        from t4dm.api.routes.system import consolidate_memory
        from fastapi import HTTPException

        mock_service = MagicMock()
        mock_service.light_consolidate = AsyncMock(
            side_effect=Exception("Consolidation failed")
        )

        with patch(
            "t4dm.api.routes.system.get_consolidation_service",
            return_value=mock_service,
        ):
            request = ConsolidationRequest(deep=False)
            services = {"session_id": "test"}

            with pytest.raises(HTTPException) as exc:
                # Pass admin_auth=True (simulating authenticated request)
                await consolidate_memory(request, services, _=True)
            assert exc.value.status_code == 500


class TestSessionEndpoint:
    """Tests for session info endpoint."""

    @pytest.mark.asyncio
    async def test_session_info(self):
        """Session info is returned with storage backend."""
        from t4dm.api.routes.system import get_session_info

        with patch("t4dm.api.routes.system.get_settings") as mock_settings:
            settings = MagicMock()
            settings.session_id = "configured-session"
            mock_settings.return_value = settings

            result = await get_session_info(session_id="custom-session")

            assert result["session_id"] == "custom-session"
            assert result["configured_session"] == "configured-session"
            assert result["storage_backend"] == "t4dx_embedded"
