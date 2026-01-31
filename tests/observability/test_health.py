"""Tests for health check module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from t4dm.observability.health import (
    HealthStatus,
    ComponentHealth,
    SystemHealth,
    HealthChecker,
    get_health_checker,
)


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """All status values exist."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_basic_creation(self):
        """Basic health creation."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.HEALTHY,
        )
        assert health.name == "test"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == ""

    def test_with_details(self):
        """Health with all fields."""
        health = ComponentHealth(
            name="database",
            status=HealthStatus.DEGRADED,
            message="High latency",
            latency_ms=150.5,
            details={"queries_per_sec": 100},
        )
        assert health.latency_ms == 150.5
        assert health.details["queries_per_sec"] == 100

    def test_to_dict(self):
        """Converts to dictionary."""
        health = ComponentHealth(
            name="cache",
            status=HealthStatus.HEALTHY,
            latency_ms=5.123,
        )
        d = health.to_dict()
        assert d["name"] == "cache"
        assert d["status"] == "healthy"
        assert d["latency_ms"] == 5.12
        assert "checked_at" in d

    def test_to_dict_none_latency(self):
        """Handles None latency."""
        health = ComponentHealth(
            name="test",
            status=HealthStatus.UNKNOWN,
        )
        d = health.to_dict()
        assert d["latency_ms"] is None


class TestSystemHealth:
    """Tests for SystemHealth dataclass."""

    def test_basic_creation(self):
        """Basic system health."""
        components = [
            ComponentHealth(name="db", status=HealthStatus.HEALTHY),
            ComponentHealth(name="cache", status=HealthStatus.HEALTHY),
        ]
        system = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=components,
        )
        assert len(system.components) == 2
        assert system.version == "1.0.0"

    def test_to_dict(self):
        """Converts to dictionary."""
        components = [
            ComponentHealth(name="db", status=HealthStatus.HEALTHY),
        ]
        system = SystemHealth(
            status=HealthStatus.HEALTHY,
            components=components,
            version="2.0.0",
            uptime_seconds=3600.5,
        )
        d = system.to_dict()
        assert d["status"] == "healthy"
        assert d["version"] == "2.0.0"
        assert d["uptime_seconds"] == 3600.5
        assert len(d["components"]) == 1
        assert "checked_at" in d


class TestHealthChecker:
    """Tests for HealthChecker class."""

    @pytest.fixture
    def checker(self):
        """Create health checker."""
        return HealthChecker(timeout=1.0)

    def test_initialization(self, checker):
        """Checker initializes correctly."""
        assert checker.timeout == 1.0
        assert checker._start_time is not None

    @pytest.mark.asyncio
    async def test_check_qdrant_healthy(self, checker):
        """Qdrant health check when healthy."""
        mock_store = MagicMock()
        mock_store.count = AsyncMock(return_value=100)
        mock_store.episodes_collection = "episodes"

        with patch("t4dm.storage.qdrant_store.get_qdrant_store", return_value=mock_store):
            result = await checker.check_qdrant()
            assert result.status == HealthStatus.HEALTHY
            assert result.name == "qdrant"
            assert result.latency_ms is not None

    @pytest.mark.asyncio
    async def test_check_qdrant_connection_error(self, checker):
        """Qdrant health check on connection error."""
        with patch("t4dm.storage.qdrant_store.get_qdrant_store", side_effect=ConnectionError("Refused")):
            result = await checker.check_qdrant()
            assert result.status == HealthStatus.UNHEALTHY
            assert "Connection failed" in result.message

    @pytest.mark.asyncio
    async def test_check_neo4j_healthy(self, checker):
        """Neo4j health check when healthy."""
        mock_store = MagicMock()
        mock_store.query = AsyncMock(return_value=[{"n": 1}])

        with patch("t4dm.storage.neo4j_store.get_neo4j_store", return_value=mock_store):
            result = await checker.check_neo4j()
            assert result.status == HealthStatus.HEALTHY
            assert result.name == "neo4j"

    @pytest.mark.asyncio
    async def test_check_neo4j_connection_error(self, checker):
        """Neo4j health check on connection error."""
        with patch("t4dm.storage.neo4j_store.get_neo4j_store", side_effect=ConnectionError("Refused")):
            result = await checker.check_neo4j()
            assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_embedding_healthy(self, checker):
        """Embedding health check when healthy."""
        mock_provider = MagicMock()
        mock_provider.embed_query = AsyncMock(return_value=[0.1] * 1024)

        with patch("t4dm.embedding.bge_m3.get_embedding_provider", return_value=mock_provider):
            result = await checker.check_embedding()
            assert result.status == HealthStatus.HEALTHY
            assert result.name == "embedding"
            assert result.details.get("dimension") == 1024

    @pytest.mark.asyncio
    async def test_check_embedding_unavailable(self, checker):
        """Embedding health check when unavailable."""
        with patch("t4dm.embedding.bge_m3.get_embedding_provider", side_effect=ImportError("No model")):
            result = await checker.check_embedding()
            assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_metrics_healthy(self, checker):
        """Metrics health check when healthy."""
        mock_metrics = MagicMock()
        mock_metrics.get_summary.return_value = {"total_operations": 1000}

        with patch("t4dm.observability.health.get_metrics", return_value=mock_metrics):
            result = await checker.check_metrics()
            assert result.status == HealthStatus.HEALTHY
            assert result.name == "metrics"

    @pytest.mark.asyncio
    async def test_check_all_all_healthy(self, checker):
        """Check all components all healthy."""
        healthy = ComponentHealth(name="test", status=HealthStatus.HEALTHY)

        with patch.object(checker, "check_qdrant", return_value=healthy):
            with patch.object(checker, "check_neo4j", return_value=healthy):
                with patch.object(checker, "check_embedding", return_value=healthy):
                    with patch.object(checker, "check_metrics", return_value=healthy):
                        result = await checker.check_all()
                        assert result.status == HealthStatus.HEALTHY
                        assert len(result.components) == 4

    @pytest.mark.asyncio
    async def test_check_all_with_degraded(self, checker):
        """Check all with one degraded."""
        healthy = ComponentHealth(name="healthy", status=HealthStatus.HEALTHY)
        degraded = ComponentHealth(name="degraded", status=HealthStatus.DEGRADED)

        with patch.object(checker, "check_qdrant", return_value=healthy):
            with patch.object(checker, "check_neo4j", return_value=degraded):
                with patch.object(checker, "check_embedding", return_value=healthy):
                    with patch.object(checker, "check_metrics", return_value=healthy):
                        result = await checker.check_all()
                        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_check_all_with_unhealthy(self, checker):
        """Check all with one unhealthy."""
        healthy = ComponentHealth(name="healthy", status=HealthStatus.HEALTHY)
        unhealthy = ComponentHealth(name="unhealthy", status=HealthStatus.UNHEALTHY)

        with patch.object(checker, "check_qdrant", return_value=unhealthy):
            with patch.object(checker, "check_neo4j", return_value=healthy):
                with patch.object(checker, "check_embedding", return_value=healthy):
                    with patch.object(checker, "check_metrics", return_value=healthy):
                        result = await checker.check_all()
                        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_liveness(self, checker):
        """Liveness check always returns True."""
        result = await checker.check_liveness()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_readiness_when_healthy(self, checker):
        """Readiness check when healthy."""
        with patch.object(checker, "check_all", return_value=SystemHealth(
            status=HealthStatus.HEALTHY,
            components=[],
        )):
            result = await checker.check_readiness()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_readiness_when_degraded(self, checker):
        """Readiness check when degraded - still ready."""
        with patch.object(checker, "check_all", return_value=SystemHealth(
            status=HealthStatus.DEGRADED,
            components=[],
        )):
            result = await checker.check_readiness()
            assert result is True

    @pytest.mark.asyncio
    async def test_check_readiness_when_unhealthy(self, checker):
        """Readiness check when unhealthy - not ready."""
        with patch.object(checker, "check_all", return_value=SystemHealth(
            status=HealthStatus.UNHEALTHY,
            components=[],
        )):
            result = await checker.check_readiness()
            assert result is False


class TestGetHealthChecker:
    """Tests for get_health_checker singleton."""

    def test_returns_singleton(self):
        """Returns same instance."""
        # Reset singleton
        import t4dm.observability.health as health_module
        health_module._health_checker = None

        h1 = get_health_checker()
        h2 = get_health_checker()
        assert h1 is h2

    def test_creates_new_if_none(self):
        """Creates new checker if none exists."""
        import t4dm.observability.health as health_module
        health_module._health_checker = None

        checker = get_health_checker()
        assert isinstance(checker, HealthChecker)
