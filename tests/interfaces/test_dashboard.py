"""Tests for system dashboard module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestSystemDashboard:
    """Tests for SystemDashboard class."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock storage backends."""
        with patch("t4dm.interfaces.dashboard.EpisodicMemory") as mock_ep, \
             patch("t4dm.interfaces.dashboard.SemanticMemory") as mock_sem, \
             patch("t4dm.interfaces.dashboard.ProceduralMemory") as mock_proc, \
             patch("t4dm.interfaces.dashboard.get_vector_store") as mock_qdrant, \
             patch("t4dm.interfaces.dashboard.get_graph_store") as mock_neo4j:

            mock_ep_instance = MagicMock()
            mock_ep_instance.initialize = AsyncMock()
            mock_ep.return_value = mock_ep_instance

            mock_sem_instance = MagicMock()
            mock_sem_instance.initialize = AsyncMock()
            mock_sem.return_value = mock_sem_instance

            mock_proc_instance = MagicMock()
            mock_proc_instance.initialize = AsyncMock()
            mock_proc.return_value = mock_proc_instance

            # Configure vector store
            mock_vs = MagicMock()
            mock_vs.episodes_collection = "episodes"
            mock_vs.entities_collection = "entities"
            mock_vs.procedures_collection = "procedures"
            mock_vs.scroll = AsyncMock(return_value=([], None))
            mock_vs.count = AsyncMock(return_value=10)
            mock_vs.circuit_breaker = MagicMock()
            mock_vs.circuit_breaker.state = MagicMock()
            mock_vs.circuit_breaker.state.name = "CLOSED"
            mock_vs.circuit_breaker.failure_count = 0
            mock_vs.circuit_breaker.success_count = 100
            mock_qdrant.return_value = mock_vs

            # Configure graph store
            mock_gs = MagicMock()
            mock_gs.circuit_breaker = MagicMock()
            mock_gs.circuit_breaker.state = MagicMock()
            mock_gs.circuit_breaker.state.name = "CLOSED"
            mock_gs.circuit_breaker.failure_count = 0
            mock_gs.circuit_breaker.success_count = 50
            mock_neo4j.return_value = mock_gs

            yield {
                "episodic": mock_ep_instance,
                "semantic": mock_sem_instance,
                "procedural": mock_proc_instance,
                "vector_store": mock_vs,
                "graph_store": mock_gs,
            }

    def test_init_without_rich(self, mock_stores):
        """Test initialization fails without rich library."""
        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", False):
            from t4dm.interfaces.dashboard import SystemDashboard

            with pytest.raises(ImportError, match="rich library required"):
                SystemDashboard(session_id="test")

    def test_init_with_rich(self, mock_stores):
        """Test initialization with rich library."""
        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.dashboard.Console") as mock_console:
            from t4dm.interfaces.dashboard import SystemDashboard

            dashboard = SystemDashboard(session_id="test")
            assert dashboard.session_id == "test"
            assert dashboard._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, mock_stores):
        """Test initialization of storage backends."""
        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.dashboard.Console"):
            from t4dm.interfaces.dashboard import SystemDashboard

            dashboard = SystemDashboard(session_id="test")
            await dashboard.initialize()

            assert dashboard._initialized is True
            mock_stores["episodic"].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_memory_counts(self, mock_stores):
        """Test getting memory counts."""
        mock_stores["vector_store"].count = AsyncMock(side_effect=[100, 50, 25])

        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.dashboard.Console"):
            from t4dm.interfaces.dashboard import SystemDashboard

            dashboard = SystemDashboard(session_id="test")
            await dashboard.initialize()

            counts = await dashboard.get_memory_counts()

            assert counts["episodes"] == 100
            assert counts["entities"] == 50
            assert counts["skills"] == 25

    @pytest.mark.asyncio
    async def test_get_storage_health(self, mock_stores):
        """Test getting storage health."""
        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.dashboard.Console"):
            from t4dm.interfaces.dashboard import SystemDashboard

            dashboard = SystemDashboard(session_id="test")
            await dashboard.initialize()

            health = await dashboard.get_storage_health()

            assert "t4dx_circuit_breaker" in health
            # neo4j circuit breaker removed
            assert health["t4dx_circuit_breaker"]["state"] == "CLOSED"
            # removed

    @pytest.mark.asyncio
    async def test_get_recent_activity(self, mock_stores):
        """Test getting recent activity."""
        now = datetime.now()
        mock_stores["vector_store"].scroll = AsyncMock(side_effect=[
            ([
                ("id1", {"timestamp": now.isoformat()}, None),
                ("id2", {"timestamp": (now - timedelta(hours=2)).isoformat()}, None),
            ], None),
            ([
                ("id3", {"name": "Entity 1", "access_count": 10}, None),
                ("id4", {"name": "Entity 2", "access_count": 5}, None),
            ], None),
        ])

        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.dashboard.Console"):
            from t4dm.interfaces.dashboard import SystemDashboard

            dashboard = SystemDashboard(session_id="test")
            await dashboard.initialize()

            activity = await dashboard.get_recent_activity()

            assert "episodes_last_hour" in activity
            assert activity["episodes_last_hour"] == 1  # Only one within last hour
            assert "top_entities" in activity

    @pytest.mark.asyncio
    async def test_get_performance_stats(self, mock_stores):
        """Test getting performance stats."""
        now = datetime.now()
        mock_stores["vector_store"].scroll = AsyncMock(side_effect=[
            ([
                ("id1", {"last_accessed": now.isoformat(), "stability": 2.0}, None),
            ], None),
            ([
                ("id2", {"success_rate": 0.9, "deprecated": False}, None),
            ], None),
        ])

        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.dashboard.Console"):
            from t4dm.interfaces.dashboard import SystemDashboard

            dashboard = SystemDashboard(session_id="test")
            await dashboard.initialize()

            stats = await dashboard.get_performance_stats()

            assert "avg_retrievability" in stats
            assert "min_retrievability" in stats
            assert "max_retrievability" in stats
            assert "avg_skill_success" in stats
            assert stats["avg_skill_success"] == 0.9

    @pytest.mark.asyncio
    async def test_get_performance_stats_empty(self, mock_stores):
        """Test performance stats with no data."""
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([], None))

        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.dashboard.Console"):
            from t4dm.interfaces.dashboard import SystemDashboard

            dashboard = SystemDashboard(session_id="test")
            await dashboard.initialize()

            stats = await dashboard.get_performance_stats()

            assert stats["avg_retrievability"] == 0.0
            assert stats["avg_skill_success"] == 0.0

    def test_build_layout(self, mock_stores):
        """Test layout building."""
        with patch("t4dm.interfaces.dashboard.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.dashboard.Console"), \
             patch("t4dm.interfaces.dashboard.Layout") as mock_layout:
            from t4dm.interfaces.dashboard import SystemDashboard

            dashboard = SystemDashboard(session_id="test")
            layout = dashboard._build_layout()

            # Layout should be created
            mock_layout.assert_called()
