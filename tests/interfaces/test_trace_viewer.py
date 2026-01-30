"""Tests for trace viewer module."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestTraceViewer:
    """Tests for TraceViewer class."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock storage backends."""
        with patch("ww.interfaces.trace_viewer.EpisodicMemory") as mock_ep, \
             patch("ww.interfaces.trace_viewer.SemanticMemory") as mock_sem, \
             patch("ww.interfaces.trace_viewer.get_qdrant_store") as mock_qdrant:

            mock_ep_instance = MagicMock()
            mock_ep_instance.initialize = AsyncMock()
            mock_ep.return_value = mock_ep_instance

            mock_sem_instance = MagicMock()
            mock_sem_instance.initialize = AsyncMock()
            mock_sem_instance.get_entity = AsyncMock(return_value=None)
            mock_sem_instance.graph_store = MagicMock()
            mock_sem_instance.graph_store.get_relationships = AsyncMock(return_value=[])
            mock_sem.return_value = mock_sem_instance

            mock_vs = MagicMock()
            mock_vs.episodes_collection = "episodes"
            mock_vs.entities_collection = "entities"
            mock_vs.scroll = AsyncMock(return_value=([], None))
            mock_qdrant.return_value = mock_vs

            yield {
                "episodic": mock_ep_instance,
                "semantic": mock_sem_instance,
                "vector_store": mock_vs,
            }

    def test_init_without_rich(self, mock_stores):
        """Test initialization fails without rich library."""
        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", False):
            from ww.interfaces.trace_viewer import TraceViewer

            with pytest.raises(ImportError, match="rich library required"):
                TraceViewer(session_id="test")

    def test_init_with_rich(self, mock_stores):
        """Test initialization with rich library."""
        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console"):
            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            assert viewer.session_id == "test"
            assert viewer._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, mock_stores):
        """Test initialization of storage backends."""
        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console"), \
             patch("ww.interfaces.trace_viewer.Progress"):
            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            await viewer.initialize()

            assert viewer._initialized is True
            mock_stores["episodic"].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_show_access_timeline(self, mock_stores):
        """Test showing access timeline."""
        now = datetime.now()
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {"last_accessed": now.isoformat(), "access_count": 5, "content": "Test episode"}, None),
            ("id2", {"last_accessed": (now - timedelta(hours=25)).isoformat(), "access_count": 3, "content": "Old episode"}, None),
        ], None))

        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console") as mock_console, \
             patch("ww.interfaces.trace_viewer.Progress"), \
             patch("ww.interfaces.trace_viewer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            await viewer.show_access_timeline(hours=24)

            # Should have printed access timeline
            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_show_decay_curves(self, mock_stores):
        """Test showing decay curves."""
        now = datetime.now()
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {
                "timestamp": now.isoformat(),
                "last_accessed": now.isoformat(),
                "stability": 2.0,
                "access_count": 5,
            }, None),
        ], None))

        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console") as mock_console, \
             patch("ww.interfaces.trace_viewer.Progress"), \
             patch("ww.interfaces.trace_viewer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            await viewer.show_decay_curves(sample_size=5)

            mock_console_instance.print.assert_called()

    def test_render_decay_curve(self, mock_stores):
        """Test rendering ASCII decay curve."""
        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console"):
            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")

            # High retrievability - should be green
            curve = viewer._render_decay_curve(0.9)
            assert "[green]" in curve

            # Medium retrievability - should be yellow
            curve = viewer._render_decay_curve(0.6)
            assert "[yellow]" in curve

            # Low retrievability - should be red
            curve = viewer._render_decay_curve(0.3)
            assert "[red]" in curve

    @pytest.mark.asyncio
    async def test_show_consolidation_events(self, mock_stores):
        """Test showing consolidation events."""
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {
                "name": "Consolidated Entity",
                "source": "episode-123",
                "created_at": datetime.now().isoformat(),
            }, None),
        ], None))

        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console") as mock_console, \
             patch("ww.interfaces.trace_viewer.Progress"), \
             patch("ww.interfaces.trace_viewer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            await viewer.show_consolidation_events(limit=10)

            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_show_consolidation_events_empty(self, mock_stores):
        """Test consolidation events with no data."""
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {"name": "User Entity", "source": "user_provided", "created_at": datetime.now().isoformat()}, None),
        ], None))

        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console") as mock_console, \
             patch("ww.interfaces.trace_viewer.Progress"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            await viewer.show_consolidation_events(limit=10)

            # Should indicate no consolidation events
            calls = [str(call) for call in mock_console_instance.print.call_args_list]
            assert any("No consolidation" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_show_access_heatmap(self, mock_stores):
        """Test showing access heatmap."""
        now = datetime.now()
        mock_stores["vector_store"].scroll = AsyncMock(return_value=([
            ("id1", {"last_accessed": now.isoformat()}, None),
            ("id2", {"last_accessed": (now - timedelta(hours=2)).isoformat()}, None),
        ], None))

        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console") as mock_console, \
             patch("ww.interfaces.trace_viewer.Progress"), \
             patch("ww.interfaces.trace_viewer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            await viewer.show_access_heatmap(hours=24, bucket_minutes=60)

            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_show_activation_history(self, mock_stores):
        """Test showing activation history."""
        mock_entity = MagicMock()
        mock_entity.id = "entity-id"
        mock_entity.name = "Test Entity"
        mock_stores["semantic"].get_entity = AsyncMock(return_value=mock_entity)
        mock_stores["semantic"].graph_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "other-id", "type": "RELATED", "properties": {"weight": 0.5, "coAccessCount": 10}},
        ])

        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console") as mock_console, \
             patch("ww.interfaces.trace_viewer.Progress"), \
             patch("ww.interfaces.trace_viewer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            await viewer.show_activation_history("entity-id")

            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_show_activation_history_not_found(self, mock_stores):
        """Test activation history for non-existent entity."""
        mock_stores["semantic"].get_entity = AsyncMock(return_value=None)

        with patch("ww.interfaces.trace_viewer.RICH_AVAILABLE", True), \
             patch("ww.interfaces.trace_viewer.Console") as mock_console, \
             patch("ww.interfaces.trace_viewer.Progress"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from ww.interfaces.trace_viewer import TraceViewer

            viewer = TraceViewer(session_id="test")
            await viewer.show_activation_history("nonexistent-id")

            # Should print not found message
            calls = [str(call) for call in mock_console_instance.print.call_args_list]
            assert any("not found" in str(c) for c in calls)
