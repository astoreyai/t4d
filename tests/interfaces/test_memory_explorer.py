"""Tests for memory explorer module."""

import pytest
from datetime import datetime
from uuid import UUID
from unittest.mock import AsyncMock, MagicMock, patch


class TestMemoryExplorer:
    """Tests for MemoryExplorer class."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock storage backends."""
        with patch("t4dm.interfaces.memory_explorer.EpisodicMemory") as mock_ep, \
             patch("t4dm.interfaces.memory_explorer.SemanticMemory") as mock_sem, \
             patch("t4dm.interfaces.memory_explorer.ProceduralMemory") as mock_proc:

            mock_ep_instance = MagicMock()
            mock_ep_instance.initialize = AsyncMock()
            mock_ep_instance.vector_store = MagicMock()
            mock_ep_instance.vector_store.episodes_collection = "episodes"
            mock_ep_instance.vector_store.scroll = AsyncMock(return_value=([], None))
            mock_ep_instance.vector_store.get = AsyncMock(return_value=[])
            mock_ep_instance.recall = AsyncMock(return_value=[])
            mock_ep.return_value = mock_ep_instance

            mock_sem_instance = MagicMock()
            mock_sem_instance.initialize = AsyncMock()
            mock_sem_instance.vector_store = MagicMock()
            mock_sem_instance.vector_store.entities_collection = "entities"
            mock_sem_instance.vector_store.scroll = AsyncMock(return_value=([], None))
            mock_sem_instance.get_entity = AsyncMock(return_value=None)
            mock_sem_instance.recall = AsyncMock(return_value=[])
            mock_sem_instance.graph_store = MagicMock()
            mock_sem_instance.graph_store.get_relationships = AsyncMock(return_value=[])
            mock_sem.return_value = mock_sem_instance

            mock_proc_instance = MagicMock()
            mock_proc_instance.initialize = AsyncMock()
            mock_proc_instance.vector_store = MagicMock()
            mock_proc_instance.vector_store.procedures_collection = "procedures"
            mock_proc_instance.vector_store.scroll = AsyncMock(return_value=([], None))
            mock_proc_instance.recall_skill = AsyncMock(return_value=[])
            mock_proc.return_value = mock_proc_instance

            yield {
                "episodic": mock_ep_instance,
                "semantic": mock_sem_instance,
                "procedural": mock_proc_instance,
            }

    def test_init_without_rich(self, mock_stores):
        """Test initialization fails without rich library."""
        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", False):
            from t4dm.interfaces.memory_explorer import MemoryExplorer

            with pytest.raises(ImportError, match="rich library required"):
                MemoryExplorer(session_id="test")

    def test_init_with_rich(self, mock_stores):
        """Test initialization with rich library."""
        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console"):
            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            assert explorer.session_id == "test"
            assert explorer._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, mock_stores):
        """Test initialization of storage backends."""
        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console"), \
             patch("t4dm.interfaces.memory_explorer.Progress"):
            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.initialize()

            assert explorer._initialized is True
            mock_stores["episodic"].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_episodes(self, mock_stores):
        """Test listing episodes."""
        now = datetime.now()
        mock_stores["episodic"].vector_store.scroll = AsyncMock(return_value=([
            ("id1", {
                "timestamp": now.isoformat(),
                "outcome": "success",
                "emotional_valence": 0.8,
                "access_count": 5,
                "content": "Test episode content",
            }, None),
        ], None))

        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console") as mock_console, \
             patch("t4dm.interfaces.memory_explorer.Progress"), \
             patch("t4dm.interfaces.memory_explorer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.list_episodes(limit=10)

            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_list_episodes_with_filters(self, mock_stores):
        """Test listing episodes with filters."""
        now = datetime.now()
        mock_stores["episodic"].vector_store.scroll = AsyncMock(return_value=([
            ("id1", {
                "timestamp": now.isoformat(),
                "outcome": "success",
                "emotional_valence": 0.8,
                "access_count": 5,
                "content": "Test episode",
            }, None),
        ], None))

        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console") as mock_console, \
             patch("t4dm.interfaces.memory_explorer.Progress"), \
             patch("t4dm.interfaces.memory_explorer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.list_episodes(
                limit=10,
                session_filter="session-1",
                outcome_filter="success",
            )

            # Filter should be applied in scroll call
            scroll_call = mock_stores["episodic"].vector_store.scroll.call_args
            assert scroll_call is not None

    @pytest.mark.asyncio
    async def test_view_episode(self, mock_stores):
        """Test viewing episode details."""
        now = datetime.now()
        mock_stores["episodic"].vector_store.get = AsyncMock(return_value=[
            ("12345678-1234-1234-1234-123456789012", {
                "session_id": "test",
                "content": "Test episode content",
                "timestamp": now.isoformat(),
                "outcome": "success",
                "emotional_valence": 0.8,
                "context": {},
                "access_count": 5,
                "last_accessed": now.isoformat(),
                "stability": 2.0,
            }),
        ])

        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console") as mock_console, \
             patch("t4dm.interfaces.memory_explorer.Progress"), \
             patch("t4dm.interfaces.memory_explorer.Layout"), \
             patch("t4dm.interfaces.memory_explorer.Panel"), \
             patch("t4dm.interfaces.memory_explorer.Table"), \
             patch("t4dm.interfaces.memory_explorer.Text"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.view_episode("12345678-1234-1234-1234-123456789012")

            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_view_episode_not_found(self, mock_stores):
        """Test viewing non-existent episode."""
        mock_stores["episodic"].vector_store.get = AsyncMock(return_value=[])

        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console") as mock_console, \
             patch("t4dm.interfaces.memory_explorer.Progress"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.view_episode("nonexistent-id")

            # Should print not found message
            calls = [str(call) for call in mock_console_instance.print.call_args_list]
            assert any("not found" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_list_entities(self, mock_stores):
        """Test listing entities."""
        mock_stores["semantic"].vector_store.scroll = AsyncMock(return_value=([
            ("id1", {
                "name": "Test Entity",
                "entity_type": "CONCEPT",
                "access_count": 10,
                "stability": 2.5,
                "summary": "A test entity for testing",
            }, None),
        ], None))

        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console") as mock_console, \
             patch("t4dm.interfaces.memory_explorer.Progress"), \
             patch("t4dm.interfaces.memory_explorer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.list_entities(limit=10)

            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_view_entity_graph(self, mock_stores):
        """Test viewing entity graph."""
        mock_entity = MagicMock()
        mock_entity.id = UUID("12345678-1234-1234-1234-123456789012")
        mock_entity.name = "Test Entity"
        mock_entity.entity_type = MagicMock()
        mock_entity.entity_type.value = "CONCEPT"
        mock_entity.summary = "A test entity"
        mock_entity.access_count = 5
        mock_entity.stability = 2.0

        mock_stores["semantic"].get_entity = AsyncMock(return_value=mock_entity)
        mock_stores["semantic"].graph_store.get_relationships = AsyncMock(return_value=[
            {"other_id": "other-id", "type": "RELATED", "properties": {"weight": 0.8}},
        ])

        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console") as mock_console, \
             patch("t4dm.interfaces.memory_explorer.Progress"), \
             patch("t4dm.interfaces.memory_explorer.Tree"), \
             patch("t4dm.interfaces.memory_explorer.Panel"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.view_entity_graph("12345678-1234-1234-1234-123456789012")

            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_list_skills(self, mock_stores):
        """Test listing skills."""
        mock_stores["procedural"].vector_store.scroll = AsyncMock(return_value=([
            ("id1", {
                "name": "Test Skill",
                "domain": "coding",
                "success_rate": 0.9,
                "execution_count": 20,
                "steps": [{"action": "do something"}],
            }, None),
        ], None))

        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console") as mock_console, \
             patch("t4dm.interfaces.memory_explorer.Progress"), \
             patch("t4dm.interfaces.memory_explorer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.list_skills(limit=10)

            mock_console_instance.print.assert_called()

    @pytest.mark.asyncio
    async def test_search(self, mock_stores):
        """Test searching all memory types."""
        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.item = MagicMock()
        mock_result.item.content = "Test content"
        mock_result.item.timestamp = datetime.now()

        mock_stores["episodic"].recall = AsyncMock(return_value=[mock_result])
        mock_stores["semantic"].recall = AsyncMock(return_value=[])
        mock_stores["procedural"].recall_skill = AsyncMock(return_value=[])

        with patch("t4dm.interfaces.memory_explorer.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.memory_explorer.Console") as mock_console, \
             patch("t4dm.interfaces.memory_explorer.Progress"), \
             patch("t4dm.interfaces.memory_explorer.Table"):
            mock_console_instance = MagicMock()
            mock_console.return_value = mock_console_instance
            mock_console_instance.status = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))

            from t4dm.interfaces.memory_explorer import MemoryExplorer

            explorer = MemoryExplorer(session_id="test")
            await explorer.search("test query", limit=5)

            # Should search all memory types
            mock_stores["episodic"].recall.assert_called_once()
            mock_stores["semantic"].recall.assert_called_once()
            mock_stores["procedural"].recall_skill.assert_called_once()
