"""Tests for CRUD manager module."""

import pytest
from datetime import datetime
from uuid import UUID
from unittest.mock import AsyncMock, MagicMock, patch


class TestCRUDManager:
    """Tests for CRUDManager class."""

    @pytest.fixture
    def mock_stores(self):
        """Create mock storage backends."""
        with patch("t4dm.interfaces.crud_manager.EpisodicMemory") as mock_ep, \
             patch("t4dm.interfaces.crud_manager.SemanticMemory") as mock_sem, \
             patch("t4dm.interfaces.crud_manager.ProceduralMemory") as mock_proc:

            mock_ep_instance = MagicMock()
            mock_ep_instance.initialize = AsyncMock()
            mock_ep_instance.vector_store = MagicMock()
            mock_ep_instance.vector_store.episodes_collection = "episodes"
            mock_ep_instance.vector_store.get = AsyncMock(return_value=[])
            mock_ep_instance.vector_store.delete = AsyncMock()
            mock_ep_instance.store = AsyncMock()
            mock_ep.return_value = mock_ep_instance

            mock_sem_instance = MagicMock()
            mock_sem_instance.initialize = AsyncMock()
            mock_sem_instance.vector_store = MagicMock()
            mock_sem_instance.vector_store.entities_collection = "entities"
            mock_sem_instance.vector_store.delete = AsyncMock()
            mock_sem_instance.graph_store = MagicMock()
            mock_sem_instance.graph_store.delete_node = AsyncMock()
            mock_sem_instance.graph_store.get_relationships = AsyncMock(return_value=[])
            mock_sem_instance.create_entity = AsyncMock()
            mock_sem_instance.get_entity = AsyncMock(return_value=None)
            mock_sem_instance.supersede = AsyncMock()
            mock_sem.return_value = mock_sem_instance

            mock_proc_instance = MagicMock()
            mock_proc_instance.initialize = AsyncMock()
            mock_proc_instance.vector_store = MagicMock()
            mock_proc_instance.vector_store.procedures_collection = "procedures"
            mock_proc_instance.vector_store.add = AsyncMock()
            mock_proc_instance.graph_store = MagicMock()
            mock_proc_instance.graph_store.create_node = AsyncMock()
            mock_proc_instance.get_procedure = AsyncMock(return_value=None)
            mock_proc_instance.update = AsyncMock()
            mock_proc_instance.deprecate = AsyncMock()
            mock_proc_instance._to_payload = MagicMock(return_value={})
            mock_proc_instance._to_graph_props = MagicMock(return_value={})
            mock_proc.return_value = mock_proc_instance

            yield {
                "episodic": mock_ep_instance,
                "semantic": mock_sem_instance,
                "procedural": mock_proc_instance,
            }

    def test_init_without_rich(self, mock_stores):
        """Test initialization fails without rich library."""
        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", False):
            from t4dm.interfaces.crud_manager import CRUDManager

            with pytest.raises(ImportError, match="rich library required"):
                CRUDManager(session_id="test")

    def test_init_with_rich(self, mock_stores):
        """Test initialization with rich library."""
        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            assert manager.session_id == "test"
            assert manager._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, mock_stores):
        """Test initialization of storage backends."""
        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            await manager.initialize()

            assert manager._initialized is True
            mock_stores["episodic"].initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_episode(self, mock_stores):
        """Test creating an episode."""
        mock_episode = MagicMock()
        mock_episode.id = UUID("12345678-1234-1234-1234-123456789012")
        mock_stores["episodic"].store = AsyncMock(return_value=mock_episode)

        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            episode = await manager.create_episode(
                content="Test episode content",
                outcome="success",
                emotional_valence=0.8,
            )

            assert episode.id == mock_episode.id
            mock_stores["episodic"].store.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_entity(self, mock_stores):
        """Test creating an entity."""
        mock_entity = MagicMock()
        mock_entity.id = UUID("12345678-1234-1234-1234-123456789012")
        mock_stores["semantic"].create_entity = AsyncMock(return_value=mock_entity)

        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            entity = await manager.create_entity(
                name="Test Entity",
                entity_type="CONCEPT",
                summary="A test entity",
            )

            assert entity.id == mock_entity.id
            mock_stores["semantic"].create_entity.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_episode(self, mock_stores):
        """Test getting an episode by ID."""
        mock_stores["episodic"].vector_store.get = AsyncMock(return_value=[
            ("12345678-1234-1234-1234-123456789012", {
                "session_id": "test",
                "content": "Test content",
                "timestamp": "2024-01-01T12:00:00",
                "outcome": "success",
                "emotional_valence": 0.8,
                "context": {},
                "access_count": 1,
                "last_accessed": "2024-01-01T12:00:00",
                "stability": 1.0,
            }),
        ])

        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            episode = await manager.get_episode("12345678-1234-1234-1234-123456789012")

            assert episode is not None
            assert episode.content == "Test content"

    @pytest.mark.asyncio
    async def test_get_episode_not_found(self, mock_stores):
        """Test getting non-existent episode."""
        mock_stores["episodic"].vector_store.get = AsyncMock(return_value=[])

        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            episode = await manager.get_episode("nonexistent-id")

            assert episode is None

    @pytest.mark.asyncio
    async def test_update_entity(self, mock_stores):
        """Test updating an entity."""
        mock_new_entity = MagicMock()
        mock_new_entity.id = UUID("22222222-2222-2222-2222-222222222222")
        mock_stores["semantic"].supersede = AsyncMock(return_value=mock_new_entity)

        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            new_entity = await manager.update_entity(
                entity_id="12345678-1234-1234-1234-123456789012",
                new_summary="Updated summary",
            )

            assert new_entity.id == mock_new_entity.id
            mock_stores["semantic"].supersede.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_episode_without_confirm(self, mock_stores):
        """Test deleting episode without confirmation."""
        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            result = await manager.delete_episode("episode-id", confirm=False)

            assert result is True
            mock_stores["episodic"].vector_store.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_entity_without_confirm(self, mock_stores):
        """Test deleting entity without confirmation."""
        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress"):
            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            result = await manager.delete_entity("entity-id", confirm=False)

            assert result is True
            mock_stores["semantic"].vector_store.delete.assert_called_once()
            mock_stores["semantic"].graph_store.delete_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_create_episodes(self, mock_stores):
        """Test batch creating episodes."""
        mock_episode = MagicMock()
        mock_episode.id = UUID("12345678-1234-1234-1234-123456789012")
        mock_stores["episodic"].store = AsyncMock(return_value=mock_episode)

        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress") as mock_progress:
            mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress.return_value)
            mock_progress.return_value.__exit__ = MagicMock(return_value=False)
            mock_progress.return_value.add_task = MagicMock(return_value=0)
            mock_progress.return_value.advance = MagicMock()

            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            episodes = await manager.batch_create_episodes([
                {"content": "Episode 1"},
                {"content": "Episode 2"},
                {"content": "Episode 3"},
            ])

            assert len(episodes) == 3
            assert mock_stores["episodic"].store.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_delete_episodes_without_confirm(self, mock_stores):
        """Test batch deleting episodes."""
        with patch("t4dm.interfaces.crud_manager.RICH_AVAILABLE", True), \
             patch("t4dm.interfaces.crud_manager.Console"), \
             patch("t4dm.interfaces.crud_manager.Progress") as mock_progress:
            mock_progress.return_value.__enter__ = MagicMock(return_value=mock_progress.return_value)
            mock_progress.return_value.__exit__ = MagicMock(return_value=False)
            mock_progress.return_value.add_task = MagicMock(return_value=0)
            mock_progress.return_value.update = MagicMock()

            from t4dm.interfaces.crud_manager import CRUDManager

            manager = CRUDManager(session_id="test")
            count = await manager.batch_delete_episodes(
                ["id1", "id2", "id3"],
                confirm=False,
            )

            assert count == 3
            mock_stores["episodic"].vector_store.delete.assert_called_once()
