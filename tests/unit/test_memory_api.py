"""
Tests for the simplified memory API.

Tests the ww.memory_api module and its exports from ww package.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from ww.memory_api import Memory, MemoryResult, memory


class TestMemoryResult:
    """Tests for MemoryResult class."""

    def test_create_memory_result(self):
        """Creates MemoryResult with all fields."""
        result = MemoryResult(
            content="test content",
            memory_type="episodic",
            score=0.95,
            id="test-id",
            metadata={"key": "value"},
        )

        assert result.content == "test content"
        assert result.memory_type == "episodic"
        assert result.score == 0.95
        assert result.id == "test-id"
        assert result.metadata == {"key": "value"}

    def test_create_memory_result_defaults(self):
        """Creates MemoryResult with default values."""
        result = MemoryResult(
            content="test",
            memory_type="semantic",
        )

        assert result.content == "test"
        assert result.memory_type == "semantic"
        assert result.score == 0.0
        assert result.id is None
        assert result.metadata == {}

    def test_memory_result_repr(self):
        """MemoryResult has informative repr."""
        result = MemoryResult(
            content="This is a longer content string for testing repr",
            memory_type="procedural",
            score=0.75,
        )

        repr_str = repr(result)
        assert "procedural" in repr_str
        assert "0.75" in repr_str


class TestMemoryClass:
    """Tests for Memory class."""

    def test_memory_init_default_session(self):
        """Memory initializes with default session."""
        m = Memory()
        assert m._session_id is None

    def test_memory_init_custom_session(self):
        """Memory initializes with custom session."""
        m = Memory(session_id="custom-session")
        assert m._session_id == "custom-session"

    def test_memory_session_property_custom(self):
        """session_id property returns custom session."""
        m = Memory(session_id="custom")
        assert m.session_id == "custom"

    @patch("ww.memory_api.get_settings")
    def test_memory_session_property_default(self, mock_settings):
        """session_id property returns default from settings."""
        mock_settings.return_value.session_id = "settings-session"
        m = Memory()
        assert m.session_id == "settings-session"


class TestMemoryStore:
    """Tests for Memory store methods."""

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_store_content(self, mock_services):
        """store() stores episodic content."""
        mock_episodic = MagicMock()
        mock_episodic.add_episode = AsyncMock(return_value=MagicMock(id="ep-123"))
        mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

        m = Memory(session_id="test-session")
        result = await m.store("Test content", importance=0.8)

        assert result == "ep-123"
        mock_episodic.add_episode.assert_called_once()
        call_args = mock_episodic.add_episode.call_args[0][0]
        assert call_args.content == "Test content"
        assert call_args.emotional_valence == 0.8

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_store_episode(self, mock_services):
        """store_episode() stores episodic memory."""
        mock_episodic = MagicMock()
        mock_episodic.add_episode = AsyncMock(return_value=MagicMock(id="ep-456"))
        mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

        m = Memory(session_id="test")
        result = await m.store_episode("Episode content", importance=0.5)

        assert result == "ep-456"

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_store_entity(self, mock_services):
        """store_entity() stores semantic entity."""
        mock_semantic = MagicMock()
        mock_semantic.add_entity = AsyncMock(return_value=MagicMock(id="ent-789"))
        mock_services.return_value = (MagicMock(), mock_semantic, MagicMock())

        m = Memory(session_id="test")
        result = await m.store_entity(
            "Python",
            description="A programming language",
            entity_type="concept",
        )

        assert result == "ent-789"
        mock_semantic.add_entity.assert_called_once()
        call_args = mock_semantic.add_entity.call_args[0][0]
        assert call_args.name == "Python"
        assert call_args.summary == "A programming language"

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_store_entity_type_mapping(self, mock_services):
        """store_entity() maps string types to enums."""
        mock_semantic = MagicMock()
        mock_semantic.add_entity = AsyncMock(return_value=MagicMock(id="ent-123"))
        mock_services.return_value = (MagicMock(), mock_semantic, MagicMock())

        m = Memory(session_id="test")

        for type_str in ["concept", "person", "project", "tool", "technique", "fact"]:
            await m.store_entity(f"Test {type_str}", entity_type=type_str)

        assert mock_semantic.add_entity.call_count == 6

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_store_skill(self, mock_services):
        """store_skill() stores procedural skill."""
        mock_procedural = MagicMock()
        mock_procedural.add_skill = AsyncMock(return_value=MagicMock(id="sk-101"))
        mock_services.return_value = (MagicMock(), MagicMock(), mock_procedural)

        m = Memory(session_id="test")
        result = await m.store_skill(
            "git_commit",
            script="git add . && git commit -m 'msg'",
            domain="coding",
        )

        assert result == "sk-101"
        mock_procedural.add_skill.assert_called_once()
        call_args = mock_procedural.add_skill.call_args[0][0]
        assert call_args.name == "git_commit"
        assert "git add" in call_args.script

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_store_skill_domain_mapping(self, mock_services):
        """store_skill() maps string domains to enums."""
        mock_procedural = MagicMock()
        mock_procedural.add_skill = AsyncMock(return_value=MagicMock(id="sk-123"))
        mock_services.return_value = (MagicMock(), MagicMock(), mock_procedural)

        m = Memory(session_id="test")

        for domain_str in ["coding", "research", "trading", "devops", "writing"]:
            await m.store_skill(f"test_{domain_str}", "script", domain=domain_str)

        assert mock_procedural.add_skill.call_count == 5


class TestMemoryRecall:
    """Tests for Memory recall methods."""

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_recall_all_types(self, mock_services):
        """recall() searches all memory types."""
        # Setup mocks
        mock_episodic = MagicMock()
        mock_ep_result = MagicMock()
        mock_ep_result.item.content = "Episode content"
        mock_ep_result.item.id = "ep-1"
        mock_ep_result.item.timestamp = datetime.utcnow()
        mock_ep_result.item.emotional_valence = 0.5
        mock_ep_result.score = 0.9
        mock_episodic.recall_similar = AsyncMock(return_value=[mock_ep_result])

        mock_semantic = MagicMock()
        mock_ent = MagicMock()
        mock_ent.name = "Entity"
        mock_ent.summary = "Entity summary"
        mock_ent.id = "ent-1"
        mock_ent.entity_type = "concept"
        mock_semantic.search_similar = AsyncMock(return_value=[mock_ent])

        mock_procedural = MagicMock()
        mock_skill = MagicMock()
        mock_skill.name = "Skill"
        mock_skill.script = "script content"
        mock_skill.id = "sk-1"
        mock_skill.domain = "coding"
        mock_procedural.find_relevant_skills = AsyncMock(return_value=[mock_skill])

        mock_services.return_value = (mock_episodic, mock_semantic, mock_procedural)

        m = Memory(session_id="test")
        results = await m.recall("query", limit=5)

        assert len(results) == 3
        types = [r.memory_type for r in results]
        assert "episodic" in types
        assert "semantic" in types
        assert "procedural" in types

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_recall_episodes_only(self, mock_services):
        """recall_episodes() only searches episodic memory."""
        mock_episodic = MagicMock()
        mock_ep_result = MagicMock()
        mock_ep_result.item.content = "Episode"
        mock_ep_result.item.id = "ep-1"
        mock_ep_result.item.timestamp = datetime.utcnow()
        mock_ep_result.item.emotional_valence = 0.5
        mock_ep_result.score = 0.8
        mock_episodic.recall_similar = AsyncMock(return_value=[mock_ep_result])

        mock_semantic = MagicMock()
        mock_procedural = MagicMock()
        mock_services.return_value = (mock_episodic, mock_semantic, mock_procedural)

        m = Memory(session_id="test")
        results = await m.recall_episodes("query")

        assert len(results) == 1
        assert results[0].memory_type == "episodic"
        mock_semantic.search_similar.assert_not_called()
        mock_procedural.find_relevant_skills.assert_not_called()

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_recall_entities_only(self, mock_services):
        """recall_entities() only searches semantic memory."""
        mock_episodic = MagicMock()
        mock_semantic = MagicMock()
        mock_ent = MagicMock()
        mock_ent.name = "Entity"
        mock_ent.summary = "Summary"
        mock_ent.id = "ent-1"
        mock_ent.entity_type = "concept"
        mock_semantic.search_similar = AsyncMock(return_value=[mock_ent])

        mock_procedural = MagicMock()
        mock_services.return_value = (mock_episodic, mock_semantic, mock_procedural)

        m = Memory(session_id="test")
        results = await m.recall_entities("query")

        assert len(results) == 1
        assert results[0].memory_type == "semantic"
        mock_episodic.recall_similar.assert_not_called()
        mock_procedural.find_relevant_skills.assert_not_called()

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_recall_skills_only(self, mock_services):
        """recall_skills() only searches procedural memory."""
        mock_episodic = MagicMock()
        mock_semantic = MagicMock()
        mock_procedural = MagicMock()
        mock_skill = MagicMock()
        mock_skill.name = "Skill"
        mock_skill.script = "script"
        mock_skill.id = "sk-1"
        mock_skill.domain = "coding"
        mock_procedural.find_relevant_skills = AsyncMock(return_value=[mock_skill])

        mock_services.return_value = (mock_episodic, mock_semantic, mock_procedural)

        m = Memory(session_id="test")
        results = await m.recall_skills("query")

        assert len(results) == 1
        assert results[0].memory_type == "procedural"
        mock_episodic.recall_similar.assert_not_called()
        mock_semantic.search_similar.assert_not_called()

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_recall_sorts_by_score(self, mock_services):
        """recall() sorts results by score descending."""
        mock_episodic = MagicMock()

        # Create results with different scores
        result1 = MagicMock()
        result1.item.content = "Low score"
        result1.item.id = "ep-1"
        result1.item.timestamp = datetime.utcnow()
        result1.item.emotional_valence = 0.5
        result1.score = 0.3

        result2 = MagicMock()
        result2.item.content = "High score"
        result2.item.id = "ep-2"
        result2.item.timestamp = datetime.utcnow()
        result2.item.emotional_valence = 0.5
        result2.score = 0.9

        mock_episodic.recall_similar = AsyncMock(return_value=[result1, result2])

        mock_semantic = MagicMock()
        mock_semantic.search_similar = AsyncMock(return_value=[])
        mock_procedural = MagicMock()
        mock_procedural.find_relevant_skills = AsyncMock(return_value=[])

        mock_services.return_value = (mock_episodic, mock_semantic, mock_procedural)

        m = Memory(session_id="test")
        results = await m.recall("query")

        # Higher score should be first
        assert results[0].score > results[1].score

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_get_recent(self, mock_services):
        """get_recent() returns recent episodes."""
        mock_episodic = MagicMock()
        mock_ep = MagicMock()
        mock_ep.content = "Recent episode"
        mock_ep.id = "ep-recent"
        mock_ep.timestamp = datetime.utcnow()
        mock_ep.emotional_valence = 0.5
        mock_episodic.get_recent_episodes = AsyncMock(return_value=[mock_ep])

        mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

        m = Memory(session_id="test")
        results = await m.get_recent(limit=10)

        assert len(results) == 1
        assert results[0].content == "Recent episode"
        mock_episodic.get_recent_episodes.assert_called_once_with(limit=10)


class TestMemorySession:
    """Tests for Memory session context manager."""

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_session_context_manager(self, mock_services):
        """session() provides context manager for custom session."""
        mock_episodic = MagicMock()
        mock_episodic.add_episode = AsyncMock(return_value=MagicMock(id="ep-ctx"))
        mock_services.return_value = (mock_episodic, MagicMock(), MagicMock())

        m = Memory()

        async with m.session("custom-session") as session_memory:
            assert session_memory.session_id == "custom-session"
            await session_memory.store("Session content")

        mock_episodic.add_episode.assert_called_once()

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_session_clears_cache_on_exit(self, mock_services):
        """session() clears services cache on exit."""
        mock_services.return_value = (MagicMock(), MagicMock(), MagicMock())

        m = Memory()

        async with m.session("test") as session_memory:
            # Force cache population
            await session_memory._get_services()
            assert session_memory._services_cache is not None

        # Cache should be cleared after exit
        assert session_memory._services_cache is None


class TestModuleLevelFunctions:
    """Tests for module-level convenience functions."""

    @pytest.mark.asyncio
    @patch("ww.memory_api.memory")
    async def test_module_store(self, mock_memory):
        """Module-level store() delegates to memory instance."""
        from ww.memory_api import store

        mock_memory.store = AsyncMock(return_value="ep-123")

        result = await store("content", importance=0.7)

        assert result == "ep-123"
        mock_memory.store.assert_called_once_with(
            "content",
            importance=0.7,
            tags=None,
            metadata=None,
        )

    @pytest.mark.asyncio
    @patch("ww.memory_api.memory")
    async def test_module_recall(self, mock_memory):
        """Module-level recall() delegates to memory instance."""
        from ww.memory_api import recall

        mock_memory.recall = AsyncMock(return_value=[])

        result = await recall("query", limit=10)

        assert result == []
        mock_memory.recall.assert_called_once_with(
            "query",
            limit=10,
            memory_types=None,
        )

    @pytest.mark.asyncio
    @patch("ww.memory_api.memory")
    async def test_module_store_episode(self, mock_memory):
        """Module-level store_episode() delegates to memory instance."""
        from ww.memory_api import store_episode

        mock_memory.store_episode = AsyncMock(return_value="ep-456")

        result = await store_episode("content", importance=0.5, tags=["tag1"])

        assert result == "ep-456"

    @pytest.mark.asyncio
    @patch("ww.memory_api.memory")
    async def test_module_store_entity(self, mock_memory):
        """Module-level store_entity() delegates to memory instance."""
        from ww.memory_api import store_entity

        mock_memory.store_entity = AsyncMock(return_value="ent-789")

        result = await store_entity("Name", description="Desc", entity_type="person")

        assert result == "ent-789"

    @pytest.mark.asyncio
    @patch("ww.memory_api.memory")
    async def test_module_store_skill(self, mock_memory):
        """Module-level store_skill() delegates to memory instance."""
        from ww.memory_api import store_skill

        mock_memory.store_skill = AsyncMock(return_value="sk-101")

        result = await store_skill("name", "script", domain="research")

        assert result == "sk-101"


class TestPackageExports:
    """Tests for ww package exports."""

    def test_memory_exported_from_ww(self):
        """memory is exported from ww package."""
        from ww import memory

        assert memory is not None
        assert isinstance(memory, Memory)

    def test_memory_class_exported_from_ww(self):
        """Memory class is exported from ww package."""
        from ww import Memory

        assert Memory is not None

    def test_memory_result_exported_from_ww(self):
        """MemoryResult is exported from ww package."""
        from ww import MemoryResult

        assert MemoryResult is not None

    def test_store_exported_from_ww(self):
        """store function is exported from ww package."""
        from ww import store

        assert callable(store)

    def test_recall_exported_from_ww(self):
        """recall function is exported from ww package."""
        from ww import recall

        assert callable(recall)

    def test_all_memory_functions_exported(self):
        """All memory functions are exported from ww package."""
        from ww import (
            store,
            recall,
            store_episode,
            store_entity,
            store_skill,
            recall_episodes,
            recall_entities,
            recall_skills,
            get_recent,
        )

        assert all(callable(f) for f in [
            store,
            recall,
            store_episode,
            store_entity,
            store_skill,
            recall_episodes,
            recall_entities,
            recall_skills,
            get_recent,
        ])


class TestServicesCaching:
    """Tests for services caching behavior."""

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_services_cached_within_instance(self, mock_services):
        """Services are cached within a Memory instance."""
        mock_services.return_value = (MagicMock(), MagicMock(), MagicMock())

        m = Memory(session_id="test")

        # Call twice
        await m._get_services()
        await m._get_services()

        # Should only initialize once
        mock_services.assert_called_once()

    @pytest.mark.asyncio
    @patch("ww.memory_api.get_services")
    async def test_different_instances_get_own_cache(self, mock_services):
        """Different Memory instances have separate caches."""
        mock_services.return_value = (MagicMock(), MagicMock(), MagicMock())

        m1 = Memory(session_id="session1")
        m2 = Memory(session_id="session2")

        await m1._get_services()
        await m2._get_services()

        # Each instance should call get_services
        assert mock_services.call_count == 2
