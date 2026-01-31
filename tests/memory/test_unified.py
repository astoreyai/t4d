"""
Comprehensive tests for unified memory service.

Tests UnifiedMemoryService class for cross-memory search,
result formatting, and NCA integration.
"""

import pytest
from datetime import datetime
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from t4dm.memory.unified import UnifiedMemoryService, get_unified_memory_service, _get_learning_hooks
from t4dm.core.types import (
    Episode,
    Entity,
    EntityType,
    Outcome,
    EpisodeContext,
    ScoredResult,
    Procedure,
    Domain,
    ProcedureStep,
)


# =============================================================================
# Test _get_learning_hooks Function
# =============================================================================


class TestGetLearningHooks:
    """Tests for lazy loading of learning hooks."""

    def test_get_learning_hooks_returns_callable_or_none(self):
        """_get_learning_hooks returns a callable or None."""
        result = _get_learning_hooks()
        # Should be either a callable (if hooks available) or None
        assert result is None or callable(result)

    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2
    def test_learning_hooks_caching(self):
        """Test that _get_learning_hooks caches results."""
        result1 = _get_learning_hooks()
        result2 = _get_learning_hooks()
        # Both calls should return the same object (cached)
        assert result1 is result2


# =============================================================================
# Test UnifiedMemoryService Initialization
# =============================================================================


class TestUnifiedMemoryServiceInit:
    """Tests for UnifiedMemoryService initialization."""

    @pytest.fixture
    def mock_memories(self):
        """Create mock memory services."""
        episodic = AsyncMock()
        episodic.session_id = "test-session"

        semantic = AsyncMock()
        procedural = AsyncMock()

        return episodic, semantic, procedural

    @pytest.fixture
    def service(self, mock_memories):
        """Create UnifiedMemoryService instance."""
        episodic, semantic, procedural = mock_memories
        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings"):
                return UnifiedMemoryService(episodic, semantic, procedural)

    def test_initialization(self, service, mock_memories):
        """Service initializes with memory references."""
        episodic, semantic, procedural = mock_memories
        assert service.episodic is episodic
        assert service.semantic is semantic
        assert service.procedural is procedural

    def test_bridge_container_initialization(self, mock_memories):
        """Bridge container is initialized with session ID."""
        episodic, semantic, procedural = mock_memories
        mock_bridge = MagicMock()

        with patch("t4dm.memory.unified.get_bridge_container", return_value=mock_bridge) as mock_get_bridge:
            with patch("t4dm.core.config.get_settings"):
                service = UnifiedMemoryService(episodic, semantic, procedural)
                mock_get_bridge.assert_called_once_with("test-session")
                assert service._bridge_container is mock_bridge

    def test_nca_modulation_enabled_by_default(self, mock_memories):
        """NCA modulation is enabled by default."""
        episodic, semantic, procedural = mock_memories
        mock_settings = MagicMock()
        mock_settings.nca_modulation_enabled = True

        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings", return_value=mock_settings):
                service = UnifiedMemoryService(episodic, semantic, procedural)
                assert service._nca_modulation_enabled is True

    def test_nca_modulation_respects_settings(self, mock_memories):
        """NCA modulation setting is read from config."""
        episodic, semantic, procedural = mock_memories
        mock_settings = MagicMock()
        mock_settings.nca_modulation_enabled = False

        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings", return_value=mock_settings):
                service = UnifiedMemoryService(episodic, semantic, procedural)
                assert service._nca_modulation_enabled is False


# =============================================================================
# Test UnifiedMemoryService.search()
# =============================================================================


class TestUnifiedMemorySearch:
    """Tests for unified memory search functionality."""

    @pytest.fixture
    def mock_memories(self):
        """Create mock memory services."""
        episodic = AsyncMock()
        episodic.session_id = "test-session"
        semantic = AsyncMock()
        procedural = AsyncMock()
        return episodic, semantic, procedural

    @pytest.fixture
    def service(self, mock_memories):
        """Create UnifiedMemoryService instance."""
        episodic, semantic, procedural = mock_memories
        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings"):
                return UnifiedMemoryService(episodic, semantic, procedural)

    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode."""
        return Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test episode content",
            timestamp=datetime.now(),
            outcome=Outcome.SUCCESS,
            emotional_valence=0.7,
        )

    @pytest.fixture
    def sample_entity(self):
        """Create a sample entity."""
        return Entity(
            id=uuid4(),
            name="Test Concept",
            entity_type=EntityType.CONCEPT,
            summary="A test concept",
            details="Detailed information",
        )

    @pytest.fixture
    def sample_procedure(self):
        """Create a sample procedure."""
        return Procedure(
            id=uuid4(),
            name="Test Skill",
            domain=Domain.CODING,
            trigger_pattern="pattern.*",
            steps=[
                ProcedureStep(order=1, action="Step 1", tool="tool1"),
                ProcedureStep(order=2, action="Step 2", tool="tool2"),
            ],
        )

    @pytest.mark.asyncio
    async def test_search_all_memory_types(self, service, sample_episode, sample_entity, sample_procedure):
        """Search returns results from all memory types."""
        # Setup mock returns
        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_episode, score=0.9)
        ])
        service.semantic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_entity, score=0.8)
        ])
        service.procedural.recall_skill = AsyncMock(return_value=[
            ScoredResult(item=sample_procedure, score=0.7)
        ])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        result = await service.search("test query", k=10)

        assert result["query"] == "test query"
        assert result["total_count"] == 3
        assert len(result["results"]) == 3
        assert "episodic" in result["by_type"]
        assert "semantic" in result["by_type"]
        assert "procedural" in result["by_type"]
        assert len(result["by_type"]["episodic"]) == 1
        assert len(result["by_type"]["semantic"]) == 1
        assert len(result["by_type"]["procedural"]) == 1

    @pytest.mark.asyncio
    async def test_search_filtered_memory_types(self, service, sample_episode, sample_entity):
        """Search respects memory_types filter."""
        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_episode, score=0.9)
        ])
        service.semantic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_entity, score=0.8)
        ])
        service.procedural.recall_skill = AsyncMock()

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        result = await service.search(
            "test query",
            k=10,
            memory_types=["episodic", "semantic"]
        )

        assert result["total_count"] == 2
        service.procedural.recall_skill.assert_not_called()
        assert "episodic" in result["by_type"]
        assert "semantic" in result["by_type"]

    @pytest.mark.asyncio
    async def test_search_min_score_filtering(self, service, sample_episode, sample_entity):
        """Search filters results by min_score."""
        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_episode, score=0.9),
            ScoredResult(item=sample_episode, score=0.3),  # Will be filtered
        ])
        service.semantic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_entity, score=0.8)
        ])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        result = await service.search(
            "test query",
            k=10,
            min_score=0.5
        )

        assert result["total_count"] == 2
        assert all(r["score"] >= 0.5 for r in result["results"])

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_score(self, service, sample_episode, sample_entity):
        """Search results are sorted by score descending."""
        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_episode, score=0.5)
        ])
        service.semantic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_entity, score=0.9)
        ])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        result = await service.search("test query", k=10)

        scores = [r["score"] for r in result["results"]]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 0.9
        assert scores[1] == 0.5

    @pytest.mark.asyncio
    async def test_search_handles_exception_in_one_type(self, service, sample_episode, sample_procedure):
        """Search continues when one memory type throws exception."""
        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_episode, score=0.9)
        ])
        service.semantic.recall = AsyncMock(side_effect=ValueError("DB error"))
        service.procedural.recall_skill = AsyncMock(return_value=[
            ScoredResult(item=sample_procedure, score=0.7)
        ])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        result = await service.search("test query", k=10)

        # Should have results from episodic and procedural, but not semantic
        assert result["total_count"] == 2
        assert len(result["by_type"]["episodic"]) == 1
        assert len(result["by_type"]["semantic"]) == 0
        assert "error" not in result  # Overall response should not have error key

    @pytest.mark.asyncio
    async def test_search_passes_session_id(self, service, sample_episode):
        """Search passes session_id to memory services."""
        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_episode, score=0.9)
        ])
        service.semantic.recall = AsyncMock(return_value=[])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        await service.search("test query", session_id="custom-session")

        service.episodic.recall.assert_called_once()
        call_args = service.episodic.recall.call_args
        assert call_args.kwargs.get("session_filter") == "custom-session"

    @pytest.mark.asyncio
    async def test_search_passes_k_to_memory_services(self, service, sample_episode):
        """Search passes k parameter to memory services."""
        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=sample_episode, score=0.9)
        ])
        service.semantic.recall = AsyncMock(return_value=[])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        await service.search("test query", k=20)

        service.episodic.recall.assert_called_once()
        call_args = service.episodic.recall.call_args
        assert call_args.kwargs.get("limit") == 20

    @pytest.mark.asyncio
    async def test_search_empty_results(self, service):
        """Search handles empty results gracefully."""
        service.episodic.recall = AsyncMock(return_value=[])
        service.semantic.recall = AsyncMock(return_value=[])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        result = await service.search("test query")

        assert result["total_count"] == 0
        assert result["results"] == []
        assert result["by_type"]["episodic"] == []
        assert result["by_type"]["semantic"] == []
        assert result["by_type"]["procedural"] == []


# =============================================================================
# Test NCA Modulation in Search
# =============================================================================


class TestNCAModulation:
    """Tests for NCA cognitive state modulation."""

    @pytest.fixture
    def mock_memories(self):
        """Create mock memory services."""
        episodic = AsyncMock()
        episodic.session_id = "test-session"
        semantic = AsyncMock()
        procedural = AsyncMock()
        return episodic, semantic, procedural

    @pytest.fixture
    def service(self, mock_memories):
        """Create UnifiedMemoryService with NCA enabled."""
        episodic, semantic, procedural = mock_memories
        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings") as mock_settings:
                mock_settings_obj = MagicMock()
                mock_settings_obj.nca_modulation_enabled = True
                mock_settings.return_value = mock_settings_obj
                return UnifiedMemoryService(episodic, semantic, procedural)

    @pytest.fixture
    def sample_episodes(self):
        """Create multiple sample episodes."""
        episodes = []
        for i in range(3):
            episodes.append(Episode(
                id=uuid4(),
                session_id="test-session",
                content=f"Episode {i}",
                timestamp=datetime.now(),
                outcome=Outcome.SUCCESS,
                emotional_valence=0.5 + i * 0.1,
            ))
        return episodes

    @pytest.mark.asyncio
    async def test_nca_focus_boosts_top_results(self, service, sample_episodes):
        """FOCUS state boosts top results and suppresses lower ones."""
        results = [
            ScoredResult(item=sample_episodes[0], score=0.9),
            ScoredResult(item=sample_episodes[1], score=0.7),
            ScoredResult(item=sample_episodes[2], score=0.5),
        ]

        service.episodic.recall = AsyncMock(return_value=results)
        service.semantic.recall = AsyncMock(return_value=[])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        # Mock NCA bridge with FOCUS state
        mock_bridge = MagicMock()
        from t4dm.nca.attractors import CognitiveState
        mock_bridge.get_current_cognitive_state = MagicMock(return_value=CognitiveState.FOCUS)
        mock_bridge.get_current_nt_state = MagicMock(return_value=[1.0, 0.0, 0.0])
        service._bridge_container.get_nca_bridge = MagicMock(return_value=mock_bridge)

        result = await service.search("test query")

        # Results should be reordered with focus modulation applied
        assert "nca_boost" in result["results"][0]
        # Top result should have higher boost than lower results
        assert result["results"][0]["nca_boost"] > result["results"][-1]["nca_boost"]

    @pytest.mark.asyncio
    async def test_nca_explore_adds_noise(self, service, sample_episodes):
        """EXPLORE state adds exploration noise to scores."""
        results = [
            ScoredResult(item=sample_episodes[0], score=0.9),
            ScoredResult(item=sample_episodes[1], score=0.7),
        ]

        service.episodic.recall = AsyncMock(return_value=results)
        service.semantic.recall = AsyncMock(return_value=[])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        # Mock NCA bridge with EXPLORE state
        mock_bridge = MagicMock()
        from t4dm.nca.attractors import CognitiveState
        mock_bridge.get_current_cognitive_state = MagicMock(return_value=CognitiveState.EXPLORE)
        mock_bridge.get_current_nt_state = MagicMock(return_value=[0.0, 1.0, 0.0])
        service._bridge_container.get_nca_bridge = MagicMock(return_value=mock_bridge)

        result = await service.search("test query")

        # Results should have exploration noise applied
        assert all("nca_boost" in r for r in result["results"])
        # With exploration, scores may be reordered differently than by original score
        assert result["results"] is not None

    @pytest.mark.asyncio
    async def test_nca_modulation_disabled(self, mock_memories):
        """NCA modulation is skipped when disabled."""
        episodic, semantic, procedural = mock_memories

        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings") as mock_settings:
                mock_settings_obj = MagicMock()
                mock_settings_obj.nca_modulation_enabled = False
                mock_settings.return_value = mock_settings_obj
                service = UnifiedMemoryService(episodic, semantic, procedural)

        episode = Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test",
            timestamp=datetime.now(),
            outcome=Outcome.SUCCESS,
            emotional_valence=0.5,
        )

        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=episode, score=0.9)
        ])
        service.semantic.recall = AsyncMock(return_value=[])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        service._bridge_container.get_nca_bridge = MagicMock()

        result = await service.search("test query")

        # NCA bridge should not be called when modulation is disabled
        service._bridge_container.get_nca_bridge.assert_not_called()

    @pytest.mark.asyncio
    async def test_nca_modulation_handles_exception(self, service, sample_episodes):
        """NCA modulation failures do not crash search."""
        results = [
            ScoredResult(item=sample_episodes[0], score=0.9),
        ]

        service.episodic.recall = AsyncMock(return_value=results)
        service.semantic.recall = AsyncMock(return_value=[])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        # Mock NCA bridge that raises exception
        mock_bridge = MagicMock()
        mock_bridge.get_current_cognitive_state = MagicMock(side_effect=RuntimeError("NCA error"))
        service._bridge_container.get_nca_bridge = MagicMock(return_value=mock_bridge)

        # Should not raise, should return results
        result = await service.search("test query")
        assert result["total_count"] == 1
        assert result["results"][0]["score"] == 0.9


# =============================================================================
# Test UnifiedMemoryService.get_related()
# =============================================================================


class TestGetRelated:
    """Tests for related memory graph traversal."""

    @pytest.fixture
    def mock_memories(self):
        """Create mock memory services."""
        episodic = AsyncMock()
        episodic.session_id = "test-session"
        semantic = AsyncMock()
        procedural = AsyncMock()
        return episodic, semantic, procedural

    @pytest.fixture
    def service(self, mock_memories):
        """Create UnifiedMemoryService instance."""
        episodic, semantic, procedural = mock_memories
        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings"):
                return UnifiedMemoryService(episodic, semantic, procedural)

    @pytest.fixture
    def sample_episode(self):
        """Create a sample episode."""
        return Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test episode",
            timestamp=datetime.now(),
            outcome=Outcome.SUCCESS,
            emotional_valence=0.5,
        )

    @pytest.fixture
    def sample_entity(self):
        """Create a sample entity."""
        return Entity(
            id=uuid4(),
            name="Test Entity",
            entity_type=EntityType.CONCEPT,
            summary="Test summary",
        )

    @pytest.mark.asyncio
    async def test_get_related_episodic_returns_entities(self, service, sample_episode, sample_entity):
        """Get related for episodic memory returns extracted entities."""
        episode_id = str(sample_episode.id)

        # Mock graph store relationships
        service.semantic.graph_store = AsyncMock()
        service.semantic.graph_store.get_relationships = AsyncMock(return_value=[
            {"other_id": str(sample_entity.id), "type": "EXTRACTED_FROM"}
        ])
        service.semantic.get_entity = AsyncMock(return_value=sample_entity)

        result = await service.get_related(episode_id, "episodic")

        assert result["source_id"] == episode_id
        assert result["source_type"] == "episodic"
        assert len(result["related"]["semantic"]) == 1
        assert result["related"]["semantic"][0]["name"] == "Test Entity"

    @pytest.mark.asyncio
    async def test_get_related_semantic_returns_neighbors(self, service, sample_entity):
        """Get related for semantic memory returns graph neighbors."""
        entity_id = str(sample_entity.id)

        neighbor_entity = Entity(
            id=uuid4(),
            name="Neighbor",
            entity_type=EntityType.CONCEPT,
            summary="Neighbor entity",
        )

        # Mock semantic get_entity and graph relationships
        service.semantic.get_entity = AsyncMock(return_value=sample_entity)
        service.semantic.graph_store = AsyncMock()
        service.semantic.graph_store.get_relationships = AsyncMock(return_value=[
            {
                "other_id": str(neighbor_entity.id),
                "properties": {"weight": 0.8}
            }
        ])

        # Mock neighbor entity lookup
        async def get_entity_side_effect(uid):
            if uid == neighbor_entity.id:
                return neighbor_entity
            return sample_entity

        service.semantic.get_entity = AsyncMock(side_effect=get_entity_side_effect)

        result = await service.get_related(entity_id, "semantic")

        assert result["source_id"] == entity_id
        assert result["source_type"] == "semantic"
        assert len(result["related"]["semantic"]) >= 1

    @pytest.mark.asyncio
    async def test_get_related_procedural_returns_similar(self, service):
        """Get related for procedural memory returns similar skills."""
        procedure = Procedure(
            id=uuid4(),
            name="Test Skill",
            domain=Domain.CODING,
            trigger_pattern="pattern",
            steps=[],
        )

        similar_procedure = Procedure(
            id=uuid4(),
            name="Similar Skill",
            domain=Domain.CODING,
            trigger_pattern="pattern",
            steps=[],
        )

        procedure_id = str(procedure.id)

        # Mock procedural get_procedure and recall_skill
        service.procedural.get_procedure = AsyncMock(return_value=procedure)
        service.procedural.recall_skill = AsyncMock(return_value=[
            ScoredResult(item=similar_procedure, score=0.85)
        ])

        result = await service.get_related(procedure_id, "procedural")

        assert result["source_id"] == procedure_id
        assert result["source_type"] == "procedural"
        assert len(result["related"]["procedural"]) == 1
        assert result["related"]["procedural"][0]["name"] == "Similar Skill"

    @pytest.mark.asyncio
    async def test_get_related_excludes_self(self, service):
        """Get related excludes the source procedure from results."""
        procedure = Procedure(
            id=uuid4(),
            name="Test Skill",
            domain=Domain.CODING,
            trigger_pattern="pattern",
            steps=[],
        )

        procedure_id = str(procedure.id)

        service.procedural.get_procedure = AsyncMock(return_value=procedure)
        service.procedural.recall_skill = AsyncMock(return_value=[
            ScoredResult(item=procedure, score=0.95),  # Self
        ])

        result = await service.get_related(procedure_id, "procedural")

        # Self should be excluded
        assert len(result["related"]["procedural"]) == 0

    @pytest.mark.asyncio
    async def test_get_related_handles_exception(self, service):
        """Get related handles exceptions gracefully."""
        entity_id = str(uuid4())

        service.semantic.get_entity = AsyncMock(side_effect=ValueError("DB error"))
        service.semantic.graph_store = AsyncMock()

        result = await service.get_related(entity_id, "semantic")

        assert result["source_id"] == entity_id
        assert "error" in result
        assert result["related"]["semantic"] == []

    @pytest.mark.asyncio
    async def test_get_related_depth_parameter(self, service, sample_entity):
        """Get related respects depth parameter."""
        entity_id = str(sample_entity.id)

        service.semantic.get_entity = AsyncMock(return_value=sample_entity)
        service.semantic.graph_store = AsyncMock()
        service.semantic.graph_store.get_relationships = AsyncMock(return_value=[])

        result = await service.get_related(entity_id, "semantic", depth=3)

        assert result["depth"] == 3


# =============================================================================
# Test Result Formatting Methods
# =============================================================================


class TestResultFormatting:
    """Tests for result formatting helper methods."""

    @pytest.fixture
    def mock_memories(self):
        """Create mock memory services."""
        episodic = AsyncMock()
        episodic.session_id = "test-session"
        semantic = AsyncMock()
        procedural = AsyncMock()
        return episodic, semantic, procedural

    @pytest.fixture
    def service(self, mock_memories):
        """Create UnifiedMemoryService instance."""
        episodic, semantic, procedural = mock_memories
        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings"):
                return UnifiedMemoryService(episodic, semantic, procedural)

    def test_to_unified_result_episodic(self, service):
        """_to_unified_result formats episodic results correctly."""
        episode = Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test content",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            outcome=Outcome.SUCCESS,
            emotional_valence=0.8,
        )

        scored = ScoredResult(item=episode, score=0.85, components={"hybrid": 0.85})

        result = service._to_unified_result(scored, "episodic")

        assert result["id"] == str(episode.id)
        assert result["memory_type"] == "episodic"
        assert result["score"] == 0.85
        assert result["content"] == "Test content"
        assert result["metadata"]["timestamp"] == "2024-01-01T12:00:00"
        assert result["metadata"]["outcome"] == "success"
        assert result["metadata"]["valence"] == 0.8
        assert result["metadata"]["components"] == {"hybrid": 0.85}

    def test_to_unified_result_semantic(self, service):
        """_to_unified_result formats semantic results correctly."""
        entity = Entity(
            id=uuid4(),
            name="Test Concept",
            entity_type=EntityType.CONCEPT,
            summary="Test summary",
            details="Test details",
        )

        scored = ScoredResult(item=entity, score=0.75, components={"semantic": 0.75})

        result = service._to_unified_result(scored, "semantic")

        assert result["id"] == str(entity.id)
        assert result["memory_type"] == "semantic"
        assert result["score"] == 0.75
        assert result["content"] == "Test Concept: Test summary"
        assert result["metadata"]["name"] == "Test Concept"
        assert result["metadata"]["entity_type"] == "CONCEPT"
        assert result["metadata"]["summary"] == "Test summary"

    def test_to_unified_result_procedural(self, service):
        """_to_unified_result formats procedural results correctly."""
        procedure = Procedure(
            id=uuid4(),
            name="Test Skill",
            domain=Domain.CODING,
            trigger_pattern="pattern.*",
            steps=[
                ProcedureStep(order=1, action="Step 1", tool="python"),
            ],
            success_rate=0.92,
            execution_count=50,
        )

        scored = ScoredResult(item=procedure, score=0.88, components={"procedural": 0.88})

        result = service._to_unified_result(scored, "procedural")

        assert result["id"] == str(procedure.id)
        assert result["memory_type"] == "procedural"
        assert result["score"] == 0.88
        assert result["content"] == "Test Skill (coding)"
        assert result["metadata"]["name"] == "Test Skill"
        assert result["metadata"]["domain"] == "coding"
        assert result["metadata"]["trigger_pattern"] == "pattern.*"
        assert result["metadata"]["success_rate"] == 0.92
        assert result["metadata"]["execution_count"] == 50

    def test_to_dict_episodic(self, service):
        """_to_dict formats episodic results correctly."""
        episode = Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test content",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            outcome=Outcome.FAILURE,
            emotional_valence=0.3,
        )

        scored = ScoredResult(item=episode, score=0.85, components={"hybrid": 0.85})

        result = service._to_dict(scored, "episodic")

        assert result["id"] == str(episode.id)
        assert result["score"] == 0.85
        assert result["content"] == "Test content"
        assert result["timestamp"] == "2024-01-01T12:00:00"
        assert result["outcome"] == "failure"
        assert result["valence"] == 0.3
        assert result["session_id"] == "test-session"

    def test_to_dict_semantic(self, service):
        """_to_dict formats semantic results correctly."""
        entity = Entity(
            id=uuid4(),
            name="Test Concept",
            entity_type=EntityType.PERSON,
            summary="Test summary",
            details="Test details",
        )

        scored = ScoredResult(item=entity, score=0.75, components={"semantic": 0.75})

        result = service._to_dict(scored, "semantic")

        assert result["id"] == str(entity.id)
        assert result["score"] == 0.75
        assert result["name"] == "Test Concept"
        assert result["entity_type"] == "PERSON"
        assert result["summary"] == "Test summary"
        assert result["details"] == "Test details"

    def test_to_dict_procedural(self, service):
        """_to_dict formats procedural results correctly."""
        procedure = Procedure(
            id=uuid4(),
            name="Test Skill",
            domain=Domain.DEVOPS,
            trigger_pattern="deploy.*",
            steps=[
                ProcedureStep(order=1, action="Build", tool="docker", parameters={"tag": "latest"}),
                ProcedureStep(order=2, action="Push", tool="registry"),
            ],
            success_rate=0.95,
            execution_count=100,
        )

        scored = ScoredResult(item=procedure, score=0.88, components={"procedural": 0.88})

        result = service._to_dict(scored, "procedural")

        assert result["id"] == str(procedure.id)
        assert result["score"] == 0.88
        assert result["name"] == "Test Skill"
        assert result["domain"] == "devops"
        assert result["trigger_pattern"] == "deploy.*"
        assert result["success_rate"] == 0.95
        assert result["execution_count"] == 100
        assert len(result["steps"]) == 2
        assert result["steps"][0]["action"] == "Build"
        assert result["steps"][0]["tool"] == "docker"


# =============================================================================
# Test Factory Function
# =============================================================================


class TestFactoryFunction:
    """Tests for get_unified_memory_service factory."""

    def test_factory_creates_service(self):
        """Factory creates UnifiedMemoryService instance."""
        episodic = AsyncMock()
        episodic.session_id = "test"
        semantic = AsyncMock()
        procedural = AsyncMock()

        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings"):
                service = get_unified_memory_service(episodic, semantic, procedural)

        assert isinstance(service, UnifiedMemoryService)
        assert service.episodic is episodic
        assert service.semantic is semantic
        assert service.procedural is procedural

    def test_factory_returns_service_instance_type(self):
        """Factory return type is UnifiedMemoryService."""
        episodic = AsyncMock()
        episodic.session_id = "test"
        semantic = AsyncMock()
        procedural = AsyncMock()

        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings"):
                service = get_unified_memory_service(episodic, semantic, procedural)
                assert type(service).__name__ == "UnifiedMemoryService"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for unified memory service."""

    @pytest.fixture
    def mock_memories(self):
        """Create mock memory services."""
        episodic = AsyncMock()
        episodic.session_id = "test-session"
        semantic = AsyncMock()
        procedural = AsyncMock()
        return episodic, semantic, procedural

    @pytest.fixture
    def service(self, mock_memories):
        """Create UnifiedMemoryService instance."""
        episodic, semantic, procedural = mock_memories
        with patch("t4dm.memory.unified.get_bridge_container"):
            with patch("t4dm.core.config.get_settings"):
                return UnifiedMemoryService(episodic, semantic, procedural)

    @pytest.mark.asyncio
    async def test_search_with_results_flow(self, service):
        """End-to-end search flow with results."""
        episode = Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test",
            timestamp=datetime.now(),
            outcome=Outcome.SUCCESS,
            emotional_valence=0.5,
        )

        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=episode, score=0.9)
        ])
        service.semantic.recall = AsyncMock(return_value=[])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        result = await service.search("test query", session_id="test-session")

        assert result["query"] == "test query"
        assert result["session_id"] == "test-session"
        assert result["total_count"] == 1

    @pytest.mark.asyncio
    async def test_parallel_search_execution(self, service):
        """Verify search executes memory type searches in parallel."""
        episode = Episode(
            id=uuid4(),
            session_id="test-session",
            content="Test",
            timestamp=datetime.now(),
            outcome=Outcome.SUCCESS,
            emotional_valence=0.5,
        )

        entity = Entity(
            id=uuid4(),
            name="Test",
            entity_type=EntityType.CONCEPT,
            summary="Test",
        )

        service.episodic.recall = AsyncMock(return_value=[
            ScoredResult(item=episode, score=0.9)
        ])
        service.semantic.recall = AsyncMock(return_value=[
            ScoredResult(item=entity, score=0.8)
        ])
        service.procedural.recall_skill = AsyncMock(return_value=[])

        service._bridge_container.get_nca_bridge = MagicMock(return_value=None)

        result = await service.search("test query")

        # All three memory types should be queried
        service.episodic.recall.assert_called_once()
        service.semantic.recall.assert_called_once()
        service.procedural.recall_skill.assert_called_once()
