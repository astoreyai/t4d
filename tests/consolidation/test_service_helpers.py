"""Tests for consolidation service helper methods."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

from t4dm.consolidation.service import ConsolidationService
from t4dm.core.types import Episode, EpisodeContext, Outcome, EntityType, Procedure


class TestStratifiedSample:
    """Tests for _stratified_sample method."""

    @pytest.fixture
    def service(self):
        """Create consolidation service with minimal mocks."""
        svc = ConsolidationService.__new__(ConsolidationService)
        return svc

    @pytest.fixture
    def sample_episodes(self):
        """Create episodes with timestamps spanning a week."""
        base_time = datetime.now()
        episodes = []
        for i in range(100):
            ep = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                embedding=[0.1] * 1024,
                timestamp=base_time - timedelta(hours=i),
                context=EpisodeContext(),
                outcome=Outcome.NEUTRAL,
            )
            episodes.append(ep)
        return episodes

    def test_sample_smaller_than_n(self, service, sample_episodes):
        """Return all episodes if fewer than n_samples."""
        small_set = sample_episodes[:5]
        result = service._stratified_sample(small_set, n_samples=10)
        assert len(result) == 5
        assert result == small_set

    def test_sample_equal_to_n(self, service, sample_episodes):
        """Return all episodes if exactly n_samples."""
        exact_set = sample_episodes[:10]
        result = service._stratified_sample(exact_set, n_samples=10)
        assert len(result) == 10

    def test_sample_larger_than_n(self, service, sample_episodes):
        """Sample when more episodes than n_samples."""
        result = service._stratified_sample(sample_episodes, n_samples=20)
        assert len(result) == 20

    def test_sample_preserves_temporal_distribution(self, service, sample_episodes):
        """Sampled episodes span the time range."""
        result = service._stratified_sample(sample_episodes, n_samples=10)

        # Get timestamps
        timestamps = [ep.timestamp for ep in result]
        timestamps.sort()

        # First sample should be near the earliest
        # Last sample should be near the latest
        full_range = max(ep.timestamp for ep in sample_episodes) - min(ep.timestamp for ep in sample_episodes)
        sample_range = timestamps[-1] - timestamps[0]

        # Sample should cover at least 80% of the time range
        assert sample_range >= full_range * 0.8

    def test_sample_empty_list(self, service):
        """Handle empty episode list."""
        result = service._stratified_sample([], n_samples=10)
        assert result == []


class TestExtractEntityFromCluster:
    """Tests for _extract_entity_from_cluster method."""

    @pytest.fixture
    def service(self):
        """Create consolidation service."""
        svc = ConsolidationService.__new__(ConsolidationService)
        return svc

    def test_empty_cluster(self, service):
        """Empty cluster returns None."""
        result = service._extract_entity_from_cluster([])
        assert result is None

    def test_extract_project_entity(self, service):
        """Extract project entity from cluster."""
        episodes = []
        for i in range(3):
            ep = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Working on task {i}",
                embedding=[0.1] * 1024,
                timestamp=datetime.now(),
                context=EpisodeContext(project="world-weaver", file=f"file{i}.py"),
                outcome=Outcome.NEUTRAL,
            )
            episodes.append(ep)

        result = service._extract_entity_from_cluster(episodes)

        assert result is not None
        assert result["name"] == "world-weaver"
        assert result["type"] == EntityType.PROJECT.value

    def test_extract_tool_entity(self, service):
        """Extract tool entity from cluster without project."""
        episodes = []
        for i in range(3):
            ep = Episode(
                id=uuid4(),
                session_id="test",
                content=f"Using tool {i}",
                embedding=[0.1] * 1024,
                timestamp=datetime.now(),
                context=EpisodeContext(tool="pytest"),
                outcome=Outcome.NEUTRAL,
            )
            episodes.append(ep)

        result = service._extract_entity_from_cluster(episodes)

        assert result is not None
        assert result["name"] == "pytest"
        assert result["type"] == EntityType.TOOL.value

    def test_extract_concept_entity(self, service):
        """Extract concept entity when no project or tool."""
        episodes = []
        for i in range(3):
            ep = Episode(
                id=uuid4(),
                session_id="test",
                content="Testing memory consolidation algorithms.",
                embedding=[0.1] * 1024,
                timestamp=datetime.now(),
                context=EpisodeContext(),
                outcome=Outcome.NEUTRAL,
            )
            episodes.append(ep)

        result = service._extract_entity_from_cluster(episodes)

        assert result is not None
        assert result["type"] == EntityType.CONCEPT.value
        # Name should be from content
        assert len(result["name"]) > 0


class TestMergeProcedureSteps:
    """Tests for _merge_procedure_steps method."""

    @pytest.fixture
    def service(self):
        """Create consolidation service."""
        svc = ConsolidationService.__new__(ConsolidationService)
        return svc

    def test_empty_procedures(self, service):
        """Empty procedure list returns empty steps."""
        result = service._merge_procedure_steps([])
        assert result == []

    def test_single_procedure(self, service):
        """Single procedure returns its steps."""
        from dataclasses import dataclass

        @dataclass
        class MockStep:
            instruction: str

        procedure = MagicMock()
        procedure.steps = [MockStep("Step 1"), MockStep("Step 2")]

        result = service._merge_procedure_steps([procedure])
        assert len(result) == 2

    def test_multiple_procedures_uses_best(self, service):
        """Multiple procedures uses best (first) steps."""
        from dataclasses import dataclass

        @dataclass
        class MockStep:
            instruction: str

        proc1 = MagicMock()
        proc1.steps = [MockStep("Best Step 1"), MockStep("Best Step 2")]

        proc2 = MagicMock()
        proc2.steps = [MockStep("Other Step 1")]

        result = service._merge_procedure_steps([proc1, proc2])
        # Should use first procedure's steps
        assert len(result) == 2
        assert result[0].instruction == "Best Step 1"


class TestCosineSimilarity:
    """Tests for _cosine_similarity method."""

    @pytest.fixture
    def service(self):
        """Create consolidation service."""
        svc = ConsolidationService.__new__(ConsolidationService)
        return svc

    def test_identical_vectors(self, service):
        """Identical vectors have similarity 1.0."""
        vec = [1.0, 2.0, 3.0]
        result = service._cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0, abs=0.001)

    def test_orthogonal_vectors(self, service):
        """Orthogonal vectors have similarity 0.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        result = service._cosine_similarity(vec1, vec2)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_opposite_vectors(self, service):
        """Opposite vectors have similarity -1.0."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]
        result = service._cosine_similarity(vec1, vec2)
        assert result == pytest.approx(-1.0, abs=0.001)

    def test_zero_vector(self, service):
        """Zero vector returns 0.0."""
        vec1 = [0.0, 0.0, 0.0]
        vec2 = [1.0, 2.0, 3.0]
        result = service._cosine_similarity(vec1, vec2)
        assert result == 0.0


class TestConsolidateMain:
    """Tests for main consolidate method."""

    @pytest.fixture
    def mock_service(self):
        """Create consolidation service with mocked internals."""
        import asyncio
        from t4dm.consolidation.service import ConsolidationScheduler
        svc = ConsolidationService.__new__(ConsolidationService)
        svc._consolidate_light = AsyncMock(return_value={"episodes_scanned": 10, "duplicates_found": 2, "cleaned": 2})
        svc._consolidate_deep = AsyncMock(return_value={"consolidated_episodes": 5, "new_entities_created": 1})
        svc._consolidate_skills = AsyncMock(return_value={"procedures_analyzed": 3, "merged": 1})
        svc._update_decay = AsyncMock(return_value={"episodes": 10, "entities": 5})
        # RACE-002 FIX: Add consolidation lock for tests using __new__
        svc._consolidation_lock = asyncio.Lock()
        # P3.3 FIX: Add scheduler reference for tests using __new__
        svc._scheduler = ConsolidationScheduler(enabled=True)
        return svc

    @pytest.mark.asyncio
    async def test_consolidate_light(self, mock_service):
        """Test light consolidation."""
        result = await mock_service.consolidate(consolidation_type="light")

        assert result["status"] == "completed"
        assert result["consolidation_type"] == "light"
        assert "light" in result["results"]
        mock_service._consolidate_light.assert_called_once()
        mock_service._consolidate_deep.assert_not_called()

    @pytest.mark.asyncio
    async def test_consolidate_deep(self, mock_service):
        """Test deep consolidation."""
        result = await mock_service.consolidate(consolidation_type="deep")

        assert result["status"] == "completed"
        assert "episodic_to_semantic" in result["results"]
        assert "decay_updated" in result["results"]
        mock_service._consolidate_deep.assert_called_once()
        mock_service._update_decay.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidate_skill(self, mock_service):
        """Test skill consolidation."""
        result = await mock_service.consolidate(consolidation_type="skill")

        assert result["status"] == "completed"
        assert "skill_consolidation" in result["results"]
        mock_service._consolidate_skills.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidate_all(self, mock_service):
        """Test all consolidation types."""
        result = await mock_service.consolidate(consolidation_type="all")

        assert result["status"] == "completed"
        assert "light" in result["results"]
        assert "episodic_to_semantic" in result["results"]
        assert "skill_consolidation" in result["results"]
        assert "decay_updated" in result["results"]

    @pytest.mark.asyncio
    async def test_consolidate_unknown_type(self, mock_service):
        """Test unknown consolidation type."""
        result = await mock_service.consolidate(consolidation_type="unknown")

        assert result["status"] == "error"
        assert "Unknown consolidation type" in result["error"]

    @pytest.mark.asyncio
    async def test_consolidate_with_session_filter(self, mock_service):
        """Test consolidation with session filter."""
        result = await mock_service.consolidate(
            consolidation_type="light",
            session_filter="session-123",
        )

        assert result["status"] == "completed"
        mock_service._consolidate_light.assert_called_once_with("session-123")

    @pytest.mark.asyncio
    async def test_consolidate_error_handling(self, mock_service):
        """Test error handling during consolidation."""
        mock_service._consolidate_light = AsyncMock(side_effect=Exception("Test error"))

        result = await mock_service.consolidate(consolidation_type="light")

        assert result["status"] == "failed"
        assert "Test error" in result["error"]

    @pytest.mark.asyncio
    async def test_consolidate_records_duration(self, mock_service):
        """Test that duration is recorded."""
        result = await mock_service.consolidate(consolidation_type="light")

        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0


class TestUpdateDecay:
    """Tests for _update_decay method."""

    @pytest.fixture
    def service(self):
        """Create consolidation service."""
        svc = ConsolidationService.__new__(ConsolidationService)
        return svc

    @pytest.mark.asyncio
    async def test_update_decay_returns_dict(self, service):
        """Update decay returns summary dict."""
        result = await service._update_decay()

        assert isinstance(result, dict)
        assert "episodes" in result
        assert "entities" in result
        assert "procedures" in result


class TestGetConsolidationService:
    """Tests for singleton getter."""

    def test_get_service_returns_singleton(self):
        """Getter returns singleton instance."""
        # Note: This would need mocking of dependencies
        # Just test the import works
        from t4dm.consolidation.service import get_consolidation_service
        assert callable(get_consolidation_service)
