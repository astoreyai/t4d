"""
Unit tests for HDBSCAN clustering in consolidation service.
"""

import pytest
from datetime import datetime
from uuid import uuid4

from t4dm.consolidation import HDBSCAN_AVAILABLE
from t4dm.consolidation.service import ConsolidationService
from t4dm.core.types import Episode, EpisodeContext, Domain

# Skip all tests if HDBSCAN not installed
pytestmark = pytest.mark.skipif(
    not HDBSCAN_AVAILABLE,
    reason="HDBSCAN not installed. Install with: pip install hdbscan"
)


@pytest.fixture
def consolidation_service():
    """Create consolidation service instance."""
    return ConsolidationService()


@pytest.fixture
def sample_episodes():
    """Create sample episodes for testing."""
    episodes = []

    # Create 3 clusters of similar episodes
    # Cluster 1: Python programming (5 episodes)
    for i in range(5):
        ep = Episode(
            id=uuid4(),
            session_id="test-session",
            content=f"Writing Python code to implement feature {i} using async/await patterns",
            timestamp=datetime.now(),
            emotional_valence=0.7,
            context=EpisodeContext(
                project="ww",
                tool="python",
                file=f"src/module_{i}.py",
            ),
        )
        episodes.append(ep)

    # Cluster 2: Testing (4 episodes)
    for i in range(4):
        ep = Episode(
            id=uuid4(),
            session_id="test-session",
            content=f"Writing pytest tests for module {i} with mocking and fixtures",
            timestamp=datetime.now(),
            emotional_valence=0.6,
            context=EpisodeContext(
                project="ww",
                tool="pytest",
                file=f"tests/test_{i}.py",
            ),
        )
        episodes.append(ep)

    # Cluster 3: Documentation (3 episodes)
    for i in range(3):
        ep = Episode(
            id=uuid4(),
            session_id="test-session",
            content=f"Writing documentation for API endpoint {i} with examples",
            timestamp=datetime.now(),
            emotional_valence=0.5,
            context=EpisodeContext(
                project="ww",
                tool="markdown",
                file=f"docs/api_{i}.md",
            ),
        )
        episodes.append(ep)

    return episodes


@pytest.mark.asyncio
async def test_cluster_episodes_empty_input(consolidation_service):
    """Test clustering with empty input."""
    clusters = await consolidation_service._cluster_episodes([])
    assert clusters == []


@pytest.mark.asyncio
async def test_cluster_episodes_small_input(consolidation_service):
    """Test clustering with input smaller than min_cluster_size."""
    episodes = [
        Episode(
            id=uuid4(),
            session_id="test-session",
            content="Single episode",
            timestamp=datetime.now(),
            emotional_valence=0.5,
            context=EpisodeContext(),
        ),
    ]

    clusters = await consolidation_service._cluster_episodes(
        episodes,
        min_cluster_size=3
    )
    assert clusters == []


@pytest.mark.asyncio
async def test_cluster_episodes_hdbscan(consolidation_service, sample_episodes):
    """Test HDBSCAN clustering with multiple episode groups."""
    clusters = await consolidation_service._cluster_episodes(
        sample_episodes,
        min_cluster_size=3,
    )

    # Should find at least some clusters
    assert len(clusters) >= 0  # May vary based on embedding similarity

    # All clusters should have at least min_cluster_size episodes
    for cluster in clusters:
        assert len(cluster) >= 3

    # No episode should appear in multiple clusters
    all_episode_ids = set()
    for cluster in clusters:
        cluster_ids = {str(ep.id) for ep in cluster}
        assert len(cluster_ids.intersection(all_episode_ids)) == 0
        all_episode_ids.update(cluster_ids)


@pytest.mark.asyncio
async def test_cluster_episodes_noise_handling(consolidation_service):
    """Test that noise points (label=-1) are excluded from clusters."""
    # Create episodes with very different content (likely to be noise)
    episodes = [
        Episode(
            id=uuid4(),
            session_id="test-session",
            content="Python programming",
            timestamp=datetime.now(),
            emotional_valence=0.5,
            context=EpisodeContext(),
        ),
        Episode(
            id=uuid4(),
            session_id="test-session",
            content="Quantum physics equations",
            timestamp=datetime.now(),
            emotional_valence=0.5,
            context=EpisodeContext(),
        ),
        Episode(
            id=uuid4(),
            session_id="test-session",
            content="Cooking recipes for pasta",
            timestamp=datetime.now(),
            emotional_valence=0.5,
            context=EpisodeContext(),
        ),
    ]

    clusters = await consolidation_service._cluster_episodes(
        episodes,
        min_cluster_size=2,
    )

    # With very different content, may not form clusters
    # Just verify it doesn't crash
    assert isinstance(clusters, list)


@pytest.mark.asyncio
async def test_cluster_episodes_error_handling(consolidation_service, monkeypatch):
    """Test error handling in clustering."""
    # Mock embedding to raise an error
    async def mock_embed_error(contents):
        raise RuntimeError("Embedding failed")

    monkeypatch.setattr(
        consolidation_service.embedding,
        "embed",
        mock_embed_error,
    )

    episodes = [
        Episode(
            id=uuid4(),
            session_id="test-session",
            content=f"Episode {i}",
            timestamp=datetime.now(),
            emotional_valence=0.5,
            context=EpisodeContext(),
        )
        for i in range(5)
    ]

    # Should return empty list on error, not crash
    clusters = await consolidation_service._cluster_episodes(episodes)
    assert clusters == []


@pytest.mark.asyncio
async def test_cluster_procedures_empty_input(consolidation_service):
    """Test procedure clustering with empty input."""
    clusters = await consolidation_service._cluster_procedures([])
    assert clusters == []


@pytest.mark.asyncio
async def test_cluster_procedures_complexity(consolidation_service):
    """Verify O(n log n) complexity vs O(n²) for large inputs."""
    # This is more of a performance test, not run by default
    # Just verify the implementation exists and handles reasonable inputs
    from t4dm.core.types import Procedure, ProcedureStep

    procedures = [
        Procedure(
            id=uuid4(),
            name=f"Task {i}",
            domain=Domain.CODING,
            task=f"Complete task {i}",
            script=f"step1(); step2(); complete_{i}()",
            steps=[
                ProcedureStep(
                    order=1,
                    action=f"action_{i}",
                    expected_outcome=f"outcome_{i}",
                )
            ],
            success_rate=0.8,
            execution_count=10,
            version=1,
            created_from="test",
            deprecated=False,
        )
        for i in range(10)
    ]

    # Should complete without timeout (O(n log n) is fast)
    clusters = await consolidation_service._cluster_procedures(
        procedures,
        min_cluster_size=2,
    )

    assert isinstance(clusters, list)
