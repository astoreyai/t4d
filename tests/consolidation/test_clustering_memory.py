"""Tests for memory-safe HDBSCAN clustering."""
import pytest
import numpy as np
import os
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import patch, MagicMock

# Set test mode to bypass password validation
os.environ["WW_TEST_MODE"] = "true"

# Skip entire module if HDBSCAN not available
hdbscan = pytest.importorskip("hdbscan", reason="HDBSCAN not installed (pip install hdbscan)")

from ww.consolidation.service import ConsolidationService
from ww.core.types import Episode


class TestStratifiedSampling:
    """Tests for stratified sampling."""

    def test_sampling_preserves_temporal_distribution(self):
        """Test that samples are evenly distributed across time."""
        # Create episodes spread across 30 days
        episodes = []
        base_time = datetime.now()
        for i in range(1000):
            episodes.append(Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                timestamp=base_time + timedelta(days=i/33),
                embedding=[0.1] * 1024,
            ))

        service = ConsolidationService()
        sampled = service._stratified_sample(episodes, 100)

        assert len(sampled) == 100

        # Check temporal distribution
        sampled_times = [e.timestamp for e in sampled]
        time_gaps = [
            (sampled_times[i+1] - sampled_times[i]).total_seconds()
            for i in range(len(sampled_times) - 1)
        ]

        # Gaps should be roughly equal (±50%)
        avg_gap = sum(time_gaps) / len(time_gaps)
        for gap in time_gaps:
            assert gap > avg_gap * 0.5 and gap < avg_gap * 1.5

    def test_sampling_returns_all_if_under_limit(self):
        """Test that no sampling occurs if under limit."""
        episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                timestamp=datetime.now(),
            )
            for i in range(50)
        ]

        service = ConsolidationService()
        sampled = service._stratified_sample(episodes, 100)

        assert len(sampled) == 50
        assert sampled == episodes

    def test_sampling_handles_mixed_timestamps(self):
        """Test sampling with mixed old and new timestamps."""
        base_time = datetime(2020, 1, 1)
        episodes = []

        # Half old, half new timestamps
        for i in range(100):
            if i % 2 == 0:
                ts = base_time + timedelta(days=i)
            else:
                ts = datetime.now() + timedelta(days=i)

            episodes.append(Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                timestamp=ts,
            ))

        service = ConsolidationService()
        sampled = service._stratified_sample(episodes, 50)

        assert len(sampled) == 50
        # Should handle the wide time range without error


class TestClusterAssignment:
    """Tests for cluster assignment."""

    @pytest.mark.asyncio
    async def test_non_sampled_assigned_to_nearest(self):
        """Test that non-sampled episodes are assigned to nearest cluster."""
        # Create clusters with distinct embeddings
        cluster1_embedding = [1.0, 0.0, 0.0] + [0.0] * 1021
        cluster2_embedding = [0.0, 1.0, 0.0] + [0.0] * 1021

        cluster1 = [
            Episode(
                id=uuid4(),
                session_id="test",
                content="C1",
                embedding=cluster1_embedding,
            )
            for _ in range(3)
        ]
        cluster2 = [
            Episode(
                id=uuid4(),
                session_id="test",
                content="C2",
                embedding=cluster2_embedding,
            )
            for _ in range(3)
        ]

        # Non-sampled episode closer to cluster1
        non_sampled = Episode(
            id=uuid4(),
            session_id="test",
            content="NS",
            embedding=[0.9, 0.1, 0.0] + [0.0] * 1021,
        )

        service = ConsolidationService()
        expanded = await service._assign_to_clusters(
            all_episodes=cluster1 + cluster2 + [non_sampled],
            clusters=[cluster1, cluster2],
            sampled_episodes=cluster1 + cluster2,
        )

        # Non-sampled should be in cluster1
        assert non_sampled in expanded[0]
        assert non_sampled not in expanded[1]

    @pytest.mark.asyncio
    async def test_assignment_handles_no_embedding(self):
        """Test that episodes without embeddings are skipped."""
        cluster1 = [
            Episode(
                id=uuid4(),
                session_id="test",
                content="C1",
                embedding=[1.0] * 1024,
            )
            for _ in range(3)
        ]

        # Episode without embedding
        no_embed = Episode(
            id=uuid4(),
            session_id="test",
            content="No embedding",
            embedding=None,
        )

        service = ConsolidationService()
        expanded = await service._assign_to_clusters(
            all_episodes=cluster1 + [no_embed],
            clusters=[cluster1],
            sampled_episodes=cluster1,
        )

        # Should still have just cluster1 (no_embed skipped)
        assert len(expanded) == 1
        assert no_embed not in expanded[0]

    @pytest.mark.asyncio
    async def test_assignment_handles_empty_clusters(self):
        """Test assignment with no clusters."""
        episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Ep {i}",
                embedding=[0.5] * 1024,
            )
            for i in range(5)
        ]

        service = ConsolidationService()
        expanded = await service._assign_to_clusters(
            all_episodes=episodes,
            clusters=[],
            sampled_episodes=episodes,
        )

        assert expanded == []

    @pytest.mark.asyncio
    async def test_assignment_handles_no_non_sampled(self):
        """Test assignment when all episodes were sampled."""
        cluster1 = [
            Episode(
                id=uuid4(),
                session_id="test",
                content="C1",
                embedding=[1.0] * 1024,
            )
            for _ in range(3)
        ]

        service = ConsolidationService()
        expanded = await service._assign_to_clusters(
            all_episodes=cluster1,
            clusters=[cluster1],
            sampled_episodes=cluster1,
        )

        # Should return clusters unchanged
        assert len(expanded) == 1
        assert len(expanded[0]) == 3


class TestHDBSCANMemoryLimit:
    """Tests for HDBSCAN memory limiting."""

    @pytest.mark.asyncio
    async def test_clustering_large_dataset_samples(self):
        """Test that large datasets are sampled before clustering."""
        # Create 10,000 episodes (more than default limit)
        episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                timestamp=datetime.now() + timedelta(minutes=i),
                embedding=np.random.rand(1024).tolist(),
            )
            for i in range(10000)
        ]

        service = ConsolidationService()

        # Mock HDBSCAN to track what it receives
        with patch("ww.consolidation.service.HDBSCAN") as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.zeros(5000)  # 5000 samples
            mock_hdbscan.return_value = mock_clusterer

            # Mock embedding to return the episode embeddings
            with patch.object(service.embedding, 'embed') as mock_embed:
                async def mock_embed_fn(contents):
                    # Return embeddings for the sampled episodes
                    return [episodes[0].embedding] * len(contents)
                mock_embed.side_effect = mock_embed_fn

                clusters = await service._cluster_episodes(episodes, max_samples=5000)

                # HDBSCAN should have received only 5000 samples
                call_args = mock_clusterer.fit_predict.call_args
                assert call_args is not None
                embeddings_passed = call_args[0][0]
                assert len(embeddings_passed) == 5000

    @pytest.mark.asyncio
    async def test_clustering_small_dataset_no_sampling(self):
        """Test that small datasets are not sampled."""
        episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                embedding=np.random.rand(1024).tolist(),
            )
            for i in range(100)
        ]

        service = ConsolidationService()

        with patch("ww.consolidation.service.HDBSCAN") as mock_hdbscan:
            mock_clusterer = MagicMock()
            mock_clusterer.fit_predict.return_value = np.zeros(100)
            mock_hdbscan.return_value = mock_clusterer

            with patch.object(service.embedding, 'embed') as mock_embed:
                async def mock_embed_fn(contents):
                    return [episodes[0].embedding] * len(contents)
                mock_embed.side_effect = mock_embed_fn

                clusters = await service._cluster_episodes(episodes, max_samples=5000)

                # HDBSCAN should receive all 100 episodes
                call_args = mock_clusterer.fit_predict.call_args
                assert call_args is not None
                embeddings_passed = call_args[0][0]
                assert len(embeddings_passed) == 100

    @pytest.mark.asyncio
    async def test_memory_error_fallback(self):
        """Test fallback when HDBSCAN raises MemoryError."""
        episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                embedding=[0.1] * 1024,
            )
            for i in range(100)
        ]

        service = ConsolidationService()

        with patch("ww.consolidation.service.HDBSCAN") as mock_hdbscan:
            mock_hdbscan.return_value.fit_predict.side_effect = MemoryError("OOM")

            with patch.object(service.embedding, 'embed') as mock_embed:
                async def mock_embed_fn(contents):
                    return [[0.1] * 1024] * len(contents)
                mock_embed.side_effect = mock_embed_fn

                clusters = await service._cluster_episodes(episodes)

                # Should return single cluster as fallback
                assert len(clusters) == 1
                assert len(clusters[0]) == 100

    @pytest.mark.asyncio
    async def test_memory_error_fallback_small_dataset(self):
        """Test fallback with dataset smaller than min_cluster_size."""
        episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Episode {i}",
                embedding=[0.1] * 1024,
            )
            for i in range(2)
        ]

        service = ConsolidationService()

        with patch("ww.consolidation.service.HDBSCAN") as mock_hdbscan:
            mock_hdbscan.return_value.fit_predict.side_effect = MemoryError("OOM")

            with patch.object(service.embedding, 'embed') as mock_embed:
                async def mock_embed_fn(contents):
                    return [[0.1] * 1024] * len(contents)
                mock_embed.side_effect = mock_embed_fn

                clusters = await service._cluster_episodes(episodes)

                # Should return empty (too small for clustering)
                assert len(clusters) == 0

    @pytest.mark.asyncio
    async def test_clustering_assigns_non_sampled(self):
        """Test that non-sampled episodes are assigned to clusters."""
        # Create 100 episodes in 2 distinct groups
        group1_episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Group 1 Episode {i}",
                timestamp=datetime.now() + timedelta(minutes=i),
                embedding=[1.0, 0.0] + [0.0] * 1022,
            )
            for i in range(50)
        ]
        group2_episodes = [
            Episode(
                id=uuid4(),
                session_id="test",
                content=f"Group 2 Episode {i}",
                timestamp=datetime.now() + timedelta(minutes=i + 50),
                embedding=[0.0, 1.0] + [0.0] * 1022,
            )
            for i in range(50)
        ]

        all_episodes = group1_episodes + group2_episodes

        service = ConsolidationService()

        with patch("ww.consolidation.service.HDBSCAN") as mock_hdbscan:
            # Mock to return 2 clusters from sampled data
            mock_clusterer = MagicMock()

            def mock_fit_predict(embeddings):
                # First 5 get label 0, next 5 get label 1
                labels = np.zeros(len(embeddings))
                labels[5:] = 1
                return labels

            mock_clusterer.fit_predict.side_effect = mock_fit_predict
            mock_hdbscan.return_value = mock_clusterer

            with patch.object(service.embedding, 'embed') as mock_embed:
                async def mock_embed_fn(contents):
                    # Return actual embeddings
                    result = []
                    for content in contents:
                        if "Group 1" in content:
                            result.append([1.0, 0.0] + [0.0] * 1022)
                        else:
                            result.append([0.0, 1.0] + [0.0] * 1022)
                    return result
                mock_embed.side_effect = mock_embed_fn

                # Sample to only 10 episodes
                clusters = await service._cluster_episodes(all_episodes, max_samples=10)

                # Should have 2 clusters
                assert len(clusters) == 2

                # All episodes should be assigned (10 sampled + 90 assigned)
                total_episodes = sum(len(c) for c in clusters)
                assert total_episodes == 100


class TestConfigParameters:
    """Tests for configuration parameters."""

    def test_hdbscan_max_samples_config(self):
        """Test that max_samples is configurable."""
        from ww.core.config import Settings

        settings = Settings(hdbscan_max_samples=10000)
        assert settings.hdbscan_max_samples == 10000

    def test_hdbscan_max_samples_default(self):
        """Test default max_samples value."""
        from ww.core.config import Settings

        settings = Settings()
        assert settings.hdbscan_max_samples == 5000

    def test_hdbscan_max_samples_validation_too_small(self):
        """Test max_samples validation - too small."""
        from ww.core.config import Settings
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            Settings(hdbscan_max_samples=50)

        assert "hdbscan_max_samples" in str(exc_info.value)

    def test_hdbscan_max_samples_validation_too_large(self):
        """Test max_samples validation - too large."""
        from ww.core.config import Settings
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            Settings(hdbscan_max_samples=100000)

        assert "hdbscan_max_samples" in str(exc_info.value)

    def test_service_uses_config_max_samples(self):
        """Test that service uses configured max_samples."""
        from ww.core.config import Settings, get_settings
        from functools import lru_cache

        # Clear cache
        get_settings.cache_clear()

        # Mock settings
        with patch("ww.consolidation.service.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.hdbscan_max_samples = 1000
            mock_settings.hdbscan_min_cluster_size = 3
            mock_settings.hdbscan_min_samples = None
            mock_settings.hdbscan_metric = "cosine"
            mock_settings.consolidation_min_similarity = 0.75
            mock_settings.consolidation_min_occurrences = 3
            mock_settings.consolidation_skill_similarity = 0.85
            mock_get_settings.return_value = mock_settings

            service = ConsolidationService()
            assert service.hdbscan_max_samples == 1000
