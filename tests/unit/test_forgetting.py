"""
Tests for PO-2: Active Forgetting System.

Tests memory forgetting mechanisms for bounded memory growth.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from t4dm.memory.forgetting import (
    ActiveForgettingSystem,
    ForgettingCandidate,
    ForgettingResult,
    ForgettingStrategy,
    RetentionPolicy,
    get_forgetting_system,
    reset_forgetting_system,
)


class TestRetentionPolicy:
    """Test RetentionPolicy configuration."""

    def test_default_policy(self):
        """Default policy has reasonable values."""
        policy = RetentionPolicy()
        assert policy.max_episodes == 100_000
        assert policy.max_age_days == 365
        assert policy.min_importance == 0.1

    def test_custom_policy(self):
        """Custom policy values are preserved."""
        policy = RetentionPolicy(
            max_episodes=50_000,
            max_age_days=180,
            min_importance=0.2,
        )
        assert policy.max_episodes == 50_000
        assert policy.max_age_days == 180
        assert policy.min_importance == 0.2


class TestDecayScore:
    """Test decay-based forgetting score."""

    @pytest.fixture
    def system(self):
        """Create forgetting system for testing."""
        return ActiveForgettingSystem(
            policy=RetentionPolicy(max_age_days=100, access_decay_days=30),
            strategy=ForgettingStrategy.DECAY,
        )

    def test_old_memory_high_score(self, system):
        """Old memories have high decay score."""
        created_at = datetime.now() - timedelta(days=90)
        score = system.compute_decay_score(
            created_at=created_at,
            last_accessed=None,
            access_count=0,
        )
        assert score > 0.7  # Old, never accessed

    def test_new_memory_low_score(self, system):
        """New, frequently accessed memories have low score."""
        created_at = datetime.now() - timedelta(days=1)
        last_accessed = datetime.now() - timedelta(hours=1)
        score = system.compute_decay_score(
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=100,
        )
        assert score < 0.3  # New, recently accessed, many accesses

    def test_access_count_reduces_score(self, system):
        """More accesses reduce decay score."""
        created_at = datetime.now() - timedelta(days=30)

        score_low = system.compute_decay_score(created_at, None, access_count=1)
        score_high = system.compute_decay_score(created_at, None, access_count=100)

        assert score_low > score_high


class TestValueScore:
    """Test value-based forgetting score."""

    @pytest.fixture
    def system(self):
        """Create forgetting system for testing."""
        return ActiveForgettingSystem(
            policy=RetentionPolicy(
                min_importance=0.1,
                critical_importance=0.9,
            ),
            strategy=ForgettingStrategy.VALUE,
        )

    def test_low_importance_high_score(self, system):
        """Low importance memories have high forgetting score."""
        score = system.compute_value_score(importance=0.05)
        assert score == 1.0  # Below min_importance

    def test_critical_importance_zero_score(self, system):
        """Critical importance memories never forgotten."""
        score = system.compute_value_score(importance=0.95)
        assert score == 0.0  # Above critical threshold

    def test_medium_importance_medium_score(self, system):
        """Medium importance has proportional score."""
        score = system.compute_value_score(importance=0.5)
        assert 0.4 < score < 0.6  # Inverted importance


class TestInterferenceScore:
    """Test interference-based forgetting score."""

    @pytest.fixture
    def system(self):
        """Create forgetting system for testing."""
        return ActiveForgettingSystem(
            policy=RetentionPolicy(
                interference_threshold=0.85,
                max_similar_memories=5,
            ),
            strategy=ForgettingStrategy.INTERFERENCE,
        )

    def test_no_similar_memories_zero_score(self, system):
        """No similar memories = no interference."""
        embedding = np.random.randn(128).astype(np.float32)
        score = system.compute_interference_score(
            memory_embedding=embedding,
            similar_embeddings=[],
            similar_importances=[],
        )
        assert score == 0.0

    def test_similar_important_memory_high_score(self, system):
        """Similar, more important memories cause interference."""
        embedding = np.ones(128, dtype=np.float32)
        similar = np.ones(128, dtype=np.float32) * 0.99  # Very similar

        score = system.compute_interference_score(
            memory_embedding=embedding,
            similar_embeddings=[similar],
            similar_importances=[0.9],  # More important
        )
        assert score > 0.3  # High interference

    def test_dissimilar_memories_low_score(self, system):
        """Dissimilar memories don't cause interference."""
        embedding = np.ones(128, dtype=np.float32)
        dissimilar = -np.ones(128, dtype=np.float32)  # Opposite

        score = system.compute_interference_score(
            memory_embedding=embedding,
            similar_embeddings=[dissimilar],
            similar_importances=[0.9],
        )
        assert score == 0.0  # No interference


class TestHybridScore:
    """Test hybrid forgetting strategy."""

    @pytest.fixture
    def system(self):
        """Create hybrid forgetting system."""
        return ActiveForgettingSystem(strategy=ForgettingStrategy.HYBRID)

    def test_hybrid_combines_scores(self, system):
        """Hybrid strategy combines decay and value scores."""
        created_at = datetime.now() - timedelta(days=60)
        score, reasons = system.compute_forgetting_score(
            created_at=created_at,
            last_accessed=None,
            access_count=0,
            importance=0.2,
        )

        # Should be moderately high (old + low importance)
        assert score > 0.5
        assert len(reasons) >= 1


class TestIdentifyCandidates:
    """Test candidate identification."""

    @pytest.fixture
    def system(self):
        """Create forgetting system."""
        return ActiveForgettingSystem(strategy=ForgettingStrategy.HYBRID)

    def test_identify_old_low_importance(self, system):
        """Identifies old, low importance memories as candidates."""
        memories = [
            {
                "id": "mem1",
                "type": "episode",
                "created_at": datetime.now() - timedelta(days=100),
                "access_count": 0,
                "importance": 0.1,
            },
            {
                "id": "mem2",
                "type": "episode",
                "created_at": datetime.now() - timedelta(days=1),
                "access_count": 50,
                "importance": 0.9,
            },
        ]

        candidates = system.identify_candidates(memories, threshold=0.5)

        # Only mem1 should be a candidate
        assert len(candidates) == 1
        assert candidates[0].memory_id == "mem1"

    def test_candidates_sorted_by_score(self, system):
        """Candidates sorted by forgetting score (highest first)."""
        memories = [
            {
                "id": "mem1",
                "created_at": datetime.now() - timedelta(days=30),
                "access_count": 5,
                "importance": 0.3,
            },
            {
                "id": "mem2",
                "created_at": datetime.now() - timedelta(days=90),
                "access_count": 0,
                "importance": 0.1,
            },
            {
                "id": "mem3",
                "created_at": datetime.now() - timedelta(days=60),
                "access_count": 2,
                "importance": 0.2,
            },
        ]

        candidates = system.identify_candidates(memories, threshold=0.0)

        # Should be sorted by score (highest first)
        for i in range(len(candidates) - 1):
            assert candidates[i].forgetting_score >= candidates[i + 1].forgetting_score


class TestForgettingCycle:
    """Test forgetting cycle execution."""

    @pytest.fixture
    def system(self):
        """Create forgetting system with low thresholds for testing."""
        return ActiveForgettingSystem(
            policy=RetentionPolicy(
                max_episodes=100,
                soft_limit_ratio=0.5,  # 50 = soft limit
            ),
        )

    @pytest.fixture
    def mock_store(self):
        """Create mock memory store."""
        store = AsyncMock()
        store.get_memory_count = AsyncMock(return_value=60)  # Above soft limit
        store.get_oldest_memories = AsyncMock(
            return_value=[
                {
                    "id": f"mem{i}",
                    "created_at": datetime.now() - timedelta(days=100 + i),
                    "access_count": 0,
                    "importance": 0.1,
                }
                for i in range(10)
            ]
        )
        store.get_least_accessed = AsyncMock(return_value=[])
        store.delete_memory = AsyncMock(return_value=True)
        store.archive_memory = AsyncMock(return_value=True)
        return store

    @pytest.mark.asyncio
    async def test_dry_run_no_delete(self, system, mock_store):
        """Dry run doesn't delete memories."""
        result = await system.run_forgetting_cycle(
            store=mock_store,
            max_to_forget=5,
            dry_run=True,
        )

        assert result.memories_forgotten == 5
        mock_store.delete_memory.assert_not_called()

    @pytest.mark.asyncio
    async def test_below_soft_limit_skips(self, system, mock_store):
        """Skips cycle when below soft limit."""
        mock_store.get_memory_count = AsyncMock(return_value=40)  # Below 50

        result = await system.run_forgetting_cycle(
            store=mock_store,
            max_to_forget=5,
        )

        assert result.candidates_evaluated == 0
        assert result.memories_forgotten == 0

    @pytest.mark.asyncio
    async def test_deletes_candidates(self, system, mock_store):
        """Actually deletes candidates when not dry run."""
        result = await system.run_forgetting_cycle(
            store=mock_store,
            max_to_forget=3,
            dry_run=False,
        )

        assert result.memories_forgotten == 3
        assert mock_store.delete_memory.call_count == 3


class TestSingleton:
    """Test singleton pattern."""

    def test_get_forgetting_system_singleton(self):
        """get_forgetting_system returns same instance."""
        reset_forgetting_system()
        s1 = get_forgetting_system()
        s2 = get_forgetting_system()
        assert s1 is s2

    def test_reset_forgetting_system(self):
        """reset_forgetting_system creates new instance."""
        s1 = get_forgetting_system()
        reset_forgetting_system()
        s2 = get_forgetting_system()
        assert s1 is not s2


class TestStats:
    """Test statistics tracking."""

    def test_stats_tracking(self):
        """Stats are tracked correctly."""
        system = ActiveForgettingSystem()

        # Simulate some forgetting
        system._total_forgotten = 10
        system._total_archived = 5
        system._cycles_run = 3

        stats = system.get_stats()

        assert stats["total_forgotten"] == 10
        assert stats["total_archived"] == 5
        assert stats["cycles_run"] == 3
        assert stats["strategy"] == "hybrid"
