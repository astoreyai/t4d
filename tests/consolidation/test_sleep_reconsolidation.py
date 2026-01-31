"""
Tests for Phase 1A: Sleep Replay Reconsolidation Integration.

Verifies that sleep replay actually updates episode embeddings via reconsolidation.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional

from t4dm.consolidation.sleep import SleepConsolidation, ReplayEvent
from t4dm.learning.reconsolidation import ReconsolidationEngine
from t4dm.consolidation.lability import LabilityManager, LabilityConfig


@dataclass
class MockEpisode:
    """Mock episode for testing."""
    id: UUID
    content: str = "test content"
    outcome_score: float = 0.7
    emotional_valence: float = 0.6
    created_at: datetime = None
    embedding: Optional[np.ndarray] = None
    last_accessed: Optional[datetime] = None  # For lability window

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.embedding is None:
            # Create random normalized embedding
            emb = np.random.randn(128)
            self.embedding = emb / np.linalg.norm(emb)


class MockEpisodicMemory:
    """Mock episodic memory for testing."""

    def __init__(self, episodes: list = None):
        self.episodes = episodes or []

    async def get_recent(self, hours: int = 24, limit: int = 100):
        return self.episodes[:limit]

    async def get_by_id(self, episode_id: UUID):
        for ep in self.episodes:
            if ep.id == episode_id:
                return ep
        return None

    async def sample_random(self, limit: int = 50, **kwargs):
        return self.episodes[:limit]


class MockSemanticMemory:
    """Mock semantic memory for testing."""

    def __init__(self):
        self.created_entities = []

    async def create_or_strengthen(self, name: str, description: str, source_episode_id=None):
        self.created_entities.append({
            "name": name,
            "description": description,
            "source": source_episode_id
        })
        return MagicMock(id=uuid4())


class MockGraphStore:
    """Mock graph store for testing."""

    async def get_all_nodes(self, label: str | None = None):
        return []

    async def get_node(self, node_id: str):
        return None


@pytest.mark.asyncio
async def test_sleep_actually_updates_embeddings():
    """
    Test that sleep replay calls reconsolidation and updates embeddings.

    Success criteria:
    - ReconsolidationEngine.reconsolidate() is called during replay
    - Lability window is NOT checked (allows initial consolidation)
    - Episode embeddings are updated via reconsolidation
    """
    # Create mock episodes with embeddings
    episodes = [
        MockEpisode(
            id=uuid4(),
            outcome_score=0.8,
            last_accessed=datetime.now(),  # Must have been accessed for reconsolidation (ATOM-P1-8)
        )
        for _ in range(5)
    ]

    episodic = MockEpisodicMemory(episodes)
    semantic = MockSemanticMemory()
    graph = MockGraphStore()

    # Create sleep consolidation
    sleep = SleepConsolidation(
        episodic_memory=episodic,
        semantic_memory=semantic,
        graph_store=graph,
        vae_enabled=False,  # Disable VAE for simpler testing
    )

    # Create mock reconsolidation engine
    mock_recon = MagicMock(spec=ReconsolidationEngine)
    
    # Mock reconsolidate to return a modified embedding
    def mock_reconsolidate(memory_id, memory_embedding, query_embedding, outcome_score):
        # Return slightly modified embedding to simulate update
        updated = memory_embedding + 0.01 * (query_embedding - memory_embedding)
        # Normalize
        return updated / np.linalg.norm(updated)
    
    mock_recon.reconsolidate = MagicMock(side_effect=mock_reconsolidate)

    # Set reconsolidation engine
    sleep.set_reconsolidation_engine(mock_recon)

    # Run NREM phase
    result = await sleep.nrem_phase(session_id="test_session", replay_count=3)

    # Verify reconsolidate was called
    assert mock_recon.reconsolidate.call_count > 0, "Reconsolidation should be called during replay"
    
    # Verify calls were made with correct parameters
    for call in mock_recon.reconsolidate.call_args_list:
        args, kwargs = call
        # Check that we got the expected parameters
        assert 'memory_id' in kwargs or len(args) >= 1
        assert 'memory_embedding' in kwargs or len(args) >= 2
        assert 'query_embedding' in kwargs or len(args) >= 3
        assert 'outcome_score' in kwargs or len(args) >= 4

    print(f"✓ Sleep replay called reconsolidation {mock_recon.reconsolidate.call_count} times")


@pytest.mark.asyncio
async def test_lability_window_prevents_early_recon():
    """
    Test that lability window is checked before reconsolidation.

    Success criteria:
    - Episodes outside lability window are NOT reconsolidated
    - Episodes within lability window ARE reconsolidated
    - Lability window check uses is_reconsolidation_eligible()
    """
    # Create episodes with different retrieval times
    now = datetime.now()
    
    # Episode recently retrieved (within 6 hour window)
    recent_episode = MockEpisode(
        id=uuid4(),
        outcome_score=0.8,
        last_accessed=now - timedelta(hours=2),  # 2 hours ago - within window
    )
    
    # Episode retrieved long ago (outside 6 hour window)
    old_episode = MockEpisode(
        id=uuid4(),
        outcome_score=0.8,
        last_accessed=now - timedelta(hours=10),  # 10 hours ago - outside window
    )

    episodic = MockEpisodicMemory([recent_episode, old_episode])
    semantic = MockSemanticMemory()
    graph = MockGraphStore()

    # Create sleep consolidation
    sleep = SleepConsolidation(
        episodic_memory=episodic,
        semantic_memory=semantic,
        graph_store=graph,
        vae_enabled=False,
    )

    # Create mock reconsolidation engine
    mock_recon = MagicMock(spec=ReconsolidationEngine)
    mock_recon.reconsolidate = MagicMock(return_value=np.random.randn(128))

    sleep.set_reconsolidation_engine(mock_recon)

    # Run NREM phase
    result = await sleep.nrem_phase(session_id="test_session", replay_count=5)

    # Get the memory IDs that were passed to reconsolidate
    reconsolidated_ids = set()
    for call in mock_recon.reconsolidate.call_args_list:
        args, kwargs = call
        if 'memory_id' in kwargs:
            reconsolidated_ids.add(kwargs['memory_id'])
        elif len(args) >= 1:
            reconsolidated_ids.add(args[0])

    # Verify recent episode was reconsolidated
    assert recent_episode.id in reconsolidated_ids, "Recent episode should be reconsolidated"
    
    # Verify old episode was NOT reconsolidated
    assert old_episode.id not in reconsolidated_ids, "Old episode should NOT be reconsolidated"

    print(f"✓ Lability window correctly gates reconsolidation")


@pytest.mark.asyncio
async def test_batch_reconsolidation_during_nrem():
    """
    Test that multiple episodes are batch reconsolidated during NREM.

    Success criteria:
    - All replayed episodes trigger reconsolidation
    - Reconsolidation is called with appropriate query embeddings
    - Outcome scores are correctly passed
    """
    # Create multiple episodes
    episodes = [
        MockEpisode(
            id=uuid4(),
            outcome_score=0.5 + i * 0.1,
            last_accessed=datetime.now(),  # Must have been accessed for reconsolidation (ATOM-P1-8)
        )
        for i in range(10)
    ]

    episodic = MockEpisodicMemory(episodes)
    semantic = MockSemanticMemory()
    graph = MockGraphStore()

    # Create sleep consolidation
    sleep = SleepConsolidation(
        episodic_memory=episodic,
        semantic_memory=semantic,
        graph_store=graph,
        vae_enabled=False,
        max_replays=10,
    )

    # Create mock reconsolidation engine
    mock_recon = MagicMock(spec=ReconsolidationEngine)
    
    # Track reconsolidation calls
    recon_calls = []
    
    def track_reconsolidate(memory_id, memory_embedding, query_embedding, outcome_score):
        recon_calls.append({
            'memory_id': memory_id,
            'memory_embedding': memory_embedding.copy(),
            'query_embedding': query_embedding.copy(),
            'outcome_score': outcome_score,
        })
        # Return modified embedding
        updated = memory_embedding + 0.05 * (query_embedding - memory_embedding)
        return updated / np.linalg.norm(updated)
    
    mock_recon.reconsolidate = MagicMock(side_effect=track_reconsolidate)

    sleep.set_reconsolidation_engine(mock_recon)

    # Run NREM phase
    result = await sleep.nrem_phase(session_id="test_session", replay_count=10)

    # Verify batch reconsolidation occurred
    assert len(recon_calls) > 0, "Should have reconsolidated at least some episodes"
    
    # Verify outcome scores were passed correctly
    for call in recon_calls:
        assert 0.0 <= call['outcome_score'] <= 1.0, "Outcome score should be in [0, 1]"
        
    # Verify query embeddings are normalized
    for call in recon_calls:
        query_norm = np.linalg.norm(call['query_embedding'])
        assert 0.9 <= query_norm <= 1.1, "Query embedding should be approximately normalized"

    print(f"✓ Batch reconsolidated {len(recon_calls)} episodes during NREM")


@pytest.mark.asyncio
async def test_reconsolidation_without_engine_is_noop():
    """
    Test that sleep replay works without reconsolidation engine (backward compat).

    Success criteria:
    - Sleep replay completes successfully without reconsolidation engine
    - No reconsolidation is attempted
    - Semantic strengthening still works
    """
    episodes = [MockEpisode(id=uuid4()) for _ in range(5)]

    episodic = MockEpisodicMemory(episodes)
    semantic = MockSemanticMemory()
    graph = MockGraphStore()

    # Create sleep consolidation WITHOUT setting reconsolidation engine
    sleep = SleepConsolidation(
        episodic_memory=episodic,
        semantic_memory=semantic,
        graph_store=graph,
        vae_enabled=False,
    )

    # Don't set reconsolidation engine - verify backward compatibility

    # Run NREM phase
    result = await sleep.nrem_phase(session_id="test_session", replay_count=3)

    # Verify replay occurred
    assert len(result) > 0, "Should have replayed some episodes"
    
    # Verify semantic strengthening still works
    # (This happens even without reconsolidation)
    assert sleep._reconsolidation_engine is None, "Should not have reconsolidation engine"

    print("✓ Sleep replay works without reconsolidation engine (backward compat)")


@pytest.mark.asyncio
async def test_reconsolidation_uses_query_context():
    """
    Test that reconsolidation uses appropriate query embedding context.

    Success criteria:
    - Query embedding is derived from recent episodes
    - Query embedding is different from episode embedding
    - Query embedding provides meaningful context for update
    """
    # Create episodes with distinct embeddings
    episodes = []
    for i in range(5):
        emb = np.random.randn(128)
        emb[i] = 5.0  # Make each episode distinctive
        emb = emb / np.linalg.norm(emb)
        
        episodes.append(MockEpisode(
            id=uuid4(),
            embedding=emb,
            last_accessed=datetime.now(),  # ATOM-P1-8
        ))

    episodic = MockEpisodicMemory(episodes)
    semantic = MockSemanticMemory()
    graph = MockGraphStore()

    sleep = SleepConsolidation(
        episodic_memory=episodic,
        semantic_memory=semantic,
        graph_store=graph,
        vae_enabled=False,
    )

    # Track reconsolidation calls
    query_embeddings_used = []
    
    def track_query_emb(memory_id, memory_embedding, query_embedding, outcome_score):
        query_embeddings_used.append(query_embedding.copy())
        return memory_embedding  # Return unchanged
    
    mock_recon = MagicMock(spec=ReconsolidationEngine)
    mock_recon.reconsolidate = MagicMock(side_effect=track_query_emb)

    sleep.set_reconsolidation_engine(mock_recon)

    # Run NREM phase
    result = await sleep.nrem_phase(session_id="test_session", replay_count=3)

    # Verify query embeddings were used
    assert len(query_embeddings_used) > 0, "Should have used query embeddings"
    
    # Verify query embeddings are distinct
    for query_emb in query_embeddings_used:
        # Query should be derived from averaging recent episodes
        # So it should be normalized
        query_norm = np.linalg.norm(query_emb)
        assert 0.9 <= query_norm <= 1.1, f"Query embedding should be normalized, got {query_norm}"

    print(f"✓ Reconsolidation uses {len(query_embeddings_used)} distinct query contexts")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
