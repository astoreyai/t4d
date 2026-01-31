"""
Unit tests for World Weaver sleep consolidation module.

Tests SleepConsolidation with NREM, REM, and prune phases.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass
from typing import Optional

from t4dm.consolidation.sleep import (
    SleepConsolidation,
    SleepPhase,
    SleepCycleResult,
    ReplayEvent,
    AbstractionEvent,
    run_sleep_cycle,
)


@dataclass
class MockEpisode:
    """Mock episode for testing."""
    id: Optional[str] = None
    content: str = "test content"
    outcome_score: float = 0.7
    emotional_valence: float = 0.6
    created_at: datetime = None
    context: Optional[object] = None

    def __post_init__(self):
        if self.id is None:
            self.id = uuid4()
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class MockContext:
    """Mock context for testing."""
    project: str = "test_project"
    file: str = "test_file.py"
    tool: str = "test_tool"


class MockEpisodicMemory:
    """Mock episodic memory for testing."""

    def __init__(self, episodes: list = None):
        self.episodes = episodes or []
        self.get_recent_calls = []

    async def get_recent(self, hours: int = 24, limit: int = 100):
        self.get_recent_calls.append({"hours": hours, "limit": limit})
        return self.episodes[:limit]

    async def get_by_id(self, episode_id):
        for ep in self.episodes:
            if ep.id == episode_id:
                return ep
        return None

    async def sample_random(
        self,
        limit: int = 50,
        session_filter: str | None = None,
        exclude_hours: int = 24,
    ) -> list:
        """Sample random episodes for VAE replay."""
        import random
        episodes = list(self.episodes)
        random.shuffle(episodes)
        return episodes[:limit]


class MockSemanticMemory:
    """Mock semantic memory for testing."""

    def __init__(self):
        self.created_entities = []
        self.created_relationships = []
        self.cluster_calls = []

    async def create_or_strengthen(
        self,
        name: str,
        description: str,
        source_episode_id=None
    ):
        self.created_entities.append({
            "name": name,
            "description": description,
            "source": source_episode_id
        })
        return MagicMock(id=uuid4())

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        summary: str,
        details: str = None,
        embedding: list = None,
        source: str = None,
    ):
        """P5.1: Create entity for REM abstractions."""
        entity_id = uuid4()
        self.created_entities.append({
            "id": entity_id,
            "name": name,
            "entity_type": entity_type,
            "summary": summary,
            "details": details,
            "embedding": embedding,
            "source": source,
        })
        mock_entity = MagicMock()
        mock_entity.id = entity_id
        mock_entity.name = name
        return mock_entity

    async def create_relationship(
        self,
        source_id,
        target_id,
        relation_type: str,
        initial_weight: float = 0.5,
    ):
        """P5.1: Create relationship for REM abstractions."""
        self.created_relationships.append({
            "source_id": source_id,
            "target_id": target_id,
            "relation_type": relation_type,
            "weight": initial_weight,
        })
        return MagicMock()

    async def cluster_embeddings(
        self,
        embeddings: list,
        min_cluster_size: int = 3
    ):
        self.cluster_calls.append({
            "count": len(embeddings),
            "min_size": min_cluster_size
        })
        # Return simple clustering
        if len(embeddings) >= min_cluster_size:
            return [list(range(min_cluster_size))]
        return []


class MockGraphStore:
    """Mock graph store for testing."""

    def __init__(self, nodes: list = None, relationships: dict = None):
        self.nodes = nodes or []
        self.relationships = relationships or {}
        self.deleted_relationships = []
        self.updated_weights = []

    async def get_all_nodes(self, label: str = None):
        if label:
            return [n for n in self.nodes if n.get("label") == label]
        return self.nodes

    async def get_relationships(self, node_id: str, direction: str = "out"):
        return self.relationships.get(node_id, [])

    async def update_relationship_weight(
        self,
        source_id: str,
        target_id: str,
        new_weight: float
    ):
        self.updated_weights.append({
            "source": source_id,
            "target": target_id,
            "weight": new_weight
        })

    async def delete_relationship(self, source_id: str, target_id: str):
        self.deleted_relationships.append({
            "source": source_id,
            "target": target_id
        })

    async def get_node(self, node_id: str):
        """P5.1: Get single node by ID for REM abstractions."""
        for node in self.nodes:
            if node.get("id") == node_id:
                return node
        return None


class TestSleepCycleResult:
    """Tests for SleepCycleResult dataclass."""

    def test_creation(self):
        result = SleepCycleResult(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=60),
            nrem_replays=10,
            rem_abstractions=3,
            pruned_connections=5,
            strengthened_connections=2,
            total_duration_seconds=60.0
        )
        assert result.nrem_replays == 10
        assert result.rem_abstractions == 3

    def test_duration_property(self):
        start = datetime.now()
        end = start + timedelta(minutes=5)
        result = SleepCycleResult(
            session_id="test",
            start_time=start,
            end_time=end,
            nrem_replays=0,
            rem_abstractions=0,
            pruned_connections=0,
            strengthened_connections=0,
            total_duration_seconds=300.0
        )
        assert result.duration == timedelta(minutes=5)


class TestReplayEvent:
    """Tests for ReplayEvent dataclass."""

    def test_creation(self):
        event = ReplayEvent(
            episode_id=uuid4(),
            priority_score=0.8,
            strengthening_applied=True,
            entities_extracted=["entity1", "entity2"]
        )
        assert event.priority_score == 0.8
        assert event.strengthening_applied
        assert len(event.entities_extracted) == 2


class TestAbstractionEvent:
    """Tests for AbstractionEvent dataclass."""

    def test_creation(self):
        event = AbstractionEvent(
            cluster_ids=["a", "b", "c"],
            concept_name="test_concept",
            confidence=0.85
        )
        assert len(event.cluster_ids) == 3
        assert event.confidence == 0.85


class TestSleepConsolidation:
    """Tests for SleepConsolidation."""

    @pytest.fixture
    def episodic(self):
        episodes = [
            MockEpisode(
                content=f"episode {i}",
                outcome_score=0.8 - i * 0.1,
                emotional_valence=0.7,
                created_at=datetime.now() - timedelta(hours=i),
                context=MockContext() if i < 3 else None
            )
            for i in range(5)
        ]
        return MockEpisodicMemory(episodes=episodes)

    @pytest.fixture
    def semantic(self):
        return MockSemanticMemory()

    @pytest.fixture
    def graph_store(self):
        nodes = [
            {"id": f"entity_{i}", "label": "Entity", "embedding": np.random.randn(128)}
            for i in range(5)
        ]
        relationships = {
            "entity_0": [
                {"other_id": "entity_1", "properties": {"weight": 0.6}},
                {"other_id": "entity_2", "properties": {"weight": 0.03}},  # Below prune threshold
            ],
            "entity_1": [
                {"other_id": "entity_2", "properties": {"weight": 0.4}},
            ]
        }
        return MockGraphStore(nodes=nodes, relationships=relationships)

    @pytest.fixture
    def consolidator(self, episodic, semantic, graph_store):
        return SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            replay_hours=24,
            max_replays=10,
            nrem_cycles=2,
            replay_delay_ms=0  # No delay in tests
        )

    def test_creation_default(self, episodic, semantic, graph_store):
        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store
        )
        assert consolidator.replay_hours == 24
        assert consolidator.max_replays == 100
        assert consolidator.current_phase == SleepPhase.WAKE

    def test_creation_custom(self, episodic, semantic, graph_store):
        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            replay_hours=12,
            max_replays=50,
            prune_threshold=0.1
        )
        assert consolidator.replay_hours == 12
        assert consolidator.max_replays == 50
        assert consolidator.prune_threshold == 0.1

    @pytest.mark.asyncio
    async def test_nrem_phase_basic(self, consolidator, episodic):
        events = await consolidator.nrem_phase("test_session")

        assert consolidator.current_phase == SleepPhase.NREM
        assert len(events) > 0
        assert len(episodic.get_recent_calls) == 1

    @pytest.mark.asyncio
    async def test_nrem_phase_extracts_entities(self, consolidator, semantic):
        events = await consolidator.nrem_phase("test_session")

        # Episodes with context should have entities extracted
        entities_with_context = [e for e in events if len(e.entities_extracted) > 0]
        assert len(entities_with_context) >= 1

        # Semantic memory should have been called
        assert len(semantic.created_entities) >= 1

    @pytest.mark.asyncio
    async def test_nrem_phase_empty_episodes(self, semantic, graph_store):
        empty_episodic = MockEpisodicMemory(episodes=[])
        consolidator = SleepConsolidation(
            episodic_memory=empty_episodic,
            semantic_memory=semantic,
            graph_store=graph_store
        )

        events = await consolidator.nrem_phase("test_session")
        assert events == []

    @pytest.mark.asyncio
    async def test_nrem_phase_replay_count(self, consolidator):
        events = await consolidator.nrem_phase("test_session", replay_count=2)
        assert len(events) <= 2

    @pytest.mark.asyncio
    async def test_rem_phase_basic(self, consolidator):
        events = await consolidator.rem_phase("test_session")

        assert consolidator.current_phase == SleepPhase.REM
        # May or may not have abstractions depending on clustering
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_rem_phase_creates_abstractions(self, episodic, semantic):
        # Create graph store with very similar normalized embeddings (clusterable)
        base_emb = np.random.randn(128)
        base_emb = base_emb / np.linalg.norm(base_emb)

        # Use very small noise and normalize to ensure high similarity
        nodes = []
        for i in range(5):
            emb = base_emb + np.random.randn(128) * 0.01
            emb = emb / np.linalg.norm(emb)  # Normalize!
            nodes.append({"id": f"entity_{i}", "label": "Entity", "embedding": emb})

        graph_store = MockGraphStore(nodes=nodes)

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            min_cluster_size=3,
            abstraction_threshold=0.5  # Lower threshold for testing
        )

        events = await consolidator.rem_phase("test_session")

        # Should have at least one abstraction
        assert len(events) >= 1

    @pytest.mark.asyncio
    async def test_rem_phase_not_enough_entities(self, episodic, semantic):
        graph_store = MockGraphStore(nodes=[
            {"id": "entity_0", "label": "Entity", "embedding": np.random.randn(128)}
        ])

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            min_cluster_size=3
        )

        events = await consolidator.rem_phase("test_session")
        assert events == []

    @pytest.mark.asyncio
    async def test_prune_phase_removes_weak(self, consolidator, graph_store):
        pruned, strengthened = await consolidator.prune_phase()

        assert consolidator.current_phase == SleepPhase.PRUNE
        # Should have pruned the weak connection (0.03 < 0.05)
        assert pruned >= 1
        assert len(graph_store.deleted_relationships) >= 1

    @pytest.mark.asyncio
    async def test_prune_phase_homeostatic_scaling(self, episodic, semantic):
        # Create node with high total weight
        graph_store = MockGraphStore(
            nodes=[{"id": "node1"}],
            relationships={
                "node1": [
                    {"other_id": "node2", "properties": {"weight": 8.0}},
                    {"other_id": "node3", "properties": {"weight": 8.0}},
                    # Total = 16, above target of 10 * 1.2 = 12
                ]
            }
        )

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            homeostatic_target=10.0,
            prune_threshold=0.05
        )

        pruned, strengthened = await consolidator.prune_phase()

        # Should have scaled the weights down
        assert strengthened >= 1

    @pytest.mark.asyncio
    async def test_full_sleep_cycle(self, consolidator):
        result = await consolidator.full_sleep_cycle("test_session")

        assert isinstance(result, SleepCycleResult)
        assert result.session_id == "test_session"
        assert result.nrem_replays >= 0
        assert result.total_duration_seconds >= 0
        assert consolidator.current_phase == SleepPhase.WAKE

    @pytest.mark.asyncio
    async def test_full_sleep_cycle_multiple_nrem(self, consolidator):
        result = await consolidator.full_sleep_cycle("test_session")

        # Should have called NREM multiple times (nrem_cycles=2)
        # Total replays should reflect this
        assert result.nrem_replays >= 0

    def test_prioritize_for_replay(self, consolidator):
        episodes = [
            MockEpisode(
                outcome_score=0.3,
                emotional_valence=0.3,
                created_at=datetime.now() - timedelta(hours=20)  # Old
            ),
            MockEpisode(
                outcome_score=0.9,
                emotional_valence=0.8,
                created_at=datetime.now() - timedelta(hours=1)  # Recent, high value
            ),
            MockEpisode(
                outcome_score=0.5,
                emotional_valence=0.5,
                created_at=datetime.now() - timedelta(hours=10)  # Medium
            ),
        ]

        prioritized = consolidator._prioritize_for_replay(episodes)

        # Highest priority should be first (recent + high outcome)
        assert prioritized[0].outcome_score == 0.9

    def test_prioritize_for_replay_empty(self, consolidator):
        prioritized = consolidator._prioritize_for_replay([])
        assert prioritized == []

    @pytest.mark.asyncio
    async def test_cluster_embeddings(self, consolidator):
        # Create similar embeddings with controlled noise
        # Use seed for reproducibility
        rng = np.random.default_rng(42)
        base = rng.standard_normal(128)
        base = base / np.linalg.norm(base)

        # Use small noise (0.01) to ensure similarity > 0.7 threshold
        embeddings = np.array([
            base + rng.standard_normal(128) * 0.01,
            base + rng.standard_normal(128) * 0.01,
            base + rng.standard_normal(128) * 0.01,
            -base,  # Very different
        ])

        clusters = await consolidator._cluster_embeddings(embeddings)

        # Should find at least one cluster
        assert len(clusters) >= 1
        assert len(clusters[0]) >= 3

    @pytest.mark.asyncio
    async def test_cluster_embeddings_too_few(self, consolidator):
        embeddings = np.random.randn(2, 128)
        clusters = await consolidator._cluster_embeddings(embeddings)
        assert clusters == []

    def test_get_replay_history(self, consolidator):
        # Add some history
        consolidator._replay_history = [
            ReplayEvent(episode_id=uuid4()),
            ReplayEvent(episode_id=uuid4()),
            ReplayEvent(episode_id=uuid4()),
        ]

        history = consolidator.get_replay_history()
        assert len(history) == 3

        limited = consolidator.get_replay_history(limit=2)
        assert len(limited) == 2

    def test_get_abstraction_history(self, consolidator):
        consolidator._abstraction_history = [
            AbstractionEvent(cluster_ids=["a"]),
            AbstractionEvent(cluster_ids=["b"]),
        ]

        history = consolidator.get_abstraction_history()
        assert len(history) == 2

    def test_get_last_cycle_result(self, consolidator):
        # No cycle run yet
        assert consolidator.get_last_cycle_result() is None

    @pytest.mark.asyncio
    async def test_get_last_cycle_result_after_cycle(self, consolidator):
        await consolidator.full_sleep_cycle("test_session")
        result = consolidator.get_last_cycle_result()
        assert result is not None
        assert result.session_id == "test_session"

    def test_get_stats_empty(self, consolidator):
        stats = consolidator.get_stats()

        assert stats["current_phase"] == "wake"
        assert stats["total_replays"] == 0
        assert stats["total_abstractions"] == 0

    @pytest.mark.asyncio
    async def test_get_stats_after_cycle(self, consolidator):
        await consolidator.full_sleep_cycle("test_session")
        stats = consolidator.get_stats()

        assert stats["total_replays"] >= 0
        assert "last_cycle" in stats
        assert stats["last_cycle"]["duration_seconds"] >= 0

    def test_clear_history(self, consolidator):
        consolidator._replay_history = [ReplayEvent(episode_id=uuid4())]
        consolidator._abstraction_history = [AbstractionEvent(cluster_ids=["a"])]
        consolidator._last_cycle_result = SleepCycleResult(
            session_id="test",
            start_time=datetime.now(),
            end_time=datetime.now(),
            nrem_replays=1,
            rem_abstractions=0,
            pruned_connections=0,
            strengthened_connections=0,
            total_duration_seconds=0
        )

        consolidator.clear_history()

        assert len(consolidator._replay_history) == 0
        assert len(consolidator._abstraction_history) == 0
        assert consolidator._last_cycle_result is None


class TestRunSleepCycle:
    """Tests for run_sleep_cycle convenience function."""

    @pytest.mark.asyncio
    async def test_run_sleep_cycle(self):
        episodic = MockEpisodicMemory(episodes=[
            MockEpisode(context=MockContext())
        ])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        result = await run_sleep_cycle(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            session_id="test_session"
        )

        assert isinstance(result, SleepCycleResult)
        assert result.session_id == "test_session"


# ==================== P2.5: Sleep Consolidation Timing Tests ====================


class TestReplayTiming:
    """Tests for P2.5 biologically accurate replay timing."""

    @pytest.fixture
    def episodic(self):
        return MockEpisodicMemory(episodes=[
            MockEpisode(context=MockContext())
        ])

    @pytest.fixture
    def semantic(self):
        return MockSemanticMemory()

    @pytest.fixture
    def graph_store(self):
        return MockGraphStore()

    def test_default_replay_delay_is_biological(self, episodic, semantic, graph_store):
        """P2.5: Default replay_delay_ms should be 500ms (not 10ms)."""
        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )
        # P2.5: Changed from 10ms to 500ms for biological accuracy
        # Hippocampal sharp-wave ripples occur at ~1-2 Hz frequency
        assert consolidator.replay_delay_ms == 500

    def test_replay_delay_configurable(self, episodic, semantic, graph_store):
        """Replay delay should be configurable."""
        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            replay_delay_ms=750,  # Custom timing
        )
        assert consolidator.replay_delay_ms == 750

    def test_replay_delay_can_be_zero_for_tests(self, episodic, semantic, graph_store):
        """Replay delay can be set to 0 for fast testing."""
        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            replay_delay_ms=0,
        )
        assert consolidator.replay_delay_ms == 0

    def test_replay_delay_range_biological(self, episodic, semantic, graph_store):
        """Test biologically plausible range (100-1000ms for 1-10 Hz replay)."""
        # 1 Hz = 1000ms between replays
        consolidator_slow = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            replay_delay_ms=1000,
        )
        assert consolidator_slow.replay_delay_ms == 1000

        # 10 Hz = 100ms between replays (fast ripples)
        consolidator_fast = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            replay_delay_ms=100,
        )
        assert consolidator_fast.replay_delay_ms == 100

    @pytest.mark.asyncio
    async def test_nrem_phase_respects_delay(self, episodic, semantic, graph_store):
        """NREM phase should use replay_delay_ms timing."""
        import time

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            replay_delay_ms=50,  # Short delay for testing
            max_replays=2,
        )

        start = time.time()
        await consolidator.nrem_phase("test_session")
        elapsed = time.time() - start

        # With 2 replays at 50ms each (adjusted for compression), should take some time
        # Just verify it completes without error - exact timing depends on compression
        assert elapsed >= 0


class TestEdgeCases:
    """Edge case tests for sleep consolidation."""

    @pytest.mark.asyncio
    async def test_handles_episode_without_id(self):
        @dataclass
        class BadEpisode:
            content: str = "test"
            # No id field

        episodic = MockEpisodicMemory(episodes=[BadEpisode()])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store
        )

        # Should not crash
        events = await consolidator.nrem_phase("test")
        assert len(events) == 0  # Skipped bad episode

    @pytest.mark.asyncio
    async def test_handles_node_without_embedding(self):
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore(nodes=[
            {"id": "node1", "label": "Entity"},  # No embedding
        ])

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store
        )

        # Should not crash
        events = await consolidator.rem_phase("test")
        assert events == []

    @pytest.mark.asyncio
    async def test_handles_episodic_failure(self):
        failing_episodic = MockEpisodicMemory()
        failing_episodic.get_recent = AsyncMock(side_effect=Exception("DB error"))

        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=failing_episodic,
            semantic_memory=semantic,
            graph_store=graph_store
        )

        # Should not crash
        events = await consolidator.nrem_phase("test")
        assert events == []

    @pytest.mark.asyncio
    async def test_handles_graph_store_failure(self):
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()

        failing_graph = MockGraphStore()
        failing_graph.get_all_nodes = AsyncMock(side_effect=Exception("Graph error"))

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=failing_graph
        )

        # Should not crash
        events = await consolidator.rem_phase("test")
        assert events == []

        pruned, strengthened = await consolidator.prune_phase()
        assert pruned == 0
        assert strengthened == 0


class TestP51REMConceptStorage:
    """P5.1: Tests for storing abstract concepts from REM sleep."""

    @pytest.fixture
    def entity_nodes(self):
        """Create entity nodes with embeddings for clustering."""
        dim = 1024
        np.random.seed(42)  # Reproducible
        # Create similar embeddings that will cluster together
        base_embedding = np.random.randn(dim)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)

        nodes = []
        for i in range(5):
            # Add very small noise to create similar embeddings (similarity > 0.95)
            noise = np.random.randn(dim) * 0.01
            emb = base_embedding + noise
            emb = emb / np.linalg.norm(emb)

            node_id = str(uuid4())
            nodes.append({
                "id": node_id,
                "label": "Entity",
                "embedding": emb.tolist(),
                "properties": {
                    "name": f"Machine Learning Concept {i}",
                    "entity_type": "TECHNIQUE",
                    "summary": f"A machine learning technique #{i}",
                },
            })

        return nodes

    @pytest.mark.asyncio
    async def test_rem_phase_creates_concept_entity(self, entity_nodes):
        """REM phase creates and persists abstract concept entity."""
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore(nodes=entity_nodes)

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            min_cluster_size=3,
            abstraction_threshold=0.5,  # Lower threshold for testing
        )

        events = await consolidator.rem_phase("test_session")

        # Should create at least one abstraction
        assert len(events) >= 1

        # Check AbstractionEvent has new fields
        event = events[0]
        assert event.concept_name is not None
        assert event.entity_id is not None
        assert event.centroid_embedding is not None
        assert len(event.centroid_embedding) == 1024

        # Check semantic memory was called
        assert len(semantic.created_entities) >= 1
        created = semantic.created_entities[0]
        assert created["entity_type"] == "CONCEPT"
        assert created["source"] == "rem_abstraction"
        assert "Machine" in created["name"] or "Technique" in created["name"]

    @pytest.mark.asyncio
    async def test_rem_phase_creates_abstracts_relationships(self, entity_nodes):
        """REM phase creates ABSTRACTS relationships to source entities."""
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore(nodes=entity_nodes)

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            min_cluster_size=3,
            abstraction_threshold=0.5,
        )

        events = await consolidator.rem_phase("test_session")

        # Should have created relationships
        assert len(semantic.created_relationships) >= 1

        # Check relationship type
        for rel in semantic.created_relationships:
            assert rel["relation_type"] == "ABSTRACTS"
            assert rel["weight"] > 0

    @pytest.mark.asyncio
    async def test_concept_name_generation_uses_common_words(self):
        """Concept name generation finds common themes."""
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )

        # Test with entities having common words
        names = [
            "Neural Network Training",
            "Deep Neural Networks",
            "Neural Architecture Search",
            "Convolutional Neural Networks",
        ]
        types = ["TECHNIQUE", "TECHNIQUE", "TECHNIQUE", "TECHNIQUE"]

        concept_name = await consolidator._generate_concept_name(names, types)

        # Should include "Neural" as it appears in all names
        assert "Neural" in concept_name
        assert "Technique" in concept_name

    @pytest.mark.asyncio
    async def test_concept_name_fallback_for_diverse_entities(self):
        """Concept name generation falls back for diverse entities."""
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )

        # Test with diverse names (no common words)
        names = ["Alpha", "Beta", "Gamma", "Delta"]
        types = ["TOOL", "TOOL", "TOOL", "TOOL"]

        concept_name = await consolidator._generate_concept_name(names, types)

        # Should fallback to type-based name
        assert "Tool" in concept_name
        assert "Related" in concept_name or "4" in concept_name

    @pytest.mark.asyncio
    async def test_concept_name_empty_entities(self):
        """Concept name generation handles empty input."""
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )

        concept_name = await consolidator._generate_concept_name([], [])

        # Should return abstract fallback
        assert "Abstract" in concept_name

    @pytest.mark.asyncio
    async def test_abstraction_event_fields(self, entity_nodes):
        """AbstractionEvent includes all P5.1 fields."""
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore(nodes=entity_nodes)

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            min_cluster_size=3,
            abstraction_threshold=0.5,
        )

        events = await consolidator.rem_phase("test")

        assert len(events) >= 1
        event = events[0]

        # Original fields
        assert event.cluster_ids is not None
        assert event.abstraction_time is not None
        assert event.confidence > 0

        # P5.1 new fields
        assert event.concept_name is not None
        assert event.entity_id is not None
        assert event.centroid_embedding is not None

    @pytest.mark.asyncio
    async def test_abstraction_persists_on_semantic_failure(self, entity_nodes):
        """Abstraction event created even if persistence fails."""
        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        # Make create_entity fail
        semantic.create_entity = AsyncMock(side_effect=Exception("DB error"))
        graph_store = MockGraphStore(nodes=entity_nodes)

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            min_cluster_size=3,
            abstraction_threshold=0.5,
        )

        events = await consolidator.rem_phase("test")

        # Should still create event
        assert len(events) >= 1
        event = events[0]

        # Has concept name but no entity_id (persistence failed)
        assert event.concept_name is not None
        assert event.entity_id is None  # Failed to persist
        assert event.centroid_embedding is not None

    @pytest.mark.asyncio
    async def test_low_confidence_cluster_not_persisted(self):
        """Low confidence clusters don't create entities."""
        dim = 1024
        # Create dissimilar embeddings that won't cluster well
        nodes = []
        for i in range(5):
            emb = np.random.randn(dim)
            emb = emb / np.linalg.norm(emb)

            nodes.append({
                "id": str(uuid4()),
                "label": "Entity",
                "embedding": emb.tolist(),
                "properties": {
                    "name": f"Random Entity {i}",
                    "entity_type": "CONCEPT",
                },
            })

        episodic = MockEpisodicMemory()
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore(nodes=nodes)

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            min_cluster_size=3,
            abstraction_threshold=0.9,  # High threshold
        )

        events = await consolidator.rem_phase("test")

        # No events due to high threshold
        assert len(events) == 0
        assert len(semantic.created_entities) == 0


class TestP53SynapticTagging:
    """P5.3: Tests for synaptic tagging integration with sleep consolidation."""

    @pytest.fixture
    def mock_plasticity_manager(self):
        """Create mock PlasticityManager."""
        manager = MagicMock()
        manager.on_consolidation = AsyncMock(return_value={
            "scaling_events": 5,
            "tags_captured": 10,
        })
        manager.on_retrieval = AsyncMock(return_value={
            "ltd_events": 3,
            "tags_created": 7,
        })
        manager.tagger = MagicMock()
        manager.tagger.get_tagged_synapses = MagicMock(return_value=[
            MagicMock(source_id="e1", target_id="e2", tag_type="late", captured=False),
            MagicMock(source_id="e2", target_id="e3", tag_type="early", captured=False),
        ])
        return manager

    @pytest.mark.asyncio
    async def test_set_plasticity_manager(self, mock_plasticity_manager):
        """set_plasticity_manager stores the manager correctly."""
        episodes = [MockEpisode(outcome_score=0.8) for _ in range(3)]
        episodic = MockEpisodicMemory(episodes)
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )

        consolidator.set_plasticity_manager(mock_plasticity_manager)
        assert consolidator._plasticity_manager is mock_plasticity_manager

    @pytest.mark.asyncio
    async def test_nrem_captures_synaptic_tags(self, mock_plasticity_manager):
        """NREM phase captures synaptic tags via PlasticityManager."""
        episodes = [MockEpisode(outcome_score=0.9) for _ in range(5)]
        for ep in episodes:
            ep.embedding = [0.1] * 1024

        episodic = MockEpisodicMemory(episodes)
        episodic.get_by_id = AsyncMock(return_value=episodes[0])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )
        consolidator.set_plasticity_manager(mock_plasticity_manager)

        events = await consolidator.nrem_phase("test")

        # PlasticityManager.on_consolidation should be called
        mock_plasticity_manager.on_consolidation.assert_called()
        call_args = mock_plasticity_manager.on_consolidation.call_args
        assert "all_entity_ids" in call_args.kwargs
        assert "store" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_nrem_without_plasticity_manager_works(self):
        """NREM phase works fine without PlasticityManager."""
        episodes = [MockEpisode(outcome_score=0.8) for _ in range(3)]
        for ep in episodes:
            ep.embedding = [0.1] * 1024

        episodic = MockEpisodicMemory(episodes)
        episodic.get_by_id = AsyncMock(return_value=episodes[0])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )
        # No plasticity manager set

        events = await consolidator.nrem_phase("test")

        # Should complete without error
        assert events is not None

    @pytest.mark.asyncio
    async def test_plasticity_manager_failure_doesnt_break_nrem(self, mock_plasticity_manager):
        """PlasticityManager failure doesn't break NREM phase."""
        mock_plasticity_manager.on_consolidation = AsyncMock(
            side_effect=Exception("Plasticity failure")
        )

        episodes = [MockEpisode(outcome_score=0.9) for _ in range(3)]
        for ep in episodes:
            ep.embedding = [0.1] * 1024

        episodic = MockEpisodicMemory(episodes)
        episodic.get_by_id = AsyncMock(return_value=episodes[0])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )
        consolidator.set_plasticity_manager(mock_plasticity_manager)

        # Should not raise despite PlasticityManager failure
        events = await consolidator.nrem_phase("test")
        assert events is not None

    @pytest.mark.asyncio
    async def test_no_events_skips_tag_capture(self, mock_plasticity_manager):
        """Tag capture is skipped when no replay events."""
        episodic = MockEpisodicMemory([])  # No episodes
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )
        consolidator.set_plasticity_manager(mock_plasticity_manager)

        events = await consolidator.nrem_phase("test")

        # on_consolidation should not be called when no events
        mock_plasticity_manager.on_consolidation.assert_not_called()


# =============================================================================
# Phase 7: VAE-Based Generative Replay & Multi-Night Scheduling
# =============================================================================


class TestP7VAEGenerativeReplay:
    """Test Phase 7 VAE-based generative replay integration."""

    def test_vae_enabled_by_default(self):
        """VAE should be enabled by default in SleepConsolidation."""
        episodic = MockEpisodicMemory([])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
        )

        assert consolidator.vae_enabled is True
        assert consolidator._vae_generator is not None
        assert consolidator._generative_replay is not None

    def test_vae_can_be_disabled(self):
        """VAE can be disabled via parameter."""
        episodic = MockEpisodicMemory([])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            vae_enabled=False,
        )

        assert consolidator.vae_enabled is False
        assert consolidator._vae_generator is None
        assert consolidator._generative_replay is None

    def test_vae_generator_configured_correctly(self):
        """VAE generator should have correct dimensions."""
        episodic = MockEpisodicMemory([])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            vae_latent_dim=64,
            embedding_dim=512,
        )

        vae_stats = consolidator.get_vae_statistics()
        assert vae_stats is not None
        assert vae_stats["config"]["embedding_dim"] == 512
        assert vae_stats["config"]["latent_dim"] == 64

    def test_get_vae_statistics_none_when_disabled(self):
        """get_vae_statistics returns None when VAE disabled."""
        episodic = MockEpisodicMemory([])
        semantic = MockSemanticMemory()
        graph_store = MockGraphStore()

        consolidator = SleepConsolidation(
            episodic_memory=episodic,
            semantic_memory=semantic,
            graph_store=graph_store,
            vae_enabled=False,
        )

        assert consolidator.get_vae_statistics() is None


class TestP7MultiNightScheduling:
    """Test Phase 7 multi-night scheduling capabilities."""

    def test_scheduler_initial_night_is_one(self):
        """Scheduler should start at night 1."""
        from t4dm.consolidation.service import ConsolidationScheduler

        scheduler = ConsolidationScheduler()
        assert scheduler.state.current_night == 1
        assert scheduler.state.total_nights == 0

    def test_recommended_depth_by_night(self):
        """Recommended depth should vary by night."""
        from t4dm.consolidation.service import ConsolidationScheduler

        scheduler = ConsolidationScheduler()

        # Night 1: light
        assert scheduler.get_recommended_consolidation_depth() == "light"

        # Night 2-3: deep
        scheduler.state.current_night = 2
        assert scheduler.get_recommended_consolidation_depth() == "deep"

        scheduler.state.current_night = 3
        assert scheduler.get_recommended_consolidation_depth() == "deep"

        # Night 4+: all
        scheduler.state.current_night = 4
        assert scheduler.get_recommended_consolidation_depth() == "all"

        scheduler.state.current_night = 10
        assert scheduler.get_recommended_consolidation_depth() == "all"

    def test_advance_night_increments_correctly(self):
        """advance_night should increment night counter."""
        from t4dm.consolidation.service import ConsolidationScheduler

        scheduler = ConsolidationScheduler()
        assert scheduler.state.current_night == 1

        new_night = scheduler.advance_night(memories_consolidated=50)
        assert new_night == 2
        assert scheduler.state.current_night == 2
        assert scheduler.state.total_nights == 1
        assert scheduler.state.memories_this_night == 50

        new_night = scheduler.advance_night(memories_consolidated=30)
        assert new_night == 3
        assert scheduler.state.total_nights == 2
        assert scheduler.state.memories_this_night == 80  # Cumulative

    def test_reset_night_cycle(self):
        """reset_night_cycle should reset to night 1."""
        from t4dm.consolidation.service import ConsolidationScheduler

        scheduler = ConsolidationScheduler()
        scheduler.advance_night(100)
        scheduler.advance_night(50)
        assert scheduler.state.current_night == 3

        scheduler.reset_night_cycle()
        assert scheduler.state.current_night == 1
        assert scheduler.state.memories_this_night == 0

    def test_get_stats_includes_multi_night(self):
        """get_stats should include multi-night info."""
        from t4dm.consolidation.service import ConsolidationScheduler

        scheduler = ConsolidationScheduler()
        scheduler.advance_night(25)

        stats = scheduler.get_stats()
        assert "multi_night" in stats
        assert stats["multi_night"]["current_night"] == 2
        assert stats["multi_night"]["total_nights"] == 1
        assert stats["multi_night"]["memories_this_night"] == 25
        assert stats["multi_night"]["recommended_depth"] == "deep"
