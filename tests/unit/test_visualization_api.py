"""
Unit tests for T4DM visualization API routes.

Tests graph, embeddings, timeline, activity, and export endpoints.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from t4dm.api.routes.visualization import (
    router,
    MemoryType,
    EdgeType,
    Position3D,
    NodeMetadata,
    MemoryNodeResponse,
    MemoryEdgeResponse,
    GraphResponse,
    EmbeddingsResponse,
    TimelineResponse,
    ActivityResponse,
    ExportRequest,
    ExportResponse,
    _compute_activation,
    _assign_3d_positions,
    _generate_gexf,
    _generate_graphml,
    # Biological mechanism types
    FSRSState,
    HebbianWeight,
    ActivationSpread,
    SleepPhaseViz,
    SleepConsolidationState,
    WorkingMemoryState,
    PatternSeparationMetrics,
    PatternCompletionMetrics,
    BiologicalMechanismsResponse,
)


class TestMemoryType:
    """Tests for MemoryType enum."""

    def test_values(self):
        assert MemoryType.EPISODIC.value == "episodic"
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_values(self):
        assert EdgeType.CAUSED.value == "CAUSED"
        assert EdgeType.SIMILAR_TO.value == "SIMILAR_TO"
        assert EdgeType.PREREQUISITE.value == "PREREQUISITE"
        assert EdgeType.CONTRADICTS.value == "CONTRADICTS"
        assert EdgeType.REFERENCES.value == "REFERENCES"
        assert EdgeType.DERIVED_FROM.value == "DERIVED_FROM"


class TestPosition3D:
    """Tests for Position3D model."""

    def test_creation(self):
        pos = Position3D(x=1.0, y=2.0, z=3.0)
        assert pos.x == 1.0
        assert pos.y == 2.0
        assert pos.z == 3.0


class TestNodeMetadata:
    """Tests for NodeMetadata model."""

    def test_creation_minimal(self):
        meta = NodeMetadata(
            created_at=1700000000.0,
            last_accessed=1700000000.0,
            access_count=5,
            importance=0.8,
        )
        assert meta.access_count == 5
        assert meta.importance == 0.8
        assert meta.tags is None
        assert meta.source is None

    def test_creation_full(self):
        meta = NodeMetadata(
            created_at=1700000000.0,
            last_accessed=1700000000.0,
            access_count=5,
            importance=0.8,
            tags=["tag1", "tag2"],
            source="project-a",
        )
        assert meta.tags == ["tag1", "tag2"]
        assert meta.source == "project-a"


class TestMemoryNodeResponse:
    """Tests for MemoryNodeResponse model."""

    def test_creation(self):
        node = MemoryNodeResponse(
            id="node-1",
            type=MemoryType.EPISODIC,
            content="Test content",
            metadata=NodeMetadata(
                created_at=1700000000.0,
                last_accessed=1700000000.0,
                access_count=5,
                importance=0.8,
            ),
        )
        assert node.id == "node-1"
        assert node.type == MemoryType.EPISODIC
        assert node.position is None

    def test_creation_with_position(self):
        node = MemoryNodeResponse(
            id="node-1",
            type=MemoryType.SEMANTIC,
            content="Test",
            metadata=NodeMetadata(
                created_at=1700000000.0,
                last_accessed=1700000000.0,
                access_count=0,
                importance=0.5,
            ),
            position=Position3D(x=1.0, y=2.0, z=3.0),
        )
        assert node.position.x == 1.0


class TestMemoryEdgeResponse:
    """Tests for MemoryEdgeResponse model."""

    def test_creation(self):
        edge = MemoryEdgeResponse(
            id="edge-1",
            source="node-a",
            target="node-b",
            type=EdgeType.CAUSED,
            weight=0.8,
        )
        assert edge.source == "node-a"
        assert edge.target == "node-b"
        assert edge.weight == 0.8
        assert edge.metadata is None


class TestComputeActivation:
    """Tests for _compute_activation helper."""

    def test_recent_high_access(self):
        activation, recency, frequency = _compute_activation(
            access_count=100,
            last_accessed=datetime.now(),
            created_at=datetime.now() - timedelta(days=1),
            importance=0.8,
        )

        assert 0.0 <= activation <= 1.0
        assert recency > 0.9  # Very recent
        assert frequency == 1.0  # Max normalized

    def test_old_low_access(self):
        activation, recency, frequency = _compute_activation(
            access_count=1,
            last_accessed=datetime.now() - timedelta(days=30),
            created_at=datetime.now() - timedelta(days=60),
            importance=0.2,
        )

        assert activation < 0.5  # Low overall
        assert recency < 0.5  # Decayed
        assert frequency < 0.1  # Low access

    def test_importance_weight(self):
        # Same recency/frequency, different importance
        act_high, _, _ = _compute_activation(
            access_count=10,
            last_accessed=datetime.now() - timedelta(days=5),
            created_at=datetime.now() - timedelta(days=10),
            importance=1.0,
        )

        act_low, _, _ = _compute_activation(
            access_count=10,
            last_accessed=datetime.now() - timedelta(days=5),
            created_at=datetime.now() - timedelta(days=10),
            importance=0.0,
        )

        assert act_high > act_low


class TestAssign3DPositions:
    """Tests for _assign_3d_positions helper."""

    def test_empty_nodes(self):
        positions = _assign_3d_positions([], [], "force-directed")
        assert len(positions) == 0

    def test_force_directed(self):
        nodes = [
            {"id": "a", "type": "episodic"},
            {"id": "b", "type": "semantic"},
            {"id": "c", "type": "procedural"},
        ]

        positions = _assign_3d_positions(nodes, [], "force-directed")

        assert len(positions) == 3
        assert "a" in positions
        assert isinstance(positions["a"], Position3D)

    def test_hierarchical(self):
        nodes = [
            {"id": "a", "type": "episodic"},
            {"id": "b", "type": "semantic"},
            {"id": "c", "type": "procedural"},
        ]

        positions = _assign_3d_positions(nodes, [], "hierarchical")

        assert len(positions) == 3
        # Different types should have different y positions
        assert positions["a"].y != positions["b"].y

    def test_circular(self):
        nodes = [
            {"id": "a", "type": "episodic"},
            {"id": "b", "type": "episodic"},
            {"id": "c", "type": "episodic"},
        ]

        positions = _assign_3d_positions(nodes, [], "circular")

        assert len(positions) == 3
        # All should have z=0 for circular layout
        for pos in positions.values():
            assert pos.z == 0.0


class TestGenerateGexf:
    """Tests for GEXF export generation."""

    def test_basic_export(self):
        graph = GraphResponse(
            nodes=[
                MemoryNodeResponse(
                    id="node-1",
                    type=MemoryType.EPISODIC,
                    content="Test content",
                    metadata=NodeMetadata(
                        created_at=1700000000.0,
                        last_accessed=1700000000.0,
                        access_count=5,
                        importance=0.8,
                    ),
                    position=Position3D(x=1.0, y=2.0, z=3.0),
                )
            ],
            edges=[],
            metrics={"total_nodes": 1},
        )

        gexf = _generate_gexf(graph)

        assert '<?xml version="1.0"' in gexf
        assert '<gexf' in gexf
        assert 'node id="node-1"' in gexf
        assert '</gexf>' in gexf

    def test_with_edges(self):
        graph = GraphResponse(
            nodes=[
                MemoryNodeResponse(
                    id="a",
                    type=MemoryType.EPISODIC,
                    content="A",
                    metadata=NodeMetadata(
                        created_at=1700000000.0,
                        last_accessed=1700000000.0,
                        access_count=1,
                        importance=0.5,
                    ),
                ),
                MemoryNodeResponse(
                    id="b",
                    type=MemoryType.SEMANTIC,
                    content="B",
                    metadata=NodeMetadata(
                        created_at=1700000000.0,
                        last_accessed=1700000000.0,
                        access_count=1,
                        importance=0.5,
                    ),
                ),
            ],
            edges=[
                MemoryEdgeResponse(
                    id="e1",
                    source="a",
                    target="b",
                    type=EdgeType.REFERENCES,
                    weight=0.5,
                )
            ],
            metrics={"total_nodes": 2, "total_edges": 1},
        )

        gexf = _generate_gexf(graph)

        assert '<edges>' in gexf
        assert 'source="a"' in gexf
        assert 'target="b"' in gexf


class TestGenerateGraphml:
    """Tests for GraphML export generation."""

    def test_basic_export(self):
        graph = GraphResponse(
            nodes=[
                MemoryNodeResponse(
                    id="node-1",
                    type=MemoryType.PROCEDURAL,
                    content="Test skill",
                    metadata=NodeMetadata(
                        created_at=1700000000.0,
                        last_accessed=1700000000.0,
                        access_count=10,
                        importance=0.9,
                    ),
                )
            ],
            edges=[],
            metrics={"total_nodes": 1},
        )

        graphml = _generate_graphml(graph)

        assert '<?xml version="1.0"' in graphml
        assert '<graphml' in graphml
        assert 'node id="node-1"' in graphml
        assert 'procedural' in graphml
        assert '</graphml>' in graphml


class MockEpisode:
    """Mock episode for testing."""

    def __init__(
        self,
        id=None,
        content="Test content",
        timestamp=None,
        emotional_valence=0.5,
        access_count=0,
        context=None,
    ):
        self.id = id or uuid4()
        self.content = content
        self.timestamp = timestamp or datetime.now()
        self.last_accessed = self.timestamp
        self.emotional_valence = emotional_valence
        self.access_count = access_count
        self.context = context or MagicMock(project=None, file=None, tool=None)
        self.embedding = np.random.randn(384)


class TestGraphResponse:
    """Tests for GraphResponse model."""

    def test_creation(self):
        response = GraphResponse(
            nodes=[],
            edges=[],
            metrics={"total_nodes": 0, "total_edges": 0},
        )
        assert len(response.nodes) == 0
        assert len(response.edges) == 0


class TestTimelineResponse:
    """Tests for TimelineResponse model."""

    def test_creation(self):
        response = TimelineResponse(
            events=[],
            start_time=1700000000.0,
            end_time=1700000000.0,
            total_events=0,
        )
        assert response.total_events == 0


class TestActivityResponse:
    """Tests for ActivityResponse model."""

    def test_creation(self):
        response = ActivityResponse(
            memories=[],
            most_active=[],
            recently_created=[],
            recently_accessed=[],
        )
        assert len(response.most_active) == 0


class TestExportRequest:
    """Tests for ExportRequest model."""

    def test_defaults(self):
        req = ExportRequest()
        assert req.format == "json"
        assert req.include_embeddings is False
        assert req.include_positions is True

    def test_custom(self):
        req = ExportRequest(format="gexf", include_embeddings=True)
        assert req.format == "gexf"
        assert req.include_embeddings is True


class TestEmbeddingsResponse:
    """Tests for EmbeddingsResponse model."""

    def test_creation(self):
        response = EmbeddingsResponse(
            points=[],
            embedding_dim=384,
            projection_method="t-SNE",
        )
        assert response.embedding_dim == 384
        assert response.projection_method == "t-SNE"


class TestVisualizationEndpoints:
    """Integration tests for visualization endpoints."""

    @pytest.fixture
    def mock_services(self):
        """Create mock memory services."""
        # Use MagicMock as base so sync method assignments work properly
        episodic = MagicMock()
        semantic = MagicMock()
        procedural = MagicMock()

        # Setup episodic async methods
        episodic.recent = AsyncMock(return_value=[
            MockEpisode(content="Episode 1", emotional_valence=0.8),
            MockEpisode(content="Episode 2", emotional_valence=0.5),
            MockEpisode(content="Episode 3", emotional_valence=0.3),
        ])

        # Setup semantic methods
        semantic.list_entities = AsyncMock(return_value=[
            {"id": "entity-1", "name": "Entity One", "importance": 0.7},
        ])
        semantic.get_relationships = AsyncMock(return_value=[
            {"source": "entity-1", "target": "entity-2", "type": "REFERENCES", "weight": 0.5},
        ])

        # Setup procedural methods
        procedural.list_skills = AsyncMock(return_value=[
            {"id": "skill-1", "name": "Skill One", "success_rate": 0.9},
        ])

        # Setup biological mechanism stats methods (sync methods - use MagicMock)
        episodic.get_dopamine_stats = MagicMock(return_value={
            "total_signals": 100,
            "positive_surprises": 60,
            "negative_surprises": 40,
            "avg_rpe": 0.1,
            "avg_surprise": 0.3,
            "memories_tracked": 50,
        })

        episodic.get_reconsolidation_stats = MagicMock(return_value={
            "total_updates": 200,
            "positive_updates": 120,
            "negative_updates": 80,
            "avg_update_magnitude": 0.05,
            "memories_in_cooldown": 10,
            "avg_learning_rate": 0.01,
        })

        episodic.get_fusion_training_stats = MagicMock(return_value={
            "enabled": True,
            "train_steps": 1000,
            "avg_loss": 0.5,
            "current_weights": None,
        })

        # Setup neuromodulator orchestra methods (sync methods - use MagicMock)
        episodic.get_orchestra_stats = MagicMock(return_value={
            "norepinephrine": {
                "current_gain": 1.2,
                "novelty_score": 0.3,
                "tonic_level": 0.5,
                "phasic_response": 0.1,
                "exploration_bonus": 0.05,
                "history_length": 50,
            },
            "acetylcholine": {
                "current_mode": "balanced",
                "encoding_level": 0.5,
                "retrieval_level": 0.5,
                "attention_weights": None,
                "mode_switches": 10,
                "time_in_mode": 30.0,
            },
            "serotonin": {
                "current_mood": 0.6,
                "total_outcomes": 100,
                "positive_outcome_rate": 0.7,
                "memories_with_traces": 25,
                "active_traces": 10,
                "active_contexts": 2,
            },
            "inhibitory": {
                "recent_sparsity": 0.8,
                "avg_sparsity": 0.75,
                "inhibition_events": 50,
                "k": 10,
                "lateral_strength": 0.5,
            },
        })

        episodic.get_current_neuromodulator_state = MagicMock(return_value={
            "dopamine_rpe": 0.1,
            "norepinephrine_gain": 1.2,
            "acetylcholine_mode": "balanced",
            "serotonin_mood": 0.6,
            "inhibition_sparsity": 0.8,
            "effective_learning_rate": 1.0,
            "exploration_balance": 0.0,
        })

        episodic.get_learned_gate_stats = MagicMock(return_value={
            "enabled": True,
            "n_observations": 50,
            "cold_start_progress": 0.5,
            "store_rate": 0.6,
            "buffer_rate": 0.2,
            "skip_rate": 0.2,
            "avg_accuracy": 0.8,
            "calibration_ece": 0.05,
        })

        return {
            "session_id": "test-session",
            "episodic": episodic,
            "semantic": semantic,
            "procedural": procedural,
        }

    @pytest.mark.asyncio
    async def test_get_memory_graph(self, mock_services):
        """Test graph endpoint returns correct structure."""
        from t4dm.api.routes.visualization import get_memory_graph

        response = await get_memory_graph(
            services=mock_services,
            layout="force-directed",
            limit=100,
            include_edges=True,
        )

        assert isinstance(response, GraphResponse)
        assert len(response.nodes) >= 3  # At least 3 episodes
        assert "total_nodes" in response.metrics

    @pytest.mark.asyncio
    async def test_get_memory_graph_hierarchical(self, mock_services):
        """Test hierarchical layout."""
        from t4dm.api.routes.visualization import get_memory_graph

        response = await get_memory_graph(
            services=mock_services,
            layout="hierarchical",
            limit=100,
            include_edges=False,
        )

        assert isinstance(response, GraphResponse)
        # Should have positions assigned
        for node in response.nodes:
            assert node.position is not None

    @pytest.mark.asyncio
    async def test_get_timeline(self, mock_services):
        """Test timeline endpoint."""
        from t4dm.api.routes.visualization import get_timeline

        response = await get_timeline(
            services=mock_services,
            days=30,
            limit=100,
        )

        assert isinstance(response, TimelineResponse)
        assert response.total_events >= 0

    @pytest.mark.asyncio
    async def test_get_activity(self, mock_services):
        """Test activity endpoint."""
        from t4dm.api.routes.visualization import get_activity

        response = await get_activity(
            services=mock_services,
            limit=100,
            top_n=10,
        )

        assert isinstance(response, ActivityResponse)
        assert len(response.most_active) <= 10

    @pytest.mark.asyncio
    async def test_get_embeddings(self, mock_services):
        """Test embeddings endpoint."""
        from t4dm.api.routes.visualization import get_embeddings

        response = await get_embeddings(
            services=mock_services,
            limit=10,
            include_raw=False,
            project_to="none",
        )

        assert isinstance(response, EmbeddingsResponse)
        # Should have found episodes with embeddings
        assert len(response.points) > 0

    @pytest.mark.asyncio
    async def test_export_json(self, mock_services):
        """Test JSON export."""
        from t4dm.api.routes.visualization import export_graph

        request = ExportRequest(format="json")
        response = await export_graph(
            request=request,
            services=mock_services,
        )

        assert isinstance(response, ExportResponse)
        assert response.format == "json"
        assert response.filename.endswith(".json")
        assert "nodes" in response.content

    @pytest.mark.asyncio
    async def test_export_gexf(self, mock_services):
        """Test GEXF export."""
        from t4dm.api.routes.visualization import export_graph

        request = ExportRequest(format="gexf")
        response = await export_graph(
            request=request,
            services=mock_services,
        )

        assert isinstance(response, ExportResponse)
        assert response.format == "gexf"
        assert response.filename.endswith(".gexf")
        assert "<gexf" in response.content

    @pytest.mark.asyncio
    async def test_export_graphml(self, mock_services):
        """Test GraphML export."""
        from t4dm.api.routes.visualization import export_graph

        request = ExportRequest(format="graphml")
        response = await export_graph(
            request=request,
            services=mock_services,
        )

        assert isinstance(response, ExportResponse)
        assert response.format == "graphml"
        assert response.filename.endswith(".graphml")
        assert "<graphml" in response.content


# ============================================================================
# Biological Mechanism Type Tests
# ============================================================================

class TestFSRSState:
    """Tests for FSRSState model."""

    def test_creation(self):
        state = FSRSState(
            memory_id="mem-1",
            stability=5.0,
            difficulty=0.3,
            retrievability=0.85,
            last_review=1700000000.0,
            next_review=1700432000.0,
            review_count=5,
            decay_curve=[(0.0, 1.0), (5.0, 0.63), (10.0, 0.37)],
        )
        assert state.stability == 5.0
        assert state.difficulty == 0.3
        assert len(state.decay_curve) == 3


class TestHebbianWeight:
    """Tests for HebbianWeight model."""

    def test_creation(self):
        weight = HebbianWeight(
            source_id="a",
            target_id="b",
            weight=0.8,
            weight_history=[(1700000000.0, 0.5), (1700001000.0, 0.8)],
            co_activation_count=10,
        )
        assert weight.weight == 0.8
        assert weight.co_activation_count == 10
        assert len(weight.weight_history) == 2

    def test_with_plasticity(self):
        weight = HebbianWeight(
            source_id="a",
            target_id="b",
            weight=0.7,
            weight_history=[],
            co_activation_count=5,
            last_potentiation=1700000000.0,
            last_depression=1699999000.0,
            eligibility_trace=0.3,
        )
        assert weight.last_potentiation > weight.last_depression
        assert weight.eligibility_trace == 0.3


class TestActivationSpread:
    """Tests for ActivationSpread model."""

    def test_creation(self):
        activation = ActivationSpread(
            source_id="src",
            target_id="tgt",
            base_level=1.5,
            spreading_activation=0.5,
            total_activation=2.0,
            decay_rate=0.5,
            time_since_activation=30.0,
        )
        assert activation.total_activation == 2.0
        assert activation.base_level + activation.spreading_activation == 2.0


class TestSleepPhaseViz:
    """Tests for SleepPhaseViz enum."""

    def test_values(self):
        assert SleepPhaseViz.NREM.value == "nrem"
        assert SleepPhaseViz.REM.value == "rem"
        assert SleepPhaseViz.PRUNE.value == "prune"
        assert SleepPhaseViz.COMPLETE.value == "complete"


class TestSleepConsolidationState:
    """Tests for SleepConsolidationState model."""

    def test_inactive(self):
        state = SleepConsolidationState(
            is_active=False,
            current_phase=None,
            phase_progress=0.0,
        )
        assert not state.is_active
        assert state.current_phase is None

    def test_active_nrem(self):
        state = SleepConsolidationState(
            is_active=True,
            current_phase=SleepPhaseViz.NREM,
            phase_progress=0.5,
            replays_completed=25,
        )
        assert state.is_active
        assert state.current_phase == SleepPhaseViz.NREM
        assert state.replays_completed == 25


class TestWorkingMemoryState:
    """Tests for WorkingMemoryState model."""

    def test_empty(self):
        state = WorkingMemoryState(
            capacity=4,
            current_size=0,
            items=[],
            attention_weights=[],
            decay_rate=0.1,
            eviction_history=[],
            is_full=False,
        )
        assert state.capacity == 4
        assert state.current_size == 0
        assert not state.is_full

    def test_full(self):
        state = WorkingMemoryState(
            capacity=4,
            current_size=4,
            items=[{"id": f"item-{i}", "priority": 0.5 + i*0.1} for i in range(4)],
            attention_weights=[0.5, 0.6, 0.7, 0.8],
            decay_rate=0.1,
            eviction_history=[],
            is_full=True,
        )
        assert state.is_full
        assert len(state.items) == 4
        assert len(state.attention_weights) == 4


class TestPatternSeparationMetrics:
    """Tests for PatternSeparationMetrics model."""

    def test_creation(self):
        metrics = PatternSeparationMetrics(
            input_similarity=0.9,
            output_similarity=0.3,
            separation_ratio=3.0,
            sparsity=0.1,
            orthogonalization_strength=0.5,
        )
        assert metrics.separation_ratio == 3.0
        assert metrics.input_similarity > metrics.output_similarity


class TestPatternCompletionMetrics:
    """Tests for PatternCompletionMetrics model."""

    def test_creation(self):
        metrics = PatternCompletionMetrics(
            input_completeness=0.5,
            output_confidence=0.9,
            convergence_iterations=7,
            best_match_id="mem-123",
            similarity_to_match=0.95,
        )
        assert metrics.convergence_iterations == 7
        assert metrics.best_match_id == "mem-123"


class TestBiologicalMechanismsResponse:
    """Tests for BiologicalMechanismsResponse model."""

    def test_creation_empty(self):
        response = BiologicalMechanismsResponse()
        assert len(response.fsrs_states) == 0
        assert len(response.hebbian_weights) == 0
        assert response.sleep_consolidation is None
        assert response.working_memory is None


class TestBiologicalMechanismEndpoints:
    """Tests for biological mechanism visualization endpoints."""

    @pytest.fixture
    def mock_services(self):
        """Create mock memory services."""
        # Use MagicMock as base so sync method assignments work properly
        episodic = MagicMock()
        semantic = MagicMock()
        procedural = MagicMock()

        # Setup episodic async methods
        episodic.recent = AsyncMock(return_value=[
            MockEpisode(content="Episode 1", emotional_valence=0.8),
            MockEpisode(content="Episode 2", emotional_valence=0.5),
        ])

        # Setup semantic methods
        semantic.get_relationships = AsyncMock(return_value=[
            {
                "source": "entity-1",
                "target": "entity-2",
                "type": "REFERENCES",
                "weight": 0.7,
                "created_at": datetime.now(),
            },
        ])

        # Setup biological mechanism stats methods (sync - use MagicMock)
        episodic.get_dopamine_stats = MagicMock(return_value={
            "total_signals": 100,
            "positive_surprises": 60,
            "negative_surprises": 40,
            "avg_rpe": 0.1,
            "avg_surprise": 0.3,
            "memories_tracked": 50,
        })

        episodic.get_reconsolidation_stats = MagicMock(return_value={
            "total_updates": 200,
            "positive_updates": 120,
            "negative_updates": 80,
            "avg_update_magnitude": 0.05,
            "memories_in_cooldown": 10,
            "avg_learning_rate": 0.01,
        })

        episodic.get_fusion_training_stats = MagicMock(return_value={
            "enabled": True,
            "train_steps": 1000,
            "avg_loss": 0.5,
            "current_weights": None,
        })

        # Setup neuromodulator orchestra methods (sync - use MagicMock)
        episodic.get_orchestra_stats = MagicMock(return_value={
            "norepinephrine": {
                "current_gain": 1.2,
                "novelty_score": 0.3,
                "tonic_level": 0.5,
                "phasic_response": 0.1,
                "exploration_bonus": 0.05,
                "history_length": 50,
            },
            "acetylcholine": {
                "current_mode": "balanced",
                "encoding_level": 0.5,
                "retrieval_level": 0.5,
                "attention_weights": None,
                "mode_switches": 10,
                "time_in_mode": 30.0,
            },
            "serotonin": {
                "current_mood": 0.6,
                "total_outcomes": 100,
                "positive_outcome_rate": 0.7,
                "memories_with_traces": 25,
                "active_traces": 10,
                "active_contexts": 2,
            },
            "inhibitory": {
                "recent_sparsity": 0.8,
                "avg_sparsity": 0.75,
                "inhibition_events": 50,
                "k": 10,
                "lateral_strength": 0.5,
            },
        })

        episodic.get_current_neuromodulator_state = MagicMock(return_value={
            "dopamine_rpe": 0.1,
            "norepinephrine_gain": 1.2,
            "acetylcholine_mode": "balanced",
            "serotonin_mood": 0.6,
            "inhibition_sparsity": 0.8,
            "effective_learning_rate": 1.0,
            "exploration_balance": 0.0,
        })

        episodic.get_learned_gate_stats = MagicMock(return_value={
            "enabled": True,
            "n_observations": 50,
            "cold_start_progress": 0.5,
            "store_rate": 0.6,
            "buffer_rate": 0.2,
            "skip_rate": 0.2,
            "avg_accuracy": 0.8,
            "calibration_ece": 0.05,
        })

        return {
            "session_id": "test-session",
            "episodic": episodic,
            "semantic": semantic,
            "procedural": procedural,
        }

    @pytest.mark.asyncio
    async def test_get_fsrs_states(self, mock_services):
        """Test FSRS states endpoint."""
        from t4dm.api.routes.visualization import get_fsrs_states

        states = await get_fsrs_states(
            services=mock_services,
            limit=10,
            include_decay_curve=True,
            forecast_days=30,
        )

        assert isinstance(states, list)
        assert len(states) == 2
        for state in states:
            assert isinstance(state, FSRSState)
            assert len(state.decay_curve) > 0

    @pytest.mark.asyncio
    async def test_get_fsrs_states_no_curve(self, mock_services):
        """Test FSRS states without decay curve."""
        from t4dm.api.routes.visualization import get_fsrs_states

        states = await get_fsrs_states(
            services=mock_services,
            limit=10,
            include_decay_curve=False,
            forecast_days=30,
        )

        assert len(states) == 2
        for state in states:
            assert state.decay_curve == []

    @pytest.mark.asyncio
    async def test_get_hebbian_weights(self, mock_services):
        """Test Hebbian weights endpoint."""
        from t4dm.api.routes.visualization import get_hebbian_weights

        weights = await get_hebbian_weights(
            services=mock_services,
            limit=100,
            min_weight=0.1,
        )

        assert isinstance(weights, list)
        assert len(weights) == 1
        assert weights[0].weight == 0.7

    @pytest.mark.asyncio
    async def test_get_activation_spreading(self, mock_services):
        """Test activation spreading endpoint."""
        from t4dm.api.routes.visualization import get_activation_spreading

        activations = await get_activation_spreading(
            services=mock_services,
            source_id=None,
            limit=50,
        )

        assert isinstance(activations, list)
        assert len(activations) == 1
        assert activations[0].source_id == "entity-1"

    @pytest.mark.asyncio
    async def test_get_sleep_consolidation_state(self, mock_services):
        """Test sleep consolidation state endpoint."""
        from t4dm.api.routes.visualization import get_sleep_consolidation_state

        state = await get_sleep_consolidation_state(services=mock_services)

        assert isinstance(state, SleepConsolidationState)
        assert not state.is_active  # Default state

    @pytest.mark.asyncio
    async def test_get_working_memory_state(self, mock_services):
        """Test working memory state endpoint."""
        from t4dm.api.routes.visualization import get_working_memory_state

        state = await get_working_memory_state(services=mock_services)

        assert isinstance(state, WorkingMemoryState)
        assert state.capacity == 4  # Default

    @pytest.mark.asyncio
    async def test_get_pattern_separation_metrics(self, mock_services):
        """Test pattern separation metrics endpoint."""
        from t4dm.api.routes.visualization import get_pattern_separation_metrics

        # Without inputs
        metrics = await get_pattern_separation_metrics(
            services=mock_services,
            input_a=None,
            input_b=None,
        )

        assert isinstance(metrics, PatternSeparationMetrics)
        assert metrics.input_similarity == 0.0

        # With inputs
        metrics = await get_pattern_separation_metrics(
            services=mock_services,
            input_a="test pattern A",
            input_b="test pattern B",
        )

        assert metrics.input_similarity > 0

    @pytest.mark.asyncio
    async def test_get_pattern_completion_metrics(self, mock_services):
        """Test pattern completion metrics endpoint."""
        from t4dm.api.routes.visualization import get_pattern_completion_metrics

        # Without input
        metrics = await get_pattern_completion_metrics(
            services=mock_services,
            partial_input=None,
        )

        assert isinstance(metrics, PatternCompletionMetrics)
        assert metrics.input_completeness == 0.0

        # With input
        metrics = await get_pattern_completion_metrics(
            services=mock_services,
            partial_input="partial pattern...",
        )

        assert metrics.input_completeness > 0

    @pytest.mark.asyncio
    async def test_get_all_biological_mechanisms(self, mock_services):
        """Test combined biological mechanisms endpoint."""
        from t4dm.api.routes.visualization import get_all_biological_mechanisms

        response = await get_all_biological_mechanisms(
            services=mock_services,
            fsrs_limit=20,
            hebbian_limit=50,
        )

        assert isinstance(response, BiologicalMechanismsResponse)
        assert len(response.fsrs_states) == 2
        assert len(response.hebbian_weights) == 1
        assert response.sleep_consolidation is not None
        assert response.working_memory is not None
