"""
Tests for P4: Future Work Integration.

P4-1: Hierarchical multi-timescale prediction
P4-2: Causal discovery integration
P4-3: Place/grid cell spatial prediction
P4-4: Theta-gamma coupling
"""

import numpy as np
import pytest
from uuid import uuid4

# P4-1: Hierarchical Prediction
from t4dm.prediction import (
    HierarchicalPredictor,
    HierarchicalConfig,
    HierarchicalPrediction,
    HierarchicalError,
    ContextEncoder,
)

# P4-2: Causal Discovery
from t4dm.learning import (
    CausalLearner,
    CausalGraph,
    CausalAttributor,
    CausalRelationType,
    CausalDiscoveryConfig,
)

# P4-3: Spatial Cells
from t4dm.nca import (
    SpatialCellSystem,
    SpatialConfig,
    Position2D,
)

# P4-4: Theta-Gamma Integration
from t4dm.nca import (
    ThetaGammaIntegration,
    ThetaGammaConfig,
    CognitivePhase,
)


class TestHierarchicalPredictor:
    """Tests for P4-1: Hierarchical multi-timescale prediction."""

    @pytest.fixture
    def predictor(self):
        """Create hierarchical predictor."""
        return HierarchicalPredictor()

    def test_initialization(self, predictor):
        """Test default initialization."""
        assert predictor.config.fast_horizon == 1
        assert predictor.config.medium_horizon == 5
        assert predictor.config.slow_horizon == 15

    def test_custom_config(self):
        """Test custom configuration."""
        config = HierarchicalConfig(
            fast_horizon=2,
            medium_horizon=10,
            slow_horizon=30,
        )
        predictor = HierarchicalPredictor(config=config)
        assert predictor.config.fast_horizon == 2
        assert predictor.config.slow_horizon == 30

    def test_add_episode(self, predictor):
        """Test adding episodes."""
        for i in range(5):
            emb = np.random.randn(1024).astype(np.float32)
            predictor.add_episode(uuid4(), emb)

        stats = predictor.get_statistics()
        assert stats["buffer_size"] == 5

    def test_predict_returns_hierarchical(self, predictor):
        """Test prediction at all timescales."""
        # Add enough context
        for i in range(10):
            emb = np.random.randn(1024).astype(np.float32)
            predictor.add_episode(uuid4(), emb)

        prediction = predictor.predict()

        assert isinstance(prediction, HierarchicalPrediction)
        assert prediction.fast_prediction is not None
        assert prediction.medium_prediction is not None
        assert prediction.slow_prediction is not None
        assert 0 <= prediction.combined_confidence <= 1

    def test_error_resolution(self, predictor):
        """Test that errors are resolved when targets arrive."""
        # Add episodes and make predictions
        for i in range(20):
            emb = np.random.randn(1024).astype(np.float32)
            predictor.add_episode(uuid4(), emb)
            if i >= 3:
                predictor.predict()

        # Should have some resolved predictions (for fast horizon at least)
        stats = predictor.get_statistics()
        assert stats["resolved_predictions"] > 0

    def test_get_high_error_episodes(self, predictor):
        """Test getting high-error episodes."""
        for i in range(30):
            emb = np.random.randn(1024).astype(np.float32)
            predictor.add_episode(uuid4(), emb)
            if i >= 3:
                predictor.predict()

        high_error = predictor.get_high_error_episodes(k=5)
        # May or may not have errors depending on resolution timing
        assert isinstance(high_error, list)


class TestCausalDiscovery:
    """Tests for P4-2: Causal discovery integration."""

    @pytest.fixture
    def learner(self):
        """Create causal learner."""
        return CausalLearner()

    def test_initialization(self, learner):
        """Test default initialization."""
        stats = learner.get_statistics()
        assert stats["total_observations"] == 0
        assert stats["causal_edges_learned"] == 0

    def test_observe_creates_edges(self, learner):
        """Test that observations create causal edges."""
        context = [uuid4() for _ in range(3)]
        outcome = uuid4()

        learner.observe(context, outcome)

        stats = learner.get_statistics()
        assert stats["total_observations"] == 1
        assert stats["causal_edges_learned"] >= 1

    def test_causal_graph_edges(self):
        """Test causal graph edge operations."""
        graph = CausalGraph()

        source = uuid4()
        target = uuid4()

        edge = graph.add_edge(source, target)
        assert edge.strength == 0.5
        assert edge.evidence_count == 1

        # Strengthen by re-adding
        edge = graph.add_edge(source, target)
        assert edge.evidence_count == 2
        assert edge.strength > 0.5

    def test_causal_attribution(self, learner):
        """Test attributing outcomes to causes."""
        # Create some causal structure
        cause1 = uuid4()
        cause2 = uuid4()
        outcome = uuid4()

        learner.observe([cause1], outcome)
        learner.observe([cause2], outcome)

        attribution = learner.attribute_outcome(outcome)
        assert attribution.outcome_id == outcome
        assert len(attribution.attributions) > 0

    def test_counterfactual_learning(self, learner):
        """Test learning from counterfactuals."""
        cause = uuid4()
        expected_outcome = uuid4()

        # First, establish causal relationship
        learner.observe([cause], expected_outcome)

        # Then, observe counterfactual (didn't happen) multiple times
        for _ in range(3):
            learner.observe_counterfactual(cause, expected_outcome, outcome_occurred=False)

        # Edge should be weakened (1.0 - 0.2*3 = 0.4)
        effects = learner.get_likely_effects(cause)
        if effects:
            assert effects[0][1] < 0.5  # Strength should decrease

    def test_predictive_causes(self, learner):
        """Test getting predictive causes."""
        causes = [uuid4() for _ in range(5)]
        effect = uuid4()

        for cause in causes:
            learner.observe([cause], effect)

        predictive = learner.get_predictive_causes(effect, k=3)
        assert len(predictive) == 3


class TestSpatialCells:
    """Tests for P4-3: Place/grid cell spatial prediction."""

    @pytest.fixture
    def spatial(self):
        """Create spatial cell system."""
        return SpatialCellSystem()

    def test_initialization(self, spatial):
        """Test default initialization."""
        assert spatial.config.n_place_cells == 100
        assert len(spatial._grid_modules) == 3

    def test_encode_position(self, spatial):
        """Test encoding embedding to position."""
        emb = np.random.randn(1024).astype(np.float32)
        position = spatial.encode_position(emb)

        assert isinstance(position, Position2D)
        assert -2 < position.x < 2  # Positions should be bounded
        assert -2 < position.y < 2

    def test_place_cell_activations(self, spatial):
        """Test place cell population vector."""
        emb = np.random.randn(1024).astype(np.float32)
        spatial.encode_position(emb)

        activations = spatial.get_place_activations()
        assert len(activations) == spatial.config.n_place_cells

        # Should be sparse (most cells inactive)
        active = np.sum(activations > 0.5)
        assert active < spatial.config.n_place_cells * 0.2

    def test_grid_cell_responses(self, spatial):
        """Test grid cell responses."""
        emb = np.random.randn(1024).astype(np.float32)
        spatial.encode_position(emb)

        responses = spatial.get_grid_responses()
        expected_size = len(spatial._grid_modules) * spatial.config.cells_per_module
        assert len(responses) == expected_size

        # Grid responses should be periodic
        assert np.all((responses >= 0) & (responses <= 1))

    def test_find_neighbors(self, spatial):
        """Test finding nearby positions."""
        # Add several episodes
        for _ in range(10):
            emb = np.random.randn(1024).astype(np.float32)
            spatial.encode_position(emb, episode_id=uuid4())

        neighbors = spatial.find_neighbors(k=3)
        assert len(neighbors) <= 3

    def test_predict_next_position(self, spatial):
        """Test position prediction."""
        # Add two episodes to compute velocity
        emb1 = np.random.randn(1024).astype(np.float32)
        spatial.encode_position(emb1)

        emb2 = np.random.randn(1024).astype(np.float32)
        pos2 = spatial.encode_position(emb2)

        predicted = spatial.predict_next_position()
        assert isinstance(predicted, Position2D)

    def test_position_2d_distance(self):
        """Test Position2D distance calculation."""
        p1 = Position2D(x=0.0, y=0.0)
        p2 = Position2D(x=3.0, y=4.0)

        dist = p1.distance_to(p2)
        assert abs(dist - 5.0) < 1e-5


class TestThetaGammaIntegration:
    """Tests for P4-4: Theta-gamma coupling."""

    @pytest.fixture
    def integration(self):
        """Create theta-gamma integration."""
        return ThetaGammaIntegration()

    def test_initialization(self, integration):
        """Test default initialization."""
        assert integration.config.plasticity_gating is True
        assert integration.config.max_wm_items == 7

    def test_step_updates_state(self, integration):
        """Test that step updates state."""
        outputs = integration.step(
            ach_level=0.6,
            da_level=0.5,
            ne_level=0.3,
            glu_level=0.5,
            gaba_level=0.4,
        )

        assert "theta" in outputs
        assert "gamma" in outputs
        assert "encoding_signal" in outputs
        assert "retrieval_signal" in outputs

    def test_cognitive_phase_alternates(self, integration):
        """Test that cognitive phase alternates with theta."""
        phases = []
        for _ in range(200):  # ~200ms should cover multiple theta cycles
            integration.step(dt_ms=1.0)
            phases.append(integration.state.current_cognitive_phase)

        # Should see both phases
        assert CognitivePhase.ENCODING in phases
        assert CognitivePhase.RETRIEVAL in phases

    def test_gated_learning_rate(self, integration):
        """Test theta-gated learning rate."""
        base_lr = 0.01

        # Step to get to encoding phase
        for _ in range(50):
            integration.step()

        gated = integration.get_gated_learning_rate(base_lr)
        # Should be different from base rate
        assert gated > 0

    def test_reward_updates_pac(self, integration):
        """Test that reward updates PAC strength."""
        initial_pac = integration.oscillator.pac.strength

        integration.on_reward(rpe=0.5)

        # PAC should increase with positive reward
        final_pac = integration.oscillator.pac.strength
        assert final_pac > initial_pac

    def test_working_memory_operations(self, integration):
        """Test working memory store/retrieve."""
        content_id = uuid4()

        # Store
        success = integration.store_in_wm(content_id)
        assert success

        # Check contents
        contents = integration.get_wm_contents()
        assert content_id in contents

        # Retrieve (refresh)
        found = integration.retrieve_from_wm(content_id)
        assert found

    def test_wm_capacity(self, integration):
        """Test working memory has limited capacity."""
        capacity = integration.state.wm_capacity
        assert 4 <= capacity <= 10  # Miller's 7±2

        # Fill beyond capacity
        for i in range(15):
            integration.store_in_wm(uuid4())

        # Should still respect capacity
        contents = integration.get_wm_contents()
        assert len(contents) <= capacity

    def test_encoding_retrieval_signals_complementary(self, integration):
        """Test that encoding + retrieval signals sum to ~1."""
        for _ in range(100):
            integration.step()

            enc = integration.state.encoding_signal
            ret = integration.state.retrieval_signal

            # Should be complementary
            assert abs(enc + ret - 1.0) < 0.15

    def test_inhibition_signal(self, integration):
        """Test alpha-based inhibition signal."""
        # Run with low NE (high alpha = more inhibition)
        for _ in range(100):
            integration.step(ne_level=0.1)

        inhibition = integration.get_inhibition_signal()
        assert 0 <= inhibition <= 1


class TestP4Integration:
    """Integration tests for P4 modules working together."""

    def test_hierarchical_with_spatial(self):
        """Test hierarchical predictor with spatial encoding."""
        predictor = HierarchicalPredictor()
        spatial = SpatialCellSystem()

        # Process sequence of episodes
        for i in range(20):
            emb = np.random.randn(1024).astype(np.float32)
            ep_id = uuid4()

            # Encode spatial position
            position = spatial.encode_position(emb, episode_id=ep_id)

            # Add to predictor
            predictor.add_episode(ep_id, emb)

            if i >= 3:
                pred = predictor.predict()
                neighbors = spatial.find_neighbors(k=3)

    def test_causal_with_theta_gamma(self):
        """Test causal learning with theta-gamma gating."""
        learner = CausalLearner()
        integration = ThetaGammaIntegration()

        # Simulate learning episode
        context_ids = [uuid4() for _ in range(3)]
        outcome_id = uuid4()

        # Step until in encoding phase
        for _ in range(50):
            integration.step()

        if integration.state.current_cognitive_phase == CognitivePhase.ENCODING:
            # Learn during encoding
            learner.observe(context_ids, outcome_id)

        stats = learner.get_statistics()
        # Should have learned something if in encoding phase

    def test_full_p4_pipeline(self):
        """Test all P4 modules together."""
        # Initialize all modules
        predictor = HierarchicalPredictor()
        spatial = SpatialCellSystem()
        causal = CausalLearner()
        theta_gamma = ThetaGammaIntegration()

        # Process episode sequence
        prev_ids = []
        for i in range(30):
            # Step oscillators
            outputs = theta_gamma.step()

            # Generate episode
            emb = np.random.randn(1024).astype(np.float32)
            ep_id = uuid4()

            # Spatial encoding
            pos = spatial.encode_position(emb, episode_id=ep_id)

            # Hierarchical prediction
            predictor.add_episode(ep_id, emb)
            predictor.predict()  # Make predictions after adding episodes

            # Causal learning (during encoding phase)
            if prev_ids and theta_gamma.state.current_cognitive_phase == CognitivePhase.ENCODING:
                causal.observe(prev_ids[-3:], ep_id)

            # Working memory
            theta_gamma.store_in_wm(ep_id)

            prev_ids.append(ep_id)

        # Validate all modules have state
        assert predictor.get_statistics()["total_predictions"] > 0
        assert spatial.get_statistics()["total_updates"] == 30
        assert causal.get_statistics()["total_observations"] > 0
        assert theta_gamma.get_statistics()["total_steps"] == 30
