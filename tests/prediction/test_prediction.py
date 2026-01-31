"""Tests for P2: Latent Prediction Module."""

import numpy as np
import pytest
from uuid import uuid4

from t4dm.prediction import (
    ContextEncoder,
    ContextEncoderConfig,
    EncodedContext,
    LatentPredictor,
    LatentPredictorConfig,
    Prediction,
    PredictionError,
    PredictionIntegration,
    PredictionIntegrationConfig,
    PredictionTracker,
    TrackerConfig,
    create_prediction_integration,
)


class TestContextEncoder:
    """Test P2-1: ContextEncoder."""

    def test_initialization(self):
        """Test default initialization."""
        encoder = ContextEncoder()
        assert encoder.config.embedding_dim == 1024
        assert encoder.config.context_dim == 1024
        assert encoder.config.aggregation == "attention"

    def test_custom_config(self):
        """Test custom configuration."""
        config = ContextEncoderConfig(
            embedding_dim=512,
            hidden_dim=256,
            context_dim=512,
            aggregation="mean",
        )
        encoder = ContextEncoder(config)
        assert encoder.config.embedding_dim == 512
        assert encoder.config.aggregation == "mean"

    def test_encode_single_embedding(self):
        """Test encoding single embedding."""
        encoder = ContextEncoder()
        embedding = np.random.randn(1024).astype(np.float32)

        result = encoder.encode([embedding])

        assert isinstance(result, EncodedContext)
        assert result.context_vector.shape == (1024,)
        assert result.n_episodes == 1
        # Should be normalized
        norm = np.linalg.norm(result.context_vector)
        assert abs(norm - 1.0) < 0.01

    def test_encode_multiple_embeddings(self):
        """Test encoding multiple embeddings."""
        encoder = ContextEncoder()
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(5)]

        result = encoder.encode(embeddings)

        assert result.n_episodes == 5
        assert result.attention_weights is not None
        assert result.attention_weights.shape == (5,)
        # Attention weights should sum to 1
        assert abs(result.attention_weights.sum() - 1.0) < 0.01

    def test_encode_empty_context(self):
        """Test encoding empty context."""
        encoder = ContextEncoder()

        result = encoder.encode([])

        assert result.n_episodes == 0
        assert np.allclose(result.context_vector, 0)

    def test_encode_as_array(self):
        """Test encoding array input."""
        encoder = ContextEncoder()
        embeddings = np.random.randn(3, 1024).astype(np.float32)

        result = encoder.encode(embeddings)

        assert result.n_episodes == 3

    def test_mean_aggregation(self):
        """Test mean aggregation mode."""
        config = ContextEncoderConfig(aggregation="mean")
        encoder = ContextEncoder(config)
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(4)]

        result = encoder.encode(embeddings)

        # Mean aggregation should have uniform weights
        expected_weight = 1.0 / 4
        assert np.allclose(result.attention_weights, expected_weight, atol=0.01)

    def test_last_aggregation(self):
        """Test last-only aggregation mode."""
        config = ContextEncoderConfig(aggregation="last")
        encoder = ContextEncoder(config)
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(4)]

        result = encoder.encode(embeddings)

        # Last aggregation: only last weight = 1
        assert result.attention_weights[-1] == 1.0
        assert np.sum(result.attention_weights[:-1]) == 0.0

    def test_truncate_long_context(self):
        """Test that long context is truncated."""
        config = ContextEncoderConfig(max_context_length=4)
        encoder = ContextEncoder(config)
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(10)]

        result = encoder.encode(embeddings)

        # Should only use last 4
        assert result.n_episodes == 4

    def test_save_load_state(self):
        """Test state persistence."""
        encoder = ContextEncoder()
        embeddings = [np.random.randn(1024).astype(np.float32) for _ in range(3)]
        result1 = encoder.encode(embeddings)

        state = encoder.save_state()

        encoder2 = ContextEncoder()
        encoder2.load_state(state)
        result2 = encoder2.encode(embeddings)

        assert np.allclose(result1.context_vector, result2.context_vector, atol=1e-5)


class TestLatentPredictor:
    """Test P2-2: LatentPredictor."""

    def test_initialization(self):
        """Test default initialization."""
        predictor = LatentPredictor()
        assert predictor.config.context_dim == 1024
        assert predictor.config.hidden_dim == 512
        assert predictor.config.output_dim == 1024

    def test_predict(self):
        """Test prediction from context."""
        predictor = LatentPredictor()
        context = np.random.randn(1024).astype(np.float32)

        result = predictor.predict(context)

        assert isinstance(result, Prediction)
        assert result.predicted_embedding.shape == (1024,)
        assert 0 <= result.confidence <= 1
        # Should be normalized
        norm = np.linalg.norm(result.predicted_embedding)
        assert abs(norm - 1.0) < 0.01

    def test_compute_error(self):
        """Test prediction error computation."""
        predictor = LatentPredictor()
        context = np.random.randn(1024).astype(np.float32)
        prediction = predictor.predict(context)
        actual = np.random.randn(1024).astype(np.float32)
        actual = actual / np.linalg.norm(actual)  # Normalize
        episode_id = uuid4()

        error = predictor.compute_error(prediction, actual, episode_id)

        assert isinstance(error, PredictionError)
        assert error.episode_id == episode_id
        assert error.error_magnitude >= 0
        assert 0 <= error.cosine_error <= 2  # Cosine error can be 0-2
        assert error.combined_error >= 0

    def test_error_zero_for_identical(self):
        """Test that error is zero for identical predictions."""
        predictor = LatentPredictor()
        context = np.random.randn(1024).astype(np.float32)
        prediction = predictor.predict(context)

        # Use predicted as actual
        error = predictor.compute_error(
            prediction, prediction.predicted_embedding, uuid4()
        )

        assert error.error_magnitude < 0.01
        assert error.cosine_error < 0.01

    def test_train_step(self):
        """Test training step."""
        predictor = LatentPredictor()
        context = np.random.randn(1024).astype(np.float32)
        target = np.random.randn(1024).astype(np.float32)
        target = target / np.linalg.norm(target)

        loss1 = predictor.train_step(context, target)
        # Train again
        loss2 = predictor.train_step(context, target)

        # Loss should decrease (or at least not increase much)
        assert loss1 >= 0
        assert loss2 >= 0

    def test_training_reduces_error(self):
        """Test that training reduces prediction error."""
        predictor = LatentPredictor()
        context = np.random.randn(1024).astype(np.float32)
        target = np.random.randn(1024).astype(np.float32)
        target = target / np.linalg.norm(target)

        # Initial prediction
        pred1 = predictor.predict(context)
        error1 = np.linalg.norm(pred1.predicted_embedding - target)

        # Train for several steps
        for _ in range(50):
            predictor.train_step(context, target, learning_rate=0.01)

        # Final prediction
        pred2 = predictor.predict(context)
        error2 = np.linalg.norm(pred2.predicted_embedding - target)

        # Error should decrease significantly
        assert error2 < error1 * 0.5

    def test_residual_connection(self):
        """Test residual connection."""
        config = LatentPredictorConfig(residual=True)
        predictor = LatentPredictor(config)
        context = np.random.randn(1024).astype(np.float32)

        result = predictor.predict(context)

        # Residual should make output not orthogonal to input
        dot = np.dot(result.predicted_embedding, context / np.linalg.norm(context))
        assert abs(dot) > 0.01  # Some correlation

    def test_statistics(self):
        """Test statistics tracking."""
        predictor = LatentPredictor()
        context = np.random.randn(1024).astype(np.float32)
        target = np.random.randn(1024).astype(np.float32)

        # Make some predictions
        for _ in range(5):
            pred = predictor.predict(context)
            predictor.compute_error(pred, target, uuid4())

        stats = predictor.get_statistics()

        assert stats["total_predictions"] == 5
        assert "mean_error" in stats
        assert "error_trend" in stats

    def test_save_load_state(self):
        """Test state persistence."""
        predictor = LatentPredictor()
        context = np.random.randn(1024).astype(np.float32)
        target = np.random.randn(1024).astype(np.float32)
        target = target / np.linalg.norm(target)

        # Train a bit
        for _ in range(10):
            predictor.train_step(context, target)

        pred1 = predictor.predict(context)
        state = predictor.save_state()

        predictor2 = LatentPredictor()
        predictor2.load_state(state)
        pred2 = predictor2.predict(context)

        assert np.allclose(pred1.predicted_embedding, pred2.predicted_embedding, atol=1e-5)


class TestPredictionTracker:
    """Test P2-3: PredictionTracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker with encoder and predictor."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        return PredictionTracker(encoder, predictor)

    def test_initialization(self, tracker):
        """Test default initialization."""
        stats = tracker.get_statistics()
        assert stats["pending_predictions"] == 0
        assert stats["tracked_errors"] == 0

    def test_make_prediction(self, tracker):
        """Test making a prediction."""
        context = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(3)
        ]

        tracked = tracker.make_prediction(context)

        assert tracked.episode_id == context[-1][0]  # Last context episode
        assert len(tracked.context_ids) == 3
        assert not tracked.resolved

    def test_resolve_prediction(self, tracker):
        """Test resolving a prediction."""
        context = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(3)
        ]

        tracked = tracker.make_prediction(context)
        before_id = tracked.episode_id

        # Resolve with actual outcome
        actual_id = uuid4()
        actual_embedding = np.random.randn(1024).astype(np.float32)

        error = tracker.resolve_prediction(before_id, actual_embedding, actual_id)

        assert error is not None
        assert error.episode_id == actual_id
        assert error.error_magnitude >= 0

    def test_resolve_nonexistent(self, tracker):
        """Test resolving nonexistent prediction."""
        error = tracker.resolve_prediction(
            uuid4(),
            np.random.randn(1024).astype(np.float32),
            uuid4(),
        )
        assert error is None

    def test_get_prediction_error(self, tracker):
        """Test getting prediction error for episode."""
        context = [(uuid4(), np.random.randn(1024).astype(np.float32))]
        tracked = tracker.make_prediction(context)

        actual_id = uuid4()
        actual = np.random.randn(1024).astype(np.float32)
        tracker.resolve_prediction(tracked.episode_id, actual, actual_id)

        error = tracker.get_prediction_error(actual_id)
        assert error is not None
        assert error >= 0

    def test_high_error_episodes(self, tracker):
        """Test getting high-error episodes."""
        # Make several predictions with varying errors
        for _ in range(10):
            context = [(uuid4(), np.random.randn(1024).astype(np.float32))]
            tracked = tracker.make_prediction(context)

            # Random actual embedding (will have varying error)
            actual_id = uuid4()
            actual = np.random.randn(1024).astype(np.float32)
            tracker.resolve_prediction(tracked.episode_id, actual, actual_id)

        high_error = tracker.get_high_error_episodes(k=5)

        # Should get at most 5
        assert len(high_error) <= 5
        # Should be sorted by priority (descending)
        if len(high_error) > 1:
            priorities = [p for _, p in high_error]
            assert priorities == sorted(priorities, reverse=True)

    def test_tag_episode_with_error(self, tracker):
        """Test tagging episode with error."""
        # Create mock episode
        class MockEpisode:
            def __init__(self):
                self.id = uuid4()
                self.prediction_error = None
                self.prediction_error_timestamp = None

        episode = MockEpisode()

        # Make and resolve prediction
        context = [(uuid4(), np.random.randn(1024).astype(np.float32))]
        tracked = tracker.make_prediction(context)
        actual = np.random.randn(1024).astype(np.float32)
        tracker.resolve_prediction(tracked.episode_id, actual, episode.id)

        # Tag episode
        tracker.tag_episode_with_error(episode)

        assert episode.prediction_error is not None
        assert episode.prediction_error_timestamp is not None

    def test_statistics(self, tracker):
        """Test statistics."""
        context = [(uuid4(), np.random.randn(1024).astype(np.float32))]
        tracked = tracker.make_prediction(context)

        stats = tracker.get_statistics()
        assert stats["pending_predictions"] == 1

        # Resolve
        actual_id = uuid4()
        actual = np.random.randn(1024).astype(np.float32)
        tracker.resolve_prediction(tracked.episode_id, actual, actual_id)

        stats = tracker.get_statistics()
        assert stats["pending_predictions"] == 0
        assert stats["tracked_errors"] == 1
        assert stats["total_predictions"] == 1

    def test_save_load_state(self, tracker):
        """Test state persistence."""
        # Make some predictions
        for _ in range(5):
            context = [(uuid4(), np.random.randn(1024).astype(np.float32))]
            tracked = tracker.make_prediction(context)
            actual_id = uuid4()
            actual = np.random.randn(1024).astype(np.float32)
            tracker.resolve_prediction(tracked.episode_id, actual, actual_id)

        state = tracker.save_state()

        # Create new tracker
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        tracker2 = PredictionTracker(encoder, predictor)
        tracker2.load_state(state)

        stats1 = tracker.get_statistics()
        stats2 = tracker2.get_statistics()

        assert stats1["tracked_errors"] == stats2["tracked_errors"]
        assert stats1["error_mean"] == stats2["error_mean"]


class TestPredictionIntegration:
    """Test P2-4: PredictionIntegration."""

    def test_initialization(self):
        """Test default initialization."""
        integration = PredictionIntegration()
        stats = integration.get_statistics()
        assert stats["predictions_made"] == 0
        assert stats["predictions_resolved"] == 0

    def test_factory_function(self):
        """Test create_prediction_integration factory."""
        integration = create_prediction_integration()
        assert isinstance(integration, PredictionIntegration)

    def test_custom_config(self):
        """Test custom configuration."""
        config = PredictionIntegrationConfig(
            context_size=3,
            training_lr=0.01,
        )
        integration = PredictionIntegration(config)
        assert integration.config.context_size == 3
        assert integration.config.training_lr == 0.01

    def test_episode_lifecycle(self):
        """Test full episode lifecycle."""
        integration = PredictionIntegration()

        class MockEpisode:
            def __init__(self):
                self.id = uuid4()
                self.embedding = np.random.randn(1024).astype(np.float32)
                self.embedding = self.embedding / np.linalg.norm(self.embedding)
                self.prediction_error = None
                self.prediction_error_timestamp = None

        # Create several episodes
        episodes = [MockEpisode() for _ in range(5)]

        for i, ep in enumerate(episodes):
            # On created
            integration.on_episode_created(ep)
            # On stored
            error = integration.on_episode_stored(ep)

            if i >= 1:
                # Should have resolved prediction after first
                assert error is not None or i == 1

        stats = integration.get_statistics()
        assert stats["predictions_made"] >= 3
        assert stats["recent_buffer_size"] == 5

    def test_on_replay(self):
        """Test replay training."""
        integration = PredictionIntegration()

        class MockEpisode:
            def __init__(self):
                self.id = uuid4()
                self.embedding = np.random.randn(1024).astype(np.float32)
                self.prediction_error = None
                self.prediction_error_timestamp = None

        episode = MockEpisode()
        context = [np.random.randn(1024).astype(np.float32) for _ in range(3)]

        loss = integration.on_replay(episode, context)

        assert loss >= 0
        stats = integration.get_statistics()
        assert stats["total_training_steps"] >= 1

    def test_predict_next(self):
        """Test predict_next for anticipatory retrieval."""
        integration = PredictionIntegration()

        context = [np.random.randn(1024).astype(np.float32) for _ in range(3)]
        predicted = integration.predict_next(context)

        assert predicted.shape == (1024,)
        # Should be normalized
        norm = np.linalg.norm(predicted)
        assert abs(norm - 1.0) < 0.01

    def test_get_priority_episodes(self):
        """Test priority episode retrieval."""
        integration = PredictionIntegration()

        class MockEpisode:
            def __init__(self):
                self.id = uuid4()
                self.embedding = np.random.randn(1024).astype(np.float32)
                self.prediction_error = None
                self.prediction_error_timestamp = None

        # Create episodes to build up predictions
        for _ in range(10):
            ep = MockEpisode()
            integration.on_episode_created(ep)
            integration.on_episode_stored(ep)

        priority = integration.get_priority_episodes(k=5)
        # May or may not have high-error episodes
        assert isinstance(priority, list)

    def test_save_load_state(self):
        """Test state persistence."""
        integration = PredictionIntegration()

        class MockEpisode:
            def __init__(self):
                self.id = uuid4()
                self.embedding = np.random.randn(1024).astype(np.float32)
                self.prediction_error = None
                self.prediction_error_timestamp = None

        # Build up some state
        for _ in range(5):
            ep = MockEpisode()
            integration.on_episode_created(ep)
            integration.on_episode_stored(ep)

        state = integration.save_state()

        integration2 = PredictionIntegration()
        integration2.load_state(state)

        stats1 = integration.get_statistics()
        stats2 = integration2.get_statistics()

        assert stats1["predictions_made"] == stats2["predictions_made"]
        assert stats1["total_training_steps"] == stats2["total_training_steps"]


class TestPipelineIntegration:
    """Integration tests for prediction pipeline."""

    def test_full_prediction_pipeline(self):
        """Test complete prediction workflow."""
        # Initialize components
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        tracker = PredictionTracker(encoder, predictor)

        # Simulate episode sequence
        embeddings = []
        episode_ids = []

        for i in range(10):
            # Generate new episode
            new_id = uuid4()
            new_emb = np.random.randn(1024).astype(np.float32)
            new_emb = new_emb / np.linalg.norm(new_emb)

            if len(embeddings) >= 3:
                # Make prediction from context
                context = list(zip(episode_ids[-3:], embeddings[-3:]))
                tracked = tracker.make_prediction(context)

                # Resolve with actual
                tracker.resolve_prediction(
                    tracked.episode_id, new_emb, new_id
                )

            embeddings.append(new_emb)
            episode_ids.append(new_id)

        # Check we tracked predictions
        stats = tracker.get_statistics()
        assert stats["tracked_errors"] == 7  # First 3 have no context

    def test_prediction_improves_with_training(self):
        """Test that predictor improves over time."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()

        # Create predictable sequence (each is shifted version)
        base = np.random.randn(1024).astype(np.float32)
        sequence = []
        for i in range(20):
            shifted = np.roll(base, i * 10)
            shifted = shifted / np.linalg.norm(shifted)
            sequence.append(shifted)

        # Train on sequence
        errors_before = []
        errors_after = []

        for i in range(3, len(sequence)):
            context_embs = sequence[i-3:i]
            context = encoder.encode(context_embs)
            target = sequence[i]

            # Before training error
            pred = predictor.predict(context.context_vector)
            error = np.linalg.norm(pred.predicted_embedding - target)
            errors_before.append(error)

            # Train
            predictor.train_step(context.context_vector, target)

        # Run through again to get after errors
        for i in range(3, len(sequence)):
            context_embs = sequence[i-3:i]
            context = encoder.encode(context_embs)
            target = sequence[i]

            pred = predictor.predict(context.context_vector)
            error = np.linalg.norm(pred.predicted_embedding - target)
            errors_after.append(error)

        # Should have lower error after training
        assert np.mean(errors_after) < np.mean(errors_before)

    def test_end_to_end_with_integration(self):
        """Test end-to-end with PredictionIntegration."""
        integration = create_prediction_integration()

        class MockEpisode:
            def __init__(self, content_seed=None):
                self.id = uuid4()
                np.random.seed(content_seed)
                self.embedding = np.random.randn(1024).astype(np.float32)
                self.embedding = self.embedding / np.linalg.norm(self.embedding)
                self.prediction_error = None
                self.prediction_error_timestamp = None

        # Simulate realistic episode flow
        for i in range(20):
            ep = MockEpisode(content_seed=i)

            # Episode created
            integration.on_episode_created(ep)

            # Episode stored (triggers prediction resolution)
            error = integration.on_episode_stored(ep)

        stats = integration.get_statistics()
        assert stats["predictions_made"] > 10
        assert stats["predictions_resolved"] > 5
        assert stats["total_training_steps"] > 0
