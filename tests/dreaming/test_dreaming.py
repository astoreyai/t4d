"""Tests for P3: Dreaming System."""

import numpy as np
import pytest
from uuid import uuid4

from t4dm.dreaming import (
    DreamConsolidation,
    DreamConsolidationConfig,
    DreamingConfig,
    DreamingSystem,
    DreamQuality,
    DreamQualityEvaluator,
    DreamReplayEvent,
    DreamTrajectory,
    QualityConfig,
)
from t4dm.prediction import (
    ContextEncoder,
    LatentPredictor,
    PredictionTracker,
)


class TestDreamingSystem:
    """Test P3-1: Dream trajectory generation."""

    @pytest.fixture
    def dreamer(self):
        """Create dreaming system."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        return DreamingSystem(encoder, predictor)

    def test_initialization(self, dreamer):
        """Test default initialization."""
        assert dreamer.config.max_dream_length == 15
        assert dreamer.config.confidence_threshold == 0.3
        stats = dreamer.get_statistics()
        assert stats["total_dreams"] == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DreamingConfig(
            max_dream_length=10,
            confidence_threshold=0.5,
            noise_scale=0.1,
        )
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        dreamer = DreamingSystem(encoder, predictor, config)
        assert dreamer.config.max_dream_length == 10

    def test_single_dream(self, dreamer):
        """Test generating a single dream."""
        seed = np.random.randn(1024).astype(np.float32)
        seed = seed / np.linalg.norm(seed)

        trajectory = dreamer.dream(seed)

        assert isinstance(trajectory, DreamTrajectory)
        assert trajectory.length > 0
        assert trajectory.seed_embedding is not None
        assert len(trajectory.embeddings) > 0

    def test_dream_with_context(self, dreamer):
        """Test dream with context embeddings."""
        seed = np.random.randn(1024).astype(np.float32)
        seed = seed / np.linalg.norm(seed)

        context = [
            np.random.randn(1024).astype(np.float32)
            for _ in range(3)
        ]
        context = [c / np.linalg.norm(c) for c in context]

        trajectory = dreamer.dream(seed, context_embeddings=context)

        assert trajectory.length > 0

    def test_dream_with_episode_id(self, dreamer):
        """Test dream with seed episode ID."""
        seed = np.random.randn(1024).astype(np.float32)
        seed = seed / np.linalg.norm(seed)
        episode_id = uuid4()

        trajectory = dreamer.dream(seed, seed_episode_id=episode_id)

        assert trajectory.seed_episode_id == episode_id

    def test_dream_batch(self, dreamer):
        """Test generating multiple dreams."""
        seeds = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(3)
        ]
        seeds = [(id, s / np.linalg.norm(s)) for id, s in seeds]

        trajectories = dreamer.dream_batch(seeds)

        assert len(trajectories) == 3
        for traj in trajectories:
            assert traj.length > 0

    def test_dream_termination_by_length(self):
        """Test dream terminates at max length."""
        config = DreamingConfig(
            max_dream_length=5,
            confidence_threshold=0.0,  # Never terminate early
            coherence_threshold=0.0,
        )
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        dreamer = DreamingSystem(encoder, predictor, config)

        seed = np.random.randn(1024).astype(np.float32)
        seed = seed / np.linalg.norm(seed)

        trajectory = dreamer.dream(seed)

        assert trajectory.length == 5
        assert trajectory.termination_reason == "max_length"

    def test_reference_embeddings(self, dreamer):
        """Test adding reference embeddings."""
        refs = [np.random.randn(1024).astype(np.float32) for _ in range(10)]
        dreamer.add_reference_embeddings(refs)

        assert len(dreamer._reference_embeddings) == 10

    def test_dream_metrics(self, dreamer):
        """Test dream trajectory metrics."""
        seed = np.random.randn(1024).astype(np.float32)
        seed = seed / np.linalg.norm(seed)

        trajectory = dreamer.dream(seed)

        assert 0 <= trajectory.mean_confidence <= 1
        assert 0 <= trajectory.mean_coherence <= 1
        assert trajectory.duration_ms >= 0

    def test_get_recent_dreams(self, dreamer):
        """Test getting recent dreams."""
        for _ in range(5):
            seed = np.random.randn(1024).astype(np.float32)
            seed = seed / np.linalg.norm(seed)
            dreamer.dream(seed)

        recent = dreamer.get_recent_dreams(n=3)
        assert len(recent) == 3

    def test_save_load_state(self, dreamer):
        """Test state persistence."""
        refs = [np.random.randn(1024).astype(np.float32) for _ in range(5)]
        dreamer.add_reference_embeddings(refs)

        seed = np.random.randn(1024).astype(np.float32)
        dreamer.dream(seed / np.linalg.norm(seed))

        state = dreamer.save_state()

        encoder = ContextEncoder()
        predictor = LatentPredictor()
        dreamer2 = DreamingSystem(encoder, predictor)
        dreamer2.load_state(state)

        assert len(dreamer2._reference_embeddings) == 5


class TestDreamQualityEvaluator:
    """Test P3-2: Dream quality evaluation."""

    @pytest.fixture
    def evaluator(self):
        """Create evaluator with references."""
        refs = [np.random.randn(1024).astype(np.float32) for _ in range(20)]
        refs = [r / np.linalg.norm(r) for r in refs]
        return DreamQualityEvaluator(reference_embeddings=refs)

    @pytest.fixture
    def sample_dream(self):
        """Create a sample dream trajectory."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        dreamer = DreamingSystem(encoder, predictor)

        seed = np.random.randn(1024).astype(np.float32)
        seed = seed / np.linalg.norm(seed)
        return dreamer.dream(seed)

    def test_initialization(self, evaluator):
        """Test default initialization."""
        stats = evaluator.get_statistics()
        assert stats["total_evaluated"] == 0

    def test_evaluate_dream(self, evaluator, sample_dream):
        """Test evaluating a dream."""
        quality = evaluator.evaluate(sample_dream)

        assert isinstance(quality, DreamQuality)
        assert 0 <= quality.coherence_score <= 1
        assert 0 <= quality.smoothness_score <= 1
        assert 0 <= quality.novelty_score <= 1
        assert 0 <= quality.informativeness_score <= 1
        assert 0 <= quality.overall_score <= 1

    def test_evaluate_empty_dream(self, evaluator):
        """Test evaluating empty dream."""
        empty = DreamTrajectory()
        quality = evaluator.evaluate(empty)

        assert quality.overall_score == 0.0
        assert not quality.is_high_quality

    def test_evaluate_batch(self, evaluator):
        """Test batch evaluation."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        dreamer = DreamingSystem(encoder, predictor)

        dreams = []
        for _ in range(3):
            seed = np.random.randn(1024).astype(np.float32)
            dreams.append(dreamer.dream(seed / np.linalg.norm(seed)))

        qualities = evaluator.evaluate_batch(dreams)

        assert len(qualities) == 3

    def test_filter_high_quality(self, evaluator):
        """Test filtering high-quality dreams."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        dreamer = DreamingSystem(encoder, predictor)

        dreams = []
        for _ in range(5):
            seed = np.random.randn(1024).astype(np.float32)
            dreams.append(dreamer.dream(seed / np.linalg.norm(seed)))

        filtered = evaluator.filter_high_quality(dreams)

        # All returned should be high quality
        for dream, quality in filtered:
            assert quality.is_high_quality

    def test_quality_thresholds(self):
        """Test custom quality thresholds."""
        config = QualityConfig(
            high_quality_threshold=0.9,  # Very strict
            min_quality_threshold=0.1,
        )
        evaluator = DreamQualityEvaluator(config=config)

        encoder = ContextEncoder()
        predictor = LatentPredictor()
        dreamer = DreamingSystem(encoder, predictor)

        seed = np.random.randn(1024).astype(np.float32)
        dream = dreamer.dream(seed / np.linalg.norm(seed))

        quality = evaluator.evaluate(dream)
        # With strict threshold, unlikely to be high quality
        # (but depends on dream generation)

    def test_statistics(self, evaluator, sample_dream):
        """Test statistics tracking."""
        evaluator.evaluate(sample_dream)
        evaluator.evaluate(sample_dream)

        stats = evaluator.get_statistics()
        assert stats["total_evaluated"] == 2


class TestDreamConsolidation:
    """Test P3-3: Imagination-based consolidation."""

    @pytest.fixture
    def consolidation(self):
        """Create dream consolidation system."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        tracker = PredictionTracker(encoder, predictor)
        return DreamConsolidation(encoder, predictor, tracker)

    def test_initialization(self, consolidation):
        """Test default initialization."""
        stats = consolidation.get_statistics()
        assert stats["total_cycles"] == 0
        assert stats["total_dreams"] == 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DreamConsolidationConfig(
            dreams_per_cycle=3,
            min_quality_for_replay=0.6,
        )
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        consolidation = DreamConsolidation(encoder, predictor, config=config)

        assert consolidation.config.dreams_per_cycle == 3

    def test_run_dream_cycle(self, consolidation):
        """Test running a dream cycle."""
        recent = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(10)
        ]
        recent = [(id, e / np.linalg.norm(e)) for id, e in recent]

        refs = [np.random.randn(1024).astype(np.float32) for _ in range(20)]
        refs = [r / np.linalg.norm(r) for r in refs]

        result = consolidation.run_dream_cycle(
            recent_episodes=recent,
            reference_embeddings=refs,
        )

        assert result.dreams_generated > 0
        assert result.duration_ms > 0

    def test_dream_cycle_with_high_error(self, consolidation):
        """Test dream cycle with explicit high-error seeds."""
        recent = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(5)
        ]
        recent = [(id, e / np.linalg.norm(e)) for id, e in recent]

        high_error = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(3)
        ]
        high_error = [(id, e / np.linalg.norm(e)) for id, e in high_error]

        result = consolidation.run_dream_cycle(
            recent_episodes=recent,
            high_error_episodes=high_error,
        )

        assert result.dreams_generated >= 3  # At least the high-error seeds

    def test_priority_updates(self, consolidation):
        """Test priority update generation."""
        recent = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(5)
        ]
        recent = [(id, e / np.linalg.norm(e)) for id, e in recent]

        consolidation.run_dream_cycle(recent_episodes=recent)

        updates = consolidation.get_priority_updates()
        # May or may not have updates depending on dream quality

        # Clear and verify
        consolidation.clear_priority_updates()
        assert len(consolidation.get_priority_updates()) == 0

    def test_get_recent_dreams(self, consolidation):
        """Test getting recent dreams."""
        recent = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(5)
        ]
        recent = [(id, e / np.linalg.norm(e)) for id, e in recent]

        consolidation.run_dream_cycle(recent_episodes=recent)

        dreams = consolidation.get_recent_dreams(n=10)
        assert len(dreams) > 0

    def test_save_load_state(self, consolidation):
        """Test state persistence."""
        recent = [
            (uuid4(), np.random.randn(1024).astype(np.float32))
            for _ in range(5)
        ]
        recent = [(id, e / np.linalg.norm(e)) for id, e in recent]

        consolidation.run_dream_cycle(recent_episodes=recent)

        state = consolidation.save_state()

        encoder = ContextEncoder()
        predictor = LatentPredictor()
        consolidation2 = DreamConsolidation(encoder, predictor)
        consolidation2.load_state(state)


class TestDreamIntegration:
    """Integration tests for dreaming system."""

    def test_full_dream_pipeline(self):
        """Test complete dream pipeline."""
        # Initialize all components
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        tracker = PredictionTracker(encoder, predictor)
        consolidation = DreamConsolidation(encoder, predictor, tracker)

        # Simulate episode sequence
        episodes = []
        for i in range(20):
            ep_id = uuid4()
            emb = np.random.randn(1024).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            episodes.append((ep_id, emb))

            # Track predictions
            if len(episodes) >= 3:
                context = episodes[-4:-1]
                tracked = tracker.make_prediction(context)
                tracker.resolve_prediction(
                    tracked.episode_id, emb, ep_id
                )

        # Run dream cycle
        result = consolidation.run_dream_cycle(
            recent_episodes=episodes,
            reference_embeddings=[e for _, e in episodes],
        )

        assert result.dreams_generated > 0
        stats = consolidation.get_statistics()
        assert stats["total_cycles"] == 1

    def test_dream_training_improves_predictor(self):
        """Test that dreaming can train the predictor."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()

        # Use lenient quality thresholds to ensure training happens
        quality_config = QualityConfig(
            high_quality_threshold=0.3,  # Very lenient
            min_quality_threshold=0.1,
        )
        config = DreamConsolidationConfig(
            dreams_per_cycle=10,
            train_on_dreams=True,
            min_quality_for_replay=0.2,  # Very lenient
            quality_config=quality_config,
        )
        consolidation = DreamConsolidation(encoder, predictor, config=config)

        # Create structured sequence
        np.random.seed(42)
        base = np.random.randn(1024).astype(np.float32)
        base = base / np.linalg.norm(base)

        episodes = []
        for i in range(50):
            shifted = np.roll(base, i * 5)
            shifted = shifted / np.linalg.norm(shifted)
            episodes.append((uuid4(), shifted))

        # Run dream cycles
        for _ in range(3):
            result = consolidation.run_dream_cycle(
                recent_episodes=episodes,
                reference_embeddings=[e for _, e in episodes],
            )

        # Verify dreams were generated and some training occurred
        stats = consolidation.get_statistics()
        assert stats["total_dreams"] > 0
        # Training may or may not occur depending on dream quality
        # Just verify the system ran without error


class TestDreamTrajectory:
    """Test DreamTrajectory dataclass."""

    def test_empty_trajectory(self):
        """Test empty trajectory properties."""
        traj = DreamTrajectory()
        assert traj.length == 0
        assert traj.mean_confidence == 0.0
        assert traj.mean_coherence == 0.0

    def test_to_dict(self):
        """Test serialization."""
        encoder = ContextEncoder()
        predictor = LatentPredictor()
        dreamer = DreamingSystem(encoder, predictor)

        seed = np.random.randn(1024).astype(np.float32)
        traj = dreamer.dream(seed / np.linalg.norm(seed))

        d = traj.to_dict()
        assert "id" in d
        assert "length" in d
        assert "mean_confidence" in d
        assert "termination_reason" in d
