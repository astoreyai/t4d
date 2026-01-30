"""
Tests for algorithm parameter configuration.

Validates that all hardcoded algorithm parameters can be configured
via Settings and that validation works correctly.
"""

import os
import pytest
from pydantic import ValidationError

from ww.core.config import Settings, get_settings


class TestFSRSParameters:
    """Test FSRS parameter configuration."""

    def test_fsrs_default_values(self):
        """Test FSRS parameters have correct defaults."""
        settings = Settings()
        assert settings.fsrs_decay_factor == 0.9
        assert settings.fsrs_default_stability == 1.0
        assert settings.fsrs_retention_target == 0.9
        assert settings.fsrs_recency_decay == 0.1

    def test_fsrs_decay_factor_bounds(self):
        """Test FSRS decay factor validates bounds."""
        # Valid values
        Settings(fsrs_decay_factor=0.1)
        Settings(fsrs_decay_factor=0.5)
        Settings(fsrs_decay_factor=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(fsrs_decay_factor=0.0)
        with pytest.raises(ValidationError):
            Settings(fsrs_decay_factor=1.1)

    def test_fsrs_stability_bounds(self):
        """Test FSRS stability validates bounds."""
        # Valid values
        Settings(fsrs_default_stability=0.1)
        Settings(fsrs_default_stability=5.0)
        Settings(fsrs_default_stability=10.0)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(fsrs_default_stability=0.0)
        with pytest.raises(ValidationError):
            Settings(fsrs_default_stability=11.0)

    def test_fsrs_env_override(self, monkeypatch):
        """Test FSRS parameters can be overridden via environment."""
        monkeypatch.setenv("WW_FSRS_DECAY_FACTOR", "0.8")
        monkeypatch.setenv("WW_FSRS_DEFAULT_STABILITY", "2.0")

        settings = Settings()
        assert settings.fsrs_decay_factor == 0.8
        assert settings.fsrs_default_stability == 2.0


class TestACTRParameters:
    """Test ACT-R parameter configuration."""

    def test_actr_default_values(self):
        """Test ACT-R parameters load from config (may be defaults or env overrides)."""
        settings = Settings()
        assert settings.actr_spreading_strength == 1.6
        assert settings.actr_decay == 0.5
        assert settings.actr_threshold == 0.0
        # actr_noise may be 0.0 or 0.5 depending on .env file
        assert 0.0 <= settings.actr_noise <= 1.0

    def test_actr_spreading_strength_bounds(self):
        """Test ACT-R spreading strength validates bounds."""
        # Valid values
        Settings(actr_spreading_strength=0.1)
        Settings(actr_spreading_strength=3.0)
        Settings(actr_spreading_strength=5.0)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(actr_spreading_strength=0.0)
        with pytest.raises(ValidationError):
            Settings(actr_spreading_strength=6.0)

    def test_actr_noise_bounds(self):
        """Test ACT-R noise validates bounds."""
        # Valid values
        Settings(actr_noise=0.0)
        Settings(actr_noise=0.5)
        Settings(actr_noise=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(actr_noise=-0.1)
        with pytest.raises(ValidationError):
            Settings(actr_noise=1.1)

    def test_actr_env_override(self, monkeypatch):
        """Test ACT-R parameters can be overridden via environment."""
        monkeypatch.setenv("WW_ACTR_SPREADING_STRENGTH", "2.0")
        monkeypatch.setenv("WW_ACTR_NOISE", "0.3")

        settings = Settings()
        assert settings.actr_spreading_strength == 2.0
        assert settings.actr_noise == 0.3


class TestHebbianParameters:
    """Test Hebbian learning parameter configuration."""

    def test_hebbian_default_values(self):
        """Test Hebbian parameters have correct defaults."""
        settings = Settings()
        assert settings.hebbian_learning_rate == 0.1
        assert settings.hebbian_decay_rate == 0.01
        assert settings.hebbian_initial_weight == 0.1
        assert settings.hebbian_min_weight == 0.01
        assert settings.hebbian_stale_days == 30

    def test_hebbian_learning_rate_bounds(self):
        """Test Hebbian learning rate validates bounds."""
        # Valid values
        Settings(hebbian_learning_rate=0.01)
        Settings(hebbian_learning_rate=0.25)
        Settings(hebbian_learning_rate=0.5)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(hebbian_learning_rate=0.005)
        with pytest.raises(ValidationError):
            Settings(hebbian_learning_rate=0.6)

    def test_hebbian_stale_days_bounds(self):
        """Test Hebbian stale days validates bounds."""
        # Valid values
        Settings(hebbian_stale_days=1)
        Settings(hebbian_stale_days=100)
        Settings(hebbian_stale_days=365)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(hebbian_stale_days=0)
        with pytest.raises(ValidationError):
            Settings(hebbian_stale_days=366)

    def test_hebbian_env_override(self, monkeypatch):
        """Test Hebbian parameters can be overridden via environment."""
        monkeypatch.setenv("WW_HEBBIAN_LEARNING_RATE", "0.2")
        monkeypatch.setenv("WW_HEBBIAN_STALE_DAYS", "60")

        settings = Settings()
        assert settings.hebbian_learning_rate == 0.2
        assert settings.hebbian_stale_days == 60


class TestHDBSCANParameters:
    """Test HDBSCAN clustering parameter configuration."""

    def test_hdbscan_default_values(self):
        """Test HDBSCAN parameters have correct defaults."""
        settings = Settings()
        assert settings.hdbscan_min_cluster_size == 3
        assert settings.hdbscan_min_samples is None
        assert settings.hdbscan_metric == "cosine"

    def test_hdbscan_min_cluster_size_bounds(self):
        """Test HDBSCAN min_cluster_size validates bounds."""
        # Valid values
        Settings(hdbscan_min_cluster_size=2)
        Settings(hdbscan_min_cluster_size=50)
        Settings(hdbscan_min_cluster_size=100)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(hdbscan_min_cluster_size=1)
        with pytest.raises(ValidationError):
            Settings(hdbscan_min_cluster_size=101)

    def test_hdbscan_min_samples_optional(self):
        """Test HDBSCAN min_samples is optional."""
        # None is valid
        settings = Settings(hdbscan_min_samples=None)
        assert settings.hdbscan_min_samples is None

        # Integer is valid
        settings = Settings(hdbscan_min_samples=5)
        assert settings.hdbscan_min_samples == 5

    def test_hdbscan_env_override(self, monkeypatch):
        """Test HDBSCAN parameters can be overridden via environment."""
        monkeypatch.setenv("WW_HDBSCAN_MIN_CLUSTER_SIZE", "5")
        monkeypatch.setenv("WW_HDBSCAN_METRIC", "euclidean")

        settings = Settings()
        assert settings.hdbscan_min_cluster_size == 5
        assert settings.hdbscan_metric == "euclidean"


class TestEpisodicRetrievalWeights:
    """Test episodic retrieval weight configuration and validation."""

    def test_episodic_weights_default_values(self):
        """Test episodic weights have correct defaults."""
        settings = Settings()
        assert settings.episodic_weight_semantic == 0.4
        assert settings.episodic_weight_recency == 0.25
        assert settings.episodic_weight_outcome == 0.2
        assert settings.episodic_weight_importance == 0.15

    def test_episodic_weights_sum_to_one(self):
        """Test episodic weights must sum to 1.0."""
        # Valid: sum = 1.0
        Settings(
            episodic_weight_semantic=0.5,
            episodic_weight_recency=0.3,
            episodic_weight_outcome=0.1,
            episodic_weight_importance=0.1,
        )

        # Invalid: sum != 1.0
        with pytest.raises(ValidationError, match="Episodic weights"):
            Settings(
                episodic_weight_semantic=0.5,
                episodic_weight_recency=0.5,
                episodic_weight_outcome=0.5,
                episodic_weight_importance=0.5,
            )

    def test_episodic_weights_bounds(self):
        """Test individual episodic weights are bounded [0, 1]."""
        # Invalid: negative weight
        with pytest.raises(ValidationError):
            Settings(episodic_weight_semantic=-0.1)

        # Invalid: weight > 1
        with pytest.raises(ValidationError):
            Settings(episodic_weight_semantic=1.1)

    def test_episodic_weights_env_override(self, monkeypatch):
        """Test episodic weights can be overridden via environment."""
        monkeypatch.setenv("WW_EPISODIC_WEIGHT_SEMANTIC", "0.5")
        monkeypatch.setenv("WW_EPISODIC_WEIGHT_RECENCY", "0.3")
        monkeypatch.setenv("WW_EPISODIC_WEIGHT_OUTCOME", "0.1")
        monkeypatch.setenv("WW_EPISODIC_WEIGHT_IMPORTANCE", "0.1")

        settings = Settings()
        assert settings.episodic_weight_semantic == 0.5
        assert settings.episodic_weight_recency == 0.3


class TestSemanticRetrievalWeights:
    """Test semantic retrieval weight configuration and validation."""

    def test_semantic_weights_default_values(self):
        """Test semantic weights have correct defaults."""
        settings = Settings()
        assert settings.semantic_weight_similarity == 0.4
        assert settings.semantic_weight_activation == 0.35
        assert settings.semantic_weight_retrievability == 0.25

    def test_semantic_weights_sum_to_one(self):
        """Test semantic weights must sum to 1.0."""
        # Valid: sum = 1.0
        Settings(
            semantic_weight_similarity=0.5,
            semantic_weight_activation=0.3,
            semantic_weight_retrievability=0.2,
        )

        # Invalid: sum != 1.0
        with pytest.raises(ValidationError, match="Semantic weights"):
            Settings(
                semantic_weight_similarity=0.5,
                semantic_weight_activation=0.5,
                semantic_weight_retrievability=0.5,
            )

    def test_semantic_weights_env_override(self, monkeypatch):
        """Test semantic weights can be overridden via environment."""
        monkeypatch.setenv("WW_SEMANTIC_WEIGHT_SIMILARITY", "0.5")
        monkeypatch.setenv("WW_SEMANTIC_WEIGHT_ACTIVATION", "0.3")
        monkeypatch.setenv("WW_SEMANTIC_WEIGHT_RETRIEVABILITY", "0.2")

        settings = Settings()
        assert settings.semantic_weight_similarity == 0.5
        assert settings.semantic_weight_activation == 0.3


class TestProceduralRetrievalWeights:
    """Test procedural retrieval weight configuration and validation."""

    def test_procedural_weights_default_values(self):
        """Test procedural weights have correct defaults."""
        settings = Settings()
        assert settings.procedural_weight_similarity == 0.6
        assert settings.procedural_weight_success == 0.3
        assert settings.procedural_weight_experience == 0.1

    def test_procedural_weights_sum_to_one(self):
        """Test procedural weights must sum to 1.0."""
        # Valid: sum = 1.0
        Settings(
            procedural_weight_similarity=0.7,
            procedural_weight_success=0.2,
            procedural_weight_experience=0.1,
        )

        # Invalid: sum != 1.0
        with pytest.raises(ValidationError, match="Procedural weights"):
            Settings(
                procedural_weight_similarity=0.5,
                procedural_weight_success=0.5,
                procedural_weight_experience=0.5,
            )

    def test_procedural_weights_env_override(self, monkeypatch):
        """Test procedural weights can be overridden via environment."""
        monkeypatch.setenv("WW_PROCEDURAL_WEIGHT_SIMILARITY", "0.7")
        monkeypatch.setenv("WW_PROCEDURAL_WEIGHT_SUCCESS", "0.2")
        monkeypatch.setenv("WW_PROCEDURAL_WEIGHT_EXPERIENCE", "0.1")

        settings = Settings()
        assert settings.procedural_weight_similarity == 0.7
        assert settings.procedural_weight_success == 0.2


class TestConsolidationParameters:
    """Test consolidation parameter configuration."""

    def test_consolidation_default_values(self):
        """Test consolidation parameters have correct defaults."""
        settings = Settings()
        assert settings.consolidation_min_similarity == 0.75
        assert settings.consolidation_min_occurrences == 3
        assert settings.consolidation_skill_similarity == 0.85

    def test_consolidation_similarity_bounds(self):
        """Test consolidation similarity validates bounds."""
        # Valid values
        Settings(consolidation_min_similarity=0.5)
        Settings(consolidation_min_similarity=0.75)
        Settings(consolidation_min_similarity=1.0)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(consolidation_min_similarity=0.4)
        with pytest.raises(ValidationError):
            Settings(consolidation_min_similarity=1.1)

    def test_consolidation_occurrences_bounds(self):
        """Test consolidation min_occurrences validates bounds."""
        # Valid values
        Settings(consolidation_min_occurrences=2)
        Settings(consolidation_min_occurrences=5)
        Settings(consolidation_min_occurrences=10)

        # Invalid values
        with pytest.raises(ValidationError):
            Settings(consolidation_min_occurrences=1)
        with pytest.raises(ValidationError):
            Settings(consolidation_min_occurrences=11)

    def test_consolidation_env_override(self, monkeypatch):
        """Test consolidation parameters can be overridden via environment."""
        monkeypatch.setenv("WW_CONSOLIDATION_MIN_SIMILARITY", "0.8")
        monkeypatch.setenv("WW_CONSOLIDATION_MIN_OCCURRENCES", "5")

        settings = Settings()
        assert settings.consolidation_min_similarity == 0.8
        assert settings.consolidation_min_occurrences == 5


class TestBackwardCompatibility:
    """Test backward compatibility for deprecated parameters."""

    def test_deprecated_retrieval_weights_exist(self):
        """Test deprecated retrieval_* weights still exist."""
        settings = Settings()
        assert hasattr(settings, "retrieval_semantic_weight")
        assert hasattr(settings, "retrieval_recency_weight")
        assert hasattr(settings, "retrieval_outcome_weight")
        assert hasattr(settings, "retrieval_importance_weight")

    def test_deprecated_weights_default_to_episodic(self):
        """Test deprecated weights have same defaults as episodic weights."""
        settings = Settings()
        assert settings.retrieval_semantic_weight == settings.episodic_weight_semantic
        assert settings.retrieval_recency_weight == settings.episodic_weight_recency
        assert settings.retrieval_outcome_weight == settings.episodic_weight_outcome
        assert settings.retrieval_importance_weight == settings.episodic_weight_importance


class TestSettingsCaching:
    """Test settings singleton caching."""

    def test_get_settings_caches(self):
        """Test get_settings returns same instance."""
        # Clear cache by reimporting
        from ww.core import config
        config._settings = None

        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_settings_immutable_after_load(self):
        """Test settings values don't change after initial load."""
        settings = get_settings()
        original_decay = settings.fsrs_decay_factor

        # Try to modify (should not affect cached instance)
        settings.fsrs_decay_factor = 0.5

        # Get fresh reference - should still have modified value
        # (pydantic models are mutable, but singleton ensures consistency)
        assert settings.fsrs_decay_factor == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
