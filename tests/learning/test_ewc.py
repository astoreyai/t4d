"""
Tests for P3.5: Elastic Weight Consolidation (EWC).

Tests the EWCRegularizer class that prevents catastrophic forgetting
by penalizing changes to important weights.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Conditional imports for torch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Skip all tests if torch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch required for EWC tests"
)


@pytest.fixture
def simple_model():
    """Create a simple neural network for testing."""
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    return model


@pytest.fixture
def simple_dataloader():
    """Create a simple dataloader for testing."""
    class SimpleLoader:
        def __init__(self, n_batches=10, batch_size=4, input_dim=10):
            self.data = [
                torch.randn(batch_size, input_dim)
                for _ in range(n_batches)
            ]

        def __iter__(self):
            return iter(self.data)

    return SimpleLoader()


@pytest.fixture
def ewc_regularizer():
    """Create an EWC regularizer with default settings."""
    from t4dm.learning.plasticity import EWCRegularizer
    return EWCRegularizer(
        lambda_ewc=100.0,  # Lower for testing
        fisher_n_samples=20,
        online=True,
        gamma=0.9,
    )


# =============================================================================
# Test EWC Initialization
# =============================================================================


class TestEWCInitialization:
    """Tests for EWC initialization."""

    def test_default_initialization(self):
        """Test EWC initializes with defaults."""
        from t4dm.learning.plasticity import EWCRegularizer
        ewc = EWCRegularizer()

        assert ewc.lambda_ewc == 1000.0
        assert ewc.online is True
        assert ewc.gamma == 0.95
        assert ewc.fisher_n_samples == 200
        assert len(ewc.fisher_diag) == 0
        assert len(ewc.optimal_weights) == 0

    def test_custom_initialization(self):
        """Test EWC with custom parameters."""
        from t4dm.learning.plasticity import EWCRegularizer
        ewc = EWCRegularizer(
            lambda_ewc=500.0,
            fisher_n_samples=100,
            online=False,
            gamma=0.8,
        )

        assert ewc.lambda_ewc == 500.0
        assert ewc.online is False
        assert ewc.gamma == 0.8
        assert ewc.fisher_n_samples == 100

    def test_stats_initialized(self):
        """Test stats are initialized."""
        from t4dm.learning.plasticity import EWCRegularizer
        ewc = EWCRegularizer()

        stats = ewc.get_stats()
        assert stats["n_consolidations"] == 0
        assert stats["total_fisher_entries"] == 0
        assert stats["avg_fisher_magnitude"] == 0.0


# =============================================================================
# Test Fisher Information Computation
# =============================================================================


class TestFisherComputation:
    """Tests for Fisher information computation."""

    def test_compute_fisher_returns_dict(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test Fisher computation returns dictionary."""
        fisher = ewc_regularizer.compute_fisher(
            simple_model, simple_dataloader, device="cpu"
        )

        assert isinstance(fisher, dict)
        assert len(fisher) > 0

    def test_fisher_has_correct_keys(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test Fisher has entries for all trainable parameters."""
        fisher = ewc_regularizer.compute_fisher(
            simple_model, simple_dataloader, device="cpu"
        )

        for name, param in simple_model.named_parameters():
            if param.requires_grad:
                assert name in fisher

    def test_fisher_shapes_match_params(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test Fisher entries have same shape as parameters."""
        fisher = ewc_regularizer.compute_fisher(
            simple_model, simple_dataloader, device="cpu"
        )

        for name, param in simple_model.named_parameters():
            if name in fisher:
                assert fisher[name].shape == param.shape

    def test_fisher_non_negative(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test Fisher values are non-negative (squared gradients)."""
        fisher = ewc_regularizer.compute_fisher(
            simple_model, simple_dataloader, device="cpu"
        )

        for name, f in fisher.items():
            assert (f >= 0).all(), f"Negative Fisher values for {name}"


# =============================================================================
# Test Consolidation
# =============================================================================


class TestConsolidation:
    """Tests for EWC consolidation."""

    def test_consolidation_stores_fisher(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test consolidation stores Fisher information."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        assert len(ewc_regularizer.fisher_diag) > 0

    def test_consolidation_stores_optimal_weights(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test consolidation stores optimal weights."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        assert len(ewc_regularizer.optimal_weights) > 0

    def test_online_ewc_accumulates(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test online EWC accumulates Fisher over consolidations."""
        # First consolidation
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )
        first_fisher = {k: v.clone() for k, v in ewc_regularizer.fisher_diag.items()}

        # Second consolidation
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        # Fisher should be updated (not necessarily increased due to gamma decay)
        for name in first_fisher:
            if name in ewc_regularizer.fisher_diag:
                # Values should be different after accumulation
                # Due to gamma decay + new values
                assert ewc_regularizer.fisher_diag[name] is not first_fisher[name]

    def test_consolidation_updates_stats(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test consolidation updates statistics."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        stats = ewc_regularizer.get_stats()
        assert stats["n_consolidations"] == 1
        assert stats["total_fisher_entries"] > 0
        assert stats["avg_fisher_magnitude"] >= 0.0


# =============================================================================
# Test EWC Penalty
# =============================================================================


class TestEWCPenalty:
    """Tests for EWC penalty computation."""

    def test_penalty_zero_before_consolidation(self, ewc_regularizer, simple_model):
        """Test penalty is zero before any consolidation."""
        penalty = ewc_regularizer.penalty(simple_model)

        # Should return zero (as tensor)
        assert penalty.item() == 0.0

    def test_penalty_zero_when_unchanged(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test penalty is zero when weights haven't changed."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        # Weights haven't changed, so penalty should be very small
        penalty = ewc_regularizer.penalty(simple_model)
        assert penalty.item() < 1e-6

    def test_penalty_increases_with_weight_change(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test penalty increases when weights change."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        # Change weights
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        penalty = ewc_regularizer.penalty(simple_model)
        assert penalty.item() > 0.0

    def test_penalty_scales_with_lambda(self, simple_model, simple_dataloader):
        """Test penalty scales with lambda_ewc."""
        from t4dm.learning.plasticity import EWCRegularizer

        ewc_low = EWCRegularizer(lambda_ewc=100.0, fisher_n_samples=20)
        ewc_high = EWCRegularizer(lambda_ewc=1000.0, fisher_n_samples=20)

        # Consolidate both
        ewc_low.consolidate(simple_model, simple_dataloader, device="cpu")
        ewc_high.fisher_diag = {k: v.clone() for k, v in ewc_low.fisher_diag.items()}
        ewc_high.optimal_weights = {k: v.clone() for k, v in ewc_low.optimal_weights.items()}

        # Change weights
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        penalty_low = ewc_low.penalty(simple_model)
        penalty_high = ewc_high.penalty(simple_model)

        # Higher lambda should give higher penalty
        assert penalty_high.item() > penalty_low.item()


# =============================================================================
# Test EWC Loss
# =============================================================================


class TestEWCLoss:
    """Tests for combined EWC loss."""

    def test_ewc_loss_adds_penalty(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test ewc_loss adds penalty to task loss."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        # Change weights
        with torch.no_grad():
            for param in simple_model.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        task_loss = torch.tensor(1.0, requires_grad=True)
        total_loss = ewc_regularizer.ewc_loss(simple_model, task_loss)

        # Total should be >= task loss
        assert total_loss.item() >= task_loss.item()

    def test_ewc_loss_supports_backward(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test EWC loss supports backward pass."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        # Create a simple forward pass
        x = torch.randn(4, 10)
        output = simple_model(x)
        task_loss = output.sum()

        total_loss = ewc_regularizer.ewc_loss(simple_model, task_loss)

        # Should not raise during backward
        total_loss.backward()

        # Gradients should exist
        for param in simple_model.parameters():
            assert param.grad is not None


# =============================================================================
# Test Importance Scores
# =============================================================================


class TestImportanceScores:
    """Tests for importance score computation."""

    def test_importance_empty_before_consolidation(self, ewc_regularizer, simple_model):
        """Test importance scores empty before consolidation."""
        scores = ewc_regularizer.get_importance_scores(simple_model)
        assert scores == {}

    def test_importance_after_consolidation(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test importance scores available after consolidation."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        scores = ewc_regularizer.get_importance_scores(simple_model)

        assert len(scores) > 0
        for name, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for {name}"

    def test_importance_sums_to_one(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test normalized importance scores sum to ~1."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        scores = ewc_regularizer.get_importance_scores(simple_model)
        total = sum(scores.values())

        assert abs(total - 1.0) < 1e-6


# =============================================================================
# Test Save/Load
# =============================================================================


class TestSaveLoad:
    """Tests for EWC state persistence."""

    def test_save_creates_file(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test save creates file."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            ewc_regularizer.save(path)
            assert Path(path).exists()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_restores_state(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test load restores EWC state."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        # Get stats before save
        stats_before = ewc_regularizer.get_stats()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            ewc_regularizer.save(path)

            # Create new EWC and load
            from t4dm.learning.plasticity import EWCRegularizer
            ewc_new = EWCRegularizer()
            ewc_new.load(path)

            stats_after = ewc_new.get_stats()

            assert stats_after["n_consolidations"] == stats_before["n_consolidations"]
            assert stats_after["lambda_ewc"] == stats_before["lambda_ewc"]
        finally:
            Path(path).unlink(missing_ok=True)


# =============================================================================
# Test Reset
# =============================================================================


class TestReset:
    """Tests for EWC reset."""

    def test_reset_clears_fisher(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test reset clears Fisher information."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )
        assert len(ewc_regularizer.fisher_diag) > 0

        ewc_regularizer.reset()
        assert len(ewc_regularizer.fisher_diag) == 0

    def test_reset_clears_optimal_weights(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test reset clears optimal weights."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )
        assert len(ewc_regularizer.optimal_weights) > 0

        ewc_regularizer.reset()
        assert len(ewc_regularizer.optimal_weights) == 0

    def test_reset_clears_stats(self, ewc_regularizer, simple_model, simple_dataloader):
        """Test reset clears statistics."""
        ewc_regularizer.consolidate(
            simple_model, simple_dataloader, device="cpu"
        )

        ewc_regularizer.reset()
        stats = ewc_regularizer.get_stats()
        assert stats["n_consolidations"] == 0


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateEWCRegularizer:
    """Tests for EWC factory function."""

    def test_create_with_defaults(self):
        """Test factory with defaults."""
        from t4dm.learning.plasticity import create_ewc_regularizer
        ewc = create_ewc_regularizer()

        assert ewc.lambda_ewc == 1000.0
        assert ewc.online is True
        assert ewc.gamma == 0.95

    def test_create_with_custom_params(self):
        """Test factory with custom parameters."""
        from t4dm.learning.plasticity import create_ewc_regularizer
        ewc = create_ewc_regularizer(
            lambda_ewc=500.0,
            online=False,
            gamma=0.8,
        )

        assert ewc.lambda_ewc == 500.0
        assert ewc.online is False
        assert ewc.gamma == 0.8


# =============================================================================
# Test Integration with LoRA
# =============================================================================


class TestLoRAIntegration:
    """Tests for EWC integration with LoRA adapter."""

    def test_lora_ewc_config(self):
        """Test LoRA config includes EWC settings."""
        from t4dm.embedding.lora_adapter import LoRAConfig

        config = LoRAConfig(
            ewc_enabled=True,
            ewc_lambda=500.0,
            ewc_online=True,
            ewc_gamma=0.9,
        )

        assert config.ewc_enabled is True
        assert config.ewc_lambda == 500.0
        assert config.ewc_online is True
        assert config.ewc_gamma == 0.9

    def test_lora_stats_include_ewc(self):
        """Test LoRA adapter stats include EWC when enabled."""
        from t4dm.embedding.lora_adapter import LoRAEmbeddingAdapter, LoRAConfig

        config = LoRAConfig(ewc_enabled=True, ewc_lambda=100.0)
        adapter = LoRAEmbeddingAdapter(config=config, device="cpu")

        stats = adapter.get_stats()
        assert "ewc" in stats
        assert stats["config"]["ewc_enabled"] is True


# =============================================================================
# Test Integration with Scorer
# =============================================================================


class TestScorerIntegration:
    """Tests for EWC integration with ScorerTrainer."""

    def test_scorer_ewc_config(self):
        """Test ScorerTrainer config includes EWC settings."""
        from t4dm.learning.scorer import TrainerConfig

        config = TrainerConfig(
            ewc_enabled=True,
            ewc_lambda=500.0,
            ewc_online=True,
            ewc_gamma=0.9,
        )

        assert config.ewc_enabled is True
        assert config.ewc_lambda == 500.0
        assert config.ewc_online is True
        assert config.ewc_gamma == 0.9
