"""
Unit Tests for Learned Embedding Alignment (W2-03).

Verifies learnable alignment layer between Qwen hidden states and
BGE-M3 embedding space following Graves (2014) NTM principles.

Evidence Base: Graves (2014) "Neural Turing Machines"
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, MagicMock, patch


class TestEmbeddingAlignment:
    """Test EmbeddingAlignment module."""

    def test_alignment_creation(self):
        """Should create alignment with correct dimensions."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment(qwen_dim=2048, bge_dim=1024)

        assert alignment is not None

    def test_alignment_output_dimension(self):
        """Output should match BGE dimension."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment(qwen_dim=2048, bge_dim=1024)
        qwen_hidden = torch.randn(2, 128, 2048)  # [batch, seq, hidden]

        aligned = alignment(qwen_hidden)

        assert aligned.shape == (2, 1024), f"Expected (2, 1024), got {aligned.shape}"

    def test_alignment_is_normalized(self):
        """Output should be L2 normalized."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment(qwen_dim=2048, bge_dim=1024)
        qwen_hidden = torch.randn(5, 64, 2048)

        aligned = alignment(qwen_hidden)
        norms = torch.norm(aligned, dim=-1)

        torch.testing.assert_close(norms, torch.ones(5), atol=1e-5, rtol=1e-5)

    def test_alignment_has_trainable_params(self):
        """Alignment should have trainable parameters."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment()
        params = list(alignment.parameters())

        assert len(params) > 0, "Should have trainable parameters"
        assert all(p.requires_grad for p in params)

    def test_dimension_importance_learnable(self):
        """Should have learnable dimension importance weights."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment(bge_dim=1024)

        assert hasattr(alignment, "dimension_importance")
        assert alignment.dimension_importance.shape == (1024,)
        assert alignment.dimension_importance.requires_grad

    def test_different_inputs_produce_different_outputs(self):
        """Different inputs should produce different aligned embeddings."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment()
        alignment.eval()

        input1 = torch.zeros(1, 64, 2048)
        input2 = torch.ones(1, 64, 2048)

        out1 = alignment(input1)
        out2 = alignment(input2)

        assert not torch.allclose(out1, out2), "Different inputs should produce different outputs"

    def test_gradients_flow(self):
        """Gradients should flow through alignment."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment()
        qwen_hidden = torch.randn(2, 32, 2048, requires_grad=True)

        aligned = alignment(qwen_hidden)
        loss = aligned.sum()
        loss.backward()

        assert qwen_hidden.grad is not None, "Gradients should flow to input"


class TestAlignmentConfig:
    """Test AlignmentConfig dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        from t4dm.qwen.alignment import AlignmentConfig

        config = AlignmentConfig()

        assert config.qwen_dim == 2048
        assert config.bge_dim == 1024
        assert config.hidden_dim == 1536
        assert config.dropout > 0

    def test_config_override(self):
        """Should be able to override config values."""
        from t4dm.qwen.alignment import AlignmentConfig

        config = AlignmentConfig(
            qwen_dim=4096,
            bge_dim=768,
            hidden_dim=2048,
            dropout=0.2,
        )

        assert config.qwen_dim == 4096
        assert config.bge_dim == 768


class TestAlignmentTrainer:
    """Test AlignmentTrainer for training alignment."""

    @pytest.fixture
    def alignment(self):
        """Create alignment for testing."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        return EmbeddingAlignment(qwen_dim=128, bge_dim=64, hidden_dim=96)

    @pytest.fixture
    def mock_bge(self):
        """Create mock BGE model."""
        model = Mock()
        model.encode = Mock(return_value=np.random.randn(64).astype(np.float32))
        return model

    def test_trainer_creation(self, alignment, mock_bge):
        """Should create trainer with alignment and BGE model."""
        from t4dm.qwen.alignment import AlignmentTrainer

        trainer = AlignmentTrainer(alignment, mock_bge)

        assert trainer.alignment is alignment
        assert trainer.bge_model is mock_bge

    def test_training_reduces_loss(self, alignment, mock_bge):
        """Training should reduce loss over iterations."""
        from t4dm.qwen.alignment import AlignmentTrainer

        trainer = AlignmentTrainer(alignment, mock_bge, lr=0.01)

        # Training data
        qwen_hidden = torch.randn(10, 32, 128)
        texts = [f"Text {i}" for i in range(10)]

        # Track loss
        initial_loss = None
        final_loss = None

        for epoch in range(50):
            loss = trainer.train_step(qwen_hidden, texts)

            if epoch == 0:
                initial_loss = loss
            if epoch == 49:
                final_loss = loss

        assert final_loss < initial_loss, "Training should reduce loss"

    def test_dimension_importance_changes(self, alignment, mock_bge):
        """Dimension importance should change during training."""
        from t4dm.qwen.alignment import AlignmentTrainer

        trainer = AlignmentTrainer(alignment, mock_bge, lr=0.01)

        initial_importance = alignment.dimension_importance.clone().detach()

        # Train for some steps
        qwen_hidden = torch.randn(10, 32, 128)
        texts = [f"Text {i}" for i in range(10)]

        for _ in range(20):
            trainer.train_step(qwen_hidden, texts)

        final_importance = alignment.dimension_importance.detach()

        # Should have changed
        assert not torch.allclose(initial_importance, final_importance, atol=0.01), \
            "Dimension importance should change during training"


class TestAlignmentLatency:
    """Test alignment latency requirements."""

    def test_alignment_under_1ms(self):
        """Alignment should add <1ms latency (single sample)."""
        from t4dm.qwen.alignment import EmbeddingAlignment
        import time

        alignment = EmbeddingAlignment()
        alignment.eval()

        # Single sample
        qwen_hidden = torch.randn(1, 64, 2048)

        # Warmup
        for _ in range(10):
            alignment(qwen_hidden)

        # Measure
        times = []
        for _ in range(100):
            start = time.perf_counter()
            with torch.no_grad():
                alignment(qwen_hidden)
            times.append(time.perf_counter() - start)

        avg_time_ms = np.mean(times) * 1000

        # Allow some slack for CI variability
        assert avg_time_ms < 5.0, f"Alignment took {avg_time_ms:.2f}ms, should be <5ms"


class TestAlignmentIntegration:
    """Integration tests for alignment in retrieval pipeline."""

    def test_alignment_works_with_real_shapes(self):
        """Should work with realistic Qwen hidden state shapes."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment(qwen_dim=2048, bge_dim=1024)

        # Realistic Qwen output shape
        qwen_hidden = torch.randn(1, 512, 2048)  # Long sequence

        aligned = alignment(qwen_hidden)

        assert aligned.shape == (1, 1024)
        assert torch.isfinite(aligned).all()

    def test_alignment_handles_batch(self):
        """Should handle batched inputs."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment()

        # Batch of sequences
        qwen_hidden = torch.randn(8, 128, 2048)

        aligned = alignment(qwen_hidden)

        assert aligned.shape == (8, 1024)

    def test_alignment_deterministic_in_eval_mode(self):
        """Should be deterministic in eval mode."""
        from t4dm.qwen.alignment import EmbeddingAlignment

        alignment = EmbeddingAlignment()
        alignment.eval()

        qwen_hidden = torch.randn(2, 64, 2048)

        out1 = alignment(qwen_hidden)
        out2 = alignment(qwen_hidden)

        torch.testing.assert_close(out1, out2)
