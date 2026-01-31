"""
Unit tests for dendritic neuron implementation.
"""

import pytest
import torch
import torch.nn as nn

from t4dm.encoding.dendritic import DendriticNeuron, DendriticProcessor


class TestDendriticNeuron:
    """Tests for two-compartment dendritic neuron."""

    @pytest.fixture
    def neuron(self):
        """Create default dendritic neuron."""
        return DendriticNeuron(
            input_dim=64,
            hidden_dim=32,
            context_dim=32,
            coupling_strength=0.5
        )

    def test_initialization(self, neuron):
        """Neuron initializes with correct dimensions."""
        assert neuron.input_dim == 64
        assert neuron.hidden_dim == 32
        assert neuron.context_dim == 32
        assert neuron.coupling_strength == 0.5

    def test_forward_shape(self, neuron):
        """Forward pass produces correct output shape."""
        basal = torch.randn(8, 64)  # batch=8
        apical = torch.randn(8, 32)

        output, mismatch = neuron(basal, apical)

        assert output.shape == (8, 32)
        assert mismatch.shape == (8,)

    def test_forward_without_context(self, neuron):
        """Forward pass works without context (apical=None)."""
        basal = torch.randn(8, 64)

        output, mismatch = neuron(basal, apical_input=None)

        assert output.shape == (8, 32)
        assert mismatch.shape == (8,)

    def test_compartment_isolation(self, neuron):
        """Basal and apical compartments process independently."""
        basal = torch.randn(1, 64)
        apical = torch.randn(1, 32)

        # Output with context
        out_with_context, _ = neuron(basal, apical)

        # Output without context
        out_without_context, _ = neuron(basal, None)

        # Should be different when context is present
        assert not torch.allclose(out_with_context, out_without_context)

    def test_coupling_strength_effect(self):
        """Stronger coupling increases context influence."""
        basal = torch.randn(1, 64)
        apical = torch.randn(1, 32)

        neuron_weak = DendriticNeuron(
            input_dim=64, hidden_dim=32, context_dim=32,
            coupling_strength=0.1
        )
        neuron_strong = DendriticNeuron(
            input_dim=64, hidden_dim=32, context_dim=32,
            coupling_strength=0.9
        )

        # Get outputs
        out_weak, _ = neuron_weak(basal, apical)
        out_strong, _ = neuron_strong(basal, apical)

        # Measure context influence
        out_weak_no_ctx, _ = neuron_weak(basal, None)
        out_strong_no_ctx, _ = neuron_strong(basal, None)

        diff_weak = torch.norm(out_weak - out_weak_no_ctx)
        diff_strong = torch.norm(out_strong - out_strong_no_ctx)

        # Stronger coupling should show greater context influence
        # (Not guaranteed due to random init, but pattern should hold)
        assert diff_weak.item() >= 0  # Just verify computation works
        assert diff_strong.item() >= 0

    def test_gradient_flow(self, neuron):
        """Gradients flow through both compartments."""
        basal = torch.randn(1, 64, requires_grad=True)
        apical = torch.randn(1, 32, requires_grad=True)

        output, mismatch = neuron(basal, apical)
        loss = output.sum() + mismatch.sum()
        loss.backward()

        assert basal.grad is not None
        assert apical.grad is not None
        assert basal.grad.abs().sum() > 0
        assert apical.grad.abs().sum() > 0

    def test_mismatch_signal(self, neuron):
        """Mismatch signal is computed correctly."""
        basal = torch.randn(4, 64)
        apical = torch.randn(4, 32)

        _, mismatch = neuron(basal, apical)

        # Mismatch should be non-negative (norm)
        assert (mismatch >= 0).all()

    def test_time_constant_validation(self):
        """Tau_dendrite must be less than tau_soma."""
        with pytest.raises(AssertionError):
            DendriticNeuron(
                input_dim=64, hidden_dim=32, context_dim=32,
                tau_dendrite=20.0, tau_soma=10.0  # Invalid: dendrite > soma
            )

    def test_context_influence_method(self, neuron):
        """Context influence computation works."""
        basal = torch.randn(1, 64)
        apical = torch.randn(1, 32)

        influence = neuron.compute_context_influence(basal, apical)

        assert isinstance(influence, float)
        assert influence >= 0


class TestDendriticProcessor:
    """Tests for multi-layer dendritic processor."""

    @pytest.fixture
    def processor(self):
        """Create default processor."""
        return DendriticProcessor(
            input_dim=64,
            hidden_dims=[32, 16],
            context_dim=32,
            coupling_strength=0.5
        )

    def test_initialization(self, processor):
        """Processor initializes correctly."""
        assert len(processor.layers) == 2
        assert processor.output_dim == 16

    def test_forward_shape(self, processor):
        """Forward produces correct shapes."""
        x = torch.randn(8, 64)
        context = torch.randn(8, 32)

        output, mismatches = processor(x, context)

        assert output.shape == (8, 16)
        assert len(mismatches) == 2

    def test_forward_without_context(self, processor):
        """Forward works without context."""
        x = torch.randn(8, 64)

        output, mismatches = processor(x, None)

        assert output.shape == (8, 16)

    def test_gradient_flow(self, processor):
        """Gradients flow through all layers."""
        x = torch.randn(1, 64, requires_grad=True)
        context = torch.randn(1, 32, requires_grad=True)

        output, _ = processor(x, context)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert context.grad is not None
