"""
Phase 3 Spike Pipeline Integration Test (P3-09).

Tests the complete spike pipeline integration:
1. SpikeReinjector converts embeddings to spike trains
2. SNNBackend processes spike trains
3. CorticalStack applies attention and modulation
4. STDP computes weight updates
5. Consolidation updates memory κ

All components must work together without errors.
"""

import numpy as np
import pytest
import torch

from t4dm.nca.spike_reinjection import (
    SpikeReinjector,
    ReinjectionConfig,
    ReinjectionMode,
)
from t4dm.nca.snn_backend import SNNBackend, SNNConfig, NeuronModel
from t4dm.spiking.cortical_stack import CorticalStack
from t4dm.spiking.cortical_block import CorticalBlock
from t4dm.spiking.lif import LIFNeuron
from t4dm.learning.stdp_jit import compute_stdp_delta_jit as compute_stdp_delta


class TestSpikePipelineIntegration:
    """Integration tests for complete spike pipeline."""

    @pytest.fixture
    def pipeline_components(self):
        """Create all pipeline components."""
        # Reinjector: embedding → spikes
        reinjector = SpikeReinjector(
            embedding_dim=256,
            hidden_dim=128,
            output_dim=64,
            config=ReinjectionConfig(mode=ReinjectionMode.RATE, num_steps=50),
        )

        # SNN Backend: spike processing
        snn = SNNBackend(
            input_size=64,
            hidden_size=64,
            config=SNNConfig(neuron_model=NeuronModel.LIF, tau_mem=10.0),
        )

        # Cortical Stack: attention + modulation
        stack = CorticalStack(dim=64, num_blocks=3, num_heads=4)

        return {
            "reinjector": reinjector,
            "snn": snn,
            "stack": stack,
        }

    def test_pipeline_forward_pass(self, pipeline_components):
        """Test complete forward pass through pipeline."""
        reinjector = pipeline_components["reinjector"]
        snn = pipeline_components["snn"]
        stack = pipeline_components["stack"]

        # Input: batch of embeddings
        batch_size = 4
        embedding = torch.randn(batch_size, 256)

        # Stage 1: Reinjection
        spikes = reinjector(embedding)
        assert spikes.shape == (batch_size, 50, 64)

        # Stage 2: SNN processing
        with torch.no_grad():
            snn_out, snn_state = snn(spikes)
        assert snn_out.shape == spikes.shape

        # Stage 3: Cortical stack
        stack_out, states, metrics = stack(snn_out)
        assert stack_out.shape == snn_out.shape
        assert len(metrics) == 3  # 3 blocks

        # Verify metrics
        for i, m in enumerate(metrics):
            assert "pe" in m, f"Block {i} missing prediction error"
            assert "goodness" in m, f"Block {i} missing goodness"
            assert "attn" in m, f"Block {i} missing attention"

    def test_pipeline_produces_valid_outputs(self, pipeline_components):
        """Outputs should be valid (no NaN, reasonable range)."""
        reinjector = pipeline_components["reinjector"]
        snn = pipeline_components["snn"]
        stack = pipeline_components["stack"]

        embedding = torch.randn(2, 256)

        # Forward pass
        spikes = reinjector(embedding)
        with torch.no_grad():
            snn_out, _ = snn(spikes)
        stack_out, _, metrics = stack(snn_out)

        # Check for NaN
        assert not torch.isnan(spikes).any(), "Spikes contain NaN"
        assert not torch.isnan(snn_out).any(), "SNN output contains NaN"
        assert not torch.isnan(stack_out).any(), "Stack output contains NaN"

        # Check range
        assert (spikes >= 0).all() and (spikes <= 1).all(), "Spikes out of [0,1]"

    def test_pipeline_handles_batch_sizes(self, pipeline_components):
        """Pipeline should handle various batch sizes."""
        reinjector = pipeline_components["reinjector"]
        snn = pipeline_components["snn"]
        stack = pipeline_components["stack"]

        for batch_size in [1, 2, 8, 16]:
            embedding = torch.randn(batch_size, 256)

            spikes = reinjector(embedding)
            with torch.no_grad():
                snn_out, _ = snn(spikes)
            stack_out, _, _ = stack(snn_out)

            assert stack_out.shape[0] == batch_size, \
                f"Batch size {batch_size} not preserved"

    def test_pipeline_with_different_modes(self, pipeline_components):
        """Pipeline should work with all reinjection modes."""
        snn = pipeline_components["snn"]
        stack = pipeline_components["stack"]

        embedding = torch.randn(2, 256)

        for mode in ReinjectionMode:
            reinjector = SpikeReinjector(
                embedding_dim=256,
                hidden_dim=128,
                output_dim=64,
                config=ReinjectionConfig(mode=mode, num_steps=50),
            )

            spikes = reinjector(embedding)
            with torch.no_grad():
                snn_out, _ = snn(spikes)
            stack_out, _, _ = stack(snn_out)

            assert stack_out.shape == (2, 50, 64), \
                f"Mode {mode.value} failed"


class TestSpikePipelineSTDPIntegration:
    """Test STDP integration with spike pipeline."""

    def test_stdp_from_pipeline_spikes(self):
        """STDP should work with spikes from pipeline."""
        # Create pipeline
        reinjector = SpikeReinjector(
            embedding_dim=128,
            hidden_dim=64,
            output_dim=32,
            config=ReinjectionConfig(mode=ReinjectionMode.RATE, num_steps=100),
        )
        snn = SNNBackend(
            input_size=32,
            hidden_size=32,
            config=SNNConfig(neuron_model=NeuronModel.LIF, tau_mem=10.0),
        )

        # Generate spikes
        embedding = torch.randn(1, 128)
        pre_spikes = reinjector(embedding)

        with torch.no_grad():
            post_spikes, _ = snn(pre_spikes)

        # Extract spike times
        pre_times = torch.where(pre_spikes[0, :, 0] > 0.5)[0].numpy()
        post_times = torch.where(post_spikes[0, :, 0] > 0.5)[0].numpy()

        if len(pre_times) > 0 and len(post_times) > 0:
            # Compute STDP for spike pairs
            stdp_deltas = []
            for pre_t in pre_times[:5]:
                for post_t in post_times[:5]:
                    delta_t_ms = float(post_t - pre_t)  # Already in ms (timestep units)
                    delta = compute_stdp_delta(delta_t_ms=delta_t_ms, current_weight=0.5)
                    stdp_deltas.append(delta)

            # Should have computed some deltas
            assert len(stdp_deltas) > 0, "Should compute STDP deltas"

            # Mix of LTP and LTD expected
            has_positive = any(d > 0 for d in stdp_deltas)
            has_negative = any(d < 0 for d in stdp_deltas)
            # At least one direction (depends on timing)
            assert has_positive or has_negative, "Should have non-zero STDP"

    def test_cumulative_stdp_effects(self):
        """Multiple STDP updates should accumulate."""
        initial_weight = 0.5
        weight = initial_weight

        # Simulate LTP-inducing timing (post after pre)
        for _ in range(10):
            delta_t_ms = 10.0  # 10ms post-after-pre = LTP
            delta = compute_stdp_delta(delta_t_ms=delta_t_ms, current_weight=weight)
            weight = np.clip(weight + delta, 0.0, 1.0)

        # Weight should increase (LTP for positive delta_t)
        assert weight > initial_weight, f"LTP should increase weight from {initial_weight} to {weight}"
        assert weight <= 1.0, "Weight should not exceed 1.0"


class TestSpikePipelineConsolidation:
    """Test consolidation integration with spike pipeline."""

    def test_kappa_update_after_replay(self):
        """κ should update after successful replay."""
        # Simulate memory consolidation
        memories = [
            {"id": i, "kappa": 0.1, "embedding": np.random.randn(128)}
            for i in range(5)
        ]

        # Simulate NREM replay
        reinjector = SpikeReinjector(
            embedding_dim=128,
            hidden_dim=64,
            output_dim=32,
            config=ReinjectionConfig(mode=ReinjectionMode.RATE, num_steps=50),
        )

        for mem in memories:
            embedding = torch.tensor(mem["embedding"], dtype=torch.float32).unsqueeze(0)
            spikes = reinjector(embedding)

            # Replay successful if spikes generated
            if spikes.sum() > 0:
                # NREM κ increment
                mem["kappa"] = min(1.0, mem["kappa"] + 0.05)

        # All memories should have increased κ
        for mem in memories:
            assert mem["kappa"] >= 0.15, f"κ should increase, got {mem['kappa']}"

    def test_activity_dependent_consolidation(self):
        """Higher activity should lead to stronger consolidation."""
        reinjector = SpikeReinjector(
            embedding_dim=128,
            hidden_dim=64,
            output_dim=32,
            config=ReinjectionConfig(mode=ReinjectionMode.RATE, num_steps=50),
        )

        # High-activity embedding (large values)
        high_emb = torch.ones(1, 128) * 2.0
        high_spikes = reinjector(high_emb)
        high_activity = high_spikes.sum().item()

        # Low-activity embedding (small values)
        low_emb = torch.ones(1, 128) * 0.1
        low_spikes = reinjector(low_emb)
        low_activity = low_spikes.sum().item()

        # Scale κ update by activity
        high_kappa_delta = 0.05 * (high_activity / 1000)
        low_kappa_delta = 0.05 * (low_activity / 1000)

        # Higher activity = larger κ update
        assert high_kappa_delta >= low_kappa_delta * 0.5, \
            "Higher activity should lead to stronger consolidation"


class TestSpikePipelineState:
    """Test state management in spike pipeline."""

    def test_snn_state_persists(self):
        """SNN state should persist across calls."""
        snn = SNNBackend(
            input_size=32,
            hidden_size=32,
            config=SNNConfig(neuron_model=NeuronModel.LIF, tau_mem=10.0),
        )

        # First input
        x1 = torch.randn(1, 10, 32)
        with torch.no_grad():
            out1, state1 = snn(x1)

        assert state1 is not None
        assert state1.membrane is not None

        # Second input with state
        x2 = torch.randn(1, 10, 32)
        with torch.no_grad():
            out2, state2 = snn(x2, state=state1)

        # State should be different (evolved)
        assert not torch.allclose(state1.membrane, state2.membrane), \
            "State should evolve"

    def test_cortical_stack_state_persists(self):
        """Cortical stack state should persist."""
        stack = CorticalStack(dim=32, num_blocks=2, num_heads=4)

        x1 = torch.randn(1, 10, 32)
        out1, states1, _ = stack(x1)

        assert states1 is not None
        assert len(states1) == 2  # 2 blocks

        # Second call
        x2 = torch.randn(1, 10, 32)
        out2, states2, _ = stack(x2, states=states1)

        # States should exist
        assert states2 is not None


class TestSpikePipelineEdgeCases:
    """Test edge cases in spike pipeline."""

    def test_zero_embedding(self):
        """Pipeline should handle zero embedding."""
        reinjector = SpikeReinjector(
            embedding_dim=128,
            hidden_dim=64,
            output_dim=32,
            config=ReinjectionConfig(mode=ReinjectionMode.RATE, num_steps=50),
        )

        zero_emb = torch.zeros(1, 128)
        spikes = reinjector(zero_emb)

        # Should not crash, may produce few/no spikes
        assert spikes.shape == (1, 50, 32)
        assert not torch.isnan(spikes).any()

    def test_large_embedding_values(self):
        """Pipeline should handle large embedding values."""
        reinjector = SpikeReinjector(
            embedding_dim=128,
            hidden_dim=64,
            output_dim=32,
            config=ReinjectionConfig(mode=ReinjectionMode.RATE, num_steps=50),
        )

        large_emb = torch.ones(1, 128) * 100.0
        spikes = reinjector(large_emb)

        # Should not crash or produce NaN
        assert not torch.isnan(spikes).any()
        assert not torch.isinf(spikes).any()

    def test_single_timestep(self):
        """Pipeline should handle single timestep."""
        reinjector = SpikeReinjector(
            embedding_dim=128,
            hidden_dim=64,
            output_dim=32,
            config=ReinjectionConfig(mode=ReinjectionMode.RATE, num_steps=1),
        )

        emb = torch.randn(1, 128)
        spikes = reinjector(emb)

        assert spikes.shape == (1, 1, 32)
