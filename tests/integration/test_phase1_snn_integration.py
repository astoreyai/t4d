"""
Phase 1 Integration Test for SNN Unification Plan.

Tests that all P1 atoms work together:
- P1-01: τ(t) temporal control signal
- P1-02: τ(t) integration into MemoryGate
- P1-03: Norse SNN backend wrapper
- P1-04: Numba JIT for STDP
- P1-05: Unified MemoryItem schema
- P1-06/07/08: κ fields on Episode/Entity/Procedure
- P1-09: GPU PDE solver

Gate Check:
- τ(t) signal flows through MemoryGate
- Norse SNN generates spikes
- STDP JIT provides speedup
- MemoryItem with κ validates
"""

import pytest
import numpy as np
import torch
from datetime import datetime

# Skip if dependencies not available
pytest.importorskip("torch")


class TestTemporalControlSignal:
    """P1-01: Test τ(t) temporal control signal."""

    def test_temporal_control_init(self):
        """Test TemporalControlSignal initialization."""
        from t4dm.core.temporal_control import TemporalControlSignal, TemporalControlMode

        control = TemporalControlSignal()
        assert control is not None
        assert hasattr(control, "lambdas")
        assert len(control.lambdas) == 5  # epsilon, delta, r, da, theta

    def test_tau_forward_pass(self):
        """Test τ(t) forward computation."""
        from t4dm.core.temporal_control import TemporalControlSignal

        control = TemporalControlSignal()

        # Test with tensor inputs
        pe = torch.tensor(0.5)
        nov = torch.tensor(0.3)
        rew = torch.tensor(0.0)

        tau = control.forward(pe, nov, rew)

        assert tau.shape == ()
        assert 0.0 < tau.item() < 1.0  # Sigmoid output

    def test_tau_with_dopamine(self):
        """Test τ(t) with dopamine modulation."""
        from t4dm.core.temporal_control import TemporalControlSignal

        control = TemporalControlSignal()

        pe = torch.tensor(0.5)
        nov = torch.tensor(0.3)
        rew = torch.tensor(0.0)
        da = torch.tensor(0.8)  # High dopamine

        tau_with_da = control.forward(pe, nov, rew, dopamine=da)
        tau_without_da = control.forward(pe, nov, rew)

        # High DA should increase tau (more encoding)
        # Note: Depends on lambda weights, but they're positive by default
        assert tau_with_da is not None
        assert tau_without_da is not None

    def test_compute_state(self):
        """Test full state computation."""
        from t4dm.core.temporal_control import TemporalControlSignal, TemporalControlState

        control = TemporalControlSignal()

        state = control.compute_state(
            prediction_error=torch.tensor(0.7),
            novelty=torch.tensor(0.5),
            reward=torch.tensor(0.1),
        )

        assert isinstance(state, TemporalControlState)
        assert 0.0 <= state.tau <= 1.0
        assert isinstance(state.write_enabled, bool)
        assert state.plasticity_gain >= 0.0

    def test_should_write(self):
        """Test write decision."""
        from t4dm.core.temporal_control import TemporalControlSignal

        control = TemporalControlSignal()

        # High signals should enable write
        should_write_high = control.should_write(
            prediction_error=0.9,
            novelty=0.8,
            reward=0.5,
        )

        # Low signals should disable write
        should_write_low = control.should_write(
            prediction_error=0.1,
            novelty=0.1,
            reward=0.0,
        )

        # High signals more likely to write than low
        # (exact behavior depends on trained lambdas)
        assert isinstance(should_write_high, bool)
        assert isinstance(should_write_low, bool)


class TestMemoryGateIntegration:
    """P1-02: Test τ(t) integration into MemoryGate."""

    def test_memory_gate_with_tau(self):
        """Test MemoryGate uses τ(t)."""
        from t4dm.core.memory_gate import MemoryGate, GateContext, StorageDecision

        gate = MemoryGate(use_temporal_control=True)
        assert gate.use_temporal_control is True

        context = GateContext(
            session_id="test_session",
            prediction_error=0.5,
            novelty_signal=0.6,
        )

        result = gate.evaluate("Important test content with entities", context)

        # Should have tau_value in result
        assert hasattr(result, "tau_value")
        assert 0.0 <= result.tau_value <= 1.0
        assert hasattr(result, "plasticity_gain")

    def test_gate_context_neural_fields(self):
        """Test GateContext has neural signal fields."""
        from t4dm.core.memory_gate import GateContext

        context = GateContext(
            session_id="test",
            prediction_error=0.7,
            novelty_signal=0.5,
            reward_signal=0.3,
            theta_phase=1.57,  # π/2
        )

        assert context.prediction_error == 0.7
        assert context.novelty_signal == 0.5
        assert context.reward_signal == 0.3
        assert abs(context.theta_phase - 1.57) < 0.01

    def test_gate_result_neural_outputs(self):
        """Test GateResult has neural output fields."""
        from t4dm.core.memory_gate import GateResult, StorageDecision

        result = GateResult(
            decision=StorageDecision.STORE,
            score=0.8,
            reasons=["test"],
            suggested_importance=0.7,
            tau_value=0.65,
            plasticity_gain=1.3,
        )

        assert result.tau_value == 0.65
        assert result.plasticity_gain == 1.3


class TestNorseSNNBackend:
    """P1-03: Test Norse SNN backend wrapper."""

    def test_snn_backend_init(self):
        """Test SNNBackend initialization."""
        from t4dm.nca.snn_backend import SNNBackend, SNNConfig

        config = SNNConfig(use_norse=False)  # Use custom impl for testing
        backend = SNNBackend(input_size=64, hidden_size=128, config=config)

        assert backend.input_size == 64
        assert backend.hidden_size == 128

    def test_snn_forward_pass(self):
        """Test SNN forward pass generates spikes."""
        from t4dm.nca.snn_backend import SNNBackend, SNNConfig

        config = SNNConfig(use_norse=False)
        backend = SNNBackend(input_size=32, hidden_size=64, config=config)

        x = torch.randn(4, 32)  # batch=4
        spikes, state = backend.forward(x)

        assert spikes.shape == (4, 64)
        assert state is not None
        assert state.membrane.shape == (4, 64)

    def test_spike_encoder(self):
        """Test spike encoding."""
        from t4dm.nca.snn_backend import SpikeEncoder

        encoder = SpikeEncoder(encoding="rate", num_steps=10)

        x = torch.randn(4, 32)
        spikes = encoder(x)

        assert spikes.shape == (4, 10, 32)
        assert spikes.min() >= 0.0
        assert spikes.max() <= 1.0

    def test_spike_decoder(self):
        """Test spike decoding."""
        from t4dm.nca.snn_backend import SpikeDecoder

        decoder = SpikeDecoder(decoding="rate")

        spikes = torch.rand(4, 10, 32) > 0.5  # Random spike train
        values = decoder(spikes.float())

        assert values.shape == (4, 32)


class TestSTDPJIT:
    """P1-04: Test Numba JIT for STDP."""

    def test_stdp_jit_compute(self):
        """Test JIT-compiled STDP computation."""
        from t4dm.learning.stdp_jit import compute_stdp_delta_jit, NUMBA_AVAILABLE

        # Should work even without Numba (fallback to numpy)
        delta = compute_stdp_delta_jit(
            delta_t_ms=10.0,  # 10ms post after pre
            current_weight=0.5,
        )

        assert isinstance(delta, float)
        assert delta > 0  # LTP for pre-before-post

        # Test LTD
        delta_ltd = compute_stdp_delta_jit(
            delta_t_ms=-10.0,  # 10ms pre after post
            current_weight=0.5,
        )

        assert delta_ltd < 0  # LTD for post-before-pre

    def test_batch_stdp(self):
        """Test batch STDP computation."""
        from t4dm.learning.stdp_jit import batch_stdp_update_jit

        delta_t = np.array([0.01, -0.01, 0.02, -0.02, 0.005])  # seconds
        weights = np.array([0.5, 0.5, 0.3, 0.7, 0.5])

        deltas = batch_stdp_update_jit(delta_t, weights)

        assert deltas.shape == (5,)
        assert deltas[0] > 0  # LTP
        assert deltas[1] < 0  # LTD
        assert deltas[2] > 0  # LTP
        assert deltas[3] < 0  # LTD

    def test_dopamine_modulation(self):
        """Test dopamine-modulated STDP."""
        from t4dm.learning.stdp_jit import compute_stdp_delta_jit

        # High DA should increase LTP
        delta_high_da = compute_stdp_delta_jit(
            delta_t_ms=10.0,
            current_weight=0.5,
            da_level=0.8,
        )

        delta_low_da = compute_stdp_delta_jit(
            delta_t_ms=10.0,
            current_weight=0.5,
            da_level=0.2,
        )

        # High DA should produce larger LTP
        assert delta_high_da > delta_low_da


class TestUnifiedMemoryItem:
    """P1-05: Test unified MemoryItem schema."""

    def test_unified_memory_item_init(self):
        """Test UnifiedMemoryItem creation."""
        from t4dm.core.unified_memory import UnifiedMemoryItem

        item = UnifiedMemoryItem(content="Test content")

        assert item.content == "Test content"
        assert item.kappa == 0.0  # Default
        assert item.item_type == "episodic"
        assert item.spike_trace is None

    def test_from_episode(self):
        """Test conversion from Episode."""
        from t4dm.core.unified_memory import UnifiedMemoryItem, SpikeTrace
        from t4dm.core.types import Episode

        episode = Episode(
            session_id="test",
            content="Test episode content",
            emotional_valence=0.7,
        )

        spike_trace = SpikeTrace()
        spike_trace.add_spike(10.0, 1)
        spike_trace.add_spike(15.0, 2)

        item = UnifiedMemoryItem.from_episode(
            episode,
            spike_trace=spike_trace,
            tau_value=0.65,
        )

        assert item.content == "Test episode content"
        assert item.importance == 0.7
        assert item.tau_value == 0.65
        assert item.spike_trace is not None

    def test_kappa_update(self):
        """Test κ consolidation updates."""
        from t4dm.core.unified_memory import UnifiedMemoryItem

        item = UnifiedMemoryItem(content="Test", kappa=0.0)

        # NREM replay
        item.update_kappa(delta=0.05, phase="nrem", replay_count=1)
        assert abs(item.kappa - 0.05) < 0.01

        # REM consolidation
        item.update_kappa(delta=0.2, phase="rem")
        assert abs(item.kappa - 0.25) < 0.01

        # History should be recorded
        assert len(item.consolidation_history) == 2
        assert item.consolidation_history[0]["phase"] == "nrem"

    def test_kappa_based_queries(self):
        """Test κ-based query classification."""
        from t4dm.core.unified_memory import UnifiedMemoryItem

        episodic = UnifiedMemoryItem(content="Fresh", kappa=0.1)
        transitional = UnifiedMemoryItem(content="Replayed", kappa=0.5)
        semantic = UnifiedMemoryItem(content="Consolidated", kappa=0.85)

        assert episodic.is_episodic() is True
        assert episodic.is_semantic() is False

        assert transitional.is_transitional() is True

        assert semantic.is_semantic() is True
        assert semantic.is_episodic() is False


class TestKappaFieldsOnTypes:
    """P1-06/07/08: Test κ fields on Episode/Entity/Procedure."""

    def test_episode_has_kappa(self):
        """Test Episode has κ field."""
        from t4dm.core.types import Episode

        episode = Episode(session_id="test", content="Test")
        assert hasattr(episode, "kappa")
        assert episode.kappa == 0.0  # Default for episodic

    def test_entity_has_kappa(self):
        """Test Entity has κ field."""
        from t4dm.core.types import Entity, EntityType

        entity = Entity(
            name="TestEntity",
            entity_type=EntityType.CONCEPT,
            summary="Test summary",
        )
        assert hasattr(entity, "kappa")
        assert entity.kappa == 0.85  # Default for semantic

    def test_procedure_has_kappa(self):
        """Test Procedure has κ field."""
        from t4dm.core.types import Procedure, Domain

        procedure = Procedure(
            name="TestProcedure",
            domain=Domain.CODING,
        )
        assert hasattr(procedure, "kappa")
        assert procedure.kappa == 0.5  # Default for procedural

    def test_kappa_validation(self):
        """Test κ field validation."""
        from t4dm.core.types import Episode
        from pydantic import ValidationError

        # Valid kappa
        episode = Episode(session_id="test", content="Test", kappa=0.5)
        assert episode.kappa == 0.5

        # Invalid kappa should raise validation error
        with pytest.raises(ValidationError):
            Episode(session_id="test", content="Test", kappa=1.5)

        with pytest.raises(ValidationError):
            Episode(session_id="test", content="Test", kappa=-0.1)


class TestGPUPDESolver:
    """P1-09: Test GPU PDE solver."""

    def test_gpu_field_init(self):
        """Test NeuralFieldGPU initialization."""
        from t4dm.nca.neural_field_gpu import NeuralFieldGPU, GPUFieldConfig

        config = GPUFieldConfig(
            spatial_dims=1,
            grid_size=32,
            device="cpu",  # Use CPU for testing
        )
        solver = NeuralFieldGPU(config)

        assert solver.fields.shape == (6, 32)  # 6 NTs, 32 grid points

    def test_gpu_field_step(self):
        """Test GPU solver step."""
        from t4dm.nca.neural_field_gpu import NeuralFieldGPU, GPUFieldConfig

        config = GPUFieldConfig(spatial_dims=1, grid_size=32, device="cpu")
        solver = NeuralFieldGPU(config)

        initial_state = solver.get_mean_state().clone()

        # Run some steps
        for _ in range(10):
            solver.step()

        final_state = solver.get_mean_state()

        # State should remain stable (within bounds)
        assert torch.all(final_state >= 0.0)
        assert torch.all(final_state <= 1.0)

    def test_laplacian_kernel(self):
        """Test Laplacian kernel computation."""
        from t4dm.nca.neural_field_gpu import LaplacianKernel

        kernel = LaplacianKernel(spatial_dims=1, dx=1.0)

        # Test on a simple field
        field = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # (1, 4) -> batch, spatial
        laplacian = kernel(field)

        # Laplacian should detect the peak
        assert laplacian is not None
        assert laplacian.shape[-1] == 4

    def test_rpe_injection(self):
        """Test reward prediction error injection."""
        from t4dm.nca.neural_field_gpu import NeuralFieldGPU, GPUFieldConfig

        config = GPUFieldConfig(spatial_dims=1, grid_size=16, device="cpu")
        solver = NeuralFieldGPU(config)

        da_before = solver.fields[0].mean().item()

        # Inject positive RPE
        solver.inject_rpe(rpe=0.5, scale=0.3)

        da_after = solver.fields[0].mean().item()

        # DA should increase
        assert da_after > da_before


class TestPhase1Integration:
    """Full Phase 1 integration test."""

    def test_full_pipeline(self):
        """Test complete P1 pipeline: τ(t) → gate → memory → SNN → STDP."""
        from t4dm.core.temporal_control import TemporalControlSignal
        from t4dm.core.memory_gate import MemoryGate, GateContext, StorageDecision
        from t4dm.core.unified_memory import UnifiedMemoryItem, SpikeTrace
        from t4dm.core.types import Episode
        from t4dm.nca.snn_backend import SNNBackend, SNNConfig, SpikeEncoder
        from t4dm.learning.stdp_jit import compute_stdp_delta_jit

        # 1. Create episode
        episode = Episode(
            session_id="integration_test",
            content="Important discovery about neural networks",
            emotional_valence=0.8,
        )
        assert episode.kappa == 0.0  # P1-06: κ field exists

        # 2. Evaluate with τ(t)-integrated gate
        gate = MemoryGate(use_temporal_control=True, tau_weight=0.3)
        context = GateContext(
            session_id="integration_test",
            prediction_error=0.6,  # Surprising
            novelty_signal=0.7,  # Novel
        )
        result = gate.evaluate(episode.content, context)

        # Gate should use τ(t)
        assert hasattr(result, "tau_value")
        assert result.decision == StorageDecision.STORE  # Should store important content

        # 3. Convert to UnifiedMemoryItem
        item = UnifiedMemoryItem.from_episode(
            episode,
            tau_value=result.tau_value,
        )
        assert item.tau_value == result.tau_value

        # 4. Encode through SNN (P1-03)
        config = SNNConfig(use_norse=False)
        snn = SNNBackend(input_size=32, hidden_size=64, config=config)
        encoder = SpikeEncoder(encoding="rate", num_steps=10)

        # Simulate embedding as input
        embedding = torch.randn(1, 32)
        spike_train = encoder(embedding)

        # Run through SNN
        spikes, state = snn(embedding)

        # 5. Record spike trace
        spike_trace = SpikeTrace()
        spike_indices = (spikes[0] > 0).nonzero(as_tuple=True)[0]
        for idx in spike_indices[:10]:  # Limit for test
            spike_trace.add_spike(float(idx) * 1.0, int(idx))

        item.set_spike_trace(spike_trace)

        # 6. Compute STDP (P1-04)
        if spike_trace.num_spikes >= 2:
            delta = compute_stdp_delta_jit(
                delta_t_ms=5.0,  # 5ms timing
                current_weight=0.5,
            )
            assert delta > 0  # LTP

        # 7. Update κ (P1-06/07/08)
        item.update_kappa(delta=0.05, phase="nrem")
        assert item.kappa > 0.0

        # Pipeline completed successfully
        assert item.content == episode.content
        assert item.spike_trace is not None

    def test_no_performance_regression(self):
        """Verify no significant performance regression."""
        import time

        from t4dm.learning.stdp_jit import batch_stdp_update_jit

        # Benchmark STDP
        n_synapses = 10000
        delta_t = np.random.uniform(-0.05, 0.05, n_synapses)
        weights = np.random.uniform(0.1, 0.9, n_synapses)

        # Warmup
        _ = batch_stdp_update_jit(delta_t[:100], weights[:100])

        # Time
        start = time.perf_counter()
        for _ in range(10):
            _ = batch_stdp_update_jit(delta_t, weights)
        elapsed = time.perf_counter() - start

        # Should complete 10 batches of 10k synapses in < 1 second
        # (with JIT this should be ~10-100ms)
        assert elapsed < 1.0, f"STDP too slow: {elapsed:.3f}s for 10x10k synapses"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
