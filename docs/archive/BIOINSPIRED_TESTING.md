# Bioinspired Systems Testing Strategy

## Overview

This document defines the testing strategy for T4DM's bioinspired neural memory components. Testing spans from unit-level component validation to system-wide biological plausibility checks.

## Testing Pyramid

```
                    ╭─────────────────╮
                    │    E2E Tests    │  ← 5%
                    │   (Full Stack)  │
                  ╭─┴─────────────────┴─╮
                  │  Integration Tests  │  ← 15%
                  │  (Multi-Component)  │
              ╭───┴─────────────────────┴───╮
              │     Biological Validation    │  ← 20%
              │   (Neuroscience Targets)     │
          ╭───┴───────────────────────────────┴───╮
          │           Unit Tests                   │  ← 60%
          │    (Individual Components)             │
          └───────────────────────────────────────┘
```

---

## 1. Unit Tests

### 1.1 Dendritic Neuron Tests

```python
# tests/unit/test_dendritic_neuron.py

class TestDendriticNeuron:
    """Two-compartment neuron model tests."""

    def test_compartment_isolation(self):
        """Verify basal/apical compartments process independently."""
        neuron = DendriticNeuron(hidden_dim=64, context_dim=32)
        basal_input = torch.randn(1, 64)
        apical_input = torch.randn(1, 32)

        # Zero apical should not affect basal processing
        out_with_context = neuron(basal_input, apical_input)
        out_without_context = neuron(basal_input, torch.zeros(1, 32))

        assert not torch.allclose(out_with_context, out_without_context)

    def test_coupling_strength_effect(self):
        """Coupling strength modulates context integration."""
        neuron_weak = DendriticNeuron(coupling_strength=0.1)
        neuron_strong = DendriticNeuron(coupling_strength=0.9)

        basal = torch.randn(1, 64)
        apical = torch.randn(1, 32)

        out_weak = neuron_weak(basal, apical)
        out_strong = neuron_strong(basal, apical)

        # Strong coupling should show greater context influence
        context_effect_weak = torch.norm(out_weak - neuron_weak(basal, torch.zeros(1, 32)))
        context_effect_strong = torch.norm(out_strong - neuron_strong(basal, torch.zeros(1, 32)))

        assert context_effect_strong > context_effect_weak

    def test_time_constants(self):
        """Verify tau_dendrite < tau_soma for proper dynamics."""
        neuron = DendriticNeuron(tau_dendrite=5.0, tau_soma=20.0)
        assert neuron.tau_dendrite < neuron.tau_soma

    def test_gradient_flow(self):
        """Ensure gradients flow through both compartments."""
        neuron = DendriticNeuron()
        basal = torch.randn(1, 64, requires_grad=True)
        apical = torch.randn(1, 32, requires_grad=True)

        out = neuron(basal, apical)
        loss = out.sum()
        loss.backward()

        assert basal.grad is not None
        assert apical.grad is not None
```

### 1.2 Sparse Encoder Tests

```python
# tests/unit/test_sparse_encoder.py

class TestSparseEncoder:
    """k-Winner-Take-All sparse encoding tests."""

    def test_sparsity_level(self):
        """Output sparsity matches target (2% default)."""
        encoder = SparseEncoder(hidden_dim=1000, sparsity=0.02)
        input_batch = torch.randn(32, 512)

        encoded = encoder(input_batch)

        # Count non-zero activations
        sparsity = (encoded != 0).float().mean()
        assert 0.015 < sparsity < 0.025  # Within 25% of target

    def test_kwta_exact_k(self):
        """k-WTA selects exactly k winners per sample."""
        encoder = SparseEncoder(hidden_dim=100, sparsity=0.05, use_kwta=True)
        input_batch = torch.randn(10, 64)

        encoded = encoder(input_batch)
        k = int(100 * 0.05)  # 5 winners

        for sample in encoded:
            active = (sample != 0).sum().item()
            assert active == k

    def test_lateral_inhibition(self):
        """Lateral inhibition produces competition."""
        encoder = SparseEncoder(lateral_inhibition=0.5)

        # Similar inputs should compete differently than dissimilar
        similar_inputs = torch.randn(2, 64)
        similar_inputs[1] = similar_inputs[0] + 0.1 * torch.randn(64)

        encoded = encoder(similar_inputs)

        # Overlap should be partial, not identical
        overlap = ((encoded[0] != 0) & (encoded[1] != 0)).float().mean()
        assert 0.3 < overlap < 0.9

    def test_pattern_orthogonality(self):
        """Different inputs produce decorrelated patterns."""
        encoder = SparseEncoder(hidden_dim=500, sparsity=0.02)

        inputs = torch.randn(100, 64)
        encoded = encoder(inputs)

        # Compute average pairwise correlation
        norm_encoded = encoded / (encoded.norm(dim=1, keepdim=True) + 1e-8)
        correlations = torch.mm(norm_encoded, norm_encoded.t())
        off_diagonal = correlations[~torch.eye(100, dtype=bool)]

        assert off_diagonal.abs().mean() < 0.1  # Low correlation
```

### 1.3 Neuromodulator System Tests

```python
# tests/unit/test_neuromodulator.py

class TestNeuromodulatorGains:
    """Neuromodulator gain parameter tests."""

    def test_learning_rate_formula(self):
        """η_eff = η_base × g_DA × g_NE × g_ACh × g_5HT"""
        gains = NeuromodGains(
            rho_da=1.2, rho_ne=1.1, rho_ach_fast=1.0,
            rho_ach_slow=0.9, alpha_ne=0.8
        )

        base_lr = 0.01
        effective_lr = gains.compute_learning_rate(base_lr)

        expected = base_lr * 1.2 * 1.1 * 1.0 * 0.9
        assert abs(effective_lr - expected) < 1e-6

    def test_gain_bounds(self):
        """Gains stay within biological plausibility."""
        gains = NeuromodGains()

        for attr in ['rho_da', 'rho_ne', 'rho_ach_fast', 'rho_ach_slow']:
            value = getattr(gains, attr)
            assert 0.1 <= value <= 10.0, f"{attr} out of biological range"

    def test_100x_learning_separation(self):
        """Fast vs slow learning should differ by ~100x."""
        fast_gains = NeuromodGains(rho_da=2.0, rho_ne=2.0, rho_ach_fast=2.0)
        slow_gains = NeuromodGains(rho_da=0.2, rho_ne=0.2, rho_ach_slow=0.2)

        base_lr = 0.01
        fast_lr = fast_gains.compute_learning_rate(base_lr)
        slow_lr = slow_gains.compute_learning_rate(base_lr)

        ratio = fast_lr / slow_lr
        assert 50 < ratio < 200  # ~100x separation
```

### 1.4 Eligibility Trace Tests

```python
# tests/unit/test_eligibility.py

class TestEligibilityTrace:
    """Temporal credit assignment trace tests."""

    def test_exponential_decay(self):
        """Trace decays exponentially with tau."""
        trace = EligibilityTrace(decay=0.9, tau_trace=10.0)

        # Spike at t=0
        trace.update(torch.ones(1, 64))
        initial = trace.get_trace().clone()

        # Decay for 10 steps
        for _ in range(10):
            trace.step()

        final = trace.get_trace()
        expected_decay = 0.9 ** 10

        assert torch.allclose(final, initial * expected_decay, atol=0.01)

    def test_trace_accumulation(self):
        """Multiple spikes accumulate in trace."""
        trace = EligibilityTrace(decay=0.9)

        trace.update(torch.ones(1, 64))
        trace.step()
        trace.update(torch.ones(1, 64))

        current = trace.get_trace()
        # Should be > 1.0 due to accumulation
        assert current.mean() > 1.0

    def test_credit_assignment(self):
        """Late reward credits earlier actions."""
        trace = EligibilityTrace(decay=0.95)

        # Action at t=0
        action_pattern = torch.zeros(1, 64)
        action_pattern[0, :10] = 1.0
        trace.update(action_pattern)

        # Wait 5 steps
        for _ in range(5):
            trace.step()

        # Reward arrives
        credit = trace.assign_credit(reward=1.0)

        # Credit should still favor original action pattern
        assert credit[0, :10].mean() > credit[0, 10:].mean()
```

### 1.5 Attractor Network Tests

```python
# tests/unit/test_attractor.py

class TestAttractorNetwork:
    """Hopfield-style attractor dynamics tests."""

    def test_pattern_storage(self):
        """Network stores and retrieves patterns."""
        net = AttractorNetwork(dim=100, capacity=10)

        # Store 5 random patterns
        patterns = [torch.randn(100) for _ in range(5)]
        for p in patterns:
            net.store(p)

        # Retrieve with noisy cue
        noisy_cue = patterns[2] + 0.3 * torch.randn(100)
        retrieved = net.retrieve(noisy_cue, steps=50)

        similarity = F.cosine_similarity(retrieved, patterns[2], dim=0)
        assert similarity > 0.9

    def test_settling_dynamics(self):
        """Energy decreases during settling."""
        net = AttractorNetwork(settling_steps=100)
        patterns = [torch.randn(100) for _ in range(3)]
        for p in patterns:
            net.store(p)

        cue = torch.randn(100)
        energies = []

        state = cue.clone()
        for step in range(100):
            state = net.step(state)
            energies.append(net.compute_energy(state))

        # Energy should decrease monotonically
        for i in range(len(energies) - 1):
            assert energies[i+1] <= energies[i] + 1e-6

    def test_capacity_limit(self):
        """Performance degrades beyond capacity."""
        net = AttractorNetwork(dim=100, capacity=14)  # Hopfield ~0.14N

        # Store up to capacity
        patterns = [torch.randn(100) for _ in range(14)]
        for p in patterns:
            net.store(p)

        # Test retrieval
        accuracies = []
        for p in patterns:
            noisy = p + 0.2 * torch.randn(100)
            retrieved = net.retrieve(noisy)
            acc = F.cosine_similarity(retrieved, p, dim=0)
            accuracies.append(acc.item())

        assert sum(accuracies) / len(accuracies) > 0.85

        # Overstuff the network
        for _ in range(20):
            net.store(torch.randn(100))

        # Retrieval should degrade
        noisy = patterns[0] + 0.2 * torch.randn(100)
        retrieved = net.retrieve(noisy)
        degraded_acc = F.cosine_similarity(retrieved, patterns[0], dim=0)

        assert degraded_acc < accuracies[0]
```

### 1.6 Fast Episodic Store Tests

```python
# tests/unit/test_fast_episodic.py

class TestFastEpisodicStore:
    """Hippocampus-like rapid memory store tests."""

    def test_one_shot_learning(self):
        """Single exposure creates retrievable memory."""
        store = FastEpisodicStore(capacity=1000, learning_rate=1.0)

        episode = torch.randn(1, 512)
        store.write(episode)

        retrieved = store.read(episode)
        similarity = F.cosine_similarity(episode, retrieved, dim=1)

        assert similarity > 0.95

    def test_capacity_management(self):
        """Store handles capacity overflow gracefully."""
        store = FastEpisodicStore(capacity=100)

        # Overfill the store
        episodes = [torch.randn(1, 512) for _ in range(150)]
        for ep in episodes:
            store.write(ep)

        assert store.size() <= 100

    def test_consolidation_threshold(self):
        """Frequent patterns flag for consolidation."""
        store = FastEpisodicStore(consolidation_threshold=3)

        # Write same pattern multiple times
        pattern = torch.randn(1, 512)
        for _ in range(4):
            store.write(pattern)

        candidates = store.get_consolidation_candidates()
        assert len(candidates) >= 1

    def test_interference_resistance(self):
        """Similar patterns don't catastrophically interfere."""
        store = FastEpisodicStore(capacity=100)

        base_pattern = torch.randn(1, 512)
        similar_patterns = [
            base_pattern + 0.1 * torch.randn(1, 512)
            for _ in range(10)
        ]

        for p in similar_patterns:
            store.write(p)

        # Original should still be partially retrievable
        retrieved = store.read(base_pattern)
        similarity = F.cosine_similarity(base_pattern, retrieved, dim=1)

        assert similarity > 0.5  # Partial retrieval
```

---

## 2. Integration Tests

### 2.1 Encoding Pipeline Tests

```python
# tests/integration/test_encoding_pipeline.py

class TestEncodingPipeline:
    """Full encoding pipeline integration tests."""

    def test_mcp_to_sparse_encoding(self):
        """MCP messages encode to sparse representations."""
        pipeline = BioinspiredPipeline(config)

        mcp_message = {
            "method": "tools/call",
            "params": {"name": "remember", "content": "test memory"}
        }

        encoded = pipeline.encode(mcp_message)

        # Verify sparsity
        sparsity = (encoded != 0).float().mean()
        assert 0.01 < sparsity < 0.05

    def test_context_modulated_encoding(self):
        """Neuromodulator state affects encoding."""
        pipeline = BioinspiredPipeline(config)

        content = "important discovery"

        # High dopamine state (reward)
        encoded_high_da = pipeline.encode_with_modulation(
            content, da_level=0.9
        )

        # Low dopamine state
        encoded_low_da = pipeline.encode_with_modulation(
            content, da_level=0.1
        )

        # High DA should produce stronger encoding
        assert encoded_high_da.norm() > encoded_low_da.norm()

    def test_end_to_end_memory_flow(self):
        """Memory flows correctly through all components."""
        system = BioinspiredMemorySystem(config)

        # Create episode
        episode = system.create_episode(
            content="Test content",
            context={"session": "test"}
        )

        # Store in fast episodic
        system.store_fast(episode)

        # Retrieve
        retrieved = system.retrieve(episode.cue)

        assert retrieved.content == episode.content

        # Trigger consolidation
        system.consolidate(episode)

        # Verify in semantic store
        semantic_result = system.semantic_query(episode.cue)
        assert semantic_result is not None
```

### 2.2 Neuromodulator-Learning Integration

```python
# tests/integration/test_neuromod_learning.py

class TestNeuromodulatorLearning:
    """Neuromodulator-gated learning integration tests."""

    def test_reward_enhances_learning(self):
        """Dopamine release enhances weight updates."""
        learner = ModulatedLearner(config)

        pre = torch.randn(64)
        post = torch.randn(64)

        # Learning without reward
        delta_no_reward = learner.compute_update(pre, post, reward=0)

        # Learning with reward
        delta_with_reward = learner.compute_update(pre, post, reward=1.0)

        assert delta_with_reward.norm() > delta_no_reward.norm() * 1.5

    def test_attention_sharpens_encoding(self):
        """Acetylcholine increases encoding precision."""
        encoder = ModulatedEncoder(config)

        input_data = torch.randn(1, 512)

        # Low attention
        encoded_low = encoder(input_data, ach_level=0.2)

        # High attention
        encoded_high = encoder(input_data, ach_level=0.9)

        # Higher ACh should produce sparser, more precise encoding
        sparsity_low = (encoded_low != 0).float().mean()
        sparsity_high = (encoded_high != 0).float().mean()

        assert sparsity_high < sparsity_low
```

---

## 3. Biological Validation Tests

### 3.1 Sparsity Targets

```python
# tests/biological/test_sparsity_targets.py

class TestBiologicalSparsity:
    """Validate sparsity matches neuroscience targets."""

    @pytest.mark.parametrize("region,target_sparsity,tolerance", [
        ("hippocampus", 0.02, 0.01),   # DG: 1-5%
        ("neocortex", 0.05, 0.02),     # Cortical sparse coding
        ("entorhinal", 0.10, 0.03),    # EC layer II
    ])
    def test_regional_sparsity(self, region, target_sparsity, tolerance):
        """Different regions maintain appropriate sparsity."""
        encoder = get_regional_encoder(region, config)

        test_inputs = torch.randn(1000, 512)
        encoded = encoder(test_inputs)

        actual_sparsity = (encoded != 0).float().mean()

        assert abs(actual_sparsity - target_sparsity) < tolerance
```

### 3.2 Learning Rate Separation

```python
# tests/biological/test_learning_dynamics.py

class TestLearningRateSeparation:
    """Validate 100x learning rate separation (Hinton target)."""

    def test_fast_slow_separation(self):
        """Fast learning 100x faster than slow learning."""
        fast_system = FastLearningSystem(config)
        slow_system = SlowLearningSystem(config)

        # Same training data
        data = generate_training_episodes(100)

        # Measure convergence speed
        fast_steps = fast_system.steps_to_convergence(data)
        slow_steps = slow_system.steps_to_convergence(data)

        ratio = slow_steps / fast_steps
        assert 50 < ratio < 200  # ~100x difference

    def test_neuromodulator_mediated_separation(self):
        """Neuromodulators control learning rate range."""
        modulator = NeuromodulatorSystem(config)

        # Maximum learning (high DA, NE, ACh)
        max_lr = modulator.compute_effective_lr(
            base_lr=0.01,
            da=1.0, ne=1.0, ach=1.0
        )

        # Minimum learning (low everything)
        min_lr = modulator.compute_effective_lr(
            base_lr=0.01,
            da=0.1, ne=0.1, ach=0.1
        )

        ratio = max_lr / min_lr
        assert ratio > 100
```

### 3.3 Pattern Capacity

```python
# tests/biological/test_capacity.py

class TestPatternCapacity:
    """Validate memory capacity matches biological targets."""

    def test_hopfield_capacity(self):
        """Attractor network holds ~0.14N patterns."""
        N = 1000
        expected_capacity = int(0.14 * N)

        network = AttractorNetwork(dim=N)

        # Store patterns up to expected capacity
        patterns = [torch.randn(N) for _ in range(expected_capacity)]
        for p in patterns:
            network.store(p)

        # Test retrieval accuracy
        correct = 0
        for p in patterns:
            noisy = p + 0.2 * torch.randn(N)
            retrieved = network.retrieve(noisy)
            if F.cosine_similarity(retrieved, p, dim=0) > 0.9:
                correct += 1

        accuracy = correct / len(patterns)
        assert accuracy > 0.85

    def test_fast_episodic_capacity(self):
        """Fast store holds ~10,000 episodes."""
        store = FastEpisodicStore(capacity=10000)

        # Fill to capacity
        for _ in range(10000):
            store.write(torch.randn(1, 512))

        assert store.size() == 10000

        # Verify retrieval still works
        test_ep = torch.randn(1, 512)
        store.write(test_ep)

        retrieved = store.read(test_ep)
        similarity = F.cosine_similarity(test_ep, retrieved, dim=1)

        assert similarity > 0.8
```

---

## 4. Performance Benchmarks

### 4.1 Encoding Latency

```python
# tests/performance/test_encoding_latency.py

class TestEncodingLatency:
    """Benchmark encoding performance."""

    def test_sparse_encoding_latency(self, benchmark):
        """Sparse encoding < 10ms for batch of 32."""
        encoder = SparseEncoder(hidden_dim=1000, sparsity=0.02)
        batch = torch.randn(32, 512)

        result = benchmark(encoder, batch)

        assert benchmark.stats['mean'] < 0.010  # 10ms

    def test_dendritic_computation_latency(self, benchmark):
        """Dendritic computation < 5ms."""
        neuron = DendriticNeuron(hidden_dim=256, context_dim=64)
        basal = torch.randn(32, 256)
        apical = torch.randn(32, 64)

        result = benchmark(neuron, basal, apical)

        assert benchmark.stats['mean'] < 0.005  # 5ms
```

### 4.2 Memory Operations

```python
# tests/performance/test_memory_ops.py

class TestMemoryOperations:
    """Benchmark memory read/write performance."""

    def test_fast_episodic_write(self, benchmark):
        """Fast store write < 1ms."""
        store = FastEpisodicStore(capacity=10000)
        episode = torch.randn(1, 512)

        benchmark(store.write, episode)

        assert benchmark.stats['mean'] < 0.001  # 1ms

    def test_attractor_settling(self, benchmark):
        """Attractor settling < 50ms for 100 steps."""
        network = AttractorNetwork(dim=500, settling_steps=100)

        # Pre-store patterns
        for _ in range(10):
            network.store(torch.randn(500))

        cue = torch.randn(500)
        benchmark(network.retrieve, cue)

        assert benchmark.stats['mean'] < 0.050  # 50ms
```

---

## 5. Regression Tests

### 5.1 Backward Compatibility

```python
# tests/regression/test_backward_compat.py

class TestBackwardCompatibility:
    """Ensure bioinspired features don't break existing behavior."""

    def test_disabled_bioinspired_matches_original(self):
        """With bioinspired disabled, behavior unchanged."""
        config_off = Config(bioinspired={'enabled': False})
        config_on = Config(bioinspired={'enabled': True})

        system_off = MemorySystem(config_off)
        system_on = MemorySystem(config_on)

        # Same operations
        test_data = generate_test_episodes(100)

        for ep in test_data:
            system_off.store(ep)
            system_on.store(ep)

        # Results should match when bioinspired disabled
        for ep in test_data:
            result_off = system_off.retrieve(ep.cue)
            result_on = system_on.retrieve(ep.cue)

            assert result_off == result_on

    def test_config_migration(self):
        """Old configs work with new bioinspired defaults."""
        old_config = {
            "neuromodulation": {"enabled": True},
            "memory": {"episodic_weight": 0.6}
        }

        # Should not raise
        system = MemorySystem.from_dict(old_config)

        # Bioinspired should be disabled by default
        assert system.config.bioinspired.enabled == False
```

---

## 6. Test Configuration

### 6.1 pytest.ini

```ini
[pytest]
markers =
    unit: Unit tests (fast)
    integration: Integration tests (medium)
    biological: Biological validation (slow)
    performance: Performance benchmarks
    regression: Regression tests

testpaths = tests
python_files = test_*.py
python_functions = test_*

addopts =
    -v
    --tb=short
    --strict-markers
    -ra

filterwarnings =
    ignore::DeprecationWarning
```

### 6.2 CI Pipeline

```yaml
# .github/workflows/bioinspired-tests.yml

name: Bioinspired Tests

on:
  push:
    paths:
      - 'src/bioinspired/**'
      - 'tests/**'
  pull_request:
    paths:
      - 'src/bioinspired/**'

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run unit tests
        run: pytest tests/unit -m unit --cov=src/bioinspired

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: pytest tests/integration -m integration

  biological-validation:
    runs-on: ubuntu-latest
    needs: integration-tests
    steps:
      - uses: actions/checkout@v3
      - name: Run biological validation
        run: pytest tests/biological -m biological

  performance:
    runs-on: ubuntu-latest
    needs: unit-tests
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: pytest tests/performance --benchmark-only
```

---

## 7. Coverage Requirements

| Component | Minimum Coverage |
|-----------|------------------|
| Dendritic Neuron | 95% |
| Sparse Encoder | 95% |
| Neuromodulator | 90% |
| Eligibility Trace | 90% |
| Attractor Network | 85% |
| Fast Episodic | 85% |
| Integration | 80% |

---

## 8. Test Data Generators

```python
# tests/fixtures/generators.py

def generate_training_episodes(n: int) -> List[Episode]:
    """Generate realistic training episodes."""
    pass

def generate_noisy_patterns(base: Tensor, noise_levels: List[float]) -> List[Tensor]:
    """Generate patterns with varying noise."""
    pass

def generate_modulation_scenarios() -> List[ModulationState]:
    """Generate diverse neuromodulator states."""
    pass
```

---

## Summary

This testing strategy ensures:
1. **Individual correctness**: Unit tests verify each component
2. **System integration**: Components work together
3. **Biological plausibility**: Matches neuroscience targets
4. **Performance**: Meets latency requirements
5. **Stability**: No regressions from existing functionality

Target: 90%+ coverage on bioinspired components with all biological validation tests passing.
