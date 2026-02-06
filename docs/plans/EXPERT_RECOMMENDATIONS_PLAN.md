# T4DM Expert Recommendations Implementation Plan

**Created**: 2026-02-06
**Status**: ACTIVE
**Source**: Expert Panel Analysis (Hinton, Bengio, Friston, O'Reilly, Graves)
**Total Atoms**: 45 across 5 waves
**Estimated Duration**: 8-12 weeks

---

## Executive Summary

This plan addresses all gaps identified by the expert panel review of T4DM's architecture. Implementation follows TDD principles with explicit success criteria for each atom.

### Expert Gap Summary

| Expert | Gap | Priority | Wave |
|--------|-----|----------|------|
| **Hinton** | Adaptive layer thresholds for FF | HIGH | 1 |
| **Hinton** | Dream-based negative samples | MEDIUM | 2 |
| **Bengio** | Energy convergence validation | HIGH | 1 |
| **Bengio** | IIT consciousness metrics (Φ) | MEDIUM | 3 |
| **Friston** | Free Energy objective (ELBO) | HIGH | 1 |
| **Friston** | Uncertainty-aware storage | HIGH | 2 |
| **Friston** | Markov blanket retrieval | MEDIUM | 3 |
| **Friston** | Variational consolidation framing | LOW | 4 |
| **O'Reilly** | Adaptive consolidation trigger | HIGH | 1 |
| **O'Reilly** | Reconsolidation lability | HIGH | 2 |
| **O'Reilly** | Generalization quality scoring | MEDIUM | 3 |
| **Graves** | Learned embedding alignment | HIGH | 2 |
| **Graves** | Edge type importance learning | MEDIUM | 3 |
| **Graves** | Verify LearnedRetrievalScorer integration | HIGH | 1 |

---

## Wave Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WAVE 1: FOUNDATIONS                              │
│                        (Critical Infrastructure)                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Free Energy │ │  Adaptive   │ │  Energy     │ │  Retrieval  │       │
│  │  Objective  │ │  Thresholds │ │ Validation  │ │  Integration│       │
│  │  (Friston)  │ │  (Hinton)   │ │  (Bengio)   │ │  (Graves)   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
│                        + Adaptive Consolidation (O'Reilly)              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         WAVE 2: UNCERTAINTY                              │
│                       (Probabilistic Memory)                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │ Vector Cov  │ │ Reconsol.   │ │  Embedding  │ │   Dream     │       │
│  │  Storage    │ │  Lability   │ │  Alignment  │ │  Negatives  │       │
│  │  (Friston)  │ │  (O'Reilly) │ │  (Graves)   │ │  (Hinton)   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         WAVE 3: INTELLIGENCE                             │
│                       (Advanced Retrieval)                               │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │
│  │   Markov    │ │    Edge     │ │  General.   │ │     IIT     │       │
│  │   Blanket   │ │    Type     │ │   Quality   │ │  Conscious  │       │
│  │  (Friston)  │ │  (Graves)   │ │  (O'Reilly) │ │  (Bengio)   │       │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         WAVE 4: INTEGRATION                              │
│                        (System Coherence)                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                       │
│  │ Variational │ │   Sleep     │ │  End-to-End │                       │
│  │   Consol.   │ │   Framing   │ │   Pipeline  │                       │
│  │  (Friston)  │ │   (All)     │ │  Validation │                       │
│  └─────────────┘ └─────────────┘ └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         WAVE 5: BENCHMARKS                               │
│                        (Validation & Proof)                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                       │
│  │ LongMemEval │ │   DMR       │ │  Bio-Plaus  │                       │
│  │  Benchmark  │ │  Benchmark  │ │  Validation │                       │
│  └─────────────┘ └─────────────┘ └─────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Wave 1: Foundations (Week 1-2)

### Gate Criteria
- All Wave 1 tests pass
- Free Energy decreasing during wake phase
- Adaptive thresholds stabilize within 1000 steps
- LearnedRetrievalScorer integrated in retrieval pipeline

### Atoms

#### W1-01: Free Energy Objective (Friston)

**File**: `src/t4dm/learning/free_energy.py`

**Evidence Base**: Friston (2010) "The free-energy principle: a unified brain theory?"

**Implementation**:
```python
@dataclass
class FreeEnergyConfig:
    """Configuration for variational free energy minimization."""
    beta: float = 1.0  # Inverse temperature (complexity weight)
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.1
    use_elbo: bool = True

class FreeEnergyMinimizer:
    """Friston's variational free energy: F = reconstruction + β·KL[q||p]."""

    def compute_free_energy(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
        q_posterior: torch.distributions.Distribution,
        p_prior: torch.distributions.Distribution,
    ) -> FreeEnergyResult:
        """
        Compute variational free energy.

        F = E_q[log p(o|s)] + β·KL[q(s|o) || p(s)]

        Where:
        - First term: reconstruction accuracy (negative log-likelihood)
        - Second term: complexity cost (deviation from prior)
        """
        reconstruction = self._reconstruction_error(prediction, observation)
        complexity = kl_divergence(q_posterior, p_prior)

        F = self.config.reconstruction_weight * reconstruction \
            + self.config.beta * self.config.kl_weight * complexity

        return FreeEnergyResult(
            free_energy=F,
            reconstruction=reconstruction,
            complexity=complexity,
            learning_rate_scale=self._compute_lr_scale(F),
        )
```

**Tests**: `tests/unit/learning/test_free_energy.py`
```python
def test_free_energy_decreases_during_wake():
    """F should decrease as predictions improve."""
    minimizer = FreeEnergyMinimizer()
    F_values = []

    for step in range(100):
        prediction = model(observation)
        result = minimizer.compute_free_energy(prediction, observation, q, p)
        F_values.append(result.free_energy)

        # Update model to minimize F
        loss = result.free_energy
        loss.backward()
        optimizer.step()

    # F should trend downward
    slope = np.polyfit(range(len(F_values)), F_values, 1)[0]
    assert slope < 0, "Free energy should decrease during wake"

def test_kl_divergence_regularizes():
    """KL term should prevent q from deviating too far from p."""
    # When q == p, KL = 0
    q = torch.distributions.Normal(0, 1)
    p = torch.distributions.Normal(0, 1)
    assert kl_divergence(q, p) < 1e-6

    # When q != p, KL > 0
    q = torch.distributions.Normal(1, 0.5)
    assert kl_divergence(q, p) > 0

def test_reconstruction_error_computation():
    """Reconstruction error should match prediction-observation gap."""
    minimizer = FreeEnergyMinimizer()

    prediction = torch.randn(1024)
    observation = prediction + 0.1 * torch.randn(1024)  # Small noise

    result = minimizer.compute_free_energy(prediction, observation, q, p)

    # Small noise → small reconstruction error
    assert result.reconstruction < 0.1
```

**Success Criteria**:
- [ ] F decreases monotonically during wake phase (100 steps)
- [ ] KL term = 0 when q matches p
- [ ] Reconstruction error correlates with prediction accuracy (r > 0.9)
- [ ] Learning rate scales with dF/dt

---

#### W1-02: Adaptive Layer Thresholds (Hinton)

**File**: `src/t4dm/nca/forward_forward.py` (extend existing)

**Evidence Base**: Hinton (2022) "The Forward-Forward Algorithm"

**Implementation**:
```python
@dataclass
class AdaptiveThresholdConfig:
    """Homeostatic threshold adaptation per FF layer."""
    target_firing_rate: float = 0.15  # Target 15% activation
    adaptation_rate: float = 0.01  # θ update rate
    min_threshold: float = 0.1
    max_threshold: float = 10.0
    window_size: int = 100  # Steps to average over

class AdaptiveThreshold:
    """Layer-specific threshold that adapts to maintain target firing rate."""

    def __init__(self, config: AdaptiveThresholdConfig):
        self.config = config
        self.theta = 1.0  # Initial threshold
        self.firing_history = deque(maxlen=config.window_size)

    def update(self, goodness: torch.Tensor) -> float:
        """
        Update threshold based on observed firing rate.

        If firing_rate > target: increase θ (make it harder to fire)
        If firing_rate < target: decrease θ (make it easier to fire)
        """
        # Compute current firing rate
        fired = (goodness > self.theta).float().mean().item()
        self.firing_history.append(fired)

        if len(self.firing_history) >= self.config.window_size:
            avg_rate = np.mean(self.firing_history)

            # Multiplicative update for stability
            ratio = self.config.target_firing_rate / (avg_rate + 1e-6)
            self.theta *= (1 - self.config.adaptation_rate) + \
                          self.config.adaptation_rate * ratio

            # Clamp to valid range
            self.theta = np.clip(
                self.theta,
                self.config.min_threshold,
                self.config.max_threshold
            )

        return self.theta

class ForwardForwardNetwork:
    """Extended with adaptive thresholds per layer."""

    def __init__(self, config: ForwardForwardConfig):
        self.config = config
        self.layers = nn.ModuleList([...])

        # One adaptive threshold per layer
        self.thresholds = [
            AdaptiveThreshold(config.adaptive_threshold)
            for _ in range(len(self.layers))
        ]

    def forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        h = self.layers[layer_idx](x)
        goodness = (h ** 2).sum(dim=-1)

        # Use layer-specific adaptive threshold
        theta = self.thresholds[layer_idx].update(goodness)

        if self.training:
            # Positive phase: maximize G > θ
            # Negative phase: minimize G < θ
            ...

        return h
```

**Tests**: `tests/unit/nca/test_adaptive_threshold.py`
```python
def test_threshold_converges_to_target_rate():
    """Threshold should adapt until firing rate ≈ target."""
    config = AdaptiveThresholdConfig(target_firing_rate=0.15)
    threshold = AdaptiveThreshold(config)

    # Simulate 1000 steps with random goodness values
    for _ in range(1000):
        goodness = torch.randn(100) * 2  # Random goodness
        threshold.update(goodness)

    # After adaptation, firing rate should be near target
    final_rates = []
    for _ in range(100):
        goodness = torch.randn(100) * 2
        fired = (goodness > threshold.theta).float().mean().item()
        final_rates.append(fired)

    avg_final_rate = np.mean(final_rates)
    assert abs(avg_final_rate - 0.15) < 0.05, \
        f"Firing rate {avg_final_rate} not near target 0.15"

def test_threshold_stays_in_bounds():
    """Threshold should not exceed min/max bounds."""
    config = AdaptiveThresholdConfig(min_threshold=0.1, max_threshold=10.0)
    threshold = AdaptiveThreshold(config)

    # Extreme goodness values
    for _ in range(1000):
        goodness = torch.ones(100) * 1000  # Very high → increase θ
        threshold.update(goodness)

    assert threshold.theta <= 10.0, "Threshold exceeded max"

    for _ in range(1000):
        goodness = torch.zeros(100)  # Very low → decrease θ
        threshold.update(goodness)

    assert threshold.theta >= 0.1, "Threshold below min"

def test_per_layer_independence():
    """Each layer should have independent threshold."""
    ff = ForwardForwardNetwork(config)

    # Different inputs to different layers
    ff.forward_layer(torch.randn(10, 512), layer_idx=0)
    ff.forward_layer(torch.ones(10, 512) * 10, layer_idx=1)

    # Thresholds should differ after adaptation
    assert ff.thresholds[0].theta != ff.thresholds[1].theta
```

**Success Criteria**:
- [ ] Firing rate converges to ±5% of target within 1000 steps
- [ ] Thresholds stay within [min, max] bounds
- [ ] Each layer adapts independently
- [ ] No gradient discontinuities from threshold updates

---

#### W1-03: Energy Landscape Validation (Bengio)

**File**: `src/t4dm/nca/energy_validation.py`

**Evidence Base**: Hopfield & Tank (1986), Bengio (2019) "On the Measure of Intelligence"

**Implementation**:
```python
@dataclass
class EnergyValidationResult:
    """Result of energy convergence validation."""
    wake_slope: float  # dE/dt during wake (should be < 0)
    sleep_variance: float  # Var(E) during sleep (should be low)
    convergence_time: int  # Steps to reach attractor
    attractor_depth: float  # Energy at attractor basin
    is_valid: bool  # Overall validation pass

class EnergyLandscapeValidator:
    """Validate Hopfield energy dynamics match biological constraints."""

    def __init__(self, energy_fn: Callable):
        self.energy_fn = energy_fn
        self.E_history = []

    def check_convergence(
        self,
        state: torch.Tensor,
        phase: str,  # "wake" or "sleep"
        steps: int = 1000
    ) -> EnergyValidationResult:
        """
        Verify energy dynamics match expected behavior.

        Wake: E should decrease (minimize prediction error)
        Sleep: E should stabilize (settled in attractor)
        """
        self.E_history = []

        for step in range(steps):
            E = self.energy_fn(state)
            self.E_history.append(E.item())
            state = self._dynamics_step(state)

        E_array = np.array(self.E_history)

        if phase == "wake":
            # Expect E to decrease
            slope = np.polyfit(range(len(E_array)), E_array, 1)[0]
            is_valid = slope < 0
            return EnergyValidationResult(
                wake_slope=slope,
                sleep_variance=np.var(E_array),
                convergence_time=self._find_convergence(E_array),
                attractor_depth=E_array[-1],
                is_valid=is_valid,
            )
        else:  # sleep
            # Expect E to stabilize (low variance in last 50%)
            late_E = E_array[len(E_array)//2:]
            variance = np.var(late_E)
            is_valid = variance < 0.01  # Small variance = stable
            return EnergyValidationResult(
                wake_slope=0,
                sleep_variance=variance,
                convergence_time=self._find_convergence(E_array),
                attractor_depth=np.mean(late_E),
                is_valid=is_valid,
            )

    def _find_convergence(self, E_array: np.ndarray, threshold: float = 0.001) -> int:
        """Find step where E stabilizes within threshold."""
        for i in range(len(E_array) - 10):
            window = E_array[i:i+10]
            if np.std(window) < threshold:
                return i
        return len(E_array)  # Did not converge
```

**Tests**: `tests/unit/nca/test_energy_validation.py`
```python
def test_wake_energy_decreases():
    """During wake, energy should decrease as predictions improve."""
    validator = EnergyLandscapeValidator(hopfield_energy)
    state = torch.randn(1024)

    result = validator.check_convergence(state, phase="wake", steps=500)

    assert result.wake_slope < 0, "Energy should decrease during wake"
    assert result.is_valid, "Wake phase validation failed"

def test_sleep_energy_stabilizes():
    """During sleep, energy should stabilize in attractor."""
    validator = EnergyLandscapeValidator(hopfield_energy)
    state = torch.randn(1024)

    result = validator.check_convergence(state, phase="sleep", steps=500)

    assert result.sleep_variance < 0.01, "Energy should stabilize during sleep"
    assert result.is_valid, "Sleep phase validation failed"

def test_attractor_depth_reflects_memory_strength():
    """Deeper attractors = stronger memories."""
    validator = EnergyLandscapeValidator(hopfield_energy)

    # Strong memory (well-consolidated)
    strong_memory = torch.ones(1024)
    result_strong = validator.check_convergence(strong_memory, "wake")

    # Weak memory (noise)
    weak_memory = torch.randn(1024) * 0.1
    result_weak = validator.check_convergence(weak_memory, "wake")

    assert result_strong.attractor_depth < result_weak.attractor_depth, \
        "Strong memories should have deeper attractors (lower energy)"
```

**Success Criteria**:
- [ ] Wake phase: dE/dt < 0 (energy decreasing)
- [ ] Sleep phase: Var(E) < 0.01 (energy stable)
- [ ] Attractor depth correlates with memory strength (r > 0.8)
- [ ] Convergence within 500 steps

---

#### W1-04: Adaptive Consolidation Trigger (O'Reilly)

**File**: `src/t4dm/consolidation/adaptive_trigger.py`

**Evidence Base**: O'Reilly et al. (2014) "Complementary Learning Systems"

**Implementation**:
```python
@dataclass
class AdaptiveConsolidationConfig:
    """CLS-based consolidation triggering."""
    memtable_saturation_threshold: float = 0.7  # Trigger at 70% full
    encoding_rate_threshold: float = 10.0  # Memories/minute
    adenosine_threshold: float = 0.6  # Default sleep pressure
    min_interval_seconds: float = 300  # Minimum 5 min between consolidations

class AdaptiveConsolidationTrigger:
    """
    CLS principle: consolidate when fast learner (MemTable) saturates.

    Don't wait for adenosine pressure if:
    - MemTable is filling faster than consolidation
    - Encoding rate exceeds capacity
    """

    def __init__(self, config: AdaptiveConsolidationConfig, engine: T4DXEngine):
        self.config = config
        self.engine = engine
        self.last_consolidation = 0
        self.encoding_times = deque(maxlen=100)

    def should_trigger(self, adenosine_pressure: float) -> ConsolidationTrigger:
        """
        Decide whether to trigger consolidation.

        Returns trigger reason or None.
        """
        now = time.time()

        # Respect minimum interval
        if now - self.last_consolidation < self.config.min_interval_seconds:
            return None

        # Check MemTable saturation
        memtable_size = self.engine.memtable_size()
        max_size = self.engine.max_memtable_size
        saturation = memtable_size / max_size

        if saturation > self.config.memtable_saturation_threshold:
            return ConsolidationTrigger(
                reason="memtable_saturation",
                urgency=saturation,
                phase="nrem",  # Start with NREM for fast learner
            )

        # Check encoding rate
        encoding_rate = self._compute_encoding_rate()
        if encoding_rate > self.config.encoding_rate_threshold:
            return ConsolidationTrigger(
                reason="high_encoding_rate",
                urgency=encoding_rate / self.config.encoding_rate_threshold,
                phase="nrem",
            )

        # Default: use adenosine pressure
        if adenosine_pressure > self.config.adenosine_threshold:
            return ConsolidationTrigger(
                reason="adenosine_pressure",
                urgency=adenosine_pressure,
                phase="full",  # Full sleep cycle
            )

        return None

    def record_encoding(self):
        """Record an encoding event for rate tracking."""
        self.encoding_times.append(time.time())

    def _compute_encoding_rate(self) -> float:
        """Compute encodings per minute."""
        if len(self.encoding_times) < 2:
            return 0.0

        duration = self.encoding_times[-1] - self.encoding_times[0]
        if duration < 1:
            return 0.0

        return len(self.encoding_times) / (duration / 60)
```

**Tests**: `tests/unit/consolidation/test_adaptive_trigger.py`
```python
def test_trigger_on_memtable_saturation():
    """Should trigger consolidation when MemTable is 70%+ full."""
    engine = MockEngine(memtable_size=7000, max_size=10000)  # 70%
    trigger = AdaptiveConsolidationTrigger(config, engine)

    result = trigger.should_trigger(adenosine_pressure=0.3)  # Low adenosine

    assert result is not None
    assert result.reason == "memtable_saturation"
    assert result.phase == "nrem"

def test_trigger_on_high_encoding_rate():
    """Should trigger when encoding rate exceeds threshold."""
    engine = MockEngine(memtable_size=1000, max_size=10000)  # Low saturation
    trigger = AdaptiveConsolidationTrigger(config, engine)

    # Simulate 20 encodings in 1 minute (above threshold of 10/min)
    for _ in range(20):
        trigger.record_encoding()

    result = trigger.should_trigger(adenosine_pressure=0.3)

    assert result is not None
    assert result.reason == "high_encoding_rate"

def test_respects_minimum_interval():
    """Should not trigger within min_interval of last consolidation."""
    trigger = AdaptiveConsolidationTrigger(config, engine)
    trigger.last_consolidation = time.time()  # Just consolidated

    result = trigger.should_trigger(adenosine_pressure=0.9)  # High pressure

    assert result is None, "Should respect minimum interval"

def test_default_to_adenosine():
    """When no urgency, use adenosine pressure."""
    engine = MockEngine(memtable_size=1000, max_size=10000)  # Low saturation
    trigger = AdaptiveConsolidationTrigger(config, engine)
    trigger.last_consolidation = 0  # Long ago

    result = trigger.should_trigger(adenosine_pressure=0.8)  # High

    assert result is not None
    assert result.reason == "adenosine_pressure"
    assert result.phase == "full"
```

**Success Criteria**:
- [ ] Triggers at 70% MemTable saturation
- [ ] Triggers when encoding rate > 10/min
- [ ] Respects 5-minute minimum interval
- [ ] Falls back to adenosine when no urgency

---

#### W1-05: Verify LearnedRetrievalScorer Integration (Graves)

**File**: `tests/integration/test_retrieval_pipeline.py`

**Evidence Base**: Graves (2016) "Hybrid computing using a neural network with dynamic external memory"

**Implementation**: Verify existing integration, add tests

```python
def test_retrieval_pipeline_uses_learned_scorer():
    """
    Verify the complete retrieval pipeline:
    1. Query embedding
    2. HNSW rough search (top 100)
    3. LearnedRetrievalScorer refines (top 10)
    4. Spike attention selects final
    """
    client = T4DMClient(session_id="test")

    # Store some memories
    for i in range(50):
        client.store_memory(f"Memory about topic {i % 5}", importance=0.5)

    # Query
    results = client.search("topic 2", k=5)

    # Verify scorer was used (check logs or metrics)
    metrics = client.get_retrieval_metrics()

    assert metrics["hnsw_candidates"] == 100, "HNSW should retrieve 100 candidates"
    assert metrics["scorer_reranked"] == True, "LearnedRetrievalScorer should rerank"
    assert metrics["final_count"] == 5, "Should return k=5 results"

    # Verify relevance ordering
    for i in range(len(results) - 1):
        assert results[i].score >= results[i+1].score, \
            "Results should be ordered by score"

def test_scorer_improves_retrieval_quality():
    """LearnedRetrievalScorer should improve over raw HNSW."""
    # Store memories with known relevance
    client = T4DMClient(session_id="test")
    client.store_memory("Python is a programming language", importance=0.9)
    client.store_memory("Java is a programming language", importance=0.5)
    client.store_memory("Coffee is a drink", importance=0.9)  # Distractor

    # Query about programming
    results_with_scorer = client.search("programming languages", k=2, use_scorer=True)
    results_without_scorer = client.search("programming languages", k=2, use_scorer=False)

    # With scorer, should prefer programming-related results
    with_prog_count = sum(1 for r in results_with_scorer if "programming" in r.content)
    without_prog_count = sum(1 for r in results_without_scorer if "programming" in r.content)

    assert with_prog_count >= without_prog_count, \
        "Scorer should improve relevance"
```

**Success Criteria**:
- [ ] HNSW retrieves 100 candidates
- [ ] LearnedRetrievalScorer reranks candidates
- [ ] Final results are properly ordered by score
- [ ] Scorer improves relevance vs raw HNSW (≥10% improvement on test set)

---

## Wave 2: Uncertainty (Week 3-4)

### Gate Criteria
- All Wave 1 tests pass
- Wave 2 tests pass
- Uncertainty estimates correlate with prediction errors
- Reconsolidation triggers on memory mismatch

### Atoms

#### W2-01: Uncertainty-Aware Memory Storage (Friston)

**File**: `src/t4dm/storage/t4dx/uncertainty.py`

**Evidence Base**: Friston (2010) Free Energy Principle

**Implementation**:
```python
@dataclass
class UncertaintyAwareItem:
    """Memory item with uncertainty quantification."""
    id: UUID
    vector_mean: np.ndarray  # 1024-dim embedding mean
    vector_cov: np.ndarray  # 1024x1024 covariance (or diagonal)
    content: str
    kappa: float
    importance: float

    @property
    def uncertainty(self) -> float:
        """Scalar uncertainty measure (trace of covariance)."""
        if self.vector_cov.ndim == 1:
            return float(np.sum(self.vector_cov))
        return float(np.trace(self.vector_cov))

    @property
    def confidence(self) -> float:
        """Confidence = 1 / (1 + uncertainty)."""
        return 1.0 / (1.0 + self.uncertainty)

class UncertaintyEstimator:
    """Estimate embedding uncertainty via dropout or ensemble."""

    def __init__(self, embedding_model, method: str = "mc_dropout"):
        self.model = embedding_model
        self.method = method
        self.n_samples = 10

    def embed_with_uncertainty(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute embedding mean and covariance.

        Uses MC Dropout: sample embeddings with dropout enabled,
        compute mean and variance across samples.
        """
        if self.method == "mc_dropout":
            self.model.train()  # Enable dropout
            samples = []

            for _ in range(self.n_samples):
                emb = self.model.encode(text)
                samples.append(emb)

            self.model.eval()
            samples = np.stack(samples)

            mean = np.mean(samples, axis=0)
            # Diagonal covariance for efficiency
            var = np.var(samples, axis=0)

            return mean, var
        else:
            raise ValueError(f"Unknown method: {self.method}")
```

**Tests**: `tests/unit/storage/test_uncertainty.py`
```python
def test_uncertainty_higher_for_ambiguous_text():
    """Ambiguous text should have higher uncertainty."""
    estimator = UncertaintyEstimator(embedding_model)

    # Clear, specific text
    clear_mean, clear_var = estimator.embed_with_uncertainty(
        "Python is a programming language created by Guido van Rossum"
    )

    # Ambiguous text
    ambig_mean, ambig_var = estimator.embed_with_uncertainty(
        "It might be something"
    )

    assert np.sum(ambig_var) > np.sum(clear_var), \
        "Ambiguous text should have higher uncertainty"

def test_confidence_correlates_with_accuracy():
    """High confidence should correlate with correct retrieval."""
    # Store items with uncertainty
    for text in test_texts:
        mean, var = estimator.embed_with_uncertainty(text)
        engine.insert(UncertaintyAwareItem(
            vector_mean=mean,
            vector_cov=var,
            ...
        ))

    # Query and measure accuracy
    results = engine.search_with_uncertainty(query)

    # High confidence results should be more accurate
    high_conf = [r for r in results if r.confidence > 0.8]
    low_conf = [r for r in results if r.confidence < 0.5]

    assert accuracy(high_conf) > accuracy(low_conf), \
        "High confidence should correlate with accuracy"
```

**Success Criteria**:
- [ ] MC Dropout produces valid covariance estimates
- [ ] Ambiguous text has higher uncertainty than clear text
- [ ] Confidence correlates with retrieval accuracy (r > 0.6)
- [ ] Storage overhead < 2x (diagonal covariance)

---

#### W2-02: Reconsolidation Lability (O'Reilly)

**File**: `src/t4dm/learning/reconsolidation.py`

**Evidence Base**: Nader et al. (2000) "Fear memories require protein synthesis"

**Implementation**:
```python
@dataclass
class ReconsolidationConfig:
    """Configuration for memory reconsolidation."""
    lability_window_seconds: float = 300  # 5 minutes
    kappa_drop_threshold: float = 0.7  # Only affect consolidated memories
    kappa_drop_amount: float = 0.4  # How much to reduce κ
    mismatch_threshold: float = 0.3  # Cosine distance for mismatch

class ReconsolidationManager:
    """
    Implements memory reconsolidation with lability windows.

    When a consolidated memory (κ > 0.7) is reactivated and there's
    a mismatch with current experience, temporarily reduce κ to make
    the memory labile for updating during next consolidation.
    """

    def __init__(self, config: ReconsolidationConfig, engine: T4DXEngine):
        self.config = config
        self.engine = engine
        self.labile_memories: dict[UUID, float] = {}  # id → lability_expires_at

    def on_reactivation(
        self,
        memory_id: UUID,
        current_context_embedding: np.ndarray
    ) -> ReconsolidationResult:
        """
        Called when a memory is retrieved/reactivated.

        Checks for mismatch and potentially opens lability window.
        """
        memory = self.engine.get(memory_id)

        # Only affect consolidated memories
        if memory.kappa < self.config.kappa_drop_threshold:
            return ReconsolidationResult(triggered=False, reason="not_consolidated")

        # Check mismatch
        mismatch = 1 - cosine_similarity(memory.vector, current_context_embedding)

        if mismatch > self.config.mismatch_threshold:
            # Open lability window
            expires_at = time.time() + self.config.lability_window_seconds
            self.labile_memories[memory_id] = expires_at

            # Temporarily reduce κ
            new_kappa = memory.kappa - self.config.kappa_drop_amount
            self.engine.update_fields(memory_id, {
                "kappa": new_kappa,
                "labile": True,
            })

            return ReconsolidationResult(
                triggered=True,
                reason="mismatch_detected",
                mismatch_score=mismatch,
                new_kappa=new_kappa,
                lability_expires=expires_at,
            )

        return ReconsolidationResult(triggered=False, reason="no_mismatch")

    def close_lability_windows(self):
        """Close expired lability windows, restore κ if not updated."""
        now = time.time()
        expired = [mid for mid, exp in self.labile_memories.items() if exp < now]

        for memory_id in expired:
            memory = self.engine.get(memory_id)
            if memory.labile:
                # Lability window closed without update → restore κ
                self.engine.update_fields(memory_id, {
                    "kappa": memory.kappa + self.config.kappa_drop_amount,
                    "labile": False,
                })
            del self.labile_memories[memory_id]
```

**Tests**: `tests/unit/learning/test_reconsolidation.py`
```python
def test_reconsolidation_on_mismatch():
    """Mismatch should open lability window."""
    manager = ReconsolidationManager(config, engine)

    # Create consolidated memory
    memory_id = engine.insert(content="Cats are mammals", kappa=0.9)

    # Reactivate with mismatched context (cats are reptiles)
    reptile_context = embed("Reptiles are cold-blooded")
    result = manager.on_reactivation(memory_id, reptile_context)

    assert result.triggered, "Should trigger on mismatch"
    assert result.reason == "mismatch_detected"

    # Verify κ dropped
    memory = engine.get(memory_id)
    assert memory.kappa == 0.5, "κ should drop by 0.4"
    assert memory.labile == True, "Should be marked labile"

def test_no_reconsolidation_for_low_kappa():
    """Low-κ memories should not trigger reconsolidation."""
    manager = ReconsolidationManager(config, engine)

    # Create unconsolidated memory
    memory_id = engine.insert(content="New memory", kappa=0.2)

    # Reactivate with mismatch
    result = manager.on_reactivation(memory_id, mismatched_context)

    assert not result.triggered
    assert result.reason == "not_consolidated"

def test_lability_window_expires():
    """Lability window should close after timeout."""
    manager = ReconsolidationManager(config, engine)
    memory_id = engine.insert(content="Test", kappa=0.9)

    # Trigger reconsolidation
    manager.on_reactivation(memory_id, mismatched_context)

    # Wait for window to expire
    time.sleep(config.lability_window_seconds + 1)
    manager.close_lability_windows()

    # κ should be restored
    memory = engine.get(memory_id)
    assert memory.kappa == 0.9, "κ should be restored"
    assert memory.labile == False
```

**Success Criteria**:
- [ ] Mismatch > 0.3 triggers lability
- [ ] κ drops by 0.4 when labile
- [ ] Window closes after 5 minutes
- [ ] κ restored if not updated during window

---

#### W2-03: Learned Embedding Alignment (Graves)

**File**: `src/t4dm/qwen/alignment.py`

**Evidence Base**: Graves (2014) "Neural Turing Machines"

**Implementation**:
```python
class EmbeddingAlignment(nn.Module):
    """
    Learnable alignment layer between Qwen hidden states and BGE-M3 embedding space.

    Addresses the gap between frozen LLM features and memory retrieval needs.
    """

    def __init__(
        self,
        qwen_dim: int = 2048,
        bge_dim: int = 1024,
        hidden_dim: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(qwen_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bge_dim),
        )

        # Learnable importance weights per dimension
        self.dimension_importance = nn.Parameter(torch.ones(bge_dim))

    def forward(self, qwen_hidden: torch.Tensor) -> torch.Tensor:
        """
        Align Qwen hidden states to BGE-M3 space.

        Args:
            qwen_hidden: [batch, seq, 2048] from Qwen layer 18

        Returns:
            aligned: [batch, 1024] aligned embedding
        """
        # Pool over sequence
        pooled = qwen_hidden.mean(dim=1)  # [batch, 2048]

        # Project to BGE space
        projected = self.projection(pooled)  # [batch, 1024]

        # Apply learned dimension importance
        aligned = projected * self.dimension_importance

        # Normalize
        aligned = F.normalize(aligned, dim=-1)

        return aligned

class AlignmentTrainer:
    """Train alignment layer to match Qwen→retrieval with BGE→retrieval."""

    def __init__(self, alignment: EmbeddingAlignment, bge_model):
        self.alignment = alignment
        self.bge_model = bge_model
        self.optimizer = torch.optim.AdamW(alignment.parameters(), lr=1e-4)

    def train_step(
        self,
        qwen_hidden: torch.Tensor,
        text: str,
        relevant_memory_ids: list[UUID],
    ) -> float:
        """
        Train alignment to match BGE retrieval ranking.

        Loss = ranking loss between aligned retrieval and BGE retrieval.
        """
        self.optimizer.zero_grad()

        # Get aligned embedding
        aligned = self.alignment(qwen_hidden)

        # Get BGE embedding
        bge_emb = self.bge_model.encode(text)

        # Compute retrieval rankings
        aligned_scores = self._compute_retrieval_scores(aligned, relevant_memory_ids)
        bge_scores = self._compute_retrieval_scores(bge_emb, relevant_memory_ids)

        # ListMLE ranking loss
        loss = listMLE_loss(aligned_scores, bge_scores)

        loss.backward()
        self.optimizer.step()

        return loss.item()
```

**Tests**: `tests/unit/qwen/test_alignment.py`
```python
def test_alignment_dimension_match():
    """Aligned output should match BGE dimension."""
    alignment = EmbeddingAlignment(qwen_dim=2048, bge_dim=1024)
    qwen_hidden = torch.randn(2, 128, 2048)  # [batch, seq, hidden]

    aligned = alignment(qwen_hidden)

    assert aligned.shape == (2, 1024), f"Expected [2, 1024], got {aligned.shape}"

def test_alignment_improves_retrieval():
    """Aligned embeddings should retrieve better than raw projection."""
    alignment = EmbeddingAlignment()
    trainer = AlignmentTrainer(alignment, bge_model)

    # Train for some steps
    initial_ndcg = evaluate_retrieval(alignment, test_set)

    for _ in range(100):
        trainer.train_step(qwen_hidden, text, relevant_ids)

    final_ndcg = evaluate_retrieval(alignment, test_set)

    assert final_ndcg > initial_ndcg, "Training should improve retrieval"

def test_dimension_importance_learns():
    """Dimension importance should diverge from uniform."""
    alignment = EmbeddingAlignment()
    trainer = AlignmentTrainer(alignment, bge_model)

    initial_importance = alignment.dimension_importance.clone()

    for _ in range(100):
        trainer.train_step(qwen_hidden, text, relevant_ids)

    final_importance = alignment.dimension_importance

    # Should have learned non-uniform importance
    assert not torch.allclose(initial_importance, final_importance, atol=0.01), \
        "Dimension importance should diverge"
```

**Success Criteria**:
- [ ] Output dimension matches BGE-M3 (1024)
- [ ] Training improves NDCG@10 by ≥5%
- [ ] Dimension importance diverges from uniform
- [ ] Alignment adds <1ms latency

---

#### W2-04: Dream-Based Negative Samples (Hinton)

**File**: `src/t4dm/learning/dream_negatives.py`

**Evidence Base**: Hinton (2022) "The Forward-Forward Algorithm"

**Implementation**:
```python
class DreamNegativeGenerator:
    """
    Generate negative samples for FF learning using VAE dreams.

    More realistic negatives than random noise → tighter decision boundaries.
    """

    def __init__(self, vae: VAEGenerator, corruption_rate: float = 0.3):
        self.vae = vae
        self.corruption_rate = corruption_rate

    def generate(
        self,
        positive_data: torch.Tensor,
        method: str = "vae_sample"
    ) -> torch.Tensor:
        """
        Generate negative samples from positive data.

        Methods:
        - vae_sample: Sample from VAE conditioned on positive
        - shuffle: Shuffle features across batch
        - hybrid: Mix VAE with random noise
        """
        if method == "vae_sample":
            # Encode positive to latent
            mu, logvar = self.vae.encode(positive_data)

            # Sample with higher variance (more dreaming)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) * 2

            # Decode
            negative = self.vae.decode(z)

            # Corrupt to ensure negativity
            mask = torch.rand_like(negative) < self.corruption_rate
            negative[mask] = torch.randn_like(negative[mask])

            return negative

        elif method == "shuffle":
            # Shuffle within batch
            idx = torch.randperm(positive_data.size(0))
            return positive_data[idx]

        elif method == "hybrid":
            vae_neg = self.generate(positive_data, "vae_sample")
            shuffle_neg = self.generate(positive_data, "shuffle")

            # Mix 50/50
            mask = torch.rand(positive_data.size(0)) < 0.5
            negative = torch.where(mask.unsqueeze(-1), vae_neg, shuffle_neg)

            return negative

        raise ValueError(f"Unknown method: {method}")
```

**Tests**: `tests/unit/learning/test_dream_negatives.py`
```python
def test_vae_negatives_realistic():
    """VAE negatives should be more realistic than random noise."""
    generator = DreamNegativeGenerator(vae)
    positive = torch.randn(32, 1024)

    vae_negative = generator.generate(positive, method="vae_sample")
    random_negative = torch.randn_like(positive)

    # VAE negatives should be closer to positive manifold
    vae_dist = torch.norm(positive - vae_negative, dim=-1).mean()
    random_dist = torch.norm(positive - random_negative, dim=-1).mean()

    assert vae_dist < random_dist, "VAE negatives should be closer to positive"

def test_negatives_sufficiently_different():
    """Negatives should be sufficiently different from positive."""
    generator = DreamNegativeGenerator(vae)
    positive = torch.randn(32, 1024)

    negative = generator.generate(positive, method="vae_sample")

    similarity = F.cosine_similarity(positive, negative, dim=-1).mean()

    assert similarity < 0.8, "Negatives should be different from positive"

def test_ff_benefits_from_dream_negatives():
    """FF should learn better with dream negatives than random."""
    ff_dream = ForwardForwardNetwork(negative_method="vae_sample")
    ff_random = ForwardForwardNetwork(negative_method="random")

    # Train both
    for batch in train_loader:
        ff_dream.train_step(batch)
        ff_random.train_step(batch)

    # Evaluate
    acc_dream = evaluate(ff_dream, test_loader)
    acc_random = evaluate(ff_random, test_loader)

    assert acc_dream >= acc_random, "Dream negatives should help or not hurt"
```

**Success Criteria**:
- [ ] VAE negatives closer to manifold than random
- [ ] Negatives have <0.8 cosine similarity to positive
- [ ] FF accuracy with dreams ≥ FF with random
- [ ] Generation adds <5ms latency

---

## Wave 3: Intelligence (Week 5-6)

### Gate Criteria
- All Wave 2 tests pass
- Markov blanket retrieval reduces irrelevant results
- Edge type learning improves traversal
- Generalization quality correlates with silhouette score

### Atoms

#### W3-01: Markov Blanket Retrieval (Friston)

**File**: `src/t4dm/storage/t4dx/markov_retrieval.py`

**Evidence Base**: Pearl (1988) "Probabilistic Reasoning in Intelligent Systems"

**Implementation**:
```python
class MarkovBlanketRetriever:
    """
    Retrieve memories prioritizing Markov blanket of query concept.

    Markov blanket = parents + children + spouses (co-parents)
    p(X | MB(X)) ⊥ p(X | non-MB) - memories outside MB are irrelevant.
    """

    def __init__(self, engine: T4DXEngine, exploration_ratio: float = 0.1):
        self.engine = engine
        self.exploration_ratio = exploration_ratio

    def search(
        self,
        query_vector: np.ndarray,
        query_concept: Optional[UUID] = None,
        k: int = 10,
    ) -> list[MemoryItem]:
        """
        Retrieve k memories, prioritizing Markov blanket.

        90% from MB, 10% from global (exploration).
        """
        if query_concept is None:
            # No concept provided, use pure vector search
            return self.engine.search(query_vector, k)

        # Get Markov blanket
        mb = self._get_markov_blanket(query_concept)

        # Retrieve from MB
        mb_k = int(k * (1 - self.exploration_ratio))
        mb_results = self._search_in_set(query_vector, mb, mb_k)

        # Retrieve from global (exploration)
        explore_k = k - len(mb_results)
        global_results = self.engine.search(query_vector, explore_k * 2)

        # Filter out MB items from global
        global_results = [r for r in global_results if r.id not in mb]
        global_results = global_results[:explore_k]

        # Combine
        return mb_results + global_results

    def _get_markov_blanket(self, concept_id: UUID) -> set[UUID]:
        """Get Markov blanket: parents + children + spouses."""
        # Parents: concepts that CAUSE this one
        parents = self.engine.traverse(
            concept_id, edge_type="CAUSES", direction="incoming", depth=1
        )

        # Children: concepts caused by this one
        children = self.engine.traverse(
            concept_id, edge_type="CAUSES", direction="outgoing", depth=1
        )

        # Spouses: other parents of children
        spouses = set()
        for child in children:
            child_parents = self.engine.traverse(
                child.id, edge_type="CAUSES", direction="incoming", depth=1
            )
            spouses.update(p.id for p in child_parents if p.id != concept_id)

        return parents | children | spouses
```

**Tests**: `tests/unit/storage/test_markov_retrieval.py`
```python
def test_markov_blanket_correctly_computed():
    """MB should include parents, children, and spouses."""
    engine = create_test_graph()
    # A → B → D
    # A → C → D
    # B is spouse of C (both parents of D)

    retriever = MarkovBlanketRetriever(engine)
    mb = retriever._get_markov_blanket(concept_id=B)

    assert A in mb, "Parent A should be in MB"
    assert D in mb, "Child D should be in MB"
    assert C in mb, "Spouse C should be in MB"

def test_mb_retrieval_more_relevant():
    """MB retrieval should return more relevant results."""
    retriever = MarkovBlanketRetriever(engine)

    # Query with known concept
    results_mb = retriever.search(query_vector, query_concept=concept_id, k=10)
    results_global = engine.search(query_vector, k=10)

    # MB results should have higher average relevance
    mb_relevance = np.mean([r.relevance for r in results_mb])
    global_relevance = np.mean([r.relevance for r in results_global])

    assert mb_relevance >= global_relevance, \
        "MB retrieval should be at least as relevant"

def test_exploration_preserves_diversity():
    """10% exploration should maintain result diversity."""
    retriever = MarkovBlanketRetriever(engine, exploration_ratio=0.1)

    results = retriever.search(query_vector, query_concept=concept_id, k=10)

    # At least 1 result should be from outside MB
    mb = retriever._get_markov_blanket(concept_id)
    non_mb_count = sum(1 for r in results if r.id not in mb)

    assert non_mb_count >= 1, "Should have some exploration results"
```

**Success Criteria**:
- [ ] MB correctly computes parents + children + spouses
- [ ] MB retrieval relevance ≥ global retrieval
- [ ] 10% exploration maintains diversity
- [ ] MB computation adds <5ms latency

---

#### W3-02: Edge Type Importance Learning (Graves)

**File**: `src/t4dm/storage/t4dx/learned_edges.py`

**Implementation**:
```python
class LearnedEdgeImportance(nn.Module):
    """
    Learn importance weights for each edge type.

    During traversal, weight edges by learned importance.
    """

    EDGE_TYPES = [
        "CAUSES", "TEMPORAL_BEFORE", "PART_OF", "SIMILAR_TO",
        "CONTRADICTS", "ELABORATES", "EXEMPLIFIES", "GENERALIZES",
        "PRECONDITION", "EFFECT", "ATTRIBUTE", "CONTEXT",
        "REFERENCES", "DERIVES_FROM", "SUPPORTS", "OPPOSES", "NEUTRAL"
    ]

    def __init__(self):
        super().__init__()
        self.importance = nn.Embedding(len(self.EDGE_TYPES), 1)
        nn.init.ones_(self.importance.weight)  # Start uniform

    def get_weight(self, edge_type: str) -> float:
        """Get learned importance weight for edge type."""
        idx = self.EDGE_TYPES.index(edge_type)
        return torch.exp(self.importance(torch.tensor(idx))).item()

    def forward(self, edge_types: torch.Tensor) -> torch.Tensor:
        """Get importance weights for batch of edge types."""
        return torch.exp(self.importance(edge_types)).squeeze(-1)

class TraversalWithLearnedEdges:
    """Traversal that weights edges by learned importance."""

    def __init__(self, engine: T4DXEngine, edge_importance: LearnedEdgeImportance):
        self.engine = engine
        self.edge_importance = edge_importance

    def traverse(
        self,
        start_id: UUID,
        depth: int = 2,
        min_weight: float = 0.1,
    ) -> list[MemoryItem]:
        """
        BFS traversal with learned edge weighting.

        Candidate score = base_score × edge_importance
        """
        visited = {start_id}
        frontier = [(start_id, 1.0)]  # (id, accumulated_weight)
        results = []

        for d in range(depth):
            new_frontier = []

            for node_id, acc_weight in frontier:
                edges = self.engine.get_edges(node_id)

                for edge in edges:
                    if edge.target_id in visited:
                        continue

                    # Apply learned importance
                    edge_weight = self.edge_importance.get_weight(edge.edge_type)
                    new_acc = acc_weight * edge_weight * edge.weight

                    if new_acc >= min_weight:
                        visited.add(edge.target_id)
                        new_frontier.append((edge.target_id, new_acc))

                        target = self.engine.get(edge.target_id)
                        target.traversal_score = new_acc
                        results.append(target)

            frontier = new_frontier

        return sorted(results, key=lambda x: x.traversal_score, reverse=True)
```

**Tests**: `tests/unit/storage/test_learned_edges.py`
```python
def test_edge_importance_learns():
    """Edge importance should diverge from uniform after training."""
    edge_importance = LearnedEdgeImportance()
    trainer = EdgeImportanceTrainer(edge_importance)

    initial_weights = [edge_importance.get_weight(e) for e in edge_importance.EDGE_TYPES]

    # Train on retrieval feedback
    for _ in range(100):
        trainer.train_step(traversal_result, relevance_feedback)

    final_weights = [edge_importance.get_weight(e) for e in edge_importance.EDGE_TYPES]

    assert not np.allclose(initial_weights, final_weights, atol=0.01), \
        "Edge weights should diverge from uniform"

def test_important_edges_get_higher_weight():
    """Edges that lead to relevant results should get higher weight."""
    edge_importance = LearnedEdgeImportance()
    trainer = EdgeImportanceTrainer(edge_importance)

    # Simulate: CAUSES edges lead to relevant results
    for _ in range(100):
        trainer.train_step(
            traversal_edges=["CAUSES", "SIMILAR_TO"],
            relevance=[0.9, 0.3],  # CAUSES more relevant
        )

    causes_weight = edge_importance.get_weight("CAUSES")
    similar_weight = edge_importance.get_weight("SIMILAR_TO")

    assert causes_weight > similar_weight, \
        "CAUSES should get higher weight"

def test_traversal_prefers_important_edges():
    """Traversal should prefer paths with high-importance edges."""
    edge_importance = LearnedEdgeImportance()
    edge_importance.importance.weight.data[0] = 2.0  # CAUSES = high
    edge_importance.importance.weight.data[3] = 0.1  # SIMILAR_TO = low

    traversal = TraversalWithLearnedEdges(engine, edge_importance)
    results = traversal.traverse(start_id, depth=2)

    # Results via CAUSES should be ranked higher
    causes_results = [r for r in results if r.via_edge == "CAUSES"]
    similar_results = [r for r in results if r.via_edge == "SIMILAR_TO"]

    if causes_results and similar_results:
        avg_causes_rank = np.mean([results.index(r) for r in causes_results])
        avg_similar_rank = np.mean([results.index(r) for r in similar_results])
        assert avg_causes_rank < avg_similar_rank
```

**Success Criteria**:
- [ ] Edge weights diverge from uniform after training
- [ ] Relevant edges get higher weights
- [ ] Traversal ranks high-importance edges first
- [ ] Edge importance lookup <0.1ms

---

#### W3-03: Generalization Quality Scoring (O'Reilly)

**File**: `src/t4dm/consolidation/generalization.py`

**Implementation**:
```python
class GeneralizationQualityScorer:
    """
    Compute generalization quality of REM-created prototypes.

    Uses silhouette score: high separation = good generalization.
    """

    def __init__(self, min_quality: float = 0.3):
        self.min_quality = min_quality

    def score_cluster(
        self,
        cluster_vectors: np.ndarray,
        all_vectors: np.ndarray,
    ) -> GeneralizationResult:
        """
        Compute generalization quality for a cluster.

        Silhouette score in [-1, 1]:
        - +1: Perfectly separated
        -  0: Overlapping
        - -1: Wrong cluster assignment
        """
        if len(cluster_vectors) < 2:
            return GeneralizationResult(quality=0.0, should_generalize=False)

        # Compute silhouette
        labels = np.zeros(len(all_vectors))
        cluster_indices = [self._find_index(v, all_vectors) for v in cluster_vectors]
        labels[cluster_indices] = 1

        if len(np.unique(labels)) < 2:
            return GeneralizationResult(quality=0.0, should_generalize=False)

        silhouette = silhouette_score(all_vectors, labels)

        should_generalize = silhouette >= self.min_quality

        return GeneralizationResult(
            quality=silhouette,
            should_generalize=should_generalize,
            reason="low_separation" if not should_generalize else "high_separation",
        )

    def filter_clusters_for_prototyping(
        self,
        clusters: list[Cluster],
        all_vectors: np.ndarray,
    ) -> list[Cluster]:
        """Filter clusters to only those suitable for prototyping."""
        suitable = []

        for cluster in clusters:
            result = self.score_cluster(cluster.vectors, all_vectors)

            if result.should_generalize:
                cluster.generalization_quality = result.quality
                suitable.append(cluster)
            else:
                logger.info(
                    f"Cluster {cluster.id} not suitable for prototyping: "
                    f"{result.reason} (quality={result.quality:.2f})"
                )

        return suitable
```

**Tests**: `tests/unit/consolidation/test_generalization.py`
```python
def test_high_separation_high_quality():
    """Well-separated clusters should have high quality."""
    scorer = GeneralizationQualityScorer()

    # Two well-separated clusters
    cluster_a = np.random.randn(20, 128) + np.array([5, 0] + [0]*126)
    cluster_b = np.random.randn(20, 128) + np.array([-5, 0] + [0]*126)
    all_vectors = np.vstack([cluster_a, cluster_b])

    result = scorer.score_cluster(cluster_a, all_vectors)

    assert result.quality > 0.5, "Well-separated should have high quality"
    assert result.should_generalize, "Should recommend generalization"

def test_overlapping_low_quality():
    """Overlapping clusters should have low quality."""
    scorer = GeneralizationQualityScorer()

    # Overlapping clusters
    cluster_a = np.random.randn(20, 128)
    cluster_b = np.random.randn(20, 128)  # Same distribution
    all_vectors = np.vstack([cluster_a, cluster_b])

    result = scorer.score_cluster(cluster_a, all_vectors)

    assert result.quality < 0.3, "Overlapping should have low quality"
    assert not result.should_generalize, "Should not recommend generalization"

def test_filter_preserves_good_clusters():
    """Filter should keep high-quality clusters."""
    scorer = GeneralizationQualityScorer(min_quality=0.3)

    good_cluster = Cluster(vectors=well_separated_vectors)
    bad_cluster = Cluster(vectors=overlapping_vectors)

    filtered = scorer.filter_clusters_for_prototyping(
        [good_cluster, bad_cluster], all_vectors
    )

    assert good_cluster in filtered
    assert bad_cluster not in filtered
```

**Success Criteria**:
- [ ] Well-separated clusters have quality > 0.5
- [ ] Overlapping clusters have quality < 0.3
- [ ] Filter correctly preserves good clusters
- [ ] Silhouette computation <10ms for 1000 items

---

#### W3-04: IIT Consciousness Metrics (Bengio)

**File**: `src/t4dm/observability/consciousness_metrics.py`

**Evidence Base**: Tononi (2004) "An information integration theory of consciousness"

**Implementation**:
```python
@dataclass
class ConsciousnessMetrics:
    """Integrated Information Theory metrics."""
    phi: float  # Integrated information (Φ)
    surprise: float  # |E_current - E_previous|
    integration: float  # Mutual info between subsystems
    differentiation: float  # Entropy of activation patterns
    conscious_threshold: float = 0.5

class IITMetricsComputer:
    """
    Compute IIT-inspired consciousness metrics.

    Φ (phi) measures how much information is generated by a system
    above and beyond its parts.
    """

    def __init__(self, energy_fn: Callable):
        self.energy_fn = energy_fn
        self.previous_energy = None

    def compute(
        self,
        spiking_output: torch.Tensor,
        memory_state: torch.Tensor,
    ) -> ConsciousnessMetrics:
        """
        Compute consciousness metrics from current state.

        Φ = Σ I(A; B) for all bipartitions - not exactly computable,
        so we use approximations.
        """
        # Surprise = energy change
        current_energy = self.energy_fn(spiking_output)
        if self.previous_energy is not None:
            surprise = abs(current_energy - self.previous_energy)
        else:
            surprise = 0.0
        self.previous_energy = current_energy

        # Integration = mutual info between spiking and memory
        integration = self._mutual_information(spiking_output, memory_state)

        # Differentiation = entropy of activation pattern
        differentiation = self._entropy(spiking_output)

        # Φ approximation (simplified)
        phi = integration * differentiation

        return ConsciousnessMetrics(
            phi=phi,
            surprise=surprise,
            integration=integration,
            differentiation=differentiation,
        )

    def _mutual_information(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Estimate mutual information via correlation."""
        # Simplified: use correlation as proxy for MI
        x_flat = x.flatten()
        y_flat = y.flatten()[:len(x_flat)]

        corr = torch.corrcoef(torch.stack([x_flat, y_flat]))[0, 1]
        return abs(corr.item())

    def _entropy(self, x: torch.Tensor) -> float:
        """Estimate entropy of activation pattern."""
        # Discretize to bins
        x_flat = x.flatten()
        hist = torch.histc(x_flat, bins=100)
        probs = hist / hist.sum()
        probs = probs[probs > 0]

        entropy = -torch.sum(probs * torch.log(probs + 1e-10))
        return entropy.item()
```

**Tests**: `tests/unit/observability/test_consciousness.py`
```python
def test_phi_increases_with_integration():
    """Φ should increase when subsystems are more integrated."""
    computer = IITMetricsComputer(energy_fn)

    # Independent subsystems
    independent_spiking = torch.randn(100)
    independent_memory = torch.randn(100)
    metrics_independent = computer.compute(independent_spiking, independent_memory)

    # Correlated subsystems
    correlated_spiking = torch.randn(100)
    correlated_memory = correlated_spiking + torch.randn(100) * 0.1
    metrics_correlated = computer.compute(correlated_spiking, correlated_memory)

    assert metrics_correlated.phi > metrics_independent.phi, \
        "Correlated systems should have higher Φ"

def test_surprise_reflects_energy_change():
    """Surprise should be high when energy changes significantly."""
    computer = IITMetricsComputer(energy_fn)

    # First state
    state1 = torch.zeros(100)
    computer.compute(state1, state1)

    # Similar state → low surprise
    state2 = torch.zeros(100) + 0.01
    metrics_low = computer.compute(state2, state2)

    # Very different state → high surprise
    state3 = torch.ones(100) * 10
    metrics_high = computer.compute(state3, state3)

    assert metrics_high.surprise > metrics_low.surprise

def test_differentiation_reflects_pattern_diversity():
    """Differentiation should be high for diverse patterns."""
    computer = IITMetricsComputer(energy_fn)

    # Uniform pattern
    uniform = torch.ones(100)
    metrics_uniform = computer.compute(uniform, uniform)

    # Diverse pattern
    diverse = torch.randn(100)
    metrics_diverse = computer.compute(diverse, diverse)

    assert metrics_diverse.differentiation > metrics_uniform.differentiation
```

**Success Criteria**:
- [ ] Φ increases with subsystem correlation
- [ ] Surprise reflects energy changes
- [ ] Differentiation reflects pattern diversity
- [ ] Metrics computation <5ms

---

## Wave 4: Integration (Week 7-8)

### Gate Criteria
- All Wave 3 tests pass
- Variational consolidation improves prototype quality
- End-to-end pipeline passes all integration tests

### Atoms

#### W4-01: Variational Consolidation Framing (Friston)

Reframe existing consolidation as variational EM:
- NREM = E-step: infer which episodes belong to same cluster
- REM = M-step: update cluster prototype (EM clustering)
- PRUNE = regularization: remove low-posterior items

#### W4-02: Sleep Phase Documentation

Document sleep phases as variational inference with proper mathematical notation.

#### W4-03: End-to-End Pipeline Validation

Integration tests verifying complete learning loop:
1. Encode → store with uncertainty
2. Retrieve → with Markov blanket
3. Outcome → three-factor learning
4. Consolidate → variational EM
5. Verify κ progression

---

## Wave 5: Benchmarks (Week 9-12)

### Gate Criteria
- LongMemEval benchmark completed
- DMR benchmark completed
- Bio-plausibility validation report generated

### Atoms

#### W5-01: LongMemEval Benchmark

**File**: `benchmarks/longmemeval/run.py`

Compare T4DM against Mem0, Letta, Zep on:
- Session-based memory (needle-in-haystack)
- Long-term retention
- Consolidation effectiveness

#### W5-02: DMR Benchmark

**File**: `benchmarks/dmr/run.py`

Deep Memory Retrieval benchmark:
- Measure retrieval accuracy
- Compare κ-gradient vs discrete stores

#### W5-03: Bio-Plausibility Validation

**File**: `benchmarks/bioplausibility/run.py`

Validate against neuroscience literature:
- CLS theory compliance
- Consolidation dynamics
- Neuromodulator effects

---

## Success Metrics Summary

| Wave | Primary Metric | Target |
|------|---------------|--------|
| **Wave 1** | Free Energy decreasing | dF/dt < 0 during wake |
| **Wave 1** | Adaptive thresholds | Firing rate ±5% of target |
| **Wave 1** | Energy validation | Wake slope < 0, sleep variance < 0.01 |
| **Wave 2** | Uncertainty correlation | Confidence-accuracy r > 0.6 |
| **Wave 2** | Reconsolidation | Lability triggers on mismatch |
| **Wave 2** | Alignment | NDCG@10 improves ≥5% |
| **Wave 3** | Markov blanket | Retrieval relevance ≥ global |
| **Wave 3** | Edge importance | Weights diverge from uniform |
| **Wave 3** | Generalization | Quality correlates with silhouette |
| **Wave 4** | End-to-end | All integration tests pass |
| **Wave 5** | LongMemEval | Competitive with Mem0 |
| **Wave 5** | Bio-plausibility | 90%+ compliance |

---

## Appendix: State Machine Updates Required

The following state machine diagrams need updates after implementation:

1. **10_learning_eligibility.mermaid**: Add Free Energy node
2. **02_consolidation_sleep_cycle.mermaid**: Add variational EM labels
3. **04_neuromodulator_states.mermaid**: Add uncertainty-gated transitions
4. **New diagram**: Reconsolidation lability state machine
5. **New diagram**: Markov blanket retrieval flow

---

*Plan created: 2026-02-06*
*Expert sources: Hinton, Bengio, Friston, O'Reilly, Graves panel analysis*
