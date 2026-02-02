# Biological Fixes Implementation Roadmap

**Date**: 2026-01-07
**Status**: Planning Phase
**Score Impact**: 92/100 → 98/100 (addressing 4 minor issues + FF enhancement)

---

## Executive Summary

This roadmap addresses the 4 minor biological issues identified in the validation report and explores biological mechanisms to enhance the Forward-Forward encoder (addressing the "frozen embeddings" concern from Hinton review).

### Issues to Fix (from BIOLOGICAL_VALIDATION_REPORT.md)
1. **VTA DA Decay** (B1): Linear decay vs exponential tau-based (92/100)
2. **STDP Weight Dependence** (B7): Missing multiplicative weight dependence in consolidation
3. **TAN Pause Mechanism** (B5): Absent cholinergic interneuron pause dynamics (90/100)
4. **Astrocyte Gap Junctions** (B8): Missing Ca2+ wave propagation between astrocytes (91/100)

### FF Enhancement Opportunities
- **Hinton Concern**: "Frozen embeddings" limit encoder plasticity
- **Biological Solution**: Implement activity-dependent neurogenesis and dendritic rewiring mechanisms
- **Expected Impact**: Dynamic encoder adaptation during consolidation

---

## Phase 1: Critical Biological Accuracy Fixes (HIGH PRIORITY)

**Estimated Effort**: 3-4 days
**Can Parallelize With**: Architecture work, documentation
**Dependencies**: None

### 1.1: VTA Dopamine Exponential Decay (DAY 1)

**Score Impact**: 92 → 95/100 for B1

#### Files to Modify
- `/mnt/projects/t4d/t4dm/src/t4dm/nca/vta.py`

#### Current Implementation (Lines 67, 407-412)
```python
# Config
da_decay_rate: float = 0.1     # Per-timestep decay back to tonic

# In _to_tonic_mode()
self.state.current_da += (
    (self.config.tonic_da_level - self.state.current_da) *
    self.config.da_decay_rate
)
```

#### Biological Issue
- **Problem**: Per-timestep decay doesn't reflect biological time constant
- **Literature**: Grace & Bunney (1984) report DA clearance tau = 0.2-0.5s
- **Impact**: Incorrect temporal dynamics, especially during rapid burst-pause transitions

#### Implementation

```python
# In VTAConfig (line 67)
da_decay_tau: float = 0.3      # Seconds, biological time constant
# Remove: da_decay_rate: float = 0.1

# In _to_tonic_mode() (line 407-412)
def _to_tonic_mode(self, dt: float) -> None:
    """Transition to/maintain tonic firing."""
    self.state.firing_mode = VTAFiringMode.TONIC
    self.state.current_rate = self.config.tonic_rate

    # Exponential decay: da(t+dt) = target + (da(t) - target) * exp(-dt/tau)
    decay_factor = np.exp(-dt / self.config.da_decay_tau)
    self.state.current_da = (
        self.config.tonic_da_level +
        (self.state.current_da - self.config.tonic_da_level) * decay_factor
    )

    # Update timing
    self.state.time_since_phasic += dt
    self.state.phasic_remaining = 0.0
```

#### Testing Requirements

**Unit Test**: `/mnt/projects/t4d/t4dm/tests/nca/test_vta.py`
```python
def test_vta_exponential_decay():
    """Validate exponential decay matches biological tau."""
    vta = VTACircuit(VTAConfig(da_decay_tau=0.3))

    # Set to high DA
    vta.state.current_da = 0.8
    initial_da = vta.state.current_da

    # Step for 300ms (1 tau)
    for _ in range(30):
        vta._to_tonic_mode(dt=0.01)

    # After 1 tau, should decay to ~63% toward baseline
    expected_da = 0.3 + (initial_da - 0.3) * np.exp(-1)
    assert abs(vta.state.current_da - expected_da) < 0.05

def test_vta_decay_time_constant():
    """Validate decay time constant is in biological range."""
    vta = VTACircuit()
    assert 0.2 <= vta.config.da_decay_tau <= 0.5  # Grace & Bunney 1984
```

**Integration Test**: Validate with RPE sequences
```python
def test_vta_burst_pause_dynamics():
    """Test rapid burst-pause transitions with exponential decay."""
    vta = VTACircuit()

    # Positive RPE -> burst
    vta.process_rpe(rpe=0.5, dt=0.01)
    burst_da = vta.state.current_da
    assert burst_da > 0.5

    # Immediate negative RPE -> pause
    vta.process_rpe(rpe=-0.5, dt=0.01)
    pause_da = vta.state.current_da

    # Decay back to tonic
    for _ in range(50):
        vta.step(dt=0.01)

    # Should approach tonic (0.3) exponentially
    assert abs(vta.state.current_da - 0.3) < 0.1
```

#### Documentation Updates
- Update `/mnt/projects/t4d/t4dm/src/t4dm/nca/README.md` with exponential decay formula
- Add reference to Grace & Bunney (1984) in VTA docstring
- Update `/mnt/projects/t4d/t4dm/docs/biological_validation.md` with fix confirmation

---

### 1.2: TAN Pause Mechanism (DAY 2)

**Score Impact**: 90 → 94/100 for B5

#### Files to Modify
- `/mnt/projects/t4d/t4dm/src/t4dm/nca/striatal_msn.py` (new TAN class)
- `/mnt/projects/t4d/t4dm/src/t4dm/nca/dopamine_integration.py` (TAN integration)

#### Biological Background
- **TANs** (Tonically Active Neurons): Cholinergic interneurons in striatum
- **Key Behavior**: Pause firing (~200ms) during unexpected rewards
- **Function**: Timing signal for learning (marks "when" reinforcement occurred)
- **Reference**: Aosaki et al. (1994), Apicella (2017)

#### Implementation

**New Class**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/striatal_msn.py` (add at end)

```python
from enum import Enum

class TANState(Enum):
    """TAN firing states."""
    TONIC = "tonic"        # Baseline 3-10 Hz
    PAUSED = "paused"      # Brief cessation during salient events

@dataclass
class TANConfig:
    """Configuration for tonically active neurons (cholinergic interneurons)."""

    # Firing parameters
    tonic_rate: float = 5.0           # Hz, baseline tonic firing
    pause_duration: float = 0.2       # Seconds, duration of pause
    pause_threshold: float = 0.3      # Minimum RPE for pause

    # Acetylcholine dynamics
    ach_per_spike: float = 0.02       # ACh release per spike
    ach_decay_tau: float = 0.1        # ACh clearance time constant

    # Rebound excitation
    rebound_magnitude: float = 1.5    # Post-pause burst magnitude
    rebound_duration: float = 0.1     # Seconds

    # Temporal precision
    pause_jitter: float = 0.02        # Timing variability (seconds)


@dataclass
class TANLayerState:
    """State of TAN layer."""

    state: TANState = TANState.TONIC
    current_rate: float = 5.0
    ach_level: float = 0.1            # Ambient ACh from tonic firing

    # Pause timing
    time_in_pause: float = 0.0
    time_since_pause: float = 1.0

    # Rebound state
    in_rebound: bool = False
    rebound_time: float = 0.0


class TANLayer:
    """
    Tonically Active Neurons (cholinergic interneurons) in striatum.

    Key Function: Pause firing during unexpected rewards to provide
    temporal learning signal. The pause marks "when" reinforcement occurred.

    Biology:
    - TANs fire tonically at ~3-10 Hz
    - Pause (~200ms) triggered by unexpected rewards (positive RPE)
    - Rebound excitation after pause
    - ACh dip during pause disinhibits dopamine release
    - Critical for reward timing and habit formation

    References:
    - Aosaki et al. (1994): TAN pause during reward
    - Apicella (2017): Role in learning and memory
    - Morris et al. (2004): ACh-DA interaction
    """

    def __init__(self, config: TANConfig | None = None):
        """Initialize TAN layer."""
        self.config = config or TANConfig()
        self.state = TANLayerState(current_rate=self.config.tonic_rate)

        logger.info(
            f"TANLayer initialized: tonic_rate={self.config.tonic_rate} Hz, "
            f"pause_duration={self.config.pause_duration}s"
        )

    def process_rpe(self, rpe: float, dt: float = 0.01) -> float:
        """
        Process RPE signal and trigger pause if unexpected reward.

        Args:
            rpe: Reward prediction error
            dt: Timestep in seconds

        Returns:
            Current ACh level
        """
        # Unexpected reward (positive RPE) triggers pause
        if rpe > self.config.pause_threshold:
            if self.state.state == TANState.TONIC:
                self._trigger_pause()

        # Update state
        self._update_state(dt)

        # Compute ACh release
        ach_release = self._compute_ach_release(dt)

        # Update ACh level with decay
        decay_factor = np.exp(-dt / self.config.ach_decay_tau)
        self.state.ach_level = (
            self.state.ach_level * decay_factor + ach_release
        )

        return self.state.ach_level

    def _trigger_pause(self):
        """Trigger TAN pause."""
        self.state.state = TANState.PAUSED
        self.state.current_rate = 0.0
        self.state.time_in_pause = 0.0
        self.state.time_since_pause = 0.0

        # Add biological jitter
        jitter = np.random.normal(0, self.config.pause_jitter)
        self.config.pause_duration += jitter

        logger.debug("TAN pause triggered")

    def _update_state(self, dt: float):
        """Update TAN state machine."""
        if self.state.state == TANState.PAUSED:
            self.state.time_in_pause += dt

            # End pause, start rebound
            if self.state.time_in_pause >= self.config.pause_duration:
                self.state.state = TANState.TONIC
                self.state.in_rebound = True
                self.state.rebound_time = 0.0

        elif self.state.in_rebound:
            self.state.rebound_time += dt

            # End rebound
            if self.state.rebound_time >= self.config.rebound_duration:
                self.state.in_rebound = False
                self.state.current_rate = self.config.tonic_rate

        self.state.time_since_pause += dt

    def _compute_ach_release(self, dt: float) -> float:
        """Compute ACh release based on firing rate."""
        if self.state.state == TANState.PAUSED:
            return 0.0

        elif self.state.in_rebound:
            # Enhanced firing during rebound
            rate = self.config.tonic_rate * self.config.rebound_magnitude
            return self.config.ach_per_spike * rate * dt

        else:
            # Tonic firing
            return self.config.ach_per_spike * self.state.current_rate * dt

    def get_ach_level(self) -> float:
        """Get current ACh level."""
        return self.state.ach_level

    def is_paused(self) -> bool:
        """Check if currently in pause state."""
        return self.state.state == TANState.PAUSED

    def get_stats(self) -> dict:
        """Get TAN statistics."""
        return {
            "state": self.state.state.value,
            "current_rate": self.state.current_rate,
            "ach_level": self.state.ach_level,
            "time_since_pause": self.state.time_since_pause,
            "in_rebound": self.state.in_rebound,
        }

    def reset(self):
        """Reset to tonic state."""
        self.state = TANLayerState(current_rate=self.config.tonic_rate)
```

**Integration**: Add to `/mnt/projects/t4d/t4dm/src/t4dm/nca/striatal_msn.py`

```python
# In StriatumCircuit class, add TAN layer
class StriatumCircuit:
    def __init__(self, ...):
        # Existing init...

        # Add TAN layer
        self.tan_layer = TANLayer()

    def process_dopamine(self, da_level: float, rpe: float = 0.0) -> dict:
        """Process dopamine with TAN pause integration."""

        # Process RPE through TAN
        ach_level = self.tan_layer.process_rpe(rpe)

        # TAN pause disinhibits D1 MSNs (facilitates learning)
        if self.tan_layer.is_paused():
            d1_boost = 1.3  # Enhanced D1 during TAN pause
        else:
            d1_boost = 1.0

        # Rest of existing code with d1_boost applied...
        d1_response = self._compute_d1_response(da_level) * d1_boost

        return {
            "d1_activation": d1_response,
            "d2_activation": self._compute_d2_response(da_level),
            "ach_level": ach_level,
            "tan_paused": self.tan_layer.is_paused(),
        }
```

#### Testing Requirements

```python
def test_tan_pause_on_positive_rpe():
    """Test TAN pause triggered by positive RPE."""
    tan = TANLayer()

    # Positive RPE should trigger pause
    ach_before = tan.state.ach_level
    tan.process_rpe(rpe=0.5, dt=0.01)

    assert tan.is_paused()
    assert tan.state.current_rate == 0.0

    # ACh should decrease during pause
    for _ in range(20):  # 200ms
        ach = tan.process_rpe(rpe=0.0, dt=0.01)

    assert ach < ach_before * 0.5  # Significant ACh dip

def test_tan_rebound_after_pause():
    """Test rebound excitation after pause."""
    tan = TANLayer()

    # Trigger pause
    tan.process_rpe(rpe=0.5, dt=0.01)

    # Wait for pause to end
    for _ in range(20):
        tan.process_rpe(rpe=0.0, dt=0.01)

    # Should be in rebound
    assert tan.state.in_rebound
    assert tan.state.current_rate > tan.config.tonic_rate

def test_tan_timing_precision():
    """Test pause duration is biologically accurate."""
    tan = TANLayer(TANConfig(pause_duration=0.2))

    tan.process_rpe(rpe=0.5, dt=0.01)

    pause_count = 0
    for _ in range(30):  # 300ms
        tan.process_rpe(rpe=0.0, dt=0.01)
        if tan.is_paused():
            pause_count += 1

    # Pause should last ~200ms (20 steps)
    assert 18 <= pause_count <= 22  # Allow jitter
```

---

## Phase 2: Consolidation and Glial Enhancements (MEDIUM PRIORITY)

**Estimated Effort**: 2-3 days
**Can Parallelize With**: Testing, documentation
**Dependencies**: Phase 1 complete

### 2.1: STDP Multiplicative Weight Dependence (DAY 3)

**Score Impact**: Sleep/Wake B7: 94 → 96/100

#### Files to Modify
- `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/stdp_integration.py`
- `/mnt/projects/t4d/t4dm/src/t4dm/learning/stdp.py` (if not already implemented)

#### Biological Background
- **Issue**: Current STDP uses additive weight updates
- **Biology**: Synaptic plasticity depends on current weight (Song et al. 2000)
  - Strong synapses: Less LTP, more LTD (prevents runaway)
  - Weak synapses: More LTP, less LTD (competitive dynamics)
- **Formula**: Δw = η * f(w) * STDP(Δt)
  - Multiplicative LTP: f(w) = (w_max - w)^μ
  - Multiplicative LTD: f(w) = w^μ

#### Implementation

```python
# In stdp_integration.py, update ConsolidationSTDPConfig
@dataclass
class ConsolidationSTDPConfig:
    # ... existing fields ...

    # Weight dependence
    use_multiplicative_stdp: bool = True
    ltp_exponent: float = 1.0         # μ for LTP: (w_max - w)^μ
    ltd_exponent: float = 1.0         # μ for LTD: w^μ
    soft_bound: bool = True           # Soft vs hard bounds

# In ConsolidationSTDP.__init__
stdp_config = self.config.stdp_config or STDPConfig(
    spike_window_ms=200.0,
    a_plus=0.008,
    a_minus=0.0084,
    weight_decay=0.01,
    # NEW: Weight-dependent plasticity
    use_weight_dependent=self.config.use_multiplicative_stdp,
    ltp_exponent=self.config.ltp_exponent,
    ltd_exponent=self.config.ltd_exponent,
    min_weight=self.config.consolidation_min_weight,
    max_weight=self.config.consolidation_max_weight,
)

# In apply_stdp_to_sequence (line 169)
current_weight = self.stdp.get_weight(pre_id, post_id)
update = self.stdp.compute_update(
    pre_id, post_id, current_weight,
    weight_dependent=self.config.use_multiplicative_stdp
)
```

**In `/mnt/projects/t4d/t4dm/src/t4dm/learning/stdp.py`** (add weight dependence):

```python
def compute_update(
    self,
    pre_id: str,
    post_id: str,
    current_weight: float,
    weight_dependent: bool = True
) -> STDPUpdate | None:
    """Compute STDP weight update with optional weight dependence."""

    # ... existing spike timing code ...

    if weight_dependent:
        # Multiplicative STDP (Song et al. 2000)
        if delta_t > 0:  # LTP
            # Stronger when weight is low
            w_factor = (self.config.max_weight - current_weight) ** self.config.ltp_exponent
            delta_w = self.config.a_plus * w_factor * np.exp(-delta_t / self.config.tau_plus)
        else:  # LTD
            # Stronger when weight is high
            w_factor = current_weight ** self.config.ltd_exponent
            delta_w = -self.config.a_minus * w_factor * np.exp(delta_t / self.config.tau_minus)
    else:
        # Additive STDP (original)
        if delta_t > 0:
            delta_w = self.config.a_plus * np.exp(-delta_t / self.config.tau_plus)
        else:
            delta_w = -self.config.a_minus * np.exp(delta_t / self.config.tau_minus)

    # Apply update
    new_weight = np.clip(
        current_weight + delta_w,
        self.config.min_weight,
        self.config.max_weight
    )

    # ... rest of existing code ...
```

#### Testing

```python
def test_stdp_weight_dependence():
    """Test multiplicative STDP weight dependence."""
    config = ConsolidationSTDPConfig(use_multiplicative_stdp=True)
    stdp = ConsolidationSTDP(config)

    # Weak synapse should potentiate more
    stdp.stdp.set_weight("A", "B", 0.2)
    stdp.apply_stdp_to_sequence(["A", "B"])
    weak_delta = stdp.stdp.get_weight("A", "B") - 0.2

    # Strong synapse should potentiate less
    stdp.stdp.set_weight("C", "D", 0.8)
    stdp.apply_stdp_to_sequence(["C", "D"])
    strong_delta = stdp.stdp.get_weight("C", "D") - 0.8

    # Weak should grow faster (competitive dynamics)
    assert weak_delta > strong_delta

def test_stdp_ltd_weight_dependence():
    """Test LTD is stronger for strong synapses."""
    config = ConsolidationSTDPConfig(use_multiplicative_stdp=True)
    stdp = ConsolidationSTDP(config)

    # Strong synapse with reverse timing (LTD)
    stdp.stdp.set_weight("A", "B", 0.8)
    stdp.apply_stdp_to_sequence(["B", "A"])  # Post before pre
    strong_ltd = 0.8 - stdp.stdp.get_weight("A", "B")

    # Weak synapse with reverse timing
    stdp.stdp.set_weight("C", "D", 0.2)
    stdp.apply_stdp_to_sequence(["D", "C"])
    weak_ltd = 0.2 - stdp.stdp.get_weight("C", "D")

    # Strong should depress more (homeostatic)
    assert strong_ltd > weak_ltd
```

---

### 2.2: Astrocyte Gap Junction Communication (DAY 4)

**Score Impact**: 91 → 95/100 for B8

#### Files to Modify
- `/mnt/projects/t4d/t4dm/src/t4dm/nca/astrocyte.py`

#### Biological Background
- **Gap Junctions**: Connexin channels between astrocytes
- **Function**: Propagate Ca2+ waves across astrocyte network
- **Speed**: ~10-20 μm/s
- **Range**: 100-500 μm
- **Reference**: Giaume & Theis (2010)

#### Implementation

```python
# In AstrocyteConfig (add after line 80)
# Gap junction parameters
gap_junction_conductance: float = 0.3  # Ca2+ diffusion rate
gap_junction_threshold: float = 0.5    # Minimum Ca2+ for propagation
wave_decay: float = 0.1                # Spatial decay per neighbor

# In AstrocyteLayer class
class AstrocyteLayer:
    def __init__(self, config: AstrocyteConfig | None = None, network_size: int = 1):
        # ... existing init ...

        # Gap junction network
        self.network_size = network_size
        if network_size > 1:
            # Each astrocyte tracks its own Ca2+
            self.ca_network = np.full(network_size, 0.1, dtype=np.float32)
        else:
            self.ca_network = None

    def propagate_calcium_wave(
        self,
        source_idx: int,
        connectivity: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Propagate Ca2+ wave through gap junction network.

        Args:
            source_idx: Index of astrocyte with elevated Ca2+
            connectivity: Adjacency matrix (default: nearest neighbors)

        Returns:
            Updated Ca2+ levels across network
        """
        if self.ca_network is None:
            return np.array([self.state.calcium])

        cfg = self.config

        # Default connectivity: nearest neighbors
        if connectivity is None:
            connectivity = self._default_connectivity()

        # Check if source exceeds threshold
        if self.ca_network[source_idx] < cfg.gap_junction_threshold:
            return self.ca_network.copy()

        # Propagate via diffusion
        ca_new = self.ca_network.copy()

        for i in range(self.network_size):
            if i == source_idx:
                continue

            # Get neighbors
            neighbors = np.where(connectivity[i] > 0)[0]

            if len(neighbors) == 0:
                continue

            # Diffusion from neighbors
            for neighbor in neighbors:
                # Ca2+ flow proportional to gradient and conductance
                gradient = self.ca_network[neighbor] - ca_new[i]

                if gradient > 0:  # Only if neighbor has higher Ca2+
                    # Apply spatial decay
                    distance = connectivity[i, neighbor]
                    decay = np.exp(-distance * cfg.wave_decay)

                    flow = (
                        cfg.gap_junction_conductance *
                        gradient *
                        decay
                    )

                    ca_new[i] += flow

        # Clip and update
        self.ca_network = np.clip(ca_new, 0.0, 1.0)

        # Update mean state
        self.state.calcium = float(np.mean(self.ca_network))

        return self.ca_network.copy()

    def _default_connectivity(self) -> np.ndarray:
        """Create default nearest-neighbor connectivity."""
        # 1D ring lattice with nearest neighbors
        conn = np.zeros((self.network_size, self.network_size))

        for i in range(self.network_size):
            # Connect to immediate neighbors (distance = 1)
            left = (i - 1) % self.network_size
            right = (i + 1) % self.network_size
            conn[i, left] = 1.0
            conn[i, right] = 1.0

        return conn

    def trigger_calcium_wave(self, source_idx: int):
        """Manually trigger Ca2+ wave from specific astrocyte."""
        if self.ca_network is not None:
            self.ca_network[source_idx] = min(
                self.ca_network[source_idx] + 0.3,
                1.0
            )
            self.propagate_calcium_wave(source_idx)
```

#### Testing

```python
def test_astrocyte_gap_junction_propagation():
    """Test Ca2+ wave propagation through gap junctions."""
    astro = AstrocyteLayer(network_size=10)

    # Elevate Ca2+ in center astrocyte
    astro.ca_network[5] = 0.8

    # Propagate
    ca_after = astro.propagate_calcium_wave(source_idx=5)

    # Neighbors should have elevated Ca2+
    assert ca_after[4] > 0.1  # Left neighbor
    assert ca_after[6] > 0.1  # Right neighbor

    # Distant astrocytes should have lower Ca2+
    assert ca_after[0] < ca_after[4]  # Far left

def test_calcium_wave_decay():
    """Test spatial decay of Ca2+ wave."""
    astro = AstrocyteLayer(
        config=AstrocyteConfig(wave_decay=0.1),
        network_size=20
    )

    # Trigger wave at one end
    astro.ca_network[0] = 0.9
    astro.propagate_calcium_wave(source_idx=0)

    # Check exponential decay with distance
    distances = np.arange(1, 10)
    ca_values = [astro.ca_network[d] for d in distances]

    # Should decrease with distance
    assert all(ca_values[i] >= ca_values[i+1] for i in range(len(ca_values)-1))
```

---

## Phase 3: Forward-Forward Encoder Enhancement (HINTON CONCERN)

**Estimated Effort**: 4-5 days
**Can Parallelize With**: Phase 2
**Dependencies**: Requires architecture discussion

### 3.1: Problem Analysis - "Frozen Embeddings"

**Hinton's Concern** (from review):
> "The FF encoder layer is trained once and then frozen. This limits the system's ability to adapt embeddings based on consolidation feedback."

**Current Architecture**:
```
Input → FF Encoder (FROZEN) → Episodic/Semantic/Procedural Memory
                ↓
          Fixed 256-dim embeddings
```

**Biological Parallel**: Adult neurogenesis and dendritic plasticity
- **Hippocampus**: Continuous neurogenesis in dentate gyrus (~700 new neurons/day)
- **Neocortex**: Dendritic spine turnover (~10% per week)
- **Function**: Allows pattern separation and encoding flexibility

### 3.2: Biological Solutions

#### Option A: Activity-Dependent Neurogenesis (Most Biological)

**Mechanism**: Simulate neurogenesis by adding/removing encoder neurons based on consolidation feedback

```python
# New file: /mnt/projects/t4d/t4dm/src/t4dm/encoding/neurogenesis.py

@dataclass
class NeurogenesisConfig:
    """Configuration for activity-dependent neurogenesis."""

    # Birth/death rates
    birth_rate: float = 0.001          # Fraction of neurons born per consolidation
    death_rate: float = 0.0005         # Fraction pruned per consolidation

    # Selection criteria
    activity_threshold: float = 0.1    # Minimum activity for survival
    novelty_boost: float = 2.0         # Birth rate multiplier during novelty

    # Network constraints
    min_neurons: int = 128             # Minimum encoder size
    max_neurons: int = 512             # Maximum encoder size

    # Integration
    maturation_time: int = 10          # Consolidation cycles before full integration
    immature_weight: float = 0.5       # Contribution of immature neurons


class NeurogenicEncoder:
    """
    FF encoder with activity-dependent neurogenesis.

    Simulates dentate gyrus neurogenesis:
    1. New neurons born during consolidation (if novelty detected)
    2. Immature neurons have reduced weights initially
    3. Low-activity neurons pruned over time
    4. Maintains pattern separation via population turnover

    Biology:
    - DG neurogenesis: ~700 new neurons/day in adult hippocampus
    - New neurons more excitable, enhance pattern separation
    - Activity-dependent survival (use it or lose it)

    References:
    - Kempermann et al. (2015): Adult hippocampal neurogenesis
    - Aimone et al. (2011): Computational influence of neurogenesis
    - Sahay et al. (2011): Increasing neurogenesis promotes forgetting
    """

    def __init__(
        self,
        base_encoder: Any,  # Existing FF encoder
        config: NeurogenesisConfig | None = None
    ):
        self.base_encoder = base_encoder
        self.config = config or NeurogenesisConfig()

        # Track neuron ages and activities
        self.neuron_ages = np.zeros(base_encoder.output_dim)
        self.neuron_activities = np.zeros(base_encoder.output_dim)

        # Immature neurons (recently born)
        self.immature_neurons: set[int] = set()

        logger.info(
            f"NeurogenicEncoder initialized: "
            f"birth_rate={self.config.birth_rate}, "
            f"death_rate={self.config.death_rate}"
        )

    def consolidation_cycle(
        self,
        novelty_score: float,
        prediction_error: float
    ) -> dict[str, int]:
        """
        Run neurogenesis update after consolidation.

        Args:
            novelty_score: Recent novelty level (0-1)
            prediction_error: Recent prediction error (0-1)

        Returns:
            Dict with neurons_born, neurons_pruned
        """
        current_size = len(self.neuron_activities)

        # Determine birth count
        birth_multiplier = 1.0 + self.config.novelty_boost * novelty_score
        base_birth = int(current_size * self.config.birth_rate * birth_multiplier)

        # Limit by max size
        max_birth = self.config.max_neurons - current_size
        neurons_to_birth = min(base_birth, max_birth)

        # Add new neurons
        neurons_born = self._add_neurons(neurons_to_birth)

        # Determine pruning based on activity
        neurons_pruned = self._prune_inactive_neurons()

        # Age existing neurons
        self.neuron_ages += 1

        # Mature immature neurons
        self._mature_neurons()

        return {
            "neurons_born": neurons_born,
            "neurons_pruned": neurons_pruned,
            "current_size": len(self.neuron_activities),
            "immature_count": len(self.immature_neurons),
        }

    def _add_neurons(self, count: int) -> int:
        """Add new neurons to encoder."""
        if count == 0:
            return 0

        # Extend neuron arrays
        self.neuron_ages = np.concatenate([
            self.neuron_ages,
            np.zeros(count)
        ])
        self.neuron_activities = np.concatenate([
            self.neuron_activities,
            np.zeros(count)
        ])

        # Mark as immature
        start_idx = len(self.neuron_ages) - count
        for i in range(start_idx, len(self.neuron_ages)):
            self.immature_neurons.add(i)

        # Initialize weights (small random)
        self._initialize_new_weights(start_idx, count)

        logger.debug(f"Born {count} new neurons (total: {len(self.neuron_ages)})")
        return count

    def _prune_inactive_neurons(self) -> int:
        """Prune neurons below activity threshold."""
        cfg = self.config

        # Can't prune below minimum
        if len(self.neuron_activities) <= cfg.min_neurons:
            return 0

        # Find inactive neurons
        inactive = np.where(
            self.neuron_activities < cfg.activity_threshold
        )[0]

        # Limit pruning rate
        max_prune = int(len(self.neuron_activities) * cfg.death_rate)
        to_prune = inactive[:max_prune]

        if len(to_prune) == 0:
            return 0

        # Remove from arrays
        keep_mask = np.ones(len(self.neuron_activities), dtype=bool)
        keep_mask[to_prune] = False

        self.neuron_ages = self.neuron_ages[keep_mask]
        self.neuron_activities = self.neuron_activities[keep_mask]

        # Remove from immature set
        self.immature_neurons = {
            i for i in self.immature_neurons if i not in to_prune
        }

        # Update base encoder weights
        self._remove_weights(to_prune)

        logger.debug(f"Pruned {len(to_prune)} inactive neurons")
        return len(to_prune)

    def _mature_neurons(self):
        """Mature immature neurons that reach maturation age."""
        to_mature = [
            i for i in self.immature_neurons
            if self.neuron_ages[i] >= self.config.maturation_time
        ]

        for i in to_mature:
            self.immature_neurons.remove(i)

        if to_mature:
            logger.debug(f"Matured {len(to_mature)} neurons")

    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """
        Encode with neurogenesis-aware weighting.

        Immature neurons contribute less to encoding.
        """
        # Get base encoding
        embedding = self.base_encoder.encode(input_data)

        # Apply maturity weights
        if len(self.immature_neurons) > 0:
            maturity_weights = np.ones(len(embedding))
            for i in self.immature_neurons:
                if i < len(maturity_weights):
                    age_factor = self.neuron_ages[i] / self.config.maturation_time
                    maturity_weights[i] = (
                        self.config.immature_weight +
                        (1.0 - self.config.immature_weight) * age_factor
                    )

            embedding = embedding * maturity_weights

        # Track activity
        self.neuron_activities = (
            0.9 * self.neuron_activities +
            0.1 * np.abs(embedding)
        )

        return embedding

    def _initialize_new_weights(self, start_idx: int, count: int):
        """Initialize weights for new neurons."""
        # Xavier initialization for new weights
        # Implementation depends on base_encoder architecture
        pass

    def _remove_weights(self, indices: np.ndarray):
        """Remove weights for pruned neurons."""
        # Implementation depends on base_encoder architecture
        pass

    def get_stats(self) -> dict:
        """Get neurogenesis statistics."""
        return {
            "total_neurons": len(self.neuron_activities),
            "immature_neurons": len(self.immature_neurons),
            "mean_age": float(np.mean(self.neuron_ages)),
            "mean_activity": float(np.mean(self.neuron_activities)),
            "min_neurons": self.config.min_neurons,
            "max_neurons": self.config.max_neurons,
        }
```

#### Integration with Consolidation

```python
# In /mnt/projects/t4d/t4dm/src/t4dm/consolidation/service.py

from t4dm.encoding.neurogenesis import NeurogenicEncoder

class ConsolidationService:
    def __init__(self, ..., neurogenic_encoder: NeurogenicEncoder | None = None):
        # ... existing init ...
        self.neurogenic_encoder = neurogenic_encoder

    async def run_consolidation_cycle(self, ...):
        # ... existing consolidation ...

        # After replay and STDP
        if self.neurogenic_encoder:
            # Compute novelty from recent episodes
            novelty = self._compute_novelty_score(episodes)

            # Compute prediction error
            pred_error = self._compute_prediction_error(episodes)

            # Run neurogenesis
            neuro_stats = self.neurogenic_encoder.consolidation_cycle(
                novelty_score=novelty,
                prediction_error=pred_error
            )

            logger.info(
                f"Neurogenesis: +{neuro_stats['neurons_born']} "
                f"-{neuro_stats['neurons_pruned']} neurons"
            )
```

#### Testing

```python
def test_neurogenesis_birth_during_novelty():
    """Test new neurons born during high novelty."""
    encoder = NeurogenicEncoder(base_encoder=MockFFEncoder(output_dim=256))

    initial_size = len(encoder.neuron_activities)

    # High novelty should increase birth rate
    stats = encoder.consolidation_cycle(novelty_score=0.9, prediction_error=0.5)

    assert stats["neurons_born"] > 0
    assert stats["current_size"] > initial_size

def test_neurogenesis_pruning():
    """Test inactive neurons are pruned."""
    encoder = NeurogenicEncoder(base_encoder=MockFFEncoder(output_dim=256))

    # Set some neurons to low activity
    encoder.neuron_activities[:10] = 0.05  # Below threshold

    stats = encoder.consolidation_cycle(novelty_score=0.1, prediction_error=0.1)

    assert stats["neurons_pruned"] > 0

def test_immature_neuron_maturation():
    """Test immature neurons mature over time."""
    encoder = NeurogenicEncoder(
        base_encoder=MockFFEncoder(output_dim=256),
        config=NeurogenesisConfig(maturation_time=5)
    )

    # Add neurons
    encoder.consolidation_cycle(novelty_score=0.8, prediction_error=0.5)
    immature_initial = len(encoder.immature_neurons)

    # Run maturation cycles
    for _ in range(5):
        encoder.consolidation_cycle(novelty_score=0.0, prediction_error=0.0)

    # Should have matured
    assert len(encoder.immature_neurons) < immature_initial
```

---

### 3.3: Alternative: Dendritic Spine Plasticity (Simpler)

**If neurogenesis is too complex**, implement dendritic plasticity:

```python
# Simpler: Adjust FF encoder weights during consolidation based on usage

class PlasticFFEncoder:
    """FF encoder with consolidation-based weight adjustment."""

    def consolidation_update(self, usage_stats: dict[str, float]):
        """
        Adjust encoder weights based on memory usage.

        High-usage dimensions: Strengthen
        Low-usage dimensions: Weaken or repurpose
        """
        for layer in self.layers:
            # Track which dimensions are active
            active_dims = usage_stats.get("active_dimensions", [])

            # Hebbian-like update: strengthen used pathways
            for dim in active_dims:
                layer.weight[:, dim] *= 1.05  # 5% increase

            # Homeostatic: weaken unused
            inactive_dims = set(range(layer.weight.shape[1])) - set(active_dims)
            for dim in inactive_dims:
                layer.weight[:, dim] *= 0.98  # 2% decrease

            # Renormalize
            layer.weight /= np.linalg.norm(layer.weight, axis=0, keepdims=True)
```

---

## Summary and Recommendations

### Prioritization

**Phase 1 (CRITICAL - Do First)**:
1. VTA exponential decay (1 day) - Fixes B1 biological accuracy
2. TAN pause mechanism (1 day) - Fixes B5, adds reward timing

**Phase 2 (IMPORTANT - Do Second)**:
3. STDP weight dependence (1 day) - Fixes consolidation biology
4. Astrocyte gap junctions (1 day) - Fixes B8, completes glial model

**Phase 3 (ENHANCEMENT - Discuss First)**:
5. Neurogenic encoder (3-4 days) - Addresses Hinton concern
   - **Recommendation**: Start with Option A (neurogenesis) for biological fidelity
   - If too complex, fall back to Option B (dendritic plasticity)

### Parallelization Strategy

```
Week 1:
  Day 1-2: Phase 1 (VTA + TAN) - Developer A
  Day 1-2: Documentation updates - Developer B
  Day 3-4: Phase 2 (STDP + Astrocyte) - Developer A
  Day 3-4: Testing infrastructure - Developer B
  Day 5: Integration testing - Both

Week 2:
  Day 1-5: Phase 3 (Neurogenesis) - Developer A
  Day 1-5: Comprehensive validation - Developer B
```

### Expected Score Impact

| Phase | Component | Before | After | Delta |
|-------|-----------|--------|-------|-------|
| 1 | VTA (B1) | 92 | 95 | +3 |
| 1 | Striatum (B5) | 90 | 94 | +4 |
| 2 | Sleep/Wake (B7) | 94 | 96 | +2 |
| 2 | Astrocytes (B8) | 91 | 95 | +4 |
| 3 | FF Architecture | N/A | +3 | +3 |
| **Total** | **Overall** | **92** | **98** | **+6** |

### Documentation Updates Required

1. **Per-phase**:
   - Update `BIOLOGICAL_VALIDATION_REPORT.md` with fix confirmations
   - Add references to `docs/biological_validation.md`
   - Update module README files

2. **Final**:
   - Create `docs/NEUROGENESIS_DESIGN.md` (if Phase 3 implemented)
   - Update `CHANGELOG.md` with biological fixes
   - Add to `docs/NEURAL_MEMORY_UPGRADE_ROADMAP.md`

---

**Next Steps**:
1. Review this plan with team
2. Confirm Phase 3 approach (neurogenesis vs dendritic plasticity)
3. Begin Phase 1 implementation
4. Set up continuous validation tests
