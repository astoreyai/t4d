# World Weaver NCA: Biologically-Grounded Visualization Telemetry Specification

**Date**: 2026-01-01
**Version**: 1.0
**Status**: Design Specification

## Executive Summary

This document specifies visualization telemetry for World Weaver's Neural Cellular Automata (NCA) memory system from a computational neuroscience perspective. The system models 6 neuromodulator systems, hippocampal circuits, and homeostatic mechanisms. Current visualization coverage is incomplete (35-68%), indicating substantial blind spots in system observability.

## I. Essential Biological Signals

### 1.1 Neuromodulator State Space (6D)

**Current Coverage**: 35% (`neuromodulator_state.py`)
**Gaps**: Missing real-time NT dynamics, opponent processes, temporal correlations

#### Critical Signals
```python
@dataclass
class NeuromodulatorTelemetry:
    """Complete NT state snapshot."""

    # Core concentrations [0, 1]
    dopamine: float           # VTA: reward prediction error
    serotonin: float          # Raphe: patience/temporal discounting
    acetylcholine: float      # Encoding vs retrieval mode
    norepinephrine: float     # Arousal/gain modulation
    gaba: float               # Inhibitory tone
    glutamate: float          # Excitatory drive

    # Firing rates (Hz) - biological validation
    vta_firing_rate: float    # DA neurons: 4-5 Hz tonic, 20-40 Hz burst
    raphe_firing_rate: float  # 5-HT neurons: 1-5 Hz

    # Opponent process dynamics
    da_5ht_ratio: float       # Reward-seeking vs impulse control
    e_i_balance: float        # Glu/GABA ratio (0.5 = balanced)

    # Arousal-modulated parameters
    hopfield_beta: float      # NE → retrieval sharpness (2-16)
    alpha_inhibition: float   # Alpha power as cortical gating

    # Homeostatic targets
    bcm_threshold: float      # LTP/LTD crossover point
    synaptic_scaling: float   # Global activity normalization
```

**Data Sources**:
- `/ww/nca/vta.py`: `VTAState.current_da`, `firing_mode`, `last_rpe`
- `/ww/nca/raphe.py`: `RapheNucleusState.extracellular_5ht`, `firing_rate`, `autoreceptor_inhibition`
- `/ww/nca/energy.py`: `EnergyLandscape.compute_arousal_beta()`, `_current_beta`
- `/ww/nca/oscillators.py`: `AlphaOscillator.get_inhibition_level()`

**Visualization Requirements**:
1. **6D State Trajectory**: PCA/UMAP projection showing NT evolution
2. **Opponent Process Phase Plane**: DA-5HT and E/I balance as 2D dynamics
3. **Firing Rate Validation**: Histogram overlays with biological bounds
4. **Arousal-Beta Coupling**: Scatter plot of NE level vs Hopfield β

---

### 1.2 Hippocampal Circuit Dynamics

**Current Coverage**: 68% (`pattern_separation.py`)
**Gaps**: Missing CA3 completion iterations, theta phase gating, novelty-dopamine coupling

#### Critical Signals
```python
@dataclass
class HippocampalTelemetry:
    """Hippocampal processing snapshot."""

    # DG Pattern Separation
    dg_sparsity: float             # Fraction active units (bio: ~0.5%)
    dg_separation_magnitude: float # Orthogonalization strength
    dg_expansion_ratio: float      # EC → DG dimension (bio: 4x)

    # CA3 Pattern Completion (Hopfield)
    ca3_stored_patterns: int       # Memory capacity usage
    ca3_completion_iters: int      # Convergence speed (< 10 is healthy)
    ca3_hopfield_energy: float     # Attractor depth
    ca3_beta_current: float        # Arousal-modulated sharpness

    # CA1 Novelty Detection
    ca1_novelty_score: float       # EC-CA3 mismatch [0, 1]
    ca1_mode: str                  # "encoding" | "retrieval" | "automatic"
    ca1_theta_phase: float         # Theta phase [0, 2π]

    # Theta Rhythm (4-8 Hz)
    theta_frequency: float         # Current Hz (ACh-modulated)
    theta_amplitude: float         # Power (encoding strength)
    theta_encoding_prob: float     # Phase-gated encoding likelihood

    # Novelty → VTA coupling
    novelty_to_da: float           # Novelty-driven RPE injection
    vta_da_response: float         # Resulting DA burst
```

**Data Sources**:
- `/ww/nca/hippocampus.py`: `HippocampalState` (all fields), `_apply_theta_gating()`
- `/ww/nca/oscillators.py`: `FrequencyBandGenerator.get_encoding_signal()`, `theta.freq`, `theta.amplitude`
- `/ww/nca/vta.py`: `VTACircuit.connect_from_hippocampus()` (novelty signal)

**Visualization Requirements**:
1. **DG Sparsity Histogram**: Before/after separation with biological 0.5% target line
2. **CA3 Energy Landscape**: 2D slice showing attractor wells and current position
3. **Theta Phase Circle**: Encoding/retrieval zones with current phase indicator
4. **Novelty-DA Coupling**: Time series of CA1 novelty → VTA dopamine latency

---

### 1.3 Oscillatory Coherence (Neural Rhythms)

**Current Coverage**: 0% (not visualized)
**Priority**: HIGH - essential for validating bio-plausibility

#### Critical Signals
```python
@dataclass
class OscillationTelemetry:
    """Neural rhythm coherence metrics."""

    # Frequency bands (Hz)
    theta_freq: float          # 4-8 Hz (memory/navigation)
    alpha_freq: float          # 8-13 Hz (cortical gating)
    beta_freq: float           # 13-30 Hz (motor/cognitive)
    gamma_freq: float          # 30-100 Hz (local processing)

    # Power (variance)
    theta_power: float         # Encoding strength
    alpha_power: float         # Inhibition level
    gamma_power: float         # Attention/binding

    # Phase-Amplitude Coupling (PAC)
    pac_strength: float        # Theta-gamma coupling [0, 1]
    pac_modulation_index: float # Kullback-Leibler PAC measure
    preferred_gamma_phase: float # Theta phase for max gamma

    # Working memory capacity
    wm_slots: int              # gamma_freq / theta_freq (bio: 4-8)

    # Cross-frequency interactions
    theta_gamma_coherence: float    # Nesting quality
    alpha_beta_suppression: float   # Competition metric
```

**Data Sources**:
- `/ww/nca/oscillators.py`: All `FrequencyBandGenerator` methods
- `OscillatorState.to_dict()`
- `PhaseAmplitudeCoupling.compute_modulation_index()`

**Visualization Requirements**:
1. **Spectral Power**: Real-time PSD with biological frequency band overlays
2. **PAC Comodulogram**: Theta phase (x) vs gamma amplitude (y) heatmap
3. **Phase Relationship Circle**: Theta, alpha, beta phases as rotating vectors
4. **WM Capacity Tracker**: Line plot of gamma/theta ratio over time

---

## II. Opponent Process Dynamics

### 2.1 Dopamine-Serotonin Balance

**Biological Basis**: VTA dopamine (reward-seeking) vs Raphe serotonin (patience/impulse control)

**Key Equation**:
```python
# Raphe → VTA inhibition via 5-HT2C receptors
vta_inhibition = serotonin_level * 0.3  # Max 30% DA reduction
effective_da = da_baseline - vta_inhibition

# DA-5HT ratio as behavioral mode
ratio = dopamine / (serotonin + 1e-6)
if ratio > 1.5:
    mode = "impulsive/reward-seeking"
elif ratio < 0.5:
    mode = "patient/aversive"
else:
    mode = "balanced"
```

**Visualization**: 2D phase plane with nullclines
- **X-axis**: Dopamine [0, 1]
- **Y-axis**: Serotonin [0, 1]
- **Nullclines**: Lines where dDA/dt = 0 or d5HT/dt = 0
- **Trajectory**: Time series of (DA, 5HT) with velocity vectors
- **Attractor Regions**: Shaded zones for stable behavioral modes

### 2.2 Excitation-Inhibition (E/I) Balance

**Biological Basis**: Glutamate (E) vs GABA (I) determines gamma frequency and stability

**Key Equation**:
```python
# E/I ratio affects gamma frequency
ei_balance = glutamate - gaba  # [-1, 1]
gamma_freq = base_gamma * (1 - 0.4 * ei_balance)  # More I → faster gamma

# Critical slowing near E/I parity
stability_margin = abs(ei_balance)  # < 0.1 = near criticality
if stability_margin < 0.1:
    warning = "Near E/I critical point - risk of runaway excitation"
```

**Visualization**: Time series with stability metric
- **Dual Y-axis**: Glu (green) and GABA (red) concentrations
- **Derived metric**: E/I balance line (blue)
- **Shaded regions**: Critical zone (|E/I| < 0.1), safe zone
- **Gamma frequency**: Right Y-axis showing frequency modulation

---

## III. Biological Invariants to Monitor

### 3.1 Firing Rate Bounds

**Purpose**: Validate neuromodulator dynamics stay within physiological ranges

| Neuron Type | Tonic (Hz) | Burst (Hz) | Pause (Hz) | Data Source |
|-------------|------------|------------|------------|-------------|
| VTA DA      | 4-5        | 20-40      | 0-2        | `VTAState.current_rate` |
| Raphe 5-HT  | 2-5        | 8          | 0.1        | `RapheNucleusState.firing_rate` |
| DG granule  | ~0.05      | N/A        | N/A        | Inferred from `dg_sparsity * 4096` |

**Violation Detection**:
```python
def validate_firing_rates(state: NeuromodulatorTelemetry) -> List[str]:
    violations = []

    if not (0.1 <= state.vta_firing_rate <= 50):
        violations.append(f"VTA firing rate {state.vta_firing_rate:.1f} Hz out of bounds [0.1, 50]")

    if not (0.1 <= state.raphe_firing_rate <= 10):
        violations.append(f"Raphe firing rate {state.raphe_firing_rate:.1f} Hz out of bounds [0.1, 10]")

    return violations
```

**Visualization**: Real-time alert system
- **Dashboard Panel**: Green/yellow/red status lights per neuron type
- **Histogram**: Distribution of firing rates over last 1000 timesteps
- **Anomaly Timeline**: Mark timesteps where violations occurred

---

### 3.2 Oscillation Coherence

**Purpose**: Ensure theta-gamma PAC matches biological literature

**Biological Targets**:
- **Theta frequency**: 4-8 Hz (human: ~6 Hz)
- **Gamma frequency**: 30-80 Hz (human: ~40 Hz)
- **PAC modulation index**: 0.2-0.5 (Tort et al., 2010)
- **WM capacity**: 4-8 items (Miller's 7±2)

**Validation Code** (from `oscillators.py:801-831`):
```python
def validate_oscillations(self) -> Dict[str, Any]:
    results = {
        "theta_freq_valid": 4.0 <= theta_freq <= 8.0,
        "gamma_freq_valid": 30.0 <= gamma_freq <= 80.0,
        "pac_present": modulation_index > 0.1,
        "wm_capacity_valid": 4 <= wm_capacity <= 10,
    }
    results["all_pass"] = all(results.values())
    return results
```

**Visualization**: Validation dashboard
- **Frequency Range Bars**: Show current freq with acceptable range shaded
- **PAC Strength Gauge**: Dial indicator with "weak" (< 0.1), "typical" (0.2-0.5), "strong" (> 0.5)
- **WM Capacity Counter**: Numeric display with color coding (red < 4, green 4-8, yellow > 8)

---

### 3.3 Neuromodulator Ratios

**Purpose**: Monitor opponent process balance for behavioral validation

**Key Ratios**:
1. **DA/5HT**: Reward-seeking vs impulse control (target: 0.7-1.3 for balanced)
2. **Glu/GABA**: Excitation/inhibition (target: 0.9-1.1 for stability)
3. **ACh/NE**: Encoding vs arousal (high ACh + low NE = consolidation)

**Homeostatic Regulation**:
```python
# Raphe autoreceptor creates 5-HT homeostasis
setpoint = 0.4
error = serotonin - setpoint
raphe_firing_adjustment = homeostatic_gain * error  # Negative feedback

# BCM threshold adapts to average activity
bcm_threshold = moving_average(postsynaptic_activity, window=100)
```

**Visualization**: Ratio time series with homeostatic setpoints
- **Three subplots**: DA/5HT, Glu/GABA, ACh/NE over time
- **Horizontal lines**: Homeostatic setpoints and acceptable ranges
- **Color coding**: Green (in range), yellow (deviation), red (critical)

---

## IV. Consolidation & Replay Visualization

### 4.1 Sharp-Wave Ripple (SWR) Sequences

**Current Coverage**: 39% (`consolidation_replay.py`)
**Gaps**: Missing priority weighting, NREM/REM phase transitions, energy-based selection

**Biological Mechanism**:
During offline periods (sleep/rest), hippocampus replays memories in compressed time (~20x speedup). Replay priority is determined by:
1. **Recency**: Recent memories replayed more
2. **Emotional salience**: High RPE → high replay
3. **Novelty**: Novel patterns consolidated first

**Data Structure**:
```python
@dataclass
class SWREvent:
    """Single sharp-wave ripple replay."""
    timestamp: datetime
    memories_replayed: List[UUID]        # CA3 pattern IDs
    replay_order: List[int]              # Temporal sequence
    priority_scores: List[float]         # Why these memories?
    phase: str                           # "nrem" | "rem"

    # Energy-based selection
    hopfield_energy_before: float        # Pre-consolidation
    hopfield_energy_after: float         # Post-consolidation (should decrease)

    # Neuromodulator context
    ach_level: float                     # Low ACh gates consolidation
    ne_level: float                      # Low NE enables SWR
```

**Data Sources**:
- `/ww/nca/hippocampus.py`: `CA3Layer._patterns`, `_pattern_ids`
- `/ww/nca/energy.py`: `HopfieldIntegration.compute_energy()`
- Needs new SWR generation logic (not yet implemented)

**Visualization Requirements**:

1. **SWR Sequence Diagram**:
   - X-axis: Position in sequence (0, 1, 2, ...)
   - Y-axis: Priority score [0, 1]
   - Points: Memory IDs (sized by salience)
   - Arrows: Temporal ordering (replay direction)
   - Color: NREM (blue) vs REM (orange)

2. **Energy Landscape Descent**:
   - 2D energy surface (PCA of pattern space)
   - Trajectory: Memory embedding movement during consolidation
   - Before/after markers: Show energy reduction
   - Attractor wells: Shaded regions of low energy

3. **Consolidation Priority Matrix**:
   - Heatmap: Memories (rows) × Features (columns)
   - Features: Recency, novelty, RPE, replay count
   - Color: Priority contribution (warmer = higher)
   - Sorting: By total priority (most consolidated at top)

---

### 4.2 NREM vs REM Phase Dynamics

**Biological Distinction**:
- **NREM (slow-wave sleep)**: Hippocampal replay, synaptic downscaling, declarative memory consolidation
- **REM (paradoxical sleep)**: Cortical integration, emotional processing, procedural memory

**Phase Indicators**:
```python
def detect_sleep_phase(state: NeuromodulatorTelemetry) -> str:
    """Infer sleep phase from NT profile."""

    # NREM: Low ACh, low NE, high 5-HT
    if state.acetylcholine < 0.3 and state.norepinephrine < 0.2 and state.serotonin > 0.5:
        return "nrem"

    # REM: High ACh, low NE, low 5-HT (similar to wake but no NE)
    elif state.acetylcholine > 0.6 and state.norepinephrine < 0.2 and state.serotonin < 0.3:
        return "rem"

    # Wake: High ACh, high NE
    elif state.acetylcholine > 0.5 and state.norepinephrine > 0.5:
        return "wake"

    else:
        return "drowsy"
```

**Visualization**: Phase timeline with NT profiles
- **Top panel**: Phase labels (wake/nrem/rem/drowsy) as colored bar
- **Middle panel**: NT levels (ACh, NE, 5-HT) as stacked area chart
- **Bottom panel**: Replay events (vertical lines) colored by phase
- **Annotations**: Phase transitions marked with duration

---

## V. Missing Visualizations (High Priority)

### 5.1 Plasticity Dynamics (LTP/LTD/Homeostasis)

**Current Coverage**: 37% (`plasticity_traces.py`)
**Missing**:
- BCM threshold evolution
- Synaptic scaling events
- Hebbian trace correlations

**Proposed**: BCM Learning Curve Live Update
- Scatter plot: Activation level (x) vs weight change (y)
- Theoretical curve: BCM function overlay
- Moving threshold line: Current θ_m position
- Color: Recent updates (fade over time)

---

### 5.2 Attractor Basin Dynamics

**Current Coverage**: 0%
**Missing**: State space trajectory through cognitive attractors

**Proposed**: 3D Attractor Landscape
- Surface: Energy as height (from `EnergyLandscape.compute_total_energy()`)
- Wells: Cognitive states (Focus, Explore, Consolidate, etc.)
- Trajectory: NT state evolution (line with time-colored gradient)
- Arrows: Energy gradient field

**Data Source**: `/ww/nca/attractors.py` + `/ww/nca/energy.py`

---

### 5.3 Arousal-Modulated Retrieval Sharpness

**Current Coverage**: 0%
**Missing**: Visualization of NE → Hopfield β coupling

**Biological Insight** (from `energy.py:192-225`):
High norepinephrine (arousal) increases Hopfield inverse temperature β, making pattern retrieval sharper (less generalization). Low NE → diffuse retrieval (creative associations).

**Proposed**: Interactive Retrieval Simulator
- Slider: NE level [0, 1]
- Display: Current β value (2-16)
- Heatmap: Softmax attention over stored patterns (sharper = darker)
- Comparison: Retrieved pattern at different NE levels

---

## VI. Implementation Roadmap

### Phase 1: Core Biological Validators (Week 1)
1. Firing rate bounds checker (Section III.1)
2. Oscillation coherence dashboard (Section III.2)
3. Neuromodulator ratio tracker (Section III.3)

**Deliverable**: `validation_dashboard.py` with real-time alerts

---

### Phase 2: Opponent Process Visualizations (Week 2)
1. DA-5HT phase plane (Section II.1)
2. E/I balance time series (Section II.2)
3. Arousal-beta coupling scatter plot (Section I.1)

**Deliverable**: `opponent_dynamics.py` with interactive phase portraits

---

### Phase 3: Consolidation & Replay (Week 3)
1. SWR sequence diagram (Section IV.1)
2. Energy landscape descent (Section IV.1)
3. NREM/REM phase timeline (Section IV.2)

**Deliverable**: `consolidation_telemetry.py` with sleep cycle analysis

---

### Phase 4: Advanced Dynamics (Week 4)
1. BCM curve live update (Section V.1)
2. 3D attractor basin (Section V.2)
3. Arousal-retrieval simulator (Section V.3)

**Deliverable**: `advanced_viz.py` with 3D/interactive components

---

## VII. Data Pipeline Architecture

### 7.1 Telemetry Collection

```python
class NCATelemeter:
    """Unified telemetry collector for NCA system."""

    def __init__(self,
                 hippocampus: HippocampalCircuit,
                 vta: VTACircuit,
                 raphe: RapheNucleus,
                 oscillator: FrequencyBandGenerator,
                 energy_landscape: EnergyLandscape):
        self.hippocampus = hippocampus
        self.vta = vta
        self.raphe = raphe
        self.oscillator = oscillator
        self.energy = energy_landscape

        # Time series buffers
        self._nt_history: Deque[NeuromodulatorTelemetry] = deque(maxlen=10000)
        self._hippo_history: Deque[HippocampalTelemetry] = deque(maxlen=10000)
        self._osc_history: Deque[OscillationTelemetry] = deque(maxlen=10000)
        self._swr_events: List[SWREvent] = []

    def collect_snapshot(self) -> SystemSnapshot:
        """Collect all telemetry at current timestep."""
        nt = self._collect_neuromodulator()
        hippo = self._collect_hippocampal()
        osc = self._collect_oscillation()

        # Store in buffers
        self._nt_history.append(nt)
        self._hippo_history.append(hippo)
        self._osc_history.append(osc)

        return SystemSnapshot(
            nt=nt, hippo=hippo, osc=osc,
            timestamp=datetime.now()
        )

    def detect_swr(self, threshold: float = 0.8) -> Optional[SWREvent]:
        """Detect sharp-wave ripple based on state."""
        # Criteria: Low ACh, low NE, high gamma power
        current_nt = self._nt_history[-1]
        current_osc = self._osc_history[-1]

        if (current_nt.acetylcholine < 0.3 and
            current_nt.norepinephrine < 0.2 and
            current_osc.gamma_power > threshold):

            # Generate replay sequence
            swr = self._generate_swr_event()
            self._swr_events.append(swr)
            return swr
        return None
```

---

### 7.2 Streaming to Visualization

**Option A: Real-time Dashboard (Streamlit/Dash)**
```python
import streamlit as st

telemeter = NCATelemeter(...)

while True:
    snapshot = telemeter.collect_snapshot()

    # Update visualizations
    st.line_chart(telemeter.get_nt_timeseries())
    st.plotly_chart(create_opponent_phase_plane())
    st.metric("VTA Firing Rate", f"{snapshot.nt.vta_firing_rate:.1f} Hz")

    time.sleep(0.1)  # 10 Hz update
```

**Option B: Batch Export (HDF5)**
```python
import h5py

with h5py.File("nca_telemetry.h5", "w") as f:
    nt_group = f.create_group("neuromodulator")
    nt_group.create_dataset("dopamine", data=[s.dopamine for s in nt_history])
    nt_group.create_dataset("timestamps", data=[s.timestamp for s in nt_history])

    # ... similar for all telemetry streams
```

---

## VIII. Validation Criteria

### 8.1 Biological Plausibility Checklist

| Metric | Target Range | Data Source | Pass/Fail |
|--------|--------------|-------------|-----------|
| VTA tonic firing | 4-5 Hz | `VTAState.current_rate` | ✓/✗ |
| VTA burst firing | 20-40 Hz | `VTAState.current_rate` (burst mode) | ✓/✗ |
| Raphe firing | 2-5 Hz | `RapheNucleusState.firing_rate` | ✓/✗ |
| Theta frequency | 4-8 Hz | `OscillatorState.theta_freq` | ✓/✗ |
| Gamma frequency | 30-80 Hz | `OscillatorState.gamma_freq` | ✓/✗ |
| PAC strength | 0.2-0.5 | `pac.compute_modulation_index()` | ✓/✗ |
| DG sparsity | 0.5-5% | `HippocampalState.dg_sparsity` | ✓/✗ |
| E/I balance | 0.9-1.1 | `glutamate / (gaba + 1e-6)` | ✓/✗ |

**Automated Test**:
```python
def test_biological_plausibility(telemeter: NCATelemeter):
    snapshot = telemeter.collect_snapshot()
    violations = []

    # Check all criteria
    if not (4.0 <= snapshot.nt.vta_firing_rate <= 5.0):
        violations.append("VTA tonic firing out of range")

    # ... similar for all metrics

    assert len(violations) == 0, f"Bio violations: {violations}"
```

---

### 8.2 Emergent Behavior Verification

**Goal**: Validate that opponent processes produce expected behavioral modes

| NT Configuration | Expected Mode | Test Scenario |
|------------------|---------------|---------------|
| High DA, low 5-HT | Impulsive/reward-seeking | Rapid pattern encoding, low theta |
| Low DA, high 5-HT | Patient/aversive | Slow consolidation, high alpha |
| High ACh, low NE | Encoding/learning | CA1 novelty → VTA coupling active |
| Low ACh, low NE | Consolidation/sleep | SWR events, synaptic scaling |
| High NE, high ACh | Aroused attention | High beta, sharp Hopfield retrieval |

**Test Case Example**:
```python
def test_consolidation_mode():
    """Verify low ACh + low NE triggers SWR."""
    # Set NT state
    telemeter.vta.state.current_da = 0.3  # Baseline
    telemeter.raphe.state.extracellular_5ht = 0.6  # High patience

    # Simulate ACh/NE drop
    ach_level = 0.2
    ne_level = 0.1

    # Collect snapshot
    snapshot = telemeter.collect_snapshot()

    # Should detect NREM phase
    phase = detect_sleep_phase(snapshot.nt)
    assert phase == "nrem", f"Expected NREM, got {phase}"

    # Should trigger SWR
    swr = telemeter.detect_swr()
    assert swr is not None, "SWR not triggered in consolidation mode"
    assert swr.phase == "nrem"
```

---

## IX. References

### Neuroscience Literature
1. **Dopamine**: Schultz et al. (1997) - Reward prediction error
2. **Serotonin**: Dayan & Huys (2009) - Opponent processes, patience
3. **Hippocampus**: Rolls (2013) - Pattern separation/completion
4. **Theta-gamma PAC**: Lisman & Jensen (2013) - Neural code
5. **Raphe autoreceptors**: Blier & de Montigny (1987) - 5-HT1A dynamics
6. **Modern Hopfield**: Ramsauer et al. (2020) - Exponential capacity
7. **BCM learning**: Bienenstock et al. (1982) - Sliding threshold
8. **Sharp-wave ripples**: Buzsáki (2015) - Memory consolidation

### World Weaver Source Files
- `/ww/nca/vta.py` - VTA dopamine circuit
- `/ww/nca/raphe.py` - Raphe serotonin circuit
- `/ww/nca/hippocampus.py` - DG/CA3/CA1 subregions
- `/ww/nca/oscillators.py` - Theta/alpha/beta/gamma rhythms
- `/ww/nca/energy.py` - Hopfield integration, arousal-beta coupling
- `/ww/visualization/neuromodulator_state.py` - Current NT viz (35% coverage)
- `/ww/visualization/pattern_separation.py` - DG viz (68% coverage)
- `/ww/visualization/plasticity_traces.py` - LTP/LTD viz (37% coverage)
- `/ww/visualization/consolidation_replay.py` - SWR viz (39% coverage)

---

## X. Summary of Recommendations

### Immediate Priorities (Sprint 5)
1. **Opponent process phase planes** (DA-5HT, E/I balance) - No current visualization
2. **Biological validation dashboard** (firing rates, oscillation bounds) - Critical for debugging
3. **Arousal-beta coupling visualization** - Enables tuning of retrieval sharpness
4. **SWR detection and replay sequence diagram** - Core consolidation mechanism invisible

### Instrumentation Gaps
- **Missing data collection**: SWR events not currently logged
- **Incomplete NT telemetry**: Missing firing rates from VTA/Raphe state objects
- **No consolidation metrics**: Energy landscape changes during replay not tracked

### Architectural Improvements
- **Unified telemeter class**: Single point of data collection across NCA modules
- **HDF5 export pipeline**: Enable offline analysis and archival
- **Real-time alerts**: Violation of biological bounds should trigger warnings
- **Interactive dashboards**: Streamlit/Dash for live system monitoring

---

**Next Steps**: Implement Phase 1 validators (Section VI) and instrument `NCATelemeter` data pipeline (Section VII.1).
