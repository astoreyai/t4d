# T4DM NCA Visualization: Biological Perspective - Quick Answers

## 1. What biological signals are ESSENTIAL to visualize?

### Core Triad (Minimum Viable Observability)

**A. Neuromodulator Opponent Processes**
- **DA/5-HT ratio**: Reward-seeking vs impulse control (current: not visualized)
  - Source: `VTAState.current_da`, `RapheNucleusState.extracellular_5ht`
  - Visualization: 2D phase plane with nullclines and trajectory
  - Why: Determines behavioral mode (impulsive/patient/balanced)

- **Glu/GABA (E/I balance)**: Stability margin (current: not visualized)
  - Source: NT state vector indices [4, 5]
  - Visualization: Time series with critical zone (|E/I| < 0.1) shaded
  - Why: Prevents runaway excitation, validates gamma frequency modulation

**B. Hippocampal Novelty → VTA Dopamine Coupling**
- **CA1 novelty signal → VTA RPE injection** (current: partially in `consolidation_replay.py`)
  - Source: `HippocampalState.novelty_score`, `VTACircuit.connect_from_hippocampus()`
  - Visualization: Dual time series with latency measurement
  - Why: Core memory-reward integration, validates one-shot learning mechanism

- **CA3 Hopfield energy landscape** (current: not visualized)
  - Source: `CA3Layer.complete()` return value `energy`
  - Visualization: 2D PCA slice showing attractor wells and current position
  - Why: Validates pattern completion convergence, detects interference

**C. Theta-Gamma Phase-Amplitude Coupling (PAC)**
- **Theta phase gating of encoding** (current: not visualized)
  - Source: `FrequencyBandGenerator.get_encoding_signal()`, `theta.phase`
  - Visualization: Circular plot with encoding/retrieval zones marked
  - Why: Validates Forward-Forward phase separation, critical for consolidation

- **PAC modulation index** (current: not visualized)
  - Source: `PhaseAmplitudeCoupling.compute_modulation_index()`
  - Visualization: Comodulogram (theta phase × gamma amplitude heatmap)
  - Why: Measures working memory capacity (4-8 gamma cycles per theta = 4-8 items)

### Biological Validation Signals

**D. Firing Rate Bounds** (current: not checked)
- VTA DA neurons: 4-5 Hz tonic, 20-40 Hz burst, 0-2 Hz pause
- Raphe 5-HT neurons: 2-5 Hz tonic, 8 Hz max
- DG granule cells: ~0.05 Hz (0.5-5% sparsity × 4096 cells)
- Visualization: Real-time dashboard with green/yellow/red status lights
- Why: Out-of-range = model bug or bio-implausibility

---

## 2. How to represent opponent process dynamics?

### DA-5HT Opponent Process (Reward vs Patience)

**Phase Plane Visualization**:
```
       Serotonin (5-HT)
            ↑ 1.0
            |
   Patient  |         Balanced
  (avoid)   |  ●      (homeostasis)
            |    ●
       0.5  + - - - - - - - - >
            |       ●  ●
 Impulsive  |          ●   Reward-seeking
  (burst)   |              (exploration)
            |
       0.0  +--------------------→ Dopamine (DA)
           0.0        0.5        1.0

Nullclines:
- dDA/dt = 0: Vertical line at DA_setpoint (VTA tonic equilibrium)
- d5HT/dt = 0: Horizontal line at 5HT_setpoint (Raphe autoreceptor equilibrium)
- Fixed point: (DA=0.3, 5HT=0.4) → healthy baseline

Trajectory:
- Real-time path showing last 100 timesteps
- Color: Time gradient (recent = bright)
- Arrows: Velocity vectors showing dDA/dt, d5HT/dt

Interaction overlay:
- Raphe → VTA inhibition: Shaded band showing 5-HT suppresses DA
- VTA → Raphe feedback: Positive RPE reduces 5-HT (reward decreases patience)
```

**Data Extraction**:
```python
# From VTA
da_level = vta.state.current_da
vta_firing_mode = vta.state.firing_mode  # TONIC/BURST/PAUSE

# From Raphe
serotonin = raphe.state.extracellular_5ht
raphe_inhibition = raphe.get_vta_inhibition()  # How much 5-HT suppresses DA

# Compute opponent ratio
ratio = da_level / (serotonin + 1e-6)
behavioral_mode = classify_mode(ratio)  # impulsive/balanced/patient
```

**Key Insights from Visualization**:
1. **Oscillations**: Healthy system shows bounded oscillation around fixed point
2. **Runaway**: Trajectory escaping bounds = broken homeostasis
3. **Mode shifts**: Abrupt changes in behavioral quadrant = context transitions

---

### E/I Balance (Glutamate vs GABA)

**Time Series with Criticality Indicator**:
```
Concentration
    ↑ 1.0     Glu ▬▬▬▬  (green)
              GABA ════  (red)
    | ●●●      E/I ─ ─   (blue)
    |  ●●
0.5 + - - - - ●● - - - - - Critical Zone (shaded yellow)
    |       ●●
    |      ●
    |     ●
0.0 +─────────────────────────→ Time

    Gamma Freq (Hz) [right Y-axis]
    ↑ 80
    | ↗↗  (faster gamma with more GABA)
    |   ↗
 40 + - - - - baseline
    |  ↘
    | ↘ ↘  (slower gamma with more Glu)
    ↓ 30

Critical Zone: |Glu - GABA| < 0.1
- Risk of instability
- Alert: "Near E/I critical point - possible runaway excitation"
```

**Data Extraction**:
```python
# From NT state vector (indices 4, 5)
glu_level = nt_state[4]  # Glutamate
gaba_level = nt_state[5]  # GABA

ei_balance = glu_level - gaba_level  # [-1, 1]
criticality_margin = abs(ei_balance)  # < 0.1 = dangerous

# Effect on gamma frequency
gamma_freq = oscillator.gamma.freq  # Modulated by E/I
# More GABA (I) → tighter interneuron loop → faster gamma
```

---

## 3. Key biological invariants to monitor

### Invariant Matrix

| Category | Metric | Biological Range | Data Source | Violation Severity |
|----------|--------|------------------|-------------|-------------------|
| **Firing Rates** | VTA tonic | 4-5 Hz | `VTAState.current_rate` | HIGH |
| | VTA burst | 20-40 Hz | `VTAState.current_rate` (burst mode) | HIGH |
| | Raphe tonic | 2-5 Hz | `RapheNucleusState.firing_rate` | HIGH |
| **Oscillations** | Theta freq | 4-8 Hz | `OscillatorState.theta_freq` | MEDIUM |
| | Gamma freq | 30-80 Hz | `OscillatorState.gamma_freq` | MEDIUM |
| | PAC MI | 0.2-0.5 | `pac.compute_modulation_index()` | LOW |
| **Hippocampus** | DG sparsity | 0.5-5% | `HippocampalState` (implicit) | MEDIUM |
| | CA3 iters | < 10 | `HippocampalState.completion_iterations` | MEDIUM |
| **Balance** | E/I ratio | 0.9-1.1 | `glu / (gaba + eps)` | HIGH |
| | DA/5HT ratio | 0.7-1.3 | `da / (serotonin + eps)` | MEDIUM |
| **Arousal** | Hopfield β | 2-16 | `EnergyLandscape._current_beta` | LOW |

### Criticality Detection

**E/I Imbalance** (most dangerous):
```python
def check_ei_criticality(glu: float, gaba: float) -> str:
    ratio = glu / (gaba + 1e-6)
    margin = abs(1.0 - ratio)

    if margin < 0.1:
        return "CRITICAL: Near E/I critical point"
    elif margin < 0.2:
        return "WARNING: E/I imbalance detected"
    else:
        return "OK"
```

**VTA Firing Rate Anomaly**:
```python
def validate_vta_firing(rate: float, mode: VTAFiringMode) -> bool:
    if mode == VTAFiringMode.TONIC:
        return 4.0 <= rate <= 5.0
    elif mode == VTAFiringMode.PHASIC_BURST:
        return 20.0 <= rate <= 40.0
    elif mode == VTAFiringMode.PHASIC_PAUSE:
        return 0.0 <= rate <= 2.0
    return False
```

**Theta-Gamma Coupling Failure**:
```python
def check_pac_health(mi: float, wm_capacity: int) -> str:
    issues = []

    if mi < 0.1:
        issues.append("PAC too weak - theta/gamma decoupled")
    if mi > 0.8:
        issues.append("PAC too strong - overly rigid")
    if not (4 <= wm_capacity <= 8):
        issues.append(f"WM capacity {wm_capacity} outside Miller's 7±2")

    return "; ".join(issues) if issues else "OK"
```

### Real-Time Dashboard Layout

```
┌─────────────────────────────────────────────────────────┐
│  BIOLOGICAL INVARIANTS DASHBOARD                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Firing Rates                                           │
│  ┌─────────┬──────────┬────────────┬────────┐          │
│  │ Neuron  │ Current  │ Range      │ Status │          │
│  ├─────────┼──────────┼────────────┼────────┤          │
│  │ VTA     │ 4.3 Hz   │ 4-5 Hz     │ ✓ PASS │          │
│  │ Raphe   │ 3.1 Hz   │ 2-5 Hz     │ ✓ PASS │          │
│  └─────────┴──────────┴────────────┴────────┘          │
│                                                          │
│  Oscillation Coherence                                  │
│  ┌─────────┬──────────┬────────────┬────────┐          │
│  │ Band    │ Freq     │ Range      │ Status │          │
│  ├─────────┼──────────┼────────────┼────────┤          │
│  │ Theta   │ 6.2 Hz   │ 4-8 Hz     │ ✓ PASS │          │
│  │ Gamma   │ 38 Hz    │ 30-80 Hz   │ ✓ PASS │          │
│  │ PAC MI  │ 0.32     │ 0.2-0.5    │ ✓ PASS │          │
│  └─────────┴──────────┴────────────┴────────┘          │
│                                                          │
│  Opponent Processes                                     │
│  ┌─────────┬──────────┬────────────┬────────┐          │
│  │ Ratio   │ Current  │ Safe Range │ Status │          │
│  ├─────────┼──────────┼────────────┼────────┤          │
│  │ DA/5HT  │ 0.85     │ 0.7-1.3    │ ✓ PASS │          │
│  │ E/I     │ 0.92     │ 0.9-1.1    │ ⚠ WARN │          │
│  └─────────┴──────────┴────────────┴────────┘          │
│                                                          │
│  Alerts: E/I margin = 0.08 (approaching critical)       │
└─────────────────────────────────────────────────────────┘
```

---

## 4. How to visualize consolidation/replay during sleep?

### Sleep Phase Detection (from NT profile)

**State Machine**:
```
   Wake → Drowsy → NREM → REM → (cycle)
    ↑                        ↓
    └────────────────────────┘

NT Signatures:
┌───────┬────────┬───────┬────────┐
│ State │ ACh    │ NE    │ 5-HT   │
├───────┼────────┼───────┼────────┤
│ Wake  │ High   │ High  │ Med    │
│ Drowsy│ Med    │ Med   │ High   │
│ NREM  │ Low    │ Low   │ High   │
│ REM   │ High   │ Low   │ Low    │
└───────┴────────┴───────┴────────┘

Detection Logic:
if ACh < 0.3 and NE < 0.2 and 5HT > 0.5:
    phase = "NREM"  # Trigger SWR detection
elif ACh > 0.6 and NE < 0.2 and 5HT < 0.3:
    phase = "REM"   # Cortical integration
```

### SWR Sequence Visualization

**Diagram Structure**:
```
    Priority Score
         ↑ 1.0
         |                 Memory IDs (hover for content)
         |     ●           ┌─────────────┐
    0.8  |  ●     ●        │ mem_abc123  │
         |           ●     │ Priority: 0.9│
    0.6  +→●→→●→→●→→●      │ Novelty: High│
         |  ↓  ↓  ↓  ↓     │ RPE: +0.7   │
    0.4  | Replay Order    └─────────────┘
         |  (arrows show
    0.2  |   temporal
         |   sequence)
       0 +────────────────→ Position in Sequence
         0  1  2  3  4  5  6  7

Color coding:
- Blue circles = NREM replay (declarative consolidation)
- Orange circles = REM replay (emotional/procedural)
- Size = Salience (recent + novel + high RPE)

Arrow thickness = Temporal association strength
```

**Data Extraction**:
```python
# Detect SWR trigger
def detect_swr(nt_state: NeuromodulatorTelemetry,
               osc_state: OscillationTelemetry) -> bool:
    """SWR occurs during NREM with high gamma bursts."""
    return (
        nt_state.acetylcholine < 0.3 and
        nt_state.norepinephrine < 0.2 and
        osc_state.gamma_power > 0.8  # Ripple = high-freq gamma burst
    )

# Generate replay sequence
def generate_swr_replay(ca3: CA3Layer) -> SWREvent:
    """Select memories for replay based on priority."""

    # Priority scoring
    priorities = []
    for i, pattern_id in enumerate(ca3._pattern_ids):
        metadata = ca3._pattern_metadata[i]

        recency = compute_recency(metadata['timestamp'])
        novelty = metadata.get('novelty_score', 0.5)
        rpe = metadata.get('vta_rpe', 0.0)

        # Weighted combination
        priority = 0.4 * recency + 0.4 * novelty + 0.2 * abs(rpe)
        priorities.append((pattern_id, priority))

    # Select top K for replay
    sorted_memories = sorted(priorities, key=lambda x: x[1], reverse=True)
    replay_sequence = sorted_memories[:7]  # Typical SWR = 5-10 memories

    return SWREvent(
        timestamp=datetime.now(),
        memories_replayed=[mem_id for mem_id, _ in replay_sequence],
        priority_scores=[score for _, score in replay_sequence],
        phase=detect_sleep_phase(nt_state)
    )
```

### Energy Landscape Descent Visualization

**Before → After Consolidation**:
```
    Energy
      ↑
      |     Before (scattered)
  0.5 |   ●  ●    ●
      |  ●     ●
  0.0 + - - - - - - - - - - → PCA Dimension 1
      |      ●●● After (converged)
 -0.5 |       ◉  ← Attractor well
      |
      └─────────────────────→ PCA Dimension 2

Trajectory arrows:
● → ● → ● → ◉  (gradient descent toward attractor)

Contour lines: Equipotential energy surfaces
Shaded wells: Stable cognitive states (attractors)
```

**Data Extraction**:
```python
# Before SWR
pre_embeddings = [ca3._patterns[i] for i in replay_indices]
pre_energy = sum([
    hopfield.compute_energy(emb) for emb in pre_embeddings
])

# Simulate consolidation (Hopfield update)
for emb in pre_embeddings:
    retrieved, _, _ = hopfield.retrieve(emb, arousal_beta=high_beta)
    # Pattern moves toward attractor

# After SWR
post_embeddings = [ca3._patterns[i] for i in replay_indices]
post_energy = sum([
    hopfield.compute_energy(emb) for emb in post_embeddings
])

# Visualize
consolidation_improvement = pre_energy - post_energy  # Should be positive
assert consolidation_improvement > 0, "Consolidation failed to reduce energy"
```

### Full Sleep Cycle Timeline

```
Time (hours) →  0    1    2    3    4    5    6    7    8
              │WAKE│ Drowsy │NREM│REM│NREM│REM│NREM│REM│Wake
              └────┴────────┴────┴───┴────┴───┴────┴───┴────

NT Levels:
   ACh  ▬▬▬▬▬╲╲╲╲____╱╱____╱╱____╱╱▬▬▬▬
   NE   ▬▬▬▬▬╲╲╲╲╲╲__╱╱╲__╱╱╲__╱╱╲╲▬▬▬
   5-HT ──────╱╱╱╱▬▬▬▬╲╲▬▬╲╲▬▬╲╲╲╲╲────

SWR Events (vertical lines):
                    | | |  |  | | |
                    ↑ NREM replay bursts

Consolidation Metrics:
   Energy ──────────╲╲╲╲╲╲╲╲╲╲╲╲──────
   (lower = better)  ↓ Consolidation happening

   CA3 Size ─────────▬▬▬▬▬▬▬▬▬▬▬▬─────
   (constant pattern count)
```

---

## Summary: Implementation Priorities

### Sprint 5 Week 1: Core Validators
1. ✓ **Firing rate dashboard** - Real-time violation detection
2. ✓ **E/I criticality monitor** - Prevent runaway excitation
3. ✓ **Oscillation bounds checker** - Validate theta/gamma/PAC

### Sprint 5 Week 2: Opponent Dynamics
1. ✓ **DA-5HT phase plane** - Behavioral mode visualization
2. ✓ **E/I time series** - Stability margin tracking
3. ✓ **Arousal-beta scatter plot** - Retrieval sharpness tuning

### Sprint 5 Week 3: Consolidation
1. ✓ **SWR detection** - Identify replay events from NT+OSC state
2. ✓ **Replay sequence diagram** - Priority-based memory selection
3. ✓ **Energy landscape descent** - Validate consolidation improves storage
4. ✓ **Sleep phase timeline** - NT profile-based state detection

### Sprint 5 Week 4: Advanced
1. ⚠ **3D attractor basin** - Cognitive state space navigation
2. ⚠ **Interactive beta simulator** - Arousal → retrieval sharpness demo
3. ⚠ **BCM curve live update** - Plasticity threshold adaptation

---

**Files Created**:
- `/mnt/projects/t4d/t4dm/docs/visualization_biology_spec.md` - Full specification (10,000+ words)
- `/mnt/projects/t4d/t4dm/docs/visualization_biology_answers.md` - This summary (concise answers)

**Next Action**: Implement `NCATelemeter` class (Section VII.1 in spec) to unify data collection across VTA, Raphe, Hippocampus, Oscillator modules.
