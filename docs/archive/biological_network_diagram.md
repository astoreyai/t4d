# World Weaver Biological Network Architecture

**Visual Network Maps and Connectivity Diagrams**

---

## 1. COMPLETE SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    WORLD WEAVER BIOLOGICAL NETWORK                      │
│                          (Phase 1-3 Complete)                           │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌───────────────────┐
                        │  CORTICAL LAYER   │
                        │   (Future Phase)  │
                        └─────────┬─────────┘
                                  │
    ┌─────────────────────────────┼─────────────────────────────┐
    │                             │                             │
    ▼                             ▼                             ▼
┌────────┐                  ┌──────────┐                  ┌────────┐
│ VTA/DA │◄────────────────►│ LC/NE    │                  │ Raphe  │
│ Reward │  Opponent        │ Arousal  │                  │ 5-HT   │
│ RPE    │  Process         │ Surprise │                  │Patience│
└───┬────┘                  └────┬─────┘                  └───┬────┘
    │                            │                            │
    │ DA modulation              │ NE modulation              │ 5-HT
    │                            │                            │
    └────────────┬───────────────┴──────────────┬─────────────┘
                 │                              │
                 ▼                              ▼
        ┌─────────────────┐          ┌──────────────────┐
        │  HIPPOCAMPUS    │          │   OSCILLATORS    │
        │  EC→DG→CA3→CA1  │          │ Theta/Gamma/Alpha│
        │  Spatial Cells  │          │  Delta/Spindles  │
        └────────┬────────┘          └────────┬─────────┘
                 │                            │
                 │ Memory                     │ Rhythm
                 │ Encoding                   │ Coordination
                 │                            │
                 └────────────┬───────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  CONSOLIDATION   │
                    │  Sleep/SWR/Glymp │
                    │  Adenosine       │
                    └──────────────────┘
```

---

## 2. NEUROMODULATOR NETWORK (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│               NEUROMODULATOR INTERACTION NETWORK                │
│                  (All connections validated)                    │
└─────────────────────────────────────────────────────────────────┘

                         ┌──────────────┐
                         │     VTA      │
                         │  (Dopamine)  │
                         │              │
                         │  Tonic: 4.5Hz│
                         │  Burst: 20Hz │
                         │  Pause: 0Hz  │
                         └──┬────────┬──┘
                            │        │
            ┌───────────────┘        └───────────────┐
            │ Inhibit                    Excite      │
            │ (-0.3 via 5HT2C)          (Novelty)    │
            │                                        │
            ▼                                        ▼
    ┌──────────────┐                        ┌──────────────┐
    │    RAPHE     │                        │ HIPPOCAMPUS  │
    │ (Serotonin)  │                        │   (CA1)      │
    │              │                        │              │
    │ Rate: 2.5Hz  │                        │ Novelty Det  │
    │ Autoinhibit  │                        │ EC-CA3 Δ     │
    └──────┬───────┘                        └──────────────┘
           │
           │ 5-HT modulation
           │ (Patience ↑ gamma ↑)
           │
           ▼
    ┌──────────────┐
    │   STRIATUM   │
    │  D1/D2 MSN   │      ┌──────────────┐
    │              │◄─────┤     LC/NE    │
    │ Action Sel   │  NE  │ (Locus Coer) │
    └──────────────┘      │              │
                          │ Tonic: 3Hz   │
                          │ Phasic: 15Hz │
                          └──────┬───────┘
                                 │
                    ┌────────────┴────────────┐
                    │ NE → ↓ Alpha            │
                    │ NE → ↑ Signal/Noise     │
                    │ NE → ↑ Pattern Sep (DG) │
                    │ NE → ↓ Glymphatic       │
                    └─────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  ACETYLCHOLINE (ACh) - Not shown above for clarity              │
│  • Basal forebrain → Cortex/Hippocampus                         │
│  • High ACh → Encoding mode (DG/CA3 separation)                 │
│  • Low ACh → Retrieval mode (CA3 completion)                    │
│  • High ACh → Blocks SWR (no replay during REM)                 │
│  • ACh → ↑ Theta power (medial septum)                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. HIPPOCAMPAL CIRCUIT

```
┌─────────────────────────────────────────────────────────────────┐
│                   HIPPOCAMPAL MEMORY CIRCUIT                    │
│              (Trisynaptic Loop + Direct Path)                   │
└─────────────────────────────────────────────────────────────────┘

    ENTORHINAL CORTEX (EC Layer II/III)
         │                    │
         │ Perforant Path     │ Temporoammonic Path (Direct)
         │                    │
         ▼                    │
    ┌──────────────┐          │
    │ DENTATE      │          │
    │ GYRUS (DG)   │          │
    │              │          │
    │ • 4096 cells │          │
    │ • 4% sparse  │          │
    │ • NE ↑ sep   │          │
    └──────┬───────┘          │
           │ Mossy Fibers     │
           │                  │
           ▼                  │
    ┌──────────────┐          │
    │     CA3      │          │
    │ Autoassoc    │          │
    │              │          │
    │ • Hopfield   │          │
    │ • Beta=8.0   │          │
    │ • Recurrent  │          │
    └──────┬───────┘          │
           │ Schaffer         │
           │ Collaterals      │
           ▼                  │
    ┌──────────────┐          │
    │     CA1      │◄─────────┘
    │ Comparator   │
    │              │
    │ EC ≠ CA3?    │
    │ → Novelty    │
    └──────┬───────┘
           │
           │ Novelty Signal
           │
           ▼
    ┌──────────────┐
    │     VTA      │
    │ (Dopamine)   │
    │ Novelty→RPE  │
    └──────────────┘

PATTERN SEPARATION (DG):
Input: [A, B] similar patterns
Output: [A', B'] orthogonalized
Metric: Cosine similarity ↓

PATTERN COMPLETION (CA3):
Input: [A'] partial cue
Output: [A] full pattern
Mechanism: Hopfield energy minimization

NOVELTY DETECTION (CA1):
Mismatch = ||EC_input - CA3_output||
If mismatch > 0.3 → Encoding mode
If mismatch < 0.3 → Retrieval mode
```

---

## 4. SPATIAL NAVIGATION SYSTEM

```
┌─────────────────────────────────────────────────────────────────┐
│              PLACE CELLS + GRID CELLS (Nobel 2014)              │
└─────────────────────────────────────────────────────────────────┘

    ENTORHINAL CORTEX (MEC)
    ┌────────────────────────────────────┐
    │  GRID CELLS (Hexagonal Lattice)    │
    │                                    │
    │  Module 1: Scale 0.3 (fine)        │
    │  Module 2: Scale 0.5 (medium)      │
    │  Module 3: Scale 0.8 (coarse)      │
    │                                    │
    │  Gridness Score:                   │
    │  min(r60,r120) - max(r30,r90,r150) │
    │  > 0.3 = valid grid                │
    └────────────┬───────────────────────┘
                 │ Grid→Place
                 │ Spatial basis
                 ▼
    ┌────────────────────────────────────┐
    │  HIPPOCAMPUS CA3/CA1               │
    │  PLACE CELLS (Location-specific)   │
    │                                    │
    │  • 100 place cells (simplified)    │
    │  • Gaussian receptive fields       │
    │  • Sigma = 0.15 spatial units      │
    │  • 4% sparsity (like DG)           │
    │                                    │
    │  Place Field:                      │
    │  P(x,y) = exp(-||loc - center||²   │
    │                  / (2σ²))          │
    └────────────────────────────────────┘

    ┌────────────────────────────────────┐
    │  HEAD DIRECTION CELLS              │
    │  (Postsubiculum/Anterodorsal Thal) │
    │                                    │
    │  • Tuned to allocentric heading    │
    │  • Ring attractor dynamics         │
    │  • Integrates angular velocity     │
    └────────────────────────────────────┘
```

---

## 5. SLEEP CONSOLIDATION NETWORK

```
┌─────────────────────────────────────────────────────────────────┐
│                  SLEEP-DEPENDENT CONSOLIDATION                  │
│              (Hinton 8.0-9.0 Sleep Framework)                   │
└─────────────────────────────────────────────────────────────────┘

    WAKE PHASE (16 hours typical)
         │
         ├─► Activity → ATP consumption
         │
         ├─► ATP → ADP → AMP → Adenosine
         │                      (accumulation)
         │
         └─► Adenosine ↑ → Sleep Pressure ↑
                 │
                 │ Threshold: 0.7
                 │
                 ▼
    ┌────────────────────────────────────┐
    │        SLEEP INITIATION            │
    │                                    │
    │  • ACh ↓ (allows SWR)              │
    │  • NE ↓ (allows glymphatic)        │
    │  • Adenosine → ↑ Delta power       │
    └────────────┬───────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────┐
    │     NREM SLEEP (Deep SWS)          │
    │                                    │
    │  DELTA (0.5-4 Hz)                  │
    │  ┌──────────┐  ┌──────────┐        │
    │  │ Up-State │  │Down-State│        │
    │  │ 200-500ms│  │ 200-500ms│        │
    │  │          │  │          │        │
    │  │ Consolid │  │ Downscale│        │
    │  └────┬─────┘  └──────────┘        │
    │       │                            │
    │       ├─► SPINDLES (11-16 Hz)      │
    │       │   • TRN-Cortex             │
    │       │   • Gate HPC→Ctx transfer  │
    │       │   • 0.5-2s bursts          │
    │       │                            │
    │       └─► SWR (150-250 Hz)         │
    │           • CA3/CA1 ripples        │
    │           • Memory replay          │
    │           • ~80ms duration         │
    │           • Compression 10x        │
    │                                    │
    │  GLYMPHATIC CLEARANCE              │
    │  • Low NE → Astrocyte shrink       │
    │  • Interstitial space ↑ 60%        │
    │  • CSF flow during delta up-states │
    │  • Waste removal 2x vs wake        │
    └────────────────────────────────────┘

    TIMING COORDINATION:
    ┌────────────────────────────────────┐
    │  Delta Up-State                    │
    │     ├─► 100-200ms delay            │
    │     ├─► Spindle initiation         │
    │     │   └─► Spindle plateau        │
    │     │       └─► SWR trigger         │
    │     │           └─► Replay          │
    │     │               └─► Consolidate │
    │     │                               │
    │     └─► Glymphatic pump            │
    │         (CSF oscillation)          │
    └────────────────────────────────────┘
```

---

## 6. OSCILLATION HIERARCHY

```
┌─────────────────────────────────────────────────────────────────┐
│           NEURAL OSCILLATION FREQUENCY HIERARCHY                │
│              (Cross-Frequency Coupling)                         │
└─────────────────────────────────────────────────────────────────┘

    FREQUENCY (Hz)
    0.01  0.1   1    10    100   1000
    │     │     │    │     │     │
    │     │     │    │     │     └─► Single Spikes (1000Hz)
    │     │     │    │     │
    │     │     │    │     └───────► Ripples (150-250Hz)
    │     │     │    │               SWR during sleep
    │     │     │    │               Memory replay
    │     │     │    │
    │     │     │    └─────────────► Gamma (30-100Hz)
    │     │     │                    E/I balance
    │     │     │                    Local processing
    │     │     │                    PAC with Theta
    │     │     │
    │     │     │    ┌─────────────► Beta (13-30Hz)
    │     │     │    │               Motor control
    │     │     │    │               DA modulated
    │     │     │    │
    │     │     │    └─────────────► Spindles (11-16Hz)
    │     │     │                    NREM stage 2
    │     │     │                    Memory gate
    │     │     │
    │     │     └──────────────────► Theta (4-8Hz)
    │     │                          Hippocampal
    │     │                          Encoding/Retrieval
    │     │                          ACh modulated
    │     │
    │     │     ┌──────────────────► Alpha (8-13Hz)
    │     │     │                    Thalamocortical
    │     │     │                    Idling/Inhibition
    │     │     │                    NE suppresses
    │     │     │
    │     └─────┴──────────────────► Delta (0.5-4Hz)
    │                                Slow-wave sleep
    │                                Up/Down states
    │                                Consolidation
    │
    └────────────────────────────────► Ultradian (90min)
                                     Sleep cycles
                                     (Not implemented)

PHASE-AMPLITUDE COUPLING (PAC):
┌────────────────────────────────────┐
│  Theta Phase → Gamma Amplitude     │
│                                    │
│     Theta Cycle (167ms @ 6Hz)      │
│     ┌──────────────────────┐       │
│     │    6-7 Gamma Bursts  │       │
│     │    (40Hz = 25ms each)│       │
│     └──────────────────────┘       │
│                                    │
│  Working Memory Capacity:          │
│  # Gamma cycles/Theta = 6-7 items  │
│  (Miller's 7±2)                    │
└────────────────────────────────────┘
```

---

## 7. NEUROTRANSMITTER DIFFUSION FIELD

```
┌─────────────────────────────────────────────────────────────────┐
│            NEURAL FIELD SOLVER (NCA Framework)                  │
│              6 NT Fields + Reaction-Diffusion                   │
└─────────────────────────────────────────────────────────────────┘

    Spatial Grid: [H x W] (e.g., 64x64)

    Field Indices:
    [0] Dopamine       (DA)  - VTA source
    [1] Serotonin      (5HT) - Raphe source
    [2] Norepinephrine (NE)  - LC source
    [3] Acetylcholine  (ACh) - Basal forebrain
    [4] GABA                 - Local inhibition
    [5] Glutamate            - Excitatory drive

    REACTION-DIFFUSION PDE:
    ∂u/∂t = D∇²u + R(u,v,w,...)
            ↑      ↑
        Diffusion  Reactions

    DIFFUSION CONSTANTS:
    D_DA  = 0.02  (slow volume transmission)
    D_5HT = 0.015 (very slow)
    D_NE  = 0.03  (moderate)
    D_ACh = 0.025 (moderate)
    D_GABA= 0.05  (fast, local)
    D_Glu = 0.08  (fastest, synapse)

    DECAY RATES:
    τ_DA  = 2.0s  (MAO/COMT)
    τ_5HT = 5.0s  (SERT, MAO)
    τ_NE  = 1.5s  (NET, MAO)
    τ_ACh = 0.5s  (AChE - fastest)
    τ_GABA= 0.2s  (GAT)
    τ_Glu = 0.1s  (EAAT - fastest)

    ASTROCYTE UPTAKE:
    • Glutamate → GLT-1/EAAT2 (80% uptake)
    • K+ buffering → Kir4.1 channels
    • Ca2+ waves → IP3-mediated propagation
```

---

## 8. STRIATAL CIRCUIT

```
┌─────────────────────────────────────────────────────────────────┐
│              BASAL GANGLIA ACTION SELECTION                     │
│              (Direct vs Indirect Pathways)                      │
└─────────────────────────────────────────────────────────────────┘

    CORTEX (State Representation)
         │
         │ Corticostriatal
         │
         ▼
    ┌────────────────────────────────┐
    │      STRIATUM (MSNs)           │
    │                                │
    │  D1-MSN        D2-MSN          │
    │  (Direct)      (Indirect)      │
    │                                │
    │  DA: ↑         DA: ↓           │
    │  (+0.7)        (-0.5)          │
    │                                │
    │  "GO"          "NO-GO"         │
    └─┬──────────────────────────┬───┘
      │                          │
      │ Direct Path              │ Indirect Path
      │ (facilitate)             │ (suppress)
      │                          │
      ▼                          ▼
    ┌────────┐              ┌────────┐
    │  GPi   │              │  GPe   │
    │ (inhib)│              │ (inhib)│
    └───┬────┘              └───┬────┘
        │                       │
        │ ↓ Inhibition          │
        │   to Thalamus         ▼
        │                   ┌────────┐
        │                   │  STN   │
        │                   │(excite)│
        │                   └───┬────┘
        │                       │
        │ ◄─────────────────────┘
        │   ↑ Excitation
        │   to GPi
        │
        ▼
    ┌────────┐
    │Thalamus│ → Cortex (Action)
    └────────┘

    OPPONENT PROCESS:
    • High DA: D1 "GO" > D2 "NO-GO" → Action
    • Low DA: D2 "NO-GO" > D1 "GO" → Inhibit
    • Balanced DA: Deliberation

    BISTABILITY:
    • MSNs have Up/Down states
    • Threshold: -50mV
    • Hysteresis in action selection
```

---

## 9. NETWORK VALIDATION CHECKLIST

### 9.1 Anatomical Connections

| Connection | Implemented | Literature | Status |
|------------|-------------|------------|--------|
| EC → DG → CA3 → CA1 | ✓ | Andersen 1971 | ✓ PASS |
| EC → CA1 (direct) | ✓ | Hasselmo 1997 | ✓ PASS |
| VTA → Striatum | ✓ | Gerfen 2011 | ✓ PASS |
| Raphe → VTA (5-HT2C) | ✓ | Di Matteo 2001 | ✓ PASS |
| LC → Cortex | ✓ | Aston-Jones 2005 | ✓ PASS |
| CA1 → VTA (novelty) | ✓ | Lisman & Grace 2005 | ✓ PASS |
| MS → HPC (ACh/theta) | ✓ | Hasselmo 2005 | ✓ PASS |
| TRN → Cortex (spindles) | ✓ | Steriade 1993 | ✓ PASS |

### 9.2 Functional Pathways

| Pathway | Function | Validated | Status |
|---------|----------|-----------|--------|
| Pattern Separation | DG orthogonalization | ✓ | ✓ PASS |
| Pattern Completion | CA3 Hopfield | ✓ | ✓ PASS |
| Novelty Detection | CA1 mismatch | ✓ | ✓ PASS |
| RPE Computation | VTA TD-error | ✓ | ✓ PASS |
| Surprise Detection | LC uncertainty | ✓ | ✓ PASS |
| Patience Modulation | Raphe gamma | ✓ | ✓ PASS |
| SWR Gating | ACh < 0.3 | ✓ | ✓ PASS |
| Glymphatic Gating | NE < 0.2 | ✓ | ✓ PASS |

### 9.3 Oscillation Coupling

| Coupling | Mechanism | Validated | Status |
|----------|-----------|-----------|--------|
| Theta-Gamma PAC | Phase-amplitude | ✓ | ✓ PASS |
| Delta-Spindle | Up-state timing | ✓ | ✓ PASS |
| Spindle-SWR | Sequential | ✓ | ✓ PASS |
| Alpha-NE | Suppression | ✓ | ✓ PASS |
| Theta-ACh | Enhancement | ✓ | ✓ PASS |

---

## 10. INTEGRATION POINTS

### 10.1 WW-Knowledge Integration
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/integration/knowledge.py`

**Flow**:
```
Hippocampus → Knowledge Graph
CA3 Patterns → Entities
CA1 Novelty → New Entity Creation
Spatial Cells → Spatial Embeddings
```

### 10.2 WW-Graph Integration
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/integration/graph.py`

**Flow**:
```
VTA RPE → Edge Weights (Hebbian + TD)
Raphe 5-HT → Temporal Links (patience)
LC NE → Salience Gating (surprise)
```

### 10.3 WW-Retriever Integration
**File**: `/mnt/projects/t4d/t4dm/src/t4dm/integration/retriever.py`

**Flow**:
```
CA3 Completion → Pattern Retrieval
DG Separation → Query Expansion
Theta Phase → Encoding/Retrieval Mode
```

---

## 11. FUTURE EXPANSION NODES

**Phase 4 Targets**:
```
┌─────────────────────────────────────┐
│  PREFRONTAL CORTEX                  │
│  • Working memory (DLPFC)           │
│  • Executive control (ACC)          │
│  • Value representation (OFC)       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  AMYGDALA                           │
│  • Threat detection                 │
│  • Emotional valence                │
│  • LC modulation                    │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  CEREBELLUM                         │
│  • Forward models                   │
│  • Error prediction                 │
│  • Sequence learning                │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  CIRCADIAN CLOCK (SCN)              │
│  • Process C (Borbély model)        │
│  • Melatonin rhythms                │
│  • Temperature coupling             │
└─────────────────────────────────────┘
```

---

**Document Version**: 1.0
**Last Updated**: 2026-01-04
**Maintained By**: World Weaver Development Team
**Related**: `biological_validation.md`, Module docstrings
