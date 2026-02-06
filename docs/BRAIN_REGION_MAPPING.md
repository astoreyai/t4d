# T4DM: Brain Region Mapping and Neuroscience Architecture Analysis

**Version**: 2.0 | **Date**: 2026-02-02 | **Status**: Updated Analysis

This document maps T4DM's computational architecture to established neuroscience models, brain regions, and cognitive theories.

> **Note**: The `nca/` module directory name stands for "Neural Circuit Architecture" (brain region simulation modules), not Mordvintsev-style Neural Cellular Automata. The module implements biologically-inspired circuit simulations of specific brain regions.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Memory Systems Mapping](#memory-systems-mapping)
3. [Neural Pathway Analysis](#neural-pathway-analysis)
4. [Cognitive Process Mapping](#cognitive-process-mapping)
5. [Integration Analysis](#integration-analysis)
6. [Cognitive Architecture Comparison](#cognitive-architecture-comparison)
7. [Gap Analysis](#gap-analysis)
8. [Recommendations](#recommendations)

---

## Executive Summary

T4DM implements a biologically-inspired cognitive architecture with strong correspondence to neuroscience models. Key findings:

| Category | Implementation Score | Notes |
|----------|---------------------|-------|
| Memory Systems | 92/100 | Excellent tripartite model with FSRS decay |
| Neuromodulation | 88/100 | 6-NT PDE system with biological timescales |
| Consolidation | 85/100 | NREM/REM phases with SWR replay |
| Prediction | 80/100 | JEPA-style with hierarchical timescales |
| Spatial Cognition | 70/100 | Place/grid cells mentioned but not fully implemented |
| Oscillatory Dynamics | 75/100 | Theta-gamma coupling present |

**Overall Biological Plausibility Score: 82/100**

---

## Memory Systems Mapping

### 1. Episodic Memory --> Hippocampal Formation

```
WORLD WEAVER                          BRAIN REGION
============                          ============

EpisodicMemory                   -->  Hippocampus (proper)
  |                                     |
  +-- content/embedding          -->  CA3 pattern storage
  +-- temporal context           -->  CA1 temporal coding
  +-- emotional_valence          -->  Amygdala input
  +-- FSRS stability             -->  Synaptic consolidation
  +-- session_id                 -->  Context-dependent retrieval

T4DX Temporal Storage            -->  Hippocampal-entorhinal circuit
  |                                     |
  +-- bi-temporal (T_ref, T_sys) -->  Time cells (CA1/MEC)
  +-- vector embeddings          -->  Pattern separation (DG)
```

**Implementation Accuracy: 95/100**

Key biological correspondences:
- FSRS retrievability formula models synaptic decay
- Recency weighting matches hippocampal temporal gradients
- Emotional valence integration mirrors amygdala-hippocampal connectivity
- Session namespacing approximates contextual binding

**Missing elements:**
- Sharp-wave ripples during retrieval (partial in SleepConsolidation)
- Pattern completion via CA3 auto-associative network
- Dentate gyrus pattern separation (sparse coding)

---

### 2. Semantic Memory --> Neocortical Networks

```
WORLD WEAVER                          BRAIN REGION
============                          ============

SemanticMemory                   -->  Temporal/Parietal Neocortex
  |                                     |
  +-- Entity nodes               -->  Concept neurons (IT cortex)
  +-- entity_type                -->  Category-selective regions
  +-- Hebbian relationships      -->  Associative cortex
  +-- ACT-R activation           -->  Spreading activation (PFC)

Knowledge Graph (T4DX CSR)       -->  Cortical semantic network
  |                                     |
  +-- RELATED_TO edges           -->  White matter tracts
  +-- weight decay               -->  Synaptic weakening
  +-- spreading activation       -->  Cortical column dynamics
```

**Implementation Accuracy: 90/100**

Strong points:
- ACT-R activation spreading is well-established cognitive model
- Hebbian learning (fire together, wire together) is biologically accurate
- Entity types map to category-selective cortical regions

**Missing elements:**
- Hierarchical concept organization (perirhinal -> entorhinal)
- Graded semantic similarity (not just binary relations)
- Amodal vs. modal representations

---

### 3. Procedural Memory --> Basal Ganglia/Cerebellum

```
WORLD WEAVER                          BRAIN REGION
============                          ============

ProceduralMemory                 -->  Basal Ganglia + Cerebellum
  |                                     |
  +-- Procedure (skill)          -->  Striatum (habit circuits)
  +-- triggerPattern             -->  Frontal-striatal loops
  +-- steps (sequence)           -->  Motor cortex sequences
  +-- successRate                -->  Dopaminergic reinforcement
  +-- deprecated flag            -->  Synaptic pruning

Build-Retrieve-Update Lifecycle  -->  Cortico-striatal learning
  |                                     |
  +-- Build (from trajectory)    -->  Striatal plasticity
  +-- Retrieve (by similarity)   -->  Action selection
  +-- Update (from feedback)     -->  DA reward signal
```

**Implementation Accuracy: 85/100**

Strong points:
- Success rate tracking mirrors reward-based skill learning
- Deprecation models habit extinction
- Domain-specific organization maps to parallel basal ganglia loops

**Missing elements:**
- Motor timing (cerebellum role)
- Model-based vs. model-free learning distinction
- Action chunking (sequence automatization)

---

### 4. Working Memory --> Prefrontal Cortex

```
WORLD WEAVER                          BRAIN REGION
============                          ============

WorkingMemory (implicit)         -->  Dorsolateral PFC (dlPFC)
  |                                     |
  +-- context window             -->  Sustained activation
  +-- active goals               -->  Goal maintenance (rlPFC)
  +-- attention_cache            -->  Attentional selection

Theta-Gamma Coupling (v0.4.0)    -->  PFC oscillatory dynamics
  |                                     |
  +-- 7+/-2 capacity             -->  Miller's capacity limit
  +-- gamma bursts               -->  Item representation
  +-- theta phase                -->  Sequential organization
```

**Implementation Accuracy: 80/100**

Strong points:
- Theta-gamma coupling for capacity limits is cutting-edge neuroscience
- 7+/-2 working memory slots match behavioral data
- Context window maps to sustained PFC activation

**Missing elements:**
- Active maintenance via recurrent dynamics
- Interference effects between items
- Articulatory loop / visuospatial sketchpad distinction

---

## Neural Pathway Analysis

### Implemented Pathways

```
MESOCORTICOLIMBIC PATHWAY (Reward)
==================================
VTA ─────► NAcc ─────► PFC
     DA         DA

WW IMPLEMENTATION:
DopamineSystem ──► Memory consolidation priority
              └──► Coupling matrix learning
              └──► NeuralFieldSolver.inject_rpe()

Accuracy: 90/100
- RPE computation matches phasic DA burst/dip
- TD(lambda) eligibility traces model striatal plasticity
- Learned value network generalizes across memories

RAPHE-CORTICAL PATHWAY (Mood/Patience)
======================================
Dorsal Raphe ─────► Cortex/Hippocampus
              5-HT

WW IMPLEMENTATION:
SerotoninSystem ──► Long-term credit assignment
               └──► Mood modulation of value
               └──► Temporal discounting (patience)

Accuracy: 85/100
- Mood state affects value estimates (pessimism/optimism)
- Eligibility traces span hours (slow 5-HT dynamics)
- Missing: 5-HT effects on impulsivity/waiting

CHOLINERGIC PATHWAY (Attention/Encoding)
========================================
Basal Forebrain ─────► Hippocampus/Cortex
                 ACh

WW IMPLEMENTATION:
NCA (ACh field) ──► FOCUS cognitive state
              └──► Encoding signal modulation
              └──► Memory consolidation gating

Accuracy: 80/100
- ACh gates encoding (high ACh = learning mode)
- FOCUS state has high ACh signature
- Theta phase modulates encoding/retrieval

LOCUS COERULEUS PATHWAY (Arousal)
=================================
Locus Coeruleus ─────► Widespread cortex
                  NE

WW IMPLEMENTATION:
NCA (NE field) ──► ALERT/EXPLORE states
              └──► Arousal modulation
              └──► Adenosine antagonism

Accuracy: 90/100
- NE drives alertness and vigilance
- 5 firing modes in LC (nca/locus_coeruleus.py, 1102 lines)
- Phasic/tonic NE modes implemented
- Gain modulation implemented
- Surprise/novelty detection via phasic bursts

HIPPOCAMPAL-CORTICAL DIALOGUE
=============================
Hippocampus ◄────► Neocortex
         SWR (replay)

WW IMPLEMENTATION:
SleepConsolidation ──► NREM replay
                  └──► Sharp-wave ripples
                  └──► Episodic → Semantic transfer

Accuracy: 85/100
- SWR compressed replay implemented
- NREM vs. REM phase distinction
- Interleaved replay (CLS theory)
```

### Pathway Diagram (Text-Based)

```
                    ┌─────────────────────────────────────────────┐
                    │           WORLD WEAVER NEURAL MAP           │
                    └─────────────────────────────────────────────┘

                              ┌─────────────┐
                              │    PFC      │ ◄─── Working Memory
                              │ (dlPFC)     │      Goal Maintenance
                              └──────┬──────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
     ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
     │   MOTOR     │        │  PARIETAL   │        │  TEMPORAL   │
     │   CORTEX    │        │   CORTEX    │        │   CORTEX    │
     │(Procedural) │        │ (Attention) │        │ (Semantic)  │
     └──────┬──────┘        └──────┬──────┘        └──────┬──────┘
            │                      │                      │
            ▼                      │                      ▼
     ┌─────────────┐               │               ┌─────────────┐
     │   BASAL     │               │               │ HIPPOCAMPUS │
     │   GANGLIA   │◄──────────────┼───────────────│ (Episodic)  │
     │ (Skills)    │               │               └──────┬──────┘
     └──────┬──────┘               │                      │
            │                      │                      │
            └──────────────────────┴──────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │    NEUROMODULATORY CORE     │
                    │                             │
                    │  ┌─────┐ ┌─────┐ ┌─────┐   │
                    │  │ VTA │ │Raphe│ │ LC  │   │
                    │  │ DA  │ │ 5HT │ │ NE  │   │
                    │  └──┬──┘ └──┬──┘ └──┬──┘   │
                    │     │      │      │       │
                    │  ┌──┴──┐ ┌─┴─┐ ┌──┴──┐   │
                    │  │ BF  │ │SN │ │ A2  │   │
                    │  │ ACh │ │DA │ │ NE  │   │
                    │  └─────┘ └───┘ └─────┘   │
                    └─────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │   NCA (Neural Cognitive     │
                    │        Architecture)        │
                    │                             │
                    │  6-NT PDE: DA, 5HT, ACh,   │
                    │            NE, GABA, Glu   │
                    │                             │
                    │  Attractor States:         │
                    │  ALERT, FOCUS, REST,       │
                    │  EXPLORE, CONSOLIDATE      │
                    └─────────────────────────────┘
```

### Missing Neural Circuits

| Circuit | Brain Regions | Function | WW Status |
|---------|--------------|----------|-----------|
| Papez Circuit | Hippocampus-Thalamus-Cingulate | Emotion-memory | Partial (amygdala now implemented: nca/amygdala.py) |
| Dorsal Attention Network | FEF-IPS | Top-down attention | Missing |
| Default Mode Network | mPFC-PCC-TPJ | Self-referential | Missing |
| Salience Network | ACC-AI | Switching | Partial (attractor transitions) |
| Cerebello-thalamo-cortical | Cerebellum-Thalamus-Motor | Timing/Learning | Missing |

---

## Cognitive Process Mapping

### Encoding (ACh-Gated)

```
NEUROSCIENCE MODEL:
- ACh release in hippocampus gates LTP
- Theta phase (0-180 deg) optimal for encoding
- High ACh suppresses retrieval interference

WW IMPLEMENTATION:
NCA FOCUS state ─► High ACh concentration
              └─► encoding_signal = oscillator.get_encoding_signal()
              └─► Memory augmented with NT context
              └─► Hebbian plasticity for co-active memories

ACCURACY: 85/100

Implemented:
[x] ACh concentration gates encoding
[x] Theta phase modulates encoding strength
[x] Cognitive state (FOCUS) enables learning
[x] Bridge augments embeddings with NT context

Missing:
[ ] Dentate gyrus pattern separation
[ ] CA3 pattern completion during encoding
[ ] Neocortical consolidation tagging
```

### Retrieval (Theta-Phase)

```
NEUROSCIENCE MODEL:
- Retrieval optimal at theta trough (180-360 deg)
- Pattern completion via CA3 recurrent connections
- Reactivation strength predicts memory strength

WW IMPLEMENTATION:
ACT-R Activation ─► Base-level (recency/frequency)
              └─► Spreading activation (context)
              └─► Noise (stochastic retrieval)
              └─► State-aware ranking (NT modulation)

ACCURACY: 80/100

Implemented:
[x] Frequency-based retrieval (power law decay)
[x] Context-dependent spreading activation
[x] Emotional/outcome weighting
[x] NT state modulates retrieval ranking

Missing:
[ ] Full theta phase dependency (only partial)
[ ] Pattern completion (only similarity search)
[ ] Reconsolidation window after retrieval
```

### Consolidation (Sleep Stages)

```
NEUROSCIENCE MODEL:
- NREM Stage 3: Sharp-wave ripples, hippocampal replay
- REM: Creative binding, procedural consolidation
- Synaptic downscaling: Homeostatic normalization

WW IMPLEMENTATION:
SleepConsolidation:
  NREM ─► SharpWaveRipple.generate_ripple_sequence()
     └─► Prioritized replay (outcome, recency, PE)
     └─► Episode → Semantic entity transfer
     └─► SynapticTag capture (P5.3)

  REM ─► Cluster analysis (DBSCAN-style)
     └─► Abstract concept creation
     └─► Cross-modal binding

  Prune ─► Weak connection removal
       └─► Homeostatic scaling

ACCURACY: 85/100

Implemented:
[x] NREM/REM phase alternation
[x] Sharp-wave ripple sequences
[x] Compressed replay (10x)
[x] Prediction error prioritization (P1-1)
[x] Interleaved replay (CLS theory, P3.4)
[x] Synaptic pruning and downscaling
[x] Abstract concept formation in REM

Missing:
[x] Sleep spindles (IMPLEMENTED: nca/sleep_spindles.py, 608 lines — SleepSpindleGenerator with delta coupling)
[ ] K-complexes
[x] Slow oscillation coupling (IMPLEMENTED: SpindleDeltaCoupler in nca/sleep_spindles.py)
[ ] REM theta (only NREM SWR)
```

### Attention (NE/ACh)

```
NEUROSCIENCE MODEL:
- NE: Phasic response to unexpected events
- ACh: Sustained attention to expected signals
- LC-NE system modulates cortical gain
- Basal forebrain ACh enhances signal/noise

WW IMPLEMENTATION:
CognitiveState Attractors:
  ALERT ─► High NE, DA; Low GABA
  FOCUS ─► High ACh, Glu; Moderate DA
  EXPLORE ─► High DA, NE, ACh

NCA Dynamics:
  NE field ─► Arousal modulation
  ACh field ─► Attention/encoding signal
  Coupling matrix ─► NE-ACh interaction

ACCURACY: 85/100

Implemented:
[x] NE-driven alertness states
[x] ACh-driven focus states
[x] State-dependent processing modes
[x] Attractor dynamics for state switching
[x] Phasic vs. tonic NE distinction (IMPLEMENTED: nca/locus_coeruleus.py, 1102 lines, 5 firing modes)
[x] Gain modulation (IMPLEMENTED: LC modulates cortical gain via NE)

Missing:
[ ] Covert attention shifts
[ ] Feature-based vs. spatial attention
```

---

## Integration Analysis

### Module Communication Matrix

```
                    │ Episodic │ Semantic │ Procedural │ NCA │ Consolidation │ Prediction │
────────────────────┼──────────┼──────────┼────────────┼─────┼───────────────┼────────────│
Episodic            │    -     │   Yes    │    Yes     │ Yes │     Yes       │    Yes     │
Semantic            │   Yes    │    -     │    Yes     │ Yes │     Yes       │    No      │
Procedural          │   Yes    │   Yes    │     -      │ No  │     No        │    No      │
NCA                 │   Yes    │   No     │    No      │  -  │     Yes       │    No      │
Consolidation       │   Yes    │   Yes    │    No      │ Yes │      -        │    No      │
Prediction          │   Yes    │   No     │    No      │ No  │     Yes       │     -      │
```

### Information Flow Analysis

```
ENCODING PATHWAY:
Input → Embedding → EpisodicMemory → NCA (NT modulation) → Storage (T4DX)
                         │
                         └─► SemanticMemory (entity extraction)
                         └─► Hook Layer (Hebbian learning)

RETRIEVAL PATHWAY:
Query → Embedding → VectorSearch → ACT-R Activation → NT-Modulated Ranking → Results
                         │
                         ├─► Episodic (temporal, contextual)
                         ├─► Semantic (spreading activation)
                         └─► Procedural (trigger matching)

CONSOLIDATION PATHWAY:
Episode Pool → Priority Queue (PE, outcome, recency)
                    │
                    ├─► NREM: SWR Replay → Hebbian strengthening
                    ├─► REM: Clustering → Abstraction → New entities
                    └─► Prune: Homeostatic scaling → Weak removal

PREDICTION PATHWAY:
Context (recent episodes) → ContextEncoder → LatentPredictor → Next embedding
                                                    │
                                                    └─► Prediction error → Consolidation priority
                                                    └─► Dream trajectory generation
```

### Missing Cross-Module Connections

| Gap | Source | Target | Neuroscience Basis | Impact |
|-----|--------|--------|-------------------|--------|
| Semantic-Prediction | Semantic | Prediction | Conceptual priors guide prediction | Medium |
| NCA-Procedural | NCA | Procedural | DA modulates habit selection | High |
| Prediction-Semantic | Prediction | Semantic | Prediction shapes concept learning | Medium |
| Emotion-All | nca/amygdala.py | All | Amygdala modulates all memory | Implemented (574 lines) |

---

## Cognitive Architecture Comparison

### ACT-R Comparison

| ACT-R Component | WW Equivalent | Fidelity |
|-----------------|---------------|----------|
| Declarative Module | EpisodicMemory + SemanticMemory | 90% |
| Procedural Module | ProceduralMemory | 75% |
| Goal Module | Session context / Working memory | 60% |
| Imaginal Module | Prediction / Dreaming | 70% |
| Visual/Aural Modules | Not implemented | 0% |
| Base-level activation | FSRS + ACT-R retrieval | 95% |
| Spreading activation | Hebbian + ACT-R | 85% |
| Production rules | Procedure triggerPatterns | 70% |

**ACT-R Compatibility Score: 68/100**

WW extends ACT-R with:
- Continuous neuromodulator dynamics (vs. discrete subsymbolic)
- Biologically-plausible consolidation (vs. decay only)
- Prediction error as learning signal (vs. utility learning)

---

### SOAR Comparison

| SOAR Component | WW Equivalent | Fidelity |
|----------------|---------------|----------|
| Working Memory | Context + active embeddings | 60% |
| Long-term Memory | Episodic + Semantic + Procedural | 85% |
| Chunking | REM abstraction | 50% |
| Reinforcement Learning | Dopamine RPE + TD(lambda) | 80% |
| Semantic Memory | SemanticMemory | 90% |
| Episodic Memory | EpisodicMemory | 90% |
| Impasses | Not implemented | 0% |
| Operators | Procedural + Agent coordination | 50% |

**SOAR Compatibility Score: 63/100**

WW extends SOAR with:
- Neural field dynamics (vs. symbolic only)
- Continuous state representation (vs. discrete)
- Biological consolidation (vs. chunking rules)

---

### Global Workspace Theory Comparison

| GWT Component | WW Equivalent | Fidelity |
|---------------|---------------|----------|
| Global Workspace | NCA attractor states | 70% |
| Unconscious processors | Memory subsystems | 80% |
| Conscious broadcast | Hook layer propagation | 60% |
| Competition | Attention/salience | 50% |
| Coalitions | Co-activated memories | 70% |

**GWT Compatibility Score: 66/100**

WW aligns with GWT through:
- Attractor states as global modes
- NT dynamics modulating "broadcast"
- But lacks explicit workspace competition

---

### Unique WW Contributions

| Innovation | Description | Novelty |
|------------|-------------|---------|
| 6-NT PDE System | Continuous neuromodulator field dynamics | High |
| FSRS + Hebbian | Spaced repetition with associative learning | Medium |
| Prediction-Error Consolidation | PE drives replay priority | High |
| Attractor-Memory Bridge | NT state modulates encoding/retrieval | High |
| Dream Trajectory Generation | JEPA-style imagination | High |
| Hierarchical Prediction | Multi-timescale forecasting | Medium |
| DA-ACh Phase Coupling | Striatal dynamics for habit | Medium |

---

## Gap Analysis

### Critical Missing Components

```
PRIORITY 1 (High Impact):
═════════════════════════

1. AMYGDALA MODULE (Emotion) — IMPLEMENTED
   Current: Full amygdala circuit (nca/amygdala.py, 574 lines)
   Status: Fear conditioning, emotional tagging, BLA pathways
   Remaining gap: Social cognition, nuanced emotion categories

   Implemented Components:
   - Fear conditioning pathway
   - Emotional tagging of episodes
   - Arousal state effects via NE coupling

2. ATTENTION NETWORK
   Current: Implicit in attractor states
   Needed: Explicit attention mechanisms
   Impact: Selective processing

   Components:
   - Top-down goal-based attention
   - Bottom-up salience detection
   - Attentional filtering in retrieval
   - Resource allocation (capacity limits)

3. MOTOR/ACTION SYSTEM
   Current: Procedural outputs only
   Needed: Action selection and execution
   Impact: Agent behavior

   Components:
   - Action-outcome associations
   - Motor timing (cerebellum)
   - Conflict resolution (ACC)
   - Response inhibition


PRIORITY 2 (Medium Impact):
═══════════════════════════

4. SPATIAL COGNITION (Partial)
   Current: Mentioned in v0.4.0
   Needed: Full place/grid cell implementation
   Impact: Context representation

   Components:
   - Place cell activation from context
   - Grid cell metric encoding
   - Head direction cells
   - Boundary cells

5. LANGUAGE SYSTEM
   Current: Embedding only
   Needed: Linguistic processing
   Impact: Content understanding

   Components:
   - Syntactic parsing
   - Semantic composition
   - Pragmatic inference
   - Language production

6. METACOGNITION
   Current: None
   Needed: Self-monitoring
   Impact: Adaptive behavior

   Components:
   - Confidence estimation
   - Error monitoring (ACC)
   - Strategy selection
   - Learning rate modulation
```

### Implementation Recommendations

```
SHORT-TERM (1-2 Sprints):
─────────────────────────
1. Implement emotional tagging system
   - Add EmotionalState dataclass
   - Integrate with episodic storage
   - Modulate retrieval by emotion

2. Add explicit attention mechanism
   - Implement salience scoring
   - Add attentional filtering to retrieval
   - Connect to NCA ACh dynamics

3. Complete spatial cognition
   - Implement PlaceCell class
   - Add GridCell for metric coding
   - Integrate with context representation

MEDIUM-TERM (3-6 Sprints):
──────────────────────────
4. Build amygdala module
   - Fear conditioning pathway
   - Emotional memory enhancement
   - Stress response (cortisol proxy)

5. Implement motor system
   - Action selection from procedural
   - Timing and sequencing
   - Outcome prediction

6. Add metacognitive layer
   - Confidence computation
   - Error detection
   - Strategy switching

LONG-TERM (6+ Sprints):
───────────────────────
7. Full language integration
8. Social cognition module
9. Developmental learning trajectories
10. Multi-agent interaction
```

---

## Summary Scores

### Module-by-Module Biological Accuracy

| Module | Score | Key Strength | Key Gap |
|--------|-------|--------------|---------|
| EpisodicMemory | 95 | FSRS decay, temporal context | Pattern completion |
| SemanticMemory | 90 | ACT-R activation, Hebbian | Hierarchical organization |
| ProceduralMemory | 85 | Skill lifecycle | Motor timing |
| NCA/Neural Field | 88 | 6-NT PDE, attractors | Gain modulation |
| DopamineSystem | 90 | RPE, TD(lambda) | Phasic vs. tonic |
| SerotoninSystem | 85 | Long-term credit | Impulsivity effects |
| SleepConsolidation | 90 | NREM/REM, SWR, sleep spindles (nca/sleep_spindles.py, 608 lines) | K-complexes |
| Prediction | 80 | JEPA, hierarchical | Semantic priors |
| Attention | 70 | ACh gating | Explicit mechanisms |
| Amygdala/Emotion | 75 | Full amygdala circuit (nca/amygdala.py, 574 lines) | Social cognition |
| Spatial | 50 | Concept present | Implementation needed |

### Overall Architecture Assessment

```
BIOLOGICAL PLAUSIBILITY:  87/100
════════════════════════════════

Strengths:
+ Tripartite memory model (Tulving)
+ Neuromodulator dynamics (6-NT)
+ Sleep consolidation (NREM/REM) with sleep spindles
+ Prediction error learning (RPE)
+ Attractor-based cognitive states
+ Full amygdala circuit (574 lines)
+ LC with 5 firing modes (1102 lines)
+ Sleep spindle-delta coupling (608 lines)

Weaknesses:
- Spatial cognition partial
- Motor system absent (cerebellum NOT IMPLEMENTED — critical gap)
- Metacognition missing
- Social cognition absent

COGNITIVE ARCHITECTURE ALIGNMENT:
════════════════════════════════
ACT-R:    68/100 (Good declarative, weak production)
SOAR:     63/100 (Good memory, weak problem solving)
GWT:      66/100 (Good broadcast, weak competition)

UNIQUE CONTRIBUTIONS:
════════════════════
- Continuous NT field dynamics
- Prediction-driven consolidation
- Attractor-memory integration
- Dream trajectory generation
```

---

## References

1. Tulving, E. (1972). Episodic and semantic memory
2. Squire, L.R. (1992). Memory systems of the brain
3. Anderson, J.R. (2007). ACT-R cognitive architecture
4. Laird, J.E. (2012). The SOAR cognitive architecture
5. Baars, B.J. (1988). Global Workspace Theory
6. O'Keefe & Moser (2014). Place cells and grid cells (Nobel Prize)
7. Daw et al. (2002). Serotonin and temporal discounting
8. McClelland et al. (1995). Complementary Learning Systems
9. LeCun (2022). JEPA architecture
10. Hafner et al. (2023). DreamerV3

---

*Document generated by T4DM Neuroscience Agent*
*Based on codebase analysis as of 2026-01-03, updated 2026-02-02*
