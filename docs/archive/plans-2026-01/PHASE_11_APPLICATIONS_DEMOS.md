# Phase 11: Applications & Demos - Implementation Plan

**Generated**: 2026-01-04 | **Agent**: ww-compbio | **Status**: PLANNING COMPLETE

---

## Executive Summary

Phase 11 creates four biologically-grounded interactive demonstrations:

1. **Interactive Memory Explorer** - Full memory lifecycle visualization
2. **Dream Viewer** - Sleep consolidation phases
3. **NT State Dashboard** - Live neuromodulator dynamics
4. **Learning Trace Replay** - Memory formation visualization

---

## Demo 1: Interactive Memory Explorer

### Biological Processes Visualized

| Process | Brain Region | Visualization |
|---------|--------------|---------------|
| Pattern Separation | Dentate Gyrus | Input/output similarity heatmap |
| Pattern Completion | CA3 | Hopfield attractor basin animation |
| Sequence Binding | CA1 | Temporal order binding |
| Memory Consolidation | HPC → Cortex | SWR replay timeline |

### Data Exposed

```python
class MemoryExplorerState:
    # Hippocampal stages
    dg_sparsity: float          # 2-5% active neurons
    ca3_attractor_state: str    # "encoding", "retrieval", "settling"
    ca1_binding_strength: float

    # Pattern separation metrics
    input_similarity: float
    output_similarity: float
    separation_gain: float      # > 3x orthogonalization

    # Memory flow
    working_memory_items: list
    episodic_recent: list
    semantic_links: list

    # Embeddings for visualization
    embedding_2d: np.ndarray    # t-SNE/UMAP projection
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/explorer/state` | GET | Current explorer state |
| `/explorer/demo/encode` | POST | Demo encoding with visualization |
| `/explorer/demo/retrieve` | POST | Pattern completion demo |
| `/ws/explorer` | WS | Real-time events |

---

## Demo 2: Dream Viewer

### Biological Processes Visualized

| Process | Sleep Stage | Visualization |
|---------|-------------|---------------|
| SWR Replay | NREM | Compressed memory sequence playback |
| Memory Recombination | REM | Novel association network |
| Glymphatic Clearance | Deep NREM | Waste removal flow animation |
| Memory Pruning | NREM→REM | Stability-based removal |

### Data Exposed

```python
class DreamViewerState:
    # Sleep phase
    current_phase: str          # "wake", "nrem_light", "nrem_deep", "rem"
    phase_progress: float       # 0.0 - 1.0
    cycle_number: int

    # Adenosine dynamics
    adenosine_level: float      # Sleep pressure
    clearance_rate: float

    # Replay events
    active_replays: list[ReplaySequence]
    replay_priorities: dict[str, float]

    # Prediction error
    prediction_errors: list[float]  # Should decrease over cycles

    # Glymphatic
    glymphatic_flow_rate: float
    aqp4_activity: float
    waste_clearance: float
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/dream/state` | GET | Current consolidation state |
| `/dream/start` | POST | Start simulated sleep |
| `/dream/replays` | GET | Replay event history |
| `/ws/dream` | WS | Phase transitions, replays, pruning |

---

## Demo 3: NT State Dashboard

### Biological Processes Visualized

| Neuromodulator | Function | Visualization |
|----------------|----------|---------------|
| DA (Dopamine) | RPE, learning rate | Spike raster, RPE plot |
| 5-HT (Serotonin) | Patience, long-term credit | Mood indicator, gamma trace |
| ACh (Acetylcholine) | Encoding/retrieval mode | Mode switch indicator |
| NE (Norepinephrine) | Arousal, exploration | Yerkes-Dodson curve |
| GABA | Inhibition, sparsity | E/I balance gauge |
| Glutamate | Excitation | E/I balance gauge |

### Data Exposed

```python
class NTDashboardState:
    # NT concentrations
    nt_levels: dict[str, float]
    nt_setpoints: dict[str, float]
    nt_deviations: dict[str, float]

    # Receptor saturation (Michaelis-Menten)
    receptor_saturation: dict[str, float]

    # Cross-NT correlation
    correlation_matrix: np.ndarray  # 6x6

    # Cognitive mode
    current_mode: str   # "explore", "exploit", "encode", "retrieve"
    mode_confidence: float

    # Oscillatory coupling
    theta_gamma_pac: float
    theta_phase: float
    gamma_amplitude: float
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/nt/state` | GET | Current NT state |
| `/nt/traces` | GET | Time series data |
| `/nt/inject` | POST | Demo NT perturbation |
| `/ws/nt` | WS | Real-time updates |

---

## Demo 4: Learning Trace Replay

### Biological Processes Visualized

| Process | Mechanism | Visualization |
|---------|-----------|---------------|
| Eligibility Traces | Decaying activity | Trace decay curves |
| Three-Factor Learning | Pre × Post × Neuromod | Factor breakdown chart |
| STDP | Timing-dependent plasticity | Timing window plot |
| BCM | Activity-dependent threshold | BCM curve with threshold |

### Data Exposed

```python
class LearningTraceState:
    # Active eligibility traces
    traces: list[EligibilityTrace]

    # Three-factor breakdown
    three_factor: dict[str, float]  # pre, post, neuromod, product

    # STDP window
    stdp_window: STDPWindow
    current_delta_t: float

    # BCM curve
    bcm_threshold: float
    activity_history: list[float]

    # Weight trajectories
    weight_updates: list[WeightUpdate]  # LTP/LTD events
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/learning/state` | GET | Current learning state |
| `/learning/traces` | GET | Active traces |
| `/learning/stdp-window` | GET | STDP parameters |
| `/learning/replay/start` | POST | Replay recorded events |
| `/learning/demo/trigger` | POST | Trigger synthetic event |
| `/ws/learning` | WS | Trace creation, weight updates |

---

## Technical Architecture

### Backend Files to Create

| File | Purpose |
|------|---------|
| `api/routes/explorer.py` | Memory explorer endpoints |
| `api/routes/dream_viewer.py` | Dream viewer endpoints |
| `api/routes/nt_dashboard.py` | NT dashboard endpoints |
| `api/routes/learning_trace.py` | Learning trace endpoints |
| `api/demo_states/` | State aggregators for each demo |

### WebSocket Extension

```python
# Extend ConnectionManager with demo channels
class DemoConnectionManager(ConnectionManager):
    async def broadcast_explorer(self, data: dict)
    async def broadcast_dream(self, data: dict)
    async def broadcast_nt(self, data: dict)
    async def broadcast_learning(self, data: dict)
```

### Frontend Stack (Recommended)

| Library | Purpose |
|---------|---------|
| React + TypeScript + Vite | Framework |
| Three.js (react-three-fiber) | 3D visualizations |
| D3.js | 2D projections, networks |
| Recharts | Time series |
| Framer Motion | Animations |

---

## Biological Accuracy Validation

| Metric | Target | Source |
|--------|--------|--------|
| DG sparsity | 2-5% | Jung & McNaughton 1993 |
| CA3 completion | >90% from 30% cue | Rolls 2013 |
| Pattern separation gain | >3x | Leutgeb 2007 |
| NREM/REM ratio | ~75:25 | Stickgold 2005 |
| E/I balance | 0.8-1.2 | Isaacson & Scanziani 2011 |
| STDP window | +/-50ms asymmetric | Bi & Poo 1998 |

---

## Implementation Timeline

| Sprint | Duration | Focus |
|--------|----------|-------|
| 11.1 | 5 days | Backend state aggregators, WebSocket |
| 11.2 | 5 days | Integration with WW components |
| 11.3 | 5-10 days | Frontend React application |

**Total**: 15-20 days
