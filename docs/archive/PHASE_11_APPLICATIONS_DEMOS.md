# Phase 11: Applications & Demos Implementation Plan

**Created**: 2026-01-04
**Status**: PLANNING
**Target Duration**: 15-20 development days
**Prerequisites**: Phases 1-10 (core visualization infrastructure)

---

## Executive Summary

Phase 11 delivers four biologically-grounded interactive demonstrations that showcase World Weaver's neural memory architecture. These demos serve as:

1. **Educational tools** - Understanding brain-inspired memory mechanisms
2. **Debugging interfaces** - Real-time system observability
3. **Research platforms** - Validating biological plausibility
4. **Marketing assets** - Demonstrating WW capabilities

---

## Existing Infrastructure

### Visualization Modules (Ready to Expose)
```
/mnt/projects/ww/src/ww/visualization/
├── telemetry_hub.py              # Multi-scale integration hub
├── nt_state_dashboard.py         # 6-channel NT visualization
├── consolidation_replay.py       # SWR replay sequences
├── neuromodulator_state.py       # DA/NE/ACh/5-HT dynamics
├── glymphatic_visualizer.py      # Sleep clearance visualization
├── ff_visualizer.py              # Forward-Forward learning
├── capsule_visualizer.py         # Capsule network routing
├── plasticity_traces.py          # STDP/BCM dynamics
├── embedding_projections.py      # t-SNE/UMAP projections
├── pattern_separation.py         # DG orthogonalization
├── stability_monitor.py          # Lyapunov/bifurcation analysis
├── pac_telemetry.py              # Phase-amplitude coupling
├── swr_telemetry.py              # Sharp-wave ripple events
└── da_telemetry.py               # Dopamine temporal structure
```

### Existing API Infrastructure
```
/mnt/projects/ww/src/ww/api/
├── websocket.py                  # WebSocket with channels (events, memory, learning, health)
├── routes/visualization.py       # REST endpoints for biological mechanisms
└── routes/                       # Other REST endpoints
```

---

## Demo 1: Interactive Memory Explorer

### Purpose
Visualize the full lifecycle of memory encoding, storage, retrieval, and consolidation in real-time. Users can:
- Watch memories flow through hippocampal stages (DG -> CA3 -> CA1)
- See pattern separation orthogonalize similar inputs
- Observe pattern completion from partial cues
- Track memory promotion through tripartite system

### Biological Processes to Highlight

| Process | Brain Region | WW Component | Visualization |
|---------|--------------|--------------|---------------|
| Pattern Separation | Dentate Gyrus | `pattern_separation.py` | Input vs output similarity heatmap |
| Pattern Completion | CA3 | `HopfieldMemory` | Attractor basin animation |
| Sequence Binding | CA1 | `BufferManager` | Temporal order binding |
| Memory Consolidation | HPC -> Cortex | `consolidation_replay.py` | Replay sequence timeline |
| Indexing | Hippocampus | `cluster_index.py` | Hierarchical cluster tree |

### Data to Expose

```python
# /mnt/projects/ww/src/ww/visualization/memory_explorer.py

@dataclass
class MemoryExplorerState:
    """Real-time state for Memory Explorer demo."""

    # Current memory operation
    active_operation: Literal["idle", "encoding", "retrieval", "consolidation"]
    operation_progress: float  # 0.0 to 1.0

    # Hippocampal stages
    dg_sparsity: float  # Dentate Gyrus activation sparsity
    ca3_attractor_state: np.ndarray  # CA3 Hopfield state
    ca1_temporal_binding: list[str]  # Bound memory IDs

    # Pattern separation metrics
    input_similarity_matrix: np.ndarray  # [N, N] input similarities
    output_similarity_matrix: np.ndarray  # [N, N] after separation
    separation_gain: float  # ratio

    # Pattern completion metrics
    query_embedding: np.ndarray  # Partial cue
    completion_trajectory: list[np.ndarray]  # Attractor convergence
    final_retrieval: str  # Retrieved memory ID
    retrieval_confidence: float

    # Memory flow tracking
    recent_memories: list[dict]  # Last N memories with timestamps
    memory_locations: dict[str, Literal["buffer", "episodic", "semantic", "procedural"]]
    promotion_events: list[dict]  # Recent promotions

    # Embedding projections
    tsne_coordinates: np.ndarray  # [N, 2] for all memories
    umap_coordinates: np.ndarray  # [N, 2] for all memories
    cluster_assignments: np.ndarray  # [N] cluster IDs
```

### WebSocket Endpoints

```python
# New WebSocket channels for Memory Explorer

@router.websocket("/ws/explorer")
async def websocket_memory_explorer(websocket: WebSocket):
    """
    Real-time memory explorer updates.

    Events:
    - explorer.encoding_start: New memory encoding begins
    - explorer.encoding_stage: DG/CA3/CA1 stage update
    - explorer.encoding_complete: Encoding finished
    - explorer.retrieval_start: Query submitted
    - explorer.attractor_step: Hopfield iteration
    - explorer.retrieval_complete: Memory retrieved
    - explorer.promotion: Memory promoted to new tier
    - explorer.consolidation_replay: SWR replay event
    """
    await manager.connect(websocket, "explorer")
    # ... event handling ...
```

### REST API Endpoints

```python
# /mnt/projects/ww/src/ww/api/routes/explorer.py

@router.get("/explorer/state")
async def get_explorer_state() -> MemoryExplorerState:
    """Get current memory explorer state."""
    ...

@router.get("/explorer/memories")
async def get_memory_landscape(
    limit: int = 100,
    projection: Literal["tsne", "umap", "pca"] = "umap"
) -> dict:
    """Get projected memory landscape for visualization."""
    ...

@router.get("/explorer/memory/{memory_id}/journey")
async def get_memory_journey(memory_id: str) -> dict:
    """Get full lifecycle of a specific memory."""
    ...

@router.post("/explorer/demo/encode")
async def demo_encode(content: str) -> dict:
    """Demo: Encode a memory with live visualization."""
    ...

@router.post("/explorer/demo/retrieve")
async def demo_retrieve(query: str, partial_cue_ratio: float = 0.3) -> dict:
    """Demo: Retrieve with partial cue pattern completion."""
    ...
```

### Frontend Technology

**Recommended Stack:**
- **React** + **TypeScript** for UI framework
- **Three.js** (via react-three-fiber) for 3D memory landscape
- **D3.js** for 2D projections (t-SNE/UMAP)
- **Framer Motion** for animations
- **Recharts** for time series (separation metrics)

**Key Components:**
```
src/demos/memory-explorer/
├── MemoryLandscape3D.tsx      # 3D scatter of memories
├── HippocampalPipeline.tsx    # DG -> CA3 -> CA1 flow
├── PatternSeparation.tsx      # Input/output heatmaps
├── PatternCompletion.tsx      # Attractor basin animation
├── MemoryTimeline.tsx         # Recent operations timeline
└── MemoryDetails.tsx          # Selected memory inspector
```

### Biological Accuracy Requirements

| Metric | Target | Validation |
|--------|--------|------------|
| DG sparsity | 2-5% active | `sparsity = active / total < 0.05` |
| CA3 completion | >90% from 30% cue | `cosine(complete, target) > 0.9` |
| Separation gain | >3x orthogonalization | `input_sim / output_sim > 3` |
| Replay rate | 100-200 Hz compressed | SWR events during consolidation |

---

## Demo 2: Dream Viewer

### Purpose
Visualize what happens during simulated "sleep" consolidation:
- Watch REM and NREM phases alternate
- See memory replays during SWR events
- Observe creative recombination in REM
- Track prediction error reduction over cycles

### Biological Processes to Highlight

| Process | Sleep Phase | WW Component | Visualization |
|---------|-------------|--------------|---------------|
| SWR Replay | NREM | `swr_telemetry.py` | Compressed timeline |
| Memory Recombination | REM | `dreaming/` | Novel combinations |
| Waste Clearance | NREM Deep | `glymphatic_visualizer.py` | Clearance rate gauge |
| Schema Extraction | REM | `consolidation/` | Abstraction emergence |
| Pruning | All | `glymphatic_visualizer.py` | Memory removal events |

### Data to Expose

```python
# /mnt/projects/ww/src/ww/visualization/dream_viewer.py

@dataclass
class DreamViewerState:
    """Real-time state for Dream Viewer demo."""

    # Sleep cycle state
    current_phase: Literal["wake", "nrem_light", "nrem_deep", "rem"]
    phase_progress: float  # 0.0 to 1.0 within phase
    cycle_number: int  # Current sleep cycle
    total_sleep_time: float  # Minutes

    # Adenosine sleep pressure
    adenosine_level: float  # 0.0 to 1.0
    sleep_pressure_history: list[float]  # Over time

    # SWR replay events
    current_replay_sequence: list[str]  # Memory IDs being replayed
    replay_speed_compression: float  # e.g., 20x faster than encoding
    replay_priority_scores: list[float]
    total_replays_this_cycle: int

    # REM recombination
    recombination_pairs: list[tuple[str, str]]  # Memories being combined
    novel_associations_created: int
    semantic_distance_of_combinations: list[float]

    # Prediction error
    prediction_errors: dict[str, float]  # memory_id -> PE
    pe_reduction_over_cycles: list[float]  # Average PE per cycle

    # Glymphatic clearance
    waste_level: float  # Metabolic waste
    clearance_rate: float  # Current clearance
    aqp4_polarization: float  # Channel activity

    # Memory pruning
    pruning_candidates: list[dict]  # Memories at risk
    pruned_this_session: list[dict]  # Actually pruned
    pruning_reasons: dict[str, int]  # Reason -> count
```

### WebSocket Events

```python
# Dream Viewer WebSocket channel

class DreamEventType(str, Enum):
    PHASE_TRANSITION = "dream.phase_transition"
    REPLAY_START = "dream.replay_start"
    REPLAY_MEMORY = "dream.replay_memory"
    REPLAY_END = "dream.replay_end"
    RECOMBINATION = "dream.recombination"
    PRUNING_CANDIDATE = "dream.pruning_candidate"
    MEMORY_PRUNED = "dream.memory_pruned"
    PE_UPDATE = "dream.prediction_error_update"
    CLEARANCE_UPDATE = "dream.clearance_update"
    CYCLE_COMPLETE = "dream.cycle_complete"
```

### REST API Endpoints

```python
@router.get("/dream/state")
async def get_dream_state() -> DreamViewerState:
    """Get current dream/consolidation state."""

@router.get("/dream/cycles")
async def get_sleep_cycles(limit: int = 10) -> list[dict]:
    """Get history of sleep cycles with metrics."""

@router.get("/dream/replays")
async def get_replay_history(cycle: int | None = None) -> list[dict]:
    """Get replay events, optionally filtered by cycle."""

@router.post("/dream/start")
async def start_consolidation(
    duration_minutes: float = 90.0,  # One full cycle
    accelerated: bool = True  # Speed up for demo
) -> dict:
    """Start simulated sleep consolidation."""

@router.post("/dream/stop")
async def stop_consolidation() -> dict:
    """Stop consolidation and return to wake state."""
```

### Frontend Components

```
src/demos/dream-viewer/
├── SleepHypnogram.tsx          # Sleep stage timeline (classic hypnogram)
├── ReplayVisualizer.tsx        # Animated memory replay sequences
├── RecombinationGraph.tsx      # Network of combined memories
├── PEReductionChart.tsx        # Prediction error over cycles
├── GlymphaticGauge.tsx         # Waste/clearance dual gauge
├── PruningQueue.tsx            # Memories at risk of pruning
├── AdenosineMeter.tsx          # Sleep pressure indicator
└── CycleStatistics.tsx         # Per-cycle summary stats
```

### Biological Accuracy Requirements

| Metric | Target | Reference |
|--------|--------|-----------|
| NREM/REM ratio | 80%/20% (early) to 50%/50% (late) | Sleep architecture |
| SWR rate | 1-3 Hz during NREM | Buzsaki (2015) |
| Replay compression | 10-20x realtime | Diekelmann & Born (2010) |
| Clearance increase | +60% during sleep | Xie et al. (2013) |
| PE reduction | Monotonic decrease | Hobson (2009) |

---

## Demo 3: NT State Dashboard

### Purpose
Live monitoring of all six neuromodulator channels plus their downstream effects:
- Real-time concentration dynamics
- Homeostatic setpoint deviations
- Receptor saturation (Michaelis-Menten)
- Cross-NT correlations
- Cognitive state transitions

### Biological Processes to Highlight

| NT | Function | Downstream Effect | Visualization |
|----|----------|-------------------|---------------|
| DA | Reward/Learning | LR modulation, gate threshold | RPE spikes, value estimates |
| 5-HT | Patience/Mood | Long-term credit assignment | Mood gauge, patience timer |
| ACh | Encoding/Retrieval | Mode switching | E/R balance slider |
| NE | Arousal/Novelty | Gain modulation, exploration | Arousal heatmap |
| GABA | Inhibition | Sparsity, winner-take-all | Inhibition strength bar |
| Glu | Excitation | Signal amplification | E/I balance gauge |

### Data to Expose

```python
# Already implemented in nt_state_dashboard.py, extend with:

@dataclass
class NTDashboardLiveState:
    """Extended NT state for live dashboard."""

    # From existing NTSnapshot
    nt_state: np.ndarray  # [6] concentrations
    setpoint_deviation: np.ndarray  # [6] from homeostatic targets
    receptor_saturation: np.ndarray  # [6] Michaelis-Menten
    ei_balance: float  # GABA/Glu ratio
    arousal_index: float  # DA + NE + Glu weighted

    # Derived cognitive state
    cognitive_mode: Literal["explore", "exploit", "encode", "retrieve", "consolidate"]
    mode_confidence: float  # 0.0 to 1.0
    mode_duration: float  # Seconds in current mode

    # Firing rate sources (if available)
    vta_firing_rate: float | None  # DA source
    raphe_firing_rate: float | None  # 5-HT source
    lc_firing_rate: float | None  # NE source (locus coeruleus)

    # Opponent processes
    da_5ht_ratio: float  # Reward-seeking vs patience
    glu_gaba_ratio: float  # E/I balance
    ne_ach_balance: float  # Arousal vs attention mode

    # Cross-NT dynamics
    correlation_matrix: np.ndarray  # [6, 6] recent correlations
    autocorrelation: np.ndarray  # [6] temporal persistence

    # Phase relationships (for PAC)
    theta_phase: float  # Current theta oscillator phase
    gamma_amplitude: float  # Current gamma power
    modulation_index: float  # PAC strength

    # Alerts
    active_alerts: list[str]
    homeostatic_violations: list[str]
```

### WebSocket Events

```python
class NTEventType(str, Enum):
    NT_UPDATE = "nt.update"  # Regular NT state update
    MODE_TRANSITION = "nt.mode_transition"  # Cognitive mode change
    ALERT_TRIGGERED = "nt.alert"  # Homeostatic violation
    ALERT_CLEARED = "nt.alert_cleared"
    FIRING_RATE_UPDATE = "nt.firing_rate"  # Neural source activity
    PAC_UPDATE = "nt.pac_update"  # Phase-amplitude coupling
```

### REST API Endpoints

```python
@router.get("/nt/state")
async def get_nt_state() -> NTDashboardLiveState:
    """Get current NT dashboard state."""

@router.get("/nt/traces")
async def get_nt_traces(
    window_seconds: float = 60.0,
    resolution_ms: int = 100
) -> dict:
    """Get NT concentration traces for plotting."""

@router.get("/nt/correlations")
async def get_nt_correlations(window: int = 100) -> dict:
    """Get cross-NT correlation matrix."""

@router.get("/nt/saturation-curves")
async def get_saturation_curves() -> dict:
    """Get Michaelis-Menten receptor saturation curves."""

@router.post("/nt/inject")
async def inject_nt(
    nt: Literal["DA", "5-HT", "ACh", "NE", "GABA", "Glu"],
    delta: float,  # Amount to add/subtract
    decay_seconds: float = 10.0  # Time to return to baseline
) -> dict:
    """Demo: Inject NT perturbation and watch system response."""
```

### Frontend Components

```
src/demos/nt-dashboard/
├── NTChannelBars.tsx           # 6-channel stacked bar chart
├── DeviationHeatmap.tsx        # Setpoint deviation over time
├── SaturationCurves.tsx        # Michaelis-Menten curves
├── CorrelationMatrix.tsx       # Cross-NT heatmap
├── CognitiveStateGauge.tsx     # Current mode with transitions
├── OpponentProcesses.tsx       # DA/5-HT, Glu/GABA balances
├── PACVisualization.tsx        # Theta-gamma coupling
├── AlertPanel.tsx              # Active homeostatic alerts
└── FiringRateMonitor.tsx       # VTA/Raphe/LC activity
```

### Biological Accuracy Requirements

| Metric | Target | Validation |
|--------|--------|------------|
| NT range | [0, 1] normalized | Clip to biological bounds |
| E/I balance | 0.8 - 1.2 ratio | Flag if outside range |
| PAC MI | 0.3 - 0.7 | Healthy theta-gamma coupling |
| Receptor saturation | Km-based curves | Michaelis-Menten kinetics |
| Autocorrelation | τ-specific persistence | Match layer timescales |

---

## Demo 4: Learning Trace Replay

### Purpose
Replay and visualize how memories are formed, showing:
- Eligibility trace accumulation
- Three-factor learning rule in action
- STDP spike timing windows
- BCM learning curves
- Weight trajectory evolution

### Biological Processes to Highlight

| Process | Mechanism | WW Component | Visualization |
|---------|-----------|--------------|---------------|
| Eligibility Traces | Activity → trace → weight | `eligibility_trace.py` | Decaying trace animation |
| Three-Factor Learning | Pre × Post × Neuromod | `three_factor.py` | Factor multiplication diagram |
| STDP | Spike timing | `stdp.py` | Timing window plot |
| BCM | Threshold plasticity | `plasticity_traces.py` | BCM curve with threshold |
| Weight Updates | LTP/LTD | `plasticity_traces.py` | Weight trajectory over time |

### Data to Expose

```python
# /mnt/projects/ww/src/ww/visualization/learning_replay.py

@dataclass
class LearningReplayState:
    """State for learning trace replay demo."""

    # Current learning event
    is_replaying: bool
    replay_speed: float  # 1.0 = realtime, 0.1 = 10x slow
    current_timestamp: float  # Replay position

    # Eligibility trace state
    active_traces: dict[str, TraceState]  # synapse_id -> trace
    trace_decay_constants: dict[str, float]  # Fast/medium/slow τ

    # Current learning event details
    current_synapse: str | None
    pre_activity: float
    post_activity: float
    neuromod_signal: float  # DA or 5-HT
    eligibility_value: float
    weight_delta: float

    # Three-factor breakdown
    factor_pre: float
    factor_post: float
    factor_neuromod: float
    factor_product: float

    # STDP window
    pre_spike_time: float | None
    post_spike_time: float | None
    timing_delta: float  # post - pre
    stdp_curve_position: float  # Where on the STDP curve

    # BCM state
    theta_m: float  # Modification threshold
    activation_history: list[float]
    weight_change_history: list[float]

    # Weight trajectory
    synapse_weights: dict[str, list[tuple[float, float]]]  # synapse -> [(time, weight)]
    ltp_count: int
    ltd_count: int
    homeostatic_count: int

@dataclass
class TraceState:
    """Single eligibility trace state."""
    trace_id: str
    source_id: str
    target_id: str
    current_value: float
    peak_value: float
    decay_constant: float
    creation_time: float
    last_activity_time: float
```

### WebSocket Events

```python
class LearningEventType(str, Enum):
    TRACE_CREATED = "learning.trace_created"
    TRACE_UPDATED = "learning.trace_updated"
    TRACE_DECAYED = "learning.trace_decayed"
    TRACE_EXPIRED = "learning.trace_expired"
    WEIGHT_UPDATE = "learning.weight_update"
    STDP_EVENT = "learning.stdp_event"
    BCM_THRESHOLD_UPDATE = "learning.bcm_threshold"
    HOMEOSTATIC_SCALING = "learning.homeostatic"
    THREE_FACTOR_UPDATE = "learning.three_factor"
```

### REST API Endpoints

```python
@router.get("/learning/state")
async def get_learning_state() -> LearningReplayState:
    """Get current learning replay state."""

@router.get("/learning/traces")
async def get_active_traces() -> list[TraceState]:
    """Get all active eligibility traces."""

@router.get("/learning/weights/{synapse_id}")
async def get_weight_history(
    synapse_id: str,
    window_seconds: float = 300.0
) -> list[tuple[float, float]]:
    """Get weight trajectory for a synapse."""

@router.get("/learning/stdp-window")
async def get_stdp_window() -> dict:
    """Get STDP timing window parameters and curve."""

@router.get("/learning/bcm-curve")
async def get_bcm_curve() -> dict:
    """Get BCM learning curve and current threshold."""

@router.post("/learning/replay/start")
async def start_learning_replay(
    start_time: float | None = None,
    speed: float = 0.5  # Half realtime for visibility
) -> dict:
    """Start replaying recorded learning events."""

@router.post("/learning/replay/pause")
async def pause_learning_replay() -> dict:
    """Pause learning replay."""

@router.post("/learning/replay/seek")
async def seek_learning_replay(timestamp: float) -> dict:
    """Seek to specific timestamp in replay."""

@router.post("/learning/demo/trigger")
async def trigger_learning_event(
    pre_activity: float = 0.8,
    post_activity: float = 0.6,
    neuromod: float = 0.7,
    delay_ms: float = 10.0  # Pre-post delay
) -> dict:
    """Demo: Trigger a synthetic learning event."""
```

### Frontend Components

```
src/demos/learning-replay/
├── EligibilityTraceViz.tsx     # Animated decaying traces
├── ThreeFactorDiagram.tsx      # Pre × Post × Neuromod visualization
├── STDPWindow.tsx              # Timing window with current position
├── BCMCurve.tsx                # BCM curve with threshold marker
├── WeightTrajectory.tsx        # Weight evolution over time
├── SynapseNetwork.tsx          # Network view with weight-coded edges
├── ReplayControls.tsx          # Play/pause/seek/speed controls
├── LTPLTDHistogram.tsx         # Distribution of potentiation/depression
└── EventTimeline.tsx           # Scrollable learning event log
```

### Biological Accuracy Requirements

| Metric | Target | Reference |
|--------|--------|-----------|
| Eligibility decay | τ = 100ms - 10s | Sutton & Barto Ch. 12 |
| STDP window | ±50ms asymmetric | Bi & Poo (1998) |
| BCM threshold | Activity-dependent | BCM theory |
| LTP/LTD ratio | ~2:1 in healthy learning | Malenka & Bear (2004) |
| Three-factor rule | δ = pre × post × neuromod | Schultz (1998) |

---

## Unified WebSocket Architecture

### Extended Connection Manager

```python
# /mnt/projects/ww/src/ww/api/websocket_v2.py

class DemoConnectionManager(ConnectionManager):
    """Extended manager for demo WebSocket channels."""

    def __init__(self):
        super().__init__()
        # Add demo-specific channels
        self._connections.update({
            "explorer": set(),     # Memory Explorer
            "dream": set(),        # Dream Viewer
            "nt": set(),           # NT Dashboard
            "learning": set(),     # Learning Replay
            "unified": set(),      # All demos unified
        })

        # Channel-specific rate limiting
        self._rate_limits = {
            "explorer": 10,   # 10 Hz max
            "dream": 2,       # 2 Hz (sleep is slow)
            "nt": 20,         # 20 Hz for smooth NT updates
            "learning": 100,  # 100 Hz for spike-level
            "unified": 30,    # 30 Hz aggregate
        }
```

### Event Aggregation

```python
class DemoEventAggregator:
    """Aggregates events from all demo visualizers."""

    def __init__(self, telemetry_hub: TelemetryHub):
        self.hub = telemetry_hub
        self._subscribers: dict[str, list[Callable]] = {}

    async def start_streaming(self, interval_ms: int = 50):
        """Stream aggregated state at fixed interval."""
        while True:
            state = {
                "timestamp": time.time(),
                "explorer": await self._get_explorer_state(),
                "dream": await self._get_dream_state(),
                "nt": await self._get_nt_state(),
                "learning": await self._get_learning_state(),
                "cross_scale": self.hub.get_summary_statistics(),
            }
            await self._broadcast("unified", state)
            await asyncio.sleep(interval_ms / 1000)
```

---

## File Structure for New Code

```
/mnt/projects/ww/
├── src/ww/
│   ├── visualization/
│   │   ├── memory_explorer.py       # NEW: Explorer state aggregation
│   │   ├── dream_viewer.py          # NEW: Dream state aggregation
│   │   └── learning_replay.py       # NEW: Learning replay state
│   ├── api/
│   │   ├── websocket_v2.py          # NEW: Extended WebSocket manager
│   │   └── routes/
│   │       ├── explorer.py          # NEW: Memory Explorer endpoints
│   │       ├── dream.py             # NEW: Dream Viewer endpoints
│   │       ├── nt_dashboard.py      # NEW: Extended NT endpoints
│   │       └── learning.py          # NEW: Learning Replay endpoints
│   └── demos/
│       ├── __init__.py              # NEW: Demo module
│       ├── orchestrator.py          # NEW: Multi-demo coordination
│       └── synthetic_data.py        # NEW: Synthetic data generators
├── frontend/                        # NEW: Frontend application
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── demos/
│   │   │   ├── memory-explorer/
│   │   │   ├── dream-viewer/
│   │   │   ├── nt-dashboard/
│   │   │   └── learning-replay/
│   │   ├── components/
│   │   │   ├── WebSocketProvider.tsx
│   │   │   ├── ThreeScene.tsx
│   │   │   └── Charts/
│   │   └── hooks/
│   │       ├── useWebSocket.ts
│   │       └── useVisualization.ts
│   └── public/
└── docs/
    └── PHASE_11_APPLICATIONS_DEMOS.md  # This file
```

---

## Implementation Tasks

### Sprint 11.1: Foundation (5 days)

| Task | Description | Effort | Risk |
|------|-------------|--------|------|
| TASK-1101 | Create `memory_explorer.py` state aggregator | Medium | Low |
| TASK-1102 | Create `dream_viewer.py` state aggregator | Medium | Low |
| TASK-1103 | Create `learning_replay.py` state aggregator | Medium | Low |
| TASK-1104 | Extend `websocket_v2.py` with demo channels | Medium | Low |
| TASK-1105 | Create REST routes for all four demos | Large | Medium |

### Sprint 11.2: Backend Integration (5 days)

| Task | Description | Effort | Risk |
|------|-------------|--------|------|
| TASK-1106 | Integrate Explorer with EpisodicMemory | Large | Medium |
| TASK-1107 | Integrate Dream with GlymphaticSystem | Large | Medium |
| TASK-1108 | Integrate NT Dashboard with NTStateDashboard | Medium | Low |
| TASK-1109 | Integrate Learning with PlasticityTracer | Medium | Low |
| TASK-1110 | Create synthetic data generators for demos | Medium | Low |

### Sprint 11.3: Frontend (5-10 days)

| Task | Description | Effort | Risk |
|------|-------------|--------|------|
| TASK-1111 | Set up React + TypeScript + Vite project | Small | Low |
| TASK-1112 | Implement WebSocket provider and hooks | Medium | Low |
| TASK-1113 | Build Memory Explorer components | Large | Medium |
| TASK-1114 | Build Dream Viewer components | Large | Medium |
| TASK-1115 | Build NT Dashboard components | Medium | Low |
| TASK-1116 | Build Learning Replay components | Large | Medium |
| TASK-1117 | Create unified demo landing page | Small | Low |

---

## Testing Strategy

### Unit Tests
- State aggregator correctness
- WebSocket event serialization
- REST endpoint responses
- Synthetic data generation

### Integration Tests
- End-to-end memory encoding → visualization
- Sleep cycle → dream viewer updates
- NT injection → dashboard response
- Learning event → trace visualization

### Visual Regression Tests
- Component snapshot tests
- Storybook for isolated component testing

### Performance Tests
- WebSocket latency < 50ms
- Frontend render < 16ms (60 FPS)
- Memory leak detection over 1hr session

---

## Biological Validation Checklist

### Memory Explorer
- [ ] DG sparsity within 2-5%
- [ ] CA3 pattern completion from 30% cue
- [ ] Separation ratio > 3x
- [ ] SWR compression 10-20x

### Dream Viewer
- [ ] NREM/REM ratio follows sleep architecture
- [ ] Adenosine rise during wake, fall during sleep
- [ ] Glymphatic clearance peaks in deep NREM
- [ ] PE reduction monotonic across cycles

### NT Dashboard
- [ ] All NTs within [0, 1] biological range
- [ ] E/I balance typically 0.8-1.2
- [ ] PAC MI in healthy range 0.3-0.7
- [ ] Mode transitions match expected triggers

### Learning Replay
- [ ] Eligibility traces decay with correct τ
- [ ] STDP window asymmetric ±50ms
- [ ] BCM threshold adapts to activity
- [ ] LTP/LTD ratio approximately 2:1

---

## Deployment

### Docker Compose Extension
```yaml
# docker-compose.demos.yml
services:
  ww-demos-frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_WS_URL=ws://localhost:8080
      - VITE_API_URL=http://localhost:8080
    depends_on:
      - ww-api
```

### URL Structure
```
http://localhost:3000/                    # Demo landing page
http://localhost:3000/explorer            # Memory Explorer
http://localhost:3000/dream               # Dream Viewer
http://localhost:3000/nt                  # NT Dashboard
http://localhost:3000/learning            # Learning Replay
```

---

## Success Metrics

| Metric | Target |
|--------|--------|
| WebSocket latency (p95) | < 100ms |
| Frontend FPS | > 30 FPS |
| State update rate | 10-100 Hz (demo-specific) |
| Biological accuracy | Pass all validation checks |
| User engagement | 5+ minute average session |
| Educational value | >80% comprehension in user study |

---

## Future Extensions

1. **VR/AR Support** - Immersive memory landscape exploration
2. **Multi-user Collaboration** - Shared demo sessions
3. **Recording/Playback** - Save and replay demo sessions
4. **Educational Curriculum** - Structured learning paths
5. **Research Export** - Export data for analysis
6. **Custom Scenarios** - User-defined demo configurations

---

**END OF PHASE 11 IMPLEMENTATION PLAN**

*This document should be updated as implementation progresses.*
