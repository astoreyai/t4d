# Temporal Module

**4 files | ~1,300 lines | Centrality: 2**

The temporal module provides unified temporal dynamics coordination for T4DM, managing session lifecycle, phase transitions, neuromodulator state, and consolidation-plasticity integration.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       TEMPORAL DYNAMICS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐│
│  │                    SESSION MANAGEMENT                                ││
│  │  SessionManager (singleton) → SessionContext (per session)          ││
│  │  start_session() → record_retrieval() → set_outcome() → end_session││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    PHASE TRANSITIONS                                 ││
│  │  ACTIVE ──(5min idle)──▶ IDLE ──(30min)──▶ CONSOLIDATING ──▶ SLEEPING││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    NEUROMODULATOR STATE                              ││
│  │  ACh: encoding/retrieval | DA: reward | NE: arousal | 5-HT: credit  ││
│  └───────────────────────────────┬─────────────────────────────────────┘│
│                                  │                                      │
│  ┌───────────────────────────────▼─────────────────────────────────────┐│
│  │                    PLASTICITY COORDINATION                           ││
│  │  Reconsolidation + Homeostatic + Salience → Modulated Updates       ││
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `session.py` | ~310 | Session lifecycle management |
| `dynamics.py` | ~545 | Unified temporal coordinator |
| `integration.py` | ~400 | Consolidation-plasticity bridge |
| `__init__.py` | ~48 | Public API exports |

## Session Management

### SessionContext

Single session state container:

```python
from t4dm.temporal import SessionContext

context = SessionContext(
    id="session-uuid",
    start_time=datetime.now(),
    initial_mode=CognitiveMode.ENCODING,
    goal="Debug authentication issue",
    project="auth-service",
    tags=["debugging", "auth"]
)

# Track operations
context.record_retrieval(memory_id)
context.record_encoding(memory_id)

# Record outcome
context.set_outcome(score=0.8, success=True)

# Close session
context.close()

# Properties
context.duration_seconds  # Elapsed time
context.is_active         # Not yet closed
```

### SessionManager

Lifecycle and history management:

```python
from t4dm.temporal import SessionManager, get_session_manager

manager = get_session_manager()  # Singleton

# Context manager style (recommended)
with manager.session(
    goal="fix bug",
    mode=CognitiveMode.RETRIEVAL,
    project="backend"
) as ctx:
    ctx.record_retrieval(memory_id)
    ctx.set_outcome(0.9, success=True)

# Manual style
ctx = manager.start_session(goal="encode facts")
# ... operations ...
manager.end_session(ctx.id, outcome_score=0.9)

# Switch mode mid-session
manager.switch_mode(ctx.id, CognitiveMode.ENCODING)

# Query sessions
active = manager.get_active_sessions()
recent = manager.get_recent_sessions(limit=10)
stats = manager.get_stats()
# {'active': 2, 'completed': 45, 'avg_outcome': 0.78, 'success_rate': 0.82}
```

## Temporal Dynamics

### TemporalPhase

System-wide operational phases:

```python
from t4dm.temporal import TemporalPhase

ACTIVE         # Active encoding/retrieval
IDLE           # Low activity, maintenance
CONSOLIDATING  # Offline consolidation
SLEEPING       # Full sleep cycle
```

### TemporalState

Complete temporal context snapshot:

```python
from t4dm.temporal import TemporalState

state = dynamics.get_state()

# Phase info
state.phase              # TemporalPhase.ACTIVE
state.cognitive_mode     # CognitiveMode.RETRIEVAL
state.phase_duration     # timedelta

# Timing
state.session_start      # datetime
state.last_activity      # datetime

# Metrics
state.retrieval_count    # int
state.encoding_count     # int
state.update_count       # int
state.active_traces      # Eligibility trace count
state.pending_reconsolidations  # int

# Neuromodulator
state.neuromodulator     # NeuromodulatorState
```

### TemporalDynamics

Unified coordinator:

```python
from t4dm.temporal import TemporalDynamics, TemporalConfig, create_temporal_dynamics

config = TemporalConfig(
    # Phase transitions (seconds)
    idle_threshold_seconds=300.0,           # 5 min → IDLE
    consolidation_threshold_seconds=1800.0, # 30 min → CONSOLIDATING

    # Neuromodulator dynamics
    ach_encoding_bias=0.8,
    ach_retrieval_bias=0.2,
    ne_decay_rate=0.1,
    da_baseline=0.5,

    # Subsystems
    reconsolidation_enabled=True,
    homeostatic_enabled=True,
    serotonin_enabled=True,

    # Background updates
    update_interval_seconds=1.0
)

dynamics = create_temporal_dynamics(config)

# Session lifecycle
dynamics.begin_session(session_id, mode, goal)
dynamics.end_session(session_id, success=True, outcome_score=0.9)

# Cognitive state
dynamics.set_cognitive_mode(CognitiveMode.ENCODING)

# Record operations
dynamics.record_retrieval(
    memory_ids=[uuid1, uuid2],
    query_embedding=embedding,
    scores=[0.9, 0.8],
    original_embeddings=[emb1, emb2]  # P3-QUALITY-001 fix
)

dynamics.record_encoding(memory_id, embedding)

# Record outcome (triggers reconsolidation)
updates = dynamics.record_outcome(outcome_score=0.8)
# Returns list[ReconsolidationUpdate]

# Phase callbacks
dynamics.register_phase_callback(on_phase_change)
dynamics.register_state_callback(on_state_update)

# Manual update (or use background loop)
dynamics.update()

# Background loop
await dynamics.start_background_updates()
await dynamics.stop_background_updates()
```

## Plasticity Coordination

### State Adapters

Convert between neuromodulator representations:

```python
from t4dm.temporal import adapt_orchestra_state

# Convert NeuromodulatorOrchestra → ModulatedEmbeddingAdapter
adapter_state = adapt_orchestra_state(orchestra_state)
# dopamine: 0.5 + clip(rpe, -0.5, 0.5)
# NE: clip(gain / 2.0, 0, 1)
# ACh: {encoding: 0.9, balanced: 0.5, retrieval: 0.2}
# 5-HT: pass-through
```

### Consolidation-Aware States

Factory functions for specific operations:

```python
from t4dm.temporal import (
    get_consolidation_state,
    get_sleep_replay_state,
    get_pattern_separation_state
)

# Offline consolidation
state = get_consolidation_state()
# ACh: 0.1 | DA: 0.7 | NE: 0.2 | 5-HT: 0.8

# Sharp-wave ripple replay
state = get_sleep_replay_state()
# ACh: 0.05 | DA: 0.6 | NE: 0.1 | 5-HT: 0.9

# Maximum encoding discrimination
state = get_pattern_separation_state()
# ACh: 0.95 | DA: 0.3 | NE: 0.5 | 5-HT: 0.3
```

### LearnedSalienceProvider

Extract learned importance from gate weights:

```python
from t4dm.temporal import LearnedSalienceProvider

provider = LearnedSalienceProvider()

# Set weights from LearnedMemoryGate
provider.set_gate_weights(gate.W_content)

# Get normalized salience weights
weights = provider.get_salience_weights(dimension=1024)
# Higher weights → more important dimensions
```

### PlasticityCoordinator

Orchestrate plasticity with embedding modulation:

```python
from t4dm.temporal import PlasticityCoordinator, PlasticityConfig, create_plasticity_coordinator

config = PlasticityConfig(
    max_update_per_outcome=10,
    cooldown_seconds=60.0,
    target_norm=1.0,
    homeostatic_interval=100,
    apply_modulation_to_updates=True,
    modulation_strength=0.5
)

coordinator = create_plasticity_coordinator(config)

# Set providers
coordinator.set_salience_provider(learned_salience)
coordinator.set_modulated_adapter(embedding_adapter)

# Process outcome
updates = await coordinator.process_outcome(
    outcome_score=0.8,
    retrieved_embeddings=[emb1, emb2],
    query_embedding=query,
    memory_ids=[uuid1, uuid2],
    current_state=neuromod_state
)
# Returns list[ReconsolidationUpdate] with modulated embeddings
```

## Integration Flows

### Session → Neuromodulator → Embedding

```
SessionManager.start_session()
    ↓
TemporalDynamics.begin_session()
    ↓
NeuromodulatorState.for_mode() [encoding/retrieval/exploration]
    ↓
ModulatedEmbeddingAdapter.set_state()
    ↓
Embeddings modulated appropriately
```

### Retrieval → Outcome → Reconsolidation

```
TemporalDynamics.record_retrieval()
    ├─ SerotoninSystem.add_eligibility() [eligibility traces]
    └─ Store pending updates [with original embeddings]

TemporalDynamics.record_outcome()
    ├─ ReconsolidationEngine.reconsolidate()
    ├─ PlasticityCoordinator.process_outcome() [with modulation]
    ├─ SerotoninSystem.receive_outcome() [credit distribution]
    └─ Update dopamine [prediction error]
```

### Phase Transitions → Consolidation

```
TemporalDynamics.update()
    ├─ ACTIVE → IDLE (5 min idle)
    ├─ IDLE → CONSOLIDATING (30 min idle)
    └─ _transition_to_phase()
        ├─ get_consolidation_state()
        └─ Notify phase_callbacks
```

## Configuration Reference

### TemporalConfig

```python
# Phase transitions (seconds)
idle_threshold_seconds: float = 300.0           # 5 min
consolidation_threshold_seconds: float = 1800.0 # 30 min

# Neuromodulator dynamics
ach_encoding_bias: float = 0.8
ach_retrieval_bias: float = 0.2
ne_decay_rate: float = 0.1                      # Per minute
da_baseline: float = 0.5

# Subsystems
reconsolidation_enabled: bool = True
homeostatic_enabled: bool = True
serotonin_enabled: bool = True

# Background updates
update_interval_seconds: float = 1.0
trace_cleanup_interval: float = 60.0
```

### PlasticityConfig

```python
max_update_per_outcome: int = 10
cooldown_seconds: float = 60.0
target_norm: float = 1.0
homeostatic_interval: int = 100
apply_modulation_to_updates: bool = True
modulation_strength: float = 0.5
```

## Testing

```bash
# Run temporal tests
pytest tests/unit/test_temporal_dynamics.py -v
pytest tests/unit/test_temporal_integration.py -v
pytest tests/temporal/ -v
```

## Public API

```python
# Session management
SessionContext, SessionManager, get_session_manager

# Temporal dynamics
TemporalPhase, TemporalState, TemporalConfig
TemporalDynamics, create_temporal_dynamics

# State adapters
adapt_orchestra_state
get_consolidation_state, get_sleep_replay_state, get_pattern_separation_state

# Plasticity coordination
SalienceProvider (Protocol), LearnedSalienceProvider
PlasticityConfig, PlasticityCoordinator, create_plasticity_coordinator
```

## Design Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| Singleton | SessionManager | Global session lifecycle |
| Factory | create_* functions | Object creation |
| Protocol | SalienceProvider | Interface definition |
| Observer | Phase/state callbacks | Event notification |
| Context Manager | session() | Session lifecycle |
| Background Loop | start_background_updates | Async updates |
