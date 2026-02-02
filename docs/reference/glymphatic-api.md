# Glymphatic System API Reference

**Module**: `t4dm.nca.glymphatic`, `t4dm.nca.glymphatic_consolidation_bridge`
**Version**: Phase 4
**Biology Score**: 89/100

---

## Overview

The glymphatic system implements sleep-gated waste clearance following the biological discovery by Nedergaard (2013). Waste clearance is ~2x higher during NREM sleep compared to waking.

### Biological Basis

- **Xie et al. (2013)**: 60% higher clearance during sleep
- **Fultz et al. (2019)**: CSF oscillations during NREM
- **Iliff et al. (2012)**: AQP4 water channels in astrocyte endfeet

### Key Mechanisms

1. **Low NE (sleep)** → Astrocyte volume decrease → Interstitial space expands
2. **Delta oscillations (0.5-4 Hz)** → Drive CSF flow
3. **High ACh (REM)** → Blocks AQP4 → Minimal clearance

---

## Core Classes

### GlymphaticConfig

Configuration for waste clearance system.

```python
from t4dm.nca.glymphatic import GlymphaticConfig

config = GlymphaticConfig(
    # Sleep-state clearance rates (biological values)
    clearance_nrem_deep=0.7,      # 70% during SWS (Xie 2013)
    clearance_nrem_light=0.5,     # 50% during light sleep
    clearance_quiet_wake=0.3,     # 30% during quiet wake
    clearance_active_wake=0.1,    # 10% during active wake
    clearance_rem=0.05,           # 5% during REM (ACh blocks AQP4)

    # Waste identification thresholds
    unused_embedding_days=30,     # Prune if not retrieved
    weak_connection_threshold=0.1,
    stale_memory_stability=0.2,   # FSRS threshold

    # Delta coupling
    clear_on_delta_upstate=True,
    delta_phase_window=0.3,

    # Neuromodulator influence
    ne_modulation=0.6,            # Low NE → high clearance
    ach_modulation=0.4,           # High ACh → low clearance

    # Safety limits
    max_clearance_fraction=0.1,   # Max 10% per cycle
    preserve_recent_hours=24,     # Never clear < 24h old
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `clearance_nrem_deep` | float | 0.7 | Clearance during deep NREM |
| `clearance_nrem_light` | float | 0.5 | Clearance during light NREM |
| `clearance_quiet_wake` | float | 0.3 | Clearance during quiet wake |
| `clearance_active_wake` | float | 0.1 | Clearance during active wake |
| `clearance_rem` | float | 0.05 | Clearance during REM (low due to ACh) |
| `unused_embedding_days` | int | 30 | Days before embedding is waste |
| `weak_connection_threshold` | float | 0.1 | Weight threshold for weak connections |
| `ne_modulation` | float | 0.6 | NE influence on clearance |
| `ach_modulation` | float | 0.4 | ACh influence on clearance |
| `max_clearance_fraction` | float | 0.1 | Safety: max 10% per cycle |

---

### WakeSleepMode

Sleep state enumeration.

```python
from t4dm.nca.glymphatic import WakeSleepMode

WakeSleepMode.ACTIVE_WAKE  # Engaged in tasks
WakeSleepMode.QUIET_WAKE   # Relaxed, drowsy
WakeSleepMode.NREM_LIGHT   # N1/N2 sleep
WakeSleepMode.NREM_DEEP    # N3/SWS - highest clearance
WakeSleepMode.REM          # REM sleep - lowest clearance
```

---

### WasteCategory

Categories of neural waste.

```python
from t4dm.nca.glymphatic import WasteCategory

WasteCategory.UNUSED_EMBEDDING   # Not retrieved in N days
WasteCategory.WEAK_CONNECTION    # Low Hebbian weight
WasteCategory.STALE_MEMORY       # Low FSRS stability
WasteCategory.ORPHAN_ENTITY      # Entity with no relations
WasteCategory.EXPIRED_EPISODE    # Old episodic memory
```

---

### WasteState

Current state of identified waste.

```python
from t4dm.nca.glymphatic import WasteState

state = WasteState()

# Waste counts
print(state.unused_embeddings)
print(state.weak_connections)
print(state.stale_memories)
print(state.total_waste)

# Clearance statistics
print(state.total_cleared)
print(state.clearance_rate)
print(state.last_clearance_time)

# Per-category breakdown
print(state.total_cleared_by_category)
```

---

### GlymphaticSystem

Main waste clearance system.

```python
from t4dm.nca.glymphatic import GlymphaticSystem, GlymphaticConfig

config = GlymphaticConfig()
glymphatic = GlymphaticSystem(config=config)
```

#### Methods

##### `get_state_clearance_rate(wake_sleep_mode: WakeSleepMode) -> float`
Get base clearance rate for a sleep state.

```python
rate = glymphatic.get_state_clearance_rate(WakeSleepMode.NREM_DEEP)
# Returns: 0.7 (70% clearance during SWS)
```

##### `compute_effective_rate(wake_sleep_mode, delta_up_state, ne_level, ach_level) -> float`
Compute effective clearance with all modulations.

```python
rate = glymphatic.compute_effective_rate(
    wake_sleep_mode=WakeSleepMode.NREM_DEEP,
    delta_up_state=True,
    ne_level=0.2,      # Low NE = more clearance
    ach_level=0.1,     # Low ACh = more clearance
)
```

##### `step(wake_sleep_mode, delta_up_state, ne_level, ach_level, dt) -> WasteState`
Execute one clearance step.

```python
state = glymphatic.step(
    wake_sleep_mode=WakeSleepMode.NREM_DEEP,
    delta_up_state=True,
    ne_level=0.2,
    ach_level=0.1,
    dt=1.0,
)
```

##### `scan(current_time: datetime) -> WasteState`
Scan for waste items.

```python
waste = glymphatic.scan()
print(f"Found {waste.total_waste} waste items")
```

##### `get_statistics() -> dict`
Get system statistics.

```python
stats = glymphatic.get_statistics()
# Returns: {
#     "total_steps": 100,
#     "current_clearance_rate": 0.7,
#     "total_cleared": 50,
#     "cleared_by_category": {...},
#     ...
# }
```

---

### WasteTracker

Tracks waste items for clearance.

```python
from t4dm.nca.glymphatic import WasteTracker, GlymphaticConfig

tracker = WasteTracker(GlymphaticConfig())

# Scan for waste
state = tracker.scan_for_waste(memory_system)

# Get candidates
candidates = tracker.get_clearance_candidates(
    category=WasteCategory.WEAK_CONNECTION,
    max_items=100
)

# Mark as cleared
tracker.mark_cleared(
    category=WasteCategory.WEAK_CONNECTION,
    item_id="conn_123",
    wake_sleep_mode=WakeSleepMode.NREM_DEEP,
    clearance_rate=0.7
)

# Get history
history = tracker.get_clearance_history(
    since=datetime.now() - timedelta(hours=8)
)
```

---

### ClearanceEvent

Record of a clearance operation.

```python
from t4dm.nca.glymphatic import ClearanceEvent

event = ClearanceEvent(
    timestamp=datetime.now(),
    category=WasteCategory.WEAK_CONNECTION,
    item_id="conn_123",
    reason="Weight below threshold",
    wake_sleep_mode=WakeSleepMode.NREM_DEEP,
    clearance_rate=0.7,
)
```

---

## Factory Function

```python
from t4dm.nca.glymphatic import create_glymphatic_system

glymphatic = create_glymphatic_system(
    clearance_nrem_deep=0.7,
    clearance_wake=0.3,
    unused_embedding_days=30,
    weak_connection_threshold=0.1,
    ne_modulation=0.6,
    memory_system=my_memory_system,  # Optional
)
```

---

## Consolidation Bridge

### GlymphaticConsolidationBridge

Couples glymphatic clearance with sleep consolidation.

```python
from t4dm.nca.glymphatic_consolidation_bridge import (
    GlymphaticConsolidationBridge,
    GlymphaticBridgeConfig,
    ClearanceMode,
)

config = GlymphaticBridgeConfig(
    # Mode thresholds
    bulk_clearance_ach=0.2,    # Low ACh → bulk mode
    selective_clearance_ne=0.5, # High NE → selective
    
    # Clearance modulation
    spindle_clearance_boost=1.3,  # 30% boost during spindles
    delta_clearance_boost=1.5,    # 50% boost during delta
    
    # Replay coordination
    replay_clearance_delay=2.0,   # Wait 2s after replay
)

bridge = GlymphaticConsolidationBridge(config=config)
```

#### Methods

##### `set_sleep_stage(mode: WakeSleepMode) -> ClearanceMode`
Set sleep stage and get clearance mode.

```python
mode = bridge.set_sleep_stage(WakeSleepMode.NREM_DEEP)
# Returns: ClearanceMode.BULK or ClearanceMode.SELECTIVE
```

##### `infer_stage_from_neuromod(nt_state) -> WakeSleepMode`
Infer sleep stage from NT levels.

```python
stage = bridge.infer_stage_from_neuromod(nt_state)
```

##### `get_clearance_gain() -> float`
Get current clearance multiplier.

```python
gain = bridge.get_clearance_gain()
# Returns: 1.0-1.5 based on oscillator phase
```

##### `on_replay_complete(replay_ids: list) -> None`
Signal that replay is complete (wait before clearing).

```python
bridge.on_replay_complete(["mem_1", "mem_2"])
```

---

### ClearanceMode

Clearance mode enumeration.

```python
from t4dm.nca.glymphatic_consolidation_bridge import ClearanceMode

ClearanceMode.MINIMAL     # Active wake - minimal clearance
ClearanceMode.SELECTIVE   # Quiet wake/REM - selective
ClearanceMode.BULK        # NREM deep - bulk clearance
```

---

## Integration Example

```python
from t4dm.nca.glymphatic import GlymphaticSystem, WakeSleepMode
from t4dm.nca.glymphatic_consolidation_bridge import GlymphaticConsolidationBridge
from t4dm.nca.oscillators import DeltaOscillator
from t4dm.nca.neural_field import NeurotransmitterState

# Create systems
glymphatic = GlymphaticSystem()
bridge = GlymphaticConsolidationBridge()
delta = DeltaOscillator(freq=1.0)  # 1 Hz delta

# Sleep simulation
for t in range(1000):
    # Get oscillator phase
    delta_phase = delta.step(dt=0.001)
    delta_up_state = delta_phase > 0.5
    
    # Get NT state (low NE, low ACh for NREM)
    nt_state = NeurotransmitterState(
        norepinephrine=0.2,
        acetylcholine=0.1,
    )
    
    # Set sleep stage
    bridge.set_sleep_stage(WakeSleepMode.NREM_DEEP)
    
    # Step glymphatic
    waste_state = glymphatic.step(
        wake_sleep_mode=WakeSleepMode.NREM_DEEP,
        delta_up_state=delta_up_state,
        ne_level=nt_state.norepinephrine,
        ach_level=nt_state.acetylcholine,
        dt=0.001,
    )

# Check results
stats = glymphatic.get_statistics()
print(f"Cleared {stats['total_cleared']} items")
```

---

## Biological Mapping

| Parameter | Biological Basis | Reference |
|-----------|------------------|-----------|
| `clearance_nrem_deep=0.7` | 60% higher than wake | Xie et al. 2013 |
| `ne_modulation=0.6` | NE contracts astrocytes | Nedergaard 2013 |
| `ach_modulation=0.4` | ACh blocks AQP4 | Iliff et al. 2012 |
| `delta_upstate` | CSF flow with delta | Fultz et al. 2019 |
| `clearance_rem=0.05` | High ACh in REM | Xie et al. 2013 |

---

## See Also

- [NCA API](nca-api.md) - Core neural field
- [Sleep API](../concepts/sleep-consolidation.md) - Sleep consolidation
- [Glymphatic Concept](../concepts/glymphatic.md) - Conceptual overview
- [Adenosine](../concepts/adenosine.md) - Sleep pressure
