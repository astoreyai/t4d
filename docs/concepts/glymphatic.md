# Glymphatic System

Sleep-gated waste clearance for T4DM memory maintenance.

**Phase 4 Implementation**: Biology B8 (glymphatic clearance)
**Phase 7 Update**: VAE generative replay integration, multi-night scheduling

## Overview

The glymphatic system models brain waste clearance that occurs primarily during sleep. Based on Xie et al. (2013) and Nedergaard (2013), clearance is ~60% higher during NREM sleep vs wakefulness.

```
Key Principle:
Low NE (sleep) → astrocyte shrinkage → interstitial space expands → CSF flow increases
```

## Biological Basis

### Key Literature
| Reference | Finding |
|-----------|---------|
| Xie et al. 2013 | 60% higher clearance during sleep |
| Fultz et al. 2019 | Delta oscillations (0.5-4 Hz) drive CSF flow |
| Iliff et al. 2012 | Glymphatic pathway discovery |
| Mestre et al. 2018 | AQP4 channels mediate clearance |

### Sleep-State Clearance Rates

```
                        Clearance Rate
Sleep State             ─────────────────────────────────────
NREM Deep (SWS)         ████████████████████████████████░ 90%
NREM Light              ██████████████░░░░░░░░░░░░░░░░░░░ 50%
Quiet Wake              ██████████░░░░░░░░░░░░░░░░░░░░░░░ 30%
Active Wake             ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 10%
REM                     ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 5%
```

REM has minimal clearance due to high ACh blocking AQP4 channels.

## Waste Categories

The system identifies and clears five categories of neural "waste":

```python
class WasteCategory(Enum):
    UNUSED_EMBEDDING = "unused_embedding"    # Not retrieved in 30 days
    WEAK_CONNECTION = "weak_connection"      # Weight < 0.1
    STALE_MEMORY = "stale_memory"            # FSRS stability < 0.2
    ORPHAN_ENTITY = "orphan_entity"          # No relations
    EXPIRED_EPISODE = "expired_episode"      # Beyond retention
```

## Usage

### Basic Glymphatic System

```python
from t4dm.nca import GlymphaticSystem, GlymphaticConfig, create_glymphatic_system

config = GlymphaticConfig(
    clearance_nrem_deep=0.9,    # 90% during SWS
    clearance_rem=0.05,          # 5% during REM
    unused_embedding_days=30,    # Prune after 30 days
    max_clearance_fraction=0.1   # Never clear >10% per cycle
)

glymphatic = create_glymphatic_system(config)

# Identify waste candidates
waste_state = glymphatic.scan_for_waste(memory_store)

print(f"Unused embeddings: {waste_state.unused_embeddings}")
print(f"Weak connections: {waste_state.weak_connections}")
print(f"Stale memories: {waste_state.stale_memories}")
```

### With Delta Oscillator Coupling

```python
from t4dm.nca import DeltaOscillator, GlymphaticSystem

delta = DeltaOscillator(frequency=1.5)  # 1.5 Hz delta waves
glymphatic = GlymphaticSystem(config)

# Clearance only during delta up-states
def on_delta_upstate(phase: float):
    if config.clear_on_delta_upstate:
        if 0.3 < phase < 0.6:  # Up-state window
            glymphatic.process_clearance_batch()
```

### Integration with Sleep Cycle

```python
from t4dm.nca import AdenosineDynamics, GlymphaticSystem, SWRCoupling

adenosine = AdenosineDynamics()
glymphatic = GlymphaticSystem(config)
swr = SWRCoupling()

# During NREM sleep
def sleep_maintenance_cycle():
    if adenosine.get_sleep_state() == SleepWakeState.NREM_DEEP:
        # High clearance + memory replay
        cleared = glymphatic.process_clearance_batch()
        swr.trigger_replay_event()
        return cleared
    return 0
```

## Configuration Parameters

| Parameter | Default | Biological Basis |
|-----------|---------|------------------|
| `clearance_nrem_deep` | 0.9 | Xie et al. 2013: peak during SWS |
| `clearance_rem` | 0.05 | High ACh blocks AQP4 |
| `ne_modulation` | 0.6 | Low NE opens interstitial space |
| `delta_phase_window` | 0.3 | Fultz et al. 2019: CSF pulses |
| `preserve_recent_hours` | 24 | Never clear recent memories |

## Glymphatic-Consolidation Bridge

The `GlymphaticConsolidationBridge` coordinates clearance with memory consolidation:

```
SWR Replay → Consolidate important memories → Clear waste
            ↓                                ↓
     Strengthen                        Remove
     traces                            unused
```

```python
from t4dm.nca import GlymphaticConsolidationBridge

bridge = GlymphaticConsolidationBridge(
    glymphatic=glymphatic,
    swr_coupling=swr_coupling,
    consolidation_engine=consolidation
)

# During sleep, bridge coordinates:
# 1. SWR identifies memories for consolidation
# 2. Consolidation strengthens selected memories
# 3. Glymphatic clears non-consolidated waste
result = bridge.sleep_cycle_step()
```

## Architecture Diagram

```
              NREM Sleep State
                    │
         ┌──────────┼──────────┐
         │          │          │
         ▼          ▼          ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Delta   │ │   SWR   │ │Adenosine│
    │Oscillator│ │Coupling │ │Dynamics │
    └────┬────┘ └────┬────┘ └────┬────┘
         │          │          │
         └──────────┼──────────┘
                    │
                    ▼
         ┌──────────────────┐
         │   Glymphatic     │
         │     System       │
         │                  │
         │ - Waste Scan     │
         │ - Batch Clear    │
         │ - Safety Limits  │
         └────────┬─────────┘
                  │
                  ▼
         ┌──────────────────┐
         │  Memory Store    │
         │  - Embeddings    │
         │  - Connections   │
         │  - Episodes      │
         └──────────────────┘
```

## Safety Limits

The system enforces strict safety constraints:

1. **Max Clearance Fraction**: Never clear >10% of memory per cycle
2. **Preserve Recent**: Never clear items <24 hours old
3. **Minimum Interval**: Wait 60s between clearance batches
4. **Batch Size**: Process max 100 items per step

```python
@dataclass
class GlymphaticConfig:
    max_clearance_fraction: float = 0.1    # 10% max
    preserve_recent_hours: int = 24        # 24h minimum
    min_clearance_interval: float = 60.0   # 60s between batches
    clearance_batch_size: int = 100        # 100 items max
```

## Phase 7: VAE Generative Replay

During sleep consolidation, the glymphatic clearance coordinates with generative replay:

```python
from t4dm.consolidation import SleepConsolidation

consolidation = SleepConsolidation(
    vae_enabled=True,       # Enable VAE-based replay
    vae_latent_dim=128,
    embedding_dim=1024,
)

# During NREM, glymphatic clearance + VAE replay work together:
# 1. Clear waste (weak connections, stale memories)
# 2. Replay real + synthetic memories (prevent forgetting)
# 3. Strengthen important traces
```

### Multi-Night Scheduling

Consolidation depth increases progressively:

| Night | Depth | Glymphatic Role |
|-------|-------|-----------------|
| 1 | `light` | Minimal clearance, recent replay |
| 2-3 | `deep` | Moderate clearance, full replay |
| 4+ | `all` | Full clearance cycle, semantic extraction |

See [CONSOLIDATION_PROTOCOL.md](../CONSOLIDATION_PROTOCOL.md) for details.

## Testing

```bash
# Run glymphatic tests
pytest tests/nca/test_glymphatic.py -v

# Run integration tests
pytest tests/integration/test_h10_cross_region_consistency.py::TestGlymphaticIntegration -v

# Phase 7 consolidation tests
pytest tests/unit/test_sleep_consolidation.py::TestP7VAEGenerativeReplay -v
pytest tests/unit/test_sleep_consolidation.py::TestP7MultiNightScheduling -v
```

## References

1. Xie et al. (2013). "Sleep Drives Metabolite Clearance from the Adult Brain"
2. Nedergaard (2013). "Garbage Truck of the Brain"
3. Fultz et al. (2019). "Coupled electrophysiological, hemodynamic, and CSF oscillations"
4. Iliff et al. (2012). "A Paravascular Pathway Facilitates CSF Flow"
5. Mestre et al. (2018). "Aquaporin-4-dependent glymphatic solute transport"
