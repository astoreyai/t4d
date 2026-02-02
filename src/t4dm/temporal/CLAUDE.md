# Temporal
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/temporal/`

## What
Temporal dynamics coordination managing neuromodulator states, memory lifecycle phases, session context, and plasticity across multiple timescales (milliseconds to days).

## How
- **Dynamics** (`dynamics.py`): `TemporalDynamics` coordinator integrating neuromodulator state (ACh/DA/NE/5-HT), state-dependent embedding modulation, reconsolidation, credit assignment, and sleep consolidation. System cycles through `TemporalPhase`: ACTIVE, IDLE, CONSOLIDATING, SLEEPING.
- **Session** (`session.py`): `SessionManager` and `SessionContext` for scoping memory operations to named sessions with temporal boundaries.
- **Integration** (`integration.py`): `PlasticityCoordinator` bridges temporal dynamics to consolidation, sleep replay, and pattern separation. `SalienceProvider` and `LearnedSalienceProvider` compute memory importance. `adapt_orchestra_state` maps system state for orchestration.
- **Lifecycle** (`lifecycle.py`): `MemoryLifecycleManager` tracks memory through phases (encoding, consolidation, long-term, decay) with `LifecycleConfig`-controlled transitions.

## Why
Biological memory operates on multiple timescales. This module models the brain's temporal coordination: neuromodulators gate encoding vs retrieval, consolidation happens offline, and memories progress through lifecycle stages with decay.

## Key Files
| File | Purpose |
|------|---------|
| `dynamics.py` | Unified temporal coordinator with neuromodulator states |
| `session.py` | Session-scoped memory context |
| `integration.py` | Plasticity coordination and salience computation |
| `lifecycle.py` | Memory lifecycle phase management |

## Data Flow
```
System State --> TemporalDynamics --> TemporalPhase (ACTIVE/IDLE/CONSOLIDATING/SLEEPING)
             --> NeuromodulatorState --> embedding modulation
             --> PlasticityCoordinator --> consolidation/replay/separation triggers
Memory Age --> LifecycleManager --> phase transitions --> decay/archive
```

## Learning Modalities
- **Neuromodulator gating**: ACh (encoding), DA (reward), NE (arousal), 5-HT (mood/patience)
- **Homeostatic plasticity**: Keeps neural activity in biological range
- **Reconsolidation**: Reactivated memories become labile and can be updated
