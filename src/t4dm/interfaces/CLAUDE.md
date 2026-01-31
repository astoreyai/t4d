# Interfaces
**Path**: `/mnt/projects/t4d/t4dm/src/ww/interfaces/`

## What
Rich terminal UI components for memory exploration, system monitoring, CRUD operations, and neural dynamics inspection. Built on the `rich` library.

## How
- **SystemDashboard**: Real-time monitoring of memory counts, storage health (Qdrant/Neo4j), circuit breaker states, cache stats, and performance metrics. Uses `rich.live` for auto-refreshing display.
- **MemoryExplorer**: Interactive browser for episodic, semantic, and procedural memories with search, filtering, and detail views.
- **CRUDManager**: Create, read, update, delete operations on memories via terminal UI.
- **LearningInspector**: Visualize learning system state -- neuromodulator levels, eligibility traces, STDP weights, three-factor signals.
- **NCAExplorer**: Inspect neural cellular automata state -- neural field dynamics, oscillator phases, attractor basins, coupling matrices.
- **TraceViewer**: View execution traces and hook activity for debugging.
- **ExportUtility**: Export memories and system state to various formats.

## Why
Provides human-readable visibility into WW's internal state for development, debugging, and demonstration. Essential for validating biological plausibility of neural dynamics.

## Key Files
| File | Purpose |
|------|---------|
| `dashboard.py` | `SystemDashboard` -- real-time health monitoring |
| `memory_explorer.py` | `MemoryExplorer` -- browse and search memories |
| `crud_manager.py` | `CRUDManager` -- memory CRUD operations |
| `learning_inspector.py` | `LearningInspector` -- neuromodulator and learning state |
| `nca_explorer.py` | `NCAExplorer` -- neural field and oscillator inspection |
| `trace_viewer.py` | `TraceViewer` -- execution trace visualization |
| `export_utils.py` | `ExportUtility` -- data export |

## Data Flow
```
WW core systems -> Interface queries -> rich terminal rendering -> user
User commands -> Interface -> WW CRUD/search operations -> display results
```

## Integration Points
- **memory/**: Reads from episodic, semantic, procedural memory stores
- **storage/**: Queries Qdrant and Neo4j health status
- **learning/**: Reads neuromodulator orchestra state, scorer weights
- **nca/**: Reads neural field state, oscillator phases, attractor basins
