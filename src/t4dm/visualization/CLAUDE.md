# Visualization
**Path**: `/mnt/projects/t4d/t4dm/src/ww/visualization/`

## What
Comprehensive visualization toolkit for analyzing neural dynamics, memory consolidation, neuromodulator state, plasticity, stability, and system telemetry. Provides both static (matplotlib) and interactive (plotly) outputs.

## How
22 visualization modules covering every subsystem:
- **Neural dynamics**: `activation_heatmap`, `pattern_separation`, `plasticity_traces`, `energy_landscape`, `stability_monitor`
- **Neuromodulators**: `neuromodulator_state`, `nt_state_dashboard`, `da_telemetry`
- **Consolidation**: `consolidation_replay`, `swr_telemetry`, `glymphatic_visualizer`
- **Learning**: `ff_visualizer` (Forward-Forward), `capsule_visualizer` (Capsule Networks), `coupling_dynamics`
- **Embeddings**: `embedding_projections` (t-SNE/UMAP)
- **System**: `persistence_state` (WAL/checkpoint), `pac_telemetry`, `telemetry_hub`
- **Validation**: `validation` (biological plausibility checks against known ranges)

Each module provides standalone plot functions and dashboard-creating factory functions.

## Why
Neural memory systems have complex internal dynamics that are impossible to debug without visualization. This module makes the invisible visible: neuromodulator interactions, consolidation replay sequences, stability bifurcations, and embedding drift.

## Key Files
| File | Purpose |
|------|---------|
| `telemetry_hub.py` | Unified cross-scale telemetry aggregation |
| `validation.py` | Biological plausibility validation against known ranges |
| `stability_monitor.py` | Lyapunov exponents, eigenvalue tracking, bifurcation detection |
| `energy_landscape.py` | Hopfield energy surface and basin occupancy |
| `embedding_projections.py` | t-SNE/UMAP dimensionality reduction plots |
| `nt_state_dashboard.py` | Neurotransmitter state with saturation curves |

## Data Flow
```
Subsystem state snapshots --> Visualizer --> matplotlib/plotly figures
TelemetryHub --> cross-scale event correlation --> unified dashboard
BiologicalValidator --> parameter range checks --> ValidationReport
```

## Integration Points
- **Temporal**: Visualizes neuromodulator and phase dynamics
- **Learning**: Plots plasticity traces, Forward-Forward goodness, capsule routing
- **Persistence**: Renders WAL timeline and checkpoint history
- **Consolidation**: Shows SWR replay sequences and priority distributions
