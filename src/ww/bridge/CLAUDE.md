# Bridge Module (Deprecated)
**Path**: `/mnt/projects/t4d/t4dm/src/ww/bridge/`

## What
DEPRECATED compatibility shim that re-exports from `ww.bridges.nca_bridge`. Contains the original Memory-NCA bridge connecting memory operations to Neural Cognitive Architecture (NCA) dynamics for state-dependent encoding and retrieval.

## How
- `__init__.py` emits a `DeprecationWarning` and re-exports `NCABridge`, `MemoryNCABridge`, `BridgeConfig`, `EncodingContext`, `RetrievalContext` from `ww.bridges.nca_bridge`
- `memory_nca.py` contains the original `MemoryNCABridge` class implementing:
  - State-dependent encoding: augments embeddings with 6-dim NT state projection, modulated by cognitive state (FOCUS +1.3x boost, EXPLORE +noise)
  - State-modulated retrieval: ranks candidates by NT similarity + cognitive state match bonus
  - Learning signal integration: combines dopamine RPE + NCA coupling gradients + state-modulated learning rate
  - Consolidation triggering: selects memories for replay by recency x energy scoring

## Why
Exists solely for backwards compatibility after Phase 10.1 migration to `ww.bridges`. New code should import from `ww.bridges` directly.

## Key Files
| File | Purpose |
|------|---------|
| `__init__.py` | Deprecation warning + re-exports from `ww.bridges` |
| `memory_nca.py` | Original MemoryNCABridge implementation (~420 lines) |

## Data Flow
```
Memory Operation -> bridge.augment_encoding() -> NT state projection -> augmented embedding
Query -> bridge.modulate_retrieval() -> state similarity ranking -> ranked memory IDs
Outcome -> bridge.compute_learning_signal() -> RPE + coupling gradient -> effective LR
```

## Integration Points
- **bridges**: Migrated to `ww.bridges.nca_bridge` (canonical location)
- **nca**: NeuralFieldSolver, LearnableCoupling, StateTransitionManager, EnergyLandscape
- **learning**: DopamineSystem for RPE computation
