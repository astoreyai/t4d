# T4DV — Observation Bus & Visualization Engine
**Path**: `/mnt/projects/t4d/t4dm/src/ww/t4dv/`

## What
In-process observation bus with async ring buffers, event emitters for spiking/storage/consolidation/neuromod components, Python renderer adapters (matplotlib + plotly), a FastAPI WebSocket server, a React 2D dashboard, and a Three.js 3D memory space viewer.

## Architecture

```
Components ──emit──► ObservationBus (per-topic deque ring buffers)
                         │
              ┌──────────┼──────────────┐
              ▼          ▼              ▼
         Aggregator   Renderers    snapshot()
         (1s/10s/60s)  (mpl/plotly)
              │
              ▼
         WebSocket Server ──100ms──► React Dashboard (2D + 3D)
```

## Key Files

| File | Purpose |
|------|---------|
| `events.py` | Pydantic event types: `SpikeEvent`, `StorageEvent`, `ConsolidationEvent`, `NeuromodEvent` |
| `bus.py` | `ObservationBus` singleton: `emit()`, `emit_sync()`, `subscribe(pattern, cb)`, ring buffers |
| `emitters/spiking.py` | PyTorch `register_forward_hook` on CorticalBlock, zero changes to spiking code |
| `emitters/storage.py` | Monkey-patches T4DX 9 ops with timing and event emission |
| `emitters/consolidation.py` | Wraps NREM/REM/PRUNE compaction methods |
| `emitters/neuromod.py` | Periodic NT level sampler |
| `renderers/protocol.py` | `RendererProtocol` + `RendererRegistry` |
| `renderers/mpl.py` | 5 matplotlib renderers (raster, neuromod, storage, kappa, energy) |
| `renderers/plotly_renderer.py` | 4 plotly renderers + kappa Sankey |
| `aggregator.py` | `SnapshotAggregator`: rolling windows, `get_dashboard_state()` → dict |
| `server.py` | FastAPI: WebSocket `/ws/observe`, REST `/api/v1/viz/*` |
| `cli.py` | `t4dm viz <view_id> [--backend mpl\|plotly] [--save path.png]` |

## Frontend (`frontend/`)

Vite + React + TypeScript + Tailwind + Recharts + D3 + Three.js

| Component | What |
|-----------|------|
| `useObservationStream` | WebSocket hook with auto-reconnect |
| `SpikeRasterView` | D3 canvas raster (time x block, color=firing_rate) |
| `KappaSankeyView` | Consolidation timeline |
| `T4dxDashboard` | Recharts bar chart for ops/sec |
| `three/MemorySpace` | Three.js 3D instanced memory nodes |
| `three/MemoryNodes` | Instanced mesh, kappa-coloring, importance-sizing |
| `three/MemoryEdges` | Line segments colored by edge type |
| `three/TimelineSync` | Canvas timeline with brush range selector |

## Usage

```python
from ww.t4dv import get_bus
from ww.t4dv.emitters.spiking import attach_spiking_hooks
from ww.t4dv.emitters.storage import attach_storage_hooks

# Attach to components
bus = get_bus()
handles = attach_spiking_hooks(cortical_stack, bus)
attach_storage_hooks(t4dx_engine, bus)

# Start visualization server
from ww.t4dv.server import run_server
run_server(port=8420)
```

## Design Decisions

1. **In-process bus** (async deque, not Kafka/Redis) — single-process system
2. **PyTorch hooks** for spiking — zero changes to `cortical_block.py`
3. **`emit_sync()`** — `call_soon_threadsafe` for tensor-producing code in hot path
4. **Adapter pattern** — wraps existing 22 viz modules via RendererRegistry
5. **Aggregator** separates raw events from dashboard payloads

## Tests

```bash
pytest tests/unit/t4dv/ -v          # 36 unit tests
pytest tests/integration/t4dv/ -v   # E2E pipeline tests
pytest tests/benchmark/t4dv/ -v -s  # Bus overhead benchmarks
```
