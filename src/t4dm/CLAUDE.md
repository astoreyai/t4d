# T4DM Package
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/`
**Version**: 2.0.0

## What
Biologically-inspired temporal memory system combining a frozen Qwen2.5-3B backbone with trainable spiking cortical blocks, backed by T4DX embedded spatiotemporal storage. Implements cognitive consolidation, neuromodulator dynamics, STDP learning, and Hopfield retrieval.

## Package Structure

```
t4dm/
├── core/              # Types, config, memory interface, MemoryItem, temporal gate
├── api/               # FastAPI REST endpoints
├── cli/               # Command-line interface (t4dm command)
├── bridges/           # High-level memory operations wrapping T4DX
├── storage/           # T4DX embedded engine, circuit breakers, archive
├── persistence/       # WAL, checkpoints, recovery, graceful shutdown
├── encoding/          # Time2Vec temporal encoding
├── embedding/         # Modulated embeddings (neuromodulator-gated)
├── learning/          # Hebbian/STDP learning, homeostatic plasticity, reconsolidation
├── consolidation/     # LSM compaction-based memory consolidation
├── dreaming/          # Sleep replay consolidation (SWR)
├── prediction/        # JEPA latent prediction, active inference, predictive coding
├── temporal/          # Temporal dynamics, session management, lifecycle
├── extraction/        # Entity/relationship extraction from text
├── nca/               # Neural Circuit Architecture (brain region simulations, NOT Mordvintsev cellular automata)
├── spiking/           # Spiking cortical blocks (LIF, thalamic gate, STDP attention)
├── qwen/              # Qwen 3B + QLoRA integration
├── observability/     # Logging, metrics, health, tracing, Prometheus
├── visualization/     # 22 visualization modules for neural dynamics
├── sdk/               # Python sync/async clients + agent SDK
├── mcp/               # Model Context Protocol integration
├── hooks/             # Lifecycle hooks
├── interfaces/        # Abstract interfaces
├── integration/       # Cross-module integration
└── integrations/      # External service integrations
```

## How It Fits Together

```
SDK/API --> core/memory --> bridges --> storage (T4DX embedded engine)
                              |              |
                              v              v
                         persistence    LSM compaction = consolidation
                              |
              temporal <------+------> learning
                  |                       |
                  v                       v
           neuromodulators          STDP/Hebbian/Hopfield
                  |                       |
                  +--------> consolidation/dreaming
                                    |
                              prediction (JEPA)
                                    |
                              visualization
```

## Key Entry Points
- `t4dm.memory_api`: Simplified `store()` / `recall()` functions
- `t4dm.api.routes`: FastAPI REST endpoints
- `t4dm.cli`: `t4dm` command-line tool
- `t4dm.sdk.client`: `T4DMClient` / `AsyncT4DMClient`

## Submodule Documentation
| Module | CLAUDE.md |
|--------|-----------|
| [observability/](observability/) | Logging, metrics, tracing, Prometheus |
| [persistence/](persistence/) | WAL, checkpoints, crash recovery |
| [prediction/](prediction/) | JEPA prediction, active inference, predictive coding |
| [sdk/](sdk/) | Python clients and agent SDK |
| [storage/](storage/) | T4DX embedded engine with LSM compaction |
| [temporal/](temporal/) | Neuromodulator dynamics, lifecycle, sessions |
| [visualization/](visualization/) | 22 visualization modules for neural dynamics |

## Integration Points
- **T4DX**: Embedded inside T4DM as the storage engine (not a separate service)
- **T4DV**: T4DV renders T4DM visualization outputs in 3D/4D
- **Kymera**: Agents use the SDK to store/recall memories
