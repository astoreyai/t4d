# ww (World Weaver) Package
**Path**: `/mnt/projects/t4d/t4dm/src/ww/`
**Version**: 0.5.0

## What
Biologically-inspired tripartite memory system (episodic, semantic, procedural) for AI agents. Implements cognitive consolidation, neuromodulator dynamics, FSRS spaced repetition, Hopfield retrieval, STDP learning, and JEPA-style prediction.

## Package Structure

```
ww/
├── core/              # Types, config, memory interface
├── api/               # FastAPI REST endpoints
├── cli/               # Command-line interface (t4dm command)
├── bridges/           # High-level Neo4j+Qdrant memory operations
├── storage/           # Neo4j + Qdrant backends, saga transactions, circuit breakers
├── persistence/       # WAL, checkpoints, recovery, graceful shutdown
├── encoding/          # Time2Vec temporal encoding
├── embedding/         # Modulated embeddings (neuromodulator-gated)
├── learning/          # Hebbian/STDP learning, homeostatic plasticity, reconsolidation
├── consolidation/     # HDBSCAN-based memory consolidation
├── dreaming/          # Sleep replay consolidation (SWR)
├── prediction/        # JEPA latent prediction, active inference, predictive coding
├── temporal/          # Temporal dynamics, session management, lifecycle
├── extraction/        # Entity/relationship extraction from text
├── nca/               # Neural Cellular Automata
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
SDK/API --> core/memory --> bridges --> storage (Neo4j + Qdrant)
                              |              |
                              v              v
                         persistence    saga transactions
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
- `ww.memory_api`: Simplified `store()` / `recall()` functions
- `ww.api.routes`: FastAPI REST endpoints
- `ww.cli`: `t4dm` command-line tool
- `ww.sdk.client`: `WorldWeaverClient` / `AsyncWorldWeaverClient`

## Submodule Documentation
| Module | CLAUDE.md |
|--------|-----------|
| [observability/](observability/) | Logging, metrics, tracing, Prometheus |
| [persistence/](persistence/) | WAL, checkpoints, crash recovery |
| [prediction/](prediction/) | JEPA prediction, active inference, predictive coding |
| [sdk/](sdk/) | Python clients and agent SDK |
| [storage/](storage/) | Neo4j + Qdrant with saga pattern |
| [temporal/](temporal/) | Neuromodulator dynamics, lifecycle, sessions |
| [visualization/](visualization/) | 22 visualization modules for neural dynamics |

## Integration Points
- **T4DX**: T4DM encodes memories, T4DX indexes the resulting vectors
- **T4DV**: T4DV renders T4DM visualization outputs in 3D/4D
- **Kymera**: Agents use the SDK to store/recall memories
