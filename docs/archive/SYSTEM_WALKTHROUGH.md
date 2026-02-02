# World Weaver System Walkthrough

**Version**: 0.1.0
**Last Updated**: 2025-12-09
**Author**: Generated with Claude Code

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Structure](#module-structure)
3. [Module Linkage](#module-linkage)
4. [Entry Points](#entry-points)
5. [Key Architectural Patterns](#key-architectural-patterns)

---

## System Overview

World Weaver is a biologically-inspired memory system for AI agents, implementing:

- **Tripartite Memory**: Episodic (events), Semantic (knowledge), Procedural (skills)
- **Neuromodulation**: 5-factor system (DA, NE, ACh, 5-HT, GABA)
- **Crash Recovery**: WAL + checkpointing for durability
- **Pattern Separation/Completion**: DG/CA3-inspired encoding

### File Count by Module

| Module | Files | Purpose |
|--------|-------|---------|
| `learning/` | 21 | Neuromodulation, plasticity, credit assignment |
| `mcp/` | 17 | Claude Code MCP tools |
| `api/` | 13 | REST API endpoints |
| `memory/` | 12 | Tripartite memory systems |
| `core/` | 11 | Types, config, protocols |
| `integrations/kymera/` | 11 | Voice interface |
| `visualization/` | 8 | Neural dynamics charts |
| `hooks/` | 8 | Lifecycle events |
| `embedding/` | 6 | BGE-M3 embeddings |
| `observability/` | 6 | Metrics, tracing, health |
| `interfaces/` | 6 | Terminal UI |
| `persistence/` | 6 | WAL, checkpoints, recovery |
| `storage/` | 5 | Qdrant + Neo4j backends |
| `encoding/` | 5 | Bioinspired neural encoding |
| `temporal/` | 4 | Session management |
| `consolidation/` | 4 | Sleep-based consolidation |
| `integration/` | 4 | llm_agents adapters |
| `sdk/` | 3 | Python client |
| `extraction/` | 2 | Entity extraction |
| **Total** | **136** | |

---

## Module Structure

```
t4dm/
├── core/              # Foundation layer
│   ├── types.py       # Episode, Entity, Procedure, Outcome
│   ├── config.py      # Settings with environment config
│   ├── protocols.py   # EmbeddingProvider, VectorStore, GraphStore
│   ├── learned_gate.py # LearnedMemoryGate (Bayesian + Thompson)
│   └── memory_gate.py  # Heuristic gating fallback
│
├── memory/            # Tripartite memory
│   ├── episodic.py    # Autobiographical events with FSRS decay
│   ├── semantic.py    # Knowledge graph with ACT-R activation
│   ├── procedural.py  # Skills with Memp lifecycle
│   ├── working_memory.py # Limited capacity buffer
│   ├── pattern_separation.py # DG orthogonalization + CA3 completion
│   ├── buffer_manager.py # LRU buffer for gated items
│   ├── cluster_index.py # Hierarchical O(log n) search
│   └── learned_sparse_index.py # Query-dependent addressing
│
├── learning/          # Plasticity & neuromodulation
│   ├── neuromodulators.py # NeuromodulatorOrchestra
│   ├── dopamine.py    # Reward prediction error
│   ├── norepinephrine.py # Arousal/novelty
│   ├── acetylcholine.py # Encoding/retrieval mode
│   ├── serotonin.py   # Long-term credit, patience
│   ├── inhibition.py  # GABA winner-take-all
│   ├── eligibility.py # Temporal credit traces
│   ├── three_factor.py # eligibility × neuromod × DA
│   ├── reconsolidation.py # Memory update on retrieval
│   ├── homeostatic.py # Synaptic scaling for stability
│   └── fsrs.py        # Spaced repetition decay
│
├── storage/           # Persistence backends
│   ├── qdrant_store.py # Vector similarity search
│   ├── neo4j_store.py  # Knowledge graph storage
│   ├── resilience.py   # Circuit breaker pattern
│   └── saga.py         # Distributed transactions
│
├── persistence/       # Crash recovery
│   ├── wal.py         # Write-ahead log
│   ├── checkpoint.py  # Periodic snapshots
│   ├── recovery.py    # Cold/warm start
│   ├── shutdown.py    # Graceful shutdown
│   └── manager.py     # Unified orchestrator
│
├── embedding/         # Text embeddings
│   ├── bge_m3.py      # BGE-M3 provider
│   ├── modulated.py   # Neuromod-adjusted embeddings
│   └── ensemble.py    # Multi-model ensemble
│
├── api/               # REST interface
│   ├── server.py      # FastAPI app + lifespan
│   ├── websocket.py   # Real-time events
│   └── routes/        # Endpoint handlers
│       ├── episodes.py
│       ├── entities.py
│       ├── skills.py
│       ├── persistence.py
│       └── visualization.py
│
├── mcp/               # Claude Code integration
│   ├── gateway.py     # MCP app, rate limiting, auth
│   ├── server.py      # Entry point
│   └── tools/         # MCP tool definitions
│       ├── episodic.py   # 4 episodic tools
│       ├── semantic.py   # 5 semantic tools
│       ├── procedural.py # 6 procedural tools
│       └── system.py     # 4 system tools
│
├── visualization/     # Neural dynamics charts
│   ├── activation_heatmap.py
│   ├── plasticity_traces.py
│   ├── neuromodulator_state.py
│   ├── pattern_separation.py
│   ├── consolidation_replay.py
│   ├── embedding_projections.py
│   └── persistence_state.py
│
└── observability/     # Monitoring
    ├── logging.py     # Structured logging
    ├── metrics.py     # Latency/count tracking
    ├── health.py      # Component health
    └── tracing.py     # OpenTelemetry
```

---

## Module Linkage

### Dependency Graph

```
                         ┌─────────────────────────────────┐
                         │        ENTRY POINTS             │
                         │   MCP Server / REST API / SDK   │
                         └─────────────┬───────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────┐          ┌───────────────────┐          ┌───────────────┐
│  mcp/tools/   │          │  api/routes/      │          │    sdk/       │
│  episodic.py  │          │  episodes.py      │          │  client.py    │
│  semantic.py  │          │  entities.py      │          └───────────────┘
│  procedural.py│          │  skills.py        │
└───────┬───────┘          └─────────┬─────────┘
        │                            │
        └────────────┬───────────────┘
                     ▼
        ┌────────────────────────────┐
        │     memory/                │
        │  ├── episodic.py           │  ◄── Autobiographical events
        │  ├── semantic.py           │  ◄── Knowledge graph
        │  └── procedural.py         │  ◄── Skills & procedures
        └────────────┬───────────────┘
                     │
    ┌────────────────┼────────────────┐
    │                │                │
    ▼                ▼                ▼
┌──────────┐  ┌─────────────┐  ┌──────────────┐
│embedding/│  │  learning/  │  │  storage/    │
│bge_m3.py │  │neuromod.py  │  │qdrant_store  │
│          │  │dopamine.py  │  │neo4j_store   │
└──────────┘  │serotonin.py │  └──────────────┘
              │eligibility  │
              └─────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │     persistence/           │
        │  ├── wal.py                │  ◄── Write-ahead log
        │  ├── checkpoint.py         │  ◄── State snapshots
        │  ├── recovery.py           │  ◄── Cold/warm start
        │  └── manager.py            │  ◄── Orchestrator
        └────────────────────────────┘
```

### Import Relationships

**Heavy Importers** (depend on many modules):
- `memory/episodic.py` → core, embedding, storage, learning, observability
- `learning/neuromodulators.py` → all 5 neuromodulator systems
- `api/server.py` → core, memory, mcp, observability, persistence
- `mcp/gateway.py` → memory, storage, learning, observability

**Well-Isolated Modules**:
- `core/*` → no internal ww dependencies
- `persistence/*` → only depends on core
- `observability/*` → mostly standalone

---

## Entry Points

### 1. MCP Server (Claude Code)

```
t4dm/mcp/server.py → t4dm/mcp/gateway.py → t4dm/mcp/tools/*.py
```

**Tools Available**:
| Category | Tools |
|----------|-------|
| Episodic | `create_episode`, `recall_episodes`, `query_at_time`, `mark_important` |
| Semantic | `create_entity`, `create_relation`, `semantic_recall`, `spread_activation`, `supersede_fact` |
| Procedural | `create_skill`, `recall_skill`, `execute_skill`, `deprecate_skill`, `build_skill`, `how_to` |
| System | `consolidate_now`, `get_provenance`, `get_session_id`, `memory_stats` |

### 2. REST API

```
t4dm/api/server.py → FastAPI → t4dm/api/routes/*.py
```

**Endpoints**:
```
/api/v1/episodes/*      - Episodic memory CRUD
/api/v1/entities/*      - Semantic memory CRUD
/api/v1/skills/*        - Procedural memory CRUD
/api/v1/persistence/*   - WAL/checkpoint status
/api/v1/viz/*           - Visualization data
/ws/*                   - WebSocket channels
```

### 3. Python SDK

```python
from ww.sdk import WorldWeaverClient

client = WorldWeaverClient("http://localhost:8080")
episode = await client.create_episode("Fixed auth bug", outcome="success")
results = await client.recall_episodes("authentication issues")
```

---

## Key Architectural Patterns

### 1. Biologically-Inspired Learning

```
Query → NE (arousal) → ACh (mode) → Retrieve → DA (RPE) → 5-HT (credit) → Update
```

All learning uses **three-factor rule**:
```
Δw = eligibility × neuromod_state × dopamine_surprise
```

### 2. Pattern Separation/Completion (Hippocampus Model)

```
Input → DG (orthogonalize) → CA3 (pattern complete) → CA1 (output)
        ↑                    ↑
        Pattern              Attractor
        Separation           Dynamics
```

### 3. Saga Pattern for Dual-Store Consistency

```
Create Episode:
  Step 1: vector_store.add() → Qdrant
  Step 2: graph_store.create_node() → Neo4j

  On failure: Execute compensations (rollback)
```

### 4. Crash Recovery (WAL + Checkpoint)

```
Operation → Log to WAL → Apply to state
                ↓
        Periodic checkpoint
                ↓
Recovery: Load checkpoint → Replay WAL
```

### 5. Learned Gating (What to Remember)

```
Content → LearnedMemoryGate → STORE | BUFFER | SKIP
          ↓
          Bayesian logistic regression
          + Thompson sampling for exploration
```

---

## See Also

- [MEMORY_STORE_RECALL_FLOW.md](MEMORY_STORE_RECALL_FLOW.md) - Detailed memory operation trace
- [NEUROMODULATION_WALKTHROUGH.md](NEUROMODULATION_WALKTHROUGH.md) - 5-factor system details
- [LEARNING_SYSTEM_WALKTHROUGH.md](LEARNING_SYSTEM_WALKTHROUGH.md) - Plasticity mechanisms
- [PERSISTENCE_ARCHITECTURE.md](PERSISTENCE_ARCHITECTURE.md) - WAL/checkpoint details
