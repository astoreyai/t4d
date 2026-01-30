# Architecture

World Weaver's architecture combines cognitive neuroscience principles with modern software engineering.

## System Overview

```mermaid
graph TB
    subgraph Interface["Interface Layer"]
        CLI[CLI<br/>typer]
        REST[REST API<br/>FastAPI]
        SDK[Python SDK<br/>httpx]
        MEM[Memory API<br/>Direct]
    end

    subgraph Hooks["Hook Layer"]
        PRE[Pre Hooks]
        ON[On Hooks]
        POST[Post Hooks]
    end

    subgraph Memory["Memory Subsystems"]
        EP[Episodic Memory<br/>FSRS Scheduling]
        SEM[Semantic Memory<br/>ACT-R Activation]
        PROC[Procedural Memory<br/>Skill Execution]
    end

    subgraph Prediction["Prediction & Dreaming"]
        HIER[Hierarchical<br/>Predictor]
        DREAM[Dreaming<br/>System]
        CAUSAL[Causal<br/>Discovery]
    end

    subgraph NCA["NCA Dynamics"]
        NT[Neural Field<br/>6-NT PDE]
        TG[Theta-Gamma<br/>Coupling]
        SPACE[Place/Grid<br/>Cells]
    end

    subgraph Learning["Learning Layer"]
        DA[Dopamine<br/>RPE]
        SERO[Serotonin<br/>Credit]
        HEBB[Hebbian<br/>Learning]
    end

    subgraph Storage["Storage Layer"]
        NEO4J[(Neo4j<br/>Graph)]
        QDRANT[(Qdrant<br/>Vector)]
    end

    Interface --> Hooks
    Hooks --> Memory
    Memory --> Prediction
    Memory --> NCA
    Memory --> Learning
    Prediction --> NCA
    Learning --> Memory
    Memory --> Storage
```

## Layer Details

### Interface Layer

Four access methods for different use cases:

| Interface | Best For | Async | Authentication |
|-----------|----------|-------|----------------|
| **CLI** | Terminal users | No | None |
| **REST API** | External services | Yes | API Key |
| **Python SDK** | Python applications | Yes/No | API Key |
| **Memory API** | Direct embedding | Yes | None |

### Hook Layer

Extensible hook system for customization:

```mermaid
sequenceDiagram
    participant Client
    participant Pre as Pre Hooks
    participant Op as Operation
    participant Post as Post Hooks

    Client->>Pre: Request
    Pre->>Pre: Validation
    Pre->>Pre: Caching check
    Pre->>Op: Validated request
    Op->>Op: Execute
    Op->>Post: Result
    Post->>Post: Audit logging
    Post->>Post: Hebbian update
    Post->>Client: Response
```

### Memory Subsystems

#### Episodic Memory

Stores autobiographical events with temporal context:

- **Storage**: Qdrant vectors + Neo4j metadata
- **Retrieval**: Embedding similarity + temporal weighting
- **Decay**: FSRS-based spaced repetition
- **Key files**: `src/ww/memory/episodic.py`

#### Semantic Memory

Knowledge graph with spreading activation:

- **Storage**: Neo4j graph
- **Retrieval**: ACT-R activation spreading
- **Learning**: Hebbian co-access strengthening
- **Key files**: `src/ww/memory/semantic.py`

#### Procedural Memory

Skills with execution tracking:

- **Storage**: Qdrant vectors + Neo4j metadata
- **Retrieval**: Task-based matching
- **Learning**: Success rate tracking
- **Key files**: `src/ww/memory/procedural.py`

### Prediction Layer

```mermaid
graph LR
    subgraph Context["Context Encoding"]
        C1[Episode 1]
        C2[Episode 2]
        C3[Episode N]
    end

    subgraph Predictor["Hierarchical Predictor"]
        FAST[Fast<br/>1-step]
        MED[Medium<br/>5-step]
        SLOW[Slow<br/>15-step]
    end

    subgraph Output["Predictions"]
        P1[Immediate]
        P2[Short-term]
        P3[Long-term]
    end

    Context --> FAST --> P1
    Context --> MED --> P2
    Context --> SLOW --> P3
```

### NCA Dynamics Layer

Six-neurotransmitter PDE system:

| NT | Full Name | Role | Timescale |
|----|-----------|------|-----------|
| DA | Dopamine | Reward, motivation | ~100ms |
| 5-HT | Serotonin | Mood, satiety | ~500ms |
| ACh | Acetylcholine | Attention, encoding | ~50ms |
| NE | Norepinephrine | Arousal, vigilance | ~200ms |
| GABA | - | Fast inhibition | ~10ms |
| Glu | Glutamate | Fast excitation | ~5ms |

### Storage Layer

Dual-store architecture with Saga pattern:

```mermaid
sequenceDiagram
    participant Client
    participant Saga
    participant Qdrant
    participant Neo4j

    Client->>Saga: Create Episode
    Saga->>Qdrant: Upsert vector
    Qdrant-->>Saga: Success
    Saga->>Neo4j: Create node
    Neo4j-->>Saga: Success
    Saga-->>Client: Episode created

    Note over Saga,Neo4j: On failure, compensate
```

## Data Flow

### Store Operation

```mermaid
flowchart TD
    A[Content] --> B[Embedding]
    B --> C{Gate Decision}
    C -->|Accept| D[Pattern Separation]
    C -->|Reject| E[Discard]
    D --> F[Buffer Add]
    F --> G{Evidence > θ?}
    G -->|Yes| H[Promote to LTM]
    G -->|No| I[Decay in Buffer]
```

### Recall Operation

```mermaid
flowchart TD
    A[Query] --> B[Query Embedding]
    B --> C[Pattern Completion]
    C --> D[Cluster Selection]
    D --> E[Vector Search]
    E --> F[Score Fusion]
    F --> G[Learned Reranking]
    G --> H[Buffer Probe]
    H --> I[Results]
```

## Module Organization

```
src/ww/
├── api/           # REST API server
├── bridge/        # Memory-NCA integration
├── cli/           # Command-line interface
├── consolidation/ # Memory consolidation
├── core/          # Types, config, schemas
├── dreaming/      # Dream trajectory generation
├── embedding/     # BGE-M3 embeddings
├── encoding/      # Sparse, dendritic encoding
├── hooks/         # Hook system
├── learning/      # Neuromodulators, STDP
├── memory/        # Episodic, semantic, procedural
├── nca/           # Neural dynamics
├── persistence/   # Checkpoint, WAL
├── prediction/    # JEPA, hierarchical
├── sdk/           # Python client
├── storage/       # Neo4j, Qdrant
└── visualization/ # Dashboards
```

## Key Design Decisions

### 1. Dual Storage

- **Vector (Qdrant)**: Fast similarity search
- **Graph (Neo4j)**: Rich relationships, temporal links

### 2. Saga Pattern

Distributed transactions across stores with compensation on failure.

### 3. Hook System

Decoupled concerns through priority-based hooks at PRE/ON/POST phases.

### 4. Session Isolation

Complete isolation between sessions using collection namespacing.

### 5. Async-First

All I/O operations are async with sync wrappers for convenience.

## MCP/Claude Integration

World Weaver integrates with Claude through the Model Context Protocol (MCP).

### Integration Architecture

```mermaid
graph TB
    subgraph Claude["Claude Desktop/Code"]
        CLAUDE[Claude AI]
        MCP_CLIENT[MCP Client]
    end

    subgraph WW_Server["World Weaver MCP Server"]
        MCP_HANDLER[MCP Handler]
        TOOLS[Tool Definitions]
        RESOURCES[Resources]
    end

    subgraph WW_Core["World Weaver Core"]
        MEMORY_API[Memory API]
        SESSION[Session Manager]
    end

    CLAUDE --> MCP_CLIENT
    MCP_CLIENT <-->|stdio/SSE| MCP_HANDLER
    MCP_HANDLER --> TOOLS
    MCP_HANDLER --> RESOURCES
    TOOLS --> MEMORY_API
    RESOURCES --> SESSION

    style CLAUDE fill:#e8f5e9
    style MCP_HANDLER fill:#e3f2fd
    style MEMORY_API fill:#fff3e0
```

### MCP Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `ww_store` | Store memory | content, importance, tags |
| `ww_recall` | Retrieve memories | query, limit, memory_type |
| `ww_forget` | Delete memory | episode_id |
| `ww_stats` | System statistics | - |
| `ww_consolidate` | Trigger consolidation | - |

### MCP Request Flow

```mermaid
sequenceDiagram
    participant Claude
    participant MCP as MCP Client
    participant Server as WW MCP Server
    participant Memory as Memory API

    Claude->>MCP: Call ww_recall("project status")
    MCP->>Server: tools/call request

    rect rgb(232, 245, 233)
        Note over Server: Process Request
        Server->>Server: Validate parameters
        Server->>Server: Get session context
        Server->>Memory: recall(query, session_id)
    end

    rect rgb(227, 242, 253)
        Note over Memory: Execute Recall
        Memory->>Memory: Generate embedding
        Memory->>Memory: Pattern completion
        Memory->>Memory: Vector search
        Memory->>Memory: Score fusion
    end

    Memory-->>Server: MemoryResult[]
    Server->>Server: Format response
    Server-->>MCP: Tool result
    MCP-->>Claude: Formatted memories
```

### MCP Resources

```mermaid
graph LR
    subgraph Resources["Available Resources"]
        R1[ww://session/current]
        R2[ww://stats/overview]
        R3[ww://memories/recent]
    end

    subgraph Data["Resource Data"]
        D1[Session ID<br/>Memory counts<br/>Status]
        D2[Backend health<br/>Cache stats<br/>Version]
        D3[Last N episodes<br/>With metadata]
    end

    R1 --> D1
    R2 --> D2
    R3 --> D3
```

### Configuration

Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "world-weaver": {
      "command": "ww",
      "args": ["mcp", "serve"],
      "env": {
        "WW_SESSION_ID": "claude-desktop"
      }
    }
  }
}
```

### Session Persistence

```mermaid
sequenceDiagram
    participant Claude1 as Claude Session 1
    participant Claude2 as Claude Session 2
    participant WW as World Weaver

    Note over Claude1,WW: Morning Session
    Claude1->>WW: ww_store("Working on auth feature")
    WW-->>Claude1: stored

    Note over Claude1,WW: Session ends, Claude restarts

    Note over Claude2,WW: Afternoon Session
    Claude2->>WW: ww_recall("what was I working on?")
    WW-->>Claude2: "Working on auth feature"

    Note over Claude2,WW: Context restored across sessions
```
