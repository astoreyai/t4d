# T4DM vs SimpleBaseline: Comparison Guide

This document compares T4DM's full biologically-inspired memory system with the SimpleBaseline (`t4dm.lite`) minimal implementation, helping you choose the right approach for your use case.

## Overview

| Aspect | SimpleBaseline (`t4dm.lite`) | Full T4DM |
|--------|------------------------------|-----------|
| **Purpose** | Quick prototyping, simple demos | Production AI memory with bio-inspired features |
| **Lines of Code** | ~189 | ~50,000+ |
| **Dependencies** | numpy only | PyTorch, Pydantic, FastAPI, and more |
| **Setup Time** | 0 seconds (in-memory) | Requires storage directory, optional GPU |
| **Memory Model** | Flat key-value with vectors | Kappa-gradient consolidation (episodic->semantic) |
| **Persistence** | None (RAM only) | WAL + LSM segments + checkpoints |

---

## Feature Comparison

### Storage and Retrieval

| Feature | SimpleBaseline | Full T4DM |
|---------|---------------|-----------|
| Store memories | Yes | Yes |
| Search by similarity | Yes (cosine) | Yes (HNSW + temporal weighting) |
| Delete memories | Yes | Yes (with tombstones) |
| Get by ID | Yes | Yes |
| Persistence | No | WAL + LSM segments |
| Temporal queries | No | Yes (time_min/time_max filters) |
| Kappa filtering | No | Yes (consolidation level queries) |
| Graph traversal | No | Yes (CSR graph structure) |

### Memory Types

| Feature | SimpleBaseline | Full T4DM |
|---------|---------------|-----------|
| Episodic memories | Implicit | Explicit type with FSRS decay |
| Semantic memories | No | Yes (entities with bi-temporal versioning) |
| Procedural memories | No | Yes (procedures with success tracking) |
| Relationships | No | Yes (Hebbian-strengthened edges) |
| Memory consolidation | No | Yes (kappa gradient 0.0->1.0) |

### Biological Features

| Feature | SimpleBaseline | Full T4DM |
|---------|---------------|-----------|
| Kappa (consolidation level) | No | Yes (0.0=raw episodic, 1.0=stable knowledge) |
| Sleep consolidation | No | Yes (NREM replay, REM abstraction, PRUNE) |
| Spiking neural network | No | Yes (6-stage cortical blocks) |
| STDP learning | No | Yes (spike-timing dependent plasticity) |
| Neuromodulators | No | Yes (DA, NE, ACh, 5-HT bus) |
| Sharp-wave ripples | No | Yes (SWR replay compression) |
| Temporal gating | No | Yes (tau(t) write gate) |
| Prediction error | No | Yes (VTA circuit for RPE) |

### Integration

| Feature | SimpleBaseline | Full T4DM |
|---------|---------------|-----------|
| Custom embeddings | Yes (embed_fn parameter) | Yes (EmbeddingProvider protocol) |
| LangChain adapter | No | Yes |
| LlamaIndex adapter | No | Yes |
| AutoGen adapter | No | Yes |
| CrewAI adapter | No | Yes |
| REST API | No | Yes (FastAPI) |
| MCP integration | No | Yes |
| Visualization | No | Yes (22 viz modules) |

---

## When to Use SimpleBaseline

Choose `t4dm.lite.Memory` when:

1. **Quick prototyping**: You want to test memory-augmented prompts without setup overhead
2. **Demos and tutorials**: Simple code that fits in a single file
3. **Lightweight applications**: Memory requirements are small (<10,000 items)
4. **No persistence needed**: You don't need memories to survive restarts
5. **Minimal dependencies**: You want to avoid PyTorch and other heavy dependencies
6. **Learning**: Understanding vector memory basics before diving into bio-inspired features

### Example Use Cases for SimpleBaseline

- Chatbot prototype with conversation history
- Quick RAG experiment
- Unit testing memory-dependent code
- Educational examples
- CLI tools with session-only memory

### Code Example: SimpleBaseline

```python
from t4dm.lite import Memory

# Create memory (in-memory, no setup needed)
mem = Memory()

# Store some memories
mem.store("Python is a programming language")
mem.store("JavaScript runs in browsers")
mem.store("Rust is memory-safe")

# Search for related memories
results = mem.search("web development")
for r in results:
    print(f"{r['score']:.3f}: {r['content']}")

# Or use module-level convenience functions
from t4dm import lite
lite.store("Quick note")
results = lite.search("note")
```

---

## When to Use Full T4DM

Choose the full T4DM system when:

1. **Production deployment**: You need reliability, persistence, and crash recovery
2. **Long-running agents**: Memories must persist across restarts and sessions
3. **Memory consolidation**: You want episodic memories to evolve into semantic knowledge
4. **Biological plausibility**: You're researching or building brain-inspired AI
5. **Large-scale memory**: You need efficient storage for millions of memories
6. **Temporal reasoning**: You need to query "what did I know at time T?"
7. **Explainability**: You need provenance tracking and audit trails
8. **Integration**: You need adapters for LangChain, LlamaIndex, etc.

### Example Use Cases for Full T4DM

- Production AI assistants with long-term memory
- Research on continual learning and catastrophic forgetting
- Agent systems that learn from experience
- Systems requiring EU AI Act compliance (provenance)
- Multi-session applications with memory consolidation
- Neuroscience-inspired AI research

### Code Example: Full T4DM

```python
from t4dm import memory_api
from t4dm.core.memory_item import MemoryItem
from t4dm.consolidation.sleep import SleepConsolidation

# Store with full metadata
memory_id = memory_api.store(
    content="User prefers dark mode",
    importance=0.8,
    session_id="session_123"
)

# Recall with temporal constraints
results = memory_api.recall(
    query="user preferences",
    time_range=(yesterday, now),
    kappa_min=0.3  # Only consolidated memories
)

# Run sleep consolidation (converts episodic->semantic)
consolidator = SleepConsolidation(
    episodic_memory=episodic_store,
    semantic_memory=semantic_store,
    graph_store=graph_store
)
result = await consolidator.full_sleep_cycle("session_123")
print(f"Replayed: {result.nrem_replays}, Abstractions: {result.rem_abstractions}")
```

---

## Performance Characteristics

### SimpleBaseline

| Metric | Value | Notes |
|--------|-------|-------|
| Store latency | ~0.1ms | Hash-based embedding |
| Search latency | O(n) | Linear scan, no index |
| Memory usage | ~1KB per item | Embedding + metadata |
| Max practical size | ~10,000 items | Search becomes slow |
| Startup time | 0ms | In-memory only |

### Full T4DM

| Metric | Value | Notes |
|--------|-------|-------|
| Store latency | ~1-5ms | WAL write + memtable |
| Search latency | O(log n) | HNSW index |
| Memory usage | ~2KB per item | Full metadata + graph edges |
| Max practical size | Millions | LSM segments scale |
| Startup time | ~100-500ms | WAL replay + index load |

### When Performance Matters

- **SimpleBaseline** is faster for small datasets (<1,000 items) due to no indexing overhead
- **Full T4DM** scales better: search remains fast even at 1M+ items
- **Full T4DM** has higher write latency due to WAL durability guarantees

---

## Migration Path: SimpleBaseline to Full T4DM

When you outgrow SimpleBaseline, here's how to migrate:

### Step 1: Install Full Dependencies

```bash
pip install t4dm[full]  # Includes PyTorch, FastAPI, etc.
```

### Step 2: Adapt Your Code

**Before (SimpleBaseline):**
```python
from t4dm.lite import Memory

mem = Memory()
mem.store("content", metadata={"key": "value"})
results = mem.search("query", k=5)
```

**After (Full T4DM):**
```python
from t4dm.sdk.client import T4DMClient

client = T4DMClient(data_dir="./memory_data")
client.store(
    content="content",
    metadata={"key": "value"},
    importance=0.5  # New: importance scoring
)
results = client.recall(
    query="query",
    k=5,
    kappa_min=0.0  # New: consolidation filtering
)
```

### Step 3: Migrate Existing Data

```python
from t4dm.lite import Memory
from t4dm.sdk.client import T4DMClient

# Load old memories
old_mem = Memory()
# ... populate from backup ...

# Migrate to new system
new_client = T4DMClient(data_dir="./memory_data")

for item_id, item in old_mem._memories.items():
    new_client.store(
        content=item.content,
        metadata=item.metadata,
        # SimpleBaseline doesn't have importance, use default
        importance=0.5
    )
```

### Step 4: Enable Consolidation (Optional)

```python
from t4dm.consolidation.service import ConsolidationService

# Run consolidation periodically
consolidation = ConsolidationService(client)
await consolidation.consolidate(mode="light")  # Quick consolidation
# Or schedule automatic consolidation
consolidation.start_scheduler(interval_hours=8)
```

---

## Architectural Differences

### SimpleBaseline Architecture

```
User Code
    |
    v
Memory class
    |
    +-- _memories: dict[str, _MemoryItem]  (in-memory storage)
    |
    +-- _embed_fn: callable  (hash-based or custom)
    |
    +-- search: O(n) linear scan
```

### Full T4DM Architecture

```
User Code / SDK / API
    |
    v
Core Types (Episode, Entity, Procedure, MemoryItem)
    |
    v
Bridges (high-level memory operations)
    |
    +-- Episodic Store
    +-- Semantic Store
    +-- Procedural Store
    |
    v
T4DX Storage Engine
    |
    +-- MemTable (working memory, in-RAM)
    +-- WAL (durability)
    +-- LSM Segments (persistent storage)
    +-- HNSW Index (vector search)
    +-- CSR Graph (relationships)
    +-- Kappa Index (consolidation queries)
    |
    v
Spiking Neural Network (optional)
    |
    +-- Cortical Stack (6 blocks x N layers)
    +-- LIF Neurons
    +-- STDP Attention
    +-- Neuromodulator Bus
    |
    v
Consolidation Service
    |
    +-- NREM: Sharp-wave ripple replay
    +-- REM: Clustering and abstraction
    +-- PRUNE: Weak connection removal
```

---

## Summary Decision Matrix

| Criterion | Choose SimpleBaseline | Choose Full T4DM |
|-----------|----------------------|------------------|
| **Project stage** | Prototype/MVP | Production |
| **Memory size** | <10K items | Any size |
| **Persistence** | Not needed | Required |
| **Dependencies** | Minimal | Can accept PyTorch |
| **Memory types** | Just vectors | Episodic/Semantic/Procedural |
| **Consolidation** | Not needed | Needed |
| **Temporal queries** | Not needed | Needed |
| **Framework integration** | DIY is fine | Need LangChain/etc |
| **Research focus** | Application | Memory systems |

---

## Further Reading

- [T4DM Full Documentation](./README.md)
- [Storage Architecture](./plans/FULL_SYSTEM_PLAN.md)
- [Biological Basis](./EQUATIONS_MAP.md)
- [API Reference](./api/README.md)
