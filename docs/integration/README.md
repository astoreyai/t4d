# T4DM Integration Guide

**Version**: 2.0.0
**Last Updated**: 2026-02-05
**Purpose**: Complete guide to integrating T4DM memory system into your applications

---

## Overview

T4DM is a biologically-inspired memory system that provides:

- **Episodic Memory**: Time-sequenced experiences with κ-based consolidation
- **Semantic Memory**: Knowledge graph with Hebbian learning and spread activation
- **Procedural Memory**: Skill patterns with execution tracking
- **Neural Dynamics**: Spiking cortical blocks, neuromodulation, oscillator phases

### Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Application                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Integration Options                     │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │  REST API  │  Python SDK  │  MCP Tools  │  Adapters │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   T4DM Server                        │    │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐           │    │
│  │  │Memory │ │Spiking│ │Consol.│ │ Viz   │           │    │
│  │  │ API   │ │Blocks │ │Engine │ │Routes │           │    │
│  │  └───────┘ └───────┘ └───────┘ └───────┘           │    │
│  └─────────────────────────────────────────────────────┘    │
│                            │                                 │
│                            ▼                                 │
│  ┌─────────────────────────────────────────────────────┐    │
│  │               T4DX Storage Engine                    │    │
│  │        (Embedded LSM + HNSW + Graph)                │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Option 1: Python SDK (Recommended)

```python
from t4dm.sdk import T4DMClient

# Connect to T4DM
with T4DMClient(base_url="http://localhost:8765", session_id="my-app") as client:
    # Store a memory
    episode = client.create_episode(
        content="User completed onboarding flow",
        context={"user_id": "123", "flow": "onboarding"},
        outcome="success",
        emotional_valence=0.8,
    )

    # Recall relevant memories
    results = client.recall_episodes("onboarding experience", limit=5)
    for ep in results.episodes:
        print(f"[{ep.timestamp}] {ep.content}")
```

### Option 2: REST API

```bash
# Store a memory
curl -X POST http://localhost:8765/api/v1/episodes/ \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: my-app" \
  -d '{"content": "User completed task", "context": {"task": "setup"}}'

# Recall memories
curl "http://localhost:8765/api/v1/episodes/search?query=completed+task&limit=5" \
  -H "X-Session-ID: my-app"
```

### Option 3: Mem0-Compatible API

For drop-in Mem0 replacement:

```python
import requests

# Uses /v1/memories/ endpoint (Mem0 format)
resp = requests.post("http://localhost:8765/v1/memories/", json={
    "messages": [{"role": "user", "content": "Remember this task"}],
    "user_id": "user-123",
    "metadata": {"project": "demo"}
})
```

---

## Integration Paths

### 1. LangChain Integration

```python
from t4dm.adapters import T4DMLangChainMemory

# Use as LangChain memory backend
memory = T4DMLangChainMemory(
    base_url="http://localhost:8765",
    session_id="langchain-agent",
)

# Works with any LangChain chain
from langchain.chains import ConversationChain
chain = ConversationChain(memory=memory)
```

### 2. LlamaIndex Integration

```python
from t4dm.adapters import T4DMLlamaIndexMemory

# Use as LlamaIndex vector store
vector_store = T4DMLlamaIndexMemory(
    base_url="http://localhost:8765",
    session_id="llamaindex-app",
)

# Works with LlamaIndex query engine
from llama_index.core import VectorStoreIndex
index = VectorStoreIndex.from_vector_store(vector_store)
```

### 3. AutoGen Integration

```python
from t4dm.adapters import T4DMAutoGenMemory

# Use as AutoGen agent memory
memory = T4DMAutoGenMemory(session_id="autogen-agents")

# Attach to AutoGen agent
agent = ConversableAgent("assistant", memory=memory)
```

### 4. CrewAI Integration

```python
from t4dm.adapters import T4DMCrewAIMemory

# Use as CrewAI memory backend
memory = T4DMCrewAIMemory(session_id="crew-tasks")

# Attach to CrewAI crew
crew = Crew(memory=memory)
```

### 5. MCP (Model Context Protocol)

For Claude Code or Claude Desktop integration:

```json
{
  "mcpServers": {
    "t4dm": {
      "command": "t4dm",
      "args": ["mcp", "server"],
      "env": {
        "T4DM_SESSION_ID": "${INSTANCE_ID}"
      }
    }
  }
}
```

---

## Core Concepts

### κ (Kappa) Gradient

T4DM uses a continuous κ value [0,1] to represent memory consolidation:

| κ Range | State | Description |
|---------|-------|-------------|
| 0.0-0.1 | Raw | Just encoded, volatile |
| 0.1-0.3 | Replayed | NREM-strengthened |
| 0.3-0.6 | Transitional | Being abstracted |
| 0.6-0.9 | Semantic | Consolidated concept |
| 0.9-1.0 | Stable | Permanent knowledge |

Query by κ band:
```python
# Get only consolidated memories
results = client.search(query, kappa_min=0.6)
```

### Neuromodulator Orchestra

Control memory dynamics via neuromodulator levels:

| Neuromodulator | Effect | Use Case |
|----------------|--------|----------|
| Dopamine (DA) | Reward gating | Positive outcomes |
| Norepinephrine (NE) | Attention boost | Important events |
| Acetylcholine (ACh) | Learning rate | Novel information |
| Serotonin (5-HT) | Mood baseline | Emotional context |

```python
# Store with elevated dopamine (rewarding event)
client.create_episode(
    content="User achieved goal",
    neuromod_override={"dopamine": 0.9}
)
```

### Sleep Consolidation

Trigger memory consolidation phases:

```python
# Light consolidation (NREM-like)
client.consolidate(phase="nrem")

# Deep consolidation (REM-like, creates semantic prototypes)
client.consolidate(phase="rem")

# Full sleep cycle
client.consolidate(phase="full_cycle")
```

---

## API Endpoints Reference

### Memory Operations

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/episodes/` | POST | Create episodic memory |
| `/api/v1/episodes/{id}` | GET | Retrieve specific episode |
| `/api/v1/episodes/search` | POST | Search episodes by query |
| `/api/v1/episodes/recent` | GET | Get recent episodes |
| `/api/v1/entities/` | POST | Create semantic entity |
| `/api/v1/entities/{id}/activate` | POST | Spread activation |
| `/api/v1/skills/` | POST | Create procedural skill |

### Visualization

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/viz/kappa/distribution` | GET | κ distribution across memories |
| `/api/v1/viz/kappa/flow` | GET | κ changes over time |
| `/api/v1/viz/t4dx/storage` | GET | LSM storage metrics |
| `/api/v1/viz/spiking/dynamics` | GET | Spike rasters and potentials |
| `/api/v1/viz/oscillator/phase` | GET | Theta/gamma/delta phases |
| `/api/v1/viz/energy/landscape` | GET | Hopfield energy surface |
| `/api/v1/viz/realtime/metrics` | GET | Aggregated live metrics |
| `/api/v1/viz/all/export` | GET | Export all viz data |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/stats` | GET | Memory statistics |
| `/api/v1/consolidate` | POST | Trigger consolidation |
| `/api/v1/viz/bio/sleep` | GET | Consolidation status |
| `/api/v1/viz/bio/neuromodulators` | GET | Neuromodulator state |

---

## WebSocket Streaming

For real-time updates:

```python
import asyncio
import websockets

async def stream_memories():
    async with websockets.connect(
        "ws://localhost:8765/ws/memory?session_id=my-app"
    ) as ws:
        async for message in ws:
            event = json.loads(message)
            print(f"Memory event: {event['type']}")

asyncio.run(stream_memories())
```

Available WebSocket channels:
- `/ws/memory` - Memory create/update/delete events
- `/ws/learning` - STDP and Hebbian learning events
- `/ws/consolidation` - Consolidation progress
- `/ws/visualization` - Live viz data stream

---

## Configuration

### Environment Variables

```bash
# Server
T4DM_HOST=0.0.0.0
T4DM_PORT=8765
T4DM_DEBUG=false

# Storage
T4DM_STORAGE_PATH=/var/lib/t4dm/data
T4DM_WAL_PATH=/var/lib/t4dm/wal

# Embedding
T4DM_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
T4DM_EMBEDDING_DEVICE=cuda:0

# Rate Limiting
T4DM_RATE_LIMIT=100  # requests per minute

# Session
T4DM_DEFAULT_SESSION=default
```

### Docker Deployment

```yaml
version: '3.8'
services:
  t4dm:
    image: astoreyai/t4dm:latest
    ports:
      - "8765:8765"
    volumes:
      - t4dm_data:/var/lib/t4dm
    environment:
      - T4DM_EMBEDDING_DEVICE=cpu

volumes:
  t4dm_data:
```

---

## Best Practices

### 1. Session Management

Use unique session IDs for multi-tenant isolation:

```python
# Each user/agent gets isolated memory namespace
client = T4DMClient(session_id=f"user-{user_id}")
```

### 2. Batch Operations

For bulk inserts, use batch endpoints:

```python
# Batch insert is 10x faster than individual inserts
client.batch_create_episodes([
    {"content": "Event 1", "context": {}},
    {"content": "Event 2", "context": {}},
    # ...
])
```

### 3. Query Optimization

Limit search scope for better performance:

```python
# Narrow time range
results = client.recall_episodes(
    query="meeting notes",
    time_range={"start": "2026-02-01", "end": "2026-02-05"},
    limit=10,
)

# Specify κ range for targeted retrieval
results = client.recall_episodes(
    query="concepts",
    kappa_min=0.6,  # Only consolidated memories
)
```

### 4. Graceful Degradation

Handle API unavailability:

```python
try:
    result = client.recall_episodes(query)
except ConnectionError:
    # Fall back to local cache or default behavior
    result = get_cached_result(query)
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `429 Too Many Requests` | Rate limit exceeded | Implement backoff, increase limit |
| `503 Service Unavailable` | Server overloaded | Scale horizontally, check resources |
| Slow queries | Large result sets | Add limits, narrow time range |
| Memory growth | No consolidation | Schedule periodic consolidation |

### Logging

Enable debug logging:

```bash
T4DM_LOG_LEVEL=DEBUG t4dm serve
```

### Health Monitoring

```bash
# Check health
curl http://localhost:8765/api/v1/health

# Get detailed stats
curl http://localhost:8765/api/v1/stats
```

---

## Next Steps

- [API Reference](../API.md) - Complete REST API documentation
- [SDK Guide](../SDK.md) - Python SDK details
- [Architecture](../ARCHITECTURE.md) - System design deep-dive
- [Brain Region Mapping](../BRAIN_REGION_MAPPING.md) - Neuroscience foundations
