# T4DM Integration Plan

**Version**: 2.0.0
**Last Updated**: 2026-02-05
**Purpose**: Complete integration roadmap for connecting T4DM with external systems

---

## Table of Contents

1. [Integration Overview](#1-integration-overview)
2. [Integration Patterns](#2-integration-patterns)
3. [Framework Integrations](#3-framework-integrations)
4. [Platform Integrations](#4-platform-integrations)
5. [API Integration](#5-api-integration)
6. [Data Migration](#6-data-migration)
7. [Production Checklist](#7-production-checklist)

---

## 1. Integration Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT APPLICATIONS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ LangChain│ │LlamaIndex│ │ AutoGen  │ │ CrewAI   │ │  Custom  │  │
│  │ Adapter  │ │ Adapter  │ │ Adapter  │ │ Adapter  │ │   App    │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       │            │            │            │            │         │
│       └────────────┴────────────┴────────────┴────────────┘         │
│                                 │                                    │
│  ┌──────────────────────────────┴───────────────────────────────┐   │
│  │                     INTEGRATION LAYER                         │   │
│  ├───────────────────────────────────────────────────────────────┤   │
│  │                                                                │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐          │   │
│  │  │Python   │  │ REST    │  │  MCP    │  │WebSocket│          │   │
│  │  │  SDK    │  │  API    │  │ Server  │  │ Stream  │          │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘          │   │
│  │       │            │            │            │                │   │
│  └───────┴────────────┴────────────┴────────────┴────────────────┘   │
│                                 │                                    │
├─────────────────────────────────┴────────────────────────────────────┤
│                           T4DM CORE                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Memory Layer  │  Spiking Blocks  │  Consolidation  │  Viz     │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                 │                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    T4DX Storage Engine                          │ │
│  │              (LSM + HNSW + Graph + Bitemporal)                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Integration Options Summary

| Method | Use Case | Latency | Complexity |
|--------|----------|---------|------------|
| **Python SDK** | Python apps, direct embedding | <1ms | Low |
| **REST API** | Any language, microservices | 1-5ms | Low |
| **MCP Server** | Claude Code, Claude Desktop | 5-10ms | Low |
| **WebSocket** | Real-time streaming, T4DV | <1ms | Medium |
| **Framework Adapters** | LangChain, LlamaIndex, etc. | 1-5ms | Low |
| **Direct Import** | Maximum performance | <0.1ms | Medium |

---

## 2. Integration Patterns

### Pattern 1: Embedded (Same Process)

Best for: Single-process applications, maximum performance

```python
from t4dm import T4DM
from t4dm.storage.t4dx import T4DXEngine

# Direct embedding - no network overhead
mem = T4DM(storage_path="./data")

# Or with explicit engine access
engine = T4DXEngine(storage_path="./data")
mem = T4DM(engine=engine)

# Use directly
mem.add("User prefers Python")
results = mem.search("programming language preference")
```

### Pattern 2: Client-Server (HTTP)

Best for: Microservices, language-agnostic access

```python
# Server (separate process)
# t4dm serve --port 8765

# Client
from t4dm.sdk import T4DMClient

client = T4DMClient(base_url="http://localhost:8765")
client.create_episode(content="Meeting notes...", context={"meeting": "standup"})
```

### Pattern 3: Sidecar (Same Host, Separate Process)

Best for: Containerized deployments, process isolation

```yaml
# docker-compose.yml
services:
  app:
    image: my-app:latest
    environment:
      - T4DM_URL=http://t4dm:8765
    depends_on:
      - t4dm

  t4dm:
    image: ghcr.io/astoreyai/t4dm:latest
    volumes:
      - t4dm-data:/data
```

### Pattern 4: MCP Tool (AI Assistant)

Best for: Claude Code, Claude Desktop, AI-driven workflows

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "t4dm": {
      "command": "python",
      "args": ["-m", "t4dm.mcp.server"]
    }
  }
}
```

---

## 3. Framework Integrations

### 3.1 LangChain

#### Memory Integration

```python
from t4dm.adapters.langchain import T4DMMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# Create T4DM-backed memory
memory = T4DMMemory(
    url="http://localhost:8765",
    session_id="langchain-demo",
    memory_key="history",
    return_messages=True,
    k=10,  # Retrieve last 10 relevant memories
    kappa_min=0.1,  # Only consolidated memories
)

# Use with LangChain
chain = ConversationChain(
    llm=ChatOpenAI(model="gpt-4"),
    memory=memory,
    verbose=True
)

# Memories are automatically stored and retrieved
response = chain.predict(input="My name is Alice and I work on ML")
response = chain.predict(input="What do you know about me?")
```

#### Retriever Integration

```python
from t4dm.adapters.langchain import T4DMRetriever
from langchain.chains import RetrievalQA

retriever = T4DMRetriever(
    url="http://localhost:8765",
    search_kwargs={
        "k": 10,
        "kappa_min": 0.3,
        "time_decay": True
    }
)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=retriever
)

answer = qa.run("What were the key decisions from last week?")
```

#### Vector Store Integration

```python
from t4dm.adapters.langchain import T4DMVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# Load and split documents
loader = TextLoader("./documents/manual.txt")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
splits = splitter.split_documents(docs)

# Store in T4DM
vectorstore = T4DMVectorStore(url="http://localhost:8765")
vectorstore.add_documents(splits)

# Similarity search
results = vectorstore.similarity_search("installation steps", k=5)
```

### 3.2 LlamaIndex

```python
from t4dm.adapters.llamaindex import T4DMVectorStore
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Create vector store
vector_store = T4DMVectorStore(
    url="http://localhost:8765",
    session_id="llamaindex-demo"
)

# Load documents
documents = SimpleDirectoryReader("./data").load_data()

# Create index
index = VectorStoreIndex.from_documents(
    documents,
    vector_store=vector_store
)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("Summarize the main findings")
print(response)
```

### 3.3 AutoGen

```python
from t4dm.adapters.autogen import T4DMAutoGenMemory
from autogen import ConversableAgent, AssistantAgent, UserProxyAgent

# Create T4DM memory
memory = T4DMAutoGenMemory(
    url="http://localhost:8765",
    session_id="autogen-demo"
)

# Create agents with shared memory
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    memory=memory
)

user_proxy = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    memory=memory
)

# Memory is shared across agents and persisted
user_proxy.initiate_chat(assistant, message="Research quantum computing")
```

### 3.4 CrewAI

```python
from t4dm.adapters.crewai import T4DMCrewMemory
from crewai import Agent, Task, Crew

# Create T4DM memory
memory = T4DMCrewMemory(
    url="http://localhost:8765",
    session_id="crewai-demo"
)

# Create agents
researcher = Agent(
    role="Researcher",
    goal="Find relevant information",
    backstory="Expert at research",
    memory=memory
)

writer = Agent(
    role="Writer",
    goal="Write clear content",
    backstory="Expert technical writer",
    memory=memory
)

# Create tasks
research_task = Task(
    description="Research T4DM architecture",
    agent=researcher
)

write_task = Task(
    description="Write documentation based on research",
    agent=writer
)

# Create crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    memory=memory  # Shared memory across crew
)

result = crew.kickoff()
```

### 3.5 Mem0 Compatibility

T4DM provides a Mem0-compatible API for drop-in replacement:

```python
# Existing Mem0 code
import requests

# Just change the URL - API is compatible
resp = requests.post("http://localhost:8765/v1/memories/", json={
    "messages": [{"role": "user", "content": "Remember this"}],
    "user_id": "user-123"
})

# Search
resp = requests.post("http://localhost:8765/v1/memories/search/", json={
    "query": "what do you remember?",
    "user_id": "user-123"
})
```

---

## 4. Platform Integrations

### 4.1 Claude Code / Claude Desktop (MCP) - Automatic Memory

T4DM provides **automatic memory** - no manual tool calls needed.

#### How It Works

| Feature | Description |
|---------|-------------|
| **Auto-load** | `memory://context` resource injected at session start |
| **Hot cache** | Frequently used memories (0ms latency) |
| **Auto-capture** | Observations extracted from Claude's outputs |
| **Auto-consolidate** | Session summaries stored at end |

#### Setup

1. **Start T4DM API server:**
   ```bash
   t4dm serve --port 8765
   ```

2. **Configure Claude Code** (`~/.claude/settings.json`):
   ```json
   {
     "mcpServers": {
       "t4dm-memory": {
         "command": "python",
         "args": ["-m", "t4dm.mcp.server"],
         "env": {
           "T4DM_API_URL": "http://localhost:8765",
           "T4DM_PROJECT": "my-project",
           "T4DM_HOT_CACHE_SIZE": "10"
         }
       }
     }
   }
   ```

3. **Restart Claude Code** - memory is now automatic.

#### MCP Resources (Auto-Loaded)

| Resource | Description |
|----------|-------------|
| `memory://context` | Session context (hot cache + project + recent) |
| `memory://hot-cache` | Frequently accessed memories |
| `memory://project/{name}` | Project-specific patterns |

#### Manual Tool (Only When Asked)

| Tool | Use Case |
|------|----------|
| `t4dm_remember` | User says "remember this" or "save this" |

#### Usage in Claude

Memory is automatic - Claude has context without visible tool calls:

```
Human: How do we handle auth?

Claude: We use JWT with refresh tokens. The auth middleware validates
tokens on each request.
[No tool call visible - context was pre-loaded]
```

See [MCP_INTEGRATION.md](MCP_INTEGRATION.md) for full documentation.

### 4.2 Kymera Platform

T4DM serves as the memory backend for Kymera agents:

```python
# Kymera agent configuration
agent_config = {
    "name": "research-agent",
    "memory": {
        "backend": "t4dm",
        "url": "http://localhost:8765",
        "session_id": "kymera-research"
    }
}
```

### 4.3 T4DV Visualization

Real-time visualization of memory state:

```python
from t4dm.api.routes.ws_viz import create_viz_connection

# Connect visualization client
async with create_viz_connection("ws://localhost:8765/ws/viz") as ws:
    # Subscribe to memory events
    await ws.subscribe(["memory.store", "memory.recall", "consolidation.*"])

    async for event in ws:
        if event.type == "memory.store":
            render_new_memory(event.data)
        elif event.type == "consolidation.progress":
            update_kappa_heatmap(event.data)
```

---

## 5. API Integration

### 5.1 REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/episodes/` | POST | Create episode |
| `/api/v1/episodes/{id}` | GET | Get episode by ID |
| `/api/v1/episodes/search` | GET | Search episodes |
| `/api/v1/entities/` | POST | Create entity |
| `/api/v1/entities/{id}` | GET | Get entity |
| `/api/v1/entities/search` | GET | Search entities |
| `/api/v1/memory/store` | POST | Generic store |
| `/api/v1/memory/search` | POST | Generic search |
| `/api/v1/consolidation/run` | POST | Trigger consolidation |
| `/api/v1/consolidation/status` | GET | Consolidation status |
| `/v1/memories/` | POST | Mem0-compatible store |
| `/v1/memories/search/` | POST | Mem0-compatible search |

### 5.2 Authentication

```python
# API Key authentication
import httpx

client = httpx.Client(
    base_url="http://localhost:8765",
    headers={"X-API-Key": "your-api-key"}
)

# Or session-based
client = httpx.Client(
    base_url="http://localhost:8765",
    headers={"X-Session-ID": "my-session"}
)
```

### 5.3 Rate Limiting

Default limits (configurable):
- 1000 requests/minute per IP
- 100 requests/minute per session for writes
- 500 requests/minute per session for reads

```python
# Handle rate limits
import time

response = client.post("/api/v1/memory/store", json=data)
if response.status_code == 429:
    retry_after = int(response.headers.get("Retry-After", 60))
    time.sleep(retry_after)
    response = client.post("/api/v1/memory/store", json=data)
```

### 5.4 Error Handling

```python
from t4dm.sdk import T4DMClient, T4DMError, RateLimitError, NotFoundError

client = T4DMClient(base_url="http://localhost:8765")

try:
    result = client.get_episode("nonexistent-id")
except NotFoundError:
    print("Episode not found")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except T4DMError as e:
    print(f"API error: {e.message}")
```

---

## 6. Data Migration

### 6.1 From Mem0

```python
import requests
from t4dm import T4DM

# Export from Mem0
mem0_memories = requests.get(
    "https://api.mem0.ai/v1/memories/",
    headers={"Authorization": f"Bearer {MEM0_API_KEY}"}
).json()

# Import to T4DM
mem = T4DM()
for memory in mem0_memories["memories"]:
    mem.add(
        content=memory["content"],
        metadata=memory.get("metadata", {}),
        timestamp=memory.get("created_at")
    )
```

### 6.2 From LangChain ChatMessageHistory

```python
from langchain.memory import ChatMessageHistory
from t4dm import T4DM

# Existing LangChain history
history = ChatMessageHistory()

# Migrate to T4DM
mem = T4DM()
for message in history.messages:
    mem.add(
        content=message.content,
        metadata={
            "role": message.type,
            "source": "langchain_migration"
        }
    )
```

### 6.3 From Vector Databases

#### From Qdrant

```python
from qdrant_client import QdrantClient
from t4dm.storage.t4dx import T4DXEngine

qdrant = QdrantClient("localhost", port=6333)
t4dx = T4DXEngine(storage_path="./data")

# Scroll through Qdrant collection
offset = None
while True:
    records, offset = qdrant.scroll(
        collection_name="memories",
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=True
    )

    if not records:
        break

    for record in records:
        t4dx.insert(
            id=str(record.id),
            vector=record.vector,
            content=record.payload.get("content", ""),
            metadata=record.payload,
            kappa=0.5  # Midpoint consolidation
        )
```

#### From Pinecone

```python
import pinecone
from t4dm.storage.t4dx import T4DXEngine

pinecone.init(api_key="your-key", environment="us-west1-gcp")
index = pinecone.Index("memories")
t4dx = T4DXEngine(storage_path="./data")

# Fetch and migrate
for ids in batch_ids(index.describe_index_stats()["namespaces"]):
    results = index.fetch(ids=ids)
    for id, data in results["vectors"].items():
        t4dx.insert(
            id=id,
            vector=data["values"],
            content=data["metadata"].get("content", ""),
            metadata=data["metadata"],
            kappa=0.5
        )
```

### 6.4 Bulk Import

```python
from t4dm.storage.t4dx import T4DXEngine
import json

engine = T4DXEngine(storage_path="./data")

# Load from JSONL
with open("memories.jsonl") as f:
    batch = []
    for line in f:
        item = json.loads(line)
        batch.append(item)

        if len(batch) >= 1000:
            engine.batch_insert(batch)
            batch = []

    if batch:
        engine.batch_insert(batch)

# Force flush
engine.flush()
```

---

## 7. Production Checklist

### Pre-Deployment

- [ ] **Hardware sizing** - Verify RAM, VRAM, disk meet requirements
- [ ] **Configuration reviewed** - All settings appropriate for production
- [ ] **Storage path** - Persistent volume mounted and writable
- [ ] **Backup strategy** - WAL and segment backup configured
- [ ] **Monitoring** - Metrics endpoint exposed, dashboards ready
- [ ] **Logging** - Structured logging to aggregator
- [ ] **Security** - API keys, TLS, network isolation

### Integration Testing

- [ ] **Smoke test** - Basic store/recall works
- [ ] **Load test** - Performance under expected load
- [ ] **Failure test** - Graceful degradation verified
- [ ] **Recovery test** - WAL replay, segment recovery tested
- [ ] **Framework adapters** - All used frameworks tested

### Monitoring

```yaml
# Prometheus scrape config
scrape_configs:
  - job_name: 't4dm'
    static_configs:
      - targets: ['localhost:8765']
    metrics_path: '/metrics'
```

Key metrics to monitor:
- `t4dm_memory_items_total` - Total items stored
- `t4dm_search_latency_seconds` - Search latency histogram
- `t4dm_consolidation_duration_seconds` - Consolidation time
- `t4dm_wal_size_bytes` - WAL size (alert if growing unbounded)
- `t4dm_segment_count` - Segment count

### Alerting

```yaml
# Example Prometheus alerts
groups:
  - name: t4dm
    rules:
      - alert: T4DMHighLatency
        expr: histogram_quantile(0.95, t4dm_search_latency_seconds) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "T4DM search latency high"

      - alert: T4DMWALGrowing
        expr: t4dm_wal_size_bytes > 1073741824  # 1GB
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "T4DM WAL size exceeding 1GB"
```

### Backup

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR=/backups/t4dm/$(date +%Y%m%d)
mkdir -p $BACKUP_DIR

# Pause writes (optional, for consistency)
curl -X POST http://localhost:8765/api/v1/admin/pause

# Copy data
cp -r /var/lib/t4dm/data/* $BACKUP_DIR/

# Resume writes
curl -X POST http://localhost:8765/api/v1/admin/resume

# Compress
tar -czf $BACKUP_DIR.tar.gz $BACKUP_DIR
rm -rf $BACKUP_DIR

# Retention (keep 30 days)
find /backups/t4dm -name "*.tar.gz" -mtime +30 -delete
```

---

## Quick Reference

### SDK Methods

```python
from t4dm import T4DM

mem = T4DM()

# Core operations
mem.add(content, metadata=None, importance=0.5)
mem.search(query, k=10, kappa_min=0.0, time_range=None)
mem.get(id)
mem.delete(id)

# Consolidation
mem.consolidate(phase="all")  # "nrem", "rem", "prune", "all"

# Status
mem.status()
mem.close()
```

### Environment Variables

```bash
T4DM_STORAGE_PATH=/var/lib/t4dm/data
T4DM_HOST=0.0.0.0
T4DM_PORT=8765
T4DM_LOG_LEVEL=INFO
T4DM_EMBEDDING_MODEL=BAAI/bge-m3
T4DM_DEVICE=auto
```

### Common curl Commands

```bash
# Health
curl http://localhost:8765/api/v1/health

# Store
curl -X POST http://localhost:8765/api/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "Remember this", "metadata": {"tag": "important"}}'

# Search
curl -X POST http://localhost:8765/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "what to remember", "k": 5}'

# Consolidate
curl -X POST http://localhost:8765/api/v1/consolidation/run
```

---

*Generated 2026-02-05*