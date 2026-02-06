# T4DM Integration Guide

**Version**: 1.0 | **Date**: 2026-02-02

How to integrate T4DM into your agent framework or application.

---

## 1. Simple API (3 Lines)

The fastest way to use T4DM:

```python
from t4dm import T4DM

mem = T4DM()
mem.add("The user prefers dark mode and Python 3.12")
results = mem.search("what are the user's preferences?")
```

The `T4DM` class handles embedding, storage, consolidation, and retrieval internally.

```python
# Store with metadata
mem.add(
    "Meeting with Alice about project Alpha",
    metadata={"session": "meeting-2026-02-01", "importance": 0.8}
)

# Search with time constraints
from datetime import datetime, timedelta
results = mem.search(
    "project Alpha status",
    time_range=(datetime.now() - timedelta(days=7), datetime.now()),
    k=5
)

# Get all memories for a session
memories = mem.get_all(session="meeting-2026-02-01")
```

---

## 2. LangChain Integration

### T4DMMemory (BaseMemory)

```python
from t4dm.adapters.langchain import T4DMMemory

memory = T4DMMemory(
    url="http://localhost:8000",  # or embedded mode
    session_id="my-agent",
    memory_key="history"
)

# Use with LangChain chains
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

chain = ConversationChain(
    llm=ChatOpenAI(),
    memory=memory
)

response = chain.predict(input="Remember that I like Python")
response = chain.predict(input="What language do I like?")
```

### T4DMRetriever (BaseRetriever)

```python
from t4dm.adapters.langchain import T4DMRetriever

retriever = T4DMRetriever(
    url="http://localhost:8000",
    search_kwargs={"k": 10, "kappa_min": 0.3}  # Only consolidated memories
)

# Use with RetrievalQA
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=retriever
)
```

---

## 3. LlamaIndex Integration

### T4DMVectorStore

```python
from t4dm.adapters.llamaindex import T4DMVectorStore
from llama_index.core import VectorStoreIndex

vector_store = T4DMVectorStore(url="http://localhost:8000")

# Build index from documents
index = VectorStoreIndex.from_vector_store(vector_store)

# Query
query_engine = index.as_query_engine()
response = query_engine.query("What did we discuss last week?")
```

---

## 4. AutoGen Integration

### T4DMAutoGenMemory

```python
from t4dm.adapters.autogen import T4DMAutoGenMemory
from autogen import ConversableAgent

memory = T4DMAutoGenMemory(url="http://localhost:8000")

agent = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    memory=memory
)

# Memory is automatically populated during conversations
# and used for context in subsequent turns
```

---

## 5. CrewAI Integration

### T4DMCrewMemory

```python
from t4dm.adapters.crewai import T4DMCrewMemory
from crewai import Agent, Crew

memory = T4DMCrewMemory(url="http://localhost:8000")

agent = Agent(
    role="Researcher",
    goal="Find relevant information",
    memory=memory
)

crew = Crew(agents=[agent], memory=memory)
result = crew.kickoff()
```

---

## 6. MCP Server (Claude Code / Claude Desktop)

T4DM provides **automatic memory** - no manual tool calls needed.

### Setup

1. **Start T4DM API server:**
   ```bash
   t4dm serve --port 8765
   ```

2. **Configure Claude Code** (add to `~/.claude/settings.json`):
   ```json
   {
     "mcpServers": {
       "t4dm-memory": {
         "command": "python",
         "args": ["-m", "t4dm.mcp.server"],
         "env": {
           "T4DM_API_URL": "http://localhost:8765",
           "T4DM_PROJECT": "my-project"
         }
       }
     }
   }
   ```

3. **Restart Claude Code** - memory is now automatic.

### How It Works

| Feature | Description |
|---------|-------------|
| **Auto-load** | Session context pre-loaded via `memory://context` resource |
| **Auto-capture** | Observations extracted from Claude's outputs |
| **Auto-consolidate** | Session summaries stored at end |
| **Hot cache** | Frequently used memories (0ms latency) |

### MCP Resources (Auto-Loaded)

| Resource | Description |
|----------|-------------|
| `memory://context` | Session context (hot cache + project + recent) |
| `memory://hot-cache` | Frequently accessed memories |
| `memory://project/{name}` | Project-specific patterns |

### Manual Tool (Only When Asked)

| Tool | Use Case |
|------|----------|
| `t4dm_remember` | User says "remember this" or "save this" |

### Claude Code Usage

Memory is automatic - Claude has context without visible tool calls:

```
Human: How do we handle auth?
Claude: We use JWT with refresh tokens. [no tool call visible]

---

## 7. REST API Direct Usage

The T4DM REST API runs on FastAPI:

```bash
# Start the server
t4dm serve --port 8000
```

### Store a memory

```bash
curl -X POST http://localhost:8000/api/v1/memory/store \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The deployment uses Kubernetes on AWS EKS",
    "metadata": {"importance": 0.7, "session": "infra-review"}
  }'
```

### Search memories

```bash
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "deployment infrastructure",
    "k": 5,
    "kappa_min": 0.0,
    "kappa_max": 1.0
  }'
```

### Get memory by ID

```bash
curl http://localhost:8000/api/v1/memory/{memory_id}
```

### Trigger consolidation

```bash
curl -X POST http://localhost:8000/api/v1/consolidation/run
```

### System status

```bash
curl http://localhost:8000/api/v1/status
```

---

## 8. Hooks / Plugin System

T4DM provides lifecycle hooks for custom extensions:

```python
from t4dm.hooks import on_store, on_recall, on_consolidate

@on_store
def my_store_hook(memory_item):
    """Called after every memory store operation."""
    print(f"Stored: {memory_item.id} with kappa={memory_item.kappa}")

@on_recall
def my_recall_hook(query, results):
    """Called after every recall operation."""
    print(f"Query '{query}' returned {len(results)} results")

@on_consolidate
def my_consolidate_hook(phase, stats):
    """Called after each consolidation phase (NREM/REM/Prune)."""
    print(f"{phase}: {stats['items_processed']} items processed")
```

### Custom scoring plugin

```python
from t4dm.hooks import register_scorer

@register_scorer(name="my_scorer", weight=0.2)
def custom_relevance(query_embedding, item):
    """Add a custom scoring dimension to retrieval."""
    # Your custom logic here
    return score  # float in [0, 1]
```

### Custom consolidation plugin

```python
from t4dm.hooks import register_consolidation_phase

@register_consolidation_phase(name="custom_merge", after="rem")
def custom_merge_phase(engine, items):
    """Run custom logic during consolidation."""
    # Access T4DX engine directly
    for item in items:
        if should_merge(item):
            engine.update_fields(item.id, {"kappa": item.kappa + 0.1})
```

---

## Configuration

All integrations share common configuration via environment variables or config file:

| Variable | Default | Description |
|----------|---------|-------------|
| `T4DM_DATA_DIR` | `./data` | Data directory for T4DX segments |
| `T4DM_HOST` | `localhost` | API server host |
| `T4DM_PORT` | `8000` | API server port |
| `T4DM_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model name |
| `T4DM_CHECKPOINT_INTERVAL` | `300` | Checkpoint interval (seconds) |
| `T4DM_CONSOLIDATION_INTERVAL` | `3600` | Auto-consolidation interval (seconds) |
| `T4DM_LOG_LEVEL` | `INFO` | Logging level |

---

*Generated 2026-02-02*