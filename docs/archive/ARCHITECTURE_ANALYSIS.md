# T4DM Architecture Analysis

**Date**: 2025-12-03
**Purpose**: Standalone, Dockerized, Programmatic API Memory System

---

## Executive Summary

T4DM is a **well-architected tripartite memory system** with solid core components but missing the glue to make it truly standalone. The foundations are strong (93%+ coverage on memory systems), but it currently requires MCP protocol and external orchestration.

### Current State: 7/10
- Core memory systems: **Working**
- Storage layer: **Working** (requires Docker services)
- MCP server: **Broken** (missing `mcp`/`fastmcp` packages)
- Programmatic API: **Partial** (tied to MCP)
- Docker infrastructure: **Good** (Neo4j + Qdrant)
- Test coverage: **79%** (793/814 pass when infra available)

---

## What Works

### 1. Storage Layer (Neo4j + Qdrant)
**Location**: `src/t4dm/storage/`

| Component | Status | Coverage | Notes |
|-----------|--------|----------|-------|
| `t4dx_graph_adapter.py` | Working | 63% | Graph operations, APOC support |
| `t4dx_vector_adapter.py` | Working | 62% | Vector operations, scroll, batch |
| `saga.py` | Working | 97% | Dual-store atomicity |

**Strengths**:
- Saga pattern for cross-store transactions
- Connection pooling
- Health checks
- Proper async/await

**Issues**:
- Requires running Docker services
- No embedded/in-memory fallback for testing

### 2. Memory Systems
**Location**: `src/t4dm/memory/`

| Component | Status | Coverage | Notes |
|-----------|--------|----------|-------|
| `episodic.py` | Working | 93% | FSRS decay, recency scoring |
| `semantic.py` | Working | 94% | ACT-R activation, Hebbian learning |
| `procedural.py` | Working | 98% | Skill patterns, execution tracking |
| `unified.py` | Not tested | 0% | Cross-memory orchestration |

**Strengths**:
- Cognitively-inspired algorithms (FSRS, ACT-R, Hebbian)
- Weighted multi-signal retrieval
- Bi-temporal versioning support
- Proper embedding integration

### 3. Embedding System
**Location**: `src/t4dm/embedding/bge_m3.py`

| Feature | Status |
|---------|--------|
| BGE-M3 model loading | Working |
| TTL-based caching | Working |
| Sentence-transformers fallback | Working |
| FP16 inference | Working |

**Strengths**:
- Lazy initialization (no GPU until first use)
- Thread-safe cache with TTL
- Multiple backend support

### 4. Configuration
**Location**: `src/t4dm/core/config.py`

**Strengths**:
- 100+ configurable parameters
- Environment variable support
- Password strength validation
- Weight validation (must sum to 1.0)

---

## What Doesn't Work

### 1. MCP Server
**Location**: `src/t4dm/mcp/`

**Problem**: Package imports fail
```
ModuleNotFoundError: No module named 'mcp.server'
```

**Root Cause**: `mcp>=1.0.0` and `fastmcp>=0.4.1` in pyproject.toml but not installed in current environment.

**Impact**:
- All 17 MCP tools unusable
- No external API surface
- 9 test files fail to import

### 2. Consolidation/Clustering
**Location**: `src/t4dm/consolidation/`

**Problem**: HDBSCAN not installed
```
ImportError: HDBSCAN not found
```

**Root Cause**: `hdbscan` is optional dependency (`[consolidation]`) but tests don't skip gracefully.

**Impact**: Memory consolidation (merging similar episodes into entities) doesn't work.

### 3. Entity Extraction
**Location**: `src/t4dm/extraction/`

**Problem**: Tests fail when LLM not configured

**Impact**: Auto-extraction from episodes disabled without OpenAI key.

### 4. Unified Memory
**Location**: `src/t4dm/memory/unified.py`

**Problem**: 0% test coverage, never exercised

**Impact**: Cross-memory queries and consolidation untested.

---

## Why It's Not Standalone

### Current Architecture
```
Claude Code ─────► MCP Protocol ─────► T4DM
                    (stdio/SSE)           │
                                          ▼
                               ┌──────────┴──────────┐
                               │  Docker Services    │
                               │  - Neo4j (7687)     │
                               │  - Qdrant (6333)    │
                               └─────────────────────┘
```

### Problems

1. **MCP Dependency**: No way to call memory functions without MCP client
2. **No REST API**: Can't call from other services
3. **No Python Client**: Must go through MCP protocol
4. **Docker Required**: No in-memory mode for simple deployments
5. **No SDK**: Other apps can't import and use WW as library

---

## Gap Analysis

| Capability | Status | Priority | Effort |
|------------|--------|----------|--------|
| REST/HTTP API | Missing | High | Medium |
| Python SDK | Partial | High | Low |
| Docker-compose fix | Needs env | High | Low |
| MCP package install | Broken | High | Low |
| In-memory mode | Missing | Medium | Medium |
| gRPC API | Missing | Medium | High |
| Async client | Missing | Medium | Medium |
| CLI interface | Missing | Low | Medium |

---

## Recommended Architecture: Standalone + Dockerized

### Target Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     T4DM Server                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  REST API   │  │  MCP Server │  │  Python SDK        │ │
│  │  (FastAPI)  │  │  (FastMCP)  │  │  (Direct Import)   │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                     │            │
│         ▼                ▼                     ▼            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Core Memory Gateway                      │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │  │
│  │  │ Episodic │  │ Semantic │  │    Procedural    │   │  │
│  │  └──────────┘  └──────────┘  └──────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
│                          │                                  │
│         ┌────────────────┼────────────────┐                │
│         ▼                ▼                ▼                │
│  ┌──────────┐     ┌──────────┐     ┌──────────────┐       │
│  │ BGE-M3   │     │ Neo4j    │     │    Qdrant    │       │
│  │ Embedder │     │ (Graph)  │     │   (Vector)   │       │
│  └──────────┘     └──────────┘     └──────────────┘       │
└─────────────────────────────────────────────────────────────┘

Docker Compose:
  - ww-server (Python 3.11)
  - ww-neo4j (Neo4j 5)
  - ww-qdrant (Qdrant)
  - ww-embedder (optional, GPU)
```

---

## Implementation Plan

### Phase 1: Fix What's Broken (1 day)

1. **Install missing packages**
   ```bash
   pip install mcp>=1.0.0 fastmcp>=0.4.1 hdbscan>=0.8.33
   ```

2. **Fix docker-compose env**
   ```bash
   cp .env.example .env
   # Set NEO4J_PASSWORD
   docker-compose up -d
   ```

3. **Verify MCP server starts**
   ```bash
   python -m t4dm.mcp.server
   ```

### Phase 2: Add REST API (2-3 days)

Create `src/t4dm/api/` with FastAPI:

```python
# src/t4dm/api/server.py
from fastapi import FastAPI
from t4dm.memory.episodic import get_episodic_memory
from t4dm.memory.semantic import get_semantic_memory
from t4dm.memory.procedural import get_procedural_memory

app = FastAPI(title="T4DM API")

@app.post("/v1/episodes")
async def create_episode(content: str, context: dict = None):
    episodic = get_episodic_memory()
    await episodic.initialize()
    return await episodic.create(content, context)

@app.get("/v1/episodes/search")
async def search_episodes(query: str, limit: int = 10):
    episodic = get_episodic_memory()
    await episodic.initialize()
    return await episodic.recall(query, limit)

# Similar for semantic, procedural
```

### Phase 3: Create Python SDK (1 day)

```python
# src/t4dm/client.py
class T4DMClient:
    """Standalone Python client for T4DM."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        qdrant_url: str = "http://localhost:6333",
        session_id: str = "default",
    ):
        self.session_id = session_id
        self._episodic = None
        self._semantic = None
        self._procedural = None

    async def connect(self):
        """Initialize all memory systems."""
        self._episodic = get_episodic_memory(self.session_id)
        self._semantic = get_semantic_memory(self.session_id)
        self._procedural = get_procedural_memory(self.session_id)

        await self._episodic.initialize()
        await self._semantic.initialize()
        await self._procedural.initialize()

    async def store(self, content: str, **kwargs) -> str:
        """Store an episode."""
        episode = await self._episodic.create(content, **kwargs)
        return str(episode.id)

    async def recall(self, query: str, limit: int = 10) -> list:
        """Recall episodes."""
        results = await self._episodic.recall(query, limit)
        return [{"id": str(r.item.id), "content": r.item.content, "score": r.score} for r in results]

    async def close(self):
        """Clean up connections."""
        from t4dm.storage.t4dx_graph_adapter import close_t4dx_graph_adapter
        from t4dm.storage.t4dx_vector_adapter import close_t4dx_vector_adapter
        await close_t4dx_graph_adapter()
        await close_t4dx_vector_adapter()

# Usage:
# async with T4DMClient() as ww:
#     await ww.store("Fixed critical bug in parser")
#     memories = await ww.recall("parser issues")
```

### Phase 4: Dockerize Everything (1 day)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install .[consolidation]

# Copy source
COPY src/ src/

# Expose ports
EXPOSE 8765  # REST API
EXPOSE 8766  # MCP Server

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
  CMD curl -f http://localhost:8765/health || exit 1

CMD ["python", "-m", "t4dm.api.server"]
```

```yaml
# docker-compose.full.yml
version: '3.8'

services:
  ww-server:
    build: .
    container_name: ww-server
    ports:
      - "8765:8765"  # REST API
      - "8766:8766"  # MCP
    environment:
      - T4DM_NEO4J_URI=bolt://neo4j:7687
      - T4DM_QDRANT_URL=http://qdrant:6333
      - T4DM_NEO4J_PASSWORD=${NEO4J_PASSWORD}
    depends_on:
      neo4j:
        condition: service_healthy
      qdrant:
        condition: service_healthy

  neo4j:
    image: neo4j:5-community
    # ... existing config

  qdrant:
    image: qdrant/qdrant:latest
    # ... existing config

  # Optional: Separate GPU embedder
  embedder:
    image: ww-embedder:latest
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
```

### Phase 5: Add Health & Metrics (1 day)

Already have `src/t4dm/observability/` but need to:
1. Add `/health` endpoint
2. Add `/metrics` endpoint (Prometheus format)
3. Enable OpenTelemetry by default

---

## Quick Start (After Implementation)

### Option 1: Docker Compose (Recommended)
```bash
# Clone and start
git clone https://github.com/astoreyai/t4dm
cd t4dm
cp .env.example .env
# Edit .env with NEO4J_PASSWORD

docker-compose -f docker-compose.full.yml up -d

# Test
curl http://localhost:8765/health
```

### Option 2: Python SDK
```python
from ww import T4DMClient

async def main():
    async with T4DMClient() as ww:
        # Store
        episode_id = await ww.store(
            "Implemented new feature X",
            context={"project": "myapp"},
            outcome="success"
        )

        # Recall
        memories = await ww.recall("feature X", limit=5)
        for m in memories:
            print(f"[{m['score']:.2f}] {m['content']}")

asyncio.run(main())
```

### Option 3: REST API
```bash
# Store
curl -X POST http://localhost:8765/v1/episodes \
  -H "Content-Type: application/json" \
  -d '{"content": "Fixed bug in parser", "outcome": "success"}'

# Recall
curl "http://localhost:8765/v1/episodes/search?query=parser&limit=5"
```

### Option 4: MCP Protocol (Claude Code)
```json
// claude_desktop_config.json
{
  "mcpServers": {
    "ww-memory": {
      "command": "docker",
      "args": ["exec", "-i", "ww-server", "python", "-m", "t4dm.mcp.server"]
    }
  }
}
```

---

## Conclusion

**T4DM has excellent bones** - cognitively-inspired memory systems, proper dual-store architecture, and comprehensive configuration. The main issues are:

1. **Missing packages** - Easy fix
2. **No standalone API** - 2-3 days to add REST
3. **MCP-only access** - Need Python SDK

**Recommendation**: Implement Phases 1-4 to create a truly standalone, dockerized memory system with programmatic API access.

**Estimated Effort**: 5-7 days total
