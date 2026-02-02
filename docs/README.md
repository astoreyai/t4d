# T4DM Documentation

**Version**: 3.1.0 | **Last Updated**: 2025-12-09 | **Tests**: 4043 passed, 79% coverage

Welcome to the T4DM documentation. This directory contains comprehensive guides for using, deploying, and developing with the T4DM tripartite memory system.

## Quick Navigation

### For Users

- **[API Reference](API.md)** - Complete reference for all 17 MCP tools
  - Episodic memory (4 tools)
  - Semantic memory (5 tools)
  - Procedural memory (4 tools)
  - Consolidation & metadata (2 tools)
  - Utilities (2 tools)

### For Operators

- **[Deployment Guide](DEPLOYMENT.md)** - Production deployment and operations
  - Quick start with Docker Compose
  - Security hardening
  - Resource optimization
  - Monitoring & logging
  - Backup & recovery
  - Troubleshooting

### For Developers

- **[Architecture](architecture.md)** - System design and cognitive science foundations
  - Tripartite memory model
  - ACT-R activation
  - FSRS retrievability
  - Hebbian learning
  - Bi-temporal versioning

- **[Algorithms](algorithms.md)** - Core algorithms and implementation details
  - Scoring functions
  - Consolidation strategies
  - Graph traversal
  - Embedding techniques

### Deep Dive Walkthroughs

- **[System Walkthrough](SYSTEM_WALKTHROUGH.md)** - Complete system overview, 136 files across 19 modules
- **[Memory Store/Recall Flow](MEMORY_STORE_RECALL_FLOW.md)** - Step-by-step data flow traces
- **[Neuromodulation Walkthrough](NEUROMODULATION_WALKTHROUGH.md)** - 5-factor neuromodulator system
- **[Learning System Walkthrough](LEARNING_SYSTEM_WALKTHROUGH.md)** - Plasticity and credit assignment
- **[API Walkthrough](API_WALKTHROUGH.md)** - REST API and MCP interface deep dive
- **[Visualization Walkthrough](VISUALIZATION_WALKTHROUGH.md)** - Neural dynamics visualization
- **[Persistence Architecture](PERSISTENCE_ARCHITECTURE.md)** - WAL, checkpoints, crash recovery
- **[Observability Walkthrough](OBSERVABILITY_WALKTHROUGH.md)** - Logging, metrics, health, tracing

## Getting Started

### Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/astoreyai/t4dm.git
cd t4dm

# Start infrastructure
docker-compose up -d

# Install Python package
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

### Configuration

Edit `.env` with your settings:

```bash
T4DM_SESSION_ID=my-session
T4DM_NEO4J_PASSWORD=your-secure-password
T4DM_QDRANT_URL=http://localhost:6333
```

### Usage

Configure Claude Code to use T4DM MCP server:

**`~/.config/claude-code/mcp_servers.json`**:
```json
{
  "mcpServers": {
    "t4dm": {
      "command": "python",
      "args": ["-m", "t4dm.mcp.memory_gateway"],
      "cwd": "/path/to/t4dm",
      "env": {
        "T4DM_SESSION_ID": "my-session"
      }
    }
  }
}
```

## Documentation Overview

| Document | Purpose | Audience |
|----------|---------|----------|
| [API.md](API.md) | Tool reference with examples | Users, integrators |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Production deployment | Operators, sysadmins |
| [architecture.md](architecture.md) | System design | Developers, architects |
| [algorithms.md](algorithms.md) | Implementation details | Developers |
| [SYSTEM_WALKTHROUGH.md](SYSTEM_WALKTHROUGH.md) | System architecture deep dive | Developers, architects |
| [MEMORY_STORE_RECALL_FLOW.md](MEMORY_STORE_RECALL_FLOW.md) | Data flow traces | Developers |
| [NEUROMODULATION_WALKTHROUGH.md](NEUROMODULATION_WALKTHROUGH.md) | Neuromodulator systems | Researchers, developers |
| [LEARNING_SYSTEM_WALKTHROUGH.md](LEARNING_SYSTEM_WALKTHROUGH.md) | Plasticity mechanisms | Researchers, developers |
| [PERSISTENCE_ARCHITECTURE.md](PERSISTENCE_ARCHITECTURE.md) | Crash recovery | Operators, developers |
| [OBSERVABILITY_WALKTHROUGH.md](OBSERVABILITY_WALKTHROUGH.md) | Monitoring & metrics | Operators, developers |

## Key Features

### Episodic Memory
- Autobiographical events with temporal-spatial context
- ACT-R activation decay
- FSRS retrievability modeling
- Bi-temporal queries ("what did we know at time T?")

### Semantic Memory
- Hebbian-weighted knowledge graph
- Entity types: CONCEPT, PERSON, PROJECT, TOOL, TECHNIQUE, FACT
- Spreading activation through relationships
- Bi-temporal versioning for facts

### Procedural Memory
- Learns from successful trajectories (>= 0.7 success threshold)
- Memp (Memory for Procedures) implementation
- Domain-specific skills: coding, research, trading, devops, writing
- Automatic deprecation of failing procedures

### Consolidation
- Light: Deduplication and cleanup
- Deep: Semantic extraction from episodes
- Skill: Procedure optimization and merging
- Provenance tracking from episodes to entities

## Tool Count Summary

| Category | Tools | Status |
|----------|-------|--------|
| Episodic | 4 | Production |
| Semantic | 5 | Production |
| Procedural | 4 | Production |
| Consolidation | 2 | Production |
| Utility | 2 | Production |
| **Total** | **17** | **Stable** |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              API / Gateway Layer                         │
│     REST API  │  MCP Gateway (17 tools)  │  Python SDK  │
├─────────────────────────────────────────────────────────┤
│              Hook Layer (Pre/On/Post)                    │
│   Caching │ Validation │ Audit │ Hebbian Learning       │
├─────────────────────────────────────────────────────────┤
│                   Memory Systems                         │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │  Episodic    │ │  Semantic    │ │  Procedural     │ │
│  │  (FSRS)      │ │  (ACT-R)     │ │  (skills)       │ │
│  └──────────────┘ └──────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Learning Layer                         │
│      Dopamine (reward)  │  Serotonin (credit)           │
├─────────────────────────────────────────────────────────┤
│                   Storage Layer                          │
│  ┌────────────────────┐    ┌──────────────────────────┐ │
│  │  Qdrant            │    │  Neo4j Graph DB          │ │
│  │  Vector embeddings │    │  Entities & relations    │ │
│  └────────────────────┘    └──────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                   Observability Layer                    │
│     WWObserver  │  Tracing  │  Citation Extraction      │
└─────────────────────────────────────────────────────────┘
```

## Cognitive Science Foundations

T4DM implements multiple cognitive architecture theories:

1. **Tripartite Memory** (Tulving 1972)
   - Episodic: Personal experiences
   - Semantic: General knowledge
   - Procedural: Skills and procedures

2. **ACT-R** (Anderson 1993)
   - Activation-based retrieval
   - Base-level activation + spreading
   - Decay over time

3. **FSRS** (Supermemo algorithm)
   - Spaced repetition scheduling
   - Retrievability modeling
   - Stability updates

4. **Hebbian Learning** (Hebb 1949)
   - "Neurons that fire together, wire together"
   - Relationship weight strengthening
   - Co-activation tracking

5. **Bi-temporal Modeling** (Snodgrass 2000)
   - Valid time vs. transaction time
   - Historical queries
   - Fact versioning

6. **Neuromodulator Systems** (Schultz 1997, Daw 2002)
   - **Dopamine**: Reward prediction errors, surprise signals, TD learning
   - **Serotonin**: Long-term credit assignment, eligibility traces, mood adaptation

7. **Eligibility Traces** (Sutton & Barto 2018)
   - Temporal credit assignment across delays
   - Decaying activation traces
   - Bridging temporal gaps in learning

## Example Workflows

### Learning from Interaction

```python
# 1. Claude Code helps user with a task
# 2. Episode is created automatically
episode = create_episode(
    content="User requested FastAPI deployment guide",
    outcome="success",
    valence=0.9
)

# 3. Semantic entities extracted during consolidation
consolidate_now(consolidation_type="deep")
# Creates: FastAPI (TOOL), Deployment (TECHNIQUE), etc.

# 4. Relationships learned
create_relation(
    source_id=fastapi_entity,
    target_id=uvicorn_entity,
    relation_type="REQUIRES"
)
```

### Skill Acquisition

```python
# 1. User completes successful workflow
trajectory = [
    {"tool": "run_tests", "result": "passed"},
    {"tool": "build_image", "result": "success"},
    {"tool": "push_image", "result": "success"}
]

# 2. Procedure learned (high success score)
create_skill(
    trajectory=trajectory,
    outcome_score=0.95,
    domain="devops",
    trigger_pattern="containerize application"
)

# 3. Later: Claude Code recalls the skill
skills = recall_skill(task="deploy Docker container")
# Returns learned procedure with steps
```

## Performance Characteristics

### Latency (p95)

| Operation | Latency | Notes |
|-----------|---------|-------|
| create_episode | <50ms | Vector embedding + Qdrant insert |
| recall_episodes | <200ms | Hybrid search with 4-component scoring |
| create_entity | <30ms | Neo4j insert |
| semantic_recall | <300ms | Vector + graph + spreading activation |
| spread_activation | <500ms | Multi-hop graph traversal |
| create_skill | <100ms | Trajectory parsing + storage |
| recall_skill | <150ms | Similarity + statistics scoring |
| consolidate_now (light) | <5s | Deduplication |
| consolidate_now (deep) | 30-300s | Depends on episode count |

### Throughput

- **Episodes**: 1000/sec sustained
- **Entities**: 500/sec sustained
- **Recalls**: 200/sec sustained (with caching)
- **Consolidation**: 10,000 episodes/min

### Storage

- **Episode**: ~2KB (vector: 1024-dim float32 = 4KB)
- **Entity**: ~1KB (Neo4j node + properties)
- **Relationship**: ~200B (Neo4j edge + weight)
- **Procedure**: ~5KB (steps + metadata)

**Scaling**:
- 100K episodes ≈ 200MB (vectors) + 50MB (metadata)
- 10K entities ≈ 10MB (nodes) + 2MB (relationships)
- 1K procedures ≈ 5MB

## API Versioning

T4DM follows semantic versioning:

- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes

**Current**: v3.0.0

**Deprecation Policy**:
- Deprecated tools supported for 2 minor versions
- Warnings emitted for deprecated features
- Migration guides provided

## Support

- **GitHub Issues**: https://github.com/astoreyai/t4dm/issues
- **Documentation**: https://github.com/astoreyai/t4dm/tree/main/docs
- **Email**: support@worldweaver.ai (if configured)

## Contributing

See `CONTRIBUTING.md` in the repository root.

## License

T4DM is released under the MIT License. See `LICENSE` file in repository root.

---

**Happy Memory Building!**
