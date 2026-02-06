# T4DM Documentation

**Version**: 2.0.0 | **Last Updated**: 2026-02-05 | **Tests**: 9,771 passed, 26% coverage

Welcome to the T4DM documentation. This directory contains comprehensive guides for using, deploying, and developing with the T4DM biologically-inspired memory system.

## Quick Navigation

### For Users

- **[API Reference](API.md)** - Complete REST API reference
  - Episodic memory endpoints
  - Semantic memory endpoints
  - Procedural memory endpoints
  - Visualization endpoints (22 modules)
  - Consolidation & system endpoints

- **[SDK Guide](SDK.md)** - Python SDK reference
- **[SDK Quickstart](sdk/QUICKSTART.md)** - Get started in 5 minutes

### For Operators

- **[Self-Hosted Guide](SELF_HOSTED_GUIDE.md)** - Production deployment
  - Single-binary deployment (no external DBs)
  - Resource optimization
  - Monitoring & logging
  - Backup & recovery

### For Developers

- **[System Architecture Master](SYSTEM_ARCHITECTURE_MASTER.md)** - Complete system design
- **[Integration Guide](integration/README.md)** - Framework adapters (LangChain, LlamaIndex, etc.)
- **[Mathematical Foundations](MATHEMATICAL_FOUNDATIONS.md)** - Core equations
- **[Brain Region Mapping](BRAIN_REGION_MAPPING.md)** - Neuroscience foundations

### Deep Dive Walkthroughs

- **[Persistence Architecture](PERSISTENCE_ARCHITECTURE.md)** - WAL, checkpoints, crash recovery
- **[Neuroscience Taxonomy](NEUROSCIENCE_TAXONOMY.md)** - Biological mapping
- **[Competitive Analysis](COMPETITIVE_ANALYSIS.md)** - T4DM vs other memory systems

## Getting Started

### Installation (2 minutes)

```bash
# Clone repository
git clone https://github.com/astoreyai/t4dm.git
cd t4dm

# Install Python package
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"

# Verify installation
pytest tests/unit/ -v
```

**Note**: T4DM uses an embedded T4DX storage engine - no external databases required.

### Configuration

Edit `.env` with your settings:

```bash
T4DM_SESSION_ID=my-session
T4DM_STORAGE_PATH=/var/lib/t4dm/data
T4DM_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Usage

**Python SDK**:
```python
from t4dm.sdk import T4DMClient

with T4DMClient(session_id="my-session") as client:
    # Store a memory
    episode = client.create_episode(
        content="User completed onboarding",
        context={"flow": "onboarding"},
        outcome="success"
    )

    # Recall memories
    results = client.recall_episodes("onboarding")
```

**MCP Server** (for Claude Code/Desktop):
```json
{
  "mcpServers": {
    "t4dm": {
      "command": "t4dm",
      "args": ["mcp", "server"],
      "env": {"T4DM_SESSION_ID": "my-session"}
    }
  }
}
```

## Documentation Overview

| Document | Purpose | Audience |
|----------|---------|----------|
| [API.md](API.md) | REST API reference | Users, integrators |
| [SDK.md](SDK.md) | Python SDK reference | Developers |
| [integration/README.md](integration/README.md) | Framework adapters | Integrators |
| [SYSTEM_ARCHITECTURE_MASTER.md](SYSTEM_ARCHITECTURE_MASTER.md) | System design | Developers, architects |
| [PERSISTENCE_ARCHITECTURE.md](PERSISTENCE_ARCHITECTURE.md) | Storage & recovery | Operators, developers |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    API / Gateway Layer                       │
│     REST API  │  MCP Server  │  Python SDK  │  Adapters     │
├─────────────────────────────────────────────────────────────┤
│                    Qwen 2.5-3B (4-bit)                       │
│  Layers 0-17 (frozen + QLoRA) │ Layers 18-35 (frozen + QLoRA)│
├─────────────────────────────────────────────────────────────┤
│              Spiking Cortical Stack (×6 blocks)              │
│  LIF │ Thalamic Gate │ Spike Attention │ Apical │ RWKV      │
├─────────────────────────────────────────────────────────────┤
│                    Memory Systems                            │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐      │
│  │  Episodic    │ │  Semantic    │ │  Procedural     │      │
│  │  (κ < 0.3)   │ │  (κ > 0.6)   │ │  (skills)       │      │
│  └──────────────┘ └──────────────┘ └─────────────────┘      │
├─────────────────────────────────────────────────────────────┤
│                    Learning Layer                            │
│  Neuromodulators (DA, NE, ACh, 5-HT) │ STDP │ Hebbian      │
├─────────────────────────────────────────────────────────────┤
│                  T4DX Storage Engine                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  WAL │ MemTable │ LSM Segments │ HNSW │ CSR Graph   │    │
│  │  κ-Index │ Bitemporal │ Provenance │ Compaction     │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                   Observability Layer                        │
│     Prometheus  │  OpenTelemetry  │  22 Viz Modules         │
└─────────────────────────────────────────────────────────────┘
```

## Core Concepts

### κ (Kappa) Gradient

Continuous consolidation level [0,1] replacing discrete memory stores:

| κ Range | State | Description |
|---------|-------|-------------|
| 0.0-0.1 | Raw | Just encoded, volatile |
| 0.1-0.3 | Replayed | NREM-strengthened |
| 0.3-0.6 | Transitional | Being abstracted |
| 0.6-0.9 | Semantic | Consolidated concept |
| 0.9-1.0 | Stable | Permanent knowledge |

### T4DX Storage Engine

Embedded LSM-style storage with:
- **9 primitives**: INSERT, GET, SEARCH, UPDATE_FIELDS, UPDATE_EDGE_WEIGHT, TRAVERSE, SCAN, DELETE, BATCH_SCALE_WEIGHTS
- **HNSW index**: 4D vector similarity (space + time)
- **CSR graph**: Compressed sparse row for edges
- **WAL**: Write-ahead log for crash recovery
- **Compaction**: Memory consolidation = LSM compaction

### Neuromodulator Orchestra

| Neuromodulator | Effect | Software Mapping |
|----------------|--------|------------------|
| Dopamine (DA) | Reward gating | Outcome-based learning |
| Norepinephrine (NE) | Attention boost | Salience weighting |
| Acetylcholine (ACh) | Learning rate | Encoding/retrieval mode |
| Serotonin (5-HT) | Mood baseline | Temporal credit assignment |

## Diagrams

See [diagrams/DIAGRAM_SUMMARY.md](diagrams/DIAGRAM_SUMMARY.md) for the complete diagram inventory:

- **9 D2 diagrams**: System architecture, data flow, spiking blocks
- **51 Mermaid diagrams**: State machines, sequences, classes
- **13 Markdown docs**: Embedded diagrams with explanations
- **173 rendered outputs**: SVG and PNG

Key diagrams:
- `t4dm_full_system.d2` - Complete system architecture
- `t4dx_storage_engine.d2` - T4DX internals
- `t4dm_spiking_block.d2` - 6-stage cortical block

## Performance Characteristics

### Latency (P95)

| Operation | Latency | Notes |
|-----------|---------|-------|
| create_episode | <50ms | Embedding + T4DX insert |
| recall_episodes | <100ms | HNSW search + reranking |
| consolidate (NREM) | <5s | LSM compaction |
| consolidate (REM) | 30-300s | Semantic prototype creation |

### Storage

- **Episode**: ~2KB (vector: 32-dim float32)
- **T4DX segment**: 64MB (configurable)
- **100K episodes**: ~200MB total

## API Versioning

T4DM follows semantic versioning:

- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes

**Current**: v2.0.0

## Support

- **GitHub Issues**: https://github.com/astoreyai/t4dm/issues
- **Documentation**: https://github.com/astoreyai/t4dm/tree/main/docs

## License

T4DM is released under the MIT License.

---

**Happy Memory Building!**
