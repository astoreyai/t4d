# T4DM (WW)

[![Release](https://img.shields.io/badge/release-v1.0.0-blue.svg)](https://github.com/astoreyai/t4d/releases/tag/v1.0.0)
[![Tests](https://img.shields.io/badge/tests-8%2C905%20passed-brightgreen.svg)](https://github.com/astoreyai/ww)
[![Coverage](https://img.shields.io/badge/coverage-81%25-green.svg)](https://github.com/astoreyai/ww)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Version**: 1.0.0 | **Status**: Production Ready | **Tests**: 8,905 passed, 81% coverage

Biologically-inspired memory for AI. A modular framework implementing tripartite neural memory (episodic, semantic, procedural) with cognitive consolidation, neuromodulator dynamics, FSRS-based spaced repetition, and world model prediction. Implements key Hinton architectures (Forward-Forward, Capsule Networks) for local learning without backpropagation.

## What's New in v1.0.0

- **100% Test Pass Rate**: All 8,905 tests passing with 81% coverage
- **Production Ready**: All critical wiring complete, no stubs in production code
- **Bug Fixes**: VAE gradient propagation, online adapter dimensions, sample tracking
- **Complete Integration**: Learning signals now persist to embeddings
- **Sleep Consolidation**: Reconsolidated embeddings properly persisted to vector store

## Features

- **Tripartite Memory Architecture**: Episodic, Semantic, and Procedural memory subsystems
- **Simplified Python API**: `from ww import memory` for intuitive memory operations
- **CLI Tool**: `ww` command for terminal-based memory management
- **REST API**: FastAPI-based HTTP interface with OpenAPI documentation
- **Dual-Store Backend**: Neo4j (graph) + Qdrant (vector) with Saga pattern for consistency
- **Session Isolation**: Complete isolation between user sessions with automatic cleanup
- **Memory Consolidation**: HDBSCAN-based clustering for memory optimization
- **Hebbian Learning**: Automatic strengthening of co-retrieved memory connections
- **Neuromodulator Systems**: Dopamine (reward prediction) and Serotonin (long-term credit)
- **FSRS Scheduler**: Free Spaced Repetition Scheduler for optimal memory retention
- **Memory Hooks**: Extensible pre/post/on hooks for memory operations
- **Observability**: OpenTelemetry-compatible tracing and metrics
- **World Model Prediction** (v0.3.0+):
  - JEPA-style latent prediction with attention-weighted context
  - 15-step dream trajectory generation for imagination-based consolidation
  - Prediction error as prioritized replay signal
- **Advanced Neuroscience** (v0.4.0):
  - Hierarchical multi-timescale prediction (fast/medium/slow horizons)
  - Causal discovery with graph-based counterfactual learning
  - Place/grid cell spatial cognition (Nobel Prize 2014 mechanisms)
  - Theta-gamma coupling with working memory slots (7±2 capacity)
- **Hinton Architectures** (v0.5.0):
  - Forward-Forward Algorithm: Local learning without backpropagation (Hinton 2022)
  - Capsule Networks: Part-whole hierarchies with routing-by-agreement
  - Three-Factor Learning: Eligibility × Neuromodulator × Dopamine
  - Glymphatic System: Sleep-based waste clearance and consolidation
  - Hippocampal Circuit: DG/CA3/CA1 pattern separation/completion
  - VTA Dopamine: Reward prediction error computation
  - Sleep Consolidation: NREM/REM cycles with SWR replay
- **Biological Integration** (v0.5.0 - Phase 1):
  - Sleep ↔ Reconsolidation: NREM replay triggers embedding updates
  - VTA ↔ STDP: Dopamine modulates synaptic plasticity
  - VAE Training Loop: Hinton wake-sleep algorithm integration
  - VTA ↔ Sleep: RPE-based replay prioritization (90% reverse, 10% forward)
  - Lability Window: 6-hour protein synthesis gate (Nader et al. 2000)
- **Production Ready** (v1.0.0):
  - All learning signals properly wired to embedding persistence
  - FFCapsuleBridge instantiated for end-to-end representation learning
  - Online embedding adapter with LoRA-style adaptation
  - Complete VAE generative replay pipeline
  - 8,905 tests passing with 81% coverage

## Quick Start

### Installation

```bash
pip install t4dm
```

### Simplified Memory API

```python
from ww import memory

# Store content
await memory.store("User prefers dark mode interfaces")

# Recall similar memories
results = await memory.recall("interface preferences")
for r in results:
    print(f"{r.memory_type}: {r.content}")

# Store structured knowledge
await memory.store_entity("Python", description="A programming language", entity_type="concept")

# Store procedural skills
await memory.store_skill("git_commit", script="git add -A && git commit -m '...'", domain="coding")

# Use session context for isolation
async with memory.session("my-project") as m:
    await m.store("Project-specific knowledge")
```

### CLI Usage

```bash
# Store a memory
ww store "Learned about decorators in Python"

# Recall memories
ww recall "Python decorators"

# Show recent episodes
ww episodic recent --limit 10

# Show system status
ww status

# Start REST API server
t4dm serve --port 8765
```

### Docker (Recommended for Infrastructure)

```bash
# Clone and configure
git clone https://github.com/astoreyai/t4dm
cd t4dm
cp .env.example .env
./scripts/setup-env.sh  # Generates secure passwords

# Start infrastructure (Neo4j + Qdrant)
docker-compose up -d

# Or start full stack with API server
docker-compose -f docker-compose.full.yml up -d

# Access API
curl http://localhost:8765/api/v1/health
# API Docs: http://localhost:8765/docs
```

### Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e ".[api,dev]"

# Start infrastructure
docker-compose up -d

# Run tests
pytest tests/ -v
```

## Configuration

T4DM supports YAML configuration with environment variable overrides:

```yaml
# t4dm.yaml (in project root or ~/.t4dm/config.yaml)
session_id: my-project
environment: development

# Storage
qdrant_host: localhost
qdrant_port: 6333
neo4j_uri: bolt://localhost:7687
neo4j_user: neo4j
neo4j_password: your-password

# Embedding
embedding_model: bge-m3
embedding_dim: 1024

# API
api_host: 0.0.0.0
api_port: 8765
```

Environment variables override YAML (prefixed with `T4DM_`):
```bash
export T4DM_SESSION_ID=my-session
export T4DM_QDRANT_HOST=localhost
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      API / INTERFACE LAYER                       │
│      CLI (typer)  │  REST API (FastAPI)  │  Python SDK          │
├─────────────────────────────────────────────────────────────────┤
│                      SIMPLIFIED MEMORY API                       │
│                  from ww import memory                           │
├─────────────────────────────────────────────────────────────────┤
│                      HOOK LAYER (Pre/On/Post)                    │
│     Caching │ Validation │ Audit │ Hebbian Learning             │
├─────────────────────────────────────────────────────────────────┤
│                      MEMORY SUBSYSTEMS                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Episodic   │  │  Semantic   │  │  Procedural │             │
│  │ (FSRS)      │  │ (ACT-R)     │  │ (skills)    │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
├─────────┴────────────────┴────────────────┴─────────────────────┤
│                   PREDICTION & DREAMING LAYER                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │ Hierarchical   │  │    Dreaming    │  │    Causal      │    │
│  │ (fast/med/slow)│  │  (15-step)     │  │   Discovery    │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
├─────────────────────────────────────────────────────────────────┤
│                      NCA DYNAMICS LAYER                          │
│  ┌──────────────────┐  ┌────────────────┐  ┌───────────────┐   │
│  │  Neural Field    │  │  Theta-Gamma   │  │ Place/Grid    │   │
│  │  (6-NT PDE)      │◄─┤  Coupling      │◄─┤ Cells         │   │
│  └────────┬─────────┘  └───────┬────────┘  └───────┬───────┘   │
│           │    Learnable Coupling Matrix (K)       │           │
├───────────┴────────────────────────────────────────┴───────────┤
│                      LEARNING LAYER                              │
│         Dopamine (reward)  │  Serotonin (credit)                │
├─────────────────────────────────────────────────────────────────┤
│                       STORAGE LAYER                              │
│         ┌─────────────┐         ┌─────────────┐                 │
│         │   Neo4j     │         │   Qdrant    │                 │
│         │ (graph)     │◄───────►│  (vector)   │                 │
│         └─────────────┘  Saga   └─────────────┘                 │
├─────────────────────────────────────────────────────────────────┤
│                    OBSERVABILITY LAYER                           │
│              Tracing │ Metrics │ Health Checks                  │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Subsystems

### Episodic Memory
Autobiographical events with temporal context:
- Stores experiences with timestamps, emotional valence, outcomes
- FSRS-based spaced repetition for optimal retention
- Enables temporal-aware retrieval (recent events weighted higher)

### Semantic Memory
Hebbian knowledge graph with automatic strengthening:
- Entities with types (CONCEPT, PERSON, PROJECT, TOOL, TECHNIQUE, FACT)
- Relationships strengthened through co-retrieval (Hebbian learning)
- ACT-R inspired activation spreading

### Procedural Memory
Skill and pattern storage:
- Parameterized skills with pre/post conditions
- Domain-specific organization (coding, research, trading, devops, writing)
- Usage tracking and success rate monitoring

## Neuro Cognitive Architecture (NCA)

KATIE-inspired neural dynamics for biologically-plausible cognitive state management:

### 6-Neurotransmitter PDE System
The neural field evolves according to: `dU/dt = -aU + D*nabla^2*U + S + K*U`
- **DA** (Dopamine): Reward, motivation (~100ms timescale)
- **5-HT** (Serotonin): Mood, satiety (~500ms timescale)
- **ACh** (Acetylcholine): Attention, encoding (~50ms timescale)
- **NE** (Norepinephrine): Arousal, vigilance (~200ms timescale)
- **GABA**: Fast inhibition (~10ms timescale)
- **Glu** (Glutamate): Fast excitation (~5ms timescale)

### Cognitive State Attractors
Five attractor basins in NT concentration space:
| State | NT Signature | Role |
|-------|-------------|------|
| ALERT | High NE, DA | Vigilance, rapid response |
| FOCUS | High ACh, Glu | Sustained attention, encoding |
| REST | High 5-HT, GABA | Default mode, low arousal |
| EXPLORE | High DA, ACh | Novelty seeking, curiosity |
| CONSOLIDATE | High GABA, 5-HT | Sleep/offline consolidation |

## Project Structure

```
t4dm/
├── src/t4dm/                 # Main source code
│   ├── api/                # REST API server (112 endpoints)
│   ├── bridge/             # Memory-NCA integration
│   ├── cli/                # Command-line interface
│   ├── consolidation/      # Sleep consolidation service
│   ├── core/               # Types, config, schemas
│   ├── dreaming/           # Dream trajectory generation
│   ├── embedding/          # Embedding service (BGE-M3)
│   ├── hooks/              # Memory operation hooks
│   ├── learning/           # Neuromodulators, eligibility, FSRS
│   │   ├── dopamine.py     # Reward prediction error
│   │   ├── eligibility.py  # Temporal credit assignment
│   │   ├── three_factor.py # Three-factor learning rule
│   │   └── fsrs.py         # Spaced repetition scheduler
│   ├── memory/             # Episodic, Semantic, Procedural
│   ├── nca/                # Neuro Cognitive Architecture
│   │   ├── forward_forward.py  # FF algorithm (Hinton 2022)
│   │   ├── capsules.py     # Capsule networks
│   │   ├── hippocampus.py  # DG/CA3/CA1 circuit
│   │   ├── vta.py          # Dopamine circuit
│   │   ├── glymphatic.py   # Waste clearance system
│   │   └── swr_coupling.py # Sharp-wave ripple replay
│   ├── observability/      # Tracing and monitoring
│   ├── persistence/        # Checkpoint, WAL, recovery
│   ├── prediction/         # JEPA-style prediction
│   ├── sdk/                # Python SDK (sync/async)
│   ├── storage/            # Neo4j + Qdrant stores
│   └── visualization/      # 20+ visualization modules
├── tests/                  # Test suite (8,905 tests, 81% coverage)
├── docs/                   # Documentation
├── docker-compose.yml      # Infrastructure
└── pyproject.toml          # Package configuration
```

## Testing

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (requires running Neo4j/Qdrant)
pytest tests/integration/ -v -m integration

# With coverage
pytest tests/ --cov=src/ww --cov-report=term-missing
```

## Dependencies

### Required
- `neo4j>=5.0.0` - Graph database client
- `qdrant-client>=1.7.0` - Vector database client
- `pydantic>=2.0.0` - Data validation
- `pydantic-settings>=2.0.0` - Settings management
- `pyyaml>=6.0.0` - YAML configuration
- `typer>=0.9.0` - CLI framework
- `rich>=13.0.0` - Rich terminal output

### Optional
- `fastapi>=0.100.0` - REST API (install with `pip install t4dm[api]`)
- `FlagEmbedding>=1.2.7` - BGE-M3 embeddings (install with `pip install t4dm[embedding]`)
- `hdbscan>=0.8.33` - Memory consolidation clustering

## Author

**Aaron W. Storey** ([@astoreyai](https://github.com/astoreyai))

PhD Candidate, Computer Science, Clarkson University. Research: memory-augmented AI architectures, explainable AI, agentic systems.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{storey2026worldweaver,
  author = {Storey, Aaron W.},
  title = {T4DM: Biologically-Inspired Memory for AI},
  year = {2026},
  url = {https://github.com/astoreyai/ww}
}
```

## Documentation

- [REST API](docs/API.md) - HTTP endpoint reference
- [Python SDK](docs/SDK.md) - Client library usage
- [Deployment](docs/DEPLOYMENT.md) - Production deployment guide
- [Architecture](docs/ARCHITECTURE.md) - Full system design
- [Biological Architecture](docs/ARCHITECTURE_BIOLOGICAL.md) - Biological integration design
- [Biological Integration](docs/BIOLOGICAL_INTEGRATION.md) - Neural mechanisms and citations
- [Memory Architecture](docs/MEMORY_ARCHITECTURE.md) - Tripartite memory specification
- [NCA Module](src/t4dm/nca/README.md) - Neuro Cognitive Architecture dynamics
- [Testing Guide](docs/TESTING_GUIDE.md) - Test suite documentation

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and contribution guidelines.

## Security

For security vulnerabilities, please see [SECURITY.md](SECURITY.md).
