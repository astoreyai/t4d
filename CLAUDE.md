# T4DM - Temporal 4D Memory

**Path**: `/mnt/projects/t4d/t4dm/`
**Version**: 1.0.0 (forked from World Weaver)
**Status**: Active Development

---

## Project Overview

T4DM (formerly World Weaver) is a biologically-inspired memory system for AI agents implementing tripartite neural memory (episodic, semantic, procedural) with cognitive consolidation, neuromodulator dynamics, FSRS-based spaced repetition, and world model prediction.

**Key Features**:
- Tripartite Memory Architecture (Episodic, Semantic, Procedural)
- Hinton architectures (Forward-Forward, Capsule Networks) for local learning
- Dual-Store Backend: Neo4j (graph) + Qdrant (vector) with Saga pattern
- 8,905 tests passing with 81% coverage

---

## T4D Stack Position

```
┌─────────┐
│  T4DV   │  ← Visualization (3D/4D rendering)
├─────────┤
│  T4DM   │  ← Memory semantics (THIS PROJECT)
├─────────┤
│  T4DX   │  ← Vector index + provenance
└─────────┘
```

---

## Scope (Added from T4DX)

| In Scope | Out of Scope |
|----------|--------------|
| Memory encoding/decoding | Vector indexing → **T4DX** |
| Time2Vec temporal encoding | Provenance graph → **T4DX** |
| Memory consolidation | 3D visualization → **T4DV** |
| Importance weighting | Query routing → **T4DX** |
| Interference detection | |
| Pattern separation | |
| STDP learning | |
| Hopfield retrieval | |

---

## Key Equations (from T4DX transfer)

### Temporal Encoding
```
t2v(τ)[i] = {
    ω_i · τ + φ_i,           if i = 0 (linear)
    sin(ω_i · τ + φ_i),      if 1 ≤ i ≤ k (periodic)
}
```

### Memory Consolidation
```
v_consolidated = Σᵢ wᵢ · vᵢ / Σᵢ wᵢ
wᵢ = importance(mᵢ) × recency(tᵢ) × relevance(mᵢ, context)
```

### STDP Learning
```
Δw = {
    A₊ · exp(-Δt/τ₊),  if Δt > 0 (LTP)
    -A₋ · exp(Δt/τ₋),  if Δt < 0 (LTD)
}
```

### Modern Hopfield Retrieval
```
update = softmax(β · Xᵀ · query) · X
```

---

## Quick Commands

```bash
# Development
make dev              # Start development server
make test             # Run all tests
make test-fast        # Run fast unit tests only
make coverage         # Generate coverage report

# Docker
docker compose up -d  # Start services
docker compose down   # Stop services

# CLI
t4dm store "memory content"   # Store a memory
t4dm recall "query"           # Recall memories
t4dm consolidate              # Run memory consolidation
```

---

## Architecture

```
src/t4dm/
├── core/              # Core memory operations
├── api/               # FastAPI REST endpoints
├── cli/               # Command-line interface
├── bridges/           # Neo4j + Qdrant backends
├── consolidation/     # Memory consolidation (HDBSCAN)
├── encoding/          # Time2Vec, multi-scale temporal (NEW)
├── learning/          # Hebbian learning, STDP
├── dreaming/          # Sleep replay consolidation
└── observability/     # Telemetry, tracing
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/t4dm/core/memory.py` | Main memory interface |
| `src/t4dm/api/routes.py` | REST API endpoints |
| `src/t4dm/bridges/saga.py` | Dual-store consistency |
| `src/t4dm/encoding/time2vec.py` | Temporal encoding (NEW) |
| `pyproject.toml` | Dependencies and config |
| `docker-compose.yml` | Service orchestration |

---

## Planning Documents

| Document | Purpose |
|----------|---------|
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | Encoding tasks (Track B) |
| [EQUATIONS_MAP.md](EQUATIONS_MAP.md) | Math requirements |

---

## Integration Points

| System | Integration |
|--------|-------------|
| **T4DX** | T4DM provides encoded vectors, T4DX indexes them |
| **T4DV** | T4DV visualizes T4DM memory structures |
| **Kymera** | Kymera uses T4DM for agent memory |

---

## Testing

```bash
# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests
pytest tests/biology/        # Biological plausibility tests
pytest -x --tb=short        # Stop on first failure
```

---

## Migration Notes (WW → T4DM)

- CLI command `ww` → `t4dm`
- Package `src/ww/` → `src/t4dm/`
- Repository will be renamed
- All tests remain valid
