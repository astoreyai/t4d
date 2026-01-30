# Tests
**Path**: `/mnt/projects/t4d/t4dm/tests/`

## What
Comprehensive test suite for T4DM's tripartite memory system (episodic, semantic, procedural). 8,905+ tests with 81% coverage spanning unit, integration, security, performance, biology, and chaos testing.

## How
- **pytest** with shared fixtures in `conftest.py`
- Property-based testing for algorithm invariants (FSRS, Hebbian, ACT-R)
- Test categories organized by subdirectory (unit, integration, security, performance, biology, chaos, e2e)
- Run: `pytest tests/unit/` (fast), `pytest tests/integration/` (needs services), `pytest tests/biology/` (biological plausibility)

## Why
Validates correctness of bio-inspired learning algorithms, dual-store saga patterns, temporal dynamics, and consolidation pipelines. Security tests prevent injection attacks on Neo4j/Qdrant backends.

## Key Files
| File | Purpose |
|------|---------|
| `conftest.py` | Shared fixtures, mock backends, test configuration |
| `unit/` | Fast isolated tests (~60+ test files) |
| `integration/` | Tests with real/mocked Neo4j and Qdrant |
| `biology/` | Biological plausibility validation |
| `chaos/` | Fault injection and resilience |
| `security/` | Input sanitization, injection prevention |
| `performance/` | Benchmarks and performance baselines |
| `e2e/` | End-to-end workflow tests |
| `encoding/` | Time2Vec and temporal encoding tests |
| `learning/` | Hebbian, STDP, eligibility trace tests |
| `consolidation/` | Memory consolidation pipeline tests |
| `dreaming/` | Sleep replay consolidation tests |

## Data Flow
```
Test Runner (pytest)
  → conftest.py (fixtures, mocks)
    → Individual test modules
      → src/t4dm/ (system under test)
```

## Integration Points
- **CI/CD**: `make test`, `make test-fast`, `make coverage`
- **Docker**: Integration tests require Neo4j + Qdrant containers
- **Makefile**: Targets for running specific test categories
