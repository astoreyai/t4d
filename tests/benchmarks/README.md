# T4DM Benchmark Tests

Pytest-compatible benchmark suite for T4DM converted from standalone scripts for CI/CD integration.

## Structure

```
tests/benchmarks/
├── conftest.py              # Pytest configuration and markers
├── test_bioplausibility.py  # Bio-plausibility validation (16 tests)
├── test_longmemeval.py      # Long-term memory evaluation (20 tests)
├── test_dmr.py              # Deep memory retrieval with κ-gradient (15 tests)
└── README.md                # This file
```

## Running Benchmarks

### All benchmarks (51 tests)
```bash
pytest tests/benchmarks/ -m benchmark -v
```

### Individual benchmark suites
```bash
# Bio-plausibility validation
pytest tests/benchmarks/test_bioplausibility.py -m bioplausibility -v

# Long memory evaluation
pytest tests/benchmarks/test_longmemeval.py -m memory -v

# Deep memory retrieval
pytest tests/benchmarks/test_dmr.py -m retrieval -v
```

### CI/CD Integration
```bash
# Run as part of test suite
pytest tests/benchmarks/ -m benchmark --tb=short

# With coverage reporting
pytest tests/benchmarks/ -m benchmark --cov=src/t4dm --cov-report=term-missing
```

## Benchmarks Overview

### 1. Bioplausibility (`test_bioplausibility.py`)

Validates T4DM against neuroscience literature on:

- **CLS Theory Compliance** (3 checks)
  - Fast hippocampal-like learning
  - Slow neocortical-like integration
  - Interleaved replay for catastrophic forgetting prevention
  - References: McClelland et al., 1995

- **Consolidation Dynamics** (4 checks)
  - NREM-dominant replay (75% of sleep)
  - REM creative integration
  - Synaptic downscaling/homeostatic pruning
  - Sharp-wave ripple compressed replay
  - References: Diekelmann & Born (2010), Foster & Wilson (2006), Tononi & Cirelli (2014)

- **Neuromodulator Effects** (4 checks)
  - Dopamine reward prediction error (Schultz, 1998)
  - Acetylcholine encoding/retrieval modulation (Hasselmo, 2006)
  - Norepinephrine arousal modulation (Sara, 2009)
  - Serotonin temporal credit assignment (Doya, 2002)
  - References: Dayan & Yu (2006)

**Pass Criteria**: ≥80% compliance rate across all checks

### 2. Long Memory Evaluation (`test_longmemeval.py`)

Evaluates memory system on realistic usage patterns:

- **Needle-in-Haystack** (5 tests)
  - Finding specific memories in large context
  - Tests positions: start, middle, end
  - Configurable haystack sizes (100, 500, 1000)
  - Measures: accuracy, latency

- **Retention** (3 tests)
  - Long-term retention after consolidation
  - Time intervals: 1h, 1d, 1w
  - Tests: memory decay, consolidation effectiveness
  - Measures: accuracy over time

- **Session Memory** (4 tests)
  - Cross-session memory continuity
  - Multiple session handling
  - Session isolation verification
  - Measures: retrieval accuracy, continuity

- **Integration** (5 tests)
  - Complete benchmark suite
  - Memory scaling properties
  - Component verification
  - Latency and accuracy thresholds

**Pass Criteria**: ≥50% pass rate, reasonable latency (<1s), valid accuracy [0,1]

### 3. Deep Memory Retrieval (`test_dmr.py`)

Measures retrieval accuracy with κ-gradient consolidation:

- **Retrieval Accuracy** (6 tests)
  - Recall@1, Recall@5, Recall@10
  - Mean Reciprocal Rank (MRR)
  - Continuous vs. discrete consolidation modes
  - Latency measurement

- **Kappa Distribution** (4 tests)
  - Retrieval across consolidation levels (κ ∈ [0, 0.3, 0.6, 0.9])
  - Episodic (κ=0.0) to semantic (κ=1.0) retrieval
  - Level completeness verification

- **Integration** (5 tests)
  - Complete benchmark suite
  - Metric validity verification
  - Recall hierarchy: R@1 ≤ R@5 ≤ R@10
  - Latency and statistics

**Pass Criteria**: Valid metrics [0,1], reasonable latency, recall hierarchy maintained

## Test Architecture

### Mock Systems

Each test suite uses mock memory systems for isolation:

- `T4DMTestSystem`: Mock with bio-plausible attributes
- `MockMemorySystem`: Simple string-based search for LongMemEval
- `MockMemorySystemDMR`: κ-aware storage for DMR tests

### Fixtures

Common fixtures in `conftest.py`:

- `pytest_configure()`: Registers custom markers
- `benchmark_results`: Collects results per benchmark type

Test-specific fixtures:

```python
@pytest.fixture
def test_system():
    """Create a test system with bio-plausible attributes."""

@pytest.fixture
def memory_system():
    """Create a mock memory system for testing."""
```

## Thresholds and Pass Criteria

| Benchmark | Metric | Threshold | Notes |
|-----------|--------|-----------|-------|
| Bio-plausibility | Compliance rate | ≥80% | Must pass 80%+ of checks |
| Bio-plausibility | Per-category | ≥80% | Each category (CLS, Consolidation, Neuro) |
| LongMemEval | Pass rate | ≥50% | Tests may have lower accuracy in mock |
| LongMemEval | Latency | <1000ms | Should be fast (mock implementation) |
| DMR | Latency | <5000ms | Reasonable for semantic search |
| DMR | Recall metrics | [0, 1] | Must be valid probabilities |
| DMR | Hierarchy | R@1 ≤ R@5 ≤ R@10 | Recall should be monotonic |

## CI Integration

### GitHub Actions Example

```yaml
- name: Run benchmarks
  run: |
    pytest tests/benchmarks/ -m benchmark \
      --tb=short \
      --cov=src/t4dm \
      --cov-report=json \
      -v

- name: Check benchmark results
  if: always()
  run: |
    python scripts/validate_benchmarks.py \
      --results coverage.json \
      --min-bio-plausibility 0.80 \
      --min-retrieval-latency 5000
```

### Make Target

```bash
make test-benchmarks          # Run all benchmarks
make test-benchmarks-bio     # Bio-plausibility only
make test-benchmarks-memory  # Memory evaluation only
make test-benchmarks-dmr     # DMR only
```

## Implementation Notes

### Markers

Tests use custom pytest markers for selective execution:

```python
@pytest.mark.benchmark       # All benchmarks
@pytest.mark.bioplausibility # Bio-plausibility suite
@pytest.mark.memory          # LongMemEval suite
@pytest.mark.retrieval       # DMR suite
```

### Classes vs. Functions

Tests are organized into classes for logical grouping:

- `TestCLSCompliance` - CLS checks
- `TestConsolidationDynamics` - Sleep consolidation
- `TestNeuromodulators` - Neuromodulator effects
- `TestBioplausibilityBenchmarkComplete` - Integration
- etc.

### Logging

All benchmark tests log summary statistics:

```
Bio-plausibility: 11/12 checks passed (91.67%), avg score: 0.80
LongMemEval: 18/20 tests passed (90.00%), avg latency: 125.45ms, avg accuracy: 0.85
DMR: 51 tests, avg recall@1: 0.75, avg MRR: 0.68, avg latency: 25.40ms
```

## Future Enhancements

1. **Real Memory System Tests**
   - Replace mock systems with T4DM instances
   - Add integration with actual storage engine
   - Measure real retrieval performance

2. **Performance Baselines**
   - Store baseline results
   - Track performance regression
   - Alert on significant changes

3. **Parallel Execution**
   - Use `pytest-xdist` for faster benchmark runs
   - Distribute across CI workers

4. **Results Reporting**
   - Generate HTML reports
   - Upload to benchmarking service
   - Track trends over time

## References

### Bio-plausibility
- McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419.
- Diekelmann, S., & Born, J. (2010). The memory function of sleep. *Nature Reviews Neuroscience*, 11(2), 114-126.
- Dayan, P., & Yu, A. J. (2006). Phasic norepinephrine: A neural interrupt signal for unexpected events. *Network*, 17(4), 335-350.

### Memory Systems
- Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554-2558.
- Marr, D. (1971). Simple memory: A theory for archicortex. *Philosophical Transactions of the Royal Society B*, 262(841), 23-81.
