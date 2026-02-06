# Quick Start Guide - Benchmark Tests

## Basic Commands

```bash
# Run ALL benchmarks (51 tests, ~5 seconds)
pytest tests/benchmarks/ -m benchmark -v

# Run with short output
pytest tests/benchmarks/ -m benchmark --tb=short

# Run with coverage
pytest tests/benchmarks/ -m benchmark --cov=src/t4dm
```

## By Benchmark Type

```bash
# Bio-plausibility only (16 tests)
pytest tests/benchmarks/test_bioplausibility.py -m bioplausibility -v

# Long memory evaluation only (20 tests)
pytest tests/benchmarks/test_longmemeval.py -m memory -v

# Deep memory retrieval only (15 tests)
pytest tests/benchmarks/test_dmr.py -m retrieval -v
```

## By Test Class

```bash
# CLS compliance only
pytest tests/benchmarks/test_bioplausibility.py::TestCLSCompliance -v

# Consolidation dynamics only
pytest tests/benchmarks/test_bioplausibility.py::TestConsolidationDynamics -v

# Neuromodulator validation only
pytest tests/benchmarks/test_bioplausibility.py::TestNeuromodulators -v

# Needle-in-haystack tests only
pytest tests/benchmarks/test_longmemeval.py::TestNeedleInHaystack -v

# Retrieval accuracy tests only
pytest tests/benchmarks/test_dmr.py::TestRetrievalAccuracy -v

# κ-distribution tests only
pytest tests/benchmarks/test_dmr.py::TestKappaDistribution -v
```

## CI Integration

```bash
# Typical CI command
pytest tests/benchmarks/ -m benchmark --tb=short -v

# With failure details
pytest tests/benchmarks/ -m benchmark --tb=long -v

# Exit on first failure
pytest tests/benchmarks/ -m benchmark -x

# Stop after N failures
pytest tests/benchmarks/ -m benchmark --maxfail=3
```

## Verbose Output

```bash
# Show print statements
pytest tests/benchmarks/ -m benchmark -v -s

# Show all assertions
pytest tests/benchmarks/ -m benchmark -vv

# Show timing
pytest tests/benchmarks/ -m benchmark -v --durations=10
```

## Filtering

```bash
# Run only bioplausibility complete tests
pytest tests/benchmarks/ -k "complete" -v

# Skip slow tests (if any marked)
pytest tests/benchmarks/ -m "benchmark and not slow" -v

# Run only tests mentioning "recall"
pytest tests/benchmarks/ -k "recall" -v

# Run only single test
pytest tests/benchmarks/test_bioplausibility.py::TestCLSCompliance::test_fast_hippocampal_learning -v
```

## Output Formats

```bash
# JSON output for CI parsing
pytest tests/benchmarks/ -m benchmark --json=results.json

# JUnit XML for CI integration
pytest tests/benchmarks/ -m benchmark --junit-xml=results.xml

# HTML report
pytest tests/benchmarks/ -m benchmark --html=report.html --self-contained-html
```

## Performance Analysis

```bash
# Show slowest 10 tests
pytest tests/benchmarks/ -m benchmark --durations=10

# Profile with cProfile
pytest tests/benchmarks/ -m benchmark --profile

# Memory profiling
pytest tests/benchmarks/ -m benchmark --memray
```

## Troubleshooting

```bash
# Show full exception details
pytest tests/benchmarks/ -m benchmark -vv --tb=long

# Drop into pdb on failure
pytest tests/benchmarks/ -m benchmark --pdb

# Show locals on failure
pytest tests/benchmarks/ -m benchmark -l

# Detailed import info
pytest tests/benchmarks/ -m benchmark --import-mode=importlib -v
```

## Examples

### Run bio-plausibility and see results
```bash
$ pytest tests/benchmarks/test_bioplausibility.py -m benchmark -v
tests/benchmarks/test_bioplausibility.py::TestCLSCompliance::test_fast_hippocampal_learning PASSED
tests/benchmarks/test_bioplausibility.py::TestCLSCompliance::test_slow_neocortical_integration PASSED
tests/benchmarks/test_bioplausibility.py::TestCLSCompliance::test_interleaved_learning PASSED
tests/benchmarks/test_bioplausibility.py::TestCLSCompliance::test_all_cls_checks PASSED
...
===================== 16 passed in 0.45s =====================
```

### Run DMR with detailed output
```bash
$ pytest tests/benchmarks/test_dmr.py::TestDMRComplete::test_complete_dmr_benchmark -v -s
tests/benchmarks/test_dmr.py::TestDMRComplete::test_complete_dmr_benchmark
DMR: 51 tests, avg recall@1: 0.75, avg MRR: 0.68, avg latency: 25.40ms
PASSED [100%]
```

### Check what tests exist
```bash
$ pytest tests/benchmarks/ -m benchmark --collect-only
collected 51 items
```

### Run only retrieval tests
```bash
$ pytest tests/benchmarks/ -m retrieval -v --tb=short
tests/benchmarks/test_dmr.py::TestRetrievalAccuracy::test_recall_at_1 PASSED
tests/benchmarks/test_dmr.py::TestRetrievalAccuracy::test_recall_at_5 PASSED
...
===================== 15 passed in 0.38s =====================
```

## Environment Variables

```bash
# Enable debug logging
PYTEST_DEBUG=1 pytest tests/benchmarks/ -m benchmark -v

# Set test timeout (seconds)
PYTEST_TIMEOUT=30 pytest tests/benchmarks/ -m benchmark

# Verbose coverage
COVERAGE_DEBUG=1 pytest tests/benchmarks/ -m benchmark --cov
```

## Configuration Files

The benchmark tests respect pytest configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
markers = ["benchmark", "bioplausibility", "memory", "retrieval"]
```

## See Also

- [Detailed README](README.md) - Full documentation
- [Test implementation](test_bioplausibility.py) - Source code
- [pytest documentation](https://docs.pytest.org/) - Pytest reference
