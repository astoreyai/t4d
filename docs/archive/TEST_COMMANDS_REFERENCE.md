# World Weaver Test Commands Reference

Quick command reference for test execution, coverage analysis, and debugging.

## Basic Test Execution

```bash
cd /mnt/projects/ww
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run with brief output
pytest tests/ -q

# Run specific module
pytest tests/api/ -v
pytest tests/unit/ -v
pytest tests/nca/ -v
pytest tests/integration/ -v

# Run specific test file
pytest tests/api/test_bio_endpoints.py -v

# Run specific test class
pytest tests/api/test_bio_endpoints.py::TestNeuromodulatorTuningModel -v

# Run specific test
pytest tests/nca/test_glutamate_signaling.py::TestGlutamatePerformance::test_sustained_simulation -v
```

## Coverage Analysis

```bash
# Get overall coverage
pytest tests/ --cov=src/ww --cov-report=term-missing

# Coverage for specific module
pytest tests/ --cov=src/ww/visualization --cov-report=term-missing
pytest tests/ --cov=src/ww/prediction --cov-report=term-missing

# Generate HTML coverage report
pytest tests/ --cov=src/ww --cov-report=html
open htmlcov/index.html

# Quick coverage summary (API tests only)
pytest tests/api/ --cov=src/ww --cov-report=term-missing | tail -50
```

## Performance Testing

```bash
# Profile failing performance test
python -m cProfile -s cumtime -m pytest \
  tests/nca/test_glutamate_signaling.py::TestGlutamatePerformance::test_sustained_simulation

# Run benchmark tests
pytest tests/ -k "benchmark" -v

# Run with timing
pytest tests/ --durations=10
pytest tests/ --durations=0  # Show all test timings
```

## Debugging

```bash
# Stop on first failure
pytest tests/ -x

# Stop after N failures
pytest tests/ --maxfail=3

# Show local variables in tracebacks
pytest tests/ -l

# Full tracebacks
pytest tests/ --tb=long

# Short tracebacks (default)
pytest tests/ --tb=short

# No tracebacks
pytest tests/ --tb=no

# Print statements during test
pytest tests/ -s

# Run with debug logging
pytest tests/ --log-cli-level=DEBUG
```

## Test Discovery

```bash
# Count total tests
pytest tests/ --co -q | wc -l

# List all test names
pytest tests/ --co -q | head -50

# List specific category
pytest tests/visualization/ --co -q

# Find tests matching pattern
pytest tests/ --co -q -k "glutamate"
```

## Filter Tests

```bash
# Run tests matching name
pytest tests/ -k "test_prediction"

# Run excluding name
pytest tests/ -k "not performance"

# Run by nodeid
pytest tests/api/test_bio_endpoints.py::TestNeuromodulatorTuningModel
```

## Integration Test Specific

```bash
# Run only integration tests
pytest tests/integration/ -v

# Run failing integration test
pytest tests/integration/test_h10_cross_region_consistency.py -v
```

## Test Maintenance

```bash
# Show skipped tests
pytest tests/ -v -rs

# Show summary of failures
pytest tests/ --tb=short -rf

# Generate test report
pytest tests/ --html=report.html --self-contained-html

# JUnit XML report (for CI/CD)
pytest tests/ --junit-xml=junit.xml
```

## Quick Analysis

```bash
# API coverage only
pytest tests/api/ --cov=src/ww/api --cov-report=term-missing | tail -80

# Consolidation coverage
pytest tests/consolidation/ --cov=src/ww/consolidation --cov-report=term-missing | tail -50

# NCA coverage
pytest tests/nca/ --cov=src/ww/nca --cov-report=term-missing | tail -50

# Get list of untested files
pytest tests/ --cov=src/ww --cov-report=term-missing 2>&1 | grep "0%"
```

## Fixing the Critical Failure

```bash
# 1. Profile the failing test
python -m cProfile -s cumtime -m pytest \
  tests/nca/test_glutamate_signaling.py::TestGlutamatePerformance::test_sustained_simulation

# 2. Check recent commits
git log -p --all -- src/ww/nca/glutamate_signaling.py | head -300

# 3. Run after fix
pytest tests/nca/test_glutamate_signaling.py::TestGlutamatePerformance::test_sustained_simulation -v
```

## Adding New Tests

```bash
# Create visualization test file
mkdir -p tests/visualization
cat > tests/visualization/test_telemetry_hub.py << 'TESTEOF'
import pytest
import numpy as np
from ww.visualization.telemetry_hub import TelemetryHub

class TestTelemetryHub:
    def test_initialization(self):
        hub = TelemetryHub()
        assert hub is not None
TESTEOF

# Run new tests
pytest tests/visualization/ -v
```

## Reports

```bash
# Generate coverage trends
pytest tests/ --cov=src/ww --cov-report=term-missing > coverage_$(date +%Y%m%d).txt

# Find slowest tests
pytest tests/ --durations=20 -q

# Count by test category
pytest tests/ --co -q | cut -d: -f1 | sort | uniq -c | sort -rn

# Test health check
pytest tests/ -q --tb=no 2>&1 | tail -5
```

## Key Metrics

```bash
# Pass rate
pytest tests/ -q --tb=no 2>&1 | tail -1

# Coverage percentage
pytest tests/ --cov=src/ww --cov-report=term-missing 2>&1 | tail -3

# Test count
pytest tests/ --co -q 2>&1 | wc -l

# Execution time
time pytest tests/ -q --tb=no
```

## CI/CD Integration

```bash
# Run test suite like CI would
pytest tests/ -v --tb=short --junit-xml=junit.xml --cov=src/ww --cov-report=term-missing

# Check what would break on commit
pytest tests/ -x --tb=short -q

# Pre-push check
pytest tests/ -x -q
```

## Resources

- Full Analysis Report: `/mnt/projects/ww/TEST_ANALYSIS_REPORT.md`
- Test Templates: `/mnt/projects/ww/TEST_TEMPLATES.md`
- pytest Docs: https://docs.pytest.org/
- Coverage Docs: https://coverage.readthedocs.io/
