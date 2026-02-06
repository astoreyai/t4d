# Makefile Integration Guide

This guide shows how to integrate the pytest benchmark tests into your Makefile.

## Current Makefile Test Targets

The Makefile already has test targets in `/mnt/projects/t4d/t4dm/Makefile`:

```makefile
test:
	$(PYTHON) -m pytest tests/ -v

test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow and not benchmark"

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src/t4dm --cov-report=term-missing --cov-report=html

test-integration:
	$(PYTHON) -m pytest tests/integration/ -v
```

Note: `test-fast` already excludes benchmarks with `-m "not slow and not benchmark"`

## Proposed Benchmark Targets

Add these targets to the Makefile:

```makefile
# Benchmark tests
test-benchmarks:
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark -v

test-benchmarks-bio:
	$(PYTHON) -m pytest tests/benchmarks/test_bioplausibility.py -m bioplausibility -v

test-benchmarks-memory:
	$(PYTHON) -m pytest tests/benchmarks/test_longmemeval.py -m memory -v

test-benchmarks-dmr:
	$(PYTHON) -m pytest tests/benchmarks/test_dmr.py -m retrieval -v

test-benchmarks-quick:
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark --tb=short -q
```

## Installation Steps

### Step 1: Update Makefile Dependencies

Add benchmark targets to the `.PHONY` declaration (line 4):

```makefile
.PHONY: help start stop restart health logs test lint clean deps api mcp docker-up docker-down install dev-install \
        frontend frontend-bg frontend-build frontend-install frontend-logs frontend-stop \
        up up-all down down-all status stats dev \
        test-benchmarks test-benchmarks-bio test-benchmarks-memory test-benchmarks-dmr test-benchmarks-quick
```

### Step 2: Add Benchmark Test Targets

Add these lines after the existing test targets (after line 262):

```makefile
# Benchmark tests
test-benchmarks:
	@echo "Running all benchmark tests..."
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark -v

test-benchmarks-bio:
	@echo "Running bio-plausibility benchmarks..."
	$(PYTHON) -m pytest tests/benchmarks/test_bioplausibility.py -m bioplausibility -v

test-benchmarks-memory:
	@echo "Running memory evaluation benchmarks..."
	$(PYTHON) -m pytest tests/benchmarks/test_longmemeval.py -m memory -v

test-benchmarks-dmr:
	@echo "Running deep memory retrieval benchmarks..."
	$(PYTHON) -m pytest tests/benchmarks/test_dmr.py -m retrieval -v

test-benchmarks-quick:
	@echo "Running quick benchmark check..."
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark --tb=short -q
```

### Step 3: Update Help Text

Update the help target to include the new commands (around line 32-34):

```makefile
	@echo "Development:"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run tests without slow markers"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-benchmarks      Run all benchmark tests (51 tests)"
	@echo "  make test-benchmarks-bio  Run bio-plausibility benchmarks (16 tests)"
	@echo "  make test-benchmarks-memory Run memory evaluation benchmarks (20 tests)"
	@echo "  make test-benchmarks-dmr  Run deep memory retrieval benchmarks (15 tests)"
	@echo "  make test-benchmarks-quick Run quick benchmark check"
	@echo "  make lint          Run linters (ruff, mypy)"
```

## Usage Examples

### Run all benchmarks
```bash
make test-benchmarks
```

### Run only bio-plausibility
```bash
make test-benchmarks-bio
```

### Run specific benchmark type
```bash
make test-benchmarks-memory
make test-benchmarks-dmr
```

### Quick check before commit
```bash
make test-benchmarks-quick
```

### Include in CI/CD pipeline
```bash
make test           # All tests
make test-benchmarks # Just benchmarks
make test-fast      # Fast tests (excludes benchmarks)
```

## Advanced Usage

### With Coverage
```makefile
test-benchmarks-cov:
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark -v \
		--cov=src/t4dm --cov-report=term-missing --cov-report=html
```

### CI/CD Variants
```makefile
test-benchmarks-ci:
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark \
		--tb=short -v --junit-xml=test-benchmarks.xml

test-benchmarks-parallel:
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark \
		-v -n auto
```

### With Performance Profiling
```makefile
test-benchmarks-profile:
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark \
		-v --durations=10 --profile
```

## Integration with Existing Targets

### test-fast (Already Updated)
The existing `test-fast` target already excludes benchmarks:
```makefile
test-fast:
	$(PYTHON) -m pytest tests/ -v -m "not slow and not benchmark"
```

This means:
- `make test` - Runs ALL tests including benchmarks
- `make test-fast` - Runs fast tests, skips benchmarks
- `make test-benchmarks` - Runs ONLY benchmarks
- `make test-cov` - Runs ALL tests with coverage (excludes benchmarks with markers)

### Combining Targets
You can create composite targets:

```makefile
test-all:
	@echo "Running comprehensive test suite..."
	$(PYTHON) -m pytest tests/unit/ -v
	$(PYTHON) -m pytest tests/integration/ -v
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark -v

test-suite:
	make test-fast
	make test-benchmarks
	make lint
```

## Makefile Best Practices

### 1. Always use `@echo` for user feedback
```makefile
test-benchmarks:
	@echo "Running benchmark tests..."
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark -v
```

### 2. Use `.PHONY` for non-file targets
Ensure targets are in `.PHONY` declaration

### 3. Quote complex commands
```makefile
test-benchmarks-complex:
	$(PYTHON) -m pytest tests/benchmarks/ \
		-m benchmark -v \
		--tb=short \
		--maxfail=3
```

### 4. Support quiet mode
```makefile
test-benchmarks-quiet:
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark -q
```

## Troubleshooting

### Target not found
```bash
# Make sure target is in .PHONY
# Make sure Makefile is saved
make help | grep benchmark
```

### Wrong Python version
```makefile
# Ensure PYTHON variable is set correctly
PYTHON ?= python3
PYTHON ?= python3.11
```

### Path issues
```bash
# Add path if needed
make test-benchmarks PYTHONPATH=/mnt/projects/t4d/t4dm/src:$$PYTHONPATH
```

## Example Complete Makefile Section

Here's a complete section to add to your Makefile:

```makefile
# ============================================================================
# BENCHMARK TESTS
# ============================================================================

test-benchmarks: ## Run all benchmark tests (51 tests)
	@echo "🔬 Running all benchmark tests..."
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark -v

test-benchmarks-bio: ## Run bio-plausibility benchmarks (16 tests)
	@echo "🧠 Running bio-plausibility benchmarks..."
	$(PYTHON) -m pytest tests/benchmarks/test_bioplausibility.py -m bioplausibility -v

test-benchmarks-memory: ## Run memory evaluation benchmarks (20 tests)
	@echo "💾 Running memory evaluation benchmarks..."
	$(PYTHON) -m pytest tests/benchmarks/test_longmemeval.py -m memory -v

test-benchmarks-dmr: ## Run deep memory retrieval benchmarks (15 tests)
	@echo "🔍 Running deep memory retrieval benchmarks..."
	$(PYTHON) -m pytest tests/benchmarks/test_dmr.py -m retrieval -v

test-benchmarks-quick: ## Quick benchmark sanity check
	@echo "⚡ Running quick benchmark check..."
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark --tb=short -q

test-benchmarks-cov: ## Run benchmarks with coverage
	@echo "🔬 Running benchmarks with coverage..."
	$(PYTHON) -m pytest tests/benchmarks/ -m benchmark -v \
		--cov=src/t4dm --cov-report=term-missing --cov-report=html
```

## CI/CD Pipeline Example

### GitHub Actions
```yaml
- name: Run benchmarks
  run: make test-benchmarks

- name: Run bio-plausibility
  run: make test-benchmarks-bio

- name: Generate coverage
  run: make test-benchmarks-cov
```

### GitLab CI
```yaml
benchmark:
  script:
    - make test-benchmarks
  artifacts:
    reports:
      junit: test-benchmarks.xml
```

### Jenkins
```groovy
stage('Benchmarks') {
    steps {
        sh 'make test-benchmarks'
    }
}
```

## Summary

After integration, you'll have:

```bash
make test              # All tests (fast + benchmarks)
make test-fast        # Fast tests only
make test-benchmarks  # Benchmarks only
make test-cov         # All tests with coverage
```

This gives you flexible test execution matching your development workflow.
