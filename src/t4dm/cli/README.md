# CLI Module

**2 files | ~400 lines | Centrality: 2**

The CLI module provides a command-line interface for World Weaver's tripartite memory system using Typer and Rich.

## Overview

```
ww (main command)
├── store         Store content in memory
├── recall        Search across memory types
├── consolidate   Optimize memory
├── status        Show system health
├── serve         Start REST API server
├── version       Show version
├── config        Manage configuration
│
├── episodic/     Episodic memory commands
│   ├── add       Add episode
│   ├── search    Search episodes
│   └── recent    List recent episodes
│
├── semantic/     Semantic memory commands
│   ├── add       Add entity
│   └── search    Search entities
│
└── procedural/   Procedural memory commands
    ├── add       Add skill
    └── search    Search skills
```

## Installation

```bash
# Install CLI entry point
pip install -e ".[cli]"

# Or directly
ww --help
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~390 | All CLI commands and logic |
| `__init__.py` | ~10 | Export Typer app |

## Core Commands

### `ww store` - Store Content

```bash
# Episodic (default)
ww store "Learned about decorator pattern" --importance 0.8

# Semantic entity
ww store "Python decorators" --type semantic --tags python,decorators

# Procedural skill
ww store "git add && git commit" --type procedural --tags git

# With metadata
ww store "API deployment" --metadata '{"env": "production"}'
```

**Options**:
- `--type`: episodic | semantic | procedural (default: episodic)
- `--importance`: 0-1 scale (default: 0.5)
- `--tags`: Comma-separated tags
- `--metadata`: JSON object

### `ww recall` - Search Memories

```bash
# Search all memory types
ww recall "Python decorators"

# Search specific type
ww recall "pytest syntax" --type episodic --k 10

# JSON output for scripting
ww recall "git workflow" --format json
```

**Options**:
- `--type`: episodic | semantic | procedural | all (default: all)
- `--k`: Number of results (default: 5)
- `--format`: table | json (default: table)

### `t4dm consolidate` - Memory Optimization

```bash
# Light consolidation (quick)
t4dm consolidate

# Deep consolidation (HDBSCAN clustering)
t4dm consolidate --full

# Preview without running
t4dm consolidate --dry-run
```

### `ww status` - System Health

```bash
ww status
```

**Output**:
```
World Weaver Status
┌──────────────────────┬─────────────────────────┐
│ Property             │ Value                   │
├──────────────────────┼─────────────────────────┤
│ Session ID           │ cli-session             │
│ Environment          │ development             │
│ Qdrant Host          │ localhost:6333          │
│ Neo4j URI            │ bolt://localhost:7687   │
│ Embedding Model      │ bge-m3                  │
└──────────────────────┴─────────────────────────┘
Memory services: Connected
```

### `t4dm serve` - Start API Server

```bash
# Development with auto-reload
t4dm serve --port 8765 --reload

# Production with workers
t4dm serve --host 0.0.0.0 --port 8765 --workers 4
```

### `ww config` - Configuration

```bash
# Show current config
t4dm config --show

# Initialize config file
t4dm config --init

# Custom path
t4dm config --init --path ~/.t4dm/custom.yaml
```

## Memory Type Commands

### Episodic (`ww episodic`)

```bash
# Add episode with emotional valence
ww episodic add "Debugged auth issue" --valence 0.9 --tags bug,auth

# Search episodes
ww episodic search "authentication" --k 10

# List recent episodes
ww episodic recent --limit 20
```

### Semantic (`ww semantic`)

```bash
# Add entity
ww semantic add "FastAPI" --desc "Modern web framework" --type tool

# Search entities
ww semantic search "web framework" --k 5
```

### Procedural (`ww procedural`)

```bash
# Add skill
ww procedural add "run_tests" --desc "Execute pytest suite"

# Search skills
ww procedural search "testing" --k 5
```

## Session Management

Set session ID via environment variable:

```bash
# Project-specific session
export WW_SESSION_ID="project-alpha"
ww store "Project-specific knowledge"

# Switch projects
export WW_SESSION_ID="project-beta"
ww recall "database queries"
```

## Usage Examples

### Daily Learning Log

```bash
# Store learning
ww store "Learned FSRS algorithm" --importance 0.9 --tags learning

# Consolidate nightly
t4dm consolidate --full

# Check recent
ww episodic recent --limit 20
```

### CI/CD Integration

```bash
# Check system health
ww status

# Log test run
ww store "Tests: 1245 passed, 8 failed" \
  --metadata '{"ci":"github_actions","run_id":"12345"}'

# Export for analysis
ww recall "test failures" --format json > failures.json
```

### Batch Export

```bash
# Export all memories
ww recall "" --type all --format json > backup.json

# Filter with jq
cat backup.json | jq '.[] | select(.type=="episodic")'
```

## Design Patterns

### Async-Sync Bridge

CLI commands are synchronous (Typer), but memory services are async:

```python
def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

# Usage in commands
run_async(_store_operation())
```

### Session Isolation

All operations scoped to session ID:

```python
session_id = os.environ.get("WW_SESSION_ID", "cli-session")
services = await get_services(session_id)
```

### Rich Output

All tables use Rich for styled terminal output:

```python
from rich.table import Table
table = Table(title="Results")
table.add_column("ID", style="cyan")
console.print(table)
```

## Entry Point

From `pyproject.toml`:

```ini
[project.scripts]
ww = "ww.cli.main:main"
```

## Testing

```bash
# Run CLI tests
pytest tests/cli/ -v

# Test specific command
pytest tests/cli/test_cli.py::TestStoreCommand -v
```

## Integration Points

- **Core Services**: `get_services(session_id)`
- **Core Types**: Episode, Entity, Procedure
- **Configuration**: `get_settings()`
- **Consolidation**: `get_consolidation_service()`
- **API Server**: `ww.api.server:app`
