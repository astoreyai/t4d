# CLI Reference

Command-line interface for T4DM.

## Installation

The CLI is included with T4DM:

```bash
pip install t4dm
```

## Global Options

```bash
ww [OPTIONS] COMMAND [ARGS]
```

| Option | Description |
|--------|-------------|
| `--help` | Show help message |
| `--version` | Show version |

## Commands

### store

Store a memory.

```bash
ww store "Your memory content here" [OPTIONS]
```

**Options:**

| Option | Type | Description |
|--------|------|-------------|
| `--importance`, `-i` | float | Importance (0.0-1.0) |
| `--tags`, `-t` | str | Comma-separated tags |
| `--type` | str | Memory type (episodic/semantic/procedural) |
| `--metadata`, `-m` | str | JSON metadata |

**Examples:**

```bash
# Basic store
ww store "Learned about CLI tools"

# With importance and tags
ww store "Important discovery" -i 0.9 -t "important,discovery"

# With metadata
ww store "Bug fix" -m '{"file": "cli.py", "line": 42}'
```

### recall

Search memories.

```bash
ww recall "search query" [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--k`, `-k` | int | 5 | Number of results |
| `--type` | str | all | Memory type filter |
| `--format`, `-f` | str | table | Output format (table/json/simple) |

**Examples:**

```bash
# Basic recall
ww recall "Python decorators"

# More results
ww recall "machine learning" -k 20

# JSON output
ww recall "API design" -f json
```

### status

Show system status.

```bash
ww status
```

**Output:**

```
T4DM Status
==================
Version: 0.4.0
Session: default

Storage:
  Neo4j: connected
  Qdrant: connected

Memory Counts:
  Episodic: 1,234
  Semantic: 456
  Procedural: 78

Health: healthy
```

### serve

Start the REST API server.

```bash
t4dm serve [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--host` | str | 0.0.0.0 | Bind host |
| `--port`, `-p` | int | 8765 | Bind port |
| `--workers`, `-w` | int | 1 | Number of workers |
| `--reload` | flag | - | Enable auto-reload |

**Examples:**

```bash
# Default server
t4dm serve

# Custom port
t4dm serve -p 8080

# Development with reload
t4dm serve --reload

# Production with workers
t4dm serve -w 4
```

### config

Manage configuration.

```bash
t4dm config [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--show` | Show current configuration |
| `--init` | Initialize default config file |
| `--path` | Show config file path |

**Examples:**

```bash
# Show current config
t4dm config --show

# Create default config
t4dm config --init

# Show config path
t4dm config --path
```

### consolidate

Run memory consolidation.

```bash
t4dm consolidate [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--full` | Run full consolidation |
| `--dry-run` | Preview without changes |

**Examples:**

```bash
# Quick consolidation
t4dm consolidate

# Full consolidation
t4dm consolidate --full

# Preview changes
t4dm consolidate --dry-run
```

### version

Show version information.

```bash
ww version
```

## Subcommands

### episodic

Episodic memory operations.

```bash
ww episodic COMMAND [OPTIONS]
```

**Commands:**

```bash
# Add episode
ww episodic add "Content" --valence 0.8 --tags "tag1,tag2"

# Search episodes
ww episodic search "query" --k 10

# List recent
ww episodic recent --limit 20

# Get by ID
ww episodic get <episode_id>

# Delete
ww episodic delete <episode_id>
```

### semantic

Semantic memory operations.

```bash
ww semantic COMMAND [OPTIONS]
```

**Commands:**

```bash
# Add entity
ww semantic add "Entity Name" --desc "Description" --type concept

# Search entities
ww semantic search "query" --k 10

# List all
ww semantic list --type concept

# Get by ID
ww semantic get <entity_id>

# Create relationship
ww semantic relate <source_id> <target_id> --type RELATED_TO
```

### procedural

Procedural memory operations.

```bash
ww procedural COMMAND [OPTIONS]
```

**Commands:**

```bash
# Add skill
ww procedural add "skill_name" --desc "Description" --domain coding

# Search skills
ww procedural search "query" --k 5

# How-to query
ww procedural howto "run the tests"

# List by domain
ww procedural list --domain coding

# Record execution
ww procedural execute <skill_id> --success --duration 1500
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `T4DM_SESSION_ID` | Default session ID |
| `T4DM_QDRANT_HOST` | Qdrant host |
| `T4DM_NEO4J_URI` | Neo4j URI |
| `T4DM_API_KEY` | API key |

## Output Formats

### Table (default)

```
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Type       ┃ Content                       ┃ Score ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ episodic   │ Learned about CLI tools...    │ 0.92  │
│ semantic   │ CLI: Command-line interface   │ 0.85  │
│ procedural │ run_cli: Execute CLI command  │ 0.78  │
└────────────┴───────────────────────────────┴───────┘
```

### JSON

```bash
ww recall "query" -f json
```

```json
[
  {
    "type": "episodic",
    "content": "Learned about CLI tools...",
    "score": 0.92,
    "id": "..."
  }
]
```

### Simple

```bash
ww recall "query" -f simple
```

```
[0.92] episodic: Learned about CLI tools...
[0.85] semantic: CLI: Command-line interface
[0.78] procedural: run_cli: Execute CLI command
```

## Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| 1 | General error |
| 2 | Configuration error |
| 3 | Connection error |
| 4 | Not found |

## Shell Completion

### Bash

```bash
# Add to ~/.bashrc
eval "$(_T4DM_COMPLETE=bash_source ww)"
```

### Zsh

```bash
# Add to ~/.zshrc
eval "$(_T4DM_COMPLETE=zsh_source ww)"
```

### Fish

```fish
# Add to ~/.config/fish/completions/t4dm.fish
_T4DM_COMPLETE=fish_source ww | source
```
