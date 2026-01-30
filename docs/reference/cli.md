# CLI Reference

Command-line interface for World Weaver.

## Installation

The CLI is included with World Weaver:

```bash
pip install world-weaver
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
World Weaver Status
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
ww serve [OPTIONS]
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
ww serve

# Custom port
ww serve -p 8080

# Development with reload
ww serve --reload

# Production with workers
ww serve -w 4
```

### config

Manage configuration.

```bash
ww config [OPTIONS]
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
ww config --show

# Create default config
ww config --init

# Show config path
ww config --path
```

### consolidate

Run memory consolidation.

```bash
ww consolidate [OPTIONS]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--full` | Run full consolidation |
| `--dry-run` | Preview without changes |

**Examples:**

```bash
# Quick consolidation
ww consolidate

# Full consolidation
ww consolidate --full

# Preview changes
ww consolidate --dry-run
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
| `WW_SESSION_ID` | Default session ID |
| `WW_QDRANT_HOST` | Qdrant host |
| `WW_NEO4J_URI` | Neo4j URI |
| `WW_API_KEY` | API key |

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
eval "$(_WW_COMPLETE=bash_source ww)"
```

### Zsh

```bash
# Add to ~/.zshrc
eval "$(_WW_COMPLETE=zsh_source ww)"
```

### Fish

```fish
# Add to ~/.config/fish/completions/ww.fish
_WW_COMPLETE=fish_source ww | source
```
