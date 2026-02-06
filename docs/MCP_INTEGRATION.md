# T4DM MCP Integration Guide

**Version**: 2.0.0
**Last Updated**: 2026-02-05
**Purpose**: Automatic memory for Claude Code and Claude Desktop

---

## Overview

T4DM provides **automatic memory** for Claude - no manual tool calls needed. Memory is:
- **Pre-loaded** via MCP resources at session start
- **Auto-captured** from Claude's outputs during sessions
- **Auto-consolidated** at session end

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Code                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              MCP Resource Layer                      │    │
│  │  memory://context    (auto-loaded at start)         │    │
│  │  memory://hot-cache  (frequently accessed)          │    │
│  │  memory://project/*  (project-specific)             │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              T4DM MCP Server                         │    │
│  │  - Hot cache (10 items, 0ms latency)                │    │
│  │  - Project context                                   │    │
│  │  - Auto-observation capture                          │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              T4DM API Server                         │    │
│  │              (localhost:8765)                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│                           ▼                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              T4DX Storage Engine                     │    │
│  │  (LSM + HNSW + Graph + Bitemporal)                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Start T4DM API Server

```bash
cd /mnt/projects/t4d/t4dm
source .venv/bin/activate
t4dm serve --port 8765
```

### 2. Configure Claude Code

The project includes `.mcp.json` which Claude Code auto-detects. Alternatively, add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "t4dm-memory": {
      "command": "python",
      "args": ["-m", "t4dm.mcp.server"],
      "env": {
        "PYTHONPATH": "/path/to/t4dm/src",
        "T4DM_API_URL": "http://localhost:8765",
        "T4DM_SESSION_ID": "my-session",
        "T4DM_PROJECT": "my-project",
        "T4DM_HOT_CACHE_SIZE": "10"
      }
    }
  }
}
```

### 3. Restart Claude Code

Claude Code will automatically:
- Start the MCP server
- Load `memory://context` resource
- Inject hot cache and project context

---

## How It Works

### Automatic Behaviors

| Behavior | Trigger | What Happens |
|----------|---------|--------------|
| **Context Load** | Session start | Hot cache + project memories injected |
| **Observation Capture** | Tool outputs | Patterns extracted and stored |
| **Session Summary** | Session end | Key decisions consolidated |
| **Hot Cache Update** | Memory access | Frequently used items promoted |

### Manual Tool (Only When Needed)

| Tool | Use Case |
|------|----------|
| `t4dm_remember` | User says "remember this", "save this", "don't forget" |

**Do NOT use** for routine observations - those are captured automatically.

---

## MCP Resources

### memory://context

Auto-loaded session context containing:
- Frequently referenced memories (hot cache)
- Project-specific patterns and decisions
- Recent session activity

```
## Frequently Referenced
- Authentication uses JWT with refresh tokens
- Database migrations run via alembic

## Project Context
- Using T4DX embedded storage, not PostgreSQL
- Spiking blocks have 6 stages

## Recent Session
- [2026-02-05T10:30:00] Fixed consolidation test failures
```

### memory://hot-cache

Fast-access cache (~10 items) of frequently used memories:
- Promoted when accessed 3+ times
- Demoted after 14 days unused
- 0ms retrieval latency

### memory://project/{name}

Project-specific memories:
- Architecture decisions
- Common patterns
- Past solutions

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `T4DM_API_URL` | `http://localhost:8765` | T4DM API server URL |
| `T4DM_SESSION_ID` | `claude-{uuid}` | Session identifier |
| `T4DM_PROJECT` | `default` | Current project name |
| `T4DM_HOT_CACHE_SIZE` | `10` | Items in hot cache |
| `T4DM_API_KEY` | (none) | Optional API key |

### Project-Specific Config

Create `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "t4dm-memory": {
      "command": "python",
      "args": ["-m", "t4dm.mcp.server"],
      "env": {
        "T4DM_PROJECT": "my-project-name"
      }
    }
  }
}
```

---

## Comparison: Before vs After

### Before (Manual Tools)

```
User: How do we handle auth?

Claude: Let me search my memory...
[calls t4dm_search with query="authentication"]
[waits for response]
[formats results]

Based on my search, we use JWT tokens.
```

**Problems:**
- User sees tool calls (breaks immersion)
- ~50ms latency per search
- Requires explicit prompting
- 7 different tools to manage

### After (Automatic)

```
User: How do we handle auth?

Claude: We use JWT with refresh tokens. The auth middleware
validates tokens on each request and the refresh endpoint
handles token renewal.
```

**Benefits:**
- No visible tool calls
- 0ms latency (pre-loaded)
- Context already available
- Single tool for explicit "remember this"

---

## Memory Lifecycle

### 1. Session Start

```
┌─────────────────────────────────────────┐
│           SessionStart Hook              │
├─────────────────────────────────────────┤
│  1. Load hot cache (10 items)           │
│  2. Load project context                │
│  3. Load recent session memories        │
│  4. Inject as memory://context          │
└─────────────────────────────────────────┘
```

### 2. During Session

```
┌─────────────────────────────────────────┐
│         Automatic Observation            │
├─────────────────────────────────────────┤
│  Claude generates output                 │
│           │                              │
│           ▼                              │
│  Pattern detection (decisions, errors)  │
│           │                              │
│           ▼                              │
│  Auto-store with low importance (0.3)   │
│           │                              │
│           ▼                              │
│  Usage promotes to hot cache            │
└─────────────────────────────────────────┘
```

### 3. Session End

```
┌─────────────────────────────────────────┐
│           SessionEnd Hook                │
├─────────────────────────────────────────┤
│  1. Summarize session activity          │
│  2. Extract key decisions               │
│  3. Store with high importance (0.8)    │
│  4. Trigger light consolidation         │
│  5. Update hot cache promotions         │
└─────────────────────────────────────────┘
```

---

## Integration with Claude Code Hooks

For full automation, add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "type": "command",
        "command": "echo 'T4DM: Context loaded'",
        "timeout": 1000
      }
    ],
    "PostToolUse": [
      {
        "type": "command",
        "command": "python3 -m t4dm.mcp.observe",
        "timeout": 2000,
        "onFailure": "ignore"
      }
    ]
  }
}
```

---

## Troubleshooting

### Memory not loading?

1. Check API server: `curl http://localhost:8765/api/v1/health`
2. Check MCP logs: `tail -f ~/.claude/logs/mcp-t4dm-memory.log`
3. Verify Python path in `.mcp.json`

### Stale context?

```bash
# Force hot cache refresh
curl -X POST http://localhost:8765/api/v1/consolidation/run
```

### Too much context?

Reduce hot cache size:
```json
{
  "env": {
    "T4DM_HOT_CACHE_SIZE": "5"
  }
}
```

---

## Best Practices

1. **Let memory be automatic** - Don't mention memory to users
2. **Use projects** - Set `T4DM_PROJECT` for context separation
3. **Trust the hot cache** - Frequently used items auto-promote
4. **Explicit only when asked** - Use `t4dm_remember` only for "remember this"
5. **Consolidate regularly** - Auto-triggered at session end

---

*Generated 2026-02-05*
