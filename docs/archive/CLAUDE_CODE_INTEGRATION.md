# World Weaver + Claude Code CLI Integration

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Code CLI                           │
├─────────────────────────────────────────────────────────────┤
│  SessionStart Hook          SessionEnd Hook                  │
│       ↓                           ↓                          │
│  ┌─────────┐                ┌─────────┐                     │
│  │Context  │                │Synthesis│                     │
│  │ Load    │                │& Store  │                     │
│  └────┬────┘                └────┬────┘                     │
│       ↓                          ↓                          │
├───────┴──────────────────────────┴──────────────────────────┤
│                    MCP Server (ww)                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ Episodic │  │ Semantic │  │Procedural│                  │
│  │  Memory  │  │  Memory  │  │  Memory  │                  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       └──────────────┼──────────────┘                       │
│                      ↓                                       │
│              ┌──────────────┐                               │
│              │ Unified API  │                               │
│              └──────┬───────┘                               │
│                     ↓                                        │
│        ┌───────────┴───────────┐                            │
│        ↓                       ↓                            │
│   ┌─────────┐            ┌─────────┐                        │
│   │ Qdrant  │            │  Neo4j  │                        │
│   │(vectors)│            │ (graph) │                        │
│   └─────────┘            └─────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## Hook Configuration

### SessionStart Hook

Location: `~/.claude/settings.json`

```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "python3 ~/.claude/skills/ww-session/session_start.py",
        "timeout": 15
      }]
    }]
  }
}
```

**What it does:**
1. Loads recent episodes from previous sessions
2. Spreads activation to prime relevant semantic entities
3. Surfaces procedures matching current working directory
4. Returns context blob for LLM injection

### SessionEnd Hook

```json
{
  "hooks": {
    "SessionEnd": [{
      "matcher": "*",
      "hooks": [{
        "type": "command",
        "command": "python3 ~/.claude/skills/ww-session/session_end.py",
        "timeout": 180
      }]
    }]
  }
}
```

**What it does:**
1. Persists key entities retrieved/created this session
2. Records unresolved tasks as procedural trigger patterns
3. Triggers lightweight consolidation
4. Updates FSRS stability for accessed memories

## MCP Server Integration

### Starting the Server

```bash
# Direct start
python -m ww.mcp.server

# Via docker
docker-compose -f docker-compose.full.yml up -d
```

### Claude Desktop Configuration

`~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "world-weaver": {
      "command": "python",
      "args": ["-m", "ww.mcp.server"],
      "cwd": "/mnt/projects/ww"
    }
  }
}
```

## Memory Operations

### Automatic (Hook-Driven)

| Operation | Trigger | Implementation |
|-----------|---------|----------------|
| Episode creation | Every tool call | Post-tool hook |
| Hebbian updates | Co-retrieval | Automatic in recall |
| Decay updates | Memory access | FSRS in retrieval path |
| Context loading | Session start | SessionStart hook |
| Consolidation | Session end / nightly | Scheduled + hook |

### Explicit (User-Invoked)

| Operation | MCP Tool | When to Use |
|-----------|----------|-------------|
| `ww_remember` | create_episode | Important learnings |
| `ww_entity` | create_entity | Named concepts |
| `ww_skill` | create_procedure | Reusable workflows |
| `ww_recall` | recall_episodes | Search memories |
| `ww_context` | get_context | Prime for task |

## Context Bridging Protocol

### Project-Based Scoping

```python
# On entering a project directory
async def on_project_enter(project_path: str):
    project_name = extract_project_name(project_path)

    # Load project-specific context
    context = await unified_memory.get_context(
        query=f"working on {project_name}",
        context={"project": project_name},
        include_episodes=True,
        include_entities=True,
        include_skills=True
    )

    return context
```

### Session Handoff

```python
# At session end
handoff = {
    "key_entities": [e.id for e in entities_this_session],
    "unresolved_tasks": extract_unresolved_tasks(conversation),
    "modified_weights": collect_hebbian_updates(),
    "project": current_project,
    "timestamp": now()
}
await episodic.create(content=json.dumps(handoff),
                      outcome="neutral",
                      importance=0.8)
```

## Hinton-Recommended Improvements

### Priority 1: Learned Retrieval Scoring

Current (hardcoded):
```python
total_score = (
    0.6 * similarity +
    0.3 * recency_score +
    0.1 * importance_score
)
```

Recommended (learned):
```python
features = torch.tensor([similarity, recency, importance, outcome_history])
total_score = scoring_model(features)  # Train on click/success data
```

### Priority 2: Implicit Feedback Collection

Add to every retrieval:
```python
async def recall_with_feedback(query, ...):
    results = await recall(query, ...)

    # Record what was shown
    await log_retrieval_event(
        query=query,
        results=[r.id for r in results],
        timestamp=now()
    )

    return results

# When outcome known
async def record_outcome(task_id, success: bool):
    retrievals = get_retrievals_for_task(task_id)
    for retrieval in retrievals:
        await update_feedback(retrieval, positive=success)
```

### Priority 3: Cross-Memory Influence

```python
# Episodes should prime semantic
async def recall_episodes_primed(query, ...):
    # First get semantic context
    entities = await semantic.recall(query, limit=5)

    # Use entity names to enrich query
    enriched_query = f"{query} {' '.join(e.name for e in entities)}"

    return await episodic.recall(enriched_query, ...)
```

## Agent Registration

The `ww-hinton` agent is now available at:
```
~/.claude/agents/ww-hinton.md
```

Use via Task tool:
```python
Task(subagent_type="ww-hinton", prompt="Analyze memory system...")
```

## Recommended Hook Points

| Hook | Purpose | Implementation |
|------|---------|----------------|
| Pre-tool | Inject relevant procedures | Query procedural memory |
| Post-tool | Record outcome | Create episode |
| On-error | Learn from failure | Negative episode + Hebbian weakening |
| On-commit | Extract skills | Trajectory → procedure promotion |
| Session-start | Load context | Spread activation |
| Session-end | Consolidate | Cluster + entity extraction |

## Status

- [x] MCP server implemented
- [x] Hook infrastructure exists
- [ ] SessionStart hook for WW (needs implementation)
- [ ] SessionEnd hook for WW (needs implementation)
- [ ] Implicit feedback collection (not implemented)
- [ ] Learned scoring model (not implemented)
- [ ] Cross-memory priming (partial)
