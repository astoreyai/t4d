# World Weaver Plugin Architecture Plan

**Date**: 2025-11-27
**Status**: Proposed

## Executive Summary

Restructure World Weaver from a standalone MCP server into a **Claude Code plugin** that provides:
- Skills for memory operations
- Hooks for automatic memory capture
- Slash commands for quick access
- Agents for complex orchestration

The MCP server remains as the storage backend; the plugin adds a UX layer.

## Current vs Proposed Architecture

### Current Architecture
```
┌─────────────────────────────────────────────┐
│              Claude Code CLI                 │
│                    │                         │
│           MCP Tool Calls                     │
│                    ▼                         │
│         ┌─────────────────┐                 │
│         │  WW MCP Server  │                 │
│         │ (ww-memory cmd) │                 │
│         └────────┬────────┘                 │
│                  │                           │
│     ┌────────────┴────────────┐             │
│     ▼                         ▼             │
│  ┌──────┐               ┌─────────┐         │
│  │Neo4j │               │ Qdrant  │         │
│  └──────┘               └─────────┘         │
└─────────────────────────────────────────────┘
```

**Problems:**
- Requires manual MCP tool calls (`mcp__ww-memory__create_episode`)
- No automatic memory capture
- No slash command shortcuts
- Agents in `.claude/agents/` aren't connected to MCP tools

### Proposed Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Claude Code CLI                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                World Weaver Plugin                       ││
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       ││
│  │  │ Skills  │ │  Hooks  │ │Commands │ │ Agents  │       ││
│  │  │ww-store │ │SessionS │ │/remember│ │ww-memory│       ││
│  │  │ww-recall│ │SessionE │ │/recall  │ │ww-retr. │       ││
│  │  │ww-ctx   │ │PostTool │ │/context │ │ww-synth │       ││
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       ││
│  │       └───────────┴───────────┴───────────┘             ││
│  │                          │                               ││
│  │                    MCP Tool Calls                        ││
│  └──────────────────────────┬──────────────────────────────┘│
│                             ▼                                │
│                   ┌─────────────────┐                       │
│                   │  WW MCP Server  │ (unchanged)           │
│                   └────────┬────────┘                       │
│              ┌─────────────┴─────────────┐                  │
│              ▼                           ▼                  │
│         ┌──────┐                   ┌─────────┐              │
│         │Neo4j │                   │ Qdrant  │              │
│         └──────┘                   └─────────┘              │
└─────────────────────────────────────────────────────────────┘
```

## Plugin Structure

```
world-weaver-plugin/
├── plugin.json                 # Plugin manifest
├── README.md                   # Installation guide
├── CHANGELOG.md
│
├── skills/
│   ├── ww-store/
│   │   └── SKILL.md           # Store episode/entity/skill
│   ├── ww-recall/
│   │   └── SKILL.md           # Multi-strategy memory recall
│   ├── ww-context/
│   │   └── SKILL.md           # Build context from memories
│   └── ww-consolidate/
│       └── SKILL.md           # Trigger memory consolidation
│
├── hooks/
│   ├── session_start.py       # Load context from WW
│   ├── session_end.py         # Store session summary
│   └── post_tool_use.py       # Learn from tool patterns
│
├── commands/
│   ├── remember.md            # /remember [content]
│   ├── recall.md              # /recall [query]
│   ├── context.md             # /context
│   ├── forget.md              # /forget [id]
│   └── consolidate.md         # /consolidate
│
├── agents/
│   ├── ww-memory.md           # Direct memory interface
│   ├── ww-retriever.md        # Multi-strategy retrieval
│   └── ww-synthesizer.md      # Cross-memory synthesis
│
└── scripts/
    ├── install.py             # Plugin installation
    ├── check_mcp.py           # Verify MCP server running
    └── requirements.txt       # Plugin dependencies
```

## Component Specifications

### 1. Skills

#### ww-store
```markdown
---
name: ww-store
description: Store memories in World Weaver (episodes, entities, skills)
allowed-tools: ['mcp__ww-memory__*']
---

# WW Store Skill

Store information in World Weaver's tripartite memory system.

## Capabilities

1. **Episode Storage**: Store autobiographical events
   - Extract context (project, file, tool)
   - Classify outcome (success/failure/partial)
   - Assess importance (valence 0-1)

2. **Entity Creation**: Add knowledge graph nodes
   - Classify entity type (CONCEPT, PERSON, PLACE, etc.)
   - Generate summary and details
   - Link to existing entities

3. **Skill Recording**: Store successful patterns
   - Extract procedure steps
   - Define pre/post conditions
   - Add parameter specifications

## Workflow

When storing an episode:
1. Call mcp__ww-memory__create_episode with content, outcome, valence
2. Extract mentioned entities, call mcp__ww-memory__create_entity for each
3. Create relations between entities
4. Return storage confirmation with IDs
```

#### ww-recall
```markdown
---
name: ww-recall
description: Recall memories using multi-strategy retrieval
allowed-tools: ['mcp__ww-memory__*']
---

# WW Recall Skill

Multi-strategy memory retrieval combining all memory subsystems.

## Strategies

1. **Episodic Recall**: Recent experiences
   - Semantic similarity to query
   - Temporal decay weighting
   - Outcome filtering

2. **Semantic Recall**: Knowledge graph
   - Vector similarity search
   - Spread activation from seed entities
   - Relationship traversal

3. **Procedural Recall**: Applicable skills
   - Match query to skill triggers
   - Check preconditions against context
   - Rank by success rate

## Workflow

1. Embed query using mcp__ww-memory__* tools
2. Search episodes: mcp__ww-memory__recall_episodes
3. Search entities: mcp__ww-memory__semantic_recall
4. Search skills: mcp__ww-memory__recall_skill
5. Merge and rank results
6. Format as context block for Claude
```

### 2. Hooks

#### session_start.py
```python
#!/usr/bin/env python3
"""
SessionStart hook: Load memory context into Claude Code session.

Fetches recent episodes, relevant entities, and applicable skills
based on current working directory and project context.
"""

import json
import subprocess
import os
from datetime import datetime, timedelta

def get_session_context():
    """Fetch context from WW MCP server."""
    cwd = os.getcwd()
    project = os.path.basename(cwd)

    # Get recent episodes (last 7 days)
    episodes = call_mcp_tool("recall_episodes", {
        "query": f"working on {project}",
        "limit": 10,
        "time_filter": {
            "after": (datetime.now() - timedelta(days=7)).isoformat()
        }
    })

    # Get related entities
    entities = call_mcp_tool("semantic_recall", {
        "query": project,
        "limit": 10
    })

    # Get applicable skills
    skills = call_mcp_tool("recall_skill", {
        "query": f"how to work with {project}",
        "limit": 5
    })

    return format_context(episodes, entities, skills)

def call_mcp_tool(tool: str, params: dict) -> dict:
    """Call WW MCP tool and return result."""
    # Implementation uses subprocess to call MCP server
    pass

def format_context(episodes, entities, skills) -> str:
    """Format memory context for injection."""
    return f"""
## Memory Context (World Weaver)

### Recent Episodes ({len(episodes)} found)
{format_episodes(episodes)}

### Related Knowledge ({len(entities)} entities)
{format_entities(entities)}

### Applicable Skills ({len(skills)} found)
{format_skills(skills)}
"""

if __name__ == "__main__":
    context = get_session_context()
    print(context)
```

#### session_end.py
```python
#!/usr/bin/env python3
"""
SessionEnd hook: Store session summary as episode.

Extracts completed tasks, git activity, and conversation summary
to create an autobiographical episode in World Weaver.
"""

import json
import subprocess
import os
from datetime import datetime

def store_session_episode():
    """Store current session as an episode."""
    # Gather session data
    session_data = {
        "cwd": os.getcwd(),
        "todos": get_completed_todos(),
        "git_activity": get_git_activity(),
        "duration": get_session_duration(),
    }

    # Create episode content
    content = format_session_content(session_data)

    # Determine outcome
    outcome = "success" if session_data["todos"]["completed"] > 0 else "neutral"

    # Calculate importance (more completed = more important)
    valence = min(1.0, session_data["todos"]["completed"] * 0.2)

    # Store episode
    result = call_mcp_tool("create_episode", {
        "content": content,
        "outcome": outcome,
        "emotional_valence": valence,
        "context": {
            "project": os.path.basename(session_data["cwd"]),
            "working_directory": session_data["cwd"],
        }
    })

    # Extract and store entities mentioned
    entities = extract_entities(content)
    for entity in entities:
        call_mcp_tool("create_entity", entity)

    return result

if __name__ == "__main__":
    result = store_session_episode()
    print(f"Session stored: {result}")
```

### 3. Slash Commands

#### /remember
```markdown
# /remember Command

Quick episode storage. Store the current context or specified content.

## Usage

```
/remember                    # Summarize recent conversation
/remember Fixed batch query  # Store specific content
/remember --important        # Mark as high importance
```

## Implementation

When invoked:
1. If content provided, use it directly
2. If not, summarize recent conversation (last 10 messages)
3. Extract context from working directory
4. Call ww-store skill to create episode
5. Report what was stored
```

#### /recall
```markdown
# /recall Command

Search World Weaver memories.

## Usage

```
/recall Neo4j queries        # Search for Neo4j-related memories
/recall --episodes only      # Only episodic memories
/recall --skills python      # Only procedural skills
/recall --graph concept      # Only semantic entities
```

## Implementation

When invoked:
1. Parse query and filters
2. Call ww-recall skill with parameters
3. Format results as context block
4. Inject into conversation
```

### 4. Agents

#### ww-memory
```markdown
---
name: ww-memory
description: Direct interface to World Weaver memory systems
tools: ['mcp__ww-memory__*', 'Read', 'Write']
model: haiku
---

You are the World Weaver memory agent. You have direct access to all
memory MCP tools and can perform complex memory operations.

## Capabilities

1. Store episodes, entities, and skills
2. Recall memories using any strategy
3. Update and strengthen memories
4. Trigger consolidation
5. Query memory statistics

## When to Use

- Complex memory operations requiring multiple tool calls
- Memory maintenance and cleanup
- Debugging memory issues
- Bulk memory operations
```

#### ww-retriever
```markdown
---
name: ww-retriever
description: Multi-strategy memory retrieval specialist
tools: ['mcp__ww-memory__*', 'Read']
model: haiku
---

You are the World Weaver retrieval agent. You specialize in finding
the most relevant memories using optimal retrieval strategies.

## Strategies

1. **Semantic Search**: Vector similarity
2. **Graph Traversal**: Spread activation
3. **Temporal Search**: Time-based filtering
4. **Hybrid Fusion**: Combine multiple strategies

## Workflow

1. Analyze query to determine optimal strategy
2. Execute retrieval with appropriate tools
3. Re-rank results based on relevance
4. Format context for main conversation
```

## Integration with Existing Hooks

World Weaver plugin hooks should **extend** Aaron's existing hooks:

### session-initializer Enhancement
```python
# In existing session_start.py, add:

from ww_plugin.hooks.session_start import get_session_context

def enhanced_session_start():
    # Existing: PhD countdown, git status, todos
    existing_output = get_existing_context()

    # New: WW memory context
    try:
        ww_context = get_session_context()
    except Exception:
        ww_context = ""  # Graceful degradation

    return existing_output + ww_context
```

### session-synthesizer Enhancement
```python
# In existing session_end.py, add:

from ww_plugin.hooks.session_end import store_session_episode

def enhanced_session_end():
    # Existing: Obsidian sync, todos, git check
    existing_result = run_existing_synthesis()

    # New: Store session in WW
    try:
        ww_result = store_session_episode()
    except Exception:
        ww_result = None  # Graceful degradation

    return combine_results(existing_result, ww_result)
```

## Implementation Phases

### Phase 1: Core Plugin (Week 1)
- [ ] Create plugin.json manifest
- [ ] Implement ww-store skill
- [ ] Implement ww-recall skill
- [ ] Create /remember command
- [ ] Create /recall command
- [ ] Write installation script
- [ ] Test with existing MCP server

### Phase 2: Hooks Integration (Week 2)
- [ ] Implement session_start.py hook
- [ ] Implement session_end.py hook
- [ ] Integrate with session-initializer
- [ ] Integrate with session-synthesizer
- [ ] Test automatic memory capture

### Phase 3: Advanced Features (Week 3)
- [ ] Create ww-context skill
- [ ] Create ww-consolidate skill
- [ ] Implement post_tool_use.py hook
- [ ] Create /context command
- [ ] Create /forget command
- [ ] Create /consolidate command

### Phase 4: Agents & Polish (Week 4)
- [ ] Create ww-memory agent
- [ ] Create ww-retriever agent
- [ ] Create ww-synthesizer agent
- [ ] Write comprehensive documentation
- [ ] Add to claude-skills repository
- [ ] Test full integration

## Configuration

### plugin.json
```json
{
  "name": "world-weaver",
  "version": "1.0.0",
  "description": "Tripartite neural memory system for Claude Code",
  "author": "Aaron Storey",
  "requires": {
    "mcpServers": ["ww-memory"]
  },
  "skills": [
    "skills/ww-store",
    "skills/ww-recall",
    "skills/ww-context",
    "skills/ww-consolidate"
  ],
  "hooks": {
    "SessionStart": "hooks/session_start.py",
    "SessionEnd": "hooks/session_end.py",
    "PostToolUse": "hooks/post_tool_use.py"
  },
  "commands": [
    "commands/remember.md",
    "commands/recall.md",
    "commands/context.md",
    "commands/forget.md",
    "commands/consolidate.md"
  ],
  "agents": [
    "agents/ww-memory.md",
    "agents/ww-retriever.md",
    "agents/ww-synthesizer.md"
  ]
}
```

### Settings Integration
```json
{
  "mcpServers": {
    "ww-memory": {
      "command": "/mnt/projects/t4d/t4dm/.venv/bin/python",
      "args": ["-m", "ww.mcp.server"],
      "env": {
        "WW_SESSION_ID": "default",
        "WW_NEO4J_URI": "bolt://localhost:7687"
      }
    }
  },
  "enabledPlugins": {
    "world-weaver@astoreyai": true
  }
}
```

## Benefits Summary

| Aspect | Current | With Plugin |
|--------|---------|-------------|
| Memory Storage | Manual MCP calls | Automatic via hooks |
| Memory Recall | Manual MCP calls | `/recall` command |
| Context Loading | None | SessionStart hook |
| Session Persistence | None | SessionEnd hook |
| Discoverability | Hidden MCP tools | Visible skills |
| User Experience | Complex | Simple commands |
| Distribution | MCP config only | Plugin install |

## Open Questions

1. **Consolidation Scheduling**: Should consolidation run automatically (cron) or only via command?
2. **Memory Limits**: How many episodes to load in SessionStart? (10? 20? configurable?)
3. **Entity Extraction**: Use LLM for entity extraction or rule-based?
4. **Skill Learning**: Automatically extract skills from repeated successful patterns?
5. **Multi-Session**: How to handle memories across different Claude Code instances?

## Next Steps

1. Review and approve this plan
2. Create plugin skeleton in `~/github/astoreyai/claude-skills/skills/world-weaver/`
3. Implement Phase 1 components
4. Test integration with existing WW MCP server
5. Iterate based on usage feedback
