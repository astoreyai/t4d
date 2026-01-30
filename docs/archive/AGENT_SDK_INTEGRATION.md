# World Weaver: Claude Agent SDK Integration

**Phase 10 Complete** | **Version**: 0.5.0

This document covers the integration between World Weaver's memory system and the Claude Agent SDK, enabling agents to learn from experience through biologically-inspired mechanisms.

---

## Quick Start

### 1. Using WWAgent (Recommended)

```python
from ww.sdk import WWAgent, AgentConfig

# Create agent with memory
agent = WWAgent(
    config=AgentConfig(
        name="my-assistant",
        model="claude-sonnet-4-5-20250929",
        memory_enabled=True,
    ),
    ww_api_url="http://localhost:8765",
)

# Start session and execute
async with agent.session():
    response = await agent.execute([
        {"role": "user", "content": "How do I implement a decorator?"}
    ])

    # Report outcome for learning
    await agent.report_outcome(
        task_id=response["task_id"],
        success=True,
        feedback="User confirmed solution worked",
    )
```

### 2. Using AgentMemoryClient Directly

```python
from ww.sdk import AgentMemoryClient

async with AgentMemoryClient(api_url="http://localhost:8765") as memory:
    # Store experience
    episode = await memory.store_experience(
        content="Implemented retry logic with exponential backoff",
        outcome="success",
        importance=0.8,
    )

    # Retrieve for task
    memories = await memory.retrieve_for_task(
        query="How to handle transient failures?",
        k=5,
    )

    # Report outcome (triggers three-factor learning)
    result = await memory.report_task_outcome(
        task_id=memories[0].task_id,
        success=True,
        reward=1.0,
    )
    print(f"Credited {result.credited} memories")
```

### 3. MCP Server for Claude Code

```bash
# Start MCP server
python -m ww.mcp.server

# Or configure in Claude Code settings
{
  "mcpServers": {
    "world-weaver": {
      "command": "python",
      "args": ["-m", "ww.mcp.server"],
      "env": {
        "WW_API_URL": "http://localhost:8765"
      }
    }
  }
}
```

---

## Architecture

```
+------------------+     +------------------+     +------------------+
|  Claude Agent    |     |   WWAgent        |     |  World Weaver    |
|  SDK             |<--->|   (Wrapper)      |<--->|  Memory API      |
+------------------+     +------------------+     +------------------+
                               |                         |
                               v                         v
                        +-------------+           +-------------+
                        | Session     |           | Three-Factor|
                        | Lifecycle   |           | Learning    |
                        | Hooks       |           +-------------+
                        +-------------+                  |
                               |                         v
                               |                  +-------------+
                               +----------------->| Eligibility |
                                                  | Traces      |
                                                  +-------------+
```

### Key Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `WWAgent` | Full agent wrapper with memory | `ww/sdk/agent.py` |
| `AgentMemoryClient` | Direct memory access with learning | `ww/sdk/agent_client.py` |
| `WorldWeaverMCPServer` | MCP server for Claude Code | `ww/mcp/server.py` |
| `SessionStartHook` | Load context at session start | `ww/hooks/session_lifecycle.py` |
| `SessionEndHook` | Persist and consolidate at end | `ww/hooks/session_lifecycle.py` |
| `TaskOutcomeHook` | Learn from task outcomes | `ww/hooks/session_lifecycle.py` |

---

## WWAgent Class

The `WWAgent` class wraps the Claude Agent SDK with automatic memory integration.

### Configuration

```python
from ww.sdk.agent import WWAgent, AgentConfig

config = AgentConfig(
    name="research-assistant",
    model="claude-sonnet-4-5-20250929",
    system_prompt="You are a helpful research assistant.",
    memory_enabled=True,
    consolidation_interval=10,  # Consolidate every 10 tasks
    max_context_memories=20,
)

agent = WWAgent(config=config, ww_api_url="http://localhost:8765")
```

### Session Management

```python
# Option 1: Context manager (recommended)
async with agent.session(session_id="project-alpha") as ctx:
    response = await agent.execute(messages)

# Option 2: Manual management
await agent.start_session(session_id="project-alpha")
try:
    response = await agent.execute(messages)
finally:
    await agent.end_session()
```

### Memory Operations

```python
# Store memory explicitly
episode = await agent.store_memory(
    content="Discovered that numpy broadcasting requires aligned shapes",
    outcome="success",
    importance=0.9,
    tags=["numpy", "debugging"],
)

# Recall memories
memories = await agent.recall_memories(
    query="array shape errors",
    limit=5,
)

# Get agent stats
stats = agent.get_stats()
print(f"Session: {stats['session_id']}, Messages: {stats['message_count']}")
```

### Outcome-Based Learning

```python
# Execute task
response = await agent.execute([
    {"role": "user", "content": "Fix the type error in auth.py"}
])

# Report success - triggers three-factor learning
result = await agent.report_outcome(
    task_id=response["task_id"],
    success=True,
    feedback="Error resolved, tests pass",
)

# Partial credit for partial success
result = await agent.report_outcome(
    task_id=response["task_id"],
    success=False,
    partial_credit=0.6,  # 60% credit
    feedback="Fixed type error but introduced new warning",
)
```

---

## AgentMemoryClient

Direct memory access with three-factor learning support.

### Initialization

```python
from ww.sdk.agent_client import AgentMemoryClient

# Default settings
client = AgentMemoryClient()

# Custom configuration
client = AgentMemoryClient(
    api_url="http://localhost:8765",
    session_id="my-session",
    api_key="optional-key",
    eligibility_decay=0.95,  # Trace decay per step
    base_learning_rate=0.01,  # Base LR for three-factor
)

# Must connect before use
await client.connect()
```

### Core Operations

```python
# Store experience with automatic eligibility tracking
episode = await client.store_experience(
    content="Implemented caching layer for API responses",
    outcome="success",
    importance=0.85,
    metadata={"project": "api-optimization"},
)

# Retrieve memories (marks eligibility traces)
memories = await client.retrieve_for_task(
    query="How to cache API responses?",
    k=5,
    task_id="task-123",  # Optional, auto-generated if omitted
)

# Each memory includes similarity score
for mem in memories:
    print(f"{mem.episode.content}: {mem.similarity_score:.2f}")
```

### Three-Factor Learning

The learning signal combines:
1. **Eligibility**: Which memories were retrieved for this task
2. **Neuromodulator**: Brain state (encoding/retrieval mode)
3. **Dopamine**: Reward prediction error (surprise)

```python
# Report outcome - this triggers learning
result = await client.report_task_outcome(
    task_id="task-123",
    success=True,
    reward=1.0,  # Optional explicit reward
)

# Learning result
print(f"Credited memories: {result.credited}")
print(f"Reconsolidated: {result.reconsolidated}")
print(f"Total LR applied: {result.total_lr_applied:.4f}")
```

### Consolidation

```python
# Light consolidation (fast, recent memories)
await client.trigger_consolidation(mode="light")

# Deep consolidation (thorough, all eligible)
await client.trigger_consolidation(mode="deep")

# Full consolidation (complete system review)
await client.trigger_consolidation(mode="full")
```

---

## MCP Server

The MCP server provides World Weaver tools to Claude Code.

### Available Tools

| Tool | Description |
|------|-------------|
| `ww_store` | Store a new memory/experience |
| `ww_recall` | Retrieve relevant memories |
| `ww_learn_outcome` | Report task success/failure |
| `ww_consolidate` | Trigger memory consolidation |
| `ww_get_context` | Get session context and stats |
| `ww_forget` | Suppress specific memories |

### Tool Schemas

```python
# ww_store
{
    "content": "string (required)",
    "outcome": "success|failure|partial (optional)",
    "importance": "float 0-1 (optional)",
    "metadata": "object (optional)"
}

# ww_recall
{
    "query": "string (required)",
    "k": "integer (optional, default 10)",
    "memory_type": "episodic|semantic|procedural (optional)"
}

# ww_learn_outcome
{
    "task_id": "string (required)",
    "success": "boolean (required)",
    "partial_credit": "float 0-1 (optional)",
    "feedback": "string (optional)"
}

# ww_consolidate
{
    "mode": "light|deep|full (optional, default light)"
}
```

### Prompts

The MCP server provides prompts for session management:

| Prompt | Arguments | Description |
|--------|-----------|-------------|
| `session_start` | project | Load context for a project |
| `session_end` | summary | Persist session with summary |
| `task_context` | task_description | Get relevant memories for task |

---

## Session Lifecycle Hooks

Integrate memory operations with Claude Code hooks.

### Creating Hooks

```python
from ww.hooks.session_lifecycle import create_session_hooks

hooks = create_session_hooks(
    api_url="http://localhost:8765",
    session_id="project-session",
    load_context=True,
    auto_consolidate=True,
)

# Individual hooks
start_hook = hooks["start"]
end_hook = hooks["end"]
outcome_hook = hooks["outcome"]
idle_hook = hooks["idle"]
```

### SessionStartHook

```python
from ww.hooks.session_lifecycle import SessionStartHook

hook = SessionStartHook(
    memory_client=client,
    context_query="current project context",
    max_context=20,
)

# Execute at session start
context = await hook.execute(
    project="my-project",
    working_dir="/path/to/project",
)

print(f"Loaded {len(context.memories)} context memories")
```

### SessionEndHook

```python
from ww.hooks.session_lifecycle import SessionEndHook

hook = SessionEndHook(
    memory_client=client,
    consolidate_on_end=True,
    consolidation_mode="light",
)

# Execute at session end
result = await hook.execute(summary="Completed feature implementation")

print(f"Session duration: {result['duration_seconds']}s")
print(f"Memories stored: {result['memories_stored']}")
```

### TaskOutcomeHook

```python
from ww.hooks.session_lifecycle import TaskOutcomeHook

hook = TaskOutcomeHook(
    memory_client=client,
    auto_store=True,  # Store experience on outcome
)

# Execute when task completes
result = await hook.execute(
    task_id="task-123",
    success=True,
    feedback="Tests pass, user confirmed working",
)

print(f"Credited: {result['credited']} memories")
```

### IdleConsolidationHook

Triggers consolidation during idle periods:

```python
from ww.hooks.session_lifecycle import IdleConsolidationHook

hook = IdleConsolidationHook(
    memory_client=client,
    idle_threshold_seconds=300,  # 5 minutes
    consolidation_mode="light",
)

# Call periodically
result = await hook.execute()
if result.get("consolidated"):
    print("Consolidated during idle")
```

---

## REST API Endpoints

Agent management via REST API.

### Agent Lifecycle

```bash
# Create agent
POST /api/v1/agents
{
    "name": "my-agent",
    "model": "claude-sonnet-4-5-20250929",
    "memory_enabled": true
}

# List agents
GET /api/v1/agents

# Get agent
GET /api/v1/agents/{agent_id}

# Delete agent
DELETE /api/v1/agents/{agent_id}
```

### Session Management

```bash
# Start session
POST /api/v1/agents/{agent_id}/sessions

# End session
DELETE /api/v1/agents/{agent_id}/sessions/{session_id}

# Get session info
GET /api/v1/agents/{agent_id}/sessions/{session_id}
```

### Execution

```bash
# Execute task
POST /api/v1/agents/{agent_id}/execute
{
    "messages": [{"role": "user", "content": "..."}],
    "include_memory_context": true
}

# Report outcome
POST /api/v1/agents/{agent_id}/outcome
{
    "task_id": "task-123",
    "success": true,
    "feedback": "..."
}
```

### Memory Operations

```bash
# Store memory
POST /api/v1/agents/{agent_id}/memories
{
    "content": "...",
    "outcome": "success",
    "importance": 0.8
}

# Recall memories
GET /api/v1/agents/{agent_id}/memories?query=...&limit=10

# Trigger consolidation
POST /api/v1/agents/{agent_id}/consolidate
{
    "mode": "deep"
}
```

---

## Three-Factor Learning Details

### How It Works

1. **Retrieval creates eligibility**: When memories are retrieved for a task, they're marked as "eligible" for credit
2. **Eligibility decays**: Traces decay over time (configurable, default 0.95/step)
3. **Outcome triggers learning**: When task outcome is reported:
   - Dopamine signal computed (reward prediction error)
   - Neuromodulator state determines mode (encoding/retrieval)
   - Three-factor rule: `LR = base_lr * eligibility * neuromod * dopamine`
4. **Weights updated**: FF encoder weights adjusted for eligible memories

### Configuration

```python
client = AgentMemoryClient(
    eligibility_decay=0.95,     # Trace decay rate
    base_learning_rate=0.01,   # Base LR before modulation
)
```

### Learning Signal Components

| Component | Source | Range | Effect |
|-----------|--------|-------|--------|
| Eligibility | Retrieval traces | 0-1 | Decays, tracks relevance |
| Neuromodulator | Brain state | 0-2 | ACh/NE gates mode |
| Dopamine | RPE signal | -1 to 1 | Surprise drives learning |
| Base LR | Configuration | 0.001-0.1 | Scales all updates |

### Example Learning Flow

```python
# 1. Retrieve memories (creates eligibility)
memories = await client.retrieve_for_task("How to handle auth?")
# Eligibility traces: {mem1: 1.0, mem2: 1.0}

# 2. Execute task (eligibility decays)
await asyncio.sleep(1)
# Eligibility traces: {mem1: 0.95, mem2: 0.95}

# 3. Report outcome (triggers learning)
result = await client.report_task_outcome("task-1", success=True)
# Learning applied:
#   mem1: LR = 0.01 * 0.95 * 1.0 * 0.8 = 0.0076
#   mem2: LR = 0.01 * 0.95 * 1.0 * 0.8 = 0.0076
```

---

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WW_API_URL` | `http://localhost:8765` | World Weaver API URL |
| `WW_SESSION_ID` | Auto-generated | Session identifier |
| `WW_API_KEY` | None | Optional API key |
| `WW_ELIGIBILITY_DECAY` | `0.95` | Eligibility trace decay |
| `WW_BASE_LR` | `0.01` | Base learning rate |
| `WW_MAX_CONTEXT` | `20` | Max context memories |

### AgentConfig Options

```python
@dataclass
class AgentConfig:
    name: str                           # Agent name
    model: str = "claude-sonnet-4-5-20250929"  # Model ID
    system_prompt: str | None = None    # System prompt
    memory_enabled: bool = True         # Enable memory
    consolidation_interval: int = 10    # Tasks between consolidation
    max_context_memories: int = 20      # Context limit
```

---

## Testing

Run Phase 10 tests:

```bash
# All Phase 10 tests
pytest tests/sdk/ tests/mcp/ tests/hooks/ tests/api/test_routes_agents.py -v

# SDK tests only
pytest tests/sdk/ -v

# MCP tests only
pytest tests/mcp/ -v

# Hook tests only
pytest tests/hooks/ -v
```

Current test counts:
- SDK: 108 tests
- MCP: 30 tests
- Hooks: 35 tests
- API Routes: 34 tests
- **Total**: 207 Phase 10 tests

---

## Troubleshooting

### Connection Errors

```python
# Check API is running
import httpx
async with httpx.AsyncClient() as client:
    r = await client.get("http://localhost:8765/health")
    print(r.json())
```

### Learning Not Applying

1. Check eligibility traces exist:
```python
print(f"Active traces: {client._eligibility.count}")
```

2. Check outcome was reported:
```python
result = await client.report_task_outcome(task_id, success=True)
print(f"Credited: {result.credited}")
```

3. Verify neuromodulator state:
```python
print(f"Mode: {client._current_nt_state.acetylcholine_mode}")
```

### MCP Server Not Responding

```bash
# Check server is running
ps aux | grep "ww.mcp.server"

# Check port
netstat -tlnp | grep 8765

# Run with debug logging
WW_LOG_LEVEL=DEBUG python -m ww.mcp.server
```

---

## Next Steps

- **Phase 11**: Documentation site and production launch
- **Helm Chart**: Kubernetes deployment
- **Monitoring**: Grafana dashboards for learning metrics
- **PyPI**: `pip install world-weaver`
