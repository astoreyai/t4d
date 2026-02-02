# MCP
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/mcp/`

## What
Model Context Protocol (MCP) server exposing T4DM memory as tools for Claude Code and Claude Desktop. Provides store, recall, learn, consolidate, context, and entity operations.

## How
- **T4DMMCPServer** (`server.py`): MCP server implementation using stdio transport. Connects to T4DM API via `AgentMemoryClient`. Handles tool execution requests from Claude, dispatching to appropriate memory operations. Configurable via environment variables (`T4DM_API_URL`, `T4DM_SESSION_ID`, `T4DM_API_KEY`).
- **MEMORY_TOOLS** (`tools.py`): Tool definitions in MCP JSON schema format:
  - `t4dm_store`: Store content in episodic memory with outcome/importance/tags
  - `t4dm_search`: Semantic search for relevant memories with project/time filtering
  - `t4dm_learn`: Report task success/failure for learning system
  - `t4dm_consolidate`: Trigger memory consolidation
  - `t4dm_context`: Get project/session context
  - `t4dm_entity`: Create/retrieve semantic entities
  - `t4dm_skill`: Create/retrieve procedural skills

## Why
MCP is the standard protocol for extending Claude with external tools. This server makes T4DM's memory system available to any MCP-compatible client, enabling persistent memory across Claude sessions.

## Key Files
| File | Purpose |
|------|---------|
| `server.py` | `T4DMMCPServer`, `create_mcp_server()`, `run_mcp_server()` |
| `tools.py` | `MEMORY_TOOLS` definitions, `MEMORY_PROMPTS`, `ToolResult` |

## Data Flow
```
Claude Code/Desktop -> MCP stdio -> T4DMMCPServer
    -> AgentMemoryClient -> T4DM API (FastAPI)
    -> Memory operations (store/search/consolidate)
    -> Response -> MCP -> Claude
```

## Integration Points
- **sdk/agent_client.py**: `AgentMemoryClient` for API communication
- **api/**: FastAPI server that MCP client connects to
- **memory/**: Underlying episodic/semantic memory operations
- **learning/**: Outcome events trigger learning feedback loop
