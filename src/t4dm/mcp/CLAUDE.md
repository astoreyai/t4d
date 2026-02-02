# MCP
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/mcp/`

## What
Model Context Protocol (MCP) server exposing WW memory as tools for Claude Code and Claude Desktop. Provides store, recall, learn, consolidate, context, and entity operations.

## How
- **WorldWeaverMCPServer** (`server.py`): MCP server implementation using stdio transport. Connects to WW API via `AgentMemoryClient`. Handles tool execution requests from Claude, dispatching to appropriate memory operations. Configurable via environment variables (`WW_API_URL`, `WW_SESSION_ID`, `WW_API_KEY`).
- **MEMORY_TOOLS** (`tools.py`): Tool definitions in MCP JSON schema format:
  - `ww_store`: Store content in episodic memory with outcome/importance/tags
  - `ww_recall`: Semantic search for relevant memories with project/time filtering
  - `ww_learn_outcome`: Report task success/failure for learning system
  - `ww_consolidate`: Trigger memory consolidation
  - `ww_get_context`: Get project/session context
  - `ww_entity`: Create/retrieve semantic entities

## Why
MCP is the standard protocol for extending Claude with external tools. This server makes WW's memory system available to any MCP-compatible client, enabling persistent memory across Claude sessions.

## Key Files
| File | Purpose |
|------|---------|
| `server.py` | `WorldWeaverMCPServer`, `create_mcp_server()`, `run_mcp_server()` |
| `tools.py` | `MEMORY_TOOLS` definitions, `MEMORY_PROMPTS`, `ToolResult` |

## Data Flow
```
Claude Code/Desktop -> MCP stdio -> WorldWeaverMCPServer
    -> AgentMemoryClient -> WW API (FastAPI)
    -> Memory operations (store/recall/consolidate)
    -> Response -> MCP -> Claude
```

## Integration Points
- **sdk/agent_client.py**: `AgentMemoryClient` for API communication
- **api/**: FastAPI server that MCP client connects to
- **memory/**: Underlying episodic/semantic memory operations
- **learning/**: Outcome events trigger learning feedback loop
