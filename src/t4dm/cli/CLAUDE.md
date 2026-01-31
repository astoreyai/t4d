# CLI Module
**Path**: `/mnt/projects/t4d/t4dm/src/ww/cli/`

## What
Typer-based command-line interface providing the `ww` command for interacting with the tripartite memory system. Supports storing, recalling, consolidating memories, checking system status, and launching the API server.

## How
- Typer app with Rich console output for styled tables
- Async-sync bridge: `run_async()` wraps async memory service calls for sync CLI commands
- Session isolation via `WW_SESSION_ID` environment variable (default: `cli-session`)
- Sub-apps for namespaced commands: `ww episodic`, `ww semantic`, `ww procedural`
- Core commands: `store`, `recall`, `consolidate`, `status`, `serve`, `config`, `version`

## Why
Provides direct terminal access to memory operations for development, debugging, scripting, and CI/CD integration without requiring the REST API to be running.

## Key Files
| File | Purpose |
|------|---------|
| `main.py` | All CLI commands and logic (~460 lines) |
| `__init__.py` | Exports Typer `app` |

## Data Flow
```
CLI Command -> Typer parser -> run_async()
    -> core.services.get_services(session_id)
    -> Memory operation (episodic/semantic/procedural)
    -> Rich table/JSON output
```

## Integration Points
- **core**: `get_services()` for memory operations, `get_settings()` for config, types (Episode, Entity, Procedure)
- **consolidation**: `get_consolidation_service()` for `ww consolidate`
- **api**: `ww.api.server:app` launched by `ww serve` via uvicorn
