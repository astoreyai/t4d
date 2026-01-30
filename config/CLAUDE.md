# Config
**Path**: `/mnt/projects/t4d/t4dm/config/`

## What
Configuration files for T4DM features, progress tracking, Claude Code settings, and session hooks.

## How
- JSON configuration files for feature flags and progress state
- Claude Code hooks in `hooks/` for session lifecycle automation

## Why
Centralizes feature management, tracks implementation progress across features, and automates Claude Code session start/end workflows.

## Key Files
| File | Purpose |
|------|---------|
| `ww-features.json` | Feature registry with verification steps and priority |
| `ww-progress.json` | Implementation progress tracking per feature |
| `claude-code-settings.json` | Claude Code project settings |
| `hooks/ww-session-start.py` | Session initialization hook |
| `hooks/ww-session-end.py` | Session teardown hook |

## Data Flow
```
ww-features.json (feature defs) → ww-progress.json (status tracking)
hooks/ → Claude Code session lifecycle
```

## Integration Points
- **Skills**: `ww-init` and `ww-session` skills read/write these configs
- **Claude Code**: Hooks auto-run on session start/end
