# Skills / Orchestration
**Path**: `/mnt/projects/t4d/t4dm/skills/orchestration/`

## What
Skills for project initialization, session management, and multi-skill coordination in T4DM development workflows.

## How
- SKILL.md files defining orchestration patterns for managing development sessions and project state

## Why
Coordinates feature tracking, session state, and multi-skill workflows to maintain development continuity across sessions.

## Key Files
| Skill | Directory | Purpose |
|-------|-----------|---------|
| **ww-conductor** | `ww-conductor/SKILL.md` | Multi-skill orchestration and task routing |
| **ww-init** | `ww-init/SKILL.md` | Project bootstrapping, feature list generation |
| **ww-session** | `ww-session/SKILL.md` | Session state loading and progress tracking |

## Data Flow
```
Session start → ww-session (load state) → ww-conductor (route tasks) → domain/memory/workflow skills
                                        ← ww-session (save state)
```

## Integration Points
- **Config**: Reads/writes `config/ww-features.json` and `config/ww-progress.json`
- **Hooks**: `config/hooks/ww-session-start.py` and `ww-session-end.py`
