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
| **t4dm-conductor** | `t4dm-conductor/SKILL.md` | Multi-skill orchestration and task routing |
| **t4dm-init** | `t4dm-init/SKILL.md` | Project bootstrapping, feature list generation |
| **t4dm-session** | `t4dm-session/SKILL.md` | Session state loading and progress tracking |

## Data Flow
```
Session start → t4dm-session (load state) → t4dm-conductor (route tasks) → domain/memory/workflow skills
                                        ← t4dm-session (save state)
```

## Integration Points
- **Config**: Reads/writes `config/t4dm-features.json` and `config/t4dm-progress.json`
- **Hooks**: `config/hooks/t4dm-session-start.py` and `t4dm-session-end.py`
