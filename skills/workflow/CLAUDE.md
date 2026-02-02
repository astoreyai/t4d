# Skills / Workflow
**Path**: `/mnt/projects/t4d/t4dm/skills/workflow/`

## What
Development workflow skills for planning, validation, synthesis, and fine-tuning within T4DM.

## How
- SKILL.md files defining structured workflows for common development tasks

## Why
Provides repeatable processes for planning features, validating implementations, synthesizing results, and fine-tuning system behavior.

## Key Files
| Skill | Directory | Purpose |
|-------|-----------|---------|
| **t4dm-planner** | `t4dm-planner/SKILL.md` | Feature planning and task decomposition |
| **t4dm-validator** | `t4dm-validator/SKILL.md` | Implementation validation against specs |
| **t4dm-synthesizer** | `t4dm-synthesizer/SKILL.md` | Result synthesis and documentation generation |
| **t4dm-finetune** | `t4dm-finetune/SKILL.md` | Parameter tuning and optimization workflows |

## Data Flow
```
Feature request → t4dm-planner (decompose) → Implementation
                → t4dm-validator (verify)  → t4dm-synthesizer (document)
                → t4dm-finetune (optimize) → Updated parameters
```

## Integration Points
- **Config**: `t4dm-features.json` drives planning and validation
- **Tests**: `t4dm-validator` references test suites for verification
- **Docs**: `t4dm-synthesizer` generates documentation updates
