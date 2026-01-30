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
| **ww-planner** | `ww-planner/SKILL.md` | Feature planning and task decomposition |
| **ww-validator** | `ww-validator/SKILL.md` | Implementation validation against specs |
| **ww-synthesizer** | `ww-synthesizer/SKILL.md` | Result synthesis and documentation generation |
| **ww-finetune** | `ww-finetune/SKILL.md` | Parameter tuning and optimization workflows |

## Data Flow
```
Feature request → ww-planner (decompose) → Implementation
                → ww-validator (verify)  → ww-synthesizer (document)
                → ww-finetune (optimize) → Updated parameters
```

## Integration Points
- **Config**: `ww-features.json` drives planning and validation
- **Tests**: `ww-validator` references test suites for verification
- **Docs**: `ww-synthesizer` generates documentation updates
