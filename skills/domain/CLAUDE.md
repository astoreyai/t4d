# Skills / Domain
**Path**: `/mnt/projects/t4d/t4dm/skills/domain/`

## What
Domain-expert skills providing specialized knowledge in algorithms, computational biology, and neuroscience for T4DM development.

## How
- Each skill directory contains a `SKILL.md` with domain-specific prompts and constraints
- Invoked when working on domain-specific T4DM features

## Why
T4DM's bio-inspired architecture requires deep domain knowledge in neuroscience and biology that general-purpose coding assistants lack.

## Key Files
| Skill | Directory | Purpose |
|-------|-----------|---------|
| **ww-algorithm** | `ww-algorithm/SKILL.md` | Algorithm design and analysis (HDBSCAN, FSRS, Hopfield, STDP) |
| **ww-compbio** | `ww-compbio/SKILL.md` | Computational biology concepts (neural dynamics, plasticity models) |
| **ww-neuro** | `ww-neuro/SKILL.md` | Neuroscience foundations (brain regions, neuromodulators, CLS theory) |

## Data Flow
```
Domain question → Skill prompt → Domain-informed code changes
```

## Integration Points
- **Agents**: `bio-memory-auditor` and `hinton-learning-validator` share domain knowledge
- **Tests**: `tests/biology/` validates claims from domain skills
