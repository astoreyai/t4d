# Skills
**Path**: `/mnt/projects/t4d/t4dm/skills/`

## What
Claude Code skill definitions organized by category for T4DM development. Each skill is a SKILL.md prompt file providing domain-specific AI assistance.

## How
- Five categories: domain, knowledge, memory, orchestration, workflow
- Each skill lives in its own directory with a `SKILL.md` file
- Invoked via Claude Code `/skill-name` commands

## Why
Provides specialized AI capabilities for working with bio-inspired memory systems, from neuroscience domain knowledge to memory consolidation workflows.

## Key Files
| Category | Skills |
|----------|--------|
| `domain/` | ww-algorithm, ww-compbio, ww-neuro |
| `knowledge/` | ww-knowledge, ww-retriever |
| `memory/` | ww-consolidate, ww-episodic, ww-graph, ww-memory, ww-procedural, ww-semantic, ww-semantic-mem |
| `orchestration/` | ww-conductor, ww-init, ww-session |
| `workflow/` | ww-finetune, ww-planner, ww-synthesizer, ww-validator |

## Data Flow
```
User invokes skill → SKILL.md (prompt) → Claude Code session
                                        → T4DM codebase operations
```

## Integration Points
- **Config**: `config/ww-features.json` and `config/ww-progress.json` used by orchestration skills
- **Agents**: Complementary to agents (skills assist, agents debug)
