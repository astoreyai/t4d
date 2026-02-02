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
| `domain/` | t4dm-algorithm, t4dm-compbio, t4dm-neuro |
| `knowledge/` | t4dm-knowledge, t4dm-retriever |
| `memory/` | t4dm-consolidate, t4dm-episodic, t4dm-graph, t4dm-memory, t4dm-procedural, t4dm-semantic, t4dm-semantic-mem |
| `orchestration/` | t4dm-conductor, t4dm-init, t4dm-session |
| `workflow/` | t4dm-finetune, t4dm-planner, t4dm-synthesizer, t4dm-validator |

## Data Flow
```
User invokes skill → SKILL.md (prompt) → Claude Code session
                                        → T4DM codebase operations
```

## Integration Points
- **Config**: `config/t4dm-features.json` and `config/t4dm-progress.json` used by orchestration skills
- **Agents**: Complementary to agents (skills assist, agents debug)
