# Agents
**Path**: `/mnt/projects/t4d/t4dm/agents/`

## What
Specialized bug-hunting agent definitions for debugging T4DM's bio-inspired memory system. Each agent is a markdown prompt file targeting specific bug categories.

## How
- Markdown files defining agent personas, inspection strategies, and bug taxonomies
- Registered in `AGENT_REGISTRY.md` with routing guidance
- Used as system prompts for AI-assisted debugging sessions

## Why
Bio-inspired memory systems have unique failure modes (plasticity bugs, CLS violations, neuromodulator errors) that require specialized domain knowledge to diagnose.

## Key Files
| File | Purpose |
|------|---------|
| `AGENT_REGISTRY.md` | Agent index with routing: which agent for which bug |
| `bio-memory-auditor.md` | Validate biological plausibility, CLS violations |
| `race-condition-hunter.md` | Detect TOCTOU, deadlocks, async state corruption |
| `memory-leak-hunter.md` | Find unbounded growth, circular refs, cache leaks |
| `hinton-learning-validator.md` | Validate Forward-Forward, locality bugs |
| `cache-coherence-analyzer.md` | Analyze staleness, stampede, poisoning |
| `eligibility-trace-debugger.md` | Debug temporal credit decay and accumulation |

## Data Flow
```
Bug report → AGENT_REGISTRY.md (routing) → Specific agent prompt
                                         → Codebase inspection
                                         → Diagnosis + fix
```

## Integration Points
- **Claude Code**: Agents loadable as system prompts for debugging sessions
- **Tests**: Agents reference test categories in `tests/biology/`, `tests/chaos/`
