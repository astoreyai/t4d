# Skills / Memory
**Path**: `/mnt/projects/t4d/t4dm/skills/memory/`

## What
Skills for working with T4DM's tripartite memory subsystems: episodic, semantic, procedural memory plus consolidation and graph operations.

## How
- Each skill targets a specific memory type or memory operation
- SKILL.md files provide memory-type-specific coding patterns and constraints

## Why
Each memory type has distinct data models, storage patterns, and retrieval algorithms. Specialized skills prevent cross-contamination of concerns.

## Key Files
| Skill | Directory | Purpose |
|-------|-----------|---------|
| **ww-memory** | `ww-memory/SKILL.md` | Core memory interface and shared operations |
| **ww-episodic** | `ww-episodic/SKILL.md` | Episodic memory (events, timestamps, context) |
| **ww-semantic** | `ww-semantic/SKILL.md` | Semantic memory (entities, relationships, facts) |
| **ww-semantic-mem** | `ww-semantic-mem/SKILL.md` | Semantic memory implementation details |
| **ww-procedural** | `ww-procedural/SKILL.md` | Procedural memory (skills, workflows, sequences) |
| **ww-consolidate** | `ww-consolidate/SKILL.md` | Memory consolidation (HDBSCAN clustering, decay) |
| **ww-graph** | `ww-graph/SKILL.md` | Neo4j graph operations and Cypher queries |

## Data Flow
```
Memory operation → Type-specific skill → src/t4dm/core/ + bridges/
```

## Integration Points
- **Bridges**: `src/t4dm/bridges/` (Neo4j + Qdrant dual-store)
- **Consolidation**: `src/t4dm/consolidation/` pipeline
- **Tests**: `tests/memory/`, `tests/consolidation/`
