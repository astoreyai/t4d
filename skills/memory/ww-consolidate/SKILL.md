---
name: ww-consolidate
description: Memory consolidation engine implementing episodic→semantic transfer, pattern extraction, and skill consolidation. Mimics biological sleep-phase consolidation for knowledge crystallization.
version: 0.1.0
---

# World Weaver Consolidation Engine

You are the consolidation engine for World Weaver. Your role is to transform episodic experiences into generalized semantic knowledge and consolidate procedural skills—mimicking the biological process of memory consolidation that occurs during sleep.

## Purpose

Orchestrate memory consolidation:
1. Transfer episodic memories to semantic knowledge
2. Extract patterns from repeated experiences
3. Consolidate similar procedures into refined skills
4. Maintain source attribution during transfer
5. Schedule consolidation based on memory stability

## Cognitive Foundation

Human memory consolidation occurs primarily during sleep, transferring hippocampal (episodic) memories to neocortical (semantic) storage. This process:
- **Abstracts** context-specific details into general knowledge
- **Strengthens** neural pathways through replay
- **Prunes** irrelevant or redundant connections

**Key Insight**: A memory episode like "We discussed Archimedes risk parameters on Tuesday" becomes semantic fact "Archimedes uses configurable risk parameters."

## Consolidation Schema

```cypher
CREATE (c:ConsolidationEvent {
  id: randomUUID(),
  eventType: $type,              // episodic_to_semantic, skill_merge, pattern_extract
  sourceIds: $sourceArray,       // Original memory IDs
  targetId: $targetId,           // Created/updated entity ID
  timestamp: datetime(),
  confidence: $score,            // Consolidation confidence [0,1]
  patternStrength: $strength,    // Number of source instances
  metadata: $context
})
```

## Core Operations

### Episodic → Semantic Transfer

```python
async def consolidate_episodes(
    episode_ids: list[str],
    min_similarity: float = 0.75,
    min_occurrences: int = 3
) -> list[Entity]:
    """
    SEMANTICIZATION: Transform episodic memories into semantic knowledge.

    Process:
    1. Cluster similar episodes
    2. Extract common patterns
    3. Abstract context-specific details
    4. Create/update semantic entities
    5. Maintain source attribution
    """
    episodes = [await get_episode(id) for id in episode_ids]

    # Cluster by embedding similarity
    clusters = cluster_by_embedding(episodes, threshold=min_similarity)

    created_entities = []
    for cluster in clusters:
        if len(cluster) < min_occurrences:
            continue  # Not enough repetition for consolidation

        # Extract common pattern
        pattern = await extract_common_pattern(cluster)

        # Check if entity already exists
        existing = await find_similar_entity(pattern.summary)

        if existing and existing.similarity > 0.9:
            # Reinforce existing knowledge
            await reinforce_entity(existing.id, cluster)
            created_entities.append(existing)
        else:
            # Create new semantic entity
            entity = await create_entity(
                name=pattern.name,
                entity_type=pattern.entity_type,
                summary=pattern.summary,
                details=pattern.details,
                source=f"consolidated_from:{len(cluster)}_episodes"
            )

            # Link to source episodes
            for ep in cluster:
                await create_relationship(
                    source_id=ep.id,
                    target_id=entity.id,
                    relation_type="CONSOLIDATED_INTO"
                )

            created_entities.append(entity)

        # Log consolidation event
        await log_consolidation(
            event_type='episodic_to_semantic',
            source_ids=[ep.id for ep in cluster],
            target_id=entity.id,
            confidence=pattern.confidence,
            pattern_strength=len(cluster)
        )

    return created_entities


async def extract_common_pattern(episodes: list[Episode]) -> Pattern:
    """
    Extract abstracted pattern from episodic cluster.

    Removes temporal/spatial context, preserves essential knowledge.
    """
    contents = [ep.content for ep in episodes]

    prompt = f"""Analyze these related memories and extract the core knowledge:

MEMORIES:
{chr(10).join(f'- {c}' for c in contents)}

Extract:
1. name: Canonical name for this knowledge
2. entity_type: CONCEPT, TECHNIQUE, FACT, TOOL, etc.
3. summary: Context-free generalized knowledge (1-2 sentences)
4. details: Additional context if relevant
5. confidence: How confident is this abstraction (0-1)

Return JSON: {{"name": "", "entity_type": "", "summary": "", "details": "", "confidence": 0.0}}
"""
    response = await llm.complete(prompt)
    return Pattern(**json.loads(response))
```

### Pattern Extraction from Trajectories

```python
async def extract_patterns_from_success(
    min_success_rate: float = 0.8,
    lookback_days: int = 7
) -> list[dict]:
    """
    Identify successful patterns worth consolidating.

    Looks for repeated successful actions that could become procedures.
    """
    # Get successful episodes from recent period
    successful = await neo4j.query("""
        MATCH (e:Episode)
        WHERE e.outcome = 'success'
          AND e.timestamp > datetime() - duration({days: $days})
        RETURN e
        ORDER BY e.timestamp DESC
    """, {'days': lookback_days})

    # Cluster by action type and context
    action_clusters = cluster_by_action_similarity(successful)

    patterns = []
    for cluster in action_clusters:
        if len(cluster) >= 3:  # Minimum repetitions
            success_rate = sum(1 for e in cluster if e.outcome == 'success') / len(cluster)

            if success_rate >= min_success_rate:
                patterns.append({
                    'episodes': cluster,
                    'success_rate': success_rate,
                    'pattern_type': infer_pattern_type(cluster),
                    'suggested_procedure': await suggest_procedure(cluster)
                })

    return patterns


async def suggest_procedure(episodes: list[Episode]) -> dict:
    """Generate procedure suggestion from successful episode pattern."""
    actions = [ep.context.get('tool') for ep in episodes]

    prompt = f"""These actions were repeatedly successful:
{chr(10).join(f'- {ep.content}' for ep in episodes[:5])}

Suggest a reusable procedure:
1. procedure_name: Short descriptive name
2. trigger_pattern: When should this procedure be used?
3. steps: High-level step sequence
4. domain: coding, research, trading, etc.

Return JSON.
"""
    return json.loads(await llm.complete(prompt))
```

### Skill Consolidation

```python
async def consolidate_procedures(
    domain: str,
    similarity_threshold: float = 0.85
) -> list[Procedure]:
    """
    Merge similar procedures into consolidated skills.

    Follows Memp pattern: consolidation occurs after sufficient
    successful executions demonstrate procedure validity.
    """
    # Get successful procedures in domain
    procedures = await neo4j.query("""
        MATCH (p:Procedure {domain: $domain, deprecated: false})
        WHERE p.successRate > 0.7 AND p.executionCount > 3
        RETURN p
    """, {'domain': domain})

    if len(procedures) < 2:
        return []

    # Cluster by embedding similarity
    clusters = cluster_by_embedding(procedures, threshold=similarity_threshold)

    consolidated = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue

        # Merge into master procedure
        master = await merge_procedures(cluster)

        # Mark originals as deprecated
        for proc in cluster:
            await neo4j.update(proc.id, {
                'deprecated': True,
                'consolidatedInto': master.id
            })

        # Log consolidation
        await log_consolidation(
            event_type='skill_merge',
            source_ids=[p.id for p in cluster],
            target_id=master.id,
            confidence=0.9,
            pattern_strength=len(cluster)
        )

        consolidated.append(master)

    return consolidated


async def merge_procedures(procedures: list[Procedure]) -> Procedure:
    """
    Create unified procedure from cluster of similar ones.

    Selects best steps from each, creates abstract script.
    """
    # Weight by success rate and execution count
    weighted = sorted(
        procedures,
        key=lambda p: p.success_rate * min(p.execution_count / 10, 1.0),
        reverse=True
    )

    # Use best procedure as base
    base = weighted[0]

    # Collect all step variations
    all_steps = [p.steps for p in procedures]

    prompt = f"""Merge these procedure variations into one optimal procedure:

BASE PROCEDURE: {base.name}
{json.dumps(base.steps, indent=2)}

VARIATIONS:
{chr(10).join(json.dumps(s, indent=2) for s in all_steps[1:3])}

Create merged procedure that:
1. Keeps most successful patterns
2. Handles edge cases from variations
3. Remains general enough to apply broadly

Return JSON: {{"steps": [...], "script": "..."}}
"""
    merged = json.loads(await llm.complete(prompt))

    return await create_procedure(
        name=f"{base.name} (Consolidated)",
        domain=base.domain,
        trigger_pattern=base.trigger_pattern,
        steps=merged['steps'],
        script=merged['script'],
        created_from='consolidated'
    )
```

### Scheduled Consolidation

```python
class ConsolidationScheduler:
    """
    Schedule consolidation runs based on memory system state.

    Mimics sleep-cycle consolidation:
    - Light consolidation: Every few hours (quick pattern check)
    - Deep consolidation: Daily (full episodic→semantic transfer)
    - Skill consolidation: Weekly (procedural optimization)
    """

    async def light_consolidation(self):
        """Quick consolidation pass - run every 2-4 hours."""
        # Check for highly similar recent episodes
        recent = await get_episodes_since(hours=4)
        patterns = await extract_patterns_from_success(
            min_success_rate=0.9,
            lookback_days=1
        )

        # Only consolidate very obvious patterns
        for pattern in patterns:
            if pattern['success_rate'] > 0.95 and len(pattern['episodes']) >= 5:
                await consolidate_episodes(
                    [e.id for e in pattern['episodes']],
                    min_similarity=0.9,
                    min_occurrences=5
                )

    async def deep_consolidation(self):
        """Full consolidation pass - run daily."""
        # Episodic → Semantic transfer
        episodes = await get_episodes_since(days=1)
        await consolidate_episodes(
            [e.id for e in episodes],
            min_similarity=0.75,
            min_occurrences=3
        )

        # Hebbian weight decay for unused connections
        await decay_unused_connections(threshold_days=30)

        # Update FSRS stability for all memories
        await update_all_stabilities()

        # Log consolidation metrics
        await log_consolidation_metrics()

    async def skill_consolidation(self):
        """Procedural optimization - run weekly."""
        domains = ['coding', 'research', 'trading', 'devops', 'writing']

        for domain in domains:
            await consolidate_procedures(domain, similarity_threshold=0.85)


async def get_consolidation_candidates() -> dict:
    """
    Identify memories ready for consolidation based on FSRS retrievability.

    Memories with declining retrievability but multiple accesses
    are prime candidates for semantic transfer.
    """
    return await neo4j.query("""
        MATCH (e:Episode)
        WHERE e.accessCount >= 3
          AND e.stability > 5  // Has been reinforced
        WITH e,
             (1 + 0.9 * duration.between(e.lastAccessed, datetime()).days / e.stability) ^ (-0.5)
             AS retrievability
        WHERE retrievability < 0.7  // Starting to decay
        RETURN e, retrievability
        ORDER BY e.accessCount DESC
        LIMIT 100
    """)
```

## Consolidation Triggers

| Trigger | Type | Threshold |
|---------|------|-----------|
| Similar episodes | Episodic→Semantic | 3+ episodes, >0.75 similarity |
| Successful pattern | Pattern extraction | 3+ successes, >0.8 rate |
| Similar procedures | Skill merge | 2+ procs, >0.85 similarity |
| Declining retrieval | Stability-based | <0.7 retrievability |
| Manual request | User-triggered | Explicit command |

## Source Attribution

```python
async def maintain_provenance(
    target_entity_id: str,
    source_episode_ids: list[str]
):
    """
    Maintain full provenance chain for consolidated knowledge.

    Enables:
    - "Where did I learn this?"
    - "What experiences led to this conclusion?"
    - Reverting if source found invalid
    """
    for ep_id in source_episode_ids:
        await neo4j.query("""
            MATCH (e:Episode {id: $ep_id})
            MATCH (ent:Entity {id: $ent_id})
            MERGE (e)-[r:SOURCE_OF]->(ent)
            SET r.consolidatedAt = datetime()
        """, {'ep_id': ep_id, 'ent_id': target_entity_id})
```

## Integration Points

### With ww-episodic

- Reads episodes for consolidation
- Updates episode metadata after transfer
- Respects session namespacing

### With ww-semantic-mem

- Creates/updates semantic entities
- Strengthens relationships via consolidation
- Triggers Hebbian updates

### With ww-procedural

- Identifies procedure consolidation candidates
- Merges similar procedures
- Tracks procedure lineage

## MCP Tools

| Tool | Description |
|------|-------------|
| `consolidate_now` | Trigger immediate consolidation cycle |
| `consolidate_episodes` | Transfer specific episodes to semantic |
| `consolidate_procedures` | Merge similar procedures |
| `get_provenance` | Trace knowledge back to source episodes |
| `schedule_consolidation` | Set consolidation schedule |

## Example Operations

### Episode Consolidation

```
Input Episodes:
- "Discussed FSRS parameters with 0.9 retention target" (Nov 25)
- "Set FSRS retention to 0.9 for memory decay" (Nov 26)
- "FSRS uses 0.9 as default retention threshold" (Nov 27)

→ consolidate_episodes([ep1, ep2, ep3])

Output Entity:
  name: "FSRS Retention Threshold"
  type: FACT
  summary: "FSRS uses 0.9 (90%) as the default retention target"
  source: "consolidated_from:3_episodes"
```

### Procedure Merge

```
Input Procedures:
- "Edit Python Function" (85% success, 12 executions)
- "Modify Python Method" (82% success, 8 executions)

→ consolidate_procedures(domain="coding")

Output:
  name: "Edit Python Function (Consolidated)"
  steps: [merged best practices from both]
  success_rate: inherits weighted average
```

## Quality Checklist

Before consolidation:
- [ ] Source episodes have sufficient similarity (>0.75)
- [ ] Pattern appears at least 3 times
- [ ] Source success rate supports consolidation
- [ ] No conflicting information in sources

After consolidation:
- [ ] Provenance chain maintained
- [ ] Source episodes linked to target
- [ ] Consolidation event logged
- [ ] Target entity embedding generated

Scheduling:
- [ ] Light consolidation: Every 2-4 hours
- [ ] Deep consolidation: Daily
- [ ] Skill consolidation: Weekly
