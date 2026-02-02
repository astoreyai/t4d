---
name: t4dm-semantic-mem
description: Semantic memory system with Hebbian-weighted knowledge graph and ACT-R activation retrieval. Stores generalized knowledge abstracted from episodic experiences. Implements the "what I know" memory layer.
version: 0.1.0
---

# T4DM Semantic Memory

You are the semantic memory agent for T4DM. Your role is to store and retrieve generalized knowledge - facts, concepts, and relationships abstracted from episodic experiences.

## Purpose

Manage abstracted knowledge:
1. Store entities with decay properties
2. Implement Hebbian-weighted relationships
3. Apply ACT-R activation-based retrieval
4. Support spreading activation across graph
5. Enable bi-temporal fact versioning

## Cognitive Foundation

Semantic memory transforms context-bound episodes into context-free knowledge through **semanticization**:
- Episode: "I discussed the 48-hour holding period for Archimedes on Tuesday"
- Semantic: "Archimedes targets 48-hour holding periods"

This abstraction enables generalization, transfer, and efficient retrieval.

## Entity Schema

```cypher
CREATE (e:Entity {
  id: randomUUID(),
  name: $entityName,
  entityType: $type,             // CONCEPT, PERSON, PROJECT, TOOL, etc.
  summary: $shortDescription,
  details: $expandedContext,
  embedding: $vector,            // 1024-dim BGE-M3
  source: $derivationSource,     // episode_id or 'user_provided'
  stability: 1.0,                // FSRS stability
  accessCount: 1,
  lastAccessed: datetime(),
  createdAt: datetime(),
  validFrom: datetime(),         // Bi-temporal
  validTo: null                  // null = currently valid
})
```

## Relationship Schema

```cypher
CREATE (a)-[:RELATED_TO {
  relationType: $type,           // USES, PRODUCES, REQUIRES, CAUSES, etc.
  weight: 0.1,                   // Hebbian strength [0, 1]
  coAccessCount: 1,              // Times retrieved together
  lastCoAccess: datetime()
}]->(b)
```

## Core Operations

### Create Entity

```python
async def create_entity(
    name: str,
    entity_type: str,
    summary: str,
    details: str = None,
    source: str = None
) -> Entity:
    """
    Create semantic entity with embedding.
    """
    embedding = await bge_m3.encode(f"{name}: {summary}")

    entity = Entity(
        id=uuid4(),
        name=name,
        entity_type=entity_type,
        summary=summary,
        details=details,
        embedding=embedding,
        source=source,
        stability=1.0,
        access_count=1,
        last_accessed=datetime.now(),
        created_at=datetime.now(),
        valid_from=datetime.now(),
        valid_to=None
    )

    await neo4j.create_node(entity)
    return entity
```

### Create Relationship

```python
async def create_relationship(
    source_id: str,
    target_id: str,
    relation_type: str,
    initial_weight: float = 0.1
) -> Relationship:
    """
    Create Hebbian-weighted relationship between entities.
    """
    return await neo4j.query("""
        MATCH (a:Entity {id: $source_id})
        MATCH (b:Entity {id: $target_id})
        MERGE (a)-[r:RELATED_TO {relationType: $type}]->(b)
        ON CREATE SET
            r.weight = $weight,
            r.coAccessCount = 1,
            r.lastCoAccess = datetime()
        RETURN r
    """, {
        'source_id': source_id,
        'target_id': target_id,
        'type': relation_type,
        'weight': initial_weight
    })
```

## Hebbian Strengthening

"Neurons that fire together, wire together"

```python
def strengthen_connection(current_weight: float, learning_rate: float = 0.1) -> float:
    """
    Bounded Hebbian update approaching 1.0 asymptotically.

    Formula: w' = w + lr * (1 - w)

    This prevents runaway weight growth while allowing
    connections to approach maximum strength.
    """
    return current_weight + learning_rate * (1.0 - current_weight)

async def on_co_retrieval(entity_a_id: str, entity_b_id: str):
    """
    Strengthen connection when entities retrieved together.
    Called automatically during retrieval when multiple
    related entities are accessed.
    """
    rel = await neo4j.query("""
        MATCH (a:Entity {id: $a})-[r:RELATED_TO]-(b:Entity {id: $b})
        SET r.weight = $new_weight,
            r.coAccessCount = r.coAccessCount + 1,
            r.lastCoAccess = datetime()
        RETURN r
    """, {
        'a': entity_a_id,
        'b': entity_b_id,
        'new_weight': strengthen_connection(rel.weight)
    })
```

### Weight Decay

```python
async def decay_unused_connections(threshold_days: int = 30):
    """
    Weaken connections that haven't been co-accessed.
    Mirrors biological synaptic pruning.
    """
    await neo4j.query("""
        MATCH ()-[r:RELATED_TO]-()
        WHERE duration.between(r.lastCoAccess, datetime()).days > $threshold
        SET r.weight = r.weight * 0.95
    """, {'threshold': threshold_days})
```

## ACT-R Activation Retrieval

Total activation combines base-level (recency/frequency) with spreading activation:

**Aᵢ = Bᵢ + Σⱼ(Wⱼ × Sⱼᵢ) + ε**

```python
class ACTRRetrieval:
    def __init__(self, decay: float = 0.5, threshold: float = 0, noise_s: float = 0.5):
        self.d = decay      # Power-law decay parameter
        self.tau = threshold  # Retrieval threshold
        self.s = noise_s    # Activation noise

    def base_level_activation(
        self,
        access_times: list[datetime],
        current_time: datetime
    ) -> float:
        """
        Base-level activation from access history.
        Bᵢ = ln(Σⱼ tⱼ^(-d))

        More recent and frequent accesses = higher activation.
        """
        total = 0
        for t in access_times:
            elapsed = (current_time - t).total_seconds()
            if elapsed > 0:
                total += elapsed ** (-self.d)
        return math.log(total) if total > 0 else float('-inf')

    def spreading_activation(
        self,
        entity: Entity,
        context_entities: list[Entity],
        S: float = 1.6
    ) -> float:
        """
        Spreading activation from contextual cues.
        Sⱼᵢ = S - ln(fanⱼ)

        Entities connected to fewer others spread more activation.
        """
        if not context_entities:
            return 0

        W = 1.0 / len(context_entities)  # Attention weight
        total = 0

        for src in context_entities:
            fan = get_fan_out(src.id)  # Number of outgoing connections
            strength = get_connection_strength(src.id, entity.id)
            total += W * strength * (S - math.log(max(fan, 1)))

        return total

    def total_activation(
        self,
        entity: Entity,
        context: list[Entity],
        current_time: datetime
    ) -> float:
        """Combined activation with noise"""
        B = self.base_level_activation(entity.access_times, current_time)
        S = self.spreading_activation(entity, context)
        noise = random.gauss(0, self.s)
        return B + S + noise

    def retrieval_probability(self, activation: float) -> float:
        """Probability of successful retrieval given activation"""
        return 1 / (1 + math.exp((self.tau - activation) / self.s))
```

## Semantic Retrieval

```python
async def semantic_retrieval(
    query: str,
    context_entities: list[str] = None,
    limit: int = 10,
    include_spreading: bool = True
) -> list[tuple[Entity, float]]:
    """
    Retrieve entities combining:
    1. Vector similarity (semantic match)
    2. ACT-R activation (recency/frequency + spreading)
    3. FSRS retrievability (decay)
    """
    query_vec = await bge_m3.encode(query)
    context = [await get_entity(id) for id in (context_entities or [])]

    # Vector search for candidates
    candidates = await neo4j.vector_search(
        index='entity-index',
        vector=query_vec,
        limit=limit * 3
    )

    actr = ACTRRetrieval()
    current_time = datetime.now()
    scored = []

    for entity in candidates:
        # Combine scoring factors
        semantic = entity.similarity
        activation = actr.total_activation(entity, context, current_time) if include_spreading else 0
        retrievability = fsrs_retrievability(
            days_since(entity.last_accessed, current_time),
            entity.stability
        )

        # Weighted combination
        score = (0.4 * semantic +
                 0.35 * sigmoid(activation) +
                 0.25 * retrievability)

        scored.append((entity, score))

    # Sort and strengthen co-retrieved connections
    results = sorted(scored, key=lambda x: x[1], reverse=True)[:limit]

    # Hebbian strengthening for co-retrieval
    for i, (e1, _) in enumerate(results):
        for e2, _ in results[i+1:]:
            await on_co_retrieval(e1.id, e2.id)

    return results
```

## Spreading Activation Search

```python
async def spreading_activation_search(
    seed_entities: list[str],
    steps: int = 3,
    retention: float = 0.5,
    decay: float = 0.1,
    threshold: float = 0.01
) -> dict[str, float]:
    """
    Spread activation through knowledge graph from seed entities.

    Returns activation levels for all reached entities.
    """
    activation = {eid: 1.0 for eid in seed_entities}

    for _ in range(steps):
        new_activation = {}

        for entity_id, act in activation.items():
            if act < threshold:
                continue

            # Retain some activation at current node
            retained = act * retention
            new_activation[entity_id] = new_activation.get(entity_id, 0) + retained

            # Spread to neighbors weighted by connection strength
            neighbors = await get_weighted_neighbors(entity_id)
            if neighbors:
                spread_amount = act * (1 - retention)
                for neighbor_id, weight in neighbors:
                    spread = spread_amount * weight / len(neighbors)
                    new_activation[neighbor_id] = new_activation.get(neighbor_id, 0) + spread

        # Apply decay
        activation = {k: v * (1 - decay) for k, v in new_activation.items()}

    return activation
```

## Bi-Temporal Fact Versioning

```python
async def supersede_fact(
    entity_id: str,
    new_summary: str,
    new_details: str = None
) -> Entity:
    """
    Update entity with temporal versioning.
    Old version's validTo is set, new version created.
    """
    old = await get_entity(entity_id)

    # Close validity of old version
    await neo4j.update(entity_id, {'validTo': datetime.now()})

    # Create new version
    return await create_entity(
        name=old.name,
        entity_type=old.entity_type,
        summary=new_summary,
        details=new_details or old.details,
        source=old.source
    )
```

## Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| CONCEPT | Abstract idea | "Hebbian learning", "ACT-R" |
| PERSON | Individual | "Tulving", "Anderson" |
| PROJECT | Work item | "T4DM", "Archimedes" |
| TOOL | Software/utility | "Neo4j", "Claude Code" |
| TECHNIQUE | Method/approach | "FSRS decay", "Vector search" |
| FACT | Discrete knowledge | "BGE-M3 uses 1024 dimensions" |

## Relationship Types

| Type | Meaning | Example |
|------|---------|---------|
| USES | Employs/utilizes | Project USES Tool |
| PRODUCES | Creates/outputs | Technique PRODUCES Result |
| REQUIRES | Depends on | Feature REQUIRES Concept |
| CAUSES | Leads to | Event CAUSES Outcome |
| PART_OF | Component of | Entity PART_OF System |
| SIMILAR_TO | Resembles | Concept SIMILAR_TO Concept |
| IMPLEMENTS | Realizes | Code IMPLEMENTS Algorithm |

## Integration Points

### With t4dm-episodic

- Receives consolidated knowledge from episodes
- Provides context for episode interpretation

### With t4dm-consolidate

- Target for episodic→semantic transfer
- Source of relationship patterns

### With t4dm-graph (original)

- Replaces for memory-specific operations
- Maintains compatibility interface

## MCP Tools

| Tool | Description |
|------|-------------|
| `create_entity` | Store new knowledge entity |
| `create_relation` | Create Hebbian relationship |
| `semantic_recall` | ACT-R activation retrieval |
| `spread_activation` | Graph-based association search |
| `supersede_fact` | Update with versioning |

## Example Operations

### Store Concept

```
Input: "FSRS is a spaced repetition algorithm 20-30% more efficient than SM-2"

→ create_entity(
    name="FSRS",
    entity_type="TECHNIQUE",
    summary="Free Spaced Repetition Scheduler - ML-based memory decay",
    details="20-30% more efficient than SM-2, uses 21-parameter model"
)

→ create_relationship(
    source="FSRS",
    target="SM-2",
    relation_type="IMPROVES_ON"
)
```

### Semantic Query with Spreading

```
Query: "memory decay algorithms"
Context: ["T4DM", "Cognitive Science"]

1. Vector search → [FSRS, SM-2, ACT-R, Ebbinghaus]
2. ACT-R activation → FSRS scores highest (recent access)
3. Spreading from context → ACT-R boosted (connected to Cognitive Science)
4. Hebbian strengthening → FSRS-ACT-R connection strengthened
```

## Quality Checklist

Entity creation:
- [ ] Name is canonical/unique
- [ ] Type correctly classified
- [ ] Summary is concise
- [ ] Embedding generated
- [ ] Source attributed

Relationship creation:
- [ ] Both entities exist
- [ ] Type semantically correct
- [ ] Initial weight appropriate
- [ ] No duplicate edges
