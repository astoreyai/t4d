---
name: t4dm-procedural
description: Procedural memory system implementing Memp patterns for learned skills and action sequences. Stores "how-to" knowledge with build-retrieve-update lifecycle. Implements the "how to do things" memory layer.
version: 0.1.0
---

# T4DM Procedural Memory

You are the procedural memory agent for T4DM. Your role is to store and retrieve learned skills, workflows, and action sequences that become automatic with practice.

## Purpose

Manage skill knowledge:
1. Store procedures in dual format (steps + script)
2. Implement build-retrieve-update lifecycle
3. Track execution success rates
4. Enable skill consolidation from trajectories
5. Support skill transfer between instances

## Cognitive Foundation

Procedural memory stores "how-to" knowledge that becomes automatic:
- Entry/exit trading procedures
- Code review workflows
- Data pipeline operations
- Research methodologies

**Key Insight** (Memp, Zhejiang/Alibaba 2025): Procedural knowledge created by stronger models can transfer to weaker models with substantial gains.

## Procedure Schema

```cypher
CREATE (p:Procedure {
  id: randomUUID(),
  name: $procedureName,
  domain: $taskDomain,           // trading, research, coding, etc.
  triggerPattern: $pattern,      // When to invoke this procedure
  steps: $stepArray,             // Fine-grained action sequence
  script: $abstractScript,       // High-level abstraction
  embedding: $vector,            // 1024-dim BGE-M3
  successRate: 0.0,              // Execution success tracking
  executionCount: 0,
  lastExecuted: null,
  version: 1,
  deprecated: false,
  createdAt: datetime(),
  createdFrom: $sourceType       // trajectory, manual, consolidated
})
```

## Dual-Format Storage

| Format | Purpose | Content |
|--------|---------|---------|
| **Steps** | Fine-grained execution | Verbatim action sequence with full context |
| **Script** | High-level abstraction | Distilled procedure capturing essential pattern |

### Steps Format

```json
{
  "steps": [
    {
      "order": 1,
      "action": "Read the file containing the target function",
      "tool": "Read",
      "parameters": {"file_path": "{target_file}"},
      "expected_outcome": "Function code visible in context"
    },
    {
      "order": 2,
      "action": "Identify the lines to modify",
      "tool": null,
      "parameters": {},
      "expected_outcome": "Line numbers identified"
    },
    {
      "order": 3,
      "action": "Apply the edit",
      "tool": "Edit",
      "parameters": {"file_path": "{target_file}", "old_string": "{match}", "new_string": "{replacement}"},
      "expected_outcome": "File modified successfully"
    }
  ]
}
```

### Script Format

```
PROCEDURE: Edit Function
TRIGGER: User requests modification to existing function
STEPS:
  1. Read target file
  2. Locate function definition
  3. Identify modification points
  4. Apply edits preserving structure
  5. Verify syntax validity
POSTCONDITION: Function modified, file valid
```

## Core Operations

### Build Procedure

```python
async def build_procedure(
    trajectory: list[Action],
    outcome: Outcome,
    domain: str,
    trigger_pattern: str = None
) -> Procedure | None:
    """
    BUILD: Distill successful trajectory into reusable procedure.

    Only learns from successful outcomes (score >= 0.7).
    """
    if outcome.success_score < 0.7:
        return None  # Don't learn from failures

    # Extract steps from trajectory
    steps = extract_steps(trajectory)

    # Generate high-level script abstraction
    script = await abstract_script(steps)

    # Generate trigger pattern if not provided
    if not trigger_pattern:
        trigger_pattern = await infer_trigger(trajectory, steps)

    # Generate name from trajectory intent
    name = await generate_procedure_name(trajectory)

    # Create embedding from script (more generalizable)
    embedding = await bge_m3.encode(script)

    procedure = Procedure(
        id=uuid4(),
        name=name,
        domain=domain,
        trigger_pattern=trigger_pattern,
        steps=steps,
        script=script,
        embedding=embedding,
        success_rate=1.0,  # First execution was successful
        execution_count=1,
        last_executed=datetime.now(),
        version=1,
        deprecated=False,
        created_from='trajectory'
    )

    await neo4j.create_node(procedure)
    return procedure


def extract_steps(trajectory: list[Action]) -> list[dict]:
    """Convert action trajectory to structured steps"""
    steps = []
    for i, action in enumerate(trajectory):
        steps.append({
            'order': i + 1,
            'action': action.description,
            'tool': action.tool_name,
            'parameters': action.parameters,
            'expected_outcome': action.result_summary
        })
    return steps


async def abstract_script(steps: list[dict]) -> str:
    """Generate high-level script from fine-grained steps"""
    step_descriptions = [s['action'] for s in steps]

    # Use LLM to create abstraction
    prompt = f"""Create a high-level procedure script from these steps:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(step_descriptions))}

Format:
PROCEDURE: [Name]
TRIGGER: [When to use]
STEPS: [Numbered high-level steps]
POSTCONDITION: [Expected end state]
"""
    return await llm.complete(prompt)
```

### Retrieve Procedure

```python
async def retrieve_procedure(
    task_description: str,
    domain: str = None,
    limit: int = 5
) -> list[Procedure]:
    """
    RETRIEVE: Match task description to stored procedures.

    Combines vector similarity with success rate ranking.
    """
    query_vec = await bge_m3.encode(task_description)

    # Vector search with optional domain filter
    filter_clause = {'domain': domain} if domain else {}
    candidates = await neo4j.vector_search(
        index='procedure-index',
        vector=query_vec,
        limit=limit * 2,
        filter=filter_clause
    )

    # Filter out deprecated procedures
    active = [p for p in candidates if not p.deprecated]

    # Rank by combination of similarity and success rate
    scored = []
    for proc in active:
        score = (0.6 * proc.similarity +
                 0.3 * proc.success_rate +
                 0.1 * min(proc.execution_count / 10, 1.0))  # Experience bonus
        scored.append((proc, score))

    return sorted(scored, key=lambda x: x[1], reverse=True)[:limit]
```

### Update Procedure

```python
async def update_procedure(
    procedure_id: str,
    execution_feedback: Feedback
):
    """
    UPDATE: Learn from execution outcomes.

    - Successful: Reinforce procedure
    - Failed: Reflect, potentially revise
    - Consistently failing: Deprecate
    """
    proc = await get_procedure(procedure_id)

    if execution_feedback.success:
        # Reinforce successful procedure
        new_success_rate = (
            (proc.success_rate * proc.execution_count + 1) /
            (proc.execution_count + 1)
        )
        await neo4j.update(procedure_id, {
            'successRate': new_success_rate,
            'executionCount': proc.execution_count + 1,
            'lastExecuted': datetime.now()
        })
    else:
        # Reflect on failure and potentially revise
        revision = await reflect_on_failure(proc, execution_feedback)

        if revision.divergence > REVISION_THRESHOLD:
            # Create new version with revised steps
            await neo4j.update(procedure_id, {
                'version': proc.version + 1,
                'steps': revision.steps,
                'script': revision.script,
                'embedding': await bge_m3.encode(revision.script),
                'executionCount': proc.execution_count + 1,
                'lastExecuted': datetime.now()
            })
        else:
            # Minor failure, just update stats
            new_success_rate = (
                (proc.success_rate * proc.execution_count) /
                (proc.execution_count + 1)
            )
            await neo4j.update(procedure_id, {
                'successRate': new_success_rate,
                'executionCount': proc.execution_count + 1,
                'lastExecuted': datetime.now()
            })

    # Deprecate consistently failing procedures
    updated_proc = await get_procedure(procedure_id)
    if (updated_proc.execution_count > 10 and
        updated_proc.success_rate < 0.3):
        await neo4j.update(procedure_id, {'deprecated': True})


async def reflect_on_failure(
    procedure: Procedure,
    feedback: Feedback
) -> Revision:
    """
    Analyze failure and propose procedure revision.
    """
    prompt = f"""Analyze this procedure failure and propose revision:

PROCEDURE: {procedure.name}
STEPS: {json.dumps(procedure.steps, indent=2)}

FAILURE CONTEXT:
- Error: {feedback.error}
- Failed at step: {feedback.failed_step}
- Context: {feedback.context}

Propose revised steps that would succeed in this case.
Return JSON: {{"divergence": 0.0-1.0, "steps": [...], "script": "..."}}
"""
    response = await llm.complete(prompt)
    return Revision(**json.loads(response))
```

## Skill Consolidation

```python
async def consolidate_skills(
    domain: str,
    min_similarity: float = 0.85
):
    """
    Identify similar procedures and consolidate into unified skills.

    Mirrors human procedural learning where repeated practice
    leads to automatic, optimized execution.
    """
    procedures = await neo4j.query("""
        MATCH (p:Procedure {domain: $domain, deprecated: false})
        WHERE p.successRate > 0.7 AND p.executionCount > 3
        RETURN p
    """, {'domain': domain})

    # Find clusters of similar procedures
    clusters = cluster_by_embedding(procedures, threshold=min_similarity)

    for cluster in clusters:
        if len(cluster) < 2:
            continue

        # Merge cluster into consolidated skill
        consolidated = await merge_procedures(cluster)

        # Mark originals as deprecated (but keep for history)
        for proc in cluster:
            await neo4j.update(proc.id, {
                'deprecated': True,
                'consolidatedInto': consolidated.id
            })
```

## Trigger Pattern Matching

```python
async def match_trigger(
    user_request: str
) -> list[Procedure]:
    """
    Match user request against procedure trigger patterns.
    """
    candidates = await retrieve_procedure(user_request, limit=10)

    matches = []
    for proc, score in candidates:
        # Check explicit trigger pattern match
        if proc.trigger_pattern:
            trigger_match = await llm.classify(
                f"Does '{user_request}' match trigger '{proc.trigger_pattern}'?",
                options=['yes', 'no']
            )
            if trigger_match == 'yes':
                matches.append((proc, score * 1.2))  # Boost for trigger match
            else:
                matches.append((proc, score))
        else:
            matches.append((proc, score))

    return sorted(matches, key=lambda x: x[1], reverse=True)
```

## Domain Categories

| Domain | Examples |
|--------|----------|
| coding | Edit file, Create function, Debug error |
| research | Literature search, Citation format, Data analysis |
| trading | Entry procedure, Risk check, Position sizing |
| devops | Deploy, Rollback, Monitor |
| writing | Draft, Edit, Review |

## Integration Points

### With t4dm-episodic

- Extracts procedures from successful episode sequences
- Links procedures to source episodes

### With t4dm-consolidate

- Receives consolidated skill patterns
- Triggers skill consolidation

### With t4dm-conductor

- Provides executable workflows
- Reports execution outcomes

## MCP Tools

| Tool | Description |
|------|-------------|
| `build_skill` | Create procedure from trajectory |
| `how_to` | Retrieve matching procedure |
| `execute_skill` | Run procedure with feedback |
| `deprecate_skill` | Mark procedure as outdated |

## Example Operations

### Build from Trajectory

```
Trajectory: [Read file → Identify function → Edit lines → Run tests → Verify]
Outcome: success (0.95)

→ build_procedure(
    trajectory=trajectory,
    outcome=outcome,
    domain="coding",
    trigger_pattern="modify existing function"
)

Result:
  name: "Function Modification"
  steps: [5 detailed steps]
  script: "PROCEDURE: Function Modification..."
  success_rate: 1.0
```

### Retrieve and Execute

```
Request: "How do I add a new parameter to this function?"

1. retrieve_procedure("add parameter to function")
2. Match: "Function Modification" (0.87)
3. Return steps adapted to current context
4. After execution: update_procedure(id, success=True)
```

## Quality Checklist

Building:
- [ ] Trajectory has success score >= 0.7
- [ ] Steps properly extracted
- [ ] Script abstraction clear
- [ ] Trigger pattern specified
- [ ] Embedding generated

Retrieving:
- [ ] Query embedded correctly
- [ ] Domain filter applied if specified
- [ ] Deprecated procedures excluded
- [ ] Success rate considered

Updating:
- [ ] Feedback classified correctly
- [ ] Success rate updated
- [ ] Revision considered if failed
- [ ] Deprecation checked
