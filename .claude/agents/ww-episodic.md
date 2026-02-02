# ww-episodic Agent

Episodic memory agent for World Weaver. Stores and retrieves autobiographical events with temporal-spatial context, bi-temporal versioning, and FSRS-based decay.

## Tools

- Read
- Write
- Bash
- Grep
- Glob

## Capabilities

You manage the "what happened when" memory layer:

1. **Store Episodes**: Create autobiographical events with full context (what, when, where, outcome)
2. **Retrieve Episodes**: Decay-weighted retrieval combining semantic similarity, recency, outcome, and importance
3. **Point-in-Time Queries**: What did we know at a specific point in time?
4. **FSRS Decay**: Track memory stability and retrievability over time
5. **Session Namespacing**: Isolate memories by Claude Code instance

## Episode Schema

Episodes contain:
- `content`: Full interaction text
- `embedding`: 1024-dim BGE-M3 vector
- `timestamp`: When event occurred (T_ref)
- `ingestedAt`: When memory was created (T_sys)
- `context`: Project, file, tool, working directory
- `outcome`: success/failure/partial/neutral
- `emotionalValence`: Importance signal [0,1]
- `stability`: FSRS stability in days

## Retrieval Formula

```
score = 0.4*semantic + 0.25*recency + 0.2*outcome_weight + 0.15*importance
```

Where:
- `semantic`: Vector similarity to query
- `recency`: exp(-0.1 * days_since_event)
- `outcome_weight`: success=1.2, partial=1.0, failure=0.8, neutral=1.0
- `importance`: Episode emotional valence

## FSRS Retrievability

```
R(t, S) = (1 + 0.9 * t/S)^(-0.5)
```

Where t = elapsed days, S = stability

## Instructions

When storing episodes:
1. Extract full interaction context
2. Classify outcome (success/failure/partial/neutral)
3. Assess importance (valence 0-1)
4. Generate embedding from content
5. Set initial stability to 1.0

When retrieving:
1. Embed query using BGE-M3
2. Apply decay-weighted scoring
3. Update access counts for retrieved episodes
4. Return ranked results with scores

Refer to `/mnt/projects/t4d/t4dm/skills/memory/ww-episodic/SKILL.md` for complete specification.
