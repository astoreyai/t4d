# Consolidation Module

**Path**: `ww/consolidation/` | **Files**: 6 | **Lines**: ~4,500

Biologically-inspired memory consolidation with sleep phases (NREM/REM/PRUNE), STDP integration, and parallel execution.

---

## Quick Start

```python
from ww.consolidation import (
    SleepConsolidation, ConsolidationService, ConsolidationScheduler,
    ParallelExecutor,
)

# Sleep-based consolidation
sleep = SleepConsolidation(episodic, semantic, graph_store)
result = await sleep.full_sleep_cycle(session_id="my-session")
print(f"Replays: {result.nrem_replays}, Abstractions: {result.rem_abstractions}")

# Service-based consolidation
service = ConsolidationService(episodic, semantic, procedural, qdrant, neo4j)
result = await service.consolidate("deep")  # light, deep, skill, all

# Automatic scheduling
scheduler = ConsolidationScheduler(interval_hours=8, memory_threshold=100)
await scheduler.start_background_task(service.consolidate)
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                   ConsolidationScheduler (P3.3)                 │
│               (Time + Load-based auto-triggering)               │
└────────────────────────────┬───────────────────────────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                   ConsolidationService                          │
├────────────────────────────────────────────────────────────────┤
│  Light (24h)   │  Deep (7d)      │  Skill           │  All     │
│  • Duplicates  │  • HDBSCAN      │  • Merge similar │          │
│  • Soft delete │  • Entity extr  │  • Deprecate     │          │
│                │  • Relationships│  • Success track │          │
└────────────────┴─────────────────┴──────────────────┴──────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│                   SleepConsolidation                            │
├────────────────────────────────────────────────────────────────┤
│  NREM Phase              │  REM Phase          │  PRUNE Phase  │
│  ┌─────────────────────┐ │ ┌─────────────────┐ │ ┌───────────┐ │
│  │ SharpWaveRipple     │ │ │ HDBSCAN cluster │ │ │ Weak del  │ │
│  │ (10x compression)   │ │ │ Concept create  │ │ │ Homeostatic│ │
│  ├─────────────────────┤ │ └─────────────────┘ │ │ scaling   │ │
│  │ Priority replay     │ │                     │ └───────────┘ │
│  │ (PE-weighted)       │ │                     │               │
│  ├─────────────────────┤ │                     │               │
│  │ STDP updates        │ │                     │               │
│  │ Synaptic tags (P5.3)│ │                     │               │
│  └─────────────────────┘ │                     │               │
└──────────────────────────┴─────────────────────┴───────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│  FESConsolidator        │  ConsolidationSTDP   │  Parallel    │
│  (Fast→LTM transfer)    │  (P7.3 weight sync)  │  Executor    │
└─────────────────────────┴──────────────────────┴──────────────┘
```

---

## File Structure

| File | Lines | Purpose | Key Classes |
|------|-------|---------|-------------|
| `sleep.py` | 1,385 | NREM/REM/PRUNE phases | `SleepConsolidation`, `SharpWaveRipple` |
| `service.py` | 1,491 | Consolidation orchestration | `ConsolidationService`, `ConsolidationScheduler` |
| `fes_consolidator.py` | 414 | Fast→LTM transfer | `FESConsolidator`, `ReplayConsolidator` |
| `stdp_integration.py` | 463 | STDP during replay | `ConsolidationSTDP`, `STDPReplayResult` |
| `parallel.py` | 378 | 10x parallel speedup | `ParallelExecutor`, `ParallelStats` |

---

## Sleep Phases

### NREM Phase (Replay)

Replays high-priority episodes with Sharp-Wave Ripple compression.

```python
events = await sleep.nrem_phase(session_id, replay_count=100)
# Returns list of ReplayEvent
```

**Priority Scoring (P1-1)**:
```
priority = outcome_weight * outcome
         + importance_weight * importance
         + recency_weight * recency
         + prediction_error_weight * |PE|
```

**Sharp-Wave Ripples**:
- 200ms window, 10-20x temporal compression
- Coherent sequence selection by cosine similarity
- Biological timing: 500ms replay intervals (P2.5)

**Interleaved Replay (P3.4 CLS)**:
```python
# Mix 60% recent + 40% older to prevent catastrophic forgetting
batch = await sleep.get_replay_batch(recent_ratio=0.6, batch_size=100)
```

### REM Phase (Abstraction)

Clusters similar entities and creates abstract concepts.

```python
abstractions = await sleep.rem_phase(session_id)
# Returns list of AbstractionEvent
```

**Process**:
1. Cluster similar semantic entities (HDBSCAN)
2. Generate concept name from cluster members
3. Create CONCEPT entity with ABSTRACTS relationships
4. Persist centroid embedding for retrieval

### PRUNE Phase (Cleanup)

Removes weak synapses and applies homeostatic scaling.

```python
pruned, strengthened = await sleep.prune_phase()
```

**Process**:
1. Find relationships below `prune_threshold` (0.05)
2. Delete weak connections
3. Apply homeostatic scaling to normalize total weight

---

## Full Sleep Cycle

```python
result = await sleep.full_sleep_cycle(session_id)
# Runs 4 NREM-REM cycles + final prune
```

```
┌──────────────────────────────────────────────────────────┐
│              Full Sleep Cycle (~4-5 cycles)              │
├──────────────────────────────────────────────────────────┤
│  Cycle 1: NREM → REM                                     │
│  Cycle 2: NREM → REM                                     │
│  Cycle 3: NREM → REM                                     │
│  Cycle 4: NREM → REM                                     │
│  Final:   PRUNE                                          │
├──────────────────────────────────────────────────────────┤
│  Returns: SleepCycleResult                               │
│  • nrem_replays: int                                     │
│  • rem_abstractions: int                                 │
│  • pruned_connections: int                               │
│  • strengthened_connections: int                         │
└──────────────────────────────────────────────────────────┘
```

---

## ConsolidationService

Higher-level consolidation with scheduling:

### Consolidation Types

| Type | Window | Purpose |
|------|--------|---------|
| `light` | 24h | Deduplication (>95% similarity) |
| `deep` | 7d | HDBSCAN + entity extraction |
| `skill` | All | Procedure merging (>85% similarity) |
| `all` | - | Light + deep + skill + decay |

```python
service = ConsolidationService(episodic, semantic, procedural, qdrant, neo4j)

# Light: Fast deduplication
result = await service.consolidate("light")

# Deep: Entity extraction + clustering
result = await service.consolidate("deep")

# Skill: Procedure consolidation
result = await service.consolidate("skill")
```

### Automatic Scheduling (P3.3)

```python
scheduler = ConsolidationScheduler(
    interval_hours=8.0,      # Time-based trigger
    memory_threshold=100,    # Load-based trigger
    check_interval_seconds=300,
)

# Check trigger conditions
trigger = scheduler.should_consolidate()
if trigger.should_run:
    await service.consolidate(trigger.reason)

# Background task
await scheduler.start_background_task(service.consolidate)
```

---

## STDP Integration (P7.3)

Weight updates during sleep replay:

```python
from ww.consolidation import ConsolidationSTDP, ConsolidationSTDPConfig

stdp = ConsolidationSTDP(
    learner=stdp_learner,
    tagger=synaptic_tagger,
    config=ConsolidationSTDPConfig(
        replay_interval_ms=50.0,
        sync_with_tags=True,
        tag_weight_influence=0.3,
    ),
)

# Apply STDP to replay sequence
result = stdp.apply_stdp_to_sequence(
    episode_ids=["ep-1", "ep-2", "ep-3"],
    interval_ms=50.0,
)
print(f"LTP: {result.ltp_count}, LTD: {result.ltd_count}")

# Consolidate weights (decay + tag capture)
stats = stdp.consolidate_weights()

# Sync with synaptic tags
synced = stdp.sync_weights_with_tags()
```

---

## FES Consolidation

Transfer from Fast Episodic Store to permanent storage:

```python
from ww.consolidation import FESConsolidator

fes = FESConsolidator(
    fast_store=fast_episodic,
    episodic_store=episodic,
    semantic_store=semantic,
    consolidation_rate=0.1,
    min_consolidation_score=0.5,
)

# Single cycle
results = await fes.consolidate_cycle(max_episodes=10)

# Continuous until done
summary = await fes.consolidate_all()

# Background task
await fes.start_background_consolidation(interval_seconds=3600)
```

**Selection Criteria**:
- High replay count
- High salience (DA + NE + ACh weighted)
- Above threshold score

---

## Parallel Execution (PO-1)

10x speedup target for consolidation:

```python
from ww.consolidation import ParallelExecutor, ParallelConfig

executor = ParallelExecutor(ParallelConfig(
    max_workers=4,
    use_process_pool=True,
    max_concurrent_embeddings=10,
    chunk_size=500,
))

# Parallel HDBSCAN clustering
results = await executor.parallel_cluster(embeddings_list, cluster_fn)

# Parallel embedding generation
embeddings = await executor.parallel_embed(texts, embed_fn)

# Parallel storage with semaphore
await executor.parallel_store(items, store_fn)

# Chunked processing with stats
results, stats = await executor.chunked_process(items, process_fn)
print(f"Speedup: {stats.speedup_factor}x")
```

### Standalone HDBSCAN

```python
from ww.consolidation import cluster_embeddings_hdbscan

labels = cluster_embeddings_hdbscan(
    embeddings,
    min_cluster_size=5,
    min_samples=3,
    metric="cosine",
)
```

---

## Configuration

### SleepConsolidation

```python
SleepConsolidation(
    # NREM
    replay_hours=24,
    max_replays=100,
    outcome_weight=0.3,
    importance_weight=0.2,
    recency_weight=0.2,
    prediction_error_weight=0.3,  # P1-1

    # REM
    min_cluster_size=3,
    abstraction_threshold=0.7,

    # PRUNE
    prune_threshold=0.05,
    homeostatic_target=10.0,

    # Cycles
    nrem_cycles=4,
    replay_delay_ms=500,  # P2.5

    # CLS
    interleave_enabled=True,
    recent_ratio=0.6,
)
```

### ConsolidationService

```python
ConsolidationService(
    min_similarity=0.75,
    min_occurrences=3,
    skill_similarity=0.85,
    hdbscan_min_cluster_size=3,
    hdbscan_min_samples=3,
    hdbscan_metric="cosine",
)
```

### ConsolidationScheduler

```python
ConsolidationScheduler(
    interval_hours=8.0,
    memory_threshold=100,
    check_interval_seconds=300,
    consolidation_type="all",  # light, deep, skill, all
    enabled=True,
)
```

---

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| NREM Replay | O(n) | Linear in episodes |
| SWR Sequence | O(n·m²) | n episodes, m sequence (8 max) |
| REM Clustering | O(n log n) | HDBSCAN with noise |
| Prune Phase | O(e) | e relationships |
| Light Consolidation | O(n log n) | ANN duplicate search |
| Deep Consolidation | O(n log n) | Stratified sampling |
| Parallel Clustering | O(n log n / p) | p workers |

---

## Biological Basis

| Mechanism | Biological Source | Implementation |
|-----------|------------------|----------------|
| Sharp-Wave Ripples | Hippocampal 100ms bursts | `SharpWaveRipple` class |
| NREM Replay | Recent memory consolidation | Priority-based replay |
| REM Abstraction | Dream pattern finding | HDBSCAN + concept creation |
| Synaptic Pruning | Homeostatic downscaling | Weight threshold deletion |
| Synaptic Tagging | CaMKII marking | `SynapticTagger` integration |
| Interleaved Replay | CLS theory | 60/40 recent/older mixing |
| STDP | Hebbian learning | Weight updates during replay |

---

## Dependencies

**Internal**:
- `ww.memory` - Episodic, semantic, procedural stores
- `ww.storage` - Qdrant, Neo4j backends
- `ww.learning` - STDP, synaptic tagging

**External**:
- `hdbscan` - Clustering
- `numpy` - Numerical operations
- `asyncio` - Async execution
- `concurrent.futures` - ProcessPool
