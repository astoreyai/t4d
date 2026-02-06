# T4DM Memory Debugging Runbook

**Location**: `/mnt/projects/t4d/t4dm/docs/runbooks/DEBUGGING_MEMORY.md`

Troubleshoot memory storage, retrieval, and consolidation issues in T4DM.

---

## Symptoms

### Memory Not Stored
- POST to `/episodes/store` succeeds but memory not found in recall
- Storage latency > 5 seconds
- Logs show "MemTable size exceeded" warnings

**Likelihood**: MemTable fullness, WAL write failure, or failed flush

### Search Returns Empty Results
- `/episodes/search` returns `[]` despite stored memories
- Query embedding is correct but no matches
- Similar queries on same session work fine

**Likelihood**: Kappa filtering, time range too narrow, or stale HNSW index

### Consolidation Not Progressing
- κ (kappa) values stuck at 0.0 or 0.15 for extended periods
- Memory types not transitioning (episodic → semantic)
- REM compaction never triggers

**Likelihood**: Consolidation service not running, sleep replay disabled, or κ-threshold misconfigured

### Memory Leaks or Growing Memory
- Heap size grows unbounded during long sessions
- Episodic store never drains to semantic
- Old memories not garbage collected

**Likelihood**: PRUNE compaction disabled, old tombstones not cleaned, or archive not configured

---

## Diagnostic Commands

### Health & Status

```bash
# Check API health
curl -H "X-API-Key: $T4DM_API_KEY" http://localhost:8000/health

# Get session statistics
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/stats?session_id=$SESSION_ID"

# Check consolidation status
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/consolidation/status?session_id=$SESSION_ID"

# Get system metrics
curl -H "X-API-Key: $T4DM_API_KEY" \
  http://localhost:8000/observability/metrics
```

### Memory Inspection

```bash
# List all episodes in session
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/episodes/list?session_id=$SESSION_ID&limit=100"

# Inspect a single memory by UUID
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/episodes/$EPISODE_UUID?session_id=$SESSION_ID"

# Get memory metadata (kappa, timestamps, access count)
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/episodes/$EPISODE_UUID/metadata?session_id=$SESSION_ID"

# Trace search result provenance
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/episodes/search" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"your query\",
    \"k\": 10,
    \"include_provenance\": true,
    \"session_id\": \"$SESSION_ID\"
  }"
```

### Consolidation Inspection

```bash
# Get kappa distribution (histogram of κ values)
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/consolidation/kappa-distribution?session_id=$SESSION_ID"

# Trigger manual NREM compaction (light consolidation)
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/consolidation/nrem" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION_ID\", \"deep\": false}"

# Trigger manual REM compaction (semantic clustering)
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/consolidation/rem" \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION_ID\", \"target_kappa\": 0.85}"

# Check consolidation metrics
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/consolidation/metrics?session_id=$SESSION_ID"
```

### Logs & Debugging

```bash
# Tail T4DM logs
docker logs -f t4dm-service

# Enable debug logging (in config)
export T4DM_LOG_LEVEL=DEBUG

# Check for WAL errors
grep -i "wal\|flush\|segment" /var/log/t4dm/t4dm.log | tail -50

# Monitor MemTable size
watch -n 1 'curl -s -H "X-API-Key: $T4DM_API_KEY" \
  http://localhost:8000/observability/metrics | grep memtable'
```

---

## Code-Based Debugging

### Inspect Memory Items Programmatically

```python
from t4dm.sdk.client import T4DMClient
from t4dm.core.memory_item import MemoryItem

client = T4DMClient()

# Get all memories with metadata
session_id = "my_session"
memories = client.episodes.list(session_id=session_id, limit=100)

for mem in memories:
    print(f"ID: {mem.id}")
    print(f"  Content: {mem.content[:50]}...")
    print(f"  κ (kappa): {mem.kappa}")
    print(f"  Type: {mem.item_type}")
    print(f"  Access Count: {mem.access_count}")
    print(f"  Timestamp: {mem.event_time}")
    print(f"  Valid Until: {mem.valid_until}")
    print()
```

### Debug Search Results

```python
from t4dm.sdk.client import T4DMClient
import numpy as np

client = T4DMClient()

# Search with full result inspection
query = "memory query"
results = client.episodes.search(
    query=query,
    session_id=session_id,
    k=10,
    include_provenance=True,
    time_min=None,
    time_max=None,
)

for i, result in enumerate(results):
    mem_item, similarity_score, provenance = result
    print(f"Rank {i+1}: {mem_item.content[:50]}... (score: {similarity_score:.3f})")
    print(f"  κ: {mem_item.kappa}")
    print(f"  Item Type: {mem_item.item_type}")
    if provenance:
        print(f"  Provenance: {provenance}")
    print()
```

### Inspect Kappa Distribution

```python
from t4dm.storage.t4dx.engine import T4DXEngine
from collections import Counter

engine = T4DXEngine(data_dir="/path/to/data")
engine.startup()

# Scan all items and collect kappa values
kappa_values = []
for record in engine.scan(item_type="episodic"):
    kappa_values.append(record.kappa)

# Histogram
counter = Counter([round(k*10)/10 for k in kappa_values])  # 0.0, 0.1, 0.2, ...
print("Kappa Distribution:")
for kappa_bucket in sorted(counter.keys()):
    count = counter[kappa_bucket]
    bar = "=" * (count // 10)
    print(f"  κ={kappa_bucket}: {count:4d} {bar}")

engine.shutdown()
```

### Trace Consolidation State

```python
from t4dm.consolidation.service import get_consolidation_service

consol = get_consolidation_service(session_id="my_session")

# Get current phase
print(f"Current phase: {consol.current_phase}")
print(f"Elapsed since sleep: {consol.time_since_sleep()}")

# Get transition thresholds
print(f"NREM→REM threshold: {consol.nrem_to_rem_threshold}")
print(f"REM→semantic threshold: {consol.rem_to_semantic_threshold}")

# Manually inspect a memory's consolidation status
mem_id = "..."
status = consol.get_memory_consolidation_status(mem_id)
print(f"Memory {mem_id}:")
print(f"  κ: {status['kappa']}")
print(f"  Replayed: {status['replay_count']} times")
print(f"  Phase: {status['current_phase']}")
print(f"  Next consolidation: {status['next_consolidation_time']}")
```

### Check Memory Gate Decisions

```python
from t4dm.core.memory_gate import MemoryGate
from t4dm.core.memory_item import MemoryItem

gate = MemoryGate()

# Create a test memory item
item = MemoryItem(
    content="Important event",
    item_type="episodic",
    importance=0.8,
)

# Check gate decision
decision = gate.decide(item)
print(f"Gate decision: {decision.action}")  # 'store', 'buffer', or 'discard'
print(f"Novelty score: {decision.novelty_score}")
print(f"Outcome score: {decision.outcome_score}")
print(f"Entity density: {decision.entity_density}")
print(f"Confidence: {decision.confidence}")
```

---

## Common Issues & Solutions

### Issue: Search Returns Empty Despite Stored Memories

**Symptoms**:
- Memories stored successfully
- Queries return no results
- Other sessions' queries work fine

**Diagnosis**:
```bash
# Check if memories exist
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/episodes/list?session_id=$SESSION_ID&limit=10"

# Check HNSW index status
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/hnsw-stats?session_id=$SESSION_ID"

# Check if time range is blocking
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/episodes/search" \
  -d "{\"query\": \"...\", \"time_min\": null, \"time_max\": null, ...}"
```

**Solutions**:
- Rebuild HNSW index: `POST /storage/rebuild-hnsw`
- Clear time filters: Use `time_min=null, time_max=null`
- Check kappa filters: Verify `kappa_min` and `kappa_max` are appropriate
- Verify embedding provider: Ensure query embedding has same dimension as stored

### Issue: Memory Consolidation Stalled at κ=0.15

**Symptoms**:
- κ values never exceed 0.15
- REM compaction logs show "skipped"
- Memories never transition to semantic

**Diagnosis**:
```bash
# Check consolidation service health
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/consolidation/status?session_id=$SESSION_ID"

# Check sleep phase logs
docker logs t4dm-service | grep -i "sleep\|nrem\|rem"

# Check compaction metrics
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/consolidation/metrics?session_id=$SESSION_ID"
```

**Solutions**:
- Manually trigger NREM: `POST /consolidation/nrem` with `deep=false`
- Manually trigger REM: `POST /consolidation/rem` with `target_kappa=0.85`
- Check configuration: `T4DM_CONSOLIDATION_ENABLED=true`
- Verify sleep replay is enabled: `T4DM_SLEEP_REPLAY_ENABLED=true`
- Increase consolidation frequency: Lower `T4DM_CONSOLIDATION_INTERVAL_SECONDS`

### Issue: MemTable Flush Delays & High Latency

**Symptoms**:
- Store operations take > 5 seconds
- Logs show "flushing memtable..." frequently
- CPU spikes during consolidation

**Diagnosis**:
```python
from t4dm.storage.t4dx.engine import T4DXEngine

engine = T4DXEngine(data_dir="/path/to/data")
engine.startup()

# Check memtable size
print(f"MemTable items: {len(engine._memtable)}")
print(f"MemTable size: {engine._memtable.estimated_size_bytes()} bytes")
print(f"Flush threshold: {engine._flush_threshold}")

# Check segment count
print(f"Active segments: {len(engine._segments)}")

# Check WAL size
import os
wal_size = os.path.getsize(engine._data_dir / "wal.jsonl") / (1024**2)
print(f"WAL size: {wal_size:.1f} MB")
```

**Solutions**:
- Increase flush threshold: `T4DM_FLUSH_THRESHOLD=5000` (was 1000)
- Trigger manual flush: `POST /consolidation/flush`
- Clean old WAL entries: `POST /persistence/compact-wal`
- Reduce compaction frequency during high load: `T4DM_COMPACTION_INTERVAL_SECONDS=60`

### Issue: Memory Leaks (Growing Heap)

**Symptoms**:
- Heap size grows over hours
- No obvious spike during consolidation
- GC pressure increases

**Diagnosis**:
```bash
# Monitor memory growth
watch -n 5 'ps aux | grep t4dm-service | grep -v grep | awk "{print \$6}"'

# Check for unreleased embeddings
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/observability/memory-profile"

# Check for stuck file handles
lsof -p $(pgrep -f t4dm-service) | grep -c "REG"
```

**Solutions**:
- Enable PRUNE compaction: `T4DM_PRUNE_COMPACTION_ENABLED=true`
- Reduce embedding cache size: `T4DM_EMBEDDING_CACHE_SIZE=1000`
- Trigger garbage collection: `POST /storage/gc`
- Archive old memories: `POST /storage/archive` with `days_old=30`
- Increase checkpoint frequency: `T4DM_CHECKPOINT_INTERVAL_SECONDS=300`

---

## Advanced Debugging

### Trace Memory Path Through System

```python
from t4dm.sdk.client import T4DMClient
from uuid import UUID

client = T4DMClient()
session_id = "my_session"

# Create memory with debug flag
result = client.episodes.store(
    content="Test memory",
    session_id=session_id,
    metadata={"debug_trace": True},
)

mem_id = result.id

# Follow memory through consolidation
from t4dm.consolidation.service import get_consolidation_service
consol = get_consolidation_service(session_id=session_id)

# Check at each consolidation phase
print("Initial state:")
status = consol.get_memory_consolidation_status(mem_id)
print(f"  κ={status['kappa']}, phase={status['current_phase']}")

# Wait for NREM compaction
print("\nAfter NREM consolidation:")
consol.consolidate_nrem(session_id=session_id)
status = consol.get_memory_consolidation_status(mem_id)
print(f"  κ={status['kappa']}, phase={status['current_phase']}")

# Wait for REM compaction
print("\nAfter REM consolidation:")
consol.consolidate_rem(session_id=session_id)
status = consol.get_memory_consolidation_status(mem_id)
print(f"  κ={status['kappa']}, phase={status['current_phase']}")
```

### Monitor Real-Time Events

```bash
# Connect to WebSocket event stream
wscat -c "ws://localhost:8000/ws/memory?session_id=$SESSION_ID&api_key=$T4DM_API_KEY"

# In another terminal, trigger a store:
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/episodes/store" \
  -H "Content-Type: application/json" \
  -d "{\"content\": \"Test\", \"session_id\": \"$SESSION_ID\"}"

# WebSocket will stream:
# {"event": "memory_stored", "id": "...", "kappa": 0.0, ...}
# {"event": "embedding_computed", "id": "...", ...}
# {"event": "index_updated", "id": "...", ...}
```

---

## Checklist for Debugging Session

- [ ] Verify API is running: `curl /health`
- [ ] Check logs for errors: `docker logs t4dm-service | grep ERROR`
- [ ] Verify session exists: `curl /stats?session_id=$SESSION_ID`
- [ ] Check memory count: `curl /episodes/list?session_id=$SESSION_ID`
- [ ] Verify kappa distribution: `curl /consolidation/kappa-distribution`
- [ ] Test search on single memory: `curl /episodes/search` with specific content
- [ ] Check consolidation running: `curl /consolidation/status`
- [ ] Monitor system metrics: `curl /observability/metrics`
- [ ] Review circuit breaker state: `curl /system/status`
- [ ] Examine error logs for XSS/validation failures
