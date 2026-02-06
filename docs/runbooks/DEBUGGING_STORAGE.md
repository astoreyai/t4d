# T4DM Storage (T4DX) Debugging Runbook

**Location**: `/mnt/projects/t4d/t4dm/docs/runbooks/DEBUGGING_STORAGE.md`

Troubleshoot T4DX embedded storage engine issues: segments, WAL, index corruption, and recovery.

---

## Symptoms

### Segment Corruption or Checksum Failure
- Logs show "segment checksum mismatch" or "invalid segment file"
- GET/SEARCH queries return stale or corrupted data
- Query planner crashes with "segment metadata invalid"

**Likelihood**: Disk I/O error, incomplete flush, or concurrent write race

### WAL (Write-Ahead Log) Growing Unbounded
- `wal.jsonl` grows to GBs despite active consolidation
- WAL replay takes minutes on startup
- "WAL size exceeded" warnings in logs

**Likelihood**: WAL not being truncated after compaction, failed archival

### MemTable Never Flushes
- MemTable size stays high despite threshold exceeded
- New writes block waiting for flush
- Compactor thread stuck or crashed

**Likelihood**: Compactor deadlock, disk full, or I/O timeout

### HNSW Index Out of Sync
- Search results miss recent vectors
- Vector count in index != stored vectors
- "HNSW refresh failed" in logs

**Likelihood**: Index not updated during flush, segment reader not loading vectors

### Storage Recovery Fails After Crash
- Startup hangs during WAL replay
- Data loss or duplicate records after recovery
- Segment files missing or incomplete

**Likelihood**: WAL corruption, lost segment metadata, or incomplete flush

### Global Index Corrupted
- Query planner can't find items in segments
- "Global index mismatch" errors
- Segment iteration loops infinitely

**Likelihood**: JSON parse error in global_index.json, concurrent writes

---

## Diagnostic Commands

### Storage Health & Status

```bash
# Check T4DX engine status
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/status?session_id=$SESSION_ID"

# Get detailed storage stats
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/stats?session_id=$SESSION_ID"

# List all segments
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/segments?session_id=$SESSION_ID"

# Get MemTable stats
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/memtable?session_id=$SESSION_ID"

# Check WAL health
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/wal-status?session_id=$SESSION_ID"

# Get HNSW index statistics
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/hnsw-stats?session_id=$SESSION_ID"

# Check global index
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/global-index?session_id=$SESSION_ID"
```

### Segment Inspection

```bash
# Inspect a specific segment
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/segment/{segment_id}?session_id=$SESSION_ID"

# Get segment metadata
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/segment/{segment_id}/metadata?session_id=$SESSION_ID"

# Verify segment checksum
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/segment/{segment_id}/verify?session_id=$SESSION_ID"

# Count items in segment
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/segment/{segment_id}/count?session_id=$SESSION_ID"

# List segment compaction history
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/compaction-history?session_id=$SESSION_ID"
```

### WAL Diagnostics

```bash
# Check WAL file size
du -h /path/to/t4dm/data/wal.jsonl

# Count WAL entries
wc -l /path/to/t4dm/data/wal.jsonl

# Verify WAL format
head -5 /path/to/t4dm/data/wal.jsonl | jq .

# Check for corruption (invalid JSON)
tail -100 /path/to/t4dm/data/wal.jsonl | jq . >/dev/null 2>&1 && echo "Valid" || echo "CORRUPTED"

# Estimate WAL replay time
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/storage/estimate-replay-time" \
  -d "{\"session_id\": \"$SESSION_ID\"}"

# Compact WAL (remove old entries after compaction)
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/storage/compact-wal" \
  -d "{\"session_id\": \"$SESSION_ID\", \"keep_entries\": 1000}"

# Truncate WAL (dangerous - only after safe backup)
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/storage/truncate-wal" \
  -d "{\"session_id\": \"$SESSION_ID\", \"confirmation\": \"I_UNDERSTAND\"}"
```

### Disk & I/O Health

```bash
# Check disk space
df -h /path/to/t4dm/data

# Monitor I/O during operations
iotop -p $(pgrep -f t4dm-service)

# Check file handles
lsof -p $(pgrep -f t4dm-service) | grep t4dm/data

# Monitor write latency
iostat -x 1 | grep sda

# Check for disk errors
dmesg | grep -i "i/o error" | tail -10
```

---

## Code-Based Debugging

### Inspect Segments Programmatically

```python
from t4dm.storage.t4dx.engine import T4DXEngine
from pathlib import Path

data_dir = "/path/to/t4dm/data"
engine = T4DXEngine(data_dir=data_dir)
engine.startup()

# List all segments
print("Active segments:")
for seg_id, segment_reader in engine._segments.items():
    print(f"\nSegment {seg_id}:")
    print(f"  Items: {len(segment_reader)}")
    print(f"  File size: {segment_reader.data_file.stat().st_size / 1024:.1f} KB")
    print(f"  Vector dim: {segment_reader.vector_dim}")

    # Verify checksum
    try:
        segment_reader.verify_checksum()
        print(f"  Checksum: OK")
    except ValueError as e:
        print(f"  Checksum: FAILED - {e}")

engine.shutdown()
```

### Inspect WAL

```python
from t4dm.storage.t4dx.wal import WAL
from pathlib import Path

wal = WAL(Path("/path/to/t4dm/data/wal.jsonl"))
wal.open()

# Count entries by operation type
entries = wal.replay()
op_counts = {}
for entry in entries:
    op_type = entry['op_type']
    op_counts[op_type] = op_counts.get(op_type, 0) + 1

print("WAL Operation Counts:")
for op_type, count in sorted(op_counts.items()):
    print(f"  {op_type}: {count}")

# Show last 10 entries
print("\nLast 10 WAL entries:")
for entry in entries[-10:]:
    print(f"  {entry['timestamp']}: {entry['op_type']}")

wal.close()
```

### Verify Data Integrity

```python
from t4dm.storage.t4dx.engine import T4DXEngine
import hashlib

data_dir = "/path/to/t4dm/data"
engine = T4DXEngine(data_dir=data_dir)
engine.startup()

# Scan all items and verify embeddings
errors = []
for segment_id, segment_reader in engine._segments.items():
    for item_bytes in segment_reader.items_by_offset.values():
        try:
            # Verify embedding is valid
            item = segment_reader.deserialize_item(item_bytes)
            if item.embedding:
                assert len(item.embedding) > 0
                assert all(-1.0 <= v <= 1.0 for v in item.embedding), "Embedding out of range"
            print(f"✓ {item.id}")
        except Exception as e:
            errors.append((segment_id, str(e)))
            print(f"✗ Segment {segment_id}: {e}")

if errors:
    print(f"\n{len(errors)} integrity errors found!")
else:
    print(f"\nAll items verified OK")

engine.shutdown()
```

### Check Global Index

```python
from t4dm.storage.t4dx.global_index import GlobalIndex
from pathlib import Path

gi = GlobalIndex()
gi.load(Path("/path/to/t4dm/data/global_index.json"))

print("Global Index Summary:")
print(f"  Total segments: {len(gi.segments)}")
print(f"  Total ID mappings: {len(gi.id_to_segment)}")

# Verify consistency
errors = []
for item_id, seg_id in gi.id_to_segment.items():
    if seg_id not in gi.segments:
        errors.append(f"ID {item_id} references missing segment {seg_id}")

if errors:
    print(f"\n{len(errors)} inconsistencies found:")
    for error in errors[:10]:
        print(f"  - {error}")
else:
    print("  Global index is consistent")

# Show segment distribution
print("\nItems per segment:")
seg_counts = {}
for seg_id in gi.id_to_segment.values():
    seg_counts[seg_id] = seg_counts.get(seg_id, 0) + 1

for seg_id in sorted(seg_counts.keys()):
    count = seg_counts[seg_id]
    bar = "=" * (count // 100)
    print(f"  Seg {seg_id}: {count:6d} {bar}")
```

### Monitor Compaction

```python
from t4dm.storage.t4dx.compactor import Compactor
from t4dm.storage.t4dx.engine import T4DXEngine
import time

data_dir = "/path/to/t4dm/data"
engine = T4DXEngine(data_dir=data_dir)
engine.startup()

# Watch a manual flush
print("Before flush:")
print(f"  MemTable items: {len(engine._memtable)}")
print(f"  Segments: {len(engine._segments)}")

start = time.time()
engine._compactor.flush()
elapsed = time.time() - start

print(f"After flush ({elapsed:.1f}s):")
print(f"  MemTable items: {len(engine._memtable)}")
print(f"  Segments: {len(engine._segments)}")
print(f"  Flush throughput: {len(engine._memtable) / elapsed:.0f} items/sec")

engine.shutdown()
```

### Rebuild HNSW Index

```python
from t4dm.storage.t4dx.engine import T4DXEngine

data_dir = "/path/to/t4dm/data"
engine = T4DXEngine(data_dir=data_dir)
engine.startup()

print("Rebuilding HNSW index...")

# Collect all vectors
all_vectors = []
all_ids = []
for record in engine.scan():
    if record.embedding and len(record.embedding) > 0:
        all_vectors.append(record.embedding)
        all_ids.append(record.id)

# Rebuild index
from t4dm.storage.t4dx.hnsw import HNSWIndex
hnsw = HNSWIndex(dim=len(all_vectors[0]) if all_vectors else 1024)
for idx, (vec, item_id) in enumerate(zip(all_vectors, all_ids)):
    hnsw.add(item_id, vec)

print(f"Rebuilt HNSW with {len(all_vectors)} vectors")
engine.shutdown()
```

---

## Common Issues & Solutions

### Issue: WAL Growing Unbounded

**Symptoms**:
- `wal.jsonl` > 500 MB
- Startup takes > 1 minute
- "WAL replay in progress..." logs hang

**Diagnosis**:
```bash
# Check WAL size
ls -lh /path/to/t4dm/data/wal.jsonl

# Count entries
wc -l /path/to/t4dm/data/wal.jsonl

# Check compaction history
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/compaction-history?session_id=$SESSION_ID"
```

**Solutions**:
1. **Compact WAL** (safe, removes old entries):
   ```bash
   curl -H "X-API-Key: $T4DM_API_KEY" \
     -X POST "http://localhost:8000/storage/compact-wal" \
     -d "{\"session_id\": \"$SESSION_ID\", \"keep_entries\": 10000}"
   ```

2. **Manual flush to segments** (consolidates MemTable):
   ```bash
   curl -H "X-API-Key: $T4DM_API_KEY" \
     -X POST "http://localhost:8000/consolidation/flush" \
     -d "{\"session_id\": \"$SESSION_ID\"}"
   ```

3. **Truncate WAL** (dangerous - only as last resort):
   ```python
   from t4dm.storage.t4dx.wal import WAL
   wal = WAL(Path("/path/to/wal.jsonl"))
   wal.truncate()  # Erases WAL - BACKUP FIRST!
   ```

### Issue: Segment Checksum Failure

**Symptoms**:
- "Segment checksum mismatch" error
- Queries on that segment fail
- `GET /segment/{id}/verify` returns false

**Diagnosis**:
```bash
# Verify specific segment
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/segment/000001/verify"

# Check segment file integrity
ls -la /path/to/t4dm/data/seg_000001/

# Check filesystem errors
fsck -n /dev/sda1  # (read-only check, don't repair yet)
```

**Solutions**:
1. **Try to recover from WAL**:
   ```python
   from t4dm.storage.t4dx.engine import T4DXEngine

   engine = T4DXEngine(data_dir="/path/to/t4dm/data")
   # Startup will replay WAL and rebuild segment
   engine.startup()
   engine.shutdown()
   ```

2. **Rebuild segment from scratch** (will re-digest WAL):
   ```bash
   # Backup corrupted segment
   mv /path/to/t4dm/data/seg_000001 /path/to/t4dm/data/seg_000001.bak

   # Restart service - engine will rebuild on startup
   systemctl restart t4dm-service
   ```

3. **Check disk health**:
   ```bash
   # Run filesystem check
   sudo fsck -n /dev/sda1

   # Check S.M.A.R.T. status
   sudo smartctl -a /dev/sda
   ```

### Issue: MemTable Never Flushes

**Symptoms**:
- MemTable size grows over threshold but doesn't flush
- Logs show no flush messages
- New writes hang or timeout

**Diagnosis**:
```python
from t4dm.storage.t4dx.engine import T4DXEngine

engine = T4DXEngine(data_dir="/path/to/t4dm/data")
engine.startup()

# Check MemTable state
print(f"MemTable items: {len(engine._memtable)}")
print(f"MemTable size: {engine._memtable.estimated_size_bytes()} bytes")
print(f"Flush threshold: {engine._flush_threshold}")
print(f"Compactor running: {engine._compactor._thread.is_alive() if hasattr(engine._compactor, '_thread') else 'N/A'}")

# Try manual flush
engine._compactor.flush()
print(f"After manual flush - MemTable items: {len(engine._memtable)}")

engine.shutdown()
```

**Solutions**:
1. **Trigger manual flush**:
   ```bash
   curl -H "X-API-Key: $T4DM_API_KEY" \
     -X POST "http://localhost:8000/storage/flush" \
     -d "{\"session_id\": \"$SESSION_ID\"}"
   ```

2. **Increase flush threshold** (if legitimate workload):
   ```bash
   export T4DM_FLUSH_THRESHOLD=5000  # was 1000
   systemctl restart t4dm-service
   ```

3. **Check for disk space**:
   ```bash
   df -h /path/to/t4dm/data
   # If < 10% free, clean up old segments
   ```

4. **Restart compactor thread**:
   ```python
   # In code
   engine._compactor.stop()
   engine._compactor.start()
   ```

### Issue: Search Missing Recent Data

**Symptoms**:
- Store succeeds but search doesn't find it for 5+ minutes
- Other searches work
- Intermittent - sometimes works, sometimes doesn't

**Diagnosis**:
```bash
# Check if item is in MemTable
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/episodes/{item_id}"  # Should be found in MemTable

# Check HNSW index freshness
curl -H "X-API-Key: $T4DM_API_KEY" \
  "http://localhost:8000/storage/hnsw-stats"

# Check query planner logic
curl -H "X-API-Key: $T4DM_API_KEY" \
  -X POST "http://localhost:8000/storage/query-plan" \
  -d "{\"query_vector\": [...], \"k\": 10}"
```

**Solutions**:
1. **Reduce flush threshold** (flush more frequently):
   ```bash
   export T4DM_FLUSH_THRESHOLD=500  # was 1000
   systemctl restart t4dm-service
   ```

2. **Rebuild HNSW index** (refresh vector search):
   ```bash
   curl -H "X-API-Key: $T4DM_API_KEY" \
     -X POST "http://localhost:8000/storage/rebuild-hnsw" \
     -d "{\"session_id\": \"$SESSION_ID\"}"
   ```

3. **Verify query planner is searching MemTable**:
   ```python
   # Check QueryPlanner logic includes MemTable
   from t4dm.storage.t4dx.query_planner import QueryPlanner
   planner = QueryPlanner(memtable, segments, hnsw)

   # Planner should check MemTable first
   assert planner.memtable is not None
   ```

### Issue: Storage Recovery Fails After Crash

**Symptoms**:
- Service won't start after unexpected shutdown
- "WAL replay failed" in logs
- Data loss or duplicate records

**Diagnosis**:
```bash
# Check startup logs
docker logs t4dm-service | tail -100 | grep -i "wal\|replay\|error"

# Verify WAL file exists and is readable
ls -l /path/to/t4dm/data/wal.jsonl
file /path/to/t4dm/data/wal.jsonl

# Check for partial/corrupt entries
tail -10 /path/to/t4dm/data/wal.jsonl | jq .
```

**Solutions**:
1. **Attempt recovery with strict mode**:
   ```bash
   # Enable strict recovery (may be slower but safer)
   export T4DM_RECOVERY_STRICT=true
   systemctl start t4dm-service
   ```

2. **Manually remove last WAL entry** (if it's partial):
   ```bash
   # Backup first!
   cp /path/to/t4dm/data/wal.jsonl /path/to/t4dm/data/wal.jsonl.bak

   # Remove last line if incomplete
   head -n -1 /path/to/t4dm/data/wal.jsonl > /tmp/wal.tmp
   mv /tmp/wal.tmp /path/to/t4dm/data/wal.jsonl

   systemctl start t4dm-service
   ```

3. **Truncate WAL as last resort** (will lose uncompacted writes):
   ```python
   from t4dm.storage.t4dx.wal import WAL
   from pathlib import Path

   wal = WAL(Path("/path/to/t4dm/data/wal.jsonl"))
   wal.truncate()  # WILL LOSE DATA - backup first!
   ```

4. **Restore from backup**:
   ```bash
   # If you have a backup of t4dm/data/
   rm -rf /path/to/t4dm/data
   cp -r /backups/t4dm/data /path/to/t4dm/
   systemctl start t4dm-service
   ```

---

## Checklist for Storage Debugging

- [ ] Verify disk space: `df -h /path/to/t4dm/data`
- [ ] Check disk errors: `dmesg | grep -i "i/o error"`
- [ ] Verify WAL format: `head -5 wal.jsonl | jq .`
- [ ] Check segment count: `ls -d seg_* | wc -l`
- [ ] Verify global index: `GET /storage/global-index`
- [ ] Check HNSW stats: `GET /storage/hnsw-stats`
- [ ] Verify MemTable is flushing: Monitor `GET /storage/memtable` repeatedly
- [ ] Verify checksums: `GET /storage/segment/{id}/verify`
- [ ] Monitor I/O latency: `iostat -x 1`
- [ ] Check file handle limits: `lsof -p $PID | wc -l`
- [ ] Trigger manual flush: `POST /storage/flush`
- [ ] Verify query planner: `POST /storage/query-plan`
