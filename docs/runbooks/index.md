# T4DM Debugging Runbooks

**Location**: `/mnt/projects/t4d/t4dm/docs/runbooks/`

Quick-reference guides for debugging T4DM components in production.

---

## Runbooks

### 1. [DEBUGGING_MEMORY.md](DEBUGGING_MEMORY.md)
**For**: Memory storage, retrieval, and consolidation issues

**When to use**:
- Memories not being stored or found
- Search returns empty despite stored data
- Consolidation not progressing (κ stuck at 0.15)
- Memory leaks or unbounded growth

**Key diagnostics**:
- `GET /stats` — Memory counts by type
- `GET /consolidation/kappa-distribution` — κ distribution
- `GET /episodes/search` with provenance — Trace search results
- Code-level: Inspect MemoryItem, trace consolidation status

---

### 2. [DEBUGGING_STORAGE.md](DEBUGGING_STORAGE.md)
**For**: T4DX embedded storage engine, WAL, and segments

**When to use**:
- Segment corruption or checksum failures
- WAL growing unbounded
- MemTable never flushing
- HNSW index out of sync
- Recovery fails after crash

**Key diagnostics**:
- `GET /storage/status` — Overall storage health
- `GET /storage/segments` — List all segments
- `GET /storage/wal-status` — WAL health check
- `GET /storage/hnsw-stats` — Vector index status
- Code-level: Inspect segments, verify checksums, rebuild HNSW

---

### 3. [DEBUGGING_SPIKING.md](DEBUGGING_SPIKING.md)
**For**: Spiking neural network blocks and learning dynamics

**When to use**:
- Spikes not firing (0% spike rate)
- Runaway spiking (> 50% rate, system unresponsive)
- STDP learning not updating weights
- Neuromodulator levels frozen
- Membrane potentials diverging (NaN/Inf)

**Key diagnostics**:
- `GET /spiking/stats` — Spike rates, statistics
- `POST /spiking/trace` — Trace block output
- `POST /spiking/trace-membrane` — Membrane potential over time
- `GET /spiking/neuromodulators` — NT levels
- Code-level: Test LIF neurons, check causal timing, verify learning gates

---

### 4. [DEBUGGING_PERFORMANCE.md](DEBUGGING_PERFORMANCE.md)
**For**: Latency, throughput, and resource optimization

**When to use**:
- Store operations > 5 seconds
- Search queries > 2 seconds
- High memory usage (> 16 GB)
- CPU saturation during consolidation
- Poor inference throughput (< 10 tokens/sec)
- Volatile latency (tail p99 >> median)

**Key diagnostics**:
- `GET /observability/latency` — Latency percentiles
- `POST /observability/query-profile` — Query execution plan
- `GET /qwen/layer-timing` — Inference bottlenecks
- Code-level: Profile operations, monitor cache behavior, tune parameters

---

## Diagnostic Workflow

### 1. Identify the Problem
- Is it memory-related? → [DEBUGGING_MEMORY.md](DEBUGGING_MEMORY.md)
- Is it storage/engine-related? → [DEBUGGING_STORAGE.md](DEBUGGING_STORAGE.md)
- Is it spiking/learning-related? → [DEBUGGING_SPIKING.md](DEBUGGING_SPIKING.md)
- Is it performance-related? → [DEBUGGING_PERFORMANCE.md](DEBUGGING_PERFORMANCE.md)

### 2. Run Quick Checks
```bash
# Health check
curl http://localhost:8000/health

# Get overall status
curl http://localhost:8000/stats?session_id=$SESSION_ID

# Check storage
curl http://localhost:8000/storage/status?session_id=$SESSION_ID

# Check spiking
curl http://localhost:8000/spiking/stats?session_id=$SESSION_ID

# Check latency
curl http://localhost:8000/observability/latency?window_seconds=60
```

### 3. Consult Appropriate Runbook
Each runbook contains:
- **Symptoms**: What to look for
- **Diagnostic commands**: API endpoints to check
- **Code-based debugging**: Programmatic inspection
- **Common issues & solutions**: Troubleshooting steps

### 4. Apply Solutions
Each issue has multiple solution paths, ordered by:
1. **Quick fixes** (configuration changes, API calls)
2. **Code changes** (enable features, adjust parameters)
3. **Last resort** (rebuild indices, recover from backup)

---

## Common Commands Reference

### API Health
```bash
curl http://localhost:8000/health
```

### Memory Status
```bash
curl http://localhost:8000/stats?session_id=$SESSION_ID
curl http://localhost:8000/episodes/list?session_id=$SESSION_ID&limit=100
curl http://localhost:8000/consolidation/status?session_id=$SESSION_ID
```

### Storage Status
```bash
curl http://localhost:8000/storage/status?session_id=$SESSION_ID
curl http://localhost:8000/storage/hnsw-stats?session_id=$SESSION_ID
curl http://localhost:8000/storage/wal-status?session_id=$SESSION_ID
```

### Spiking Status
```bash
curl http://localhost:8000/spiking/stats?session_id=$SESSION_ID
curl http://localhost:8000/spiking/neuromodulators?session_id=$SESSION_ID
curl http://localhost:8000/learning/stdp-metrics?session_id=$SESSION_ID
```

### Performance Metrics
```bash
curl http://localhost:8000/observability/latency?window_seconds=60
curl http://localhost:8000/observability/ops-count?session_id=$SESSION_ID
curl http://localhost:8000/observability/cache-stats?session_id=$SESSION_ID
```

---

## Emergency Procedures

### System Unresponsive
1. Check CPU: `top -p $(pgrep -f t4dm-service)`
2. Check memory: `ps aux | grep t4dm-service`
3. Check disk: `df -h /path/to/t4dm/data`
4. Check logs: `docker logs t4dm-service | tail -100`
5. If consolidation blocking: Restart service

### Data Loss Risk
1. Backup data: `cp -r /path/to/t4dm/data /backups/`
2. Check WAL: `tail -10 /path/to/t4dm/data/wal.jsonl | jq .`
3. Verify checksums: `GET /storage/segment/{id}/verify`
4. If corrupted: Restore from backup or replay WAL

### Cascade Failure
1. Check circuit breaker: `GET /system/status`
2. Check error logs: `docker logs t4dm-service | grep ERROR`
3. Verify dependencies: Database, embedding service, external APIs
4. Reset circuit breaker: `POST /system/reset-breaker`

---

## Integration

These runbooks are designed to work together:

```
Problem occurs
    ↓
Quick health check (this page)
    ↓
Identify component → Select runbook
    ↓
Run diagnostics (API or code-level)
    ↓
Apply solutions (quick fix → code change → rebuild)
    ↓
Verify fix (rerun diagnostics)
    ↓
Document root cause + update runbooks
```

---

## Environment Variables

Key T4DM configuration variables for debugging:

```bash
# Logging
export T4DM_LOG_LEVEL=DEBUG  # More verbose

# Storage
export T4DM_FLUSH_THRESHOLD=500  # Flush more frequently
export T4DM_HNSW_M=24  # Better HNSW connectivity

# Consolidation
export T4DM_CONSOLIDATION_BLOCKING=false  # Non-blocking
export T4DM_CONSOLIDATION_INTERVAL_SECONDS=30  # More frequent

# Performance
export T4DM_EMBEDDING_CACHE_SIZE=1000  # Tune cache size
export T4DM_CHECKPOINT_INTERVAL_SECONDS=300  # More frequent checkpoints

# Spiking
export T4DM_STDP_ENABLED=true  # Enable learning
export T4DM_NEUROMOD_ENABLED=true  # Enable modulation

# Recovery
export T4DM_RECOVERY_STRICT=true  # Safe but slower recovery
```

---

## Links to Related Documentation

- **Architecture**: [../ARCHITECTURE.md](../ARCHITECTURE.md)
- **Full setup**: [../FULL_SETUP_GUIDE.md](../FULL_SETUP_GUIDE.md)
- **API reference**: [../API.md](../API.md)
- **Integration guide**: [../INTEGRATION_PLAN.md](../INTEGRATION_PLAN.md)
- **Testing**: [../guides/testing.md](../guides/testing.md)
- **Performance tuning**: [../guides/performance.md](../guides/performance.md)

---

## Version & Maintenance

- **Last updated**: 2026-02-06
- **T4DM version**: 2.0.0
- **Maintained by**: T4DM engineering team
- **Feedback**: Update these runbooks when encountering new issues

---

## Quick Decision Tree

**Memories not showing up?**
→ Check [DEBUGGING_MEMORY.md § Search Returns Empty Results](DEBUGGING_MEMORY.md#issue-search-returns-empty-despite-stored-memories)

**Slow store operations?**
→ Check [DEBUGGING_PERFORMANCE.md § Store Operations Slow](DEBUGGING_PERFORMANCE.md#issue-store-operations-take-5-seconds)

**High memory usage?**
→ Check [DEBUGGING_PERFORMANCE.md § High Memory Usage](DEBUGGING_PERFORMANCE.md#issue-high-memory-usage--16-gb)

**Crash during recovery?**
→ Check [DEBUGGING_STORAGE.md § Storage Recovery Fails](DEBUGGING_STORAGE.md#issue-storage-recovery-fails-after-crash)

**No spikes firing?**
→ Check [DEBUGGING_SPIKING.md § No Spikes](DEBUGGING_SPIKING.md#issue-no-spikes-spike-rate--0)

**Consolidation stuck?**
→ Check [DEBUGGING_MEMORY.md § Consolidation Stalled](DEBUGGING_MEMORY.md#issue-memory-consolidation-stalled-at-κ015)

**WAL growing unbounded?**
→ Check [DEBUGGING_STORAGE.md § WAL Growing Unbounded](DEBUGGING_STORAGE.md#issue-wal-growing-unbounded)

**Inference slow?**
→ Check [DEBUGGING_PERFORMANCE.md § Poor Inference Throughput](DEBUGGING_PERFORMANCE.md#issue-poor-inference-throughput--10-tokenssec)
