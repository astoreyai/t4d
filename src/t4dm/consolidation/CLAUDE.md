# Consolidation Module
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/`

## What
Biologically-inspired memory consolidation implementing sleep phases (NREM/REM/PRUNE), HDBSCAN clustering, STDP weight updates, lability windows, parallel execution, and automatic scheduling. Mirrors hippocampal-neocortical memory transfer.

## How
- **SleepConsolidation**: Full sleep cycle with 4 NREM-REM cycles + final prune
  - NREM: Sharp-Wave Ripple replay (10x compression, PE-weighted priority, interleaved CLS 60/40)
  - REM: HDBSCAN clustering of similar entities, abstract concept creation
  - PRUNE: Delete weak synapses below threshold, homeostatic scaling
- **ConsolidationService**: Higher-level orchestrator with light/deep/skill/all modes
- **ConsolidationScheduler** (P3.3): Time-based (8h) and load-based (100 memories) auto-triggering
- **FESConsolidator**: Fast Episodic Store to permanent storage transfer
- **ConsolidationSTDP** (P7.3): STDP weight updates during replay sequences with synaptic tag sync
- **LabilityManager**: Protein synthesis gate controlling reconsolidation eligibility
- **ParallelExecutor** (PO-1): 10x speedup via ProcessPool for clustering and embedding

## Why
Prevents unbounded memory growth, extracts semantic knowledge from episodic experience, removes weak connections, and strengthens important memories. Biological plausibility through sleep-phase modeling and STDP learning.

## Key Files
| File | Purpose |
|------|---------|
| `sleep.py` | NREM/REM/PRUNE phases, SharpWaveRipple (~1,385 lines) |
| `service.py` | ConsolidationService, ConsolidationScheduler (~1,491 lines) |
| `fes_consolidator.py` | Fast->LTM transfer (~414 lines) |
| `stdp_integration.py` | STDP during replay, synaptic tag sync (~463 lines) |
| `lability.py` | Reconsolidation eligibility, lability windows |
| `parallel.py` | ParallelExecutor, HDBSCAN clustering (~378 lines) |

## Data Flow
```
Scheduler triggers -> ConsolidationService.consolidate(mode)
    -> SleepConsolidation.full_sleep_cycle()
        -> NREM: priority replay (PE-weighted) + SWR compression + STDP
        -> REM: HDBSCAN clustering -> concept entity creation
        -> PRUNE: weak synapse deletion + homeostatic scaling
    -> FESConsolidator: fast store -> permanent storage
```

## Integration Points
- **core**: Episode, Entity, Procedure types
- **memory**: Episodic, semantic, procedural stores for read/write
- **learning**: STDP learner, synaptic tagger for weight updates
- **bridges**: NCA bridge container for subsystem integration
- **storage**: Qdrant (vector), Neo4j (graph) backends
