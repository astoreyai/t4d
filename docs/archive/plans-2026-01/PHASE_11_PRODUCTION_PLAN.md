# World Weaver: Phase 11+ Production Completion Plan

**Created**: 2026-01-05 | **Status**: DETAILED PLANNING COMPLETE
**Based on**: Expert Agent Analysis (Hinton, CompBio, Architecture, Cleanup)
**Target**: Production-ready, pip-installable, learning memory system

---

## Executive Summary

This plan synthesizes findings from 4 expert agent analyses to complete World Weaver for production deployment.

### Expert Assessment Scores

| Agent | Score | Critical Finding |
|-------|-------|------------------|
| **Hinton** | 6/10 Learning | Learning loop not closed - signals computed but not applied |
| **CompBio** | 87/100 Biology | 3 blocking gaps (protein synthesis, ripple, replay direction) |
| **Architecture** | 90% Ready | SDK/MCP/K8s complete, minor gaps (Dockerfile, version sync) |
| **Cleanup** | Good Structure | 1 deprecated module, 4 duplicate classes, 50 docs to archive |

### Current State
- **Tests**: 6,540+ passing, 80% coverage
- **Code**: 242 files, 119k LOC
- **Deployment**: Helm/K8s/Docker ready
- **Integration**: SDK + REST API + MCP server complete

### Critical Path to Production

1. **Sprint 11A** (1 week): Close Learning Loop (Hinton 6/10 → 9/10)
2. **Sprint 11B** (1 week): Fix Biological Gaps (CompBio 87 → 93/100)
3. **Sprint 11C** (3 days): Production Polish (Architecture 90% → 98%)
4. **Sprint 11D** (2 days): Codebase Cleanup (Organization)

**Total**: ~3 weeks to production-ready

---

## Sprint 11A: Close the Learning Loop (Hinton Recommendations)

**Duration**: 5 working days
**Goal**: Transform from "stores representations" to "learns representations"
**Target**: Learning Effectiveness 6/10 → 9/10

### Background (Hinton Agent Analysis)

The system has excellent learning machinery, but the loops are not closed:
- Three-factor signal is computed but applied to wrong target
- Reconsolidation updates embeddings but never persists them
- Eligibility traces exist but aren't populated during retrieval
- Batch processing uses only first memory ID

### Task 11A.1: Fix FFEncoder Learning Target (Critical)

**File**: `src/ww/memory/episodic.py:2946-2961`

**Current (WRONG)**:
```python
# Applies learning to the QUERY embedding, not retrieved memories
ff_learning_stats = self._ff_encoder.learn_from_outcome(
    embedding=query_emb_np,  # ← WRONG: This is the query
    outcome_score=outcome_score,
    three_factor_signal=three_factor_signal,
)
```

**Target (CORRECT)**:
```python
# Apply learning to each RETRIEVED memory embedding
for eid in episode_ids:
    memory_emb = await self._get_episode_embedding(eid)
    signal = self.three_factor.compute(
        memory_id=eid,
        base_lr=0.03,
        outcome=per_memory_rewards.get(str(eid), outcome_score),
    )
    self._ff_encoder.learn_from_outcome(
        embedding=memory_emb,  # ← CORRECT: Memory embedding
        outcome_score=per_memory_rewards.get(str(eid), outcome_score),
        three_factor_signal=signal,
    )
```

**Effort**: 4 hours
**Tests**: Add to `tests/memory/test_episodic_learning.py`

---

### Task 11A.2: Persist Reconsolidated Embeddings (Critical)

**File**: `src/ww/learning/credit_flow.py:186-193`

**Current (DISCARDED)**:
```python
updated = self.reconsolidation.batch_reconsolidate(
    memories=retrieved_memories,
    query_embedding=query_embedding,
    outcome_score=base_outcome,
    per_memory_rewards=per_memory_rewards,
    per_memory_importance=per_memory_importance,
    per_memory_lr_modulation=per_memory_lr_modulation
)
# `updated` contains new embeddings but is NEVER WRITTEN BACK
```

**Target (PERSIST)**:
```python
updated = self.reconsolidation.batch_reconsolidate(...)

# NEW: Persist updated embeddings to vector store
if updated and len(updated) > 0:
    await self._persist_reconsolidated_embeddings(updated)
```

**New Method** (add to `episodic.py`):
```python
async def _persist_reconsolidated_embeddings(
    self,
    updated: list[ReconsolidatedMemory]
) -> int:
    """Write reconsolidated embeddings back to Qdrant."""
    count = 0
    for mem in updated:
        await self.vector_store.update_embedding(
            id=str(mem.episode_id),
            embedding=mem.new_embedding,
            metadata={"reconsolidated_at": datetime.now().isoformat()}
        )
        count += 1
    return count
```

**Effort**: 6 hours
**Tests**: Add to `tests/learning/test_reconsolidation_persistence.py`

---

### Task 11A.3: Add Eligibility Marking in Recall (Critical)

**File**: `src/ww/memory/episodic.py:1231-1380` (recall method)

**Current**: Retrieval doesn't mark memories as eligible for credit

**Target**: Add eligibility marking after retrieval
```python
async def recall(self, query: str, limit: int = 10, ...) -> list[Episode]:
    # ... existing retrieval logic ...

    results = await self._execute_retrieval(query_embedding, limit)

    # NEW: Mark retrieved memories as active for eligibility
    for result in results:
        self.three_factor.mark_active(
            memory_id=str(result.episode.id),
            activation_strength=result.similarity_score,
        )

    return results
```

**Effort**: 2 hours
**Tests**: Add to `tests/learning/test_eligibility_integration.py`

---

### Task 11A.4: Batch Three-Factor Computation (Important)

**File**: `src/ww/memory/episodic.py:2946-2954`

**Current (SINGLE)**:
```python
three_factor_signal = self.three_factor.compute(
    memory_id=episode_ids[0],  # Only uses FIRST episode
    base_lr=0.03,
    outcome=outcome_score,
)
```

**Target (BATCH)**:
```python
three_factor_signals = self.three_factor.batch_compute(
    memory_ids=episode_ids,  # All retrieved memories
    base_lr=0.03,
    outcomes=per_memory_rewards,
)
```

**Effort**: 3 hours
**Tests**: Add to `tests/learning/test_three_factor_batch.py`

---

### Sprint 11A Deliverables

| Task | File | Effort | Impact |
|------|------|--------|--------|
| 11A.1 | episodic.py:2946-2961 | 4h | CRITICAL |
| 11A.2 | credit_flow.py + episodic.py | 6h | CRITICAL |
| 11A.3 | episodic.py:1231-1380 | 2h | CRITICAL |
| 11A.4 | episodic.py:2946-2954 | 3h | HIGH |
| Testing | tests/learning/*.py | 8h | REQUIRED |
| Documentation | docs/LEARNING_FLOW.md | 2h | REQUIRED |

**Total**: 25 hours (~5 days with reviews)

**Success Criteria**:
- [ ] FFEncoder weights change after positive outcomes
- [ ] Reconsolidated embeddings visible in Qdrant
- [ ] Eligibility traces populated during retrieval
- [ ] All memories in batch receive three-factor signals
- [ ] 15+ new tests passing

---

## Sprint 11B: Fix Biological Gaps (CompBio Recommendations)

**Duration**: 5 working days
**Goal**: Fix 3 blocking biological gaps for production
**Target**: Biological Score 87/100 → 93/100

### Background (CompBio Agent Analysis)

The system has excellent biological grounding (STDP, glutamate, glymphatic all correct).
Three gaps are blocking production deployment:

| Gap ID | Feature | Impact | Effort |
|--------|---------|--------|--------|
| B37 | Protein synthesis gate | Core biological mechanism | 3 days |
| B40 | Ripple oscillator 150-250Hz | Key consolidation marker | 2 days |
| B41 | Replay directionality | Functional differentiation | 3 days |

---

### Task 11B.1: Add Protein Synthesis Gate (B37)

**File**: `src/ww/learning/reconsolidation.py:124`

**Current**: Only lability window, no protein synthesis constraint

**Target**: Add PSI timing constraint per Nader et al. (2000)

```python
@dataclass
class ReconsolidationConfig:
    lability_window_hours: float = 6.0  # Existing
    protein_synthesis_window_hours: float = 4.0  # NEW
    protein_synthesis_required: bool = True  # NEW

class ReconsolidationEngine:
    def can_reconsolidate(
        self,
        memory_id: UUID,
        retrieval_time: datetime
    ) -> tuple[bool, str]:
        """Check if memory can be reconsolidated."""
        current_time = datetime.now()
        hours_since = (current_time - retrieval_time).total_seconds() / 3600

        # Must be within lability window
        if hours_since > self.config.lability_window_hours:
            return False, "Outside lability window"

        # Must be within protein synthesis window (NEW)
        if self.config.protein_synthesis_required:
            if hours_since > self.config.protein_synthesis_window_hours:
                return False, "Protein synthesis window closed"

        return True, "Eligible for reconsolidation"
```

**Biological Reference**: Nader et al. (2000) Nature 406:722-726

**Effort**: 1 day
**Tests**: `tests/learning/test_protein_synthesis_gate.py`

---

### Task 11B.2: Add Ripple Oscillator (B40)

**File**: `src/ww/nca/oscillators.py`

**Current**: Missing 150-250 Hz ripple frequency band

**Target**: Add ripple oscillator coupled to SWR events

```python
@dataclass
class RippleOscillatorConfig:
    """Sharp-wave ripple oscillator (150-250 Hz)."""
    center_freq_hz: float = 200.0
    min_freq_hz: float = 150.0
    max_freq_hz: float = 250.0
    burst_duration_ms: float = 80.0  # ~80ms ripple bursts
    inter_ripple_interval_ms: float = 200.0
    coupling_strength: float = 0.8  # Coupling to SWR events

class RippleOscillator:
    """150-250 Hz ripple oscillations during SWR."""

    def __init__(self, config: RippleOscillatorConfig = None):
        self.config = config or RippleOscillatorConfig()
        self._phase = 0.0
        self._in_ripple = False

    def generate_ripple(self, swr_event: bool, dt_ms: float) -> float:
        """Generate ripple oscillation during SWR events."""
        if not swr_event:
            self._in_ripple = False
            return 0.0

        # Start new ripple burst
        if not self._in_ripple:
            self._in_ripple = True
            self._phase = 0.0

        # Generate oscillation
        freq = self.config.center_freq_hz
        self._phase += 2 * np.pi * freq * (dt_ms / 1000)
        amplitude = np.sin(self._phase)

        # Envelope: Gaussian burst shape
        burst_center = self.config.burst_duration_ms / 2
        envelope = np.exp(-((dt_ms - burst_center)**2) / (2 * 20**2))

        return amplitude * envelope * self.config.coupling_strength
```

**Biological Reference**: Buzsaki (2015) Neuron 85:935-945

**Effort**: 0.5 day
**Tests**: `tests/nca/test_ripple_oscillator.py`

---

### Task 11B.3: Add Replay Directionality (B41)

**File**: `src/ww/consolidation/sleep.py`

**Current**: All replay treated identically

**Target**: Distinguish forward vs reverse replay (Foster & Wilson 2006)

```python
from enum import Enum
from typing import Literal

class ReplayDirection(Enum):
    FORWARD = "forward"    # ~40-50% of replays (learning new sequences)
    REVERSE = "reverse"    # ~20-30% of replays (credit assignment)
    RANDOM = "random"      # ~20-30% of replays (exploration)

@dataclass
class ReplayEvent:
    episode_ids: list[UUID]
    replay_time: datetime
    direction: ReplayDirection  # NEW
    priority_score: float
    compression_ratio: float

class ReplayDirectionScheduler:
    """Schedule replay direction based on memory type and state."""

    def __init__(
        self,
        forward_ratio: float = 0.45,  # ~45% forward
        reverse_ratio: float = 0.25,  # ~25% reverse
        random_ratio: float = 0.30,   # ~30% random
    ):
        self.forward_ratio = forward_ratio
        self.reverse_ratio = reverse_ratio
        self.random_ratio = random_ratio

    def select_direction(
        self,
        memory_age_hours: float,
        novelty_score: float,
    ) -> ReplayDirection:
        """Select replay direction based on memory properties."""
        # New memories: prefer forward (sequence learning)
        if memory_age_hours < 1 and novelty_score > 0.7:
            return ReplayDirection.FORWARD

        # Old memories needing credit: prefer reverse
        if memory_age_hours > 6:
            return ReplayDirection.REVERSE

        # Otherwise: weighted random
        r = random.random()
        if r < self.forward_ratio:
            return ReplayDirection.FORWARD
        elif r < self.forward_ratio + self.reverse_ratio:
            return ReplayDirection.REVERSE
        else:
            return ReplayDirection.RANDOM
```

**Biological Reference**: Foster & Wilson (2006) Nature 440:680-683

**Effort**: 1 day
**Tests**: `tests/consolidation/test_replay_directionality.py`

---

### Task 11B.4: Update Documentation (Doc Drift Fix)

**Files to Update**:
- `docs/science/biological-parameters.md` - Current values (tau_minus=34ms, glymphatic=0.7)
- `MASTER_BIOLOGICAL_VALIDATION_ISSUES.md` - Archive with fix notes

**Issues Found**:
| Report Claims | Actual Code | Status |
|---------------|-------------|--------|
| tau_minus = 20ms | 34ms | CODE CORRECT |
| glymphatic = 0.9 | 0.7 | CODE CORRECT |
| DG sparsity = 4% | 1% | CODE CORRECT |

**Effort**: 0.5 day

---

### Sprint 11B Deliverables

| Task | File | Effort | Impact |
|------|------|--------|--------|
| 11B.1 | reconsolidation.py | 1d | CRITICAL |
| 11B.2 | oscillators.py | 0.5d | HIGH |
| 11B.3 | sleep.py | 1d | HIGH |
| 11B.4 | docs/*.md | 0.5d | MEDIUM |
| Testing | tests/biology/*.py | 1.5d | REQUIRED |
| Integration | End-to-end consolidation | 0.5d | REQUIRED |

**Total**: 5 days

**Success Criteria**:
- [ ] Protein synthesis gate blocks late reconsolidation
- [ ] Ripple frequency in 150-250 Hz range during SWR
- [ ] Forward:reverse replay ratio ~45:25
- [ ] All biology tests passing
- [ ] Documentation matches code

---

## Sprint 11C: Production Polish (Architecture Recommendations)

**Duration**: 3 days
**Goal**: Fix minor production gaps
**Target**: Production Readiness 90% → 98%

### Task 11C.1: Create Main Dockerfile

**File**: `deploy/docker/Dockerfile` (NEW)

```dockerfile
# Multi-stage build for production API
FROM python:3.11-slim as builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir build && \
    python -m build --wheel

FROM python:3.11-slim as production

WORKDIR /app
COPY --from=builder /app/dist/*.whl ./

RUN pip install --no-cache-dir *.whl[api,observability] && \
    rm -f *.whl

# Non-root user
RUN useradd -m ww
USER ww

EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8765/api/v1/health || exit 1

CMD ["ww-api"]
```

**Effort**: 2 hours

---

### Task 11C.2: Sync Version Numbers

**Files**:
- `src/ww/__init__.py:__version__` → "0.5.0"
- `pyproject.toml:version` → "0.5.0"

```python
# src/ww/__init__.py
__version__ = "0.5.0"  # Was "0.2.0"
```

**Effort**: 15 minutes

---

### Task 11C.3: Add Worker Optional Dependency

**File**: `pyproject.toml`

```toml
[project.optional-dependencies]
worker = [
    "hdbscan>=0.8.33",
    "uvloop>=0.17.0",
]
```

**Effort**: 15 minutes

---

### Task 11C.4: Update CHANGELOG

**File**: `CHANGELOG.md`

Add Phase 11 changes:
- Learning loop closure
- Biological gap fixes
- Production infrastructure

**Effort**: 1 hour

---

### Sprint 11C Deliverables

| Task | File | Effort | Impact |
|------|------|--------|--------|
| 11C.1 | Dockerfile | 2h | HIGH |
| 11C.2 | Version sync | 15m | LOW |
| 11C.3 | Worker deps | 15m | MEDIUM |
| 11C.4 | CHANGELOG | 1h | MEDIUM |
| Docker build test | CI | 1h | REQUIRED |

**Total**: 0.5 day

---

## Sprint 11D: Codebase Cleanup (Cleanup Agent Recommendations)

**Duration**: 2 days
**Goal**: Clean organization, remove deprecated code

### Task 11D.1: Archive Deprecated Bridge Module

**Current**: `src/ww/bridge/` deprecated, superseded by `bridges/`

**Actions**:
1. Move `src/ww/bridge/` → `docs/archive/deprecated-bridge/`
2. Update `tests/bridge/test_memory_nca.py` to use `bridges/`
3. Remove bridge import from `src/ww/__init__.py`

**Effort**: 2 hours

---

### Task 11D.2: Consolidate Duplicate Context Classes

**Current**: `RetrievalContext` defined in 4 files

**Target**: Single canonical definition

**Create**: `src/ww/core/contexts.py`

```python
"""Canonical context dataclasses for cross-module communication."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

@dataclass
class RetrievalContext:
    """Context for memory retrieval operations."""
    query: str
    query_embedding: list[float] | None = None
    retrieved_ids: list[UUID] = field(default_factory=list)
    similarity_scores: dict[UUID, float] = field(default_factory=dict)
    retrieval_time: datetime = field(default_factory=datetime.now)
    task_id: str | None = None
    neuromodulator_state: dict[str, float] = field(default_factory=dict)
    eligibility_marked: bool = False

@dataclass
class EncodingContext:
    """Context for memory encoding operations."""
    content: str
    embedding: list[float] | None = None
    ff_encoded: bool = False
    capsule_encoded: bool = False
    storage_location: str | None = None
    encoding_time: datetime = field(default_factory=datetime.now)
    importance: float = 0.5
    emotional_valence: float = 0.0
```

**Update Imports**:
- `sdk/agent_client.py` → `from ww.core.contexts import RetrievalContext`
- `bridges/nca_bridge.py` → `from ww.core.contexts import RetrievalContext`
- `learning/collector.py` → `from ww.core.contexts import RetrievalContext`

**Effort**: 4 hours

---

### Task 11D.3: Rename Integration Module

**Current**: `src/ww/integration/` (CC-API) conflicts with `integrations/`

**Target**: `src/ww/cc_api/` for clarity

**Actions**:
1. `mv src/ww/integration src/ww/cc_api`
2. Update all imports (grep for `from ww.integration`)
3. Update `__init__.py` exports

**Effort**: 2 hours

---

### Task 11D.4: Archive Large Planning Documents

**Move to `docs/archive/planning/`**:
1. `NEURAL_MEMORY_INTEGRATION_PLAN.md` (60KB)
2. `NEURAL_MEMORY_UPGRADE_ROADMAP.md` (60KB)
3. `RETRIEVAL_OPTIMIZATION_PLAN.md` (68KB)
4. `V0.2.0_RELEASE_PLAN.md` (92KB)
5. `IMPLEMENTATION_PLAN.md` (50KB)

**Move to `docs/archive/biology/`**:
1. BIOINSPIRED_DIAGRAMS.md
2. BIOINSPIRED_INTEGRATION.md
3. BIOINSPIRED_MONITORING.md
4. BIOINSPIRED_TESTING.md
5. BIOINSPIRED_TOOLS.md
6. biological_memory_analysis.md
7. biological_network_diagram.md
8. biological_quick_reference.md

**Keep at root**: `BIOLOGICAL_PLAUSIBILITY_ANALYSIS.md` (primary)

**Effort**: 1 hour

---

### Task 11D.5: Clean Root-Level Master Files

**Move to `docs/archive/tracking/`**:
1. `BIOLOGICAL_VALIDATION_REPORT.md`
2. `MASTER_BIOLOGICAL_VALIDATION_ISSUES.md`
3. `MASTER_FIX_LIST.md`

**Effort**: 15 minutes

---

### Task 11D.6: Clean Build Artifacts

```bash
# Clean bytecode (1,557 files)
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Clean syncthing metadata
rm -rf .stfolder/

# Clean old API doc
rm docs/API.md  # Superseded by API_WALKTHROUGH.md
```

**Effort**: 15 minutes

---

### Sprint 11D Deliverables

| Task | Impact | Effort |
|------|--------|--------|
| 11D.1 Archive bridge/ | LOW | 2h |
| 11D.2 Consolidate contexts | MEDIUM | 4h |
| 11D.3 Rename integration/ | MEDIUM | 2h |
| 11D.4 Archive planning docs | LOW | 1h |
| 11D.5 Archive master files | LOW | 15m |
| 11D.6 Clean artifacts | LOW | 15m |

**Total**: 1.5 days

---

## Integration & Usage Guide

### How to Integrate World Weaver (3 Patterns)

#### Pattern 1: MCP Server (Claude Code/Desktop)

**Setup** (5 minutes):
```bash
export WW_API_URL=http://localhost:8765
python -m ww.mcp.server
```

**Add to Claude Code** (`settings.json`):
```json
{
  "mcpServers": {
    "world-weaver": {
      "command": "python",
      "args": ["-m", "ww.mcp.server"],
      "env": {"WW_API_URL": "http://localhost:8765"}
    }
  }
}
```

**Available Tools**:
- `ww_store` - Store experience
- `ww_recall` - Retrieve memories
- `ww_learn_outcome` - Report task success/failure
- `ww_consolidate` - Trigger consolidation
- `ww_get_context` - Get session context

#### Pattern 2: Python SDK

**Install**:
```bash
pip install world-weaver[api]
```

**Usage**:
```python
from ww.sdk import AgentMemoryClient

async with AgentMemoryClient(api_url="http://localhost:8765") as memory:
    # Store experience
    await memory.store_experience(
        content="Implemented retry logic",
        outcome="success",
        importance=0.8,
    )

    # Retrieve for task
    memories = await memory.retrieve_for_task(
        query="How to handle failures?",
        k=5,
    )

    # Report outcome (triggers three-factor learning)
    await memory.report_task_outcome(
        task_id=memories[0].task_id,
        success=True,
    )
```

#### Pattern 3: WWAgent (Full Agent Wrapper)

```python
from ww.sdk import WWAgent, AgentConfig

agent = WWAgent(
    config=AgentConfig(
        name="my-assistant",
        model="claude-sonnet-4-5-20250929",
        memory_enabled=True,
    ),
    ww_api_url="http://localhost:8765",
)

async with agent.session():
    response = await agent.execute([
        {"role": "user", "content": "How do I implement X?"}
    ])

    await agent.report_outcome(
        task_id=response["task_id"],
        success=True,
    )
```

---

## Deployment Checklist

### Pre-Production

- [ ] Complete Sprint 11A (learning loop)
- [ ] Complete Sprint 11B (biology gaps)
- [ ] Complete Sprint 11C (production polish)
- [ ] Complete Sprint 11D (cleanup)
- [ ] All tests passing (6,540+)
- [ ] Coverage maintained (80%+)

### Docker Deployment

```bash
# Build images
docker build -f deploy/docker/Dockerfile -t world-weaver:0.5.0 .
docker build -f deploy/docker/Dockerfile.worker -t world-weaver-worker:0.5.0 .

# Deploy with compose
docker compose -f deploy/docker/docker-compose.prod.yml up -d

# Verify
curl http://localhost:8765/api/v1/health
```

### Kubernetes Deployment

```bash
helm install ww deploy/helm/world-weaver/ \
  --namespace world-weaver \
  --create-namespace \
  --values custom-values.yaml
```

### PyPI Publication

```bash
# Build
python -m build

# Upload to PyPI
twine upload dist/*
```

---

## Timeline Summary

| Sprint | Duration | Focus | Target |
|--------|----------|-------|--------|
| 11A | 5 days | Learning Loop | Hinton 6→9/10 |
| 11B | 5 days | Biology Gaps | CompBio 87→93/100 |
| 11C | 0.5 days | Production Polish | Arch 90→98% |
| 11D | 1.5 days | Cleanup | Organization |

**Total**: ~3 weeks

**After Completion**:
- `pip install world-weaver` works
- MCP server provides 6+ tools
- Learning actually updates representations
- Biological fidelity validated
- Production infrastructure ready

---

## Success Metrics

### Learning Effectiveness (Hinton Target: 9/10)
- [ ] FFEncoder weights change through use
- [ ] Reconsolidated embeddings persisted
- [ ] Eligibility populated on retrieval
- [ ] Batch three-factor applied

### Biological Fidelity (CompBio Target: 93/100)
- [ ] Protein synthesis gate functional
- [ ] Ripple oscillator 150-250 Hz
- [ ] Forward:reverse replay ratio correct
- [ ] Documentation matches code

### Production Readiness (Architecture Target: 98%)
- [ ] Main Dockerfile exists
- [ ] Versions synchronized
- [ ] `pip install world-weaver` works
- [ ] `helm install` works
- [ ] All tests passing

### Code Organization (Cleanup Complete)
- [ ] No deprecated modules in src/
- [ ] Single canonical context classes
- [ ] Clear module naming (cc_api/ not integration/)
- [ ] <50 docs at root level

---

**Plan Status**: DETAILED PLANNING COMPLETE
**Next Step**: Begin Sprint 11A (Learning Loop Closure)
**Estimated Completion**: 3 weeks from start
