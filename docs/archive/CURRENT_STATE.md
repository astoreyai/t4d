# World Weaver Current State

**Last Updated**: 2026-01-05 | **Version**: 0.5.0

This document provides an honest assessment of what works, what doesn't, and what's in progress.

## What Works

### Fully Functional

| Component | Status | Notes |
|-----------|--------|-------|
| REST API | **Working** | 112 endpoints, FastAPI, OpenAPI docs |
| Persistence | **Working** | WAL, checkpoints, crash recovery |
| HippocampalCircuit | **Working** | DG/CA3/CA1 pattern separation/completion |
| VTACircuit | **Working** | TD error, reward prediction |
| GlymphaticSystem | **Working** | Sleep waste clearance simulation |
| SleepConsolidation | **Working** | NREM/REM cycles, SWR replay |
| ForwardForwardLayer | **Working** | Local learning, positive/negative phases |
| CapsuleLayer | **Working** | Routing-by-agreement, pose matrices |
| EligibilityTrace | **Working** | Temporal credit assignment |
| DopamineSystem | **Working** | RPE computation, value learning |
| ThreeFactorRule | **Working** | eligibility × neuromod × dopamine |
| FSRS | **Working** | Spaced repetition scheduling |
| Oscillators | **Working** | Theta/gamma/delta/alpha bands |
| Astrocyte Layer | **Working** | Tripartite synapse modulation |

### Requires External Services

| Component | Requires | Notes |
|-----------|----------|-------|
| EpisodicMemory | Qdrant | Vector similarity search |
| SemanticMemory | Neo4j + Qdrant | Graph + vector hybrid |
| ProceduralMemory | Qdrant | Skill storage |
| Embedding Service | BGE-M3 model | Downloads on first use |

### Test Coverage

```
Total Tests: 6,785+
Passing: 6,785+
Skipped: ~25
Coverage: 80%
```

## Known Limitations

### Learning System Gap

The core learning infrastructure exists but the final wiring is incomplete:

```
Task Outcome → VTA RPE → Dopamine Signal
                              ↓
Eligibility Trace → Credit Assignment
                              ↓
Three-Factor Rule → Embedding Update ← NOT FULLY WIRED
```

**Issue**: The three-factor learning rule computes effective learning rates but these rarely update actual memory embeddings. The system *orchestrates* learning signals beautifully but doesn't *apply* them to change stored memories.

**Impact**: Memories don't strengthen/weaken based on outcomes. The learning system is more of a simulation than actual adaptation.

**Fix Planned**: Phase 2 of production roadmap addresses this.

### Biological Parameter Calibration

Some parameters need alignment with neuroscience literature:

| Parameter | Current | Target | Citation Needed |
|-----------|---------|--------|-----------------|
| STDP tau_minus | 20ms | 25-30ms | Bi & Poo 1998 |
| DG sparsity | 4% | 0.5-2% | Jung & McNaughton 1993 |
| SWR frequency | ~100Hz | 150-250Hz | Buzsaki 1992 |

**Biological Plausibility Score**: 94/100 (target: 96+)

### Storage Backend Requirements

World Weaver requires:
- **Qdrant**: Vector database for embeddings
- **Neo4j**: Graph database for relationships

These must be running for full functionality. Docker Compose is provided.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        API LAYER                                 │
│    CLI │ REST (112 endpoints) │ SDK │ MCP (planned)            │
├─────────────────────────────────────────────────────────────────┤
│                      MEMORY LAYER                                │
│  Episodic (FSRS) │ Semantic (ACT-R) │ Procedural (skills)      │
├─────────────────────────────────────────────────────────────────┤
│                      NCA LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Hippocampus  │  │    VTA       │  │  Glymphatic  │          │
│  │ DG/CA3/CA1   │  │  Dopamine    │  │  Clearance   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Forward-Fwd  │  │  Capsules    │  │ Oscillators  │          │
│  │ (Hinton 22)  │  │  (H8/H9)     │  │ θ/γ/δ/α      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
├─────────────────────────────────────────────────────────────────┤
│                     LEARNING LAYER                               │
│  Dopamine │ Eligibility │ Three-Factor │ FSRS │ Reconsolidation│
├─────────────────────────────────────────────────────────────────┤
│                     STORAGE LAYER                                │
│              Neo4j (graph) │ Qdrant (vector)                    │
├─────────────────────────────────────────────────────────────────┤
│                   OBSERVABILITY                                  │
│          OpenTelemetry │ Prometheus │ Health Checks             │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Verification

```python
# Test that core components work
import sys
sys.path.insert(0, 'src')

from ww.nca import (
    HippocampalCircuit, create_hippocampal_circuit,
    VTACircuit, create_vta_circuit,
    GlymphaticSystem, create_glymphatic_system,
    ForwardForwardLayer, ForwardForwardConfig,
    CapsuleLayer, create_capsule_layer,
)
from ww.learning import (
    EligibilityTrace,
    DopamineSystem,
    ThreeFactorLearningRule,
)
from ww.consolidation import SleepConsolidation

import numpy as np

# Hippocampus
hpc = create_hippocampal_circuit()
state = hpc.encode(np.random.randn(64))
print(f"Hippocampus: {state.mode}")  # → ENCODING

# VTA
vta = create_vta_circuit()
td = vta.compute_td_error(reward=1.0, current_state=np.random.randn(64))
print(f"VTA TD error: {td:.3f}")

# Forward-Forward
ff = ForwardForwardLayer(ForwardForwardConfig(input_dim=64))
stats = ff.train_positive(np.random.randn(4, 64))
print(f"FF goodness: {stats['goodness']:.2f}")

# Capsules
caps = create_capsule_layer(input_dim=64, num_capsules=8)
activations, poses = caps.forward(np.random.randn(2, 64))
print(f"Capsule activations: {activations.shape}")

print("✓ All core components working")
```

## Roadmap

See [Production Plan](/home/aaron/.claude/plans/streamed-toasting-coral.md) for the 12-phase roadmap to production.

**Current Phase**: Phase 1 (Foundation Cleanup)
**Target**: pip installable, Docker deployable, MCP server integration

## Files Index

### Core Modules (Working)

```
src/ww/nca/hippocampus.py      # DG/CA3/CA1 circuit
src/ww/nca/vta.py              # Dopamine/VTA circuit
src/ww/nca/forward_forward.py  # FF algorithm
src/ww/nca/capsules.py         # Capsule networks
src/ww/nca/glymphatic.py       # Waste clearance
src/ww/nca/swr_coupling.py     # Sharp-wave ripples
src/ww/learning/dopamine.py    # RPE computation
src/ww/learning/eligibility.py # Trace management
src/ww/learning/three_factor.py # Learning rule
src/ww/consolidation/sleep.py  # Sleep cycles
```

### API Modules (Working)

```
src/ww/api/server.py           # FastAPI app
src/ww/api/routes/             # 112 endpoints
src/ww/api/websocket.py        # Real-time updates
```

### Needs Work

```
src/ww/memory/episodic.py      # Needs storage backend
src/ww/memory/semantic.py      # Needs storage backend
src/ww/sdk/client.py           # Needs cleanup
```
