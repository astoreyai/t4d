# T4DM State Machine Diagrams

**Version**: 2.0.0
**Last Updated**: 2026-02-05

This directory contains state machine diagrams for all T4D components.

## Diagram Index

| # | Diagram | Description | States |
|---|---------|-------------|--------|
| 01 | [Memory κ-Gradient](01_memory_kappa_gradient.mermaid) | Memory consolidation levels | Raw → Replayed → Transitional → Semantic → Stable |
| 02 | [Consolidation Sleep Cycle](02_consolidation_sleep_cycle.mermaid) | NREM/REM/PRUNE phases | WAKE → NREM → REM → PRUNE → WAKE |
| 03 | [Spiking Cortical Block](03_spiking_cortical_block.mermaid) | 6-stage spiking pipeline | Thalamic → LIF → Attention → Apical → RWKV → Output |
| 04 | [Neuromodulator States](04_neuromodulator_states.mermaid) | Cognitive mode switching | Balanced ↔ Encoding ↔ Retrieval ↔ Exploration |
| 05 | [T4DX Storage Engine](05_t4dx_storage_engine.mermaid) | LSM write/read/compact | WAL → MemTable → Segments → Compaction |
| 06 | [Circuit Breaker](06_circuit_breaker.mermaid) | Resilience pattern | CLOSED → OPEN → HALF_OPEN → CLOSED |
| 07 | [Session Lifecycle](07_session_lifecycle.mermaid) | Session states | Active → Idle → Consolidating → Closed |
| 08 | [Agent Phases](08_agent_phases.mermaid) | SDK agent states | IDLE → RETRIEVAL → ENCODING → EXECUTING |
| 09 | [Shutdown Phases](09_shutdown_phases.mermaid) | Graceful shutdown | RUNNING → DRAINING → CHECKPOINTING → CLEANING |
| 10 | [Learning Eligibility](10_learning_eligibility.mermaid) | Three-factor learning | Fresh → Tagged → Decaying → Captured |
| 11 | [Hippocampal Modes](11_hippocampal_modes.mermaid) | NCA hippocampus | ENCODING ↔ RETRIEVAL ↔ CONSOLIDATION ↔ SLEEP |
| 12 | [Sleep Stages](12_sleep_stages.mermaid) | Adenosine system | AWAKE → N1 → N2 → N3 → REM |
| 13 | [Attractor States](13_attractor_states.mermaid) | Cognitive attractors | FOCUSED ↔ FLEXIBLE ↔ CHAOTIC |
| 14 | [Oscillator Phases](14_oscillator_phases.mermaid) | Neural oscillations | THETA ↔ GAMMA ↔ ALPHA ↔ DELTA |
| 15 | [SWR Phases](15_swr_phases.mermaid) | Sharp-wave ripples | OFF → PRE → ON → POST |
| 16 | [MCP Memory Lifecycle](16_mcp_memory_lifecycle.mermaid) | Automatic memory | Start → Active → Remember → End |

## State Machine Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM LEVEL                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Shutdown   │  │  Circuit    │  │   Session   │             │
│  │   Phases    │  │  Breaker    │  │  Lifecycle  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    MEMORY LEVEL                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  κ-Gradient │  │Consolidation│  │   T4DX      │             │
│  │   States    │  │ Sleep Cycle │  │  Storage    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    NEURAL LEVEL                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Spiking    │  │Neuromodulator│ │  Learning   │             │
│  │   Block     │  │   States     │ │ Eligibility │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    NCA LEVEL (Brain Regions)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │Hippocampus│ │  Sleep   │ │Attractors│ │Oscillators│          │
│  │  Modes   │ │ Stages   │ │  States  │ │  Phases  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────┴───────────────────────────────────┐
│                    INTEGRATION LEVEL                             │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │    Agent    │  │     MCP     │                               │
│  │   Phases    │  │  Lifecycle  │                               │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

## Key Transitions

### Memory Consolidation Flow
```
encode() → κ=0.0 (Raw)
    │
    ▼ NREM phase (κ += 0.05/cycle)
κ=0.15 (Replayed)
    │
    ▼ NREM continues
κ=0.4 (Transitional)
    │
    ▼ REM clustering
κ=0.85 (Semantic prototype)
    │
    ▼ Repeated access
κ=1.0 (Stable knowledge)
```

### Neuromodulator Gating
```
ACh > 0.7 → ENCODING mode (DG/CA3)
ACh < 0.3 → RETRIEVAL mode (CA1)
NE > 0.7  → EXPLORATION mode
DA > 0.7  → EXPLOITATION mode
Idle      → CONSOLIDATION mode
```

### Sleep Cycle
```
WAKE → NREM(×4) → REM → PRUNE → WAKE
         │
         └── Sharp-wave ripples
             10x temporal compression
             STDP strengthening
```

## Rendering

Generate SVG/PNG renders:

```bash
# Single diagram
npx -p @mermaid-js/mermaid-cli mmdc -i 01_memory_kappa_gradient.mermaid -o 01_memory_kappa_gradient.svg

# All diagrams
for f in *.mermaid; do
    npx -p @mermaid-js/mermaid-cli mmdc -i "$f" -o "${f%.mermaid}.svg"
    npx -p @mermaid-js/mermaid-cli mmdc -i "$f" -o "${f%.mermaid}.png"
done
```

## Usage

These diagrams document the internal state machines of T4DM for:
- Architecture documentation
- Debugging state transitions
- Understanding component behavior
- Onboarding new developers

---

*Generated 2026-02-05*
