# World Weaver Functional Architecture

**Generated**: 2025-12-06
**Purpose**: Functional decomposition by system role and data flow

---

## Functional Decomposition by Role

### F1. ENCODING PATHWAY (Input → Storage)
*"What should be remembered and how?"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENCODING PATHWAY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │  Input   │───▶│   Embedding  │───▶│   Memory    │───▶│  Storage  │ │
│  │ (Content)│    │   (BGE-M3)   │    │    Gate     │    │ Decision  │ │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────────┘ │
│                                              │                         │
│                                              ▼                         │
│                                     ┌─────────────┐                   │
│                                     │ Neuromod    │                   │
│                                     │ Orchestra   │                   │
│                                     │ (DA/NE/5HT) │                   │
│                                     └─────────────┘                   │
│                                                                         │
│  Components:                                                            │
│  - BGEM3Embedding: Dense/sparse/colbert embeddings                     │
│  - LearnedMemoryGate: Bayesian storage decision                        │
│  - NeuromodulatorOrchestra: Context-dependent modulation               │
│  - PatternSeparation (DentateGyrus): Orthogonalize similar inputs      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Classes**:
- `BGEM3Embedding` → Multi-vector embeddings
- `LearnedMemoryGate` → Adaptive gating with Bayesian learning
- `DentateGyrus` → Pattern separation (~4% sparsity, biological)
- `NeuromodulatorOrchestra` → DA, NE, 5-HT, ACh coordination

---

### F2. STORAGE PATHWAY (Memory → Persistence)
*"How is information organized and persisted?"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STORAGE PATHWAY                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                    MEMORY SYSTEMS                            │       │
│  ├──────────────┬──────────────┬──────────────┬────────────────┤       │
│  │   Episodic   │   Semantic   │  Procedural  │    Working     │       │
│  │   Memory     │   Memory     │   Memory     │    Memory      │       │
│  │  (Episodes)  │  (Entities)  │   (Skills)   │   (Active)     │       │
│  └──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘       │
│         │              │              │                │               │
│         ▼              ▼              ▼                ▼               │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                    INDEX LAYER (HSA)                        │       │
│  ├──────────────┬──────────────┬──────────────────────────────┤       │
│  │ ClusterIndex │ LearnedSparse│    FeatureAligner            │       │
│  │   (CA3)      │    Index     │  (Joint Optimization)        │       │
│  └──────┬───────┴──────┬───────┴──────────────┬───────────────┘       │
│         │              │                      │                        │
│         └──────────────┼──────────────────────┘                        │
│                        ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                  PERSISTENCE LAYER                          │       │
│  ├────────────────────────┬────────────────────────────────────┤       │
│  │      QdrantStore       │         Neo4jStore                 │       │
│  │   (Vector Embeddings)  │    (Graph Relationships)           │       │
│  │   ┌──────────────┐     │     ┌──────────────┐               │       │
│  │   │Circuit Break.│     │     │Circuit Break.│               │       │
│  │   │ (Resilience) │     │     │ (Resilience) │               │       │
│  │   └──────────────┘     │     └──────────────┘               │       │
│  └────────────────────────┴────────────────────────────────────┘       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Classes**:
- `EpisodicMemory`, `SemanticMemory`, `ProceduralMemory` → Memory types
- `WorkingMemory` → Capacity-limited active buffer
- `ClusterIndex` → CA3-like hierarchical clustering
- `LearnedSparseIndex` → Query-dependent sparse addressing
- `FeatureAligner` → Joint gating-retrieval optimization
- `QdrantStore`, `Neo4jStore` → Persistence backends with circuit breaker protection
- `CircuitBreaker` → Fault tolerance and graceful degradation

---

### F3. RETRIEVAL PATHWAY (Query → Results)
*"How is relevant information found?"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PATHWAY                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────┐ │
│  │  Query   │───▶│  Embedding   │───▶│   Index     │───▶│ Candidates│ │
│  │          │    │   Encode     │    │   Lookup    │    │           │ │
│  └──────────┘    └──────────────┘    └─────────────┘    └─────┬─────┘ │
│                                                               │       │
│                                                               ▼       │
│  ┌───────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────┐ │
│  │  Results  │◀───│   Reranker   │◀───│  Inhibition │◀───│ Fusion  │ │
│  │           │    │   (Learned)  │    │   Network   │    │ Scoring │ │
│  └───────────┘    └──────────────┘    └─────────────┘    └─────────┘ │
│                                                                       │
│  Components:                                                          │
│  - Multi-signal fusion: dense + sparse + graph + recency             │
│  - InhibitoryNetwork: Lateral inhibition for sparsity                │
│  - LearnedReranker: Neural reranking from feedback                   │
│  - PatternCompletion: CA3-like pattern completion                    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Classes**:
- `LearnedFusionWeights` → Adaptive score fusion
- `LearnedReranker` → Neural reranking
- `InhibitoryNetwork` → Winner-take-all competition
- `PatternCompletion` → Cue-based retrieval

---

### F4. LEARNING PATHWAY (Feedback → Adaptation)
*"How does the system improve from experience?"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LEARNING PATHWAY                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    SIGNAL COLLECTION                         │      │
│  ├──────────────┬──────────────┬──────────────┬────────────────┤      │
│  │  Retrieval   │   Outcome    │   Implicit   │    Explicit    │      │
│  │   Events     │   Feedback   │   Signals    │    Feedback    │      │
│  └──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘      │
│         │              │              │                │              │
│         └──────────────┼──────────────┼────────────────┘              │
│                        ▼              ▼                               │
│               ┌─────────────┐  ┌─────────────┐                        │
│               │   Event     │  │  Dopamine   │                        │
│               │  Collector  │  │    RPE      │                        │
│               └──────┬──────┘  └──────┬──────┘                        │
│                      │                │                               │
│                      ▼                ▼                               │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │                    ADAPTATION TARGETS                        │     │
│  ├──────────────┬──────────────┬──────────────┬────────────────┤     │
│  │  Gate        │  Retrieval   │   Fusion     │   Hebbian      │     │
│  │  Weights     │   Scorer     │   Weights    │   Strengths    │     │
│  └──────────────┴──────────────┴──────────────┴────────────────┘     │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐     │
│  │          DOPAMINE-MODULATED RECONSOLIDATION                  │     │
│  │  Memory retrieval → DA modulation → Labilization → Update   │     │
│  └──────────────────────────────────────────────────────────────┘     │
│                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Classes**:
- `EventCollector` → Collect learning signals
- `DopamineSystem` → Reward prediction error
- `DopamineModulatedReconsolidation` → DA-gated memory updating
- `LearnedRetrievalScorer` → Neural ranking model
- `ScorerTrainer` → Online training
- `PlasticityManager` → Synaptic plasticity coordination

---

### F5. CONSOLIDATION PATHWAY (Short-term → Long-term)
*"How are memories stabilized over time?"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      CONSOLIDATION PATHWAY                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                    SLEEP CONSOLIDATION                       │      │
│  │                                                              │      │
│  │  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │      │
│  │  │  WAKE   │───▶│  NREM   │───▶│   REM   │───▶│  WAKE   │  │      │
│  │  │(Encode) │    │(Replay) │    │(Integrate)   │(Retrieve)│  │      │
│  │  └─────────┘    └─────────┘    └─────────┘    └─────────┘  │      │
│  │                      │              │                       │      │
│  │                      ▼              ▼                       │      │
│  │              ┌─────────────┐  ┌─────────────┐               │      │
│  │              │  Episodic   │  │  Semantic   │               │      │
│  │              │   Replay    │  │ Abstraction │               │      │
│  │              └─────────────┘  └─────────────┘               │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │                   RECONSOLIDATION                            │      │
│  │  Memory reactivation → Labilization → Update → Restabilize  │      │
│  └──────────────────────────────────────────────────────────────┘      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Classes**:
- `SleepConsolidation` → Sleep cycle simulation
- `ConsolidationService` → Orchestrates consolidation
- `ReconsolidationEngine` → Memory update on reactivation
- `DopamineModulatedReconsolidation` → DA-gated memory updating during retrieval
- `BufferManager` → Working memory → LTM promotion

**Reconsolidation Process**:
1. Memory retrieval triggers labilization
2. Dopamine signal modulates update strength
3. Memory becomes temporarily malleable
4. New information integrated
5. Memory restabilized with updates

---

### F6. INTERFACE PATHWAY (External → Internal)
*"How do external systems interact?"*

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       INTERFACE PATHWAY                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────┐       │
│  │                    EXTERNAL INTERFACES                       │       │
│  ├──────────────┬──────────────┬──────────────┬────────────────┤       │
│  │  MCP Server  │  REST API    │  Claude Code │    Kymera      │       │
│  │  (Claude)    │  (Web/App)   │ Integration  │   (Voice)      │       │
│  └──────┬───────┴──────┬───────┴──────┬───────┴────────┬───────┘       │
│         │              │              │                │               │
│         └──────────────┼──────────────┼────────────────┘               │
│                        ▼              ▼                                │
│                ┌─────────────────────────────────────┐                 │
│                │         VALIDATION LAYER            │                 │
│                │  (Session, Input, Rate Limiting)    │                 │
│                └─────────────────────────────────────┘                 │
│                                 │                                      │
│                                 ▼                                      │
│                ┌─────────────────────────────────────┐                 │
│                │       UNIFIED MEMORY SERVICE        │                 │
│                │    (Cross-memory coordination)      │                 │
│                └─────────────────────────────────────┘                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Classes**:
- MCP: `gateway.py`, tool handlers
- API: FastAPI routes with configurable CORS
- Claude Code: `WWMemory`, `WWObserver`
- Kymera: `VoiceMemoryBridge`, `JarvisMemoryHook`

**Security Features**:
- Configurable CORS policies (see Resilience Patterns section)
- Session validation
- Input sanitization
- Rate limiting
- Privacy filtering

---

## Agent-Based Decomposition

### A1. MEMORY AGENTS
*Agents responsible for memory operations*

| Agent | Role | Key Classes |
|-------|------|-------------|
| **Episodic Agent** | Store/recall experiences | `EpisodicMemory`, `Episode` |
| **Semantic Agent** | Manage knowledge graph | `SemanticMemory`, `Entity`, `Relationship` |
| **Procedural Agent** | Store/execute skills | `ProceduralMemory`, `Procedure` |
| **Working Memory Agent** | Active context management | `WorkingMemory`, `BufferManager` |

### A2. LEARNING AGENTS
*Agents responsible for adaptation*

| Agent | Role | Key Classes |
|-------|------|-------------|
| **Dopamine Agent** | Reward prediction error | `DopamineSystem`, `RewardPredictionError` |
| **Norepinephrine Agent** | Arousal/attention | `NorepinephrineSystem`, `ArousalState` |
| **Serotonin Agent** | Temporal discounting | `SerotoninSystem`, `EligibilityTrace` |
| **Acetylcholine Agent** | Encoding/retrieval mode | `AcetylcholineSystem`, `CognitiveMode` |
| **Plasticity Agent** | Synaptic updates | `PlasticityManager`, `SynapticTagger` |

### A3. INDEXING AGENTS
*Agents responsible for efficient retrieval*

| Agent | Role | Key Classes |
|-------|------|-------------|
| **Cluster Agent** | Hierarchical organization | `ClusterIndex`, `ClusterMeta` |
| **Sparse Agent** | Query-dependent addressing | `LearnedSparseIndex`, `SparseAddressingResult` |
| **Alignment Agent** | Joint optimization | `FeatureAligner`, `JointLossWeights` |
| **Pattern Agent** | Separation/completion | `DentateGyrus`, `PatternCompletion` |

### A4. CONSOLIDATION AGENTS
*Agents responsible for memory stabilization*

| Agent | Role | Key Classes |
|-------|------|-------------|
| **Sleep Agent** | Offline consolidation | `SleepConsolidation`, `SleepPhase` |
| **Replay Agent** | Memory replay | `ReplayEvent`, `AbstractionEvent` |
| **Reconsolidation Agent** | Memory updates | `ReconsolidationEngine`, `DopamineModulatedReconsolidation` |

### A5. INTERFACE AGENTS
*Agents responsible for external communication*

| Agent | Role | Key Classes |
|-------|------|-------------|
| **MCP Agent** | Claude tool interface | Gateway, tool handlers |
| **API Agent** | REST endpoint handling | FastAPI routes |
| **Voice Agent** | Kymera integration | `VoiceMemoryBridge` |
| **Observer Agent** | Event tracking | `WWObserver`, `EventCollector` |

---

## Master Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         WORLD WEAVER ARCHITECTURE                             ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                        INTERFACE LAYER                                  │ ║
║  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐           │ ║
║  │  │    MCP    │  │    API    │  │  Claude   │  │  Kymera   │           │ ║
║  │  │  Server   │  │  Server   │  │   Code    │  │  Voice    │           │ ║
║  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘           │ ║
║  └────────┼──────────────┼──────────────┼──────────────┼────────────────┘ ║
║           └──────────────┴──────────────┴──────────────┘                   ║
║                                    │                                        ║
║                                    ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                      VALIDATION & ROUTING                               │ ║
║  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ ║
║  │  │  Session    │  │   Input     │  │    Rate     │  │   Privacy   │   │ ║
║  │  │ Validation  │  │ Sanitize    │  │   Limiter   │  │   Filter    │   │ ║
║  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                        ║
║                                    ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                        MEMORY LAYER                                     │ ║
║  │                                                                         │ ║
║  │  ┌───────────────────────────────────────────────────────────────────┐ │ ║
║  │  │                    MEMORY SYSTEMS                                 │ │ ║
║  │  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐             │ │ ║
║  │  │  │Episodic │  │Semantic │  │Procedur.│  │ Working │             │ │ ║
║  │  │  │ Memory  │  │ Memory  │  │ Memory  │  │ Memory  │             │ │ ║
║  │  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘             │ │ ║
║  │  └───────┼────────────┼────────────┼────────────┼────────────────────┘ │ ║
║  │          └────────────┴────────────┴────────────┘                      │ ║
║  │                              │                                          │ ║
║  │  ┌───────────────────────────┴───────────────────────────────────────┐ │ ║
║  │  │                      HSA INDEX LAYER                              │ │ ║
║  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │ │ ║
║  │  │  │  Cluster    │  │   Sparse    │  │  Feature    │               │ │ ║
║  │  │  │   Index     │  │   Index     │  │  Aligner    │               │ │ ║
║  │  │  │   (CA3)     │  │  (Learned)  │  │  (Joint)    │               │ │ ║
║  │  │  └─────────────┘  └─────────────┘  └─────────────┘               │ │ ║
║  │  └───────────────────────────────────────────────────────────────────┘ │ ║
║  │                              │                                          │ ║
║  │  ┌───────────────────────────┴───────────────────────────────────────┐ │ ║
║  │  │                   BIOLOGICAL LAYER                                │ │ ║
║  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │ │ ║
║  │  │  │   Pattern   │  │  Inhibition │  │  Completion │               │ │ ║
║  │  │  │ Separation  │  │   Network   │  │   (CA3)     │               │ │ ║
║  │  │  │ (DG ~4%)    │  │             │  │             │               │ │ ║
║  │  │  └─────────────┘  └─────────────┘  └─────────────┘               │ │ ║
║  │  └───────────────────────────────────────────────────────────────────┘ │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                    │                                        ║
║           ┌────────────────────────┼────────────────────────┐              ║
║           ▼                        ▼                        ▼              ║
║  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        ║
║  │  LEARNING LAYER │    │CONSOLIDATION    │    │  ENCODING LAYER │        ║
║  │                 │    │     LAYER       │    │                 │        ║
║  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │        ║
║  │ │ Neuromod    │ │    │ │   Sleep     │ │    │ │  Embedding  │ │        ║
║  │ │ Orchestra   │ │    │ │ Consolidate │ │    │ │   BGE-M3    │ │        ║
║  │ ├─────────────┤ │    │ ├─────────────┤ │    │ ├─────────────┤ │        ║
║  │ │ DA│NE│5HT│ACh│ │    │ │NREM│REM│Wake│ │    │ │Dense│Sparse│ │        ║
║  │ └─────────────┘ │    │ └─────────────┘ │    │ │  │ColBERT   │ │        ║
║  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ └─────────────┘ │        ║
║  │ │  Retrieval  │ │    │ │DA-Modulated │ │    │ ┌─────────────┐ │        ║
║  │ │   Scorer    │ │    │ │ Reconsol.   │ │    │ │   Memory    │ │        ║
║  │ └─────────────┘ │    │ └─────────────┘ │    │ │    Gate     │ │        ║
║  │ ┌─────────────┐ │    │ ┌─────────────┐ │    │ └─────────────┘ │        ║
║  │ │ Plasticity  │ │    │ │   Buffer    │ │    │                 │        ║
║  │ │  Manager    │ │    │ │  Manager    │ │    │                 │        ║
║  │ └─────────────┘ │    │ └─────────────┘ │    │                 │        ║
║  └─────────────────┘    └─────────────────┘    └─────────────────┘        ║
║           │                        │                        │              ║
║           └────────────────────────┼────────────────────────┘              ║
║                                    ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                       PERSISTENCE LAYER                                 │ ║
║  │  ┌─────────────────────────┐    ┌─────────────────────────┐            │ ║
║  │  │      QdrantStore        │    │       Neo4jStore        │            │ ║
║  │  │   (Vector Embeddings)   │    │   (Graph Relations)     │            │ ║
║  │  │                         │    │                         │            │ ║
║  │  │  - Dense vectors        │    │  - Entities             │            │ ║
║  │  │  - Sparse vectors       │    │  - Relationships        │            │ ║
║  │  │  - Payload filtering    │    │  - Hebbian weights      │            │ ║
║  │  │  - Circuit breaker      │    │  - Circuit breaker      │            │ ║
║  │  └─────────────────────────┘    └─────────────────────────┘            │ ║
║  │                         │                │                              │ ║
║  │                         └────────┬───────┘                              │ ║
║  │                                  ▼                                      │ ║
║  │                    ┌─────────────────────────┐                          │ ║
║  │                    │         Saga            │                          │ ║
║  │                    │  (Transaction Coord.)   │                          │ ║
║  │                    │  (Vector preservation)  │                          │ ║
║  │                    └─────────────────────────┘                          │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────────┐ ║
║  │                      OBSERVABILITY LAYER                                │ ║
║  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐           │ ║
║  │  │  Metrics  │  │  Logging  │  │  Tracing  │  │  Health   │           │ ║
║  │  └───────────┘  └───────────┘  └───────────┘  └───────────┘           │ ║
║  └─────────────────────────────────────────────────────────────────────────┘ ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## Data Flow Diagram

```
                            USER INPUT
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Content + Context   │
                    └───────────┬───────────┘
                                │
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │   ENCODING    │   │   RETRIEVAL   │   │   FEEDBACK    │
    │    PATH       │   │     PATH      │   │     PATH      │
    └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
            │                   │                   │
            ▼                   ▼                   ▼
    ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
    │   Embed →     │   │   Query →     │   │   Outcome →   │
    │   Gate →      │   │   Index →     │   │   RPE →       │
    │   Store       │   │   Rank →      │   │   Update      │
    └───────┬───────┘   │   Return      │   └───────┬───────┘
            │           └───────┬───────┘           │
            │                   │                   │
            └───────────────────┼───────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   PERSISTENCE LAYER   │
                    │   (Qdrant + Neo4j)    │
                    └───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
            ┌───────────────┐       ┌───────────────┐
            │ CONSOLIDATION │       │   LEARNING    │
            │  (Offline)    │       │   (Online)    │
            └───────────────┘       └───────────────┘
```

---

## Resilience Patterns

### Circuit Breaker Pattern

The circuit breaker protects storage operations from cascading failures by monitoring error rates and latency.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CIRCUIT BREAKER STATES                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐                                                           │
│  │  CLOSED  │  ◄────── Normal operation, requests flow through         │
│  │ (Normal) │                                                           │
│  └────┬─────┘                                                           │
│       │                                                                 │
│       │ Error threshold exceeded (5 failures in 60s)                   │
│       ▼                                                                 │
│  ┌──────────┐                                                           │
│  │   OPEN   │  ◄────── Fast-fail mode, reject requests immediately     │
│  │ (Failing)│                                                           │
│  └────┬─────┘                                                           │
│       │                                                                 │
│       │ Timeout period (30s) elapses                                   │
│       ▼                                                                 │
│  ┌──────────┐                                                           │
│  │HALF-OPEN │  ◄────── Test mode, allow limited requests               │
│  │ (Testing)│                                                           │
│  └────┬─────┘                                                           │
│       │                                                                 │
│       ├─── Success → CLOSED                                             │
│       └─── Failure → OPEN                                               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Configuration**:
- Failure threshold: 5 failures
- Timeout window: 60 seconds
- Recovery timeout: 30 seconds
- Half-open test requests: 1

**Benefits**:
- Prevents resource exhaustion during outages
- Provides fast failure detection
- Enables graceful degradation
- Allows automatic recovery testing

### Saga Compensation Pattern

The saga pattern ensures transactional consistency across Qdrant and Neo4j with vector preservation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SAGA TRANSACTION FLOW                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FORWARD FLOW (Happy Path):                                            │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│  │  Begin   │───▶│  Qdrant  │───▶│  Neo4j   │───▶│  Commit  │         │
│  │  Saga    │    │  Write   │    │  Write   │    │  Saga    │         │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘         │
│                                                                         │
│  COMPENSATION FLOW (Rollback):                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                          │
│  │  Failure │───▶│  Restore │───▶│  Rollback│                          │
│  │ Detected │    │  Vectors │    │  Changes │                          │
│  └──────────┘    └──────────┘    └──────────┘                          │
│                       │                                                 │
│                       │ Vectors preserved from original operation       │
│                       ▼                                                 │
│                  ┌─────────┐                                            │
│                  │ Vector  │                                            │
│                  │ Cache   │                                            │
│                  └─────────┘                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Features**:
- Preserves vector embeddings during rollback
- Ensures both stores remain consistent
- Handles partial failures gracefully
- Maintains data integrity across distributed operations

### CORS Configuration

CORS (Cross-Origin Resource Sharing) is now configurable via settings for secure API access.

**Settings Location**: `src/ww/settings.py`

**Default Configuration**:
```python
cors_origins: List[str] = ["http://localhost:3000"]
cors_allow_credentials: bool = True
cors_allow_methods: List[str] = ["*"]
cors_allow_headers: List[str] = ["*"]
```

**Benefits**:
- Environment-specific CORS policies
- Secure default (localhost only)
- Production-ready configuration support
- Flexible authentication handling

---

## Testing Strategy by Functional Area

| Area | Test Focus | Current Coverage | Priority |
|------|------------|------------------|----------|
| F1. Encoding | Gate decisions, embedding quality | ~75% | HIGH |
| F2. Storage | Memory CRUD, index operations | ~85% | MEDIUM |
| F3. Retrieval | Recall accuracy, ranking quality | ~70% | HIGH |
| F4. Learning | Weight updates, convergence | ~60% | HIGH |
| F5. Consolidation | Sleep cycles, replay | ~65% | MEDIUM |
| F6. Interface | API contracts, validation | ~80% | LOW |

---

## Next: Agent Analysis Tasks

1. **Hinton Agent**: Analyze neural architecture, distributed representations, learning dynamics
2. **CompBio Agent**: Validate biological plausibility, pathway modeling
3. **AGI Agent**: Assess integration, emergent behavior, system coherence
