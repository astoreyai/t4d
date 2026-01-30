# World Weaver Architecture Diagrams

**Version**: 0.1.0 | **Status**: Complete | **Last Updated**: 2025-12-06

This directory contains comprehensive Mermaid-based architecture diagrams for the World Weaver memory system. Each diagram is self-contained with embedded descriptions, metrics, and implementation details.

## Quick Navigation

| Diagram | Focus | Key Concepts |
|---------|-------|--------------|
| [System Architecture](system_architecture.md) | Complete system overview | Layers, components, data flow |
| [Memory Subsystems](memory_subsystems.md) | Tripartite memory interactions | Episodic, Semantic, Procedural, Working |
| [Neural Pathways](neural_pathways.md) | Neuromodulator orchestra | DA, NE, 5-HT, ACh, GABA, plasticity |
| [Storage Resilience](storage_resilience.md) | Fault tolerance patterns | Circuit breakers, saga, fallback |
| [Embedding Pipeline](embedding_pipeline.md) | Vector generation & caching | BGE-M3, L1/L2 cache, adapters |
| [Consolidation Flow](consolidation_flow.md) | Sleep-based memory transfer | NREM, REM, SWR, pruning |

## Diagram Hierarchy

```
System Architecture (Top-level)
├── Memory Subsystems
│   ├── Episodic Memory
│   ├── Semantic Memory
│   ├── Procedural Memory
│   └── Working Memory
├── Neural Pathways
│   ├── Neuromodulator Systems
│   ├── Learned Memory Gate
│   └── Plasticity Mechanisms
├── Storage Resilience
│   ├── Circuit Breakers
│   ├── Saga Coordinator
│   └── Graceful Degradation
├── Embedding Pipeline
│   ├── Provider Adapters
│   ├── Caching Strategy
│   └── Integration
└── Consolidation Flow
    ├── NREM Phase (SWR Replay)
    ├── REM Phase (Abstraction)
    └── Pruning Phase (Homeostasis)
```

## Detailed Breakdown

### 1. System Architecture

**Purpose**: 30,000-foot view of the entire World Weaver system

**Layers**:
- Client Layer: MCP, SDK, REST API
- Memory Orchestration: Working Memory, Pattern Separation, Learned Gate
- Tripartite Memory: Episodic, Semantic, Procedural
- Neural Mechanisms: Neuromodulators, Plasticity, Reconsolidation
- Consolidation: NREM, REM, Pruning
- Storage: Neo4j, Qdrant, Circuit Breakers
- Embedding: BGE-M3, Caching

**Use Cases**:
- Onboarding new developers
- Architecture presentations
- System documentation
- Planning discussions

**Key Metrics**:
- 1259 tests, 79% coverage
- <5ms learned gate latency
- 10-20x SWR compression
- >80% embedding cache hit rate

### 2. Memory Subsystems

**Purpose**: Deep dive into how the four memory types interact

**Memory Types**:
1. **Working Memory** (7±2 items)
   - Prefrontal cortex analog
   - Attentional blink modeling
   - FIFO + priority eviction

2. **Episodic Memory** (Autobiographical events)
   - Hippocampus analog
   - Bi-temporal versioning (T_ref, T_sys)
   - Multi-factor retrieval scoring

3. **Semantic Memory** (Knowledge graph)
   - Neocortex analog
   - Hebbian learning on co-retrieval
   - ACT-R activation-based retrieval

4. **Procedural Memory** (Skills & workflows)
   - Basal ganglia analog
   - Memp Build-Retrieve-Update lifecycle
   - Success rate tracking

**Interactions**:
- Episodic → Semantic (Consolidation)
- Semantic → Episodic (Context for retrieval)
- Procedural → Episodic (Skill building from trajectories)
- Working → All (Active processing hub)

**Retrieval Algorithms**:
- Episodic: 0.4×semantic + 0.25×recency + 0.2×outcome + 0.15×importance
- Semantic: ACT-R (base-level + spreading activation)
- Procedural: 0.6×semantic + 0.4×success_rate

### 3. Neural Pathways

**Purpose**: Neuromodulator systems and their effects on learning/plasticity

**Neuromodulator Systems**:

1. **Dopamine** (Reward Prediction Error)
   - Range: [0, 2], baseline 1.0
   - RPE: δ = actual - expected
   - Effect: Reinforcement signal, exploration

2. **Norepinephrine** (Arousal & Attention)
   - Range: [0, 2], baseline 0.5
   - Arousal states: Drowsy, Alert, Vigilant, Hyperaroused
   - Effect: Novelty detection, exploration/exploitation

3. **Acetylcholine** (Encoding/Retrieval Mode)
   - Range: [0, 1], baseline 0.5
   - Modes: ENCODING (>0.6), RETRIEVAL (<0.4), BALANCED (0.4-0.6)
   - Effect: Plasticity modulation

4. **Serotonin** (Long-term Credit Assignment)
   - Range: [0, 1], baseline 0.5
   - Eligibility traces for delayed rewards
   - Effect: Patience, temporal credit

5. **GABA/Inhibition** (Competition & Sparsity)
   - Range: [0, 1]
   - Lateral inhibition, winner-take-all
   - Effect: Sparse retrieval (2-5% active)

**Learning Components**:
- Learned Memory Gate: Online Bayesian LR, Thompson sampling
- Learned Retrieval Scorer: Neural reranker
- Reconsolidation Engine: Dopamine-modulated updating
- Homeostatic Plasticity: Synaptic scaling

**Integration**:
- Orchestra computes unified state
- State modulates gate, scorer, plasticity
- Feedback from outcomes updates systems

### 4. Storage Resilience

**Purpose**: Fault tolerance and graceful degradation patterns

**Resilience Patterns**:

1. **Circuit Breaker**
   - States: CLOSED → OPEN → HALF_OPEN → CLOSED
   - Failure threshold: 5 consecutive failures
   - Reset timeout: 60 seconds
   - Success threshold: 2 in HALF_OPEN

2. **Saga Pattern** (Distributed Transactions)
   - Two-phase commit across Neo4j + Qdrant
   - Automatic rollback on failure
   - Ensures atomicity

3. **Graceful Degradation**
   - Level 0: Normal (both backends up)
   - Level 1: No graph features (Neo4j down)
   - Level 2: No vector search (Qdrant down)
   - Level 3: In-memory only (both down)

4. **Fallback Cache**
   - LRU cache (1000 items)
   - Pending queue for replay
   - Drain to backends on recovery

5. **Health Monitoring**
   - Periodic health checks (30s interval)
   - Metrics collection
   - Alert system on state transitions

**Performance Impact**:
- Circuit breaker: <100μs overhead
- Saga coordinator: ~1-5ms overhead
- Fallback cache: <50μs overhead
- Total: <5ms latency impact

### 5. Embedding Pipeline

**Purpose**: Vector generation, caching, and provider abstraction

**Architecture**:

1. **Adapter Layer**
   - Abstract EmbeddingProvider protocol
   - Statistics tracking
   - Health monitoring

2. **Provider Implementations**
   - **BGE-M3** (Primary): Local GPU, 1024-dim, FP16
   - **Mock**: Deterministic for testing
   - **Future**: Sentence Transformer, OpenAI Ada-003

3. **Caching Strategy**
   - **L1**: In-memory LRU (1000 items, ~1μs hit)
   - **L2**: Disk SQLite (100K items, ~1ms hit)
   - **Combined hit rate**: 95-98%

4. **Integration**
   - All memory systems use adapter
   - Automatic cache invalidation
   - Batch processing support

**Performance**:
- GPU latency: ~10ms/query, ~200ms/batch-32
- CPU latency: ~100ms/query, ~3s/batch-32
- Cache hit latency: ~1μs (L1), ~1ms (L2)
- Throughput: ~100 queries/sec (GPU), ~10 queries/sec (CPU)

**Configuration**:
- Provider: bge_m3, mock, sentence_transformer, openai
- Device: cuda, cpu
- Precision: FP16 (GPU), FP32 (CPU)
- Cache: L1 + L2 enabled by default

### 6. Consolidation Flow

**Purpose**: Sleep-based memory consolidation with biological inspiration

**Sleep Phases**:

1. **NREM Sleep** (~75% of cycle)
   - Sharp-Wave Ripple (SWR) replay
   - Temporal compression (10-20x)
   - Priority-based episode selection
   - Entity extraction from recurring patterns
   - Episodic → Semantic transfer

2. **REM Sleep** (~25% of cycle)
   - HDBSCAN clustering of semantic memories
   - LLM-based concept abstraction
   - Creative cross-cluster integration
   - Pattern discovery

3. **Pruning Phase**
   - Synaptic downscaling (homeostatic)
   - Weak connection removal
   - Strong connection preservation (synaptic tagging)
   - Target: 3% network activity

**Key Algorithms**:

- **Priority Scoring**: 0.3×outcome + 0.25×importance + 0.25×recency + 0.2×novelty
- **SWR Sequence**: Coherent episodes with similarity >0.5
- **Clustering**: HDBSCAN with min_cluster_size=3
- **Abstraction**: LLM extracts common concept from cluster
- **Pruning**: Multiplicative scaling to target activity

**Statistics** (typical cycle):
- Episodes processed: 100-1000
- SWR sequences: 10-50
- Entities created: 5-20
- Concepts abstracted: 2-5
- Connections pruned: 50-200
- Connections strengthened: 100-500
- Duration: 10-60 seconds

**Configuration**:
- NREM: Priority top-k=100, min_recurrence=3
- REM: Min_cluster_size=3, abstraction_threshold=0.7
- Prune: Target_activity=0.03, weak_threshold=0.1

## Rendering the Diagrams

### In GitHub/GitLab

All diagrams use Mermaid syntax and render automatically in:
- GitHub Markdown preview
- GitLab Markdown preview
- VS Code with Mermaid extension
- Obsidian with Mermaid plugin

### Standalone Rendering

To render diagrams as PNG/SVG:

```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Render all diagrams
for file in docs/diagrams/*.md; do
    mmdc -i "$file" -o "${file%.md}.png"
done
```

### In Documentation Sites

For Sphinx/MkDocs/Docusaurus:

```bash
# Install mermaid plugin
pip install sphinxcontrib-mermaid  # Sphinx
pip install mkdocs-mermaid2-plugin  # MkDocs
npm install @docusaurus/plugin-mermaid  # Docusaurus
```

## Color Scheme

Diagrams use consistent color coding:

| Color | Hex | Usage |
|-------|-----|-------|
| Light Blue | #e1f5ff | Client/API layer, orchestration |
| Light Yellow | #fff4e1 | Working memory, active processing |
| Light Green | #e8f5e9| Memory storage (episodic, semantic, procedural) |
| Light Purple | #f3e5f5 | Neural mechanisms, neuromodulators |
| Light Orange | #ffe0b2 | Consolidation, sleep phases |
| Light Red | #ffebee | Storage backends, infrastructure |
| Light Teal | #e0f2f1 | Embeddings, caching |

## Integration with Documentation

These diagrams complement:

1. **Code Documentation**
   - `/mnt/projects/ww/src/ww/` - Source code with docstrings
   - `/mnt/projects/ww/tests/` - Test suite (1259 tests)

2. **Written Documentation**
   - `/mnt/projects/ww/ARCHITECTURE.md` - Original architecture design
   - `/mnt/projects/ww/MEMORY_ARCHITECTURE.md` - Tripartite memory spec
   - `/mnt/projects/ww/docs/FUNCTIONAL_ARCHITECTURE.md` - Functional overview
   - `/mnt/projects/ww/docs/LEARNING_ARCHITECTURE.md` - Learning system details
   - `/mnt/projects/ww/docs/NEUROTRANSMITTER_ARCHITECTURE.md` - Neuromodulator deep dive

3. **API Documentation**
   - `/mnt/projects/ww/docs/API.md` - REST API reference
   - `/mnt/projects/ww/docs/SDK.md` - Python SDK guide
   - `http://localhost:8765/docs` - Interactive OpenAPI docs (when running)

## Maintenance

**Updating Diagrams**:
1. Edit the Mermaid code blocks directly in the `.md` files
2. Preview in VS Code or GitHub
3. Ensure consistency with color scheme
4. Update metrics/numbers if implementation changes
5. Keep synchronized with code changes

**Version Control**:
- Diagrams live in `/mnt/projects/ww/docs/diagrams/`
- Tracked in Git with rest of documentation
- Update timestamps in this README when modified

**Review Checklist**:
- [ ] Mermaid syntax is valid
- [ ] Colors follow scheme
- [ ] Metrics are accurate
- [ ] Code examples are correct
- [ ] Links to related docs work
- [ ] Renders in GitHub preview

## Contributing

When adding new diagrams:

1. Follow existing naming convention: `<topic>_<aspect>.md`
2. Include comprehensive description sections
3. Add metrics and performance data
4. Provide implementation examples
5. Use consistent color scheme
6. Update this README with new entry
7. Link to related documentation

## Questions & Feedback

For questions about the diagrams or World Weaver architecture:

1. Check related documentation (links above)
2. Review source code and tests
3. Open GitHub issue with `[docs]` tag
4. Contact maintainer

---

**Generated**: 2025-12-07 | **World Weaver Version**: 0.1.0 | **Diagram Count**: 16

---

## Complete Diagram Hierarchy (26 Diagrams)

### Level 0: System Overview (10 diagrams)
| # | Diagram | Description |
|---|---------|-------------|
| 01 | System Architecture | Complete system layers |
| 02 | Bioinspired Components | Encoding, attractor, eligibility |
| 03 | Data Flow | End-to-end processing |
| 04 | MCP Tools API | MCP tools & middleware |
| 05 | Storage Architecture | Qdrant, Neo4j, PostgreSQL |
| 06 | Memory Systems | All 4 memory types |
| 07 | Class Bioinspired | Bioinspired UML classes |
| 08 | Consolidation Pipeline | Sleep & memory transfer |
| 09 | MCP Request Sequence | Request lifecycle |
| 10 | Observability | Tracing, metrics, logging |

### Level 1: Subsystem Diagrams (5 diagrams)
| # | Diagram | Description |
|---|---------|-------------|
| 11 | Memory Subsystem | Working, Episodic, Semantic, Procedural |
| 12 | Learning Subsystem | Gate, Eligibility, Hebbian, Retrieval |
| 13 | Neuromodulation Subsystem | DA, NE, ACh, 5-HT, GABA orchestra |
| 14 | Storage Subsystem | Saga, Neo4j, Qdrant, CircuitBreaker |
| 15 | API Subsystem | MCP, REST, SDK, Visualization |

### Level 2: Class Diagrams (4 diagrams)
| # | Diagram | Description |
|---|---------|-------------|
| 21 | Class Memory | Memory subsystem class hierarchy |
| 22 | Class Learning | Learning components UML |
| 23 | Class Storage | Storage backends & coordination |
| 24 | Class Neuromod | Neuromodulator system classes |

### Level 3: State Diagrams (4 diagrams)
| # | Diagram | Description |
|---|---------|-------------|
| 31 | State Circuit Breaker | CLOSED→OPEN→HALF_OPEN states |
| 32 | State Consolidation | AWAKE→NREM→REM→PRUNING cycle |
| 33 | State Neuromod | DA/NE/ACh state transitions |
| 34 | State Memory Gate | Evaluation→Pending→Update cycle |

### Level 4: Sequence Diagrams (3 diagrams)
| # | Diagram | Description |
|---|---------|-------------|
| 41 | Seq Store Memory | Full store flow with gate & saga |
| 42 | Seq Retrieve Memory | Retrieval with Hebbian updates |
| 43 | Seq Consolidation | NREM/REM/Pruning sequence |

### Regenerating PNGs

```bash
cd docs/diagrams
for f in *.mmd; do
    mmdc -i "$f" -o "${f%.mmd}.png" -b transparent -w 1600 -H 1200
done
```
