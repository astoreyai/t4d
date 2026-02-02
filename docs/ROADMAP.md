# World Weaver Development Roadmap

> **NOTICE (2026-01-17)**: For active execution, see `/FINAL_PLAN.md` - the single source of truth.
> 17 planning documents have been archived to `/docs/archive/plans-2026-01/`.
> This roadmap is retained for long-term vision and historical context.

**Version**: 0.1.0 → 1.0.0
**Last Updated**: 2026-01-02
**Status**: 6,784 tests passing, 80% coverage (All gaps resolved ✅)

---

## Table of Contents

1. [Current State Assessment](#current-state-assessment)
2. [Immediate Fixes (This Week)](#immediate-fixes-this-week)
3. [Short-term Roadmap (1 Month)](#short-term-roadmap-1-month)
4. [Medium-term Roadmap (3 Months)](#medium-term-roadmap-3-months)
5. [Long-term Vision (6+ Months)](#long-term-vision-6-months)
6. [Version Milestones](#version-milestones)

---

## Current State Assessment

### Version Information

**Current Version**: 0.1.0
**Release Type**: Alpha
**Last Major Release**: v2.0.1 (2025-11-27)

### Implementation Status

#### Completed Features ✓

**Phase 1-5 (v2.0.0) - COMPLETE**:
- ✓ Critical bug fixes (Cypher injection, saga failure handling)
- ✓ Performance optimizations (HDBSCAN clustering, batch queries, LRU cache)
- ✓ Security enhancements (rate limiting, input sanitization, RBAC)
- ✓ Test coverage expansion (1,259 tests, 79% coverage)
- ✓ API cleanup (standardized naming, TypedDict responses, OpenAPI schema)

**Neuromodulation System - COMPLETE**:
- ✓ LearnedMemoryGate (Bayesian logistic + Thompson sampling)
- ✓ NeuromodulatorOrchestra (DA, NE, ACh, 5-HT, GABA coordination)
- ✓ BufferManager (CA1-like temporary storage with evidence accumulation)
- ✓ Cold start priming (StatePersister + ContextLoader + PopulationPrior)
- ✓ Learning loop closed (implicit + explicit feedback, soft negatives)

**Memory Architecture - COMPLETE**:
- ✓ Tripartite memory (Episodic, Semantic, Procedural)
- ✓ Dual-store backend (Neo4j graph + Qdrant vector)
- ✓ Saga pattern for cross-store consistency
- ✓ Session isolation with automatic cleanup
- ✓ Hebbian learning for semantic relationships
- ✓ FSRS decay-weighted retrieval
- ✓ ACT-R activation-based scoring

**Infrastructure - COMPLETE**:
- ✓ REST API with FastAPI
- ✓ Python SDK (sync + async clients)
- ✓ MCP gateway for Claude Code integration
- ✓ Docker deployment (full stack + separate components)
- ✓ Provider-agnostic embedding interface (BGE-M3 default)
- ✓ Memory consolidation engine (HDBSCAN clustering, optional)

### Known Issues and Gaps

#### Critical Gaps (Blocking Production)

**G1: Embedding Pipeline Inefficiencies** (Priority: P0) - ✅ RESOLVED
- **Issue**: Raw 1024-dim BGE-M3 used without learned projection
- **Resolution**: Learned content projection (1024→128 dim) with tanh activation
- **State persistence**: W_content and b_content saved/loaded in save_state()/load_state()
- **Tests**: 8 projection tests + 8 state persistence tests (51 total in learned_gate)
- **Commit**: `07424b9` (2026-01-02)

**G2: Hash-Based Context Embedding** (Priority: P0) - ✅ RESOLVED
- **Issue**: Context strings used pure hash-based random projection
- **Resolution**: N-gram feature hashing with semantic locality preservation
- **Method**: Character 2-grams, 3-grams, and word unigrams with feature hashing
- **Tests**: test_ngram_semantic_similarity, test_ngram_caching
- **Commit**: `816bd74` (2026-01-02)

**G3: Learned Components Not Integrated** (Priority: P0) - ✅ RESOLVED
- **Issue**: `LearnedFusion` and `LearnedRetrievalScorer` existed but not used
- **Resolution**: Both fully integrated with `recall()`:
  - `LearnedFusionWeights` (lines 59-242): Query-dependent fusion weights
  - `LearnedReranker` (lines 244-429): Post-retrieval learned re-ranking
  - Both enabled by default (`learned_fusion_enabled=True`, `reranking_enabled=True`)
  - State persistence via `save_state()`/`load_state()`
  - Cold start blending with fixed weights (50 updates threshold)
- **Tests**: 47 tests in `test_learned_fusion.py` and `test_reranker.py`
- **Verified**: 2026-01-02

#### Performance Bottlenecks

**B1: Flat k-NN Search** (Priority: High) - ✅ RESOLVED
- **Issue**: O(n) search didn't scale beyond 100K episodes
- **Resolution**: ClusterIndex implemented with hierarchical two-stage search:
  - `ClusterIndex` class in `/mnt/projects/t4d/t4dm/src/t4dm/memory/cluster_index.py` (584 lines)
  - NE-modulated cluster selection (high arousal = more clusters)
  - ACh-mode affects exploration/exploitation balance
  - Per-cluster statistics enable learned routing
  - Complexity: O(K + k*n/K) where K=clusters, k=selected (~67x speedup for 100K episodes)
- **Integration**: Fully integrated with `EpisodicMemory.recall()` (lines 1183-1234)
- **Tests**: 56 tests in `test_cluster_index.py` + `test_hierarchical_retrieval.py`
- **Verified**: 2026-01-02

**B2: Fixed Sparsity** (Priority: Medium) - ✅ RESOLVED
- **Issue**: Hardcoded 10% sparsity in pattern completion
- **Resolution**: LearnedSparseIndex implemented with adaptive addressing:
  - `LearnedSparseIndex` class in `/mnt/projects/t4d/t4dm/src/t4dm/memory/learned_sparse_index.py` (474 lines)
  - Query-dependent cluster attention, feature attention, sparsity level
  - Online gradient descent from retrieval outcomes
  - Neuromodulator guidance (NE, ACh)
- **Integration**: Integrated with `EpisodicMemory.learn_from_outcome()` (lines 2536-2563)
- **Tests**: 27 tests in `test_learned_sparse_index.py`
- **Verified**: 2026-01-02

#### Test Coverage Status (All Verified ✅)

**T1: Integration Tests** - 87 tests passing
**T2: Performance Benchmarks** - 69 tests passing
**T3: HSA Biological Validation** - 38 tests passing

See Short-term Roadmap section for details.

### Test Statistics

- **Total Tests**: 6,784
- **Status**: All passing
- **Coverage**: 80% overall
  - Core modules: ~85%
  - Memory subsystems: ~80%
  - Learning components: ~75%
  - Integration: 87 tests
  - Benchmarks: 69 tests
  - HSA Biological: 38 tests
- **Test Framework**: pytest + pytest-asyncio + Hypothesis (property-based)

### Code Metrics

- **Lines of Code**: ~26,000
- **Modules**: 45+
- **API Endpoints**: 47 (REST API)
- **MCP Tools**: 17 (JSON-RPC gateway)
- **Agent Definitions**: 15+ (planned, see `AGENTS_AND_SKILLS.md`)

---

## Immediate Fixes (This Week)

### Week of 2026-01-02 - COMPLETED ✅

All P0 critical gaps and B1/B2 performance bottlenecks have been resolved:

#### WK1-1: Fix Embedding Pipeline (P0a) ✅ COMPLETE
- Learned projection layer: 1024→128 with tanh activation
- Feature dimensions reduced: 1143→247
- State persistence: `save_state()`/`load_state()` for W_content, b_content
- 8 projection tests + 8 state persistence tests
- Commit: `07424b9`

#### WK1-2: Semantic Context Embedding (P0b) ✅ COMPLETE
- N-gram feature hashing with semantic locality preservation
- Character 2-grams, 3-grams, and word unigrams
- LRU caching for performance
- 2 semantic similarity tests added
- Commit: `816bd74`

#### WK1-3: Learned Components Integration (P0c) ✅ VERIFIED
- LearnedFusionWeights: Query-dependent fusion already integrated
- LearnedReranker: Post-retrieval re-ranking already integrated
- Both enabled by default with cold start blending
- 47 tests passing

#### B1: Hierarchical Cluster Search ✅ VERIFIED
- ClusterIndex with two-stage retrieval
- NE-modulated cluster selection
- 56 tests passing

#### B2: Learned Sparse Addressing ✅ VERIFIED
- LearnedSparseIndex with adaptive sparsity
- Online gradient descent learning
- 27 tests passing

**Files**:
- `/mnt/projects/t4d/t4dm/src/t4dm/core/learned_gate.py`
- `/mnt/projects/t4d/t4dm/tests/unit/test_learned_gate.py`

**Acceptance Criteria**:
- ✓ Context strings have semantic embeddings
- ✓ Similar contexts produce similar embeddings (cosine > 0.7)
- ✓ Caching reduces redundant computation
- ✓ All existing tests pass

### Week Summary

**Status**: All P0 critical gaps and performance bottlenecks RESOLVED
**Verified**: 2026-01-02

---

## Short-term Roadmap (1 Month)

### January 2026

#### Test Coverage Improvements

The following test coverage gaps remain:

**T1: Integration Tests** ✅ VERIFIED (87 tests passing)
- `test_biology_integration.py`: P5 biology feature integration
- `test_hebbian_strengthening.py`: Semantic learning integration
- `test_neural_integration.py`: Neuromodulator coordination
- `test_session_isolation.py`: Cross-session isolation

**T2: Performance Benchmarks** ✅ VERIFIED (69 tests passing)
- `tests/performance/test_benchmarks.py`: Core performance
- `tests/performance/test_p5_benchmarks.py`: P5 feature benchmarks
- `tests/nca/test_biology_benchmarks.py`: Biology component benchmarks

**T3: HSA Biological Validation** ✅ VERIFIED (38 tests passing)
- `test_biological_validation.py`: 16 tests (DG/CA3/CA1/consolidation)
- `test_hierarchical_retrieval.py`: Pattern completion, clustering
- `test_sparse_addressing.py`: Sparsity, interference resistance
- `test_joint_optimization.py`: Gate-retrieval correlation
- See: `/mnt/projects/t4d/t4dm/docs/HSA_TESTING_PROTOCOLS.md`

#### Future Work

**Phase 3: Joint Optimization** (Planned)
- FeatureAligner for gate-retrieval correlation
- Periodic alignment during consolidation

**Phase 4: Biological Validation Tests** (Planned)
- Pattern separation/completion tests
- Interference resistance tests
- See: `HSA_TESTING_PROTOCOLS.md`

---

### Progress Summary

| Milestone | Version | Status | Deliverable |
|-----------|---------|--------|-------------|
| Embedding Pipeline Fixed | v0.1.1 | ✅ Complete | P0 gaps closed |
| Scalable Retrieval | v0.2.0 | ✅ Verified | Hierarchical search |
| Adaptive Sparsity | v0.3.0 | ✅ Verified | Learned sparse addressing |
| Learned Fusion | - | ✅ Verified | Query-dependent weights |
| Learned Re-ranking | - | ✅ Verified | Post-retrieval re-ranking |

---

## Medium-term Roadmap (3 Months)

### January 2026

#### Performance Optimization Sprint

**Goal**: Production-grade performance at scale

**Milestones**: v0.5.0 - Production Ready

##### Task PO-1: Parallel Consolidation ✅ COMPLETE

**Status**: Full parallelization implemented

**Implemented**:
- ✅ Async batch processing for Hebbian updates
- ✅ Pipeline optimization for episodic→semantic transfer
- ✅ `batch_create_relationships()` in graph store
- ✅ Configurable `extraction_batch_size`
- ✅ Multi-process cluster computation (`ParallelExecutor`)
- ✅ Parallel embedding with semaphore concurrency control
- ✅ Chunked processing for memory efficiency

**Files**:
- `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/parallel.py` (350+ lines)
- `/mnt/projects/t4d/t4dm/tests/unit/test_parallel_consolidation.py` (18 tests)

##### Task PO-2: Memory Compression ✅ COMPLETE

**Status**: Full forgetting and cold storage system implemented

**Implemented**:
- ✅ `prune_phase()` in SleepConsolidation (synaptic downscaling)
- ✅ `prune_weak_synapses()` in STDP integration
- ✅ `prune_stale_clusters()` in ClusterIndex
- ✅ FSRS decay-weighted retrieval (memory strength decay)
- ✅ Memory limits in dopamine.py, reconsolidation.py
- ✅ `ActiveForgettingSystem` class with multi-strategy scoring
- ✅ Cold storage archival with `ColdStorageManager`
- ✅ Configurable retention policies

**Files**:
- `/mnt/projects/t4d/t4dm/src/t4dm/memory/forgetting.py` (450+ lines, 20 tests)
- `/mnt/projects/t4d/t4dm/src/t4dm/storage/archive.py` (350+ lines, 10 tests)

##### Task PO-3: Distributed Deployment ✅ COMPLETE

**Status**: Full Kubernetes deployment manifests ready

**Implemented**:
- ✅ Kubernetes deployment manifests (8 files)
- ✅ Multi-replica API servers (3-10 with HPA)
- ✅ Shared Neo4j/Qdrant cluster configuration
- ✅ Session affinity (ClientIP + cookie-based)
- ✅ Health checks (readiness/liveness probes)
- ✅ Rate limiting and CORS at ingress
- ✅ Pod disruption budget for availability

**Files** (in `/mnt/projects/t4d/t4dm/deploy/kubernetes/`):
- `namespace.yaml`, `configmap.yaml`, `secrets.yaml`
- `deployment.yaml` (health probes, anti-affinity)
- `service.yaml` (ClusterIP + headless)
- `ingress.yaml` (NGINX with session cookies)
- `hpa.yaml` (CPU/memory autoscaling)
- `storage.yaml` (PVC + PDB)
- `kustomization.yaml` (orchestration)

##### Task PO-4: Monitoring and Observability ✅ COMPLETE

**Status**: Full production telemetry implemented

**Implemented**:
- ✅ OpenTelemetry tracing (tracing.py - 382 lines)
- ✅ Prometheus metrics (prometheus.py - 585 lines)
- ✅ Health checks (health.py)
- ✅ Custom metrics: retrieval latency, gate decisions, neuromodulator levels
- ✅ Configurable via settings (otel_enabled, otel_endpoint, etc.)

---

### February 2026

#### Feature Completeness Sprint

**Goal**: Fill gaps in memory subsystems and learning

**Milestones**: v0.6.0 - Feature Complete

##### Task FC-1: Abstraction Engine (1 week)

**Problem**: No episodic→semantic abstraction beyond clustering

**Tasks**:
- Implement `AbstractionEngine` for concept extraction
  - Frequent pattern mining in successful episodes
  - Automatic entity/relationship discovery
  - Semantic concept creation with provenance
- Integrate with consolidation pipeline
- Add abstraction quality metrics (coherence, coverage)

**Expected Impact**:
- Automated knowledge graph growth
- Better semantic memory coverage
- Reduced manual knowledge curation

##### Task FC-2: Multi-Agent Coordination (1 week)

**Problem**: No native support for agent-to-agent memory sharing

**Tasks**:
- Agent identity and namespace isolation
- Shared vs private memory partitioning
- Cross-agent memory transfer protocols
- Collaborative memory consolidation
- Agent reputation/trust scoring

**Expected Impact**:
- Multi-agent workflows supported
- Knowledge sharing without interference
- Foundation for agent orchestration layer

##### Task FC-3: Learned Projection Expansion (3 days)

**Problem**: Only content projection learned, context still uses PCA

**Tasks**:
- Replace PCA with learned projections for all context types
- Multi-task learning for projection layers
- Gradient-based optimization during consolidation
- Adaptive projection dimensionality

**Expected Impact**:
- Better task-specific representations
- Improved context sensitivity
- Reduced feature dimensionality

##### Task FC-4: ToonJSON Integration (3 days)

**Problem**: Context injection uses full JSON (wastes tokens)

**Tasks**:
- Integrate `ToonJSON` into `ContextInjector`
- Configurable compression levels (aggressive/balanced/conservative)
- Adaptive compression based on context window usage
- Decompression for explicit recall

**Expected Impact**:
- ~50% token reduction in context
- More memories fit in context window
- Lower API costs (for Claude/OpenAI APIs)

---

### March 2026

#### Documentation and Developer Experience

**Goal**: Production-ready documentation and tooling

**Milestones**: v0.7.0 - Developer Ready

##### Task DX-1: Comprehensive Documentation (1 week)

**Tasks**:
- Architectural decision records (ADRs) for all major components
- Tutorial series (beginner → advanced)
- API reference auto-generation (OpenAPI + Sphinx)
- MCP gateway cookbook (common patterns)
- Video walkthrough of memory subsystems

**Deliverables**:
- `docs/tutorials/` - Step-by-step guides
- `docs/adrs/` - Design rationale
- `docs/cookbook/` - Common recipes
- `docs/api/` - Auto-generated reference

##### Task DX-2: Development Tooling (3 days)

**Tasks**:
- Interactive memory explorer (web UI)
  - Graph visualization (D3.js)
  - Vector space projection (UMAP)
  - Timeline view for episodes
- Memory profiler (analyze storage/retrieval patterns)
- Consolidation simulator (test sleep cycles)
- Load testing harness (benchmark at scale)

**Deliverables**:
- `ww-explore` - Memory visualization tool
- `ww-profile` - Performance profiler
- `ww-simulate` - Consolidation tester
- `ww-bench` - Load testing CLI

##### Task DX-3: Example Applications (1 week)

**Tasks**:
- Personal knowledge assistant (Obsidian integration)
- Research assistant (literature review + note-taking)
- Code assistant (project memory for IDE)
- Trading bot memory (market pattern learning)

**Deliverables**:
- `examples/knowledge-assistant/` - Full app
- `examples/research-assistant/` - Jupyter notebooks
- `examples/code-assistant/` - VS Code extension
- `examples/trading-bot/` - IB API integration

##### Task DX-4: Plugin System (3 days)

**Tasks**:
- Plugin architecture for memory providers
- Custom consolidation strategies (plugin hooks)
- Embedding provider plugins (easy swapping)
- Neo4j/Qdrant alternatives (e.g., Weaviate, Milvus)

**Deliverables**:
- `ww.plugins` - Plugin interface
- Example plugins (OpenAI embeddings, Weaviate store)
- Plugin development guide

---

### Q1 2026 Milestones

| Milestone | Version | Date | Key Deliverable |
|-----------|---------|------|----------------|
| Production Performance | v0.5.0 | Jan 15 | Distributed, monitored, optimized |
| Feature Complete | v0.6.0 | Feb 15 | Abstraction engine, multi-agent, ToonJSON |
| Developer Ready | v0.7.0 | Mar 15 | Docs, tooling, examples, plugins |

**Target**: v0.7.0 by end of Q1 2026, ready for external contributors

---

## Long-term Vision (6+ Months)

### April - June 2026 (v0.8.0 - v1.0.0)

#### LTV-1: Advanced Learning Features

**Meta-Learning**:
- Cross-session meta-learning (learn how to learn)
- Automatic hyperparameter tuning (gate thresholds, consolidation frequency)
- Transfer learning between domains
- Few-shot skill acquisition

**Curriculum Learning**:
- Progressive difficulty in memory challenges
- Adaptive task scheduling based on performance
- Self-paced consolidation cycles

**Expected Impact**:
- Faster adaptation to new domains
- Better generalization
- Reduced manual tuning

---

#### LTV-2: Multi-Modal Memory

**Vision**:
- Image embeddings (CLIP/SigLIP)
- Audio embeddings (Whisper)
- Video memory (frame sequences)
- Cross-modal retrieval

**Use Cases**:
- "Show me screenshots where I saw X"
- "What was said in meetings about Y?"
- "Recall video demos of Z"

**Expected Impact**:
- Richer autobiographical memory
- Better context understanding
- Support for vision/audio assistants

---

#### LTV-3: Neuro-Symbolic Reasoning

**Goal**: Integrate symbolic reasoning with neural memory

**Components**:
- Logic programming layer (Prolog/ASP)
- Constraint satisfaction for retrieval
- Rule learning from episodes
- Analogical reasoning

**Use Cases**:
- "What's analogous to this problem?"
- "Apply the pattern from X to solve Y"
- "What rules explain these outcomes?"

**Expected Impact**:
- Explainable reasoning
- Better transfer learning
- Human-like problem solving

---

#### LTV-4: Federated Memory

**Goal**: Privacy-preserving multi-user memory

**Components**:
- Differential privacy for memory sharing
- Federated learning for gate/retrieval
- Encrypted memory storage (homomorphic)
- User consent and data governance

**Use Cases**:
- Team knowledge bases (privacy-preserving)
- Cross-organization learning
- Personal data sovereignty

**Expected Impact**:
- Enterprise-ready privacy
- Collaborative learning without data sharing
- GDPR/CCPA compliance

---

#### LTV-5: Cognitive Augmentation

**Goal**: Seamless human-AI memory integration

**Components**:
- Browser extension (capture web context)
- Mobile app (on-the-go memory)
- Wearable integration (lifelogging)
- Proactive memory suggestions

**Use Cases**:
- "You might want to revisit X from last week"
- Auto-tag important moments
- Contextual memory triggers (location, time, people)

**Expected Impact**:
- True personal AI assistant
- Ambient intelligence
- Enhanced human memory

---

### v1.0.0 Release Criteria (June 2026)

**Feature Completeness**:
- ✓ All memory subsystems production-ready
- ✓ Scalable to 1M+ memories
- ✓ Sub-100ms retrieval latency (p95)
- ✓ 99.9% uptime in production
- ✓ Comprehensive test coverage (>90%)

**Developer Experience**:
- ✓ Complete documentation (tutorials, API ref, cookbook)
- ✓ Active community (Discord, GitHub Discussions)
- ✓ 10+ example applications
- ✓ Plugin ecosystem (5+ contributed plugins)

**Production Deployments**:
- ✓ 5+ production users (beta program)
- ✓ Case studies published
- ✓ Performance benchmarks public
- ✓ Security audit completed

**Roadmap**:
- ✓ v1.x roadmap published (next 6 months)
- ✓ Long-term vision (2-3 years)
- ✓ Contributor guide
- ✓ Governance model

---

## Version Milestones

### v0.1.x - Alpha Series (Current)

**v0.1.0** (Nov 2025) - Initial release
**v0.1.1** (Dec 2025) - P0 embedding pipeline fixes

**Focus**: Core functionality, internal testing

---

### v0.2.x - Beta Series (Dec 2025 - Jan 2026)

**v0.2.0** (Dec 2025) - Hierarchical retrieval
**v0.3.0** (Dec 2025) - Adaptive sparsity
**v0.4.0** (Dec 2025) - Joint optimization
**v0.4.1** (Dec 2025) - Biological validation tests

**Focus**: Performance and scalability

---

### v0.5.x - RC Series (Jan - Mar 2026)

**v0.5.0** (Jan 2026) - Production performance
**v0.6.0** (Feb 2026) - Feature complete
**v0.7.0** (Mar 2026) - Developer ready

**Focus**: Production readiness, polish, documentation

---

### v0.8.x - Pre-Release (Apr - May 2026)

**v0.8.0** (Apr 2026) - Advanced learning
**v0.9.0** (May 2026) - Beta testing program
**v0.9.5** (May 2026) - Release candidate

**Focus**: External beta testing, final polish

---

### v1.0.0 - Production (June 2026)

**Release Date**: June 30, 2026
**Focus**: Stable, production-ready, well-documented

**Major Features**:
- Tripartite memory with 1M+ scale
- Sub-100ms retrieval (p95)
- Distributed deployment
- Multi-modal memory
- Plugin ecosystem
- Comprehensive documentation

---

## Appendix A: Priority Framework

| Priority | Criteria | Timeline | Resources |
|----------|----------|----------|-----------|
| **P0** (Critical) | Blocks production use | This week | All available |
| **P1** (High) | Significant impact | 1 month | Primary focus |
| **P2** (Medium) | Important but not blocking | 3 months | Secondary focus |
| **P3** (Low) | Nice to have | 6+ months | Opportunistic |

---

## Appendix B: Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Embedding changes break backward compat | Medium | High | Versioned state files, migration script |
| Hierarchical retrieval quality degradation | Low | High | Extensive testing, fallback to flat search |
| Performance optimization doesn't scale | Low | Medium | Benchmark early, iterative optimization |
| Community adoption slow | Medium | Medium | Example apps, tutorials, outreach |
| Neo4j/Qdrant breaking changes | Low | High | Version pinning, adapter pattern |

---

## Appendix C: Dependencies

### External Dependencies

**Required**:
- Neo4j >= 5.0.0 (graph database)
- Qdrant >= 1.7.0 (vector database)
- PyTorch >= 2.0.0 (embeddings)
- BGE-M3 model (1024-dim embeddings)

**Optional**:
- HDBSCAN >= 0.8.33 (clustering)
- FastAPI >= 0.109.0 (REST API)
- Flash Attention >= 2.4.0 (GPU acceleration)

### Internal Dependencies

**Blocking**:
- P0 fixes block v0.2.0+
- Hierarchical retrieval blocks learned sparsity
- Joint optimization needs both gate and retrieval stable

**Non-blocking**:
- Documentation can proceed in parallel
- Example apps can use current API
- Plugin system independent of core

---

## Appendix D: Related Documentation

- **Architecture**: `/mnt/projects/t4d/t4dm/ARCHITECTURE.md`
- **Memory Design**: `/mnt/projects/t4d/t4dm/MEMORY_ARCHITECTURE.md`
- **Agents**: `/mnt/projects/t4d/t4dm/AGENTS_AND_SKILLS.md`
- **HSA Implementation**: `/mnt/projects/t4d/t4dm/docs/IMPLEMENTATION_PLAN_HSA.md`
- **Testing Protocols**: `/mnt/projects/t4d/t4dm/docs/HSA_TESTING_PROTOCOLS.md`
- **Gap Analysis**: `/mnt/projects/t4d/t4dm/docs/RETRIEVAL_EXPRESSION_GAP_ANALYSIS.md`
- **Session State**: `/mnt/projects/t4d/t4dm/docs/SESSION_STATE.md`
- **Changelog**: `/mnt/projects/t4d/t4dm/CHANGELOG.md`

---

## Appendix E: Quick Reference

### Key Commands

```bash
# Environment setup
cd /mnt/projects/ww
source .venv/bin/activate
pip install -e ".[dev,api,consolidation]"

# Testing
pytest tests/ -v                          # All tests
pytest tests/unit/ -v                     # Unit only
pytest tests/integration/ -v -m integration  # Integration only
pytest --cov=src/ww --cov-report=html     # Coverage report

# Infrastructure
docker-compose up -d                      # Neo4j + Qdrant
docker-compose -f docker-compose.full.yml up -d  # Full stack

# Services
python -m ww.api.server                   # REST API
ww-memory                                 # MCP gateway
ww-explore                                # Memory explorer
ww-dashboard                              # Monitoring dashboard

# Maintenance
git log --oneline -20                     # Recent changes
git status                                # Current state
pytest tests/ --co                        # List tests
```

### Key Metrics to Track

**Performance**:
- Retrieval latency (p50, p95, p99)
- Consolidation duration
- Memory footprint (Neo4j + Qdrant + RAM)
- API throughput (req/s)

**Quality**:
- Gate decision accuracy (precision/recall)
- Retrieval relevance (P@5, P@10, NDCG@10)
- Pattern completion success rate
- Buffer promotion accuracy

**System**:
- Test coverage %
- Lines of code
- Technical debt ratio
- Documentation coverage

---

**End of Roadmap**

*This roadmap is a living document and will be updated as priorities shift and new insights emerge. Last updated: 2026-01-02*
