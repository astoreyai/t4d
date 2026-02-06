# T4DM Limitations

This document honestly describes the current limitations and development status of T4DM. We believe in transparent communication about what the system can and cannot do.

## Validation Status

### What We Have Validated

| Component | Type | Status |
|-----------|------|--------|
| Unit tests | 9,600+ tests | Passing |
| Type annotations | 100% coverage | Mypy clean |
| Bio-plausibility benchmark | 11/11 checks | Passing |
| LongMemEval benchmark | Implemented | Synthetic only |
| DMR benchmark | Implemented | Synthetic only |

### What Requires Further Validation

| Claim | Current Evidence | What's Needed |
|-------|-----------------|---------------|
| FSRS decay improves retention | Parameter tuning only | Longitudinal user studies |
| Sleep consolidation helps | Benchmark comparisons | Multi-day deployment tests |
| Neuromodulators improve learning | Unit tests pass | A/B testing with/without |
| Spiking blocks add value | Architecture works | Comparison vs. standard attention |

## Architectural Limitations

### Not Production-Ready For

1. **Multi-tenant deployment**: No tenant isolation, shared global state
2. **High-availability**: No clustering, replication, or failover
3. **Large-scale deployment**: Single-node, ~1M memories practical limit
4. **Edge/mobile**: Requires Qwen 3B (10GB+ VRAM)

### Known Technical Limitations

| Area | Limitation |
|------|------------|
| Storage | Single-node LSM, no distributed sharding |
| Embeddings | Qwen 3B required; smaller models untested |
| Latency | Cold start ~30s (model loading) |
| Memory | 16GB+ RAM required for full system |
| VRAM | 10GB inference, 16GB training |

## Scientific Rigor

### Bio-Inspired ≠ Biologically Validated

Our components are *inspired by* neuroscience, not *validated against* biological systems:

| Component | Inspiration | Limitation |
|-----------|-------------|------------|
| FSRS decay | Ebbinghaus forgetting curve | Uses simplified power-law, not individual learning curves |
| Sleep consolidation | CLS theory | Simulated sleep phases, not actual circadian timing |
| Neuromodulators | DA/NE/ACh/5-HT systems | Scalar modulation, not receptor-level simulation |
| Spiking neurons | LIF model | Discrete timesteps, not continuous dynamics |
| STDP | Spike-timing plasticity | Simplified pre/post timing, no dendritic complexity |

### IIT-Inspired Metrics Disclaimer

The `IntegrationMetrics` (formerly `ConsciousnessMetrics`) module computes integration scores inspired by Integrated Information Theory. These metrics measure *computational coupling* between subsystems, NOT consciousness or awareness. High Φ indicates tight spiking-memory coupling, which is a useful engineering diagnostic.

### "Dreaming" Terminology

The `dreaming/` module implements generative replay during simulated REM sleep, inspired by DreamerV3 world-model training. This is a computational consolidation process, not a claim about subjective experience.

## Benchmark Limitations

### Current Benchmark Suite

| Benchmark | Status | Limitation |
|-----------|--------|------------|
| LongMemEval | Implemented | Synthetic data only, not real conversations |
| DMR | Implemented | Synthetic graph traversal, not real-world retrieval |
| Bio-plausibility | Implemented | Checks presence of features, not effectiveness |

### Missing Benchmarks

- **vs. Mem0/Zep/MemGPT**: No head-to-head comparisons yet
- **Ablation studies**: No component-by-component contribution analysis
- **Longitudinal**: No multi-session user studies
- **Production load**: No stress testing beyond unit tests

## What T4DM Is Good For

Despite these limitations, T4DM is well-suited for:

1. **Research prototype**: Exploring biologically-inspired memory architectures
2. **Single-user agents**: Personal AI assistants with persistent memory
3. **Development**: Building and testing memory-augmented LLM applications
4. **Education**: Learning about cognitive-inspired AI architectures

## Honest Comparisons

### vs. Simpler Alternatives

For many use cases, simpler solutions may be more appropriate:

| Use Case | Consider Instead |
|----------|------------------|
| Simple key-value memory | Redis/SQLite |
| Basic vector search | Chroma/Pinecone |
| Production chatbots | Mem0/Zep (battle-tested) |
| Large-scale systems | PostgreSQL + pgvector |

### When T4DM Makes Sense

- You want temporal decay and consolidation
- You're researching spiking neural networks + LLMs
- You need κ-gradient-based memory transitions
- You want glass-box observability of memory operations

## Roadmap to Address Limitations

### Phase 1 (Current)
- [x] Create SimpleBaseline for comparison
- [x] Add input validation layer
- [x] Add persistence checksums
- [x] Add decision tracing
- [x] Document limitations (this file)

### Phase 2 (Next)
- [ ] Convert benchmarks to CI pytest suite
- [ ] Increase core layer test coverage to 80%
- [ ] Create debugging runbooks

### Phase 3 (Future)
- [ ] Head-to-head benchmark vs. Mem0
- [ ] Ablation study on neuromodulators
- [ ] Longitudinal pilot with real users

---

*Last updated: 2026-02-06*

*We welcome community contributions to help address these limitations. See CONTRIBUTING.md for how to get involved.*
