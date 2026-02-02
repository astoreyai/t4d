# T4DM Competitive Analysis

**Version**: 1.0 | **Date**: 2026-02-02

---

## Memory Types Supported

| System | Episodic | Semantic | Procedural | Working | Temporal (first-class) |
|--------|----------|----------|-----------|---------|----------------------|
| **T4DM** | Yes | Yes | Yes | Yes | Yes (bitemporal, kappa gradient) |
| Mem0 | Yes | Yes | No | No | Partial (timestamps only) |
| Letta/MemGPT | Yes | Yes | No | Yes (context mgmt) | Partial |
| Zep | Yes | Yes | No | No | Yes (temporal knowledge graph) |
| Supermemory | Yes | Yes | No | No | No |
| LangMem | Yes | Yes | No | No | No |
| CrewAI Memory | Yes | Yes | No | Yes | No |
| AutoGen Memory | Yes | No | No | Yes | No |

---

## Consolidation Mechanisms

| System | Consolidation | Mechanism | Biological Basis |
|--------|--------------|-----------|-----------------|
| **T4DM** | NREM/REM/Prune sleep phases | LSM compaction = biological sleep; kappa gradient 0 to 1 | Yes (Frankland & Bontempi 2005) |
| Mem0 | LLM-based extraction | Prompt LLM to extract/merge facts | No |
| Letta/MemGPT | LLM self-editing | LLM decides what to archive/forget | No |
| Zep | Graph extraction | Entity/relationship extraction pipeline | No |
| Supermemory | Chunking + dedup | RAG-style chunking | No |
| LangMem | Thread-based summarization | LLM summarizes conversations | No |
| CrewAI Memory | Entity extraction | Simple entity linking | No |
| AutoGen Memory | None | Append-only | No |

---

## Learning Mechanisms

| System | Learning | Approach |
|--------|---------|---------|
| **T4DM** | Adaptive (STDP, Hebbian, three-factor, BCM, Forward-Forward) | Trainable parameters (~65-95M), continuous online learning |
| Mem0 | Static | LLM extraction at write time, no adaptation |
| Letta/MemGPT | Semi-adaptive | LLM rewrites memory blocks |
| Zep | Static | Graph built at ingest, no weight learning |
| Supermemory | Static | Fixed RAG pipeline |
| LangMem | Static | LLM summarization only |
| CrewAI Memory | Static | Fixed entity extraction |
| AutoGen Memory | Static | Append-only log |

---

## Architecture Comparison

| System | Storage | Trainable Params | Custom Engine | Neuromodulation |
|--------|---------|-----------------|---------------|----------------|
| **T4DM** | T4DX (embedded LSM) | 65-95M (QLoRA + spiking) | Yes | DA, NE, ACh, 5-HT, GABA, Glu |
| Mem0 | External vector DB | 0 | No | No |
| Letta/MemGPT | PostgreSQL + pgvector | 0 | No | No |
| Zep | Custom graph + vector | 0 | Partial | No |
| Supermemory | Vector DB | 0 | No | No |
| LangMem | Any LangChain store | 0 | No | No |
| CrewAI Memory | External vector DB | 0 | No | No |
| AutoGen Memory | In-memory / external | 0 | No | No |

---

## T4DM Unique Differentiators

1. **Spiking neural adapter on frozen LLM**: QLoRA + 6 cortical blocks with LIF neurons. No published work combines these.
2. **LSM compaction = biological consolidation**: Storage engine maintenance IS the memory consolidation process (NREM/REM/Prune).
3. **Continuous kappa-gradient**: Memories flow from kappa=0.0 (raw episodic) to kappa=1.0 (stable semantic) without cross-store transactions.
4. **Neuromodulator dynamics**: 6-NT PDE system (DA, NE, ACh, 5-HT, GABA, Glu) with brain-region-specific circuits (VTA, LC, Raphe, Nucleus Basalis).
5. **Bitemporal queries**: "What did we know when" via event_time + record_time + valid_from/valid_until.
6. **50-80M trainable memory parameters**: The memory system itself learns, not just the LLM it wraps.
7. **Custom embedded storage engine**: Zero network hops, co-located vectors + edges + metadata + temporal indices.
8. **Glass-box observability**: Every activation in 36 Qwen layers + 6 spiking blocks + all neuromodulator states observable.

---

## Gaps to Address

| Gap | Impact | Planned |
|-----|--------|---------|
| No framework adapters (LangChain, LlamaIndex, AutoGen, CrewAI) | Adoption barrier | Phase E (OPTIMIZATION_PLAN.md) |
| No competitive benchmarks (LongMemEval) | Credibility | Phase G-01 |
| Cerebellum module missing | Bio-plausibility completeness | Phase G-06 |
| No multi-user/multi-tenant support | Enterprise use | Not yet planned |
| Single-node only | Scale | Not yet planned (embedded by design) |

---

## Market Context

| System | Funding | Open Source | Production Users |
|--------|---------|-------------|-----------------|
| Mem0 | $24M Series A | Yes (MIT) | Yes |
| Letta/MemGPT | $10M Seed | Yes (Apache-2.0) | Yes |
| Zep | $6.5M Seed | Yes (Apache-2.0) | Yes |
| Supermemory | Unknown | Yes | Early |
| LangMem | Part of LangChain | Yes | Early |
| **T4DM** | Research project | Yes | Research only |

T4DM is not competing on commercial maturity. Its positioning is as a research-grade, biologically-inspired memory system that pushes the boundary of what AI memory can be. The competitive analysis is for architectural comparison, not market positioning.

---

*Generated 2026-02-02*
