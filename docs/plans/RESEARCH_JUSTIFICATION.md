# Research Justification: T4DM v2.0 Architectural Decisions

**Date**: 2026-01-30
**Purpose**: Evidence supporting key design decisions in the T4DM system plan

---

## 1. LSM Compaction as Biological Memory Consolidation

### Evidence

**Complementary Learning Systems (CLS) Theory** — McClelland, McNaughton & O'Reilly (1995)
- Brain uses two systems: hippocampus (fast, sparse, pattern-separated) and neocortex (slow, distributed, overlapping)
- Memories transfer from hippocampus to neocortex during sleep via replay
- **T4DX parallel**: MemTable (hippocampus, fast writes) → Segments (neocortex, immutable, distributed)
- Source: https://pubmed.ncbi.nlm.nih.gov/22141588/

**Sleep Consolidation via NREM/REM Interplay** — Communications Biology (2025)
- NREM and REM play complementary roles: NREM strengthens connections via replay, REM restructures and abstracts
- Norepinephrine and dopamine regulate timing and prioritization
- **T4DX parallel**: NREM compaction = merge + κ update + STDP; REM compaction = HDBSCAN cluster + prototype
- Source: https://www.nature.com/articles/s42003-025-07868-5

**Language Models Need Sleep** — OpenReview (2025)
- Two-step sleep (consolidation + dreaming) is more robust to catastrophic forgetting than continuous learning
- First direct application of sleep metaphor to computational memory
- Source: https://openreview.net/pdf/05bbb74851e965f5199f45f83937d1c396f048c8.pdf

**LSM Compaction Research** — ACM SIGMOD (2025)
- LSM compaction is resource-intensive but essential for read performance
- Custom compaction policies significantly impact system behavior
- **T4DX**: We replace generic size-tiered compaction with biologically-motivated NREM/REM/PRUNE semantics
- Source: https://dl.acm.org/doi/10.1145/3725344

### Assessment
**Strong foundation.** CLS theory (1995) is one of the most cited models in memory neuroscience. The LSM-compaction-as-consolidation mapping is novel but well-grounded. Publication potential.

---

## 2. Custom Embedded Storage vs Existing Solutions

### Evidence

**Network Overhead**
- Embedded (in-process) = zero network latency; queries are function calls
- External stores add 100s of milliseconds per call, severely reduce QPS
- Source: https://superlinked.com/vectorhub/articles/choosing-vdb

**Embedded DB Advantages**
- No network calls, simplified development, full data control
- Best for: single application, low latency, single-tenant
- LanceDB (Rust core): 100x faster than parquet for random access, disk-based ANN with ~95% accuracy at millisecond latency
- Source: https://thedataquarry.com/blog/embedded-db-1/

**hnswlib Performance**
- 37.9M vectors, 512 dimensions: hnswlib 22.2s for 10K queries vs FAISS 2m42s (6x faster)
- Highly optimized for in-memory CPU performance
- Source: https://zilliz.com/blog/faiss-vs-hnswlib-choosing-the-right-tool-for-vector-search

**Qdrant/Milvus Overhead at Reddit Scale**
- Filtering affects latency significantly on both
- For real-time UX < 50ms: need HNSW + RAM-heavy configs
- Ingestion and query compete for resources on same nodes
- Source: https://milvus.io/blog/choosing-a-vector-database-for-ann-search-at-reddit.md

**Why Not PostgreSQL + pgvector**
- External service, network hops, no custom compaction semantics
- pgvector HNSW rebuild is expensive, not designed for frequent small updates
- No native concept of κ-gradient, bitemporal versioning, or graph adjacency

### Assessment
**Justified.** The dual-store overhead (Neo4j + Qdrant + Saga = 570 lines of compensation logic + two network round-trips per write) is eliminated. Learning writes (Hebbian Δw, STDP, κ mutations) need O(1) latency — impossible with external stores. Custom engine co-locates vectors, edges, metadata, temporal indices in one process.

---

## 3. Spiking Neural Networks as LLM Adapters

### Evidence

**SpikingBrain** — arXiv 2509.05276 (2025)
- Converts Qwen2.5-7B to spiking with lightweight training (~150B tokens, 2% of from-scratch)
- Recovers ~90% of base model performance
- 69.15% sparsity, 100x speedup for 4M-token sequences
- **Gap**: Converts entire model, not an adapter alongside frozen LLM
- Source: https://arxiv.org/html/2509.05276v1

**SpikeGPT** — arXiv 2302.13939 (Zhu 2023)
- RWKV-inspired spiking language model with O(N) complexity
- 20x fewer operations on neuromorphic hardware
- Largest backprop-trained SNN at time of publication
- **Gap**: Trained from scratch, not on frozen LLM
- Source: https://arxiv.org/abs/2302.13939

**BrainTransformers** — arXiv 2410.14687 (2024)
- BrainTransformers-3B-Chat: SNN-compatible Transformer components
- MMLU 63.2, BBH 54.1, GSM8K 76.3
- Three-stage training including synaptic plasticity
- **Gap**: Full SNN model, not adapter on frozen LLM
- Source: https://arxiv.org/html/2410.14687v2

**S²TDPT** — arXiv 2511.14691 (2025)
- Attention via STDP: eliminates softmax, uses spike timing
- CIFAR-10: 94.35%, CIFAR-100: 78.08% with 4 timesteps
- 88.47% energy reduction vs standard Transformer
- **Gap**: Vision task only, STDP for attention mechanism
- Source: https://arxiv.org/html/2511.14691

**Three-Factor Learning** — Frontiers in Neural Circuits (2015)
- Eligibility traces bridge temporal gap between neural activity and reward
- Weight change requires neuromodulatory signal M in addition to pre/post activation
- DA, 5-HT, ACh serve as third factor signaling reward/punishment/novelty
- Instantiates policy-gradient and TD-learning with local synaptic operations
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/

**Experimental Validation** — Frontiers in Neural Circuits (2018)
- Eligibility traces decay over timescale τe, bridging activity-reward temporal gap
- Provides experimental evidence for three-factor rules in biological systems
- Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/

### Novelty Assessment
| Approach | Author | Spiking | Frozen LLM | Adapter | Memory | Three-factor |
|----------|--------|---------|------------|---------|--------|-------------|
| SpikingBrain | 2025 | Yes | No (converts) | No | No | No |
| SpikeGPT | 2023 | Yes | No (from scratch) | No | No | No |
| BrainTransformers | 2024 | Yes | No (full SNN) | No | No | Partial |
| QLoRA | 2023 | No | Yes | Yes (LoRA) | No | No |
| **T4DM** | **Ours** | **Yes** | **Yes** | **Yes (spiking+QLoRA)** | **Yes (T4DX)** | **Yes** |

**No published work combines QLoRA + spiking adapter + three-factor learning on a frozen LLM.** This is a novel architecture with publication potential.

---

## 4. QLoRA on Qwen 3B

### Evidence

**QLoRA** — NeurIPS 2023 (Dettmers et al.)
- 4-bit NormalFloat (NF4): information-theoretically optimal for normally distributed weights
- Double Quantization: reduces memory overhead from 0.5 to 0.127 bits per parameter
- Matches full 16-bit finetuning and 16-bit LoRA up to 3B parameters
- Source: https://arxiv.org/abs/2305.14314

**Memory Budget for Rank 16 on 3B Model**
- 2 targets (q_proj+v_proj) × 36 layers × 2048 dim × 16 rank = ~15M trainable params
- ~60MB for adapter weights + ~0.5GB total with gradients and optimizer states
- Fits within VRAM budget (0.5GB of 24GB)

**Multi-Adapter Deployment**
- Frozen base LLM enables parallel training of multiple LoRA adapters
- S-LoRA: simultaneous deployment of thousands of adapters sharing one base model
- X-LoRA: mixture-of-experts for LoRA with dynamic gating
- Source: https://www.inferless.com/learn/how-to-serve-multi-lora-adapters

### Assessment
**Well-proven.** QLoRA is the standard for efficient LLM fine-tuning. Memory overhead is well-characterized and fits within budget. Stacking QLoRA + spiking adapter follows established multi-adapter patterns.

---

## 5. Overlay/Delta Pattern for Learning Writes

### Design Rationale

Learning generates the most frequent writes in the system:
- Hebbian Δw after each retrieval
- STDP weight changes during consolidation replay
- κ increments after each consolidation phase
- Access count increments on every retrieval
- Lability flag changes during reconsolidation windows

**T4DX overlay pattern**:
```
field_overlays[id] = {kappa: 0.15}    # O(1) dict insert
edge_deltas[(a,b)] = +0.05            # O(1) dict insert
```

Reads merge overlays on the fly. Overlays consumed during compaction (= consolidation).

### Supporting Patterns

**LSM MemTable**: Standard pattern — all writes go to in-memory mutable buffer, flushed to immutable segments periodically. T4DX extends this with typed overlays for field and edge mutations.

**Copy-on-Write in databases**: Immutable segments with delta layers is standard in LSM (LevelDB, RocksDB, Cassandra).

**Batch gradient accumulation in ML**: Gradients accumulated over mini-batch before weight update. T4DX accumulates learning deltas over waking period, applies during consolidation.

**Biological plausibility**: Memories are labile during waking (overlays accumulate in synaptic tags), consolidated during sleep (compaction merges them into stable engrams). This maps to the synaptic tagging and capture hypothesis (Frey & Morris, 1997).

### Assessment
**Standard database technique, novel application.** Overlay pattern is well-established in LSM literature. Applying it to neural learning dynamics (Hebbian Δw as dict inserts consumed during sleep-compaction) is novel and biologically plausible.

---

## 6. CSR for Graph Adjacency in Segments

### Evidence

**Memory Efficiency**
- CSR total memory: sizeof(int) × (V+E) — near-optimal for sparse graphs
- Cache-friendly sequential access for neighbor iteration
- Source: https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf

**Performance Characteristics**
- CSR excellent for static traversal, poor for dynamic insertion/deletion
- Packed CSR (PCSR): only 2x slower than CSR on traversals but supports dynamic updates
- Source: https://itshelenxu.github.io/files/papers/pcsr.pdf

**Why CSR works for T4DX**
- Segments are **immutable** — CSR's weakness (dynamic updates) doesn't apply
- Edge mutations go through MemTable overlay → consumed during compaction → new CSR built
- Typical graph density: <100 edges per node (sparse) — ideal for CSR
- Traversal is the primary graph operation (1-3 hops for context retrieval)

### Assessment
**Ideal match.** Immutable segments eliminate CSR's dynamic update weakness. Edge mutations buffered in MemTable. Memory efficiency and cache performance are optimal for the sparse, read-heavy graph access pattern.

---

## Risk Matrix

| Decision | Risk Level | Mitigation |
|----------|-----------|------------|
| Custom storage engine | Medium | Protocol adapters provide drop-in compatibility; dual-write migration path |
| Spiking adapters on frozen LLM | Medium | STE validated in SpikingBrain (2025), S²TDPT (2025); fallback to pure QLoRA |
| LSM compaction = consolidation | Low | CLS theory well-established; compaction semantics are additive to standard LSM |
| QLoRA rank 16 | Low | Proven at NeurIPS 2023; rank search (P3-11) will validate optimal rank |
| CSR for segment graph | Low | Standard format; immutable segments avoid dynamic update weakness |
| Overlay pattern | Low | Standard LSM MemTable pattern; biologically plausible |

---

## References

### Memory Consolidation & CLS
- McClelland, McNaughton, O'Reilly. "Why There Are Complementary Learning Systems in the Hippocampus and Neocortex." Psychological Review (1995). https://pubmed.ncbi.nlm.nih.gov/22141588/
- "Both slow wave and rapid eye movement sleep contribute to emotional memory consolidation." Communications Biology (2025). https://www.nature.com/articles/s42003-025-07868-5
- "Language Models Need Sleep: Learning to Self-Modify." OpenReview (2025). https://openreview.net/pdf/05bbb74851e965f5199f45f83937d1c396f048c8.pdf
- Frey & Morris. "Synaptic tagging and long-term potentiation." Nature (1997).

### Storage & Databases
- "Rethinking The Compaction Policies in LSM-trees." ACM SIGMOD (2025). https://dl.acm.org/doi/10.1145/3725344
- "A Practical Guide for Choosing a Vector Database." SuperLinked. https://superlinked.com/vectorhub/articles/choosing-vdb
- "Embedded databases: DuckDB, Kùzu, LanceDB." The Data Quarry. https://thedataquarry.com/blog/embedded-db-1/
- "Faiss vs HNSWlib on Vector Search." Zilliz. https://zilliz.com/blog/faiss-vs-hnswlib-choosing-the-right-tool-for-vector-search
- "Choosing a vector database for ANN search at Reddit." Milvus. https://milvus.io/blog/choosing-a-vector-database-for-ann-search-at-reddit.md

### Spiking Neural Networks
- "SpikingBrain: Spiking Brain-inspired Large Models." arXiv 2509.05276 (2025). https://arxiv.org/html/2509.05276v1
- Zhu et al. "SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks." arXiv 2302.13939 (2023). https://arxiv.org/abs/2302.13939
- "BrainTransformers: SNN-LLM." arXiv 2410.14687 (2024). https://arxiv.org/html/2410.14687v2
- "Attention via Synaptic Plasticity is All You Need." arXiv 2511.14691 (2025). https://arxiv.org/html/2511.14691

### Three-Factor Learning
- Gerstner et al. "Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules." Frontiers in Neural Circuits (2015). https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/
- "Eligibility Traces and Plasticity on Behavioral Time Scales." Frontiers in Neural Circuits (2018). https://pmc.ncbi.nlm.nih.gov/articles/PMC6079224/

### QLoRA & Adapters
- Dettmers et al. "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS (2023). https://arxiv.org/abs/2305.14314
- "Efficiently Deploying LoRA Adapters." Inferless. https://www.inferless.com/learn/how-to-serve-multi-lora-adapters

### Graph Storage
- "Compressed Sparse Row Format for Representing Graphs." USENIX ;login: (2020). https://www.usenix.org/system/files/login/articles/login_winter20_16_kelly.pdf
- "Packed Compressed Sparse Row: A Dynamic Graph Representation." https://itshelenxu.github.io/files/papers/pcsr.pdf
