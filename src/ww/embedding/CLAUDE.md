# Embedding Module
**Path**: `/mnt/projects/t4d/t4dm/src/ww/embedding/`

## What
Composable vector embedding pipeline built on BGE-M3 (1024-dim multilingual). Provides base embedding generation, TTL caching, and five adaptation strategies: LoRA fine-tuning, contrastive projection (InfoNCE), ensemble voting, neuromodulator-dependent modulation, and asymmetric query-memory separation.

## How
- **BGEM3Embedding**: Core provider with FP16 CUDA inference, query instruction prefix, hybrid dense+sparse vectors, heap-based TTL cache (O(log n) eviction)
- **EmbeddingAdapter**: Abstract interface with health monitoring (error rate < 10%), stats tracking, numpy convenience methods
- **LoRAAdapter**: Low-rank adaptation (`adapted = x + (alpha/r) * B(dropout(A(x)))`) with asymmetric query/memory adapters, trains on RetrievalOutcome feedback
- **ContrastiveAdapter**: InfoNCE loss projection (1024->256), supports temporal sequence training
- **EnsembleAdapter**: Multi-adapter combination with 5 strategies (mean, weighted_mean, concat, voting, best), health-aware fallback
- **ModulatedAdapter**: NT-state-dependent transformation (ACh gating, DA amplification, NE noise, sparsification, L2 norm)
- **QueryMemorySeparator**: Asymmetric projections inspired by hippocampal CA1 (query) / CA3 (memory), triplet loss training
- **SemanticMockAdapter**: Testing mock preserving semantic similarity relationships between concept clusters

## Why
Raw embeddings from BGE-M3 are general-purpose. The adaptation layers personalize representations to the user's memory domain, cognitive state, and retrieval feedback without modifying the base model. Centrality: 9.

## Key Files
| File | Purpose |
|------|---------|
| `bge_m3.py` | BGE-M3 provider with TTL cache (~550 lines) |
| `adapter.py` | Abstract adapter, factory, registry (~350 lines) |
| `lora_adapter.py` | LoRA fine-tuning from retrieval outcomes (~700 lines) |
| `contrastive_trainer.py` | InfoNCE projection learning (~650 lines) |
| `modulated.py` | NT-state-dependent embedding transformation (~300 lines) |
| `ensemble.py` | Multi-adapter voting with health awareness (~350 lines) |
| `query_memory_separation.py` | Asymmetric CA1/CA3-inspired projections (~280 lines) |
| `semantic_mock.py` | Semantic-aware testing mock (~250 lines) |

## Data Flow
```
Text -> BGEM3Embedding (dense+sparse, cached)
    -> [Optional] Adaptation layer (LoRA / Contrastive / Modulated / Ensemble)
    -> [Optional] QueryMemorySeparator (asymmetric query vs memory projection)
    -> Qdrant (vector storage)
```

## Integration Points
- **core**: Implements `EmbeddingProvider` protocol
- **memory**: `EpisodicMemory.store()` uses `embed()`, search uses `embed_query()`
- **storage**: Dense vectors -> Qdrant collection, sparse vectors -> Qdrant sparse index
- **learning**: `RetrievalOutcome` feeds LoRA training, `NeuromodulatorState` drives modulation
- **bridges**: NCA bridge NT state feeds ModulatedAdapter

## Learning Modalities
- **LoRA**: Online adaptation from retrieval feedback (positive/negative examples)
- **Contrastive**: InfoNCE loss with optional temporal sequence signal
- **Modulated**: State-dependent transformation (no explicit training, driven by NT state)
