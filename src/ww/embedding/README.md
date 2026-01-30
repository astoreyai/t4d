# Embedding Module

**9 files | ~2,500 lines | Centrality: 9**

The embedding module provides a sophisticated, composable system for generating, caching, and adapting vector representations using BGE-M3 (1024-dimensional multilingual embeddings).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          EMBEDDING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│  Input Text                                                              │
│      ↓                                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  BGE-M3 Provider (1024-dim)                                       │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │   │
│  │  │  Dense Vectors │  │ Sparse Vectors │  │   TTL Cache        │  │   │
│  │  │  (semantic)    │  │ (lexical)      │  │   O(log n) evict   │  │   │
│  │  └────────────────┘  └────────────────┘  └────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│      ↓                                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Adaptation Layer (Optional)                                      │   │
│  │  ┌─────────┐  ┌───────────┐  ┌──────────┐  ┌─────────────────┐  │   │
│  │  │  LoRA   │  │Contrastive│  │ Ensemble │  │    Modulated    │  │   │
│  │  │ rank=16 │  │  InfoNCE  │  │  Voting  │  │  NT-dependent   │  │   │
│  │  └─────────┘  └───────────┘  └──────────┘  └─────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│      ↓                                                                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Query-Memory Separation (Asymmetric)                             │   │
│  │  ┌─────────────────────┐  ┌─────────────────────────────────────┐│   │
│  │  │ Query Projection    │  │ Memory Projection                   ││   │
│  │  │ (CA1-inspired)      │  │ (CA3-inspired)                      ││   │
│  │  └─────────────────────┘  └─────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────────┘   │
│      ↓                                                                   │
│  Vector Database (Qdrant)                                                │
└─────────────────────────────────────────────────────────────────────────┘
```

## File Organization

| File | Lines | Purpose |
|------|-------|---------|
| `bge_m3.py` | ~550 | BGE-M3 embedding provider with TTL caching |
| `adapter.py` | ~350 | Abstract adapter + factory functions |
| `lora_adapter.py` | ~700 | Low-rank adaptation for task-specific tuning |
| `contrastive_trainer.py` | ~650 | Learnable projection with InfoNCE loss |
| `modulated.py` | ~300 | Neuromodulator-dependent transformation |
| `ensemble.py` | ~350 | Multi-adapter voting with health-aware weighting |
| `query_memory_separation.py` | ~280 | Asymmetric query/memory projections |
| `semantic_mock.py` | ~250 | Semantic-aware mock for testing |
| `__init__.py` | ~100 | 40+ public exports |

## Core Components

### BGE-M3 Provider

```python
from ww.embedding import BGEM3Embedding

provider = BGEM3Embedding()

# Single query (with instruction prefix)
query_emb = await provider.embed_query("What is memory?")

# Batch documents
doc_embs = await provider.embed(["Memory is...", "Learning is..."])

# Hybrid: dense + sparse vectors
dense, sparse = await provider.embed_hybrid(texts)
```

**Features**:
- Query instruction prefix: "Represent this sentence for searching relevant passages: "
- Sparse vector extraction for hybrid search
- FP16 inference on CUDA
- Fallback to sentence-transformers

### TTL Cache

O(log n) heap-based eviction with thread safety:

```python
# Cache configuration (default)
max_size: int = 1000
ttl_seconds: int = 3600  # 1 hour

# Automatic cache key: MD5(text) -> 32 chars
```

### Adapter Interface

```python
from ww.embedding import EmbeddingAdapter, create_adapter, EmbeddingBackend

class EmbeddingAdapter(ABC):
    async def embed_query(query: str) -> list[float]
    async def embed(texts: list[str]) -> list[list[float]]
    async def embed_query_np(query: str) -> np.ndarray
    async def embed_np(texts: list[str]) -> np.ndarray

    def is_healthy() -> bool  # Error rate < 10%
    @property
    def stats -> EmbeddingStats

# Create default adapter
adapter = create_adapter(backend=EmbeddingBackend.BGE_M3)
```

## Adaptation Strategies

### LoRA (Low-Rank Adaptation)

Fine-tune embeddings without modifying base model:

```python
from ww.embedding import LoRAEmbeddingAdapter, LoRAConfig, RetrievalOutcome

config = LoRAConfig(
    rank=16,              # Low-rank dimension
    alpha=16.0,           # Scaling factor
    use_asymmetric=True,  # Separate query/memory adapters
    ewc_enabled=False     # Elastic Weight Consolidation
)

adapter = LoRAEmbeddingAdapter(base_adapter, config)

# Record outcome for training
outcome = RetrievalOutcome(
    query_embedding=query_emb,
    positive_embeddings=[pos1, pos2],
    negative_embeddings=[neg1]
)
adapter.record_outcome(outcome)

# Train on accumulated outcomes
losses = adapter.train_on_outcomes(epochs=10)
```

**Architecture**:
```
adapted = x + (α/r) × B(dropout(A(x)))
# A: 1024 → r (down-projection)
# B: r → 1024 (up-projection, initialized to zero)
```

### Contrastive Trainer

Learn projection with InfoNCE loss:

```python
from ww.embedding import ContrastiveAdapter

adapter = ContrastiveAdapter(input_dim=1024, output_dim=256)

# Training step
loss, accuracy = adapter.update(
    anchors=query_embeddings,
    positives=positive_embeddings,
    negatives=negative_embeddings,
    temporal_sequence=episode_sequence  # Optional
)

# Inference
projected = adapter.forward(embeddings)
```

**Loss Function**:
```
L = -log(exp(sim(a,p)/τ) / Σ exp(sim(a,n)/τ))
```

### Ensemble Voting

Health-aware multi-adapter combination:

```python
from ww.embedding import EnsembleEmbeddingAdapter, EnsembleStrategy

ensemble = EnsembleEmbeddingAdapter(
    adapters=[adapter1, adapter2, adapter3],
    strategy=EnsembleStrategy.WEIGHTED_MEAN,
    fallback_on_failure=True
)

# Embedding uses weighted combination
embedding = await ensemble.embed_query("query")

# Check per-adapter stats
stats = ensemble.get_ensemble_stats()
```

**Strategies**:
- `MEAN`: Simple average
- `WEIGHTED_MEAN`: Health-weighted average
- `CONCAT`: Concatenate (increases dimension)
- `VOTING`: Weighted voting on similarity
- `BEST`: Use healthiest adapter

### State-Dependent Modulation

Transform embeddings based on neuromodulator state:

```python
from ww.embedding import ModulatedEmbeddingAdapter, NeuromodulatorState

modulated = ModulatedEmbeddingAdapter(base_adapter)

# Encoding mode (high ACh)
modulated.set_state(NeuromodulatorState(
    acetylcholine=0.8,  # High = encoding mode
    dopamine=0.6,       # Salience amplification
    norepinephrine=0.5, # Exploration noise
    serotonin=0.4
))
encoding_emb = await modulated.embed_query("new memory")

# Retrieval mode (low ACh)
modulated.set_state(NeuromodulatorState(acetylcholine=0.2))
retrieval_emb = await modulated.embed_query("same query")
# Different embedding based on cognitive state!
```

**Modulation Pipeline**:
1. ACh gating (encoding vs retrieval dimensions)
2. DA amplification (salient dimensions)
3. NE noise (exploration)
4. Sparsification (top-k)
5. L2 normalization

### Query-Memory Separation

Asymmetric projections inspired by hippocampal CA1/CA3:

```python
from ww.embedding import QueryMemorySeparator

separator = QueryMemorySeparator(embedding_dim=1024, hidden_dim=256)

# Different projections for query vs memory
query_proj = separator.project_query(query_embedding)
memory_proj = separator.project_memory(memory_embedding)

# Training with triplet loss
loss = separator.train_step(query, positive, negatives, margin=0.2)
```

## Configuration

```python
# Environment variables / Settings
embedding_model: str = "BAAI/bge-m3"
embedding_device: str = "cuda:0"
embedding_use_fp16: bool = True
embedding_batch_size: int = 32
embedding_max_length: int = 512
embedding_dimension: int = 1024
embedding_cache_size: int = 10000
embedding_cache_ttl: int = 3600
```

## Usage Patterns

### Basic Retrieval Pipeline

```python
from ww.embedding import create_adapter, register_adapter, get_adapter

# Initialize and register
adapter = create_adapter()
register_adapter(adapter, name="default")

# Use anywhere
adapter = get_adapter("default")
query_emb = await adapter.embed_query("search query")
doc_embs = await adapter.embed(documents)

# Compute similarity
from ww.embedding import cosine_similarity
scores = [cosine_similarity(query_emb, doc) for doc in doc_embs]
```

### Adaptive Retrieval with Learning

```python
from ww.embedding import create_lora_adapter, RetrievalOutcome

# Create LoRA-adapted provider
adapted = create_lora_adapter(rank=32, use_asymmetric=True)

# Retrieval loop
for query, feedback in retrieval_stream:
    # Get embeddings
    query_emb = await adapted.embed_query(query)
    results = await search(query_emb)

    # Record outcome based on user feedback
    if feedback.helpful:
        outcome = RetrievalOutcome(
            query_embedding=query_emb,
            positive_embeddings=[results[feedback.selected].embedding],
            negative_embeddings=[r.embedding for r in results if not r.selected]
        )
        adapted.record_outcome(outcome)

# Periodic training
adapted.train_on_outcomes(epochs=5)
adapted.save("checkpoints/lora_latest.pt")
```

## Testing

### Mock Adapters

```python
from ww.embedding import MockEmbeddingAdapter, SemanticMockAdapter

# Deterministic mock (hash-seeded)
mock = MockEmbeddingAdapter(dimension=1024)

# Semantic-aware mock (preserves similarity relationships)
semantic_mock = SemanticMockAdapter(dimension=128, seed=42)
# "code" and "programming" will have high similarity
```

### Test Commands

```bash
# Run embedding tests
pytest tests/embedding/ -v

# With coverage
pytest tests/embedding/ --cov=ww.embedding --cov-report=term-missing

# Performance benchmarks
pytest tests/embedding/ -v -m benchmark
```

## Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| Single embedding (GPU) | 50-100ms | FP16, includes tokenization |
| Batch 32 (GPU) | 80-150ms | Amortized model overhead |
| Cache lookup | O(1) | MD5 key, dict lookup |
| Cache eviction | O(log n) | Heap-based TTL |
| LoRA forward | ~1-2ms | rank=16, on CPU |
| Contrastive project | ~0.5ms | 1024→256 MLP |

## Public API

```python
# Core
BGEM3Embedding, EmbeddingAdapter, create_adapter
EmbeddingBackend, EmbeddingStats

# Adapters
BGEM3Adapter, MockEmbeddingAdapter, CachedEmbeddingAdapter
LoRAEmbeddingAdapter, LoRAConfig, RetrievalOutcome
ContrastiveAdapter
EnsembleEmbeddingAdapter, EnsembleStrategy
ModulatedEmbeddingAdapter, NeuromodulatorState
QueryMemorySeparator, SemanticMockAdapter

# Registry
register_adapter, get_adapter, clear_adapters

# Utilities
cosine_similarity, euclidean_distance, normalize_embedding

# Factory functions
create_lora_adapter, create_adapted_provider
create_ensemble_adapter, create_modulated_adapter
create_contrastive_adapter, create_semantic_mock
```

## Design Patterns

| Pattern | Example | Purpose |
|---------|---------|---------|
| Lazy Loading | `BGEM3Embedding._ensure_initialized()` | Defer model loading |
| Adapter | `EmbeddingAdapter` ABC | Swap providers |
| Factory | `create_adapter()` | Configurable creation |
| Strategy | `EnsembleStrategy` enum | Pluggable algorithms |
| Registry | Global `_adapters` dict | Singleton access |
| Residual | LoRA, QueryMemorySeparator | Stability |

## Integration Points

### With Memory
- `EpisodicMemory.store()` uses `embed()` for content vectors
- `UnifiedMemory.search()` uses `embed_query()` for query vectors

### With Storage
- Dense vectors → Qdrant collection
- Sparse vectors → Qdrant sparse index (hybrid search)

### With Learning
- `RetrievalOutcome` → LoRA training
- `NeuromodulatorState` → Modulated embeddings
