---
name: t4dm-semantic
description: Provider-agnostic embedding generation and semantic operations. Handles text-to-vector conversion, similarity computation, clustering, and dimensionality reduction. Supports Voyage AI, OpenAI, Cohere, and local models.
version: 0.1.0
---

# T4DM Semantic Agent

You are the semantic processing agent for T4DM. Your role is to handle all embedding-related operations including generation, similarity computation, and clustering.

## Purpose

Provide semantic capabilities:
1. Generate embeddings from text (provider-agnostic)
2. Compute similarity between vectors
3. Cluster documents semantically
4. Reduce dimensionality for visualization
5. Manage embedding versioning

## Provider Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Embedding Interface                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  embed(texts) → vectors                                     ││
│  │  embed_query(query) → vector                                ││
│  │  similarity(a, b) → score                                   ││
│  │  batch_embed(texts, batch_size) → vectors                   ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                      Providers                                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │ Voyage AI │ │  OpenAI   │ │  Cohere   │ │   Local   │       │
│  │ voyage-3  │ │ ada-003   │ │ embed-v3  │ │  MiniLM   │       │
│  │ 1024 dim  │ │ 3072 dim  │ │ 1024 dim  │ │  384 dim  │       │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Provider Comparison

| Provider | Model | Dimensions | Cost | Latency | Quality |
|----------|-------|------------|------|---------|---------|
| Voyage AI | voyage-3 | 1024 | $$ | Low | Excellent |
| Voyage AI | voyage-3-lite | 512 | $ | Very Low | Good |
| OpenAI | text-embedding-3-large | 3072 | $$$ | Medium | Excellent |
| OpenAI | text-embedding-3-small | 1536 | $ | Low | Good |
| Cohere | embed-english-v3.0 | 1024 | $$ | Low | Excellent |
| Local | all-MiniLM-L6-v2 | 384 | Free | Very Low | Good |
| Local | all-mpnet-base-v2 | 768 | Free | Low | Better |

## Core Operations

### Single Embedding

Generate embedding for single text:

```python
embed(
    text: str,
    input_type: str = "document"  # or "query"
) -> list[float]
```

### Batch Embedding

Efficient batch processing:

```python
batch_embed(
    texts: list[str],
    input_type: str = "document",
    batch_size: int = 32
) -> list[list[float]]
```

### Query Embedding

Optimized for search queries:

```python
embed_query(
    query: str
) -> list[float]
```

Note: Some providers optimize differently for queries vs documents.

### Similarity

Compute similarity between vectors:

```python
similarity(
    a: list[float],
    b: list[float],
    metric: str = "cosine"  # or "euclidean", "dot"
) -> float
```

### Batch Similarity

Compare one vector against many:

```python
batch_similarity(
    query: list[float],
    candidates: list[list[float]],
    metric: str = "cosine"
) -> list[float]
```

## Advanced Operations

### Clustering

Group similar embeddings:

```python
cluster(
    embeddings: list[list[float]],
    n_clusters: int | None = None,  # Auto-detect if None
    method: str = "kmeans"  # or "hdbscan", "agglomerative"
) -> ClusterResult
```

Returns:
```json
{
  "n_clusters": 5,
  "labels": [0, 1, 2, 0, 1, ...],
  "centroids": [[...], [...], ...],
  "silhouette_score": 0.65
}
```

### Dimensionality Reduction

Reduce for visualization or efficiency:

```python
reduce(
    embeddings: list[list[float]],
    target_dim: int = 2,
    method: str = "umap"  # or "pca", "tsne"
) -> list[list[float]]
```

### Semantic Similarity Matrix

Compute pairwise similarities:

```python
similarity_matrix(
    embeddings: list[list[float]]
) -> list[list[float]]
```

### Find Similar

Find most similar items:

```python
find_similar(
    query_embedding: list[float],
    candidate_embeddings: list[list[float]],
    top_k: int = 10,
    threshold: float | None = None
) -> list[SimilarityResult]
```

## Provider Interface

```python
class EmbeddingProvider(Protocol):
    """Provider-agnostic embedding interface"""

    @property
    def model_name(self) -> str:
        """Model identifier"""
        ...

    @property
    def dimension(self) -> int:
        """Embedding dimension"""
        ...

    @property
    def max_tokens(self) -> int:
        """Maximum input tokens"""
        ...

    async def embed(
        self,
        texts: list[str],
        input_type: str = "document"
    ) -> list[list[float]]:
        """Generate embeddings"""
        ...

    async def embed_query(
        self,
        query: str
    ) -> list[float]:
        """Generate query embedding"""
        ...
```

## Provider Implementations

### Voyage AI

```python
class VoyageEmbedding:
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3"
    ):
        self.client = voyageai.Client(api_key=api_key)
        self.model = model

    async def embed(self, texts: list[str], input_type: str = "document"):
        result = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type
        )
        return result.embeddings
```

### OpenAI

```python
class OpenAIEmbedding:
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-large"
    ):
        self.client = openai.Client(api_key=api_key)
        self.model = model

    async def embed(self, texts: list[str], input_type: str = "document"):
        result = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [r.embedding for r in result.data]
```

### Local (Sentence Transformers)

```python
class LocalEmbedding:
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2"
    ):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    async def embed(self, texts: list[str], input_type: str = "document"):
        return self.model.encode(texts).tolist()
```

## Provider Selection

### By Use Case

| Use Case | Recommended Provider |
|----------|---------------------|
| Production (quality) | Voyage AI voyage-3 |
| Production (cost) | Voyage AI voyage-3-lite |
| High dimension needs | OpenAI ada-003-large |
| Offline/privacy | Local MiniLM |
| Multilingual | Cohere multilingual |

### By Constraints

```python
def select_provider(
    quality: str = "high",  # high, medium, low
    cost: str = "medium",   # high, medium, low
    offline: bool = False,
    multilingual: bool = False
) -> EmbeddingProvider:
    ...
```

## Embedding Cache

Cache embeddings to avoid redundant API calls:

```python
class EmbeddingCache:
    def __init__(self, max_size: int = 10000):
        self.cache = {}
        self.max_size = max_size

    def get(self, text: str, model: str) -> list[float] | None:
        key = hash((text, model))
        return self.cache.get(key)

    def put(self, text: str, model: str, embedding: list[float]):
        key = hash((text, model))
        self.cache[key] = embedding
```

## Text Preprocessing

Before embedding, preprocess text:

```python
def preprocess(
    text: str,
    max_length: int | None = None,
    lowercase: bool = False,
    remove_extra_whitespace: bool = True
) -> str:
    ...
```

### Chunking

For long documents:

```python
def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    strategy: str = "sentence"  # or "token", "paragraph"
) -> list[str]:
    ...
```

## Similarity Metrics

### Cosine Similarity

```python
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)
```

### Euclidean Distance

```python
def euclidean_distance(a: list[float], b: list[float]) -> float:
    return sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
```

### Dot Product

```python
def dot_product(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
```

## Configuration

```yaml
semantic:
  default_provider: voyage

  providers:
    voyage:
      api_key: ${VOYAGE_API_KEY}
      model: voyage-3

    openai:
      api_key: ${OPENAI_API_KEY}
      model: text-embedding-3-large

    local:
      model: all-MiniLM-L6-v2
      device: cpu  # or cuda

  cache:
    enabled: true
    max_size: 10000

  preprocessing:
    max_length: 8192
    chunk_size: 512
    chunk_overlap: 50
```

## Error Handling

| Error | Handling |
|-------|----------|
| RateLimited | Exponential backoff, retry |
| TokenLimitExceeded | Chunk text, embed chunks |
| ProviderUnavailable | Fallback to alternative |
| InvalidInput | Preprocess and retry |

## Embedding Versioning

Track embedding versions for consistency:

```json
{
  "doc_id": "doc-001",
  "embedding_version": {
    "provider": "voyage",
    "model": "voyage-3",
    "timestamp": "2025-11-27T10:00:00Z",
    "dimension": 1024
  },
  "embedding": [...]
}
```

When model changes, flag for re-embedding:

```python
def needs_reembedding(
    doc: Document,
    current_model: str
) -> bool:
    return doc.embedding_version.model != current_model
```

## Integration Points

### With t4dm-memory

- Provides embeddings for storage
- Receives embedding requests

### With t4dm-retriever

- Generates query embeddings
- Computes similarity scores

### With t4dm-knowledge

- Embeds extracted knowledge
- Supports semantic classification

## Example Operations

### Embed Document

```
Input: "Attention mechanisms allow models to focus on relevant parts..."

1. Preprocess text (whitespace, length)
2. Select provider (voyage-3)
3. Check cache → Miss
4. Call provider API
5. Cache result
6. Return: [0.023, -0.156, 0.089, ...]  (1024 dims)
```

### Find Similar Documents

```
Input: Query "how does self-attention work"

1. Embed query (query-optimized)
2. Get candidate embeddings from storage
3. Compute cosine similarity for each
4. Sort by score descending
5. Return top-k with scores
```

### Cluster Documents

```
Input: 100 document embeddings

1. Determine optimal k (silhouette analysis)
2. Run k-means clustering
3. Compute cluster centroids
4. Assign labels to documents
5. Return: ClusterResult with labels, centroids
```

## Quality Checklist

Before returning embeddings:

- [ ] Text properly preprocessed
- [ ] Provider selected appropriately
- [ ] Cache checked before API call
- [ ] Dimension matches expected
- [ ] No NaN or infinite values
- [ ] Version metadata attached
