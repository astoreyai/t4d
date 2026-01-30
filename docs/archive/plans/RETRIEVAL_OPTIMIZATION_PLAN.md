# World Weaver Retrieval Optimization Plan

**Version**: 2.0 | **Status**: Complete | **Target**: Best-in-class self-contained retrieval + adaptive learning

## Executive Summary

Transform WW from basic vector-only retrieval to a state-of-the-art hybrid retrieval system with adaptive learning, incorporating insights from the ACE (Agentic Context Engineering) framework:

**Retrieval Pipeline**:
- **Hybrid Search**: Dense + Sparse vectors (BGE-M3 native)
- **Semantic Chunking**: Sentence-boundary aware, overlap-based
- **Local NER**: GLiNER zero-shot extraction (no external APIs)
- **Cross-Encoder Reranking**: BGE-reranker for top-k refinement
- **Retrieval Fusion**: RRF-based multi-signal combination

**Adaptive Learning** (ACE-Inspired):
- **Skillbook System**: Evolving playbook with helpful/harmful counters
- **Three-Role Architecture**: Agent → Reflector → SkillManager
- **Delta Updates**: Incremental changes, no context collapse
- **Semantic Deduplication**: SemHash for efficient consolidation
- **Async Learning**: Parallel reflection, serial updates

---

## Current State Analysis

### What's Working
| Component | Status | Notes |
|-----------|--------|-------|
| BGE-M3 Dense | ✅ Good | 1024-dim, FP16, cached |
| TTL Cache | ✅ Good | MD5 keys, LRU eviction |
| FSRS Decay | ✅ Good | Proper retrievability calc |
| Session Isolation | ✅ Good | Per-session namespacing |
| Dual Store | ✅ Good | Qdrant + Neo4j with Saga |
| Procedural Memory | ✅ Basic | Skills with success_rate |

### Critical Gaps
| Component | Issue | Impact |
|-----------|-------|--------|
| **Chunking** | None - truncation at 512 tokens | Lost information |
| **Sparse Search** | Disabled despite BGE-M3 support | No exact matching |
| **NER** | LLM-only (OpenAI required) | Not self-contained |
| **Reranking** | None | Suboptimal ranking |
| **Hybrid Fusion** | Single-signal only | Miss complementary matches |
| **Adaptive Learning** | No reflection/curation cycle | Skills don't improve |
| **Deduplication** | None | Redundant memories |

---

## Research Foundation

### ACE Paper (Stanford/SambaNova)
Source: [arxiv.org/abs/2510.04618](https://arxiv.org/abs/2510.04618)

**Key Insights**:
1. **Contexts as Evolving Playbooks**: Not static prompts, but accumulating strategies
2. **Brevity Bias Problem**: LLM summarization loses domain-specific details
3. **Context Collapse**: Monolithic rewrites erode information (18k → 122 tokens = 10% accuracy drop)
4. **Grow-and-Refine**: Incremental delta updates preserve history
5. **LLMs prefer long context**: Let model filter relevance, don't pre-compress

### kayba-ai Implementation
Source: [github.com/kayba-ai/agentic-context-engine](https://github.com/kayba-ai/agentic-context-engine)

**Production Patterns**:
1. **Skillbook Structure**: Skills with `helpful/harmful/neutral` counters
2. **Three Roles**: Agent (execute) → Reflector (analyze) → SkillManager (curate)
3. **Update Operations**: ADD, UPDATE, TAG, REMOVE with validation
4. **Quality Gates**: Atomicity scoring (reject <70%), semantic dedup
5. **Async Learning**: Parallel reflectors, serial skill manager
6. **Thread Safety**: Lock-free reads, locked writes

### SemHash (MinishLab)
Source: [github.com/MinishLab/semhash](https://github.com/MinishLab/semhash)

**Fast Deduplication**:
- Model2Vec (8M params) + ANN for O(log n) similarity
- 1.8M records in ~83s
- Configurable thresholds (default 0.95)
- Cross-dataset dedup for train/test leakage

---

## Technology Decisions

### GLiNER vs Ollama for NER

**Decision: GLiNER (Option B)**

| Criterion | GLiNER | Ollama | Winner |
|-----------|--------|--------|--------|
| **Speed** | 50-200ms/text | 1-5s/text | GLiNER |
| **Memory** | ~500MB-2GB | ~4-6GB | GLiNER |
| **Determinism** | Yes (testable) | No | GLiNER |
| **Cold Start** | ~5s model load | Daemon always running | GLiNER |
| **Flexibility** | Zero-shot any type | Full generation | Ollama |
| **Self-Contained** | Yes | Requires daemon | GLiNER |
| **Batch Support** | Native | Sequential | GLiNER |

**Rationale**: NER is extraction, not generation. GLiNER is 10-50x faster, deterministic (testable), and purpose-built for the task.

### Deduplication: SemHash vs Custom

**Decision: SemHash-Inspired Custom Implementation**

- Use Model2Vec (lightweight) for dedup embeddings (separate from BGE-M3)
- ANN backend for scalability
- Integrate with Skillbook for skill deduplication
- Threshold: 0.90 for skills (more aggressive), 0.95 for episodes

---

## Architecture

### Full System Architecture

```
                              ┌─────────────────────────────────────┐
                              │         ADAPTIVE LEARNING           │
                              │                                     │
    Task Execution ──────────►│  Agent ─► Reflector ─► SkillManager│
                              │    │         │              │       │
                              │    ▼         ▼              ▼       │
                              │ Episode   Lessons      Skillbook   │
                              │                        (Playbook)   │
                              └───────────────┬─────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INGESTION PIPELINE                              │
│                                                                              │
│    Content Input                                                             │
│         │                                                                    │
│         ▼                                                                    │
│    ┌────────────────────────┐                                               │
│    │   Semantic Chunker     │                                               │
│    │  (sentence boundaries) │                                               │
│    │  (512 tok, 64 overlap) │                                               │
│    └───────────┬────────────┘                                               │
│                │                                                             │
│    ┌───────────┼───────────────────────┐                                    │
│    │           │                       │                                    │
│    ▼           ▼                       ▼                                    │
│ ┌──────────┐ ┌──────────┐ ┌────────────────────┐                           │
│ │  Dense   │ │  Sparse  │ │      GLiNER        │                           │
│ │  BGE-M3  │ │  BGE-M3  │ │    Local NER       │                           │
│ │ (1024d)  │ │(lexical) │ │ (zero-shot types)  │                           │
│ └────┬─────┘ └────┬─────┘ └─────────┬──────────┘                           │
│      │            │                 │                                       │
│      ▼            ▼                 ▼                                       │
│ ┌─────────────────────────────────────────────────────┐                    │
│ │                 STORAGE LAYER                        │                    │
│ │  ┌──────────────┐           ┌──────────────┐        │                    │
│ │  │    Qdrant    │           │    Neo4j     │        │                    │
│ │  │  • dense_vec │           │  • entities  │        │                    │
│ │  │  • sparse_vec│           │  • relations │        │                    │
│ │  │  • chunks    │           │  • episodes  │        │                    │
│ │  │  • dedup_vec │           │  • skills    │        │                    │
│ │  └──────────────┘           └──────────────┘        │                    │
│ └─────────────────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ Query Time
┌─────────────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL PIPELINE                                  │
│                                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Dense    │  │   Sparse   │  │   Graph    │  │  Skillbook │            │
│  │   Search   │  │   Search   │  │  Traverse  │  │   Lookup   │            │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘            │
│        │               │               │               │                    │
│        └───────────────┴───────────────┴───────────────┘                    │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                               │
│                    │      RRF Fusion       │                               │
│                    │   (rank-based merge)  │                               │
│                    └───────────┬───────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                               │
│                    │    Cross-Encoder      │                               │
│                    │   Reranker (top-20)   │                               │
│                    │   BGE-reranker-v2-m3  │                               │
│                    └───────────┬───────────┘                               │
│                                │                                            │
└────────────────────────────────┼────────────────────────────────────────────┘
                                 │
                                 ▼
                          Final Results (top-k)
```

### ACE Three-Role Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE LEARNING LOOP                            │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         AGENT (Generator)                         │   │
│  │  • Executes tasks using current skillbook context                │   │
│  │  • Records actions, outcomes, tool usage                         │   │
│  │  • Produces episode with success/failure signal                  │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                       REFLECTOR (Analyzer)                        │   │
│  │  • Diagnoses execution: what worked, what failed, why            │   │
│  │  • Extracts atomic lessons (atomicity score ≥70%)                │   │
│  │  • Tags strategies: helpful / harmful / neutral                  │   │
│  │  • Rejects vague directives ("be careful with...")               │   │
│  │  • Produces structured delta update                              │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    SKILL MANAGER (Curator)                        │   │
│  │  • Validates quality gates before applying updates               │   │
│  │  • Semantic deduplication (threshold 0.90)                       │   │
│  │  • Applies operations: ADD, UPDATE, TAG, REMOVE                  │   │
│  │  • Maintains counter integrity (helpful/harmful/neutral)         │   │
│  │  • Thread-safe: lock-free reads, locked writes                   │   │
│  └────────────────────────────┬─────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                         SKILLBOOK                                 │   │
│  │  • Organized by sections (domain, task type)                     │   │
│  │  • Skills with counters: helpful, harmful, neutral               │   │
│  │  • Embeddings for semantic dedup                                 │   │
│  │  • TOON format for LLM consumption                               │   │
│  │  • Status: active / invalid (soft delete)                        │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    ASYNC LEARNING PIPELINE                        │   │
│  │                                                                   │   │
│  │  Main Thread ──► Submit Task ──► Return Immediately              │   │
│  │                       │                                           │   │
│  │                       ▼                                           │   │
│  │  ThreadPool ────► Parallel Reflectors (N workers)                │   │
│  │                       │                                           │   │
│  │                       ▼                                           │   │
│  │  Queue ─────────► Serial SkillManager (1 worker)                 │   │
│  │                       │                                           │   │
│  │                       ▼                                           │   │
│  │  Skillbook ─────► Updated (eventual consistency)                 │   │
│  │                                                                   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Hybrid Search (BGE-M3 Sparse) [4 hrs]

**Goal**: Enable native hybrid search without new models.

**Files Modified**:
- `src/ww/embedding/bge_m3.py` - Enable sparse vectors
- `src/ww/storage/qdrant_store.py` - Sparse vector storage + search
- `src/ww/memory/episodic.py` - Hybrid recall integration

**Key Changes**:

```python
# bge_m3.py - Enable sparse vectors
async def embed_hybrid(self, texts: list[str]) -> tuple[list[list[float]], list[dict]]:
    """Return (dense_vecs, sparse_dicts) for hybrid search."""
    result = self._model.encode(
        texts,
        return_dense=True,
        return_sparse=True,  # Enable lexical weights
        return_colbert_vecs=False,
    )
    return result['dense_vecs'].tolist(), self._convert_sparse(result['lexical_weights'])

def _convert_sparse(self, lexical_weights: list) -> list[dict[int, float]]:
    """Convert BGE-M3 lexical weights to Qdrant sparse format."""
    sparse_vectors = []
    for weights in lexical_weights:
        # weights is dict of {token_id: weight}
        sparse_vectors.append({int(k): float(v) for k, v in weights.items()})
    return sparse_vectors
```

```python
# qdrant_store.py - Collection with sparse vectors
async def create_hybrid_collection(self, name: str, dimension: int = 1024):
    """Create collection supporting both dense and sparse vectors."""
    await self._client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            ),
        },
    )

async def search_hybrid(
    self,
    collection: str,
    dense_vector: list[float],
    sparse_vector: dict[int, float],
    limit: int = 10,
    filter: Optional[dict] = None,
) -> list[tuple[str, float, dict]]:
    """Hybrid search using Qdrant's native prefetch + RRF."""
    prefetch = [
        models.Prefetch(query=dense_vector, using="dense", limit=limit * 2),
        models.Prefetch(
            query=models.SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values()),
            ),
            using="sparse",
            limit=limit * 2,
        ),
    ]

    results = await self._client.query_points(
        collection_name=collection,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),  # Native RRF
        limit=limit,
        query_filter=self._build_filter(filter) if filter else None,
        with_payload=True,
    )

    return [(str(r.id), r.score, r.payload) for r in results.points]
```

---

### Phase 2: Semantic Chunking [6 hrs]

**Goal**: Handle long content without information loss.

**New Files**:
- `src/ww/chunking/__init__.py`
- `src/ww/chunking/semantic_chunker.py`

**Key Implementation**:

```python
"""Semantic chunking for World Weaver."""

import re
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    id: UUID = field(default_factory=uuid4)
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int
    parent_id: Optional[UUID] = None  # Link to parent episode


class SemanticChunker:
    """
    Sentence-boundary aware chunking with overlap.

    Based on ACE insight: "LLMs are more effective with long, detailed
    contexts" - we chunk to preserve information, not compress it.
    """

    SENTENCE_PATTERN = re.compile(
        r'(?<=[.!?])\s+(?=[A-Z])|'  # After sentence-ending punctuation
        r'(?<=\n)\s*(?=\S)|'        # After newlines
        r'(?<=:)\s*(?=\n)'          # After colons before newlines
    )

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        min_chunk_tokens: int = 50,
        chars_per_token: float = 4.0,  # Conservative estimate
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.chars_per_token = chars_per_token

    def chunk(self, text: str, parent_id: Optional[UUID] = None) -> list[Chunk]:
        """Split text into overlapping semantic chunks."""
        if not text or not text.strip():
            return []

        max_chars = int(self.max_tokens * self.chars_per_token)

        # Short text: single chunk
        if len(text) <= max_chars:
            return [Chunk(
                text=text,
                start_char=0,
                end_char=len(text),
                chunk_index=0,
                total_chunks=1,
                parent_id=parent_id,
            )]

        # Split into sentences
        sentences = self._split_sentences(text)

        # Build chunks with overlap
        chunks = []
        current_sentences = []
        current_chars = 0
        chunk_start = 0
        overlap_chars = int(self.overlap_tokens * self.chars_per_token)

        for sentence, start, end in sentences:
            sentence_chars = end - start

            # Would exceed max?
            if current_chars + sentence_chars > max_chars and current_sentences:
                # Emit chunk
                chunk_text = self._join_sentences(current_sentences, text)
                chunk_end = current_sentences[-1][2]

                chunks.append(Chunk(
                    text=chunk_text,
                    start_char=chunk_start,
                    end_char=chunk_end,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    parent_id=parent_id,
                ))

                # Keep overlap from end of current chunk
                overlap_sentences, overlap_total = self._get_overlap(
                    current_sentences, overlap_chars
                )
                current_sentences = overlap_sentences
                current_chars = overlap_total
                chunk_start = overlap_sentences[0][1] if overlap_sentences else start

            current_sentences.append((sentence, start, end))
            current_chars += sentence_chars

        # Final chunk
        if current_sentences:
            chunk_text = self._join_sentences(current_sentences, text)
            chunks.append(Chunk(
                text=chunk_text,
                start_char=chunk_start,
                end_char=current_sentences[-1][2],
                chunk_index=len(chunks),
                total_chunks=0,
                parent_id=parent_id,
            ))

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into (sentence, start, end) tuples."""
        sentences = []
        last_end = 0

        for match in self.SENTENCE_PATTERN.finditer(text):
            sentence = text[last_end:match.start()].strip()
            if sentence:
                sentences.append((sentence, last_end, match.start()))
            last_end = match.end()

        # Final segment
        final = text[last_end:].strip()
        if final:
            sentences.append((final, last_end, len(text)))

        return sentences

    def _join_sentences(
        self,
        sentences: list[tuple[str, int, int]],
        original: str
    ) -> str:
        """Join sentences preserving original spacing."""
        if not sentences:
            return ""
        start = sentences[0][1]
        end = sentences[-1][2]
        return original[start:end]

    def _get_overlap(
        self,
        sentences: list[tuple[str, int, int]],
        target_chars: int
    ) -> tuple[list[tuple[str, int, int]], int]:
        """Get sentences from end totaling ~target_chars."""
        overlap = []
        total = 0

        for sentence in reversed(sentences):
            chars = sentence[2] - sentence[1]
            if total + chars <= target_chars:
                overlap.insert(0, sentence)
                total += chars
            else:
                break

        return overlap, total
```

**Storage Schema Update**:

```python
# New: EpisodeChunk model
class EpisodeChunk(BaseModel):
    """Chunk of a parent episode for long content."""
    id: UUID = Field(default_factory=uuid4)
    parent_id: UUID  # Parent episode
    chunk_index: int
    total_chunks: int
    content: str
    embedding: Optional[list[float]] = None
    start_char: int
    end_char: int
```

---

### Phase 3: Local NER with GLiNER [4 hrs]

**Goal**: Self-contained entity extraction without external APIs.

**New Files**:
- `src/ww/extraction/gliner_extractor.py`

**Key Implementation**:

```python
"""GLiNER-based local entity extraction for World Weaver."""

import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GLiNERConfig:
    """Configuration for GLiNER extractor."""
    model_name: str = "urchade/gliner_large-v2.1"
    device: str = "cuda:0"
    threshold: float = 0.5
    flat_ner: bool = True
    multi_label: bool = False
    max_length: int = 1024  # GLiNER context window


class GLiNERExtractor:
    """
    Zero-shot NER using GLiNER.

    Chosen over Ollama for:
    - 10-50x faster inference (50-200ms vs 1-5s)
    - Deterministic outputs (testable)
    - Self-contained (no daemon)
    - Purpose-built for NER
    """

    # Entity types aligned with WW semantic memory
    DEFAULT_LABELS = [
        "PERSON",
        "ORGANIZATION",
        "PROJECT",
        "TECHNOLOGY",
        "CONCEPT",
        "TECHNIQUE",
        "TOOL",
        "LOCATION",
        "DATE",
        "FILE_PATH",
    ]

    def __init__(self, config: Optional[GLiNERConfig] = None):
        self.config = config or GLiNERConfig()
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load model with thread-safe double-check."""
        if self._initialized:
            return

        try:
            from gliner import GLiNER

            logger.info(f"Loading GLiNER: {self.config.model_name}")
            self._model = GLiNER.from_pretrained(self.config.model_name)

            if "cuda" in self.config.device:
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.to(self.config.device)
                else:
                    logger.warning("CUDA not available, using CPU")

            self._initialized = True
            logger.info("GLiNER loaded successfully")

        except ImportError:
            logger.error("GLiNER not installed: pip install gliner")
            raise

    async def extract(
        self,
        text: str,
        labels: Optional[list[str]] = None,
    ) -> list['ExtractedEntity']:
        """Extract entities from text."""
        from ww.extraction.entity_extractor import ExtractedEntity

        self._ensure_initialized()

        if not text or not text.strip():
            return []

        labels = labels or self.DEFAULT_LABELS

        # Truncate if needed (GLiNER has context limit)
        if len(text) > self.config.max_length * 4:  # ~4 chars/token
            text = text[:self.config.max_length * 4]

        # GLiNER extraction
        entities = self._model.predict_entities(
            text,
            labels,
            threshold=self.config.threshold,
            flat_ner=self.config.flat_ner,
            multi_label=self.config.multi_label,
        )

        # Convert to ExtractedEntity format
        results = []
        for entity in entities:
            # Extract context window
            start = max(0, entity["start"] - 50)
            end = min(len(text), entity["end"] + 50)
            context = text[start:end]

            results.append(ExtractedEntity(
                name=entity["text"],
                entity_type=self._map_entity_type(entity["label"]),
                confidence=entity["score"],
                span=(entity["start"], entity["end"]),
                context=context,
            ))

        logger.debug(f"GLiNER extracted {len(results)} entities")
        return results

    def _map_entity_type(self, label: str) -> str:
        """Map GLiNER labels to WW EntityType."""
        mapping = {
            "ORGANIZATION": "PROJECT",  # Often projects in code context
            "TECHNIQUE": "TECHNIQUE",
            "TOOL": "TOOL",
        }
        return mapping.get(label, label)

    async def extract_batch(
        self,
        texts: list[str],
        labels: Optional[list[str]] = None,
    ) -> list[list['ExtractedEntity']]:
        """Batch extraction for efficiency."""
        self._ensure_initialized()
        labels = labels or self.DEFAULT_LABELS

        # GLiNER supports batch via list input
        all_entities = self._model.batch_predict_entities(
            texts,
            labels,
            threshold=self.config.threshold,
        )

        # Convert each text's entities
        all_results = []
        for text, entities in zip(texts, all_entities):
            results = []
            for entity in entities:
                from ww.extraction.entity_extractor import ExtractedEntity
                start = max(0, entity["start"] - 50)
                end = min(len(text), entity["end"] + 50)

                results.append(ExtractedEntity(
                    name=entity["text"],
                    entity_type=self._map_entity_type(entity["label"]),
                    confidence=entity["score"],
                    span=(entity["start"], entity["end"]),
                    context=text[start:end],
                ))
            all_results.append(results)

        return all_results
```

---

### Phase 4: Cross-Encoder Reranking [3 hrs]

**Goal**: Refine top-k results with cross-attention scoring.

**New Files**:
- `src/ww/retrieval/__init__.py`
- `src/ww/retrieval/reranker.py`

**Key Implementation**:

```python
"""Cross-encoder reranking for World Weaver."""

import logging
from typing import Optional, TypeVar
import asyncio

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CrossEncoderReranker:
    """
    BGE-reranker-v2-m3 for cross-encoder reranking.

    Uses full cross-attention between query and document,
    more accurate than bi-encoder similarity but slower.
    Applied only to top-k candidates for efficiency.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda:0",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy load reranker."""
        if self._initialized:
            return

        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker: {self.model_name}")
        self._model = CrossEncoder(
            self.model_name,
            max_length=self.max_length,
            device=self.device,
        )
        self._initialized = True
        logger.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Returns:
            List of (original_index, score) sorted by score descending
        """
        self._ensure_initialized()

        if not documents:
            return []

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score in batches
        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            scores = self._model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores)

        # Sort by score (descending)
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        if top_k:
            indexed_scores = indexed_scores[:top_k]

        return indexed_scores

    async def rerank_async(
        self,
        query: str,
        documents: list[str],
        top_k: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """Async wrapper using thread executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.rerank(query, documents, top_k)
        )

    def rerank_with_items(
        self,
        query: str,
        items: list[T],
        content_fn: callable,
        top_k: Optional[int] = None,
    ) -> list[tuple[T, float]]:
        """
        Rerank arbitrary items by extracting content.

        Args:
            query: Search query
            items: List of items to rerank
            content_fn: Function to extract text from item
            top_k: Return top k results

        Returns:
            List of (item, score) sorted by score descending
        """
        documents = [content_fn(item) for item in items]
        ranked = self.rerank(query, documents, top_k)
        return [(items[idx], score) for idx, score in ranked]
```

---

### Phase 5: Retrieval Fusion (RRF) [2 hrs]

**Goal**: Combine multiple retrieval signals optimally.

**New Files**:
- `src/ww/retrieval/fusion.py`

**Key Implementation**:

```python
"""Reciprocal Rank Fusion for multi-signal retrieval."""

from typing import TypeVar, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class FusionResult:
    """Result from fusion with component scores."""
    id: str
    score: float
    components: dict[str, float]


def reciprocal_rank_fusion(
    *result_lists: list[tuple[str, float]],
    k: int = 60,
    limit: Optional[int] = None,
) -> list[FusionResult]:
    """
    Reciprocal Rank Fusion (RRF) for combining ranked lists.

    RRF score = Σ 1/(k + rank_i) for each list

    Based on: "Reciprocal Rank Fusion outperforms Condorcet and
    individual Rank Learning Methods" (Cormack et al., 2009)

    Args:
        result_lists: Multiple (id, score) lists, each sorted by score desc
        k: RRF constant (default 60, optimal per original paper)
        limit: Max results to return

    Returns:
        Fused results with component breakdown
    """
    rrf_scores = defaultdict(float)
    components = defaultdict(dict)

    for list_idx, results in enumerate(result_lists):
        list_name = f"signal_{list_idx}"
        for rank, (id_, original_score) in enumerate(results, start=1):
            contribution = 1.0 / (k + rank)
            rrf_scores[id_] += contribution
            components[id_][list_name] = original_score
            components[id_][f"{list_name}_rank"] = rank

    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    if limit:
        sorted_ids = sorted_ids[:limit]

    return [
        FusionResult(
            id=id_,
            score=rrf_scores[id_],
            components=components[id_],
        )
        for id_ in sorted_ids
    ]


def weighted_score_fusion(
    *result_lists: list[tuple[str, float]],
    weights: Optional[list[float]] = None,
    normalize: bool = True,
    limit: Optional[int] = None,
) -> list[FusionResult]:
    """
    Weighted linear combination of scores.

    Args:
        result_lists: Multiple (id, score) lists
        weights: Weight per list (default: equal)
        normalize: Normalize scores to [0,1] per list
        limit: Max results
    """
    if weights is None:
        weights = [1.0 / len(result_lists)] * len(result_lists)

    assert len(weights) == len(result_lists)

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Optionally normalize scores per list
    normalized_lists = []
    for results in result_lists:
        if normalize and results:
            max_score = max(s for _, s in results)
            min_score = min(s for _, s in results)
            range_score = max_score - min_score if max_score != min_score else 1.0
            normalized = [(id_, (s - min_score) / range_score) for id_, s in results]
        else:
            normalized = results
        normalized_lists.append(normalized)

    # Aggregate
    scores = defaultdict(float)
    components = defaultdict(dict)

    for list_idx, (results, weight) in enumerate(zip(normalized_lists, weights)):
        list_name = f"signal_{list_idx}"
        for id_, score in results:
            scores[id_] += weight * score
            components[id_][list_name] = score

    # Sort
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    if limit:
        sorted_ids = sorted_ids[:limit]

    return [
        FusionResult(id=id_, score=scores[id_], components=components[id_])
        for id_ in sorted_ids
    ]
```

---

### Phase 6: ACE-Inspired Skillbook System [8 hrs]

**Goal**: Implement adaptive learning with evolving skillbook.

**New Files**:
- `src/ww/learning/__init__.py`
- `src/ww/learning/skillbook.py`
- `src/ww/learning/reflector.py`
- `src/ww/learning/skill_manager.py`
- `src/ww/learning/deduplicator.py`
- `src/ww/learning/async_pipeline.py`

#### 6.1 Skillbook Data Structure

```python
"""Skillbook: Evolving playbook for procedural knowledge."""

import json
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Any
from uuid import UUID, uuid4
from enum import Enum


class SkillStatus(str, Enum):
    """Skill lifecycle status."""
    ACTIVE = "active"
    INVALID = "invalid"  # Soft delete for audit


class SkillTag(str, Enum):
    """Skill effectiveness tags (from ACE)."""
    HELPFUL = "helpful"
    HARMFUL = "harmful"
    NEUTRAL = "neutral"


@dataclass
class Skill:
    """
    Individual skill in the skillbook.

    Based on ACE Skill structure with:
    - Counter-based feedback (not continuous weights)
    - Embedding for semantic dedup
    - Soft delete support
    """
    id: str  # Format: {section}-{number:05d}
    section: str  # Domain/task grouping
    content: str  # The actual skill/strategy

    # Counters (ACE pattern)
    helpful: int = 0
    harmful: int = 0
    neutral: int = 0

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Deduplication
    embedding: Optional[list[float]] = None
    status: SkillStatus = SkillStatus.ACTIVE

    @property
    def usefulness(self) -> float:
        """Calculate usefulness score from counters."""
        total = self.helpful + self.harmful + self.neutral
        if total == 0:
            return 0.5  # Neutral default
        # Helpful contributes positively, harmful negatively
        return (self.helpful - self.harmful * 0.5) / total

    @property
    def confidence(self) -> float:
        """Confidence based on total observations."""
        total = self.helpful + self.harmful + self.neutral
        # Sigmoid-like: approaches 1.0 as observations increase
        return total / (total + 10)  # 10 = half-confidence point

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "id": self.id,
            "section": self.section,
            "content": self.content,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "neutral": self.neutral,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding": self.embedding,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Skill':
        """Deserialize from dict."""
        return cls(
            id=data["id"],
            section=data["section"],
            content=data["content"],
            helpful=data.get("helpful", 0),
            harmful=data.get("harmful", 0),
            neutral=data.get("neutral", 0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            embedding=data.get("embedding"),
            status=SkillStatus(data.get("status", "active")),
        )


class Skillbook:
    """
    Central repository of learned skills/strategies.

    Based on kayba-ai/agentic-context-engine Skillbook:
    - Organized by sections
    - Counter-based feedback
    - Semantic deduplication
    - TOON format for LLM consumption
    """

    def __init__(self):
        self._skills: dict[str, Skill] = {}
        self._sections: dict[str, set[str]] = {}  # section -> skill_ids
        self._next_id: dict[str, int] = {}  # section -> counter
        self._similarity_cache: dict[frozenset, tuple[str, float]] = {}

    def add_skill(
        self,
        section: str,
        content: str,
        embedding: Optional[list[float]] = None,
    ) -> Skill:
        """Add new skill to skillbook."""
        # Generate ID
        if section not in self._next_id:
            self._next_id[section] = 1
            self._sections[section] = set()

        skill_id = f"{section}-{self._next_id[section]:05d}"
        self._next_id[section] += 1

        # Create skill
        skill = Skill(
            id=skill_id,
            section=section,
            content=content,
            embedding=embedding,
        )

        self._skills[skill_id] = skill
        self._sections[section].add(skill_id)

        return skill

    def update_skill(self, skill_id: str, content: str) -> Optional[Skill]:
        """Update skill content."""
        if skill_id not in self._skills:
            return None

        skill = self._skills[skill_id]
        skill.content = content
        skill.updated_at = datetime.now().isoformat()
        skill.embedding = None  # Invalidate embedding

        return skill

    def tag_skill(self, skill_id: str, tag: SkillTag) -> Optional[Skill]:
        """Increment tag counter for skill."""
        if skill_id not in self._skills:
            return None

        skill = self._skills[skill_id]
        if tag == SkillTag.HELPFUL:
            skill.helpful += 1
        elif tag == SkillTag.HARMFUL:
            skill.harmful += 1
        elif tag == SkillTag.NEUTRAL:
            skill.neutral += 1

        skill.updated_at = datetime.now().isoformat()
        return skill

    def remove_skill(self, skill_id: str, soft: bool = True) -> bool:
        """Remove skill (soft delete by default for audit)."""
        if skill_id not in self._skills:
            return False

        if soft:
            self._skills[skill_id].status = SkillStatus.INVALID
        else:
            skill = self._skills.pop(skill_id)
            self._sections[skill.section].discard(skill_id)

        return True

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        """Get skill by ID."""
        return self._skills.get(skill_id)

    def skills(self, section: Optional[str] = None, active_only: bool = True) -> list[Skill]:
        """List skills, optionally filtered."""
        if section:
            skill_ids = self._sections.get(section, set())
        else:
            skill_ids = self._skills.keys()

        skills = [self._skills[sid] for sid in skill_ids]

        if active_only:
            skills = [s for s in skills if s.status == SkillStatus.ACTIVE]

        return sorted(skills, key=lambda s: s.usefulness, reverse=True)

    def as_prompt(self, sections: Optional[list[str]] = None) -> str:
        """
        Format skillbook for LLM consumption (TOON format).

        Example output:
        ## coding
        ✓ [coding-00001] Use pytest fixtures for test setup (helpful: 5, harmful: 0)
        ✗ [coding-00002] Avoid global state in tests (helpful: 3, harmful: 2)
        """
        lines = []
        target_sections = sections or list(self._sections.keys())

        for section in sorted(target_sections):
            if section not in self._sections:
                continue

            lines.append(f"## {section}")

            for skill in self.skills(section):
                # Tag indicator
                if skill.helpful > skill.harmful:
                    tag = "✓"
                elif skill.harmful > skill.helpful:
                    tag = "✗"
                else:
                    tag = "○"

                stats = f"(+{skill.helpful}/-{skill.harmful})"
                lines.append(f"{tag} [{skill.id}] {skill.content} {stats}")

            lines.append("")

        return "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        """Get skillbook statistics."""
        active = [s for s in self._skills.values() if s.status == SkillStatus.ACTIVE]
        return {
            "total_skills": len(self._skills),
            "active_skills": len(active),
            "sections": len(self._sections),
            "total_helpful": sum(s.helpful for s in active),
            "total_harmful": sum(s.harmful for s in active),
            "total_neutral": sum(s.neutral for s in active),
        }

    def to_dict(self) -> dict:
        """Serialize skillbook."""
        return {
            "skills": {sid: s.to_dict() for sid, s in self._skills.items()},
            "sections": {sec: list(ids) for sec, ids in self._sections.items()},
            "next_id": self._next_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Skillbook':
        """Deserialize skillbook."""
        sb = cls()
        sb._skills = {sid: Skill.from_dict(s) for sid, s in data.get("skills", {}).items()}
        sb._sections = {sec: set(ids) for sec, ids in data.get("sections", {}).items()}
        sb._next_id = data.get("next_id", {})
        return sb

    def save(self, path: str) -> None:
        """Save to file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'Skillbook':
        """Load from file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))


class ThreadSafeSkillbook:
    """
    Thread-safe wrapper for Skillbook.

    Pattern from ACE: Lock-free reads, locked writes.
    """

    def __init__(self, skillbook: Optional[Skillbook] = None):
        self._skillbook = skillbook or Skillbook()
        self._lock = threading.RLock()

    # Read operations: NO LOCK (eventual consistency acceptable)
    def get_skill(self, skill_id: str) -> Optional[Skill]:
        return self._skillbook.get_skill(skill_id)

    def skills(self, section: Optional[str] = None) -> list[Skill]:
        return self._skillbook.skills(section)

    def as_prompt(self, sections: Optional[list[str]] = None) -> str:
        return self._skillbook.as_prompt(sections)

    def stats(self) -> dict:
        return self._skillbook.stats()

    # Write operations: LOCKED
    def add_skill(self, section: str, content: str, embedding: Optional[list[float]] = None) -> Skill:
        with self._lock:
            return self._skillbook.add_skill(section, content, embedding)

    def update_skill(self, skill_id: str, content: str) -> Optional[Skill]:
        with self._lock:
            return self._skillbook.update_skill(skill_id, content)

    def tag_skill(self, skill_id: str, tag: SkillTag) -> Optional[Skill]:
        with self._lock:
            return self._skillbook.tag_skill(skill_id, tag)

    def remove_skill(self, skill_id: str, soft: bool = True) -> bool:
        with self._lock:
            return self._skillbook.remove_skill(skill_id, soft)

    def save(self, path: str) -> None:
        with self._lock:
            self._skillbook.save(path)
```

#### 6.2 Update Operations

```python
"""Update operations for skillbook mutations."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any
import json


class OperationType(str, Enum):
    """Types of skillbook mutations."""
    ADD = "ADD"
    UPDATE = "UPDATE"
    TAG = "TAG"
    REMOVE = "REMOVE"


@dataclass
class UpdateOperation:
    """Single skillbook update operation."""
    type: OperationType
    section: str
    content: Optional[str] = None
    skill_id: Optional[str] = None
    tag: Optional[str] = None  # For TAG operations
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def validate(self) -> tuple[bool, str]:
        """Validate operation."""
        if self.type == OperationType.ADD:
            if not self.content:
                return False, "ADD requires content"
        elif self.type == OperationType.UPDATE:
            if not self.skill_id or not self.content:
                return False, "UPDATE requires skill_id and content"
        elif self.type == OperationType.TAG:
            if not self.skill_id or not self.tag:
                return False, "TAG requires skill_id and tag"
            if self.tag not in ("helpful", "harmful", "neutral"):
                return False, f"Invalid tag: {self.tag}"
        elif self.type == OperationType.REMOVE:
            if not self.skill_id:
                return False, "REMOVE requires skill_id"
        return True, ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "section": self.section,
            "content": self.content,
            "skill_id": self.skill_id,
            "tag": self.tag,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'UpdateOperation':
        return cls(
            type=OperationType(data["type"].upper()),
            section=data["section"],
            content=data.get("content"),
            skill_id=data.get("skill_id"),
            tag=data.get("tag"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class UpdateBatch:
    """Batch of update operations with reasoning."""
    operations: list[UpdateOperation]
    reasoning: str = ""

    def validate_all(self) -> list[tuple[int, str]]:
        """Validate all operations, return list of (index, error)."""
        errors = []
        for i, op in enumerate(self.operations):
            valid, error = op.validate()
            if not valid:
                errors.append((i, error))
        return errors

    def to_dict(self) -> dict:
        return {
            "operations": [op.to_dict() for op in self.operations],
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'UpdateBatch':
        return cls(
            operations=[UpdateOperation.from_dict(op) for op in data.get("operations", [])],
            reasoning=data.get("reasoning", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'UpdateBatch':
        return cls.from_dict(json.loads(json_str))
```

#### 6.3 Semantic Deduplicator

```python
"""Semantic deduplication for skillbook."""

import logging
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """
    Semantic deduplication using lightweight embeddings.

    Inspired by SemHash: Uses smaller model (Model2Vec or MiniLM)
    for fast dedup embeddings, separate from main BGE-M3.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.90,
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.device = device
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading dedup model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._initialized = True

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        self._ensure_initialized()
        return self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def find_duplicates(
        self,
        new_text: str,
        existing_texts: list[str],
        existing_ids: list[str],
    ) -> Optional[tuple[str, float]]:
        """
        Find if new_text is duplicate of any existing text.

        Returns:
            (duplicate_id, similarity) if duplicate found, else None
        """
        if not existing_texts:
            return None

        self._ensure_initialized()

        # Embed new text
        new_emb = self._model.encode([new_text], convert_to_numpy=True)[0]

        # Embed existing (could cache these)
        existing_embs = self._model.encode(existing_texts, convert_to_numpy=True)

        # Compute similarities
        similarities = np.dot(existing_embs, new_emb) / (
            np.linalg.norm(existing_embs, axis=1) * np.linalg.norm(new_emb)
        )

        # Find max
        max_idx = np.argmax(similarities)
        max_sim = similarities[max_idx]

        if max_sim >= self.threshold:
            return (existing_ids[max_idx], float(max_sim))

        return None

    def deduplicate_batch(
        self,
        texts: list[str],
        ids: list[str],
    ) -> list[tuple[str, list[str]]]:
        """
        Deduplicate a batch, returning clusters.

        Returns:
            List of (representative_id, [member_ids])
        """
        if not texts:
            return []

        self._ensure_initialized()

        # Embed all
        embeddings = self._model.encode(texts, convert_to_numpy=True)

        # Build similarity matrix
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms
        similarities = np.dot(normalized, normalized.T)

        # Greedy clustering
        clusters = []
        used = set()

        for i in range(len(texts)):
            if i in used:
                continue

            # Start new cluster
            cluster = [ids[i]]
            used.add(i)

            # Find similar items
            for j in range(i + 1, len(texts)):
                if j in used:
                    continue
                if similarities[i, j] >= self.threshold:
                    cluster.append(ids[j])
                    used.add(j)

            clusters.append((ids[i], cluster))

        return clusters
```

#### 6.4 Async Learning Pipeline

```python
"""Async learning pipeline for background skill updates."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue, Empty
from threading import Thread
from typing import Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class LearningTask:
    """Task for async learning pipeline."""
    episode_id: str
    content: str
    outcome: str
    context: dict


@dataclass
class ReflectionResult:
    """Result from reflector."""
    episode_id: str
    lessons: list[dict]  # Extracted lessons
    tags: list[tuple[str, str]]  # (skill_id, tag)
    update_batch: Optional['UpdateBatch'] = None


class AsyncLearningPipeline:
    """
    Async learning with parallel reflection, serial updates.

    Pattern from ACE:
    - Main thread: Submit tasks, return immediately
    - ThreadPool: Run reflectors in parallel
    - Single thread: Apply skill updates serially
    """

    def __init__(
        self,
        reflector: 'Reflector',
        skill_manager: 'SkillManager',
        skillbook: 'ThreadSafeSkillbook',
        max_reflector_workers: int = 4,
        queue_size: int = 100,
    ):
        self.reflector = reflector
        self.skill_manager = skill_manager
        self.skillbook = skillbook
        self.max_workers = max_reflector_workers
        self.queue_size = queue_size

        self._reflection_queue: Queue[ReflectionResult] = Queue(maxsize=queue_size)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._manager_thread: Optional[Thread] = None
        self._running = False

    def start(self):
        """Start the learning pipeline."""
        if self._running:
            return

        self._running = True
        self._manager_thread = Thread(target=self._skill_manager_loop, daemon=True)
        self._manager_thread.start()
        logger.info("Async learning pipeline started")

    def stop(self, timeout: float = 30.0):
        """Stop the pipeline gracefully."""
        self._running = False
        if self._manager_thread:
            self._manager_thread.join(timeout=timeout)
        self._executor.shutdown(wait=True)
        logger.info("Async learning pipeline stopped")

    def submit(self, task: LearningTask) -> None:
        """Submit task for async learning (fire-and-forget)."""
        if not self._running:
            logger.warning("Pipeline not running, task dropped")
            return

        # Submit to thread pool
        self._executor.submit(self._process_reflection, task)

    def _process_reflection(self, task: LearningTask) -> None:
        """Run reflection in worker thread."""
        try:
            # Get current skillbook context
            context = self.skillbook.as_prompt()

            # Run reflector
            result = self.reflector.reflect(
                episode_id=task.episode_id,
                content=task.content,
                outcome=task.outcome,
                context=task.context,
                skillbook_context=context,
            )

            # Queue for skill manager
            try:
                self._reflection_queue.put(result, timeout=10.0)
            except:
                logger.warning(f"Reflection queue full, dropping result for {task.episode_id}")

        except Exception as e:
            logger.error(f"Reflection failed for {task.episode_id}: {e}")

    def _skill_manager_loop(self) -> None:
        """Serial skill manager loop."""
        while self._running:
            try:
                result = self._reflection_queue.get(timeout=1.0)

                # Apply updates via skill manager
                self.skill_manager.apply(result, self.skillbook)

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Skill manager error: {e}")

    def wait_for_completion(self, timeout: float = 60.0) -> bool:
        """Wait for all pending work to complete."""
        import time
        start = time.time()

        # Wait for executor
        self._executor.shutdown(wait=True)

        # Wait for queue to drain
        while not self._reflection_queue.empty():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)

        return True
```

---

## Dependencies

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
# Existing
consolidation = ["hdbscan>=0.8.33"]
api = ["fastapi>=0.109.0", "uvicorn[standard]>=0.27.0"]

# New
retrieval = [
    "gliner>=0.2.0",  # Zero-shot NER
]

learning = [
    "gliner>=0.2.0",
]

full = [
    "hdbscan>=0.8.33",
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "gliner>=0.2.0",
]
```

---

## Configuration

Add to `src/ww/core/config.py`:

```python
class Settings(BaseSettings):
    # ... existing

    # Phase 1: Hybrid Search
    hybrid_enabled: bool = Field(default=True)
    hybrid_alpha: float = Field(default=0.7, ge=0.0, le=1.0)

    # Phase 2: Chunking
    chunk_max_tokens: int = Field(default=512, ge=128, le=2048)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=256)
    chunk_min_tokens: int = Field(default=50, ge=10, le=256)

    # Phase 3: NER
    ner_use_gliner: bool = Field(default=True)
    ner_use_llm: bool = Field(default=False)
    ner_gliner_model: str = Field(default="urchade/gliner_large-v2.1")
    ner_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Phase 4: Reranking
    rerank_enabled: bool = Field(default=True)
    rerank_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    rerank_top_k: int = Field(default=20, ge=5, le=100)

    # Phase 6: Skillbook
    skillbook_enabled: bool = Field(default=True)
    skillbook_path: str = Field(default="data/skillbook.json")
    skillbook_dedup_threshold: float = Field(default=0.90, ge=0.5, le=1.0)
    skillbook_atomicity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    learning_async: bool = Field(default=True)
    learning_reflector_workers: int = Field(default=4, ge=1, le=16)
```

---

## Testing Strategy

### Unit Tests
- Each component in isolation
- Mocked dependencies
- Deterministic assertions (GLiNER enables this)

### Integration Tests
```python
@pytest.mark.integration
async def test_full_retrieval_pipeline():
    """End-to-end retrieval with all phases."""
    # Create episode with long content
    content = "..." * 1000  # >512 tokens

    # Ingest (triggers chunking, embedding, NER)
    episode = await episodic.create(content=content, ...)

    # Recall (triggers hybrid search, fusion, reranking)
    results = await episodic.recall("query", limit=10)

    # Verify
    assert len(results) > 0
    assert results[0].components.get("rerank") is not None
```

### Benchmark Tests
```python
@pytest.mark.benchmark
def test_retrieval_latency(benchmark):
    """Verify p95 latency <300ms."""
    results = benchmark(episodic.recall, "test query", limit=10)
    assert benchmark.stats["mean"] < 0.3
```

### Quality Tests
```python
def test_retrieval_quality():
    """Verify MRR@10 meets target."""
    queries = load_test_queries()
    ground_truth = load_ground_truth()

    results = [episodic.recall(q) for q in queries]
    mrr = calculate_mrr(results, ground_truth)

    assert mrr >= 0.80, f"MRR {mrr} below target 0.80"
```

---

## Rollout Plan

| Phase | Duration | Deliverable | Risk | Dependencies |
|-------|----------|-------------|------|--------------|
| **1: Hybrid** | 4 hrs | BGE-M3 sparse enabled | Low | None |
| **2: Chunking** | 6 hrs | Semantic chunker | Medium | Phase 1 |
| **3: GLiNER** | 4 hrs | Local NER | Low | None |
| **4: Reranking** | 3 hrs | Cross-encoder | Low | Phase 1 |
| **5: Fusion** | 2 hrs | RRF combination | Low | Phase 1, 4 |
| **6: Skillbook** | 8 hrs | ACE learning | Medium | Phase 3 |

**Total**: ~27 hours

**Parallel execution possible**: Phases 1-5 can proceed independently. Phase 6 builds on all.

---

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **MRR@10** | ~0.65 | 0.80+ | Test query set |
| **Long content recall** | Truncated | Full | >512 token episodes |
| **NER coverage** | LLM-only | Local + LLM | Entity extraction count |
| **Self-contained** | No (OpenAI) | Yes | No external APIs |
| **Latency p95** | ~200ms | <300ms | Retrieval benchmark |
| **Skill usefulness** | N/A | >0.6 avg | Skillbook stats |
| **Learning latency** | N/A | <5s async | Pipeline benchmark |

---

## Migration Notes

### Breaking Changes
- **Phase 2**: Multi-chunk episodes require schema migration
- **Phase 6**: Skillbook is new persistent state

### Backward Compatibility
- All phases feature-flagged
- Existing API unchanged
- GLiNER/reranker are optional dependencies
- Skillbook starts empty, grows organically

### Migration Script
```python
async def migrate_to_hybrid():
    """Migrate existing collections to hybrid format."""
    # 1. Create new hybrid collection
    # 2. Re-embed existing records with sparse vectors
    # 3. Swap collections atomically
    pass
```

---

## Model Sizes

| Model | Size | Load Time | Memory |
|-------|------|-----------|--------|
| BGE-M3 | 2.3GB | ~30s | ~4GB GPU |
| GLiNER large | 400MB | ~5s | ~1GB GPU |
| BGE-reranker-v2-m3 | 1.1GB | ~10s | ~2GB GPU |
| MiniLM (dedup) | 80MB | ~2s | ~200MB GPU |
| **Total** | ~3.9GB | ~47s cold | ~7.2GB GPU |

All models share CUDA context; actual memory ~5-6GB with sharing.

---

## References

- [ACE Paper](https://arxiv.org/abs/2510.04618) - Stanford/SambaNova
- [kayba-ai/agentic-context-engine](https://github.com/kayba-ai/agentic-context-engine) - Production implementation
- [SemHash](https://github.com/MinishLab/semhash) - Fast semantic deduplication
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - Multi-lingual multi-granularity embeddings
- [GLiNER](https://github.com/urchade/GLiNER) - Zero-shot NER
