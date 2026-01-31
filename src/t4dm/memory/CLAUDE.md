# Memory
**Path**: `/mnt/projects/t4d/t4dm/src/ww/memory/`

## What
Tripartite memory system implementing episodic, semantic, and procedural memory types plus working memory, pattern separation (dentate gyrus), Modern Hopfield retrieval, active forgetting, and learned sparse indexing.

## How
### Core Memory Types
- **EpisodicMemory** (`episodic.py`): Stores experiences with temporal context. Supports hybrid retrieval (vector similarity + recency + importance). Uses Saga pattern for dual-store consistency. Split into focused submodules: `episodic_storage.py`, `episodic_retrieval.py`, `episodic_saga.py`, `episodic_learning.py`, `episodic_fusion.py`.
- **SemanticMemory** (`semantic.py`): Entity and concept storage with relationship graphs. Knowledge extracted from episodes during consolidation.
- **ProceduralMemory** (`procedural.py`): Learned skills and action sequences. Reinforced through successful outcomes.
- **WorkingMemory** (`working_memory.py`): Capacity-limited active buffer with attentional blink, priority-based eviction, and item decay.

### Retrieval & Indexing
- **PatternSeparation** (`pattern_separation.py`): Dentate gyrus simulation for orthogonalizing similar memories. Modern Hopfield networks (`softmax(beta * X^T * query) * X`) for pattern completion with configurable modes (standard, sparse, exponential).
- **LearnedSparseIndex** (`learned_sparse_index.py`): Neural sparse retrieval for efficient memory lookup.
- **ClusterIndex** (`cluster_index.py`): Clustering-based memory organization.
- **FeatureAligner** (`feature_aligner.py`): Aligns features across memory types.
- **BufferManager** (`buffer_manager.py`): Manages memory buffers and overflow.

### Forgetting
- **ActiveForgettingSystem** (`forgetting.py`): Biologically-inspired forgetting with retention policies. Identifies candidates by decay, interference, and redundancy. Supports multiple strategies.

## Why
The tripartite architecture mirrors human memory organization: episodic for "what happened," semantic for "what I know," procedural for "how to do things." Working memory provides the active workspace. Pattern separation prevents catastrophic interference.

## Key Files
| File | Purpose |
|------|---------|
| `episodic.py` | `EpisodicMemory` main class |
| `semantic.py` | `SemanticMemory` entity/concept store |
| `procedural.py` | `ProceduralMemory` skill store |
| `working_memory.py` | `WorkingMemory` capacity-limited buffer |
| `pattern_separation.py` | `DentateGyrus`, `PatternCompletion`, Modern Hopfield |
| `forgetting.py` | `ActiveForgettingSystem`, retention policies |
| `unified.py` | Unified memory interface across all types |

## Data Flow
```
Input -> WorkingMemory (active processing)
    -> EpisodicMemory (experience storage via Saga)
    -> Consolidation -> SemanticMemory (entity extraction)
    -> Outcome feedback -> ProceduralMemory (skill refinement)

Retrieval query -> PatternSeparation (orthogonalize)
    -> Hopfield completion -> hybrid scoring -> ranked results
    -> ActiveForgetting (prune low-value memories)
```

## Integration Points
- **bridges/**: Neo4j + Qdrant dual-store via Saga pattern
- **learning/**: Retrieval events feed learning system; outcomes update memory strength
- **consolidation/**: HDBSCAN clustering triggers semantic extraction
- **encoding/**: Time2Vec temporal encoding for memory timestamps
- **nca/hippocampus.py**: Hippocampal circuit drives encoding/retrieval mode
