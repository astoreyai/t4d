# World Weaver Algorithms

## Overview
World Weaver implements cognitive science-inspired algorithms for memory management across three subsystems: episodic, semantic, and procedural memory.

## 1. FSRS (Free Spaced Repetition Scheduler)

### Purpose
Models memory decay in episodic memory using spaced repetition principles adapted from SuperMemo and Anki.

### Formula
```
R(t, S) = (1 + 0.9 * t/S)^(-0.5)
```
Where:
- **R** = Retrievability (probability of successful recall)
- **t** = Time elapsed since last access (in days)
- **S** = Stability (resistance to forgetting, increases with successful recalls)

### Implementation
- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` (lines 305-309)
- **Used in**: Episode scoring during recall operations
- **Deviation from standard**: Uses 0.9 factor (vs standard 1.0) for slightly slower decay

### Update Algorithm
On successful recall:
```python
new_stability = stability * (1 + 0.1 * (1 - R))
```

On failed recall:
```python
new_stability = stability * 0.8
```

### Parameters
| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| initial_stability | 1.0 | [0.5, 10.0] | Starting stability for new episodes |
| decay_factor | 0.9 | [0.1, 1.0] | Multiplier for time/stability ratio |

### Context
Applied during episodic recall to downweight old memories unless they've been frequently accessed. Stability increases with successful retrievals, creating a self-reinforcing pattern for important memories.

---

## 2. ACT-R Activation

### Purpose
Models semantic memory activation based on recency, frequency, and spreading activation from associated chunks.

### Total Activation Formula
```
A_i = B_i + Σ(W_j * S_ji) + ε
```
Where:
- **A_i** = Total activation of chunk i
- **B_i** = Base-level activation (recency + frequency)
- **W_j** = Attention weight for source chunk j
- **S_ji** = Strength of association from j to i
- **ε** = Gaussian noise term (mean=0, σ=0.1)

### Base-Level Activation
```
B_i = ln(access_count) - d * ln(elapsed_time / 3600)
```
Where:
- **d** = Decay parameter (default: 0.5)
- **elapsed_time** = Seconds since last access

### Spreading Activation
```
S_ji = W * strength * (S_base - ln(max(fan, 1)))
```
Where:
- **W** = 1 / context_size (attention divided among context entities)
- **strength** = Hebbian weight on relationship (see below)
- **S_base** = Base spreading strength (default: 1.6)
- **fan** = Number of outgoing connections from j (fan effect)

### Implementation
- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` (lines 268-318)
- **Optimizations**: Batch relationship queries to eliminate N+1 pattern (lines 320-358)

### Normalization
Raw activation is passed through sigmoid for scoring:
```
norm_activation = 1 / (1 + exp(-activation))
```

### Context
ACT-R (Adaptive Control of Thought—Rational) is a cognitive architecture used to model human memory retrieval. Spreading activation allows context entities to boost related memories, mirroring priming effects in human cognition.

---

## 3. Hebbian Learning

### Purpose
Strengthens connections between co-activated entities following Hebb's rule: "Neurons that fire together, wire together."

### Weight Update Formula
```
w' = w + η * (1 - w)
```
Where:
- **w** = Current weight [0, 1]
- **w'** = New weight [0, 1]
- **η** = Learning rate (default: 0.1)

### Properties
- **Bounded**: Weights asymptotically approach 1.0, never exceed
- **Convergence**: After n updates: `w_n ≈ 1 - (1-w_0) * (1-η)^n`
- **Co-retrieval**: Applied when entities are retrieved together in a single recall

### Implementation
- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` (lines 385-447)
- **Method**: `_strengthen_co_retrieval()`
- **Optimization**: Parallel strengthening with batch relationship queries

### Decay (Not Yet Implemented)
Future versions will add decay for unused relationships:
```
w_decay = w * (1 - decay_rate * dt)
```

### Context
Hebbian learning models synaptic plasticity. In World Weaver, entities that are frequently recalled together develop stronger associative bonds, enabling better context-aware retrieval.

---

## 4. HDBSCAN Clustering

### Purpose
Groups similar episodes or procedures for consolidation without requiring pre-specified cluster count.

### Why HDBSCAN over K-Means?
| Feature | HDBSCAN | K-Means |
|---------|---------|---------|
| Cluster count | Auto-detected | Must specify K |
| Noise handling | Labels outliers as noise (-1) | Forces all points into clusters |
| Density variation | Handles varying densities | Assumes spherical, equal-variance clusters |
| Complexity | O(n log n) | O(n² k) naive, O(nk) optimized |
| Metric | Cosine (for embeddings) | Euclidean |

### Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| min_cluster_size | 3 (episodes), 2 (procedures) | Minimum points per cluster |
| metric | cosine | Distance metric for embeddings |
| cluster_selection_method | eom | Excess of Mass for stable clusters |
| min_samples | 1 | Lenient to capture small patterns |

### Implementation
- **File**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/service.py`
- **Episodes**: Lines 428-507 (`_cluster_episodes`)
- **Procedures**: Lines 509-585 (`_cluster_procedures`)

### Algorithm Overview
1. Compute mutual reachability distances between all points
2. Build minimum spanning tree (MST)
3. Construct cluster hierarchy
4. Extract flat clusters using EOM (Excess of Mass)
5. Label noise points as -1

### Noise Handling
Points that don't fit any cluster (label = -1) are skipped during consolidation. This prevents forced grouping of unrelated memories.

### Context
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) is ideal for semantic clustering because:
- Episode similarities form natural density clusters
- No need to guess optimal K
- Cosine metric works well with text embeddings (BGE-M3)

---

## 5. Multi-Component Scoring

### Episodic Recall Scoring
```
score = 0.4*semantic + 0.25*recency + 0.2*outcome + 0.15*importance
```

**Components**:
- **Semantic** (40%): Vector similarity to query
- **Recency** (25%): `exp(-0.1 * days_elapsed)`
- **Outcome** (20%): SUCCESS=1.0, PARTIAL=0.83, NEUTRAL=0.83, FAILURE=0.67
- **Importance** (15%): Emotional valence [0, 1]

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` (lines 184-189)

### Semantic Recall Scoring
```
score = 0.4*semantic + 0.35*activation + 0.25*retrievability
```

**Components**:
- **Semantic** (40%): Vector similarity to query
- **Activation** (35%): Normalized ACT-R activation (sigmoid)
- **Retrievability** (25%): FSRS decay (same as episodes)

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` (lines 243-247)

### Procedural Retrieval Scoring
```
score = 0.5*semantic + 0.3*success_rate + 0.2*recency
```

**Components**:
- **Semantic** (50%): Vector similarity to task description
- **Success Rate** (30%): `successful_executions / total_executions`
- **Recency** (20%): `exp(-0.1 * days_since_last_use)`

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/procedural.py`

---

## 6. Consolidation Algorithms

### Light Consolidation
**Purpose**: Quick deduplication and cleanup

**Algorithm**:
1. Retrieve last 1000 episodes
2. Find pairs with identical content (exact match)
3. Mark newer duplicate with `emotional_valence=0.0`
4. Add `duplicate_of` pointer to original

**Complexity**: O(n²) pairwise comparison (optimized with content hashing in future)

### Deep Consolidation
**Purpose**: Extract semantic knowledge from episodic clusters

**Algorithm**:
1. Retrieve last 500 episodes
2. Cluster using HDBSCAN (cosine similarity on embeddings)
3. For each cluster with ≥ `min_occurrences` episodes:
   - Extract entity candidates (project, tool, or concept)
   - Create or update semantic entity
   - Link episodes to entity with SOURCE_OF relationship
4. Calculate confidence: `min(1.0, consolidated_count / total_episodes)`

**Complexity**: O(n log n) for clustering + O(k²) for entity extraction (k = cluster size)

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/service.py` (lines 185-284)

### Skill Consolidation
**Purpose**: Merge similar procedures

**Algorithm**:
1. Retrieve all active procedures
2. Cluster using HDBSCAN (cosine similarity on scripts)
3. For each cluster with ≥2 procedures:
   - Sort by success rate (descending)
   - Keep best procedure, merge steps
   - Deprecate others with CONSOLIDATED_INTO relationship
   - Estimate success improvement: `+0.05 per merged procedure`

**Complexity**: O(n log n) for clustering + O(k²) for merging (k = cluster size)

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/service.py` (lines 286-368)

---

## 7. Bi-Temporal Versioning

### Purpose
Track both when events occurred (reference time) and when they were recorded (system time).

### Timestamps
- **T_ref** (`timestamp`): When event occurred in real world
- **T_sys** (`ingested_at`): When memory was created in system
- **Valid_from** (`valid_from`): When version became valid
- **Valid_to** (`valid_to`): When version was superseded (None = current)

### Supersede Algorithm
1. Close old version: Set `valid_to = now()`
2. Create new version: Copy fields, update summary/details
3. Inherit provenance: Same `source` reference

### Use Cases
- **Historical queries**: "What did we know about X on date Y?"
- **Audit trails**: Track how understanding evolved
- **Debugging**: Recover from incorrect updates

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` (lines 462-505)

---

## Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| Episode recall | O(log n) + O(k) | O(k) |
| Semantic recall | O(log n) + O(k*d) | O(k) |
| HDBSCAN clustering | O(n log n) | O(n) |
| Hebbian update | O(1) | O(1) |
| FSRS update | O(1) | O(1) |
| ACT-R activation | O(d) | O(d) |
| Light consolidation | O(n²) | O(n) |
| Deep consolidation | O(n log n + k²) | O(n) |

Where:
- **n** = Total items in collection
- **k** = Number of results returned
- **d** = Average degree (connections per node)

---

## References

1. **FSRS**: Piotr Wozniak, SuperMemo Algorithm SM-2 (1987)
2. **ACT-R**: Anderson, J.R., Bothell, D., Byrne, M.D., et al. (2004). "An integrated theory of the mind." *Psychological Review*, 111(4), 1036-1060.
3. **Hebbian Learning**: Hebb, D.O. (1949). *The Organization of Behavior*. Wiley.
4. **HDBSCAN**: McInnes, L., Healy, J., Astels, S. (2017). "hdbscan: Hierarchical density based clustering." *JOSS*, 2(11), 205.
5. **Bi-temporal Data**: Snodgrass, R.T. (1999). *Developing Time-Oriented Database Applications in SQL*. Morgan Kaufmann.
