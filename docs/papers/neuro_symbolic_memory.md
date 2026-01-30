# Neuro-Symbolic Memory Representations for Adaptive Retrieval Systems

**Draft v0.1** | December 2025

---

## Abstract

We present a hybrid neuro-symbolic memory representation that combines dense vector embeddings with structured knowledge graph triples to enable adaptive, learnable memory retrieval. The system addresses key challenges in retrieval-augmented generation (RAG): (1) credit assignment for retrieved memories, (2) token-efficient context injection, and (3) explainable relevance scoring. Our compact representation format achieves 90% token reduction while preserving identity, reliability signals, and relationship structure. We analyze the complete lifecycle—creation, storage, retrieval, learning, and validation—and identify optimization opportunities in graph traversal and score fusion.

**Keywords**: Neuro-symbolic AI, knowledge graphs, memory augmentation, credit assignment, retrieval-augmented generation

---

## 1. Introduction

Large language models (LLMs) demonstrate remarkable reasoning capabilities but suffer from hallucinations, outdated knowledge, and inability to learn from deployment experience (Lewis et al., 2020). Retrieval-augmented generation (RAG) addresses knowledge limitations by grounding outputs in external documents, but traditional RAG systems retrieve isolated text chunks and ignore relationships among them, weakening multi-hop reasoning (Gao et al., 2024).

Knowledge graph-based RAG offers a structural solution, representing knowledge as typed relationships that support graph traversal and inference (Pan et al., 2024). However, existing approaches lack mechanisms for:

1. **Adaptive learning**: Updating retrieval weights based on task outcomes
2. **Credit assignment**: Attributing success/failure to specific retrieved memories
3. **Token efficiency**: Injecting structured knowledge without exhausting context windows
4. **Neural-symbolic fusion**: Principled combination of embedding similarity and graph structure

This paper presents a neuro-symbolic memory representation addressing these gaps. Our contributions include:

- A **compact format** reducing memory representation from ~500 to ~40 tokens (90% reduction)
- **Triple extractors** that automatically build knowledge graphs from retrieval patterns
- **Credit attribution** via causal graph paths with TD(λ) eligibility traces
- **Score fusion** combining neural embeddings with symbolic relevance
- Analysis of **lifecycle stages** and **optimization opportunities**

---

## 2. Related Work

### 2.1 Neuro-Symbolic AI

The integration of neural and symbolic approaches has seen renewed interest, with Garcez et al. (2022) identifying three paradigms: (1) neural networks informed by symbolic knowledge, (2) symbolic reasoning enhanced by neural representations, and (3) hybrid architectures with bidirectional information flow.

Recent surveys (Mao et al., 2024; Hamilton et al., 2024) categorize neuro-symbolic KG reasoning into:
- **Logically-informed embeddings**: TransE, RotatE, ComplEx encode relations as geometric transformations
- **Embedding with logical constraints**: Neural models constrained by symbolic rules
- **Differentiable rule learning**: DRUM, Neural LP learn symbolic rules via gradient descent

Our work falls in category (3), combining learned embeddings with explicit symbolic structure that evolves through experience.

### 2.2 Knowledge Graph Embeddings

Knowledge graph embedding methods learn continuous representations of entities and relations. Key approaches include:

| Method | Relation Modeling | Expressiveness |
|--------|-------------------|----------------|
| TransE (Bordes et al., 2013) | Translation: h + r ≈ t | Simple, scalable |
| RotatE (Sun et al., 2019) | Rotation in complex space | Handles symmetry/antisymmetry |
| ComplEx (Trouillon et al., 2016) | Hermitian products | Captures asymmetric relations |

These methods excel at link prediction but lack mechanisms for learning from downstream task feedback. Our approach maintains explicit symbolic structure alongside embeddings, enabling interpretable updates.

### 2.3 Memory-Augmented Neural Networks

Memory-augmented architectures provide differentiable read/write access to external memory:

- **Neural Turing Machines** (Graves et al., 2014): Content and location-based addressing
- **Differentiable Neural Computers** (Graves et al., 2016): Dynamic memory allocation
- **Memory Networks** (Weston et al., 2015): Attention over memory slots

These approaches use fully neural memory representations. We instead maintain hybrid representations with explicit symbolic structure for interpretability and credit attribution.

### 2.4 Retrieval-Augmented Generation

RAG systems (Lewis et al., 2020) augment LLMs with retrieved context. Recent advances include:

- **HippoRAG** (Gutierrez et al., 2024): Neurobiologically-inspired long-term memory
- **GraphRAG** (Microsoft, 2024): Community detection for summarization
- **RAPTOR** (Sarthi et al., 2024): Recursive summarization trees

Our work extends RAG with adaptive learning: memories that were retrieved for successful tasks receive positive updates, enabling the system to improve retrieval quality over time.

### 2.5 Credit Assignment in Cognitive Architectures

Cognitive architectures address credit assignment differently:

- **ACT-R** (Anderson, 2007): Base-level activation + spreading activation
- **SOAR** (Laird, 2012): Chunking and reinforcement learning
- **CLARION** (Sun, 2006): Dual explicit/implicit representations

Our approach draws from reinforcement learning, using TD(λ) eligibility traces (Sutton & Barto, 2018) to propagate outcome signals across retrieval events separated in time.

---

## 3. System Architecture

### 3.1 Memory Representation

Each memory unit has dual representations:

**Neural Component:**
- `embedding`: 1024-dimensional BGE-M3 vector for similarity search
- `learned_features`: Scoring features learned from outcomes
- `success_rate`: Running average of positive outcomes when retrieved

**Symbolic Component:**
- `memory_id`: Unique identifier (UUID)
- `memory_type`: Categorical type (Episodic/Semantic/Procedural)
- `triples`: Set of (subject, predicate, object, weight) relationships

### 3.2 Predicate Taxonomy

We define 18 predicate types organized by semantic category:

```
TEMPORAL:     PRECEDED_BY, FOLLOWED_BY, CONCURRENT_WITH
SEMANTIC:     SIMILAR_TO, CONTRASTS_WITH, ELABORATES, SUMMARIZES
CAUSAL:       CAUSED, CONTRIBUTED_TO, BLOCKED
STRUCTURAL:   HAS_TYPE, BELONGS_TO, DERIVED_FROM, INSTANCE_OF
LEARNING:     CO_RETRIEVED, STRENGTHENS, INHIBITS
CITATION:     CITED_BY, USED_IN
```

Each predicate carries:
- **weight**: Learned importance [0, 1]
- **confidence**: Certainty of relationship
- **count**: Observation frequency (for Hebbian updates)

### 3.3 Compact Representation Format

For token-efficient LLM context injection, we define a compact format:

```
[{id8}|{type}|{success%}|{rel1},{rel2},{rel3}]
```

**Example:** `[0c17c544|P|50%|cau→f0245f,sim→398cd0]`

| Field | Meaning | Source |
|-------|---------|--------|
| `0c17c544` | First 8 chars of UUID | Identity |
| `P` | Procedural memory | Type (E/S/P) |
| `50%` | Success rate | Learned signal |
| `cau→f0245f` | CAUSED outcome f0245f | Top relationship |
| `sim→398cd0` | SIMILAR_TO memory 398cd0 | Top relationship |

**Token Analysis:**
- Full JSON representation: ~500 characters (~125 tokens)
- Compact format: ~40 characters (~10 tokens)
- **Reduction: 90%**

---

## 4. Lifecycle Analysis

### 4.1 Creation

Memories are created through three pathways:

1. **Episode Recording**: User interactions create episodic memories with content, timestamp, outcome, emotional valence
2. **Entity Extraction**: NLP extracts semantic entities (people, concepts, tools) from episodes
3. **Skill Building**: Successful action sequences are compiled into procedural memories

**Triple Creation** occurs via extractors:

```python
class CausalExtractor:
    def extract(self, memory, context):
        if reward > 0.5:
            return Triple(memory_id, CAUSED, outcome_id, weight=reward)
        elif reward > 0:
            return Triple(memory_id, CONTRIBUTED_TO, outcome_id, weight=reward)
        elif reward < 0:
            return Triple(memory_id, BLOCKED, outcome_id, weight=abs(reward))
```

### 4.2 Storage

The system uses heterogeneous storage:

| Store | Data | Purpose |
|-------|------|---------|
| **Qdrant** | Embeddings (1024-dim) | Vector similarity search |
| **Neo4j** | Triples, relationships | Graph traversal |
| **SQLite** | Learning events, experiences | Credit assignment |

**Current Gap**: Triples exist in-memory (`TripleSet`) but lack persistent Neo4j integration. The `NeuroSymbolicMemory` wrapper is not yet connected to storage layer.

### 4.3 Retrieval

Retrieval follows a two-phase process:

**Phase 1 - Neural Retrieval:**
```python
candidates = qdrant.search(query_embedding, limit=k)
neural_scores = {m.id: m.score for m in candidates}
```

**Phase 2 - Symbolic Expansion:**
```python
for candidate in candidates:
    # Graph traversal to find related memories
    related = neo4j.traverse(candidate.id, depth=2)
    symbolic_scores[candidate.id] = compute_symbolic_score(candidate.triples)
```

**Score Fusion:**
```python
final_score = 0.6 * neural_score + 0.4 * symbolic_score
```

**Optimization Opportunities:**
1. The 60/40 split is currently fixed—should be learned
2. Graph traversal is O(degree^depth)—needs pruning strategies
3. No caching of frequent traversal patterns

### 4.4 Learning

Learning occurs when outcomes are observed:

**Step 1 - Event Recording:**
```python
retrieval_event = RetrievalEvent(query, memory_type, retrieved_ids, scores)
outcome_event = OutcomeEvent(success_score, context_hash, citations)
```

**Step 2 - Experience Matching:**
Context hashes link retrievals to outcomes occurring within a time window.

**Step 3 - Credit Assignment:**
```python
advantage = outcome_score - baseline
time_discount = 1.0 / (1.0 + 0.1 * hours_delay)
attention_weight = retrieval_score / sum(all_scores)
citation_bonus = 1.5 if explicitly_cited else 1.0

reward = advantage * time_discount * attention_weight * citation_bonus
```

**Step 4 - Weight Updates:**

*Eligibility Traces (TD-λ):*
```python
trace[memory_id] = γλ * trace[memory_id] + retrieval_score
```

*Hebbian Triple Updates:*
```python
triple.weight += α * (reward * triple.count / total_count)
triple.count += 1
```

*Neural Scorer Training:*
```python
loss = ListMLE(predicted_scores, ground_truth_rewards)
optimizer.step()
```

### 4.5 Validation

Current validation coverage:

| Component | Test Coverage | Validation Method |
|-----------|---------------|-------------------|
| Events | ✓ Complete | Unit tests, serialization round-trips |
| Collector | ✓ Complete | Integration tests with SQLite |
| Neuro-Symbolic | ⚠ Partial | In-memory tests only |
| Neo4j Integration | ✗ Missing | No triple persistence tests |
| Score Fusion | ✗ Missing | No end-to-end retrieval tests |

**Validation Gaps:**
1. No tests verifying triples persist to Neo4j
2. No tests for graph traversal performance
3. No ablation studies on fusion weights

---

## 5. Score Fusion Analysis

### 5.1 Current Approach

The reasoner combines neural and symbolic scores linearly:

```python
fused = 0.6 * neural_score + 0.4 * symbolic_score
```

**Symbolic Score Components:**
- Query entity matching: 0.3 weight
- Outcome path connection: 0.5 weight
- Co-retrieval bonus: 0.1 per co-retrieved memory

### 5.2 Theoretical Considerations

The linear fusion has limitations:

1. **Fixed weights**: Cannot adapt to query type or domain
2. **Independence assumption**: Treats neural/symbolic as independent signals
3. **No uncertainty**: Ignores confidence in each score

**Alternative Approaches:**

*Learned Gating:*
```python
gate = sigmoid(W @ [query_embedding, symbolic_features])
fused = gate * neural + (1 - gate) * symbolic
```

*Attention-Based Fusion:*
```python
attention = softmax(Q @ K.T / sqrt(d))
fused = attention @ V  # V contains both neural and symbolic
```

*Bayesian Combination:*
```python
p(relevant | neural, symbolic) ∝ p(neural | relevant) * p(symbolic | relevant) * p(relevant)
```

### 5.3 Graph Traversal Optimization

Current traversal is naive BFS with depth limit:

```python
def find_paths(start, end, max_depth=3):
    # O(degree^depth) complexity
    ...
```

**Optimization Strategies:**

1. **Bidirectional Search**: Search from both start and end, meet in middle
2. **A* with Learned Heuristic**: Use embedding distance as heuristic
3. **Materialized Views**: Pre-compute common traversal patterns
4. **Pruning by Edge Weight**: Skip edges below threshold

---

## 6. Alternative Representations

### 6.1 Current Formats

We implemented three representation formats:

| Format | Tokens | Use Case |
|--------|--------|----------|
| FullJSON | ~125 | Debugging, full fidelity |
| ToonJSON | ~40 | Moderate compression |
| NeuroSymbolicTriples | ~50 | Graph structure emphasis |
| Compact | ~10 | LLM context injection |

### 6.2 Potential Additions

**Uncertainty-Aware Format:**
```
[0c17c544|P|50±15%|cau→f0245f(0.9),sim→398cd0(0.85)]
```
Adds confidence intervals and edge weights.

**Hierarchical Format:**
```
[0c17c544|P|50%|parent:abc123|children:3|depth:2]
```
For memories with part-whole relationships (inspired by Hinton's capsules).

**Temporal Context Format:**
```
[0c17c544|P|50%|age:2h|trend:↑|last_success:0.5h]
```
Includes recency and performance trend.

---

## 7. Discussion

### 7.1 Symbol Grounding

A key question in neuro-symbolic AI is symbol grounding: how do symbolic predicates acquire meaning? In our system:

- **SIMILAR_TO** is grounded in embedding cosine similarity (threshold 0.8)
- **CAUSED** is grounded in temporal precedence + outcome correlation
- **CO_RETRIEVED** is grounded in retrieval co-occurrence statistics

This provides operational grounding but may miss semantic nuances. Future work could learn predicate embeddings jointly with entity embeddings.

### 7.2 Credit Attribution Plausibility

Our path-based credit attribution (product of edge weights) assumes:

1. Credit flows backward through causal chains
2. Intermediate nodes partially transmit credit
3. Multiple paths contribute additively

This resembles backpropagation through a graph, but with learned edge weights rather than gradients. The biological plausibility is debated—Hebbian learning is local, while our credit attribution requires global path information.

### 7.3 Limitations

1. **Cold Start**: New memories have no relationships until retrieved
2. **Scalability**: Graph traversal is expensive for dense graphs
3. **Staleness**: Old relationships may not reflect current relevance
4. **Single Graph**: No mechanism for context-dependent subgraphs

---

## 8. Recommendations

### 8.1 Immediate Improvements

1. **Persist triples to Neo4j**: Connect `NeuroSymbolicMemory` to storage
2. **Add fusion weight learning**: Replace 60/40 with learned gate
3. **Implement traversal caching**: Memoize frequent paths
4. **Add confidence to compact format**: `[id|type|rate±conf|rels]`

### 8.2 Research Directions

1. **Differentiable graph operations**: End-to-end learning through graph
2. **Attention over triples**: Dynamic predicate weighting per query
3. **Hierarchical memory**: Part-whole relationships (capsule-inspired)
4. **Continual consolidation**: Sleep-like offline reorganization

---

## 9. Mathematical Formalization

### 9.1 Memory Representation

A neuro-symbolic memory unit $m$ is a tuple:

$$m = (\mathbf{e}, \mathcal{T}, \theta)$$

Where:
- $\mathbf{e} \in \mathbb{R}^{d}$ is the neural embedding (d=1024 for BGE-M3)
- $\mathcal{T} = \{(s, p, o, w)\}$ is the symbolic triple set
- $\theta = (\text{type}, \text{success\_rate}, \text{retrieval\_count})$ is metadata

### 9.2 Score Fusion

Given query embedding $\mathbf{q}$, neural score $\phi_n$ and symbolic score $\phi_s$:

$$\phi_n(m, \mathbf{q}) = \frac{\mathbf{e}_m \cdot \mathbf{q}}{||\mathbf{e}_m|| \cdot ||\mathbf{q}||}$$

$$\phi_s(m, \mathcal{G}) = \sum_{p \in \text{paths}(\mathcal{T}_m, \mathcal{G})} \prod_{(s,p,o,w) \in p} w$$

**Current (Fixed) Fusion:**
$$\phi(m, \mathbf{q}) = \alpha \cdot \phi_n + (1-\alpha) \cdot \phi_s, \quad \alpha = 0.6$$

**Recommended (Learned) Fusion:**
$$\alpha(\mathbf{q}, \mathbf{e}_m) = \sigma(\mathbf{W}[\mathbf{q}; \mathbf{e}_m; \mathbf{r}] + b)$$

Where $\mathbf{r}$ is the one-hot relation type encoding and $\sigma$ is sigmoid.

### 9.3 Credit Assignment (TD-λ)

For a retrieval event at time $t$ with memories $M_t = \{m_1, ..., m_k\}$ and outcome $O_{t+\tau}$ with success score $s$:

**Eligibility Trace Update:**
$$z_t(m) = \gamma \lambda \cdot z_{t-1}(m) + \mathbb{1}[m \in M_t] \cdot \phi(m)$$

Where:
- $\gamma = 0.99$ is the discount factor
- $\lambda = 0.9$ is the trace decay

**Reward Assignment:**
$$r(m) = z_t(m) \cdot (s - \bar{s}) \cdot \beta_{cite}(m)$$

Where:
- $\bar{s}$ is the running baseline (exponential moving average)
- $\beta_{cite}(m) = 1.5$ if explicitly cited, else $1.0$

### 9.4 Hebbian Weight Update

For co-retrieved memories $m_i, m_j$:

$$\Delta w_{ij} = \eta \cdot (1 - w_{ij}) \cdot r_{\text{outcome}}$$

With bounded update approaching asymptote:
$$w'_{ij} = w_{ij} + \Delta w_{ij}, \quad w \in [0, 1]$$

**Anti-Hebbian (LTD) for failures:**
$$\Delta w_{ij} = -\eta' \cdot w_{ij} \cdot (1 - r_{\text{outcome}})$$

### 9.5 ListMLE Loss

For learning retrieval ranking with ground truth relevance $y_1 > y_2 > ... > y_n$:

$$\mathcal{L}_{\text{ListMLE}} = -\sum_{i=1}^{n} \log \frac{\exp(\phi(m_i))}{\sum_{j=i}^{n} \exp(\phi(m_j))}$$

This is the negative log-likelihood of the correct ranking under a Plackett-Luce model.

---

## 10. System Accountability and Auditability

### 10.1 Decision Traceability

Every retrieval decision can be traced through:

1. **Query Context Hash**: Links retrieval to conversation state
2. **Retrieved Memory IDs**: Which memories were surfaced
3. **Component Scores**: Why each memory ranked as it did
4. **Outcome Linkage**: What happened after retrieval

**Audit Query Example:**
```sql
SELECT r.query, r.retrieved_ids, r.retrieval_scores,
       o.outcome_type, o.success_score,
       e.per_memory_rewards
FROM retrieval_events r
JOIN outcome_events o ON r.context_hash = o.context_hash
JOIN experiences e ON e.retrieval_id = r.id
WHERE r.session_id = 'session-001'
ORDER BY r.timestamp;
```

### 10.2 Memory Provenance

Each memory's compact representation encodes its history:

```
[0c17c544|P|50%|cau→f0245f,sim→398cd0]
         ↑   ↑      ↑
         │   │      └── Causal link to specific outcome
         │   └── Reliability: 50% success rate (neutral)
         └── Type: Procedural skill
```

**Questions This Answers:**
- *Where did this memory come from?* → Check DERIVED_FROM edges
- *How reliable is it?* → Success rate from learning history
- *What outcomes did it influence?* → CAUSED/CONTRIBUTED_TO edges
- *What else is it related to?* → SIMILAR_TO/CO_RETRIEVED edges

### 10.3 Bias Detection

Systematic biases can be detected via:

1. **Retrieval Disparity**: Are certain memory types over/under-retrieved?
2. **Reward Correlation**: Do certain predicates correlate with success?
3. **Temporal Drift**: Are older memories systematically down-weighted?

**Bias Audit Query:**
```python
# Check if procedural memories are over-credited
procedural_avg_reward = mean(r for r in rewards if type == 'P')
episodic_avg_reward = mean(r for r in rewards if type == 'E')
bias_ratio = procedural_avg_reward / episodic_avg_reward
```

### 10.4 Explainability via Graph Paths

For any retrieval decision, the system can generate explanations:

```python
def explain_retrieval(memory_id, outcome_id):
    paths = find_paths(memory_id, outcome_id, max_depth=3)
    explanations = []
    for path in paths:
        edges = " → ".join([f"{t.predicate}({t.weight:.2f})" for t in path])
        explanations.append(f"Memory {memory_id[:8]} {edges} Outcome {outcome_id[:8]}")
    return explanations

# Example output:
# Memory 0c17c544 CAUSED(0.90) Outcome f0245fc7
# Memory 0c17c544 SIMILAR_TO(0.85) Memory 398cd091 CONTRIBUTED_TO(0.60) Outcome f0245fc7
```

---

## 11. Practical Applications

### 11.1 Adaptive RAG

The neuro-symbolic representation enables RAG systems that improve with use:

```python
# Before: Static retrieval
results = retriever.search(query, k=10)

# After: Adaptive retrieval with learned ranking
results = retriever.search(query, k=10)
for result in results:
    result.score = neural_scorer.score([
        result.similarity,
        result.recency,
        result.success_rate,  # Learned signal
        result.outcome_history,  # Historical performance
    ])
results.sort(by='score', reverse=True)
```

### 11.2 Multi-Session Learning

Memories learned in one session transfer to others:

```
Session 1: User asks about Python logging → Memory M1 retrieved → Task succeeds
           → M1.success_rate updated, CAUSED edge added

Session 2: Different user asks about Python logging → M1 retrieved with higher score
           → Benefits from Session 1's learning
```

### 11.3 Memory Consolidation

Periodically consolidate episodic memories into semantic knowledge:

```python
# Cluster similar episodes
clusters = hdbscan.fit(episodic_embeddings)

# Extract common patterns
for cluster in clusters:
    if len(cluster) > threshold:
        semantic_entity = synthesize_entity(cluster.episodes)
        semantic_entity.add_triple(DERIVED_FROM, cluster.episode_ids)
```

### 11.4 Debugging Failed Tasks

When tasks fail, trace back through memory:

```python
def debug_failure(session_id):
    # Find all retrievals in failed session
    retrievals = get_retrievals(session_id)
    outcome = get_outcome(session_id)

    # Identify potential culprits
    for r in retrievals:
        if r.success_rate < 0.3:
            print(f"Low-reliability memory retrieved: {r.memory_id}")
        if BLOCKED in r.edges:
            print(f"Memory with negative history: {r.memory_id}")
```

---

## 12. Future Directions

### 12.1 Differentiable Graph Operations

Current graph operations are discrete. Future work should explore:
- **Differentiable message passing** (GNN-style updates)
- **Soft path attention** instead of hard path enumeration
- **End-to-end training** through retrieval + reasoning

### 12.2 Uncertainty Quantification

Add epistemic uncertainty to all components:
- **Embedding uncertainty**: Monte Carlo dropout at encoding
- **Edge weight distributions**: Beta distributions instead of point estimates
- **Retrieval confidence**: Calibrated probability that memory is relevant

### 12.3 Hierarchical Memory (Capsule-Inspired)

Implement part-whole relationships:
```
[abstract_concept|S|80%|has_part→concrete1,concrete2,concrete3]
```

### 12.4 Continual Reconsolidation

Memories should update on retrieval:
```python
async def recall_with_reconsolidation(query, context):
    results = await recall(query)
    for result in results:
        # Context may update our understanding
        updated = contextualize(result.embedding, context)
        if distance(updated, result.embedding) > threshold:
            await update_embedding(result.id, updated)
    return results
```

---

## 13. Conclusion

We presented a neuro-symbolic memory representation that addresses key challenges in adaptive retrieval systems:

1. **Token Efficiency**: 90% reduction via compact format `[id|type|rate|edges]`
2. **Credit Assignment**: TD-λ eligibility traces + outcome-weighted Hebbian learning
3. **Explainability**: Graph paths provide interpretable decision traces
4. **Accountability**: Full audit trail from query to outcome

The system represents a pragmatic middle ground between fully neural approaches (flexible but opaque) and fully symbolic systems (interpretable but brittle). By maintaining explicit symbolic structure alongside learned embeddings, we enable both human understanding and machine learning.

**Key Trade-offs:**
- Interpretability vs. end-to-end learning (we favor interpretability)
- Fixed structure vs. learned structure (we use fixed predicates with learned weights)
- Read-only vs. dynamic memories (gap: reconsolidation not implemented)

Future work will address learned fusion weights, Bayesian uncertainty, and reconsolidation to create truly adaptive memory systems.

---

## References

Anderson, J. R. (2007). How Can the Human Mind Occur in the Physical Universe? Oxford University Press.

Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating embeddings for modeling multi-relational data. NeurIPS.

Gao, Y., et al. (2024). Retrieval-Augmented Generation for AI-Generated Content: A Survey. arXiv:2402.19473.

Garcez, A., Lamb, L. C., & Gabbay, D. M. (2022). Neural-Symbolic Cognitive Reasoning. Springer.

Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. arXiv:1410.5401.

Graves, A., et al. (2016). Hybrid computing using a neural network with dynamic external memory. Nature.

Gutierrez, B. J., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. NeurIPS.

Hamilton, W., et al. (2024). Neurosymbolic AI for Reasoning over Knowledge Graphs: A Survey. IEEE TNNLS.

Laird, J. E. (2012). The Soar Cognitive Architecture. MIT Press.

Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.

Mao, J., et al. (2024). Neuro-Symbolic AI in 2024: A Systematic Review. arXiv:2501.05435.

Pan, S., et al. (2024). Unifying Large Language Models and Knowledge Graphs: A Roadmap. IEEE TKDE.

Sarthi, P., et al. (2024). RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. ICLR.

Sun, R. (2006). The CLARION cognitive architecture. Cognitive Systems Research.

Sun, Z., et al. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. ICLR.

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.

Trouillon, T., et al. (2016). Complex Embeddings for Simple Link Prediction. ICML.

Weston, J., Chopra, S., & Bordes, A. (2015). Memory Networks. ICLR.

---

## Appendix A: Hinton Critique (Synthesized)

This appendix captures analysis from a Geoffrey Hinton-inspired perspective on the system's architecture.

### A.1 The 60/40 Neural-Symbolic Fusion Is Arbitrary

**Critique**: "These weights should be learned from data, not hand-engineered. The whole point of neural networks is that we do not have to specify how features combine - that emerges from training."

The system treats fusion as a hyperparameter when it should be a **learned gate**. From capsule networks and mixture-of-experts: routing between components should be input-dependent, not fixed.

**Recommendation**: Implement a gating network:
```python
class NeuralSymbolicFusion(nn.Module):
    def __init__(self, embedding_dim=1024, hidden_dim=64):
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim * 2 + NUM_RELATIONS, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # neural, symbolic, decay
            nn.Softmax(dim=-1)
        )

    def forward(self, query_emb, item_emb, relation_one_hot):
        features = torch.cat([query_emb, item_emb, relation_one_hot], dim=-1)
        return self.gate(features)  # [w_neural, w_symbolic, w_decay]
```

### A.2 Hebbian Updates Are Plausible but Incomplete

**What's Missing**:
1. **No LTD (Long-Term Depression)**: Biological synapses weaken when pre-synaptic activity occurs without post-synaptic response. Our system only strengthens.
2. **Symmetric Strengthening**: We update both directions equally. Biological connections are often asymmetric.

**Recommendation**: Add outcome-weighted updates:
```python
# Success: strengthen associations that led here
# Failure: weaken or don't strengthen
delta_w = learning_rate * (1 - w) * outcome_signal
```

### A.3 Path-Based Credit Attribution Is Problematic

**Critique**: "Product of probabilities becomes vanishingly small over long chains. This is the credit assignment problem."

Current approach cannot distinguish between "weak at every step" vs "weak at one critical step."

**Recommendation**: Consider the **forward-forward algorithm** approach (Hinton, 2022):
1. During retrieval, record the path taken
2. On success: do a "goodness" pass strengthening each edge
3. On failure: do a "badness" pass (weaker or neutral update)

### A.4 The Symbol Grounding Problem Is Unaddressed

Predicates like `CAUSES`, `SIMILAR_TO` are **syntactic labels** for graph traversal, not semantic entities. The embedding model has no knowledge of these predicate types.

**Critique**: "True understanding requires grounding symbols in sensorimotor experience or learned representations. Your predicates are not grounded - they are programmer-defined categories."

**Recommendation**: Learn relation embeddings via TransE/RotatE and add relation type embeddings to scoring.

### A.5 Comparison to Attention Mechanisms

| Aspect | Our ACT-R Spreading | Transformer Attention |
|--------|---------------------|----------------------|
| Weights | Pre-computed (Hebbian) | Computed per-query |
| Scope | Fixed graph neighbors | All items in context |
| Learning | Separate from use | End-to-end differentiable |

**What's Missing**:
- Multi-head aspect: Only one spreading activation mechanism
- Query-key-value separation: Same weights for all queries

### A.6 Biological Memory: Critical Gaps

| Property | Implementation | Gap |
|----------|---------------|-----|
| Fast hippocampal learning | Episode storage | OK |
| Slow neocortical consolidation | HDBSCAN clustering | Missing incremental |
| **Reconsolidation** | Not implemented | **CRITICAL** |
| Pattern separation | Sparse embeddings | Could be stronger |

**Most Critical Gap**: Every retrieval should update the memory based on current context. Our retrieval is read-only.

### A.7 Uncertainty Is Missing

The compact format reports point estimates everywhere. "Neural networks should know what they do not know."

**Recommendation**: Add Bayesian treatment:
- Edge weights as distributions: `weight ~ Beta(α, β)`
- Report variance: `[0c17c544|P|50%±12%|...]`

### A.8 Final Assessment

"This is a well-engineered symbolic AI system augmented with neural embeddings. It is not a neural system that learns its structure. Both approaches have value. For a PhD-level system requiring interpretability and auditability, the symbolic approach is defensible."

**Fundamental Tensions**:
- Symbolic structure vs learned representations
- Hand-tuned weights vs end-to-end learning
- Read-only retrieval vs memory dynamics

---

## Appendix B: Implementation Status

| Component | File | Status |
|-----------|------|--------|
| Events | `learning/events.py` | ✓ Complete |
| Representations | `learning/events.py` | ✓ Complete |
| Collector | `learning/collector.py` | ✓ Complete |
| Neuro-Symbolic | `learning/neuro_symbolic.py` | ✓ Complete |
| Hooks | `learning/hooks.py` | ✓ Complete |
| Neural Scorer | `learning/scorer.py` | ✓ Complete |
| Neo4j Integration | `storage/neo4j_store.py` | ⚠ Partial |
| End-to-End Pipeline | - | ✗ Missing |

## Appendix B: Test Coverage

```
Total tests: 1,273 passing
Coverage: 58%
Learning module: Not yet in coverage (new code)
```
