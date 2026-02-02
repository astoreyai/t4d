# T4DM - Mathematical Foundations

**Version**: 0.2.0
**Last Updated**: 2025-12-06
**Status**: Complete specification of all mathematical models in production

---

## Table of Contents

1. [Memory Decay Models](#1-memory-decay-models)
2. [Hebbian Learning](#2-hebbian-learning)
3. [ACT-R Activation](#3-act-r-activation)
4. [Retrieval Scoring](#4-retrieval-scoring)
5. [Learned Memory Gating](#5-learned-memory-gating)
6. [Neuromodulation](#6-neuromodulation)
7. [Pattern Separation & Clustering](#7-pattern-separation--clustering)
8. [Similarity & Fusion](#8-similarity--fusion)

---

## 1. Memory Decay Models

### 1.1 FSRS (Free Spaced Repetition Scheduler)

**Purpose**: Model memory retrievability over time with stability-based decay

**Core Formula**:
```
R(t, S) = (1 + 0.9 × t/S)^(-0.5)
```

**Parameters**:
- `t`: Elapsed time in days since last access
- `S`: Stability (learned parameter, in days)
- `R`: Retrievability ∈ [0, 1]
- `0.9`: Decay factor (slower than flashcard FSRS which uses 1.0)
- `-0.5`: Power law exponent (cognitive forgetting curve)

**LaTeX**:
```latex
R(t, S) = \left(1 + \frac{0.9t}{S}\right)^{-0.5}
```

**Stability Update on Successful Retrieval**:
```
S' = S × (1 + exp(D) × (11 - d) × S^(-0.2) × (exp(0.1 × (1 - R)) - 1))
```

Where:
- `d`: Difficulty ∈ [1, 10] (learned from retrieval attempts)
- `D`: Base difficulty adjustment = 0.5
- Higher `R` at retrieval → larger stability increase

**Stability Update on Failed Retrieval**:
```
S' = D × S^0.2 × (exp(0.1 × d) - 1)
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/core/types.py` lines 128-144

**Evidence**: FSRS shows 20-30% improvement over SM-2 algorithm ([Jarrett Ye et al., 2024](https://github.com/open-spaced-repetition/fsrs4anki))

---

### 1.2 Exponential Recency Decay

**Purpose**: Weight recent memories higher in retrieval

**Formula**:
```
w_recency(t) = exp(-λ × t)
```

**Parameters**:
- `t`: Days since episode timestamp
- `λ`: Decay rate = 0.1 (configurable)
- `w_recency`: Weight ∈ (0, 1]

**LaTeX**:
```latex
w_{\text{recency}}(t) = e^{-\lambda t}
```

**Example Values**:
- Day 0: `w = 1.0`
- Day 7: `w = 0.497`
- Day 30: `w = 0.050`

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` retrieval scoring

---

## 2. Hebbian Learning

### 2.1 Asymptotic Weight Update

**Purpose**: Strengthen co-retrieved entity relationships ("fire together, wire together")

**Bounded Update Formula**:
```
w' = w + η × (1 - w)
```

**Parameters**:
- `w`: Current weight ∈ [0, 1]
- `w'`: Updated weight
- `η`: Learning rate = 0.1 (default)
- Asymptotically approaches 1.0

**LaTeX**:
```latex
w' = w + \eta(1 - w) = w(1 - \eta) + \eta
```

**Convergence**:
After `n` co-retrievals:
```
w_n = 1 - (1 - w_0)(1 - η)^n
```

**Example Progression** (η = 0.1, w_0 = 0.1):
- Co-retrieval 1: `w = 0.19`
- Co-retrieval 5: `w = 0.41`
- Co-retrieval 10: `w = 0.61`
- Co-retrieval 20: `w = 0.79`
- Limit: `w → 1.0`

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` lines 50-56

---

### 2.2 Fan-Out Normalization

**Purpose**: Prevent hub nodes from dominating spreading activation

**Normalized Strength**:
```
S_norm(A → B) = w(A → B) / √(fan_out(A))
```

Where `fan_out(A)` = number of outgoing edges from entity A

**LaTeX**:
```latex
S_{\text{norm}}(A \to B) = \frac{w(A \to B)}{\sqrt{\text{fan}(A)}}
```

**Rationale**: Prevents entities with many connections from flooding activation

---

## 3. ACT-R Activation

### 3.1 Base-Level Activation

**Purpose**: Frequency and recency based activation

**Formula**:
```
B_i = ln(Σ_j t_j^(-d))
```

**Parameters**:
- `t_j`: Time since jth access (in seconds)
- `d`: Decay parameter = 0.5
- `B_i`: Base-level activation (log scale)

**LaTeX**:
```latex
B_i = \ln\left(\sum_{j=1}^{n} t_j^{-d}\right)
```

**Interpretation**:
- More frequent accesses → higher sum
- Recent accesses weighted more (power law)
- Log transform prevents extreme values

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` ACTRRetrieval class

---

### 3.2 Spreading Activation

**Purpose**: Context-sensitive retrieval via graph structure

**Formula**:
```
A_spread = Σ_j (W_j × S_ji)
```

Where:
```
W_j = 1 / |context_entities|
S_ji = S_max - ln(fan_out(j))
```

**Parameters**:
- `S_max`: Maximum association strength = 1.6
- `fan_out(j)`: Number of edges from source entity j
- Context divided equally among source entities

**LaTeX**:
```latex
A_{\text{spread}} = \sum_{j \in C} W_j \times S_{ji}
```
```latex
W_j = \frac{1}{|C|}, \quad S_{ji} = S_{\max} - \ln(\text{fan}(j))
```

---

### 3.3 Total Activation

**Combined Formula**:
```
A_i = B_i + A_spread + ε
```

**Parameters**:
- `B_i`: Base-level activation (frequency/recency)
- `A_spread`: Spreading activation from context
- `ε ~ N(0, σ)`: Gaussian noise, σ = 0.5
- `τ`: Retrieval threshold = 0.0

**Retrieval Probability**:
```
P(retrieve) = 1 if A_i ≥ τ, else 0
```

**LaTeX**:
```latex
A_i = B_i + \sum_{j \in C} W_j S_{ji} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` ACTRRetrieval class

---

## 4. Retrieval Scoring

### 4.1 Fixed-Weight Episode Scoring (Current Production)

**Multi-Factor Score**:
```
score = α × s_semantic + β × s_recency + γ × s_outcome + δ × s_importance
```

**Default Weights**:
- α (semantic) = 0.4
- β (recency) = 0.25 (was 0.3, updated)
- γ (outcome) = 0.2
- δ (importance) = 0.15

**Component Scores**:
```
s_semantic = cosine_similarity(q, e)
s_recency = exp(-0.1 × days_since_timestamp)
s_outcome = 1.2 if outcome == "success" else 0.8
s_importance = emotional_valence
```

**LaTeX**:
```latex
\text{score} = \alpha \cdot \text{sim}(q, e) + \beta \cdot e^{-\lambda t} + \gamma \cdot o + \delta \cdot v
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` recall method

---

### 4.2 Learned Fusion Weights (Implemented, Not Yet Deployed)

**Query-Adaptive Weights**:
```
[α', β', γ', δ'] = softmax(W_2 × ReLU(W_1 × φ_query + b_1) + b_2)
```

**Architecture**:
- Input: Query embedding φ_query (1024-dim)
- Hidden: 32 neurons, ReLU activation
- Output: 4 weights via softmax
- Xavier initialization

**Training**:
- Online gradient descent from retrieval outcomes
- Learning rate = 0.01
- Cold start: Uses fixed weights until 50 updates

**LaTeX**:
```latex
h = \text{ReLU}(W_1 \phi_q + b_1), \quad w = \text{softmax}(W_2 h + b_2)
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` LearnedFusionWeights class

**Status**: ⚠️ **NOT INTEGRATED** - Exists but not called in recall()

---

### 4.3 Reciprocal Rank Fusion (RRF)

**Purpose**: Merge vector search and keyword search results

**Formula**:
```
RRF_score(d) = Σ_r (1 / (k + rank_r(d)))
```

**Parameters**:
- `r`: Retrieval method (vector, keyword, etc.)
- `k`: Rank constant = 60 (standard value)
- `rank_r(d)`: Rank of document d in retrieval method r

**Example**:
Document ranked #1 in vector, #3 in keyword:
```
RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
```

**LaTeX**:
```latex
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/semantic.py` semantic recall

---

## 5. Learned Memory Gating

### 5.1 Bayesian Logistic Regression

**Purpose**: Predict memory utility to gate storage decisions

**Model**:
```
p(store | φ) = σ(w^T φ + b)
```

Where:
- `φ`: Feature vector (1143-dim)
- `w ~ N(μ, Σ)`: Weight distribution (Bayesian)
- `σ(x) = 1 / (1 + exp(-x))`: Sigmoid function

**LaTeX**:
```latex
p(\text{store} \mid \phi) = \sigma(w^T \phi + b), \quad w \sim \mathcal{N}(\mu, \Sigma)
```

---

### 5.2 Thompson Sampling

**Exploration-Exploitation Balance**:
```
w_sample ~ N(μ, Σ)
p_sample = σ(w_sample^T φ + b)
decision = p_sample ≥ θ_ACh
```

**Norepinephrine Boost** (arousal → exploration):
```
Σ' = Σ × (1 + γ_NE × NE)
```

**LaTeX**:
```latex
w \sim \mathcal{N}(\mu, \Sigma'), \quad \Sigma' = \Sigma(1 + \gamma_{\text{NE}} \cdot \text{NE})
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/learning/learned_gate.py`

---

### 5.3 Online Bayesian Update

**Posterior Update from Utility Signal**:
```
∇ = Σ × ∇_w log p(u | w, φ)
μ' = μ - η × ∇
Σ' = Σ - η × ∇∇^T
```

Where for binary outcome:
```
∇_w log p(u | w, φ) = (σ(w^T φ) - u) × φ
```

**LaTeX**:
```latex
\mu' = \mu - \eta \Sigma \nabla_w \log p(u \mid w, \phi)
```
```latex
\Sigma' = \Sigma - \eta \nabla \nabla^T
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/learning/learned_gate.py` update() method

---

## 6. Neuromodulation

### 6.1 Dopamine (DA) - Reward Prediction Error

**Formula**:
```
DA = utility - baseline
```

**Parameters**:
- `utility`: Observed outcome utility ∈ [0, 1]
- `baseline`: Moving average (α = 0.1)
- `baseline' = 0.9 × baseline + 0.1 × utility`

**Effect**: Modulates learning rate in gate updates

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/learning/neuromodulator_orchestra.py`

---

### 6.2 Serotonin (5-HT) - Long-Term Value

**Time-Integrated Utility**:
```
5-HT = (1 - β) × 5-HT_prev + β × utility_smoothed
```

**Parameters**:
- β (integration rate) = 0.05
- Exponential moving average with long time constant

**Effect**: Stabilizes learning, prevents overreaction to noise

---

### 6.3 Acetylcholine (ACh) - Attention/Novelty

**Entropy-Based Novelty**:
```
ACh = -Σ_i p_i log p_i + novelty_boost
```

**Threshold Modulation**:
```
θ_effective = θ_base × (1 - γ_ACh × ACh)
```

**Effect**: Lowers threshold when attention high (stores more)

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/learning/neuromodulator_orchestra.py`

---

### 6.4 Combined Signal

**Multi-Timescale Integration**:
```
utility_combined = 0.7 × DA + 0.3 × 5-HT
```

**Rationale**:
- DA: Fast adaptation to immediate outcomes
- 5-HT: Slow integration for stable long-term value
- 70/30 split balances reactivity and stability

**LaTeX**:
```latex
u_{\text{comb}} = 0.7 \cdot \text{DA} + 0.3 \cdot \text{5-HT}
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/memory/episodic.py` learn_from_outcome()

---

## 7. Pattern Separation & Clustering

### 7.1 HDBSCAN Density-Based Clustering

**Purpose**: Group similar episodes for consolidation

**Algorithm**: Hierarchical DBSCAN with automatic cluster count

**Distance Metric**: Cosine distance on embeddings
```
d_cosine(u, v) = 1 - (u · v) / (||u|| ||v||)
```

**Parameters**:
- `min_cluster_size`: Minimum episodes per cluster = 3
- `min_samples`: Core point threshold = 2
- `cluster_selection_epsilon`: Merge threshold = 0.0

**LaTeX**:
```latex
d_{\cos}(u, v) = 1 - \frac{u \cdot v}{\|u\| \|v\|}
```

**Implementation**: `/mnt/projects/t4d/t4dm/src/t4dm/consolidation/episode_deduplication.py`

**Evidence**: HDBSCAN superior to k-means for non-spherical clusters ([McInnes et al., 2017](https://joss.theoj.org/papers/10.21105/joss.00205))

---

### 7.2 Orthogonalization (Pattern Separation)

**Purpose**: Reduce interference between similar memories

**Gram-Schmidt Process**:
```
v'_i = v_i - Σ_{j<i} proj(v_i, v'_j)
proj(u, v) = (u · v / ||v||^2) × v
```

**LaTeX**:
```latex
v'_i = v_i - \sum_{j < i} \text{proj}_{v'_j}(v_i)
```
```latex
\text{proj}_v(u) = \frac{u \cdot v}{\|v\|^2} v
```

**Effect**: Decorrelates embeddings within a cluster

**Status**: Planned for Phase 1 (HSA improvements)

---

## 8. Similarity & Fusion

### 8.1 Cosine Similarity

**Definition**:
```
sim(u, v) = (u · v) / (||u|| ||v||)
```

**Properties**:
- Range: [-1, 1]
- 1 = identical direction
- 0 = orthogonal
- -1 = opposite direction

**LaTeX**:
```latex
\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|} = \frac{\sum_i u_i v_i}{\sqrt{\sum_i u_i^2} \sqrt{\sum_i v_i^2}}
```

**Implementation**: Used throughout for vector similarity

---

### 8.2 Euclidean Distance

**Definition**:
```
d(u, v) = ||u - v|| = √(Σ_i (u_i - v_i)^2)
```

**LaTeX**:
```latex
d(u, v) = \|u - v\| = \sqrt{\sum_{i=1}^{n} (u_i - v_i)^2}
```

**Use Case**: HDBSCAN clustering (after cosine distance conversion)

---

### 8.3 Softmax Temperature Scaling

**Purpose**: Control sharpness of probability distributions

**Formula**:
```
p_i = exp(x_i / T) / Σ_j exp(x_j / T)
```

**Parameters**:
- `T`: Temperature
  - `T → 0`: Winner-take-all (argmax)
  - `T = 1`: Standard softmax
  - `T → ∞`: Uniform distribution

**LaTeX**:
```latex
p_i = \frac{e^{x_i / T}}{\sum_j e^{x_j / T}}
```

**Implementation**: Used in learned fusion weights

---

## Mathematical Validation

### Convergence Properties

1. **Hebbian Weights**: Converge to 1.0 asymptotically (proven)
2. **FSRS Stability**: Converges to optimal spacing (empirically validated)
3. **Bayesian Gate**: Posterior converges to true utility function (under standard assumptions)

### Numerical Stability

1. **Log-Sum-Exp Trick**: Used in softmax to prevent overflow
2. **Xavier Initialization**: Prevents gradient explosion/vanishing
3. **Covariance Clipping**: Prevents Σ from becoming singular

### Biological Plausibility

1. **FSRS**: Matches Ebbinghaus forgetting curve
2. **ACT-R**: 30+ years of cognitive science validation
3. **Hebbian**: "Neurons that fire together wire together" (Hebb, 1949)
4. **Neuromodulation**: Matches known DA/5-HT roles in learning

---

## References

1. **FSRS**: Jarrett Ye et al. (2024). "A Stochastic Shortest Path Algorithm for Optimizing Spaced Repetition Scheduling"
2. **ACT-R**: Anderson, J.R. (2007). "How Can the Human Mind Occur in the Physical Universe?"
3. **Hebbian Learning**: Hebb, D.O. (1949). "The Organization of Behavior"
4. **HDBSCAN**: McInnes, L., Healy, J., Astels, S. (2017). "hdbscan: Hierarchical density based clustering"
5. **Neuromodulation**: Schultz, W. (2002). "Getting Formal with Dopamine and Reward"
6. **Thompson Sampling**: Chapelle, O., Li, L. (2011). "An Empirical Evaluation of Thompson Sampling"

---

## Formula Index

| Formula | Section | Implementation |
|---------|---------|----------------|
| `R(t, S) = (1 + 0.9t/S)^(-0.5)` | 1.1 | Episode.retrievability() |
| `w' = w + η(1 - w)` | 2.1 | SemanticMemory.strengthen_connection() |
| `B_i = ln(Σ t_j^(-d))` | 3.1 | ACTRRetrieval.base_level() |
| `A_spread = Σ W_j S_ji` | 3.2 | ACTRRetrieval.spreading() |
| `p = σ(w^T φ + b)` | 5.1 | LearnedMemoryGate.predict() |
| `w ~ N(μ, Σ')` | 5.2 | LearnedMemoryGate (Thompson) |
| `DA = u - baseline` | 6.1 | NeuromodulatorOrchestra |
| `RRF = Σ 1/(k + rank)` | 4.3 | SemanticMemory.semantic_recall() |

---

**Document Status**: Complete ✓
**Last Verified**: 2025-12-06 against production code
**Next Review**: When new mathematical models added
