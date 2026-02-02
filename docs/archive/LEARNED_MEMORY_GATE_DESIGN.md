# LearnedMemoryGate Algorithm Design

**Document Version**: 1.0
**Date**: 2025-12-06
**Author**: T4DM Algorithm Design Agent

---

## 1. Mathematical Formulation

### 1.1 Problem Statement

At memory encoding time, predict the probability that a memory will be both:
1. Retrieved in the future (P(retrieval))
2. Lead to positive outcomes when retrieved (P(positive | retrieval))

**Objective**: Learn a gating function `g(x, c, n) → [0, 1]` where:
- `x`: Content embedding (vector ∈ R^d)
- `c`: Context features (project, task, temporal state)
- `n`: Neuromodulator state (DA, 5-HT, NE, ACh)

**Output**: Storage probability `p = g(x, c, n)`

**Decision Rule**:
```
if p ≥ θ_store:     STORE
elif p ≥ θ_buffer:  BUFFER (for later review)
else:               SKIP
```

### 1.2 Target Label Construction

The fundamental challenge: **we only observe positive examples** (retrieved memories).

**Solution**: Define a composite utility score after retrieval occurs:

```
U(m, t) = α·I(retrieved at t) + β·V_DA(m) + γ·V_5HT(m) + δ·R(m, t)

where:
  I(retrieved) = 1 if memory m was retrieved, 0 otherwise
  V_DA(m)      = Dopamine expected value (immediate utility)
  V_5HT(m)     = Serotonin long-term value (delayed utility)
  R(m, t)      = Recency-weighted access count

  α, β, γ, δ ∈ [0, 1], α + β + γ + δ = 1
```

**Recommended weights**:
- α = 0.4 (retrieval is primary signal)
- β = 0.3 (immediate value matters)
- γ = 0.2 (long-term value matters)
- δ = 0.1 (recency/frequency bonus)

### 1.3 Training Signal Generation

**Positive examples** (U ≥ 0.5): Memories that were useful
**Negative examples** (U < 0.3): Construct via:

1. **Temporal negative sampling**: Old memories never retrieved (t > 7 days)
2. **Low-value memories**: Retrieved but V_DA, V_5HT < 0.3
3. **Explicit deletions**: User-deleted memories
4. **Failed retrievals**: Retrieved but outcome = 0

**Label smoothing** to handle uncertainty:
```
y_target = 0.1 + 0.8 · U(m)  // Maps [0, 1] → [0.1, 0.9]
```

This prevents overconfident predictions and allows for uncertainty.

---

## 2. Algorithm Choice: Online Linear Model with Thompson Sampling

### 2.1 Why Not Neural Networks?

| Requirement | Neural Network | Linear Model |
|-------------|----------------|--------------|
| Online learning | Requires experience replay | Native online updates |
| Low latency | 5-10ms forward pass | <1ms forward pass |
| Cold start | Poor without pretraining | Reasonable with priors |
| Interpretability | Black box | Weight inspection |
| Memory overhead | Model + replay buffer | Weights only |

**Decision**: Start with online logistic regression, upgrade to neural if needed.

### 2.2 Model Architecture

**Feature representation**:
```
φ(x, c, n) = [
  x,                    // Content embedding (1024-dim)
  embed_context(c),     // Project + task encoding (64-dim)
  n.dopamine_rpe,       // Scalar
  n.norepinephrine_gain,// Scalar
  n.serotonin_mood,     // Scalar
  n.acetylcholine_mode, // One-hot (3-dim)
  temporal_features(t), // Time-of-day, day-of-week (16-dim)
  interaction_features  // x · n (cross-product, 32-dim)
]
```

**Total dimensionality**: ~1139 dimensions

**Model**:
```
p = σ(w^T φ(x, c, n) + b)

where σ(z) = 1 / (1 + exp(-z))  // Logistic sigmoid
```

### 2.3 Thompson Sampling for Exploration

Instead of deterministic prediction, sample from posterior:
```
w_t ~ N(μ_t, Σ_t)  // Sample weights from posterior
p_t = σ(w_t^T φ)    // Predict with sampled weights
```

This provides:
1. **Automatic exploration**: Uncertainty → wider sampling → occasional storage
2. **Confidence-aware decisions**: High certainty → consistent predictions
3. **Graceful uncertainty handling**: Unknown contexts → wider exploration

**Update rule** (Online Bayesian Logistic Regression):
```
// After observing (φ_t, y_t):
Σ_t+1^(-1) = Σ_t^(-1) + λ·φ_t φ_t^T  // Precision update (Hessian approx)
μ_t+1 = μ_t + η·Σ_t+1·∇L(w; φ_t, y_t) // Mean update (gradient step)

where:
  λ = learning rate for precision (controls exploration decay)
  η = learning rate for mean (controls adaptation speed)
  ∇L = gradient of log-loss
```

**Practical approximation** (diagonal covariance):
```
Σ_t = diag(σ_1^2, ..., σ_d^2)  // Diagonal only (saves memory)
```

### 2.4 Cold Start Strategy

**Initialize with informed priors**:

```python
# Feature means (zero-initialized)
μ_0 = zeros(d)

# Feature variances (high uncertainty)
Σ_0 = diag([
  0.01,  # Content embedding (low variance, embeddings are informative)
  0.05,  # Context features (medium variance)
  0.1,   # Neuromodulator states (high variance, need to learn)
  ...
])

# Bias initialization
b_0 = logit(0.5)  # Start at 50% probability
```

**Fallback heuristics** (first 100 decisions):
```python
if n_observations < 100:
    p_learned = σ(w^T φ)
    p_heuristic = rule_based_gate.evaluate(content, context)  # Existing gate

    # Blend: Start with heuristics, fade to learned
    α = min(1.0, n_observations / 100)
    p_final = (1 - α) · p_heuristic + α · p_learned
```

This ensures reasonable behavior during the learning phase.

---

## 3. Pseudocode

### 3.1 Initialization

```python
class LearnedMemoryGate:
    def __init__(self, feature_dim: int = 1139):
        # Model parameters
        self.μ = np.zeros(feature_dim)  # Weight mean
        self.Σ = np.diag([0.01] * 1024 + [0.05] * 64 + [0.1] * 51)  # Variance
        self.b = 0.0  # Bias

        # Hyperparameters
        self.λ_precision = 0.01  # Precision learning rate
        self.η_mean = 0.1        # Mean learning rate
        self.θ_store = 0.6       # Storage threshold
        self.θ_buffer = 0.3      # Buffer threshold

        # Cold start
        self.n_observations = 0
        self.cold_start_threshold = 100
        self.fallback_gate = RuleBasedMemoryGate()  # Existing gate

        # Training buffer
        self.pending_labels: Dict[UUID, Tuple[np.ndarray, datetime]] = {}

        # Statistics
        self.decisions = {"store": 0, "buffer": 0, "skip": 0}
        self.accuracy_buffer = deque(maxlen=1000)
```

### 3.2 Prediction (Forward Pass)

```python
def predict(
    self,
    content_embedding: np.ndarray,
    context: GateContext,
    neuromod_state: NeuromodulatorState,
    explore: bool = True
) -> GateDecision:
    """
    Predict storage probability with optional exploration.

    Args:
        content_embedding: Content vector (1024-dim)
        context: Session context
        neuromod_state: Current neuromodulator state
        explore: If True, use Thompson sampling

    Returns:
        GateDecision with probability and action
    """
    # 1. Feature extraction
    φ = self._extract_features(content_embedding, context, neuromod_state)

    # 2. Sample weights (Thompson sampling)
    if explore and self.n_observations > 0:
        w = np.random.multivariate_normal(self.μ, self.Σ)
    else:
        w = self.μ  # Use mean for deterministic prediction

    # 3. Compute probability
    logit = np.dot(w, φ) + self.b
    p_learned = sigmoid(logit)

    # 4. Cold start blending
    if self.n_observations < self.cold_start_threshold:
        p_heuristic = self.fallback_gate.evaluate(
            content_embedding, context
        ).score

        α = self.n_observations / self.cold_start_threshold
        p = (1 - α) * p_heuristic + α * p_learned
    else:
        p = p_learned

    # 5. Make decision
    if p >= self.θ_store:
        action = StorageDecision.STORE
    elif p >= self.θ_buffer:
        action = StorageDecision.BUFFER
    else:
        action = StorageDecision.SKIP

    self.decisions[action.value] += 1

    return GateDecision(
        action=action,
        probability=p,
        features=φ,  # Save for later training
        timestamp=datetime.now()
    )
```

### 3.3 Feature Extraction

```python
def _extract_features(
    self,
    content_embedding: np.ndarray,
    context: GateContext,
    neuromod_state: NeuromodulatorState
) -> np.ndarray:
    """Extract feature vector φ(x, c, n)."""

    # Content embedding (1024-dim)
    content_features = content_embedding

    # Context encoding (64-dim)
    project_embed = self._embed_string(context.project or "default", dim=32)
    task_embed = self._embed_string(context.current_task or "general", dim=32)
    context_features = np.concatenate([project_embed, task_embed])

    # Neuromodulator features (7-dim)
    neuromod_features = np.array([
        neuromod_state.dopamine_rpe,
        neuromod_state.norepinephrine_gain,
        neuromod_state.serotonin_mood,
        1.0 if neuromod_state.acetylcholine_mode == "encoding" else 0.0,
        1.0 if neuromod_state.acetylcholine_mode == "balanced" else 0.0,
        1.0 if neuromod_state.acetylcholine_mode == "retrieval" else 0.0,
        neuromod_state.inhibition_sparsity
    ])

    # Temporal features (16-dim)
    now = datetime.now()
    hour_sin = np.sin(2 * np.pi * now.hour / 24)
    hour_cos = np.cos(2 * np.pi * now.hour / 24)
    day_sin = np.sin(2 * np.pi * now.weekday() / 7)
    day_cos = np.cos(2 * np.pi * now.weekday() / 7)

    time_since_last = 0.0
    if context.last_store_time:
        elapsed = (now - context.last_store_time).total_seconds() / 3600
        time_since_last = np.clip(elapsed / 24, 0, 1)  # Normalize to [0, 1]

    temporal_features = np.array([
        hour_sin, hour_cos, day_sin, day_cos, time_since_last,
        *np.zeros(11)  # Reserved for future temporal features
    ])

    # Interaction features (32-dim) - cross products
    # Use content embedding mean with neuromodulator states
    content_mean = content_embedding.mean()
    interactions = np.array([
        content_mean * neuromod_state.dopamine_rpe,
        content_mean * neuromod_state.norepinephrine_gain,
        content_mean * neuromod_state.serotonin_mood,
        *np.zeros(29)  # Reserved
    ])

    # Concatenate all features
    φ = np.concatenate([
        content_features,
        context_features,
        neuromod_features,
        temporal_features,
        interactions
    ])

    return φ
```

### 3.4 Online Update (Training)

```python
def update(self, memory_id: UUID, utility: float) -> None:
    """
    Online update after observing memory utility.

    Args:
        memory_id: ID of memory we predicted for
        utility: Observed utility score [0, 1]
    """
    # 1. Retrieve stored features
    if memory_id not in self.pending_labels:
        logger.warning(f"No features found for {memory_id}")
        return

    φ, timestamp = self.pending_labels.pop(memory_id)

    # 2. Construct target with label smoothing
    y = 0.1 + 0.8 * utility

    # 3. Compute prediction
    logit = np.dot(self.μ, φ) + self.b
    p = sigmoid(logit)

    # 4. Compute gradient (logistic regression)
    error = p - y
    ∇w = error * φ
    ∇b = error

    # 5. Update precision (Σ^(-1) ← Σ^(-1) + λ·φφ^T)
    # For diagonal approximation:
    Σ_inv = np.diag(1.0 / np.diag(self.Σ))
    Σ_inv += self.λ_precision * np.outer(φ, φ)
    self.Σ = np.diag(1.0 / np.diag(Σ_inv))  # Invert back

    # 6. Update mean (μ ← μ - η·Σ·∇w)
    self.μ -= self.η_mean * self.Σ @ ∇w
    self.b -= self.η_mean * ∇b

    # 7. Update statistics
    self.n_observations += 1
    accuracy = 1.0 - abs(p - y)
    self.accuracy_buffer.append(accuracy)

    logger.debug(
        f"Updated gate: n={self.n_observations}, "
        f"error={error:.3f}, accuracy={accuracy:.3f}"
    )
```

### 3.5 Negative Sampling

```python
def generate_negative_samples(
    self,
    memory_store: MemoryStore,
    lookback_days: int = 7
) -> List[Tuple[UUID, float]]:
    """
    Generate negative training examples from old, unused memories.

    Args:
        memory_store: Access to stored memories
        lookback_days: Consider memories older than this

    Returns:
        List of (memory_id, utility_score) for training
    """
    cutoff = datetime.now() - timedelta(days=lookback_days)
    negatives = []

    # Query memories created before cutoff
    old_memories = memory_store.query_by_date(before=cutoff, limit=1000)

    for memory in old_memories:
        # Check if never retrieved
        retrieval_count = memory_store.get_retrieval_count(memory.id)

        if retrieval_count == 0:
            # Never retrieved → utility = 0.0
            negatives.append((memory.id, 0.0))

        elif retrieval_count > 0:
            # Retrieved but check value signals
            da_value = self.neuromod.dopamine.get_expected_value(memory.id)
            sht_value = self.neuromod.serotonin.get_long_term_value(memory.id)

            # If both are low, treat as negative
            if da_value < 0.3 and sht_value < 0.3:
                utility = 0.5 * (da_value + sht_value)
                negatives.append((memory.id, utility))

    return negatives
```

### 3.6 Batch Training (Periodic)

```python
def batch_train(
    self,
    memory_store: MemoryStore,
    n_epochs: int = 1
) -> Dict[str, float]:
    """
    Periodic batch training on accumulated data.

    Called during low-activity periods to:
    1. Process negative samples
    2. Reweight features based on importance
    3. Prune low-confidence predictions

    Args:
        memory_store: Access to memories
        n_epochs: Number of passes over data

    Returns:
        Training statistics
    """
    # Generate negative samples
    negatives = self.generate_negative_samples(memory_store)

    # Retrieve positive samples from neuromodulator values
    positives = []
    for mem_id_str, value in self.neuromod.serotonin._long_term_values.items():
        if value > 0.5:
            positives.append((UUID(mem_id_str), value))

    # Combine and shuffle
    all_samples = positives + negatives
    random.shuffle(all_samples)

    # Train
    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for memory_id, utility in all_samples:
            # Retrieve features
            memory = memory_store.get(memory_id)
            if memory is None:
                continue

            φ = self._extract_features(
                memory.embedding,
                memory.context,
                self.neuromod.get_current_state()
            )

            # Update (same as online update)
            y = 0.1 + 0.8 * utility
            logit = np.dot(self.μ, φ) + self.b
            p = sigmoid(logit)

            error = p - y
            epoch_loss += error ** 2

            # Gradient update
            self.μ -= self.η_mean * error * φ
            self.b -= self.η_mean * error

        epoch_loss /= len(all_samples)
        losses.append(epoch_loss)

    return {
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "final_loss": losses[-1] if losses else 0.0,
        "avg_loss": np.mean(losses) if losses else 0.0
    }
```

---

## 4. Complexity Analysis

### 4.1 Time Complexity

**Forward Pass** (prediction):
```
Feature extraction: O(d)  where d = feature_dim = 1139
  - Content: O(1) (already embedded)
  - Context: O(1) (hash-based embedding)
  - Neuromodulator: O(1)
  - Temporal: O(1)
  - Interactions: O(1)

Weight sampling: O(d²) for full covariance, O(d) for diagonal
  - Diagonal approximation: O(d) ✓

Dot product: O(d)

Total: O(d) ≈ O(1139) ≈ 1-2ms on modern CPU
```

**Backward Pass** (update):
```
Gradient computation: O(d)
Precision update: O(d²) full, O(d) diagonal
Mean update: O(d²) full, O(d) diagonal

Total (diagonal): O(d) ≈ O(1139) ≈ 1-2ms
```

**Total latency per decision**: ~2-4ms (negligible compared to embedding generation)

### 4.2 Space Complexity

**Model parameters**:
```
μ: d floats             = 1139 × 8 bytes ≈ 9 KB
Σ (diagonal): d floats  = 1139 × 8 bytes ≈ 9 KB
b: 1 float              = 8 bytes

Total: ~18 KB
```

**Pending labels buffer** (for delayed feedback):
```
Assume 1000 pending predictions
Features: 1000 × 1139 × 8 bytes ≈ 9 MB
Timestamps: 1000 × 8 bytes ≈ 8 KB

Total: ~9 MB
```

**Total memory footprint**: ~9 MB (very lightweight)

### 4.3 Scalability

| Metric | Value | Bottleneck? |
|--------|-------|-------------|
| Latency per decision | 2-4ms | No (embedding is 50-100ms) |
| Memory overhead | 9 MB | No (negligible vs Neo4j/Qdrant) |
| Throughput | ~250-500 decisions/sec | No (typical load << 10/sec) |
| Training latency | 1-2ms/sample | No (async background) |

**Conclusion**: Algorithm is lightweight and will not bottleneck the system.

---

## 5. Cold Start Strategy

### 5.1 Phase 1: Heuristic-Dominant (0-100 observations)

```python
# Blend learned model with rule-based heuristics
α = min(1.0, n_observations / 100)
p_final = (1 - α) · p_heuristic + α · p_learned

# This ensures:
# - 0 obs: 100% heuristic (existing gate)
# - 50 obs: 50% blend
# - 100 obs: 100% learned
```

**Heuristic sources**:
1. Existing `MemoryGate` rule-based system (pattern matching)
2. Entity density
3. Outcome indicators
4. Explicit user triggers ("remember this")

### 5.2 Phase 2: Exploration-Heavy (100-1000 observations)

```python
# Increase Thompson sampling variance
exploration_boost = max(1.0, 3.0 - n_observations / 500)
Σ_boosted = exploration_boost · Σ

# Sample from boosted posterior
w ~ N(μ, Σ_boosted)
```

This causes wider exploration during learning phase, then tightens as confidence grows.

### 5.3 Phase 3: Confident (1000+ observations)

```python
# Use standard Thompson sampling
w ~ N(μ, Σ)

# Periodically validate performance
if n_observations % 1000 == 0:
    accuracy = self.validate_holdout()
    if accuracy < 0.6:
        logger.warning("Gate performance degraded, increasing exploration")
        self.Σ *= 1.5  # Boost uncertainty
```

### 5.4 Initialization Priors

**Informative priors** from domain knowledge:

```python
# Bias toward storing when:
# 1. High neuromodulator activity (novel, important)
# 2. Explicit encoding mode (ACh)
# 3. High emotional valence (if available)

# Initialize weights with slight positive bias for these features
μ_0[dopamine_rpe_idx] = 0.5     # Positive RPE → store
μ_0[ne_gain_idx] = 0.3          # High arousal → store
μ_0[ach_encoding_idx] = 0.4     # Encoding mode → store
μ_0[sht_mood_idx] = 0.1         # Slight mood bias

# High variance (uncertain)
Σ_0 = 0.1 · I  # Identity matrix scaled
```

---

## 6. Integration with Neuromodulators

### 6.1 Dopamine Integration

**Dopamine provides immediate feedback**:

```python
# After retrieval with outcome
rpe = dopamine.compute_rpe(memory_id, outcome)

# Use RPE surprise magnitude as utility component
utility = 0.3 · dopamine.get_expected_value(memory_id) +
          0.7 · (outcome if rpe.surprise_magnitude > 0.1 else 0.5)

# Update gate
learned_gate.update(memory_id, utility)
```

This prioritizes learning from surprising outcomes (high |δ|).

### 6.2 Serotonin Integration

**Serotonin provides long-term value**:

```python
# Periodically (e.g., session end)
for memory_id in session_memories:
    long_term_value = serotonin.get_long_term_value(memory_id)

    # Update gate with delayed feedback
    utility = 0.5 · immediate_utility + 0.5 · long_term_value
    learned_gate.update(memory_id, utility)
```

This ensures memories valuable across sessions are learned.

### 6.3 Norepinephrine Integration

**NE modulates exploration** (already handled by Thompson sampling):

```python
# During prediction
if neuromod_state.norepinephrine_gain > 1.5:
    # High arousal → more exploration
    Σ_temp = 1.5 · self.Σ
    w ~ N(μ, Σ_temp)
else:
    # Normal arousal → standard sampling
    w ~ N(μ, Σ)
```

### 6.4 Acetylcholine Integration

**ACh signals encoding mode**:

```python
# ACh state is a feature in φ(x, c, n)
# Model learns: encoding mode → higher storage probability

# Additionally, can use ACh to modulate threshold
if neuromod_state.acetylcholine_mode == "encoding":
    θ_store_adjusted = 0.5  # Lower threshold (easier to store)
elif neuromod_state.acetylcholine_mode == "retrieval":
    θ_store_adjusted = 0.7  # Higher threshold (harder to store)
else:
    θ_store_adjusted = 0.6  # Balanced
```

---

## 7. Training Loop Integration

### 7.1 Encoding-Time Decision

```python
# In EpisodicMemory.create_episode()

def create_episode(self, content: str, context: GateContext) -> Optional[Episode]:
    # 1. Generate embedding
    embedding = self.embedding_service.embed(content)

    # 2. Get neuromodulator state
    neuromod_state = self.neuromod.process_query(
        embedding,
        is_question=False
    )

    # 3. Gate decision
    decision = self.learned_gate.predict(
        content_embedding=embedding,
        context=context,
        neuromod_state=neuromod_state,
        explore=True  # Use Thompson sampling
    )

    if decision.action == StorageDecision.SKIP:
        logger.debug(f"Gate skipped storage: p={decision.probability:.3f}")
        return None

    # 4. Store episode
    episode = Episode(
        content=content,
        embedding=embedding,
        importance=decision.probability,  # Use gate confidence
        ...
    )

    episode_id = self.storage.create(episode)

    # 5. Save features for later training
    self.learned_gate.pending_labels[episode_id] = (
        decision.features,
        datetime.now()
    )

    return episode
```

### 7.2 Retrieval-Time Feedback

```python
# In EpisodicMemory.recall()

def recall(self, query: str, top_k: int = 10) -> List[Episode]:
    # ... retrieval logic ...

    retrieved_episodes = self.storage.search(query_embedding, top_k)

    # Add eligibility traces (for serotonin)
    for episode in retrieved_episodes:
        self.neuromod.serotonin.add_eligibility(
            episode.id,
            strength=episode.score
        )

    return retrieved_episodes
```

### 7.3 Outcome-Time Training

```python
# In NeuromodulatorOrchestra.process_outcome()

def process_outcome(
    self,
    memory_outcomes: Dict[str, float],
    session_outcome: Optional[float] = None
) -> Dict[str, float]:
    # ... existing neuromodulator updates ...

    # Compute composite utility for gate training
    for mem_id_str, immediate_outcome in memory_outcomes.items():
        mem_id = UUID(mem_id_str)

        # Combine signals
        da_value = self.dopamine.get_expected_value(mem_id)
        sht_value = self.serotonin.get_long_term_value(mem_id)

        utility = (
            0.4 * 1.0 +  # Was retrieved (positive signal)
            0.3 * da_value +
            0.2 * sht_value +
            0.1 * immediate_outcome
        )

        # Update learned gate
        self.learned_gate.update(mem_id, utility)

    return learning_signals
```

### 7.4 Background Training (Periodic)

```python
# Scheduled task (e.g., nightly at 3 AM)

async def nightly_gate_training():
    """Background training on accumulated data."""

    # 1. Generate negative samples
    stats = learned_gate.batch_train(
        memory_store=t4dx_graph_adapter,
        n_epochs=3
    )

    logger.info(
        f"Nightly training complete: "
        f"pos={stats['n_positives']}, "
        f"neg={stats['n_negatives']}, "
        f"loss={stats['final_loss']:.4f}"
    )

    # 2. Validate performance
    accuracy = validate_gate_performance()

    # 3. Adjust exploration if needed
    if accuracy < 0.6:
        learned_gate.Σ *= 1.3  # Increase exploration
    elif accuracy > 0.85:
        learned_gate.Σ *= 0.9  # Decrease exploration
```

---

## 8. Performance Metrics

### 8.1 Primary Metrics

1. **Precision at storage**: Of memories stored, what fraction was useful?
   ```
   Precision = |{stored ∩ useful}| / |{stored}|
   ```

2. **Recall of useful memories**: Of useful memories, what fraction was stored?
   ```
   Recall = |{stored ∩ useful}| / |{useful}|
   ```

3. **F1 score**: Harmonic mean
   ```
   F1 = 2 · (Precision · Recall) / (Precision + Recall)
   ```

### 8.2 Operational Metrics

1. **Storage efficiency**: Reduction in stored memories vs baseline
   ```
   Efficiency = 1 - (n_stored_learned / n_stored_baseline)
   ```

2. **Decision latency**: Time to make gate decision
   ```
   Target: < 5ms (p99)
   ```

3. **False negative rate**: Useful memories incorrectly skipped
   ```
   FNR = |{useful \ stored}| / |{useful}|
   Target: < 10%
   ```

### 8.3 Learning Metrics

1. **Calibration error**: How well do probabilities match outcomes?
   ```
   ECE = Σ |p_predicted - p_empirical| / n_bins
   ```

2. **Uncertainty quality**: Does uncertainty correlate with errors?
   ```
   Correlation(σ_prediction, |error|)
   Target: > 0.5
   ```

3. **Exploration effectiveness**: Diversity of stored memories
   ```
   Diversity = avg_pairwise_distance(stored_embeddings)
   ```

---

## 9. Implementation Checklist

- [ ] Core `LearnedMemoryGate` class
  - [ ] Feature extraction pipeline
  - [ ] Thompson sampling prediction
  - [ ] Online Bayesian logistic regression update
  - [ ] Negative sampling logic
  - [ ] Batch training method

- [ ] Integration points
  - [ ] Hook into `EpisodicMemory.create_episode()`
  - [ ] Hook into `NeuromodulatorOrchestra.process_outcome()`
  - [ ] Add eligibility trace tracking
  - [ ] Background training scheduler

- [ ] Cold start strategy
  - [ ] Heuristic blending (0-100 obs)
  - [ ] Exploration boost (100-1000 obs)
  - [ ] Informed prior initialization
  - [ ] Fallback to existing `MemoryGate`

- [ ] Monitoring and metrics
  - [ ] Precision/recall tracking
  - [ ] Calibration monitoring
  - [ ] Decision latency profiling
  - [ ] A/B test framework (learned vs rule-based)

- [ ] Tests
  - [ ] Unit tests for feature extraction
  - [ ] Unit tests for update logic
  - [ ] Integration test with neuromodulators
  - [ ] Performance benchmarks
  - [ ] Cold start behavior tests

---

## 10. Future Enhancements

### 10.1 Neural Network Upgrade Path

If linear model plateaus:

```python
# Small MLP for nonlinear patterns
class NeuralGate(nn.Module):
    def __init__(self, feature_dim: int = 1139):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, φ):
        return self.net(φ)

# Use online gradient descent with experience replay
```

### 10.2 Multi-Objective Optimization

Optimize for multiple goals:
```
L = λ_1 · BCE(p, y) +           # Accuracy
    λ_2 · |stored_count - budget| +  # Storage budget
    λ_3 · diversity_loss            # Coverage diversity
```

### 10.3 Contextual Bandits

Frame as contextual bandit for tighter exploration:
```python
# LinUCB algorithm
p_mean = w^T φ
p_ucb = p_mean + β · sqrt(φ^T Σ φ)  # Upper confidence bound

if p_ucb >= θ_store:
    STORE
```

### 10.4 Meta-Learning

Learn to adapt gate for different users/contexts:
```python
# MAML-style meta-learning
for user in users:
    w_user = w_base + adapt(user_data, n_steps=5)
    # Use w_user for this user's decisions
```

---

## 11. References

1. **Thompson Sampling**: Russo et al. (2018) "A Tutorial on Thompson Sampling"
2. **Online Learning**: Hazan et al. (2016) "Introduction to Online Convex Optimization"
3. **Bayesian Logistic Regression**: Murphy (2012) "Machine Learning: A Probabilistic Perspective"
4. **Eligibility Traces**: Sutton & Barto (2018) "Reinforcement Learning: An Introduction"
5. **Neuromodulation**: Doya (2002) "Metalearning and neuromodulation"

---

**End of Design Document**
