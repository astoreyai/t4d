# Learning System Walkthrough

**Version**: 0.1.0
**Last Updated**: 2025-12-09

This document covers T4DM's learning and plasticity mechanisms.

---

## Table of Contents

1. [Learning Philosophy](#learning-philosophy)
2. [Plasticity Mechanisms](#plasticity-mechanisms)
3. [Reconsolidation](#reconsolidation)
4. [Learned Components](#learned-components)
5. [Training Loops](#training-loops)
6. [Cold Start & Warm Start](#cold-start--warm-start)
7. [File Reference](#file-reference)

---

## Learning Philosophy

T4DM follows these biological principles:

### 1. Memories Update on Retrieval

Unlike traditional databases, memories are not frozen after storage. Each retrieval is an opportunity to update based on current context:

```
Retrieval → Reconsolidation → Updated embedding
```

### 2. Learning Scales with Surprise

Dopamine-modulated learning means:
- Unexpected outcomes (high RPE) → Strong learning
- Expected outcomes (low RPE) → Minimal learning

### 3. Storage Decisions are Learned

The memory gate learns what to store based on retrieval outcomes:
- Stored memory retrieved successfully → Reinforce storage decision
- Stored memory never retrieved → Weaken storage tendency

### 4. Complementary Learning Systems (CLS)

Two learning rates for different memory types:

| System | Learning Rate | Purpose |
|--------|--------------|---------|
| Hippocampal (Episodic) | 0.1 (fast) | One-shot learning, specific episodes |
| Neocortical (Semantic) | 0.01 (slow) | Gradual abstraction, generalization |

---

## Plasticity Mechanisms

### 1. Three-Factor Learning Rule

**File**: `learning/three_factor.py`

The core learning principle:

```python
Δw = eligibility × neuromodulator × dopamine
```

| Factor | Meaning | Source |
|--------|---------|--------|
| Eligibility | Temporal credit | Serotonin traces |
| Neuromodulator | Learning mode | ACh, NE state |
| Dopamine | Surprise signal | RPE magnitude |

```python
class ThreeFactorLearningRule:
    def compute_weight_update(
        self,
        eligibility: float,
        neuromod_state: NeuromodulatorState,
        dopamine_rpe: float,
        base_lr: float = 0.01
    ) -> float:
        # All factors gate each other
        update = (
            base_lr *
            eligibility *
            neuromod_state.effective_learning_rate *
            abs(dopamine_rpe)
        )

        # Sign from dopamine
        if dopamine_rpe < 0:
            update *= -1  # LTD (weakening)

        return update
```

### 2. Eligibility Traces

**File**: `learning/eligibility.py`

Temporal credit assignment - "what was active when outcome arrived?"

```python
class EligibilityTraceManager:
    def add_trace(self, memory_id: UUID, strength: float):
        """Mark memory as eligible for credit."""
        self.traces[memory_id] = EligibilityTrace(
            memory_id=memory_id,
            strength=strength,
            timestamp=datetime.now()
        )

    def get_decayed_eligibility(self, memory_id: UUID) -> float:
        """Get eligibility with temporal decay."""
        trace = self.traces.get(memory_id)
        if not trace:
            return 0.0

        age_seconds = (datetime.now() - trace.timestamp).total_seconds()
        decay = np.exp(-age_seconds / self.decay_constant)

        return trace.strength * decay

    def distribute_credit(self, outcome: float) -> Dict[UUID, float]:
        """Assign credit proportional to decayed eligibility."""
        credits = {}
        for mem_id, trace in self.traces.items():
            eligibility = self.get_decayed_eligibility(mem_id)
            credits[mem_id] = outcome * eligibility
        return credits
```

### 3. Homeostatic Plasticity

**File**: `learning/homeostatic.py`

Prevents runaway excitation/inhibition:

```python
class HomeostaticPlasticity:
    def __init__(self, target_activity: float = 0.5, tau: float = 1000.0):
        self.target = target_activity
        self.tau = tau  # Time constant

    def compute_scaling_factor(self, current_activity: float) -> float:
        """Synaptic scaling to maintain target activity."""
        error = self.target - current_activity
        scaling = 1.0 + error / self.tau
        return np.clip(scaling, 0.5, 2.0)

    def apply_scaling(self, weights: np.ndarray, activity: float) -> np.ndarray:
        """Scale all weights toward target activity."""
        factor = self.compute_scaling_factor(activity)
        return weights * factor
```

**Biological Basis**: Turrigiano (2008) - TNFα-mediated synaptic scaling

### 4. FSRS Decay

**File**: `learning/fsrs.py`

Spaced Repetition System for memory retention:

```python
class FSRSScheduler:
    def compute_retrievability(
        self,
        stability: float,
        elapsed_days: float
    ) -> float:
        """Compute probability of successful recall."""
        return np.exp(-elapsed_days / stability)

    def update_stability(
        self,
        old_stability: float,
        rating: int,  # 1=Again, 2=Hard, 3=Good, 4=Easy
        retrievability: float
    ) -> float:
        """Update stability based on review outcome."""
        # Successful recall at low retrievability = big stability boost
        if rating >= 3:  # Good or Easy
            difficulty_bonus = (1 - retrievability) * self.difficulty_factor
            new_stability = old_stability * (1 + difficulty_bonus)
        else:
            # Failed recall = stability decrease
            new_stability = old_stability * self.lapse_penalty

        return new_stability
```

---

## Reconsolidation

**File**: `learning/reconsolidation.py`

When memories are retrieved, they become labile and can be updated:

```python
class ReconsolidationEngine:
    def __init__(
        self,
        base_learning_rate: float = 0.1,
        max_update_magnitude: float = 0.2,
        cooldown_hours: float = 0.5
    ):
        self.base_lr = base_learning_rate
        self.max_magnitude = max_update_magnitude
        self.cooldown = timedelta(hours=cooldown_hours)
        self.last_update: Dict[UUID, datetime] = {}

    async def update(
        self,
        memory_id: UUID,
        current_embedding: np.ndarray,
        context_embedding: np.ndarray,
        learning_signal: float,
        vector_store: T4DXVectorAdapter
    ) -> Optional[np.ndarray]:
        """Update memory embedding based on retrieval context."""
        # Check cooldown
        if self._in_cooldown(memory_id):
            return None

        # Compute update direction (toward context)
        direction = context_embedding - current_embedding
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Compute update magnitude
        magnitude = self.base_lr * learning_signal
        magnitude = min(magnitude, self.max_magnitude)

        # Apply update
        new_embedding = current_embedding + magnitude * direction
        new_embedding = new_embedding / np.linalg.norm(new_embedding)

        # Update in vector store
        await vector_store.update_vector(
            collection=vector_store.episodes_collection,
            id=str(memory_id),
            vector=new_embedding.tolist()
        )

        self.last_update[memory_id] = datetime.now()
        return new_embedding
```

**When Reconsolidation Happens**:
1. Memory is retrieved
2. Outcome is observed (success/failure)
3. If surprise is high (RPE > threshold)
4. And cooldown has passed
5. → Update embedding toward retrieval context

---

## Learned Components

### 1. Learned Memory Gate

**File**: `core/learned_gate.py`

Bayesian logistic regression + Thompson sampling:

```python
class LearnedMemoryGate:
    def predict(
        self,
        content_embedding: np.ndarray,
        context: GateContext,
        neuromod_state: NeuromodulatorState,
        explore: bool = True
    ) -> GateDecision:
        # Extract features
        features = self._extract_features(content_embedding, context, neuromod_state)

        # Bayesian prediction with uncertainty
        mean = self.weights @ features
        variance = features @ self.covariance @ features
        std = np.sqrt(variance)

        if explore:
            # Thompson sampling: sample from posterior
            sample = np.random.normal(mean, std)
            exploration_boost = sample - mean
        else:
            sample = mean
            exploration_boost = 0.0

        probability = self._sigmoid(sample)

        # Decision thresholds
        if probability > self.store_threshold:
            action = GateAction.STORE
        elif probability > self.buffer_threshold:
            action = GateAction.BUFFER
        else:
            action = GateAction.SKIP

        return GateDecision(
            action=action,
            probability=probability,
            exploration_boost=exploration_boost,
            features=features
        )

    def update(self, memory_id: UUID, outcome: float):
        """Bayesian update from retrieval outcome."""
        features = self.pending_labels.pop(memory_id)

        # Online Bayesian logistic regression update
        pred = self._sigmoid(self.weights @ features)
        gradient = (outcome - pred) * features

        # Update weights
        self.weights += self.learning_rate * gradient

        # Update covariance (shrink uncertainty)
        outer = np.outer(features, features)
        self.covariance -= self.learning_rate * outer * pred * (1 - pred)
```

### 2. Learned Fusion Weights

**File**: `memory/episodic.py:LearnedFusionWeights`

Query-dependent scoring weights:

```python
class LearnedFusionWeights:
    def compute_weights(self, query_embedding: np.ndarray) -> np.ndarray:
        """Query → [semantic, recency, outcome, importance] weights."""
        hidden = np.tanh(self.W1 @ query_embedding + self.b1)
        logits = self.W2 @ hidden + self.b2
        weights = self._softmax(logits)

        # Blend with defaults during cold start
        if self.n_updates < self.cold_start_threshold:
            blend = self.n_updates / self.cold_start_threshold
            weights = blend * weights + (1 - blend) * self.default_weights

        return weights

    def update(
        self,
        query_embedding: np.ndarray,
        component_scores: Dict[str, float],
        outcome_utility: float
    ):
        """Update from retrieval outcome."""
        # Gradient: emphasize components correlated with utility
        scores = np.array([component_scores[k] for k in self.component_names])
        utility_centered = outcome_utility - 0.5
        target_shift = utility_centered * (scores - scores.mean())

        # Backprop through MLP
        # ... (standard gradient descent)
```

### 3. Learned Reranker

**File**: `memory/episodic.py:LearnedReranker`

Second-pass scoring with cross-component interactions:

```python
class LearnedReranker:
    def rerank(self, scored_results: List, query_embedding: np.ndarray) -> List:
        """Re-rank results using learned model."""
        query_context = self._compress_query(query_embedding)

        for result in scored_results:
            features = np.concatenate([
                [result.components[k] for k in self.component_names],
                query_context
            ])

            # MLP forward pass
            hidden = np.tanh(self.W1 @ features + self.b1)
            adjustment = (self.W2 @ hidden + self.b2)[0]
            adjustment = np.tanh(adjustment) * 0.2

            # Residual blend
            result.score = (
                (1 - self.residual_weight) * result.score +
                self.residual_weight * (result.score + adjustment)
            )

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results
```

### 4. Learned Sparse Index

**File**: `memory/learned_sparse_index.py`

Query-dependent cluster attention:

```python
class LearnedSparseIndex:
    def compute_attention(
        self,
        query_embedding: np.ndarray,
        cluster_centroids: np.ndarray
    ) -> np.ndarray:
        """Compute soft attention over clusters."""
        # Project query to attention space
        query_proj = self.query_projection(query_embedding)

        # Compute attention scores
        scores = cluster_centroids @ query_proj
        attention = self._sparse_softmax(scores, sparsity=0.1)

        return attention  # Only attend to ~10% of clusters
```

---

## Training Loops

### Online Learning Flow

```
1. User creates episode
   └─► Gate predicts STORE/BUFFER/SKIP
   └─► Register pending label

2. User queries memory
   └─► Retrieve episodes
   └─► Add eligibility traces
   └─► Neuromodulator state recorded

3. Outcome observed
   └─► Compute dopamine RPE
   └─► Distribute serotonin credit
   └─► Update gate weights
   └─► Update fusion weights
   └─► Update reranker
   └─► Reconsolidate if high surprise
```

### Batch Learning (Consolidation)

```python
# In consolidation/sleep_consolidation.py

async def consolidate(self):
    """Sleep-like consolidation with replay."""
    # 1. Select memories for replay (priority sampling)
    replay_queue = self.prioritized_replay.sample(batch_size=32)

    # 2. Compute cluster structure
    await self.cluster_index.recompute_clusters()

    # 3. Update sparse index
    self.sparse_index.train_on_clusters(self.cluster_index.centroids)

    # 4. Hebbian updates for frequently co-activated memories
    for pair in self.coactivation_buffer:
        await self.hebbian_update(pair.source, pair.target, pair.strength)

    # 5. Homeostatic scaling
    await self.homeostatic.apply_global_scaling()
```

---

## Cold Start & Warm Start

### Cold Start

When no prior state exists:

```python
# All learned weights initialized randomly
gate.weights = xavier_init(feature_dim)
fusion.W1 = xavier_init(hidden_dim, embed_dim)
reranker.W1 = xavier_init(hidden_dim, input_dim)

# Conservative defaults
gate.store_threshold = 0.6
gate.buffer_threshold = 0.3
fusion.default_weights = [0.4, 0.3, 0.2, 0.1]

# Extended exploration period
gate.cold_start_threshold = 100  # samples before trusting learned weights
```

### Warm Start

When checkpoint exists:

```python
# Load all learned state
gate.load_state(checkpoint['gate_state'])
fusion.load_state(checkpoint['fusion_state'])
reranker.load_state(checkpoint['reranker_state'])
neuromod.load_state(checkpoint['neuromod_state'])

# Eligibility traces restored
serotonin.traces = checkpoint['eligibility_traces']

# Dopamine expectations restored
dopamine.expectations = checkpoint['dopamine_expectations']
```

### Persistence Integration

```python
# WAL operations for learning events
WALOperation.GATE_WEIGHT_UPDATE
WALOperation.FUSION_WEIGHT_UPDATE
WALOperation.RERANKER_WEIGHT_UPDATE
WALOperation.DOPAMINE_EXPECTATION
WALOperation.ELIGIBILITY_UPDATE

# Checkpoint includes all learned state
checkpoint = {
    'gate_state': gate.save_state(),
    'fusion_state': fusion.save_state(),
    'reranker_state': reranker.save_state(),
    'neuromod_state': {
        'dopamine_expectations': dopamine.expectations,
        'serotonin_traces': serotonin.traces,
        'serotonin_values': serotonin.long_term_values,
    },
    'lsn': current_lsn,
}
```

---

## File Reference

### Core Learning

| File | Class | Purpose |
|------|-------|---------|
| `learning/three_factor.py` | `ThreeFactorLearningRule` | Combined learning rule |
| `learning/eligibility.py` | `EligibilityTrace` | Temporal credit |
| `learning/reconsolidation.py` | `ReconsolidationEngine` | Memory update on retrieval |
| `learning/homeostatic.py` | `HomeostaticPlasticity` | Stability maintenance |
| `learning/fsrs.py` | `FSRSScheduler` | Spaced repetition |

### Learned Components

| File | Class | Purpose |
|------|-------|---------|
| `core/learned_gate.py` | `LearnedMemoryGate` | Storage decisions |
| `memory/episodic.py` | `LearnedFusionWeights` | Query-dependent scoring |
| `memory/episodic.py` | `LearnedReranker` | Second-pass ranking |
| `memory/learned_sparse_index.py` | `LearnedSparseIndex` | Cluster attention |
| `memory/cluster_index.py` | `ClusterIndex` | Hierarchical search |

### Neuromodulation

| File | Class | Purpose |
|------|-------|---------|
| `learning/dopamine.py` | `DopamineSystem` | Reward prediction error |
| `learning/norepinephrine.py` | `NorepinephrineSystem` | Arousal/novelty |
| `learning/acetylcholine.py` | `AcetylcholineSystem` | Mode switching |
| `learning/serotonin.py` | `SerotoninSystem` | Long-term credit |
| `learning/inhibition.py` | `InhibitoryNetwork` | GABA dynamics |
| `learning/neuromodulators.py` | `NeuromodulatorOrchestra` | Coordination |

### Consolidation

| File | Class | Purpose |
|------|-------|---------|
| `consolidation/sleep_consolidation.py` | `SleepConsolidation` | Offline learning |
| `consolidation/sharp_wave_ripple.py` | `SharpWaveRipple` | Memory replay |

### Persistence

| File | Class | Purpose |
|------|-------|---------|
| `learning/persistence.py` | `StatePersister` | Save/load learned state |
| `persistence/checkpoint.py` | `CheckpointManager` | Full state snapshots |

---

## Visualization

| File | Function | Purpose |
|------|----------|---------|
| `visualization/plasticity_traces.py` | `plot_bcm_curve()` | Learning curve visualization |
| `visualization/plasticity_traces.py` | `plot_ltp_ltd_distribution()` | Weight change distribution |
| `visualization/neuromodulator_state.py` | `plot_neuromodulator_traces()` | Neuromod timeline |
