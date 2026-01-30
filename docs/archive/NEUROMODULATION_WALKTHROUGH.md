# Neuromodulation System Walkthrough

**Version**: 0.1.0
**Last Updated**: 2025-12-09

This document details the five neuromodulator systems that coordinate World Weaver's brain-like dynamics.

---

## Table of Contents

1. [Overview](#overview)
2. [The Five Neuromodulators](#the-five-neuromodulators)
3. [Orchestra Coordination](#orchestra-coordination)
4. [Integration with Memory](#integration-with-memory)
5. [Learning Signals](#learning-signals)
6. [File Reference](#file-reference)

---

## Overview

World Weaver implements a biologically-inspired neuromodulation system based on five key neurotransmitters:

| System | Neurotransmitter | Role |
|--------|-----------------|------|
| **Dopamine** | DA | Reward prediction error, surprise |
| **Norepinephrine** | NE | Arousal, attention, novelty |
| **Acetylcholine** | ACh | Encoding vs retrieval mode |
| **Serotonin** | 5-HT | Long-term credit, patience |
| **GABA** | Inhibition | Sparse coding, winner-take-all |

These systems don't operate independently - they form a coordinated ensemble through the `NeuromodulatorOrchestra`.

---

## The Five Neuromodulators

### 1. Dopamine System (DA)

**File**: `learning/dopamine.py`

**Purpose**: Reward Prediction Error (RPE) - learning scales with surprise

**Core Equation**:
```
RPE = actual_outcome - expected_value
```

**Key Class**: `DopamineSystem`

```python
class DopamineSystem:
    def compute_rpe(self, memory_id: UUID, outcome: float) -> RewardPredictionError:
        expected = self.get_expected_value(memory_id)
        rpe = outcome - expected
        surprise = abs(rpe)
        return RewardPredictionError(
            rpe=rpe,
            surprise_magnitude=surprise,
            expected=expected,
            actual=outcome
        )

    def update_expectations(self, memory_id: UUID, outcome: float):
        # Exponential moving average
        self.expectations[memory_id] = (
            (1 - self.learning_rate) * self.expectations[memory_id] +
            self.learning_rate * outcome
        )
```

**Biological Basis**: Schultz (1998) - Dopamine neurons encode prediction errors

**Effect on Learning**:
- High surprise (unexpected success/failure) → Strong learning signal
- Low surprise (expected outcome) → Minimal learning
- Prevents overwriting memories that are already well-calibrated

---

### 2. Norepinephrine System (NE)

**File**: `learning/norepinephrine.py`

**Purpose**: Arousal modulation, novelty detection, exploration/exploitation balance

**Key Class**: `NorepinephrineSystem`

```python
class NorepinephrineSystem:
    def update(self, query_embedding: np.ndarray) -> NEState:
        # Compute novelty relative to recent queries
        novelty = self._compute_novelty(query_embedding)

        # Update arousal gain
        gain = 1.0 + (novelty - self.baseline_novelty) * self.gain_factor
        gain = np.clip(gain, self.min_gain, self.max_gain)

        return NEState(
            novelty_score=novelty,
            combined_gain=gain,
            exploration_boost=self._compute_exploration(novelty)
        )

    def modulate_retrieval_threshold(self, base_threshold: float) -> float:
        # High arousal → lower threshold → broader search
        return base_threshold / self.current_gain
```

**Biological Basis**: Aston-Jones & Cohen (2005) - LC-NE system and attentional control

**Effect on Retrieval**:
- High novelty → Lower similarity threshold → More exploration
- Low novelty → Higher threshold → Focused, efficient retrieval

---

### 3. Acetylcholine System (ACh)

**File**: `learning/acetylcholine.py`

**Purpose**: Mode switching between encoding (storage) and retrieval

**Key Class**: `AcetylcholineSystem`

```python
class AcetylcholineMode(Enum):
    ENCODING = "encoding"    # Prioritize storage
    RETRIEVAL = "retrieval"  # Prioritize recall
    BALANCED = "balanced"    # Equal priority

class AcetylcholineSystem:
    def update(
        self,
        encoding_demand: float,
        retrieval_demand: float,
        arousal_gain: float
    ) -> ACHState:
        # Arousal modulates mode sensitivity
        enc_signal = encoding_demand * arousal_gain
        ret_signal = retrieval_demand

        if enc_signal > ret_signal + self.mode_threshold:
            mode = AcetylcholineMode.ENCODING
        elif ret_signal > enc_signal + self.mode_threshold:
            mode = AcetylcholineMode.RETRIEVAL
        else:
            mode = AcetylcholineMode.BALANCED

        return ACHState(mode=mode, ...)

    def compute_encoding_demand(
        self,
        query_novelty: float,
        is_statement: bool,
        explicit_importance: Optional[float]
    ) -> float:
        demand = 0.0
        if is_statement:
            demand += 0.3
        if query_novelty > 0.7:
            demand += 0.3 * query_novelty
        if explicit_importance:
            demand += 0.2 * explicit_importance
        return demand
```

**Biological Basis**: Hasselmo (2006) - ACh and encoding/retrieval dynamics

**Effect on Memory**:
- Encoding mode: Prioritize storage, enhance plasticity
- Retrieval mode: Prioritize recall, suppress interference

---

### 4. Serotonin System (5-HT)

**File**: `learning/serotonin.py`

**Purpose**: Long-term credit assignment, temporal discounting, patience

**Key Class**: `SerotoninSystem`

```python
class SerotoninSystem:
    def add_eligibility(self, memory_id: UUID, strength: float):
        """Add eligibility trace for later credit assignment."""
        self.eligibility_traces[memory_id] = EligibilityTrace(
            memory_id=memory_id,
            strength=strength,
            timestamp=datetime.now()
        )

    def receive_outcome(self, outcome: float, context_id: Optional[str] = None):
        """Distribute credit based on eligibility traces."""
        credits = {}
        for mem_id, trace in self.eligibility_traces.items():
            # Temporal discount
            age = (datetime.now() - trace.timestamp).total_seconds()
            discount = np.exp(-self.discount_rate * age)

            # Credit = outcome × eligibility × discount
            credit = outcome * trace.strength * discount
            credits[mem_id] = credit

            # Update long-term value
            self.long_term_values[mem_id] = (
                (1 - self.value_lr) * self.long_term_values.get(mem_id, 0.5) +
                self.value_lr * credit
            )

        return credits

    def get_long_term_value(self, memory_id: UUID) -> float:
        """Get accumulated long-term value for a memory."""
        return self.long_term_values.get(memory_id, 0.5)
```

**Biological Basis**: Doya (2002) - Serotonin and temporal discounting

**Effect on Learning**:
- Assigns credit to memories that preceded successful outcomes
- Enables learning from delayed rewards
- "Patience" signal for long-term value estimation

---

### 5. Inhibitory Network (GABA)

**File**: `learning/inhibition.py`

**Purpose**: Competitive dynamics, sparse representations, winner-take-all

**Key Class**: `InhibitoryNetwork`

```python
class InhibitoryNetwork:
    def apply_inhibition(
        self,
        scores: Dict[str, float],
        embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> InhibitionResult:
        """Apply lateral inhibition to sharpen score distribution."""
        score_arr = np.array(list(scores.values()))

        # Softmax with temperature for competitive dynamics
        exp_scores = np.exp(score_arr / self.temperature)
        inhibited = exp_scores / exp_scores.sum()

        # Winner-take-all enhancement
        max_idx = np.argmax(inhibited)
        inhibited[max_idx] *= self.winner_boost

        # Compute sparsity
        sparsity = 1.0 - (np.count_nonzero(inhibited > self.threshold) / len(inhibited))

        return InhibitionResult(
            inhibited_scores=dict(zip(scores.keys(), inhibited)),
            sparsity=sparsity,
            winner_id=list(scores.keys())[max_idx]
        )
```

**Biological Basis**: Douglas & Martin (2004) - Cortical inhibition

**Effect on Retrieval**:
- Sharpens score distributions (reduces noise)
- Enhances strongest candidates
- Suppresses weak competitors
- Creates sparse, selective responses

---

## Orchestra Coordination

**File**: `learning/neuromodulators.py`

The `NeuromodulatorOrchestra` coordinates all five systems:

```python
class NeuromodulatorOrchestra:
    def __init__(self):
        self.dopamine = DopamineSystem()
        self.norepinephrine = NorepinephrineSystem()
        self.acetylcholine = AcetylcholineSystem()
        self.serotonin = SerotoninSystem()
        self.inhibitory = InhibitoryNetwork()
```

### Interaction Pattern

```
Query Input
    │
    ▼
┌───────────────────────────────────────┐
│         process_query()               │
│                                       │
│  1. NE: Compute novelty/arousal       │
│         ↓                             │
│  2. ACh: Set mode based on novelty    │
│         (high novelty → encoding)     │
│         ↓                             │
│  3. Return combined state             │
└───────────────────────────────────────┘
    │
    ▼
Retrieval
    │
    ▼
┌───────────────────────────────────────┐
│       process_retrieval()             │
│                                       │
│  1. Add eligibility traces (5-HT)     │
│  2. Apply inhibition (GABA)           │
│  3. Update state with sparsity        │
└───────────────────────────────────────┘
    │
    ▼
Outcome Received
    │
    ▼
┌───────────────────────────────────────┐
│        process_outcome()              │
│                                       │
│  1. DA: Compute RPE for each memory   │
│  2. DA: Update expectations           │
│  3. 5-HT: Distribute credit           │
│  4. Combine: DA × 5-HT × eligibility  │
└───────────────────────────────────────┘
```

### Key Methods

```python
class NeuromodulatorOrchestra:

    def process_query(
        self,
        query_embedding: np.ndarray,
        is_question: bool = False,
        explicit_importance: Optional[float] = None
    ) -> NeuromodulatorState:
        """Process query through NE and ACh."""
        # 1. Update NE (novelty/arousal)
        ne_state = self.norepinephrine.update(query_embedding)

        # 2. Update ACh (encoding/retrieval mode)
        encoding_demand = self.acetylcholine.compute_encoding_demand(...)
        retrieval_demand = self.acetylcholine.compute_retrieval_demand(is_question)
        ach_state = self.acetylcholine.update(
            encoding_demand, retrieval_demand, ne_state.combined_gain
        )

        return NeuromodulatorState(
            dopamine_rpe=0.0,  # Updated on outcome
            norepinephrine_gain=ne_state.combined_gain,
            acetylcholine_mode=ach_state.mode.value,
            serotonin_mood=self.serotonin.get_current_mood(),
            inhibition_sparsity=0.0
        )

    def process_retrieval(
        self,
        retrieved_ids: list[UUID],
        scores: dict[str, float],
        embeddings: Optional[dict[str, np.ndarray]] = None
    ) -> dict[str, float]:
        """Process retrieval through 5-HT eligibility and GABA inhibition."""
        # Add eligibility traces
        for mem_id in retrieved_ids:
            self.serotonin.add_eligibility(mem_id, scores.get(str(mem_id), 0.5))

        # Apply inhibition
        result = self.inhibitory.apply_inhibition(scores, embeddings)
        return result.inhibited_scores

    def process_outcome(
        self,
        memory_outcomes: Dict[str, float],
        session_outcome: Optional[float] = None
    ) -> Dict[str, float]:
        """Process outcomes through DA and 5-HT."""
        # Compute dopamine RPEs
        rpes = self.dopamine.batch_compute_rpe(memory_outcomes)
        self.dopamine.batch_update_expectations(memory_outcomes)

        # Distribute serotonin credit
        if session_outcome is not None:
            self.serotonin.receive_outcome(session_outcome)

        # Multiplicative combination
        learning_signals = {}
        for mem_id in memory_outcomes:
            dopamine_surprise = rpes[mem_id].surprise_magnitude
            serotonin_patience = self.serotonin.get_long_term_value(UUID(mem_id))
            eligibility = self.serotonin.get_eligibility(UUID(mem_id))

            # All factors gate each other
            learning_signals[mem_id] = dopamine_surprise * serotonin_patience * eligibility

        return learning_signals
```

---

## Integration with Memory

### During Storage (Encoding)

```python
# In episodic.py:create()

# Process as encoding operation
neuromod_state = self.orchestra.process_query(
    query_embedding=embedding,
    is_question=False,  # Storage = encoding
    explicit_importance=valence
)

# Use for gate decision
gate_decision = self.learned_gate.predict(
    content_embedding=embedding,
    neuromod_state=neuromod_state,
    ...
)
```

**Effect**: High novelty → encoding mode → higher storage probability

### During Retrieval

```python
# In episodic.py:recall()

# Process as retrieval operation
neuromod_state = self.orchestra.process_query(
    query_embedding=query_emb,
    is_question=True,  # Recall = retrieval
)

# Modulate search threshold
threshold = self.orchestra.get_retrieval_threshold(base_threshold)

# After retrieval, apply inhibition
inhibited_scores = self.orchestra.process_retrieval(
    retrieved_ids, scores, embeddings
)
```

**Effect**: High arousal → lower threshold → broader exploration

### After Outcome

```python
# In learn_from_outcome()

learning_signals = self.orchestra.process_outcome(
    memory_outcomes={str(mem_id): outcome for mem_id in retrieved_ids},
    session_outcome=session_success
)

# Apply to reconsolidation
for mem_id, signal in learning_signals.items():
    if signal > self.learning_threshold:
        await self.reconsolidation.update(mem_id, signal)
```

---

## Learning Signals

### LearningParams Structure

```python
@dataclass
class LearningParams:
    effective_lr: float  # Combined from NE, ACh, 5-HT
    eligibility: float   # Temporal credit [0, 1]
    surprise: float      # Dopamine |RPE|
    patience: float      # Serotonin long-term value
    rpe: float           # Signed dopamine RPE

    @property
    def combined_learning_signal(self) -> float:
        """Multiplicative gating with bootstrap."""
        multiplicative = (
            self.effective_lr *
            self.eligibility *
            self.surprise *
            self.patience
        )
        # Bootstrap prevents zero-learning deadlock
        bootstrap = 0.01 * self.effective_lr * max(0.1, self.surprise)
        return multiplicative + bootstrap
```

### Three-Factor Learning Rule

The core learning principle:

```
Δw = eligibility × neuromod_state × dopamine_surprise
```

| Factor | Source | Meaning |
|--------|--------|---------|
| Eligibility | 5-HT traces | "Was this memory recently active?" |
| Neuromod State | ACh, NE | "Is the system in learning mode?" |
| Dopamine Surprise | DA RPE | "Was the outcome unexpected?" |

All factors must align for strong learning - this prevents:
- Updating inactive memories (low eligibility)
- Updating during retrieval mode (low neuromod)
- Updating on expected outcomes (low surprise)

---

## File Reference

| File | Class | Purpose |
|------|-------|---------|
| `learning/dopamine.py` | `DopamineSystem` | Reward prediction error |
| `learning/norepinephrine.py` | `NorepinephrineSystem` | Arousal/novelty |
| `learning/acetylcholine.py` | `AcetylcholineSystem` | Encoding/retrieval mode |
| `learning/serotonin.py` | `SerotoninSystem` | Long-term credit |
| `learning/inhibition.py` | `InhibitoryNetwork` | GABA dynamics |
| `learning/neuromodulators.py` | `NeuromodulatorOrchestra` | Coordination |
| `learning/eligibility.py` | `EligibilityTrace` | Temporal credit traces |
| `learning/three_factor.py` | `ThreeFactorLearningRule` | Combined learning |

### Visualization

| File | Function | Purpose |
|------|----------|---------|
| `visualization/neuromodulator_state.py` | `plot_neuromodulator_traces()` | Timeline of all modulators |
| `visualization/neuromodulator_state.py` | `plot_neuromodulator_radar()` | Current state snapshot |

---

## References

- Schultz, W. (1998). Predictive reward signal of dopamine neurons. *J Neurophysiol*
- Aston-Jones, G., & Cohen, J. D. (2005). LC-NE system and attentional control. *Annu Rev Neurosci*
- Hasselmo, M. E. (2006). ACh and encoding/retrieval dynamics. *Curr Opin Neurobiol*
- Doya, K. (2002). Metalearning and neuromodulation. *Neural Netw*
- Douglas, R. J., & Martin, K. A. (2004). Neuronal circuits of the neocortex. *Annu Rev Neurosci*
