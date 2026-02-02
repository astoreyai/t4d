# Learning Module

**Path**: `t4dm/learning/` | **Files**: 24 | **Lines**: ~8,500

Biologically-inspired adaptive learning framework combining eligibility traces, neuromodulator dynamics, and reward prediction errors.

---

## Quick Start

```python
from ww.learning import (
    NeuromodulatorOrchestra, create_neuromodulator_orchestra,
    ThreeFactorLearningRule, ReconsolidationEngine,
    emit_retrieval_event,
)

# Create neuromodulator orchestra
orchestra = create_neuromodulator_orchestra()

# Process query (updates NE, ACh)
orchestra.process_query(query_embedding, context)

# Process retrieval (adds eligibility traces)
orchestra.process_retrieval(retrieved_ids, scores)

# Process outcome (computes RPE)
signals = orchestra.process_outcome(
    memory_outcomes={"mem-1": 0.8, "mem-2": 0.3},
    session_outcome=0.7,
)

# Get combined learning rate
params = orchestra.get_learning_params("mem-1")
effective_lr = params.combined_learning_signal
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                  Three-Factor Learning Rule                     │
│                                                                 │
│  effective_lr = base_lr × eligibility × neuromod × DA_surprise  │
└────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Eligibility  │    │  Neuromod     │    │   Dopamine    │
│    Traces     │    │   Orchestra   │    │    System     │
├───────────────┤    ├───────────────┤    ├───────────────┤
│ Fast (τ=5s)   │    │ DA (RPE)      │    │ RPE compute   │
│ Slow (τ=60s)  │    │ NE (arousal)  │    │ TD(λ) update  │
│ Decay exp     │    │ ACh (mode)    │    │ Value network │
│               │    │ 5-HT (mood)   │    │               │
│               │    │ GABA (inhib)  │    │               │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                   ┌──────────────────────┐
                   │  ReconsolidationEngine │
                   │  (Embedding Updates)   │
                   └──────────────────────┘
```

---

## File Structure

### Core Learning

| File | Purpose | Key Classes |
|------|---------|-------------|
| `three_factor.py` | Combined learning rule | `ThreeFactorLearningRule`, `ThreeFactorSignal` |
| `eligibility.py` | Temporal credit | `EligibilityTrace`, `LayeredEligibilityTrace` |
| `dopamine.py` | Reward prediction | `DopamineSystem`, `LearnedValueEstimator` |
| `reconsolidation.py` | Embedding updates | `ReconsolidationEngine` |
| `scorer.py` | Learned ranking | `LearnedRetrievalScorer`, `ScorerTrainer` |

### Neuromodulators

| File | Purpose | Key Classes |
|------|---------|-------------|
| `neuromodulators.py` | Unified coordinator | `NeuromodulatorOrchestra`, `LearningParams` |
| `norepinephrine.py` | Arousal/attention | `NorepinephrineSystem`, `ArousalState` |
| `acetylcholine.py` | Encoding/retrieval | `AcetylcholineSystem`, `CognitiveMode` |
| `serotonin.py` | Long-term value | `SerotoninSystem`, `TemporalContext` |
| `inhibition.py` | Competitive dynamics | `InhibitoryNetwork` |

### Events & Persistence

| File | Purpose | Key Classes |
|------|---------|-------------|
| `events.py` | Event types | `RetrievalEvent`, `OutcomeEvent`, `Experience` |
| `collector.py` | Event storage | `EventCollector`, `EventStore` |
| `hooks.py` | Integration points | `RetrievalHookMixin` |
| `persistence.py` | State save/load | Checkpoint support |

### Advanced

| File | Purpose |
|------|---------|
| `stdp.py` | Spike-timing plasticity |
| `plasticity.py` | LTD, homeostatic scaling, EWC |
| `neuro_symbolic.py` | Triple-based reasoning |
| `causal_discovery.py` | Causal graph learning |
| `self_supervised.py` | Implicit credit signals |
| `cold_start.py` | Population priors |
| `fsrs.py` | Spaced repetition |

---

## Three-Factor Learning

All three factors must align for strong learning:

```
effective_lr = base_lr × eligibility × neuromod_gate × dopamine_surprise
```

### Factor 1: Eligibility Traces

Which synapses were recently active?

```python
from ww.learning import EligibilityTrace

traces = EligibilityTrace(decay=0.95, tau=20.0)

# Memory activated
traces.update("memory-123", activity=1.0)

# Time passes
traces.step(dt=1.0)

# Reward arrives → credit assignment
credits = traces.assign_credit(reward=0.8)
# credits["memory-123"] = 0.8 * trace_strength
```

**Layered Traces**:
- Fast layer (τ=5s): Recent activity
- Slow layer (τ=60s): Extended eligibility window

### Factor 2: Neuromodulator Gate

Should we learn now?

```python
# Combined gate from 3 systems
gate = ACh_boost × NE_gain × 5HT_mood_factor

# ACh modes
ENCODING: 2.0x boost (new information)
RETRIEVAL: 1.2x boost (familiar context)
BALANCED: 1.0x (default)

# NE arousal
gain = 0.5 + novelty_burst  # Range: 0.5-2.0x

# 5-HT mood (inverted U)
factor = 1.0 - abs(mood - 0.5)  # Optimal at mood=0.5
```

### Factor 3: Dopamine Surprise

How unexpected was this?

```python
from ww.learning import DopamineSystem

da = DopamineSystem()

# Compute reward prediction error
rpe = da.compute_rpe("memory-123", actual_outcome=0.9)
# rpe = actual - expected

# Update expectations
da.update_expectations("memory-123", actual_outcome=0.9)

# TD(λ) credit distribution
da.update_with_td_lambda(td_error=rpe)
```

**LearnedValueEstimator**: MLP that predicts value from embedding:
```
embedding [1024] → FC1 → ReLU → LayerNorm → FC2 → Sigmoid [0,1]
```

---

## Neuromodulator Orchestra

Unified coordinator for all 5 neuromodulator systems:

```python
orchestra = create_neuromodulator_orchestra()

# Full query-retrieval-outcome flow
orchestra.process_query(query_emb, context)      # Updates NE, ACh
orchestra.process_retrieval(ids, scores)          # Adds eligibility
signals = orchestra.process_outcome(outcomes)     # Computes RPE

# Get integrated learning parameters
params = orchestra.get_learning_params("mem-123")
print(params.combined_learning_signal)  # Final multiplier
```

### Individual Systems

| System | Signal | Effect |
|--------|--------|--------|
| **Dopamine** | RPE (δ) | Scales learning by surprise |
| **Norepinephrine** | Arousal | Exploration vs exploitation |
| **Acetylcholine** | Mode | Encoding vs retrieval |
| **Serotonin** | Mood | Long-term temporal credit |
| **GABA** | Inhibition | Winner-take-all sharpening |

---

## Reconsolidation

Update memory embeddings based on outcomes:

```python
from ww.learning import ReconsolidationEngine

engine = ReconsolidationEngine(base_lr=0.01)

# Positive outcome → move toward query
# Negative outcome → move away from query
update = engine.update_embedding(
    memory_id="mem-123",
    query_embedding=query_emb,
    outcome=0.9,  # Success
)
```

**Three-Factor Integration**:
```python
engine = ReconsolidationEngine(
    base_lr=0.01,
    three_factor_rule=ThreeFactorLearningRule(orchestra),
)
# Now lr = base_lr × eligibility × neuromod × surprise
```

---

## Learned Retrieval Scorer

Neural network for ranking memories:

```python
from ww.learning import LearnedRetrievalScorer, ScorerTrainer

scorer = LearnedRetrievalScorer()
trainer = ScorerTrainer(scorer)

# Score memories
scores = scorer.forward(features)  # [similarity, recency, importance, outcome]

# Train from experience
loss = trainer.train_step(experiences)
```

**Architecture**:
```
[4 features] → FC1(32) → ReLU → FC2(32) → FCOut(1) → Sigmoid
```

**Training**: ListMLE ranking loss with prioritized replay.

---

## STDP (Spike-Timing-Dependent Plasticity)

```python
from ww.learning import STDPLearner

stdp = STDPLearner(
    a_plus=0.005,   # LTP rate
    a_minus=0.00525, # LTD rate (slightly higher for stability)
    tau_plus=20.0,
    tau_minus=20.0,
)

# Record spikes
stdp.record_spike("pre-123", time=100)
stdp.record_spike("post-456", time=105)  # 5ms after → LTP

# Get weight update
delta_w = stdp.get_weight_update("pre-123", "post-456")
```

---

## Events System

Capture and store learning events:

```python
from ww.learning import emit_retrieval_event, EventCollector

# Emit retrieval event
emit_retrieval_event(
    query="python debugging",
    retrieved_ids=["mem-1", "mem-2"],
    scores=[0.9, 0.7],
    component_scores={"mem-1": {"similarity": 0.9, "recency": 0.8}},
)

# Query stored events
collector = EventCollector()
experiences = collector.get_experiences(limit=100)
```

**Representation Formats**:
- `FullJSON`: Complete fidelity for storage
- `ToonJSON`: Token-optimized (~50% reduction) for LLM context
- `NeuroSymbolicTriples`: Graph-based for reasoning

---

## Configuration

```python
# Eligibility
decay=0.95
tau_trace=20.0

# Dopamine
default_expected=0.5
value_learning_rate=0.1
surprise_threshold=0.05

# Neuromodulators
NE_baseline_arousal=0.5
ACh_baseline=0.5
5HT_baseline_mood=0.5

# Inhibition
inhibition_strength=0.5
sparsity_target=0.2
```

---

## Bug Fixes Applied

| Bug | Fix |
|-----|-----|
| BUG-004 | Added `_rpe_cache` to NeuromodulatorOrchestra |
| BUG-005 | Changed mood modulation to inverted-U |
| BIO-002 | Both encoding AND retrieval modes boost learning |
| LOGIC-007 | Separated long-term value from eligibility |
| LOGIC-010 | Added `get_signed_rpe()` for depression |
| DATA-005 | NaN/Inf validation in three_factor and reconsolidation |
| MEM-004 | Size limits (10K history, 100K memories) |

---

## Biological Basis

| Mechanism | Biological Source |
|-----------|------------------|
| Eligibility Traces | Synaptic tags (Frey & Morris) |
| Dopamine RPE | VTA (Schultz 1998) |
| Norepinephrine | Locus Coeruleus (Aston-Jones) |
| Acetylcholine | NBM (Hasselmo) |
| Serotonin | Raphe nuclei (Doya patience) |
| STDP | Bi & Poo 1998 |
| Three-Factor | Neuromodulated plasticity (Pawlak 2010) |
