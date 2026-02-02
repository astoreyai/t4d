# T4DM Learning System - Quick Reference

## Core Formulas

### Retrieval Reward
```
R(memory) = (outcome - baseline) × time_discount × attention_weight × citation_bonus

where:
  time_discount = 1 / (1 + 0.1 × hours_delay)    # Hyperbolic
  attention_weight = score[memory] / Σscores
  citation_bonus = 1.5 if explicitly cited, else 1.0
```

### Eligibility Traces (TD-λ)
```
On retrieval:
  trace[memory] += retrieval_score

On time passing:
  trace[memory] *= (γ × λ)^elapsed_hours
  where γ = 0.99, λ = 0.9

On outcome:
  credit[memory] = reward × trace[memory] / Σtraces
```

### ListMLE Ranking Loss
```
L = -Σᵢ log(exp(scoreᵢ) / Σⱼ∈remaining exp(scoreⱼ))

where i iterates in target order (sorted by reward)
```

### EWC Penalty
```
L_EWC = λ × Σᵢ Fᵢ × (θᵢ - θᵢ*)²

where:
  Fᵢ = Fisher information diagonal
  θ* = saved parameters after task completion
  λ = 1000 (regularization strength)
```

### Entity Quality Score
```
Q = 0.3 × usage_freq + 0.4 × outcome_utility + 0.15 × centrality + 0.15 × specificity

where:
  usage_freq = 1 - exp(-retrievals / 5)
  outcome_utility = Σ(success × weight) / n
  centrality = min(pagerank × 10, 1.0)
  specificity = 1 / (1 + log(1 + fan_out))
```

### Forgetting Regret
```
If regretted (re-learned within window):
  R = -1 / (1 + days_since_forgotten / 7)

If not regretted:
  R = min(0.1 + 0.01 × days_since_forgotten, 0.5)
```

---

## Key Data Structures

```python
@dataclass
class RetrievalEvent:
    retrieval_id: UUID
    query: str
    retrieved_ids: list[UUID]
    retrieval_scores: dict[UUID, float]
    timestamp: datetime
    context_hash: str
    session_id: str

@dataclass
class OutcomeEvent:
    outcome_id: UUID
    success_score: float  # [0, 1]
    context_hash: str
    timestamp: datetime
    explicit_citations: list[UUID]

@dataclass
class Experience:
    query: str
    retrieved_ids: list[UUID]
    retrieval_scores: list[float]
    component_vectors: list[list[float]]
    outcome_score: float
    per_memory_rewards: dict[str, float]
```

---

## Learning Rates

| Component | Learning Rate | Update Frequency |
|-----------|---------------|------------------|
| Retrieval Scorer | 1e-4 | Every batch (offline) |
| Consolidation Policy | 1e-4 | Daily |
| Decay Rates | 1e-5 | Weekly |
| Hebbian (online) | 0.05 × reward | Every outcome |

---

## Implicit Feedback Signals

| Signal | Implied Reward | Detection |
|--------|----------------|-----------|
| Accept | +0.3 | "yes", "thanks", "perfect" |
| Reject | -0.5 | "no", "that's not right", "try again" |
| Modify | 0.0 | User edits Claude output |
| Explicit positive | +0.8 | "that was helpful" |
| Explicit negative | -0.8 | "that didn't help" |
| Repetition | -0.3 | Same query within 24h |

---

## Training Schedule

```
Online (during session):
  - Eligibility trace updates: Every retrieval
  - Experience collection: Every outcome
  - Hebbian strengthening: If reward > 0

Offline (between sessions):
  - Retrieval scorer training: Hourly (if >50 new experiences)
  - Consolidation evaluation: Daily
  - Decay optimization: Weekly
  - EWC Fisher update: After each domain/project switch
```

---

## Checkpoint Contents

```python
{
    'retrieval_scorer': scorer.state_dict(),
    'consolidation_policy': policy.state_dict(),
    'decay_policy': decay.state_dict(),
    'ewc_params': ewc.saved_params,
    'ewc_fisher': ewc.fisher_diagonal,
    'training_step': step_count,
    'metrics_history': [...],
}
```

---

## Quick Debugging

### Check if learning is happening:
```python
scorer = load_scorer()
print(scorer.get_weights())  # Should change over time
```

### Check reward attribution:
```python
events = load_events(last_24h)
for e in events:
    if e.type == "outcome":
        print(f"Outcome: {e.success_score}")
        retrievals = match_retrievals(e.context_hash)
        rewards = compute_rewards(retrievals, e)
        print(f"Rewards: {rewards}")
```

### Check trace decay:
```python
traces = tracer.traces
for mem_id, trace in traces.items():
    print(f"{mem_id}: {trace:.4f}")
```

---

## Phase Completion Checklist

- [ ] Phase 0: All tests passing, infrastructure stable
- [ ] Phase 1: Events being collected, SQLite populated
- [ ] Phase 2: Rewards computed, traces updating, Hebbian weighted
- [ ] Phase 3: Scorer training, weights changing, loss decreasing
- [ ] Phase 4: Hooks installed, sessions tracked, feedback detected
- [ ] Phase 5: Offline training running, metrics logged
- [ ] Phase 6: EWC active, MAML adapting, no forgetting
- [ ] Phase 7: 90%+ coverage, benchmarks met
- [ ] Phase 8: Docs complete, released
