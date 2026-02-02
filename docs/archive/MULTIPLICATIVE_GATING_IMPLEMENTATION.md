# Multiplicative Gating Implementation

## Overview

Implemented multiplicative gating for the neuromodulator orchestra system, replacing additive combination with biologically-inspired multiplicative combination of learning signals.

## Key Changes

### 1. LearningParams Dataclass (`neuromodulators.py`)

Created new `LearningParams` dataclass that integrates all neuromodulatory signals:

```python
@dataclass
class LearningParams:
    effective_lr: float      # Combined learning rate (NE, ACh, 5-HT)
    eligibility: float       # Eligibility trace strength [0, 1]
    surprise: float          # Dopamine surprise magnitude (|RPE|)
    patience: float          # Serotonin long-term value
    rpe: float              # Raw dopamine RPE (signed)
```

**Key property**: `combined_learning_signal` computes multiplicative combination:
```python
combined_signal = effective_lr * eligibility * surprise * patience
```

This implements biological gating: all signals must align for strong learning. If any signal is zero/weak, the combined signal is zero/weak.

### 2. Updated `process_outcome()` (`neuromodulators.py`)

Changed from **additive** combination:
```python
# OLD: Weighted sum
learning_signals[mem_id] = 0.7 * dopamine_rpe + 0.3 * serotonin_credit
```

To **multiplicative** gating:
```python
# NEW: Multiplicative gates
dopamine_surprise = rpe.surprise_magnitude
serotonin_patience = serotonin_credits.get(mem_id, 0.0)
eligibility_strength = self.serotonin.get_eligibility(mem_uuid)

combined_signal = dopamine_surprise * serotonin_patience * eligibility_strength
```

### 3. Integration Methods

Added two new methods to `NeuromodulatorOrchestra`:

#### `get_learning_params(memory_id: UUID) -> LearningParams`
Returns learning parameters without outcome (surprise/RPE are 0).

#### `get_learning_params_with_outcome(memory_id: UUID, outcome: float) -> LearningParams`
Computes complete learning parameters including dopamine RPE and surprise based on the provided outcome.

### 4. NeuromodulatorIntegratedReconsolidation (`reconsolidation.py`)

New class that integrates the full neuromodulator orchestra with reconsolidation:

```python
class NeuromodulatorIntegratedReconsolidation:
    def update(self, memory_id, memory_embedding, query_embedding, outcome_score, importance=0.0):
        # Get integrated learning parameters
        params = self.orchestra.get_learning_params_with_outcome(memory_id, outcome_score)

        # Use multiplicative gating
        lr_modulation = params.combined_learning_signal

        # Apply reconsolidation
        updated_embedding = self.reconsolidation.reconsolidate(
            memory_id, memory_embedding, query_embedding,
            outcome_score, importance, lr_modulation
        )
```

## Biological Rationale

### Multiplicative vs Additive

**Additive** (old):
- Signals sum together
- Even if one signal disagrees, combined signal can still be strong
- Less selective learning

**Multiplicative** (new):
- Signals gate each other
- ALL signals must agree for strong learning
- If any signal is weak/absent, combined signal is weak/absent
- More selective, biologically plausible learning

### Components

1. **Dopamine Surprise** (`|RPE|`): Learn more from unexpected outcomes
2. **Serotonin Patience**: Long-term value estimation, temporal discounting
3. **Eligibility Traces**: Temporal credit assignment (recently active synapses)
4. **Effective LR**: Baseline from norepinephrine (arousal), acetylcholine (mode), serotonin (mood)

### Example

```python
# Scenario: High surprise, but no eligibility trace
dopamine_surprise = 0.8  # Very surprising!
serotonin_patience = 0.5  # Moderate long-term value
eligibility = 0.0  # Memory wasn't recently active
effective_lr = 1.0

# Additive (old): 0.8 + 0.5 + 0.0 + 1.0 = 2.3 (strong signal)
# Multiplicative (new): 0.8 * 0.5 * 0.0 * 1.0 = 0.0 (no update)
# Biological: Don't update synapses that weren't active!
```

## Files Modified

1. `/mnt/projects/t4d/t4dm/src/t4dm/learning/neuromodulators.py`
   - Added `LearningParams` dataclass
   - Updated `process_outcome()` to use multiplicative gating
   - Added `get_learning_params()` and `get_learning_params_with_outcome()` methods
   - Updated documentation

2. `/mnt/projects/t4d/t4dm/src/t4dm/learning/reconsolidation.py`
   - Added `NeuromodulatorIntegratedReconsolidation` class
   - Integrates orchestra with reconsolidation engine

3. `/mnt/projects/t4d/t4dm/src/t4dm/learning/serotonin.py`
   - Fixed exports (removed `EligibilityTrace` from `__all__`)

## Tests

Created comprehensive test suite: `/mnt/projects/t4d/t4dm/tests/learning/test_multiplicative_gating.py`

Test categories:
1. **LearningParams** (4 tests)
   - Creation, multiplicative combination, zero-gating, serialization

2. **Multiplicative Gating** (2 tests)
   - `process_outcome()` uses multiplicative combination
   - Zero eligibility results in zero signal (gating property)

3. **Get Learning Params** (3 tests)
   - Basic parameter retrieval
   - With outcome (computes RPE)
   - Neutral params when no state

4. **NeuromodulatorIntegratedReconsolidation** (3 tests)
   - Import, creation, update with orchestra

**All tests passing**: 12/12 in new test file, 235/238 in full learning module (3 pre-existing failures unrelated to this change)

## Usage Example

```python
from t4dm.learning.reconsolidation import NeuromodulatorIntegratedReconsolidation
import numpy as np
from uuid import uuid4

# Create integrated system
recon = NeuromodulatorIntegratedReconsolidation()

# Process query
query_emb = np.random.randn(128)
recon.orchestra.process_query(query_emb)

# Retrieve memory (sets eligibility trace)
mem_id = uuid4()
mem_emb = np.random.randn(128)
mem_emb /= np.linalg.norm(mem_emb)

recon.orchestra.process_retrieval(
    retrieved_ids=[mem_id],
    scores={str(mem_id): 0.9}
)

# Update with outcome (multiplicative gating applies)
updated_emb = recon.update(
    memory_id=mem_id,
    memory_embedding=mem_emb,
    query_embedding=query_emb,
    outcome_score=0.85
)

# Check learning parameters
params = recon.orchestra.get_learning_params_with_outcome(mem_id, 0.85)
print(f"Effective LR: {params.effective_lr}")
print(f"Eligibility: {params.eligibility}")
print(f"Surprise: {params.surprise}")
print(f"Patience: {params.patience}")
print(f"Combined: {params.combined_learning_signal}")
```

## API Compatibility

**Backward compatible**: Existing code using `NeuromodulatorOrchestra.process_outcome()` continues to work. The return type is still `Dict[str, float]`, but the values are now computed using multiplicative gating instead of additive combination.

**New APIs**:
- `LearningParams` dataclass (exported from `t4dm.learning.neuromodulators`)
- `NeuromodulatorOrchestra.get_learning_params(memory_id)`
- `NeuromodulatorOrchestra.get_learning_params_with_outcome(memory_id, outcome)`
- `NeuromodulatorIntegratedReconsolidation` class (exported from `t4dm.learning.reconsolidation`)

## Future Work

1. **Empirical validation**: Test multiplicative vs additive on real memory tasks
2. **Parameter tuning**: Optimize balance between different signals
3. **Adaptive thresholds**: Learn optimal gating thresholds per memory type
4. **Higher-order interactions**: Explore non-linear combinations beyond simple multiplication

## References

- Hebb's rule + timing: Eligibility traces capture synaptic tagging
- Dopamine modulation: RPE scales plasticity (Schultz 1997)
- Serotonin patience: Temporal discounting (Daw et al. 2002)
- Three-factor learning: Pre/post activity + neuromodulator (Fremaux & Gerstner 2016)
