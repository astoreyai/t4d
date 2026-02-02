# World Weaver Learning Flow

**Phase 5 Documentation** | **Last Updated**: 2026-01-05

This document describes how World Weaver **actually learns** representations through use, addressing the critical "Learning Gap" identified by the Hinton agent analysis.

---

## The Learning Gap (Pre-Phase 5)

Before Phase 5, World Weaver had a fundamental limitation:

```
BEFORE (No Learning):
    Input Text
        ↓
    [Frozen BGE-M3 Embedder] ─→ Fixed 1024-dim vector
        ↓
    Store in Qdrant
        ↓
    Retrieve by cosine similarity

    Problem: The system STORES representations but never LEARNS them.
    Embeddings are frozen at creation time.
```

The system had sophisticated learning infrastructure (three-factor rule, eligibility traces, dopamine RPE) but **the loop was not closed** - nothing updated the stored representations.

---

## The Solution: Learnable FF Encoder

Phase 5 introduces a trainable Forward-Forward encoder stack between the frozen embedder and storage:

```
AFTER (Learns):
    Input Text
        ↓
    [Frozen BGE-M3 Embedder] ─→ Raw 1024-dim vector
        ↓
    [FFEncoder (LEARNABLE)]  ─→ Refined 1024-dim vector
        ↓                           ↑
    Store in Qdrant          Three-Factor Learning Signal
        ↓                    (eligibility × neuromod × dopamine)
    Retrieve → Use → Outcome → Learn
```

---

## Complete Learning Flow

### 1. Encoding (Storage)

```python
# In EpisodicMemory.create()

# Step 1: Get frozen embedding
embedding = await self.embedding.embed_query(content)

# Step 2: Transform through learnable FF layers (NEW)
if self._ff_encoder_enabled:
    encoded = self._ff_encoder.encode(embedding)
    embedding = encoded

# Step 3: Store the LEARNED representation
episode = Episode(embedding=embedding, ...)
await self.vector_store.add(...)
```

### 2. Retrieval

```python
# In EpisodicMemory.recall()

# Step 1: Get frozen query embedding
query_emb = await self.embedding.embed_query(query)

# Step 2: Transform query through SAME FF layers (NEW)
if self._ff_encoder_enabled:
    query_emb = self._ff_encoder.encode(query_emb)

# Step 3: Search in learned embedding space
results = await self.vector_store.search(query_emb, ...)
```

### 3. Learning from Outcomes

```python
# In EpisodicMemory.learn_from_outcome()

# Step 1: Compute three-factor learning signal
for episode_id in episode_ids:
    # Mark as active (eligibility trace)
    self.three_factor.mark_active(episode_id)

    # Get combined signal
    signal = self.three_factor.compute(
        memory_id=episode_id,
        base_lr=0.03,
        outcome=outcome_score,
    )
    # signal.effective_lr_multiplier = eligibility × neuromod × dopamine

    # Step 2: Update FF encoder weights (THE KEY STEP)
    self._ff_encoder.learn_from_outcome(
        embedding=query_embedding,
        outcome_score=outcome_score,
        three_factor_signal=signal,
    )
```

---

## Three-Factor Learning Rule

The learning rate is modulated by three biological factors:

```
effective_lr = base_lr × eligibility × neuromod_gate × dopamine_surprise

Where:
- eligibility   [0,1]: Was this memory recently active? (temporal credit)
- neuromod_gate [0.5-2.0]: Should we learn now? (ACh, NE, 5-HT)
- dopamine      [0.1-2.0]: How surprising was this? (|RPE|)
```

### Eligibility Traces (What to Update)

```python
# Recently active memories have high eligibility
trace = exp(-elapsed_time / tau)

# Only recently active memories get substantial updates
if eligibility < 0.01:
    effective_lr *= 0.1  # Minimal learning
```

### Neuromodulator Gate (When to Learn)

```python
# ACh: Encoding mode boosts learning
if acetylcholine_mode == "encoding":
    ach_factor = 1.5
elif acetylcholine_mode == "retrieval":
    ach_factor = 0.6

# NE: Arousal modulates directly
ne_factor = norepinephrine_gain  # [0.5, 2.0]

# 5-HT: Moderate mood is optimal
serotonin_factor = 1.0 - 0.5 * |mood - 0.5|

# Combined gate
neuromod_gate = 0.4*ach + 0.35*ne + 0.25*serotonin
```

### Dopamine Surprise (How Much to Learn)

```python
# Prediction error = actual - expected
rpe = outcome - expected_value

# Surprise is magnitude of RPE
dopamine_surprise = max(|rpe|, 0.1)

# Expected outcomes (low surprise) → minimal updates
# Surprising outcomes (high |rpe|) → larger updates
```

---

## Forward-Forward Learning

The FFEncoder uses Hinton's Forward-Forward algorithm instead of backpropagation:

### Layer Architecture

```
Input (1024)
    ↓
[FF Layer 1] (512) ─→ ReLU → Normalize
    ↓
[FF Layer 2] (256) ─→ ReLU → Normalize
    ↓
[Output Projection] (1024)
    +
[Residual Connection] (skip from input)
    ↓
Output (1024, normalized)
```

### Local Learning Rule

Each layer learns independently via contrastive goodness:

```python
# Goodness = sum of squared activations
goodness = (activations ** 2).sum()

# Positive phase (helpful retrieval): increase goodness
if outcome > 0.5:
    W += lr * (threshold - goodness) * outer(x, grad)

# Negative phase (unhelpful retrieval): decrease goodness
if outcome < 0.5:
    W -= lr * (goodness - threshold) * outer(x, grad)
```

**Key advantage**: No backpropagation through layers, more biologically plausible.

---

## Consolidation (Sleep Replay)

During consolidation, generative replay prevents catastrophic forgetting:

```python
# In FFEncoder.replay_consolidation()

# 1. Sample real patterns from history
real_patterns = sample(self._encode_history, n_positive)

# 2. Generate negative patterns (corrupted versions)
negative_patterns = [corrupt(p) for p in sample(history, n_negative)]

# 3. Interleaved learning
for pattern in real_patterns:
    layer.learn_positive(pattern)  # Increase goodness

for pattern in negative_patterns:
    layer.learn_negative(pattern)  # Decrease goodness
```

This mimics hippocampal replay during sleep, where real memories are interleaved with "negative" examples to maintain discrimination.

---

## Biological Mapping

| World Weaver Component | Brain Analog |
|------------------------|--------------|
| Frozen BGE-M3 Embedder | Sensory cortex (feature extraction) |
| FFEncoder | Hippocampal CA3 (pattern completion/separation) |
| Three-Factor Rule | Neuromodulated Hebbian plasticity |
| Eligibility Traces | Synaptic tags |
| Sleep Replay | Systems consolidation |
| Dopamine RPE | VTA reward prediction |

---

## Configuration

Enable/disable the learnable encoder in settings:

```python
# In config or environment
ff_encoder_enabled = True  # Default: True

# FFEncoder configuration
FFEncoderConfig(
    input_dim=1024,           # Match embedding dimension
    hidden_dims=(512, 256),   # FF layer sizes
    output_dim=1024,          # Match input for storage
    learning_rate=0.03,       # Base learning rate
    use_residual=True,        # Skip connection for stability
    use_neuromod_gating=True, # NT integration
)
```

---

## Verification

To verify the system is actually learning:

```python
from ww.encoding import get_ff_encoder

encoder = get_ff_encoder()
stats = encoder.get_stats()

# Check that learning is happening
print(f"Total encodes: {stats['state']['total_encodes']}")
print(f"Positive updates: {stats['state']['total_positive_updates']}")
print(f"Negative updates: {stats['state']['total_negative_updates']}")
print(f"Mean goodness: {stats['state']['mean_goodness']:.3f}")

# Weights should change after learning
# Positive outcomes should increase goodness for similar patterns
# System should improve retrieval quality over time
```

---

## Key Files

- `src/t4dm/encoding/ff_encoder.py` - FFEncoder implementation
- `src/t4dm/learning/three_factor.py` - Three-factor learning rule
- `src/t4dm/memory/episodic.py` - Integration in create/recall/learn
- `tests/encoding/test_ff_encoder.py` - Tests

---

## Success Metrics

The learning gap is fixed when:

1. **Weights Change**: `encoder.state.total_positive_updates > 0`
2. **Goodness Increases**: Positive outcomes increase pattern goodness
3. **Retrieval Improves**: Quality metrics improve with use
4. **No Catastrophic Forgetting**: Old patterns retained after new learning

---

## References

- Hinton, G. (2022). The Forward-Forward Algorithm. arXiv:2212.13345
- Frémaux, N. & Gerstner, W. (2016). Neuromodulated STDP
- McClelland, J.L. et al. (1995). Complementary Learning Systems
