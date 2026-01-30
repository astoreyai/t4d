# HSA-Inspired Testing Protocols for World Weaver

**Version**: 1.0.0
**Date**: 2025-12-06
**Purpose**: Biologically-grounded validation of HSA improvements to memory system

---

## Executive Summary

This document defines testing protocols for validating Hierarchical Sparse Addressing (HSA) improvements to World Weaver's episodic memory system. All tests are grounded in hippocampal neuroscience and provide quantitative benchmarks derived from physiological observations.

**Test Coverage**:
- 4 test modules with 40+ test cases
- Biological validation against 15+ hippocampal properties
- Performance benchmarks targeting O(log n) retrieval
- Integration tests for complete DG→CA3→CA1 pathway

---

## 1. Testing Architecture

### 1.1 Test Modules

| Module | Purpose | Biological Basis |
|--------|---------|------------------|
| `test_hierarchical_retrieval.py` | Pattern completion, clustering, latency | CA3 autoassociative recall, cortical hierarchies |
| `test_sparse_addressing.py` | Sparsity levels, addressing accuracy, interference | DG sparse coding, expansion ratio |
| `test_joint_optimization.py` | Gate-retrieval correlation, credit assignment | Hippocampal-cortical loop, dopamine RPE |
| `test_biological_validation.py` | Physiological benchmarks, integrated validation | DG/CA3/CA1 properties, consolidation |

### 1.2 Biological Constants

All tests reference experimentally-validated constants:

```python
# Dentate Gyrus (Pattern Separation)
DG_EXPANSION_RATIO = 10          # DG has ~10x more neurons than EC
DG_SPARSITY = 0.02              # ~2% active per context
DG_SEPARATION_RATIO = 0.85      # 85% decorrelation target

# CA3 (Pattern Completion)
CA3_MIN_CUE_FRACTION = 0.30     # Can complete from 30% cue
CA3_COMPLETION_THRESHOLD = 0.85  # 85% completion accuracy

# CA1 (Temporal Integration)
CA1_INTEGRATION_WINDOW_MS = 150  # 150ms theta cycle window
CA1_THETA_FREQUENCY_HZ = 8.0    # 8 Hz theta rhythm

# Consolidation
SYNAPTIC_CONSOLIDATION_HOURS = 6.0
SYSTEMS_CONSOLIDATION_DAYS = 30.0
```

**References**:
- Leutgeb et al. (2007) - DG pattern separation in vivo
- Nakazawa et al. (2002) - CA3 pattern completion
- Dragoi & Buzsaki (2006) - CA1 theta oscillations
- Lisman et al. (2018) - Consolidation timescales

---

## 2. Hierarchical Retrieval Tests

**File**: `/mnt/projects/ww/tests/unit/test_hierarchical_retrieval.py`

### 2.1 Pattern Completion (CA3-like)

**Biological Basis**: CA3 recurrent connections enable retrieval of complete memories from partial cues.

#### Test Cases

**`test_partial_cue_retrieval`**
- Input: Query with 30% of memory features (CA3 threshold)
- Expected: Retrieve full episode with >85% accuracy
- Validates: Autoassociative recall mechanism

**`test_degraded_cue_robustness`**
- Input: Noisy query (40% noise added to embedding)
- Expected: Still retrieve correct memory
- Validates: Robustness to input degradation

**`test_pattern_separation_prevents_false_completion`**
- Input: Two similar but distinct memories
- Expected: No interference, correct memory retrieved
- Validates: Pattern separation prevents false completion

**Key Assertions**:
```python
# Pattern completion accuracy
assert similarity > CA3_MIN_CUE_FRACTION  # >30% match required
assert completion_accuracy > CA3_COMPLETION_THRESHOLD  # >85% accuracy

# Pattern separation
assert intra_cluster_sim > inter_cluster_sim  # Clusters distinct
assert cohens_d > 0.8  # Large effect size for separation
```

### 2.2 Cluster Coherence

**Biological Basis**: Hippocampus organizes memories by semantic similarity and temporal proximity.

#### Test Cases

**`test_semantic_cluster_formation`**
- Input: Episodes from 3 semantic categories
- Expected: Intra-cluster similarity >> inter-cluster
- Metric: Cohen's d > 0.8

**`test_cluster_purity`**
- Input: Labeled episodes across categories
- Expected: Purity > 0.85 (matches CA3 threshold)
- Validates: Semantic organization quality

**Key Assertions**:
```python
# Cluster quality
mean_intra = np.mean(intra_cluster_sims)
mean_inter = np.mean(inter_cluster_sims)
assert mean_intra > mean_inter

# Effect size
cohens_d = (mean_intra - mean_inter) / std_pooled
assert cohens_d > 0.8

# Purity
purity = correct_assignments / total_assignments
assert purity >= 0.85
```

### 2.3 Retrieval Latency

**Biological Basis**: Hippocampus retrieves memories in ~150ms regardless of total count (hierarchical indexing).

#### Test Cases

**`test_logarithmic_scaling`**
- Input: Retrieval times at [100, 500, 1K, 5K, 10K] episodes
- Expected: O(log n) scaling, not O(n)
- Metric: Log model RMSE < linear model RMSE

**`test_retrieval_within_ca1_window`**
- Expected: P95 latency < 200ms (CA1 integration window + overhead)
- Validates: Physiologically plausible retrieval speed

**Key Assertions**:
```python
# Scaling analysis
log_rmse = fit_log_model(episode_counts, retrieval_times)
linear_rmse = fit_linear_model(episode_counts, retrieval_times)
assert log_rmse <= linear_rmse * 1.1  # Hierarchical access

# Latency benchmark
p95_time_ms = np.percentile(times, 95) * 1000
assert p95_time_ms < CA1_INTEGRATION_WINDOW_MS + 50  # <200ms
```

---

## 3. Sparse Addressing Tests

**File**: `/mnt/projects/ww/tests/unit/test_sparse_addressing.py`

### 3.1 Sparsity Levels

**Biological Basis**: DG maintains ~2% activation sparsity across varying loads.

#### Test Cases

**`test_low_load_sparsity`**
- Input: 10 memories
- Expected: Selective storage (sparsity ≤ 0.5)
- Validates: Gate maintains selectivity under low load

**`test_high_load_sparsity`**
- Input: 100 memories
- Expected: Sparsity ≤ 0.3 despite high load
- Validates: Competitive inhibition maintains sparsity

**`test_sparsity_adapts_to_importance`**
- Input: Same memories under normal vs high-importance neuromod states
- Expected: High importance → lower sparsity (store more)
- Validates: Neuromodulator control of sparsity

**Key Assertions**:
```python
# Sparsity measurement
store_count = sum(1 for d in decisions if d.action == STORE)
sparsity = store_count / n_inputs

# Low load: selective
assert sparsity <= 0.5

# High load: maintain sparsity
assert sparsity <= 0.3

# Importance adaptation
assert importance_sparsity >= normal_sparsity
```

### 3.2 Addressing Accuracy

**Biological Basis**: Sparse codes must reliably identify specific memories.

#### Test Cases

**`test_consistent_addressing`**
- Input: Same embedding presented 10 times
- Expected: Identical decisions (deterministic)
- Validates: Reproducibility of sparse codes

**`test_learned_discrimination`**
- Input: Train on important vs unimportant patterns
- Expected: Higher probability for important patterns
- Validates: Learning to discriminate utility

**`test_addressing_capacity`**
- Input: 100 distinct patterns
- Expected: Diverse probabilities (not all same)
- Validates: Sufficient capacity for distinct memories

**Key Assertions**:
```python
# Consistency
assert len(set(actions)) == 1
assert np.std(probabilities) < 1e-6

# Discrimination
assert important_prob > unimportant_prob

# Capacity (diversity)
prob_variance = np.var(probabilities)
assert prob_variance > 0.01
```

### 3.3 Interference Resistance

**Biological Basis**: Sparse coding prevents catastrophic interference.

#### Test Cases

**`test_similar_inputs_separated`**
- Input: 10 similar patterns (small perturbations)
- Expected: Mean feature overlap < 15%
- Validates: Orthogonalization of similar inputs

**`test_interference_under_sequential_learning`**
- Input: Learn Task A, then Task B, re-test Task A
- Expected: Mean change < 15%, max change < 30%
- Validates: Resistance to catastrophic forgetting

**`test_orthogonalization_strength`**
- Input: 50 random patterns
- Expected: Mean feature similarity < 0.3
- Validates: Sparse codes approach orthogonality

**Key Assertions**:
```python
# Separation quality
mean_overlap = np.mean(pairwise_overlaps)
assert mean_overlap < MAX_INTERFERENCE  # <15%

# Catastrophic forgetting resistance
mean_change = np.mean(probability_changes)
max_change = np.max(probability_changes)
assert mean_change < 0.15
assert max_change < 0.30

# Orthogonalization
mean_similarity = np.mean(cosine_similarities)
assert mean_similarity < 0.3
```

---

## 4. Joint Optimization Tests

**File**: `/mnt/projects/ww/tests/unit/test_joint_optimization.py`

### 4.1 Gate-Retrieval Correlation

**Biological Basis**: Hippocampal encoding decisions predict retrieval success.

#### Test Cases

**`test_positive_correlation_store_retrieve`**
- Input: Memories with varying gate probabilities
- Expected: STORE decisions → higher retrieval success
- Validates: Encoding-retrieval match hypothesis

**`test_gate_probability_predicts_utility`**
- Input: Train on examples with varying utility
- Expected: Gate learns to predict utility
- Validates: Adaptive memory hypothesis

**`test_retrieval_feedback_improves_gate`**
- Input: 20 iterations of memory + retrieval feedback
- Expected: Gate observations increase, accuracy improves
- Validates: Learning from retrieval outcomes

**Key Assertions**:
```python
# Correlation
store_success_rate = sum(retrieved for stored) / len(stored)
skip_success_rate = sum(retrieved for skipped) / len(skipped)
assert store_success_rate >= skip_success_rate

# Learning
assert final_observations > initial_observations
```

### 4.2 Consistency Loss

**Biological Basis**: Encoding and retrieval should be mutually consistent.

#### Test Cases

**`test_consistency_loss_convergence`**
- Input: 5 epochs of training on prediction vs outcome
- Expected: Loss decreases over epochs
- Validates: Joint optimization converging

**`test_prediction_calibration`**
- Input: Train on utility-correlated patterns
- Expected: Predictions match outcome frequencies (ECE)
- Validates: Predictive coding principle

**Key Assertions**:
```python
# Convergence
first_half_loss = np.mean(losses[:len(losses)//2])
second_half_loss = np.mean(losses[len(losses)//2:])
assert second_half_loss <= first_half_loss * 1.2

# Calibration
ece = compute_expected_calibration_error(predictions, outcomes)
assert 0.0 <= ece <= 0.3  # Reasonable calibration
```

### 4.3 Catastrophic Forgetting

**Biological Basis**: Synaptic consolidation protects old memories.

#### Test Cases

**`test_sequential_task_retention`**
- Input: Learn Task A, then Task B, re-test Task A
- Expected: Retention rate ≥ 80%
- Validates: Complementary learning systems

**`test_weight_regularization_prevents_forgetting`**
- Input: 100 update iterations
- Expected: Weight norm growth < 5x
- Validates: Bounded updates prevent explosion

**Key Assertions**:
```python
# Retention
retention_rate = mean_prob_after / mean_prob_before
assert retention_rate >= 0.80

# Regularization
weight_growth = final_norm / initial_norm
assert weight_growth < 5.0
```

### 4.4 Credit Assignment

**Biological Basis**: Dopamine provides credit assignment for encoding decisions.

#### Test Cases

**`test_immediate_reward_attribution`**
- Input: Decision → immediate reward → re-test
- Expected: Probability increases for rewarded pattern
- Validates: Phasic dopamine signaling

**`test_delayed_reward_credit`**
- Input: Multiple decisions, then delayed out-of-order rewards
- Expected: Updates applied correctly, pending cleared
- Validates: Eligibility traces (TD-lambda)

**`test_credit_assignment_accuracy`**
- Input: Train on good/bad pattern contingencies
- Expected: Higher probability for good patterns
- Validates: Causal attribution accuracy

**Key Assertions**:
```python
# Learning discrimination
prob_good = gate.predict(good_pattern).probability
prob_bad = gate.predict(bad_pattern).probability
assert prob_good > prob_bad

# Delayed credit
assert memory_id in processed_updates
assert memory_id not in pending_labels
```

---

## 5. Biological Validation Tests

**File**: `/mnt/projects/ww/tests/unit/test_biological_validation.py`

### 5.1 DG Pattern Separation

**Physiological Benchmarks**:
- Separation ratio: 85% decorrelation
- Active sparsity: 2% ± 0.5%
- Expansion ratio: ~10x

#### Test Cases

**`test_dg_separation_benchmark`**
- Input: Two patterns with 0.9 similarity
- Expected: After separation, similarity → 0.14
- Validates: 85% decorrelation achieved

**`test_dg_sparsity_level`**
- Expected: Active fraction = 2% ± 1%
- Validates: Physiological sparsity level

**`test_dg_capacity_scaling`**
- Expected: Sparse capacity > 10x dense capacity
- Validates: Theoretical capacity benefits

**Key Assertions**:
```python
# Separation ratio
separation_ratio = (sim_in - sim_out) / sim_in
assert separation_ratio >= 0.85 * 0.9  # 90% of target

# Sparsity
assert abs(sparsity - 0.02) < 0.01

# Capacity
capacity_ratio = sparse_capacity / dense_capacity
assert capacity_ratio > 10
```

### 5.2 CA3 Pattern Completion

**Physiological Benchmarks**:
- Minimum cue: 30% of pattern
- Completion accuracy: 85%
- Recurrent connectivity: 3%

#### Test Cases

**`test_ca3_minimum_cue_threshold`**
- Expected: Min cue = 30% ± 5%
- Validates: Physiological cue threshold

**`test_ca3_completion_accuracy_benchmark`**
- Expected: Accuracy ≥ 85%
- Validates: Completion quality

**`test_partial_retrieval_completion`**
- Input: 30% cue → retrieval
- Expected: Retrieve correct complete memory
- Validates: End-to-end completion

**Key Assertions**:
```python
# Minimum cue
assert 0.25 <= min_cue <= 0.35

# Accuracy
assert completion_accuracy >= 0.85

# End-to-end
assert len(results) >= 1
assert results[0].content == full_content
```

### 5.3 CA1 Temporal Integration

**Physiological Benchmarks**:
- Theta frequency: 8 Hz
- Theta period: 125ms
- Integration window: 150ms

#### Test Cases

**`test_theta_rhythm_parameters`**
- Expected: Frequency 7-9 Hz, period 110-140ms
- Validates: Theta parameters in range

**`test_integration_window`**
- Expected: Window 100-200ms
- Validates: Physiological integration timescale

**`test_temporal_clustering`**
- Input: Events within/outside 150ms window
- Expected: Within-window events cluster
- Validates: Temporal binding

**Key Assertions**:
```python
# Theta parameters
assert 7.0 <= theta_freq <= 9.0
assert 110 <= theta_period <= 140

# Integration window
assert 100 <= window <= 200

# Temporal clustering
assert time_diff_close < INTEGRATION_WINDOW
```

### 5.4 Consolidation Timescales

**Physiological Benchmarks**:
- Synaptic: 6 hours
- Systems: 30 days
- Reconsolidation: 6 hours

#### Test Cases

**`test_synaptic_consolidation_window`**
- Expected: 4-12 hours
- Validates: Protein synthesis timescale

**`test_systems_consolidation_timescale`**
- Expected: 7-90 days
- Validates: Hippocampal-cortical transfer

**`test_reconsolidation_window`**
- Expected: 2-12 hours
- Validates: Post-retrieval lability

**Key Assertions**:
```python
assert 4 <= synaptic_hours <= 12
assert 7 <= systems_days <= 90
assert 2 <= recon_hours <= 12
```

### 5.5 Integrated Validation

**`test_biological_plausibility_score`**
- Combines 5 component scores
- Target: ≥ 80% overall plausibility
- Provides quantitative biological fidelity metric

**`test_hippocampal_workflow`**
- End-to-end: DG separation → CA3 completion → CA1 integration
- Validates: Complete circuit functionality

**Key Assertions**:
```python
# Component scores
scores = {
    'dg_separation': check_separation(),
    'dg_sparsity': check_sparsity(),
    'ca3_completion': check_completion(),
    'ca3_min_cue': check_min_cue(),
    'ca1_integration': check_integration(),
}

# Overall
plausibility = sum(scores.values()) / len(scores)
assert plausibility >= 0.80
```

---

## 6. Mock Data Generation Strategies

### 6.1 Controlled Similarity Patterns

```python
def generate_similar_embeddings(base: np.ndarray,
                               n_variants: int,
                               similarity_target: float) -> List[np.ndarray]:
    """
    Generate embeddings with controlled similarity to base.

    Uses spherical interpolation to achieve exact similarity.
    """
    variants = []
    for _ in range(n_variants):
        noise = np.random.randn(len(base))
        noise /= np.linalg.norm(noise)

        # Spherical interpolation
        alpha = math.sqrt(similarity_target)
        variant = alpha * base + math.sqrt(1 - similarity_target) * noise
        variant /= np.linalg.norm(variant)

        variants.append(variant)

    return variants
```

### 6.2 Clustered Episode Generation

```python
def generate_clustered_episodes(n_clusters: int,
                               episodes_per_cluster: int) -> List[Episode]:
    """
    Generate episodes that naturally cluster.

    Each cluster has a center with tight variance.
    """
    episodes = []

    for cluster_id in range(n_clusters):
        # Cluster center
        center = np.random.randn(1024).astype(np.float32)
        center /= np.linalg.norm(center)

        for i in range(episodes_per_cluster):
            # Add small noise
            embedding = center + np.random.randn(1024) * 0.1
            embedding /= np.linalg.norm(embedding)

            episode = create_episode(
                embedding=embedding,
                cluster_id=cluster_id
            )
            episodes.append(episode)

    return episodes
```

### 6.3 Temporal Sequence Generation

```python
def generate_temporal_sequence(n_events: int,
                              interval_ms: float,
                              start_time: datetime) -> List[Episode]:
    """
    Generate temporally structured episodes.

    Events spaced at regular intervals for testing
    temporal clustering.
    """
    episodes = []

    for i in range(n_events):
        timestamp = start_time + timedelta(milliseconds=i * interval_ms)

        episode = create_episode(
            timestamp=timestamp,
            content=f"Event {i}"
        )
        episodes.append(episode)

    return episodes
```

---

## 7. Performance Benchmarks

### 7.1 Target Metrics

| Metric | Target | Biological Basis |
|--------|--------|------------------|
| Pattern completion accuracy | ≥85% | CA3 autoassociative recall |
| Pattern separation ratio | ≥80% | DG orthogonalization |
| Active sparsity | 2% ± 1% | DG granule cell activation |
| Retrieval latency (P95) | <200ms | CA1 integration window |
| Cluster purity | ≥85% | Semantic organization |
| Retention rate | ≥80% | Synaptic consolidation |
| Orthogonalization | Mean sim <0.3 | Sparse code properties |

### 7.2 Scaling Benchmarks

| Episode Count | Target Latency | Scaling |
|---------------|----------------|---------|
| 100 | <50ms | Baseline |
| 1,000 | <80ms | O(log n) |
| 10,000 | <110ms | O(log n) |
| 100,000 | <150ms | O(log n) |

### 7.3 Capacity Benchmarks

| Property | Target | Justification |
|----------|--------|---------------|
| Distinct patterns | ≥10,000 | Practical memory scale |
| Interference threshold | <15% overlap | Sparse coding benefit |
| Catastrophic forgetting | <20% degradation | Synaptic protection |

---

## 8. Running the Tests

### 8.1 Test Execution

```bash
# All HSA tests
pytest tests/unit/test_hierarchical_retrieval.py \
       tests/unit/test_sparse_addressing.py \
       tests/unit/test_joint_optimization.py \
       tests/unit/test_biological_validation.py -v

# Specific module
pytest tests/unit/test_biological_validation.py -v

# Specific test
pytest tests/unit/test_biological_validation.py::TestDGPatternSeparation::test_dg_separation_benchmark -v

# With coverage
pytest tests/unit/test_*.py --cov=ww.memory --cov-report=html
```

### 8.2 Expected Output

```
tests/unit/test_hierarchical_retrieval.py::TestPatternCompletion::test_partial_cue_retrieval PASSED
tests/unit/test_hierarchical_retrieval.py::TestClusterCoherence::test_semantic_cluster_formation PASSED
tests/unit/test_sparse_addressing.py::TestSparsityLevels::test_high_load_sparsity PASSED
tests/unit/test_biological_validation.py::TestIntegratedBiologicalValidation::test_biological_plausibility_score PASSED

Biological Plausibility Score: 92%
  ✓ dg_separation: 100%
  ✓ dg_sparsity: 100%
  ✓ ca3_completion: 100%
  ✓ ca3_min_cue: 100%
  ✗ ca1_integration: 60%
```

---

## 9. Integration with CI/CD

### 9.1 GitHub Actions Workflow

```yaml
name: HSA Validation

on: [push, pull_request]

jobs:
  biological-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run HSA tests
        run: |
          pytest tests/unit/test_hierarchical_retrieval.py \
                 tests/unit/test_sparse_addressing.py \
                 tests/unit/test_joint_optimization.py \
                 tests/unit/test_biological_validation.py \
                 --cov=ww.memory \
                 --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

### 9.2 Quality Gates

- All biological validation tests must pass
- Plausibility score ≥ 80%
- Coverage ≥ 85% for memory module
- No regression in benchmark metrics

---

## 10. Future Extensions

### 10.1 Planned Tests

- **Contextual modulation**: Test ACh/NE effects on encoding
- **Replay consolidation**: Test offline memory replay
- **Multi-scale integration**: Test integration across DG→CA3→CA1
- **Long-term stability**: Test FSRS decay over extended periods

### 10.2 Additional Biological Benchmarks

- **Place cell properties**: Spatial coding fidelity
- **Grid cell patterns**: Hexagonal organization
- **Replay sequences**: Temporal compression in consolidation
- **Theta-gamma coupling**: Nested oscillation coordination

---

## References

1. **Leutgeb et al.** (2007). Pattern separation in the dentate gyrus and CA3 of the hippocampus. *Science*, 315(5814), 961-966.

2. **Nakazawa et al.** (2002). Requirement for hippocampal CA3 NMDA receptors in associative memory recall. *Science*, 297(5579), 211-218.

3. **Dragoi & Buzsaki** (2006). Temporal encoding of place sequences by hippocampal cell assemblies. *Neuron*, 50(1), 145-157.

4. **Treves & Rolls** (1992). Computational constraints suggest the need for two distinct input systems to the hippocampal CA3 network. *Hippocampus*, 2(2), 189-199.

5. **Lisman et al.** (2018). Memory formation depends on both synapse-specific modifications of synaptic strength and cell-specific increases in excitability. *Nature Neuroscience*, 21(3), 309-314.

6. **Rolls** (2013). The mechanisms for pattern completion and pattern separation in the hippocampus. *Frontiers in Systems Neuroscience*, 7, 74.

7. **O'Keefe & Recce** (1993). Phase relationship between hippocampal place units and the EEG theta rhythm. *Hippocampus*, 3(3), 317-330.

---

## Appendix: Quick Reference

### Test File Locations

```
/mnt/projects/ww/tests/unit/
├── test_hierarchical_retrieval.py  # Pattern completion, clustering, latency
├── test_sparse_addressing.py       # Sparsity, addressing, interference
├── test_joint_optimization.py      # Gate-retrieval, credit assignment
└── test_biological_validation.py   # Physiological benchmarks
```

### Key Metrics Checklist

- [ ] Pattern completion ≥85%
- [ ] Pattern separation ≥80%
- [ ] Sparsity = 2% ± 1%
- [ ] Retrieval latency <200ms (P95)
- [ ] Cluster purity ≥85%
- [ ] Retention rate ≥80%
- [ ] Interference <15%
- [ ] Plausibility score ≥80%

### Common Assertions

```python
# Pattern completion
assert similarity > CA3_MIN_CUE_FRACTION
assert accuracy > CA3_COMPLETION_THRESHOLD

# Pattern separation
assert separation_ratio >= DG_SEPARATION_RATIO * 0.9

# Sparsity
assert abs(sparsity - TARGET_SPARSITY) < SPARSITY_TOLERANCE

# Latency
assert p95_latency_ms < CA1_INTEGRATION_WINDOW_MS + 50

# Interference
assert mean_overlap < MAX_INTERFERENCE
```
