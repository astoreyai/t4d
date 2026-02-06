# Ablation Study Framework for T4DM

**Author**: T4DM Development Team
**Status**: Methodology Document
**Last Updated**: 2026-02-06

---

## Overview

This document provides a systematic framework for conducting ablation studies on T4DM's biologically-inspired components. Ablation studies help identify the contribution of each component to overall system performance, validate design decisions, and guide future development priorities.

### What is an Ablation Study?

An ablation study systematically removes or disables individual components of a system to measure their contribution to overall performance. In neuroscience, this mirrors lesion studies where specific brain regions are inactivated to understand their function.

### Goals

1. **Quantify component contributions** to retrieval accuracy, consolidation efficiency, and learning dynamics
2. **Validate biological hypotheses** about neuromodulator functions and spiking mechanisms
3. **Identify critical vs. optional components** for deployment optimization
4. **Guide hyperparameter tuning** by understanding component interactions

---

## Ablatable Components

### 1. Neuromodulator Systems

The NeuromodulatorOrchestra coordinates five systems that modulate learning and memory dynamics.

| Component | File | Function | Ablation Method |
|-----------|------|----------|-----------------|
| **Dopamine (DA)** | `learning/dopamine.py` | Reward prediction error, surprise-driven learning | Set `dopamine_rpe=0.0` in NeuromodulatorState |
| **Norepinephrine (NE)** | `learning/norepinephrine.py` | Arousal, attention, novelty detection | Set `norepinephrine_gain=1.0` (neutral) |
| **Acetylcholine (ACh)** | `learning/acetylcholine.py` | Encoding/retrieval mode switching | Set `acetylcholine_mode="balanced"` always |
| **Serotonin (5-HT)** | `learning/serotonin.py` | Long-term credit assignment, patience | Set `serotonin_mood=0.5` (neutral) |
| **GABA/Inhibition** | `learning/inhibition.py` | Competitive dynamics, sparse retrieval | Bypass `InhibitoryNetwork.apply_inhibition()` |

**Implementation**: Create `AblatedNeuromodulatorOrchestra` subclass:

```python
class AblatedNeuromodulatorOrchestra(NeuromodulatorOrchestra):
    """Orchestra with selective component ablation."""

    def __init__(
        self,
        ablate_dopamine: bool = False,
        ablate_norepinephrine: bool = False,
        ablate_acetylcholine: bool = False,
        ablate_serotonin: bool = False,
        ablate_inhibition: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._ablate_da = ablate_dopamine
        self._ablate_ne = ablate_norepinephrine
        self._ablate_ach = ablate_acetylcholine
        self._ablate_5ht = ablate_serotonin
        self._ablate_gaba = ablate_inhibition

    def process_query(self, query_embedding, **kwargs) -> NeuromodulatorState:
        state = super().process_query(query_embedding, **kwargs)
        if self._ablate_da:
            state.dopamine_rpe = 0.0
        if self._ablate_ne:
            state.norepinephrine_gain = 1.0
        if self._ablate_ach:
            state.acetylcholine_mode = "balanced"
        if self._ablate_5ht:
            state.serotonin_mood = 0.5
        return state
```

### 2. Spiking Cortical Block Stages

The CorticalBlock has 6 stages that can be individually ablated.

| Stage | Component | File | Function | Ablation Method |
|-------|-----------|------|----------|-----------------|
| 1 | **Thalamic Gate** | `spiking/thalamic_gate.py` | ACh-modulated input masking | Pass-through (return input unchanged) |
| 2 | **LIF Integration** | `spiking/lif.py` | Spike generation | Return input as continuous values |
| 3 | **Spike Attention** | `spiking/spike_attention.py` | STDP-weighted attention | Use uniform attention weights |
| 4 | **Apical Modulation** | `spiking/apical_modulation.py` | Prediction error + FF goodness | Pass-through (no modulation) |
| 5 | **RWKV Recurrence** | `spiking/rwkv_recurrence.py` | Linear recurrence | Pass-through (no recurrence) |
| 6 | **LIF Output** | `spiking/lif.py` | Output spike generation | Return continuous values |

**Implementation**: Create ablation flags in CorticalBlock:

```python
class AblatedCorticalBlock(CorticalBlock):
    """Cortical block with selective stage ablation."""

    def __init__(
        self,
        dim: int,
        ablate_thalamic: bool = False,
        ablate_lif_input: bool = False,
        ablate_attention: bool = False,
        ablate_apical: bool = False,
        ablate_rwkv: bool = False,
        ablate_lif_output: bool = False,
        **kwargs
    ):
        super().__init__(dim, **kwargs)
        self.ablations = {
            "thalamic": ablate_thalamic,
            "lif_input": ablate_lif_input,
            "attention": ablate_attention,
            "apical": ablate_apical,
            "rwkv": ablate_rwkv,
            "lif_output": ablate_lif_output,
        }
```

### 3. Learning Rules

| Component | File | Function | Ablation Method |
|-----------|------|----------|-----------------|
| **STDP** | `learning/stdp.py` | Spike-timing plasticity | Set `a_plus=0, a_minus=0` |
| **Three-Factor Rule** | `learning/three_factor.py` | Eligibility * neuromod * DA | Return `effective_lr=base_lr` |
| **Hebbian** | `learning/plasticity.py` | Correlation-based learning | Disable weight updates |
| **Anti-Hebbian** | `learning/anti_hebbian.py` | Decorrelation learning | Disable weight updates |
| **BCM Metaplasticity** | `learning/bcm_metaplasticity.py` | Sliding LTP/LTD threshold | Fix threshold at 0.5 |
| **Homeostatic** | `learning/homeostatic.py` | Activity regulation | Disable scaling |
| **Eligibility Traces** | `learning/eligibility.py` | Temporal credit assignment | Return eligibility=1.0 (no decay) |

### 4. Consolidation Components

| Component | File | Function | Ablation Method |
|-----------|------|----------|-----------------|
| **NREM Phase** | `consolidation/sleep.py` | Sharp-Wave Ripple replay | Skip NREM cycles |
| **REM Phase** | `consolidation/sleep.py` | Clustering, abstraction | Skip REM cycles |
| **PRUNE Phase** | `consolidation/sleep.py` | Weak synapse deletion | Skip pruning |
| **Lability Window** | `consolidation/lability.py` | Reconsolidation eligibility | Always return eligible |
| **Adaptive Trigger** | `consolidation/adaptive_trigger.py` | Smart consolidation timing | Use fixed intervals |

### 5. Memory Architecture Components

| Component | Location | Function | Ablation Method |
|-----------|----------|----------|-----------------|
| **Time2Vec Encoding** | `encoding/time2vec.py` | Temporal embedding | Use raw timestamps |
| **Temporal Gate (tau)** | `core/temporal_gate.py` | Memory write gating | Always allow writes |
| **Kappa Gradient** | `core/memory_item.py` | Continuous consolidation level | Use discrete categories |
| **Pattern Separation** | `memory/pattern_separation.py` | Input orthogonalization | Pass-through |

### 6. NeuromodBus Routing

| Mapping | Effect | Ablation |
|---------|--------|----------|
| DA -> L2/3, L5 | Attention, apical stages | Neutral RPE (0.0) |
| NE -> L5 | LIF, output stages | Neutral gain (1.0) |
| ACh -> L1/L4 | Thalamic gate | Balanced mode |
| 5-HT -> L5/6 | RWKV patience | Neutral mood (0.5) |

---

## Baseline Configuration

The baseline (full system) should have all components enabled with standard hyperparameters.

```python
BASELINE_CONFIG = {
    # Neuromodulators
    "neuromodulators": {
        "dopamine": {"enabled": True, "learning_rate": 0.1},
        "norepinephrine": {"enabled": True, "baseline_arousal": 0.5},
        "acetylcholine": {"enabled": True, "encoding_threshold": 0.7},
        "serotonin": {"enabled": True, "discount_rate": 0.99},
        "inhibition": {"enabled": True, "strength": 0.5},
    },

    # Spiking
    "spiking": {
        "thalamic_gate": {"enabled": True},
        "lif_neurons": {"enabled": True, "alpha": 0.9, "v_thresh": 1.0},
        "spike_attention": {"enabled": True, "num_heads": 8},
        "apical_modulation": {"enabled": True},
        "rwkv_recurrence": {"enabled": True},
    },

    # Learning
    "learning": {
        "stdp": {"enabled": True, "a_plus": 0.01, "a_minus": 0.0105},
        "three_factor": {"enabled": True},
        "hebbian": {"enabled": True},
        "eligibility_traces": {"enabled": True, "decay": 0.95},
    },

    # Consolidation
    "consolidation": {
        "nrem_phases": 4,
        "rem_enabled": True,
        "prune_enabled": True,
        "lability_window_hours": 6.0,
    },

    # Memory
    "memory": {
        "time2vec": {"enabled": True, "periodic_dims": 16},
        "temporal_gate": {"enabled": True},
        "kappa_gradient": {"enabled": True},
        "pattern_separation": {"enabled": True, "strength": 0.3},
    },
}
```

---

## Ablation Variants

### Single-Component Ablations

Each variant disables exactly one component from the baseline.

| Variant ID | Component Ablated | Expected Impact |
|------------|-------------------|-----------------|
| A1-DA | Dopamine | Reduced reward-based learning |
| A1-NE | Norepinephrine | Reduced novelty detection, flat arousal |
| A1-ACh | Acetylcholine | No encoding/retrieval mode switching |
| A1-5HT | Serotonin | Impaired long-term credit assignment |
| A1-GABA | Inhibition | Less sparse, noisier retrieval |
| A2-THA | Thalamic Gate | No input gating |
| A2-LIF | LIF Neurons | No spiking dynamics |
| A2-ATT | Spike Attention | Uniform attention |
| A2-API | Apical Modulation | No prediction error feedback |
| A2-RWK | RWKV Recurrence | No recurrent context |
| A3-STDP | STDP | No timing-based plasticity |
| A3-3FAC | Three-Factor | No eligibility-gated learning |
| A3-ELI | Eligibility Traces | No temporal credit assignment |
| A4-NREM | NREM Phase | No replay consolidation |
| A4-REM | REM Phase | No clustering/abstraction |
| A4-PRU | Pruning | No weak synapse removal |
| A5-T2V | Time2Vec | No temporal encoding |
| A5-TAU | Temporal Gate | Unfiltered memory writes |
| A5-KAP | Kappa Gradient | Discrete memory types |

### Multi-Component Ablations (Interaction Studies)

| Variant ID | Components Ablated | Purpose |
|------------|-------------------|---------|
| AM-NEURO | All neuromodulators | Baseline without neuromodulation |
| AM-SPIKE | All spiking components | Baseline without spiking |
| AM-LEARN | STDP + Three-Factor | Baseline without bioplausible learning |
| AM-SLEEP | NREM + REM + Prune | Baseline without consolidation |
| AM-DA-5HT | Dopamine + Serotonin | Reward system interaction |
| AM-NE-ACH | NE + ACh | Attention/mode interaction |
| AM-LIF-STDP | LIF + STDP | Spiking + plasticity interaction |

---

## Metrics to Track

### Primary Metrics

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Retrieval Accuracy** | Fraction of retrievals returning relevant memories | `relevant_retrieved / total_retrieved` |
| **Recall@K** | Fraction of relevant memories in top-K results | Standard IR metric |
| **MRR** | Mean Reciprocal Rank of first relevant result | `1/rank` averaged |
| **Consolidation Rate** | Memories consolidated per cycle | `consolidated_count / cycle_time` |
| **Kappa Progression** | Average kappa increase per consolidation | `mean(kappa_after - kappa_before)` |
| **Prune Ratio** | Fraction of memories pruned | `pruned / total` |

### Neuromodulator-Specific Metrics

| Metric | Component | Expected Behavior |
|--------|-----------|-------------------|
| **RPE Variance** | Dopamine | Higher variance = more surprise-driven learning |
| **Arousal Range** | NE | Wider range = better novelty detection |
| **Mode Switch Rate** | ACh | Frequency of encoding/retrieval transitions |
| **Eligibility Decay** | 5-HT | Longer traces = better long-term credit |
| **Sparsity** | GABA | Higher sparsity = more selective retrieval |

### Spiking-Specific Metrics

| Metric | Component | Expected Behavior |
|--------|-----------|-------------------|
| **Spike Rate** | LIF | Rate should stabilize under homeostasis |
| **Attention Entropy** | Spike Attention | Lower entropy = more focused attention |
| **Prediction Error** | Apical | Should decrease with learning |
| **Recurrence Norm** | RWKV | Bounded growth = stable dynamics |

### Learning-Specific Metrics

| Metric | Component | Expected Behavior |
|--------|-----------|-------------------|
| **Weight Drift** | STDP | Net weight change over time |
| **LTP/LTD Ratio** | STDP | Should balance for stability |
| **Effective LR** | Three-Factor | Should vary with context |
| **Credit Assignment Lag** | Eligibility | Time from action to credit |

### Consolidation-Specific Metrics

| Metric | Component | Expected Behavior |
|--------|-----------|-------------------|
| **Replay Count** | NREM | Memories replayed per cycle |
| **Cluster Purity** | REM | Semantic coherence of clusters |
| **Prune Threshold** | PRUNE | Adaptive threshold dynamics |
| **Memory Survival** | Overall | Retention curve over time |

---

## Test Procedure

### Phase 1: Baseline Establishment

1. **Initialize System**
   ```python
   system = T4DMSystem(config=BASELINE_CONFIG)
   ```

2. **Load Training Data**
   - Use standardized dataset (e.g., synthetic episodic sequences)
   - Ensure reproducibility with fixed random seed

3. **Training Run**
   ```python
   for epoch in range(N_EPOCHS):
       for batch in training_data:
           system.encode(batch)
           system.retrieve(batch.queries)
           system.update(batch.outcomes)
       if epoch % CONSOLIDATION_INTERVAL == 0:
           system.consolidate()
   ```

4. **Record Baseline Metrics**
   - All primary and component-specific metrics
   - Save model checkpoints

### Phase 2: Ablation Runs

For each ablation variant:

1. **Configure Ablation**
   ```python
   ablation_config = BASELINE_CONFIG.copy()
   ablation_config[component]["enabled"] = False
   system = T4DMSystem(config=ablation_config)
   ```

2. **Run Identical Training**
   - Same data, same epochs, same random seed
   - Same consolidation schedule

3. **Record Ablation Metrics**
   - Delta from baseline for each metric
   - Statistical significance tests

### Phase 3: Analysis

1. **Compute Deltas**
   ```python
   delta = ablation_metrics - baseline_metrics
   relative_delta = delta / baseline_metrics
   ```

2. **Statistical Tests**
   - Paired t-test for significance
   - Effect size (Cohen's d)
   - 95% confidence intervals

3. **Generate Reports**
   - Per-component contribution scores
   - Interaction effects for multi-ablations
   - Visualization of metric distributions

---

## Expected Hypotheses

### Neuromodulator Hypotheses

| Ablation | Hypothesis | Rationale |
|----------|------------|-----------|
| **A1-DA** | Retrieval accuracy decreases by 15-25% | Dopamine drives surprise-based learning; without it, system cannot adapt to prediction errors |
| **A1-NE** | Recall@K decreases for novel queries by 20-30% | NE modulates novelty detection; ablation impairs response to new situations |
| **A1-ACh** | Encoding speed decreases by 30-40% | ACh controls encoding mode; fixed balanced mode is suboptimal for new memories |
| **A1-5HT** | Long-term retention decreases by 25-35% | Serotonin handles credit assignment; ablation breaks temporal reward propagation |
| **A1-GABA** | Retrieval precision decreases by 20-30% | Inhibition sharpens retrieval; ablation leads to noisy, less selective results |

### Spiking Hypotheses

| Ablation | Hypothesis | Rationale |
|----------|------------|-----------|
| **A2-THA** | Encoding noise increases by 40% | Thalamic gate filters inputs; ablation admits irrelevant information |
| **A2-LIF** | Temporal dynamics lost entirely | Spiking is fundamental; ablation converts to continuous (non-bioplausible) |
| **A2-ATT** | Context sensitivity decreases by 25% | Attention focuses on relevant inputs; uniform attention is less selective |
| **A2-API** | Prediction accuracy unchanged but learning slower | Apical modulation provides feedback; ablation removes one learning signal |
| **A2-RWK** | Sequence processing accuracy decreases by 35% | RWKV handles recurrence; ablation breaks temporal integration |

### Learning Hypotheses

| Ablation | Hypothesis | Rationale |
|----------|------------|-----------|
| **A3-STDP** | Temporal correlation learning fails | STDP is timing-based; ablation removes bioplausible Hebbian mechanism |
| **A3-3FAC** | Learning becomes context-insensitive | Three-factor gates by eligibility; ablation makes all weights equally updatable |
| **A3-ELI** | Credit assignment becomes immediate-only | Eligibility traces bridge time; ablation breaks distal reward learning |

### Consolidation Hypotheses

| Ablation | Hypothesis | Rationale |
|----------|------------|-----------|
| **A4-NREM** | Memory retention at 24h decreases by 50% | NREM replay strengthens memories; ablation breaks consolidation |
| **A4-REM** | Semantic abstraction fails | REM clustering creates concepts; ablation keeps all memories episodic |
| **A4-PRU** | Memory usage grows unboundedly | Pruning removes weak memories; ablation leads to memory bloat |

### Interaction Hypotheses

| Ablation | Hypothesis | Rationale |
|----------|------------|-----------|
| **AM-DA-5HT** | Reward learning fails completely | DA (immediate) + 5-HT (long-term) form complete reward system |
| **AM-NE-ACH** | System becomes statically configured | NE (arousal) + ACh (mode) provide dynamic adaptation |
| **AM-LIF-STDP** | Bioplausibility lost, reverts to backprop-like | LIF + STDP are core bioplausible mechanisms |

---

## Results Template

### Single-Component Ablation Results

```markdown
## Ablation: [COMPONENT_NAME]

**Variant ID**: A[X]-[CODE]
**Date**: YYYY-MM-DD
**Run ID**: [UUID]

### Configuration
- Baseline config with [COMPONENT] disabled
- Training epochs: [N]
- Random seed: [SEED]

### Primary Metrics

| Metric | Baseline | Ablated | Delta | Relative | p-value |
|--------|----------|---------|-------|----------|---------|
| Retrieval Accuracy | 0.XX | 0.XX | -0.XX | -XX% | 0.XXX |
| Recall@10 | 0.XX | 0.XX | -0.XX | -XX% | 0.XXX |
| MRR | 0.XX | 0.XX | -0.XX | -XX% | 0.XXX |
| Consolidation Rate | X.XX | X.XX | -X.XX | -XX% | 0.XXX |
| Kappa Progression | 0.XX | 0.XX | -0.XX | -XX% | 0.XXX |

### Component-Specific Metrics

| Metric | Baseline | Ablated | Delta | Notes |
|--------|----------|---------|-------|-------|
| [METRIC1] | X.XX | X.XX | -X.XX | [OBSERVATION] |
| [METRIC2] | X.XX | X.XX | -X.XX | [OBSERVATION] |

### Hypothesis Evaluation

**Hypothesis**: [STATED HYPOTHESIS]
**Result**: [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
**Effect Size (Cohen's d)**: X.XX ([SMALL/MEDIUM/LARGE])
**Notes**: [OBSERVATIONS]

### Qualitative Observations

- [Observation 1]
- [Observation 2]
- [Observation 3]
```

### Summary Table Template

```markdown
## Ablation Study Summary

| Variant | Component | Accuracy Delta | Recall@10 Delta | Consolidation Delta | Hypothesis |
|---------|-----------|----------------|-----------------|---------------------|------------|
| A1-DA | Dopamine | -XX% | -XX% | -XX% | Supported |
| A1-NE | Norepinephrine | -XX% | -XX% | -XX% | Partial |
| ... | ... | ... | ... | ... | ... |

### Key Findings

1. **Critical Components**: [LIST]
2. **Redundant Components**: [LIST]
3. **Interaction Effects**: [SUMMARY]
4. **Recommendations**: [ACTIONS]
```

---

## Code Scaffolding

### Ablation Test Runner

```python
# tests/ablation/runner.py

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any

@dataclass
class AblationResult:
    variant_id: str
    component: str
    baseline_metrics: Dict[str, float]
    ablated_metrics: Dict[str, float]
    delta_metrics: Dict[str, float]
    p_values: Dict[str, float]
    hypothesis_supported: bool
    notes: str

class AblationRunner:
    """Runs ablation experiments systematically."""

    def __init__(self, baseline_config: dict, output_dir: Path):
        self.baseline_config = baseline_config
        self.output_dir = output_dir
        self.results: List[AblationResult] = []

    def run_baseline(self, training_data, epochs: int, seed: int):
        """Run baseline experiment."""
        system = T4DMSystem(config=self.baseline_config)
        metrics = self._train_and_evaluate(system, training_data, epochs, seed)
        return metrics

    def run_ablation(
        self,
        variant_id: str,
        component_path: str,
        training_data,
        epochs: int,
        seed: int,
        baseline_metrics: dict
    ) -> AblationResult:
        """Run single ablation experiment."""
        # Create ablated config
        config = self._ablate_component(self.baseline_config, component_path)

        # Train and evaluate
        system = T4DMSystem(config=config)
        ablated_metrics = self._train_and_evaluate(system, training_data, epochs, seed)

        # Compute deltas and statistics
        delta_metrics, p_values = self._compute_statistics(
            baseline_metrics, ablated_metrics
        )

        result = AblationResult(
            variant_id=variant_id,
            component=component_path,
            baseline_metrics=baseline_metrics,
            ablated_metrics=ablated_metrics,
            delta_metrics=delta_metrics,
            p_values=p_values,
            hypothesis_supported=False,  # To be filled manually
            notes=""
        )

        self.results.append(result)
        return result

    def save_results(self):
        """Save all results to JSON."""
        output_file = self.output_dir / "ablation_results.json"
        with open(output_file, "w") as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)

    def _ablate_component(self, config: dict, path: str) -> dict:
        """Disable component at given path."""
        config = config.copy()
        parts = path.split(".")
        target = config
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]]["enabled"] = False
        return config

    def _train_and_evaluate(self, system, data, epochs, seed) -> dict:
        """Training loop with metric collection."""
        # Implementation depends on T4DM training API
        pass

    def _compute_statistics(self, baseline, ablated) -> tuple:
        """Compute delta metrics and p-values."""
        from scipy import stats
        delta = {}
        p_values = {}
        for key in baseline:
            delta[key] = ablated[key] - baseline[key]
            # Would need multiple runs for proper t-test
            p_values[key] = 0.0  # Placeholder
        return delta, p_values
```

### Pytest Integration

```python
# tests/ablation/test_neuromodulators.py

import pytest
from t4dm.learning.neuromodulators import NeuromodulatorOrchestra

class TestNeuromodulatorAblations:
    """Ablation tests for neuromodulator components."""

    @pytest.fixture
    def baseline_orchestra(self):
        return NeuromodulatorOrchestra()

    @pytest.fixture
    def training_data(self):
        # Load or generate test data
        pass

    def test_dopamine_ablation(self, baseline_orchestra, training_data):
        """A1-DA: Ablate dopamine, expect reduced reward learning."""
        # Create ablated version
        ablated = AblatedNeuromodulatorOrchestra(ablate_dopamine=True)

        # Run experiments
        baseline_metrics = run_experiment(baseline_orchestra, training_data)
        ablated_metrics = run_experiment(ablated, training_data)

        # Assert expected behavior
        assert ablated_metrics["retrieval_accuracy"] < baseline_metrics["retrieval_accuracy"]
        # Hypothesis: 15-25% decrease
        delta = (baseline_metrics["retrieval_accuracy"] - ablated_metrics["retrieval_accuracy"])
        relative = delta / baseline_metrics["retrieval_accuracy"]
        assert 0.15 <= relative <= 0.25, f"Expected 15-25% decrease, got {relative*100:.1f}%"
```

---

## Best Practices

### Reproducibility

1. **Fix Random Seeds**: Use identical seeds for baseline and ablation runs
2. **Version Control**: Track config versions and code commits
3. **Environment Logging**: Record Python/PyTorch/CUDA versions
4. **Checkpoint Saving**: Save model states at regular intervals

### Statistical Rigor

1. **Multiple Runs**: Run each experiment 5+ times with different seeds
2. **Confidence Intervals**: Report 95% CIs, not just point estimates
3. **Effect Sizes**: Use Cohen's d to quantify practical significance
4. **Correction**: Apply Bonferroni correction for multiple comparisons

### Documentation

1. **Pre-registration**: Document hypotheses before running experiments
2. **Raw Data**: Save all raw metrics, not just summaries
3. **Negative Results**: Report null findings with same rigor as positive
4. **Visualizations**: Include learning curves, not just final metrics

---

## References

### Neuroscience Background

- Bi & Poo (1998): STDP in hippocampal neurons
- Schultz et al. (1997): Dopamine reward prediction
- Hasselmo (2006): ACh and memory encoding
- Aston-Jones & Cohen (2005): NE and adaptive gain
- Daw et al. (2002): Serotonin and temporal discounting

### Machine Learning Ablation Studies

- Meyes et al. (2019): "Ablation Studies in Artificial Neural Networks"
- Lipton & Steinhardt (2018): "Troubling Trends in Machine Learning Scholarship"

### T4DM Architecture

- NEUROTRANSMITTER_ARCHITECTURE.md: Detailed neuromodulator design
- BRAIN_REGION_MAPPING.md: Software-to-neuroscience mapping
- LEARNING_ARCHITECTURE.md: Learning rule specifications

---

## Appendix: Quick Reference

### Ablation ID Scheme

- `A1-XXX`: Neuromodulator ablations
- `A2-XXX`: Spiking component ablations
- `A3-XXX`: Learning rule ablations
- `A4-XXX`: Consolidation ablations
- `A5-XXX`: Memory architecture ablations
- `AM-XXX`: Multi-component ablations

### File Locations

| Component Category | Source Directory |
|-------------------|------------------|
| Neuromodulators | `src/t4dm/learning/` |
| Spiking | `src/t4dm/spiking/` |
| Consolidation | `src/t4dm/consolidation/` |
| Core Memory | `src/t4dm/core/` |
| Encoding | `src/t4dm/encoding/` |

### Metric Formulas

| Metric | Formula |
|--------|---------|
| Retrieval Accuracy | `TP / (TP + FP)` |
| Recall@K | `(relevant in top-K) / total_relevant` |
| MRR | `mean(1 / rank_of_first_relevant)` |
| Cohen's d | `(mean1 - mean2) / pooled_std` |
