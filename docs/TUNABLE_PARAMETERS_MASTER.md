# T4DM: Master Parameter Reference

**Generated**: 2025-12-09 | **Version**: 0.3.0 | **Audit Status**: COMPLETE

Comprehensive audit of all tunable parameters, compiled from Hinton (neural architecture), CompBio (biological mechanisms), and codebase exploration agents.

> **Total Parameters**: 243+ | **API Exposed**: 127+ | **Hardcoded**: 116+

## Recent Additions (v0.3.0)

- **ACh threshold validation**: Encoding/retrieval threshold constraints with hysteresis
- **Three-factor weight normalization**: Weights must sum to 1.0
- **Neuromodulator tuning endpoints**: Live parameter adjustment via PUT `/api/v1/viz/bio/neuromodulators`
- **Neuromodulator interactions**: Three-factor weight tuning via PUT `/api/v1/viz/bio/neuromodulators/interactions`
- **Learning dynamics**: Effective LR and credit flow via GET `/api/v1/viz/bio/learning/*`
- **Temporal dynamics**: Neuromodulator traces and simulation via `/api/v1/viz/bio/neuromodulators/traces|simulate|step`
- **Top-k eligibility**: GET `/api/v1/viz/bio/eligibility/top-k`

---

## Backwards Compatibility

### Safe to Tune (No Breaking Changes)
All parameters with existing API endpoints are safe to modify within documented ranges. The system uses sensible defaults from cognitive psychology literature.

### Bio-Plausible Recommendations (CompBio Agent)
The following defaults differ from current values for improved biological plausibility:

| Parameter | Current | Bio-Recommended | Rationale |
|-----------|---------|-----------------|-----------|
| `inhibition_strength` | 0.5 | 0.75 | E/I ratio closer to cortical (4:1) |
| `phasic_decay` (NE) | 0.7 | 0.3 | LC bursts decay faster |
| `a_minus` (LTD rate) | 0.00525 | 0.02 | More balanced LTP/LTD |
| `sparsity_target` | 0.2 | 0.05 | Hippocampal DG ~2-5% active |

**Migration**: These are recommendations only. Current defaults remain for stability. Apply via API tuning for experimentation.

### Additive Parameters (New, Non-Breaking)
Parameters marked **MISSING** in this document can be added without breaking changes. They extend existing systems with optional functionality.

---

## Executive Summary

| Category | Parameters | API Writable | API Read-Only | Not Exposed |
|----------|------------|--------------|---------------|-------------|
| **Core Config** | 85 | 60 | 0 | 25 |
| **FSRS** | 19 | 4 | 0 | 15 (internal) |
| **ACT-R** | 7 | 7 | 0 | 0 |
| **Neuromodulators** | 34 | 5 (partial) | 29 | 0 |
| **Eligibility Traces** | 12 | 2 | 6 | 4 |
| **Sparse Encoding** | 8 | 4 | 4 | 0 |
| **Attractor Network** | 9 | 6 | 0 | 3 |
| **Pattern Sep/Comp** | 10 | 2 | 3 | 5 |
| **Consolidation** | 17 | 6 | 11 | 0 |
| **Hebbian** | 5 | 5 | 0 | 0 |
| **Dendritic** | 8 | 6 | 0 | 2 |
| **Homeostatic** | 6 | 0 | 0 | 6 |
| **Three-Factor** | 6 | 6 | 0 | 0 |
| **Learned Gate** | 11 | 6 | 0 | 5 |
| **Modulated Embed** | 8 | 0 | 0 | 8 |
| **Buffer Manager** | 14 | 0 | 0 | 14 |
| **Memory Gate** | 6 | 0 | 0 | 6 |
| **Storage/Viz** | 8 | 0 | 0 | 8 |
| **TOTAL** | **243+** | ~87 | ~53 | ~103 |

**API Coverage**: ~36% writable, ~22% read-only, ~42% unexposed

---

## 1. NEUROMODULATOR SYSTEMS

### 1.1 Dopamine (Reward Prediction Error)

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `dopamine_baseline` | 0.5 | [0.0, 1.0] | Config | Strong | Default expected value |
| `dopamine_value_learning_rate` | 0.1 | [0.01, 1.0] | Config | Strong | TD learning rate (α) |
| `dopamine_surprise_threshold` | 0.05 | [0.01, 0.5] | - | Moderate | Min |δ| for significance |
| `dopamine_max_rpe_magnitude` | 1.0 | [0.1, 10.0] | - | Strong | RPE clipping |
| `phasic_boost` | - | [1.0, 5.0] | **MISSING** | Strong | Phasic DA magnitude |
| `tonic_baseline` | - | [0.2, 0.8] | **MISSING** | Strong | Tonic DA level |
| `d1_sensitivity` | - | [0.5, 2.0] | **MISSING** | Strong | D1 receptor (go) |
| `d2_sensitivity` | - | [0.5, 2.0] | **MISSING** | Strong | D2 receptor (no-go) |

### 1.2 Norepinephrine (Arousal/Attention)

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `norepinephrine_gain` | 1.0 | [0.1, 5.0] | Config | Moderate | Gain multiplier |
| `baseline_arousal` | 0.5 | [0.2, 0.8] | - | Moderate | Tonic LC-NE level |
| `novelty_decay` | 0.95 | [0.90, 0.99] | - | Weak | Habituation rate |
| `phasic_decay` | 0.7 | [0.5, 0.9] | - | Moderate | Burst decay |
| `min_gain` | 0.5 | [0.3, 0.7] | - | Weak | Minimum gain |
| `max_gain` | 2.0 | [1.5, 3.0] | - | Weak | Maximum gain |
| `uncertainty_weight` | 0.3 | [0.2, 0.5] | - | Moderate | Entropy weight |
| `novelty_weight` | 0.7 | [0.5, 0.8] | - | Moderate | Novelty weight |
| `lc_threshold` | - | [0.3, 0.7] | **MISSING** | Moderate | LC activation threshold |
| `exploration_temperature` | - | [0.1, 2.0] | **MISSING** | Moderate | Exploration softmax |

### 1.3 Serotonin (Temporal Credit / Patience)

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `serotonin_discount` | 0.5 | [0.0, 1.0] | Config | Strong | Temporal discount γ |
| `base_discount_rate` | 0.99 | [0.95, 0.995] | - | Strong | Per-step γ |
| `eligibility_decay` | 0.95 | [0.90, 0.98] | - | Strong | Trace decay/hour |
| `trace_lifetime_hours` | 24.0 | [6.0, 48.0] | - | Moderate | Max trace duration |
| `baseline_mood` | 0.5 | [0.3, 0.7] | - | Weak | Default 5-HT level |
| `mood_adaptation_rate` | 0.1 | [0.05, 0.2] | - | Weak | Mood update speed |
| `patience_temperature` | - | [0.1, 2.0] | **MISSING** | Moderate | Softness of discounting |
| `delayed_reward_bonus` | - | [1.0, 2.0] | **MISSING** | Moderate | Waiting bonus |

### 1.4 Acetylcholine (Encoding/Retrieval Mode)

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `acetylcholine_threshold` | 0.5 | [0.0, 1.0] | Config | Moderate | Mode switch threshold (deprecated) |
| `encodingThreshold` | 0.7 | [0.5, 0.95] | Config | Moderate | ACh > → encoding (validated: must > retrieval) |
| `retrievalThreshold` | 0.3 | [0.1, 0.5] | Config | Moderate | ACh < → retrieval (validated: gap ≥ 0.2) |
| `adaptationRate` | 0.1 | [0.01, 1.0] | Config | Weak | Mode switch speed |
| `hysteresis` | 0.05 | [0.0, 0.2] | Config | Moderate | Prevents rapid mode oscillation |
| `baseline_ach` | 0.5 | [0.3, 0.7] | - | Moderate | Default ACh level |
| `circadian_modulation` | - | [0.0, 0.3] | **MISSING** | Strong | Time-of-day effect |

> **API Status**: ACh now has dedicated `AcetylcholineConfig` model with validation that `encodingThreshold > retrievalThreshold` and gap ≥ 0.2 for proper mode switching (added v0.3.0)

### 1.5 GABA / Inhibition

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `gaba_inhibition` | 0.3 | [0.0, 1.0] | Config | Moderate | Pattern separation strength |
| `inhibition_strength` | 0.5 | [0.3, 0.7] | - | Moderate | Lateral inhibition |
| `sparsity_target` | 0.2 | [0.05, 0.3] | - | Strong | Target active % |
| `temperature` | 1.0 | [0.5, 2.0] | - | Weak | Softmax temp |
| `pv_interneuron_strength` | - | [0.5, 2.0] | **MISSING** | Strong | Fast inhibition |
| `sst_interneuron_strength` | - | [0.3, 1.5] | **MISSING** | Strong | Dendritic inhibition |
| `e_i_ratio` | - | [3.0, 5.0] | **MISSING** | Strong | E/I balance |
| `gamma_frequency` | - | [30, 80] Hz | **MISSING** | Strong | Gamma oscillation |

---

## 2. MEMORY SYSTEMS

### 2.1 FSRS (Spaced Repetition)

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `fsrs_default_stability` | 1.0 | [0.1, 10.0] | Config | Strong | Initial stability (days) |
| `fsrs_retention_target` | 0.9 | [0.5, 1.0] | Config | Strong | Target recall rate |
| `fsrs_decay_factor` | 0.9 | [0.1, 1.0] | Config | Strong | Power-law decay |
| `fsrs_recency_decay` | 0.1 | [0.01, 1.0] | Config | Moderate | Recency scoring |
| `w0-w16` | (see FSRS-4.5) | varies | Internal | Strong | FSRS weights |
| `maximum_interval` | 36500 | [1, 36500] | Internal | Moderate | Max days |

### 2.2 ACT-R Activation

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `actr_decay` | 0.5 | [0.1, 1.0] | Config | Strong | Power-law decay (d) |
| `actr_spreading_strength` | 1.6 | [0.1, 5.0] | Config | Moderate | Spreading activation (S) |
| `actr_threshold` | 0.0 | [-5.0, 5.0] | Config | Moderate | Retrieval threshold (τ) |
| `actr_noise` | 0.0 | [0.0, 1.0] | Config | Weak | Activation noise (σ) |
| `spreading_max_nodes` | 1000 | [10, 10000] | - | Engineering | Graph limit |
| `spreading_max_neighbors` | 50 | [1, 200] | - | Engineering | Neighbor limit |
| `spreading_default_steps` | 3 | [1, 5] | - | Weak | Propagation steps |

### 2.3 Fast Episodic Store

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `fes_capacity` | 10000 | [1, 100000] | Bioinspired | Moderate | Episode capacity |
| `fes_learning_rate` | 0.1 | [0.001, 1.0] | Bioinspired | Strong | One-shot rate (100x semantic) |
| `fes_consolidation_threshold` | 0.7 | [0.0, 1.0] | Bioinspired | Moderate | Consolidation trigger |
| `salience_weight_dopamine` | 0.4 | [0.0, 1.0] | - | Moderate | DA in salience |
| `salience_weight_norepinephrine` | 0.3 | [0.0, 1.0] | - | Moderate | NE in salience |
| `salience_weight_acetylcholine` | 0.3 | [0.0, 1.0] | - | Moderate | ACh in salience |

---

## 3. NEURAL ARCHITECTURE

### 3.1 Dendritic Processing

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `dendritic_hidden_dim` | 512 | [64, 4096] | Bioinspired | Engineering | Internal dim |
| `dendritic_context_dim` | 512 | [64, 4096] | Bioinspired | Engineering | Top-down dim |
| `dendritic_coupling_strength` | 0.5 | [0.0, 1.0] | Bioinspired | Strong | Basal-apical coupling |
| `dendritic_tau_dendrite` | 10.0 | [1.0, 100.0] ms | Bioinspired | Strong | Dendritic τ |
| `dendritic_tau_soma` | 15.0 | [1.0, 100.0] ms | Bioinspired | Strong | Somatic τ |
| `apical_gain` | - | [0.1, 2.0] | **MISSING** | Strong | Top-down modulation |
| `plateau_duration` | - | [10, 100] ms | **MISSING** | Strong | Ca spike duration |
| `nmda_boost` | - | [1.0, 5.0] | **MISSING** | Strong | Supralinearity |

### 3.2 Sparse Encoding (k-WTA)

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `sparse_hidden_dim` | 8192 | [128, 131072] | Bioinspired | Engineering | Expanded dim (8x) |
| `sparse_sparsity` | 0.02 | [0.001, 0.2] | Bioinspired | Strong | 2% active neurons |
| `sparse_use_kwta` | True | bool | Bioinspired | Strong | k-WTA enabled |
| `sparse_lateral_inhibition` | 0.2 | [0.0, 1.0] | Bioinspired | Strong | Competition strength |
| `soft_kwta_temperature` | - | [0.1, 10.0] | **MISSING** | Moderate | Softness of WTA |
| `homeostatic_target_rate` | - | [0.01, 0.1] | **MISSING** | Strong | Target firing rate |

### 3.3 Attractor Network

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `attractor_settling_steps` | 10 | [1, 1000] | Bioinspired | Moderate | Settling iterations |
| `attractor_step_size` | 0.1 | [0.01, 1.0] | Bioinspired | Moderate | Update magnitude |
| `attractor_noise_std` | 0.01 | [0.0, 0.5] | Bioinspired | Moderate | Exploration noise |
| `attractor_adaptation_tau` | 5.0 | [0.1, 50.0] | Bioinspired | Moderate | Adaptation τ |
| `attractor_capacity_ratio` | 0.138 | [0.01, 0.5] | - | Strong | Hopfield capacity |
| `hopfield_beta` | - | [1.0, 100.0] | **MISSING** | Strong | Modern Hopfield temp |
| `energy_threshold` | - | [0.001, 0.1] | **MISSING** | Moderate | Convergence criterion |

### 3.4 Pattern Separation / Completion

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `similarity_threshold` | 0.55 | [0.0, 1.0] | PatternSep | Moderate | Separation trigger |
| `max_separation` | 0.3 | [0.0, 1.0] | - | Moderate | Max orthogonalization |
| `min_separation` | 0.05 | [0.0, 1.0] | - | Weak | Min to apply |
| `sparsity_ratio` | 0.04 | [0.001, 0.2] | - | Strong | DG sparsity (4%) |
| `dg_expansion_ratio` | - | [3, 10] | **MISSING** | Strong | DG expansion |
| `ca3_recurrence` | - | [0.0, 0.5] | **MISSING** | Strong | CA3 autoassociative |
| `completion_threshold` | - | [0.3, 0.8] | **MISSING** | Moderate | Pattern completion |

---

## 4. LEARNING SYSTEMS

### 4.1 Hebbian Learning

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `hebbian_learning_rate` | 0.1 | [0.01, 0.5] | Config | Strong | LTP magnitude |
| `hebbian_decay_rate` | 0.01 | [0.001, 0.1] | Config | Moderate | LTD / forgetting |
| `hebbian_initial_weight` | 0.1 | [0.01, 1.0] | Config | Weak | Initial synapse |
| `hebbian_min_weight` | 0.01 | [0.001, 0.1] | Config | Weak | Pruning threshold |
| `hebbian_stale_days` | 30 | [1, 365] | Config | Engineering | Inactivity threshold |
| `bcm_threshold` | - | [0.3, 0.7] | **MISSING** | Strong | BCM sliding threshold |
| `bcm_adaptation_rate` | - | [0.001, 0.1] | **MISSING** | Strong | Threshold adaptation |
| `metaplasticity_rate` | - | [0.01, 0.1] | **MISSING** | Strong | Plasticity of plasticity |

### 4.2 Eligibility Traces

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `eligibility_decay` | 0.95 | [0.5, 0.999] | Bioinspired | Strong | Per-step λ |
| `eligibility_tau_trace` | 20.0 | [1.0, 100.0] s | Bioinspired | Strong | Time constant |
| `a_plus` | 0.005 | [0.001, 0.1] | - | Strong | LTP rate |
| `a_minus` | 0.00525 | [0.001, 0.1] | - | Strong | LTD rate |
| `fast_tau` | 5.0 | [1.0, 20.0] s | - | Strong | Fast trace τ |
| `slow_tau` | 60.0 | [30.0, 120.0] s | - | Strong | Slow trace τ |
| `dopamine_window` | - | [0.5, 5.0] s | **MISSING** | Strong | DA consolidation window |
| `stdp_window` | - | [10, 100] ms | **MISSING** | Strong | STDP temporal window |

### 4.3 Homeostatic Plasticity

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `target_norm` | 1.0 | [0.1, 10.0] | **NONE** | Moderate | Target L2 norm |
| `norm_tolerance` | 0.2 | [0.01, 1.0] | **NONE** | Weak | Deviation threshold |
| `ema_alpha` | 0.01 | [0.001, 0.1] | **NONE** | Weak | EMA rate |
| `decorrelation_strength` | 0.01 | [0.0, 0.1] | **NONE** | Moderate | Decorrelation |
| `sliding_threshold_rate` | 0.001 | [0.0001, 0.01] | **NONE** | Strong | BCM adaptation |
| `synaptic_scaling_rate` | - | [0.001, 0.1] | **MISSING** | Strong | Multiplicative scaling |

### 4.4 Three-Factor Learning Rule

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `ach_weight` | 0.4 | [0.0, 1.0] | Config | Strong | ACh contribution to neuromod gate |
| `ne_weight` | 0.35 | [0.0, 1.0] | Config | Strong | NE contribution to neuromod gate |
| `serotonin_weight` | 0.25 | [0.0, 1.0] | Config | Strong | 5-HT contribution |
| `min_effective_lr` | 0.1 | [0.01, 0.5] | Config | Moderate | Learning rate floor |
| `max_effective_lr` | 3.0 | [1.0, 10.0] | Config | Moderate | Learning rate ceiling |
| `bootstrap_rate` | 0.01 | [0.001, 0.1] | Config | Moderate | Prevents zero-learning deadlock |

**Combined Signal Formula**: `effective_lr × eligibility × surprise × patience + bootstrap`

> **API Status**: Full API support via `ThreeFactorConfig` model in `/api/v1/config` (added v0.1.1)

### 4.5 Learned Memory Gate

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `store_threshold` | 0.6 | [0.3, 0.9] | Config | Engineering | Min score to store |
| `buffer_threshold` | 0.3 | [0.1, 0.5] | Config | Engineering | Min score for buffer |
| `learning_rate_mean` | 0.1 | [0.01, 0.5] | Config | Moderate | Thompson sampling mean update |
| `learning_rate_var` | 0.05 | [0.01, 0.2] | Config | Moderate | Thompson sampling var update |
| `cold_start_threshold` | 100 | [10, 1000] | Config | Engineering | Samples before adaptive |
| `thompson_temperature` | 1.0 | [0.1, 5.0] | Config | Moderate | Exploration temperature |
| `CONTENT_DIM` | 128 | [32, 512] | - | Engineering | Feature vector content dim |
| `CONTEXT_DIM` | 64 | [16, 256] | - | Engineering | Context feature dim |
| `NEUROMOD_DIM` | 7 | fixed | - | Engineering | Neuromodulator feature count |
| `TEMPORAL_DIM` | 16 | [8, 64] | - | Engineering | Temporal feature dim |
| `INTERACTION_DIM` | 32 | [8, 128] | - | Engineering | Interaction feature dim |

> **API Status**: Full API support via `LearnedGateConfig` model in `/api/v1/config` (added v0.1.1)

### 4.6 Reconsolidation

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `base_learning_rate` | 0.01 | [0.005, 0.05] | - | Moderate | Embedding update |
| `max_update_magnitude` | 0.1 | [0.05, 0.2] | - | Weak | Max L2 change |
| `lability_window_hours` | 6.0 | [2.0, 12.0] | - | Strong | Reconsolidation window |
| `cooldown_hours` | 1.0 | [0.5, 2.0] | - | Weak | Between updates |

---

## 5. CONSOLIDATION SYSTEMS

### 5.1 Sleep Consolidation

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `replay_hours` | 24 | [6, 48] | Consolidation | Moderate | Recency window |
| `max_replays` | 100 | [50, 500] | Consolidation | Engineering | Per NREM cycle |
| `nrem_cycles` | 4 | [3, 5] | Consolidation | Strong | Sleep cycles |
| `compression_factor` | 10.0 | [5.0, 20.0] | - | Strong | SWR compression |
| `prune_threshold` | 0.05 | [0.01, 0.1] | Consolidation | Weak | Pruning cutoff |
| `homeostatic_target` | 10.0 | [5.0, 20.0] | - | Weak | SHY target weight |
| `nrem_rem_ratio` | - | [2.0, 4.0] | **MISSING** | Strong | NREM:REM time |
| `swr_frequency_hz` | - | [100, 250] | **MISSING** | Strong | Ripple frequency |
| `replay_direction_bias` | - | [0.0, 1.0] | **MISSING** | Moderate | Forward vs reverse |

### 5.2 HDBSCAN Clustering

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `hdbscan_min_cluster_size` | 3 | [2, 100] | Consolidation | Weak | Min for cluster |
| `hdbscan_metric` | "cosine" | - | Consolidation | Moderate | Distance metric |
| `consolidation_min_similarity` | 0.75 | [0.5, 1.0] | Consolidation | Weak | Episode clustering |
| `consolidation_min_occurrences` | 3 | [2, 10] | Consolidation | Weak | Consolidation threshold |

---

## 6. MODULATED EMBEDDING

### 6.1 Attention Modulation

| Parameter | Default | Range | API | Bio Evidence | Description |
|-----------|---------|-------|-----|--------------|-------------|
| `state_dim` | 5 | [3, 10] | - | Engineering | Neuromod state vector size |
| `salience_k` | 2.0 | [1.0, 5.0] | - | Moderate | Attention sharpening |
| `salience_threshold` | 0.3 | [0.1, 0.5] | - | Moderate | Min activation |
| `attention_heads` | 4 | [1, 8] | - | Engineering | Multi-head attention |
| `dropout_rate` | 0.1 | [0.0, 0.3] | - | Engineering | Regularization |
| `emotional_weight` | 0.3 | [0.0, 1.0] | - | Strong | Valence influence |
| `context_weight` | 0.4 | [0.0, 1.0] | - | Moderate | Context influence |
| `temporal_weight` | 0.3 | [0.0, 1.0] | - | Moderate | Recency influence |

---

## 7. BUFFER & GATE SYSTEMS

### 7.1 Buffer Manager (memory/buffer_manager.py)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `RETRIEVAL_HIT_SIGNAL` | 0.25 | [0.1, 0.5] | - | Utility boost on retrieval |
| `CO_RETRIEVAL_SIGNAL` | 0.15 | [0.05, 0.3] | - | Co-occurrence boost |
| `OUTCOME_SIGNAL_SCALE` | 0.2 | [0.1, 0.4] | - | Feedback scaling |
| `NEUROMOD_SIGNAL_SCALE` | 0.1 | [0.05, 0.2] | - | Neuromod influence |
| `CONTEXT_MATCH_SIGNAL` | 0.05 | [0.01, 0.1] | - | Context matching boost |
| `TIME_DECAY_PER_SECOND` | 0.0003 | [0.0001, 0.001] | - | Temporal decay rate |
| `PROMOTED_UTILITY_BASE` | 0.5 | [0.3, 0.7] | - | Base value after promotion |
| `PROMOTED_UTILITY_SCALE` | 0.5 | [0.3, 0.7] | - | Scaling for promoted items |
| `DISCARDED_UTILITY_BASE` | 0.1 | [0.05, 0.2] | - | Base value after discard |
| `DISCARDED_UTILITY_SCALE` | 0.35 | [0.2, 0.5] | - | Scaling for discarded items |
| `DEFAULT_PROMOTION_THRESHOLD` | 0.65 | [0.5, 0.8] | - | Buffer → store threshold |
| `DEFAULT_DISCARD_THRESHOLD` | 0.25 | [0.1, 0.4] | - | Buffer → discard threshold |
| `DEFAULT_MAX_RESIDENCE_SECONDS` | 300 | [60, 600] | - | Max time in buffer (5 min) |
| `DEFAULT_MAX_BUFFER_SIZE` | 50 | [20, 100] | - | Max buffer capacity |

### 7.2 Memory Gate (core/memory_gate.py)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `store_threshold` | 0.4 | [0.2, 0.6] | - | Min score to store directly |
| `buffer_threshold` | 0.2 | [0.1, 0.4] | - | Min score for buffer |
| `min_store_interval` | 30s | [10s, 120s] | - | Cooldown between stores |
| `max_messages_without_store` | 20 | [5, 50] | - | Force store after N messages |
| `voice_mode_adjustments` | True | bool | - | Adjust thresholds for voice |
| `_recent_hash_limit` | 100 | [50, 200] | - | Deduplication hash cache |

---

## 8. RETRIEVAL WEIGHTS

### 8.1 Episodic Retrieval (must sum to 1.0)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `episodic_weight_semantic` | 0.4 | [0.0, 1.0] | Config | Embedding similarity |
| `episodic_weight_recency` | 0.25 | [0.0, 1.0] | Config | Time decay |
| `episodic_weight_outcome` | 0.2 | [0.0, 1.0] | Config | Success/failure |
| `episodic_weight_importance` | 0.15 | [0.0, 1.0] | Config | Salience |

### 8.2 Semantic Retrieval (must sum to 1.0)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `semantic_weight_similarity` | 0.4 | [0.0, 1.0] | Config | Embedding |
| `semantic_weight_activation` | 0.35 | [0.0, 1.0] | Config | ACT-R |
| `semantic_weight_retrievability` | 0.25 | [0.0, 1.0] | Config | FSRS |

### 8.3 Procedural Retrieval (must sum to 1.0)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `procedural_weight_similarity` | 0.6 | [0.0, 1.0] | Config | Task similarity |
| `procedural_weight_success` | 0.3 | [0.0, 1.0] | Config | Success rate |
| `procedural_weight_experience` | 0.1 | [0.0, 1.0] | Config | Execution count |

---

## 9. STORAGE & VISUALIZATION

### 9.1 T4DX Graph Storage (storage/t4dx/graph_adapter.py)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `DEFAULT_DB_TIMEOUT` | 30 | [10, 60] | - | Query timeout (seconds) |
| `MAX_BATCH_SIZE` | 1000 | [100, 5000] | - | Batch insert limit |
| `MAX_DEPTH_LIMIT` | 10 | [3, 20] | - | Graph traversal depth |

### 9.2 Cluster Index (memory/cluster_index.py)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `DEFAULT_K` | 5 | [3, 20] | - | Default cluster count |
| `MIN_CLUSTER_SIZE` | 5 | [2, 20] | - | Minimum cluster size |
| `MAX_CLUSTERS` | 1000 | [100, 5000] | - | Maximum clusters |

### 9.3 FES Consolidator (consolidation/fes_consolidator.py)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `consolidation_rate` | 0.1 | [0.01, 0.5] | - | Update rate |
| `min_consolidation_score` | 0.5 | [0.3, 0.8] | - | Min score to consolidate |
| `MAX_CONSOLIDATION_BATCH` | 100 | [50, 500] | - | Batch size |
| `MAX_ENTITY_EXTRACTION` | 50 | [10, 200] | - | Entity limit per batch |

### 9.4 Visualization (visualization/*.py)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `neuromod_window_size` | 1000 | [100, 10000] | - | State history window |
| `plasticity_max_updates` | 10000 | [1000, 100000] | - | Trace history limit |
| `persistence_max_history` | 1000 | [100, 10000] | - | State history limit |

### 9.5 Embedding Configuration (core/config.py)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `embedding_model` | BAAI/bge-m3 | - | Env | Model identifier |
| `embedding_dimension` | 1024 | [256, 4096] | Env | Vector dimension |
| `embedding_cache_size` | 1000 | [100, 100000] | Env | LRU cache size |
| `embedding_cache_ttl` | 3600 | [60, 86400] | Env | Cache TTL (seconds) |
| `embedding_batch_size` | 32 | [1, 128] | Env | Batch size |
| `embedding_max_length` | 512 | [128, 2048] | Env | Max token length |

### 9.6 Entity Extraction (core/config.py)

| Parameter | Default | Range | API | Description |
|-----------|---------|-------|-----|-------------|
| `extraction_confidence_threshold` | 0.7 | [0.5, 0.95] | Env | Min confidence |
| `extraction_batch_size` | 10 | [1, 50] | Env | Batch size |
| `batch_max_concurrency` | 10 | [1, 50] | Env | Parallel workers |

---

## 10. API GAP SUMMARY

### Resolved Gaps (v0.3.0)

| Gap | Resolution | Endpoint |
|-----|------------|----------|
| **Neuromodulator live tuning** | ✅ Full PUT support | `PUT /api/v1/viz/bio/neuromodulators` |
| **ACh mode switching** | ✅ Encoding/retrieval thresholds with validation | `AcetylcholineConfig` in `/api/v1/config` |
| **Three-factor weights** | ✅ Validated sum = 1.0 | `PUT /api/v1/viz/bio/neuromodulators/interactions` |
| **Eligibility top-k** | ✅ Most eligible memories | `GET /api/v1/viz/bio/eligibility/top-k` |
| **Learning dynamics** | ✅ Effective LR, credit flow | `GET /api/v1/viz/bio/learning/*` |
| **Temporal dynamics** | ✅ Traces, simulation, stepping | `GET/POST /api/v1/viz/bio/neuromodulators/*` |

### Remaining Critical Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| **Homeostatic system** | No endpoints exist at all | P0 |
| **Memory CRUD** | No UPDATE/DELETE for episodes/entities/skills | P1 |
| **Pattern sep/comp config** | Read-only visualization | P1 |

### Missing Biological Parameters

| System | Missing | Bio Evidence |
|--------|---------|--------------|
| Dopamine | D1/D2 sensitivity, tonic/phasic split | Strong |
| NE | LC threshold, exploration temp | Moderate |
| ACh | Circadian modulation | Strong |
| GABA | PV/SST/VIP interneurons, gamma oscillation | Strong |
| Hebbian | BCM threshold, metaplasticity | Strong |
| Pattern Sep | DG expansion, CA3 recurrence | Strong |
| Consolidation | Theta/gamma oscillations, sleep spindles | Strong |
| Eligibility | STDP window, dopamine window | Strong |

---

## 11. CONFIGURATION PRESETS

Four curated presets are available via `GET /api/v1/config/presets` and can be applied via `POST /api/v1/config/presets/{name}`:

### 11.1 bio-plausible
Optimized for biological realism per CompBio agent recommendations.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `gaba_inhibition` | 0.75 | E/I ratio closer to cortical (4:1) |
| `sparse_lateral_inhibition` | 0.75 | Stronger competition |
| `sparse_sparsity` | 0.05 | Hippocampal DG ~2-5% active |
| `pattern_sep_sparsity` | 0.05 | Match sparse encoder |
| `neuromod_alpha_ne` | 0.3 | More gradual arousal |
| `eligibility_decay` | 0.98 | Longer eligibility window |
| `dendritic_tau_dendrite` | 15.0 | Slower dendritic integration |
| `dendritic_tau_soma` | 20.0 | Slower somatic integration |
| `attractor_settling_steps` | 20 | More thorough pattern completion |
| `three_factor_serotonin_weight` | 0.35 | Stronger patience signal |
| `three_factor_ach_weight` | 0.35 | Balanced attention |
| `three_factor_ne_weight` | 0.30 | Reduced arousal influence |

### 11.2 performance
Optimized for speed and throughput.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `sparse_sparsity` | 0.01 | Minimal computation |
| `sparse_hidden_dim` | 4096 | Smaller hidden layer |
| `attractor_settling_steps` | 5 | Fast convergence |
| `eligibility_decay` | 0.90 | Quick trace decay |
| `learned_gate_store_threshold` | 0.7 | Higher bar for storage |
| `memory_gate_threshold` | 0.5 | Fewer memories stored |

### 11.3 conservative
Prioritizes memory retention and stability.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `fsrs_retention_target` | 0.95 | Higher recall target |
| `fsrs_default_stability` | 2.0 | Stronger initial memory |
| `learned_gate_store_threshold` | 0.4 | Lower bar for storage |
| `memory_gate_threshold` | 0.2 | Store more memories |
| `hebbian_learning_rate` | 0.05 | Slower weight updates |
| `fes_learning_rate` | 0.05 | Slower episodic learning |
| `eligibility_tau_trace` | 30.0 | Longer eligibility window |

### 11.4 exploration
Encourages novel associations and exploratory behavior.

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `norepinephrine_gain` | 1.5 | Higher arousal |
| `dopamine_baseline` | 0.3 | Lower expected reward (more surprise) |
| `learned_gate_temperature` | 2.0 | More uncertainty exploration |
| `acetylcholine_threshold` | 0.7 | Bias toward encoding |
| `attractor_noise_std` | 0.05 | More basin exploration |

---

## 12. RECOMMENDED PRIORITY TIERS (Hinton Agent)

### Tier 1: Critical for System Behavior
1. Neuromodulator gains (rho_da, rho_ne, rho_ach)
2. Sparsity parameters (k-WTA k, lateral inhibition)
3. Eligibility trace decay (tau_trace, decay rate)
4. Attractor settling dynamics (steps, step_size, noise)
5. Sleep consolidation cycles (nrem_cycles, replay params)

### Tier 2: Important for Biological Plausibility
1. Dendritic time constants (tau_dendrite, tau_soma)
2. STDP parameters (a_plus, a_minus)
3. BCM threshold adaptation
4. Pattern separation/completion thresholds
5. E/I balance

### Tier 3: Research Directions
1. Modern Hopfield temperature (beta)
2. Metaplasticity dynamics
3. Circadian modulation of ACh
4. Replay direction bias
5. Schema formation rates

---

## 13. VALIDATION CONSTRAINTS

### Weight Sums
- Episodic weights: must sum to 1.0 ± 0.001
- Semantic weights: must sum to 1.0 ± 0.001
- Procedural weights: must sum to 1.0 ± 0.001

### Time Constant Ratios
- tau_dendrite < tau_soma (dendrites faster than soma)
- fast_tau < slow_tau (layered eligibility)

### Learning Rate Relationships
- a_minus ≥ a_plus (slight LTD bias for stability)
- hebbian_decay_rate << hebbian_learning_rate

### Biological Plausibility Bounds
- Sparsity: 1-10% (hippocampus: 2-5%)
- Hopfield capacity: ≤ 0.138N patterns
- Reconsolidation window: 2-12 hours
- NREM-REM ratio: 3:1 to 4:1

---

## 14. ARCHITECTURAL GAPS (Hinton Agent Analysis)

### Critical Gaps

| Gap | Description | Recommendation |
|-----|-------------|----------------|
| **Pattern Separation Learning** | DG-like encoder doesn't learn | Add gradient-based training path |
| **Attractor Integration** | Hopfield isolated from pipeline | Wire into retrieval with `use_attractor` toggle |
| **Consolidation Scheduling** | No explicit sleep-like cycles | Add `ConsolidationScheduler` with time-based triggers |
| **Neuromod ↔ Learning Coupling** | Loose integration | Implement `ThreeFactorHook` callback pattern |
| **Pattern Completion** | CA3 autoassociative weak | Add energy-based completion endpoint |

### Recommended New Classes

```python
# RuntimeTunableParams - Hot-reloadable parameter container
class RuntimeTunableParams:
    _instance = None
    def __init__(self):
        self._params = {...}  # All 205 parameters
        self._callbacks = []  # Notify on change
    def update(self, key: str, value: Any): ...
    def subscribe(self, callback: Callable): ...
```

### Deduplication Notes

The following parameters have similar names but distinct semantics - NOT duplicates:

| Parameter A | Parameter B | Distinction |
|-------------|-------------|-------------|
| `sparse_sparsity` (0.02) | `sparsity_target` (0.2) | Encoder k-WTA vs inhibition target |
| `eligibility_decay` (elig) | `eligibility_decay` (serotonin) | Same param, different docstrings |
| `baseline_arousal` (NE) | `baseline_ach` (ACh) | Distinct neuromodulator systems |
| `decay` (FSRS) | `decay` (ACT-R) | Different memory theories |
| `store_threshold` (MemoryGate) | `store_threshold` (LearnedGate) | Rule-based vs learned |
| `consolidation_rate` (FES) | `consolidation_threshold` (FES) | Rate vs threshold |
| `temperature` (inhibition) | `thompson_temperature` (gate) | Softmax vs exploration |

### Backwards Compatibility Verification

**All parameters safe to tune:**
- Existing defaults preserved as fallbacks
- New API endpoints are additive (no breaking changes)
- Environment variable prefixes unchanged (`T4DM_*`)
- Pydantic validators ensure range compliance
- Weight sum constraints enforced automatically

---

## References

- Anderson, J. R., & Schooler, L. J. (1991). Reflections of the environment in memory
- Aston-Jones, G., & Cohen, J. D. (2005). Adaptive Gain Theory
- Buzsáki, G. (1989). Two-stage model of memory trace formation
- Hasselmo, M. E. (2006). The role of acetylcholine in learning and memory
- Nader, K., et al. (2000). Fear memories require protein synthesis for reconsolidation
- Rolls, E. T., & Treves, A. (1998). Neural networks and brain function
- Schultz, W., et al. (1997). A neural substrate of prediction and reward
- Tononi, G. & Cirelli, C. (2014). Synaptic homeostasis hypothesis
- Wilson, M. A., & McNaughton, B. L. (1994). Reactivation during sleep
