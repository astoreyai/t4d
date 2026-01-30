# Phase 9: Attention & Binding - Implementation Plan

**Generated**: 2026-01-04 | **Agent**: ww-hinton | **Status**: PLANNING COMPLETE

---

## Executive Summary

Phase 9 implements a unified attention and binding system:

1. **Unified Attention System** - Merge capsule routing with transformer attention
2. **Temporal Attention** - Sequence modeling with theta-gamma positional encoding
3. **Cross-Modal Binding** - Binding across episodic/semantic/procedural memories
4. **Working Memory Gating** - Theta-gamma based attention gating

---

## Component 1: Unified Attention System

### Theoretical Foundation

Capsule routing-by-agreement and transformer attention are complementary:
- **Capsule routing**: Part-whole relationships, configuration agreement
- **Transformer attention**: Associative content-addressing, parallel queries

### Mathematical Formulation

**Fusion Attention:**
```
Attention(Q, K, V) = softmax(A_capsule + A_transformer) * V

A_capsule[i,j] = exp(-||pose_i @ W_ij - pose_j||_F / tau)
A_transformer[i,j] = (q_i @ k_j) / sqrt(d_k)
A_combined = alpha * A_capsule + (1 - alpha) * A_transformer
```

### File: `/mnt/projects/ww/src/ww/nca/unified_attention.py`

```python
@dataclass
class UnifiedAttentionConfig:
    embed_dim: int = 1024
    num_heads: int = 8
    head_dim: int = 64
    pose_dim: int = 4
    capsule_weight: float = 0.5
    temperature: float = 1.0
    use_ff_learning: bool = True
    attention_dropout: float = 0.1

class UnifiedAttentionHead:
    def compute_capsule_attention(self, query_poses, key_poses) -> np.ndarray
    def compute_transformer_attention(self, query, key) -> np.ndarray
    def forward(self, query, key, value, poses=None, mask=None) -> tuple
    def learn_fusion_weight(self, utility, contributions) -> None

class UnifiedAttentionSystem:
    def attend(self, query, keys, values, poses=None, mask=None) -> np.ndarray
    def extract_poses(self, embeddings) -> np.ndarray
    def get_attention_stats(self) -> dict
```

---

## Component 2: Temporal Attention

### Mathematical Formulation

**Theta-Phase Positional Modulation:**
```
PE_theta(pos, theta_cycle, gamma_slot) =
    PE_base(pos) + sin(gamma_slot / gamma_slots_per_theta) * W_theta
```

**Relative Positional Attention:**
```
A[i,j] = (q_i @ k_j + q_i @ r_{i-j}) / sqrt(d_k)
```

### File: `/mnt/projects/ww/src/ww/nca/temporal_attention.py`

```python
@dataclass
class TemporalAttentionConfig:
    embed_dim: int = 1024
    max_sequence_length: int = 512
    positional_type: str = "learnable"
    use_theta_modulation: bool = True
    gamma_slots_per_theta: int = 7
    theta_position_weight: float = 0.3
    use_relative_positions: bool = True
    max_relative_distance: int = 128
    temporal_decay_rate: float = 0.1

class PositionalEncoding:
    def encode_positions(self, length, theta_phase=None, gamma_phases=None) -> np.ndarray
    def get_theta_modulated_encoding(self, position, theta_cycle, gamma_slot) -> np.ndarray

class RelativePositionEmbedding:
    def get_relative_embeddings(self, length) -> np.ndarray

class TemporalAttention:
    def attend_sequence(self, query_seq, memory_seq, positions=None, causal=False) -> np.ndarray
    def encode_temporal_context(self, items, timestamps, theta_cycles) -> np.ndarray
    def compute_temporal_similarity(self, pos1, pos2, time1, time2) -> float
```

---

## Component 3: Cross-Modal Binding

### Mathematical Formulation

**Cross-Modal Attention:**
```
Q = f_q(episodic)
A_semantic = softmax(Q @ K_semantic.T / sqrt(d))
A_procedural = softmax(Q @ K_procedural.T / sqrt(d))
```

**Binding via Synchrony:**
```
Bound = gamma_sync * (A_semantic @ V_semantic + A_procedural @ V_procedural)
gamma_sync = PLV(gamma_phase_episodic, gamma_phase_semantic)
```

### File: `/mnt/projects/ww/src/ww/nca/cross_modal_binding.py`

```python
@dataclass
class CrossModalBindingConfig:
    embed_dim: int = 1024
    binding_dim: int = 256
    binding_temperature: float = 0.5
    synchrony_threshold: float = 0.3
    contrastive_temperature: float = 0.07
    orthogonality_weight: float = 0.1

class ModalityProjector:
    def project(self, embedding) -> np.ndarray
    def update(self, embedding, target, lr) -> None

class GammaSynchronyDetector:
    def compute_synchrony(self, phases_1, phases_2) -> float
    def is_synchronized(self, synchrony) -> bool

class CrossModalBinding:
    def bind(self, episodic, semantic, procedural, gamma_phases=None) -> dict
    def query_across_modalities(self, query, query_mod, target_mod, candidates) -> list
    def learn_binding(self, anchor, anchor_mod, positive, pos_mod, negatives) -> float

class TripartiteMemoryAttention:
    async def holistic_recall(self, query, embedding, top_k=5) -> dict
    def compute_cross_modal_coherence(self, episodic, semantic, procedural) -> float
```

---

## Component 4: Working Memory Gating

### Mathematical Formulation

**Encoding Gate:**
```
G_encode = sigmoid(encoding_signal - threshold) * (1 - alpha_inhibition)
```

**Retrieval Gate:**
```
G_retrieve = sigmoid(retrieval_signal - threshold) * attention_weight
```

**Maintenance via Rehearsal:**
```
activation(t+1) = activation(t) * decay + G_rehearse * attention
G_rehearse = (1 - alpha_inhibition) * gamma_amplitude
```

### File: `/mnt/projects/ww/src/ww/nca/wm_gating.py`

```python
@dataclass
class WMGatingConfig:
    wm_capacity: int = 7
    encoding_threshold: float = 0.5
    retrieval_threshold: float = 0.5
    eviction_threshold: float = 0.2
    decay_rate: float = 0.1
    rehearsal_boost: float = 0.3
    attention_learning_rate: float = 0.1
    alpha_inhibition_weight: float = 0.5
    gamma_maintenance_weight: float = 0.3

class EncodingGate:
    def compute_gate(self, encoding_signal, alpha, capacity, max_cap) -> float
    def should_encode(self, item_priority, gate_value) -> bool

class RetrievalGate:
    def compute_gate(self, retrieval_signal, attention, activation) -> float
    def retrieve_strength(self, gate_value, activation) -> float

class MaintenanceController:
    def update_activations(self, activations, attention, gamma, alpha, dt) -> np.ndarray
    def select_for_rehearsal(self, items, attention, gamma_phase) -> int

class WorkingMemoryGating:
    def step(self, new_items=None, query=None, dt_ms=1.0) -> dict
    def compute_attention_modulated_priority(self, item_emb, context_emb) -> float
    def synchronize_with_theta_gamma(self) -> None
```

---

## Implementation Order

| Sprint | Component | Duration | Dependencies |
|--------|-----------|----------|--------------|
| 9.1 | Unified Attention | 1 week | capsules.py, pose.py |
| 9.2 | Temporal Attention | 1 week | unified_attention.py, theta_gamma.py |
| 9.3 | Cross-Modal Binding | 1 week | unified_attention.py |
| 9.4 | WM Gating | 1 week | All above + working_memory.py |
| 9.5 | Integration | 1 week | All components |

---

## Integration Points

### Files to Modify

1. **`/mnt/projects/ww/src/ww/nca/__init__.py`**: Export new classes
2. **`/mnt/projects/ww/src/ww/memory/episodic.py`**: Add cross-modal binding support
3. **`/mnt/projects/ww/src/ww/memory/working_memory.py`**: Add gating integration

---

## File Summary

| File | Classes | LOC |
|------|---------|-----|
| `nca/unified_attention.py` | 3 | ~400 |
| `nca/temporal_attention.py` | 4 | ~450 |
| `nca/cross_modal_binding.py` | 5 | ~500 |
| `nca/wm_gating.py` | 5 | ~450 |
| Tests (4 files) | - | ~800 |

**Total**: ~2,600 LOC
