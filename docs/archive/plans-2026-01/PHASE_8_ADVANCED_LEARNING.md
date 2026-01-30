# Phase 8: Advanced Learning Mechanisms - Implementation Plan

**Generated**: 2026-01-04 | **Agent**: ww-hinton | **Status**: PLANNING COMPLETE

---

## Executive Summary

Phase 8 introduces four advanced learning mechanisms inspired by computational neuroscience:

1. **Predictive Coding Hierarchy** - Full hierarchical prediction error minimization with precision weighting
2. **Continuous Neural Dynamics** - Neural ODE formulation for capsule temporal evolution
3. **Anti-Hebbian Learning** - Foldiak-style lateral inhibition for sparse, decorrelated representations
4. **BCM Metaplasticity** - Sliding threshold with synaptic tagging and capture

---

## Component 1: Predictive Coding Hierarchy

### Mathematical Formulation

Following Rao & Ballard (1999) and Friston (2005):

**Level Dynamics:**
```
dmu_l/dt = D_mu * [-mu_l + f(mu_{l+1}) + Sigma_l^-1 * epsilon_l]
epsilon_l = x_{l-1} - g(mu_l)  # Prediction error at level l
```

**Precision Dynamics (adaptive gain):**
```
dPi_l/dt = D_pi * [-Pi_l + Pi_prior + expected_precision(epsilon_l)]
expected_precision = 1 / (epsilon_l^2 + epsilon_0)
```

**Free Energy:**
```
F = sum_l [0.5 * epsilon_l^T * Pi_l * epsilon_l + 0.5 * log|Sigma_l|]
```

### Files to Create

#### `/mnt/projects/ww/src/ww/prediction/active_inference.py`

```python
@dataclass
class ActiveInferenceConfig:
    n_levels: int = 4
    dims: list[int]  # [1024, 512, 256, 128]
    precision_lr: float = 0.01
    belief_lr: float = 0.005
    free_energy_threshold: float = 0.1
    temporal_horizon: int = 5

@dataclass
class BeliefState:
    mean: np.ndarray      # mu - current belief
    precision: np.ndarray # Pi - confidence
    prediction: np.ndarray
    error: np.ndarray
    free_energy: float

class PrecisionWeightedLevel:
    def compute_prediction_error(self, target: np.ndarray) -> np.ndarray
    def update_precision(self, error: np.ndarray) -> float
    def compute_free_energy(self) -> float
    def belief_update(self, error_below: np.ndarray, pred_above: np.ndarray) -> np.ndarray

class ActiveInferenceHierarchy:
    def infer(self, sensory: np.ndarray, n_iterations: int = 10) -> list[BeliefState]
    def learn(self) -> dict[str, float]
    def compute_expected_free_energy(self, action: np.ndarray) -> float
    def generate_action(self, goal_state: np.ndarray) -> np.ndarray
```

### Integration Points

1. **CapsuleNetwork**: Route predictions through capsule poses
2. **NeuralFieldSolver**: Inject prediction errors into ACh field
3. **DopamineSystem**: Free energy as alternative RPE signal

### Literature Citations

- Rao & Ballard (1999): Predictive coding in visual cortex
- Friston (2005): Theory of cortical responses
- Bogacz (2017): Tutorial on free-energy framework

---

## Component 2: Continuous Neural Dynamics (Neural ODE)

### Mathematical Formulation

**Neural ODE for Capsule States:**
```
dz/dt = f_theta(z, t)
```

Where `z = [activations, poses]` is the capsule state vector.

**Capsule-specific dynamics:**
```
da_i/dt = -lambda * a_i + sigma(W_a * x + sum_j c_ij * v_j)
dP_i/dt = -gamma * (P_i - I) + W_p * x + sum_j c_ij * T_ij * P_j
dc_ij/dt = eta * (agreement(P_i, T_ij * P_j) - c_ij)
```

### Files to Create

#### `/mnt/projects/ww/src/ww/nca/neural_ode_capsules.py`

```python
@dataclass
class NeuralODECapsuleConfig:
    input_dim: int = 1024
    num_capsules: int = 32
    capsule_dim: int = 16
    pose_dim: int = 4
    time_span: tuple[float, float] = (0.0, 1.0)
    solver: str = "dopri5"
    rtol: float = 1e-3
    atol: float = 1e-4
    adjoint: bool = True
    activation_decay: float = 0.1
    pose_regularization: float = 0.01
    routing_rate: float = 0.5

class CapsuleODEFunc:
    def forward(self, t: Tensor, state: Tensor) -> Tensor
    def unpack_state(self, state: Tensor) -> tuple
    def pack_state(self, activations, poses, routing) -> Tensor

class NeuralODECapsuleLayer:
    def forward(self, x: Tensor, t_span: tuple = None) -> tuple[Tensor, Tensor]
    def get_trajectory(self, x: Tensor, n_points: int = 10) -> list
    def compute_energy(self, state: Tensor) -> float
```

### Literature Citations

- Chen et al. (2018): Neural ordinary differential equations
- Massaroli et al. (2020): Dissecting neural ODEs

---

## Component 3: Anti-Hebbian Learning (Foldiak)

### Mathematical Formulation

**Foldiak Anti-Hebbian Rule (1990):**
```
dW_lat_ij/dt = eta * (y_i * y_j - delta_ij * target_covariance)
```

**Combined Hebbian/Anti-Hebbian:**
```
dW_ff/dt = eta_ff * y * x^T           # Feedforward: Hebbian
dW_lat/dt = -eta_lat * (y * y^T - I)  # Lateral: Anti-Hebbian
```

### Files to Create

#### `/mnt/projects/ww/src/ww/learning/anti_hebbian.py`

```python
@dataclass
class AntiHebbianConfig:
    input_dim: int = 1024
    output_dim: int = 512
    learning_rate_ff: float = 0.01
    learning_rate_lat: float = 0.001
    sparsity_target: float = 0.05
    decorrelation_strength: float = 1.0
    n_iterations: int = 5
    weight_decay: float = 1e-4

class FoldiakLayer:
    def forward(self, x: np.ndarray) -> np.ndarray
    def _settle(self, x: np.ndarray) -> np.ndarray
    def learn(self, x: np.ndarray, y: np.ndarray) -> dict
    def compute_correlation_matrix(self) -> np.ndarray
    def get_sparsity(self) -> float

class AntiHebbianNetwork:
    def forward(self, x: np.ndarray) -> list[np.ndarray]
    def learn(self, x: np.ndarray) -> dict
    def get_decorrelation_stats(self) -> dict
```

### Literature Citations

- Foldiak (1990): Forming sparse representations
- Olshausen & Field (1996): Sparse code for natural images
- Bell & Sejnowski (1995): Information-maximization for ICA

---

## Component 4: BCM Metaplasticity with Synaptic Tagging

### Mathematical Formulation

**BCM Sliding Threshold:**
```
dphi/dt = eta * y * (y - theta_m) * x
d(theta_m)/dt = tau^-1 * (y^2 - theta_m)
```

**Synaptic Tagging and Capture:**
```
Tag_i(t) = exp(-(t - t_induction) / tau_tag) * tag_strength_i
PRPs(t) = base_rate + sum(strong_inputs) * synthesis_factor
Consolidation_i = Tag_i(t) * min(PRPs(t), capture_capacity)
```

### Files to Create

#### `/mnt/projects/ww/src/ww/learning/bcm_metaplasticity.py`

```python
@dataclass
class BCMConfig:
    theta_m_init: float = 0.5
    theta_m_tau: float = 100.0
    theta_m_min: float = 0.1
    theta_m_max: float = 0.9
    ltp_rate: float = 0.01
    ltd_rate: float = 0.005
    tag_decay_tau: float = 7200.0
    early_tag_threshold: float = 0.3
    late_tag_threshold: float = 0.7
    prp_base_rate: float = 0.1
    prp_synthesis_threshold: float = 0.8
    prp_decay_rate: float = 0.001
    capture_capacity: float = 1.0

class BCMLearningRule:
    def compute_update(self, pre: float, post: float, synapse_id: str) -> tuple[float, str]
    def update_threshold(self, synapse_id: str, post_activity: float) -> float
    def get_threshold(self, synapse_id: str) -> float

class SynapticTaggingAndCapture:
    def process_input(self, synapse_id: str, strength: float, timestamp: datetime) -> dict
    def attempt_capture(self, timestamp: datetime) -> list[str]
    def decay_tags(self, timestamp: datetime) -> list[str]

class BCMMetaplasticityManager:
    def on_retrieval(self, synapse_ids: list, activities: list, timestamp: datetime) -> dict
    def on_consolidation(self, timestamp: datetime) -> dict
    def get_plasticity_state(self, synapse_id: str) -> BCMState
```

### Literature Citations

- Bienenstock, Cooper & Munro (1982): BCM theory
- Frey & Morris (1997): Synaptic tagging and LTP
- Redondo & Morris (2011): Synaptic tagging and capture hypothesis
- Clopath et al. (2008): Tag-trigger-consolidation model

---

## Implementation Priority

| Sprint | Component | Duration | Dependencies |
|--------|-----------|----------|--------------|
| 8.1 | BCM Metaplasticity | 2 weeks | None |
| 8.2 | Anti-Hebbian Learning | 2 weeks | 8.1 |
| 8.3 | Neural ODE Capsules | 2 weeks | None (parallel) |
| 8.4 | Active Inference | 2 weeks | 8.1, 8.2 |

---

## File Summary

| File | Classes | LOC |
|------|---------|-----|
| `prediction/active_inference.py` | 4 | ~600 |
| `nca/neural_ode_capsules.py` | 3 | ~450 |
| `learning/anti_hebbian.py` | 3 | ~400 |
| `learning/bcm_metaplasticity.py` | 4 | ~550 |
| Tests (4 files) | - | ~950 |

**Total**: ~2,950 LOC
