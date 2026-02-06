# Variational Sleep Phase Documentation (W4-02)

## Overview

This document provides mathematical notation for sleep-based memory consolidation as variational inference, following Friston's Free Energy Principle.

## Variational Free Energy

Sleep consolidation minimizes the variational free energy:

```
F = E_q[log q(z) - log p(x,z)]
  = KL(q(z) || p(z|x)) - log p(x)
  = -ELBO
```

Where:
- **x** = observed episodic memories (data)
- **z** = latent cluster assignments (hidden variables)
- **q(z)** = approximate posterior (variational distribution)
- **p(x,z)** = generative model (joint distribution)
- **ELBO** = Evidence Lower Bound

## Sleep Phases as EM Steps

### NREM Phase (E-step)

**Goal**: Update q(z|x) to minimize KL divergence

```
q*(z|x) = argmin_q KL(q(z) || p(z|x))
```

**Implementation**: Soft cluster assignment via softmax

```
q(z_i = k | x_i) = exp(s_ik / τ) / Σ_j exp(s_ij / τ)
```

Where:
- s_ik = similarity between memory i and cluster k
- τ = temperature (lower = sharper assignments)

**Biological Correspondence**:
- Sharp-wave ripples (SWR) replay compressed memory sequences
- Hippocampal-neocortical dialogue assigns memories to cortical traces
- ~75% of sleep time

### REM Phase (M-step)

**Goal**: Update p(x|z) parameters to maximize ELBO

```
θ* = argmax_θ E_q[log p(x|z; θ)]
```

**For Gaussian clusters**:

```
μ_k = Σ_i q(z_i=k) x_i / Σ_i q(z_i=k)
σ²_k = Σ_i q(z_i=k) (x_i - μ_k)² / Σ_i q(z_i=k)
```

Where:
- μ_k = cluster k centroid (prototype)
- σ²_k = cluster k variance

**Biological Correspondence**:
- Dreams combine disparate memories
- Pattern finding creates abstract concepts
- Theta rhythm coordinates memory integration
- ~25% of sleep time

### PRUNE Phase (Regularization)

**Goal**: Enforce sparsity prior on cluster assignments

```
z_i = ∅ if max_k q(z_i=k) < threshold
```

**Regularized objective**:

```
F_reg = F + λ Σ_i H(q(z_i))
```

Where:
- H(q(z_i)) = entropy of assignment (uncertainty)
- λ = regularization strength

**Biological Correspondence**:
- Synaptic downscaling during slow-wave sleep
- Weak connections pruned (LTD)
- Homeostatic normalization
- Occurs during deep NREM

## ELBO Decomposition

The ELBO decomposes into:

```
ELBO = E_q[log p(x|z)] - KL(q(z) || p(z))
     = [Log-likelihood] - [KL divergence]
```

### Log-likelihood Term

For Gaussian clusters:

```
log p(x_i | z_i=k) = -1/2 [(x_i - μ_k)ᵀ Σ_k⁻¹ (x_i - μ_k) + log|Σ_k|]
```

### KL Divergence Term

For categorical assignments:

```
KL(q(z) || p(z)) = Σ_i Σ_k q(z_i=k) log[q(z_i=k) / p(z_i=k)]
```

## Convergence Criteria

EM converges when:

```
|ELBO_t - ELBO_{t-1}| < ε
```

Or equivalently:

```
|F_t - F_{t-1}| < ε
```

Where ε is a small threshold (e.g., 0.001).

## Full Sleep Cycle

A typical cycle alternates:

1. **NREM₁** (E-step): Initial cluster assignment
2. **REM₁** (M-step): Update prototypes
3. **NREM₂** (E-step): Refine assignments
4. **REM₂** (M-step): Further prototype refinement
5. **NREM₃** (E-step): Final assignment
6. **PRUNE** (Reg): Remove uncertain memories

This typically runs 4-5 NREM-REM cycles per night with final pruning.

## Implementation Mapping

| Sleep Phase | EM Step | T4DM Method |
|-------------|---------|-------------|
| NREM | E-step | `VariationalConsolidation.e_step()` |
| REM | M-step | `VariationalConsolidation.m_step()` |
| PRUNE | Regularization | `VariationalConsolidation.regularization_step()` |
| Wake | Encoding | `memory.store()` |

## References

1. Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

2. Diekelmann, S., & Born, J. (2010). The memory function of sleep. *Nature Reviews Neuroscience*, 11(2), 114-126.

3. McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex. *Psychological Review*, 102(3), 419.

4. Neal, R. M., & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. In *Learning in Graphical Models* (pp. 355-368).
