# Encoding Module
**Path**: `/mnt/projects/t4d/t4dm/src/ww/encoding/`

## What
Biologically-inspired neural encoding implementing three hippocampal computations: sparse coding (dentate gyrus pattern separation), dendritic computation (L5 pyramidal prediction error), and associative memory (CA3 Hopfield pattern completion). Also includes a Forward-Forward encoder for local learning without backpropagation.

## How
- **SparseEncoder**: 8x expansion (1024->8192) with k-Winner-Take-All (2% sparsity = ~164 active units), lateral inhibition, straight-through gradient estimator. Decorrelates similar inputs for pattern separation.
- **DendriticNeuron**: Two-compartment model (basal=feedforward, apical=top-down context). Sigmoid gate controls coupling. Mismatch signal = `||h_basal - h_apical||` serves as prediction error. DendriticProcessor stacks multiple layers.
- **AttractorNetwork**: Classical Hopfield with Hebbian storage (`E = -0.5 * s^T W s`), settling-based retrieval, 0.14N capacity limit. Basin of attraction estimation and pattern removal (unlearning).
- **FFEncoder** (Phase 5): Forward-Forward algorithm for local learning -- goodness-based layer training without backpropagation, used by FF bridges for novelty detection.
- **Neurogenesis**: Dynamic addition of encoding units
- **Online Adapter**: Adapter training for encoding pipeline
- **Utilities**: Sparsity computation, cosine similarity matrices, straight-through estimator, exponential decay

## Why
Standard embeddings (BGE-M3) lack the pattern separation, completion, and prediction error signals needed for biologically plausible memory. These encoding layers transform dense embeddings into representations supporting hippocampal-style memory operations.

## Key Files
| File | Purpose |
|------|---------|
| `sparse.py` | k-WTA sparse encoder, DG-inspired (~275 lines) |
| `dendritic.py` | Two-compartment pyramidal neurons (~240 lines) |
| `attractor.py` | Hopfield associative memory network (~485 lines) |
| `ff_encoder.py` | Forward-Forward encoder (Phase 5) |
| `neurogenesis.py` | Dynamic unit addition |
| `online_adapter.py` | Online adapter training |
| `utils.py` | Sparsity, similarity, gradient utilities (~175 lines) |

## Data Flow
```
BGE-M3 embedding [1024]
    -> SparseEncoder [8192, 2% active] (pattern separation)
    -> DendriticNeuron (prediction error from context mismatch)
    -> AttractorNetwork (pattern completion from partial cues)
    -> Memory storage / retrieval
```

## Integration Points
- **embedding**: Receives dense vectors from BGE-M3 adapter
- **bridges**: FFEncoder used by ff_encoding_bridge and ff_retrieval_scorer
- **memory**: Sparse codes stored/retrieved via vector stores
- **consolidation**: Attractor network supports replay-based reconsolidation

## Learning Modalities
- **Hebbian**: AttractorNetwork stores patterns via outer product rule
- **Forward-Forward**: FFEncoder trains layers locally using goodness signal
- **k-WTA**: No learning, fixed competitive inhibition
- **Dendritic**: Mismatch signal provides unsupervised prediction error
