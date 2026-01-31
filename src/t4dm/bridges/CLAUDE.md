# Bridges Module
**Path**: `/mnt/projects/t4d/t4dm/src/ww/bridges/`

## What
Integration bridges connecting neural subsystems (NCA, Forward-Forward, Capsule Networks, Predictive Coding) to memory operations. Six bridge implementations providing state-dependent encoding, retrieval scoring, novelty detection, and learning signal routing.

## How
Each bridge connects a neural subsystem to the memory pipeline:
- **NCABridge** (P10.1): Augments embeddings with 6-dim NT state, modulates retrieval by cognitive state similarity, routes RPE learning signals
- **DopamineBridge** (P5.2): Converts hierarchical prediction errors (Rao & Ballard) into dopamine-like RPE signals for learning modulation
- **CapsuleBridge** (P6.2): Uses capsule network part-whole representations for compositional retrieval scoring beyond cosine similarity
- **FFEncodingBridge** (P6.3): Forward-Forward goodness as novelty detector -- low goodness = novel = enhanced encoding priority
- **FFRetrievalScorer** (P6.4): Forward-Forward confidence scoring for retrieval result quality
- **FFCapsuleBridge** (Phase 6A): Combined FF goodness + capsule routing agreement for joint scoring

## Why
Memory operations should be state-dependent. The same input encoded in FOCUS vs EXPLORE creates different memories. These bridges make encoding, retrieval, and learning context-aware by integrating biologically-inspired neural dynamics.

## Key Files
| File | Purpose |
|------|---------|
| `nca_bridge.py` | Memory-NCA bridge (state-dependent encoding/retrieval) |
| `dopamine_bridge.py` | PredictiveCoding -> DopamineSystem RPE conversion |
| `capsule_bridge.py` | CapsuleLayer -> retrieval scoring via routing agreement |
| `ff_encoding_bridge.py` | ForwardForward -> encoding novelty detection |
| `ff_retrieval_scorer.py` | ForwardForward -> retrieval confidence scoring |
| `ff_capsule_bridge.py` | Combined FF goodness + capsule routing |

## Data Flow
```
Embedding -> [NCABridge] -> NT-augmented embedding -> Storage
Query -> [CapsuleBridge] -> compositional similarity -> Retrieval reranking
Prediction Error -> [DopamineBridge] -> RPE -> [ThreeFactorLearning]
Embedding -> [FFEncodingBridge] -> goodness/novelty -> Encoding priority
```

## Integration Points
- **nca**: NeuralFieldSolver, LearnableCoupling, StateTransitionManager, EnergyLandscape
- **learning**: DopamineSystem, ThreeFactorLearningRule, EligibilityTrace
- **encoding**: FFEncoder, CapsuleLayer
- **bridge**: Backwards-compatible re-exports (deprecated)
