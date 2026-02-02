# Learning
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/learning/`

## What
Biologically-inspired adaptive learning system implementing neuromodulator dynamics, three-factor learning rules, FSRS spaced repetition, causal discovery, and retrieval feedback loops. The largest module in WW (~35 files).

## How
### Neuromodulator Systems
- **Dopamine** (`dopamine.py`): Reward prediction error (RPE) -- `delta = actual - expected`. Drives surprise-based learning.
- **Norepinephrine** (`norepinephrine.py`): Arousal/attention modulation, novelty detection, exploration vs exploitation.
- **Acetylcholine** (`acetylcholine.py`): Encoding vs retrieval mode switching (cognitive mode gating).
- **Serotonin** (`serotonin.py`): Long-term credit assignment, patience, temporal discounting.
- **Inhibition** (`inhibition.py`): GABA-like lateral inhibition, winner-take-all, sparse retrieval.
- **NeuromodulatorOrchestra** (`neuromodulators.py`): Unified orchestration of all neuromodulator systems.

### Learning Rules
- **Three-Factor** (`three_factor.py`): `effective_lr = base_lr * eligibility * neuromod_gate * dopamine_surprise`. Biologically plausible learning.
- **STDP** (`stdp.py`): Spike-timing dependent plasticity with DA modulation.
- **Hebbian/Anti-Hebbian** (`plasticity.py`, `anti_hebbian.py`): Correlation-based and decorrelation-based learning.
- **BCM Metaplasticity** (`bcm_metaplasticity.py`): Sliding threshold for LTP/LTD transitions.
- **Homeostatic** (`homeostatic.py`): Keeps neural activity within bounds.
- **Eligibility Traces** (`eligibility.py`): Temporal credit assignment across layered traces.

### Retrieval & Scoring
- **LearnedRetrievalScorer** (`scorer.py`): Neural network that re-ranks retrieval results. Trained with ListMLE loss and prioritized replay.
- **FSRS** (`fsrs.py`): Free Spaced Repetition Scheduler for memory strength tracking.
- **RetrievalFeedbackCollector** (`retrieval_feedback.py`): Implicit feedback from user interactions.
- **FeedbackSignalProcessor** (`feedback_signals.py`): Converts feedback to three-factor compatible signals.

### Other
- **CreditFlowEngine** (`credit_flow.py`): Bridges neuromodulator signals to actual weight updates.
- **CausalDiscovery** (`causal_discovery.py`): Learns causal graphs from event sequences.
- **GenerativeReplay** (`generative_replay.py`): Wake-sleep replay for consolidation (Hinton-inspired).
- **VAEGenerator** (`vae_generator.py`, `vae_training.py`): Variational autoencoder for memory generation.
- **SelfSupervisedCredit** (`self_supervised.py`): Credit estimation without explicit outcomes.
- **UnifiedLearningSignal** (`unified_signals.py`): Combines all learning signals (three-factor + FF goodness + capsule agreement).

## Why
Enables WW to learn from experience without explicit labels. Memories that lead to successful outcomes get strengthened; failures cause reconsolidation. The neuromodulator systems provide biologically-grounded gating of when and how much to learn.

## Key Files
| File | Purpose |
|------|---------|
| `events.py` | `RetrievalEvent`, `OutcomeEvent`, `Experience` data structures |
| `collector.py` | `EventCollector` with SQLite persistence |
| `neuromodulators.py` | `NeuromodulatorOrchestra` unified system |
| `three_factor.py` | `ThreeFactorLearningRule` main learning rule |
| `scorer.py` | `LearnedRetrievalScorer` neural re-ranker |
| `fsrs.py` | `FSRS` spaced repetition scheduler |
| `unified_signals.py` | `UnifiedLearningSignal` deep integration |
| `hooks.py` | Learning hook mixins, `emit_retrieval_event()` |

## Data Flow
```
Retrieval -> RetrievalEvent -> EventCollector -> SQLite
Outcome   -> OutcomeEvent  -> Experience (retrieval+outcome pair)
Experience -> NeuromodulatorOrchestra -> ThreeFactorLearningRule
    -> CreditFlowEngine -> weight updates -> reconsolidation
    -> ScorerTrainer -> updated retrieval scoring
```

## Learning Modalities
| Modality | Mechanism | File |
|----------|-----------|------|
| Reward-based | Dopamine RPE | `dopamine.py` |
| Timing-based | STDP | `stdp.py` |
| Correlation | Hebbian | `plasticity.py` |
| Decorrelation | Anti-Hebbian | `anti_hebbian.py` |
| Spaced repetition | FSRS | `fsrs.py` |
| Self-supervised | Implicit credit | `self_supervised.py` |
| Causal | Granger + transfer entropy | `causal_discovery.py` |
| Generative | VAE + wake-sleep replay | `generative_replay.py`, `vae_generator.py` |
