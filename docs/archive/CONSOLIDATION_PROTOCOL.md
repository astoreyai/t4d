# Consolidation Protocol

Sleep-based memory consolidation with VAE generative replay and multi-night scheduling.

**Phase 7 Implementation**: Complete sleep consolidation pipeline

## Overview

World Weaver implements a biologically-inspired sleep consolidation system based on Complementary Learning Systems (CLS) theory. The system integrates:

1. **Sharp-Wave Ripple (SWR) Replay** - Compressed hippocampal replay
2. **Generative Replay (VAE)** - Synthetic memory generation to prevent catastrophic forgetting
3. **Multi-Night Scheduling** - Progressive consolidation depth across nights
4. **Lability Window** - 6-hour reconsolidation window for memory modification

## Architecture

```
Awake Phase
    │
    ▼ (encoding via FFEncoder)
Episodic Storage
    │
    ▼ (trigger consolidation)
┌────────────────────────────────────────────────────┐
│               Sleep Consolidation                   │
│                                                     │
│  ┌─────────────┐    ┌─────────────┐               │
│  │ Real Memory │───▶│ VAE Encoder │               │
│  │  Embeddings │    └──────┬──────┘               │
│  └─────────────┘           │                       │
│         │                  ▼                       │
│         │         ┌──────────────┐                │
│         │         │ Latent Space │                │
│         │         └──────┬───────┘                │
│         │                │                         │
│         │                ▼                         │
│         │         ┌──────────────┐                │
│         │         │ VAE Decoder  │                │
│         │         └──────┬───────┘                │
│         │                │                         │
│         ▼                ▼                         │
│  ┌─────────────────────────────────────┐          │
│  │     Interleaved Training (CLS)      │          │
│  │  Real + Synthetic = No Forgetting   │          │
│  └──────────────────────────────────────┘         │
│                     │                              │
│                     ▼                              │
│           ┌─────────────────┐                     │
│           │ FF Layer Update │◀── Three-Factor LR  │
│           └─────────────────┘                     │
└────────────────────────────────────────────────────┘
```

## VAE Generative Replay

The system uses a Variational Autoencoder (VAE) to generate synthetic memories during consolidation. This prevents catastrophic forgetting by interleaving old patterns with new learning.

### Configuration

```python
from ww.consolidation import SleepConsolidation

consolidation = SleepConsolidation(
    # VAE parameters
    vae_enabled=True,           # Enable generative replay
    vae_latent_dim=128,         # Latent space dimensionality
    embedding_dim=1024,         # Input embedding dimension

    # Replay parameters
    replay_ratio=0.5,           # 50% synthetic, 50% real
    batch_size=32,              # Memories per replay batch
)
```

### VAE Statistics

```python
# Get VAE training statistics
stats = consolidation.get_vae_statistics()

print(f"Reconstruction loss: {stats['reconstruction_loss']:.4f}")
print(f"KL divergence: {stats['kl_divergence']:.4f}")
print(f"Training steps: {stats['training_steps']}")
```

### Wake-Sleep Algorithm

During consolidation, the system runs a modified wake-sleep algorithm:

1. **Wake Phase**: Train on real memories (positive phase)
2. **Sleep Phase**: Train on VAE-generated memories (prevents forgetting)
3. **Interleave**: Alternate real/synthetic to maintain old representations

```python
from ww.learning import GenerativeReplaySystem

# The system automatically interleaves:
# - 50% real memories (recent episodes)
# - 50% synthetic memories (VAE-generated)
# This prevents catastrophic forgetting of old patterns
```

## Multi-Night Scheduling

Consolidation depth increases progressively across multiple "nights" (consolidation cycles).

### Night Schedule

| Night | Depth | Focus |
|-------|-------|-------|
| 1 | `light` | Recent memories, weak consolidation |
| 2-3 | `deep` | Medium-priority memories, stronger replay |
| 4+ | `all` | Full consolidation, semantic extraction |

### Usage

```python
from ww.consolidation import ConsolidationService

service = ConsolidationService()

# Check current night
stats = service.get_stats()
print(f"Current night: {stats['multi_night']['current_night']}")

# Get recommended depth for this night
depth = service.get_recommended_consolidation_depth()
print(f"Recommended depth: {depth}")  # 'light', 'deep', or 'all'

# Advance to next night after consolidation
service.advance_night(memories_consolidated=150)

# Reset cycle (e.g., after major milestone)
service.reset_night_cycle()
```

### Automatic Depth Selection

```python
# Run consolidation with auto depth
async def nightly_consolidation(service: ConsolidationService):
    depth = service.get_recommended_consolidation_depth()

    if depth == "light":
        # Quick consolidation: recent memories only
        result = await service.consolidate(
            max_episodes=100,
            priority_only=True,
        )
    elif depth == "deep":
        # Full episode processing
        result = await service.consolidate(
            max_episodes=500,
            extract_entities=True,
        )
    else:  # "all"
        # Complete consolidation with semantic extraction
        result = await service.consolidate(
            max_episodes=1000,
            extract_entities=True,
            abstract_concepts=True,
            prune_connections=True,
        )

    # Advance night counter
    service.advance_night(result.consolidated_count)
```

## Lability Window (Reconsolidation)

When memories are retrieved, they enter a labile state for ~6 hours. During this window, the memory can be updated with new information (memory reconsolidation).

### Biological Basis

From Nader et al. (2000): Retrieved memories require protein synthesis for re-stabilization. The 6-hour window reflects this biological constraint.

Key findings:
- Sevenster et al. (2012): Prediction error gates reconsolidation
- McGaugh (2004): Emotional memories have extended windows
- Dudai (2012): The "restless engram" - consolidation never truly ends

### Configuration

```python
from ww.consolidation import (
    LabilityManager,
    LabilityConfig,
    is_reconsolidation_eligible,
    compute_reconsolidation_strength,
)

# Create manager with custom config
config = LabilityConfig(
    window_hours=6.0,                 # Default: 6 hours
    min_retrieval_strength=0.3,       # Weak retrieval doesn't destabilize
    emotional_modulation=True,        # High emotion extends window
    require_prediction_error=True,    # Surprise gates lability
    prediction_error_threshold=0.1,
)

manager = LabilityManager(config)

# On memory retrieval
state = manager.on_retrieval(
    memory_id=episode.id,
    strength=recall_score,
    emotional_valence=episode.emotional_valence,
    prediction_error=prediction_error,  # From dopamine bridge
)

# Check if memory can be updated
if manager.is_labile(episode.id):
    # Memory can be updated
    strength = compute_reconsolidation_strength(
        retrieval_strength=recall_score,
        emotional_valence=episode.emotional_valence,
        prediction_error=prediction_error,
        hours_elapsed=1.0,
    )
    # Update embedding with modulated strength
    new_embedding = ff_encoder.encode(new_content)
    await update_memory(episode.id, new_embedding, strength)
    manager.on_reconsolidation(episode.id, success=True)
else:
    # Memory is stable, update creates new trace instead
    await episodic.create(new_content)

# Quick eligibility check (stateless)
if is_reconsolidation_eligible(episode.last_accessed):
    # Safe to update
    pass
```

### Lability Flow

```
Retrieval
    │
    ▼
┌─────────────────┐
│ Trigger Lability│
│ (mark as labile)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  6-hour Window  │◀── protein_synthesis_equivalent
│ (memory is soft)│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐  ┌───────┐
│Update │  │ No    │
│Memory │  │Update │
└───┬───┘  └───┬───┘
    │          │
    ▼          ▼
┌───────┐  ┌───────┐
│ Re-   │  │Stable │
│stable │  │ Trace │
└───────┘  └───────┘
```

## Complete Consolidation Example

```python
import asyncio
from ww.consolidation import SleepConsolidation, ConsolidationService
from ww.memory import EpisodicMemory

async def nightly_routine():
    # Initialize
    episodic = EpisodicMemory(...)
    service = ConsolidationService()

    consolidation = SleepConsolidation(
        vae_enabled=True,
        vae_latent_dim=128,
        embedding_dim=1024,
    )

    # Get recommended depth
    depth = service.get_recommended_consolidation_depth()
    print(f"Night {service.state.current_night}: {depth} consolidation")

    # Run consolidation
    result = await consolidation.run_sleep_cycle(
        episodic_memory=episodic,
        depth=depth,
    )

    # Check VAE stats
    if consolidation.get_vae_statistics():
        stats = consolidation.get_vae_statistics()
        print(f"VAE reconstruction loss: {stats['reconstruction_loss']:.4f}")

    # Advance night
    service.advance_night(result.consolidated_count)

    return result

# Run nightly
result = asyncio.run(nightly_routine())
```

## Integration with Three-Factor Learning

During consolidation, the three-factor learning signal modulates weight updates:

```python
# Three-factor integration in consolidation:
# 1. Eligibility trace: Which synapses were recently active
# 2. Neuromodulator gate: DA/NE signal from outcome
# 3. Learning rate: Base LR × eligibility × gate

# In consolidation:
effective_lr = base_lr * eligibility * dopamine_gate * serotonin_patience

# VAE training uses this modulated LR
vae.train_step(embeddings, learning_rate=effective_lr)
```

## SWR-Theta Phase Locking

Sharp-Wave Ripple replay is gated by theta oscillations:

```python
from ww.nca import SWRCoupling

swr = SWRCoupling(
    theta_phase_window=(0.2, 0.4),  # Optimal replay window
    ripple_frequency=150,           # Hz
)

# Replay only occurs in proper theta phase
if swr.is_replay_phase():
    swr.trigger_replay_event(memories)
```

## Testing

```bash
# Run all Phase 7 tests (99 tests)
pytest tests/consolidation/test_lability.py \
       tests/learning/test_vae_generator.py \
       tests/nca/test_swr_coupling.py -v

# Lability window tests (35 tests)
pytest tests/consolidation/test_lability.py -v

# VAE generator tests (36 tests)
pytest tests/learning/test_vae_generator.py -v

# SWR coupling tests (28 tests)
pytest tests/nca/test_swr_coupling.py -v

# Sleep consolidation tests
pytest tests/unit/test_sleep_consolidation.py -v

# Service tests
pytest tests/unit/test_consolidation_service.py -v

# Full integration
pytest tests/unit/test_parallel_consolidation.py -v
```

## References

1. Kumaran, Hassabis, McClelland (2016). "What Learning Systems do Intelligent Agents Need?"
2. McClelland et al. (1995). "Why There Are Complementary Learning Systems"
3. Nader, Schafe, LeDoux (2000). "Fear memories require protein synthesis in the amygdala"
4. Shin et al. (2017). "Continual Learning with Deep Generative Replay"
5. Xie et al. (2013). "Sleep Drives Metabolite Clearance from the Adult Brain"
6. Girardeau & Zugaro (2011). "Hippocampal ripples and memory consolidation"
