# Cold Start Priming System

## Overview

The cold start system addresses the "chicken and egg" problem in learned memory systems:
- The gate needs examples to learn what to store
- Without storing, there are no examples to learn from

## Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ColdStartManager                             │
│  Orchestrates initialization, checkpointing, and finalization   │
└──────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  StatePersister │  │  ContextLoader  │  │ PopulationPrior │
│  ~/.ww/state/   │  │  CLAUDE.md etc  │  │  Default weights│
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Components

### 1. StatePersister (`learning/persistence.py`)

Handles saving/loading learned state across sessions:

```python
from ww.learning.persistence import StatePersister

persister = StatePersister(storage_path="~/.ww/learned_state")

# Save gate state
persister.save_gate_state(learned_gate)

# Restore from disk
persister.restore_gate(learned_gate)

# Save neuromodulator state
persister.save_neuromodulator_state(orchestra)
```

**Stored State:**
- `LearnedGateState`: Weight means, covariances, bias, observation count
- `ScorerState`: MLP layer weights and training statistics
- `NeuromodulatorState`: Dopamine expectations, serotonin values, NE reference

### 2. ContextLoader (`learning/cold_start.py`)

Extracts context from project files to inform priors:

```python
from ww.learning.cold_start import ContextLoader

loader = ContextLoader(working_dir="/home/user/project")
signals = loader.load_context(
    claude_md_path="~/.claude/CLAUDE.md",
    include_git=True
)
```

**Context Sources:**
- Project type detection (Python, TypeScript, Rust, etc.)
- CLAUDE.md priority patterns ("IMPORTANT", "CRITICAL")
- README keywords and structure
- Recent git activity

### 3. PopulationPrior (`learning/cold_start.py`)

Default priors based on observed patterns:

```python
from ww.learning.cold_start import PopulationPrior

prior = PopulationPrior()
weights = prior.get_prior_weights(feature_dim=1143)
```

**Prior Biases:**
- Neuromodulator section:
  - DA RPE positive (+0.3) - surprise = valuable
  - NE gain positive (+0.2) - novelty = valuable
  - ACh encoding mode (+0.4) - should store in encoding mode
  - ACh retrieval mode (-0.3) - don't store when retrieving

### 4. ColdStartManager (`learning/cold_start.py`)

Orchestrates the initialization lifecycle:

```python
from ww.learning.cold_start import ColdStartManager

manager = ColdStartManager(gate, orchestra, persister)

# Session start: load persisted state or apply priors
result = manager.initialize_session(
    working_dir="/path/to/project",
    claude_md_path="~/.claude/CLAUDE.md"
)

# Periodic checkpoint
manager.checkpoint()

# Session end: save state
manager.finalize_session(session_outcome=0.8)
```

## Initialization Strategy

Priority order:
1. **Persisted State** (if available): Load previous session's learned weights
2. **Context Priors**: Bias weights based on project type and CLAUDE.md
3. **Population Priors**: Apply default biases for neuromodulator features

## Cold Start Blending

During cold start, the system linearly interpolates between heuristic and learned predictions:

```
α = n_observations / cold_start_threshold
p_final = (1 - α) × p_heuristic + α × p_learned
```

This ensures smooth transition as the gate learns.

## Progress Tracking

```python
progress = manager.get_cold_start_progress()  # [0, 1]

weights = manager.get_current_blend_weights()
# {"heuristic": 0.75, "learned": 0.25}  # at 25% progress
```

## File Locations

State is stored in `~/.ww/learned_state/`:
- `learned_gate.json.gz` - Gate weights and statistics
- `learned_scorer.pkl.gz` - Scorer MLP weights
- `neuromodulators.pkl.gz` - Neuromodulator system state

## Integration Example

```python
from ww.core.learned_gate import LearnedMemoryGate
from ww.learning.neuromodulators import NeuromodulatorOrchestra
from ww.learning.cold_start import ColdStartManager

# Create components
orchestra = NeuromodulatorOrchestra()
gate = LearnedMemoryGate(neuromod_orchestra=orchestra)
manager = ColdStartManager(gate, orchestra)

# Initialize with priming
result = manager.initialize_session(
    working_dir=".",
    claude_md_path="~/.claude/CLAUDE.md"
)
print(f"Strategy: {result['strategy']}")  # "persisted" or "cold_start_priors"

# ... normal operation ...

# Save at session end
manager.finalize_session(session_outcome=0.85)
```

## Tests

The cold start system is tested in `tests/unit/test_cold_start.py` with 18 tests covering:
- State serialization and restoration
- Context loading from CLAUDE.md and README
- Population prior generation
- Manager lifecycle (initialize → checkpoint → finalize)
- Cold start progress calculation
