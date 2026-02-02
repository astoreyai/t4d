# NCA (Neural Circuit Architecture)
**Path**: `/mnt/projects/t4d/t4dm/src/t4dm/nca/`

> **Naming Note**: NCA in this project stands for "Neural Circuit Architecture" -- biologically-inspired brain region simulation modules. This is NOT Mordvintsev-style Neural Cellular Automata (self-organizing pixel patterns). The module implements computational models of specific brain regions and neural circuits.

## What
Biologically-inspired neural field dynamics implementing brain-region simulations, neurotransmitter systems, oscillatory networks, Hinton architectures (Forward-Forward, Capsule Networks), and sleep/wake consolidation. The most neuroscience-heavy module (~40 files).

## How
### Core Neural Dynamics
- **NeuralFieldSolver** (`neural_field.py`): 3D spatiotemporal PDE solver for neurotransmitter concentration dynamics. Configurable grid, diffusion, and decay parameters.
- **LearnableCoupling** (`coupling.py`): Coupling matrix between neurotransmitter systems with biological bounds (e.g., DA-5HT antagonism).
- **AttractorBasins** (`attractors.py`): Cognitive state attractors (FOCUSED, EXPLORATORY, CONSOLIDATING, etc.) with transition dynamics.
- **EnergyLandscape** (`energy.py`): Hopfield-inspired energy function with contrastive learning phases.

### Brain Regions
- **HippocampalCircuit** (`hippocampus.py`): DG -> CA3 -> CA1 circuit for encoding/retrieval. Pattern separation in DG, auto-association in CA3, output in CA1.
- **VTACircuit** (`vta.py`): Ventral tegmental area dopamine circuit. Phasic/tonic firing modes, reward prediction.
- **LocusCoeruleus** (`locus_coeruleus.py`): Norepinephrine source with surprise model for uncertainty signaling.
- **RapheNucleus** (`raphe.py`): Serotonin source with patience/temporal discounting model.
- **StriatalMSN** (`striatal_msn.py`): Medium spiny neurons for action selection (D1 go / D2 no-go pathways).

### Oscillators & Sleep
- **FrequencyBandGenerator** (`oscillators.py`): Theta, gamma, alpha, delta oscillations. Phase-amplitude coupling.
- **SleepSpindleGenerator** (`sleep_spindles.py`): Thalamocortical spindles for memory consolidation.
- **SWRNeuralFieldCoupling** (`swr_coupling.py`): Sharp-wave ripple coupling for replay during sleep.
- **AdenosineDynamics** (`adenosine.py`): Sleep pressure accumulation and wake/sleep state transitions.
- **GlymphaticSystem** (`glymphatic.py`): Metabolic waste clearance during sleep.
- **ThetaGammaIntegration** (`theta_gamma_integration.py`): Working memory slot binding via theta-gamma nesting.

### Hinton Architectures
- **ForwardForwardNetwork** (`forward_forward.py`): Hinton's 2022 FF algorithm -- local learning without backprop using goodness functions.
- **CapsuleNetwork** (`capsules.py`): Part-whole relationships via pose matrices and dynamic routing.
- **PoseLearner** (`pose_learner.py`): Emergent semantic dimension discovery for capsule poses.

### Glial & Signaling
- **AstrocyteLayer** (`astrocyte.py`): Tripartite synapse modulation (glia influence on synaptic transmission).
- **GlutamateSignaling** (`glutamate_signaling.py`): NMDA/AMPA receptor dynamics, LTP/LTD direction.
- **SpatialCells** (`spatial_cells.py`): Place cells and grid cells for cognitive spatial mapping.

## Why
Provides the biological substrate for WW's learning and memory. Neural oscillations gate encoding/retrieval, dopamine modulates plasticity, sleep consolidates memories, and Hinton architectures enable local (non-backprop) learning. This is what makes WW biologically plausible rather than just another vector database.

## Key Files
| File | Purpose |
|------|---------|
| `neural_field.py` | Core PDE solver for NT dynamics |
| `hippocampus.py` | DG-CA3-CA1 encoding/retrieval circuit |
| `vta.py` | Dopamine reward circuit |
| `oscillators.py` | Theta/gamma/alpha/delta oscillations |
| `forward_forward.py` | Hinton FF local learning |
| `capsules.py` | Capsule networks with routing |
| `connectome.py` | Brain region connectivity graph |
| `energy.py` | Hopfield energy landscape |
| `swr_coupling.py` | Sharp-wave ripple memory replay |

## Data Flow
```
Sensory input -> NeuralField (NT concentration update)
    -> Oscillators (phase gating: theta=encode, sharp-wave=replay)
    -> Hippocampus (DG separation -> CA3 association -> CA1 output)
    -> VTA (dopamine RPE) -> Coupling (modulate plasticity)
    -> ForwardForward / Capsules (local learning)
    -> Energy landscape (settle into attractor)
    -> Sleep: Adenosine pressure -> spindles + SWR -> glymphatic clearance
```

## Integration Points
- **learning/**: Dopamine/NE/ACh/5HT systems connect to neuromodulator orchestra
- **learning/stdp.py**: VTA dopamine modulates STDP via integration/stdp_vta_bridge
- **memory/**: Hippocampal circuit drives episodic encoding and retrieval
- **memory/pattern_separation.py**: DentateGyrus maps to nca/hippocampus DG layer
- **bridge/**: Memory-NCA binding for neural field <-> memory store coupling
