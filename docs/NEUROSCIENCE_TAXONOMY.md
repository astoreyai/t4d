# T4DM Neuroscience Taxonomy

**Version**: 1.0 | **Date**: 2026-02-02 | **Status**: Complete

A formal 4-level mapping from neuroscience to software implementation.

---

## Level 1: Brain Regions to Software Modules

| Brain Region | Subregion | Module | File(s) | Lines | Accuracy | Fidelity Level |
|-------------|-----------|--------|---------|-------|----------|---------------|
| Hippocampus | DG/CA3/CA1 | HippocampalCircuit | nca/hippocampus.py, encoding/sparse.py, encoding/attractor.py | 2,030 | Excellent | Circuit-level |
| Neocortex | PFC (dlPFC) | WorkingMemory | nca/wm_gating.py, nca/theta_gamma_integration.py | 1,056 | Good | Mechanism-level |
| Thalamus | Reticular/Relay | ThalamicGate, SleepSpindles | spiking/thalamic_gate.py, nca/sleep_spindles.py | 647 | Good | Mechanism-level |
| Basal Ganglia | Striatum (D1/D2) | StriatalMSN | nca/striatal_msn.py, nca/substantia_nigra.py | 1,368 | Excellent | Circuit-level |
| Amygdala | BLA/CeA | AmygdalaCircuit | nca/amygdala.py | 574 | Good | Mechanism-level |
| VTA | DA neurons | VTACircuit | nca/vta.py, learning/dopamine.py | 1,838 | Excellent | Circuit-level |
| Locus Coeruleus | NE neurons | LocusCoeruleus | nca/locus_coeruleus.py, learning/norepinephrine.py | 1,566 | Excellent | Circuit-level |
| Raphe Nuclei | 5-HT neurons | RapheNucleus | nca/raphe.py, learning/serotonin.py | 1,522 | Good | Mechanism-level |
| Nucleus Basalis | ACh neurons | NucleusBasalis | nca/nucleus_basalis.py, learning/acetylcholine.py | 1,055 | Good | Mechanism-level |
| Astrocytes | Tripartite synapse | AstrocyteLayer | nca/astrocyte.py | 518 | Impressive | Functional-analogy |
| Glymphatic System | CSF clearance | GlymphaticSystem | nca/glymphatic.py | 740 | Novel | Engineering |
| **Cerebellum** | **Purkinje/granule** | **NOT IMPLEMENTED** | -- | -- | **Critical gap** | -- |

### Fidelity Level Definitions

| Level | Definition | Example |
|-------|-----------|---------|
| **Circuit-level** | Models specific neural populations, connectivity, firing patterns | Hippocampal DG-CA3-CA1 trisynaptic circuit |
| **Mechanism-level** | Captures core mechanism but simplifies circuitry | Amygdala fear conditioning without full BLA microcircuit |
| **Functional-analogy** | Reproduces function via different computational means | Astrocyte Ca2+ signaling as modulatory gating |
| **Engineering** | Inspired by biology but primarily an engineering solution | Glymphatic waste clearance as garbage collection |

---

## Level 2: Neural Mechanisms to Algorithms

| Neural Mechanism | Algorithm | Implementation | Accuracy | Reference |
|-----------------|-----------|----------------|----------|-----------|
| LTP/LTD | Bounded Hebbian + BCM sliding threshold | learning/hebbian.py, learning/homeostatic.py | Simplified but stable | Hebb 1949, Bienenstock 1982 |
| STDP | Exponential window (tau+=17ms, tau-=34ms) + DA modulation | learning/stdp.py | Accurate | Bi & Poo 1998 |
| Sharp-wave ripples | 10x compressed, PE-weighted, forward/reverse replay | consolidation/sleep.py, nca/swr_coupling.py | Good | Foster & Wilson 2006 |
| Theta-gamma coupling | PAC with WM capacity estimation (7+/-2 slots) | nca/theta_gamma_integration.py, nca/oscillators.py | Good | Lisman & Jensen 2013 |
| Pattern separation | 8x expansion + k-WTA (4% sparsity) in DG | encoding/sparse.py, nca/hippocampus.py | Accurate | Treves & Rolls 1994 |
| Pattern completion | Modern Hopfield attractor network in CA3 | encoding/attractor.py | Good | Ramsauer et al. 2020 |
| Three-factor learning | Eligibility x neuromodulator x DA surprise | learning/stdp.py, spiking/neuromod_bus.py | Accurate | Fremaux & Gerstner 2016 |
| Sleep spindles | Thalamocortical 11-16 Hz, coupled to delta up-states | nca/sleep_spindles.py | Good | Diekelmann & Born 2010 |
| Dopamine RPE | TD(lambda) prediction error, phasic burst/dip | learning/dopamine.py, nca/vta.py | Excellent | Schultz 1997 |
| NE gain modulation | Phasic/tonic modes, 5 firing patterns | nca/locus_coeruleus.py | Excellent | Aston-Jones & Cohen 2005 |
| ACh encoding/retrieval | Mode switch (high=encode, low=retrieve) | learning/acetylcholine.py | Good | Hasselmo 2006 |
| Memory consolidation | kappa gradient [0,1] via LSM compaction | storage/t4dx/, consolidation/ | Novel and defensible | Frankland & Bontempi 2005 |
| Synaptic homeostasis | BCM sliding threshold + firing rate targets | learning/homeostatic.py | Good | Turrigiano 2008 |
| Fear conditioning | Amygdala BLA pathway | nca/amygdala.py | Good | LeDoux 2000 |

---

## Level 3: Computational Models to Equations

| Model | Equation | Implementation File | Reference |
|-------|---------|-------------------|-----------|
| LIF neuron | `u(t+1) = alpha*u(t) + I(t) - beta*spike` | spiking/lif.py | Zenke & Ganguli 2018 |
| FSRS decay | `R(t,S) = (1 + 0.9*t/S)^(-0.5)` | core/types.py | Wixted & Ebbesen 1991 |
| Hebbian update | `w' = w + eta*(1 - w)` | learning/hebbian.py (via semantic.py) | Hebb 1949 |
| ACT-R activation | `A_i = ln(sum t_j^(-d)) + sum W_j*S_ji + epsilon` | memory/semantic.py | Anderson 2007 |
| Tau temporal gate | `tau(t) = sigma(lambda_e*e + lambda_delta*novelty + lambda_r*reward)` | core/temporal_gate.py | Novel |
| Time2Vec encoding | `t2v(t)[i] = omega_i*t + phi_i (linear) or sin(omega_i*t + phi_i)` | encoding/time2vec.py | Kazemi et al. 2019 |
| Modern Hopfield | `p = softmax(beta * X^T * xi)` | encoding/attractor.py | Ramsauer et al. 2020 |
| Forward-Forward goodness | `G(h) = sum h_i^2` | nca/forward_forward.py | Hinton 2022 |
| STDP window | `dw = A+ * exp(-dt/tau+) if dt>0 else -A- * exp(dt/tau-)` | learning/stdp.py | Bi & Poo 1998 |
| Dopamine RPE | `delta = r + gamma*V(s') - V(s)` | learning/dopamine.py | Schultz 1997 |
| RRF fusion | `RRF(d) = sum 1/(k + rank_r(d))` | memory/semantic.py | Robertson 2009 |

---

## Level 4: Neuroengineering to Implementation

| Neuroengineering Concept | T4DM Implementation | Purpose | Fidelity |
|--------------------------|-------------------|---------|----------|
| BCI spike decoding | Spiking cortical blocks (LIF + STDP attention) | Memory adapter between Qwen layers | Engineering |
| Optogenetics (targeted activation) | Neuromodulator bus (layer-specific NT injection) | DA/NE/ACh/5-HT to specific cortical block layers | Engineering |
| Neural prosthetics | QLoRA adapters on frozen Qwen | Trainable memory interface on frozen LLM | Engineering |
| Deep brain stimulation | Oscillatory phase bias (theta/gamma/delta currents) | Bias LIF thresholds for encoding/consolidation | Engineering |
| Neural dust (distributed sensing) | Activation visibility hooks (36 Qwen + 6 spiking layers) | Glass-box observability | Engineering |
| EEG oscillation monitoring | FrequencyBandGenerator + ThetaGammaIntegration | Phase-amplitude coupling for WM capacity | Functional-analogy |
| fMRI BOLD signal | Visualization modules (22 renderers) | Neural activity visualization | Functional-analogy |
| Connectome mapping | nca/connectome.py (brain region connectivity graph) | Inter-module communication topology | Mechanism-level |

---

## Gap Analysis

### Implemented Brain Regions (12/13 target)

```
IMPLEMENTED:
  Hippocampus ............. Excellent (DG/CA3/CA1 trisynaptic circuit)
  VTA ..................... Excellent (5 firing modes, RPE)
  Locus Coeruleus ........ Excellent (5 firing modes, phasic/tonic)
  Striatum ............... Excellent (D1/D2 MSN pathways)
  Amygdala ............... Good (fear conditioning, emotional tagging)
  Raphe Nuclei ........... Good (5-HT, temporal discounting)
  Nucleus Basalis ........ Good (ACh, encoding/retrieval switching)
  Thalamus ............... Good (thalamic gate + sleep spindles)
  Neocortex/PFC .......... Good (WM gating, theta-gamma)
  Astrocytes ............. Impressive (tripartite synapse)
  Glymphatic ............. Novel (waste clearance)
  Substantia Nigra ....... Good (DA output, basal ganglia)

NOT IMPLEMENTED:
  Cerebellum ............. CRITICAL GAP
    - Motor timing and error-driven learning
    - Predictive internal models
    - Planned for Phase G-06 (OPTIMIZATION_PLAN.md)
```

### Mechanism Coverage

| Category | Implemented | Missing |
|----------|-----------|---------|
| Memory encoding | Pattern separation, Hopfield completion, sparse coding | Dentate gyrus neurogenesis |
| Memory consolidation | NREM/REM/Prune, SWR, sleep spindles, kappa gradient | K-complexes |
| Learning rules | Hebbian, STDP, BCM, three-factor, Forward-Forward | Reservoir computing |
| Neuromodulation | DA, NE, ACh, 5-HT, GABA, Glu | Endocannabinoids, oxytocin |
| Oscillations | Theta, gamma, alpha, delta, spindles | Beta oscillations |
| Spatial cognition | Place cells, grid cells (basic) | Head direction cells, boundary cells |

---

## References

1. Anderson, J.R. (2007). How Can the Human Mind Occur in the Physical Universe?
2. Aston-Jones, G. & Cohen, J.D. (2005). An integrative theory of locus coeruleus-norepinephrine function.
3. Bi, G.Q. & Poo, M.M. (1998). Synaptic modifications in cultured hippocampal neurons.
4. Bienenstock, E.L., Cooper, L.N., & Munro, P.W. (1982). Theory for the development of neuron selectivity.
5. Diekelmann, S. & Born, J. (2010). The memory function of sleep.
6. Foster, D.J. & Wilson, M.A. (2006). Reverse replay of behavioural sequences in hippocampal place cells.
7. Frankland, P.W. & Bontempi, B. (2005). The organization of recent and remote memories.
8. Fremaux, N. & Gerstner, W. (2016). Neuromodulated spike-timing-dependent plasticity.
9. Hasselmo, M.E. (2006). The role of acetylcholine in learning and memory.
10. Hebb, D.O. (1949). The Organization of Behavior.
11. Hinton, G. (2022). The Forward-Forward Algorithm.
12. Kazemi, S.M. et al. (2019). Time2Vec: Learning a general-purpose representation of time.
13. LeDoux, J.E. (2000). Emotion circuits in the brain.
14. Lisman, J.E. & Jensen, O. (2013). The theta-gamma neural code.
15. Ramsauer, H. et al. (2020). Hopfield Networks is All You Need.
16. Schultz, W. (1997). A neural substrate of prediction and reward.
17. Treves, A. & Rolls, E.T. (1994). Computational analysis of the role of the hippocampus in memory.
18. Turrigiano, G.G. (2008). The self-tuning neuron.
19. Zenke, F. & Ganguli, S. (2018). SuperSpike: Supervised learning in multilayer spiking neural networks.

---

*Generated 2026-02-02 from codebase analysis*
