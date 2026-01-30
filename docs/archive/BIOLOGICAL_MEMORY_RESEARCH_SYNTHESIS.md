# Biological Memory Systems: Research Synthesis for AI Architecture
**Date**: 2025-12-06
**Purpose**: Neuroscience-informed design principles for World Weaver memory system

---

## Executive Summary

This synthesis examines biological memory mechanisms across seven domains to inform artificial memory architecture design. Key findings reveal that biological systems solve catastrophic forgetting through complementary learning systems, multi-timescale dynamics, neuromodulatory gating, and dendritic compartmentalization. These principles suggest specific computational strategies for implementing stable, continual learning in AI systems.

---

## 1. Hippocampal Memory Systems

### Pattern Separation (DG-CA3)
**Mechanism**: Dentate granule cells use competitive learning to convert grid-like entorhinal cortex firing to sparse, place-like fields. Mossy fiber inputs provide randomizing effect enabling orthogonalization of similar inputs.

**Recent Findings (2024-2025)**:
- **Temporal persistence**: Hippocampal pattern completion contributes to holistic episodic retrieval both immediately and 24h post-encoding, working additively with neocortical processes rather than competitively
- **Extra-hippocampal contributions**: Prefrontal and parietal regions (CHiPS framework) contribute to pattern separation by resolving interference in sensory regions before hippocampal input
- **Naturalistic stimuli**: DG-CA3 pair shows pattern separation during real-world stimuli, while CA3-CA1 and CA1-SUB exhibit pattern completion

**Computational Principles**:
1. **Sparse coding in episodic buffer**: Use competitive learning for orthogonalization
2. **Multi-stage separation**: Pre-hippocampal interference resolution + hippocampal separation
3. **Persistent completion**: Pattern completion mechanisms remain active during consolidation
4. **Complementary processes**: Separation and completion operate in parallel, not sequentially

### Pattern Completion (CA3-CA1)
**Mechanism**: Recurrent CA3 connections enable completion from partial cues. CA3 auto-associative network recalls full patterns from fragments.

**AI Implementation**:
- Hopfield-style recurrent dynamics in episodic memory
- Asymmetric connectivity: strong recurrence in CA3-analog, feed-forward CA3→CA1-analog
- Cue-based retrieval with degraded/partial inputs

### Memory Consolidation
**Systems Consolidation**:
- Hippocampus stores rapid associations, gradually transfers to neocortex
- Selective consolidation: Not all memories persist; importance/relevance determines fate
- Schema extraction: Gradual abstraction from episodic to semantic representations

**Recent Findings**:
- **Awake consolidation** (2025): Brief rest periods facilitate consolidation through hippocampal-cortical dialogue, reducing interference
- **Reinforcement learning perspective** (2024): Consolidation viewed as value-based selection and schema formation
- **Additive, not compensatory**: Hippocampus and neocortex work together during retrieval after consolidation

**AI Principles**:
- **Dual-store architecture required**: Fast episodic + slow semantic stores
- **Importance-weighted consolidation**: Use value/utility signals to select memories
- **Schema extraction**: Cluster episodic memories to build semantic abstractions
- **Offline replay**: Use idle periods for consolidation, not just sleep

---

## 2. Synaptic Plasticity Mechanisms

### Long-Term Potentiation (LTP) and Depression (LTD)
**Temporal Dynamics**:
- LTP: Presynaptic spike precedes postsynaptic by ~0-20ms → strengthening
- LTD: Postsynaptic leads presynaptic by ~20-100ms → weakening
- Sharp transition window: 1-5ms between LTP and LTD

**Functional Roles** (2024):
- **LTP**: Generates spatial experience records, creates associative schemas for rapid re-use
- **LTD**: Enables dynamic updating and modification of representations
- **Interplay**: LTP/LTD balance supports complex associative memories resistant to generalization

**Variability**:
- Symmetric, anti-Hebbian, frequency-dependent variants exist
- Anti-Hebbian STDP in striatum (pre-before-post → LTD) supports sequence learning
- Neuromodulation can reverse STDP sign

### Spike-Timing Dependent Plasticity (STDP)
**Recent Developments** (2024):
- **Anti-Hebbian sequence learning**: Striatal medium spiny neurons combine anti-Hebbian STDP with non-associative potentiation to learn spatiotemporal sequences
- **Three basic mechanisms**: Anti-Hebbian learning + spike latency + collateral inhibition = efficient sequence acquisition

**Neuromodulatory Control**:
- Cholinergic modulation regulates GABAergic inhibition of dendritic Ca²⁺
- Dopamine, ACh, NE can modulate or reverse STDP magnitude and sign
- Context-dependent plasticity rules

**AI Principles**:
1. **Temporal contiguity**: Weight updates based on precise spike timing, not just correlation
2. **Bidirectional plasticity**: Both strengthening and weakening required
3. **Multiple STDP rules**: Different learning rules for different subsystems (Hebbian vs anti-Hebbian)
4. **Neuromodulated plasticity**: Learning rate and rule type controlled by modulatory signals

---

## 3. Neuromodulation and Learning Gates

### Dopamine System
**Dual Role**:
- **D1 antagonism** (2024-2025): Slows probabilistic learning, reduces corticocortical and frontostriatal connectivity
- **D2 antagonism**: Improves learning, enhances cortical connectivity
- **Classical role**: Reward prediction error signaling

**Computational Theory**:
- Dopamine signals **error in reward prediction**
- Gates learning in striatal and cortical networks
- Modulates working memory maintenance in dlPFC

### Acetylcholine System
**Attentional Gating**:
- Critical for working memory through enhanced attention
- α4β2R receptors support persistent firing against distractions
- Slows fear extinction when BLA terminals stimulated during training

**Sleep and Consolidation** (2024):
- Orchestrates oscillatory activity during NREM sleep
- Suppresses sharp-wave ripples, slow waves, spindles
- Infraslow fluctuations time oscillatory events

### Norepinephrine System
**Functions**:
- Arousal and alertness
- Randomness in action selection
- Memory consolidation during sleep
- Novelty encoding: Hippocampal NE release encodes novel contextual information

### Serotonin (5-HT)
**Computational Role**:
- Controls **time scale of reward prediction**
- Modulates learning rates
- Combined with DA signals for integrated neuromodulation

### Astrocytes as Neuromodulators (2025)
**Emerging Role**:
- Express receptors for classical neuromodulators
- Calcium elevations stimulate gliotransmitter release
- Spatially and temporally integrate neuronal and neuromodulatory signals
- Local support function for neuromodulatory systems

**AI Principles**:
1. **Multi-signal integration**: Multiple neuromodulators provide orthogonal control dimensions
2. **Context-dependent learning**: DA for value, NE for novelty, ACh for attention, 5-HT for timescale
3. **Gated plasticity**: Neuromodulators gate when learning occurs, not just what is learned
4. **Hierarchical modulation**: Fast synaptic + slow neuromodulatory dynamics
5. **Sleep-based orchestration**: Neuromodulators control consolidation timing

---

## 4. Memory Consolidation and Replay

### Sleep-Dependent Consolidation
**Oscillatory Coordination**:
- **Slow oscillations (SOs)**: Set global excitability/inhibition windows
- **Spindles**: Partially reactivate cortical networks, facilitate hippocampal ripples
- **Ripples**: Activate local memory circuits, drive hippocampal-cortical pattern completion
- **SO-spindle-ripple coupling**: Governs both synaptic and systems consolidation

**Mechanisms** (2024-2025):
- **Active systems consolidation**: Repeated hippocampal replay transfers to neocortex
- **Synaptic downselection**: Widespread synaptic pruning accompanies consolidation
- **Memory transformation**: Abstraction to semantic representations during transfer
- **Temporal compression**: Experiences replayed at accelerated timescales

### Memory Replay
**Sharp-Wave Ripples (SWRs)**:
- Neural sequences from waking reactivate during sleep in same order
- 150-250Hz ripple oscillations during slow-wave sleep
- Causal role in consolidation confirmed in rats and humans
- Replay both forward and reverse sequences

**Awake Replay** (2025):
- Brief rest periods enable consolidation without sleep
- Cortical reactivation predicts subsequent memory
- Retrieval-mediated consolidation as alternative to sleep

**AI Principles**:
1. **Offline replay required**: Use idle/rest periods for memory strengthening
2. **Nested oscillations**: Multi-timescale coordination for consolidation (fast ripples nested in slow oscillations)
3. **Temporal compression**: Replay experiences faster than real-time
4. **Bidirectional replay**: Both forward and backward temporal sequences
5. **Selective consolidation**: Value/importance signals determine what gets replayed

---

## 5. Complementary Learning Systems (CLS)

### Original Theory (McClelland, McNaughton, O'Reilly 1995)
**Core Principle**: Brain requires two specialized learning systems to avoid catastrophic interference:

**Hippocampal System**:
- Sparse, pattern-separated representations
- Rapid learning of episodic memories
- Supports one-shot learning
- Temporary storage

**Neocortical System**:
- Distributed, overlapping representations
- Gradual integration across episodes
- Extracts latent semantic structure
- Permanent storage

**Consolidation Mechanism**:
- Hippocampal reinstatement of recent memories in neocortex
- Neocortical synapses change incrementally on each reinstatement
- Remote memory based on accumulated neocortical changes

### Updated Theory (2016)
**Schema-Consistent Rapid Learning**:
- Neocortex can rapidly learn new information **if consistent with prior schemas**
- Catastrophic interference only occurs when inconsistent information learned quickly
- Schema frameworks enable fast neocortical integration

### Recent Extensions (2024-2025)
**Emergence of CLS** (CCN 2024):
- CLS principles emerging in artificial systems through proper architectural constraints
- Not just biological quirk, but fundamental solution to stability-plasticity dilemma

**AI Implementation Principles**:
1. **Dual-store architecture mandatory**: Episodic (sparse, fast) + Semantic (distributed, slow)
2. **Separated learning rates**: 10-100x faster for episodic than semantic
3. **Schema-gated consolidation**: Rapid neocortical learning when schema-consistent
4. **Replay-based transfer**: Hippocampal replay drives neocortical updates
5. **Complementary, not redundant**: Systems solve different computational problems

---

## 6. Predictive Coding Framework

### Hierarchical Prediction Error Minimization
**Core Mechanism**:
- Higher levels predict lower-level activity
- Prediction errors (PE) propagate upward
- Minimize free energy through prediction improvement

**Recent Challenges** (2024-2025):
- **PFC-centric PEs**: Genuine prediction errors emerge in prefrontal cortex, not V1
- **Predictive routing alternative**: PE signals may route information rather than just update predictions
- **Implementation debate**: Unclear how to implement with spiking neurons

### Layer 2/3 Circuit Mechanisms (2025)
**Dendritic Computation Model**:
- Feedforward input targets soma
- Top-down feedback reaches distal apical dendrite
- Local somato-dendritic comparison computes PE
- Sign-specific PEs when excitation/inhibition balance disrupted in one compartment
- Three inhibitory subtypes interact with two-compartment pyramidal neurons

### Predictive Coding Light (2025)
**Alternative Framework**:
- Does **not** transmit prediction errors upward
- Suppresses most predictable spikes
- Transmits compressed representation of input
- Efficient coding through spike suppression

### Crossmodal Predictive Coding (2024)
**Hierarchical Networks**:
- Unimodal predictions processed by distributed networks
- Crossmodal knowledge formed through alpha-band interactions
- Rapid redirection to central-parietal electrodes during learning
- Spatio-spectro-temporal signatures of PE across hierarchies

**AI Principles**:
1. **Hierarchical predictions**: Each level predicts activity of level below
2. **Error-based learning**: Update based on prediction errors, not raw inputs
3. **Compressed transmission**: Send only unpredicted/surprising information
4. **Dendritic compartments**: Separate processing of feedforward vs feedback
5. **Multi-timescale hierarchy**: Slower predictions at higher levels
6. **Precision weighting**: PE magnitude weighted by confidence/uncertainty

---

## 7. Dendritic Computation and Local Learning

### Dendritic Compartmentalization
**Biological Structure**:
- Separate soma and dendritic compartments with non-linear integration
- Local calcium dynamics enable branch-specific learning
- NMDA and L-type calcium channels for local plasticity signals
- Backpropagating action potentials from soma to dendrites

**Computational Advantages** (2024-2025):
- **91% vs 88% accuracy**: Dendritic models outperform purely synaptic models (MNIST)
- **Compartmentalized processing**: Different dendritic branches process different features
- **Non-linear integration**: Enables solving non-linear feature binding problem
- **Memory linking**: Same dendritic segments preferentially activated by linked memories

### Local Learning Rules
**Calcium-Based Plasticity**:
- Local NMDA and L-type Ca²⁺ channel dynamics
- Dopaminergic reward signals for three-factor learning
- Metaplasticity ensures stability of individual weights
- Branch-specific allocation of memory traces

### Dendritic Target Propagation (2024-2025)
**Biologically Plausible Backprop Alternative**:
- Separate soma and dendrite compartments in each unit
- Distal dendrites process top-down signals
- Local error computation through somato-dendritic mismatch
- No need for symmetric weights or global error signals

### Temporal Hierarchy Through Dendritic Heterogeneity
**Multi-Timescale Learning**:
- Different dendritic branches have different temporal integration windows
- Heterogeneous dendritic time constants
- Enables learning at multiple timescales simultaneously

**AI Principles**:
1. **Two-compartment neurons**: Separate soma (feedforward) and dendrite (feedback/context)
2. **Local learning**: No global error backpropagation required
3. **Non-linear subunits**: Dendritic branches as independent computational units
4. **Branch-specific plasticity**: Different learning rules per dendritic compartment
5. **Temporal heterogeneity**: Different timescales in different branches
6. **Inhibitory shaping**: Three inhibitory subtypes for compartmentalization

---

## 8. Avoiding Catastrophic Forgetting

### Biological Mechanisms

**Complementary Learning Systems** (see Section 5):
- Primary biological solution to catastrophic forgetting
- Dual-store architecture with separated learning rates
- Hippocampal replay prevents neocortical interference

**Metaplasticity from Synaptic Uncertainty (2025)**:
- **Bayesian synapses**: Maintain uncertainty "error bars" on weights
- Adjust learning rate based on confidence
- High-confidence synapses resist change, low-confidence adapt quickly
- Resonates with concept of synaptic consolidation
- Avoids both catastrophic forgetting and catastrophic remembering

**Corticohippocampal Hybrid Networks (2024-2025)**:
- **CH-HNN architecture**: Combines ANNs and SNNs
- Dual representation of specific and generalized memories
- Episode inference facilitates new learning using prior knowledge
- Task-agnostic without increasing memory demands
- Significant mitigation in task-incremental and class-incremental scenarios

### Biological Forgetting Types
**Passive Mechanisms**:
- Natural decay over time
- Transient forgetting (reversible)

**Active Mechanisms**:
- Intentional forgetting
- Retrieval-induced forgetting
- **Retroactive interference**: New learning during consolidation interferes with old memories

### Synaptic Consolidation
**Elastic Weight Consolidation (EWC)**:
- Inspired by synaptic consolidation in brain
- Synapses carry multiple pieces of information:
  - Short-term plasticity: Current synaptic strength
  - Long-term plasticity: Mean weight (consolidated)
  - Variance: Uncertainty estimate

**Eligibility Traces**:
- Molecular memory of recent activity
- Allows delayed credit assignment (seconds to minutes)
- Protects recently active synapses from modification

### AI Implementation Principles
1. **Dual-store architecture**: Episodic + semantic with different learning rates
2. **Bayesian weight uncertainty**: Track confidence per parameter
3. **Importance weighting**: Protect important weights from change (EWC)
4. **Replay-based consolidation**: Interleave old and new examples
5. **Metaplasticity**: Adjust learning rates based on weight history
6. **Eligibility traces**: Molecular-level credit assignment
7. **Schema-consistent learning**: Fast learning only when consistent with existing knowledge
8. **Selective consolidation**: Not everything needs long-term storage

---

## 9. Multi-Timescale Neural Dynamics

### Temporal Hierarchy in Cortex
**Robust Finding** (2024-2025):
- **Hierarchical organization**: Timescales increase along cortical hierarchy
- Fast in primary sensory/motor areas (tens of milliseconds)
- Slow in transmodal association areas (seconds)
- Consistent across species (rodents, primates, humans)
- Consistent across modalities (spiking, LFP, ECoG, fMRI)

**Recent Evidence** (2025):
- Zeisler et al.: Consistent hierarchies across mice, macaques, humans
- Cusinato et al.: Sleep modulates neural timescales and spatiotemporal integration
- Temporal hierarchy provides 2-6% accuracy gains in SNNs
- Under iso-accuracy, hierarchy reduces parameters by 5x

### Multiple Timescales Within Neurons
**Parallel Hierarchies**:
- Most neurons exhibit **multiple timescales** simultaneously
- Timescales not correlated across neurons in same area
- Independent parallel hierarchies of temporal integration
- Enables simultaneous processing at different rates

### Mechanisms
**Unimodal-Transmodal Gradient**:
- Shorter intrinsic neural timescales (INT) in unimodal regions (V1, A1)
- Longer INT in transmodal regions (default mode network)
- Central to temporal integration and segregation of inputs

**Multisensory Integration** (2024):
- Multi-timescale dynamics within and across brain networks
- Oscillatory mechanisms: Power modulations, phase resetting, phase-amplitude coupling
- Enables simultaneous integration, segregation, hierarchical structuring, and selection
- Information processed in different time windows concurrently

### AI Principles
1. **Hierarchical timescales**: Faster dynamics in lower layers, slower in higher layers
2. **Multiple timescales per unit**: Each neuron integrates at multiple rates
3. **Independent parallel hierarchies**: Timescales not synchronized across population
4. **Intrinsic dynamics**: Built into neuron properties, not just network architecture
5. **Temporal receptive fields**: Units have characteristic integration windows
6. **Oscillatory coordination**: Nested oscillations at multiple frequencies
7. **Context-dependent timescales**: Modulated by task demands and neuromodulation

---

## 10. Sparse Coding and Efficiency

### Theoretical Foundation
**Efficient Population Code**:
- Small fraction of neurons active at any time
- Non-negative sparse coding (NSC) as emergent property
- Dimensionality reduction + sparsity constraints
- Employed by sensory areas for efficient stimulus encoding

**Computational Advantages**:
- **Overcomplete representations**: More features than inputs
- **Decorrelation**: Sparse codes are approximately decorrelated
- **Memory capacity**: Increases beyond complete codes
- **Learning speed**: Faster weight learning with decorrelated inputs
- **Metabolic efficiency**: Minimizes energy consumption

### Recent Developments (2024-2025)
**Sparse Autoencoders (SAEs)**:
- Surge in SAE use for LLM interpretability
- Extraction of interpretable features from activations
- **Limitation**: SAEs fail to achieve optimal recovery (O'Neill et al. 2025)
- Linear-nonlinear encoder lacks complexity for full recovery

**Improved Sparsity** (2024):
- Anisotropic Gaussian priors for sparse codes
- Improves convexity of sparse coding problem
- Models feature correlation
- Reduces error in noisy scenarios

### Visual Cortex Evidence
**Exponential Distributions**:
- V1, V2, V4 firing rates described by exponential distributions
- Consistent with maximizing information transmission
- Subject to metabolic constraints on mean firing rate
- Higher firing during challenging tasks than metabolic predictions

### Relationship to Predictive Coding
**Efficiency Principle**:
- Transmit only mismatch (prediction error)
- Similar to sparse coding's efficiency principle
- Only unexpected information propagates upward
- Compressed representations at each level

### AI Principles
1. **Sparse activations**: Small fraction active at any time (1-5%)
2. **Overcomplete basis**: More units than input dimensions
3. **Non-negative constraints**: Rectification (ReLU-like)
4. **Decorrelation objective**: Minimize feature correlations
5. **Metabolic costs**: Penalize high firing rates
6. **Learned sparsity**: Not hardcoded, emerges from training
7. **Hierarchical sparsity**: Increasing sparsity up hierarchy

---

## 11. Credit Assignment Problem

### The Challenge
**Definition**: Determining how much credit/blame each parameter (synapse) should receive for outcomes.

**Two Types**:
- **Structural**: Which parameters (weights) are responsible?
- **Temporal**: Which past activities led to delayed outcomes?

### Eligibility Traces
**Standard Approach**:
- Molecular memory of recent neuronal activity
- Renders synapses malleable for several seconds
- Mechanisms: Elevated dendritic spine Ca²⁺ or sustained activity
- Enables delayed reward learning (seconds delay)

**Limitation**:
- Exponentially decaying traces mix events during delay
- Poor temporal precision for long delays

### Cascading Eligibility Traces (2025)
**Innovation**:
- State-space model inspired by biochemical reaction cascades
- Temporally precise memory for arbitrary delays
- Works at behavioral timescales (seconds to minutes)
- Applicable across layers with stacked delays
- Enables credit assignment for retrograde axonal signals or neuropeptides

### Dendritic Solutions
**Spatial Segregation**:
- Use dendritic spatial layout to distinguish credit signals
- Non-linear dendritic properties separate credit from non-credit inputs
- Credit signals to distal dendrites, task inputs to soma
- Local comparison generates learning signal

### Three-Factor Learning
**Neuromodulatory Gating**:
- STDP creates eligibility trace
- Neuromodulators (DA, NE, ACh) transcribe trace into synaptic change
- Feedback about external reward modulates plasticity
- Can revert sign of synaptic change

### Prefrontal Cortex Role
**Credit Assignment Hub**:
- PFC implicated in solving credit assignment
- Maintains representations during delays
- Links actions to outcomes
- Provides teaching signals to other areas

### AI Principles
1. **Eligibility traces required**: Cannot rely on immediate feedback
2. **Cascading traces**: Multi-stage biochemical models for long delays
3. **Three-factor learning**: Activity × eligibility × neuromodulation
4. **Dendritic compartments**: Spatial segregation of credit signals
5. **Hierarchical credit**: Different timescales at different levels
6. **Retrospective credit**: Ability to assign credit after-the-fact
7. **Neuromodulatory gates**: Global signals modulate local plasticity

---

## Key Biological Principles for AI Memory Systems

### 1. Dual-Store Architecture (CLS)
**Mandatory Separation**:
- **Fast episodic store**: Sparse, pattern-separated, rapid learning
- **Slow semantic store**: Distributed, overlapping, gradual integration
- **Learning rate ratio**: ~10-100x difference
- **Transfer mechanism**: Replay-based consolidation

### 2. Neuromodulatory Gating
**Multi-Signal Control**:
- **Dopamine**: Reward prediction error, value-based gating
- **Norepinephrine**: Novelty, arousal, long-term potentiation
- **Acetylcholine**: Attention, working memory, consolidation timing
- **Serotonin**: Learning rate, temporal discount
- **Combined signals**: Orthogonal control dimensions

**Implementation**:
- Gates determine **when** to learn, not just **what**
- Context-dependent plasticity rules
- Sleep/wake state modulation
- Importance/salience weighting

### 3. Multi-Timescale Dynamics
**Hierarchical Integration**:
- Fast timescales in early processing (10-100ms)
- Slow timescales in higher cognition (seconds)
- Multiple timescales per unit
- Independent parallel hierarchies

**Implementation**:
- Heterogeneous time constants
- Recurrent connections with varying delays
- Nested oscillatory dynamics
- Temporal receptive fields

### 4. Dendritic Compartmentalization
**Two-Compartment Neurons**:
- **Soma**: Feedforward task input
- **Dendrites**: Feedback, context, credit signals
- **Local learning**: Somato-dendritic mismatch
- **Non-linear integration**: Branch-specific computation

**Advantages**:
- No global error backpropagation needed
- Biologically plausible credit assignment
- Enhanced representational capacity
- Multi-timescale integration per neuron

### 5. Sparse, Efficient Coding
**Population Sparsity**:
- 1-5% activation at any time
- Overcomplete basis (more units than inputs)
- Decorrelated features
- Metabolic cost penalties

**Benefits**:
- Increased memory capacity
- Faster learning
- Better generalization
- Interpretable features

### 6. Replay-Based Consolidation
**Offline Learning**:
- Replay during rest/sleep, not just online
- Temporal compression (faster than real-time)
- Bidirectional replay (forward and reverse)
- Nested oscillatory coordination
- Selective replay based on value

**Mechanisms**:
- Sharp-wave ripples (150-250Hz)
- Slow oscillations (~1Hz)
- Spindles (10-15Hz)
- Coordinated multi-frequency dynamics

### 7. Metaplasticity and Uncertainty
**Bayesian Weights**:
- Maintain uncertainty estimate per parameter
- High confidence → low learning rate
- Low confidence → high learning rate
- Prevents both catastrophic forgetting and remembering

**Implementation**:
- Weight mean and variance
- Importance weighting (EWC)
- Eligibility traces
- Synaptic tagging and capture

### 8. Predictive Hierarchies
**Error-Based Learning**:
- Each level predicts level below
- Propagate prediction errors upward
- Update to minimize errors
- Precision-weighted learning

**Compression**:
- Transmit only unpredicted information
- Suppress predictable activity
- Efficient coding through prediction

### 9. Schema-Modulated Learning
**Consistency Gating**:
- Rapid learning when schema-consistent
- Slow learning when schema-inconsistent
- Catastrophic interference only with rapid inconsistent learning
- Neocortex can do fast learning with proper schemas

### 10. Active Forgetting
**Selective Consolidation**:
- Not all memories deserve long-term storage
- Value/importance-based selection
- Active interference mechanisms
- Retrieval-induced forgetting

---

## Computational Implementation Roadmap

### Phase 1: Core Dual-Store CLS
- [ ] Episodic buffer with sparse coding
- [ ] Semantic store with distributed representations
- [ ] 100x learning rate separation
- [ ] Replay-based consolidation loop

### Phase 2: Neuromodulatory Control
- [ ] Dopamine: Value-based gating
- [ ] Norepinephrine: Novelty detection
- [ ] Acetylcholine: Attention weighting
- [ ] Serotonin: Learning rate modulation
- [ ] Combined multi-signal integration

### Phase 3: Multi-Timescale Dynamics
- [ ] Heterogeneous time constants per unit
- [ ] Hierarchical timescale gradients
- [ ] Multiple timescales per neuron
- [ ] Nested oscillatory coordination

### Phase 4: Dendritic Computation
- [ ] Two-compartment neuron model
- [ ] Soma: Feedforward processing
- [ ] Dendrites: Feedback/context integration
- [ ] Local somato-dendritic learning rule
- [ ] Branch-specific plasticity

### Phase 5: Advanced Consolidation
- [ ] Multi-frequency replay (ripples, spindles, SOs)
- [ ] Temporal compression during replay
- [ ] Bidirectional replay
- [ ] Value-based selective replay
- [ ] Sleep/wake state modulation

### Phase 6: Metaplasticity
- [ ] Bayesian weight uncertainty
- [ ] Importance weighting (EWC-style)
- [ ] Cascading eligibility traces
- [ ] Synaptic tagging and capture
- [ ] Three-factor learning rules

### Phase 7: Predictive Coding
- [ ] Hierarchical prediction generation
- [ ] Prediction error computation
- [ ] Error-driven learning
- [ ] Precision weighting
- [ ] Compressed representations

### Phase 8: Active Forgetting
- [ ] Value-based consolidation selection
- [ ] Retrieval-induced forgetting
- [ ] Interference-based updating
- [ ] Schema-gated learning rates

---

## References

### Hippocampal Memory Systems
- [Pattern completion and pattern separation in the hippocampal circuit during naturalistic stimuli](https://onlinelibrary.wiley.com/doi/am-pdf/10.1002/hbm.70150) - Human Brain Mapping 2025
- [An Enduring Role for Hippocampal Pattern Completion](https://www.jneurosci.org/content/44/18/e1740232024) - Journal of Neuroscience 2024
- [Memory consolidation from a reinforcement learning perspective](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1538741/full) - Frontiers 2024
- [Awake reactivation of cortical memory traces](https://faculty.wcas.northwestern.edu/~paller/Progress in Neurobiology 2025.pdf) - Progress in Neurobiology 2025

### Synaptic Plasticity
- [Interplay of hippocampal LTP and LTD in enabling memory representations](https://royalsocietypublishing.org/doi/10.1098/rstb.2023.0229) - Phil Trans Royal Society B 2024
- [Anti-Hebbian plasticity drives sequence learning in striatum](https://www.nature.com/articles/s42003-024-06203-8) - Communications Biology 2024
- [Neuromodulated Spike-Timing-Dependent Plasticity](https://pmc.ncbi.nlm.nih.gov/articles/PMC4717313/)

### Neuromodulation
- [Pharmacological Modulation of Dopamine Receptors](https://www.jneurosci.org/content/45/6/e1301242024.full.pdf) - Journal of Neuroscience 2025
- [Emerging Functions of Neuromodulation during Sleep](https://www.jneurosci.org/content/44/40/e1277242024) - Journal of Neuroscience 2024
- [The Duality of Astrocyte Neuromodulation](https://pubmed.ncbi.nlm.nih.gov/40191899/) - 2025

### Memory Consolidation and Replay
- [Sleep's contribution to memory formation](https://journals.physiology.org/doi/full/10.1152/physrev.00054.2024) - Physiological Reviews 2024
- [Systems memory consolidation during sleep](https://pmc.ncbi.nlm.nih.gov/articles/PMC12576410/) - 2024
- [Mechanisms of systems memory consolidation during sleep](https://www.nature.com/articles/s41593-019-0467-3) - Nature Neuroscience
- [Coupled sleep rhythms for memory consolidation](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(24)00029-9) - Trends in Cognitive Sciences 2024

### Complementary Learning Systems
- [Why there are complementary learning systems](https://stanford.edu/~jlmcc/papers/McCMcNaughtonOReilly95.pdf) - Psychological Review 1995
- [What Learning Systems do Intelligent Agents Need?](https://pubmed.ncbi.nlm.nih.gov/27315762/?dopt=Abstract) - Updated CLS Theory 2016
- [Incorporating rapid neocortical learning](https://pubmed.ncbi.nlm.nih.gov/23978185/) - Schema extension
- [Emergence of complementary learning systems](https://2024.ccneuro.org/pdf/426_Paper_authored_CCN-Authored.pdf) - CCN 2024

### Predictive Coding
- [Dynamic predictive coding: hierarchical sequence learning](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011801) - PLOS Comp Bio 2024
- [Crossmodal hierarchical predictive coding](https://www.nature.com/articles/s42003-024-06677-6) - Communications Biology 2024
- [Where is the error? Hierarchical predictive coding through dendritic error computation](https://www.cell.com/trends/neurosciences/fulltext/S0166-2236(22)00186-2) - Trends in Neurosciences
- [Predictive Coding Light](https://www.nature.com/articles/s41467-025-64234-z) - Nature Communications 2025
- [Modelling Predictive Coding in V1](https://www.biorxiv.org/content/10.1101/2025.11.01.686040v1) - bioRxiv 2025

### Dendritic Computation
- [Advancing neural computation: dendritic learning in feedforward tree networks](https://pmc.ncbi.nlm.nih.gov/articles/PMC11751443/) - 2024
- [Local, calcium- and reward-based synaptic learning rule](https://elifesciences.org/articles/97274) - eLife 2024
- [Compartmentalized dendritic plasticity in retrosplenial cortex](https://www.nature.com/articles/s41593-025-01876-8) - Nature Neuroscience 2025
- [Dendritic Localized Learning](https://github.com/Lvchangze/Dendritic-Localized-Learning) - ICML 2025
- [A Neural Model for V1 with dendritic nonlinearities](https://www.jneurosci.org/content/45/43/e1975242025.full.pdf) - Journal of Neuroscience 2025

### Catastrophic Forgetting
- [Bayesian continual learning and forgetting](https://www.nature.com/articles/s41467-025-64601-w) - Nature Communications 2025
- [Hybrid neural networks for continual learning inspired by corticohippocampal circuits](https://www.nature.com/articles/s41467-025-56405-9) - Nature Communications 2025
- [Overcoming catastrophic forgetting in neural networks](https://www.pnas.org/doi/10.1073/pnas.1611835114) - PNAS

### Multi-Timescale Dynamics
- [Multi-timescale neural dynamics for multisensory integration](https://www.nature.com/articles/s41583-024-00845-7) - Nature Reviews Neuroscience 2024
- [Temporal dendritic heterogeneity with SNNs](https://www.nature.com/articles/s41467-023-44614-z) - Nature Communications
- [Neural timescales from a computational perspective](https://arxiv.org/html/2409.02684v2) - 2025
- [Multiple timescales across cortex](https://www.pnas.org/doi/10.1073/pnas.2005993117) - PNAS

### Sparse Coding
- [From superposition to sparse codes](https://arxiv.org/html/2503.01824v1) - 2025
- [Toward a unified theory of efficient, predictive, and sparse coding](https://www.pnas.org/doi/10.1073/pnas.1711114115) - PNAS
- [Neural correlates of sparse coding](https://pmc.ncbi.nlm.nih.gov/articles/PMC6597036/)
- [Relating sparse and predictive coding to divisive normalization](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013059) - PLOS 2024

### Credit Assignment
- [Learning From the Past with Cascading Eligibility Traces](https://arxiv.org/abs/2506.14598) - arXiv 2025
- [Solving the Credit Assignment Problem With the Prefrontal Cortex](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00182/full) - Frontiers
- [Dendritic solutions to the credit assignment problem](https://pubmed.ncbi.nlm.nih.gov/30205266/)
- [Brain-Inspired Machine Intelligence: Neurobiologically-Plausible Credit Assignment](https://arxiv.org/html/2312.09257v2)

---

## Conclusion

Biological memory systems employ a sophisticated suite of mechanisms that jointly solve the stability-plasticity dilemma. The core insight is that **no single mechanism is sufficient**—rather, it is the **interaction of multiple systems** that enables robust continual learning:

1. **Complementary Learning Systems** provide architectural separation
2. **Neuromodulation** provides adaptive gating and context sensitivity
3. **Multi-timescale dynamics** enable simultaneous fast and slow processing
4. **Dendritic computation** enables local, biologically plausible learning
5. **Replay-based consolidation** transfers knowledge offline without interference
6. **Metaplasticity** adjusts learning based on weight history and importance
7. **Sparse coding** increases capacity and interpretability
8. **Predictive coding** enables efficient, error-driven learning

Implementing these principles in artificial systems requires moving beyond single-store, uniform-timescale, backpropagation-based architectures toward heterogeneous, multi-store, locally-computed learning systems that mirror the brain's sophisticated memory organization.

The World Weaver system is well-positioned to implement these principles through its tripartite memory architecture, neuromodulatory orchestration, and consolidation engine. The next phase should focus on dendritic computation, multi-timescale dynamics, and replay-based consolidation to fully realize biologically-inspired continual learning.
